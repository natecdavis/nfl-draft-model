"""Net gain of a pick: does a rookie add less to his team when the snaps he displaces are good?

Earlier tests measured the rookie's own value against his incumbents (scripts/quality_need.py: each good incumbent
starter-equivalent cost the rookie ~13% of his value, each weak one ~11%), and the realized-title yardstick scores the
draftee's own career. Neither nets out the snaps he takes from teammates. Here the outcome is the team's whole room:

  room value  = sum over everyone who played the position for the team that season of value (% of cap): snap share
                priced at his quality on the veteran pay curve (scripts/wr_stacking.py player-seasons). Rooms are
                ~fixed snap totals, so a rookie raises it only by (his quality - displaced quality) x his snaps.
  unit        team x season x group (RB, WR, TE, OL, DL, EDGE, LB, DB), drafts 2014-25 (horizons: season 1,
                seasons 1-2 for drafts <= 2024, seasons 1-4 for drafts <= 2022).
  treatment   E = expected own value over the horizon of the team's round 1-3 picks at the group in that draft
                (fit: rookie's own value ~ group x log pick + group x log consensus rank).
  room        draft-day incumbents' starter-equivalents by quality tier (scripts/displacement.py: DARKO-for-PFF
                tercile; contracts as roster_need.signed_by_draft): hq, mq, lq, plus unknown.
  model       room value ~ E + E x hq + E x lq (centred at the mean room) + hq + mq + lq + unk + room value last
                season + group x season FE; SEs clustered by team.
  Reading: the coefficient on E is the room's gain per 1.0 of the rookie's expected own value in a typical room
  (1 = all of it is net gain); E x hq < 0 means good incumbents shrink the net gain.

Part 2 (selection): teams rarely spend early picks where they're strong unless they know something (an incumbent
about to fade, a special prospect), which flatters rookies drafted into strong rooms. Fallers (picked well after
their consensus rank: log(pick + 10) - log(rank + 10) in the top third) were taken mostly because they were there,
so room quality at their picks is closer to chance. E and its interactions are split into fallers / others.

Usage: python scripts/net_room_value.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import displacement as dz  # noqa: E402
import roster_need as rn  # noqa: E402
import wr_stacking as ws  # noqa: E402
from outlook import OUT, RAW  # noqa: E402

GROUPS = ["RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]
TO_G = {"RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "IDL": "DL", "ED": "EDGE", "LB": "LB",
        "CB": "DB", "S": "DB"}
HORIZONS = {"season 1": (1, 2025), "seasons 1-2": (2, 2024), "seasons 1-4": (4, 2022)}


def build():
    ps = ws.player_seasons()
    ps = ps[ps.g.isin(GROUPS)]
    room = ps.groupby(["season", "team", "g"]).v_now.sum()
    own = ps.groupby(["pfr", "season", "team"]).v_now.sum()
    # rookies, rounds 1-3, from the review's picks (consensus rank, team after draft-night trades)
    R = pd.read_parquet(OUT / "review_picks.parquet")
    R = R[R["round"].between(1, 3) & R.year.between(2014, 2025)].copy()
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pid = {(int(y), int(p)): i for y, p, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    R["pfr"] = [pid.get((y, p)) for y, p in zip(R.year, R.pick)]
    g_of = ps.groupby("pfr").g.agg(lambda x: x.mode().iloc[0])
    R["g"] = [g_of.get(p) if p in g_of.index else TO_G.get(g) for p, g in zip(R.pfr, R.grp)]
    R = R[R.g.isin(GROUPS)]
    R["lp"], R["lr"] = np.log(R.pick + 10), np.log(R.board_rank.fillna(330) + 10)
    R["fall"] = R.lp - R.lr
    R["faller"] = R.fall >= R.fall.quantile(2 / 3)
    for h, (n, last) in HORIZONS.items():
        R[f"own_{n}"] = [sum(float(own.get((p, y + k, tm), 0.0)) for k in range(n)) if y <= last else np.nan
                         for p, y, tm in zip(R.pfr, R.year, R.team)]
        d = R[R[f"own_{n}"].notna()]
        m = smf.ols(f"own_{n} ~ C(g) * lp + C(g) * lr", d).fit()
        R[f"E_{n}"] = m.predict(R)
    # incumbents by quality tier (draft day)
    A = dz.attributes()
    s, ps_rn, games, _ = rn.load()
    I = dz.types(pd.concat([dz.incumbents(ps_rn, y, A) for y in range(2014, 2026)], ignore_index=True))
    I["g"] = I.grp.map(TO_G)
    inc = I.pivot_table(index=["year", "team", "g"], columns="qt", values="se", aggfunc="sum", fill_value=0.0)
    # team-season-group table
    teams = sorted(ps.team.unique())
    idx = pd.MultiIndex.from_product([range(2014, 2026), teams, GROUPS], names=["year", "team", "g"])
    U = pd.DataFrame(index=idx).reset_index()
    for c in ["high", "mid", "low", "unk"]:
        U[c] = [float(inc[c].get(k, 0.0)) if c in inc else 0.0 for k in zip(U.year, U.team, U.g)]
    U["v_last"] = [float(room.get((y - 1, t, g), 0.0)) for y, t, g in zip(U.year, U.team, U.g)]
    for h, (n, last) in HORIZONS.items():
        U[f"y_{n}"] = [sum(float(room.get((y + k, t, g), 0.0)) for k in range(n)) if y <= last else np.nan
                       for y, t, g in zip(U.year, U.team, U.g)]
        for who, mask in [("all", R.faller | ~R.faller), ("fall", R.faller), ("other", ~R.faller)]:
            e = R[mask].groupby(["year", "team", "g"])[f"E_{n}"].sum()
            U[f"E{who}_{n}"] = [float(e.get(k, 0.0)) for k in zip(U.year, U.team, U.g)]
    U["n_rook"] = [int(x) for x in [R.groupby(["year", "team", "g"]).size().get(k, 0) for k in zip(U.year, U.team, U.g)]]
    return U, R


def fit(U, n, split=False):
    d = U[U[f"y_{n}"].notna()].copy()
    # centre room quality at the mean of rooms that received a rookie
    tr = d[d[f"Eall_{n}"] > 0]
    for c in ["high", "low"]:
        d[f"{c}_c"] = d[c] - tr[c].mean()
    whos = ["fall", "other"] if split else ["all"]
    terms = []
    for w in whos:
        E = f"E{w}_{n}"
        d[f"{w}_x_high"], d[f"{w}_x_low"] = d[E] * d.high_c, d[E] * d.low_c
        terms += [E, f"{w}_x_high", f"{w}_x_low"]
    f = f"y_{n} ~ " + " + ".join(terms) + " + high + mid + low + unk + v_last + C(g):C(year)"
    m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.team})
    return m, terms, d


def main():
    U, R = build()
    pd.set_option("display.width", 200)
    print(f"rookies rounds 1-3 (non-QB) 2014-25: {len(R)}; fallers (top third of pick - rank): {R.faller.sum()} "
          f"(fall >= {R.fall.quantile(2 / 3):.2f} in log units, ~{(np.exp(R.fall.quantile(2 / 3)) - 1) * 100:.0f}% later than rank)")
    tr = U[U["Eall_1"] > 0]
    print(f"rooms receiving a rookie: mean incumbent se high {tr.high.mean():.2f}, mid {tr.mid.mean():.2f}, "
          f"low {tr.low.mean():.2f} (all rooms: {U.high.mean():.2f} / {U.mid.mean():.2f} / {U.low.mean():.2f})")
    print(f"  rooms receiving a faller: high {U[U.Efall_1 > 0].high.mean():.2f}; other picks: {U[U.Eother_1 > 0].high.mean():.2f}")
    sdh = tr.high.std()
    print(f"  sd of high-tier se across rooms receiving a rookie: {sdh:.2f}")
    rows = []
    for h, (n, _) in HORIZONS.items():
        for split in [False, True]:
            m, terms, d = fit(U, n, split)
            ci = m.conf_int()
            for t in terms:
                rows.append({"horizon": h, "term": t.replace(f"_{n}", ""), "coef": m.params[t], "lo": ci.loc[t, 0],
                             "hi": ci.loc[t, 1], "p": m.pvalues[t]})
            if split:
                tst = m.t_test("fall_x_high - other_x_high = 0")
                rows.append({"horizon": h, "term": "fall_x_high - other_x_high", "coef": float(tst.effect[0]),
                             "lo": float(tst.conf_int()[0, 0]), "hi": float(tst.conf_int()[0, 1]), "p": float(tst.pvalue)})
        mean_e = R[f"E_{n}"].mean()
        rows.append({"horizon": h, "term": "(mean E per rookie)", "coef": mean_e})
    T = pd.DataFrame(rows)
    print("\nRoom value (% of cap, summed over the horizon) per 1.0 of rookies' expected own value (E), and per 1.0 x one more")
    print("starter-equivalent of high / low-tier incumbents than the average room that drafted there:")
    print(T.round(3).to_string(index=False))
    # by group, season 1-2, pooled E only
    out = []
    for g in GROUPS:
        m, terms, d = fit(U[U.g == g], 2)
        ci = m.conf_int()
        out.append({"group": g, "E": m.params["Eall_2"], "lo": ci.loc["Eall_2", 0], "hi": ci.loc["Eall_2", 1],
                    "E x high": m.params["all_x_high"], "lo ": ci.loc["all_x_high", 0], "hi ": ci.loc["all_x_high", 1]})
    print("\nby group, seasons 1-2:")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    U.to_parquet(OUT / "net_room_value.parquet")


if __name__ == "__main__":
    main()
