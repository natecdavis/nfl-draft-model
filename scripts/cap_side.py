"""Cap side of a pick: does drafting at a position cut veteran spending there, and does the money buy value elsewhere?

scripts/net_room_value.py: only ~5-25% of a rookie's expected own value (snap share priced at quality on the veteran
pay curve) shows up as a better room on the field; the rest is snaps veterans would have played. The pay-curve
currency counts a rookie's value as what the team would otherwise pay a veteran for that production, so the gain should
come back as cap money. Same units and treatment as net_room_value (team x season x group, drafts 2014-25, non-QB
groups; E = expected own value of the team's round 1-3 picks at the group, % of cap; controls: draft-day incumbents by
quality tier, last season's room value, group x season FE; team-clustered), outcomes summed over the horizon:

  vet_spend    cap % paid to players at the group other than this draft class (OTC season history, cap hits of
               every player the team carried, including injured / inactive; contract position -> group)
  class_cost   cap % of this draft class at the group (the rookies' own cost)
  other_spend  cap % paid at the team's other positions (QB and specialists included)
  other_value  room value at the team's other non-QB groups (field)
  plus last season's group spend as a control.
Reading: vet_spend per 1.0 of E near -0.8 would mean the rookie replaces veteran purchases about as the currency
assumes; other_spend / other_value rising would mean the money is redeployed.

Usage: python scripts/cap_side.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import net_room_value as nrv  # noqa: E402
import roster_need as rn  # noqa: E402
import wr_stacking as ws  # noqa: E402
from outlook import OUT, RAW  # noqa: E402

POS_G = {"WR": "WR", "CB": "DB", "S": "DB", "IDL": "DL", "LB": "LB", "ED": "EDGE", "RB": "RB", "FB": "RB", "TE": "TE",
         "LG": "OL", "LT": "OL", "RT": "OL", "RG": "OL", "C": "OL", "QB": "QB", "K": "SPEC", "P": "SPEC", "LS": "SPEC"}


def cap_hits() -> pd.DataFrame:
    """One row per player x season x team: cap % (cap hit / league cap), group, pfr id, draft year."""
    c = pd.read_parquet(RAW / "contracts.parquet", columns=["gsis_id", "position", "draft_year", "season_history"])
    c = c.dropna(subset=["gsis_id"])
    c["n"] = c.season_history.map(lambda x: len(x) if x is not None else 0)
    c = c.sort_values("n").drop_duplicates("gsis_id", keep="last")
    nick = dict(zip(*pd.read_parquet(RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": nick.get("Commanders", "WAS"), "Redskins": nick.get("Commanders", "WAS"),
                 "Oilers": nick.get("Titans", "TEN")})
    rows = []
    for g, pos, dy, h in zip(c.gsis_id, c.position, c.draft_year, c.season_history):
        for x in h if h is not None else []:
            try:
                y = int(x["year"])
            except (TypeError, ValueError):
                continue
            if x.get("cap_number") is None or x.get("team") is None:
                continue
            rows.append((g, POS_G.get(pos), dy, y, rn.team(nick.get(x["team"], x["team"])), float(x["cap_number"]),
                         x.get("cap_percent")))
    H = pd.DataFrame(rows, columns=["gsis", "g", "draft_year", "season", "team", "cap", "pct"])
    H["team"] = H.team.replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
    lc = (H.cap / H.pct.astype(float)).groupby(H.season).median()
    H["pct"] = H.cap / H.season.map(lc) * 100
    return H


def build():
    U, R = nrv.build()
    H = cap_hits()
    H = H[H.season.between(2013, 2028)]
    spend = H.groupby(["season", "team", "g"]).pct.sum()
    cls = H[H.draft_year == H.season].copy()  # not used directly; class membership by draft year below
    tot = H.groupby(["season", "team"]).pct.sum()
    ps = ws.player_seasons()
    room = ps[ps.g.isin(nrv.GROUPS)].groupby(["season", "team", "g"]).v_now.sum()
    room_tot = ps[ps.g.isin(nrv.GROUPS)].groupby(["season", "team"]).v_now.sum()
    by_class = H.groupby(["season", "team", "g", "draft_year"]).pct.sum()
    for name, (n, last) in nrv.HORIZONS.items():
        def summ(f):
            return [sum(f(y + k, t, g, y) for k in range(n)) if y <= last else np.nan for y, t, g in zip(U.year, U.team, U.g)]
        cc = lambda s, t, g, y: float(by_class.get((s, t, g, y), 0.0))
        U[f"class_{n}"] = summ(cc)
        U[f"vet_{n}"] = summ(lambda s, t, g, y: float(spend.get((s, t, g), 0.0)) - cc(s, t, g, y))
        U[f"oth_{n}"] = summ(lambda s, t, g, y: float(tot.get((s, t), 0.0)) - float(spend.get((s, t, g), 0.0)))
        U[f"othv_{n}"] = summ(lambda s, t, g, y: float(room_tot.get((s, t), 0.0)) - float(room.get((s, t, g), 0.0)))
        U[f"tot_{n}"] = summ(lambda s, t, g, y: float(tot.get((s, t), 0.0)))
    U["spend_last"] = [float(spend.get((y - 1, t, g), 0.0)) for y, t, g in zip(U.year, U.team, U.g)]
    U["tot_last"] = [float(tot.get((y - 1, t), 0.0)) for y, t in zip(U.year, U.team)]
    U["othv_last"] = [float(room_tot.get((y - 1, t), 0.0)) - float(room.get((y - 1, t, g), 0.0)) for y, t, g in zip(U.year, U.team, U.g)]
    return U, R


def main():
    U, R = build()
    pd.set_option("display.width", 200)
    cov = U[U.year.between(2014, 2025)].groupby("year").apply(lambda d: d.drop_duplicates("team").tot_1.mean(), include_groups=False)
    print("team cap % covered by OTC cap hits, season 1, mean by draft year:", cov.round(0).to_dict())
    rows = []
    for h, (n, _) in nrv.HORIZONS.items():
        d = U[U[f"vet_{n}"].notna()]
        base = "high + mid + low + unk + v_last + spend_last + C(g):C(year)"
        for out, extra in [(f"y_{n}", ""), (f"vet_{n}", ""), (f"class_{n}", ""), (f"oth_{n}", " + tot_last"),
                           (f"othv_{n}", " + othv_last"), (f"tot_{n}", " + tot_last")]:
            for E in [f"Eall_{n}", f"Efall_{n}"]:
                f = f"{out} ~ {E}" + (f" + Eother_{n}" if "fall" in E else "") + f" + {base}{extra}"
                m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.team})
                lo, hi = m.conf_int().loc[E]
                rows.append({"horizon": h, "outcome": out.split("_")[0], "E": "fallers" if "fall" in E else "all",
                             "per 1.0 E": m.params[E], "lo": lo, "hi": hi, "p": m.pvalues[E]})
    T = pd.DataFrame(rows)
    lab = {"y": "room value (field)", "vet": "veteran spend at group", "class": "rookies' own cap", "oth": "spend at other positions",
           "othv": "room value at other positions", "tot": "team total spend"}
    T["outcome"] = T.outcome.map(lab)
    print("\nper 1.0 of rookies' expected own value (% of cap), summed over the horizon:")
    print(T.round(3).to_string(index=False))
    U.to_parquet(OUT / "cap_side.parquet")


if __name__ == "__main__":
    main()
