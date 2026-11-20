"""Team needs in free agency: does an outside free agent add less to his new team's room when the room is full?

Draft version (scripts/room_fullness_value.py): a round 1-3 pick added the same to thin and full rooms (veterans
absorb the gap). Free agents play at once and displace directly, so it's asked again for them. Specified before
running, same units and outcome as the draft test (scripts/net_room_value.py):
  unit       team x season x group (RB WR TE OL DL EDGE LB DB), seasons 2015-25
  outcome    room value: everyone who played the group for the team, snap share x pay-curve price (% of cap),
             season 1 and seasons 1-2 summed
  treatment  E_fa = forecast own value (src/veteran.Comps, other players' futures) of the team's OUTSIDE free-agent
             signings at the group that offseason (OTC 'UFA', APY >= 0.5% of cap, no snaps for the team the season
             before), summed over the horizon; % of cap
  fullness   ret = returning starter-equivalents before free agency: last season's snaps of players still with the
             team after departures (roster_need.departures_and_adds), over the group mean (1 = typical room)
  controls   ret, room value last season, the team's round 1-3 rookies' expected value at the group (draft test's
             E), group x season FE; SEs clustered by team
  reading    coefficient on E_fa = share of the signee's forecast value that shows up as room gain in a typical
             room (1 = all net gain, 0 = all displaced); E_fa x (ret - mean) < 0 = full rooms gain less. Also by
             tercile of ret (thin / typical / full). Second outcome: the signees' own value with the team.
Run: .venv/bin/python scripts/fa_room_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import fa_market as fm  # noqa: E402
import net_room_value as nv  # noqa: E402
import roster_need as rn  # noqa: E402
import veteran as vt  # noqa: E402

YEARS = range(2015, 2026)


LABELS = "snap" if "--snap-labels" in sys.argv else "otc"


def otc_groups():
    """pfr -> pricing group (the OTC contract position the value model uses; same as room values and signees)."""
    v = vt.season_table()
    return v.sort_values("season").groupby("pfr_id").group.last().to_dict()


def returning(ps_rn, year, gmap=None):
    """Returning starter-equivalents by team x group. Labels: OTC pricing group (default since Nov 20 2026; the first
    run used draft / snap-count labels, which file many edge rushers as LB or DL), snap label as fallback."""
    prev = ps_rn[(ps_rn.season == year - 1) & ps_rn.grp.isin(rn.GROUPS)]
    signed, away, _ = rn.departures_and_adds(year)
    left = np.array([(p, tm) in away or (p in signed and tm not in signed[p])
                     for p, tm in zip(prev.pfr_player_id, prev.team)], bool)
    r = prev[~left].assign(g=lambda d: d.grp.map(nv.TO_G))
    if gmap is not None:
        r["g"] = [gmap.get(p) or g for p, g in zip(r.pfr_player_id, r.g)]
    r = r[r.g.isin(nv.GROUPS)]
    return r.groupby(["team", "g"]).se.sum()


def signings():
    nick = dict(zip(*pd.read_parquet(vt.RAW / "teams.parquet")[["team_nick", "team_abbr"]].T.values))
    nick.update({"Washington": "WAS", "Redskins": "WAS", "Football Team": "WAS", "Commanders": "WAS"})
    played = rn._played_for()
    rows = []
    for p, hist in fm.history().items():
        for (y, kind, apy, yrs, gtd, team) in hist:
            if kind == "UFA" and y in YEARS and apy >= fm.MIN_APY:
                tm = rn.team(nick.get(team, team))
                if tm not in played.get((p, y - 1), set()):
                    rows.append((p, y, tm, apy))
    S = pd.DataFrame(rows, columns=["pfr_id", "Y", "team", "apy"]).drop_duplicates(["pfr_id", "Y"], keep="last")
    A = pd.read_parquet(vt.OUT / "vet_anchors.parquet")
    S = S.merge(A, on=["pfr_id", "Y"])
    S = S[S.grp.isin(nv.GROUPS)]
    C = vt.comps()
    F = pd.DataFrame([C.forecast(r) for r in S.to_dict("records")], index=S.index)
    S = S.join(F)
    for n in (1, 2):
        S[f"E_{n}"] = sum(S[f"f{k}"] for k in range(1, n + 1)) * 100
        S[f"own_{n}"] = np.nan
    return S


def main():
    pd.set_option("display.width", 200)
    U, R = nv.build()
    U = U[U.year.isin(YEARS)].copy()
    _, ps_rn, _, _ = rn.load()
    gmap = otc_groups() if LABELS == "otc" else None
    print(f"position labels for returning snaps: {LABELS}")
    ret = {y: returning(ps_rn, y, gmap) for y in YEARS}
    U["ret_se"] = [float(ret[y].get((t, g), 0.0)) for y, t, g in zip(U.year, U.team, U.g)]
    U["ret"] = U.ret_se / U.groupby("g").ret_se.transform("mean")
    U["terc"] = U.groupby("g").ret.transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["thin", "typical", "full"])).astype(str)
    S = signings()
    ps = nv.ws.player_seasons()
    own = ps.groupby(["pfr", "season", "team"]).v_now.sum()
    for n in (1, 2):
        S[f"own_{n}"] = [sum(float(own.get((p, y + k, tm), 0.0)) for k in range(n)) for p, y, tm in zip(S.pfr_id, S.Y, S.team)]
        e = S.groupby(["Y", "team", "grp"])[[f"E_{n}", f"own_{n}"]].sum()
        U[f"Efa_{n}"] = [float(e[f"E_{n}"].get(k, 0.0)) for k in zip(U.year, U.team, U.g)]
        U[f"ownfa_{n}"] = [float(e[f"own_{n}"].get(k, 0.0)) for k in zip(U.year, U.team, U.g)]
    tr = U[U.Efa_1 > 0]
    print(f"outside UFA signings matched: {len(S)}; rooms receiving one: {len(tr)} of {len(U)}")
    print(f"teams sign for need: share of rooms with an outside signing by tercile "
          f"{U.assign(d=U.Efa_1 > 0).groupby('terc').d.mean().round(3).to_dict()}")
    print(f"signees' forecast vs own realized value, season 1 (% of cap per room): {tr.Efa_1.mean():.2f} vs {tr.ownfa_1.mean():.2f}")
    rows = []
    for n, last in ((1, 2025), (2, 2024)):
        d = U[U.year <= last].copy()
        d["ret_c"] = d.ret - d[d[f"Efa_{n}"] > 0].ret.mean()
        d["x"] = d[f"Efa_{n}"] * d.ret_c
        for t in ["thin", "typical", "full"]:
            d[f"E_{t}"] = d[f"Efa_{n}"] * (d.terc == t)
        base = f"ret + v_last + Eall_{n} + C(g):C(year)"
        for y, lab in ((f"y_{n}", "room value"), (f"ownfa_{n}", "signees' own value")):
            m = smf.ols(f"{y} ~ Efa_{n} + x + {base}", d).fit(cov_type="cluster", cov_kwds={"groups": d.team})
            m3 = smf.ols(f"{y} ~ E_thin + E_typical + E_full + {base}", d).fit(cov_type="cluster", cov_kwds={"groups": d.team})
            tf = m3.t_test("E_thin - E_full = 0")
            ci, ci3 = m.conf_int(), m3.conf_int()
            rows.append({"horizon": f"seasons 1-{n}" if n > 1 else "season 1", "outcome": lab,
                         "per 1.0 of E": f"{m.params[f'Efa_{n}']:.2f} ({ci.loc[f'Efa_{n}', 0]:.2f} to {ci.loc[f'Efa_{n}', 1]:.2f})",
                         "E x fullness": f"{m.params['x']:+.2f} ({ci.loc['x', 0]:+.2f} to {ci.loc['x', 1]:+.2f})",
                         "thin": round(m3.params["E_thin"], 2), "typical": round(m3.params["E_typical"], 2),
                         "full": round(m3.params["E_full"], 2),
                         "thin - full": f"{float(tf.effect[0]):+.2f} ({float(tf.conf_int()[0, 0]):+.2f} to {float(tf.conf_int()[0, 1]):+.2f})"})
    print(pd.DataFrame(rows).to_string(index=False))
    # by group, season 1 room value
    out = []
    for g in nv.GROUPS:
        d = U[U.g == g].copy()
        d["ret_c"] = d.ret - d[d.Efa_1 > 0].ret.mean()
        d["x"] = d.Efa_1 * d.ret_c
        m = smf.ols("y_1 ~ Efa_1 + x + ret + v_last + Eall_1 + C(year)", d).fit(cov_type="cluster", cov_kwds={"groups": d.team})
        out.append({"group": g, "n rooms w/ FA": int((d.Efa_1 > 0).sum()), "per 1.0 of E": m.params["Efa_1"],
                    "E x fullness": m.params["x"], "p": m.pvalues["x"]})
    print("\nby group, season 1 room value:")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    cap_side(U, S)
    U.to_parquet(vt.OUT / ("fa_room_test.parquet" if LABELS == "otc" else "fa_room_test_snaplabels.parquet"))


def cap_side(U, S):
    """Where the signee's money comes from: spend on the group's other players (OTC cap hits), the signees' own cap
    hits, and the team's total spend, per 1.0 of E_fa (same controls + last season's group spend). For draft picks
    (scripts/cap_side.py) veteran spend at the group fell 0.54 per 1.0 of E: the rookie's value came back as cap."""
    import cap_side as cs
    H = cs.cap_hits()
    pl = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id"])
    p2g = dict(zip(pl.pfr_id, pl.gsis_id))
    H["g"] = H.g.replace({"ED": "EDGE"}) if "ED" in set(H.g.dropna()) else H.g
    spend = H.groupby(["season", "team", "g"]).pct.sum()
    tot = H.groupby(["season", "team"]).pct.sum()
    hit = H.groupby(["gsis", "season", "team"]).pct.sum()
    S = S.assign(gsis=S.pfr_id.map(p2g))
    rows = []
    for n, last in ((1, 2025), (2, 2024)):
        S[f"cap_{n}"] = [sum(float(hit.get((g, y + k, tm), 0.0)) for k in range(n)) for g, y, tm in zip(S.gsis, S.Y, S.team)]
        sc = S.groupby(["Y", "team", "grp"])[f"cap_{n}"].sum()
        d = U[U.year <= last].copy()
        d["sign_cap"] = [float(sc.get(k, 0.0)) for k in zip(d.year, d.team, d.g)]
        d["grp_spend"] = [sum(float(spend.get((y + k, t, g), 0.0)) for k in range(n)) for y, t, g in zip(d.year, d.team, d.g)]
        d["others"] = d.grp_spend - d.sign_cap
        d["team_tot"] = [sum(float(tot.get((y + k, t), 0.0)) for k in range(n)) for y, t in zip(d.year, d.team)]
        d["spend_last"] = [float(spend.get((y - 1, t, g), 0.0)) for y, t, g in zip(d.year, d.team, d.g)]
        for y, lab in (("sign_cap", "signees' own cap"), ("others", "group's other players' cap"), ("team_tot", "team total cap")):
            m = smf.ols(f"{y} ~ Efa_{n} + ret + v_last + spend_last + Eall_{n} + C(g):C(year)", d).fit(
                cov_type="cluster", cov_kwds={"groups": d.team})
            ci = m.conf_int()
            rows.append({"horizon": n, "outcome": lab, "per 1.0 of E": round(m.params[f"Efa_{n}"], 2),
                         "lo": round(ci.loc[f"Efa_{n}", 0], 2), "hi": round(ci.loc[f"Efa_{n}", 1], 2)})
    print("\nCap side (% of cap over the horizon, per 1.0 of the signees' forecast value):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
