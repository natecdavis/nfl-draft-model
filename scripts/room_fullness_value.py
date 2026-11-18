"""Team-level room test: does a pick add more to its team's room when the room is thin than when it is full?

Follow-up to scripts/net_room_value.py (which asked about the QUALITY of incumbents and found no effect). Same units,
outcome, treatment and controls; the interaction here is the QUANTITY of returning snaps:

  ret   draft-day incumbents' starter-equivalents at the group (last season's snaps by players still on the team plus
        veterans added before the draft; scripts/displacement.py), divided by the group's mean, so 1 = a typical room.
  model room value ~ E + E x (ret - mean ret of rooms that drafted) + ret + tiers + room value last season
        + group x season FE; SEs clustered by team. E x ret < 0 means a fuller room shrinks the pick's net gain.
  Also by tercile of ret within group (thin / typical / full), and split into fallers / other picks (selection:
  fallers were taken mostly because they were there).
  Second outcome: the rookies' own realized value (does a fuller room cut what the rookie himself plays / earns?).

Usage: python scripts/room_fullness_value.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import net_room_value as nv  # noqa: E402


def prep(U, R):
    U = U.copy()
    U["tot"] = U.high + U.mid + U.low + U.unk
    U["ret"] = U.tot / U.groupby("g").tot.transform("mean")
    U["terc"] = U.groupby("g").ret.transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["thin", "typical", "full"])).astype(str)
    for n in [1, 2, 4]:
        for who, mask in [("all", slice(None)), ("fall", R.faller), ("other", ~R.faller)]:
            o = R[mask].groupby(["year", "team", "g"])[f"own_{n}"].sum(min_count=1)
            U[f"own{who}_{n}"] = [float(o.get(k, 0.0)) for k in zip(U.year, U.team, U.g)]
    return U


def fit(U, n, y, whos=("all",), by_terc=False):
    d = U[U[f"y_{n}"].notna()].copy()
    tr = d[d[f"Eall_{n}"] > 0]
    d["ret_c"] = d.ret - tr.ret.mean()
    terms = []
    for w in whos:
        E = f"E{w}_{n}"
        if by_terc:
            for t in ["thin", "typical", "full"]:
                d[f"{w}_{t}"] = d[E] * (d.terc == t)
                terms.append(f"{w}_{t}")
        else:
            d[f"{w}_x_ret"] = d[E] * d.ret_c
            terms += [E, f"{w}_x_ret"]
    f = f"{y}_{n} ~ " + " + ".join(terms) + " + ret + high + mid + low + v_last + C(g):C(year)"
    m = smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.team})
    return m, terms, d, tr


def table(U, y, label):
    rows = []
    for h, (n, _) in nv.HORIZONS.items():
        for whos, bt in [(("all",), False), (("all",), True), (("fall", "other"), False), (("fall", "other"), True)]:
            m, terms, d, tr = fit(U, n, y, whos, bt)
            ci = m.conf_int()
            for t in terms:
                rows.append({"horizon": h, "term": t.replace(f"_{n}", ""), "coef": m.params[t], "lo": ci.loc[t, 0],
                             "hi": ci.loc[t, 1], "p": m.pvalues[t]})
            if bt:
                for w in whos:
                    tst = m.t_test(f"{w}_thin - {w}_full = 0")
                    rows.append({"horizon": h, "term": f"{w}: thin - full", "coef": float(tst.effect[0]),
                                 "lo": float(tst.conf_int()[0, 0]), "hi": float(tst.conf_int()[0, 1]), "p": float(tst.pvalue)})
    print(f"\n{label}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main():
    U, R = nv.build()
    U = prep(U, R)
    pd.set_option("display.width", 200)
    tr = U[U.Eall_1 > 0]
    print(f"team x season x group rooms: {len(U)}; received a round 1-3 rookie: {len(tr)}")
    print(f"returning snaps (1 = typical room for the group): all rooms sd {U.ret.std():.2f}; rooms that drafted mean "
          f"{tr.ret.mean():.2f}, sd {tr.ret.std():.2f}; fallers' rooms {U[U.Efall_1 > 0].ret.mean():.2f}, others' {U[U.Eother_1 > 0].ret.mean():.2f}")
    print("by tercile (rooms that drafted): n, mean ret, mean starter-equivalents returning, mean E (season 1)")
    print(tr.groupby("terc").agg(n=("ret", "size"), ret=("ret", "mean"), se=("tot", "mean"), E=("Eall_1", "mean")).round(2).to_string())
    print("share of rooms that drafted R1-3, by tercile (all rooms):")
    print(U.assign(d=U.Eall_1 > 0).groupby("terc").d.mean().round(3).to_string())
    table(U, "y", "TEAM ROOM VALUE (% of cap over the horizon) per 1.0 of rookies' expected own value E")
    table(U, "ownall", "ROOKIES' OWN realized value per 1.0 of E")
    out = []
    for g in nv.GROUPS:
        m, terms, d, _ = fit(U[U.g == g], 2, "y")
        ci = m.conf_int()
        out.append({"group": g, "E": m.params["Eall_2"], "E x ret": m.params["all_x_ret"], "lo": ci.loc["all_x_ret", 0],
                    "hi": ci.loc["all_x_ret", 1]})
    print("\nby group, seasons 1-2 (room value):")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    U.to_parquet(nv.OUT / "room_fullness_value.parquet")


if __name__ == "__main__":
    main()
