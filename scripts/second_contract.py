"""Complaint #5: the model values only rookie-contract years (plus round 1's option year). How much do drafted
players add for the team that drafted them in seasons 5-7?

Data: drafted players of 2011-2020 (season 6 played by 2025; season 7 for 2011-2019), season values from
scripts/second_contract_label.py (same pricing as the label: snap share x veteran pay curve, cfdarko units;
/ outcomes_v2.CF_OVER_OURS_ALL to this project's units). A season counts for the drafting team only if he played
it for that team (contracts season_history); otherwise 0 (gone: comp picks ignored).
Cost of a retained season: APY (% of cap) of the latest contract with the drafting team signed by then (extension,
tag, tender, option, rookie deal); "cap" version: that season's cap hit instead (backloaded, so lower).

Two currencies, per season, % of cap:
  surplus  value - cost (what a below-market deal saves)
  model    the model's season points: rate x (value - VET_MIN) - COST_RATE x (cost - VET_MIN), rate 0.19 non-QB /
           0.34 QB, cap charge 0.10 (pick_value.pick_effect), / rate_nonqb to read in % of cap of non-QB value.
           This is how the model would credit the season: a player kept at market pay still adds (0.19 - 0.10) x pay.
Compared with seasons 1-4 (and round 1's option year as modeled) in the same currency.

Run: .venv/bin/python scripts/second_contract.py
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outcomes_v2 import CF_OVER_OURS_ALL  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from pick_value import COST_RATE  # noqa: E402
from value import VET_MIN  # noqa: E402

RATES = json.load(open(OUT / "bridge.json"))["preseason"]
CLASSES = range(2011, 2021)
ALIAS = {"Redskins": "Commanders", "Washington": "Commanders", "Football Team": "Commanders"}
nt = lambda t: ALIAS.get(t, t)


def load():
    L = pd.read_parquet(OUT / "labels_seasons5_7.parquet")
    base = pd.read_parquet(OUT / "labels_pickadj_narrow_darko.parquet").set_index("pfr_id")
    chk = L.set_index("pfr_id")[["value_y1", "value_y4", "value_y5"]].join(base[["value_y1", "value_y4", "value_y5"]].add_suffix("_b"))
    for c in ("value_y1", "value_y4", "value_y5"):
        assert np.allclose(chk[c].fillna(-9), chk[c + "_b"].fillna(-9)), c
    d = L[L.drafted & L.draft_year.isin(CLASSES) & L.gsis.notna()].copy()
    C = pd.read_parquet(RAW / "contracts.parquet")
    C = C[C.gsis_id.notna()]
    cap = (C.apy / C.apy_cap_pct).replace([np.inf], np.nan).groupby(C.year_signed).median()
    hist = {}
    for g, sh, ch, dt in zip(C.gsis_id, C.season_history, C.contract_history, C.draft_team):
        if g in hist or sh is None:
            continue
        hist[g] = (nt(dt), {int(x["year"]): (nt(x["team"]), x["cap_percent"]) for x in sh if str(x.get("year", "")).isdigit()},
                   [(int(x["year_signed"]), nt(x["team"]), x["contract_type"], x["apy"]) for x in (ch if ch is not None else [])
                    if x.get("year_signed") == x.get("year_signed") and x.get("year_signed") is not None])
    rows = []
    for r in d.itertuples():
        h = hist.get(r.gsis)
        for k in (5, 6, 7):
            y = r.draft_year + k - 1
            v = getattr(r, f"v{k}")
            if pd.isna(v):
                continue
            stay, c_apy, c_cap, kind = False, np.nan, np.nan, "gone"
            if h:
                dt, seasons, contracts = h
                team, cp = seasons.get(y, (None, None))
                stay = team is not None and team == dt
                if stay:
                    c_cap = cp if cp is not None else np.nan
                    mine = [c for c in contracts if c[1] == dt and c[0] <= y]
                    if mine:
                        ys, _, kind, apy = max(mine, key=lambda c: c[0])
                        c_apy = apy / cap.get(ys, np.nan) if apy is not None else np.nan
                    kind = {"Drafted": "rookie/option"}.get(kind, kind)
            rows.append({"pfr_id": r.pfr_id, "name": r.player_name, "year": r.draft_year, "pick": r.overall_pick,
                         "round": r.round, "grp": r.group, "k": k, "value": v / CF_OVER_OURS_ALL, "stay": stay,
                         "kind": kind if stay else "gone", "cost_apy": c_apy, "cost_cap": c_cap})
    S = pd.DataFrame(rows)
    S["cost_apy"] = S.cost_apy.fillna(S.cost_cap)
    # rookie seasons (and the option year) in the same units, from the production label
    R = d.assign(value14=d[[f"value_y{i}" for i in range(1, 5)]].sum(axis=1, min_count=4) / CF_OVER_OURS_ALL,
                 cost14=d[[f"cost_y{i}" for i in range(1, 5)]].sum(axis=1, min_count=4),
                 opt_v=d.value_y5.fillna(0) / CF_OVER_OURS_ALL, opt_c=d.cost_y5.fillna(0))
    return S, R


def model_pts(value, cost, grp, stay):
    rate = np.where(grp == "QB", RATES["qb"], RATES["nonqb"])
    p = rate * np.maximum(value - VET_MIN, 0) - COST_RATE * np.maximum(cost - VET_MIN, 0)
    return np.where(stay, p, 0.0) / RATES["nonqb"]


def main():
    S, R = load()
    S["surplus"] = np.where(S.stay, S.value - S.cost_apy, 0.0)
    S["surplus_cap"] = np.where(S.stay, S.value - S.cost_cap, 0.0)
    S["model"] = model_pts(S.value, S.cost_apy.fillna(0), S.grp, S.stay)
    R["rookie_surplus"] = R.value14 - R.cost14
    R["rookie_model"] = sum(model_pts(R[f"value_y{i}"] / CF_OVER_OURS_ALL, R[f"cost_y{i}"], R.group, True) for i in range(1, 5))
    R["opt_model"] = model_pts(R.opt_v, R.opt_c, R.group, R.opt_v > 0)
    bins = [0, 10, 32, 64, 100, 160, 262]
    S["bin"], R["bin"] = pd.cut(S.pick, bins), pd.cut(R.overall_pick, bins)
    print(f"{R.shape[0]} drafted players 2011-2020; % of cap, this project's units\n")
    print("Who is still with the drafting team, and on what:")
    print(pd.crosstab([S.k], S.kind, normalize="index").round(2).to_string(), "\n")
    w = S.pivot_table(index="bin", columns="k", values=["stay", "surplus", "surplus_cap", "model"], aggfunc="mean", observed=True)
    rk = R.groupby("bin", observed=True)[["rookie_surplus", "rookie_model", "opt_model"]].mean()
    out = pd.concat([rk, w], axis=1)
    out.columns = [c if isinstance(c, str) else f"{c[0]}_s{c[1]}" for c in out.columns]
    print("Per drafted player, by pick (seasons 5-7 for the drafting team; 'model' = model points / 0.19):")
    print((out * 100).round(2).to_string(), "\n")
    print("Season 5 model points as a share of seasons 1-4 (+ option) model points, by pick:")
    s5 = S[S.k == 5].groupby("bin", observed=True).model.mean()
    print((s5 / (rk.rookie_model + rk.opt_model)).round(2).to_string(), "\n")
    print("By position, picks 1-100, season 5 (model points x100; rookie seasons 1-4 for scale):")
    t = S[(S.k == 5) & (S.pick <= 100)].groupby("grp").agg(n=("model", "size"), stay=("stay", "mean"), s5=("model", "mean"),
                                                             surplus5=("surplus", "mean"))
    t["rookie14"] = R[R.overall_pick <= 100].groupby("group").rookie_model.mean()
    t["ratio"] = t.s5 / t.rookie14
    print(t.assign(s5=t.s5 * 100, surplus5=t.surplus5 * 100, rookie14=t.rookie14 * 100).round(2).to_string(), "\n")
    print("Biggest season-5 model points (retained):")
    print(S[S.k == 5].nlargest(12, "model")[["name", "year", "pick", "grp", "kind", "value", "cost_apy", "model"]].round(3).to_string(index=False))
    S.drop(columns="bin").to_parquet(OUT / "second_contract_seasons.parquet")


if __name__ == "__main__":
    main()
