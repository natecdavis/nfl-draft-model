"""How alike are the suggested free-agent signings across teams, with the depth switch off (default) and on?

For every team, the Offseason plan page's default greedy plan (off.plan) as the page builds it before anything is
clicked: team situation and picks from teams.team_inputs, the projected cap, saved dead money, off-limits players,
suggested cuts / restructures applied to the budget, open roster slots credited, most signings 8, QB only when the
team needs one. Ranked twice per team: need=False (default) and need=True (experimental depth factor).
Titles only (strict price and break-even skipped: the plan doesn't use them).
Reports: how many teams' plans each player is in, pairwise Jaccard overlap of plans, rank correlation of titles per $
across teams, and how much each team's plan changes with the depth factor.
Run: .venv/bin/python scripts/fa_plan_overlap.py   (writes reports/fa_plan_overlap.csv)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd

sys.path[:0] = ["scripts", "src"]
import fa_market as fm  # noqa: E402
import offseason as off  # noqa: E402
import vet_trade as vtr  # noqa: E402
import veteran as vt  # noqa: E402

SIGN_SEASON = vt.LAST + 2
MAX_N = 8


def cap_projection():
    c = pd.read_parquet(vt.RAW / "contracts.parquet", columns=["year_signed", "apy", "apy_cap_pct"])
    cap_last = float((c.apy / c.apy_cap_pct).replace([np.inf], np.nan).groupby(c.year_signed).median().loc[SIGN_SEASON - 1])
    return float(round(cap_last * vtr.CAP_GROWTH))


def one_team(args):
    team, sit, picks = args
    from teams import team_inputs  # noqa: F401  (warm imports in the worker)
    cap_m = cap_projection()
    qb_need = sit["qb_from"] is not None
    picks = sorted({int(x) for x in (picks or [40 + 32 * r for r in range(6)])})
    dead = off.load_dead().get(team, (0.0, "", ""))[0]
    B = off.budget(team, cap_m, picks, 0.0, dead, 0.03)
    T, _ = off.roster_moves(team, sit, cap_m)
    sug_c, sug_r, sug_v = off.suggest(T, B["budget"]) if len(T) else (set(), set(), set())
    saved = (T[T.pfr_id.isin(sug_c)].saved_cut.sum() + T[T.pfr_id.isin(sug_r - sug_v)].saved_restr.sum()
             + T[T.pfr_id.isin(sug_v)].saved_restr_v.sum()) if len(T) else 0.0
    n_open = off.open_slots(B["n_signed"], len(sug_c), len(picks))
    b_after = B["budget"] + saved - (n_open * off.slot_credit(cap_m) - B["fill"])
    offl = set(off.load_off_limits(team))
    FA = fm.free_agents(SIGN_SEASON)
    out = []
    for need in (False, True):
        R = fm.rank(FA, sit, team=team, need=need, strict=False, break_even_=False)
        R["note"] = [fm.notes(r) for r in R.itertuples()]
        R = R[~R.pfr_id.isin(offl)]
        P = off.plan(R, b_after, cap_m, MAX_N, qb_need=qb_need, open_slots=n_open) if b_after > 0 else R.iloc[0:0]
        C = R[(R.titles > 0) & ~R.restricted].copy()
        C["per_10m"] = C.titles * 100 / (C.apy_pred * cap_m) * 10
        for r in C.itertuples():
            out.append({"team": team, "need": need, "pfr_id": r.pfr_id, "player": r.display_name,
                        "grp": r.grp, "titles": r.titles, "per_10m": r.per_10m, "need_factor": r.need_factor,
                        "in_plan": r.pfr_id in set(P.pfr_id), "budget": b_after, "start": sit["start"]})
    return out


def main():
    from teams import team_inputs
    TI = team_inputs()
    jobs = [(t, TI.loc[t].situation, list(TI.loc[t].picks) if TI.loc[t].picks else None) for t in TI.index]
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(one_team, jobs)):
            rows += r
            print(f"{i + 1}/{len(jobs)} {jobs[i][0]}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(vt.ROOT / "reports" / "fa_plan_overlap.csv", index=False)
    report(D)


def report(D):
    pd.set_option("display.width", 200)
    teams = sorted(D.team.unique())
    print(f"\n{len(teams)} teams; budgets ${D.drop_duplicates('team').budget.median():.0f}M median "
          f"({(D.drop_duplicates('team').budget <= 0).sum()} teams with no budget left -> empty plan)")
    for need in (False, True):
        lab = "depth ON " if need else "depth OFF"
        d = D[D.need == need]
        plans = {t: set(d[(d.team == t) & d.in_plan].pfr_id) for t in teams}
        nonempty = [t for t in teams if plans[t]]
        jac = [len(plans[a] & plans[b]) / len(plans[a] | plans[b]) for a, b in combinations(nonempty, 2)]
        cnt = d[d.in_plan].groupby(["player", "grp"]).size().sort_values(ascending=False)
        piv = d.pivot_table(index="pfr_id", columns="team", values="per_10m")
        rho = piv.rank().corr().values[np.triu_indices(piv.shape[1], 1)]
        print(f"\n[{lab}] plans non-empty {len(nonempty)}; distinct players across all plans {len(cnt)}; "
              f"mean plan size {np.mean([len(plans[t]) for t in nonempty]):.1f}")
        print(f"  pairwise Jaccard of plans: median {np.median(jac):.2f}, 10-90% {np.percentile(jac, 10):.2f}-"
              f"{np.percentile(jac, 90):.2f}")
        print(f"  rank correlation of titles per $ across teams: median {np.median(rho):.3f}, min {rho.min():.3f}")
        print("  players in the most plans:")
        print("   ", "; ".join(f"{p} ({g}) {n}" for (p, g), n in cnt.head(12).items()))
    a = D[~D.need & D.in_plan].groupby("team").pfr_id.apply(set)
    b = D[D.need & D.in_plan].groupby("team").pfr_id.apply(set)
    ch = pd.Series({t: len(a.get(t, set()) ^ b.get(t, set())) / 2 for t in teams})
    print(f"\nper-team change with depth ON: signings swapped median {ch.median():.1f}, max {ch.max():.1f} "
          f"(of up to {MAX_N}); teams with any change {(ch > 0).sum()}/{len(teams)}")
    nf = D[D.need].drop_duplicates(["team", "grp"]).need_factor
    print(f"depth factor across team x group: range {nf.min():.2f}-{nf.max():.2f}, sd {nf.std():.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(pd.read_csv(vt.ROOT / "reports" / "fa_plan_overlap.csv"))
    else:
        main()
