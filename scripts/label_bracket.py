"""Bracket check: pick values and real-trade verdicts under the stored label vs the grade-only label.

The stored label credits every snap a rookie played (an upper bound on what high picks earn, if some of their
playing time is teams protecting a draft investment); the grade-only label prices rookie seasons at the snap
share their PFF quality earns (a lower bound: it also discards real playing-time information and compresses
toward the middle). See README "Grade-only label". Live outputs are not touched; results go to
data/derived/bracket_<label>_{pick_value,trades}.parquet.

Decision rule (fixed before running): leave the labels alone if (1) no real trade verdict flips, a flip being a
team-side whose title change has opposite signs under the two labels with both more than 2 se from zero, and
(2) pick values relative to pick 32 move by less than ~10%. Otherwise build the pick-attributable correction.
Usage: python scripts/label_bracket.py [stored|grade] (default both)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import outcomes_v2
import pick_value as pv
import prospect_model as pm
import trade

LABELS = {"stored": pm.LABEL_FILE, "grade": pm.OUT / "labels_gradeonly_t20.parquet"}
SHOW = [1, 5, 10, 20, 32, 64, 96, 160]


def grid(ctx):
    rows = []
    for sname, sit in pv.SITUATIONS.items():
        base = pv.baseline_paths(sit, np.random.default_rng(11))
        base_titles = ctx.sb(base)
        for grp in pv.GROUPS:
            for pick in pv.PICKS:
                shares, price = pv.sample_outcomes(ctx.lib, ctx.curves, grp, pick, np.random.default_rng(pick))
                pts = pv.pick_effect(shares, price, pick, grp, sit, ctx.cost, ctx.bar, ctx.rates)
                if pts is None:
                    continue
                d = ctx.sb(base + pts) - base_titles
                rows.append({"situation": sname, "grp": grp, "pick": pick, "titles_added": d.sum(axis=1).mean(),
                             "se": d.sum(axis=1).std() / np.sqrt(pv.N)})
    res = pd.DataFrame(rows)
    nq = res[res.grp != "QB"].assign(w=lambda d: d.grp.map(dict(zip(ctx.nonqb_groups, ctx.nonqb_w))))
    nq = (nq.groupby(["situation", "pick"], sort=False)
          .apply(lambda d: np.average(d.titles_added, weights=d.w), include_groups=False)
          .rename("titles_added").reset_index().assign(grp="nonqb"))
    return pd.concat([res, nq], ignore_index=True)


def run(name):
    pm.LABEL_FILE = LABELS[name]
    outcomes_v2._MODEL = None
    ctx = trade.Context()
    g = grid(ctx)
    g.to_parquet(pm.OUT / f"bracket_{name}_pick_value.parquet")
    t = trade.real_trades(ctx)
    t.to_parquet(pm.OUT / f"bracket_{name}_trades.parquet")
    print(f"{name}: {len(g)} grid cells, {t.trade_id.nunique()} trades")


def compare():
    G = {n: pd.read_parquet(pm.OUT / f"bracket_{n}_pick_value.parquet") for n in LABELS}
    live = pd.read_parquet(pm.OUT / "pick_value_titles.parquet")
    chk = G["stored"].merge(live, on=["situation", "grp", "pick"], suffixes=("", "_live"))
    print(f"stored-label rerun vs live pick_value_titles: max abs diff "
          f"{(chk.titles_added - chk.titles_added_live).abs().max() * 100:.4f} (% of a title)\n")
    for grp in ["nonqb", "QB"]:
        print(f"{'Typical non-QB' if grp == 'nonqb' else 'QB'} pick, titles added (% of one Super Bowl), "
              f"stored / grade-only, and value relative to pick 32")
        for sname in pv.SITUATIONS:
            s = {n: G[n][(G[n].grp == grp) & (G[n].situation == sname)].set_index("pick").titles_added * 100
                 for n in LABELS}
            if s["stored"].empty:
                continue
            rel = {n: s[n] / s[n][32] for n in LABELS}
            print(f"  {sname}")
            print("    pick            " + "".join(f"{p:>8d}" for p in SHOW))
            print("    stored          " + "".join(f"{s['stored'][p]:8.2f}" for p in SHOW))
            print("    grade-only      " + "".join(f"{s['grade'][p]:8.2f}" for p in SHOW))
            print("    rel-32 change   " + "".join(f"{rel['grade'][p] / rel['stored'][p] - 1:+8.0%}" for p in SHOW))
    T = {n: pd.read_parquet(pm.OUT / f"bracket_{n}_trades.parquet") for n in LABELS}
    m = T["stored"].merge(T["grade"], on=["trade_id", "season", "team", "role"], suffixes=("_s", "_g"))
    sign = np.sign(m.change_s) != np.sign(m.change_g)
    firm = (m.change_s.abs() > 2 * m.se_s) & (m.change_g.abs() > 2 * m.se_g)
    print(f"\nREAL TRADES 2013-25: {m.trade_id.nunique()} trades, {len(m)} team-sides")
    print(f"  any sign change: {int(sign.sum())}; flips (both > 2 se from zero): {int((sign & firm).sum())}")
    print(f"  mean change, trade-up side: stored {m[m.role == 'up'].change_s.mean():+.2f}, grade-only "
          f"{m[m.role == 'up'].change_g.mean():+.2f} (% of a title); correlation {m.change_s.corr(m.change_g):.3f}")
    print(f"  share of trade-ups that gain titles: stored {(m[m.role == 'up'].change_s > 0).mean():.0%}, "
          f"grade-only {(m[m.role == 'up'].change_g > 0).mean():.0%}")
    if sign.any():
        print(m[sign][["trade_id", "season", "team", "role", "change_s", "se_s", "change_g", "se_g"]]
              .round(2).to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1:] or list(LABELS)
    for n in which:
        if n in LABELS:
            run(n)
    compare()
