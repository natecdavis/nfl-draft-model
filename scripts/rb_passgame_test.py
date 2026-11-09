"""Running backs: do pass-game traits pick the RBs worth drafting, and do pass-catching backs help on net?

Follow-up to the RB finding (README "Running backs"): RB on-field effect on pass plays carries over to a new
team (RAPM movers 0.25), on run plays it doesn't (0.00). Competing claim (user): pass-game RBs lose value
because their targets displace more valuable WR targets.

Pre-specified (written before running).
PART A, draft (same harness as scripts/lit_bias_test.py): RBs only.
  Primary traits (z within RB, by class):
    1. college receiving share = career receptions / career touches (touches >= 100)
    2. PFF college receiving grade (pass_route), best season
    3. PFF college pass-block grade, best season
  Pricing: drafted board RBs 2014-24, log((board rank + 10) / (pick + 10)).
  Value: drafted 2016-22 vs slot, board 2018-22 vs consensus (Poisson, class fixed effects), label = the
  current default (narrow). 3 traits x 2 yardsticks: p < 0.008 survives Bonferroni.
  Transfer check: do the college traits predict NFL pass-play RAPM and NFL receiving / pass-block grades
  (years 1-4 averages, drafted RBs 2016-22 with NFL data)?
PART B, NFL mechanism (2016-25):
  B1. EPA per target by the targeted player's position (RB / WR / TE); same plays as scripts/pbp_metrics.py.
  B2. RB pass-play RAPM (pass plays only, ridge penalty 3200 as in pbp_metrics) on the RB's same-season
      PFF traits (z): target rate (targets per route), receiving grade, pass-block grade, pass-block rate
      (pass-block snaps per pass snap); RBs with >= 100 pass plays. The displacement claim predicts a
      negative target-rate coefficient. Also next-season pass RAPM on season-t traits (the part that persists).
Usage: python scripts/rb_passgame_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import pbp_metrics as pbm  # noqa: E402
import prospect_model as pm  # noqa: E402
from multisport_test import SLOT_CLASSES, expectations  # noqa: E402
from size_bias_test import reach, value_cont  # noqa: E402

PFF_NFL = Path.home() / "cfdarko/data/raw/pff_nfl"
TRAITS = [("rec_share_z", "1 college receiving share"), ("route_z", "2 PFF college receiving grade (best)"),
          ("pblock_z", "3 PFF college pass-block grade (best)")]


# ---------------------------------------------------------------- PART B data
def pass_rapm():
    rows = []
    for y in pbm.PART:
        m, _, _ = pbm.rapm_data(y, pbm.plays(y))
        mm = m[m.dropback].reset_index(drop=True)
        o = mm.offense_players.str.split(";").explode(); e = mm.defense_players.str.split(";").explode()
        X, ou, _ = pbm.rapm_X(mm, o, e, 3200)
        r = Ridge(alpha=1.0).fit(X, mm.epa.to_numpy())
        rows.append(pd.DataFrame({"gsis_id": ou, "season": y, "pass_rapm": r.coef_[:len(ou)] / np.sqrt(3200),
                                  "pass_n": o.value_counts().reindex(ou).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def nfl_rb_traits():
    rows = []
    for y in pbm.PART:
        r = pd.read_csv(PFF_NFL / str(y) / "rushing_summary.csv").drop_duplicates("player_id")
        r = r[r.position.isin(["HB", "FB"])]
        ob = pd.read_csv(PFF_NFL / str(y) / "offense_blocking.csv").drop_duplicates("player_id").set_index("player_id")
        r["pb_rate"] = r.player_id.map(ob.snap_counts_pass_block) / r.player_id.map(ob.snap_counts_pass_play)
        r["tgt_rate"] = r.targets / r.routes.where(r.routes > 0)
        rows.append(r.assign(season=y)[["player_id", "season", "tgt_rate", "grades_pass_route", "grades_pass_block",
                                        "pb_rate", "routes", "targets"]])
    t = pd.concat(rows, ignore_index=True)
    pl = pd.read_parquet(ROOT / "data/raw/players.parquet")[["pff_id", "gsis_id"]].dropna()
    pl = pl.assign(pff_id=pl.pff_id.astype(int)).drop_duplicates("pff_id")
    return t.merge(pl, left_on="player_id", right_on="pff_id").drop(columns=["pff_id"])


def epa_by_target_pos():
    pos = pd.read_parquet(ROOT / "data/raw/players.parquet")[["gsis_id", "position"]].dropna().drop_duplicates("gsis_id")
    pos = pos.set_index("gsis_id").position
    rows = []
    for y in pbm.PART:
        d = pbm.plays(y)
        d = d[d.dropback & d.receiver_player_id.notna()]
        rows.append(d.assign(pos=d.receiver_player_id.map(pos))[["pos", "epa"]])
    d = pd.concat(rows)
    d["pos"] = d.pos.replace({"FB": "RB"})
    g = d[d.pos.isin(["RB", "WR", "TE"])].groupby("pos").epa.agg(["count", "mean", "sem"])
    print("\nB1. EPA per target by targeted position, 2016-25 (win prob 0.05-0.95):")
    print(g.round(3).to_string())


def part_b():
    epa_by_target_pos()
    R = pass_rapm()
    T = nfl_rb_traits()
    d = T.merge(R, on=["gsis_id", "season"])
    d = d[d.pass_n >= 100].copy()
    X = ["tgt_rate", "grades_pass_route", "grades_pass_block", "pb_rate"]
    for c in X:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    Z = [c + "_z" for c in X]
    n = d[["gsis_id", "season", "pass_rapm"]].assign(season=lambda x: x.season - 1).rename(columns={"pass_rapm": "next"})
    d = d.merge(n, on=["gsis_id", "season"], how="left")
    print(f"\nB2. RB pass-play RAPM (EPA per play, x100) on same-season traits (per sd); RB-seasons with >= 100 pass plays")
    print(f"    mean pass RAPM x100 {100 * d.pass_rapm.mean():+.2f}; top-quartile target rate {100 * d.pass_rapm[d.tgt_rate >= d.tgt_rate.quantile(.75)].mean():+.2f}, "
          f"bottom quartile {100 * d.pass_rapm[d.tgt_rate <= d.tgt_rate.quantile(.25)].mean():+.2f}")
    for y, lab in [("pass_rapm", "same season"), ("next", "next season")]:
        e = d.dropna(subset=Z + [y])
        groups = pd.factorize(e.gsis_id)[0]
        m = sm.WLS(100 * e[y], sm.add_constant(e[Z]), weights=np.sqrt(e.pass_n)).fit(cov_type="cluster", cov_kwds={"groups": groups})
        print(f"    {lab:12s} n={len(e)}: " + "; ".join(f"{c[:-2]} {m.params[c]:+.2f} (t {m.tvalues[c]:+.1f})" for c in Z))
        for c in Z:  # one at a time
            m1 = sm.WLS(100 * e[y], sm.add_constant(e[[c]]), weights=np.sqrt(e.pass_n)).fit(cov_type="cluster", cov_kwds={"groups": groups})
            print(f"        alone: {c[:-2]:18s} {m1.params[c]:+.2f} (t {m1.tvalues[c]:+.1f})")
    return d


# ---------------------------------------------------------------- PART A
def zcls(d, col):
    return d.groupby("cls")[col].transform(lambda s: (s - s.mean()) / s.std())


def load_a():
    uni = pm.load()
    u = uni[uni.grp == "RB"][["player_id", "cls", "grp", "hasan_rank", "pick", "drafted", "on_board", "pfr_id",
                               "career_touches", "career_rush_att", "pff_grades_pass_route_peak",
                               "pff_grades_pass_block_peak"]].copy()
    u["rec_share"] = ((u.career_touches - u.career_rush_att) / u.career_touches).where(u.career_touches >= 100)
    u["rec_share_z"] = zcls(u, "rec_share")
    u["route_z"] = zcls(u, "pff_grades_pass_route_peak")
    u["pblock_z"] = zcls(u, "pff_grades_pass_block_peak")
    return uni, u


def transfer(u, d):
    """College traits vs NFL pass-game measures, years 1-4, drafted RBs 2016-22."""
    p = pd.read_parquet(ROOT / "data/raw/players.parquet")[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    x = u[u.drafted.astype(bool) & u.cls.between(2016, 2022)].merge(p, on="pfr_id")
    nfl = d.merge(x[["gsis_id", "cls"]], on="gsis_id")
    nfl = nfl[nfl.season.between(nfl.cls, nfl.cls + 3)]
    a = nfl.groupby("gsis_id")[["pass_rapm", "grades_pass_route", "grades_pass_block", "tgt_rate"]].mean()
    x = x.join(a, on="gsis_id")
    print(f"\n  transfer (drafted RBs 2016-22 with >= 100 NFL pass plays in a season; n={x.pass_rapm.notna().sum()}): corr")
    rows = {}
    for t, _ in TRAITS:
        rows[t] = {c: x[[t, c]].dropna().corr().iloc[0, 1] for c in ["pass_rapm", "grades_pass_route", "grades_pass_block", "tgt_rate"]}
    print(pd.DataFrame(rows).T.round(2).to_string())


def part_a(d):
    uni, u = load_a()
    dr = u[u.drafted.astype(bool)]
    print(f"\nPART A. RBs in universe {len(u)}, drafted {len(dr)}; coverage among drafted: "
          + ", ".join(f"{t} {dr[t].notna().mean():.0%}" for t, _ in TRAITS))
    print("  Pricing (drafted board RBs 2014-24; > 1 = taken earlier than board rank):")
    for t, lab in TRAITS:
        reach(u, t, lab)
    R = expectations(uni, SLOT_CLASSES)
    R = R[["player_id", "cls", "actual", "exp_slot", "exp_cons"]].merge(u.drop(columns=["pfr_id"]), on=["player_id", "cls"])
    for e, s, nm in [("exp_slot", R[R.drafted.astype(bool) & R.exp_slot.notna()], "DRAFTED RBs vs SLOT 2016-22"),
                     ("exp_cons", R[(R.on_board == 1) & R.exp_cons.notna()], "BOARD RBs vs CONSENSUS 2018-22")]:
        print(f"  Value, {nm} (< 1 = trait overvalued):")
        for t, lab in TRAITS:
            value_cont(s, e, t, lab)
    transfer(u, d)


def main():
    pd.set_option("display.width", 200)
    d = part_b()
    part_a(d)


if __name__ == "__main__":
    main()
