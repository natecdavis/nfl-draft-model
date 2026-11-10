"""Old-class replication, part 2: recruit-to-combine weight gain and press-conference word length (vs draft slot).

Same design as scripts/old_class_replication.py: drafted classes 2006-2015 (never in the original tests, careers
complete), expected value from the slot model fit leave-one-class-out within 2006-15, seasons 1-4.

Specified before looking (originals in brackets):
  C. Weight gain (scripts/growth_test.build: combine weight minus recruit listing, z within position group in this
     sample; recruits from 2006, so drafted classes ~2008-15). Controls combine weight z, years since recruiting
     class, class fixed effects.
     C1 primary: all positions, per sd                 [x1.06 vs slot and consensus, p ~0.08-0.10]
     C2 OL/DL/EDGE/TE vs other; picks 1-100; top quarter of gain (secondary, as in the original)
  D. Mean word length (scripts/transcripts_test.features; ASAP transcripts begin with the 2011 bowl season, so
     classes 2012-15), players >= 200 words, z with the DISCOVERY sample's (2016-22) mean and sd, controls log words
     + class fixed effects, prediction value < 1, one-sided p.
     D1 primary: drafted vs slot                        [x0.85 discovery; fresh 2023-25 x0.95 failed]
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from growth_test import BIG, build, cont
from multisport_test import compare
from old_class_replication import expected_slot
from transcripts_test import features

OUT = pm.OUT


def growth_part(d):
    R = d.assign(drafted=True, player_id=d.player_id.astype(str))[
        ["player_id", "cls", "pfr_id", "grp", "drafted", "pick", "actual", "exp_slot"]]
    g = build(R, None)
    s = g.dropna(subset=["wt_gain_z"])
    fe = [f"c{c}" for c in sorted(s.cls.unique())[1:]]  # classes with recruit data; first as baseline
    for c in sorted(s.cls.unique()):
        g[f"c{c}"] = (g.cls == c).astype(float)
    s = g.dropna(subset=["wt_gain_z"])
    print(f"\nC. WEIGHT GAIN vs SLOT: drafted 2006-15 {len(g)}; with recruit listing + combine weight {len(s)}; by class "
          f"{s.groupby('cls').size().to_dict()}; median gain {s.wt_gain.median():.0f} lb")
    covs = ["cmb_wt_z", "years", *fe]
    cont(g, "exp_slot", "wt_gain_z", covs, "C1 primary: all positions")
    cont(g[g.grp.isin(BIG)], "exp_slot", "wt_gain_z", covs, "C2 OL / DL / EDGE / TE")
    cont(g[~g.grp.isin(BIG)], "exp_slot", "wt_gain_z", covs, "C2 other positions")
    cont(g[g.pick <= 100], "exp_slot", "wt_gain_z", covs, "C2 picks 1-100")
    cont(g[g.cls <= 2011], "exp_slot", "wt_gain_z", ["cmb_wt_z", "years"], "   classes 2008-11")
    cont(g[g.cls >= 2012], "exp_slot", "wt_gain_z", ["cmb_wt_z", "years"], "   classes 2012-15")
    compare(s, "gain_top", "exp_slot", covs=("cmb_wt_z", "years", *fe), label="C2 top quarter of gain at position")
    print("  height gained (same combine height), context")
    cont(g, "exp_slot", "ht_gain_z", ["cmb_ht_z", "years", *fe], "all positions")
    g.to_parquet(OUT / "old_class_growth.parquet")


def transcripts_part(d):
    a = pd.read_parquet(OUT / "asap_answers.parquet")
    a["player_id"] = a.player_id.astype(str)
    f = features(a)
    f = f[f.n_words >= 200]
    disc = f[f.cls.between(2016, 2022)].word_len
    f["z"] = (f.word_len - disc.mean()) / disc.std()
    f["log_words"] = np.log(f.n_words)
    s = d.assign(player_id=d.player_id.astype(str)).merge(f, on=["player_id", "cls"])
    print(f"\nD. WORD LENGTH vs SLOT: drafted transcript players {len(s)}; by class {s.groupby('cls').size().to_dict()}; "
          f"mean z {s.z.mean():+.2f}")
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    m = sm.GLM(s.actual, sm.add_constant(pd.concat([s[["z", "log_words"]], fe], axis=1)), family=sm.families.Poisson(),
               offset=np.log(s.exp_slot.clip(lower=1e-4))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc["z"])
    print(f"    D1 primary: per 1 sd value x{np.exp(m.params.z):.2f} (95% {lo:.2f}-{hi:.2f}; one-sided p "
          f"{stats.norm.cdf(m.params.z / m.bse.z):.3f})")
    top = s.z > s.z.median()
    r = lambda x: x.actual.sum() / x.exp_slot.sum()
    print(f"    actual / expected: longer-word half {r(s[top]):.2f}, shorter-word half {r(s[~top]):.2f}")
    for c, t in s.groupby("cls"):
        mc = sm.GLM(t.actual, sm.add_constant(t[["z", "log_words"]]), family=sm.families.Poisson(),
                    offset=np.log(t.exp_slot.clip(lower=1e-4))).fit(cov_type="HC1")
        print(f"      class {c}: n={len(t):3d} x{np.exp(mc.params.z):.2f} (p {mc.pvalues.z:.2f})")
    s.to_parquet(OUT / "old_class_transcripts.parquet")


def main():
    d = expected_slot()
    growth_part(d)
    transcripts_part(d)


if __name__ == "__main__":
    main()
