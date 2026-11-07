"""Do multi-sport athletes or MLB draftees beat their expectation?

Expectations are out of sample (each class predicted by models trained on earlier classes, pm.rolling):
  drafted players  -> 'slot' model (draft slot + position): did teams under/over-rate them?
  board players    -> 'consensus_pre' model (consensus rank + position): did the public board?
Outcome: 4-season rookie-deal value (fraction of cap), cfdarko labels.

Signals:
  MLB drafted (src/multisport.py): vs slot 2016-2022, vs consensus 2018-2022.
  Other sports in the guides' BACKGROUND text (Brugler, 2019-2022): basketball, track, throws,
  baseball, wrestling, soccer/lacrosse/rugby, any. Each also tested at the positions where a
  benefit is plausible (basketball: TE/WR/EDGE; track: WR/DB/RB; wrestling and throws: OL/DL/EDGE/LB;
  baseball: QB/WR). Text flags control for background length (longer bios mention more sports).
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import multisport as ms
import prospect_model as pm
from outcomes_v2 import REG
from outlook import OUT

SLOT_CLASSES = tuple(range(2016, 2023))  # slot model needs 4 earlier classes (from 2012)
CONS_CLASSES = tuple(range(2018, 2023))  # consensus model needs board classes (from 2014) with 4 seasons played
POS = {"basketball": {"TE", "WR", "EDGE"}, "track": {"WR", "DB", "RB"}, "throws": {"OL", "DL", "EDGE", "LB"},
       "wrestling": {"OL", "DL", "EDGE", "LB"}, "baseball": {"QB", "WR"}}


def expectations(uni, classes):
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("slot",), test_classes=classes)
        C = pm.rolling(uni, versions=("consensus_pre",), test_classes=tuple(c for c in classes if c in CONS_CLASSES))
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R = R.merge(C[["player_id", "cls"] + [f"consensus_pre_p{s}" for s in pm.SEASONS]], on=["player_id", "cls"], how="left")
    R = R.merge(uni[["player_id", "cls", "log_rank", "on_board", "age_at_draft", "pfr_id"]], on=["player_id", "cls"], how="left")
    R["actual"] = R[[f"y{s}" for s in pm.SEASONS]].sum(axis=1)
    R["exp_slot"] = R[[f"slot_p{s}" for s in pm.SEASONS]].sum(axis=1)
    R["exp_cons"] = R[[f"consensus_pre_p{s}" for s in pm.SEASONS]].sum(axis=1)
    return R


def compare(d, flag, e, covs=(), label="", min_n=8):
    d = d.dropna(subset=[e]).copy()
    f, o = d[d[flag]], d[~d[flag]]
    if len(f) < min_n:
        print(f"    {label:38s} n={len(f):3d}  (too few)")
        return
    d["beat"] = (d.actual > d[e]).astype(int)
    d["lexp"] = np.log(d[e].clip(lower=1e-4))
    d["flag"] = d[flag].astype(float)
    X = sm.add_constant(d[["lexp", "flag", *covs]].astype(float))
    lg = sm.Logit(d.beat, X).fit(disp=0, cov_type="HC1")
    g = sm.GLM(d.actual, sm.add_constant(d[["flag", *covs]].astype(float)), family=sm.families.Poisson(),
               offset=d.lexp).fit(cov_type="HC1")
    r = lambda x: x.actual.sum() / x[e].sum()
    rng = np.random.default_rng(0)
    bs = [r(f.sample(len(f), replace=True, random_state=int(rng.integers(1e9)))) /
          r(o.sample(len(o), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(1000)]
    print(f"    {label:38s} n={len(f):3d} vs {len(o):4d} | beat {(f.actual > f[e]).mean():4.0%} vs {(o.actual > o[e]).mean():4.0%} "
          f"(odds x{np.exp(lg.params['flag']):.2f}, p {lg.pvalues['flag']:.2f}) | value x{r(f) / r(o):.2f} "
          f"(95% {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}; Poisson x{np.exp(g.params['flag']):.2f}, p {g.pvalues['flag']:.2f})")


def mlb_part(uni, R):
    flags = ms.mlb_flags(uni)
    R = R.merge(flags, on="player_id", how="left")
    R["mlb"] = R.mlb_drafted.fillna(False).astype(bool)
    R["mlb_top10"] = R.mlb & (R.mlb_best_round <= 10)
    print("\nMLB-DRAFTED: drafted players vs slot 2016-2022, board players vs consensus 2018-2022 (4 NFL seasons each)")
    dr = R[R.drafted.astype(bool)]
    bd = R[R.on_board == 1]
    for flag, lab in [("mlb", "any MLB draft"), ("mlb_top10", "MLB rounds 1-10")]:
        print(f"  {lab}")
        compare(dr, flag, "exp_slot", label="drafted, vs draft slot")
        compare(bd, flag, "exp_cons", label="board, vs consensus")
        compare(dr[dr.grp == "QB"], flag, "exp_slot", label="QBs drafted, vs draft slot", min_n=3)
    show = R[R.mlb & (R.drafted.astype(bool) | (R.on_board == 1))].sort_values(["cls", "pick"])
    print(show[["cls", "player_name", "grp", "hasan_rank", "pick", "mlb_best_round", "mlb_debut", "exp_slot", "exp_cons", "actual"]]
          .assign(**{c: lambda x, c=c: (x[c] * 100).round(1) for c in ["exp_slot", "exp_cons", "actual"]}).to_string(index=False))
    return R


def text_part(uni):
    from beast_test import match
    R = expectations(uni, (2019, 2020, 2021, 2022))
    beast = pd.read_parquet(OUT / "beast_profiles.parquet")
    beast = beast[beast.year <= 2022].reset_index(drop=True)
    f = ms.text_sports(beast)
    b = pd.concat([beast, f], axis=1)
    d = match(b, R)
    d = d[d.player_id.notna()].drop_duplicates("player_id").reset_index(drop=True)
    d["drafted"] = d.drafted.astype(bool)
    d["log_bg"] = np.log(d.background.fillna("").str.len().clip(lower=100))
    print(f"\nOTHER SPORTS IN THE GUIDES' BACKGROUND TEXT, 2019-2022: {len(beast)} profiles, {len(d)} matched; "
          f"drafted {d.drafted.sum()}, board {int((d.on_board == 1).sum())}")
    print("  share flagged: " + ", ".join(f"{k} {d[k].mean():.0%}" for k in list(ms.SPORTS) + ["any_other"]))
    for k in list(ms.SPORTS) + ["any_other"]:
        d[k] = d[k].astype(bool)
        print(f"  {k}")
        compare(d[d.drafted], k, "exp_slot", covs=("log_bg",), label="drafted, vs draft slot")
        compare(d[d.on_board == 1], k, "exp_cons", covs=("log_bg",), label="board, vs consensus")
        if k in POS:
            pos = d.grp.isin(POS[k])
            compare(d[d.drafted & pos], k, "exp_slot", covs=("log_bg",), label=f"  {'/'.join(sorted(POS[k]))} drafted, vs slot")
            compare(d[(d.on_board == 1) & pos], k, "exp_cons", covs=("log_bg",), label=f"  {'/'.join(sorted(POS[k]))} board, vs consensus")
    d.drop(columns=[c for c in d.columns if d[c].dtype == object and c not in ("name", "player_name", "school", "grp")]) \
        .to_parquet(OUT / "multisport_text.parquet")


def main():
    uni = pm.load()
    which = sys.argv[1:] or ["mlb", "text"]
    if "mlb" in which:
        R = expectations(uni, SLOT_CLASSES)
        mlb_part(uni, R).drop(columns=["mlb_pos"]).to_parquet(OUT / "multisport_mlb.parquet")
    if "text" in which:
        text_part(uni)


if __name__ == "__main__":
    main()
