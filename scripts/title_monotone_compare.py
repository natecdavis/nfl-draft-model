"""Review before (TITLE_MONOTONE=off, data/derived/before_title_monotone) vs after (min, data/derived/title_monotone_min).
README "Titles non-increasing in rank". Run: python scripts/title_monotone_compare.py"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from review import once_per_draft
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
A = pd.read_parquet("data/derived/before_title_monotone/review_picks.parquet")
B = pd.read_parquet("data/derived/title_monotone_min/review_picks.parquet")
rng = np.random.default_rng(0)
def boot(x):
    x = np.asarray(x, float)
    if len(x) == 0: return "n/a"
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f}) n={len(x)}"
def card(d, label):
    d = d[d.real_team.notna()].copy(); d["d_t"] = d.real_model - d.real_team
    o = once_per_draft(d)
    print(" ", label)
    for gl in ["R1", "R2-4", "R5-7"]:
        x = o[o.band == gl]
        print(f"    {gl:5s} titles {boot(x.d_t)} | QB recs {(x.rec_grp == 'QB').mean():.0%} | QB recs only {boot(x[x.rec_grp == 'QB'].d_t)} | non-QB {boot(x[x.rec_grp != 'QB'].d_t)}")
    print(f"    all   titles {boot(o.d_t)}")
card(A, "before (off)"); card(B, "after (min)")
m = A.merge(B, on=["year", "pick"], suffixes=("_a", "_b"))
ch = m[m.rec_player_a != m.rec_player_b]
print(f"\nrecs changed: {len(ch)} of {len(m)}; by round {ch.round_a.value_counts().sort_index().to_dict()}; from QB {int((ch.rec_grp_a=='QB').sum())}, to QB {int((ch.rec_grp_b=='QB').sum())}")
cm = ch[ch.real_model_a.notna() & ch.real_model_b.notna()]
print("changed recs with outcomes: realized new - old:", boot(cm.real_model_b - cm.real_model_a))
print("  same, each new/old pair once per draft:", boot(cm.drop_duplicates(["year", "rec_player_a", "rec_player_b"]).pipe(lambda d: d.real_model_b - d.real_model_a)))
q = ch[(ch.rec_grp_a == "QB") | (ch.rec_grp_b == "QB")]
print(q[["year", "pick", "team_a", "player_a", "rec_player_a", "rec_rank_a", "value_rec_a", "real_model_a", "rec_player_b", "rec_rank_b", "value_rec_b", "real_model_b", "real_team_a"]].round(2).to_string(index=False))
print("\nvalue_actual changed:", int((~np.isclose(m.value_actual_a, m.value_actual_b, equal_nan=True, atol=1e-6)).sum()), "mean change", float((m.value_actual_b - m.value_actual_a).mean()))
print("QB recs:", int((A.rec_grp == "QB").sum()), "->", int((B.rec_grp == "QB").sum()), "; R1:", int(((A.rec_grp == "QB") & (A["round"] == 1)).sum()), "->", int(((B.rec_grp == "QB") & (B["round"] == 1)).sum()))
print(B[(B.year == 2020) & (B.pick.isin([1, 5, 6]))][["pick", "team", "player", "value_actual", "rec_player", "value_rec"]].round(2).to_string(index=False))
