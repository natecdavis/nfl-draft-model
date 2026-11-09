"""Does the talent-model label (scripts/darko_label.py) track outside yardsticks better than the narrow label?

Drafted players, classes 2011-2021. Label value = value_y14 (years 1-4, % of cap). Yardsticks:
  second  second-contract pay: max APY % of cap over contracts signed draft_year+2 .. draft_year+5 (0 if none);
          the market's verdict after seeing the player, not built from PFF
  board   consensus board rank (hasan_rank, lower = better), pre-draft signal of talent
  pick    overall pick (contaminated: playing time follows the investment; shown for reference)
All variables are percentile-ranked within (pricing group, class); correlation pooled. 95% CIs and the
difference darko - narrow from a player bootstrap (1,000 draws).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/derived"
LABELS = {"stored": Path.home() / "cfdarko/data/processed/rookie_value_labels.parquet",
          "narrow": D / "labels_pickadj_narrow.parquet",
          "darko": D / "labels_pickadj_narrow_darko.parquet"}
if (D / "labels_pickadj_narrow_darko_ctx.parquet").exists():
    LABELS["darko_ctx"] = D / "labels_pickadj_narrow_darko_ctx.parquet"
CLASSES = (2011, 2021)


def main():
    base = pd.read_parquet(LABELS["narrow"])
    base = base[base.drafted & base.draft_year.between(*CLASSES)][["pfr_id", "gsis", "draft_year", "overall_pick", "group"]]
    for k, f in LABELS.items():
        base = base.merge(pd.read_parquet(f)[["pfr_id", "value_y14"]].rename(columns={"value_y14": k}), on="pfr_id",
                          how="left")
    c = pd.read_parquet(ROOT / "data/raw/contracts.parquet")[["gsis_id", "year_signed", "apy_cap_pct"]].dropna()
    b = base.merge(c, left_on="gsis", right_on="gsis_id", how="left")
    w = b.year_signed.between(b.draft_year + 2, b.draft_year + 5)
    second = b[w].groupby("pfr_id").apy_cap_pct.max()
    base["second"] = base.pfr_id.map(second).fillna(0.0)
    u = pd.read_parquet(D / "pm_universe.parquet")[["pfr_id", "hasan_rank"]].dropna().drop_duplicates("pfr_id")
    base["board"] = -base.pfr_id.map(u.set_index("pfr_id").hasan_rank)
    base["pick"] = -base.overall_pick
    base = base.dropna(subset=list(LABELS)).copy()
    for col in list(LABELS) + ["second", "board", "pick"]:
        base[col] = base.groupby(["group", "draft_year"])[col].rank(pct=True)
    print(f"drafted players {CLASSES[0]}-{CLASSES[1]}: {len(base)}; with second contract "
          f"{(base.pfr_id.map(second).notna()).sum()}; with board rank {base.board.notna().sum()}")
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(base), len(base)) for _ in range(1000)]
    rows = []
    for y in ["second", "board", "pick"]:
        d = base.dropna(subset=[y])
        idx = [rng.integers(0, len(d), len(d)) for _ in range(1000)] if len(d) != len(base) else B
        corr = lambda dd, k: np.corrcoef(dd[k], dd[y])[0, 1]
        row = {"yardstick": y, "n": len(d)}
        boots = {k: np.array([corr(d.iloc[i], k) for i in idx]) for k in LABELS}
        for k in LABELS:
            row[k] = f"{corr(d, k):.3f}"
        diff = boots["darko"] - boots["narrow"]
        row["darko-narrow"] = f"{corr(d, 'darko') - corr(d, 'narrow'):+.3f} ({np.percentile(diff, 2.5):+.3f} to {np.percentile(diff, 97.5):+.3f})"
        rows.append(row)
    pd.set_option("display.width", 200)
    print("\ncorrelation of label value (years 1-4) with yardstick, within group x class percentiles:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nby group (second-contract pay):")
    g = base.groupby("group").apply(lambda d: pd.Series({k: np.corrcoef(d[k], d.second)[0, 1] for k in LABELS} | {"n": len(d)}))
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
