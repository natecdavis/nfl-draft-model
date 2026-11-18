"""Lead #3 follow-up: is age overpay hidden in cut players (dead money), which the team test can't see?

Veteran contracts signed 2013-2021 (non-rookie deals, APY >= 1% of cap): age at signing, contracted years,
guaranteed share of total value, and how many of the contracted seasons the player actually logged snaps for the
signing team (snap_counts team match, by season). Cost proxy of early exits = guaranteed money for seasons not
played, as a share of APY x years, assuming guarantees are paid evenly (rough upper-bound-ish proxy).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
sys.path.insert(0, str(ROOT / "src"))
from simulate import RELOCATED  # noqa: E402

c = pd.read_parquet(RAW / "contracts.parquet")
c = c[c.apy_cap_pct.notna() & ~(c.year_signed == c.draft_year) & c.year_signed.between(2013, 2021)
      & (c.apy_cap_pct >= 0.01) & (c.years >= 1)]
pl_ = pd.read_parquet(RAW / "players.parquet")[["gsis_id", "pfr_id", "birth_date", "position"]].dropna(subset=["gsis_id"])
c = c.merge(pl_.drop(columns="position"), on="gsis_id", how="left")
c["age"] = (pd.to_datetime(c.year_signed.astype(int).astype(str) + "-09-01") - pd.to_datetime(c.birth_date)).dt.days / 365.25
s = pd.read_parquet(RAW / "snap_counts.parquet")
s = s[s.game_type == "REG"]
s["team"] = s.team.replace(RELOCATED)
played = s.groupby(["pfr_player_id", "season"]).team.agg(lambda x: set(x)).to_dict()
# OTC team names -> abbreviations via a snap team (first alphabetically if several) in the signing season or the next
rows = []
for r in c.itertuples():
    if pd.isna(r.pfr_id) or pd.isna(r.age):
        continue
    yrs = int(min(r.years, 2025 - r.year_signed + 1))
    first = played.get((r.pfr_id, int(r.year_signed)))
    if not first:
        first = played.get((r.pfr_id, int(r.year_signed) + 1))
    if not first:
        rows.append((r.age, r.position, r.apy_cap_pct, yrs, r.guaranteed / r.value if r.value else np.nan, 0))
        continue
    tm = sorted(first)[0]  # deterministic (set order changes between runs)
    n = sum(1 for k in range(yrs) if tm in played.get((r.pfr_id, int(r.year_signed) + k), set()))
    rows.append((r.age, r.position, r.apy_cap_pct, yrs, r.guaranteed / r.value if r.value else np.nan, n))
T = pd.DataFrame(rows, columns=["age", "pos", "apy", "years", "gtd_share", "played"])
T["frac"] = T.played / T.years
T["dead_proxy"] = (T.gtd_share * T.years - T.played).clip(lower=0) / T.years  # unplayed guaranteed seasons, share
T["qb"] = np.where(T.pos == "QB", "QB", "non-QB")
T["ageb"] = pd.cut(T.age, [0, 27, 30, 33, 50], right=False, labels=["<=26", "27-29", "30-32", "33+"])
pd.set_option("display.width", 200)
print(T.groupby(["qb", "ageb"], observed=True).agg(
    n=("age", "size"), apy=("apy", "mean"), years=("years", "mean"), gtd_share=("gtd_share", "mean"),
    frac_seasons_played=("frac", "mean"), dead_proxy=("dead_proxy", "mean")).round(3).to_string())
