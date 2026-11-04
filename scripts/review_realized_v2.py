"""Recompute realized values on saved review results with cfdarko season labels."""
import sys; sys.path.insert(0, "src")
import pandas as pd
from review import add_realized_v2
from trade import Context
p = pd.read_parquet("data/derived/review_picks.parquet")
s = pd.read_parquet("data/derived/review_states.parquet")
p = add_realized_v2(p, s, Context())
p.to_parquet("data/derived/review_picks.parquet")
print("realized values recomputed for", p.real_team.notna().sum(), "picks")
