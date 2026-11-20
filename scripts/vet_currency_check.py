"""Which currency turns veterans into team strength at the bridge's rates? (free agency step 1 check)

Team-season (2013-25) rating ~ roster value, QB vs non-QB, season means removed, clustered by team (as
scripts/age_pricing.fit). Three currencies:
  curve  season value from the pay curve (scripts/vet_season_values.py; snap share x price(share, quality) x games),
         this project's units - what the draft model credits a pick's seasons with
  apy    snap share x APY of the contract in force (the bridge's pre-season currency; rookie deals at slot price)
  vet split: curve value of veterans (contract signed after the rookie deal) vs rookie-deal players.
If curve buys ~0.19 / 0.34 per 1% (the pick model's rates), a veteran's forecast value can use the same rates.
Run: .venv/bin/python scripts/vet_currency_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["scripts", "src"]
import age_pricing as ap  # noqa: E402

ps = ap.player_seasons()  # pfr_player_id, season, team, share, grp, price (% cap), value (share x price), vet
v = pd.read_parquet("data/derived/vet_season_values.parquet")
v = v.sort_values("value_ours").drop_duplicates(["pfr_id", "season"], keep="last")
# split a player's curve value across his teams by snap share that season
tot = ps.groupby(["pfr_player_id", "season"]).share.transform("sum")
ps["w"] = np.where(tot > 0, ps.share / tot, 0)
ps = ps.merge(v[["pfr_id", "season", "value_ours"]].rename(columns={"pfr_id": "pfr_player_id"}),
              on=["pfr_player_id", "season"], how="left")
ps["curve"] = ps.value_ours.fillna(0) * ps.w * 100
ps["qb"] = np.where(ps.grp == "qb", "qb", "nonqb")
ps["qv"] = ps.qb + np.where(ps.vet, "_vet", "_rookie")
rat = ap.ratings()
print("linked curve value share of snaps:", round((ps.value_ours.notna() * ps.share).sum() / ps.share.sum(), 3))
ap.fit(ps, rat, "value", "qb", "bridge currency: share x APY in force")
ap.fit(ps, rat, "curve", "qb", "curve currency: pay-curve season value")
ap.fit(ps, rat, "curve", "qv", "curve currency, veterans vs rookie deals", test=["nonqb_vet", "nonqb_rookie"])
v_ = ps[ps.vet & (ps.share > 0)]
print("\nveterans with snaps: sum curve / sum (share x APY) =", round(v_.curve.sum() / v_.value.sum(), 3),
      "| QB", round(v_[v_.grp == "qb"].curve.sum() / v_[v_.grp == "qb"].value.sum(), 3))
