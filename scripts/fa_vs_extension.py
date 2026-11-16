"""Lead #4 of the contract-efficiency list: do outside free-agent signings buy fewer points per dollar than
re-signings / extensions with the player's own team? (winner's curse: the own team knows more)

Same bridge-style team test as scripts/age_pricing.py (416 team-seasons 2013-25, pre-season prices, snap share x
price). Each veteran's current contract (latest signed by that season) is classed by where he played (most snaps):
  own  : same team in the season before signing and in the signing season (re-signed / extended / tagged)
  new  : different team in the signing season than the season before (outside free agent, or traded and re-done)
  other: no snaps the season before signing (out of the league, injured, rookie-deal edge cases)
Pre-specified test: equal slopes own vs new for non-QB veterans; the lead predicts new < own. Splits: by APY
(>= 5% of cap = the big auctions, where a winner's curse should be strongest) and QB. Also early exits: contracts
signed 2013-21, unplayed guaranteed seasons by class (scripts/age_pricing_tenure.py proxy).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import age_pricing as ap  # noqa: E402


def main():
    ps = ap.player_seasons()
    main_team = (ps.sort_values("share").groupby(["pfr_player_id", "season"]).team.last())
    prev = [main_team.get((p, int(s) - 1)) if pd.notna(s) else None for p, s in zip(ps.pfr_player_id, ps.signed)]
    at_sign = [main_team.get((p, int(s))) if pd.notna(s) else None for p, s in zip(ps.pfr_player_id, ps.signed)]
    ps["cls"] = np.select(
        [~ps.vet, pd.isna(pd.Series(prev, index=ps.index)),
         pd.Series(prev, index=ps.index) == pd.Series(at_sign, index=ps.index)],
        ["rookie_deal", "other", "own"], "new")
    ps["big"] = ps.price >= 5
    ps["b"] = np.where(ps.grp == "qb", "qb_" + ps.cls, np.where(ps.cls.isin(["own", "new", "other"]), "vet_" + ps.cls,
                                                                  ps.cls))
    rat = ap.ratings()
    v = ps[ps.vet]
    print("snap-weighted veteran value by class (non-QB):")
    print((v[v.grp != "qb"].groupby("cls").value.sum() / v[v.grp != "qb"].value.sum()).round(3).to_string())
    for money in ["value", "cap"]:
        ap.fit(ps, rat, money, "b", "veteran contracts: own team vs new team", test=["vet_own", "vet_new"])
    ps["b2"] = np.where(ps.b.isin(["vet_own", "vet_new"]), ps.b + np.where(ps.big, "_big", "_small"), ps.b)
    ap.fit(ps, rat, "value", "b2", "by contract size (>= 5% of cap = big)", test=["vet_own_big", "vet_new_big"])
    # first season of the deal vs later (the winner's curse should show immediately)
    ps["b3"] = np.where(ps.b.isin(["vet_own", "vet_new"]),
                        ps.b + np.where(ps.season == ps.signed, "_yr1", "_later"), ps.b)
    ap.fit(ps, rat, "value", "b3", "first season of the deal vs later")

    # early exits by class, contracts signed 2013-21 with APY >= 1%
    first = ps[(ps.season == ps.signed) & ps.vet & ps.signed.between(2013, 2021)].copy()
    c = pd.read_parquet(ap.RAW / "contracts.parquet")
    pl_ = pd.read_parquet(ap.RAW / "players.parquet")[["gsis_id", "pfr_id"]].dropna()
    c = c[c.apy_cap_pct.notna() & ~(c.year_signed == c.draft_year) & (c.apy_cap_pct >= 0.01)].merge(pl_, on="gsis_id")
    c = c.sort_values("apy_cap_pct").drop_duplicates(["pfr_id", "year_signed"], keep="last")
    f = first.sort_values("share").drop_duplicates(["pfr_player_id", "season"], keep="last").merge(
        c[["pfr_id", "year_signed", "years", "guaranteed", "value"]].rename(columns={"value": "tot"}),
        left_on=["pfr_player_id", "signed"], right_on=["pfr_id", "year_signed"])
    played = ps.groupby(["pfr_player_id", "season"]).team.agg(set).to_dict()
    n = []
    for r in f.itertuples():
        yrs = int(min(r.years, 2025 - r.signed + 1))
        n.append((yrs, sum(1 for k in range(yrs) if r.team in played.get((r.pfr_player_id, int(r.signed) + k), set()))))
    f["yrs"], f["played"] = zip(*n)
    f["gtd"] = (f.guaranteed / f.tot).where(f.tot > 0)
    f["dead_proxy"] = (f.gtd * f.yrs - f.played).clip(lower=0) / f.yrs
    f["qb"] = np.where(f.grp == "qb", "QB", "non-QB")
    print("\nContracts signed 2013-21 (APY >= 1% of cap), first season on the deal:")
    print(f.groupby(["qb", "cls"]).agg(n=("yrs", "size"), apy=("price", "mean"), years=("yrs", "mean"),
                                        gtd_share=("gtd", "mean"), frac_played=("played", lambda x: 0),
                                        dead_proxy=("dead_proxy", "mean"))
          .assign(frac_played=f.groupby(["qb", "cls"]).apply(lambda g: (g.played / g.yrs).mean()))
          .round(3).to_string())


if __name__ == "__main__":
    main()
