"""Revealed team opinion: (A) draft-night trade-ups, (B) undrafted free-agent guarantees.

A. Does a player a team traded up to take beat his draft slot? (conviction = private information)
   Universe: drafted players 2006-2022 (4 NFL seasons). Expected value from draft slot, out of sample:
   2006-15 leave-one-class-out slot model (old_class_replication.expected_slot), 2016-22 the rolling slot model
   (multisport_test.expectations); each class rescaled so actual = expected (class fixed effect).
   Trade-up pick: the team that made pick k got it in a draft-night trade (review.draft_night) and, in that trade,
   gave no pick of this draft earlier than k (it moved up, paying with later picks / future picks).
   Specified before looking:
     A1 primary: trade-up picks vs all other picks, value vs slot
     A2 round 1 / A3 picks 33+ / A4 non-QB
     A5 intensity among trade-ups: value vs slot per unit log premium (chart value paid / received; Massey-Thaler
        market curve exp(-0.146 (pick-1)^0.698); a future pick = the middle of its round, one round later)
     A6 2018-22 vs the consensus-board model (does it add beyond the board, not just slot?)
B. Do undrafted free agents with more guaranteed money do better, beyond what the board said?
   UDFA contract = OverTheCap contract with no draft round, signed in the player's draft year (draft_year ==
   year_signed), first such contract. Matched to the prospect universe by normalized name + class (unique only).
   Outcome: value seasons 1-4 (0 if he never played). Classes 2018-2022.
   Specified before looking:
     B1 primary: all matched UDFA signers, Poisson value ~ z(log1p guarantee) + position group + class + on board
        + log board rank (0 off board)
     B2 board-ranked UDFA signers vs the consensus-board model (offset), per sd
     B3 top quartile of guarantee vs the rest (B1 controls)
     B4 (added after B1-B3, before looking at it) all matched signers vs the public pre-draft model (full_pre:
        board + college stats + combine + recruiting, no draft position), per sd, position + class effects
   Caveat: guaranteed money can itself buy roster time (sunk cost), so B is an upper bound on information.
Run: python scripts/conviction_test.py
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
import review as rv
from multisport_test import SLOT_CLASSES, compare, expectations
from old_class_replication import expected_slot
from public_board import _norm

rv.DRAFT_START.update({2006: "2006-04-29", 2007: "2007-04-28", 2008: "2008-04-26", 2009: "2009-04-25",
                       2010: "2010-04-22", 2011: "2011-04-28", 2012: "2012-04-26", 2013: "2013-04-25"})
mt = lambda k: np.exp(-0.146 * (np.asarray(k, float) - 1) ** 0.698)


def trade_ups(years):
    rows = []
    for y in years:
        d, night = rv.draft_night(y)
        for k, tm in zip(d.pick[d.traded_on_night], d.team[d.traded_on_night]):
            got = night[(night.received == tm) & (night.pick_season == y) & (night.pick_number == k)]
            if got.empty:
                continue
            tid = got.trade_id.iloc[-1]
            tr = night[night.trade_id == tid]
            paid, recv = tr[tr.gave == tm], tr[tr.received == tm]
            now_paid = paid[(paid.pick_season == y) & paid.pick_number.notna()].pick_number
            up = not (now_paid < k).any()
            val = lambda g: sum(mt(r.pick_number) if r.pick_season == y and pd.notna(r.pick_number)
                                else mt(32 * (r.pick_round + 1) - 16) if pd.notna(r.pick_round) else np.nan
                                for r in g.itertuples())
            vp, vr = val(paid), val(recv)
            rows.append({"cls": y, "pick": int(k), "up": up, "premium": np.log(vp / vr) if vp > 0 and vr > 0 else np.nan})
    return pd.DataFrame(rows)


def part_a():
    old = expected_slot()[["pfr_id", "cls", "pick", "grp", "actual", "exp_slot"]]
    uni = pm.load()
    R = expectations(uni, SLOT_CLASSES)
    R = R[R.drafted.astype(bool) & R.exp_slot.notna()]
    new = R.rename(columns={"position_group": "grp_"})[["pfr_id", "cls", "actual", "exp_slot", "exp_cons"]]
    dp = pd.read_parquet("data/raw/draft_picks.parquet")[["season", "pick", "pfr_player_id", "position"]]
    new = new.merge(dp, left_on=["pfr_id", "cls"], right_on=["pfr_player_id", "season"], how="inner")
    new["grp"] = new.position
    d = pd.concat([old.assign(exp_cons=np.nan), new[["pfr_id", "cls", "pick", "grp", "actual", "exp_slot", "exp_cons"]]])
    d["exp_slot"] *= d.groupby("cls").actual.transform("sum") / d.groupby("cls").exp_slot.transform("sum")
    tu = trade_ups(range(2006, 2023))
    d = d.merge(tu, on=["cls", "pick"], how="left")
    d["up"] = d.up.fillna(False).astype(bool)
    d["qb"] = d.grp.astype(str).str.upper().eq("QB")
    print(f"\nA. TRADE-UPS: drafted 2006-22 with outcomes {len(d)}; draft-night acquired picks {tu.shape[0]}, "
          f"trade-ups {int(tu.up.sum())} (matched {int(d.up.sum())}; QB {int((d.up & d.qb).sum())}, round 1 {int((d.up & (d.pick <= 32)).sum())})")
    compare(d, "up", "exp_slot", label="A1 trade-up picks vs other picks")
    compare(d[d.pick <= 32], "up", "exp_slot", label="A2 round 1")
    compare(d[d.pick > 32], "up", "exp_slot", label="A3 picks 33+")
    compare(d[~d.qb], "up", "exp_slot", label="A4 non-QB")
    compare(d[d.cls <= 2015], "up", "exp_slot", label="  2006-15 only")
    compare(d[d.cls >= 2016], "up", "exp_slot", label="  2016-22 only")
    u = d[d.up & d.premium.notna()].copy()
    u["lexp"] = np.log(u.exp_slot.clip(lower=1e-4))
    g = sm.GLM(u.actual, sm.add_constant(u[["premium"]]), family=sm.families.Poisson(), offset=u.lexp).fit(cov_type="HC1")
    lo, hi = np.exp(g.conf_int().loc["premium"]) ** u.premium.std()
    print(f"    A5 intensity: {len(u)} trade-ups, median premium x{np.exp(u.premium.median()):.2f} of chart; "
          f"value per sd of log premium x{np.exp(g.params.premium * u.premium.std()):.2f} (95% {lo:.2f}-{hi:.2f}, p {g.pvalues.premium:.2f})")
    c = d[d.exp_cons.notna()].copy()
    c["exp_cons"] *= c.groupby("cls").actual.transform("sum") / c.groupby("cls").exp_cons.transform("sum")
    compare(c, "up", "exp_cons", label="A6 2018-22 vs consensus board")
    compare(c, "up", "exp_slot", label="  same players vs slot")
    d.to_parquet(pm.OUT / "conviction_tradeups.parquet")


def part_b():
    uni = pm.load()
    u = uni[uni.cls.between(2018, 2022) & ~uni.drafted.astype(bool)].copy()
    for t in pm.SEASONS:
        u[f"y{t}"] = u[f"value_y{t}"].fillna(0.0).clip(lower=0)
    u["actual"] = u[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    u["k"] = u.player_name.map(_norm)
    u = u[~u.duplicated(["k", "cls"], keep=False)]
    c = pd.read_parquet("data/raw/contracts.parquet")
    c = c[c.draft_round.isna() & (c.draft_year == c.year_signed) & c.year_signed.between(2018, 2022)].copy()
    c = c.sort_values(["player", "year_signed", "guaranteed"], ascending=[True, True, False])
    c["k"] = c.player.map(_norm)
    c = c.drop_duplicates(["k", "year_signed"])
    c["gtd_k"] = c.guaranteed.fillna(0) * 1000
    m = u.merge(c[["k", "year_signed", "gtd_k", "team"]], left_on=["k", "cls"], right_on=["k", "year_signed"], how="inner")
    print(f"\nB. UDFA GUARANTEES 2018-22: UDFA contracts {len(c)}; matched to the universe {len(m)} "
          f"(on board {int(m.hasan_rank.notna().sum())}); guarantee > 0: {(m.gtd_k > 0).mean():.0%}, "
          f"median of those ${m.gtd_k[m.gtd_k > 0].median():.0f}k, 90th pct ${m.gtd_k.quantile(0.9):.0f}k")
    m["lg"] = np.log1p(m.gtd_k)
    m["z"] = (m.lg - m.lg.mean()) / m.lg.std()
    m["on_board"] = m.hasan_rank.notna().astype(float)
    m["lrank"] = np.where(m.hasan_rank.notna(), np.log(m.hasan_rank.fillna(1)), 0.0)
    m["grp"] = m.position_group.astype(str)
    m["top"] = (m.gtd_k >= m.gtd_k.quantile(0.75)).astype(float)
    for var, lab in [("z", "B1 value per sd of log guarantee"), ("top", "B3 top quartile vs rest")]:
        f = smf.glm(f"actual ~ {var} + C(grp) + C(cls) + on_board + lrank", m, family=sm.families.Poisson()).fit(cov_type="HC1")
        lo, hi = np.exp(f.conf_int().loc[var])
        print(f"    {lab:40s} n={len(m)}  x{np.exp(f.params[var]):.2f} (95% {lo:.2f}-{hi:.2f}, p {f.pvalues[var]:.3f})")
    print(f"    mean value: guarantee 0 {m.actual[m.gtd_k == 0].mean():.3f}, top quartile {m.actual[m.top == 1].mean():.3f}; "
          f"played at all: {(m.actual[m.gtd_k == 0] > 0).mean():.0%} vs {(m.actual[m.top == 1] > 0).mean():.0%}")
    R = expectations(uni, tuple(range(2018, 2023)))
    b = m.merge(R[["player_id", "exp_cons"]], on="player_id", how="inner").dropna(subset=["exp_cons"])
    b = b[b.on_board == 1]
    if len(b) > 30:
        b["lexp"] = np.log(b.exp_cons.clip(lower=1e-4))
        f = sm.GLM(b.actual, sm.add_constant(b[["z"]]), family=sm.families.Poisson(), offset=b.lexp).fit(cov_type="HC1")
        lo, hi = np.exp(f.conf_int().loc["z"])
        print(f"    {'B2 board UDFAs vs consensus model, per sd':40s} n={len(b)}  x{np.exp(f.params.z):.2f} (95% {lo:.2f}-{hi:.2f}, p {f.pvalues.z:.3f}); "
              f"actual / consensus overall x{b.actual.sum() / b.exp_cons.sum():.2f}")
    saved = dict(pm.PARAMS); pm.PARAMS.update(__import__("multisport_test").REG)
    try:
        F = pm.rolling(uni, versions=("full_pre",), test_classes=tuple(range(2018, 2023)))
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    F["exp_pub"] = F[[f"full_pre_p{t}" for t in pm.SEASONS]].sum(axis=1)
    q = m.merge(F[["player_id", "exp_pub"]], on="player_id", how="inner").dropna(subset=["exp_pub"])
    q["lexp"] = np.log(q.exp_pub.clip(lower=1e-4))
    f = smf.glm("actual ~ z + C(grp) + C(cls)", q, family=sm.families.Poisson(), offset=q.lexp).fit(cov_type="HC1")
    lo, hi = np.exp(f.conf_int().loc["z"])
    print(f"    {'B4 all signers vs public pre-draft model':40s} n={len(q)}  x{np.exp(f.params.z):.2f} (95% {lo:.2f}-{hi:.2f}, p {f.pvalues.z:.3f}); "
          f"actual / public model overall x{q.actual.sum() / q.exp_pub.sum():.2f}")
    for nm, g in [("on board", q[q.on_board == 1]), ("off board", q[q.on_board == 0])]:
        f = smf.glm("actual ~ z + C(grp) + C(cls)", g, family=sm.families.Poisson(), offset=g.lexp).fit(cov_type="HC1")
        print(f"      {nm:38s} n={len(g)}  x{np.exp(f.params.z):.2f} (p {f.pvalues.z:.3f})")
    m.to_parquet(pm.OUT / "conviction_udfa.parquet")


if __name__ == "__main__":
    if sys.argv[1:] != ["b"]:
        part_a()
    part_b()
