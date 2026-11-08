"""Position-specific replacement level, priced on the same pay curves as the rookie-value label.

pick_value.pick_effect credits a pick with share x (price - VET_MIN): every position's replacement is
worth 0.5% of cap per full season of snaps. Here we measure what freely available players actually
produce when they play, priced like any other season: cfdarko's veteran pay curve price_g(share, q)
(q = within group-season percentile of the shrunk PFF grade), captured by running cfdarko's
build_label (read only) up to the curve fit.

Replacement pool: player-seasons 2013-2025 on a contract in force paying <= POOL_MULT x that year's
veteran minimum (veteran minimum-salary deals and undrafted rookie deals; drafted rookie deals excluded).
For each group:
  q_repl  = snap-weighted median quality of pool seasons with share >= 0.3
  repl(s) = price_g(s, q_repl) / 1.10   (cfdarko -> this project's units, outcomes_v2 single units)
compared with VET_MIN x s. Then: drafted players 2014-2022, 4-year value over replacement under each rule.

Usage: python scripts/replacement_level.py
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

CF = Path.home() / "cfdarko"
sys.path.insert(0, str(CF))
_nfl = types.ModuleType("nfl_data_py")
_nfl.import_ids = lambda: pd.read_parquet(CF / "data/raw/rookie_value/nflverse_ids.parquet")
sys.modules.setdefault("nfl_data_py", _nfl)
from src.models import rookie_value as rv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VET_MIN, UNITS = 0.005, 1.10
POOL_MULT = 1.25
CAP = {}


class Stop(Exception):
    pass


_curves, _quality = rv.pay_curves_wide, rv.quality_table_pff


def grab_quality(pos):
    q, neutral = _quality(pos)
    CAP.update(q=q, neutral=neutral, pos=pos)
    return q, neutral


def grab_curves(snaps, pfr2gsis, c, qv, qgroups, fold=None, k=5):
    CAP.update(curves=_curves(snaps, pfr2gsis, c, qv, qgroups, fold, k), pfr2gsis=pfr2gsis, qv=qv)
    raise Stop


def main():
    rv.quality_table_pff, rv.pay_curves_wide = grab_quality, grab_curves
    try:
        rv.build_label(out_path=ROOT / "data/derived/_unused.parquet")
    except Stop:
        pass
    curves, qv, p2g, pos = CAP["curves"], CAP["qv"], CAP["pfr2gsis"], CAP["pos"]

    s = rv.load_snaps()
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["gfrac"] = (s["n_games"] / s["season"].map(rv.reg_season_games)).clip(upper=1.0)
    s = (s.sort_values("share").groupby(["pfr_player_id", "season"])
         .agg(share=("share", "last"), gfrac=("gfrac", "last")).reset_index())
    s = s[s.share > 0]
    s["gsis"] = s.pfr_player_id.map(p2g)
    s["g"] = s.gsis.map(pos)
    s = s.dropna(subset=["gsis", "g"])
    s = s[s.g.isin(curves)]
    s["q"] = [qv(a, g, y) for a, g, y in zip(s.gsis, s.g, s.season)]

    # contract in force each season
    c = pd.read_parquet(rv.RAW_DIR / "nfl_contracts.parquet")
    for col in ("year_signed", "years", "apy", "draft_year", "draft_round", "implied_cap"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    cap = rv.cap_by_year(c)
    c["pct"] = c.apy / c.year_signed.map(cap)
    c = c.dropna(subset=["gsis_id", "year_signed", "years", "pct"])
    rl = pd.read_parquet(rv.PROCESSED_DIR / "replacement_levels.parquet")
    vmin = rl[rl.scope == "global_year"].assign(y=lambda d: d.key.astype(int)).set_index("y").base_apy_pct
    c["vmin"] = c.year_signed.map(vmin)
    c["rookie_drafted"] = (c.year_signed == c.draft_year) & c.draft_round.notna()
    m = s.merge(c[["gsis_id", "year_signed", "years", "pct", "vmin", "rookie_drafted", "draft_round", "draft_year"]],
                left_on="gsis", right_on="gsis_id")
    m = m[(m.year_signed <= m.season) & (m.season < m.year_signed + m.years)]
    m = m.sort_values("year_signed").drop_duplicates(["gsis", "season"], keep="last")
    pool = m[~m.rookie_drafted & (m.pct <= POOL_MULT * m.vmin)].copy()
    pool["kind"] = np.where(pool.year_signed == pool.draft_year, "UDFA rookie", "vet min")
    print(f"\nPool: {len(pool)} player-seasons 2013-2025 ({pool.kind.value_counts().to_dict()}); "
          f"all linked seasons {len(m)}")

    rows = []
    for g, G in pool.groupby("g"):
        P = G[G.share >= 0.3]
        w = P.share * P.gfrac
        order = np.argsort(P.q.to_numpy())
        cw = np.cumsum(w.to_numpy()[order]) / w.sum()
        q_repl = float(P.q.to_numpy()[order][np.searchsorted(cw, 0.5)])
        allg = m[m.g == g]
        snaps_from_pool = (G.share * G.gfrac).sum() / (allg.share * allg.gfrac).sum()
        r = {"group": g, "pool seasons": len(G), "share>=0.3": len(P),
             "pool % of snaps": round(100 * snaps_from_pool, 1), "q_repl": round(q_repl, 2)}
        for sh in (0.3, 0.6, 0.9):
            r[f"repl@{sh}"] = round(100 * curves[g](sh, q_repl) / UNITS, 2)
            r[f"now@{sh}"] = round(100 * VET_MIN * sh, 2)
        # realized: what pool players' seasons were worth per full season of snaps
        val = sum(curves[g](a, b) * f for a, b, f in zip(P.share, P.q, P.gfrac)) / UNITS
        r["pool realized per full season"] = round(100 * val / w.sum(), 2)
        rows.append(r)
        CAP.setdefault("qrepl", {})[g] = q_repl
    T = pd.DataFrame(rows).set_index("group")
    print("\nReplacement value, % of cap (this project's units). repl@s = price of a replacement-quality player "
          "playing share s; now@s = VET_MIN x s (current model):")
    print(T.to_string())

    # effect on drafted players' value over replacement, drafts 2014-2022
    L = pd.read_parquet(ROOT / "data/derived/labels_pickadj_narrow.parquet")
    L = L[L.drafted & L.draft_year.between(2014, 2022)].dropna(subset=["value_y14"])
    sh = s.set_index(["pfr_player_id", "season"])
    out = []
    for r in L.itertuples():
        g = pos.get(r.gsis) if isinstance(r.gsis, str) else None
        if g not in CAP["qrepl"]:
            continue
        old = new = 0.0
        for k in range(4):
            key = (r.pfr_id, r.draft_year + k)
            if key in sh.index:
                a, f = sh.loc[key, ["share", "gfrac"]]
                old += VET_MIN * a * f
                new += curves[g](a, CAP["qrepl"][g]) / UNITS * f
        out.append({"g": g, "round": r.round, "value": r.value_y14 / UNITS, "old": old, "new": new})
    O = pd.DataFrame(out)
    O["vor_old"], O["vor_new"] = O.value - O.old, O.value - O.new
    S = O.groupby("g")[["value", "old", "new", "vor_old", "vor_new"]].mean() * 100
    S["n"] = O.groupby("g").size()
    S["vor change %"] = 100 * (S.vor_new / S.vor_old - 1)
    print("\nDrafted 2014-2022, 4-year value per player (% of cap): gross, replacement charged (old / new), "
          "value over replacement (old / new):")
    print(S.round(2).to_string())
    R = O.assign(rd=np.where(O["round"] == 1, "R1", np.where(O["round"] <= 3, "R2-3", "R4-7")))
    S2 = R.groupby(["rd", "g"])[["vor_old", "vor_new"]].mean().mul(100).unstack(0).round(2)
    print("\nValue over replacement by round group (old / new):")
    print(S2.to_string())
    T.to_csv(ROOT / "data/derived/replacement_levels_by_group.csv")

    # q_repl sensitivity: all pool seasons weighted by snaps, 25th / 50th percentile
    print("\nq_repl sensitivity (pool, all shares, snap-weighted): 25th / 50th pct, and repl@0.9 at each")
    for g, G in pool.groupby("g"):
        w = (G.share * G.gfrac).to_numpy(); o = np.argsort(G.q.to_numpy()); cw = np.cumsum(w[o]) / w.sum()
        qs = [float(G.q.to_numpy()[o][np.searchsorted(cw, x)]) for x in (0.25, 0.5)]
        print(f"   {g:5s} q {qs[0]:.2f} / {qs[1]:.2f}   repl@0.9 {100 * curves[g](0.9, qs[0]) / UNITS:.2f} / "
              f"{100 * curves[g](0.9, qs[1]) / UNITS:.2f}   (curve at q=0.0: {100 * curves[g](0.9, 0.0) / UNITS:.2f})")

    team_check(curves, qv, p2g, pos)

    # points of team strength over 4 seasons, drafted 2014-2022: current rule (0.19 x (value - VET_MIN x share),
    # QB 0.34) vs quality-over-replacement at the same snaps (q_ref = 0.1) x team-check slope
    QREF, RATE_Q = 0.1, 0.48
    out = []
    for r in L.itertuples():
        g = pos.get(r.gsis) if isinstance(r.gsis, str) else None
        if g not in curves:
            continue
        old = new = 0.0
        for k in range(4):
            key = (r.pfr_id, r.draft_year + k)
            if key in sh.index:
                a, f = sh.loc[key, ["share", "gfrac"]]
                old += VET_MIN * a * f
                new += curves[g](a, QREF) / UNITS * f
        out.append({"g": g, "round": r.round, "value": r.value_y14 / UNITS, "old": old, "new": new})
    O = pd.DataFrame(out)
    O["rd"] = np.where(O["round"] == 1, "R1", np.where(O["round"] <= 3, "R2-3", "R4-7"))
    O["pts_old"] = 100 * (O.value - O.old) * np.where(O.g == "QB", 0.34, 0.19)
    O["pts_new"] = 100 * (O.value - O.new) * RATE_Q
    P = O.groupby(["g", "rd"])[["pts_old", "pts_new"]].mean().unstack(1)
    print(f"\nPoints of team strength, 4 seasons summed, per drafted player 2014-22 (old rule / quality over q={QREF} "
          f"x {RATE_Q}):")
    print(P.round(2).to_string())
    tot = O.groupby("rd")[["pts_old", "pts_new"]].mean()
    print(tot.round(2).to_string())
    print("Ratio new/old by position:", (O.groupby("g").pts_new.mean() / O.groupby("g").pts_old.mean()).round(2).to_dict())


def team_check(curves, qv, p2g, pos):
    """Which part of curve-priced value tracks team strength: the playing-time part or the quality part?
    base = sum over players of price_g(share, q_ref) (what replacement-quality players would be credited for
    the same snaps); qual = sum of price_g(share, q) - base. If base buys no points, a pick's credit should be
    measured over a replacement at the same snaps."""
    import statsmodels.formula.api as smf
    sys.path.insert(0, str(ROOT / "scripts"))
    from synergy_test import ratings, RELOC
    sc = pd.read_parquet(ROOT / "data/raw/snap_counts.parquet")
    sc = sc[(sc.game_type == "REG") & sc.season.between(2013, 2025)].copy()
    sc["team"] = sc.team.replace(RELOC)
    games = np.where(sc.season >= 2021, 17, 16)
    sc["g_"] = games
    ps = (sc.groupby(["pfr_player_id", "season", "team"])
          .agg(off=("offense_pct", "sum"), dfn=("defense_pct", "sum"), G=("g_", "first")).reset_index())
    ps["share"] = (ps[["off", "dfn"]].max(axis=1) / ps.G).clip(0, 1)
    ps["gsis"] = ps.pfr_player_id.map(p2g)
    ps["g"] = ps.gsis.map(pos)
    ps = ps[ps.g.isin(curves) & (ps.share > 0)].copy()
    ps["q"] = [qv(a, g, y) for a, g, y in zip(ps.gsis, ps.g, ps.season)]
    ps["gross"] = [curves[g](a, b) for g, a, b in zip(ps.g, ps.share, ps.q)]
    R = ratings()
    for qref in (0.1, 0.3, 0.5):
        ps["base"] = [curves[g](a, qref) for g, a in zip(ps.g, ps.share)]
        T = ps.groupby(["season", "team"])[["gross", "base"]].sum().mul(100 / UNITS).reset_index()
        T["qual"] = T.gross - T.base
        T = T.merge(R, on=["season", "team"])
        for c in ["gross", "base", "qual", "net"]:
            T[c] = T[c] - T.groupby("season")[c].transform("mean")
        m1 = smf.ols("net ~ gross", T).fit(cov_type="cluster", cov_kwds={"groups": T.team})
        m2 = smf.ols("net ~ base + qual", T).fit(cov_type="cluster", cov_kwds={"groups": T.team})
        ci = m2.conf_int()
        print(f"\nTeam check, q_ref={qref} (n={len(T)}; sd gross {T.gross.std():.1f}, base {T.base.std():.1f}, "
              f"qual {T.qual.std():.1f} % cap): gross {m1.params.gross:+.3f} pts per 1% (R2 {m1.rsquared:.2f}); "
              f"base {m2.params.base:+.3f} [{ci.loc['base', 0]:+.3f},{ci.loc['base', 1]:+.3f}], "
              f"qual {m2.params.qual:+.3f} [{ci.loc['qual', 0]:+.3f},{ci.loc['qual', 1]:+.3f}] (R2 {m2.rsquared:.2f})")


if __name__ == "__main__":
    main()
