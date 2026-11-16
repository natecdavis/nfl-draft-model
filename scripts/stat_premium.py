"""Lead #2 of the contract-efficiency list: does the market pay for counting stats beyond quality, and do those
dollars buy team strength?

Step 1 (pay at signing). Veteran contracts (non-rookie deals) signed 2014-2025 with a prior season of 30%+ snaps
and a label quality q (cfdarko quality, via scripts/replacement_units.capture). Per pricing group:
  APY (% of cap) ~ q + q^2 + share + age + age^2  [+ counting stats, prior season, per 17 games, standardized]
Stats: EDGE / DL sacks, QB hits; LB tackles, sacks; DB interceptions, passes defended; WR / TE receiving yards, TDs;
RB rushing yards, rushing TDs, receiving yards. QB passing yards, TDs. OL has none (control group).
Stat premium = APY per SD of the stat holding quality fixed.
Step 1b (is it information?). Next season's q on this season's q and stats: do stats predict future quality?
Step 2 (do the dollars buy points?). Each contract's APY = quality part (fitted without stats... the q / share / age
terms of the full model) + stat part (stat terms) + residual. Team-seasons 2014-25: rating on roster sums of snap
share x each part while the contract is in force, plus rookie-deal value and unmodeled veteran value. Pre-specified:
stat-part slope = quality-part slope; the lead predicts the stat part buys less. Franchise fixed-effects check.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_units as ru  # noqa: E402
import age_pricing as ap  # noqa: E402

RAW, OUT = ROOT / "data/raw", ROOT / "data/derived"
STATS = {
    "EDGE": ["def_sacks", "def_qb_hits"], "DL": ["def_sacks", "def_qb_hits"],
    "LB": ["tackles", "def_sacks"], "DB": ["def_interceptions", "def_pass_defended"],
    "WR": ["receiving_yards", "receiving_tds"], "TE": ["receiving_yards", "receiving_tds"],
    "RB": ["rushing_yards", "rushing_tds", "receiving_yards"], "QB": ["passing_yards", "passing_tds"], "OL": [],
}


def load():
    C = ru.capture()
    pos, q = C["pos"], C["q"]
    pfr2gsis = C["pfr2gsis"]
    sc = pd.read_parquet(RAW / "snap_counts.parquet")
    sc = sc[sc.game_type == "REG"].copy()
    sc["G"] = np.where(sc.season >= 2021, 17, 16)
    sh = (sc.groupby(["pfr_player_id", "season"]).agg(off=("offense_pct", "sum"), dfn=("defense_pct", "sum"),
                                                      G=("G", "first")).reset_index())
    sh["share"] = (sh[["off", "dfn"]].max(axis=1) / sh.G).clip(0, 1)
    sh["gsis"] = sh.pfr_player_id.map(pfr2gsis)
    share = sh.dropna(subset=["gsis"]).groupby(["gsis", "season"]).share.max()
    st = pd.read_parquet(RAW / "player_stats.parquet")
    st["tackles"] = st.def_tackles_solo.fillna(0) + st.def_tackle_assists.fillna(0)
    cols = sorted({c for v in STATS.values() for c in v})
    st = st.groupby(["player_id", "season"])[cols].sum()
    G = pd.Series(np.where(st.index.get_level_values(1) >= 2021, 17, 16), index=st.index)
    st = st.mul(17 / G, axis=0)  # per 17 games of season length
    return C, pos, q, share, st


def contracts(pos, q, share, st):
    c = pd.read_parquet(RAW / "contracts.parquet")
    c = c[c.apy_cap_pct.notna() & ~(c.year_signed == c.draft_year) & c.year_signed.between(2014, 2025)
          & c.gsis_id.notna()].copy()
    c = c.sort_values("apy_cap_pct").drop_duplicates(["gsis_id", "year_signed"], keep="last")
    bd = pd.read_parquet(RAW / "players.parquet")[["gsis_id", "birth_date"]].dropna().drop_duplicates("gsis_id")
    c = c.merge(bd, on="gsis_id", how="left")
    c["age"] = (pd.to_datetime(c.year_signed.astype(int).astype(str) + "-09-01")
                - pd.to_datetime(c.birth_date)).dt.days / 365.25
    c["g"] = c.gsis_id.map(pos)
    k = list(zip(c.gsis_id, c.year_signed - 1))
    c["q"] = [q.get(x, np.nan) for x in k]
    c["q_next"] = [q.get((a, y), np.nan) for a, y in zip(c.gsis_id, c.year_signed)]
    c["share"] = [share.get(x, np.nan) for x in k]
    for col in st.columns:
        c[col] = [st[col].get(x, 0.0) for x in k]
    c["apy"] = c.apy_cap_pct * 100
    return c[c.g.notna() & c.q.notna() & (c.share >= 0.3) & c.age.notna()].copy()


def pay_models(c):
    rows, parts = [], []
    for g, stats in STATS.items():
        d = c[c.g == g].copy()
        if len(d) < 60:
            continue
        for s in stats:
            d[f"z_{s}"] = (d[s] - d[s].mean()) / d[s].std()
        zs = [f"z_{s}" for s in stats]
        base = "apy ~ q + I(q**2) + share + age + I(age**2) + C(year_signed)"
        m0 = smf.ols(base, d).fit(cov_type="HC1")
        m1 = smf.ols(base + "".join(f" + {z}" for z in zs), d).fit(cov_type="HC1") if zs else m0
        rec = dict(group=g, n=len(d), mean_apy=d.apy.mean(), r2_quality=m0.rsquared, r2_with_stats=m1.rsquared,
                   q_slope=m1.params["q"])
        for z in zs:
            lo, hi = m1.conf_int().loc[z]
            rec[z[2:]] = f"{m1.params[z]:+.2f} ({lo:+.2f},{hi:+.2f})"
        rows.append(rec)
        # decomposition (centered so parts are deviations; quality part carries the level)
        stat_part = sum(m1.params[z] * d[z] for z in zs) if zs else 0 * d.apy
        d["stat_part"] = stat_part
        d["resid_part"] = m1.resid
        d["qual_part"] = d.apy - d.stat_part - d.resid_part
        # info check: next season quality on q + stats
        dn = d.dropna(subset=["q_next"])
        if zs and len(dn) > 50:
            mi = smf.ols("q_next ~ q + share + age" + "".join(f" + {z}" for z in zs), dn).fit(cov_type="HC1")
            for z in zs:
                rec[f"info_{z[2:]}"] = f"{mi.params[z]:+.3f} (p {mi.pvalues[z]:.2f})"
        parts.append(d[["gsis_id", "year_signed", "g", "apy", "qual_part", "stat_part", "resid_part"]])
    return pd.DataFrame(rows), pd.concat(parts)


def team_test(parts, cap):
    ps = ap.player_seasons()
    ps["gsis"] = ps.pfr_player_id.map(cap["pfr2gsis"])
    k = parts.set_index(["gsis_id", "year_signed"])
    hit = [(a, s) in k.index if pd.notna(s) else False for a, s in zip(ps.gsis, ps.signed)]
    ps["modeled"] = hit
    for p in ["qual_part", "stat_part", "resid_part"]:
        ps[p] = [k[p].get((a, s), 0.0) if h else 0.0 for a, s, h in zip(ps.gsis, ps.signed, hit)]
    nonqb = ps.grp != "qb"
    ps["v_qual"] = np.where(ps.modeled & nonqb, ps.share * ps.qual_part, 0)
    ps["v_stat"] = np.where(ps.modeled & nonqb, ps.share * ps.stat_part, 0)
    ps["v_resid"] = np.where(ps.modeled & nonqb, ps.share * ps.resid_part, 0)
    ps["v_other_vet"] = np.where(ps.vet & nonqb & ~ps.modeled, ps.value, 0)
    ps["v_rookie"] = np.where(~ps.vet & nonqb, ps.value, 0)
    ps["v_qb"] = np.where(~nonqb, ps.value, 0)
    print(f"\nnon-QB veteran value covered by the decomposition: "
          f"{ps[ps.vet & nonqb & ps.modeled].value.sum() / ps[ps.vet & nonqb].value.sum():.0%}")
    cols = ["v_qual", "v_stat", "v_resid", "v_other_vet", "v_rookie", "v_qb"]
    T = ps.groupby(["season", "team"])[cols].sum().reset_index().merge(ap.ratings(), on=["season", "team"])
    T[cols] = T[cols] - T.groupby("season")[cols].transform("mean")
    print("sd across team-seasons (% cap):", T[cols].std().round(2).to_dict())
    for fe in [False, True]:
        m = smf.ols("rating ~ " + " + ".join(cols) + (" + C(team)" if fe else ""), T).fit(
            cov_type="cluster", cov_kwds={"groups": T.team})
        print(f"\n--- team test{' (franchise fixed effects)' if fe else ''}: pts per 1% of cap ---")
        for col in cols:
            lo, hi = m.conf_int().loc[col]
            print(f"  {col:12s} {m.params[col]:+.3f} ({lo:+.3f} to {hi:+.3f})")
        t = m.t_test("v_stat - v_qual = 0")
        print(f"  stat - quality: {float(np.ravel(t.effect)[0]):+.3f}, p {float(np.ravel(t.pvalue)[0]):.3f}")


def main():
    C, pos, q, share, st = load()
    c = contracts(pos, q, share, st)
    print(f"veteran contracts with a prior season (30%+ snaps, quality): {len(c)}")
    R, parts = pay_models(c)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nStep 1: APY (% of cap) per SD of each stat, holding quality / share / age / year fixed; "
          "info_ = next season's quality per SD of the stat (q on 0-1 scale)")
    print(R.round(3).to_string(index=False))
    R.to_csv(OUT / "stat_premium_pay.csv", index=False)
    team_test(parts, C)


if __name__ == "__main__":
    main()
