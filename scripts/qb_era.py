"""Is QB value by draft slot stable across eras? Drafted QBs 1999-2022, seasons 1-4.

The outcome engine's QB careers are the 2012-2022 classes only (cfdarko labels need snap counts,
2012+), and the value of a QB at a given consensus rank roughly doubled as the 2016-18 classes
entered the pool (README, 'Holdout check'). Here every class is measured the same way, from
nflverse player stats (1999+), so eras can be compared:

  season proxy = a x plays + b x EPA above replacement (EPA - plays x replacement EPA/play, where
  replacement = that season's plays-weighted EPA/play of QBs outside each team's top passer), with
  a, b >= 0 fit to cfdarko's QB season values (cap share; 2012-2022 classes, seasons 1-4).
  plays = pass attempts + sacks + carries; EPA = passing + rushing EPA (regular season).

Draft slot stands in for consensus rank (no boards before 2014): fine for comparing eras with each
other, not for valuing picks.

Run:  python scripts/qb_era.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from outlook import RAW  # noqa: E402

OLD = RAW / "player_stats_1999_2011.parquet"
ERAS = [(1999, 2005), (2006, 2011), (2012, 2015), (2016, 2018), (2019, 2022)]
BANDS = [(1, 10), (11, 32), (33, 100), (101, 300)]


def stats() -> pd.DataFrame:
    if not OLD.exists():
        import nflreadpy as nfl
        nfl.load_player_stats(list(range(1999, 2012)), summary_level="reg").to_pandas().to_parquet(OLD)
    new = pd.read_parquet(RAW / "player_stats.parquet")
    new = new[new.season_type == "REG"] if "season_type" in new else new
    cols = ["player_id", "player_name", "position", "season", "recent_team", "attempts", "sacks_suffered", "carries",
            "passing_epa", "rushing_epa"]
    s = pd.concat([pd.read_parquet(OLD)[cols], new[cols]], ignore_index=True)
    s = s[s.position == "QB"].copy()
    for c in cols[5:]:
        s[c] = s[c].fillna(0.0)
    s["plays"] = s.attempts + s.sacks_suffered + s.carries
    s["epa"] = s.passing_epa + s.rushing_epa
    # replacement: QBs who weren't their team's main passer that season
    top = s.groupby(["season", "recent_team"]).attempts.transform("max") == s.attempts
    bk = s[~top].groupby("season")[["epa", "plays"]].sum()
    s["repl"] = s.season.map(bk.epa / bk.plays)
    s["var"] = s.epa - s.plays * s.repl
    return s.groupby(["player_id", "season"], as_index=False)[["plays", "var"]].sum()


def main():
    S = stats()
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    q = dp[(dp.position == "QB") & dp.season.between(1999, 2022) & dp.gsis_id.notna()][
        ["season", "pick", "gsis_id", "pfr_player_id", "pfr_player_name"]].rename(columns={"season": "cls"})
    rows = []
    for t in (1, 2, 3, 4):
        x = q.assign(t=t, season=q.cls + t - 1).merge(S, left_on=["gsis_id", "season"], right_on=["player_id", "season"],
                                                      how="left")
        rows.append(x)
    L = pd.concat(rows, ignore_index=True)
    L[["plays", "var"]] = L[["plays", "var"]].fillna(0.0)

    # calibrate to cfdarko QB labels (cap share), 2012-2022 classes
    import prospect_model as pm
    cf = pm.load()
    cf = cf[(cf.grp == "QB") & cf.drafted]
    lab = {}
    for _, r in cf.iterrows():
        for t in (1, 2, 3, 4):
            if pd.notna(r[f"y{t}"]) and isinstance(r.pfr_id_label, str):
                lab[(r.pfr_id_label, t)] = r[f"y{t}"]
    L["cf"] = [lab.get((p, t)) for p, t in zip(L.pfr_player_id, L.t)]
    C = L[L.cf.notna() & L.cls.between(2012, 2022)]
    X = np.column_stack([C.plays, C["var"].clip(lower=0), C["var"].clip(upper=0)])
    coef, _ = nnls(X, C.cf.to_numpy())
    L["proxy"] = np.column_stack([L.plays, L["var"].clip(lower=0), L["var"].clip(upper=0)]) @ coef
    L["proxy"] = L.proxy.clip(lower=0)
    C = L[L.cf.notna() & L.cls.between(2012, 2022)]
    tot = L.groupby(["cls", "pick", "pfr_player_name"], as_index=False).agg(proxy=("proxy", "sum"), plays=("plays", "sum"))
    ct = C.groupby(["cls", "pfr_player_name"]).agg(proxy=("proxy", "sum"), cf=("cf", "sum"))
    print(f"proxy = {coef[0] * 1000:.3f}e-3 x plays + {coef[1] * 1000:.3f}e-3 x EPA above repl (+) "
          f"{coef[2] * 1000:.3f}e-3 x (-); season r = {np.corrcoef(C.proxy, C.cf)[0, 1]:.2f}, "
          f"4-season r = {np.corrcoef(ct.proxy, ct.cf)[0, 1]:.2f}")

    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, float)
        if len(x) < 3:
            return f"{x.mean() if len(x) else float('nan'):.3f} (n={len(x)})"
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():.3f} ({np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f}) n={len(x)}"

    tot["era"] = pd.cut(tot.cls, [e[0] - 1 for e in ERAS] + [ERAS[-1][1]], labels=[f"{a}-{b}" for a, b in ERAS])
    print("\nQB value, seasons 1-4 (cap share, proxy units), by draft slot and era:")
    for lo, hi in BANDS:
        print(f" picks {lo}-{hi}")
        for e in tot.era.cat.categories:
            print(f"   {e}: {ci(tot[(tot.era == e) & tot.pick.between(lo, hi)].proxy)}")
    # same eras with cfdarko labels where they exist (validation of the proxy)
    ctt = C.groupby(["cls", "pick"], as_index=False).cf.sum()
    print("\ncfdarko labels, picks 1-32: " + ", ".join(
        f"{a}-{b} {ctt[ctt.cls.between(a, b) & (ctt.pick <= 32)].cf.mean():.3f}" for a, b in ERAS[2:]))
    print("proxy, picks 1-32:          " + ", ".join(
        f"{a}-{b} {tot[tot.cls.between(a, b) & (tot.pick <= 32)].proxy.mean():.3f}" for a, b in ERAS[2:]))
    # playing time by season: do rookies start sooner now?
    L["era"] = pd.cut(L.cls, [e[0] - 1 for e in ERAS] + [ERAS[-1][1]], labels=[f"{a}-{b}" for a, b in ERAS])
    r1 = L[L.pick <= 32]
    print("\nRound-1 QBs, share of seasons with 300+ plays, by season 1-4:")
    print((r1.assign(start=r1.plays >= 300).pivot_table(index="era", columns="t", values="start", aggfunc="mean",
                                                         observed=True) * 100).round(0).to_string())
    a = tot[tot.pick <= 32]
    print("\nround-1 by class (proxy, mean):", a.groupby("cls").proxy.mean().round(2).to_dict())
    print("long-run (1999-2022) / 2012-2022, picks 1-32:",
          round(a.proxy.mean() / a[a.cls >= 2012].proxy.mean(), 2), "| picks 33-100:",
          round(tot[tot.pick.between(33, 100)].proxy.mean() / tot[tot.pick.between(33, 100) & (tot.cls >= 2012)].proxy.mean(), 2))
    tot.to_parquet(RAW.parent / "derived" / "qb_era.parquet")


if __name__ == "__main__":
    main()
