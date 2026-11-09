"""Do play-level metrics (scripts/pbp_metrics.py) add to the PFF talent model (scripts/darko_pff.py)?

Pairs: player-season t -> t+1 in the same grade family (darko_pff chains), snaps_t >= 100, target seasons
through 2025. Base = PFF filtered talent through t (talent_post, no context) + log snaps t. Add the season-t
play metrics for the group:
  QB  passer effect (qb_epa), RAPM offense      RB  rusher, receiver effects, RAPM offense
  WR/TE receiver effect, RAPM offense            OL  RAPM offense       DL/EDGE/LB/DB  RAPM defense
(plus log play counts). Missing metric -> 0 + flag. Targets: next-season PFF grade (z, weight sqrt snaps),
next-season log snaps, and next-season play metric (a target built without PFF; for that one the
comparison is metric alone vs PFF talent alone vs both). Out-of-sample R^2: 5-fold CV grouped by player,
best of ridge / gradient boosting. Two windows: involved-player metrics 2007-25 targets, RAPM 2017-25.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/derived"
FAM = {"QB": "QB", "RB": "RB", "WR": "REC", "TE": "REC", "OL": "OL", "DL": "DEF", "EDGE": "DEF", "LB": "DEF", "DB": "DEF"}
INV = {"QB": ["qb_epa"], "RB": ["rush_epa", "rec_epa"], "WR": ["rec_epa"], "TE": ["rec_epa"]}
RAPM = {"QB": ["rapm_off"], "RB": ["rapm_off"], "WR": ["rapm_off"], "TE": ["rapm_off"], "OL": ["rapm_off"],
        **{g: ["rapm_def"] for g in ["DL", "EDGE", "LB", "DB"]}}
NCOL = {"qb_epa": "qb_n", "rec_epa": "rec_n", "rush_epa": "rush_n", "rapm_off": "off_n", "rapm_def": "def_n"}
PRIMARY = {"QB": "qb_epa", "RB": "rush_epa", "WR": "rec_epa", "TE": "rec_epa", "OL": "rapm_off",
           **{g: "rapm_def" for g in ["DL", "EDGE", "LB", "DB"]}}


def oos(X, y, w, grp):
    best = -np.inf
    for mk in [lambda: RidgeCV(alphas=np.logspace(-2, 3, 12)),
               lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8,
                                                     min_samples_leaf=40)]:
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = mk(); m.fit(X[tr], y[tr], sample_weight=w[tr]); pred[te] = m.predict(X[te])
        r2 = 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
        if r2 > best:
            best, bp = r2, pred
    return best, bp


def r2w(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def load():
    t = pd.read_parquet(D / "darko_pff.parquet")
    pl = pd.read_parquet(ROOT / "data/raw/players.parquet")[["pff_id", "gsis_id"]].dropna()
    pl = pl.assign(pff_id=pl.pff_id.astype(int)).drop_duplicates("pff_id")
    t = t.merge(pl, left_on="player_id", right_on="pff_id", how="left")
    m = pd.read_parquet(D / "pbp_player_seasons.parquet")
    t = t.merge(m, on=["gsis_id", "season"], how="left")
    t["fam"] = t.g.map(FAM)
    nxt = t[["player_id", "season", "fam", "g", "z", "snaps", "tm"] + list(NCOL) + list(NCOL.values())] \
        .assign(season=lambda d: d.season - 1)
    P = t.merge(nxt, on=["player_id", "season", "fam"], suffixes=("", "_n"))
    P = P[P.snaps >= 100].copy()
    P["moved"] = P.tm != P.tm_n
    P["lsn"], P["lsn_n"] = np.log1p(P.snaps), np.log1p(P.snaps_n)
    return P


def main():
    pd.set_option("display.width", 220)
    P = load()
    print(f"pairs: {len(P):,}; gsis matched {P.gsis_id.notna().mean():.1%}; "
          f"RAPM present (2016-24 seasons) {P.loc[P.season >= 2016, 'rapm_off'].notna().mean() + P.loc[P.season >= 2016, 'rapm_def'].notna().mean():.1%}")
    rows, stab = [], []
    for win, first, feats in [("involved", 2006, INV), ("rapm", 2016, RAPM)]:
        for g, mets in feats.items():
            d = P[(P.g == g) & (P.season >= first)].copy()
            extra = []
            for c in mets:
                d[c + "_has"] = d[c].notna().astype(float)
                d[c] = d[c].fillna(0.0)
                d[c + "_ln"] = np.log1p(d[NCOL[c]].fillna(0))
                extra += [c, c + "_has", c + "_ln"]
            base = ["talent_post", "lsn"]
            grp = d.player_id.to_numpy()
            mv = d.moved.to_numpy()
            for tgt in ["z_n", "lsn_n"]:
                y = d[tgt].to_numpy()
                w = np.sqrt(d.snaps_n.to_numpy()) if tgt == "z_n" else np.ones(len(d))
                rb, pb = oos(d[base].to_numpy(float), y, w, grp)
                ra, pa = oos(d[base + extra].to_numpy(float), y, w, grp)
                rows.append(dict(window=win, g=g, target=tgt, n=len(d), base=rb, plus_pbp=ra, gain=ra - rb,
                                 movers_base=r2w(y[mv], pb[mv], w[mv]), movers_plus=r2w(y[mv], pa[mv], w[mv])))
            # PFF-free target: next season's primary play metric (players with it in both seasons)
            c = PRIMARY[g] if PRIMARY[g] in mets else mets[0]
            e = d[(d[c + "_has"] == 1) & d[c + "_n"].notna() & (d[NCOL[c] + "_n"].fillna(0) >= 50)]
            if len(e) >= 100:
                y = e[c + "_n"].to_numpy(); w = np.sqrt(e[NCOL[c] + "_n"].to_numpy()); gg = e.player_id.to_numpy()
                r_m, _ = oos(e[[c, c + "_ln"]].to_numpy(float), y, w, gg)
                r_p, _ = oos(e[base].to_numpy(float), y, w, gg)
                r_b, _ = oos(e[base + [c, c + "_ln"]].to_numpy(float), y, w, gg)
                stab.append(dict(window=win, g=g, metric=c, n=len(e), metric_alone=r_m, pff_alone=r_p, both=r_b,
                                 yoy_corr=np.corrcoef(e[c], y)[0, 1]))
            print(f"  {win} {g} done", flush=True)
    R = pd.DataFrame(rows)
    for tgt, lab in [("z_n", "next-season PFF grade"), ("lsn_n", "next-season log snaps")]:
        print(f"\n== {lab}: out-of-sample R^2, base (PFF talent + snaps) vs + play metrics ==")
        print(R[R.target == tgt].drop(columns="target").round(3).to_string(index=False))
    print("\n== next-season play metric (PFF-free target), players with >= 50 plays next season ==")
    print(pd.DataFrame(stab).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
