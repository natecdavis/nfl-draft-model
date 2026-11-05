"""Prospect model v2: each prospect's value in each season of his rookie deal, from
cfdarko's data (~/cfdarko, read only).

Outcome (cfdarko's v4 rookie-value label, data/processed/rookie_value_labels.parquet):
  value_y1..y4 = what his role was worth that season at veteran market prices, as a
  fraction of the cap: priced on snap share and PFF grade, per position group.
  Players who never reached the NFL count as 0. (Undrafted players who made the NFL
  are in the labels from the 2016 class on, so undrafted prospects are only used from
  2016; drafted players from 2012.)

Inputs, by version:
  slot       position group + where he was drafted (undrafted = pick 300)
  consensus  + Arif Hasan consensus rank (off the board = 330, plus a flag)
  full       + age, combine testing, recruiting, college PFF grades (peak/last by
               facet), college team strength and opponent quality, career workload
  (cfdarko's stored off-board scores are NOT used: they were fit on all years, which
  would leak test classes. Their raw inputs are here instead.)

One gradient-boosted model per season (Tweedie loss: lots of zeros, long right tail),
with predictions forced to fall as board rank and pick get worse.

Honest test: each class 2019-2022 predicted from earlier classes, using only seasons
that had been played before that draft (class c's model sees season t of class k only
if k + t - 1 <= c - 1).
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

CF = Path.home() / "cfdarko"
OUT = Path("data/derived")
SEASONS = [1, 2, 3, 4]
LAST_SEASON_PLAYED = 2025
OFF_BOARD = 330
FIRST_UDFA_CLASS = 2016  # first class whose undrafted NFL players are labeled
FIRST_BOARD_CLASS = 2014  # first class with a consensus board (Hasan); earlier classes have no ranks
UNDRAFTED = 300

FEATURES = {
    "slot": ["log_pick"],
    "consensus": ["log_pick", "log_rank", "on_board"],
}
PLAYER = ["age_at_draft", "combine_invited", "recruiting_composite", "recruiting_stars",
          "team_sp_offense", "team_sp_defense", "career_opp_def_ppa", "career_opp_off_ppa",
          "career_touches", "career_rush_att"] + \
         [f"combine_{t}_z" for t in ["forty", "vertical", "broad_jump", "bench", "cone", "shuttle", "ht", "wt",
                                     "arm", "hand", "wingspan"]] + \
         [f"pff_grades_{g}_{w}" for g in ["offense", "defense", "pass", "run", "pass_route", "pass_block",
                                          "run_block", "pass_rush_defense", "run_defense", "coverage_defense",
                                          "tackle"] for w in ["peak", "last"]]
FEATURES["full"] = FEATURES["consensus"] + PLAYER
# Pre-draft only (no draft position): the pick carries what 31 other teams knew, so it can't
# be used to ask "what if we took him at pick k?" (it would credit an unknown player taken
# at #5 with the outcomes of players teams valued at #5). At decision time the draft-day
# information is "still available at k", applied as fall shading (FALL_WEIGHT).
FEATURES["consensus_pre"] = ["log_rank", "on_board"]
FEATURES["full_pre"] = FEATURES["consensus_pre"] + PLAYER
FALL_WEIGHT = 0.85
GROUP_MAP = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "OL": "OL", "OT": "OL", "IOL": "OL",
             "DL": "DL", "EDGE": "EDGE", "LB": "LB", "DB": "DB", "CB": "DB", "S": "DB"}


def load() -> pd.DataFrame:
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(CF / "data/output/projections/latest.parquet").schema.names)
    cols = ["player_id", "player_name", "position_group", "last_college_season_year", "hasan_rank",
            "merged_value"] + [c for c in PLAYER if c in have]
    X = pd.read_parquet(CF / "data/output/projections/latest.parquet", columns=cols)
    B = pd.read_parquet(CF / "data/processed/bridge_players.parquet",
                        columns=["player_id", "draft_year", "overall_pick", "is_udfa", "pfr_id"]).dropna(subset=["player_id"])
    B = B.drop_duplicates("player_id")
    L = pd.read_parquet(CF / "data/processed/rookie_value_labels.parquet")
    L = L.dropna(subset=["player_id"]).drop_duplicates("player_id")
    df = X.merge(B, on="player_id", how="left")
    df["drafted"] = df.overall_pick.notna() & ~df.is_udfa.fillna(False).astype(bool)
    df["cls"] = np.where(df.draft_year.notna(), df.draft_year, df.last_college_season_year + 1).astype(int)
    # universe: cfdarko's scored prospects, board players and drafted players
    df = df[(df.merged_value.notna() | df.hasan_rank.notna() | df.drafted) & df.cls.between(2012, 2025)]
    # Undrafted players' NFL careers are only labeled from the 2016 class. Earlier undrafted
    # players are kept (dropping them made off-board and low-ranked board players look better
    # than they are: only the drafted ones were left) and given the average season values of
    # 2016-18 undrafted players with the same board status and rank band (see below).
    df["imputed"] = ~df.drafted & (df.cls < FIRST_UDFA_CLASS)
    df = df.merge(L[["player_id"] + [f"value_y{t}" for t in SEASONS + [5]] + ["group", "pfr_id"]]
                  .rename(columns={"pfr_id": "pfr_id_label"}), on="player_id", how="left")
    for t in SEASONS:
        played = df.cls + t - 1 <= LAST_SEASON_PLAYED
        # never reached NFL = 0; tiny negatives (cfdarko depth-chart pricing artifacts, > -1% of cap) -> 0
        df[f"y{t}"] = np.where(played, df[f"value_y{t}"].fillna(0.0).clip(lower=0), np.nan)
    df["grp"] = df.group.fillna(df.position_group.map(GROUP_MAP)).fillna(df.position_group)
    df["pick"] = np.where(df.drafted, df.overall_pick, UNDRAFTED)
    df["log_pick"] = np.log(df.pick + 10)
    df["on_board"] = df.hasan_rank.notna().astype(int)
    df["log_rank"] = np.log(df.hasan_rank.fillna(OFF_BOARD) + 10)
    df["grp_code"] = df.grp.astype("category").cat.codes
    # fill the unlabeled pre-2016 undrafted rows with cell averages (on/off board x rank band)
    band = lambda d: np.where(d.hasan_rank.isna(), "off",
                              pd.cut(d.hasan_rank, [0, 100, 200, 250, 10_000]).astype(str))
    ref = df[~df.drafted & df.cls.between(FIRST_UDFA_CLASS, FIRST_UDFA_CLASS + 2)]
    for t in SEASONS:
        means = ref.groupby(band(ref))[f"y{t}"].mean()
        fill = pd.Series(band(df[df.imputed]), index=df.index[df.imputed]).map(means)
        played = df.cls + t - 1 <= LAST_SEASON_PLAYED
        df.loc[df.imputed & played, f"y{t}"] = fill[df.imputed & played]
    return df.reset_index(drop=True)


PARAMS = dict(objective="tweedie", tweedie_variance_power=1.3, n_estimators=500, learning_rate=0.03,
              num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, verbose=-1)


def fit(train: pd.DataFrame, version: str, t: int, seed=0):
    feats = FEATURES[version] + ["grp_code"]
    mono = [-1 if f in ("log_pick", "log_rank") else 0 for f in feats]
    tr = train[train[f"y{t}"].notna()]
    if "log_rank" in feats:  # before 2014 nobody has a board rank, so 'off board' meant nothing
        tr = tr[tr.cls >= FIRST_BOARD_CLASS]
    m = lgb.LGBMRegressor(**PARAMS, monotone_constraints=mono, random_state=seed)
    m.fit(tr[feats], tr[f"y{t}"], categorical_feature=["grp_code"])
    return m, feats


def predict_seasons(train, test, version, cutoff_cls=None):
    """Predicted value for seasons 1-4 for test rows. If cutoff_cls is given, training
    only sees seasons played before that draft."""
    out = np.zeros((len(test), len(SEASONS)))
    for i, t in enumerate(SEASONS):
        tr = train
        if cutoff_cls is not None:
            tr = train[train.cls + t - 1 <= cutoff_cls - 1]
        m, feats = fit(tr, version, t)
        out[:, i] = m.predict(test[feats])
    return out


def rolling(df, versions=("slot", "consensus", "full"), test_classes=(2019, 2020, 2021, 2022)):
    rows = []
    for c in test_classes:
        te = df[df.cls == c]
        tr = df[df.cls < c]
        res = te[["player_id", "player_name", "cls", "grp", "pick", "drafted", "hasan_rank"] +
                 [f"y{t}" for t in SEASONS]].copy()
        for v in versions:
            p = predict_seasons(tr, te, v, cutoff_cls=c)
            for i, t in enumerate(SEASONS):
                res[f"{v}_p{t}"] = p[:, i]
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def score(r: pd.DataFrame, versions=("slot", "consensus", "full")) -> pd.DataFrame:
    r = r.assign(y14=r[[f"y{t}" for t in SEASONS]].sum(axis=1), y12=r.y1 + r.y2)
    out = {}
    for v in versions:
        p14 = r[[f"{v}_p{t}" for t in SEASONS]].sum(axis=1)
        p12 = r[f"{v}_p1"] + r[f"{v}_p2"]
        cap = lambda k: np.mean([g.assign(p=p14[g.index]).nlargest(k, "p").y14.sum() / g.nlargest(k, "y14").y14.sum()
                                 for _, g in r.groupby("cls")])
        out[v] = {"4-yr value error (RMSE, % of cap)": np.sqrt(((p14 - r.y14) ** 2).mean()) * 100,
                  "early value error (yrs 1-2)": np.sqrt(((p12 - r.y12) ** 2).mean()) * 100,
                  "rank correlation, 4-yr": r.assign(p=p14).groupby("cls").apply(
                      lambda g: g.p.corr(g.y14, method="spearman"), include_groups=False).mean(),
                  "top-32 value captured": cap(32), "top-100 value captured": cap(100)}
    return pd.DataFrame(out)


def main():
    df = load()
    df.to_parquet(OUT / "pm_universe.parquet")
    print(f"Universe: {len(df)} prospects, classes {df.cls.min()}-{df.cls.max()}; drafted {df.drafted.sum()}; "
          f"reached NFL (any value>0 in seen seasons): {(df[[f'y{t}' for t in SEASONS]].fillna(0).sum(axis=1) > 0).mean():.0%}")
    print("Per class:", df.groupby("cls").size().to_dict())
    r = rolling(df)
    r.to_parquet(OUT / "pm_rolling.parquet")
    pd.set_option("display.width", 200)
    subsets = {"All prospects": r, "Drafted, rounds 1-4 (picks 1-~140)": r[r.drafted & (r.pick <= 140)],
               "Drafted, rounds 5-7": r[r.drafted & (r.pick > 140)], "On Hasan's board": r[r.hasan_rank.notna()]}
    for name, s in subsets.items():
        print(f"\n{name} ({len(s)} players, test classes 2019-2022):")
        print(score(s).round(3).to_string())


if __name__ == "__main__":
    main()


# ---------- decision test: who would each version take at each pick? ----------

def choice_test(df, params_override=None, versions=("consensus", "full"), test_classes=(2019, 2020, 2021, 2022),
                picks=None):
    """At each pick k of each test draft, every version scores all prospects still
    available (actual pick >= k, or undrafted) as if taken at k, and takes the best.
    Records the chosen player's actual 4-season value, next to what the team took.
    Pure value: no team needs or positions; the same player can't be taken twice by
    the same version within a draft."""
    global PARAMS
    saved = dict(PARAMS)
    if params_override:
        PARAMS.update(params_override)
    try:
        rows = []
        for c in test_classes:
            te = df[df.cls == c].copy()
            tr = df[df.cls < c]
            te["y14"] = te[[f"y{t}" for t in SEASONS]].sum(axis=1)
            models = {v: [fit(tr[tr.cls + t - 1 <= c - 1], v, t) for t in SEASONS] for v in versions}
            ks = picks or range(1, int(te[te.drafted].pick.max()) + 1)
            taken = {v: set() for v in versions}
            for k in ks:
                pool = te[(te.pick >= k)]
                actual = te[te.pick == k]
                row = {"cls": c, "pick": k, "team_value": actual.y14.iloc[0] if len(actual) else np.nan}
                for v in versions:
                    cand = pool[~pool.player_id.isin(taken[v])].copy()
                    cand["log_pick"] = np.log(k + 10)
                    if v.endswith("_pre"):  # board players who fell past their rank: shade toward k
                        lk = np.log(k + 10)
                        fell = cand.on_board.astype(bool) & (cand.log_rank < lk)
                        cand.loc[fell, "log_rank"] += FALL_WEIGHT * (lk - cand.loc[fell, "log_rank"])
                    pred = sum(m.predict(cand[f]) for m, f in models[v])
                    i = int(np.argmax(pred))
                    taken[v].add(cand.player_id.iloc[i])
                    row[f"{v}_value"] = cand.y14.iloc[i]
                    row[f"{v}_on_board"] = bool(cand.hasan_rank.notna().iloc[i])
                rows.append(row)
        return pd.DataFrame(rows)
    finally:
        PARAMS = saved


def summarize_choice(ch, versions=("consensus", "full")):
    ch = ch.assign(rnd=pd.cut(ch.pick, [0, 32, 140, 262], labels=["R1", "R2-4", "R5-7"]))
    cols = ["team_value"] + [f"{v}_value" for v in versions]
    t = ch.groupby("rnd", observed=True)[cols].mean() * 100
    t.loc["All"] = ch[cols].mean() * 100
    return t
