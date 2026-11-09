"""Play-level player metrics from nflverse play-by-play (follow-up to scripts/darko_pff.py).

Plays: regular season, pass/run with EPA, no two-point tries, kneels or spikes, win probability 0.05-0.95
(no garbage time). Outcome = EPA (already adjusts for down, distance, field position). Fit season by season.

A. Involved players (2006-2025, play-by-play only)
   dropbacks (incl. sacks, scrambles): EPA = mu + passer + receiver (targeted plays) + defense team
   designed runs:                      EPA = mu + rusher + offense team (line, scheme) + defense team
   Ridge with a penalty per role (column scaling); passer and receiver are adjusted for each other, the
   QB <-> receiver context the PFF tests found strongest.
B. RAPM (2016-2025, nflverse participation: the 22 players on the field)
   EPA = mu + pass-play flag + sum(offense players' o_i) + sum(defense players' d_j); separate offense and
   defense coefficients (d_j = EPA allowed, negative = good). Ridge.
Penalties picked by 5-fold cross-validation over plays in 2016-2018 (squared error), then fixed for all seasons.

Writes data/derived/pbp_player_seasons.parquet: gsis_id, season, metric columns with play counts
(qb_epa/qb_n, rec_epa/rec_n, rush_epa/rush_n, rapm_off/off_n, rapm_def/def_n), effects in EPA per play.
Usage: python scripts/pbp_metrics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/nflverse_pbp"
OUT = ROOT / "data/derived/pbp_player_seasons.parquet"
SEASONS = range(2006, 2026)
PART = range(2016, 2026)
CV_SEASONS = [2016, 2017, 2018]
COLS = ["game_id", "play_id", "season_type", "play_type", "epa", "wp", "two_point_attempt", "qb_kneel", "qb_spike",
        "qb_dropback", "qb_scramble", "passer_player_id", "receiver_player_id", "rusher_player_id", "posteam", "defteam"]


def plays(y):
    d = pd.read_parquet(RAW / f"play_by_play_{y}.parquet", columns=COLS)
    d = d[(d.season_type == "REG") & d.play_type.isin(["pass", "run"]) & d.epa.notna() & d.wp.between(0.05, 0.95)
          & (d.two_point_attempt != 1) & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
    d["season"] = y
    d["dropback"] = (d.qb_dropback == 1)
    d["passer"] = d.passer_player_id.where(d.passer_player_id.notna(), d.rusher_player_id.where(d.qb_scramble == 1))
    return d


def design(blocks, n):
    """blocks: list of (codes array with -1 = absent, penalty). Returns sparse X (columns scaled by
    1/sqrt(penalty) so one Ridge alpha=1 applies each role's penalty), and column offsets."""
    mats, offs, o = [], [], 0
    for codes, lam in blocks:
        k = codes.max() + 1
        ok = codes >= 0
        m = sp.csr_matrix((np.full(ok.sum(), 1 / np.sqrt(lam)), (np.flatnonzero(ok), codes[ok])), shape=(n, k))
        mats.append(m); offs.append((o, k, lam)); o += k
    return sp.hstack(mats).tocsr(), offs


def fit_blocks(y, blocks, extra=None):
    X, offs = design(blocks, len(y))
    if extra is not None:  # unpenalized-ish fixed columns (tiny penalty via large scale)
        X = sp.hstack([X, sp.csr_matrix(extra * 100.0)]).tocsr()
    r = Ridge(alpha=1.0, fit_intercept=True, solver="auto").fit(X, y)
    coefs = [r.coef_[o:o + k] / np.sqrt(lam) for o, k, lam in offs]
    return r, X, coefs


def cv_loss(y, blocks, extra=None, k=5):
    loss = 0.0
    for tr, te in KFold(k, shuffle=True, random_state=0).split(y):
        X, offs = design(blocks, len(y))
        if extra is not None:
            X = sp.hstack([X, sp.csr_matrix(extra * 100.0)]).tocsr()
        r = Ridge(alpha=1.0).fit(X[tr], y[tr])
        loss += np.sum((y[te] - r.predict(X[te])) ** 2)
    return loss / len(y)


def codes(s):
    c, u = pd.factorize(s)
    return c, u


# ---------------------------------------------------------------- A: involved players
def model_a_blocks(d, lam):
    dp = d[d.dropback & d.passer.notna()]
    pc, pu = codes(dp.passer); rc, ru = codes(dp.receiver_player_id); dc, _ = codes(dp.defteam)
    drop = (dp.epa.to_numpy(), [(pc, lam["qb"]), (rc, lam["rec"]), (dc, lam["team"])], pu, ru)
    dr = d[~d.dropback & d.rusher_player_id.notna()]
    uc, uu = codes(dr.rusher_player_id); oc, _ = codes(dr.posteam); dc2, _ = codes(dr.defteam)
    run = (dr.epa.to_numpy(), [(uc, lam["rush"]), (oc, lam["team"]), (dc2, lam["team"])], uu)
    return drop, run, dp, dr


def tune_a(D):
    grid = [25, 50, 100, 200, 400, 800, 1600]
    best = {}
    for role in ["qb", "rec", "rush"]:
        res = []
        for g in grid:
            lam = {"qb": 200, "rec": 200, "rush": 200, "team": 100, role: g}
            loss = 0.0
            for d in D:
                drop, run, _, _ = model_a_blocks(d, lam)
                loss += cv_loss(*drop[:2]) if role != "rush" else cv_loss(*run[:2])
            res.append(loss)
        best[role] = grid[int(np.argmin(res))]
        print(f"  A {role}: CV loss by penalty {dict(zip(grid, np.round(res, 5)))} -> {best[role]}", flush=True)
    best["team"] = 100
    return best


def run_a(d, lam):
    drop, run, dp, dr = model_a_blocks(d, lam)
    _, _, (cq, cr, _) = fit_blocks(*drop[:2])
    _, _, (cu, _, _) = fit_blocks(*run[:2])
    out = []
    for ids, coef, cnt, name in [(drop[2], cq, dp.passer.value_counts(), "qb"),
                                 (drop[3], cr, dp.receiver_player_id.value_counts(), "rec"),
                                 (run[2], cu, dr.rusher_player_id.value_counts(), "rush")]:
        out.append(pd.DataFrame({"gsis_id": ids, f"{name}_epa": coef, f"{name}_n": cnt.reindex(ids).to_numpy()}))
    return out


# ---------------------------------------------------------------- B: RAPM
def rapm_data(y, d):
    p = pd.read_parquet(RAW / f"pbp_participation_{y}.parquet",
                        columns=["nflverse_game_id", "play_id", "offense_players", "defense_players"])
    p = p.rename(columns={"nflverse_game_id": "game_id"})
    m = d.merge(p, on=["game_id", "play_id"])
    m = m[(m.offense_players.str.count(";") >= 9) & (m.defense_players.str.count(";") >= 9)].reset_index(drop=True)
    off = m.offense_players.str.split(";").explode()
    de = m.defense_players.str.split(";").explode()
    return m, off, de


def rapm_X(m, off, de, lam):
    n = len(m)
    oc, ou = pd.factorize(off.to_numpy()); dc, du = pd.factorize(de.to_numpy())
    Xo = sp.csr_matrix((np.full(len(oc), 1 / np.sqrt(lam)), (off.index.to_numpy(), oc)), shape=(n, len(ou)))
    Xd = sp.csr_matrix((np.full(len(dc), 1 / np.sqrt(lam)), (de.index.to_numpy(), dc)), shape=(n, len(du)))
    X = sp.hstack([Xo, Xd, sp.csr_matrix(m.dropback.to_numpy(float)[:, None] * 100.0)]).tocsr()
    return X, ou, du


def tune_b(R):
    grid = [100, 200, 400, 800, 1600, 3200, 6400]
    res = []
    for g in grid:
        loss = 0.0
        for m, off, de in R:
            X, _, _ = rapm_X(m, off, de, g)
            y = m.epa.to_numpy()
            for tr, te in KFold(5, shuffle=True, random_state=0).split(y):
                r = Ridge(alpha=1.0).fit(X[tr], y[tr])
                loss += np.sum((y[te] - r.predict(X[te])) ** 2) / len(y)
        res.append(loss)
    best = grid[int(np.argmin(res))]
    print(f"  B RAPM: CV loss by penalty {dict(zip(grid, np.round(res, 5)))} -> {best}", flush=True)
    return best


def run_b(m, off, de, lam):
    X, ou, du = rapm_X(m, off, de, lam)
    r = Ridge(alpha=1.0).fit(X, m.epa.to_numpy())
    k = len(ou)
    co = r.coef_[:k] / np.sqrt(lam); cd = r.coef_[k:k + len(du)] / np.sqrt(lam)
    return [pd.DataFrame({"gsis_id": ou, "rapm_off": co, "off_n": off.value_counts().reindex(ou).to_numpy()}),
            pd.DataFrame({"gsis_id": du, "rapm_def": cd, "def_n": de.value_counts().reindex(du).to_numpy()})]


def main():
    print("loading plays ...", flush=True)
    P = {y: plays(y) for y in SEASONS}
    print(f"  {sum(len(d) for d in P.values()):,} plays {min(SEASONS)}-{max(SEASONS)}")
    lam_a = tune_a([P[y] for y in CV_SEASONS])
    R = {y: rapm_data(y, P[y]) for y in PART}
    print("  RAPM plays with 11 v 11 participation: " + ", ".join(f"{y} {len(R[y][0]):,}" for y in PART))
    lam_b = tune_b([R[y] for y in CV_SEASONS])
    rows = []
    for y in SEASONS:
        parts = run_a(P[y], lam_a) + (run_b(*R[y], lam_b) if y in PART else [])
        df = parts[0]
        for p in parts[1:]:
            df = df.merge(p, on="gsis_id", how="outer")
        rows.append(df.assign(season=y))
        print(f"  {y}: {len(df):,} players", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(out):,} player-seasons); penalties A {lam_a}, B {lam_b}")


if __name__ == "__main__":
    main()
