"""The draft simulator as one self-contained HTML page: the app's main simulator page (built live,
src/pages/2_Draft_simulator.py) and the private Claude artifact (this script's output).

Same model as the Past drafts review: the value tables per team situation
(data/derived/sim_tables), each draft's consensus board plus drafted players not on it, each team's
real picks and draft-day state, and the team's values from the Past drafts review. The page does
the table lookups in JavaScript (same interpolation as review.Table.lookup).

"Actually delivered" (drafts through 2022): what each available player really did in his first four
seasons, at the cost of each of the team's picks, as titles for that team (review.RealizedV2, same as
the Past drafts page), precomputed for every player at every pick of every team and cached in
data/derived/sim_realized.npz (rebuilt when the review changes).

The output embeds Arif Hasan's consensus boards: keep it private (data/ is not in git).
Run:  python scripts/export_simulator_artifact.py  -> data/derived/artifact/draft_simulator.html
"""
import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from review import OFF_BOARD_RANK, K_GRID, T_GRID  # noqa: E402
from simulator import FALL_WEIGHT, TABLE_DIR, _key, pool, situation  # noqa: E402
from outlook import GROUPS, OUT, RAW  # noqa: E402
from positions import weights  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "simulator_artifact.html"
DEST = OUT / "artifact" / "draft_simulator.html"
SCALE = 100_000  # tables stored as int16 = value (fraction of a title) x SCALE
REAL_SCALE = 1000  # realized titles stored as int16 = % of a title x REAL_SCALE; missing = -32768
REAL_CACHE = OUT / "sim_realized_v3.npz"  # v2: recent drafts scored "so far"; v3: real positions


def realized_arrays(S, R, verbose=False) -> dict:
    """{"year_team": (picks, pool players) realized titles in % of one}, every draft with at least
    one season played; drafts after 2022 are scored on the seasons played so far."""
    stamp = max((OUT / f).stat().st_mtime for f in ["review_states.parquet", "review_picks.parquet"])
    if REAL_CACHE.exists() and REAL_CACHE.stat().st_mtime > stamp:
        z = np.load(REAL_CACHE)
        return {k: z[k] for k in z.files}
    from review import RealizedV2
    from trade import Context
    RV, out = RealizedV2(Context()), {}
    for y in sorted(y for y in R.year.unique() if RV.seasons_played(int(y)) > 0):
        P = pool(int(y))
        g, at, nm = P.grp_record.tolist(), P.drafted_at.tolist(), P["name"].tolist()  # his real position
        for _, r in S[S.year == y].iterrows():
            ks = sorted(int(k) for k in R[(R.year == y) & (R.team == r.team)].pick)
            if ks:
                out[f"{y}_{r.team}"] = np.array([RV.titles_many(situation(r), int(y), k, g, at, nm, partial=True) for k in ks])
        if verbose:
            print(f"realized {y} done", flush=True)
    out["_seasons"] = np.array([[y, RV.seasons_played(int(y))] for y in sorted(R.year.unique())])
    np.savez_compressed(REAL_CACHE, **out)
    return out


def build_html(theme: str | None = None, verbose: bool = False) -> str:
    """The whole page as one HTML string, from the current review data and simulator tables.
    theme: "light" / "dark" pins the palette (e.g. to match the Streamlit app); None follows the viewer."""
    S = pd.read_parquet(OUT / "review_states.parquet")
    R = pd.read_parquet(OUT / "review_picks.parquet")
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    num = lambda x, nd=3: None if pd.isna(x) else round(float(x), nd)

    real = realized_arrays(S, R, verbose)
    keys, blobs = {}, []
    teams, rblobs, roff = {}, [], 0
    for _, r in S.iterrows():
        sit = situation(r)
        k = _key(sit, r.year)
        if k not in keys:
            z = np.load(TABLE_DIR / f"{k}.npz")
            arr = np.stack([z[g] for g in GROUPS])  # (groups, 3 levels, T, K)
            q = np.round(arr * SCALE)
            assert np.abs(q).max() < 32767
            keys[k] = len(blobs)
            blobs.append(q.astype("<i2").tobytes())
        mine = R[(R.year == r.year) & (R.team == r.team)].sort_values("pick")
        teams.setdefault(str(int(r.year)), {})[r.team] = {
            "qb": r.qb, "status": r.status, "strength": num(r.strength_prior, 1),
            "qb_from": None if pd.isna(r.qb_from) else int(r.qb_from), "p_need": num(r.p_need, 2),
            "table": keys[k], "real": None,
            "picks": [[int(p.pick), int(p["round"]), p.original_team, int(bool(p.traded_on_night)), p.player, p.grp,
                       num(p.board_rank, 0), num(p.value_actual), num(p.real_team)] for _, p in mine.iterrows()],
        }
    played = {str(int(y)): int(n) for y, n in real.get("_seasons", [])}
    pool_cache = {}
    for yt, a in real.items():
        if yt.startswith("_"):
            continue
        y, tm = yt.split("_")
        if tm in teams.get(y, {}):
            q = np.where(np.isnan(a), -32768, np.clip(np.round(a * REAL_SCALE), -32767, 32767)).astype("<i2")
            T = teams[y][tm]
            T["real"], T["seasons"] = roff, played.get(y, 4)
            rblobs.append(q.tobytes()); roff += q.size
            if int(y) > 2022:  # the review doesn't score these: the team's own pick, from the same array
                P = pool_cache.setdefault(y, pool(int(y)))
                at = P.drafted_at.to_numpy()
                for j, pk in enumerate(T["picks"]):
                    i = np.flatnonzero(at == pk[0])
                    pk[8] = num(a[j, i[0]]) if len(i) else None
    pools, blends = {}, {}
    for y in sorted(R.year.unique()):
        P = pool(int(y))
        pools[str(int(y))] = [[n, g, num(rk, 0), sc if isinstance(sc, str) else "", num(at, 0), by if isinstance(by, str) else None,
                               num(av.get((y, int(at))) if pd.notna(at) else None, 0),
                               num(gs.get((y, int(at))) if pd.notna(at) else None, 0),
                               lab if isinstance(lab, str) else None]
                              for n, g, rk, sc, at, by, lab in zip(P["name"], P.grp, P["rank"], P.school, P.drafted_at,
                                                                   P.drafted_by, P.label)]
        W = weights(int(y))
        blends[str(int(y))] = {lab: [[g, round(w, 4)] for g, w in ws.items()] for lab, ws in W.items()
                               if not (len(ws) == 1)}
        if verbose:
            print(f"{y}: {len(P)} players", flush=True)
    meta = {"groups": GROUPS, "T": T_GRID.tolist(), "K": K_GRID.tolist(), "fw": FALL_WEIGHT, "off": OFF_BOARD_RANK,
            "scale": SCALE, "real_scale": REAL_SCALE, "blends": blends, "n_tables": len(blobs), "latest_av_season": int(dp.season.max())}
    data = json.dumps({"meta": meta, "teams": teams, "pools": pools}, separators=(",", ":"), ensure_ascii=False)
    b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
    html = TEMPLATE.read_text().replace("__DATA_GZ__", b64(data.encode())).replace("__TABLES_GZ__", b64(b"".join(blobs))) \
        .replace("__REAL_GZ__", b64(b"".join(rblobs)))
    if theme in ("light", "dark"):
        html = html.replace('<html lang="en">', f'<html lang="en" data-theme="{theme}">', 1)
    if verbose:
        print(f"{len(blobs)} tables; data {len(data) / 1e6:.1f} MB raw", flush=True)
    return html


def main():
    html = build_html(verbose=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(html)
    print(f"page {len(html) / 1e6:.2f} MB -> {DEST}")


if __name__ == "__main__":
    main()
