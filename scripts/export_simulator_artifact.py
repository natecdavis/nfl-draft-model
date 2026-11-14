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
from simulator import FALL_WEIGHT, TABLE_DIR, _key, _spec, get_table, pool, situation  # noqa: E402
from specialists import POS_TO_GROUP, SPEC_GROUPS  # noqa: E402
from outlook import GROUPS, OUT, RAW  # noqa: E402
from positions import weights  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "simulator_artifact.html"
DEST = OUT / "artifact" / "draft_simulator.html"
SCALE = 100_000  # tables stored as int16 = value (fraction of a title) x SCALE
REAL_SCALE = 1000  # realized titles stored as int16 = % of a title x REAL_SCALE; missing = -32768
REAL_CACHE = OUT / "sim_realized_v4.npz"  # v2: recent drafts scored "so far"; v3: real positions; v4: K/P
EXTRA_CACHE = OUT / "sim_realized_extra_v1.npz"  # picks a team could own if it nixes draft-night trades
NIGHT_CACHE = OUT / "sim_night_trades_v3.json"  # draft-night trades per team: verdicts, market prices


def realized_arrays(S, R, verbose=False) -> dict:
    """{"year_team": (picks, pool players) realized titles in % of one}, every draft with at least
    one season played; drafts after 2022 are scored on the seasons played so far."""
    stamp = max((OUT / f).stat().st_mtime for f in ["review_states.parquet", "review_picks.parquet",
                                                     "specialist_entrants.parquet", "specialist_engine.json"])
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


def _trade_order(night, y) -> list:
    """Draft-night trade ids in the order they happened: by date, and within a day by how picks flowed (a trade
    that hands on a pick comes after the trade that brought it in; trade ids aren't in time order within a day)."""
    import graphlib
    cur = night[(night.pick_season == y) & night.pick_number.notna()]
    ts = graphlib.TopologicalSorter()
    dates = night.groupby("trade_id").trade_date.first()
    for tid in dates.index:
        ts.add(tid)
    for k, g in cur.groupby("pick_number"):
        for a in g.itertuples():  # a: someone receives k; b: that team gives k on later
            for b in g.itertuples():
                if a.trade_id != b.trade_id and b.gave == a.received and dates[a.trade_id] <= dates[b.trade_id]:
                    ts.add(b.trade_id, a.trade_id)
    try:
        flow = list(ts.static_order())
    except graphlib.CycleError:
        flow = sorted(dates.index)
    rank = {t: i for i, t in enumerate(flow)}
    return sorted(dates.index, key=lambda t: (dates[t], rank[t]))


def night_trades(S, verbose=False) -> dict:
    """{"year": {"team": {"start": picks held when the draft began, "trades": [...]}}}: every draft-night
    trade the team made, in order, with what each side gave (picks this draft by number, future picks
    [years ahead, round], players by name), the trade calculator's verdict for this team (titles over 5
    seasons; pick-only trades; a pick it received counts as a QB pick if it took a QB there, as in the review),
    the model's value of each future pick, and the market price (src/market.py; pick-only trades). Cached."""
    stamp = max((OUT / f).stat().st_mtime for f in ["review_states.parquet", "review_trades.parquet", "market_chart.json"])
    if NIGHT_CACHE.exists() and NIGHT_CACHE.stat().st_mtime > stamp:
        return json.load(open(NIGHT_CACHE))
    import market
    from review import draft_night
    from trade import Context, Pick, evaluate
    ctx = Context()
    RT = pd.read_parquet(OUT / "review_trades.parquet")
    verdict = {(int(r.year), int(r.trade_id), r.team): float(r.titles_change) for r in RT.itertuples()}
    fut_cache, out = {}, {}
    for y in sorted(S.year.unique()):
        y = int(y)
        d, night = draft_night(y)
        sits = {r.team: situation(r) for _, r in S[S.year == y].iterrows()}
        owner0 = dict(zip(d.pick.astype(int), d.original_team))  # owner when the draft began (review.draft_night)
        start = {}
        for k, tm in owner0.items():
            start.setdefault(tm, []).append(k)
        start = {tm: sorted(v) for tm, v in start.items()}
        pos_at = dict(zip(d.pick, d.grp))
        Y = out.setdefault(str(y), {})
        for tm, sit in sits.items():
            Y[tm] = {"start": start.get(tm, []), "trades": []}
        for tid in _trade_order(night, y):
            g = night[night.trade_id == tid]
            for tm in sorted(set(g.gave) | set(g.received)):
                if tm not in sits:
                    continue
                partner = next(iter((set(g.gave) | set(g.received)) - {tm}), None)
                sit = sits[tm]

                def side(rows_, giving):
                    now = sorted(int(r.pick_number) for r in rows_.itertuples() if r.pick_season == y)
                    fut, futp = [], []
                    for r in rows_.itertuples():
                        if pd.notna(r.pick_season) and r.pick_season > y:
                            yr, rnd = int(r.pick_season - y), int(r.pick_round)
                            src = "own" if giving else {"start": sits.get(partner, {"start": 0.0})["start"]}
                            ck = (_key(sit, y), yr, rnd, json.dumps(src))
                            if ck not in fut_cache:
                                fut_cache[ck] = evaluate(ctx, sit, [], [Pick(rnd=rnd, year=yr, src=src)], n=4000,
                                                         seed=yr * 10 + rnd)["trade"] * 100
                            fut.append([yr, rnd, round(fut_cache[ck], 3)])
                            futp.append(Pick(rnd=rnd, year=yr, src=src))
                    players = [r.pfr_name for r in rows_.itertuples() if pd.isna(r.pick_season)]
                    return now, fut, futp, players
                gn, gf, gfp, gp = side(g[g.gave == tm], True)
                tn, tf, tfp, tp = side(g[g.received == tm], False)
                m = None
                if (gn or gf) and (tn or tf) and not (gp or tp):  # players: no price for the picks alone
                    r = market.price(sit, gn, tn, gfp, tfp, n=2000, seed=int(tid))
                    m = {k: (None if pd.isna(r[k]) else round(float(r[k]), 3)) for k in ("ratio", "ratio_lo", "ratio_hi",
                                                                                       "give_eq", "get_eq", "premium")}
                    m["moving_up"] = r["moving_up"]
                v = verdict.get((y, int(tid), tm))
                Y[tm]["trades"].append({"id": int(tid), "date": str(g.trade_date.iloc[0]), "partner": partner,
                                        "give": gn, "get": tn, "give_fut": gf, "get_fut": tf,
                                        "give_players": gp, "get_players": tp,
                                        "qb_get": [k for k in tn if pos_at.get(k) == "QB"],
                                        "verdict": None if v is None else round(v, 3), "market": m})
        if verbose:
            print(f"night trades {y} done", flush=True)
    json.dump(out, open(NIGHT_CACHE, "w"))
    return out


def realized_extra(S, R, night, verbose=False) -> dict:
    """Realized titles (as realized_arrays) at the picks a team could own if it nixes draft-night trades:
    picks it held when the draft began and picks in its trades, other than the picks it really made.
    {"year_team": (picks list, array (picks, pool players))}. Cached."""
    stamp = max(REAL_CACHE.stat().st_mtime, NIGHT_CACHE.stat().st_mtime)
    if EXTRA_CACHE.exists() and EXTRA_CACHE.stat().st_mtime > stamp:
        z = np.load(EXTRA_CACHE)
        return {k[:-2]: (z[k].tolist(), z[k[:-2] + "_a"]) for k in z.files if k.endswith("_k")}
    from review import RealizedV2
    from trade import Context
    RV, out, save = RealizedV2(Context()), {}, {}
    for y in sorted(int(y) for y in R.year.unique() if RV.seasons_played(int(y)) > 0):
        P = pool(y)
        g, at, nm = P.grp_record.tolist(), P.drafted_at.tolist(), P["name"].tolist()
        for _, r in S[S.year == y].iterrows():
            info = night.get(str(y), {}).get(r.team)
            if not info:
                continue
            real = set(int(k) for k in R[(R.year == y) & (R.team == r.team)].pick)
            allk = set(info["start"]) | {k for t in info["trades"] for k in t["give"] + t["get"]}
            ks = sorted(allk - real)
            if ks:
                a = np.array([RV.titles_many(situation(r), y, k, g, at, nm, partial=True) for k in ks])
                out[f"{y}_{r.team}"] = (ks, a)
                save[f"{y}_{r.team}_k"], save[f"{y}_{r.team}_a"] = np.array(ks), a
        if verbose:
            print(f"realized extra {y} done", flush=True)
    np.savez_compressed(EXTRA_CACHE, **save)
    return out


def _delta(arr) -> bytes:
    """A table (groups, 3 levels, T, K) as int16 (value x SCALE), differenced along K then T within each
    group and level (gzip packs the differences ~2.5x smaller); the page undoes it with running sums, in
    int16 arithmetic, so any wrap-around cancels exactly."""
    q = np.round(arr * SCALE)
    assert np.abs(q).max() < 32767
    q = q.astype(np.int64)
    d = np.diff(np.diff(q, axis=-1, prepend=0), axis=-2, prepend=0)
    return d.astype("<i2").tobytes()  # astype wraps mod 2^16, like the page's Int16Array


def _family_key(sit, year) -> str:
    """The family-units table for this situation (outcomes_v2.UNITS = "family"; built by
    OUTCOME_UNITS=family python src/simulator.py build)."""
    import outcomes_v2 as o
    prev, o.UNITS = o.UNITS, "family"
    try:
        return _key(sit, year)
    finally:
        o.UNITS = prev


def build_html(theme: str | None = None, verbose: bool = False) -> str:
    """The whole page as one HTML string, from the current review data and simulator tables.
    theme: "light" / "dark" pins the palette (e.g. to match the Streamlit app); None follows the viewer."""
    S = pd.read_parquet(OUT / "review_states.parquet")
    R = pd.read_parquet(OUT / "review_picks.parquet")
    from review import Careers
    car = Careers()  # starts / AV, undrafted players included
    dp = pd.read_parquet(RAW / "draft_picks.parquet")  # latest AV season
    dpos = {y: dict(zip(g.pick, g.position)) for y, g in dp.groupby("season")}
    num = lambda x, nd=3: None if pd.isna(x) else round(float(x), nd)

    real = realized_arrays(S, R, verbose)
    night = night_trades(S, verbose)
    extra = realized_extra(S, R, night, verbose)
    keys, blobs = {}, []
    fkeys, fblobs = {}, []  # family-units tables (positions only): the "depends on position pricing" label
    teams, rblobs, roff = {}, [], 0
    for _, r in S.iterrows():
        sit = situation(r)
        k = _key(sit, r.year)
        if k not in keys:
            z = np.load(TABLE_DIR / f"{k}.npz")
            sp = _spec(sit, r.year)  # kickers / punters (src/specialists.py)
            arr = np.stack([z[g] for g in GROUPS] + [sp[g] for g in SPEC_GROUPS])  # (groups, 3 levels, T, K)
            keys[k] = len(blobs)
            blobs.append(_delta(arr))
        fk = _family_key(sit, r.year)
        if fk not in fkeys:
            z = np.load(TABLE_DIR / f"{fk}.npz")
            fkeys[fk] = len(fblobs)
            fblobs.append(_delta(np.stack([z[g] for g in GROUPS])))
        mine = R[(R.year == r.year) & (R.team == r.team)].sort_values("pick").copy()
        # the team's own kickers / punters: their group and draft-night value (the review leaves them out)
        spos = mine.pick.map(dpos.get(int(r.year), {})).astype(str).str.upper().map(POS_TO_GROUP)
        if spos.notna().any():
            tab = get_table(sit, year=int(r.year))
            for i in spos.index[spos.notna()]:
                mine.at[i, "grp"] = spos[i]
                mine.at[i, "value_actual"] = tab.lookup(spos[i], 0, mine.at[i, "pick"], mine.at[i, "pick"]) * 100
        teams.setdefault(str(int(r.year)), {})[r.team] = {
            "qb": r.qb, "status": r.status, "strength": num(r.strength_prior, 1),
            "qb_from": None if pd.isna(r.qb_from) else int(r.qb_from), "p_need": num(r.p_need, 2),
            "table": keys[k], "ftable": fkeys[fk], "real": None,
            "picks": [[int(p.pick), int(p["round"]), p.original_team, int(bool(p.traded_on_night)), p.player, p.grp,
                       num(p.board_rank, 0), num(p.value_actual), num(p.real_team)] for _, p in mine.iterrows()],
            "start": night.get(str(int(r.year)), {}).get(r.team, {}).get("start", []),
            "trades": night.get(str(int(r.year)), {}).get(r.team, {}).get("trades", []),
        }
    played = {str(int(y)): int(n) for y, n in real.get("_seasons", [])}
    pool_cache = {}
    for yt, a in real.items():
        if yt.startswith("_"):
            continue
        y, tm = yt.split("_")
        if tm in teams.get(y, {}):
            T = teams[y][tm]
            kp = [pk[0] for pk in T["picks"]]
            if yt in extra:  # picks it could own by nixing draft-night trades, after its real picks
                kp += extra[yt][0]
                a = np.vstack([a, extra[yt][1]])
            q = np.where(np.isnan(a), -32768, np.clip(np.round(a * REAL_SCALE), -32767, 32767)).astype("<i2")
            T["real"], T["seasons"], T["kp"] = roff, played.get(y, 4), kp
            rblobs.append(q.tobytes()); roff += q.size
            # the review doesn't score drafts after 2022 or specialists: the team's own pick, from the same array
            P = pool_cache.setdefault(y, pool(int(y)))
            at = P.drafted_at.to_numpy()
            for j, pk in enumerate(T["picks"]):
                if int(y) > 2022 or (pk[8] is None and pk[5] in SPEC_GROUPS):
                    i = np.flatnonzero(at == pk[0])
                    pk[8] = num(a[j, i[0]]) if len(i) else None
    pools, blends = {}, {}
    for y in sorted(R.year.unique()):
        P = pool(int(y))
        pools[str(int(y))] = [[n, g, num(rk, 0), sc if isinstance(sc, str) else "", num(at, 0), by if isinstance(by, str) else None,
                               num(car.get("av", y, at, n), 0), num(car.get("gs", y, at, n), 0),
                               lab if isinstance(lab, str) else None]
                              for n, g, rk, sc, at, by, lab in zip(P["name"], P.grp, P["rank"], P.school, P.drafted_at,
                                                                   P.drafted_by, P.label)]
        W = weights(int(y))
        blends[str(int(y))] = {lab: [[g, round(w, 4)] for g, w in ws.items()] for lab, ws in W.items()
                               if not (len(ws) == 1)}
        if verbose:
            print(f"{y}: {len(P)} players", flush=True)
    rounds = {str(int(y)): g.groupby("round").pick.min().astype(int).tolist() for y, g in dp.groupby("season")}
    meta = {"rounds": rounds, "groups": GROUPS + SPEC_GROUPS, "T": T_GRID.tolist(), "K": K_GRID.tolist(), "fw": FALL_WEIGHT, "off": OFF_BOARD_RANK,
            "scale": SCALE, "real_scale": REAL_SCALE, "blends": blends, "n_tables": len(blobs), "fgroups": GROUPS, "latest_av_season": int(dp.season.max())}
    data = json.dumps({"meta": meta, "teams": teams, "pools": pools}, separators=(",", ":"), ensure_ascii=False)
    b64 = lambda b: base64.b64encode(gzip.compress(b, 9)).decode()
    html = TEMPLATE.read_text().replace("__DATA_GZ__", b64(data.encode())).replace("__TABLES_GZ__", b64(b"".join(blobs))) \
        .replace("__FTABLES_GZ__", b64(b"".join(fblobs))) \
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
