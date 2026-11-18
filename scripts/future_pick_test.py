"""Future picks: how much does the trade market discount next year's picks, and how does that compare with the model?

Model (src/trade.py): a future pick's player arrives a year later, so one fewer of his seasons falls inside the
5-season title window (the fifth-year option season drops out), and its slot is drawn from the original owner's
likely finish. The discount therefore depends on the team's situation.

Market: draft-window trades (March-May) made only of picks, two teams, no conditional picks, 2002-2025 (nflverse).
Value chart for this year's picks v(k) = exp(-lam * (k-1)^beta) (v(1) = 1), a premium c paid by the team
moving up (the one receiving the best current pick), and future picks worth delta x E[v(slot)], the slot drawn
from the original owner's previous-season strength (same rule as the model). Fit jointly by least squares on
log(value the up team gives) - log(value it gets) - c; 95% ranges by bootstrapping trades.
  delta_1R1   next year's 1st          delta_1L  next year's rounds 2-7          delta_2  two years out (few)
Reported in "equivalent current pick": the pick number this year worth the same as next year's 1st / 2nd / 3rd
from a middle team, to the market and to the model in each situation.

Run:  python scripts/future_pick_test.py          (chart only, for the app:  ... chart)
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, "src")
from outlook import OUT, RAW  # noqa: E402
from trade import CARRY, SITUATIONS, Context, Pick, _slots, evaluate, trade_log  # noqa: E402

TEAM = {"STL": "LA", "SD": "LAC", "OAK": "LV"}


def load_trades(ctx):
    t = trade_log()
    for c in ("gave", "received"):
        t[c] = t[c].replace(TEAM)
    t["date"] = pd.to_datetime(t.trade_date)
    picks_only = t.groupby("trade_id").pick_season.apply(lambda s: s.notna().all())
    t = t[t.trade_id.isin(picks_only[picks_only].index) & t.date.dt.month.between(3, 5)]
    rat = ctx.params["ratings"]
    rng = np.random.default_rng(0)
    base = np.zeros((4000, 5))
    slot_cache = {}
    out = []
    for tid, g in t.groupby("trade_id"):
        S = int(g.season.iloc[0])
        if (g.conditional.fillna(0) > 0).any() or len(set(g.gave) | set(g.received)) != 2:
            continue
        cur = g[(g.pick_season == S) & g.pick_number.notna()]
        if cur.empty or ((g.pick_season > S) & g.pick_round.isna()).any() or (g.pick_season < S).any():
            continue
        if ((g.pick_season == S) & g.pick_number.isna()).any():
            continue
        up = cur.loc[cur.pick_number.idxmin(), "received"]
        assets = []
        for _, r in g.iterrows():
            yr = int(r.pick_season - S)
            if yr == 0:
                item = ("cur", int(r.pick_number), None)
            else:
                prev = rat.get(S - 1, {}).get(r.gave, 0.0)
                key = (yr, int(r.pick_round), round(prev, 1))
                if key not in slot_cache:
                    slot_cache[key] = _slots(Pick(rnd=int(r.pick_round), year=yr, src={"start": CARRY * 0.73 * prev}),
                                             {}, base, rng)
                item = (f"fut{min(yr, 2)}", int(r.pick_round), slot_cache[key])
            out.append({"trade_id": tid, "season": S, "side": "up_gets" if r.received == up else "up_gives",
                        "kind": item[0], "number": item[1], "slots": item[2]})
    A = pd.DataFrame(out)
    has_fut = A.groupby("trade_id").kind.apply(lambda k: (k != "cur").any())
    print(f"Trades: {A.trade_id.nunique()} ({has_fut.sum()} with future picks; "
          f"{A[A.kind == 'fut1'].trade_id.nunique()} next-year, {A[A.kind == 'fut2'].trade_id.nunique()} two years out)")
    return A


def chart(k, lam, beta):
    return np.exp(-lam * np.maximum(np.asarray(k, float) - 1, 0) ** beta)


class Packed:
    """Trades as arrays for fast fitting."""
    def __init__(self, A):
        both = A.groupby("trade_id").side.nunique() == 2
        A = A[A.trade_id.isin(both[both].index)].reset_index(drop=True)
        self.A = A
        self.ids, self.tix = np.unique(A.trade_id.astype(str), return_inverse=True)
        self.gives = (A.side == "up_gives").to_numpy()
        self.cur = (A.kind == "cur").to_numpy()
        self.num = A.number.to_numpy(float)
        fut = ~self.cur
        self.F = np.stack([A.slots[i] for i in np.flatnonzero(fut)]) if fut.any() else np.zeros((0, 1))
        k = A.kind[fut].to_numpy(); r = A.number[fut].to_numpy()
        self.fcode = np.where(k == "fut2", 2, np.where(r == 1, 0, 1))

    def subset(self, trade_ids):
        """Bootstrap: rows of the chosen trades (repeats allowed), relabelled."""
        new = PackedView(self, trade_ids)
        return new


class PackedView:
    def __init__(self, P, trade_ids):
        rows, tix = [], []
        for j, t in enumerate(trade_ids):
            rr = P.rows_of[t]
            rows.append(rr); tix.append(np.full(len(rr), j))
        rows = np.concatenate(rows)
        self.tix = np.concatenate(tix); self.ids = np.arange(len(trade_ids))
        self.gives, self.cur, self.num = P.gives[rows], P.cur[rows], P.num[rows]
        fpos = np.cumsum(~P.cur) - 1
        fr = rows[~P.cur[rows]]
        self.F, self.fcode = P.F[fpos[fr]], P.fcode[fpos[fr]]


def residuals(theta, P):
    lam, beta, c, d1r1, d1l, d2 = theta
    v = np.zeros(len(P.cur))
    v[P.cur] = chart(P.num[P.cur], lam, beta)
    if (~P.cur).any():
        ev = chart(P.F, lam, beta).mean(axis=1)
        v[~P.cur] = ev * np.array([d1r1, d1l, d2])[P.fcode]
    n = len(P.ids)
    gv = np.bincount(P.tix[P.gives], v[P.gives], minlength=n)
    gt = np.bincount(P.tix[~P.gives], v[~P.gives], minlength=n)
    return np.log(gv) - np.log(gt) - c


def fit(P, x0=(0.15, 0.6, 0.05, 0.6, 0.6, 0.4)):
    r = least_squares(residuals, x0, args=(P,), bounds=([1e-3, 0.2, -1, 0.05, 0.05, 0.02], [2, 1.5, 1, 3, 3, 3]))
    return r.x


def equivalent(v_target, lam, beta):
    k = np.arange(1, 260)
    return float(np.interp(-v_target, -chart(k, lam, beta), k))


def model_side(ctx):
    rows = []
    ks = [1, 3, 6, 10, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200]
    for name, sit in SITUATIONS.items():
        cur = np.array([evaluate(ctx, sit, [], [Pick(k)], n=6000, seed=k)["trade"] for k in ks])
        cur = np.maximum.accumulate(cur[::-1])[::-1]  # monotone in pick (noise)
        for rnd in (1, 2, 3):
            for src in ("Middle of the pack", "own"):
                v = evaluate(ctx, sit, [], [Pick(rnd=rnd, year=1, src=src)], n=6000, seed=100 + rnd)["trade"]
                vnow = evaluate(ctx, sit, [], [Pick(rnd=rnd, year=0, src=src, number=16 + 32 * (rnd - 1))], n=6000,
                                seed=200 + rnd)["trade"]
                eq = float(np.interp(-v, -cur, ks))
                rows.append({"situation": name, "round": rnd, "source": src, "titles_future": v * 100,
                             "titles_same_slot_now": vnow * 100, "ratio": v / vnow if vnow > 0 else np.nan,
                             "equivalent_current_pick": eq})
    return pd.DataFrame(rows)


def main():
    ctx = Context()
    A = load_trades(ctx)
    P = Packed(A)
    P.rows_of = {t: np.flatnonzero(P.ids[P.tix] == t) for t in P.ids}
    print(f"Trades with both sides recorded: {len(P.ids)}")
    th = fit(P)
    lam, beta, c, d1r1, d1l, d2 = th
    print(f"\nMarket chart: v(k) = exp(-{lam:.3f} (k-1)^{beta:.3f}); v(10) {chart(10, lam, beta):.2f}, v(32) "
          f"{chart(32, lam, beta):.2f}, v(64) {chart(64, lam, beta):.2f}, v(100) {chart(100, lam, beta):.3f}; up-team premium "
          f"x{np.exp(c):.2f}")
    rng = np.random.default_rng(1)
    B = []
    for _ in range(300):
        s = rng.choice(P.ids, len(P.ids), replace=True)
        B.append(fit(PackedView(P, s), x0=th))
    B = np.array(B)
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    for i, nm in [(3, "next year's 1st (delta_1R1)"), (4, "next year's rounds 2-7 (delta_1L)"), (5, "two years out (delta_2)")]:
        print(f"  {nm:36s} {th[i]:.2f} (95% {lo[i]:.2f}-{hi[i]:.2f}) x E[value of its slot now]")
    # equivalents for a middle team's future picks
    base = np.zeros((20000, 5))
    print("\nMarket: equivalent pick this year for next year's pick from a middle team (95% range):")
    for rnd in (1, 2, 3):
        sl = _slots(Pick(rnd=rnd, year=1, src="Middle of the pack"), {}, base, np.random.default_rng(rnd))
        def eq(t):
            d = t[3] if rnd == 1 else t[4]
            return equivalent(d * chart(sl, t[0], t[1]).mean(), t[0], t[1])
        e = eq(th)
        eb = np.percentile([eq(b) for b in B], [2.5, 97.5])
        print(f"  next year's round {rnd} (avg slot {sl.mean():.0f}): like #{e:.0f} now (95% #{eb[0]:.0f}-#{eb[1]:.0f})")
    M = model_side(ctx)
    print("\nModel: next year's pick vs the same expected slot now, and the equivalent pick this year:")
    print(M.round(2).to_string(index=False))
    A.drop(columns="slots").to_parquet(OUT / "future_pick_trades.parquet")
    M.to_parquet(OUT / "future_pick_model.parquet")
    pd.DataFrame(B, columns=["lam", "beta", "c", "d1r1", "d1l", "d2"]).to_parquet(OUT / "future_pick_boot.parquet")
    save_chart(th, B, len(P.ids))


def save_chart(th, B, n):
    """The market chart for the app's Trade offer tab (src/market.py): point fit and bootstrap draws."""
    import json
    keys = ["lam", "beta", "c", "d1r1", "d1l", "d2"]
    json.dump({"fit": dict(zip(keys, map(float, th))), "boot": [dict(zip(keys, map(float, b))) for b in B],
               "n_trades": int(n), "years": "2002-2025"}, open(OUT / "market_chart.json", "w"))


def chart_only():
    """Refit the market chart and save it (no model side): python scripts/future_pick_test.py chart"""
    A = load_trades(Context())
    P = Packed(A)
    P.rows_of = {t: np.flatnonzero(P.ids[P.tix] == t) for t in P.ids}
    th = fit(P)
    rng = np.random.default_rng(1)
    B = np.array([fit(PackedView(P, rng.choice(P.ids, len(P.ids), replace=True)), x0=th) for _ in range(300)])
    save_chart(th, B, len(P.ids))
    print(dict(zip(["lam", "beta", "c", "d1r1", "d1l", "d2"], th.round(3))))


if __name__ == "__main__":
    chart_only() if sys.argv[1:2] == ["chart"] else main()
