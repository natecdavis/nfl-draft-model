"""Complaint #6: draft-night trade-ups are valued as a typical pick at the slot, but a team trading on the clock
knows which players are on the board. Does valuing the pick as the player actually taken (his board rank, shaded
toward the pick as usual, and position: draft-day information) predict how real trades turned out better?

Trades: draft-night (review.DRAFT_START to +3 days) pick-only trades, two teams, 2014-2021, current picks and next
year's picks (scripts/future_pick_realized.trades). Each team's draft-day state from the past-drafts review.
Realized: each player's real seasons through the model's title machinery (as scripts/future_pick_realized.py), for
the team's situation.

Predictions (trade.evaluate, one per side per trade):
  slot    today: current picks as a typical player at the slot (a received pick valued as a QB if a QB was taken)
  clock   the trade's earliest current pick (the one on the clock when the deal is made) valued as the player
          taken there (trade.Pick talent = shaded board rank, pos = his position); other picks as today
  all     every current pick valued as the player taken there (later picks' players weren't known yet: an upper
          bound on what knowing the board could add)

Run: .venv/bin/python scripts/trade_target.py
"""
import os
import sys

os.environ.setdefault("OUTCOME_UNITS", "single")
sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import future_pick_realized as F  # noqa: E402
import review as Rv  # noqa: E402
from outlook import GROUPS, OUT, RAW  # noqa: E402
from pick_value import baseline_paths, pick_effect  # noqa: E402
from simulator import FALL_WEIGHT  # noqa: E402
from trade import CARRY, Context, Pick, evaluate  # noqa: E402

YEARS = range(2014, 2022)
VERSIONS = ["slot", "clock", "all"]
rng = np.random.default_rng(0)


def shade(r, k):
    if k <= r:
        return float(r)
    lt, lp = np.log(r + 10), np.log(k + 10)
    return float(np.exp(lt + FALL_WEIGHT * (lp - lt)) - 10)


def board_info():
    """(year, pick) -> (consensus rank or OFF_BOARD_RANK, decision position) of the player taken there."""
    out = {}
    for y in YEARS:
        picks, _ = Rv.draft_night(y)
        b = Rv.board(y, picks)
        on = {int(a): (float(r), g) for a, r, g in zip(b.drafted_at, b["rank"], b.grp_label) if pd.notna(a)}
        for k, g in zip(picks.pick, picks.grp):
            out[(y, int(k))] = on.get(int(k), (float(Rv.OFF_BOARD_RANK), g))
    return out


def main():
    F.load_states()
    ctx = Context()
    T = F.trades()
    T = T[T.season.isin(YEARS)]
    T = T[[pd.Timestamp(Rv.DRAFT_START[s]) <= d <= pd.Timestamp(Rv.DRAFT_START[s]) + pd.Timedelta(days=3)
           for s, d in zip(T.season, T.date)]].copy()
    L = F.labels()
    rat = ctx.params["ratings"]
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pos = {p: Rv.DRAFT_POS.get(x) for p, x in zip(dp.pfr_player_id, dp.position)}
    T["grp"] = [pos.get(p) for p in T.pfr_id]
    T["yr"] = (T.pick_season - T.season).astype(int)
    B = board_info()
    print(f"draft-night pick-only trades 2014-21: {T.trade_id.nunique()}")
    from outcomes_v2 import FAMILY, pay_factor, unit_factor
    rows = []
    for tid, g in T.groupby("trade_id"):
        S = int(g.season.iloc[0])
        cur = g[g.yr == 0]
        first = int(cur.pick_number.min()) if len(cur) else None
        for tm in sorted(set(g.gave) | set(g.received)):
            if (S, tm) not in F.STATES:
                continue
            sit_pred = F.STATES[(S, tm)]
            sit = dict(sit_pred, qb_from=1)  # outcomes scored as played (as future_pick_realized states)
            base = baseline_paths(sit, np.random.default_rng(0), 2000)
            b0 = ctx.sb(base).sum(axis=1).mean()

            def realized(rows_):
                tot = np.zeros((1, 5))
                for r in rows_.itertuples():
                    if r.pfr_id not in L.index or r.grp not in ctx.bar:
                        continue
                    row = L.loc[r.pfr_id]
                    y = np.array([row[f"y{k}"] for k in range(1, 5)]) / unit_factor(FAMILY.get(r.grp, r.grp), "single") * pay_factor(r.grp, None)
                    y5 = np.r_[y, y[-1]]
                    price = np.array([max(y5.max(), 1e-6)])
                    pts = pick_effect((y5 / price[0])[None, :], price, int(r.pick_number), r.grp, sit, ctx.cost, ctx.bar, ctx.rates)
                    if pts is None:
                        continue
                    if r.yr == 1:
                        tot[:, 1:] += pts[:, :4]
                    else:
                        tot += pts
                return (ctx.sb(base + tot).sum(axis=1).mean() - b0) * 100

            give, get = g[g.gave == tm], g[g.received == tm]
            row = {"trade_id": tid, "season": S, "team": tm, "realized": realized(get) - realized(give),
                   "up": bool(len(cur)) and first in set(get[get.yr == 0].pick_number.astype(int)),
                   "qb_first": bool(len(cur)) and B.get((S, first), (0, ""))[1] == "QB"}

            def to_picks(rows_, receiving, version):
                out = []
                for r in rows_.itertuples():
                    if r.yr != 0:
                        out.append(Pick(rnd=int(r.pick_round), year=1, src={"start": CARRY * 0.73 * rat[S - 1].get(r.gave, 0.0)}))
                        continue
                    k = int(r.pick_number)
                    known = version == "all" or (version == "clock" and k == first)
                    rk, gl = B.get((S, k), (float(Rv.OFF_BOARD_RANK), r.grp))
                    gl = gl if gl in GROUPS else (r.grp if r.grp in GROUPS else None)
                    if known and gl is not None:
                        out.append(Pick(k, pos=gl, talent=shade(min(rk, Rv.OFF_BOARD_RANK), k)))
                    else:
                        out.append(Pick(k, pos="QB" if receiving and r.grp == "QB" else "nonqb"))
                return out
            for v in VERSIONS:
                row[v] = evaluate(ctx, sit_pred, to_picks(give, False, v), to_picks(get, True, v), n=3000, seed=int(tid))["change"] * 100
            rows.append(row)
        print(f"trade {tid} done", flush=True) if len(rows) % 50 == 0 else None
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / "trade_target.parquet")
    report(R)


def report(R):
    print(f"\n{len(R)} team-sides, {R.trade_id.nunique()} trades. Predicted vs realized change in titles (% of one):")
    one = R.sort_values("team").drop_duplicates("trade_id")  # one side per trade (the other mirrors it)
    for lab, d in [("all trades, one side", one), ("trade-up side (got the earliest pick)", R[R.up]),
                   ("trade-ups, first pick a QB", R[R.up & R.qb_first]), ("trade-ups, non-QB", R[R.up & ~R.qb_first])]:
        parts = []
        for v in VERSIONS:
            c = np.corrcoef(d[v], d.realized)[0, 1]
            bs = []
            for _ in range(1000):
                s = d.sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
                bs.append(np.corrcoef(s[v], s.realized)[0, 1])
            slope = np.polyfit(d[v], d.realized, 1)[0]
            parts.append(f"{v} corr {c:+.2f} ({np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f}) mean {d[v].mean():+.2f} slope {slope:.2f}")
        print(f"  {lab} (n={len(d)}, realized mean {d.realized.mean():+.2f}):\n    " + "\n    ".join(parts))
    d = R[R.up]
    for v in VERSIONS[1:]:
        diff = (d[v] - d.realized).abs() - (d.slot - d.realized).abs()
        bs = [diff.sample(frac=1, replace=True, random_state=i).mean() for i in range(2000)]
        print(f"  trade-ups: |error| {v} minus slot {diff.mean():+.2f} ({np.percentile(bs, 2.5):+.2f} to {np.percentile(bs, 97.5):+.2f})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["report"]:
        report(pd.read_parquet(OUT / "trade_target.parquet"))
    else:
        main()
