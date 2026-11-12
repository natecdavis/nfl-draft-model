"""How did real trades involving next year's picks turn out? (follow-up to scripts/future_pick_test.py)

Trades: draft-window (March-May) pick-only trades, two teams, no conditional picks, current picks drafted 2006-2021
and next year's picks drafted by 2022 (labels need four seasons). Each pick -> the player drafted with it
(nflverse trade log), his seasons 1-4 value (pm.LABEL_FILE, % of cap), pick number.

1. Player value, no title model: next year's 1sts / 2nds that were traded, realized value vs (a) the average
   drafted player at the same actual slot, (b) the average player at the market's equivalent current slot
   (next 1st ~ #42, next 2nd ~ #74; scripts/future_pick_test.py), (c) the model's expected value at the slot
   predicted from the original owner's strength. Also predicted vs actual slot.
2. Realized titles per team per trade: each player's real seasons through the model's title machinery
   (pick_value.pick_effect, single units as in review.RealizedV2), next year's players arriving a season later,
   for the team's situation at trade time. Compared with the model's pre-trade verdict (trade.evaluate, as in
   trade.real_trades). This checks the model's player expectations, not its title-window assumption.

Run:  python scripts/future_pick_realized.py          (2014-21 with each team's review state:  ... states)
"""
import os
import sys

os.environ.setdefault("OUTCOME_UNITS", "single")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
import prospect_model as pm  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from pick_value import baseline_paths, pick_effect  # noqa: E402
from trade import CARRY, Context, Pick, _slots, evaluate  # noqa: E402

TEAM = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
MARKET_EQ = {1: 42, 2: 74}


def trades():
    t = pd.read_parquet(RAW / "trades.parquet")
    for c in ("gave", "received"):
        t[c] = t[c].replace(TEAM)
    t["date"] = pd.to_datetime(t.trade_date)
    po = t.groupby("trade_id").pick_season.apply(lambda s: s.notna().all())
    t = t[t.trade_id.isin(po[po].index) & t.date.dt.month.between(3, 5) & t.season.between(2006, 2021)]
    keep = []
    for tid, g in t.groupby("trade_id"):
        S = int(g.season.iloc[0])
        if (g.conditional.fillna(0) > 0).any() or len(set(g.gave) | set(g.received)) != 2:
            continue
        if not ((g.pick_season == S) & g.pick_number.notna()).any() or (g.pick_season > S + 1).any() or (g.pick_season < S).any():
            continue
        if g.pick_number.isna().any():
            continue
        keep.append(tid)
    return t[t.trade_id.isin(keep)].copy()


def labels():
    L = pd.read_parquet(pm.LABEL_FILE).dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    for k in range(1, 5):
        L[f"y{k}"] = L[f"value_y{k}"].fillna(0).clip(lower=0)
    return L


def slot_curve(L):
    """Average seasons 1-4 value by overall pick, drafted classes 2006-2022 (smoothed)."""
    d = L[L.draft_year.between(2006, 2022) & L.drafted.astype(bool)].dropna(subset=["overall_pick"])
    v = (d.y1 + d.y2 + d.y3 + d.y4).groupby(d.overall_pick.astype(int)).mean().reindex(range(1, 263))
    return v.rolling(9, center=True, min_periods=1).mean() * 100


def main():
    ctx = Context()
    T = trades()
    L = labels()
    curve = slot_curve(L)
    rat = ctx.params["ratings"]
    T["value"] = [(L.loc[p, ["y1", "y2", "y3", "y4"]].sum() * 100) if p in L.index else 0.0 for p in T.pfr_id]
    from review import DRAFT_POS
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pos = {p: DRAFT_POS.get(x) for p, x in zip(dp.pfr_player_id, dp.position)}
    T["grp"] = [pos.get(p) for p in T.pfr_id]
    T["yr"] = (T.pick_season - T.season).astype(int)
    print(f"Trades: {T.trade_id.nunique()}; with next year's picks: {T[T.yr == 1].trade_id.nunique()}; "
          f"next-year picks: {(T.yr == 1).sum()} (1sts {((T.yr == 1) & (T.pick_round == 1)).sum()})")

    # 1. player value of next year's picks
    rng = np.random.default_rng(0)
    base0 = np.zeros((4000, 5))
    F = T[T.yr == 1].copy()
    F["pred_slot"] = [_slots(Pick(rnd=int(r.pick_round), year=1, src={"start": CARRY * 0.73 * rat.get(r.season, {}).get(r.gave, 0.0)}),
                             {}, base0, rng).mean() if r.season in rat else np.nan for r in F.itertuples()]
    # owner's strength known at trade time = previous season
    F["pred_slot"] = [_slots(Pick(rnd=int(r.pick_round), year=1, src={"start": CARRY * 0.73 * rat.get(r.season - 1, {}).get(r.gave, 0.0)}),
                             {}, base0, rng).mean() for r in F.itertuples()]
    F["at_slot"] = F.pick_number.astype(int).map(curve)
    F["at_pred"] = F.pred_slot.round().astype(int).clip(1, 262).map(curve)
    F["at_market"] = F.pick_round.astype(int).map(lambda r: curve.get(MARKET_EQ.get(r, 32 * r + 10)))
    print("\n1. Next year's picks that were traded: realized value (seasons 1-4, % of cap, sum over picks):")
    for r in (1, 2):
        g = F[F.pick_round == r]
        print(f"  round {r}: n={len(g)}, slot predicted {g.pred_slot.mean():.0f} vs actual {g.pick_number.mean():.0f} "
              f"(corr {np.corrcoef(g.pred_slot, g.pick_number)[0, 1]:.2f}) | realized {g.value.sum():.1f} vs avg player at actual slot "
              f"{g.at_slot.sum():.1f} (x{g.value.sum() / g.at_slot.sum():.2f}), at predicted slot {g.at_pred.sum():.1f}, "
              f"at the market's equivalent #{MARKET_EQ[r]} now {g.at_market.sum():.1f} (x{g.value.sum() / g.at_market.sum():.2f})")
    bs = [(lambda g: g.value.sum() / g.at_market.sum())(F[F.pick_round == 1].sample(frac=1, replace=True, random_state=i)) for i in range(2000)]
    print(f"  round 1 realized / market-equivalent value: 95% {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}")

    # 2. realized titles per team per trade
    from outcomes_v2 import FAMILY, pay_factor, unit_factor
    rows = []
    for tid, g in T.groupby("trade_id"):
        S = int(g.season.iloc[0])
        if (S - 1) not in rat:
            continue
        teams = sorted(set(g.gave) | set(g.received))
        has_fut = (g.yr == 1).any()
        if STATES and S not in STATES_YEARS:
            continue
        for tm in teams:
            prev = rat[S - 1].get(tm, 0.0)
            sit = {"start": CARRY * 0.73 * prev, "shift": {}, "qb_from": 1}
            sit_pred = sit
            if STATES:  # the model's draft-day view (review states); outcomes still scored as played
                if (S, tm) not in STATES:
                    continue
                sit_pred = STATES[(S, tm)]
                sit = dict(sit_pred, qb_from=1)
            base = baseline_paths(sit, np.random.default_rng(0), 2000)
            b0 = ctx.sb(base).sum(axis=1).mean()

            def realized(rows_):
                tot = np.zeros((1, 5))
                for r in rows_.itertuples():
                    if r.pfr_id not in L.index:
                        continue
                    row = L.loc[r.pfr_id]
                    grp = r.grp if r.grp in ctx.bar else None
                    if grp is None:  # specialists / unknown: not valued
                        continue
                    y = np.array([row[f"y{k}"] for k in range(1, 5)]) / unit_factor(FAMILY.get(grp, grp), "single") * pay_factor(grp, None)
                    y5 = np.r_[y, y[-1]]
                    price = np.array([max(y5.max(), 1e-6)])
                    pts = pick_effect((y5 / price[0])[None, :], price, int(r.pick_number), grp, sit, ctx.cost, ctx.bar, ctx.rates)
                    if pts is None:
                        continue
                    if r.yr == 1:
                        tot[:, 1:] += pts[:, :4]
                    else:
                        tot += pts
                return (ctx.sb(base + tot).sum(axis=1).mean() - b0) * 100

            give, get = g[g.gave == tm], g[g.received == tm]
            real = realized(get) - realized(give)

            def to_picks(rows_, receiving):
                out = []
                for r in rows_.itertuples():
                    if r.yr == 0:
                        # as in trade.real_trades: a pick received is valued by who was drafted there (QB or not);
                        # a pick given up as a typical non-QB
                        out.append(Pick(int(r.pick_number), pos="QB" if receiving and r.grp == "QB" else "nonqb"))
                    else:
                        out.append(Pick(rnd=int(r.pick_round), year=1,
                                        src={"start": CARRY * 0.73 * rat[S - 1].get(r.gave, 0.0)}))
                return out
            pred = evaluate(ctx, sit_pred, to_picks(give, False), to_picks(get, True), n=3000, seed=int(tid))["change"] * 100
            net_fut = (get.yr == 1).sum() - (give.yr == 1).sum()
            rows.append({"trade_id": tid, "season": S, "team": tm, "start": sit["start"], "has_future": has_fut,
                         "qb_need": sit_pred.get("qb_from") is not None,
                         "net_future_picks": net_fut, "predicted": pred, "realized": real,
                         "raw_value_net": get.value.sum() - give.value.sum(),
                         "qb_up": bool(((g.yr == 0) & (g.grp == "QB") & (g.received == tm)).any())})
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / ("future_pick_realized_states.parquet" if STATES else "future_pick_realized.parquet"))
    print("\n2. Realized vs predicted change in titles (% of one), per team per trade:")
    qbt = set(R[R.qb_up].trade_id)
    R["qb_trade"] = R.trade_id.isin(qbt)
    for lab, m in [("all trades", R.index == R.index), ("trades with next year's picks", R.has_future),
                   ("current picks only", ~R.has_future), ("no QB drafted with a current pick", ~R.qb_trade),
                   ("next year's picks, no QB", R.has_future & ~R.qb_trade)]:
        x = R[m]
        print(f"  {lab:32s} n={len(x):4d}  corr(pred, realized) {np.corrcoef(x.predicted, x.realized)[0, 1]:+.2f}  "
              f"slope {np.polyfit(x.predicted, x.realized, 1)[0]:+.2f}")
    one = R.drop_duplicates("trade_id")
    for lab, x in [("all", one), ("next year's picks", one[one.has_future]), ("no QB", one[~one.qb_trade]),
                   ("next year's picks, no QB", one[one.has_future & ~one.qb_trade])]:
        print(f"  one side per trade, {lab:26s} n={len(x):3d}: corr {np.corrcoef(x.predicted, x.realized)[0, 1]:+.2f} "
              f"(expected if exactly right ~{x.predicted.std() / x.realized.std():+.2f}, se ~{1 / np.sqrt(len(x)):.2f})")
    F2 = R[R.has_future & (R.net_future_picks != 0)].copy()
    F2["role"] = np.where(F2.net_future_picks < 0, "sold future", "bought future")
    F2["strength"] = pd.cut(F2.start, [-20, -1.5, 1.5, 20], labels=["weaker", "average", "stronger"])
    rng = np.random.default_rng(1)

    def ci(x):
        x = np.asarray(x)
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"
    print("\n  Teams that sold / bought next year's picks (net), by strength at trade time:")
    for (role, st), g in F2.groupby(["role", "strength"], observed=True):
        print(f"   {role:13s} {st:8s} n={len(g):3d}  predicted {g.predicted.mean():+.2f}  realized {ci(g.realized)}  "
              f"raw value net {g.raw_value_net.mean():+.1f}% cap")
    for role, g in F2[~F2.qb_trade].groupby("role"):
        print(f"   {role:13s} no-QB    n={len(g):3d}  predicted {g.predicted.mean():+.2f}  realized {ci(g.realized)}  "
              f"raw value net {ci(g.raw_value_net)}% cap")
    for role, g in F2.groupby("role"):
        print(f"   {role:13s} all      n={len(g):3d}  predicted {g.predicted.mean():+.2f}  realized {ci(g.realized)}  "
              f"raw value net {ci(g.raw_value_net)}% cap")


STATES, STATES_YEARS = {}, set()


def load_states():
    """Each team's draft-day state from the past-drafts review (2014+): strength, scheduled shifts and QB need
    (qb_from None = no QB need in the window, so a drafted QB adds nothing in the model)."""
    from simulator import situation
    st = pd.read_parquet(OUT / "review_states.parquet")
    STATES.update({(int(r.year), r.team): situation(r) for _, r in st.iterrows()})
    STATES_YEARS.update(int(y) for y in st.year.unique())


if __name__ == "__main__":
    if sys.argv[1:2] == ["states"]:
        load_states()
    main()
