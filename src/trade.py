"""Trade calculator: is a pick trade worth it, in Super Bowl odds, for this team?

A trade is two packages of picks: what the team gives and what it gets. Each
package is added to the team's 5-season strength path (pick_value.py) all at once,
because gains combine non-linearly on the title curve. The answer is the change
in expected Super Bowls over the same 5 seasons.

Picks:
  Pick(10)                       this year's #10, typical non-QB
  Pick(3, pos="QB")              this year's #3, used on a QB
  Pick(rnd=1, year=1)            the team's own first next year
  Pick(rnd=1, year=1, src="middle")  next year's first from a middle-of-the-pack team

Future picks: the slot is unknown, so it's drawn from where the source team is
likely to finish. For the team's own pick, that's tied to the same simulated
season it plays, so a bad season gives a high pick. A player drafted a year later
helps only in the later seasons of the 5-season window.

Also: a check on real pick-for-pick trades 2013-2025, scored for each team in its
own situation at the time (strength from the previous season), to see whether
trading up at market prices tends to gain or lose title odds.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from outlook import GROUPS, OUT, RAW, star_thresholds
from pick_value import (CARRY, SITUATIONS, decay, START_SD, baseline_paths, pick_effect,
                        sample_outcomes, strength_curve)
from simulate import league_params, load_schedules
from value import rookie_cost_curve

TRUE_SD = 5.3
LUCK_SD = TRUE_SD * np.sqrt(1 / 0.73 - 1)  # season luck around true strength (simulate.py)


@dataclass
class Pick:
    number: int | np.ndarray | None = None  # overall pick (or one per simulation), this year
    pos: str = "nonqb"            # "QB", "nonqb" (typical mix), or a position group
    year: int = 0                 # 0 = this draft, 1 = next year, 2 = the year after
    rnd: int | None = None        # round, for future picks
    src: str | dict = "own"       # future picks: "own", a situation name, or "typical"


class Context:
    """Everything the calculator needs, loaded once."""
    def __init__(self):
        sched = load_schedules()
        teams_df = pd.read_parquet(RAW / "teams.parquet")
        self.params = league_params(sched)
        curve = strength_curve(sched, teams_df, self.params)
        self.sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)
        c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
        self.cost = rookie_cost_curve(c)
        self.bar = star_thresholds(c).xs(2026, level="season").to_dict()
        self.lib = pd.read_parquet(OUT / "value_library.parquet")
        self.curves = pd.read_parquet(OUT / "outlook_curves.parquet")
        self.rates = json.load(open(OUT / "bridge.json"))["preseason"]
        out = pd.read_parquet(OUT / "pick_outcomes.parquet")
        w = out[out.grp.isin(GROUPS) & (out.grp != "QB")].grp.value_counts(normalize=True)
        self.nonqb_groups, self.nonqb_w = w.index.to_numpy(), w.to_numpy()


# Compensatory picks sit at the end of rounds 3-7, so later rounds start later than 32 x (round - 1): a future
# round-r pick is numbered this many spots later (mean net extra picks per round 2017-2026, after forfeits, from
# nflverse draft_picks; round 3 +7, 4 +5, 5 +6, 6 +8, 7 +7, cumulative). Barely moves the model's values
# (late picks are worth about the same 10-25 spots apart) but lowers late future picks' market prices 12-32%.
COMP_OFFSET = {4: 7, 5: 12, 6: 18, 7: 26}


def _slots(pick: Pick, sit: dict, base: np.ndarray, rng) -> np.ndarray:
    """Overall pick number per simulation."""
    n = len(base)
    if pick.year == 0:  # a fixed slot, or one slot per simulation
        return np.broadcast_to(np.asarray(pick.number), (n,)).copy()
    # strength in the season before that draft
    if pick.src == "own":
        x = base[:, pick.year - 1]
    else:
        start = SITUATIONS[pick.src]["start"] if isinstance(pick.src, str) and pick.src in SITUATIONS else (
            pick.src["start"] if isinstance(pick.src, dict) else 0.0)
        sd = TRUE_SD if pick.src == "typical" else START_SD
        x = rng.normal(start * decay(pick.year - 1), sd, n)
    observed = x + rng.normal(0, LUCK_SD, n)
    others = rng.normal(0, np.hypot(TRUE_SD, LUCK_SD), (n, 31))
    slot = 1 + (others < observed[:, None]).sum(axis=1)  # worst record picks first
    return slot + 32 * (pick.rnd - 1) + COMP_OFFSET.get(pick.rnd, 0)


def package_points(ctx: Context, sit: dict, picks: list[Pick], base: np.ndarray, rng) -> np.ndarray:
    """(n, 5) points of team strength a package adds each season."""
    n = len(base)
    total = np.zeros((n, 5))
    for p in picks:
        numbers = _slots(p, sit, base, rng)
        if p.pos == "nonqb":
            grps = rng.choice(ctx.nonqb_groups, size=n, p=ctx.nonqb_w)
        else:
            grps = np.full(n, p.pos)
        s = dict(sit)
        if s.get("qb_from") is None:
            s["qb_from"] = 99  # no QB need: a drafted QB doesn't play in the window
        s["qb_from"] = max(1, s["qb_from"] - p.year)
        keys = pd.DataFrame({"g": grps, "k": numbers})
        for (g, k), idx in keys.groupby(["g", "k"]).groups.items():
            idx = np.asarray(idx)
            shares, price = sample_outcomes(ctx.lib, ctx.curves, g, int(k), rng, n=len(idx))
            pts = pick_effect(shares, price, int(k), g, s, ctx.cost, ctx.bar, ctx.rates)
            if p.year:
                total[idx, p.year:] += pts[:, : 5 - p.year]
            else:
                total[idx] += pts
    return total


def evaluate(ctx: Context, sit: dict | str, give: list[Pick], get: list[Pick], n=20000, seed=0) -> dict:
    """Change in expected Super Bowls over 5 seasons from making the trade."""
    sit = SITUATIONS[sit] if isinstance(sit, str) else sit
    rng = np.random.default_rng(seed)
    base = baseline_paths(sit, rng, n)
    t_give = ctx.sb(base + package_points(ctx, sit, give, base, rng)).sum(axis=1)
    t_get = ctx.sb(base + package_points(ctx, sit, get, base, rng)).sum(axis=1)
    d = t_get - t_give
    return {"change": d.mean(), "se": d.std() / np.sqrt(n), "base": ctx.sb(base).sum(axis=1).mean(),
            "keep": t_give.mean() - ctx.sb(base).sum(axis=1).mean(),
            "trade": t_get.mean() - ctx.sb(base).sum(axis=1).mean()}


EXAMPLES = [
    ("Trade up for a QB, Mahomes-style (2017 KC price: #27, #91, next year's 1st for #10)",
     ["Contender, veteran QB", "Middle of the pack"],
     [Pick(27), Pick(91), Pick(rnd=1, year=1)], [Pick(10, pos="QB")]),
    ("Trade up for a QB, Lance-style (2021 SF price: #12, next two 1sts, next year's 3rd for #3)",
     ["Middle of the pack", "Contender, veteran QB"],
     [Pick(12), Pick(rnd=1, year=1), Pick(rnd=1, year=2), Pick(rnd=3, year=1)], [Pick(3, pos="QB")]),
    ("Trade up for a non-QB (#20 + #52 for #10)",
     list(SITUATIONS), [Pick(20), Pick(52)], [Pick(10)]),
    ("Trade down (#10 for #20 + #52) - the mirror image", list(SITUATIONS), [Pick(10)], [Pick(20), Pick(52)]),
    ("Sell next year's own 1st for this year's #30",
     list(SITUATIONS), [Pick(rnd=1, year=1)], [Pick(30)]),
    ("Trade #8 for #20 + next year's 1st from a middle team",
     list(SITUATIONS), [Pick(8)], [Pick(20), Pick(rnd=1, year=1, src="middle")]),
]


def run_examples(ctx):
    rows = []
    for title, sits, give, get in EXAMPLES:
        for sname in sits:
            r = evaluate(ctx, sname, give, get)
            rows.append({"trade": title, "situation": sname, "change": r["change"] * 100, "se": r["se"] * 100,
                         "keep": r["keep"] * 100, "trade_pkg": r["trade"] * 100})
    df = pd.DataFrame(rows)
    for title, g in df.groupby("trade", sort=False):
        print(title)
        for _, r in g.iterrows():
            verdict = "GAIN" if r.change > 2 * r.se else ("LOSS" if r.change < -2 * r.se else "about even")
            print(f"   {r.situation:27s} {r.change:+6.2f}% of a title (±{2 * r.se:.2f})  {verdict}"
                  f"   [keep: {r.keep:5.2f}, trade: {r.trade_pkg:5.2f}]")
        print()
    df.to_csv(OUT / "trade_examples.csv", index=False)


# ---------- real trades ----------

def real_trades(ctx, n=4000):
    """Pick-only trades made before or during the 2013-2025 drafts, scored for both teams."""
    t = pl.read_parquet(RAW / "trades.parquet").to_pandas()
    for col in ["gave", "received"]:
        t[col] = t[col].replace({"STL": "LA", "SD": "LAC", "OAK": "LV"})
    t = t[t.season.between(2013, 2025)]
    picks_only = t.groupby("trade_id").pick_season.apply(lambda s: s.notna().all())
    t = t[t.trade_id.isin(picks_only[picks_only].index)]
    t = t[pd.to_datetime(t.trade_date).dt.month <= 5]  # offseason through the draft
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pos_at = dict(zip(zip(dp.season, dp.pick), dp.position))
    rat = ctx.params["ratings"]

    rows = []
    for tid, g in t.groupby("trade_id"):
        S = int(g.season.iloc[0])
        cur = g[(g.pick_season == S) & g.pick_number.notna()]
        if cur.empty or (S - 1) not in rat:
            continue
        teams = sorted(set(g.gave) | set(g.received))
        if len(teams) != 2:
            continue
        up = cur.loc[cur.pick_number.idxmin(), "received"]  # team getting the best current pick
        for team in teams:
            other = [x for x in teams if x != team][0]

            def to_picks(rows_, receiving):
                out = []
                for _, r in rows_.iterrows():
                    yr = int(r.pick_season - S)
                    if yr < 0 or yr > 2:
                        continue
                    if yr == 0 and pd.notna(r.pick_number):
                        # Picks a team receives: valued by who was drafted there. Picks it gives
                        # up: a typical non-QB, since it wasn't necessarily going to take a QB.
                        qb = receiving and pos_at.get((S, int(r.pick_number))) == "QB"
                        pos = "QB" if qb else "nonqb"
                        out.append(Pick(int(r.pick_number), pos=pos))
                    elif yr > 0 and pd.notna(r.pick_round):
                        src_team = r.gave  # the original owner's strength sets the slot
                        prev = rat[S - 1].get(src_team, 0.0)
                        out.append(Pick(rnd=int(r.pick_round), year=yr,
                                        src={"start": CARRY * 0.73 * prev}))
                return out
            give = to_picks(g[g.gave == team], False)
            get = to_picks(g[g.received == team], True)
            if not give or not get:
                continue
            prev = rat[S - 1].get(team, 0.0)
            sit = {"start": CARRY * 0.73 * prev, "shift": {}, "qb_from": 1}
            r = evaluate(ctx, sit, give, get, n=n, seed=int(tid))
            got_qb = any(p.pos == "QB" for p in get)
            rows.append({"trade_id": tid, "season": S, "team": team, "role": "up" if team == up else "down",
                         "start": sit["start"], "got_qb": got_qb,
                         "change": r["change"] * 100, "se": r["se"] * 100})
    return pd.DataFrame(rows)


def summarize_real(df):
    qb_trades = set(df[df.got_qb & (df.role == "up")].trade_id)
    df["qb_up"] = df.trade_id.isin(qb_trades)
    df["strength"] = pd.cut(df.start, [-20, -1.5, 1.5, 20], labels=["weaker team", "average team", "stronger team"])
    print(f"Real pick-only trades 2013-2025: {df.trade_id.nunique()} trades\n")
    for qb in [False, True]:
        d = df[df.qb_up == qb]
        label = "Trade-ups for a QB" if qb else "Trade-ups for a non-QB"
        print(f"{label} ({d.trade_id.nunique()} trades), change in titles over 5 seasons, % of one title:")
        tab = d.groupby(["role", "strength"], observed=True).change.agg(
            avg="mean", median="median", share_gaining=lambda x: (x > 0).mean(), n="size")
        print(tab.round(2), "\n")


def main(examples=True):
    ctx = Context()
    if examples:
        run_examples(ctx)
    df = real_trades(ctx)
    df.to_parquet(OUT / "real_trades_scored.parquet")
    summarize_real(df)


if __name__ == "__main__":
    import sys
    main(examples="--real-only" not in sys.argv)
