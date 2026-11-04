"""Milestone 4: what a draft pick is worth in Super Bowl odds, by team situation.

For a team in a given situation, compare the next 5 seasons with and without the
pick. The answer is the change in expected Super Bowls won over those 5 seasons
(in percentage points of one title).

Team strength without the pick (points per game vs average):
  expected path m_t starts at the situation's level and drifts back toward
  average at the carryover rate from simulate.py (60% kept per year), plus any
  scheduled shift; actual strength adds uncertainty (sd 3 pts for next season,
  growing each year at the same carryover rate).

The pick adds, in season t (bridge.py rates):
  + position rate (QB 0.34, others 0.19) x snap share x (market price - veteran min)
      the player's value over a minimum-salary replacement
  - 0.19 x (rookie cost - veteran min)
      what the cap space would otherwise buy in free agency: non-QB veterans at
      market rates (QBs are rarely available in free agency, so the QB rate and the
      league-wide 0.22, which includes QBs, would overstate it)
Key assumption: the pick fills a need. The snaps he plays would otherwise go to a
minimum-salary player. If the team already has a good starter there, the pick is
worth much less. This matters most for QBs.
Season 5: round 1 picks are kept on the fifth-year option if worth it; others are
on market-rate deals or gone, so they add nothing.

Each pick's outcome is sampled, not averaged: tier from outlook.py chances for that
pick and position, then a real historical player's season-by-season playing time
and market price from the same position and tier (same pick range when possible).
Title odds come from the simulator's strength-to-title curve. With and without the
pick share the same random draws, so differences aren't swamped by noise.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from outlook import GROUPS, OUT, RAW, TIERS, star_thresholds
from simulate import league_params, load_schedules, title_curve
from value import VET_MIN, rookie_cost_curve

N = 20000
CARRY = 0.60
START_SD = 3.0  # uncertainty in next season's true strength at draft time

# Situations: expected strength next season, scheduled shifts, and QB need.
#   qb_from: first season a drafted QB can start (None = team has its QB, no QB need)
SITUATIONS = {
    "Contender, veteran QB": {"start": 7, "shift": {}, "qb_from": 3},
    "Contender, rookie-deal QB": {"start": 7, "shift": {3: -3.0}, "qb_from": None},
    "Middle of the pack": {"start": 0, "shift": {}, "qb_from": 1},
    "Rebuilding": {"start": -5, "shift": {}, "qb_from": 1},
}
NOTES = {
    "Contender, veteran QB": "QB ages out after season 2; a drafted QB can start from season 3",
    "Contender, rookie-deal QB": "QB's big second contract arrives in season 3 (-3 pts: ~15% of cap moved to him)",
}
PICKS = [1, 2, 3, 5, 8, 10, 11, 13, 16, 20, 24, 28, 32, 33, 40, 48, 56, 64, 65, 80, 96, 112, 128, 160, 192, 224, 256]


def strength_curve(sched, teams_df, params) -> pd.DataFrame:
    path = OUT / "title_curve_fine.parquet"
    if path.exists():
        return pd.read_parquet(path)
    c = title_curve(params, teams_df, sched, levels=list(range(-16, 19)), n=20000, seed=7)
    c.to_parquet(path)
    return c


def baseline_paths(sit: dict, rng, n: int = N) -> np.ndarray:
    """(n, 5) true strength without the pick."""
    # Expected path drifts back toward average; scheduled shifts (e.g. a QB's big
    # second contract) last from their season on, since that money is paid every year.
    m = sit["start"] * CARRY ** np.arange(5)
    for season, delta in sit["shift"].items():
        m[season - 1:] += delta
    u = np.zeros((n, 5)); u[:, 0] = rng.normal(0, START_SD, n)
    innov = np.sqrt(1 - CARRY ** 2) * 5.3
    for t in range(1, 5):
        u[:, t] = CARRY * u[:, t - 1] + rng.normal(0, innov, n)
    return m + u


def sample_outcomes(lib, curves, grp, pick, rng, bw=0.4, n=N):
    """(N, 5) snap shares and (N,) market prices for a pick: draw a tier from the
    outlook chances, then a real player of that position and tier, favoring players
    picked near this slot (weights fall off with distance in log(pick + 10), so
    there are no jumps at arbitrary pick-range edges)."""
    probs = curves[(curves.grp == grp) & (curves.pick == pick)][TIERS].to_numpy()[0]
    tier = rng.choice(len(TIERS), size=n, p=probs / probs.sum())
    shares = np.zeros((n, 5)); price = np.zeros(n)
    cols = [f"share_{t}" for t in range(1, 6)]
    for k, tname in enumerate(TIERS):
        idx = np.where(tier == k)[0]
        pool = lib[(lib.grp == grp) & (lib.tier == tname)]
        if len(idx) == 0 or len(pool) == 0:
            continue
        w = np.exp(-0.5 * ((np.log(pool.pick + 10) - np.log(pick + 10)) / bw) ** 2)
        draw = rng.choice(len(pool), size=len(idx), p=w / w.sum())
        shares[idx] = pool[cols].to_numpy()[draw]
        price[idx] = pool["price"].to_numpy()[draw]
    return shares, price


def pick_effect(shares, price, pick, grp, sit, cost, bar, rates):
    """(N, 5) points of team strength the pick adds each season."""
    shares = shares.copy()
    if grp == "QB":
        if sit["qb_from"] is None:
            return None
        shares[:, : sit["qb_from"] - 1] = 0.0  # sits behind the starter
    rate = rates["qb"] if grp == "QB" else rates["nonqb"]
    value = shares * np.maximum(price - VET_MIN, 0)[:, None] * 100
    rc = cost[pick]
    on_roster = (pick <= 64) | (shares[:, :4] > 0)
    c = np.where(on_roster, rc - VET_MIN, 0) * 100
    pts = np.zeros_like(value)
    fa = rates["nonqb"]
    pts[:, :4] = rate * value[:, :4] - fa * c
    if pick <= 32:  # fifth-year option, used only when worth it
        opt = bar[grp] * (1.0 if pick <= 10 else 0.7)
        v5 = shares[:, 4] * price
        keep = v5 > opt
        pts[:, 4] = np.where(keep, rate * (v5 - VET_MIN) * 100 - fa * (opt - VET_MIN) * 100, 0)
    return pts


def main():
    sched = load_schedules()
    teams_df = pd.read_parquet(RAW / "teams.parquet")
    params = league_params(sched)
    curve = strength_curve(sched, teams_df, params)
    sb = lambda x: np.interp(x, curve.strength, curve.super_bowl)

    c = pl.read_parquet(RAW / "contracts.parquet").filter(pl.col("apy_cap_pct").is_not_null())
    cost = rookie_cost_curve(c)
    bar = star_thresholds(c).xs(2026, level="season").to_dict()
    lib = pd.read_parquet(OUT / "value_library.parquet")
    curves = pd.read_parquet(OUT / "outlook_curves.parquet")
    rates = json.load(open(OUT / "bridge.json"))["preseason"]

    rows = []
    for sname, sit in SITUATIONS.items():
        rng = np.random.default_rng(11)
        base = baseline_paths(sit, rng)
        base_titles = sb(base)
        for grp in GROUPS:
            for pick in PICKS:
                shares, price = sample_outcomes(lib, curves, grp, pick, np.random.default_rng(pick))
                pts = pick_effect(shares, price, pick, grp, sit, cost, bar, rates)
                if pts is None:
                    continue
                d = sb(base + pts) - base_titles  # (N, 5) change in title chance per season
                rows.append({"situation": sname, "grp": grp, "pick": pick,
                             "titles_added": d.sum(axis=1).mean(),
                             "se": d.sum(axis=1).std() / np.sqrt(N),
                             "pts_per_season": pts.mean(),
                             "base_titles": base_titles.sum(axis=1).mean()})
    res = pd.DataFrame(rows)
    res.to_parquet(OUT / "pick_value_titles.parquet")

    pd.set_option("display.width", 170)
    print("Expected Super Bowls over 5 seasons WITHOUT the pick:")
    print(res.groupby("situation", sort=False)["base_titles"].first().mul(100).round(1).to_string(), "\n")
    for sname in SITUATIONS:
        t = res[res.situation == sname].pivot(index="grp", columns="pick", values="titles_added") * 100
        print(f"{sname}: titles added over 5 seasons, in % of one Super Bowl")
        print(t[[p for p in [1, 5, 10, 20, 32, 64, 100 if 100 in t else 96, 160]]].round(2), "\n")
    q = res[(res.grp == "QB") & res.pick.isin([1, 10, 32])][["situation", "pick", "pts_per_season"]]
    print("QB picks, avg points of team strength added per season:\n", q.round(2).to_string(index=False))
    print("Largest simulation noise (se, % of a title):", res.se.max().round(4) * 100)

    # "Typical non-QB pick": average over non-QB positions, weighted by how often each is drafted
    out = pd.read_parquet(OUT / "pick_outcomes.parquet")
    w = out[out.grp.isin(GROUPS) & (out.grp != "QB")].grp.value_counts(normalize=True)
    nonqb = (res[res.grp != "QB"].assign(w=lambda d: d.grp.map(w))
             .groupby(["situation", "pick"], sort=False)
             .apply(lambda d: np.average(d.titles_added, weights=d.w), include_groups=False)
             .rename("titles_added").reset_index())
    nonqb.to_parquet(OUT / "pick_value_titles_nonqb.parquet")
    print("\nTypical non-QB pick, titles added (% of one Super Bowl):")
    print((nonqb.pivot(index="situation", columns="pick", values="titles_added") * 100)
          [[1, 5, 10, 20, 32, 64, 96, 160]].round(2))
    chart(nonqb, res)


def chart(nonqb, res):
    """Two panels (non-QB, QB picks); each has its own y-scale, said in the footnote."""
    colors = dict(zip(SITUATIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # palette slots 1-4
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor=surface)
    panels = [("Typical non-QB pick", nonqb), ("QB pick (for teams that need a QB)", res[res.grp == "QB"])]
    for ax, (title, d) in zip(axes, panels):
        ax.set_facecolor(surface)
        for sname in SITUATIONS:
            s = d[d.situation == sname].sort_values("pick")
            if not s.empty:
                ax.plot(s.pick, s.titles_added * 100, color=colors[sname], lw=2, label=sname,
                        solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
        ax.set_xlim(0.85, 300)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=grid, lw=0.8)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2)
        ax.set_title(title, loc="left", color=ink, fontsize=12, fontweight="bold")
        ax.set_xlabel("Draft pick (log scale)", color=ink2)

    # Direct labels: non-QB lines are well separated just right of their peak
    for sname in SITUATIONS:
        s = nonqb[nonqb.situation == sname].set_index("pick").titles_added * 100
        axes[0].annotate(sname, (40, s.loc[40]), xytext=(0, 7), textcoords="offset points",
                         color=ink2, fontsize=9, ha="left")
    q = res[res.grp == "QB"].pivot(index="pick", columns="situation", values="titles_added") * 100
    axes[1].annotate("Middle of the pack", (2, q.loc[2, "Middle of the pack"] if 2 in q.index else q.iloc[1]["Middle of the pack"]),
                     xytext=(0, 7), textcoords="offset points", color=ink2, fontsize=9)
    axes[1].annotate("Contender (veteran QB) and Rebuilding:\nnearly the same", (16, q.loc[16].drop("Middle of the pack").max()),
                     xytext=(4, 10), textcoords="offset points", color=ink2, fontsize=9)
    axes[0].set_ylabel("Super Bowls added over 5 seasons\n(% of one title)", color=ink2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.935), ncol=4, frameon=False,
               fontsize=9.5, labelcolor=ink2)
    fig.suptitle("What a draft pick is worth in Super Bowl odds, by team situation",
                 x=0.06, y=0.975, ha="left", color=ink, fontsize=14, fontweight="bold")
    fig.text(0.06, 0.01, "The two panels use different vertical scales. A contender with a rookie-deal QB "
             "has no QB need, so it has no QB line. Assumes the pick fills a need. Model v1, public data.",
             color=ink2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.9))
    fig.savefig("reports/pick_value_titles.png", dpi=160, facecolor=surface)
    print("\nChart saved: reports/pick_value_titles.png")


if __name__ == "__main__":
    main()
