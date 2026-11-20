"""Real teams: the 2027 draft from each team's actual situation.

For each of the 32 teams:
  1. Strength now: ratings fit to the betting lines of 2026 games (recent weeks
     weighted more). The market already prices injuries and QB changes.
     Expected 2027 strength = carryover (0.60) x strength now.
  2. QB situation, from the QB with the most 2026 snaps, his contract and age:
       rookie deal, round 1   -> no QB need; when his second contract starts, the
                                 team loses ~3 pts (about 15% of cap moves to him)
       rookie deal, later pick -> unproven; a drafted QB could take over in season 2
       veteran aged 36+       -> successor needed as he ages out
       veteran paid < 7% of cap -> stopgap; QB needed now
       veteran paid more      -> no QB need, unless he played below the league median
                                 last season -> upgrade wanted; a drafted QB starts in season 2
  3. 2027 draft slot: the rest of the 2026 season simulated from actual results so
     far and the market ratings. Draft order is by wins only (playoff results aren't used).
     2027 firsts already traded (per the trade data, which ends April 2026) move
     with their new owners.
Then for each team: its first-round picks' value in Super Bowl odds, and verdicts
on three moves (trade down, trade up for a QB if needed, sell next year's 1st).

Everything here is a rule of thumb that the user can override per team via
OVERRIDES (e.g. a QB injury the snap counts misread).
"""
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

from outlook import OUT, RAW
from simulate import RELOCATED, load_schedules
from trade import Context, Pick, evaluate

SEASON = 2026
CARRY = 0.60
QB_SHIFT = -3.0
OVERRIDES: dict[str, dict] = {}  # e.g. {"NYJ": {"qb_from": 1}} after checking the news
N = 10000


# ---------- 1. strength now ----------

def market_ratings(sched: pd.DataFrame) -> tuple[pd.Series, float]:
    """Least-squares ratings from 2026 betting lines; recent weeks count more."""
    g = sched[(sched.season == SEASON) & (sched.game_type == "REG") & sched.spread_line.notna()].copy()
    last = g[g.result.notna()].week.max()
    w = 0.5 ** ((last - g.week).clip(lower=0) / 4)  # half weight every 4 weeks back
    teams = sorted(set(g.home_team) | set(g.away_team))
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(g), len(teams) + 1))
    X[np.arange(len(g)), g.home_team.map(idx)] = 1
    X[np.arange(len(g)), g.away_team.map(idx)] = -1
    X[:, -1] = (g.location != "Neutral").astype(float)
    sw = np.sqrt(w.to_numpy())
    A = np.vstack([X * sw[:, None], np.r_[np.ones(len(teams)), 0] * 100])
    y = np.r_[g.spread_line.to_numpy() * sw, 0]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    sign = np.sign(np.corrcoef(g.spread_line[g.result.notna()], g.result[g.result.notna()])[0, 1])
    return pd.Series(beta[:-1] * sign, index=teams), beta[-1] * sign


# ---------- 2. QB situation ----------

def qb_play(gsis):
    """EPA per dropback in the latest full season vs that season's QB median (nflverse);
    None if he had under 200 dropbacks in each of the last two seasons."""
    ps = pd.read_parquet(RAW / "player_stats.parquet")
    ps = ps[(ps.position == "QB") & (ps.season_type == "REG")].copy()
    ps["db"] = ps.attempts + ps.sacks_suffered
    ps["epa_db"] = ps.passing_epa / ps.db.clip(lower=1)
    med = ps[ps.db >= 200].groupby("season").epa_db.median()
    for y in (SEASON - 1, SEASON - 2):
        r = ps[(ps.player_id == gsis) & (ps.season == y) & (ps.db >= 200)]
        if len(r):
            return float(r.epa_db.iloc[0] - med[y])
    return None


def qb_situations(teams: list[str]) -> pd.DataFrame:
    s = pl.read_parquet(RAW / "snap_counts_2026.parquet").filter(
        (pl.col("position") == "QB") & (pl.col("game_type") == "REG")).to_pandas()
    s["team"] = s.team.replace(RELOCATED)
    snaps = s.groupby(["team", "pfr_player_id", "player"]).offense_snaps.sum().reset_index()
    tot = snaps.groupby("team").offense_snaps.transform("sum")
    snaps["share"] = snaps.offense_snaps / tot
    players = pd.read_parquet(RAW / "players.parquet")
    players = players.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")
    c = pd.read_parquet(RAW / "contracts.parquet")
    # Intended starter, in order:
    #   1. a first-round QB from the last 3 drafts with 15%+ of snaps (most snaps if two)
    #   2. the snap leader, if he has 60%+ of snaps
    #   3. otherwise the highest-paid QB with 20%+ (a backup filling in for an injury)
    latest_pay = c.sort_values("year_signed").groupby("otc_id").apy_cap_pct.last()
    otc_of = pd.to_numeric(players.otc_id, errors="coerce")
    snaps["pay"] = snaps.pfr_player_id.map(otc_of).map(latest_pay).fillna(0)
    rd = snaps.pfr_player_id.map(players.draft_round)
    yr = snaps.pfr_player_id.map(players.draft_year)
    snaps["young_r1"] = (rd == 1) & (yr >= SEASON - 2)
    picks = {}
    for team, d in snaps.groupby("team"):
        y = d[d.young_r1 & (d.share >= 0.15)]
        lead = d.loc[d.offense_snaps.idxmax()]
        if len(y):
            picks[team] = y.loc[y.offense_snaps.idxmax()]
        elif lead.share >= 0.6:
            picks[team] = lead
        else:
            picks[team] = d[d.share >= 0.2].sort_values("pay").iloc[-1]
    top = pd.DataFrame(picks).T
    rows = []
    for team in teams:
        r = top.loc[team]
        p = players.loc[r.pfr_player_id] if r.pfr_player_id in players.index else None
        otc = pd.to_numeric(p.otc_id, errors="coerce") if p is not None else np.nan
        k = c[c.otc_id == otc].sort_values("year_signed") if pd.notna(otc) else c.iloc[:0]
        latest = k.iloc[-1] if len(k) else None
        draft_year = p.draft_year if p is not None else np.nan
        rnd = p.draft_round if p is not None else np.nan
        birth = pd.to_datetime(p.birth_date) if p is not None and pd.notna(p.birth_date) else None
        age27 = (pd.Timestamp("2027-09-01") - birth).days / 365.25 if birth is not None else np.nan
        on_rookie = latest is not None and latest.year_signed == latest.draft_year
        pay = latest.apy_cap_pct if latest is not None else np.nan

        shift, qb_from = {}, None
        if on_rookie and rnd == 1:
            status = "rookie deal (1st round)"
            start_season = int(draft_year) + 4 - SEASON  # 1 = 2027
            if 1 <= start_season <= 5:
                shift = {start_season: QB_SHIFT}
        elif on_rookie:
            status, qb_from = "rookie deal (later round)", 2
        elif pd.notna(age27) and age27 >= 36:
            status, qb_from = f"veteran, {age27:.0f} in 2027", int(np.clip(39 - age27, 1, 4))
        elif pd.notna(pay) and pay < 0.07:
            status, qb_from = "stopgap veteran", 1
        else:
            status = "paid veteran"
            play = qb_play(p.gsis_id if p is not None else None)
            if play is not None and play < 0:
                status, qb_from = "paid veteran, below-average play", 2
        rows.append({"team": team, "qb": r.player, "qb_snap_share": r.share, "qb_status": status,
                     "qb_pay": pay, "qb_age_2027": age27, "qb_from": qb_from, "shift": shift})
    return pd.DataFrame(rows).set_index("team")


# ---------- 3. 2027 draft slots ----------

def draft_slots(sched: pd.DataFrame, rating: pd.Series, hfa: float, game_sd: float, rng) -> pd.DataFrame:
    """(N, 32) simulated 2027 first-round slot for each team's own pick."""
    g = sched[(sched.season == SEASON) & (sched.game_type == "REG")]
    teams = list(rating.index)
    idx = {t: i for i, t in enumerate(teams)}
    done, left = g[g.result.notna()], g[g.result.isna()]
    wins0 = np.zeros(len(teams))
    for h, a, res in zip(done.home_team, done.away_team, done.result):
        wins0[idx[h]] += 1 if res > 0 else (0.5 if res == 0 else 0)
        wins0[idx[a]] += 1 if res < 0 else (0.5 if res == 0 else 0)
    h = left.home_team.map(idx).to_numpy(); a = left.away_team.map(idx).to_numpy()
    edge = np.where(left.location.to_numpy() == "Neutral", 0, hfa)
    r = rating.to_numpy()
    m = r[h] - r[a] + edge + game_sd * rng.standard_normal((N, len(h)))
    wins = np.tile(wins0, (N, 1))
    np.add.at(wins, (slice(None), h), m > 0)
    np.add.at(wins, (slice(None), a), m < 0)
    key = wins + rng.random(wins.shape) * 0.5
    slot = np.argsort(np.argsort(key, axis=1), axis=1) + 1  # fewest wins -> 1
    return pd.DataFrame(slot, columns=teams), wins


def main():
    sched = load_schedules()
    ctx = Context()
    rating, hfa = market_ratings(sched)
    teams = list(rating.index)
    qb = qb_situations(teams)
    rng = np.random.default_rng(2027)
    slots, wins = draft_slots(sched, rating, ctx.params["hfa_recent"], ctx.params["game_sd"], rng)

    owner = {t: t for t in teams}
    owner.update({"GB": "DAL", "IND": "NYJ", "DAL": "NYJ"})  # 2027 firsts traded (trade data to Apr 2026)
    rows = []
    for team in teams:
        q = qb.loc[team]
        qf = q["qb_from"]  # pandas turns None into NaN and ints into floats
        sit = {"start": CARRY * rating[team], "shift": q["shift"],
               "qb_from": None if pd.isna(qf) else int(qf)}
        sit.update(OVERRIDES.get(team, {}))
        need = sit["qb_from"] is not None
        own_firsts = [t for t in teams if owner[t] == team]
        med = {t: int(np.median(slots[t])) for t in own_firsts}
        base = evaluate(ctx, sit, [], [], n=N, seed=1)["base"]  # titles expected without any picks
        # value of the firsts the team owns (slot as a distribution)
        firsts = [Pick(slots[t].to_numpy()[:N]) for t in own_firsts]
        v_firsts = evaluate(ctx, sit, [], firsts, n=N, seed=2)["change"] if firsts else 0.0
        out = {"team": team, "market_rating": rating[team], "exp_2027": sit["start"],
               "wins_now": wins[:, teams.index(team)].mean(), "titles_base": base * 100, "qb": q.qb, "qb_share": q.qb_snap_share, "qb_status": q.qb_status,
               "qb_need": ("now" if sit["qb_from"] == 1 else f"season {sit['qb_from']}") if need else "no",
               "firsts": ", ".join(f"#{med[t]}" + ("" if t == team else f" (from {t})") for t in own_firsts) or "none",
               "titles_value_firsts": v_firsts * 100}
        if own_firsts:
            k = min(med.values())
            # Trade down ~10 spots plus a 2nd-round pick (roughly the market rate)
            d = evaluate(ctx, sit, [Pick(k)], [Pick(min(k + 10, 40)), Pick(k + 42)], n=N, seed=3)
            out["trade_down"] = d["change"] * 100
            if need:
                if k <= 5:
                    qbm = evaluate(ctx, sit, [Pick(k)], [Pick(k, pos="QB")], n=N, seed=4)
                    out["qb_move"] = f"take a QB at #{k}"
                else:
                    qbm = evaluate(ctx, sit, [Pick(k), Pick(k + 32), Pick(rnd=1, year=1)],
                                   [Pick(5, pos="QB")], n=N, seed=4)
                    out["qb_move"] = f"#{k}, #{k + 32}, next 1st -> #5 QB"
                out["qb_move_value"] = qbm["change"] * 100
            s = evaluate(ctx, sit, [Pick(rnd=1, year=1)], [Pick(30)], n=N, seed=5)
            out["sell_next_1st"] = s["change"] * 100
        rows.append(out)
    df = pd.DataFrame(rows).sort_values("exp_2027", ascending=False)
    df.to_csv(OUT / "teams_2027.csv", index=False)
    write_report(df, qb)


def write_report(df: pd.DataFrame, qb: pd.DataFrame):
    f = lambda x: "" if pd.isna(x) else f"{x:+.1f}"
    lines = [f"# 2027 draft in Super Bowl odds, team by team (as of {date.today():%b %d, %Y})", "",
             "Values are the change in expected Super Bowls over 2027-2031, in % of one title.",
             "Trade down = own top first for a pick ~10 spots later + a pick ~42 spots later.",
             "Sell next 1st = own 2028 first for this year's #30.",
             "QB = the team's intended starter: a 1st-round QB from the last 3 drafts with 15%+ of 2026 snaps; else the snap leader if 60%+; else the highest-paid QB with 20%+ (injury fill-in). Snap share in brackets; 'check' = under 70%, likely injury or QB change.",
             "", "| Team | Strength now | 2027 exp. | QB (status) | QB need | 1sts (median slot) | Value of 1sts | Trade down | QB move | Sell next 1st |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        qbm = "" if pd.isna(r.get("qb_move")) else f"{r.qb_move}: {f(r.qb_move_value)}"
        lines.append(f"| {r.team} | {r.market_rating:+.1f} | {r.exp_2027:+.1f} | {r.qb} [{r.qb_share:.0%}{', check' if r.qb_share < 0.7 else ''}] ({r.qb_status}) | {r.qb_need} "
                     f"| {r.firsts} | {r.titles_value_firsts:.1f} | {f(r.get('trade_down'))} | {qbm} | {f(r.get('sell_next_1st'))} |")
    lines += ["", "Caveats: QB status is a rule of thumb from 2026 snaps, contracts and age; an injured starter can be misread. "
              "2027 pick ownership uses trade data through April 2026. Draft order is by record only."]
    open("reports/teams_2027.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()


def pick_ownership(year: int = 2027, teams=None) -> tuple[dict, list]:
    """Who holds each team's `year` picks (rounds 1-7), replaying every trade of a `year` pick in the trade log in date
    order (nflverse trades; ends April 2026). The log names the giving and receiving teams and the round, not whose
    pick it originally was: a team trading a round-r pick gives its own if it still has it, else the first one it
    acquired. Conditional picks count as conveyed. Returns ({holder: [(original team, round), ...]}, unresolved rows)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from roster_need import team as norm
    t = pd.read_parquet(RAW / "trades.parquet")
    t = t[(t.pick_season == year) & t.pick_round.between(1, 7)].copy()
    t["gave"], t["received"] = t.gave.map(norm), t.received.map(norm)
    t = t.sort_values(["trade_date", "trade_id"])
    teams = teams if teams is not None else sorted(set(t.gave) | set(t.received))
    hold = {tm: [(tm, r) for r in range(1, 8)] for tm in teams}
    bad = []
    for r in t.itertuples():
        rnd = int(r.pick_round)
        have = [p for p in hold.get(r.gave, []) if p[1] == rnd]
        if not have:
            bad.append((r.trade_date, r.gave, r.received, rnd))
            continue
        p = next((x for x in have if x[0] == r.gave), have[0])
        hold[r.gave].remove(p)
        hold.setdefault(r.received, []).append(p)
    return hold, bad


def team_inputs(sims: int = 4000) -> pd.DataFrame:
    """Each team's situation and projected 2027 first-round picks, without the slow
    title-odds evaluations. Used by the draft-day app."""
    global N
    sched = load_schedules()
    from simulate import league_params
    params = league_params(sched)
    rating, _ = market_ratings(sched)
    teams = list(rating.index)
    qb = qb_situations(teams)
    old, N = N, sims
    try:
        slots, _ = draft_slots(sched, rating, params["hfa_recent"], params["game_sd"], np.random.default_rng(2027))
    finally:
        N = old
    owner = {t: t for t in teams}
    owner.update({"GB": "DAL", "IND": "NYJ", "DAL": "NYJ"})
    from trade import COMP_OFFSET
    holdings, _ = pick_ownership(2027, teams)
    rows = []
    for t in teams:
        q = qb.loc[t]
        qf = q["qb_from"]
        sit = {"start": CARRY * rating[t], "shift": q["shift"], "qb_from": None if pd.isna(qf) else int(qf)}
        sit.update(OVERRIDES.get(t, {}))
        firsts = sorted(int(np.median(slots[o])) for o in teams if owner[o] == t)
        # every 2027 pick the team holds, projected: original team's first-round slot + 32 per round + compensatory
        # picks ahead of it (trade.COMP_OFFSET, rounds 4-7)
        own_slot = {o: int(np.median(slots[o])) for o in teams}
        held = sorted(own_slot[o] + 32 * (r - 1) + COMP_OFFSET.get(r, 0) for o, r in holdings.get(t, []) if o in own_slot)
        rows.append({"team": t, "situation": sit, "qb": q.qb, "qb_status": q.qb_status,
                     "strength_now": rating[t], "firsts": firsts, "picks": held,
                     "picks_from": sorted((own_slot[o] + 32 * (r - 1) + COMP_OFFSET.get(r, 0), o, r)
                                          for o, r in holdings.get(t, []) if o in own_slot)})
    return pd.DataFrame(rows).set_index("team")
