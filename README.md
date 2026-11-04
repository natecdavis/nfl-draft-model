# NFL Draft Model: picks valued in Super Bowl odds

Goal: for general team situations, measure how much each draft decision (a pick,
trading up/down, trading future picks) changes the team's chance of winning a
Super Bowl over the next 5 seasons.

Team situations to start: contender with veteran QB, contender with rookie-deal QB,
middle-of-the-pack team, rebuilding team. Real teams come later.

## The chain
1. Player outlook: chance of bust / starter / star, by pick slot and position, by year.
   v1 = historical outcomes by slot+position. Later: "what the market misses" model.
2. Player -> team strength: value from second-contract pay (share of salary cap),
   minus rookie-contract cost, over the years the player is cheap.
3. Team strength -> title odds: season + playoff simulator.
4. Decision layer: compare options per team situation, scored in 5-year title odds.

## Data (all free, via nflreadpy)
- draft_picks: 1980-2026, 12.9k picks
- contracts (OverTheCap): 53.6k contracts, includes draft slot and apy_cap_pct
- schedules: 1999-2026 games (for calibrating the simulator)

## Milestones
- [x] M0: confirm data exists (scripts/check_data.py)
- [x] M1: player outlook v1 (src/outlook.py) - star / starter / bust chance by pick and position, drafts 2012-2021. Star = paid top-10/20/24 at position vs that year's market; starter = 50%+ snaps in 2+ seasons. Beats pick-only on rolling test (0.766 vs 0.781 log loss); well calibrated except 2nd half of round 1 slightly too optimistic on stars
- [x] M2: pick value (src/value.py) - value per season = snap share x later market price; minus rookie cost; 5th-year option for rd 1. Outputs value_library (per historical pick) and pick_value (expected, by pick x position)
- [x] M3: simulator (src/simulate.py) - real schedule + seeding + bracket. Estimated: game noise 12.6 pts, home edge 2.0, true strength sd 5.3, carryover 0.60. Playoff-game Brier 0.226 vs betting line 0.219; champion rank pattern matches history
- [x] M3b: bridge (src/bridge.py) - points of team strength per 1% of cap in player value: QB ~0.31-0.34, everyone else ~0.19, league-wide ~0.22. Same answer with leak-free pre-season prices. Position splits beyond QB vs non-QB do not predict better out of sample
- [x] M4: pick value in Super Bowl odds (src/pick_value.py, chart in reports/pick_value_titles.png). Assumes: the pick fills a need (his snaps would otherwise go to a min-salary player); saved cap buys non-QB veterans at 0.19 pts per 1%; situations start at +7 / +7 then -3 from season 3 / 0 / -5, drifting to average at 60% carryover
- [x] Trade calculator (src/trade.py): evaluate(ctx, situation, give=[Pick...], get=[Pick...]) -> change in Super Bowls over 5 seasons. Future picks' slots drawn from the source team's likely finish. Scored 394 real pick-only trades 2013-2025 (`python src/trade.py --real-only`).
- [x] Real teams (src/teams.py -> reports/teams_2027.md): market ratings from 2026 betting lines, QB situation from snaps/contracts/age, 2027 slots from simulating the rest of 2026. Per-team overrides in OVERRIDES. Note: QB cap hit from a rookie-deal QB's second contract now lasts every season (was fading); M4 numbers for that situation moved by <=0.2.
- [x] Better player outlook (src/prospects.py, src/college.py, src/college_features.py). Rolling test 2016-21 vs slot-only (log loss 0.7655): age + athletic testing 0.7606 (0.6% better, borderline vs chance; gains at TE, ED, S); adding college production (free play-by-play 2008-21, 85-98% matched) 0.7615 = no extra gain. Free college play-by-play host has no 2022+ seasons as of Nov 2026.
- [x] Board signal test: ESPN pre-draft big board 2012-21 (github.com/JackLich10/nfl-draft-data, no license stated: used for analysis only, not redistributed; file kept out of git under data/). Board-vs-pick gap has no signal (fallers: 13.4% stars vs 13.0% expected from slot; reaches: 23.3% vs 24.0%). Rolling test: slot+board 0.7647 vs slot 0.7655; age+athletic+board 0.7600 (best overall, 0.7% better than slot, 95% range of gain barely above zero).

## Draft day

    uv run streamlit run src/app.py

Runs locally in the browser; the board never leaves the machine.

1. Pick the team (situation, QB status and projected picks load from 2026 data) or "Custom". Adjust strength, QB need, the QB cap-hit season, and position needs.
2. Load the board: a CSV with `name, pos, rank, adp`.
   - `rank`: the team's own talent order (1 = best). This is where scouting comes in; the model can't beat the market on talent with public data.
   - `adp`: where the market expects the player to go (consensus mocks, intel). Drives "will he still be there?"
   - Positions: QB RB WR TE OT OG/IOL C EDGE/DE/OLB DT/DL LB/ILB CB S; K/P/LS are kept so others can draft them, never picked by us.
3. As picks are made, add players to "already drafted" and set the current pick.
4. Tabs: **On the clock** (titles added by each player at our pick, chance he's there at our next pick), **Trade offer** (any mix of picks now and future picks), **Trade down?** (titles lost by moving down N spots = the minimum an extra pick must be worth).

Engine: src/draftday.py. A player ranked r on our board gets the outcome odds of a typical r-th pick at his position; his cost is the contract of the pick where we take him. After we take a player at a position, later picks there count at a lower need (after one QB, the QB need is filled). Values from several picks are added, not simulated jointly.
Replay test: scripts/replay_2021.py (49ers 2021 with ESPN's board). App test: scripts/test_app.py.
- [x] Hasan consensus check (scripts/hasan_check.py; board from ~/cfdarko/hasan.csv, 2014-2024, not committed). Availability model recalibrated to it: actual picks scatter widely around consensus, most at the top (sd 0.67 in log(pick+10) for top-10 ranks, 0.29-0.48 elsewhere; players go ~10-30% later than rank on average). Reaches (taken 1+ round before consensus) became stars less often than their slot implies in-sample (19% vs 27%, p 0.01), but neither reaches nor falls improve year-by-year forecasts (2016-21), so no adjustment is applied.
- [x] Fall shading (scripts/fall_weight.py): players who fell past consensus turned out ~80-87% like the pick where they went (Hasan 2014-21; ~/cfdarko's 01f draft-slide experiment found the same with a rookie-value label). The app shades a falling player's talent toward the pick, weight 0.8 by default (Advanced setting).
- Relation to ~/cfdarko (the user's college prospect pipeline): its board-player values come from consensus rank + position; its own features didn't beat the board either. Reused here: Hasan consensus boards (calibration only), the draft-slide finding. Not used: its PFF / PFR-scraped inputs.
