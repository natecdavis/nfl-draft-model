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
- [x] Past drafts review (src/review.py -> app page "Past drafts"): drafts 2014-2024 pick by pick, owner at start of draft night vs team that picked, each team's state on draft day (prior-season strength; QB on hand incl. March moves, his contract known by draft day, age, and prior-season EPA vs median), the model's recommendation from Hasan's board, draft-night trades scored for both sides. Scorecard on realized value (2014-21): round 1 model's choices +4.49 vs teams +2.40 (mostly QB picks for QB-needy teams); rounds 2-4 tie; rounds 5-7 model worse (candidates limited to Hasan's board; 60% of late recs went undrafted) -> don't use recommendations in rounds 5-7. By star rate (independent of the model's QB valuation) teams' picks did better.
- [x] Prospect model v2 test (src/prospect_model.py, scripts/pm_choice.py, scripts/pm_timing.py; data read-only from ~/cfdarko: rookie-value labels by season y1-y4, prospect features, bridge). Findings:
  - Draft position can't be an input for "what if we take him at k": it carries other teams' information (an off-board player "taken at #5" inherits #5 outcomes). Value is predicted from pre-draft info; draft-day info enters only as fall shading.
  - Pick-by-pick choice test 2019-22: consensus-only (board rank + position) matches teams in round 1 (16.4 vs 16.4 % of cap over 4 yrs); adding player features (age, combine, PFF college grades, recruiting, team context) makes choices WORSE (-0.55/pick, 95% -1.0 to -0.1), no late-round gain. Same conclusion as cfdarko's own experiments.
  - Timing: nothing beyond rank/position predicts the early share of a player's value (corr ~0 for age, college polish, full features).
  - Outcome ranges from 60 nearest drafted comps are calibrated (6% below p10, 9% above p90).
- [x] Outcome engine v2 is now the default (src/outcomes_v2.py; pick_value.OUTCOMES = "v2", "v1" kept). Season-by-season values from cfdarko labels (classes through 2025), consensus rank + position only, careers sampled from similar drafted players (smooth weights; QB pool ~20), rescaled to this project's value units per position family (scripts/units_check.py). Effect: each season is priced on what the player did that year (v1 credited future stars with second-contract pay from year 1), so top-5 non-QB picks now lose value on average and non-QB value peaks ~picks 20-64; QB value peaks ~#15-20 (consensus top-10 QBs haven't outperformed 10-20). Trade-down and QB-need conclusions hold.
- [x] Review rebuilt on engine v2; realized value from cfdarko season labels (2014-2022 drafts; undrafted NFL players matched by name from 2016). Each recommended player counted once per draft (the model can recommend one available player to many teams). Model minus team where they disagreed: round 1 +1.41 (95% +0.10 to +2.90; non-QB recs alone +0.97, +0.16 to +1.92), rounds 2-4 +0.31 (tie), rounds 5-7 -0.17 (tie / slightly worse). QB recommendations: large point estimates, too few to be sure.

## Scouting text test (Brugler's "The Beast", 2019-2022)

`src/beast.py` parses the user's own downloaded guides (kept local in `data/`, not in git) into
one row per player: grade, STRENGTHS / WEAKNESSES / SUMMARY. `scripts/beast_test.py` matches
1,535 of 1,594 profiles to the prospect universe and asks, out of sample (each class predicted
by models that never saw it), whether the text adds anything to consensus rank + position:

| Question | Result |
|---|---|
| Overall value beyond consensus | ~1% of the miss explained, range includes 0 (grade, word groups, full bag-of-words alike) |
| Brugler grade vs consensus as a ranking | 0.605 vs 0.619 rank correlation with actual value; no gain combined |
| Boom/bust spread at the same expectation | no word group predicts wider outcomes |
| Year-1 share (timing) | nothing once rank is controlled |
| Bust risk (bottom quarter at the same expectation), board players | modest real signal: AUC 0.573 -> 0.625. "Elite/rare/explosive" language = fewer busts; "upside/raw/project" = more; worse grade than rank = more |
| Big hits (top 10% vs expectation) | no reliable signal |

Takeaway: the reports mostly restate what the consensus already prices in. What's left helps
flag likely busts a little, not stars, and stars drive title odds. 2026 uses a different
(bulleted) layout and isn't parsed yet; 2024-25 PDFs don't allow text copying and are skipped.

**Does the bust signal improve actual choices?** (`scripts/beast_choice.py`) At every pick of the
2019-22 drafts, the consensus model vs the same model times a text multiplier (fit on the other
three drafts; a forward-only version for 2021-22). The multiplier learned almost nothing from
the words (weights ~0.01-0.03) and mostly rewards being in the guide at all. Result, per pick:
+0.03% of cap in 4-season value (95% -0.42 to +0.51) and -0.03% of a title for a middle team
(95% -0.18 to +0.12); forward-only -0.04 (-0.21 to +0.13); round 1 slightly worse (noise).
No improvement, so older guides aren't worth buying for this.

## Sons of NFL players (`scripts/nfl_fathers.py`)

Fathers hand-coded from the guides' BACKGROUND text, 2019-22 (every father+NFL sentence read;
coaches, executives, tryout-only and uncles excluded): 61 sons, 60 matched to outcomes.

| Compared with | Sons' 4-season value vs expected, relative to others | Beat their expectation |
|---|---|---|
| Consensus rank (all 60) | x1.05 (95% 0.79-1.31) | 37% vs 33% |
| Consensus rank (38 board players) | x1.11 (0.83-1.39) | 42% vs 32% (p 0.26) |
| Draft slot (31 drafted) | x1.16 (0.86-1.51) | **61% vs 38%** (p ~0.02, holds controlling for slot and rank) |

Teams picked sons only slightly earlier than their consensus rank (1.08x vs 1.03x). So drafted
sons beat their slot more often than others, but not by enough on average to be clearly worth
more, and it's 31 players with several comparisons tried (p ~0.02 is borderline after that).
Needs more years to confirm: same-name sons (Jr./II matched to an earlier NFL player in
nflverse) could extend it free to 2014-18.

**Adding 2018** (14 sons, read page by page from the Kindle web reader; `python scripts/nfl_fathers.py wide`,
2018-22, whole prospect universe as the comparison group): drafted sons beat their slot 56% vs 38%
(odds x1.98, p 0.04), but 2018's sons didn't (38% vs 42%), nor did 2019's (38% vs 39%); the
effect comes from 2020-22 (23 sons). Value vs slot x1.15 (0.88-1.45), vs consensus x1.09
(0.81-1.36), both indistinguishable from 1. The new year weakened rather than confirmed it:
not used in the model.

## Deeper rankings: Brugler grades for players off Hasan's board (`scripts/beast_offboard.py`)

Among off-board players 2019-22, Brugler's grade sorts them somewhat (rank correlation 0.15;
his round 3-4 grades averaged 5.2% of cap, graded free agents 0.7%, players not in his guide
0.02%). But in a fair late-round choice test (picks 100+, candidates = Hasan's board, like the
app, vs board + Brugler-graded off-board players with a grade multiplier fit on the other drafts),
adding them was slightly worse: -0.008% of a title per pick (95% -0.042 to +0.028). Deeper boards
aren't worth pursuing.

**Found along the way (training data):** for 2008-15 classes only *drafted* off-board players are
in the data (undrafted aren't labeled before 2016), so the model's expectation for an off-board
player is too high (~0.6% of cap vs ~0.04% actual for 2016+ off-board players). The app and the
past-drafts review only recommend board players, so they're unaffected; but the research choice
tests that let the model pick any college player (`pm.choice_test`, `beast_choice.py`) handicapped
it in late rounds: restricted to board players it does better (0.190 vs 0.162% of a title per
pick, picks 100+).

**Fixed (training data):** `prospect_model.load()` now keeps pre-2016 undrafted players
(16,603 rows) with the average season values of 2016-18 undrafted players in the same board
status and rank band, and models that use consensus rank train only on classes with a board
(2014+; before that every player looked "off board", first-rounders included). Off-board
expectations are now on target (2019: 0.061% of cap expected vs 0.057 actual; 2022: 0.054 vs
0.043; before, ~0.6 in the research tests). Engine change: off board 0.09-0.19 -> 0.04-0.06
(% of cap over 4 seasons), rank 250 slightly lower, rank 50 slightly higher. Everything rebuilt
(chart, trades, 2027 teams, 2021 replay, app, review; `scripts/review_parallel.py` runs the
review one draft per core, ~30 min). Review scorecard after the fix (each recommendation once,
model minus team): round 1 +1.24 (95% -0.01 to +2.57; non-QB +0.76, +0.04 to +1.48),
rounds 2-4 +0.42 (-0.23 to +1.24), rounds 5-7 -0.18 (-0.40 to +0.01).

### Retests after the training-data fix

- **Player features** (combine, recruiting, PFF grades), `scripts/pm_choice_fixed.py`: still worse than rank-only.
  Per pick: −0.79% of cap (−1.26 to −0.31), or −0.35 (−0.84 to +0.10) with heavier regularization.
  The rank-only model now picks off-board players 12% of the time (26% before the fix).
  The player-features model still picks them 22–36% of the time, so it isn't used.
- **Brugler off-board test**, fair version: −0.008 titles per pick (−0.120 to +0.104), unchanged.
- The bust signal, scouting text level and NFL fathers tests are about board or drafted players, whom the fix
  barely touched, so they weren't rerun.
