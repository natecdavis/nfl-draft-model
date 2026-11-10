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
- [x] M4: pick value in Super Bowl odds (src/pick_value.py, chart in reports/pick_value_titles.png). Assumes: the pick fills a need (his snaps would otherwise go to a min-salary player); cap is charged at 0.10 pts per 1% (was 0.19, the bridge's non-QB rate; see 'Cap cost'); situations start at +7 / +7 then -3 from season 3 / 0 / -5, drifting to average at 60% carryover
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
- [x] Model changes adopted (Nov 2026), review and simulator rebuilt: (1) outcome units "single":
  cfdarko's within-position pricing with one overall factor (1.10), replacing per-family factors that
  discounted rotational positions and favored OL (`outcomes_v2.UNITS`, override with OUTCOME_UNITS=family);
  this also changes the draft-day app and trade calculator, which compute outcomes live (reports/
  teams_2027.md and the pick-value chart were built before and need a rerun); (2) fall shading applied
  once (`review.Table` shaded twice before); (3) QB need as a probability (`review.p_need`: no starter
  1.0, stopgap 0.79, later-round rookie 0.62, below-median paid veteran 0.54, age 36+ 0.52).
  Not adopted: roster need, lookahead (rollout / position drop-off). Past drafts scorecard, model
  minus team where they disagree (each rec once, 2014-2022), titles in each version's own units:
  round 1 +1.24 -> +1.16, rounds 2-4 +0.42 -> +0.48, rounds 5-7 -0.18 -> -0.11 (none outside noise);
  seasons started: round 1 -1.32 -> -1.68, rounds 2-4 -0.76 -> -1.04, rounds 5-7 -0.43 -> -0.36.
  Recommendations: round 1 QB 40% / WR 58%; rounds 2-4 WR 34%, EDGE 31%, QB 23%; OL from 54% to
  6% of rounds 2-4 recommendations. QBs in round 1 still dominate: the open question is whether a
  rookie-deal QB's cap savings are worth as many titles as the model says.
- [x] Board position labels adopted (Nov 2026; `src/positions.py`): decisions use the consensus board's own
  position, valued as a blend over the NFL families that label historically moved to (leave-one-draft-out;
  e.g. DL5T 61% DL / 39% EDGE, ER 73% EDGE / 27% LB), instead of the position a player's team listed when
  it drafted him (hindsight for players not yet taken). The team's own pick keeps its listed position, and
  "actually delivered" uses each player's real position (the board's if the draft record has none).
  Review rebuilt: recommendations changed at 18% of picks; scorecard (model minus team where they
  disagree, titles) round 1 +1.16 -> +1.08, rounds 2-4 +0.48 -> +0.55, rounds 5-7 -0.11 -> -0.16; seasons
  started round 1 -1.68 -> -1.61, rounds 2-4 -1.04 -> -0.89, rounds 5-7 -0.36 -> -0.42 (all within noise).
  Simulator and artifact show blended labels like "DL/EDGE".
- [x] Prospect model v2 test (src/prospect_model.py, scripts/pm_choice.py, scripts/pm_timing.py; data read-only from ~/cfdarko: rookie-value labels by season y1-y4, prospect features, bridge). Findings:
  - Draft position can't be an input for "what if we take him at k": it carries other teams' information (an off-board player "taken at #5" inherits #5 outcomes). Value is predicted from pre-draft info; draft-day info enters only as fall shading.
  - Pick-by-pick choice test 2019-22: consensus-only (board rank + position) matches teams in round 1 (16.4 vs 16.4 % of cap over 4 yrs); adding player features (age, combine, PFF college grades, recruiting, team context) makes choices WORSE (-0.55/pick, 95% -1.0 to -0.1), no late-round gain. Same conclusion as cfdarko's own experiments.
  - Timing: nothing beyond rank/position predicts the early share of a player's value (corr ~0 for age, college polish, full features).
  - Outcome ranges from 60 nearest drafted comps are calibrated (6% below p10, 9% above p90).
- [x] Outcome engine v2 is now the default (src/outcomes_v2.py; pick_value.OUTCOMES = "v2", "v1" kept). Season-by-season values from cfdarko labels (classes through 2025), consensus rank + position only, careers sampled from similar drafted players (smooth weights; QB pool ~20), rescaled to this project's value units per position family (scripts/units_check.py). Effect: each season is priced on what the player did that year (v1 credited future stars with second-contract pay from year 1), so top-5 non-QB picks now lose value on average and non-QB value peaks ~picks 20-64; QB value peaks ~#15-20 (consensus top-10 QBs haven't outperformed 10-20). Trade-down and QB-need conclusions hold.
- [x] Review rebuilt on engine v2; realized value from cfdarko season labels (2014-2022 drafts; undrafted NFL players matched by name from 2016). Each recommended player counted once per draft (the model can recommend one available player to many teams). Model minus team where they disagreed: round 1 +1.41 (95% +0.10 to +2.90; non-QB recs alone +0.97, +0.16 to +1.92), rounds 2-4 +0.31 (tie), rounds 5-7 -0.17 (tie / slightly worse). QB recommendations: large point estimates, too few to be sure.

## Draft simulator (app page "Draft simulator", `src/simulator.py`)

Pick a team and a past draft (2014-2026) and make your own choice at each pick the team actually
made (including picks it traded for on draft night; trades themselves aren't simulated). At each
pick: what the team really did, the model's top 15 for that team at that moment, and a searchable
list of every available player (Hasan board players plus drafted players who weren't on it).
Other teams draft as they really did: at pick k, everyone actually taken before k is gone, including
the team's own earlier picks you passed on; players you took are gone at your later picks. Your
choices set your position needs. Same values as the Past drafts review (choosing the team's own
picks reproduces its numbers exactly); drafts through 2022 also show what your players actually
delivered vs the team's (`review.RealizedV2`). Value tables per team situation (current model: units, shading, QB-need probability) are cached in
`data/derived/sim_tables/` (`python src/simulator.py build` builds all 105, ~6 min on 6 workers;
otherwise ~20s the first time a situation is opened). Test: `scripts/test_simulator_page.py`.

The main "Draft simulator" page is a self-contained HTML page (`scripts/simulator_artifact.html`, packed
by `scripts/export_simulator_artifact.build_html` from the current review data and tables, rebuilt when
they change): the model's top 25 beside a searchable list of everyone available, and your draft vs the
team's across the full width, including what each pick actually delivered over the rookie contract
(every available player precomputed at every pick of every team with
`review.RealizedV2.titles_many`, cached in data/derived/sim_realized_v2.npz; drafts since 2023 count the
seasons played so far, value and cost alike). Its lookups match the Python
ones to rounding. "Draft simulator classic" keeps the Streamlit version. The same page, exported
with `python scripts/export_simulator_artifact.py`, is published as a private Claude artifact (it embeds
Hasan's boards: keep it private). The app's look (`.streamlit/config.toml`) matches it. Tests:
`scripts/test_simulator_html.py`, `scripts/test_simulator_page.py` (classic).

## Roster need test (`scripts/roster_need.py`)

The model ignores the non-QB roster: every first pick at a position is treated as filling a starting
spot; only picks earlier in the same draft lower need (2nd at a group x0.6, 3rd x0.3 of years 1-2
playing time). Test: open starting spots on draft day per team and position group (last season's
snaps by players still on the team, plus veterans added before the draft, vs the league-typical
amount; 2014-2025 drafts, 2,825 non-QB rookies), against what each rookie did, controlling for
position x pick, position x consensus rank, team strength and year (SEs clustered by team-draft).

| +1 open starting spot at the group | Effect (95%) | Average |
|---|---|---|
| Year-1 snap share | +2.0 pts (+0.9 to +3.0) | 27% |
| Years 1-2 snap share | +1.9 pts (+0.8 to +3.0) | 29% |
| Value, year 1 (% of cap) | +0.09 (+0.03 to +0.15) | 1.15 |
| Value, years 1-4 | +0.50 (+0.18 to +0.82) | 5.8 |

A rookie drafted after another rookie at the same group in that draft: years 1-2 snap share
-2.2 pts (-4.3 to -0.1). Roster need is real but small: about 7-9% more playing time and value per
open spot, from a fully covered position to a wide-open one roughly 15-20%. The model's within-draft
discount (-40% for the 2nd rookie at a group, -70% for the 3rd) is several times larger than the
measured ~-8%. Strongest in rounds 2-7 and at CB, RB, WR; none detectable in round 1 (first-rounders
play regardless) or at OT/C (small samples).

**Do need-aware choices beat teams' picks?** (`scripts/need_choice.py`, drafts 2014-2022) At every pick,
the model's choice under four rules: current (within-draft x0.6 / x0.3, no roster), measured
within-draft effect, measured roster effect, both; effects re-estimated leaving out the draft being
scored (roster +7-10% of value per open spot; 2nd / 3rd rookie at a group -2 to -9% / -8 to -18%).
Scored on realized titles as in the review; "adjusted" rescales a recommended player's real career
for this team's open spots vs those of the team that actually drafted him.

| Rule | Model minus team, disagreements (each rec once) | vs current rule, all picks | Same choice as current |
|---|---|---|---|
| current | +0.57 (+0.04 to +1.23) | | |
| within-draft measured | +0.61 (+0.07 to +1.23) | +0.004 (-0.005 to +0.013) | 97% |
| roster | +0.53 (+0.07 to +1.08); adjusted +0.58 | +0.008 (-0.025 to +0.040) | 70% |
| both | +0.53 (+0.05 to +1.12); adjusted +0.57 | -0.007 (-0.038 to +0.026) | 74% |

Roster need changes about 30% of choices but not their realized value, and doesn't change the position
mix (OL, QB and WR still make up nearly all recommendations). Not adopted. Rounds 5-7 stay slightly
worse than teams under every rule.

Found on the way: `review.Table` applies the fall shading twice (once when the grid is built, again on
lookup), so a player who falls far past his rank is shaded with an effective weight of ~0.97 instead
of 0.85 (e.g. #50 at pick 150 valued 0.64 vs 0.80). The tables in this test shade once; with the
current need rule they pick a different player than the stored review at 24% of picks. The review and
the draft simulator still use the double-shaded tables.

## Units fix: within-position pricing (`scripts/units_choice.py`)

Outcome engine v2 rescales cfdarko's season values per position family to match this project's
measure (snap share x the player's own later pay). That measure multiplies by absolute snap share, so
rotational positions are discounted (typical starter snap share vs the family factor: correlation
0.74; RB x0.61, TE x0.67, EDGE x0.78 ... OL x1.15). cfdarko prices each season at what veterans at
that position earn for that snap share and grade, so a 56%-snap DT is priced as a starter.
`OUTCOME_UNITS=single` uses one overall factor (1.10) instead; default stays "family".

Choice test, drafts 2014-2022, current need rule, fall shading applied once:

| Single minus family model, paired at every pick | All | Round 1 | Rounds 2-4 | Rounds 5-7 |
|---|---|---|---|---|
| Titles, family units | +0.00 (-0.05 to +0.05) | +0.09 | -0.03 | +0.01 |
| Titles, single units | +0.10 (+0.05 to +0.14) | +0.24 | +0.13 | +0.02 |
| Second-contract pay, % of cap (drafted) | +0.44 (+0.27 to +0.62) | +0.27 | +0.59 | +0.19 |
| PFR AV (drafted) | +0.47 (-0.44 to +1.51) | | | |
| Seasons started (drafted) | -0.10 (-0.22 to +0.04) | | | |

The choice changes at 36% of picks. Rounds 2-4 move from OL (45% of recommendations) to EDGE (24%), WR (28%) and
TE (6%); rounds 5-7 stay mostly OL (54%); CB, LB, IDL and RB are still almost never recommended. Each
title yardstick favors its own model; on outside yardsticks the single-units choices earned more later
pay (partly because EDGE / WR are paid more than OL) and tie on AV and starts. Against teams' actual
picks, both models lose on AV and seasons started (round 1 starts: -1.2 family, -1.6 single), while
winning on title value: teams' picks play more than the model's.

**Why the model's round-1 picks start less than teams' picks (2014-2022):** it's the QB recommendations.
Where the model disagreed in round 1, its non-QB recommendations matched or beat the teams' picks (seasons
started 4.3 vs 3.8, AV 34 vs 33); its 18 distinct QB recommendations started 2.4 seasons vs 6.5 (AV 29 vs
59), while scoring higher in titles (+5.5 vs +3.5, driven by Watson, Lamar Jackson, Herbert and Wentz;
misses such as Manziel, Hundley, Lynch, Rosen, Haskins and Willis cost about -1 each). Two causes:
- Greedy picks: the review recommends the best player for this pick only, ignoring whether he'd still be
  there at the team's next pick. Recommended QBs were still available at the team's next pick 42% of the
  time in round 1 (non-QBs 23%) and sat 13 spots below the pick on the consensus board (e.g. Hundley
  recommended at #4, went #147; Willis at #2, went #86). The draft-day app has the availability model;
  the review doesn't use it.
- QB need is treated as certain: 45% of team-years are flagged. Flags do separate (different main QB two
  seasons later: stopgap vet 79%, later-round rookie 62%, one below-median season of a paid vet 54%, age
  36+ 52%, vs 25-29% for unflagged teams), but a flag counts as a 100% need where about half of those teams
  kept their QB, and unflagged teams changed QBs a quarter of the time.

## Whole-draft choice test: lookahead, units, QB need (`scripts/plan_choice.py`)

The model drafts each team's whole draft at its real picks (other teams as they really drafted),
2014-2022, five versions, compared with the team's real draft (sums per team-draft, 288 drafts):
A greedy, family units, QB need certain (today, shading once) | B greedy, single units, QB need as a
probability | C position drop-off (value now minus expected best at his position at the next pick) |
D rollout (value now + simulated value of the team's later picks if we take him; later picks greedy;
other teams simulated around consensus rank) | E rollout, family units, certain QB need.

| Per team-draft | Titles (single) | Titles (family) | AV | Seasons started | 2nd-contract pay |
|---|---|---|---|---|---|
| Model (B) minus team | +2.31 | +2.30 | -24.7 | -3.4 | +0.07 |
| B - A (units + QB prob.) | +0.85 (+0.55 to +1.18) | -0.25 (-0.55 to +0.04) | -1.6 (-6.4 to +3.0) | -0.86 (-1.50 to -0.21) | +2.40 (+1.61 to +3.21) |
| D - B (rollout) | -0.25 (-0.64 to +0.09) | -0.31 (-0.81 to +0.10) | -0.8 (-3.7 to +2.3) | -0.25 (-0.68 to +0.19) | -0.46 (-1.01 to +0.06) |
| C - B (drop-off) | -0.77 (-1.27 to -0.31) | -0.96 (-1.57 to -0.44) | -1.9 (-6.3 to +2.5) | -0.97 (-1.49 to -0.44) | -0.83 (-1.62 to +0.00) |
| E - A (rollout) | +0.02 | -0.24 | -1.0 | -0.41 | +0.30 |

- Looking ahead doesn't help: the rollout ties greedy, the position drop-off is worse (it chases
  positions whose talent falls off fastest even when they're worth little).
- QB need as a probability changed no picks: every flagged team still takes a QB (104 in round 1
  over 288 drafts, vs teams' 0.35 QBs per draft in all rounds). A round-1 QB for a needy team is worth
  ~10x a non-QB in title value, so halving the need doesn't flip the choice. That value is calibrated
  in its own currency (teams' actual round-1 QBs for needy teams: predicted +10.1, delivered +11.2).
- Units fix: more second-contract pay, fewer seasons started (EDGE / WR / TE instead of OL).
- Against teams: the model wins on title value and loses on AV and starts. Part of that is
  measurement: the model's undrafted picks count 0 on AV / starts / pay (rounds 5-7 hold about half
  the AV gap), and its players' real careers happened as later picks with less playing time
  invested in them.
- Scoring note: one team using the model against 31 real teams is the right counterfactual, so the same
  faller can appear on many teams' model drafts (La'el Collins on 32 in 2015); dividing his value by
  the number of drafts he appears on understates the model and isn't used.

## Holdout check: is the backtest leaking? (`scripts/holdout_review.py`, Nov 2026)

The outcome engine is fit on all classes through 2025, so the review valued each past pick with that
draft's own careers among the comparisons. Rerun with the engine held out (`outcomes_v2.HOLDOUT`; the live
model is unchanged): leave-one-draft-out (LODO, 2014-2022) and an expanding window (only classes and
seasons known on draft day; 2019-2022, earlier drafts have too few comparison careers). Model minus team
where they disagree, each recommendation once (titles, % of one; 95% range):

| | Full engine (in-sample) | Leave one draft out | Expanding window, 2019-22 |
|---|---|---|---|
| Round 1 | +1.08 (-0.13 to +2.33) | +0.92 (-0.52 to +2.32) | +1.17 (-0.31 to +2.80) |
| Rounds 2-4 | +0.55 (-0.06 to +1.39) | -0.11 (-0.48 to +0.35) | +0.70 (-0.57 to +2.73) |
| Rounds 5-7 | -0.16 (-0.31 to -0.03) | -0.11 (-0.26 to +0.02) | +0.60 (-0.10 to +1.66) |
| Seasons started, R1 / R2-4 | -1.61 / -0.89 | -1.67 / -1.52 | -0.20 / -1.59 |

- Round 1 holds up out of sample (still mostly QB recommendations, still not certain).
- The rounds 2-4 edge was leakage: it came from QB recommendations (Hurts +29, Carr +15, Purdy +13)
  whose own careers raised the value of QBs at their rank. Without them the model's rounds 2-4 picks trail
  teams' on every yardstick; non-QB recommendations in rounds 2-7 trail teams under every engine.
- The expanding window's late-round gain is two QB hits (Purdy +18, Minshew +8); 19% of its rounds 5-7
  recommendations are QBs. Noise, not skill.
- The engine is sensitive to which classes it sees: LODO changes 27% of recommendations, the expanding
  window 53%. QB value at consensus rank 15 on draft day rose from 0.18 to 0.36 (cap share over 4 seasons) between the
  2019 and 2022 drafts as the 2016-18 QB classes (Wentz, Watson, Mahomes, Allen, Lamar) entered the pool;
  leaving out any one class moves it by up to 10%.
- Use: round 1 recommendations (especially whether to take a QB) are the model's tested edge; in rounds
  2-7 treat it as one input, not better than teams.
- Adopted (Nov 2026): the Past drafts review, its trade decisions and the simulator value each past draft
  with the engine fit without that draft class (`review.engine_for`, `review.REVIEW_HOLDOUT`); simulator
  tables are now one per situation and draft year (336; the artifact page is ~10 MB). Drafts after the
  last labeled class (2026) and the live draft-day app, trade calculator and 2027 team report use the
  full engine. The Past drafts scorecard is now the LODO column above.

## QB value across eras (`scripts/qb_era.py`, Nov 2026)

Drafted QBs 1999-2022, seasons 1-4, all measured the same way from nflverse player stats: a proxy
(plays and EPA above that season's backup level) fit to cfdarko's QB season values (r 0.96 by season,
0.99 over 4 seasons; matches cfdarko's era means for 2012-22). Draft slot stands in for consensus rank.

| Mean value, seasons 1-4 (cap share) | 1999-2005 | 2006-2011 | 2012-2015 | 2016-2018 | 2019-2022 |
|---|---|---|---|---|---|
| Picks 1-10 | 0.28 (n 11) | 0.31 (10) | 0.46 (6) | 0.44 (8) | 0.40 (8) |
| Picks 11-32 | 0.24 (11) | 0.31 (6) | 0.14 (4) | 0.39 (3) | 0.21 (5) |
| Picks 33-100 | 0.08 (11) | 0.11 (17) | 0.19 (10) | 0.07 (7) | 0.11 (9) |
| Round-1 QBs starting (300+ plays) in season 1 | 23% | 69% | 90% | 64% | 77% |

- Round 1 over 1999-2022 delivered 0.89x the 2012-22 average (picks 33-100: 0.88x); 2016-18 is the best
  stretch (0.43 vs 0.33 for 2012-15 and 2019-22), so the engine's pool runs ~10% high from it.
- Older QBs were worth less mostly because they sat (season-1 starters 23% in 1999-2005 vs 64-90% since
  2012, the rookie wage scale era); that's a structural change, so pre-2012 classes are a poor guide.
- Conclusion: the engine's QB level is about right (within ~10%); a 10% trim wouldn't flip round-1 QB
  calls (worth ~10x a non-QB for a needy team). Not adopted; older consensus boards not worth pursuing
  for QB value.

## Calibration, fall shading and cap cost (Nov 2026)

**Calibration out of sample** (leave-one-draft-out review, 2014-2022; realized / predicted titles):
team picks round 1 1.53, rounds 2-4 1.23, rounds 5-7 0.83 (teams know more than consensus rank; QB
picks 1.55). Model recommendations (each once): round 1 1.15, rounds 2-4 0.73, rounds 5-7 0.36. Recs at or
above the current pick on the board are on target (non-QB 1.10); recs who had already fallen past their
consensus rank deliver 37-72% of prediction, and recs who went undrafted 16%.

**Stronger fall shading** (`REVIEW_FALL_WEIGHT`, LODO review): doesn't fix it. Model minus team, titles:

| Fall weight | Round 1 | Rounds 2-4 | Rounds 5-7 | Same recs as 0.85 |
|---|---|---|---|---|
| 0.85 (current) | +0.92 | -0.11 | -0.11 | |
| 1.0 | +1.15 | -0.07 | -0.13 | 83% |
| 1.2 | +1.43 | -0.22 | -0.13 | 51% |

Late recs still go far later or undrafted (76-79% in rounds 5-7) and calibrate at 0.34-0.37 under every
weight: what's wrong isn't the shading formula but that players teams keep passing on are worse than
consensus says (an optimizer's curse on the board). Not changed. Consistent with "no tested edge after
round 1".

**Cap cost** (`scripts/qb_savings.py`, team-seasons 2013-2025). The model charges every contract its cap
at 0.19 points of team strength per 1% (cap spent on him can't buy non-QB value). Tested on QB
contracts, holding the QB's on-field value fixed:

| | Model | Data | Within team |
|---|---|---|---|
| Non-QB roster value per 1% of QB cap hit | -1.00 | -0.26 (-0.47 to -0.05) | -0.23 (-0.46 to +0.00) |
| Team rating per 1% of QB cap hit | -0.19 | +0.11 (-0.05 to +0.27) | -0.02 (-0.16 to +0.12) |
| Team rating per 1% of QB value (check) | +0.31 to +0.34 | +0.27 | +0.29 |
| Team rating per 1% of non-QB value (check) | +0.19 | +0.19 | +0.20 |

Same over 3-season blocks (+0.06 within team) and with 1-2 season lags (no negative effect). Teams with good
rookie-deal QBs rated no better than teams with equally good paid QBs (+2.27 vs +2.44) despite spending
~8% of cap less on him. The on-field links hold; the cost of cap doesn't show up (restructures, rollover,
void years make the cap soft in practice, or teams don't spend savings well). This matters well beyond
QBs: the cap charge is what makes top-10 non-QB picks nearly worthless in the model (#3 WR: 0.45 points
over 5 seasons with the charge, 4.0 without; #3 QB 7.9 vs 12.8). Implied charge: savings buy ~0.25% of non-QB value per 1% (0 to 0.46), so ~0.05 points per 1% of cap
(0 to ~0.09) instead of 0.19.

Smaller charge in the leave-one-draft-out review (`COST_RATE`, `scripts/cost_compare.py`): recommendations
are 98-99% the same at 0.10 and 0 (at a given pick every candidate costs the same rookie contract, so the
charge mostly cancels); scorecards on every yardstick move by <= 0.15 (round 1 titles +0.92 / +0.90 / +0.86,
starts -1.67 / -1.63 / -1.61 at 0.19 / 0.10 / 0). The charge matters for comparing picks (pick values, trade
up / down, the trade calculator), not for which player to take. **Adopted 0.10** (Nov 2026; `pick_value.COST_RATE`):
a cautious step toward the data's ~0.05, since the evidence is QB contracts only.

## Deferring to the board, and WR value (Nov 2026)

The model takes the consensus board's best available player only 8-16% of the time: it reweights board
rank by position (outcomes, pay) and team situation, and 53% of its round-1 recommendations are WRs
(3% of board-best players). Two tests, past drafts 2014-2022, leave-one-draft-out engine.

**Deferring to the board** (`scripts/defer_test.py`; every pick each team made, every available player
valued as in the review): the model's choice vs the board's best, a margin rule (model's choice only if it
beats the board's best by m) and a rank window (model's top among the k best-ranked available), QB calls
kept; m and k tuned leave-one-draft-out. Titles vs the team's pick, per pick:

| | Model | Board | Margin (tuned) | Window (tuned) |
|---|---|---|---|---|
| Round 1 | +1.56 (+0.92 to +2.24) | -0.35 | +1.56 | +1.56 |
| Rounds 2-4 | +0.06 (-0.04 to +0.17) | -0.08 | +0.01 | +0.06 |
| Rounds 5-7 | +0.04 (-0.02 to +0.09) | -0.04 | +0.02 | -0.01 |

Tuning lands on the model in nearly every year; board-best loses on titles in every round (it does better
on starts, +0.31 vs -1.39 in round 1, which don't price position). Not adopted. Rounds 2-7 are within
noise of teams under every rule, so the simulator now labels them "no tested edge". (The replay's top
choice matches the review's recommendation at 89% of picks, not 100%: small differences in how earlier
picks set needs.)

**WR value** (`scripts/wr_check.py`). Predicted vs delivered, out of sample: the model's WR
recommendations delivered 1.52x prediction in round 1 (0.94-2.13), 1.04x in rounds 2-4: the outcome
engine doesn't overrate WR careers. Team rating per 1% of cap in on-field value, WRs split out:

| | QB | WR | Other non-QB | WR minus other |
|---|---|---|---|---|
| Career-best prices | +0.32 | +0.11 | +0.21 | -0.10 (-0.23 to +0.04) |
| Within team | +0.31 | +0.14 | +0.20 | -0.05 (-0.19 to +0.08) |
| Pre-season prices | +0.34 | +0.09 | +0.21 | -0.12 (-0.31 to +0.06) |
| Within team | +0.29 | +0.16 | +0.21 | -0.06 (-0.23 to +0.11) |

WR pay buys 45-75% as much rating as other non-QB pay, not significant; a separate WR rate doesn't predict
2021-25 better. Tried anyway: WR value x0.75 (`WR_FACTOR`, `scripts/wr_compare.py`) cuts WRs from 22% to 9%
of recommendations (round 1: edge rushers and DTs instead; QBs unchanged at 46%), and the review gets worse:
round 1 +0.92 -> +0.30 in titles (+0.48 -> +0.26 scored with WRs at 0.75 too), starts -1.63 -> -1.67;
rounds 2-4 -0.11 -> +0.21 (noise). The WRs it liked delivered (Thomas, Jefferson, Lamb, Metcalf, Pickens):
the changed round-1 picks delivered 2.21 vs 1.18 for their replacements. Not adopted (`WR_FACTOR` = 1.0).

## Team strength to titles, and carryover (Nov 2026)

Pick values are points of team strength x how fast title odds rise with strength (the season simulator on a
generic league). Checked on real seasons 2003-2025, 736 team-seasons, each simulated on its real schedule
(`scripts/title_check.py`). Calibration slope = logistic regression of the outcome on the model's log-odds
(1 = odds rise with strength as fast as the model says; < 1 = too fast):

| | Playoffs | Conference title | Super Bowl |
|---|---|---|---|
| Pre-season (last season's rating x carryover) | 0.95 (0.71 to 1.20) | 1.19 (0.73 to 1.65) | 1.35 (0.73 to 1.98) |
| In-season luck-adjusted rating | (circular) | 1.17 (0.84 to 1.50) | 0.82 (0.49 to 1.16) |

Teams rated +6 or better in-season (n=80): 11 Super Bowls vs 12.7 predicted, 23 conference titles vs
21.9. Fine as is.

Carryover: correlation of ratings k seasons apart, actual 0.44 / 0.32 / 0.20 / 0.15 (k = 1-4) vs the
model's 0.44 / 0.26 / 0.16 / 0.10: strength fades a little more slowly after year one (each gap ~1.5
standard errors). Best fit: 0.58 per season plus a permanent 11%. Sensitivity (`CARRY_MODEL=persist`,
outputs in data/derived/holdout/carry/): pick values move ~5-10% (contenders up, rebuilders down; a
contender's #20 non-QB 2.54 -> 2.64, a rebuilder's 0.74 -> 0.67), no team's trade-down, QB-move or
value-of-firsts advice changes sign (QB moves 7.49 -> 7.04 on average). Not adopted (default `ar`, 0.60).

## Position pay within families, and projecting position changes

**Pay within OL and DB** (`scripts/paypos_choice.py`, `outcomes_v2.POS_PAY`, default off). cfdarko prices
a season on one veteran pay curve per family, but veteran starters (60%+ of snaps, 2016-2025) earn, vs
their family: OT x1.27 (LT 6.7% / RT 4.6% of cap), OG x0.86, C x0.77, CB x1.11, S x0.89. At the same
consensus rank, though, positions turn out differently under pooled pricing (drafted board players
2014-2022, actual / expected): OT 0.90, OG 1.07, C 1.35, CB 0.85, S 1.17, so the net expectation factor
is OT 1.14, OG 0.92, C 1.04, CB 0.94, S 1.04. Whole-draft test vs today's model (per team-draft; realized
titles scored with position pay):

| | Titles | Seasons started | 2nd-contract pay | AV | Same choice |
|---|---|---|---|---|---|
| Pay factor only | +0.26 (+0.06 to +0.44) | -0.33 (-0.82 to +0.18) | -0.81 (-1.32 to -0.30) | +0.1 | 42% |
| Pay x turnout | -0.01 (-0.15 to +0.14) | -0.56 (-0.97 to -0.14) | -1.09 (-1.52 to -0.66) | -2.4 | 60% |

Both move rounds 2-4 toward OTs (7% of picks -> 37% / 26%), and those picks deliver less on every
yardstick. Not adopted.

**Projecting position changes** (`scripts/posproj_choice.py`). 4.9% of drafted board players were priced
in a different NFL family than their board label (5-techniques: 39% became EDGE; edge rushers -> LB 13-22%;
off-ball LB -> EDGE 8%; S -> LB 5%); the value at stake is ~1.1% of all board players' expected value.
Whole-draft test: L0 = board label only (pre-draft), L1 = board label valued as a blend over the families
it historically moved to (leave-one-draft-out), cur = today's model, which takes a drafted player's
position from his draft record (the position his team listed: hindsight for players not yet taken).

| Per team-draft | Titles | AV | Seasons started | 2nd-contract pay | Same choice |
|---|---|---|---|---|---|
| L1 - L0 (blend) | +0.00 (-0.08 to +0.08) | +1.2 (-0.2 to +2.7) | +0.14 (-0.07 to +0.38) | +0.07 | 83% |
| cur - L0 (draft-record labels) | -0.05 (-0.15 to +0.05) | +0.0 | -0.26 (-0.52 to -0.00) | +0.12 | 77% |

The blend changes 17% of choices and is at best slightly better (rounds 2-4 AV +1.4, starts +0.18); the
draft-record labels, even with hindsight, don't help. Position labels matter little for choices, and the
review's use of draft-record positions isn't flattering its results.

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

## Biggest misses (`scripts/misses.py`, Nov 2026)

Past drafts 2014-2022, leave-one-draft-out engine. Each available player at each pick has the model's titles and
what he actually delivered at that pick's cost (`holdout/defer_candidates.parquet`). Model recommendations are
counted once per player per draft.

**Where the error is.** QBs are 22% of the model's distinct recommendations and 55% of its absolute error; the 20
largest misses are 46%. Flops: Manziel (predicted 8.2 titles, delivered -0.5), Lance, Haskins, Lynch, Willis, Rosen,
Kizer, Cook; the worst non-QBs are WRs (Skyy Moore, Jaelen Strong, Treadwell, Ross, ~2 below each). Hits: Watson
(+18), Lamar Jackson, Herbert, Carr, Love, Michael Thomas. At the picks where players were really drafted, the
biggest under-predictions are QBs (Watson, Prescott at 0.1 predicted, Herbert, Allen, Lamar, Jones, Wentz, Goff)
and then top-10 non-QBs the model ranked far down its list: Khalil Mack (#5, predicted -0.14, the model's 83rd
choice there), Micah Parsons (56th), T.J. Watt, Aaron Donald, Kenny Clark.

**1. Top-10 non-QBs are under-valued (engine flaw).** Team non-QB picks, delivered minus predicted titles (95% over
draft years):

| Consensus rank | n | Predicted | Delivered | Miss | Below prediction |
|---|---|---|---|---|---|
| 1-5 | 38 | 0.59 | 1.31 | +0.72 (+0.35 to +1.05) | 32% |
| 6-10 | 40 | 0.96 | 1.62 | +0.65 (+0.24 to +1.07) | 35% |
| 11-20 | 82 | 1.17 | 1.52 | +0.35 (+0.05 to +0.64) | 49% |
| 21-32 | 98 | 1.04 | 0.99 | -0.05 (-0.36 to +0.25) | 55% |
| 33-120 | 732 | 0.66 | 0.76 | +0.10 | 57% |

Cause: the comparison careers. The engine values a player by sampling the 60 drafted players of his family whose
expected value is closest; at the top of a family every neighbour is below him, so the sample reaches down to
rank ~30-50. Drafted board players 2014-2022, 4-season value (full engine):

| | Actual | Regression | Comparison careers (used) | Local-linear correction |
|---|---|---|---|---|
| Non-QB rank 1-10 (n 76) | 0.173 | 1.10x | **0.71x** (every family 0.60-0.85) | 0.94x |
| Non-QB rank 11-32 / 33-64 / 65-120 / 121+ | | | 0.99 / 1.00 / 0.98 / 1.06x | 1.08 / 1.01 / 1.00 / 1.02x |
| QB rank 1-10 / 11-32 | | 0.80 / 0.61x | 0.98 / 0.93x | 1.01 / 0.91x |
| QB rank 33-64 (n 10) | 0.095 | 1.25x | **1.86x** | 1.71x |

The regression itself is close for top non-QBs; the comparison sample loses ~30% of their value. A local-linear
correction (weighted regression of comparison values on expected value, read at the player's own expectation)
removes the gap without moving other bands. Affects pick values, trade-up/down and "top-5 non-QB picks lose value"
more than which player to take (the shortfall is similar across families). Tested and adopted below
("Comparison-career correction").

**2. QB reaches bust.** QB recommendations by how far below the current pick the QB sat on the consensus board
(delivered / predicted, each player once per bucket): at or above the pick 1.00 (n 20), 1-15 below 0.95 (30), 16-40
below 0.65 (15), 40+ below 0.24 (17, no hits). By where they actually went: taken at or before the recommended pick
1.37, within 32 picks after 1.17, 33+ picks after 0.22, undrafted 0.08 (non-QBs: 1.14 / 0.57 / 0.29). This is
knowable on draft day (board distance), unlike where he goes. Part of it is the engine: QB comparison careers for
ranks 33-64 run 1.86x actual (only 10 such QBs drafted 2014-22), because the regression barely separates QB ranks
(QBs share tree leaves with other families) and the comparison window borrows from QBs ranked 11-32. Round-1 QB
recommendations from near the top of the board are on target.

## Comparison-career correction (`outcomes_v2.COMP_CORRECT`, `scripts/comp_correct.py`, Nov 2026)

`COMP_CORRECT=ll` (default since Nov 2026) rescales each player's sampled careers so their weighted mean equals a local-linear fit of
comparison value on expected value, read at the player's own expectation (factor clipped to 0.5-2; shape of the
distribution kept). Factors, full engine: consensus #1-10 non-QBs x1.2-1.65 (OL lowest, RB highest), ranks 15-20
x1.05-1.3, ranks 32-120 ~1, the bottom of the board (rank ~300) x0.5-0.9; QBs ~1 except ranks 32-50 (x0.91).
Simulator tables carry a `_ccll` suffix when on; `COMP_CORRECT=off` reproduces the earlier engine.

**Engine, leave-one-draft-out** (each class valued by an engine fit without it; drafted players 2014-2022,
comparison mean / actual 4-season value):

| | Off | ll |
|---|---|---|
| Non-QB rank 1-10 (n 76) | 0.70 | 0.93 |
| Non-QB 11-32 / 33-64 / 65-120 / 121-200 / 201-300 | 0.99 / 1.00 / 0.98 / 1.03 / 1.14 | 1.08 / 1.01 / 1.00 / 1.02 / 1.04 |
| QB 1-10 / 11-32 / 33-64 (n 10) | 0.99 / 0.92 / 1.93 | 1.02 / 0.90 / 1.76 |
| Board players: mean abs error / rmse (cap share) | 0.0441 / 0.0654 | 0.0436 / 0.0645 |

Not fixed: QBs ranked 33-64 (small sample), and drafted players who weren't on the board (comparison careers
1.78x actual for non-QBs: the comparison pool is drafted-only, and off-board drafted players do worse than it).

**Past drafts review, leave-one-draft-out** (model minus team where they disagree, each rec once; realized value
doesn't use the engine, so both versions are scored on the same yardstick):

| | Titles | Seasons started | AV | 2nd-contract pay |
|---|---|---|---|---|
| Round 1, off / ll | +0.92 / +1.07 | -1.63 / -1.24 | -12.0 / -8.1 | -1.00 / -0.46 |
| Rounds 2-4 | -0.11 / +0.42 | -1.52 / -1.01 | -10.4 / -6.5 | -0.69 / +0.04 |
| Rounds 5-7 | -0.08 / -0.11 | -0.74 / -0.68 | -6.6 / -6.3 | -0.81 / -0.71 |

Paired at each pick (ll's rec minus off's rec, realized titles): round 1 +0.05 per pick (-0.16 to +0.28), rounds 2-4
+0.005, rounds 5-7 -0.002; recommendations change at 18% / 14% / 11% of picks. The title gain is QB swaps (Wentz for
Lynch, Bridgewater for Manziel; Rosen for Mayfield the other way) and Hurts in round 2: luck either way. Changed
non-QB recommendations, each swap once: round 1 (n 20) titles -0.16, starts +0.50, AV +5.3, pay +0.89 (DL / EDGE such
as Donald, Allen, Hutchinson instead of some WRs; also Ed Oliver over Metcalf); rounds 2-4 (n 29) titles +0.50,
starts +0.14, AV +4.1, pay +0.99. Calibration: team non-QB picks ranked 1-5 / 6-10 delivered minus predicted
+0.66 / +0.71 -> +0.06 / +0.28; model recs ranked 1-10 delivered / predicted 1.61 -> 0.92 (18 recs instead of 8).

**Whole-draft replay** (simulator tables, both leave-one-draft-out; the model drafts each team's whole draft at its
real picks, greedy, its own choices setting needs; ll minus off per team-draft, 288 drafts): titles +0.16 (-0.10 to
+0.45), seasons started +0.08 (-0.27 to +0.43), AV +1.9 (-1.0 to +5.0), 2nd-contract pay +0.38 (-0.10 to +0.87);
same player at 70% of picks (round 1 78%); round-1 mix WR 59 -> 52%, QB 39 -> 38%, IDL / EDGE 3 -> 10%.

**Pick values** (titles added over 5 seasons, % of one): the big change. Middle of the pack, average non-QB position:
#1 0.48 -> 1.41, #5 0.64 -> 1.42, #10 1.03 -> 1.72, #20 1.19 -> 1.39, #32 1.12 -> 1.18, #64 unchanged. Non-QB value now
peaks around #10 and #1 is worth about what #16-20 is (before: a third of it). EDGE / DL / WR at #1 roughly double;
LB, RB, TE, CB, S at the top go from ~0 or negative to positive. QB picks #1-10 +2-8%, #32 10.4 -> 8.8, #48
7.5 -> 6.3. Trade-down advice and the trade calculator weaken at the top of round 1 for non-QBs.

Verdict: the engine is better calibrated out of sample, and choices tie or improve slightly on every yardstick
(none significant). **Adopted** (Nov 2026; `COMP_CORRECT` default "ll"). Rebuilt: past drafts review and trade
decisions (2014-2022 identical to the leave-one-draft-out "ll" run above, so the Past drafts scorecard is now its
column: round 1 +1.07, rounds 2-4 +0.42, rounds 5-7 -0.11 titles), simulator tables and artifact, pick-value chart,
real trades, 2027 team report, 2021 replay. Earlier sections quote pick values from before the change (e.g.
"top-5 non-QB picks now lose value on average" no longer holds: a top-5 non-QB is now worth about a #16-20 pick).
Effect on the 2027 team report: value of firsts up (most for top-10 slots: LAC #4 0.7 -> 1.5, MIA #2 0.5 -> 1.1),
trade-down still positive for every team but smaller at the top (LAC +1.7 -> +1.3, NO / TB +1.0 -> +0.6), QB moves
down slightly (TB +5.9 -> +5.1), selling next year's first from ~0 to -0.1 to -0.3. Real trades: teams trading down
for a non-QB still gain in 93-97% of trades.

## Board-less draft classes and monotone value (`outcomes_v2.NO_BOARD_CLASSES`, `MONOTONE`, `scripts/no_board.py`, Nov 2026)

**The off-board miss.** Drafted players who weren't on Hasan's board got comparison careers worth 1.78x what they
delivered (QBs 10.7x; leave-one-draft-out, 2014-2022). Cause: the board starts with the 2014 class, so every drafted
player in 2012-2013 (first-rounders included) read as off-board. The comparison set for an off-board player is
exactly the past off-board drafted players at his family (they share one expected value), and 2012-2013 were ~2/3 of
them (off-board 2012-13 drafted players averaged 0.05-0.06 of cap over 4 seasons vs 0.01-0.025 for 2014-22; off-board
QBs drew Luck, Wilson, Cousins). `NO_BOARD_CLASSES=drop` (default) leaves classes where fewer than half the drafted
players are on the board (2012, 2013) out of the regression and the comparison pool.

**The late-QB bump.** Dropping those classes exposed a second problem, already flagged: QBs ranked 250-297 drew ~0.08
of cap (vs 0.024 at rank 200) from one comparison, Brock Purdy (rank 290, 0.44), in a pool of ~14 late board QBs. With
drop alone, leave-one-draft-out reviews recommended QBs ranked ~270-300 in rounds 2-4 (Huntley at #37 over Hurts;
Silvers, Rush, Buechele, Bonner). `MONOTONE=min` (default) makes each family's mean value non-increasing in consensus
rank: each board rank capped at the lowest value of any better rank (only lowers values; off-board keeps its own
value). Tried first, `MONOTONE=iso` (isotonic fit, pooling violators) spread the bump over QBs ranked 150-250 (0.030
-> 0.038), tripled QB recommendations ranked 121+ (8 -> 24; they delivered 3% of prediction) and worsened rounds 2-4
starts (-1.01 -> -1.49): not adopted. Isotonic including the off-board point pulled off-board players to 0.78x actual
(they did better than board players ranked 251-300), so off-board is left out of the fit.

**Engine, leave-one-draft-out** (engine mean / actual 4-season value; before = keep + no monotone):

| Drafted players 2014-2022 | n | before | drop | drop + iso | drop + min (adopted) |
|---|---|---|---|---|---|
| Non-QB, off board | 360 | 1.78 | 0.93 | 0.93 | 0.93 |
| Non-QB, ranked 201-250 / 251-300 | 185 / 127 | 0.96 / 1.22 | 0.96 / 1.11 | 0.94 / 1.15 | 0.87 / 1.00 |
| Non-QB, ranked 121-200 | 496 | 1.02 | 1.01 | 1.02 | 0.96 |
| Non-QB, ranked 1-120 | 964 | 0.93-1.08 | same | same | same |
| QB, off board | 11 | 10.7 | 1.69 | 1.69 | 1.69 |
| QB, ranked 121-200 / 201-250 / 251-300 | 15 / 10 / 4 | 0.93 / 3.35 / 0.49 | 0.88 / 3.08 / 0.50 | 1.25 / 3.43 / 0.24 | 0.44 / 1.24 / 0.10 |
| Board players, mean abs error / rmse | | 0.0433 / 0.0644 | 0.0432 / 0.0644 | 0.0432 / 0.0641 | 0.0426 / 0.0642 |

Off-board by family, before -> adopted: DB 1.45 -> 0.97, DL 3.11 -> 0.86, EDGE 1.88 -> 0.84, LB 1.84 -> 1.02, OL 1.53
-> 0.87, RB 1.89 -> 0.98, TE 2.48 -> 0.93, WR 1.86 -> 0.97. Late-QB numbers rest on a handful of players (Purdy alone
is the 251-300 actual).

**Past-drafts review, leave-one-draft-out** (model minus team where they disagree, each recommendation once; before =
production before this change):

| | before | drop | drop + iso | drop + min (adopted) |
|---|---|---|---|---|
| Round 1 titles / starts | +1.07 / -1.24 | +1.11 / -1.33 | +1.35 / -1.30 | +1.07 / -1.24 |
| Rounds 2-4 titles / starts | +0.42 / -1.01 | -0.06 / -0.98 | +0.23 / -1.49 | +0.11 / -1.09 |
| Rounds 2-4 titles without Hurts, Carr | -0.10 | -0.06 | -0.10 | -0.23 |
| Rounds 5-7 titles / starts / pay | -0.11 / -0.68 / -0.71 | -0.12 / -0.81 / -0.81 | -0.10 / -0.52 / -0.68 | -0.07 / -0.40 / -0.45 |
| QB recommendations ranked 121+ (delivered / predicted) | 8 (0.01) | 8 (0.01) | 24 (0.03) | 1 (0) |
| Same recommendation as before | | 86% | 66% | 59% (R1 90%) |

Rounds 2-4 titles swing on two hits (Hurts at #37, +29.5; Carr at #33, +15.1, who under "min" is recommended in round
1 instead, +15.5 there); without them the versions are within noise of each other (ranges ~+-0.6). Round 1 is
unchanged; rounds 5-7 improve on every yardstick. **Adopted** (Nov 2026).
Rebuilt: past drafts review and trade decisions (2014-2022 identical to the leave-one-draft-out "drop + min" run, so
the Past drafts scorecard is now its column), simulator tables (suffix `_nbdrop_monomin`) and artifact, pick-value
chart, real trades, 2027 team report, 2021 replay. Pick values barely move (middle of the pack, typical non-QB #1
1.46 -> 1.48, #10 1.72 -> 1.77). 2027 team report: value of firsts and trade-down within +-0.1; QB trade-up moves
~0.5 smaller (LA +10.9 -> +10.3, TB +5.1 -> +4.6, NO +4.9 -> +4.4). No off-board player is a top recommendation in
any reviewed draft (2014-2022 before, 2014-2026 after), so the off-board fix matters for the engine's calibration and for
off-board players' values in the simulator, not for the past recommendations.

## QB reach penalty (`scripts/qb_reach.py`, Nov 2026): not adopted

Prompted by the misses study (QB recommendations 16-40 spots below the pick delivered 0.65x prediction, 40+ below
0.24x) and the QB ranks 33-64 gap. First, the 33-64 gap itself (leave-one-draft-out engine, drafted QBs 2014-2022):
it is five players ranked 33-48 (Lock, Love, Kizer, Ridder, Corral: engine 0.197 of cap over 4 seasons, actual
0.073) next to Mahomes, Allen, Watson, Jackson and Carr at 21-32. Not significant (ranks 33-64 actual / engine 95%
0.21-1.21; ranks 33-120 engine 1.18x actual), and no smoothing removes it: comparison players by consensus rank
instead of expected value 1.74 -> 1.63x; QB window of 5-20 nearest leaves ranks 33-48 at 2.1-2.7x with no change in
QB error (all within +-0.003). Left as is.

**Reach penalty.** A QB ranked below the pick has his value multiplied by ((pick + 10) / (rank + 10)) ^ lam
(off-board = rank 330), tuned leave-one-draft-out over every pick each team made 2014-2022 (candidates rebuilt with the
current engine). With the current engine the reach pattern is gone: the model's QB choices deliver 0.78x / 0.77x /
0.81x of prediction at the pick or above / 1-15 below / 16-40 below (40+ below: 4 picks, 0.45x), so the misses-study
pattern came from the engine before the comparison-career and off-board fixes. (All available QBs: 1.16 / 1.00 /
1.18; the ~20% shortfall of QB choices at every distance is ordinary selection.) Tuning picks lam = 0 in 8 of 9
drafts (0.25 in one, which swaps Hurts at #37 for Denzel Mims: -30.5). Fixed lam, round 1 titles vs teams per pick:
0 +1.56, 0.25 +1.62, 0.5 +1.51, 1 +1.36, 2 +1.32, 3 +1.07; starts improve (-1.33 -> -0.74 at 3) because fewer QBs are taken (44% ->
23%). Rounds 2-4 and 5-7 unchanged. Not adopted.

## Undrafted players of 2014-2015 in realized value (Nov 2026)

La'el Collins (board #16 in 2015, undrafted during a police investigation, then a starting tackle for Dallas) showed
as delivering 0. Realized value (review.RealizedV2) finds a player in cfdarko's season labels by draft slot, or for
undrafted players by name and class, and cfdarko labels undrafted players only from the 2016 class: every 2014-2015
undrafted player counted as zero. `review.udfa_pre2016_labels()` (cached in data/derived/udfa_pre2016_labels.parquet)
fills them in with this project's measure: snap share in seasons 1-4 x market price (best contract signed after his
first, share of cap; veteran minimum if none), times the bridge's 1.10 (outcomes_v2.CF_OVER_OURS_ALL). Check on 2016-2018
undrafted players, where both exist: correlation 0.86; cfdarko / ours 1.22 for players with real careers (cfdarko >
0.05 of cap over 4 seasons, n 41), so the filled-in values run ~10% conservative (more for fringe players, whom the
veteran-minimum price understates). 328 players, led by Norwell, Malcolm Butler, Hurns, Collins (0.17 of cap over 4
seasons), Tyrell Williams, Shaquil Barrett.

Effect: the model recommended Collins at 134 picks in 2015 (rounds 2-5): those rows now deliver +2.2% of a title
instead of 0 (and he was the board's best available at 196 picks). Recommendations unchanged. Scorecard (each
recommendation once): rounds 2-4 +0.11 -> +0.14, others unchanged. Review picks rescored for 2014-2015, simulator
realized cache and artifact rebuilt. Not changed: the training labels (prospect_model still gives pre-2016 undrafted
players cell averages) and the starts / AV / second-contract-pay yardsticks, which come from draft records and so
count any undrafted player as zero in every year.

## Training-data links and undrafted players' yardsticks (`prospect_model.LINK_FIX`, `review.Careers`, Nov 2026)

Two more data gaps found while fixing La'el Collins.

**Training data** (`prospect_model._repair_links`, `LINK_FIX=on` default). cfdarko's prospect rows are linked to draft
picks and to Hasan's board imperfectly: 115 picks 2012-2025 had no prospect row (mostly small schools: Bolles #20,
Dugger #37, Marpet, Judon, Mailata, Duvernay-Tardif, Meinerz, Oluokun; cfdarko labels them, but without a prospect id,
and load() dropped labels without one) or sat on an "undrafted" row (Maurice Hurst Jr. #140 and board #19,
Okoronkwo, Spencer Brown, Buck Allen: name mismatches); board-only rows could carry the wrong class (Hasan's 2015 #12
"Alvin" Dupree was a 2014 board player who went undrafted, separate from the drafted Bud Dupree with no rank); and
2014-2015 undrafted players had cell averages. The repair matches board rows to Hasan's board (rank + name, class
+-1; class from the board), links them to picks with the review's own matcher (review.board), merges a board-only
row into the drafted row that holds its pick, adds unlinked picks (labels from cfdarko by PFR id) and unmatched
board players, and gives 2014-2015 undrafted players their careers (review.udfa_pre2016_labels) or 0. Result:
110 picks added, 34 board players added, 26 classes corrected, 9 board-only rows merged, 5 relinked; drafted rows
per class now match the draft (e.g. 2014: 238 -> 256), no pick on two rows. Links cached in
data/derived/board_links.parquet; simulator tables carry `_lf`.

Engine, leave-one-draft-out (both scored on the repaired actuals): every band within +-0.02 of before; board players
mean abs error 0.0424 both. Past-drafts review, leave-one-draft-out (before rescored with the 2014-2015 undrafted
careers and the new yardsticks below): recommendations change at 22% of picks (round 1 4%); scorecard titles round 1
+1.07 -> +0.76, rounds 2-4 +0.14 -> +0.01, rounds 5-7 -0.06 -> -0.10, mostly bookkeeping and one coin flip: Carr's
recommendation moves from #22 to #33 (his +15.6 moves to round 2), and Hurts at #37 loses a near tie to Denzel Mims
(-29.5; his value 3.38 -> just under Mims's 3.06). Paired at changed picks: round 1 -0.06 titles in total (11
picks), rounds 3-4 +46, rounds 5-7 ~0. Adopted as a data fix.

**Yardsticks** (`review.Careers`). Seasons started, AV and second-contract pay came from draft records by (year,
pick), so any undrafted player scored 0 in every year (Collins: 0 starts). Now undrafted players (by name and rookie
season in nflverse's player list) get seasons with >= 50% of snaps (tracks PFR's count for drafted players, corr
0.94), AV = 8.25 x career snap-share sum (drafted players: R2 0.82) and best contract after his first season
(Collins: 6 starter seasons, AV 42.5, 5.3% of cap). Used by the scorecards (comp_correct, holdout_review, defer_test,
qb_reach) and the simulator's player pool. Effect on the past-drafts scorecard, rounds 5-7 (model minus team): starts
-0.40 -> -0.07, AV -4.5 -> -1.5, pay -0.45 -> -0.01 (the model's late picks who went undrafted had careers); rounds
1-4 within 0.1. Earlier sections quote starts / AV / pay on the old yardsticks.
Rebuilt: past drafts review (2014-2022 identical to the leave-one-draft-out run; Past drafts scorecard now round 1
+0.76, rounds 2-4 +0.01, rounds 5-7 -0.10 titles), simulator tables and artifact, pick-value chart, real trades, 2027
team report (every number within +-0.1), 2021 replay. Pick values within +-0.04.

## Earlier decisions on the new yardsticks (`scripts/rescore_check.py`, Nov 2026)

The saved leave-one-draft-out reviews behind each decision, rescored without rerunning any engine: realized titles
for 2014-2015 undrafted players (at each file's own cost / WR settings, detected by reproducing its stored values) and
starts / AV / pay from review.Careers. Model minus team, each recommendation once; titles, then starts:

| Decision | Version | R1 old -> new | R2-4 old -> new | R5-7 old -> new | R5-7 starts old -> new |
|---|---|---|---|---|---|
| Fall shading | 0.85 / 1.0 / 1.2 | +0.92 / +1.15 / +1.43, same | -0.11 -> -0.08 / -0.07 -> -0.04 / -0.22 -> -0.22 | -0.11 -> -0.09 / -0.13 -> -0.12 / -0.13 -> -0.11 | -0.79 -> -0.47 / -0.16 -> +0.11 / -0.43 -> -0.13 |
| Cap cost | 0.19 / 0.10 / 0 | +0.92 / +0.92 / +0.88, same | -0.11 -> -0.08 / -0.11 -> -0.08 / -0.09 -> -0.06 | -0.11 -> -0.09 / -0.08 -> -0.06 / -0.05 -> -0.04 | -0.79 -> -0.47 / -0.74 -> -0.41 / -0.64 -> -0.32 |
| WR x0.75 | 1.0 / 0.75 | +0.92 / +0.26, same | -0.11 -> -0.08 / +0.16 -> +0.18 | -0.08 -> -0.06 / -0.06 -> -0.04 | -0.74 -> -0.41 / -0.69 -> -0.35 |
| Comparison-career | off / ll | +0.92 / +1.07, same | -0.11 -> -0.08 / +0.42 -> +0.45 | -0.08 -> -0.06 / -0.11 -> -0.10 | -0.74 -> -0.41 / -0.68 -> -0.34 |
| Board-less / monotone | keep / drop / iso / min | +1.07 / +1.11 / +1.35 / +1.07, same | +0.42 -> +0.45 / -0.06 -> -0.03 / +0.23 -> +0.25 / +0.11 -> +0.14 | -0.11 -> -0.10 / -0.12 -> -0.12 / -0.10 -> -0.10 / -0.07 -> -0.06 | -0.68 -> -0.34 / -0.81 -> -0.48 / -0.52 -> -0.29 / -0.40 -> -0.07 |

Every version moves about the same (titles +0.00 to +0.03; rounds 5-7 starts +0.3, AV +3, pay +0.4), so no comparison
changes order and no verdict flips. QB reach penalty (rescored candidates): lam = 0 still chosen in 8 of 9 drafts.

Board deference (per-pick scoring, every pick each team made): rounds 1-4 unchanged in order (model ahead of or tied
with the board); rounds 5-7 the board's best now scores +0.19 vs the model's +0.13 per pick, but that is La'el Collins
"taken" at 100 different 2015 picks (the per-pick test lets one undrafted player be chosen many times). Without those
picks: model +0.06, board -0.01, as before (+0.04 / -0.04). Verdict unchanged; for this test, scoring each player once
per draft would be the better yardstick.

**Board deference rescored once per draft** (`scripts/defer_test.py`, Nov 2026). The test scored every pick each
team made, so one player could be "taken" at many picks (the board rule took La'el Collins at 100 picks in 2015,
once his career was scored). Now: picks where the rule's choice differs from the team's, each chosen player once per
draft, averaged over the picks where he was the choice; tuning uses the same score. (Counting him at his first such
pick instead, the review scorecard's convention, is fragile when comparing rules: a rule that moves a player to a
later pick changes his score without changing the choice, e.g. Watson at #1 vs Garrett +13, at #11 vs Lattimore
+32.) Candidates rebuilt with the current engine and yardsticks (model top = review recommendation at 98% of picks).
Titles vs the team's pick:

| | Model | Board | Margin (tuned) | Window (tuned) |
|---|---|---|---|---|
| Round 1 | +1.95 (+0.58 to +3.49) | -0.86 (-1.98 to +0.18) | +1.95 | +1.95 |
| Rounds 2-4 | +0.29 (-0.04 to +0.73) | -0.58 (-1.50 to +0.05) | +0.27 | +0.25 |
| Rounds 5-7 | -0.04 (-0.20 to +0.10) | +0.09 (-0.10 to +0.31, n 28) | +0.05 | +0.03 |

Rounds 1-4: the model ahead, tuning keeps it. Rounds 5-7: the board and board-leaning rules edge ahead within noise
(28 distinct board choices). Verdict unchanged: not adopted; rounds 2-7 stay "no tested edge". (Per-pick numbers are
still printed; old candidates kept as holdout/defer_candidates_nov5.parquet.)

**QB reach penalty, same scoring** (`scripts/qb_reach.py`, current candidates). Tuning now picks lam = 0.75 in 8 of
9 drafts (1.5 in 2017), but the tuned rule loses out of sample in round 1 (+1.95 -> +1.59; it drops Watson in 2017);
rounds 2-4 +0.29 -> +0.33, rounds 5-7 unchanged. Fixed lam 0.75 vs 0, round 1: +0.28 (-0.20 to +0.67 over draft
years), an isolated bump (0.5: +0.00, 1.0: -0.36). Scored at each player's first pick instead, the tuned rule
looked better (+0.85 -> +1.39), almost half of it Watson's re-anchoring alone; paired at the 20 changed choices
the penalty costs -2.2 titles each (-5.7 to +0.9). Verdict unchanged: not adopted.

## Scorecard: first pick and averaged (`review.once_per_draft`, Nov 2026)

The scorecard counts each recommended player once per draft (the model can recommend one player to many teams, but
only one could have drafted him). Two ways, now both reported, each player once per draft and round band:

- **First pick** (`how="first"`): at the first pick where the model recommended him, what a team following the model
  would have got. The conservative level vs teams; the Past drafts headline.
- **Averaged** (`how="mean"`): over every pick where he was the recommendation. Each comparison is real (each of
  those teams passed on him), but it flatters the model's level (later picks in a run face weaker team picks at a lower
  cost). Stable for comparing versions and rules, where the first pick is fragile (a change that moves a player's
  first recommendation re-anchors him: Watson vs Garrett at #1 +13, vs Lattimore at #11 +32). Used by
  comp_correct / no_board, holdout_review, rescore_check, defer_test, qb_reach; comp_correct prints the first-pick
  titles alongside.

Production (2014-2022), model minus team, titles: round 1 first pick +0.76 / averaged +1.89 (+0.47 to +3.43), rounds
2-4 +0.04 / +0.31 (-0.03 to +0.73), rounds 5-7 -0.07 / -0.07.

Earlier decisions on the averaged count (scripts/rescore_check.py, new yardsticks), titles R1 / R2-4 / R5-7:

| Decision | Versions |
|---|---|
| Fall shading (not adopted) | 0.85: +1.77 / +0.18 / -0.05; 1.0: +1.80 / +0.22 / -0.08; 1.2: +2.15 / +0.08 / -0.09 |
| Cap cost (0.10) | 0.19: +1.77 / +0.18 / -0.05; 0.10: +1.88 / +0.19 / -0.04; 0: +2.02 / +0.20 / -0.03 |
| WR x0.75 (not adopted) | 1.0: +1.88 / +0.19 / -0.04; 0.75: +0.93 / +0.26 / -0.03 |
| Comparison-career (adopted) | off: +1.88 / +0.19 / -0.04; ll: +1.56 / +0.63 / -0.03 (starts R2-4 -0.71 -> -0.45) |
| Board-less / monotone (drop + min) | keep: +1.56 / +0.63 / -0.03; drop: +2.01 / +0.36 / -0.05; iso: +2.09 / +0.58 / -0.07; min: +2.08 / +0.53 / -0.04 (R5-7 starts -0.15 / -0.21 / -0.30 / -0.06) |

No verdict reverses. One softens: the comparison-career correction's choices go from "slightly better in rounds 1 and
2-4" to mixed (round 1 worse, rounds 2-4 better), a tie; its adoption rested on the engine's held-out calibration
(top-10 non-QBs 0.70 -> 0.93 of actual), which stands. The training-data repair is slightly behind on this count too
(round 1 +2.08 -> +1.89, rounds 2-4 +0.53 -> +0.31, within noise; "Carr bookkeeping" explained part of the first-pick
drop, not all of it); kept as a data fix.

## Multi-sport athletes, MLB draftees, coded scouting language (Nov 2026)

`src/multisport.py`, `scripts/multisport_test.py`, `scripts/stereotype_words.py`. Expectations out of sample
(each class predicted by models of earlier classes): drafted players vs draft slot, board players vs consensus;
outcome = 4-season rookie-deal value.

**MLB-drafted** (every MLB draft 2005-2023 from statsapi.mlb.com, saved under data/raw/mlb_draft, not in git;
matched by name + MLB draft within 7 years before the NFL class + same birth date, or age when no birth date):
only 11 board / 7 drafted players 2016-22 (Mahomes, Kyler Murray, A.J. Brown, Hurst, Driskel, ...), almost all
late-round courtesy picks out of high school. Only 7 board players 2018-22 can be compared with consensus (Murray and
A.J. Brown beat it easily). Too few to use. (An earlier version reported x2.2 vs consensus; it wrongly counted 2016-17
board players, whom the consensus model can't predict, with an expectation of 0. Fixed in `expectations()`.)

**Other sports in the guides' BACKGROUND text** (2019-22, 1,553 matched; flags: basketball 32%, track 24%,
baseball 10%, wrestling 3%; controls for bio length):

| Flag | Drafted vs slot | Board vs consensus | Notes |
|---|---|---|---|
| Track | x0.84 (p 0.01) | x0.83 (p 0.01) | all from 2019-20 (x0.64-0.85); 2021-22 x0.94-0.98; same with 40 time + recruiting controls |
| Baseball | x1.23 (p 0.07) | x1.18 (p 0.13) | 3 of 4 years above 1 |
| Basketball | x0.93 | x0.89 | no gain at TE / WR / EDGE (x0.88) |
| Wrestling (OL / DL / EDGE / LB) | x1.19 (p 0.27) | x0.92 | 32 players |
| Any other sport | x0.89 | x0.86 | |

About 30 comparisons were run, so p 0.01 that holds in only two of four drafts isn't reliable. Not used in the model.

**Coded language by position** ("cerebral / instinctive", "gritty / motor", "athletic / explosive", "raw / project";
above vs below the position median): no pattern consistent with stereotyped evaluation. "Smart / gritty"
descriptions at C / LB / S and "athletic" ones at WR / CB / RB land within x0.92-1.17 of expectation (all p > 0.18).
"Raw" language goes with lower value at WR / CB / RB (x0.75 vs slot, p 0.02), matching the earlier bust signal.
QB "cerebral" x1.97 vs consensus (p 0.05) rests on 52 QBs. Nothing adopted.

## Coaching-staff connections (`src/coaching.py`, `scripts/coaching_test.py`, Nov 2026)

Staffs from Wikipedia's "{year} {team} season" pages 2012-2024 (8,475 coach rows; 1,233 coaches with pages,
1,192 careers parsed from their infoboxes: 84% of staff rows, 95% of head coaches / coordinators; raw wikitext
cached under data/raw/wiki, not in git). A pick is connected when someone on the drafting team's staff coached
at the player's final college in the 5 seasons before the draft. 220 of 3,327 picks 2012-24 (6.6%).

| Question | Result |
|---|---|
| Do teams draft connected players more? | a little: 6.6% of picks vs 5.7% if connections didn't matter (x1.15; round 2 x1.35) |
| Do they reach for them? | barely: taken x1.06 earlier than board rank vs x1.02 for other picks |
| Value vs draft slot, 2016-22 (128 connected picks) | x1.06 (95% 0.87-1.25); same with only players whose school had a link to some NFL staff |
| Coach there in the player's last 2 seasons (53) | x1.18 (0.85-1.49); rounds 1-3 x1.25 (p 0.16), rounds 4-7 x0.83 |
| Via head coach / coordinator (21) | x1.03 |
| Value vs consensus, board players 2018-22 (92 connected) | x1.05 (0.86-1.24); beat consensus 48% vs 40% (p 0.09) |

Teams lean slightly toward players their coaches know, and those picks turn out about like any other pick at
that slot: no sign that the inside knowledge is worth much, nor that familiarity misleads. Recent links in rounds
1-3 are the one hint (28 players). 2023-24 connected picks can be added once they have 4 seasons.

**Passes by connected teams** (`scripts/connected_pass_test.py`): does it mean something when a team whose
coaches know a player picks someone else while he's available? Drafted players 2016-22, value vs slot, controlling
for how many other staffs knew him.

| Measure | Value per unit | Placebo (same number of random unconnected teams) |
|---|---|---|
| Connected teams that picked before him and passed | x0.95 per pass (p 0.03) | x0.93 (p 0.00): same, so mechanical |
| Same, weighted by the chance a team picking there takes him given he's available (availability model around his consensus rank), controlling for the same sum over all picks | x2.9 per expected pick (95% 0.67-12.8) | x2.3 (0.45-11.8) |
| Connected teams that passed during his fall below board rank | x0.98 (p 0.65) | |
| Undrafted board players 2018-22, per connected staff | x0.94 vs consensus (p 0.38) | |

The unweighted "signal" is an artifact: with the number of connected teams fixed, more passes just means a later
pick, where the slot model is a bit high, and random teams show it equally. Weighted properly, informative passes
are rare: summed over the connected teams, a player has on average 0.05 expected chances to be taken (of 0.63
across all teams), so there is almost nothing to learn from, and the estimate is pure noise. A need-weighted
version (the team's chance to take his position) would sharpen the weights but can't create more passes.

## Body growth since high school, position changes (`scripts/growth_test.py`, Nov 2026)

Recruit height / weight / position from the College Football Data API (high-school recruits 2006-22, cached
under data/raw/cfbd_recruits, not in git), joined on the CFBD athlete id. Combine height / weight joined on PFR id.
Drafted players gained a median 15-23 lb from their recruiting listing, depending on position. Height listings
are no help: recruits average 0.3 in taller than they measure at the combine.

Per standard deviation (SD) of weight gained, at the same combine weight, years since recruiting and class:

| | Drafted vs slot, 2016-22 | Board vs consensus, 2018-22 |
|---|---|---|
| All positions | x1.06 (0.99-1.12, p 0.10; 955) | x1.06 (0.99-1.13, p 0.08; 714) |
| OL / DL / EDGE / TE | x1.01 | x1.02 |
| Other positions | x1.08 (p 0.05) | x1.07 (p 0.15) |
| Top quarter of gain at the position | x1.14 (p 0.08) | x1.13 (p 0.35) |
| Height gained | x0.99 | x1.01 |
| Recruited as "athlete" (ATH) | x1.03 (86) | x1.08 (60) |
| Recruited at another position group | x0.94 (119) | x0.92 (106) |

Without class fixed effects the consensus result looked strong (x1.15, p < 0.01). Most of that came from 2021-22:
those classes gained less (no 2021 combine), and the out-of-sample models miss by class. Teams already take
high-gain players slightly earlier than the board (x1.03 per SD, p 0.03). Weight gain is a weak, consistent lead,
mostly at skill positions; not enough to adopt. Recheck when 2023-24 have 4 seasons.

Bug found and fixed along the way: `expectations()` summed per-season predictions with NaN treated as 0, so
2016-17 board players got a consensus expectation of 0. The MLB and coaching consensus rows above are corrected.
The scouting-text, connected-pass and NFL-father tests only use 2018+ classes and were unaffected.

## Pre-draft team visits (`scripts/visits_test.py`, Nov 2026)

Visits from kentweyrauch.com's draft-visits dashboard as scraped by cfdarko (~/cfdarko/data/raw/draft_visits.parquet,
2015-26; top-30 private visits, combine and pro-day meetings, workouts; 2020-21 thin). 86% of rows matched on name +
class. cfdarko already showed visit counts add nothing beyond board rank and pick (experiments/00w, 00zh). Two new
questions, drafted players 2016-22, value vs draft slot:

| Question | Result |
|---|---|
| Drafted by a team that met him vs one that didn't (1,295 players with a reported meeting; 423 by a team that met him) | x1.09 (0.97-1.23, p 0.14); top-30 host x1.09 (p 0.21); picks 1-100 x1.13 (1.01-1.29, p 0.05); 101+ x0.99; without 2020-21 x1.07 |
| Passes by teams that met him, weighted by the chance a team picking there takes him | x0.75 per expected pick (0.45-1.24); top-30 hosts x0.61 (0.30-1.24); placebo with random teams x1.20 |

Both point the way inside information would (teams that met a player do a little better taking him; their passes
hint at worse outcomes, unlike the placebo), but neither is significant, and a typical player has only 0.10
expected picks' worth of passes by teams that met him. Caveat: when a player has one reported meeting, that team
drafted him 7% of the time (vs ~3% by chance), so some meetings may have been reported after the draft.
Same pattern as coaching connections (x1.06): picks made with inside contact run 5-10% better than the slot, never
reliably. Not adopted.

**Pooled direct contact** (`scripts/contact_test.py`, specified before looking): drafting team had a staff link to
his college OR a reported meeting with him. Drafted 2016-22 with some possible contact (1,636; 523 by a contact
team; visits dominate: 423 meeting, 128 coaching, 28 both):

| | Value vs slot |
|---|---|
| Contact drafter vs not (primary) | x1.10 (0.98-1.23; Poisson p 0.06); beat slot 45% vs 39% (p 0.09) |
| Picks 1-100 / 101+ | x1.14 (p 0.03) / x1.03 |
| Without 2020-21 | x1.08 (p 0.13) |
| By class 2016-22 | x1.20, 1.17, 1.17, 0.94, 1.00, 1.29, 0.93 |

Pooling adds little, since coaching links are few. The direction holds but is borderline: about 10% more value when
the drafting team had direct contact, concentrated in the top 100 picks, about 1.0 in three of seven classes. Even
if real, it's only known once the player's team is known, so it could adjust post-draft player outlooks, not a
pre-draft board. Recheck when 2023-24 have 4 seasons.

## Height and size cutoffs: mispricing tests (`scripts/size_bias_test.py`, Nov 2026)

A bias toward a trait would show up twice: the trait moves draft position relative to the board, and players with
it then miss their slot. Raw height / arm / hand from cfdarko's combine_metrics (arm and hand from 2011).

**Pricing** (drafted board players 2014-24): height within position barely moves picks relative to the board
(x1.00 per standard deviation (SD); taller DBs go slightly later, x0.95, p 0.03). Short QBs (under 6'1", 11),
QB hands under 9 in (4), OT arms under 33 in (16) and CBs under 5'11" (51) go about where the board has them.
Teams and the board agree on size.

**Value:**

| | Drafted vs slot 2016-22 | Board vs consensus 2018-22 |
|---|---|---|
| Height per SD, all positions | x1.00 | x1.01 |
| EDGE | x1.13 (p 0.06); x1.18 (p 0.01) with arm or weight held fixed | **x1.26 (1.10-1.44, p 0.001)**; 2018-19 x1.31, 2020-22 x1.22 |
| WR / QB | x1.15 (p 0.06) / x0.78 (p 0.08) | x1.00 / x0.91 |
| Short QB, QB hand < 9 in | 4 players: too few | too few |
| OT arm < 33 in (10 / 7) | x1.31 (p 0.13) | x1.18 |
| CB under 5'11" (27 / 25) | x0.89 | x1.08 |

The cutoffs can't be tested: too few drafted QBs fall below them, and OT / CB show nothing reliable. One result
stands out: **tall edge rushers are undervalued, short ones overvalued.** Value vs consensus rises steadily with
height (6'3" and under x0.75, 6'4" x0.86, 6'5" x1.04, 6'6"+ x1.23). It holds in both halves of the classes and
with arm length or weight controlled, and survives correction for the ~20 tests run here. It runs opposite to a
general bias toward height: the board and teams seem to overrate undersized, bendy pass rushers. Caveats: 137
board players; part may be what the value measure rewards (tall ends play more snaps, including run downs), not
pass-rush quality. Candidate model feature: EDGE height. Recheck with 2023-24.

**EDGE height in the model** (`scripts/pm_edge_height.py`, test classes 2018-22, consensus_pre with REG parameters):
(a) edge height as a feature, (b) prediction x exp(b x height z), b fit only on earlier classes' out-of-sample errors
(x1.07 in 2019, then x1.18-1.30 per SD, consistent with the residual test).

| | consensus_pre | + EDGE height feature | placebo (heights shuffled) | rolling adjustment |
|---|---|---|---|---|
| EDGE board players (137): rank correlation | 0.706 | 0.707 | 0.697 | 0.714 |
| EDGE: Poisson deviance (lower = better) | 0.0499 | 0.0499 | 0.0509 | 0.0512 |
| All board: rank correlation | 0.628 | 0.626 | 0.625 | 0.629 |
| All board: top-32 / top-100 value captured | 0.751 / 0.813 | 0.756 / 0.811 | 0.751 / 0.814 | 0.748 / 0.814 |

The trees barely use the feature (about 30 board edge rushers a class against a 150-player minimum leaf). The
adjustment ranks edge rushers slightly better (+0.008, better in 3 of 4 adjusted classes) but fits their value levels
slightly worse, and board-wide numbers move within noise. Real in the residuals, too small to change rankings
reliably: not adopted. Worth keeping as a tiebreaker note on edge rushers.

## Race: audit with name- and place-based probabilities (`scripts/race_audit.py`, Nov 2026)

Race is in none of our data and is not a model input. Each board / drafted player gets a probability of each race
from public aggregates (Census 2010 surnames, Tzioumis 2018 first names, Census county population by race via the
recruiting hometown), the BIFSG method used in fair-lending audits. Only group-level results are reported; the
derived file stays in data/ (not in git). Inputs found: surname 93%, first name 64%, home county 59%.

The proxy ranks positions sensibly (mean P(Black): K 0.05, C 0.09, QB 0.17 ... CB 0.37) but is badly compressed:
its general-population prior puts CBs at 0.37, far below their real share. So it separates players weakly, and the
slopes below (player certainly Black vs certainly not) are not interpretable as sizes; only direction and
significance mean anything.

| Group | Pricing: taken vs board, 2014-24 | Value vs slot, 2016-22 | Value vs consensus, 2018-22 |
|---|---|---|---|
| All positions | x0.96 (p 0.11) | x0.90 (p 0.24) | x0.83 (p 0.09) |
| QB (125 / 81 / 70) | x0.86 (p 0.38) | x1.76 (1.00-3.07, p 0.05) | x1.59 (p 0.34) |
| C / LB / S | x0.99 | x1.08 | x0.86 |
| WR / CB / RB | x0.93 (p 0.06) | x0.81 | x0.82 |
| All others | x0.98 | x0.80 (p 0.05) | x0.82 |

No reliable evidence of racial mispricing. The QB row points the way a bias against Black QBs would (taken a bit
later than the board, then beating the slot), but rests on 81 QBs, is one of 15 tests, and doesn't hold against
consensus. With a proxy this weak a null isn't evidence of no bias. A real test needs self-identified or
researcher-coded race, e.g. from a published study's replication data.

## Press-conference language (`src/transcripts.py`, `scripts/asap_collect.py`, `scripts/transcripts_test.py`, Nov 2026)

ASAP Sports transcripts of college bowl / title-game / kickoff press conferences 2011-2022 (3,675 interview links,
287 events; conference media days transcribe coaches only; no combine). 843 interviews matched to board / drafted
players by name + class window: 22,373 answers from 518 players (median 1,247 words, 1 interview). Raw pages cached
under data/raw/asap (not in git). Transcriber notes ("(Inaudible.)", "(Laughter.)") stripped.

Features fixed before looking; per standard deviation (SD), controlling for amount of text and class:

| Feature | Pricing (taken vs board) | Drafted vs slot 2016-22 (260) | Board vs consensus 2018-22 (210) |
|---|---|---|---|
| Words per answer | x1.02 | x0.90 (p 0.05) | x0.94 |
| Technical football vocabulary | x1.01 | x0.91 (p 0.11) | x0.90 (p 0.10) |
| "We" share of I / we | x1.03 | x0.96 | x0.98 |
| Cliches | x1.00 | x0.98 | x0.97 |
| **Mean word length** | x0.99 | **x0.85 (0.77-0.93, p < 0.001)** | **x0.83 (0.75-0.92, p < 0.001)** |
| Secondary: categorical-dynamic index | x1.06 (p 0.02) | x0.88 (p 0.03) | x0.89 (p 0.09) |
| Secondary: hedges / fillers | x1.03 | x1.00 | x1.02 |

Mean word length survives correction for the 10 primary tests and every check: position, age, event type and
years before the draft, >= 600 words, excluding QBs, dropping each class in turn (x0.82-0.85), excluding proper
nouns (x0.84 / x0.81), and controlling for the race-proxy probability (correlation -0.07; x0.85 / x0.83). Players
who use longer words (more formal register: "testament", "grateful", "necessarily", "character") turn out worse
than both their slot and their board rank, while going about where the board has them. Neither the board nor teams
price it. Opposite to the essay literature (Pennebaker's analytic style predicts higher GPA). QBs: everything
larger (26 players). Unexplained; see the replication below.

Is it the board rewarding polish that doesn't matter on the field (`scripts/polish_test.py`)? The pricing column
only compares teams with the board, so it can't tell. Two direct checks: (1) against a stats-only model of board
rank (LightGBM on measurables, production, recruiting, age; cross-fitted on 3,474 board players, corr 0.71), word
length doesn't buy a better rank (x0.99 of stats-expected, 95% 0.92-1.07, 446 players); (2) against a stats-only
outcome model (no rank or pick), long-word players still fall short (x0.84, p 0.01; vs consensus x0.83). So the
board isn't overrating polished players; long words flag something the stats miss too.

Fresh replication, classes 2023-25 (`scripts/transcripts_replication.py`, fixed before outcomes were seen; 2023-24
calendar years added to the crawl: 4,467 links, 1,034 matched interviews). Word length alone, standardized with the
discovery mean and sd, value over seasons played so far: drafted vs slot x0.95 (95% 0.84-1.07, one-sided p 0.20,
165 players); board vs consensus x0.97 (0.86-1.09, p 0.32, 193). By class vs slot: 2023 (3 seasons) x0.87, 2024
x1.03, 2025 x1.03. In the discovery data the effect is already x0.86 after two seasons (x0.92 after one), so the
short window doesn't explain the miss for 2024. **Not replicated.** The interval still includes the discovery
estimate, but the finding should be treated as likely overstated, perhaps a chance result; recheck as 2024-25 mature.

## Draft biases from the literature (`scripts/lit_bias_test.py`, `scripts/lit_bias_replication.py`, Nov 2026)

A literature search turned up six documented or claimed biases not yet tested here. Each was tested like the size
tests: pricing (drafted board players 2014-24; > 1 = teams take them earlier than the board) and value out of
sample (drafted 2016-22 vs slot, board 2018-22 vs consensus; < 1 = overvalued). One primary measure each, fixed
before running; Bonferroni bar p < 0.004.

| Bias (source) | Measure | Pricing vs board | Value vs slot | Value vs consensus |
|---|---|---|---|---|
| Program prestige (Kitchens 2015, *Econ Inquiry*) | final-season team SP+, per SD | x0.95 (p<0.01) | x1.01 | x0.96 |
| Small school (Hendricks, DeBrock & Koenker 2003, *JOLE*) | non-Power-5 | x1.07 (p<0.01) | x0.97 | x1.13 (p 0.12) |
| | FCS / lower | x1.13 (p<0.01) | x1.01 | x1.02 |
| Recruiting pedigree (anchoring) | 247 composite, per SD in position | x0.94 (p<0.01) | x0.98 | **x0.91 (p 0.01)** |
| | five-star | x0.89 (p<0.01) | x1.05 | x1.02 |
| Final-season jump (PFF one-year-wonder studies) | last PFF grade minus best earlier, per SD | x1.01 | x0.97 | x1.00 |
| | 80+ final season after < 70 (54 players) | x1.03 | x0.78 (p 0.04) | x0.80 (p 0.13) |
| Showcase exposure | Senior Bowl roster | x1.01 | x0.94 | x1.13 (p 0.06) |
| Relative age (Heneghan & Herron 2019; Beyer et al. 2016) | born Sep-Dec vs May-Aug, age controlled | x0.95 (p 0.01) | x1.01 | n/a (birth dates only for NFL players) |

Nothing passes the multiple-testing bar on value. Leads checked further:
- **Recruiting pedigree vs consensus: not real.** All of it comes from 2018-20 (x0.83); 2021-22 x1.02; fresh
  replication on 2023-25 (fixed in advance, seasons played so far) x1.06 (one-sided p 0.88), vs slot x1.18.
- **One-year wonder: not replicated** (2023-25 x1.02, 28 players). The continuous measure was null from the start.
- **Non-Power-5 vs consensus: not replicated** (x1.00).
- **Senior Bowl vs consensus: consistent but not proven.** Replication x1.16 (one-sided p 0.03); with age at draft
  controlled (Senior Bowl players are older, and older players pay off sooner) x1.13 in discovery (p 0.13) and x1.11
  in replication (p 0.12). Against draft slot it's ~x1.0 in both: teams already price it. So it is at most a small
  board miss that teams correct, like the other information that lives in draft slot.

The pricing column is the solid result: relative to the media board, teams take small-school players earlier and
players from strong programs or with high recruiting ratings later. That is the opposite of the prestige bias in
Kitchens (2015). Since value vs slot is flat for all of these, teams appear to price them about right, and the
board leans slightly toward pedigree. That matters for predicting where players go (availability), not for
player value. Not adopted. Untested from the search: local/home-region drafting (Tufts thesis: some teams overdraft
local players, no performance cost; needs a random-team placebo) and Keefer's (2015, 2017) sunk-cost effect, where
higher picks get more playing time at equal ability. That one would bias outcome labels toward high picks rather
than show a draft mistake.

## Sunk-cost playing time (`scripts/sunk_cost_test.py`, Nov 2026)

Keefer (2015 *J Labor Res*; 2017 *J Sports Econ*) finds that teams start first-round picks more than their play
warrants, because they paid more for them. That matters here because the outcome labels are priced on snap share
as well as PFF grade, so extra snaps given to high picks would count as draft-slot "value". Data: all 4,190 drafted
non-specialists 2006-22, years 1-4, PFF NFL season files (99% linked). Snap share = snaps / a typical team's full
season on that side of the ball.

**A. Round-boundary jumps** (local linear in pick, |r| <= 16, position + class fixed effects + drafting team's
prior win %). Draft order puts the best teams at the end of each round and the worst at the start of the next, so
the test is the 1/2 jump minus the average 2/3 and 3/4 jumps:

| Outcome (years 1-4) | 1/2 jump | 2/3 | 3/4 | 1/2 minus others |
|---|---|---|---|---|
| Snap share, 2006-22 | -0.07 | +0.08 | -0.12 | -0.05 (z -0.9) |
| Snap share, 2006-10 (no wage scale) | +0.13 | +0.17 | -0.11 | +0.11 (z +1.1); grade jump +4.5 too |
| Snap share, 2011-22 | -0.15 | +0.05 | -0.12 | -0.12 (z -1.6) |
| PFF grade, 2006-22 | -0.1 | +2.1 | -0.4 | -0.9 (z -0.4) |
| Label value y1-4 (cap share), 2012-22 | -0.031 | +0.006 | -0.022 | -0.023 (z -0.9) |

Same at |r| <= 10 and 24. **No first-round sunk-cost jump**, and since 2011 the sign is the wrong way.

**B. Feedback in years 2-4** (same snap share and grade the season before; clustered by player). Per log-unit of
pick (pick 10 vs 100 = 1.7 units), a higher pick predicts more snaps next season (0.067 share) and also a better
next-season grade (1.8 points), so the pick carries real information. Controlling for same-season grade (200+ snaps
both years): pick still predicts 0.035 share (se 0.004), against a total gradient of 0.085 among these regulars and
0.23 among all players. The first-round dummy on top of that is -0.014 (2011-22: -0.03): no contract effect.

Reading: there is no discontinuity of the kind Keefer describes after the 2011 rookie wage scale. What's left is a
smooth premium: at the same PFF grade, higher picks play somewhat more. That could be teams knowing more than PFF
grades show (the same information draft slot carries everywhere else here) or reputation. These data can't
separate the two. **Upper bound on label inflation:** roughly 15% of the pick-to-playing-time gradient in years 2-4
(40% among regulars) isn't explained by grade. Year 1 can only be checked at the boundaries, which show nothing.
Labels not changed. Caveat for the "teams' slot beats the board" results: some of slot's edge could be teams giving
their own high picks snaps. A grade-only (snap-free) label would test this directly.

## Grade-only label: how much of draft slot's edge is playing time? (`scripts/grade_only_label.py`, `scripts/grade_only_check.py`, Nov 2026)

Follow-up to the sunk-cost test. The grade-only label runs cfdarko's own label builder
(`rookie_value.build_label`, read only, output in data/derived) with one change: in years 1-4, any season with
snap share >= 0.2 is priced at the share that veterans (year 5 on) of the same PFF quality play in his position
(isotonic fit), not the share his team gave him. Pay curves, quality, games, costs and the pre-2013 depth-chart
model are otherwise identical. Check: with the threshold set so nothing is replaced, the builder reproduces the
stored label exactly (0 difference on all 6,795 players). 8,639 rookie seasons replaced; correlation with the
stored 4-year value 0.82-0.89 by season. Biggest drops: top-3 QBs who started on poor grades (Bortles, Darnold,
Trubisky); biggest gains: good mid-round players held below what their grades earn (Godwin, Andrews, Renfrow).
`prospect_model.LABEL_FILE` switches labels (default unchanged). Caveat: pricing at E[share | quality] pulls
everyone toward the middle; a good DL who earns 85% of snaps is priced at the 64% veterans of his grade play.

Primary (fixed in advance): board players drafted 2018-22, value vs out-of-sample consensus expectation (refit on
each label) against reach = log((rank + 10) / (pick + 10)):

| Label | b (reach) | Taken 2x earlier than board -> value vs consensus | Rises b | Falls b | Slot vs consensus Spearman |
|---|---|---|---|---|---|
| Stored | +0.74 (se 0.09) | x1.67 | +0.73 | +0.75 | 0.620 vs 0.580 (+0.040) |
| Grade-only, share >= 0.2 | +0.63 (se 0.09) | x1.54 | +0.59 | +0.67 | 0.584 vs 0.552 (+0.032) |
| Grade-only, share >= 0.35 | +0.67 (se 0.09) | x1.59 | +0.65 | +0.70 | 0.596 vs 0.561 (+0.035) |

Paired bootstrap, grade-only minus stored: -0.11 (95% -0.19 to -0.04) at 0.2; -0.07 (-0.13 to 0.00) at 0.35.
**About 85-90% of the teams' edge over the board survives** without team-chosen playing time; 10-15% of it was
teams playing the players they chose. The drop is larger for rises (players teams took early, -20%) than for falls
(-10%), as expected if teams give snaps to their own reaches. Top-pick premium (mean 4-year value, picks 1-10 vs
33-64): 2.27 -> 2.14 (part of that is the compression above).

Reading: team information in draft slot is real and the main earlier conclusions stand (fall shading, board
deference, "public features don't beat slot"). The stored label overstates slot's edge over the board by roughly a
tenth and the top-pick premium by ~5%. Stored label kept as the default: the grade-only one discards real
playing-time information (durability, role) along with the sunk-cost part. Candidate follow-up: recheck the fall
weight (0.85) on the grade-only label.

### Fall weight on the grade-only label (`scripts/fall_weight_gradeonly.py`)

The fall weight w (how far a falling player's log rank is shaded toward the log of his actual pick; 0.85 in the app,
review and simulator) re-estimated on 4-year value: board players 2014-22 who fell, Poisson value ~ a log(pick+10) +
b log(rank+10) + position + class, w = a / (a + b), same players under each label, paired bootstrap (500).

| Fallers | n | Stored | Grade-only 0.2 | Grade-only 0.35 | Grade-only 0.2 minus stored |
|---|---|---|---|---|---|
| All (undrafted = pick 300) | 1,529 | 0.73 (0.46-1.08) | 0.66 (0.34-1.03) | 0.66 (0.36-1.03) | -0.07 (-0.20 to +0.06) |
| Drafted | 864 | 0.86 (0.51-1.29) | 0.82 (0.45-1.30) | 0.81 (0.45-1.28) | -0.05 (-0.25 to +0.16) |
| Picked in rounds 1-3 | 353 | 0.95 (0.40-1.68) | 0.84 (0.11-1.60) | 0.83 (0.14-1.60) | -0.11 (-0.50 to +0.36) |

Taking out team-chosen playing time lowers w by 0.05-0.1, as predicted, but no difference is distinguishable from
zero, and 0.85 sits inside every interval on both labels. The weight is poorly pinned down on value labels in
general (intervals ~0.5 wide); earlier tests showed recommendations and scorecards barely move between 0.85 and 1.0.
**Kept at 0.85.**

### Bracket check: pick values and real trades on both labels (`scripts/label_bracket.py`)

Stored label = upper bound for high picks (every snap counts); grade-only = lower bound (discards team-chosen
playing time, real role information and the top of the snap distribution). Full pick-value grid and all 394
pick-only trades 2013-25 rescored on each; the stored rerun reproduces the live pick values exactly (0 difference).
Decision rule fixed before running: leave labels alone if no trade verdict flips (opposite signs, both > 2 se)
and values relative to pick 32 move < ~10%.

Typical non-QB pick, titles added (% of one Super Bowl), contender with a veteran QB (other situations alike):

| Pick | 1 | 5 | 10 | 20 | 32 | 64 | 96 | 160 |
|---|---|---|---|---|---|---|---|---|
| Stored | 2.80 | 2.85 | 3.59 | 2.92 | 2.48 | 1.87 | 1.26 | 0.78 |
| Grade-only | 1.37 | 1.44 | 2.33 | 2.07 | 1.74 | 1.54 | 0.98 | 0.71 |
| Relative to pick 32 | -30% | -28% | -8% | +1% | 0 | +17% | +11% | +29% |

QB picks (relative to pick 32): picks 1-20 -27% to -37%, pick 96 -65% to -70%. Absolute values fall more than
4-year label value (top-10 label -26%) because the rookie contract cost is fixed: net value falls faster than gross.

Real trades: correlation 0.986; trade-ups gain titles 10% of the time on both labels; 22 sign changes of 788
team-sides, 3 firm flips (both > 2 se): Carolina 2022 trade-up for Matt Corral (#94: +0.57 -> -0.37), New England
2019 trade-down (-> +0.11 to -0.12; pick 167 later used on a QB) and Detroit 2025 trade-up (-0.18 -> +0.12).
(Corrected: an earlier version of this note named the NYJ 2015 Bryce Petty and Oakland 2016 Connor Cook trade-ups
as flips; they go from clear gains to ~0, inside 2 se, so they are sign changes, not firm flips.)

**The rule trips on both counts**, so the labels can't be left alone without a narrower estimate. But the bracket
is wide for a reason beyond sunk cost: the grade-only label prices every full-time player at the average share of
his grade, which hits exactly the top picks and starting QBs. The sunk-cost test put the pick-attributable part at
<= ~15% of the playing-time gradient. Next: the pick-attributable correction (rookie share modeled on quality,
prior snaps and pick; seasons priced at a mid-round pick's share), which should land between the two bounds.

### Pick-attributable correction (`scripts/pickadj_label.py`, bracket in `scripts/label_bracket.py`)

Same harness (reproduces the stored label exactly with nothing changed). Rookie-deal seasons of drafted players:
share' = share + b (log(110) - log(pick + 10)), i.e. priced at the share a pick-100 player of the same quality gets;
each player keeps his own residual. Fit by OLS on share ~ log(pick+10) (+ QB slope) + quality, quality^2, prior
share, prior quality, career year, group, season.

- **Spec fixed in advance ("pre", rejected):** all seasons with snaps. b = -0.117, but quality entered U-shaped
  (cfdarko's quality is a percentile of a grade shrunk toward average by snaps, so low-snap players sit near
  average), so pick absorbed ability. Kept for the record (`labels_pickadj_pre.parquet`), not used.
- **Measured-quality seasons (share >= 0.2, + prior quality):** b = -0.075 (QB -0.051). Split by year:
  **rookie season -0.126, years 2-4 -0.049** (the sunk-cost test found -0.035 to -0.042). In a rookie season there
  is no NFL evidence yet, so draft position is the team's scouting information and can't be told apart from sunk cost.
- **Narrow (years 2-4 only):** b = -0.049 (QB -0.027); shifts a #1 pick's share in years 2-4 by -0.11, #32 by
  -0.05, #250 by +0.04. This is the part the evidence supports; it is still an upper bound (b also carries what
  teams know beyond PFF grades).

Value relative to pick 32 vs the stored label (contender with a veteran QB; other situations within a few points):

| Label | #1 | #5 | #10 | #20 | #64 | #96 | #160 | QB #1 | QB #10 | Trade flips |
|---|---|---|---|---|---|---|---|---|---|---|
| Narrow (years 2-4) | -15% | -13% | -7% | -4% | +7% | +8% | +14% | -19% | -7% | 0 (corr 0.997) |
| Pick-adjusted, all years | -42% | -37% | -21% | -3% | +17% | +23% | +34% | -59% | -39% | 0 (corr 0.937) |
| Grade-only | -30% | -28% | -8% | +1% | +17% | +11% | +29% | -29% | -30% | 3 (corr 0.986) |

Trade-ups gain titles 10% of the time under all four labels. No trade verdict flips under either pick correction.
Reading: the supported (narrow) correction trims top-5 picks by ~15% relative to the late first round and lifts
picks 64-160 by 7-18%; the rookie-season premium is the open question, and counting it all as bias roughly triples
the trim. **Adopted Nov 2026** (next section).

### Narrow label adopted (`prospect_model.LABEL`, Nov 2026)

Default outcome label is now "narrow" (`data/derived/labels_pickadj_narrow.parquet`; data/ isn't in git, so rebuild
it with `python scripts/pickadj_label.py narrow`). `LABEL=stored` restores cfdarko's label. The engine, the realized
values of past picks (`review.RealizedV2`) and the position-moves table read the same file; simulator tables carry
the label in their cache key (`_lbnarrow`). Results in the sections above were computed on the stored label
unless they say otherwise; rerun with `LABEL=stored` to reproduce them.

Rebuilt: pick values and chart, real trades, 2027 team report, past drafts 2014-26 (picks, trades, trade decisions),
336 simulator tables, simulator page. App tests pass.
- **Pick values** match the narrow bracket (non-QB #1 2.80 -> 2.15, #32 2.48 -> 2.25, #160 0.78 -> 0.81 % of a title,
  contender with a veteran QB; QB #1 11.9 -> 9.6, middle of the pack).
- **Real trades:** correlation 0.997 with before, no firm flips; trade-ups gain titles 10% of the time.
- **2027 teams:** value of each team's firsts 1.27 -> 1.06 on average; trading down stays positive for every
  team and gains a little for most (top picks are worth less).
- **Past-drafts scorecard** (2014-22, model minus team, each recommendation at its first pick): round 1 +0.76 ->
  +1.02 per recommendation, rounds 2-4 +0.05 -> +0.42, rounds 5-7 -0.11 -> -0.15. Recommendations unchanged at
  94% / 91% / 77% of picks (rounds 1 / 2-4 / 5-7). On unchanged recommendations the new yardstick moves rounds 2-4
  only +0.09 -> +0.11; most of the round 2-4 jump is one new recommendation: 2020 New England #37 takes Jalen Hurts
  instead of Kyle Dugger (+28.7). That's one QB hit, not evidence of a round 2-4 edge; guidance unchanged
  ("no tested edge after round 1").

**Undrafted players** (`scripts/udfa_slope_test.py`): should the year 2-4 shift extend past the draft (pick 300)?
Same fit with undrafted players added (1,015 seasons, 600 players). At equal quality they play -0.001 share vs the
line extended to pick 300 (95% -0.017 to +0.014) and -0.008 vs the last pick of the draft (-0.023 to +0.007): they
sit on the line, but the line only falls 0.006 between picks 262 and 300. Rule fixed in advance (extend only if they
play clearly less than the last picks): not met. Left unchanged.

## Position synergies and build order (`scripts/synergy_test.py`, Nov 2026)

Folk beliefs ("build the trenches first", "get the QB weapons", "pass rush and coverage feed each
other", "a line is only as good as its weakest player") imply interactions between units. Unit
quality = PFF grade, snap-weighted, z-scored within season (QB passing; OL; WR/TE routes; RB; ED/DI
pass rush; front run defense; CB/S coverage), 2006-2025, 640 team-seasons. Outcomes =
opponent-adjusted offensive / defensive points per game from schedules.

| Offense (pts/game per SD) | Main effect | Interaction | 95% |
|---|---|---|---|
| QB | +2.03 | QB x OL | +0.19 (-0.08 to +0.47); team FE +0.04 |
| WR/TE | +1.27 | QB x receivers | +0.02 (-0.19 to +0.22) |
| OL | +0.89 | OL x RB | -0.07 (-0.27 to +0.13) |
| RB | +0.48 | OL x receivers | +0.12 (-0.16 to +0.41) |
| **Defense** | | | |
| Coverage | +1.85 | rush x coverage | -0.05 (-0.27 to +0.18) |
| Run defense | +0.70 | run D x coverage | -0.08 (-0.26 to +0.10) |
| Pass rush | +0.54 | | |

- Units add up. Interactions raise R2 by <=0.004 and predict 2018-25 slightly worse than the
  additive model (offense RMSE 2.33 vs 2.31). No curvature in OL, pass rush or coverage; QB slightly
  convex (+0.19, -0.03 to +0.40).
- Only hint: the OL slope rises with QB quality (bottom / middle / top QB tercile +0.72 / +0.81 /
  +1.12), but it disappears within teams (team FE). Can't rule out a moderate QB-OL complement.
- Weakest link: OL mean is what matters; with the mean in the model, worst starter -0.06 and best
  starter -0.21 (both n.s.). Neither weak-link nor strong-link.
- Rookies (drafts 2014-2022, 4-year value on the narrow label, controls log pick + log consensus by
  position): value doesn't depend on the complementary unit he joins (last season's grade, per SD):
  WR/TE with QB +0.34 (-0.42 to +1.10) and OL -0.34; RB with OL +0.15; OL with QB +0.29; ED/IDL with
  coverage -0.38; CB/S with pass rush +0.23; QB with OL +1.37 (-2.8 to +5.5, uninformative). Means
  ~4-7, so effects beyond ~+/-12% per SD are ruled out for non-QBs.
- Build order (rounds 1-2, 3-draft windows, 160 team-windows): trench share of early-pick capital vs
  net rating 2 seasons later, controlling for start rating and capital: 25% -> 75% trench = +0.5
  pts/game (-2.3 to +3.4). No evidence either way.
- Caveats: grades are partly shared (receiver grades track QB, r 0.68; coverage grades reflect
  the rush), which muddies main effects more than interactions; main effects aren't per dollar.
  The model's additive treatment of positions (bridge, pick values) stands.

## Replacement level by position (`scripts/replacement_level.py`, Nov 2026)

`pick_effect` credits a pick with snap share x (price - 0.5% of cap) and charges rookie pay - 0.5%: one
minimum-salary replacement level for every position. Since outcomes_v2, though, a season's value is
cfdarko's veteran pay curve price_g(share, quality) (/1.10), and that curve pays mostly for playing time.
A full-time player (90% of snaps) at the very bottom of his position's PFF quality (q = 0) is priced at
DL 6.4, EDGE 6.3, QB 5.4, RB 4.6, WR 3.7, TE 3.7, DB 3.0, LB 2.3, OL 1.8% of cap, vs the 0.45% the model
subtracts.

Freely available players (contract in force <= 1.25x the veteran minimum, or an undrafted rookie deal;
4,103 seasons 2013-25, 3-9% of each group's snaps) priced the same way earn 2.4-6.0% of cap per full
season of snaps. Their median quality when they play 30%+ (q 0.23-0.74) is selected upward (the ones who
play are the ones who turned out good), so fixed low reference qualities are used below.

**Team check (416 team-seasons 2013-25).** Split curve-priced roster value into a playing-time part
(each player's snaps priced at reference quality q_ref) and a quality part (the rest), season-demeaned,
against opponent-adjusted net rating:

| q_ref | Playing-time part, pts per 1% of cap | Quality part | Gross value |
|---|---|---|---|
| 0.1 | -0.05 (-0.15 to +0.05) | +0.48 (+0.41 to +0.54) | +0.31 |
| 0.3 | -0.07 (-0.16 to +0.03) | +0.48 (+0.42 to +0.54) | |
| 0.5 | -0.01 (-0.08 to +0.07) | +0.49 (+0.43 to +0.55) | |

The playing-time part buys nothing: crediting a pick for the curve price of his snaps, net of only 0.5%,
counts value a replacement in the same snaps would also have produced. The right comparison is quality
over a replacement at the same snaps.

**What it would do** (drafted players 2014-22, 4-season team-strength points: current rule vs quality
over q = 0.1 x 0.48): all picks x0.70-0.74 (R1 2.95 -> 2.07, R2-3 1.22 -> 0.87, R4-7 0.47 -> 0.35), and a
big position reshuffle: OL x1.28, WR x1.13, QB x0.67, DL x0.65, LB x0.57, TE x0.59, EDGE x0.54, DB x0.49,
RB x0.32. Round-1 RBs and LBs fall to ~0.1-0.2 points.

Not adopted. Caveats before any choice test:
- The reshuffle follows how steeply each position's pay curve rises with PFF grade (steep at OL / WR,
  nearly flat at DB / LB / RB). Where PFF grades measure quality poorly, the quality part understates.
- Team check is same-season (grades partly reflect team results, inflating the quality slope) and pools
  positions under one slope.
- The 0.48 rate is in a different currency from the bridge's 0.19 (contract prices); the cost side
  (rookie pay at 0.10 per 1%) is unchanged.
Next: per-unit team check with prior-season quality; then the whole-draft choice test.

### Per-unit check (`scripts/replacement_units.py`; `points` argument for the drafted-player table)

Same split per pricing group, offense units vs offensive rating and defense vs defensive rating,
team-seasons 2014-25, quality from the same season ("now", leaks) or the player's prior season ("prior";
85% of snaps have one, else group-neutral). Pts/game per 1% of cap:

| Unit | Playing time, now | Quality, now | Playing time, prior | Quality, prior |
|---|---|---|---|---|
| QB | -0.01 | +0.50 | +0.48 | +0.18 |
| RB | +0.03 | +1.08 | +0.30 | +0.57 |
| WR | -0.22 | +0.44 | -0.09 | +0.18 |
| TE | -0.52 | +0.92 | -0.07 | +0.74 |
| OL | -0.08 | +0.19 | +0.13 | +0.17 |
| DL | -0.05 | +0.20 | -0.03 | +0.24 |
| EDGE | +0.09 | +0.20 | +0.17 | +0.24 |
| LB | -0.39 | +0.89 | -0.12 | +0.41 |
| DB | -0.13 | +1.12 | +0.27 | +0.60 |

- Same season, playing time buys ~0 everywhere, but quality slopes differ by unit (equal slopes rejected,
  p < 0.001): a curve-dollar of DB / RB / LB / TE quality buys ~1 pt, of OL / DL / EDGE ~0.2. The pay curves'
  quality pricing doesn't convert to points at one rate, so the pooled reshuffle (OL / WR up, RB / DB down)
  isn't supported; for DB and RB it's likely the wrong direction.
- With leak-free prior quality, playing time does buy points (offense pooled +0.31, QB +0.48, DB +0.27):
  who plays carries information about current quality. Pricing over a replacement at the same snaps
  overcorrects.
- Points per drafted player (round 1 / 2-3 / 4-7): current 2.95 / 1.22 / 0.47; pooled 2.04 / 0.85 / 0.34;
  unit slopes now 1.06 / 0.27 / 0.14; unit slopes prior 2.60 / 1.11 / 0.48. Position ratios swing wildly
  across versions (LB and TE negative, WR x0.11-x1.13, DB x0.47-x1.99): unit slopes are too noisy to price
  positions, consistent with the bridge (no gain beyond QB vs non-QB out of sample).

Decision: keep the current rule. The leak-free version moves the overall level only ~10% and offers no
stable position reshuffle; no choice test.

## Betting-line absence test (`scripts/absence_test.py [carry]`, `scripts/absence_curve.py`, Nov 2026)

What is a starter worth over the player who replaces him, by position, in points? Regular-season games
2013-25 (2,992 with closing spreads). Regular starter = played >= 75% of the team's previous up-to-4 games
at >= 60% of snaps; absent = no offensive/defensive snaps. "carry" keeps counting a starter absent until
he returns (otherwise the replacement becomes the regular after ~4 games and long absences are lost;
20,781 absent starter-games). Model: spread = home edge + team-season rating difference + sum over
positions of b x (absent starters home - away). Team-season ratings absorb strength, so b is the market's
starter-minus-backup estimate. Note: the backup is the team's actual backup, usually at least as good as a
freely available player.

| Per absent starter, pts/game (carry) | Market (spread) | Actual margin | Current rule, contract prices | Current rule, rookie currency (pay curve) | Quality over replacement (0.48) |
|---|---|---|---|---|---|
| QB | 2.21 (1.95 to 2.48) | 1.58 | 2.27 | 2.53 | 1.89 |
| RB | 0.33 (-0.05 to 0.71) | -0.11 | 0.38 | 0.67 | 0.18 |
| WR | 0.80 (0.64 to 0.95) | 0.58 | 0.58 | 0.84 | 1.02 |
| TE | 0.65 (0.38 to 0.92) | 0.64 | 0.41 | 0.62 | 0.44 |
| OL | 0.47 (0.34 to 0.60) | 0.60 | 0.44 | 0.50 | 0.57 |
| DL | 0.57 (0.34 to 0.79) | 0.29 | 0.64 | 0.89 | 0.76 |
| LB | 0.34 (0.17 to 0.50) | 0.80 | 0.41 | 0.46 | 0.26 |
| CB | 0.36 (0.21 to 0.51) | 0.28 | 0.40 | 0.44 | 0.23 |
| S | 0.32 (0.13 to 0.52) | -0.42 | 0.33 | 0.49 | 0.28 |

(Rule columns: what each rule credits the absent starter per game, rate 0.34 QB / 0.19 others; contract
prices = bridge.preseason_prices; pay curve uses his prior-season PFF quality and recent snap share.)

- The market prices absences close to the actual results (margin minus spread ~0 for every position
  except safeties, +0.75), so it's a usable yardstick. Without carry the numbers are similar
  (QB 2.11, non-QB 0.38-0.66).
- The current rule with a flat 0.5% replacement, in contract prices, matches the market at almost every
  position (QB 2.27 vs 2.21; non-QB within 0.1 except WR / TE, which the market values more).
- In the rookie currency the same rule over-credits RB (x2.0), DL (x1.6), S (x1.5), LB (x1.35), CB (x1.2),
  QB (x1.15); WR, TE, OL about right. Quality over replacement under-credits RB, LB, CB (x0.5-0.75) and
  over-credits WR (x1.3): the market rejects its reshuffle.
- Within positions every rule spreads credit more than the market (calibration slope 0.66-0.73 on the
  spread; pay-scaled: non-QB 0.15 + 0.08 per 1% of price, QB 0.90 + 0.15 per 1%, vs the rule's ~0.17 / 0.31
  per 1%). Expensive starters are worth less over their backup than the rule credits, cheap ones more;
  part of this is noise in price and quality as measures of the player (which flattens slopes).

### Choice test: market-calibrated position factors (`scripts/market_choice.py`, `outcomes_v2.POS_PAY="market"`)

Factors = market starter-minus-backup / the current rule's mean credit in the rookie currency, by pricing
group (`scripts/absence_curve.py factors`): QB 0.89, RB 0.51, WR 0.95, TE 1.06, OL 0.94, IDL 0.65,
EDGE 0.42 (market 0.38, 0.19 to 0.56, vs credit 0.89), LB 1.21, CB 0.82, S 0.65. Applied to realized and
expected seasons. Baseline B = today's model with tables rebuilt on the narrow label
(`units_tables_narrow`; the old `units_tables` predate it). Drafts 2014-22, per team-draft (288):

| M - B | Full factors | Halfway to 1 (`MARKET_SHRINK=0.5`) |
|---|---|---|
| Titles, market-factor scoring | +0.60 (+0.43 to +0.82) | +0.16 (+0.07 to +0.26) |
| Titles, current scoring | +0.08 (-0.10 to +0.31) | -0.07 (-0.19 to +0.04) |
| PFR AV | +2.4 (-0.1 to +4.8) | +1.9 (-0.2 to +3.9) |
| Seasons started | +1.06 (+0.74 to +1.38) | +1.07 (+0.78 to +1.35) |
| 2nd-contract pay, % cap | -0.94 (-1.41 to -0.49) | -1.09 (-1.52 to -0.68) |
| Same choice as B | 46% | 51% |

- Nearly all of the change is rounds 2-4 EDGE (30% of B's picks -> 0-1%) going to OL and TE, plus rounds
  5-7 safeties going to LB / TE. Even the halfway version drops EDGE: in B, EDGE and OL are near-ties in
  rounds 2-4, so small factors flip them.
- M ties B on B's own title yardstick and wins seasons started (and AV, borderline); it loses
  second-contract pay, which is expected since it moves away from the positions the pay market rewards
  (the currency the absence test says over-credits EDGE).
- Round 1 is unchanged (QB / WR only in both, a known quirk of the model).
Not adopted yet: the result rests mostly on one factor (EDGE) and the outside yardsticks split (starts up,
pay down). Default stays POS_PAY="off".

### Edge rushers: why the market prices their absences low (`scripts/edge_check.py`)

Carry-forward absences; market effect split by the absent starter's prior-season quality (tercile within
pricing group), vs what the current rule credits him in the rookie currency:

| Pts/game | Low quality: market / credit | Mid | High |
|---|---|---|---|
| EDGE | -0.07 / 0.67 | 0.48 / 0.85 | 0.83 / 1.18 |
| IDL (DL) | 0.36 / 0.65 | 0.51 / 0.89 | 1.05 / 1.16 |
| DB | 0.05 / 0.36 | 0.25 / 0.46 | 0.71 / 0.60 |
| OL | 0.30 / 0.31 | 0.58 / 0.42 | 0.72 / 0.78 |
| WR | 0.48 / 0.50 | 0.66 / 0.82 | 1.27 / 1.24 |

- The market does value good edge rushers (high tercile 0.83, 0.46 to 1.19; by price, >6% of cap 0.71).
  The gap is at the bottom: a low-quality edge starter is worth ~0 over his backup but credited 0.67.
  Same pattern, smaller, at IDL and DB. OL and WR match the rule at every quality level.
- So it isn't a flat position discount; the pay curve's floor (what full-time snaps earn regardless of
  quality, EDGE 6.3% of cap at q = 0) is too high at EDGE / IDL / DB. Equivalently, those positions need a
  higher replacement level: matching the terciles implies roughly +4% of cap per full season of snaps at
  EDGE, +2% at IDL, +1% at DB, ~0 at OL / WR (on top of the 0.5%).
- Rotation: edge absences whose snaps go to players already in the rotation (prior share >= 0.3) cost
  0.15 (-0.10 to 0.39); those filled from deeper on the roster 0.58 (0.30 to 0.86). IDL similar (0.37 vs
  0.73); OL, WR, DB show no such difference. Edge and interior lines rotate, so a missing starter's snaps
  mostly go to someone already playing.
- Replacements look alike everywhere (gain-weighted prior share 0.29-0.36, quality q ~0.5, 15-25% new),
  so the edge result isn't from unusually good backups.
The x0.42 flat factor over-penalizes good edge rushers; a position-specific replacement level fits better.

### Position replacement levels fit to the market, and choice test (`scripts/replacement_market.py`, `scripts/repl_label.py`, `CHOICE_VARIANT=repl scripts/market_choice.py`)

Fit: each absent starter's worth over replacement = current credit (rookie currency, prior-season quality)
- rate x r_g x snap share; spread + credit difference on team-season ratings and share-weighted absences.
Extra replacement level r_g, % of cap per full season of snaps (spread; 95%):

| QB | RB | WR | TE | OL | IDL | EDGE | LB | DB |
|---|---|---|---|---|---|---|---|---|
| 2.45 (1.7 to 3.2) | 3.19 (0.5 to 5.9) | 2.08 (1.0 to 3.2) | 0.02 (-1.9 to 1.9) | 0.63 (-0.2 to 1.5) | 2.23 (-0.6 to 5.0) | 5.21 (3.8 to 6.7) | 0.48 (-0.9 to 1.9) | 1.10 (0.3 to 1.9) |

(Actual margins give similar or larger levels, noisier. Fit consistently through the regression, the
per-position gaps differ a little from the simple means used for the x factors above: e.g. WR credit
exceeds the market by 0.24, not 0.04. Freeing the slope on the credit gives 0.31 (0.16 to 0.46): within
positions the curve spreads credit far more than the market does, which no replacement level fixes.)

Label `repl` (prospect_model.LABEL; value_t minus r_g x share_t x games fraction, floored at 0; pre-2013
seasons use the share implied by value) cuts 4-season value of drafted players 2014-22: EDGE x0.25, RB x0.30,
IDL x0.55, WR x0.65, DB x0.70, QB x0.79, OL x0.83, LB x0.84, TE x1.00.

| R - B, per team-draft (288) | |
|---|---|
| Titles, repl scoring | +0.62 (+0.41 to +0.82) |
| Titles, current scoring | -0.10 (-0.39 to +0.17) |
| PFR AV | +1.4 (-2.0 to +4.8); rounds 2-4 +3.7, rounds 5-7 -2.5 |
| Seasons started | +1.31 (+0.82 to +1.78) |
| 2nd-contract pay | -1.17 (-1.86 to -0.51) |
| Same choice | 32% |

Same verdict as the flat factors: ties today's model on its titles, more starts, less pay. EDGE vanishes
again, and TEs (the one position with r ~ 0, CI +-1.9) jump to 26% of rounds 2-4 and 36% of rounds 5-7
picks: the choices chase whichever position's noisy estimate is lowest. Not adopted.

### Shrunk and pooled replacement levels (`scripts/repl_label.py eb|pooled`, labels `repl_eb` / `repl_pooled`)

Non-QB levels shrunk toward their precision-weighted mean 1.46% (between-position sd 1.35; Q = 37 on 7 df,
so positions do differ): eb = RB 2.30, WR 1.99, TE 0.50, OL 0.71, IDL 1.82, EDGE 4.35, LB 0.70, DB 1.13;
pooled = 1.46 for every non-QB. QB keeps 2.45 in both.

| Minus today's model, per team-draft (288) | Shrunk (eb) | Pooled |
|---|---|---|
| Titles, own scoring | +0.42 (+0.22 to +0.62) | +0.27 (+0.14 to +0.41) |
| Titles, current scoring | -0.03 (-0.30 to +0.24) | **+0.31 (+0.13 to +0.50)** |
| PFR AV | **+3.4 (+0.2 to +6.6)** | +1.6 (-1.1 to +4.3) |
| Seasons started | **+1.22 (+0.82 to +1.61)** | -0.14 (-0.51 to +0.21) |
| 2nd-contract pay | -0.97 (-1.63 to -0.32) | **+0.76 (+0.29 to +1.27)** |
| Same choice | 36% | 49% |

- Shrunk: EDGE still drops out of rounds 2-4 (OL / TE / WR instead); same trade as before (starts and AV
  up, pay down); ties on current titles.
- Pooled: the first version to beat today's model on today's own title yardstick, and on pay, without
  losing on the others. Rounds 2-4 move toward EDGE (30% -> 50%) and away from OL (32% -> 6%): a uniform
  per-snap charge hits positions with little value per snap (OL, LB) hardest. By year, current-scoring
  titles are positive in 7 of 9 classes but weaker lately (2020 -0.13, 2021 +0.06, 2022 -0.34), and AV is
  negative in 2018 and 2020-22; much of the gain is 2014-2018.
Not adopted yet; the pooled version is a candidate (one parameter, fit from betting lines rather than
the choice outcomes).

### Fresh classes: pooled replacement level on 2023-24 (`CHOICE_YEARS=2023,2024 CHOICE_VARIANT=repl_pooled scripts/market_choice.py choose`)

Same test on the 2023 class (3 seasons played) and 2024 (2 seasons), scored on seasons played so far
(`RealizedV2.titles_many(partial=True)`); no second contracts yet. Pooled minus today's model, per
team-draft (32 each):

| | 2023 | 2024 |
|---|---|---|
| Titles, current scoring | +0.04 (-0.39 to +0.47) | -0.09 (-0.17 to +0.01) |
| Titles, pooled scoring | +0.12 (-0.14 to +0.40) | -0.04 (-0.09 to +0.02) |
| PFR AV | -1.8 (-4.8 to +1.1) | -1.4 (-3.3 to +0.4) |
| Seasons started | -0.84 (-1.47 to -0.28) | -0.50 (-0.69 to -0.31) |

The gain doesn't carry over: titles tie, AV leans negative, and starts are lower (rounds 2-4 again move
from OL to EDGE, who start less). Together with the fade in 2020-22, the 2014-22 advantage looks like it
came from the early classes. Not adopted; today's replacement rule stays.

## Within-position spread (`scripts/within_spread.py`, `resid`, `qb`; Nov 2026)

The fits above found the market responds to our credit with slope ~0.3 inside positions. Retest with
position levels absorbed (count of absent starters by group) and quality from the prior season, the same
season (grades over the games he did play), or their mean:

| Within-position slope on credit | Prior-season q | Same-season q | Mean |
|---|---|---|---|
| Spread | 0.35 (0.24 to 0.45) | 0.40 (0.29 to 0.51) | 0.44 (0.33 to 0.55) |
| Actual margin | 0.61 (0.17 to 1.06) | 1.03 (0.61 to 1.44) | 1.01 (0.55 to 1.46) |

- Quality measures are noisy: year-to-year correlation of q among these starters is 0.1-0.5 (RB 0.11,
  DB 0.19, QB 0.37, others 0.40-0.49), so slopes on any one season's grade are attenuated.
- The market isn't under-reacting to what it could know: margin minus spread on prior-season credit
  +0.26 (-0.19 to +0.72), non-QB -0.01. On same-season credit +0.63 (+0.19 to +1.06), i.e. information the
  market didn't have yet (plus some leakage: grades are higher in games the team played well).
- Against actual results with same-season quality, the curve's spread is about right (slope ~1.0).
  By group (spread, same-season q): WR 0.88, OL 0.72, DL 0.77, DB 0.78, TE 0.76, LB 1.46; EDGE 0.35,
  RB 0.30 (wide); QB 0.31 (0.15 to 0.47).
- QB by pre-season price (market / actual margin / rule's credit in contract prices): <3% of cap 1.25 /
  -0.67 / 0.23; 3-8% 1.74 / 1.87 / 1.00; 8-15% 2.34 / 1.17 / 3.33; >15% 3.96 / 5.07 / 4.85. Elite QBs:
  the rule matches actual results (and is ~20% above the market); the flat QB slope comes from cheap
  starters (rookie deals, where price says little about quality) whom the market rates above backups by
  more than the rule credits.
Conclusion: the low within-position slope is mostly noise in the quality and price measures, not
over-spread pay curves. No change.

## Season-level talent model: "DARKO for PFF" (`scripts/darko_pff.py`, Nov 2026)

Goal: a talent estimate that carries over from season to season better than one season's shrunk PFF grade
(the label's q), with context removed. Per group, a state-space model on the PFF season grade (centered by
group-season): talent follows an AR(1) path with an aging drift; the season grade = talent + beta x teammates'
**pre-season** talent (Kalman prior; QB <- OL, receivers; RB <- OL, QB; WR/TE <- QB, OL, other receivers;
OL <- QB; defense <- pass rushers, coverage) + gamma x schedule (opponents' pre-season talent) + noise
(a + b x 500/snaps). Maximum likelihood on 2006-17 seasons; test on 2018-25, snaps >= 300. Output:
`data/derived/darko_pff.parquet` (all-season fit; pre/post-season talent and SD, context term).

Next-season grade, out-of-sample R^2 (all groups pooled):

| | Label's q (last season, shrunk) | Filter, no context | Context-adjusted talent | + new team's context |
|---|---|---|---|---|
| Movers (1,655) | 0.082 | 0.139 | 0.138 | 0.141 |
| Stayers (5,300) | 0.245 | 0.303 | 0.304 | 0.308 |

- The filter (multi-season history, snap-based reliability, aging, mean reversion rho 0.79 RB to 0.97 QB)
  beats the label's q in every group, movers and stayers; also predicts movers' next-season snaps a
  little better (e.g. DL 0.112 -> 0.137, WR 0.145 -> 0.160).
- Context is real but small when measured from teammates' pre-season talent (grade points per SD):
  WR <- QB +1.2, other receivers -1.1 (target competition); TE <- QB +0.9; RB <- QB +0.8; QB <- OL +0.6,
  receivers +0.5; LB/DB <- coverage +0.8; EDGE <- pass rush +0.5; schedule -0.1 to -0.6. The earlier
  team-changer regression on teammates' same-season grades gave 2-4x larger effects (QB <- receivers +4.5):
  mostly shared team-season shocks / credit sharing, not transferable context.
- Adding context barely moves prediction (movers +0.002); WR movers get worse (0.264 -> 0.221).
Conclusion: the gain is the history filter, not context adjustment. Candidate to replace q in the
label (not adopted yet; the entrant prior must not use draft slot).

### Talent-model quality in the label (`scripts/darko_label.py`, `scripts/darko_label_check.py`)

Narrow label rebuilt with q = within-(group x season) percentile of the smoothed talent (rookie-deal seasons
look ahead only to the end of year 4; entrant prior is one group-wide mean, no draft slot; seasons with snaps
but no PFF row keep the old neutral q). Chains now follow a player across DL/EDGE/LB/DB and WR/TE (438 position
flips had restarted his history); movers-test results unchanged. Pick correction re-estimated: -0.047 per
log(pick+10) (was -0.049). New q correlates 0.735 with the old; the label's 4-year value 0.985 (rank 0.989).

Drafted classes 2011-21, label value vs yardsticks (percentiles within group x class; bootstrap CI):

| Yardstick | Stored | Narrow | Talent-model label | Difference vs narrow |
|---|---|---|---|---|
| Second-contract pay (2,736) | 0.787 | 0.788 | 0.791 | +0.003 (-0.001 to +0.007) |
| Consensus board rank (1,641) | 0.516 | 0.494 | 0.485 | -0.009 (-0.016 to -0.001) |

- The label is mostly playing time; a better quality measure barely moves it. The biggest changes are
  heavily used, poorly graded QBs (Darnold 0.25 -> 0.12, Bortles 0.37 -> 0.25, Brissett, Trubisky).
- Value falls 5-9% outside the top 10 picks, so top-10 picks gain slightly relative to the rest.
Not adopted at first: ties the outside yardsticks. Available as `LABEL=darko`.

### Talent-model label adopted (`prospect_model.LABEL`, Nov 2026)

Later adopted as the default label anyway: it ties on outside yardsticks, and its quality term is the better
measure of talent (it beats one season's grade at predicting next season, movers included). The default is
now "darko" (`data/derived/labels_pickadj_narrow_darko.parquet`). To rebuild it, run `python scripts/darko_pff.py`,
then `python scripts/darko_label.py`. `LABEL=narrow` restores the previous default. Simulator tables carry
`_lbdarko` in their cache key. Results in sections above this one used narrow or stored unless they say otherwise.

Rebuilt the same outputs as for the narrow label. App tests pass. Changes are small:
- **Pick values** in % of a title, for a contender with a veteran QB:
  - non-QB #1: 2.15 -> 2.36
  - non-QB #32: 2.25 -> 2.07
  - non-QB #160: 0.81 -> 0.72
  - QB #1 (middle of the pack): 9.6 -> 9.9

  Top picks gain a little, as the label check implied. Surplus in cap terms is unchanged: it doesn't read the label.
- **Real trades:** correlation 0.997 with before; no firm flips (firm calls 740 -> 735).
- **2027 teams:** the average value of each team's firsts goes 1.06 -> 1.08. The trade-down gain is smaller
  (mean 0.73 -> 0.64) but stays positive for every team.
- **Past-drafts scorecard** (2014-22, model minus team, each recommendation at its first pick):

  | Rounds | Before | After |
  |---|---|---|
  | 1 | +1.02 | +0.77 |
  | 2-4 | +0.42 | +0.50 |
  | 5-7 | -0.16 | +0.02 |

  Recommendations unchanged at 85% / 83% / 74% of picks. Most of the round 1 drop is the yardstick, not the
  choices: on unchanged recommendations, round 1 goes +1.10 -> +0.79, because the talent model rates the
  hits of heavily used QBs lower. Guidance unchanged: there is no tested edge after round 1.
- **Trade decisions:** the model's choice is unchanged in 53% of trade decisions. Most of these are near-ties
  between keeping the pick and the same trade.

## Play-level context from nflverse play-by-play (`scripts/pbp_metrics.py`, `scripts/pbp_value_test.py`, Nov 2026)

Can play-level data separate a player from his context better than PFF season grades? nflverse play-by-play
2006-25 and participation (all 22 players on the field) 2016-25, in `data/raw/nflverse_pbp` (not in git).
Regular-season pass/run plays, win probability 0.05-0.95, outcome EPA; fit season by season with ridge
(penalties by 5-fold CV over plays, 2016-18):
- Involved players (2006-25): dropbacks EPA = passer + receiver + defense team; designed runs = rusher +
  offense team + defense team (passer and receiver adjusted for each other).
- RAPM (2016-25): EPA = sum of the 11 offensive players' effects + the 11 defenders' effects (about 27k plays a season).

Added to the PFF talent model (filtered talent + snaps; 5-fold CV grouped by player), the play metrics change
out-of-sample R^2 for next-season PFF grade and next-season snaps by -0.006 to +0.016 in every group (movers
included); the one +0.016 (QB RAPM, 393 QBs) is within noise.

Why: the play metrics carry over only when the player stays with his team (season-to-season correlation,
players with >= 100 plays both seasons):

| | PFF grade, movers / stayers | Play metric, movers / stayers |
|---|---|---|
| OL (RAPM) | 0.38 / 0.60 | 0.07 / 0.33 |
| DL (RAPM) | 0.42 / 0.58 | 0.03 / 0.19 |
| EDGE (RAPM) | 0.45 / 0.52 | 0.07 / 0.20 |
| LB (RAPM) | 0.30 / 0.39 | -0.01 / 0.22 |
| DB (RAPM) | 0.15 / 0.26 | 0.03 / 0.18 |
| QB (passer EPA) | 0.26 / 0.53 | -0.13 / 0.36 |
| WR (RAPM) | 0.41 / 0.52 | 0.08 / 0.32 |
| RB (RAPM) | 0.19 / 0.39 | 0.23 / 0.27 |

- EPA-based player effects mostly measure the team: teammates play together nearly every snap, so ridge
  regression can't pull them apart, and what persists for stayers is the unit and scheme.
- PFF grades carry over to a new team far better: the context they hold is small next to EPA's. They do
  also drop for movers (QB 0.53 -> 0.26), which is part context, part selection (teams let go the players
  they rate lower).
- Receiver EPA per target sticks for stayers (0.46 at >= 50 targets) but PFF talent doesn't predict it (R^2 0.01):
  role (target depth) more than skill.
Conclusion: no play-level gain over PFF; PFF is the better per-player signal. Untested: RAPM with a PFF
prior (would mostly reproduce PFF), multi-season RAPM.

## Running backs: pass game (`scripts/rb_passgame_test.py`, Nov 2026)

Starting point: an RB's on-field effect on pass plays follows him to a new team (RAPM, movers 0.25 vs stayers
0.27; 89 movers), on run plays it doesn't (-0.00 / 0.07), and the market prices a missing starting RB at only
0.33 pts (actual margin -0.11). Two questions, tests pre-specified in the script.

**Do pass-catching backs help on net, or do their targets displace better WR targets?** (2016-25)
- EPA per target: RB 0.03, TE 0.21, WR 0.23. RB targets are worth far less (partly checkdowns when
  nothing else is open, so the alternative isn't always a WR target).
- RB pass-play RAPM on same-season PFF traits (per SD, EPA x100, 690 RB-seasons): target rate -0.19 (t -6.0),
  receiving grade +0.15 (t +5.4), pass-block grade +0.03, pass-block rate -0.09 (t -3.5). Next season (what
  persists): target rate -0.12 (t -2.8), receiving grade +0.07 (t +1.6). Top-quartile target-rate RBs average
  -0.06 vs bottom quartile +0.16. Volume hurts, receiving skill helps: the displacement claim holds for
  target share at the same skill, but a good receiving back is still a plus. Sizes are tiny (about -0.002
  EPA per pass play per SD, under 0.1 pts/game; ridge shrinkage understates them).

**Do college pass-game traits pick RBs who beat slot / consensus?** (label: narrow)

| Per SD (RBs) | Pricing vs board | Value vs slot 2016-22 | Value vs consensus 2018-22 |
|---|---|---|---|
| College receiving share (receptions / touches) | x1.03 (p 0.09) | x0.97 (p 0.67) | x1.04 (p 0.56) |
| PFF college receiving grade (best) | x1.00 | x1.15 (1.04-1.27, p 0.01) | x1.11 (p 0.14) |
| PFF college pass-block grade (best) | x1.03 | x1.01 (p 0.94) | x1.33 (1.06-1.66, p 0.01) |

- Nothing passes the pre-set bar (p < 0.008 for 3 traits x 2 yardsticks); ~150 RBs per test.
- Transfer (92 drafted RBs, NFL years 1-4): college receiving share predicts NFL target rate (0.44) and
  receiving grade (0.33), not pass-play RAPM (-0.05). College pass-block grade predicts nothing in the NFL
  (NFL pass-block grade -0.09), so its x1.33 vs consensus isn't pass blocking: likely noise.
- The label credits snaps at veteran pay, so receiving-grade RBs' x1.15 may be third-down playing time
  rather than wins.
Conclusion: RB value in the pass game is real but small; volume (targets) is the wrong thing to want.
Lead to recheck when 2023-24 mature: PFF college receiving grade vs slot. No model change.

## Kickers and punters: value in points, drafted vs undrafted (`scripts/specialist_value.py`, Nov 2026)

Specialists aren't in the label. PFF doesn't grade them, they have no pay curve, and only about 20 have ever
been on the consensus board. So they get their own measure, in team points, the engine's unit. Data: nflverse
play-by-play 2006-25, regular season.
- **Kickers:** field-goal points over expected, 3 x (made - P(make)), with P(make) by distance x era, indoor,
  wind and cold. Extra points the same way at 1 point, by era (the kick moved back in 2015).
- **Punters:** punt EPA over expected by yardline x era, indoor and wind. This includes coverage and returns.
- **Left out:** kickoffs. The touchback rules changed in 2011, 2016, 2018 and 2024, and coverage dominates.

| | Year-to-year corr (team changers) | Talent sd | Replacement (mid-season pickups) |
|---|---|---|---|
| K (20k FG, 24k XP) | 0.11 (0.13) | 0.034 pts/kick = 2.2 pts/season | -0.025 pts/kick |
| P (46k punts) | 0.17 (0.31) | 0.065 pts/kick = 4.5 pts/season | -0.078 pts/kick |

Draft test, classes 2006-22, years 1-4. Value = points over replacement, from each season's per-kick value shrunk
by its reliability, times kicks. It's shown in the pick-value unit: team points per game per season, averaged
over 5 seasons. Undrafted entrants = those who kicked or punted in the NFL; veterans who entered before 2006 are
excluded.

| | n | Drafted | Undrafted | Difference (se) |
|---|---|---|---|---|
| K, rounds 2-5 | 14 | 0.048 | 0.034 | +0.015 (0.010) |
| K, rounds 6-7 | 19 | 0.042 | 0.034 | +0.009 (0.011) |
| P, rounds 2-5 | 16 | 0.242 | 0.145 | +0.097 (0.026) |
| P, rounds 6-7 | 17 | 0.159 | 0.145 | +0.014 (0.028) |

For scale, a non-QB pick adds 0.125 at #96, 0.101 at #128, 0.086 at #160 and 0.055 at #256. That value is also
measured against a minimum-salary replacement and assumes the pick fills a need.
- **Kickers:** small, barely persistent edge (0.13 pts/game per season for a 1-sd kicker). Drafted kickers do
  no better than undrafted ones. The best of the period were often undrafted (Tucker, Bailey, Lutz, Dicker) or
  7th-rounders (Butker).
- **Punters:** the talent is larger and carries over to a new team. Punters drafted in rounds 2-5 beat
  undrafted ones by about 0.10 pts/game per season, which is about a #120 non-QB pick's whole value. They are
  better per punt, not just used more: raw value per punt is +0.01 to +0.05 for drafted vs -0.04 for undrafted.
  Rounds 6-7 punters are no better than undrafted. Samples are small (16 and 17 players).
- **Caveats:** the undrafted group is selected (only those who reached the NFL). Drafted players get more chances
  (investment). Punt EPA carries some coverage-team noise.

### In the simulator (`src/specialists.py`)

Kickers and punters are now options in the draft simulator (artifact page and the classic page). Long snappers
still aren't valued: there's no measure of snapping.
- **Signal:** where the specialist was actually drafted. The board almost never ranks them, and this is the
  market's pre-draft read, not his outcome. Undrafted specialists add nothing, since a team can sign one after
  the draft.
- **Value:** the per-season gain over an undrafted specialist (team points per game, years 1-4), minus the rookie
  contract's cost over an undrafted signing. The cost is zero after about pick #110, where the rookie minimum
  applies. This runs through each team's situation into titles, like every pick.
- **Need:** a second kicker or punter in the same draft adds nothing.
- **Gains used** (`data/derived/specialist_engine.json`, from `scripts/specialist_value.py`):
  - punters drafted by pick 176: 0.089 per season (18 punters); after 176: 0.043 (15)
  - kickers: 0.014 (33)

  Splitting punters by pick instead of round moves three punters across the line. That lowers the early gain
  from the round-based 0.097 to 0.071 in the 5-season unit, within its se of 0.026.
- **Delivered:** the player's points over replacement minus the undrafted average, per season, at the pick's
  cost.

The value tables are small per-situation arrays (`sim_tables/spec_<key>.npz`), so the main tables didn't need a
rebuild. The Past drafts review is unchanged: its recommendations still leave specialists out, as do the research
scripts (`simulator.options(specialists=False)` by default).

Example: 2018 Seattle, Michael Dickson at #149 is worth +0.31% of a title on draft night and delivered +0.72%.
He ranks 91st of 188 options at that pick.

Across every real team pick 2014-26 (3,334 picks), a kicker or punter is never the model's top choice. The best
available specialist's median rank among the options is 248 in round 1, 147 in round 4 and 89 in round 7. Teams
used 2-3% of their round 5-7 picks on them. By the model, a specialist is always worth less than the best player
left at another position.

Caveat: after round 1, the model's choices did no better than teams' in past drafts, so the late-round values
it compares against are themselves untested.

## WR stacking: is a second / third good WR worth less? (`scripts/wr_stacking.py`, Nov 2026)

Question: the simulator often takes 2-3 WRs with a team's first few picks. The model values each WR on his
own; its only stacking adjustment is within one draft (2nd rookie at a group: years 1-2 playing time x0.6,
3rd x0.3). Does a WR lose value when the team already has good or young WRs, and does a new WR take value
from the incumbents? Every group run the same way for comparison. Values are the pay-curve currency (% of
cap), player-seasons 2013-25 (cached in data/derived/wr_stacking_ps.parquet).

**A. Incumbents** (played for the team last season, didn't leave before the draft; their summed value this
season vs a round 1-2 pick at their group, controlling for their value / share / count / best player last
season, season FE). Placebo = a round 1-2 pick at the group in NEXT year's draft.

| % of cap | WR | OL | EDGE | DL | TE | DB | RB | LB |
|---|---|---|---|---|---|---|---|---|
| Round 1-2 pick this year | -0.74 (-1.70 to +0.22) | -1.37 | -1.34 | -0.84 | -0.94 | -0.92 | -0.72 | -0.08 |
| Placebo: next year | -1.26 (-2.11 to -0.41) | +0.07 | -0.22 | -0.01 | -0.23 | -0.29 | -0.40 | +0.10 |

WR is the one group where the placebo is as large as the real effect: teams draft WRs after their WRs fade,
and no displacement shows beyond that. At OL / EDGE / DL / TE a high pick does cut incumbents' value
(~1 % of cap): a model issue, but it lowers those positions, not WR.

**B. The rookie's own value** (round 1-3, years 1-2, group x season, log pick): per other round 1-3 WR
taken this or last draft -0.54 (-1.14 to +0.06) on a mean of 2.84 (~-19%); per 1% of cap of incumbent WR
value -0.07 (-0.16 to +0.02), ~-10% per SD. Other groups ~0 except DL "young" -0.39. The model's
same-draft discount (-40% of years 1-2) is already larger than this; it ignores last year's picks and
incumbents.

**C. Team points** (offensive rating on WR1 / WR2 / WR3 sorted by value, plus other units): no diminishing
returns. Prior-season quality: WR1 0.14, WR2 -0.06, WR3 0.17 pts per 1% (all wide); same-season: 0.24 /
0.21 / 0.59. WR1 - WR3 p 0.90 / 0.06 (the wrong direction for stacking).

**D. Market** (absence test, carry; absent starter x his own price and the summed price of the group's
starters still playing, both centred). Points lost per absent starter:

| | Avg | Per 1% own price | Per 1% of teammates still playing |
|---|---|---|---|
| WR, market | 0.73 | +0.084 (0.039 to 0.130) | -0.048 (-0.077 to -0.019) |
| WR, actual margin | 0.52 | +0.009 | -0.047 (-0.185 to +0.091) |
| OL, market | 0.37 | +0.046 | -0.030 (-0.046 to -0.014) |
| CB, market | 0.35 | +0.117 | -0.017 (n.s.) |
| DL, market | 0.61 | +0.100 | +0.010 (n.s.) |

The market does see WR substitution: +1 SD of teammates' price (4.5% of cap) cuts a missing WR's worth by
~0.22 pts/game (~30%); each 1% of teammates' price offsets ~57% of 1% of his own. Results agree on the point
estimate but can't confirm it. OL shows the same in the market (not in results). This is short-run
substitution (the others absorb targets for a game), the closest the data get to stacking.

Conclusion: weak evidence of WR stacking costs: rookie-side ~-19% per other young WR (p 0.08), market
substitution real, team points and incumbents show none. The model's within-draft discount is already at
least this large in years 1-2; what it lacks is any discount in years 3-4 and for WRs already on the roster.
Not enough to explain the simulator's WR-heavy drafts on its own. Next if pursued: a choice-test variant
that discounts a WR by teammates' expected WR value in every year (market ratio ~0.5), scored on the
whole-draft test.

### Choice test: teammate discount for WR (and OL) (`scripts/stack_choice.py`)

Variant of the whole-draft test (2014-22, 288 team-drafts, greedy, single units, QB need probability,
talent-model label; tables in data/derived/stack_tables). A WR's playing time in every season is scaled by
m = 1 - 0.066 x (X - 10.9), clipped 0.4-1.5, replacing the same-draft rule for WRs; X = summed pre-season
price (% of cap) of the team's top 2 WRs on draft day (returning players, veterans added before the draft,
and WRs the model already took, at their slot's expected price); 10.9 = mean X of WRs actually drafted, so
the average drafted WR keeps m = 1. Slope from the market (test D above). WO adds OL (top 4, 8.1% per 1%,
mean 18.4). W- / WO- cap m at 1 (penalty only). "Teammate-adjusted" titles rescale each chosen player's real
seasons by m at this team / m at the team that really drafted him (built on the same assumption, so
partly circular).

| Version minus B, per team-draft | Titles as played | Titles, teammate-adjusted | AV | Seasons started | 2nd-contract pay | Same choice |
|---|---|---|---|---|---|---|
| W | -0.25 (-0.65 to +0.12); R1 -0.32 | -0.02 (-0.41 to +0.35) | +1.0 | +0.07 | +0.09 | 82% |
| WO | -0.47 (-0.88 to -0.08) | +0.02 | -3.6 (-6.9 to -0.3) | -0.34 | -0.81 | 47% |
| W- | -0.12 (-0.40 to +0.10) | +0.10 (-0.14 to +0.32) | +1.4 (-0.4 to +3.2) | +0.06 | +0.25 | 87% |
| WO- | -0.10 | +0.17 (-0.11 to +0.40) | -2.3 | -0.38 (-0.75 to -0.01) | +0.13 | 59% |

WRs among each team's first 3 picks: B 1.75 (2+ in 67% of drafts), W 1.64 (62%), WO 1.52 (56%), W- 1.58
(58%), WO- 1.61 (60%). Round 1 stays QB / WR (B 35% / 62%; W 34% / 57%).

- The discount barely changes the pattern: even with stacking priced at the market's rate, the model takes
  ~1.6 WRs with its first 3 picks. The round-1 WR edge is far larger than any teammate effect.
- No version beats B: titles as played go down (W round 1 -0.32), teammate-adjusted titles tie, outside
  yardsticks split (W- slightly up on AV / pay, n.s.; adding OL loses AV, starts and pay).
Not adopted. WR stacking doesn't explain the gap with teams' drafts.

Found on the way: `roster_need.departures_and_adds` keeps only the last team of OTC's multi-team contract
strings ("BUF/DAL/CLE"), so a player who re-signed and was later traded looks like he left (e.g. Amari Cooper,
DAL 2020). stack_choice.py uses every listed team; roster_need.py, need_choice.py and wr_stacking.py test A
still use the old rule (affects which incumbents count, not the conclusions' direction; unchecked).

## Round-1 WR edge: one class or a few hits? (`scripts/wr_class_check.py`, Nov 2026)

Round-1 picks where the review recommends a WR and the team took someone else; each WR once per draft (mean
over his picks). Compared with the team's pick, the model's best available non-QB non-WR player at that pick
(stack_tables, version B values) and the board's best. Realized titles (% of a title); 2023-25 scored on
seasons played so far.

| 2014-22, 28 WRs (95% CI resampling classes) | WR rec minus |
|---|---|
| Team's pick | +1.48 (+0.70 to +2.41) |
| Best non-WR (16 of 28 EDGE, 5 IDL, 4 TE) | +0.81 (-0.12 to +1.93) |
| Board best | +1.25 (+0.32 to +2.27) |
| AV / starts / pay vs team | +0.6 / -0.34 / +1.16 (all n.s.) |
| AV / starts / pay vs non-WR | -5.7 / +0.08 / -0.50 (all n.s.) |

- **Not a 2020 artifact.** Leaving out any one class: vs team +1.15 to +1.72, vs non-WR +0.45 (without
  2020) to +1.22. 2016 (Michael Thomas) moves it as much as 2020.
- **A few hits carry the edge over other positions.** Thomas 13.0, Lamb 7.0, Pickens 6.9, Chase 6.0,
  Evans 5.2, Jefferson 4.6, Metcalf 4.6. Median WR minus non-WR 0.00 (WR better in 50%); Wilcoxon p 0.51.
  Capping every player's titles at 4 (both sides): WR 1.60 vs non-WR 1.41 (+0.19); at 3: +0.08. Hit rate
  (> 3% of a title) WR 25% vs non-WR 21%; busts (< 0.5) 29% vs 36%.
- **Against teams' picks it holds up:** capped at 3, still +0.54; Wilcoxon p 0.08; none of the teams'
  picks reached 3.
- **Fresh classes 2023-25 so far (8 WRs):** vs team +0.74 (+0.16 to +2.37), vs non-WR +0.60 (-0.13 to
  +2.40); 2023 (Smith-Njigba, Downs) carries it; 2024-25 ~0.

Reading: the model's round-1 WRs beat what teams took, but over the best alternative position (mostly edge
rushers) the gain is a right tail: WRs and edge rushers are about even typically, and a handful of WR stars
make the difference. That is how title value is supposed to work (stars dominate), but 28 WRs with 4-7 deciding
it is thin, and outside yardsticks don't back the WR-over-EDGE preference. The model's lean is "WR ≥ EDGE in
round 1", not "WR >> everything"; teams choosing edge rushers there are not clearly wrong. No change.

## Grade noise and flat quality pricing at DB / LB / RB (`scripts/grade_noise_test.py`, Nov 2026)

Lead: PFF grades carry over poorly at RB / DB / LB, and the label's pay curves price quality almost flat there,
so the model may see elite prospects at those positions as barely better than average ones.

**Part 1. How fast value falls with board rank**, drafted board players 2012-20 (Poisson GLM on log(rank+10);
x5/50 = expected value at rank 5 / rank 50; label/pay = ratio of elasticities, 90% bootstrap):

| | WR | TE | OL | DL | EDGE | RB | LB | DB |
|---|---|---|---|---|---|---|---|---|
| x5/50, label | 3.47 | 2.18 | 2.37 | 2.72 | 2.42 | 2.96 | 1.95 | 2.23 |
| x5/50, 2nd-contract pay | 3.28 | 2.20 | 2.31 | 2.77 | 2.61 | 3.20 | 2.26 | 2.57 |
| label / pay | 1.05 (0.98-1.12) | 0.99 | 1.03 | 0.98 | 0.92 (0.75-1.07) | 0.93 (0.86-1.03) | 0.82 (0.72-1.02) | 0.85 (0.76-0.95) |
| label / starts | 0.98 | 1.10 | 1.05 | 1.21 | 0.90 | 0.79 (0.71-0.88) | 0.77 (0.67-0.90) | 0.93 |
| label / AV | 1.11 | 0.87 | 1.14 | 1.26 | 1.05 | 0.99 | 0.88 | 0.98 |

The label's rank spread is flatter than pay at DB (-15%), LB (-18%), RB and EDGE (-7 to -8%), and about right or
steeper at WR. Against AV only LB is flatter. Within-group rank correlation of label with pay is 0.79-0.89
everywhere (lowest LB 0.79, TE 0.78).

**Part 2. Why: the curve, not single-season noise.** Veteran contracts 2014-25, APY (% of cap) on the player's
quality percentile in the season before signing. Last season's grade vs the talent model's multi-season talent:
attenuation 0.78 (RB) to 1.00, so grade noise explains little. But full-time players (>= 750 snaps), quality
0.1 -> 0.9:

| | DB | LB | RB | TE | OL | DL | EDGE | WR |
|---|---|---|---|---|---|---|---|---|
| Market (at signing) | 3.71 | 2.49 | 4.65 (n 24) | 3.86 | 5.74 | 8.77 | 7.80 | 9.23 |
| Label's curve (90% share) | 0.96 | 0.49 | 0.47 | 1.76 | 3.29 | 2.64 | 3.01 | 5.34 |
| Curve / market | 0.26 | 0.20 | 0.10 | 0.46 | 0.57 | 0.30 | 0.39 | 0.58 |

The curves under-price quality everywhere, and least at WR / OL. cfdarko fits price(share, q) on the contract
in force each season (often signed years earlier) against that season's quality; where quality persists poorly
(DB, LB, RB; also DL / EDGE) this season's play says little about a price set earlier, so the fitted slope
flattens. It's the timing of pay vs performance, which acts like noise and hits low-persistence positions hardest.

Reading: the model's currency tilts toward WR relative to defense. A top-5 vs #50 prospect is worth +6% more
than pay says at WR, -7% at EDGE, -13% at DB, -14% at LB. With WR vs EDGE a near-tie in round 1 (section above),
a ~13% tilt is enough to decide many of those picks. Not changed yet; next: a choice test with the rank spread
tilted to match pay by group (cheap), or curves refit at signing (cfdarko change, full rebuild).

### Choice test: rank spread tilted to match pay (`scripts/tilt_choice.py`)

Each position's playing time (value, not cost) x ((t + 10) / (r0 + 10)) ^ (b_pay - b_label), elasticities from
part 1 above, r0 set so the position's average over drafted board players is unchanged; QB untouched. Tilt at
rank 5 / 50 / 150: LB 1.25 / 1.07 / 0.96, DB 1.21 / 1.05 / 0.95, RB 1.13, EDGE 1.09, DL 1.04, OL 0.96, WR 0.93 /
0.98 / 1.02. Drafts 2014-22, 288 team-drafts, vs B (today's model):

| Minus B, per team-draft | Titles | AV | Seasons started | 2nd-contract pay | Same choice (R1) |
|---|---|---|---|---|---|
| Full tilt | -0.01 (-0.25 to +0.18) | +0.8 (-0.7 to +2.2) | +0.08 (-0.10 to +0.26) | +0.15; R1 +0.24 (+0.00 to +0.50) | 79% (91%) |
| Half tilt | -0.06 | +0.0 | -0.00 | +0.03 | 88% (94%) |

Round 1 barely moves (WR 62% -> 57%, EDGE 2% -> 5%; no CB / S / LB); WRs among each team's first 3 picks 1.75 ->
1.71. Ties everywhere; the pay gain is the tilt's own target. Not adopted.

**Why so little: the gap is level, not spread.** At the 177 round-1 picks where B takes a WR, the best available
player at other positions is worth, as a share of the WR's value (median): EDGE 0.52, DL 0.44, OL 0.45, TE 0.44,
DB 0.37, RB 0.33, LB 0.15, though the best EDGE is usually ranked higher (median board rank 17 vs the WR's 21).
A #5 prospect at pick 10 for a contender: EDGE 0.60 of a #5 WR, IDL 0.51, CB 0.33, S 0.35, T / G 0.30, LB 0.07.
A ~1.1-1.25x tilt can't close a 2x gap.

Is the level gap right in the model's own currency? The round-1 WR recommendations vs the best non-WR at the
same pick (wr_class_check pairs, 2014-22, each WR once per draft): predicted 2.17 vs 1.20 (gap +0.97), delivered
2.29 vs 1.48 (gap +0.81); delivered / predicted WR 1.05, alternative 1.23 (difference +0.18, -0.34 to +0.67).
EDGE pairs: predicted 2.14 vs 1.14, delivered 1.86 vs 1.07. Roughly calibrated: the alternatives do a bit better
than predicted, not nearly enough to close the gap. The remaining question is the currency's position levels
(pay as the measure of wins), which the betting-line absence test already checked: the market values a WR
starter at 0.80 pts/game vs EDGE 0.38, CB 0.36, LB 0.34, OL 0.47.
