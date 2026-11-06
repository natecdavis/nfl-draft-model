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
