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
- [x] Trade calculator (src/trade.py): evaluate(ctx, situation, give=[Pick...], get=[Pick...]) -> change in Super Bowls over 5 seasons. Future picks' slots drawn from the source team's likely finish. Scored 396 real pick-only trades 2013-2025 (`python src/trade.py --real-only`).
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
- [x] Fall shading (scripts/fall_weight.py): players who fell past consensus turned out ~80-87% like the pick where they went (Hasan 2014-21; ~/cfdarko's 01f draft-slide experiment found the same with a rookie-value label). The app shades a falling player's talent toward the pick, weight 0.85 by default (Advanced setting; 0.8 at first, 0.85 since undrafted fallers were included).
- Relation to ~/cfdarko (the user's college prospect pipeline): its board-player values come from consensus rank + position; its own features didn't beat the board either. Reused here: Hasan consensus boards (calibration only), the draft-slide finding. Not used: its PFF / PFR-scraped inputs.
- [x] Past drafts review (src/review.py -> app page "Past drafts"): drafts 2014-2024 pick by pick, owner at start of draft night vs team that picked, each team's state on draft day (prior-season strength; QB on hand incl. March moves, his contract known by draft day, age, and prior-season EPA vs median), the model's recommendation from Hasan's board, draft-night trades scored for both sides. Scorecard on realized value (2014-21): round 1 model's choices +4.49 vs teams +2.40 (mostly QB picks for QB-needy teams); rounds 2-4 tie; rounds 5-7 model worse (candidates limited to Hasan's board; 60% of late recs went undrafted) -> don't use recommendations in rounds 5-7. By star rate (independent of the model's QB valuation) teams' picks did better.
- [x] Model changes adopted (Nov 2026), review and simulator rebuilt: (1) outcome units "single":
  cfdarko's within-position pricing with one overall factor (1.10), replacing per-family factors that
  discounted rotational positions and favored OL (`outcomes_v2.UNITS`, override with OUTCOME_UNITS=family);
  this also changes the draft-day app and trade calculator, which compute outcomes live (reports/
  teams_2027.md and the pick-value chart were rebuilt afterwards, last on Nov 9 2026 with the darko label); (2) fall shading applied
  once (`review.Table` shaded twice before); (3) QB need as a probability (`review.p_need`: no starter
  1.0, stopgap 0.79, later-round rookie 0.62, below-median paid veteran 0.54, age 36+ 0.52; since Nov 15 the two veteran
  rates are split by his contract: 0.39 under contract next year, 0.68 expiring, see 'QB need while the veteran
  is under contract'). Review and simulator only: the draft-day app, trade calculator and 2027 team report
  still treat a flagged QB need as certain.
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
made (the classic page; the main page also offers the team's draft-night trades, below). At each
pick: what the team really did, the model's top 15 for that team at that moment, and a searchable
list of every available player (Hasan board players plus drafted players who weren't on it).
Other teams draft as they really did: at pick k, everyone actually taken before k is gone, including
the team's own earlier picks you passed on; players you took are gone at your later picks. Your
choices set your position needs. Same values as the Past drafts review (choosing the team's own
picks reproduces its numbers exactly); drafts through 2022 also show what your players actually
delivered vs the team's (`review.RealizedV2`). Value tables per team situation (current model: units, shading, QB-need probability) are cached in
`data/derived/sim_tables/` (`python src/simulator.py build` builds all 336: 105 situations x draft years, each past draft with its
leave-one-draft-out engine; run it again with OUTCOME_UNITS=family for the tables behind the 'firm / depends on
position pricing' label;
otherwise ~20s the first time a situation is opened). Test: `scripts/test_simulator_page.py`.

The main "Draft simulator" page is a self-contained HTML page (template `scripts/simulator_artifact.html`, filled
by `build_html()` in `scripts/export_simulator_artifact.py` from the current review data and tables and written to
data/derived/artifact/draft_simulator.html, rebuilt when they change): the model's top 25 beside a searchable list of everyone available, and your draft vs the
team's across the full width, including what each pick actually delivered over the rookie contract
(every available player precomputed at every pick of every team with
`review.RealizedV2.titles_many`, cached in data/derived/sim_realized_v4.npz; drafts since 2023 count the
seasons played so far, value and cost alike). Its lookups match the Python
ones to rounding. "Draft simulator classic" keeps the Streamlit version. The same page, exported
with `python scripts/export_simulator_artifact.py`, is published as a private Claude artifact (it embeds
Hasan's boards: keep it private). The app's look (`.streamlit/config.toml`) matches it. Tests:
`scripts/test_simulator_html.py`, `scripts/test_simulator_page.py` (classic).

**Draft-night trades (main page, Nov 2026).** The main page now starts from the picks the team held when the draft
began and offers each draft-night trade it really made (declined offers aren't public), before the first pick the
trade involves: what each side gives (picks, future picks, players), the trade calculator's verdict for this team
(`review_trades`; pick-only trades; a received pick counts as a QB pick if the team took a QB there) and the market
price (`src/market.py`; pick-only trades). OK it or nix it. Data: `export_simulator_artifact.night_trades` (cached
in data/derived/sim_night_trades_v3.json) and realized values at every pick the team could own
(`realized_extra`, data/derived/sim_realized_extra_v1.npz).
- Starting picks: each pick's owner at the root of its draft-night chain (gave it, never received it that night).
  `review.draft_night` used to take the first giver by date, which mislabeled 66 picks 2014-2026 on same-day chains and
  loops (labels only; fixed Nov 2026, a pick that loops back to its owner keeps the team that made it).
- Order: by date, and within a day by pick flow (a trade that hands on a pick comes after the one that brought it
  in; nflverse trade ids aren't in time order within a day). A trade you can no longer make after an earlier choice
  (you don't hold a pick it gives, or already hold one it gets) is skipped and marked "not possible".
- Board after a nix, two rules (a selector at the top): **slot** (default): everyone really taken at a pick is gone
  at that pick, so the partner takes the player really picked at the slot it keeps and the board at every later pick
  is the real one; **partner**: the partner makes the choices it really made with our picks, in order, at the slots it
  keeps, and players really taken at those slots go back on the board. The two bracket what the partner would have
  done; your result under both is the range. E.g. nixing KC's 2017 trade-up for Mahomes: under slot, BUF takes Mahomes
  at #10 and KC picks at #27 with Tre'Davious White available; under partner, BUF takes White at #10 and Mahomes is
  there at #27.
- Totals: your picks plus the calculator's value of future picks from trades you OK'd, vs the team's real draft and
  all its trades; "delivered" covers this draft's picks only.
- Check: OK every trade and take the team's real player -> the team's real picks in 398 of 416 team-drafts; the other
  18 differ by a late pick where the trade log disagrees with who picked (e.g. 2018: the log sends #154 BUF -> BAL,
  BUF made the pick). Test: `node scripts/test_simulator_trades.mjs` (after building the page).

## Roster need test (`scripts/roster_need.py`)

The model ignores the non-QB roster: every first pick at a position is treated as filling a starting
spot; only picks earlier in the same draft lower need (at the time of this test: 2nd at a group x0.6, 3rd
x0.3 of years 1-2 playing time; replaced by the measured x0.95 / x0.88 in Nov 2026, see 'Within-draft need
step' below). Test: open starting spots on draft day per team and position group (last season's
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

**Within-draft need step adopted (Nov 13 2026; `draftday.NEED_STEP`, `scripts/need_step_check.py`).** The old
x0.6 / x0.3 was a guess several times the measured effect, and the measured rule tied it above, so the model now
uses the measured one: the 2nd / 3rd+ rookie the team takes at a group keeps x0.95 / x0.88 of his playing time in
every season (pooled 2014-22 regression with roster need, value years 1-4: -4.8%, 95% -14% to +5%; -12%, -27% to
+3%). E.g. a consensus #40 WR at #40 as the team's 2nd WR now loses 6% of his title value (was 19%), as the 3rd
15% (was 32%). `NEED_STEP=early` restores the old rule; the user-set need levels in the draft-day app
(starter / rotation / depth) are unchanged. Review, simulator tables (`_nsm` in the cache key) and the simulator
page rebuilt. Model minus team where they disagree, each rec once, 2014-22, old -> new: round 1 +1.33 -> +1.33,
rounds 2-4 +0.66 (+0.10 to +1.39) -> +0.73 (+0.13 to +1.51), rounds 5-7 -0.05 -> -0.09; recommendation changed at
7% of picks (none in rounds 1-2); paired, all picks +0.01 (-0.00 to +0.02); rounds 2-4 position mix unchanged
(WR 37%, QB 20%, G 18%, EDGE 14%). Research scripts that build `review.Table` now get the measured rule unless
NEED_STEP=early is set.

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

### Position replacement levels fit to the market, and choice test (`scripts/replacement_market.py`, `scripts/repl_label.py`, `CHOICE_VARIANT=repl python scripts/market_choice.py`)

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

### Fresh classes: pooled replacement level on 2023-24 (`CHOICE_YEARS=2023,2024 CHOICE_VARIANT=repl_pooled python scripts/market_choice.py choose`)

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

## Old-class replication: coaching contact and EDGE height (`scripts/old_class_replication.py`, Nov 2026)

Classes 2006-2015 never entered the 2016-22 tests and have complete careers. No consensus board before 2014, so
only the "vs draft slot" versions replicate. Slot expectation: the project's slot model fit leave-one-class-out
within 2006-15 (overall actual/expected x0.99). Staffs 2006-11 added from Wikipedia (`scripts/staffs_pre2012.py`,
separate files `nfl_staffs_2006` / `coach_careers_2006`; 167 of 192 team-seasons parsed, picks by the other 25 dropped).
Coaching only (visits start 2015). Hypotheses fixed before running.

| Test (vs slot) | Original 2016-22 | Replication 2006-15 | Pooled 2006-22, class FE |
|---|---|---|---|
| Connected drafter, linked schools | x1.06 (coaching), x1.10 p 0.06 (pooled contact) | x1.21 (0.95-1.48, p 0.10); random-team placebo 6% >= real | x1.12-1.14 (p 0.08) |
| Same, picks 1-100 | x1.14 p 0.03 (contact) | x1.22 (p 0.15); placebo 8% | x1.12-1.15 (p 0.12) |
| Recent link, rounds 1-3 | x1.25 (28) | x1.38 (14, p 0.20); placebo 3% | x1.31-1.34 (42, p 0.02) |
| EDGE height per sd | x1.13-1.18 (x1.26 vs consensus) | x0.98 (0.84-1.15); x1.02 with weight; 2006-10 x1.05, 2011-15 x0.93 | - |

- Coaching contact replicated in direction and somewhat larger, both halves (2006-10 x1.24, 2011-15 x1.18),
  and beats a random-team placebo. Still no single result clears p 0.05 except the pooled recent-link cut (which
  was a secondary lead, 42 players). Reading: a real but modest effect (~x1.1) - teams that coached a player
  know something the slot doesn't. Usable only post-draft / for team-specific adjustments; the board can't see it.
- EDGE height failed: no effect vs slot in 2006-15, no other group shows one either. The original's strongest
  part (vs consensus) can't be checked here, but the vs-slot half doesn't hold. Treat the tall-EDGE lead as dropped.

**Part 2: weight gain and word length** (`scripts/old_class_replication2.py`, same design, hypotheses fixed first).

| Test (vs slot) | Original 2016-22 | Replication | |
|---|---|---|---|
| Weight gained since recruit listing, per sd (same combine weight, years, class FE) | x1.06 (p ~0.08-0.10) | x1.02 (0.93-1.12), 468 players 2009-15 | failed |
| Same, OL / DL / EDGE / TE | - | x0.87 (p 0.04; opposite direction) | |
| Press-conference mean word length, per sd (prediction < 1) | x0.85 (p < 0.001) | x1.20 (0.93-1.54), 105 players 2012-15; one-sided p 0.92 | failed |

- Weight gain: recruit listings only reach classes ~2009+, and only 468 of 2,432 drafted players link. No effect
  overall; the big-position cut runs the other way. Dropped.
- Word length: wrong sign (longer-word half 1.17x slot, shorter 0.72x), classes split 2012-13 positive,
  2014-15 negative. With the failed 2023-25 replication (x0.95), two independent samples don't support the
  discovery result: treat it as a 2016-22 artifact. Dropped.
- Of the four leads replicated on old classes, only coaching contact held.

**Part 3: remaining borderline leads** (`scripts/old_class_replication3.py`). Most can't reach the old classes:

| Lead (vs slot) | Original | 2006-15 | |
|---|---|---|---|
| MLB-drafted (1999-2004 MLB drafts fetched) | 7 drafted 2016-22, too few; guide-text baseball x1.23 (p 0.07) | x1.42 (0.66-2.76), beat slot 50% vs 35% (p 0.13-0.16); 22 players (Wilson, Winston, Kaepernick, Tate, Decker ...) | same direction, underpowered |
| OT arm < 33 in | x1.31 (p 0.13, 10) | 3 of 71 drafted OTs 2011-15 | too few |
| One-year wonder (PFF) | x0.78 (p 0.04) | cfdarko PFF college grades start with the 2014 season (raw 2010-13 has no grade files) | untestable |
| Sons of NFL players | x1.15-1.16, beat 56-61% vs 38% | name rule (Jr./II/III + earlier NFL player) fails validation on 2018-22 (precision 56%, recall 11%), and 2006-15 draft names carry no suffixes; Wikipedia's second-generation list is selected on outcome (lists 83% of hand-coded sons with good careers, 40% of weaker, 29% of undrafted) | untestable without hand coding |
| WR / QB height, weight-gain subgroups | x1.15 / x0.78; other positions x1.08, top quarter x1.14 | WR x1.03, QB x0.90 (part 1); other positions x1.09 (p 0.20), top quarter x0.94 (part 2) | not replicated |

Overall, of the borderline leads replicated on old classes, coaching contact is the only one that held; MLB-drafted
points the same way but 22 players can't settle it. Sons could be extended only by hand-coding older guides
(Brugler's earlier books or NFL.com bios) or a source that lists sons regardless of how their careers went.

## What inside contact is worth, through whom, and why (`scripts/contact_value.py`, Nov 2026)

Drafted players 2006-22, slot model fit leave-one-class-out, fair group = players whose school had a link to some
NFL staff (2,530; 219 connected). Questions fixed before running.

**Value.** Connected picks beat slot 45% vs 36% (odds x1.47, p 0.01); value x1.11 (0.97-1.26). At the 2025 cap
($279.2M): +$1.9M per connected pick over the rookie deal (90% -$0.1M to +$4.0M); rounds 1-2 x1.15, +$5.4M per pick
(68); rounds 3-7 x1.05, +$0.4M (151). Teams make 0.42 connected picks a season (~$0.8M a year, ~1% of a class's
value). 27% of NFL coach-seasons carry a college link from the last 5 years (~4.6 coaches per team); each sees
~3.2 draftees a year from his schools, and his team drafts 0.11 of them: about **$0.2M of draft value per linked
coach-season**.

**Role** (each vs unconnected): role matches the player's position or side x1.27 (p 0.07, 52); coached there in the
last 2 seasons x1.21-1.24 (p 0.04, 98); head coach / coordinator x1.25 (24, n.s.); assistant / QC x1.16 (82); position
coach whose role doesn't match x1.00-1.06. NFL staff role stands in for the college role (careers aren't parsed by
role).

**Mechanism: information, not favoritism.** At the same pre-season talent estimate, connected players get the same
snaps (x1.02, p 0.75); they grade +1.1 PFF points better (p 0.14). The edge doesn't fade when the connected coach
leaves: seasons 2-4 coach still on staff x1.07, gone x1.12 (gone vs present x1.06, p 0.63). It's knowledge of the
player at draft time.

**Senior Bowl staff contact** (NFL staffs coached the teams; 12 of 17 years 2007-23 parsed, 728 drafted players):
coaching teams drafted their players at their normal rate (3.3% vs 3.4% expected) and those 24 picks were x1.04 vs
slot. A week of practice, watched by every team, gives no edge. Visits (2015-22): x1.09 (n.s.).

Reading: the edge comes from multi-season, position-specific, recent knowledge (a coach who coached the player),
most valuable on early picks; worth ~$0.2M a coach-season at today's draft rates, so it's a side benefit of a hire,
not a reason for one. Since the information survives the coach's departure, it could be bought without the hire
(consulting the player's college position coach).

**Follow-ups** (`scripts/contact_followup.py`, fixed before running):

*Jobs.* College-linked coaches skew junior: of linked coach-seasons, 12% quality control / intern (7% of all staff),
33% assistant (27%), 44% position coach (41%), 8% coordinator (17%), 2% head coach (6%). Coaches fresh from college
(coached there the season before): 16% QC, 36% assistant, 40% position coach. Draft value per linked coach-season is
about flat by role (~0.06-0.10 connected picks, $0.11-0.20M; QC $0.13M; fresh from college $0.22M), so the
information comes as cheaply in a junior hire as in a senior one.

*Avoiding players.* (B1) Connected teams' picks of someone else in the 10 picks before a player went: x0.97 per pass
(0.91-1.03, p 0.28; random-team placebo at or below that in 11% of draws); 32-pick window x1.02 (= placebo). Faint, in
the right direction, not significant. (B2) Connected picks bust less (23% vs 28% below a quarter of expectation,
odds x0.82, p 0.25) without more stars (12% vs 15%); the edge is mainly fewer misses. (B3) Among schoolmates from the
same class, the connected team's pick beat slot 47% vs 34% for teammates taken by others (odds x1.75, p < 0.01;
value x1.06): connected staffs pick the right teammate, consistent with better valuation at the margin.

*Does the coach need to be on staff?* Links through a coach who left the team in the 2 seasons before the draft
(still reachable): x0.84, beat slot 33% vs 36% (67 picks). Placebo, coach hired after the draft: x1.05 (63, null).
On staff at the draft: x1.11, beat 44% vs 36% (p 0.03). The edge needs the coach on staff - informal access to a
former colleague doesn't reproduce it. (Correction to the reading above: the information can't simply be bought
without the hire.)

## 2026 example: which college coaches to pair with which teams (`scripts/coach_pairing_2026.py`, Nov 2026)

Ex ante: Hasan 2026 board top 120; each prospect's 2025 position coach (Wikipedia team-season page, or the program
page as of 2025-10-15; 34 schools looked up from school / press sources, `coaches_*.jsonl` passed as argv[1], a few
medium-confidence); each team's picks at the start of draft night; P(team takes him) from the availability model;
value = P x edge (x1.15 rounds 1-2, x1.05 later) x expected value at the pick, $ at the 2025 cap, summed per coach.
Needs and team preferences are ignored, so the chances are market-average.

Top pairs: Ohio State LB coach James Laurinaitis (Reese #2, Styles #5) with NYJ $1.2M, LV $1.1M, NYG $1.1M; Miami
DL / DE coaches Damione Lewis / Jason Taylor (Bain, Mesidor) with NYJ / NYG ~$0.85M; Notre Dame RB Ja'Juan Seider
(Love, Price) NYJ $0.75M; Utah OL Jim Harding (Fano, Lomu) $0.7M; Indiana QB Chandler Whitmer (Mendoza) LV $0.65M;
Tennessee DBs Willie Martinez (McCoy, Hood), Texas Tech OLB C.J. Ah You (Bailey, Height), LSU secondary Corey Raymond
(Delane) ~$0.6M. Value concentrates in position rooms with two top prospects, paired with teams holding top-10 picks.
Even the best pair is ~$1M: any one team's chance of taking a given prospect is ~5-14%.

Ex post: several pairings would have been live (LV took Mendoza #1, NYJ Bailey #2, ARI Love #3, NYG Reese #5 and
Hood #37, CLE Fano #9). Actual 2026 hires from these staffs: Whitmer -> TB QBs, Etheridge (Miami CB) and Matt Merritt
(Miami RB) -> ARI, Lou Esposito (Michigan DL) -> BAL DL, Doug Belk (USC DBs) -> DEN, Jonathan Decoster (Florida asst
OL) -> NE. Connected 2026 picks: BAL Rayshaun Benny (Michigan DT, #250, role-matched), ARI Carson Beck (Miami QB, #65).

## Availability: needs, runs and fallers (`scripts/availability_test.py`, Nov 2026)

Question: would modeling other teams' needs, position runs and fallers predict "will he still be there at
our next pick?" better than the app's scatter around consensus rank? Pick-choice model (conditional logit
over the available Hasan-board players plus an off-board option, every pick 2014-2026): consensus rank,
how far he has fallen, position x round, the picking team's open starting spots at his position, QB need
(now / later), what that team already took this draft, board players at his position taken in the last 8
picks, how many better players at his position remain; variant with top-30 visits.

Fit on all drafts (z-scores): rank dominant; fallers keep falling (-6.3); open spots +5.8; QB need now
+6.7, later +3.5; team already took his position -12 (QB -4.2); runs +2.2 (weak); depth at his position ~0;
visit +24.

Rolling test (fit on earlier drafts, test 2017-2026; every pick <= 160 by a team that picks again; board
players still available with rank <= next pick + 32; 108k player-decisions), log loss:

| | scatter (app) | scatter, conditioned on still being there | rank+position | + needs | + runs | + visits |
|---|---|---|---|---|---|---|
| all | 0.552 | **0.536** | 0.538 | 0.538 | 0.538 | 0.533 |
| round-1 decisions | 0.487 | 0.487 | 0.488 | 0.488 | 0.489 | 0.492 |
| QBs | 0.644 | 0.560 | 0.556 | **0.548** | 0.548 | 0.552 |
| fallers (rank < pick) | 0.618 | 0.552 | 0.551 | 0.551 | 0.551 | 0.547 |
| 2025-26 | 0.562 | 0.542 | 0.538 | 0.538 | 0.539 | 0.529 |

- Nearly all of the gain is one fix: the app drew every remaining player's draft spot from scratch, so a
  faller (consensus #5 still there at #20) was mostly drawn before #20 and treated as gone. Drawing the
  spot given he's still on the board: +0.015 log loss (95% +0.014 to +0.017), better in 9/10 drafts,
  calibration fixed (predicted <10% available: actual 12.5% -> 6.4%). **Adopted in `draftday.simulate`.**
- Team needs don't help overall out of sample (-0.001 vs the conditioned scatter, non-QB -0.002). For QBs
  they help (+0.012, 95% +0.005 to +0.020), but in only 4/10 drafts (one case: Shedeur Sanders 2025).
  Not adopted.
- Runs and position depth: nothing.
- Visits: +0.005 over needs in picks 33-160 (7-8/10 drafts), worse in round 1. Where the visit list
  came from and when it was published isn't known (some visits are reported only after the draft), so
  treat this as an upper bound. A team's own visit tracker could be an optional input later; not built.
- Round 1: nothing beats the plain scatter. Top-of-draft surprises (trades, medicals) aren't in these inputs.

## Looking ahead on draft day, rechecked (`scripts/lookahead_recheck.py`, `DraftEngine.lookahead`, `position_dropoff`, Nov 2026)

Question: instead of one player's chance of lasting, compare what waiting costs: (a) rollout = value now +
expected value of our later picks if we take him; (b) position drop-off = best at a position now minus the
expected best there at our next pick. Both were tested in `scripts/plan_choice.py` (versions C, D) with the
old availability draws; rerun here with the conditioned draws (versions Ct, Dt), same tables, boards and seeds,
drafts 2014-2022, 288 team-drafts. Per team-draft vs greedy (B), raw scoring (the right counterfactual: one
team uses the model):

| | Titles (single) | Titles (family) | AV | Seasons started | 2nd-contract pay |
|---|---|---|---|---|---|
| Rollout, old draws (D) | -0.28 (-0.67 to +0.07) | -0.34 | -1.2 | -0.32 | -0.52 |
| Rollout, conditioned (Dt) | -0.36 (-0.75 to -0.03) | -0.44 (-0.92 to -0.01) | -0.3 | -0.36 | -0.50 |
| Drop-off, conditioned (Ct) | -0.96 (-1.50 to -0.49) | -1.16 | -2.0 | -0.98 | -0.94 |

- The availability fix doesn't rescue looking ahead: the rollout ties or slightly trails greedy on titles;
  Dt vs D -0.08 (-0.20 to +0.02). It changes 24% of round-1 choices and ~50% in rounds 4-7.
- Drop-off still clearly worse (chases positions whose talent falls fastest).
- With shared scoring (a player on n teams' model drafts counts 1/n), rollout ties on titles and gains
  AV +3.2 (+1.3 to +5.3) and starts +0.35: it avoids the faller that greedy hands to many teams. Not the
  draft-day case.
- App: "On the clock" now also shows both tables (rollout per candidate with the same simulations for
  every row; drop-off by position, with the chance of a big drop). The ranking stays value-now; the tables
  show what a choice costs at later picks.

## Injury and medical history: public proxies (`scripts/injury_test.py`, Nov 2026)

No medical grades are public. Four proxies, pre-specified with one primary flag each (Bonferroni p < 0.006 over 4
flags x 2 yardsticks), tested like the literature biases (value vs slot 2016-22, vs consensus 2018-22, out of
sample; pricing = drafted vs board rank, position FE):

| Proxy (flag) | Flagged share of drafted | Taken vs board | Value vs slot | Value vs consensus |
|---|---|---|---|---|
| A final-season PFF snaps < 60% of best earlier season (2015+; 2020 opt-outs excluded) | ~10% | x1.01 (p 0.64) | x1.11 (0.91-1.35) | x1.08 (0.86-1.31) |
| B missed >= 25% of team games, last two seasons (game logs; not OL) | ~27% | x1.03 (p 0.13) | x1.04 (0.89-1.19) | x1.00 (0.80-1.22) |
| C at combine, no drills at all (2017+; earlier files barely record non-testers) | ~12% (R1 19%) | x0.94 (p 0.04) | x0.98 (0.76-1.21) | x0.93 (0.69-1.17) |
| D injury / medical concern in Brugler's Beast weaknesses (2019-22) | ~53% | x0.95 (p 0.03) | x1.05 (0.91-1.23) | x1.02 (0.87-1.19) |
| D2 serious (surgery, tear, fracture, season-ending; secondary) | ~27% | x0.97 (p 0.29) | x1.05 (0.89-1.24) | x1.03 (0.87-1.19) |

- No proxy predicts value beyond slot or the board; nothing passes even p < 0.05 on value. Point estimates
  sit slightly above 1 (injured players, if anything, deliver a bit more than their slot), not below.
- Pricing: teams take Beast-flagged and non-testing players ~5% later than the board (p 0.03-0.04, not
  Bonferroni); snap drops and missed games aren't priced beyond the board. Too small to matter for availability.
- Medical fallers vs the app's fall shading: among extreme fallers (pick > ~2x board rank) Beast-flagged
  players beat slot x2.8, but on 14 vs 11 players. Over all fall sizes the effect largely disappears
  (fallers past 1.3x rank: flagged x1.29 vs unflagged x1.14 of slot; fall x flag interactions n.s., signs
  disagree across proxies: snap drop and Beast +, no combine testing -). Fall shading unchanged.
- Reading: public injury history is priced by the board and slot. What teams know beyond it (medical
  grades, re-checks) isn't in any public proxy here. Nothing adopted.

## Team drafting skill: franchises, GMs, head coaches (`scripts/draft_skill_test.py`, Nov 2026)

Question: do some franchises or decision-makers consistently beat draft slot? If so, a skilled team's reach
would be information (fall shading could depend on who passed) and trade partners could be judged. Drafted
players 2006-22 (4,188; 544 team-drafts), value seasons 1-4 vs the slot model fit leave-one-class-out, class-demeaned;
team-draft score = sum of its picks' residuals (% of cap; SD 17.1). GM / head coach from the cached Wikipedia
team-season infoboxes (GM 98% of team-seasons; gaps filled only when both sides agree or the head coach is unchanged,
e.g. Belichick; Bengals = Mike Brown). Placebo: team labels shuffled among team-drafts within each year.

| Test | Result | Placebo / null |
|---|---|---|
| Franchise spread, value vs slot | SD of franchise means 4.82 | 4.17 (p 0.10); est. true SD 2.4 per draft vs 17.1 noise |
| Persistence (vs previous 3 drafts) | r -0.01 | -0.00 (p 0.57) |
| Rounds 1-2 / 3-7 | persistence -0.03 / -0.01 | n.s. |
| 2006-14 vs 2015-22 franchise means | r +0.11 | 32 franchises |
| GM tenures (4+ drafts), odd vs even drafts | r +0.12 (-0.15 to +0.38) | 56 tenures |
| Head coach tenures | r -0.36 (-0.58 to -0.08) | 53 tenures; wrong sign |
| GMs at 2+ franchises (first vs later) | r -0.27 | 5 GMs |
| Predict next draft from shrunk past record | R^2 0.003 (team), 0.002 (GM) | |
| Seasons started vs pick | true SD 0 | |
| AV vs pick | spread p 0.02, persistence +0.11 (p 0.01) | gone net of team margin in seasons 1-4 (spread p 0.96) |

- No usable drafting skill: franchise differences are about what chance produces, nothing persists from one
  draft to the next, and it doesn't follow GMs. Best/worst franchise means (DAL +10, CLE -10 per draft) are
  mostly luck. Matches Massey-Thaler and later work.
- The AV signal is team quality (AV credits players on good teams), not drafting.
- So: the app's uniform treatment stands. No reason to shade a faller by who passed on him, or to judge a
  trade partner's reach by its history.

## Rookie QBs: sitting behind an incumbent (`scripts/qb_sit_test.py`, Nov 2026)

The model's rule (`pick_value.pick_effect`): a drafted QB's seasons before the team's QB need begins (`qb_from`)
are set to zero; later seasons keep comparable QBs' shares (most of whom started early). Two claims tested on
QBs drafted 2006-22 (193; rounds 1-3: 93), playing time = share of his team's dropbacks, value vs the slot
model per season (leave-one-class-out):

**1. "He sits": mostly wrong.** Using the model's own draft-day status (2014-22), rounds 1-3 rookies the model
has sitting in season 1 averaged 36% of team dropbacks, 47% played 25%+, and delivered 0.92 of slot expectation
that season (model: 0). Misses come from the weak-incumbent flags ("paid veteran, below-average play",
"rookie deal (later round)": Wentz, Lawrence, Mayfield, Allen, Rosen started early); rookies behind a 37-year-old
veteran really sat (share 0). In title terms the zero costs a QB 9-17% of his value at an average team (qb_from 2
vs 1, picks 3-64) and 14-26% at a +3 team. Teams with no QB need flag (model: a drafted QB never plays) also got
real value from round 1-3 QBs (0.6-1.1 of slot, 9 players; partly with other teams).

**2. Sitting and development: can't tell.** Rookies who sat year 1 did worse later (seasons 2-4 x0.64 vs x1.14
of slot; starter in season 3-4 21% vs 64%), but teams sit the raw ones. Using a solid incumbent (previous-season
main QB with 300+ dropbacks, above-median EPA, still on the team) as the instrument: first stage +0.29 (F 7, weak);
reduced form on seasons 2-4 value -1.2 (se 4.3, mean 16.7); IV -4.0 (se 14.4); EPA per dropback +0.11 (se 0.12).
Rookies behind a solid incumbent: value x0.34 / 0.78 / 0.90 / 1.19 of slot in seasons 1-4 (rounds 1-3, 22). No sign
sitting helps or hurts; the "later seasons like comparables" half of the rule stands.

### Choice test: weak incumbents' need starts now (`scripts/qb_sit_choice.py`): not adopted

Fix: statuses "rookie deal (later round)" and "..., below-average play" get qb_from = 1 instead of 2 (p_need
unchanged). Greedy whole-draft choice test 2014-22, both versions on value tables rebuilt on the current engine
(data/derived/qbsit_tables). Scored with today's rule and "as played" (every flagged need starts now, so a
rookie's real seasons count). Per team-draft, fix - base:

| | All 288 team-drafts | 62 weak-incumbent team-drafts |
|---|---|---|
| Titles, model rule | -0.23 (-0.52 to -0.02) | -1.09 (-2.30 to -0.08) |
| Titles, as played | -0.24 (-0.51 to -0.02) | -1.11 (-2.38 to -0.11) |
| AV | -1.2 (-2.7 to +0.1) | -5.5 (-12.2 to +0.0) |
| Seasons started | -0.11 (-0.23 to -0.01) | -0.50 (-1.03 to -0.06) |
| 2nd-contract pay | -0.22 (-0.52 to +0.05) | -1.01 (-2.50 to +0.18) |

- The fix changes 17 picks, all at weak-incumbent teams, and the same number of QBs (0.97 per draft vs teams'
  0.39). It moves the QB pick earlier: Manziel over Carr (NYG #12), Rosen over Lamar Jackson (BUF #7, ARI #10),
  Hundley at #17 / #21, Paxton Lynch at #6, Jordan Love over CeeDee Lamb (ATL #16), Malik Willis at #2 (DET),
  each time giving up a WR the base model took first (Michael Thomas, Allen Robinson, Lamb, Garrett Wilson).
- Worse on every yardstick, even the one that credits rookie seasons. The year-1 zero is wrong as a forecast of
  playing time, but it was acting as a brake: the model already values early QBs too highly (README 'QB reach
  penalty', 'Whole-draft choice test'), and making them worth more makes it reach sooner. Kept today's rule.

## Future picks: market discount vs the model (`scripts/future_pick_test.py`, Nov 2026)

Market: 591 pick-only draft-window trades (March-May, two teams, no conditional picks, 2002-25; 135 with next
year's picks). Fit jointly: chart for this year's picks v(k) = exp(-0.089 (k-1)^0.729) (v(10) 0.64, v(32) 0.34,
v(64) 0.16, v(100) 0.08), up-team premium x1.11, and next year's picks worth delta x the expected value of their
slot now (slot drawn from the original owner's prior-season strength):
- next year's 1st: delta 0.48 (95% 0.39-0.54); rounds 2-7: 0.57 (0.51-0.65). Two years out: 5 trades, not usable.
- A middle team's next 1st trades like **#42 this year** (95% #37-#53); next 2nd like #74 (#69-#78); next 3rd
  like #109. The market discounts about one round, as in Massey-Thaler.
- Refit after numbering future late-round picks with comp picks (below): chart exp(-0.081 (k-1)^0.747), premium
  x1.11, next 1st 0.48 (#43 now), rounds 2-7 0.64 (0.56-0.74; was 0.57 because rounds 4-7 were numbered too early):
  next 2nd ~#68, 3rd ~#103. The app uses the refit.
- Refit again on the corrected trade log (Nov 18, 593 trades; "Trade log corrections" below): chart
  exp(-0.078 (k-1)^0.756), premium x1.10, next 1st 0.48, rounds 2-7 0.63; equivalents unchanged (#43, #68, #103).

Model (`src/trade.py`, titles over 5 seasons): equivalent pick this year for next year's 1st (middle team's / own):

| Situation | Next year's 1st | Next year's 2nd | Next year's 3rd |
|---|---|---|---|
| Contender, veteran QB | #53 / #69 | #73 / #83 | #98 / #138 |
| Contender, rookie-deal QB | #71 / #81 | #87 / #101 | #137 / #177 |
| Middle of the pack | #19 / #20 | #50 / #53 | #75 / #78 |
| Rebuilding | #13 / #11 | #22 / #21 | #54 / #51 |

Trades at the market rate, change in titles (% of one; se ~0.02):

| Situation | Sell own next 1st for #42 | Buy a middle team's next 1st with #42 | Sell own next 2nd for #74 | Buy next 2nd with #74 |
|---|---|---|---|---|
| Contender, veteran QB | +0.60 | -0.25 | +0.25 | -0.10 |
| Contender, rookie-deal QB | +0.73 | -0.48 | +0.40 | -0.30 |
| Middle of the pack | -0.22 | +0.24 | -0.22 | +0.22 |
| Rebuilding | -0.33 | +0.31 | -0.31 | +0.25 |

- The market applies one discount to everyone; the model's depends on the title window. Contenders should sell
  future picks at the market rate, middle and rebuilding teams should buy them (a rebuilding team's next 1st is
  worth about #11-13 now to it, but the market prices it at #42). Rebuilding teams value next year's picks above
  the same slot now (ratio 1.0-1.6): the player arrives as the team improves.
- These are the model's implications, not checked against outcomes: the market's flat discount may also price
  things the model leaves out (GM job security, uncertainty about the slot). Nothing changed; the trade
  calculator already values future picks this way. Possible addition: show the market price of each side in the
  app's Trade offer tab, so the gap is visible.

### How real future-pick trades turned out (`scripts/future_pick_realized.py`)

Draft-window pick-only trades 2006-21 with players' seasons 1-4 known (404 trades; 81 with next year's picks).
Each pick -> the player drafted with it. Realized titles: his real seasons through the model's title machinery
(next year's players arrive a season later), for the team's situation at trade time; predicted: trade.evaluate
before the trade (a received pick valued as a QB when a QB was taken there, as in trade.real_trades).

- Player value, no title model: the 15 traded next-year 1sts delivered x1.25 of the average player at the
  market's equivalent #42 (95% 0.75-1.81), x0.75 of the average at their actual slot; slots came out as predicted
  (17 vs 18). Next-year 2nds x1.33 of #74 (13 picks). Leans toward the market discounting too steeply; too few to say.
- The trade calculator tracks outcomes: corr(predicted, realized) +0.36 per trade (the most it could be with this
  much luck, if exactly right: ~+0.51); trades with next year's picks +0.30 (max ~+0.38).
- Non-QB trades with next year's picks (71): teams that bought future picks were predicted +0.50 and realized
  +0.38 (95% -0.06 to +0.88; raw player value +2.2% of cap, -0.2 to +4.9). The model's "buy future picks at the
  market price" held up roughly as predicted, at the edge of significance.
- QB trade-ups paid for with next year's picks were the big wins for the seller of the future pick (HOU for
  Watson +28, KC for Mahomes +22, BAL for Lamar Jackson +18); with them, future-pick buyers realized -0.53
  (-1.55 to +0.31).
- The situation dependence (contenders sell, rebuilders buy) can't be checked: 18-34 trades per cell, ranges +-2.
- Caveat and recheck (Nov 2026, `... states`): the run above sets every team's situation from its prior-season
  rating with a QB need starting now (qb_from 1), so a QB pick looks valuable to every team. Rerun for 2014-21 with
  each team's draft-day state from the past-drafts review (strength, scheduled shifts, QB need; outcomes still
  scored as played): corr(predicted, realized) one side per trade +0.27 -> +0.28 (95% +0.07 to +0.51; n 224; the
  2006-13 trades that lifted the earlier +0.36 have no review state), QB trades +0.32 -> +0.31, others +0.15 -> +0.16.
  The one large change is KC 2017 (Mahomes): predicted +9.95 -> -4.18, because KC had Alex Smith (no QB need in the
  window); it realized +22. Teams with no flagged QB need that traded up for a QB (11): predicted -0.71, realized
  +2.05. A trade-up for a QB may itself signal a need the draft-day flag misses; 11 trades, mostly KC, too few to
  act on. Conclusions above unchanged.

## QB valuation: where the model is off (Nov 2026)

**Diagnosis** (leave-one-draft-out engine, `holdout/defer_candidates.parquet`; titles delivered / predicted, each
player-year once, 95% over draft years). No broad QB over-valuation:

| | Players | Delivered / predicted |
|---|---|---|
| Every QB available to a QB-needy team, rounds 1-2 | 56 | 1.39 (0.81-2.10) |
| Non-QBs at the same picks | 1,082 | 0.96 (0.90-1.03) |
| Model's QB choices, round 1 / 2 / 3 | 21 / 11 / 14 | 1.20 (0.77-1.81) / 0.56 (0.05-1.01) / 0.28 (0.08-0.54) |
| Teams that traded up for a QB, 2006-21 (`future_pick_realized`) | 59 | predicted +3.3, realized +2.5, corr 0.51 |

- Day-2 QBs are over-valued: comparison careers for QBs ranked 33-64 ~1.5x what they did, and the engine's chance
  of a starter-level hit (4-season value > 0.30 of cap) at ranks 40-65 is 0.17-0.24 vs 0.07-0.17 for QBs drafted
  around there 1999-2022. Title odds rise faster than linearly with strength, so the fat tail costs more than the
  mean error (rank 80, middle team: 4.1 titles with the engine's shape vs 2.3 with history's, same mean).
- Round-1 QB busts (Manziel, Lynch, Willis, Haskins, Rosen) are teams knowing more: the same 2014-22 QBs by draft
  slot fall 0.35 / 0.23 / 0.16 (picks 1-10 / 11-20 / 21-32, 4-season value), by consensus rank 0.29 / 0.27 / 0.31.
  The engine's flat round-1 curve is right for what it sees. (So 1999-2022 history by slot can't be used as a
  consensus-rank yardstick: it made ranks 20-30 look 1.5x over-valued.)

**Tested fix: QB comparison careers by consensus rank** (`outcomes_v2.QB_COMPS=rank`, `scripts/qb_comps.py`; default
`ev`, nearest by expected value). Leave-one-draft-out board QBs 2014-22: ranks 33-100 mean error +0.035 -> +0.025
of cap, rmse 0.146 -> 0.144, hit Brier 0.106 -> 0.105; ranks 33-50 still 1.5x (n 6). Pick values barely move at
48-64, rise at 24-32 (Lamar Jackson, Watson, Carr become the direct comparisons; a rank-24 QB worth more than a
rank-5 one) and get bumpier at 80-128. Whole-draft replay (288 team-drafts, rank minus ev): titles +0.47 per
draft by team-draft bootstrap (+0.07 to +0.86), but it is one player: 2014 (+143 of the +135 total) is Derek Carr
taken by 10 teams instead of Manziel / Bortles / Bridgewater. By draft year: -0.28 to +1.56; without 2014 -0.03
(-0.39 to +0.26). Round 2 worse (-0.11 per pick; QB choices 0.67x -> 0.17x, loses Hurts at NE #37). Not adopted:
the 2014-22 pool has too few QBs ranked 33-100 (~25) to set their tail; rank matching reshuffles noise.
(This repeats the earlier rank-matching check under 'QB reach penalty' (1.74 -> 1.63x, left as is) with a
whole-draft replay added; same verdict. That section also tested a penalty on QBs ranked below the pick: not
adopted.)

**App: QB reach flag** (`draftday.qb_reach_note`, lookup from `scripts/qb_reach_flag.py` -> `data/derived/qb_reach.json`).
When the best-value player on the clock is a QB, the "On the clock" tab adds a note. It changes no values.
The model's QB choices 2014-22 (leave-one-draft-out, each player once per bucket; titles delivered / predicted,
95% over draft years; starters = 3+ seasons):

| QB's consensus rank | n | Delivered / predicted | Starters |
|---|---|---|---|
| 1-32 | 17 | 1.37 (0.99-2.01) | 71% |
| 33-64 | 8 | 0.80 (0.16-1.75) | 25% |
| 65+ | 14 | 0.21 (0.01-0.43) | 0% |

By distance below the pick the signal is weak (at or above 1.02, 1-15 below 1.08, 16+ below 0.72, 0.11-1.36).
So: ranked 33+ -> warning with that band's record, the chance he's still there at our next pick and the best
non-QB; ranked 1-32 but 16+ spots below the pick -> note that teams know more about QBs than boards.

## App: market price in the Trade offer tab (`src/market.py`, Nov 2026)

Under the model's verdict, the Trade offer tab now shows what the trade market would charge. Each side is priced
on the chart fitted to 591 pick-only draft-window trades, 2002-2025 (`scripts/future_pick_test.py chart` ->
`data/derived/market_chart.json`). Pick k this year is worth exp(-0.081 (k-1)^0.747). Next year's 1st is worth
0.48x the expected value of its slot, and rounds 2-7 0.64x (refit with comp-pick numbering, below); the slot comes from the original owner's likely finish.
Picks two or more years out are extrapolated as delta^years, since only 5 such trades exist.

The tab shows:
- get / give, with a 95% range from refitting the chart on 300 resampled trade sets;
- each side as an equivalent single pick this year;
- who is moving up, and the usual ~11% premium paid by the team moving up.

Examples (middle team):

| Trade | Get / give | Range |
|---|---|---|
| Mahomes-style: #27, #91 and next year's 1st for #10 | 0.87 (moving up, so about typical) | 0.81-0.94 |
| #10 for #20 + #52 | 1.05 | 1.00-1.11 |
| Next year's 1st (middle team) for #42 | 1.00 | 0.88-1.28 |

It is information, not a second verdict. The model's title change is what to act on; the market price says whether a
partner is likely to accept, and where the model and the market disagree (contenders' future 1sts) is the arbitrage.

## Comp picks in future pick numbers (`trade.COMP_OFFSET`, Nov 2026)

Compensatory picks (end of rounds 3-7, ~32 a year) push every later round back, but the calculator numbered a
future round-r pick as slot + 32 x (r - 1). Net extra picks per round 2017-2026 (nflverse draft_picks, after
forfeits): round 3 +7.0, 4 +5.1, 5 +6.3, 6 +7.9, 7 +6.7, so future rounds 4 / 5 / 6 / 7 now start 7 / 12 / 18 / 26
spots later (checked: rounds actually start at #104 / 141 / 179 / 219 on average). This-year picks are entered by
number, so comp picks the team holds already count correctly; future comp picks can't be traded (tradable only once
awarded, in March).

Effect: the model's value of future late picks barely moves (middle team, titles: #112 -> #119 0.415 -> 0.419, #144 ->
#156 0.367 -> 0.331, #208 -> #234 0.249 -> 0.234; within simulation noise). Market prices of late future picks fall
12-32% on the old chart, and the refit chart raises the next-year discount factor for rounds 2-7 from 0.57 to 0.64
(next 1st unchanged at ~#43). App examples: Mahomes-style trade 0.87 -> 0.86, #10 for #20 + #52 1.05 -> 1.06,
middle team's next 1st for #42 1.00 -> 1.02.

## Ledger of every lead tested (`reports/ledger.csv`, Nov 2026)

One row per hypothesis or model change tested in this README (242 rows as of Nov 17, split where a section tests several things).
Each row has the headline effect and p-value or 95% range copied from its section, n, significant at 5%, whether a
fresh sample replicated it, the verdict, the section and the script. "Replicated" counts only a fresh or independent
sample, or a held-out test; a year split inside the same sample is "not tried". Rows were extracted from this README and
the project notes and spot-checked against the sections. Rerun or edit by hand when a section changes.

| Verdict | Rows |
|---|---|
| Not adopted | 185 |
| Adopted | 23 |
| Adopted in the simulator room check (information only) | 4 |
| Fix applied | 15 |
| Information only (shown in the app) | 7 |
| Parked | 6 |
| No change | 2 |

**Leads for information the board misses** (90 rows: scouting text, multi-sport, coaching contact, visits, growth,
size, transcripts, literature biases, injuries, team skill, ...):
- 58 have a stated test. 22 are significant at 5%; about 3 would be expected by chance if all were null. Most of the
  rest are null without a p-value. These tests share players and years and include subgroup cuts of the same idea, so
  they are not independent and the excess is smaller than it looks.
- Of the 22, 8 were put to a fresh sample: 6 failed (Hasan reach and stars, sons of NFL players, EDGE height, transcript
  word length, recruiting rank, one-year wonders). Two held: coaching or visit contact in picks 1-100 (x1.14 vs slot)
  and the round-1 WR edge over teams' picks. Neither changes the board: contact is only known after the draft, and the
  WR edge sits inside the model's existing recommendations.
- 12 were never retested (track athletes x0.84, "raw" language x0.75, RB receiving grade x1.15, RB pass blocking
  x1.33, recent coaching links x1.21, and others). Treat each as one draw from many tests until a fresh sample says
  otherwise. The obvious rechecks come when the 2023-24 classes mature.
- No private-information lead was adopted into the model. What changed the model is engine and yardstick work:
  comparison-career correction, monotone value, board-less classes dropped, training links fixed, the darko label,
  within-position units, cap cost, conditioned availability draws and comp-pick numbering.

## Realized QB value as played (`review.REALIZED_QB`, Nov 2026)

The realized yardstick (`review.RealizedV2`) used the model's draft-day QB rule: seasons before the team's qb_from
counted zero, and a team with no flagged QB need got nothing from a QB it drafted. So Mahomes delivered -0.81 titles
for KC 2017 (Alex Smith on draft day), Brock Purdy 0 for SF, Jalen Hurts -0.06 for PHI. Now "played" (default; env
REALIZED_QB=rule restores it): a drafted QB's real seasons count wherever he went; his season values already show
whether he started, so a rookie who sat still counts ~0. Mahomes +29.10, Purdy +23.09, Hurts +22.55, Kyler Murray
-0.43 -> +6.96; 24 team picks and 75 model recommendations change. The model's valuation is unchanged (it still
uses the draft-day flag); only the scoring of what happened.

Scorecard, model minus team, round 1 (95% over draft years). The production review is itself leave-one-draft-out
(`review.REVIEW_HOLDOUT`, since the HTML simulator commit): each draft valued by an engine fit without it.

| | Old scoring | As played |
|---|---|---|
| Review 2014-2022 (leave-one-draft-out, current engine), first pick | +0.77 (-0.36 to +1.61) | -0.01 (-1.21 to +1.06) |
| Same, averaged | +1.74 (+1.00 to +2.46) | +1.33 (+0.47 to +2.18) |
| Nov 6 engine, rescored: first / averaged | +0.70 / +1.74 | +0.50 / +1.49 |

Rounds 2-4 and 5-7 barely move (R2-4 averaged +0.66, +0.10 to +1.39; R5-7 -0.05). The drop is teams' QB picks
that the old rule zeroed, mostly Mahomes at #10, now counting against the model's alternative. Round 1 stays the
model's best-supported edge on the averaged count; the first-pick count is a tie. Also rescored: the simulator's
"delivered" columns. Not rescored: older study files built on RealizedV2 (misses / defer candidates, qb_sit_choice).

**Held-out review rerun (`scripts/holdout_review.py`, Nov 12, current engine, as-played QBs).** Found: since the
production review became leave-one-draft-out, `review.engine_for` overrode the engine each job set, so "expand"
(only what was known on draft day: classes before Y, seasons through Y-1) silently ran as leave-one-draft-out and
the report's "full engine (in-sample)" column was the same leave-one-draft-out review. Fixed (each job keeps its
engine; `python scripts/holdout_review.py expand [workers]` reruns only those drafts). Round 1, model minus team:

| Drafts | Version | First pick | Averaged |
|---|---|---|---|
| 2014-2022 (n 50) | leave-one-draft-out | -0.01 (-1.21 to +1.06) | +1.33 (+0.47 to +2.18) |
| 2019-2022 (n 24) | leave-one-draft-out | +0.70 (-0.70 to +1.89) | +1.17 (-0.27 to +2.76) |
| 2019-2022 (n 23) | expanding window (draft-day knowledge) | +0.57 (-0.98 to +1.68) | +1.45 (+0.44 to +2.66) |

The expanding window makes the same choice as leave-one-draft-out at 51% of picks (round 1 65%) and does as well or
better in round 1 (non-QB recommendations averaged +1.25, +0.26 to +2.30); rounds 2-4 +0.61 (-0.12 to +1.70), rounds
5-7 -0.09. The round-1 edge holds out of sample on the averaged count, not on the first-pick count.

## Round-1 QBs: the yardstick was flattering them (`scripts/qb_need_yardstick.py`, `scripts/qb_haircut.py`, Nov 13 2026)

The complaint a team would make: "you want us to take a QB in round 1 every year." Round 1 of 2014-2026: the model
recommends a QB at 37% of picks, teams take one at 10%. It's all teams with a flagged QB need: model 69-86% of their
round-1 picks (veteran 36+, below-average paid veteran, stopgap, later-round rookie, no starter), teams 6-22%
(no starter 64%); teams with a paid veteran or a first-round rookie: model 0%, teams 1-2%. The model also recommends
a QB at 25% of round-1 picks of teams that took a QB in rounds 1-2 in the previous two drafts (teams 5%).

**Yardstick bias.** The review scores a recommended QB "as played": his real career, at the team that really drafted
him (usually one that started him). But the model recommended him to a team whose need was real only with
probability p_need (0.52-0.79); otherwise the incumbent kept the job and the rookie sat. The prediction discounts
by p_need, the yardstick didn't. Fair score: p_need x (as played) + (1 - p_need) x (sat through the window, cost
charged), the same branches the model uses. Team picks unchanged. Drafts 2014-22, model minus team where they
disagree, each rec once (95% over draft years):

| Round 1 | As played | Fair |
|---|---|---|
| Averaged | +1.33 (+0.47 to +2.18) | +0.64 (-0.06 to +1.31) |
| First pick | -0.01 (-1.21 to +1.06) | -0.47 (-1.76 to +0.65) |
| QB recs (18) / non-QB recs (32), averaged | +2.49 / +0.67 | +0.60 / +0.67 |
| Model's QB recs, delivered / predicted | 0.99x | 0.57x |

Rounds 2-4: +0.73 -> +0.60 (+0.21 to +1.01); QB recs 1.18x -> 0.90x. So the round-1 edge credited to QB picks was
mostly the yardstick: scored fairly, QB recommendations do no better than non-QB ones, and round-1 QBs for teams
with an uncertain need deliver ~0.57x what the model predicts. This also applies to the 'QB valuation' diagnosis
above ("model's QB choices round 1 1.20x", "no broad QB over-valuation"), scored as played.

**Tested fix: shrink QB value** (`scripts/qb_haircut.py`: the review's choices replayed with the positive part of
every QB value x h; same boards, states, tables (h = 1 reproduces the review at 100% of picks); fair yardstick).

| h | R1 averaged | R1 first pick | R2-4 averaged | R1 QB recs | Paired vs h=1 per draft, R1 / R2-4 |
|---|---|---|---|---|---|
| 1.0 | +0.64 | -0.47 | +0.60 | 41% | |
| 0.8 | +0.84 (+0.19 to +1.44) | -0.14 | +0.35 | 38% | +2.4 (-1.6 to +7.2) / -0.8 |
| 0.65 | +0.83 (+0.11 to +1.43) | +0.50 | +0.41 | 32% | -3.3 (-13.0 to +5.9) / +5.3 |
| 0.5 | +0.55 | +0.21 | +0.44 | 23% | -10.5 (-34.7 to +9.2) / +9.2 (-1.2 to +18.0) |
| 0.35 | +0.31 | -0.30 | +0.50 | 10% | -11.7 / +14.9 (+3.9 to +24.6) |

No h wins on every yardstick: round 1 gains a little on the each-rec-once counts at 0.65-0.8 and loses on the paired
count; rounds 2-4 gain on the paired count only. Not adopted. Round-1 recommendations are not a tested edge over
teams once QBs are scored fairly (+0.64, range touches zero); non-QB round-1 recommendations (+0.67) are the part
that holds. Found on the way: a first-round QB in his rookie window whose contract is missing from the data fell
to "veteran (contract not in data)" (WAS 2020, Haskins: p_need 0.54 instead of a filled need); fixed (a drafted
player with no contract in the data, 1-4 years after his draft, is on his rookie deal; only WAS 2020 changes).

**Adopted (Nov 13 2026): fair scoring is the review's default** (`review.REALIZED_REC = "fair"`, `RealizedV2.titles_rec`;
`REALIZED_REC=played` restores the old scoring). It applies to players the team didn't draft (the model's
recommendation, the best player on the board); team picks and the simulator's "delivered" columns stay as played.
`python src/review.py rescore` recomputes the realized columns of the saved review. Production review after the
Haskins fix, drafts 2014-22, model minus team where they disagree, each rec once (95% over draft years): round 1
averaged +0.60 (-0.06 to +1.20), first pick -0.49 (-1.77 to +0.62); rounds 2-4 averaged +0.59 (+0.21 to +1.01),
first pick +0.40 (+0.03 to +0.80). Round-1 QB recs +0.46 (n 18) vs non-QB recs +0.67 (n 32); QB recs delivered /
predicted 0.59x (round 1), 0.90x (rounds 2-4). Earlier scorecards in this README were scored as played.

## Position pricing robustness (`scripts/units_robust.py`, Nov 14 2026)

The complaint a team would make: "your position mix is an artifact of how you price positions." The model prices
each season at what veterans at that position earn for that role (outcome units "single", adopted Nov 2026); the
other defensible choice ("family") rescales each position family to match its own later pay, which marks down
rotational positions (RB, TE, EDGE, DL) and favors OL. The production review (leave-one-draft-out, 2014-2022) was
rerun under family units (`OUTCOME_UNITS=family python scripts/holdout_review.py`) and each recommendation compared:

| Production rec vs family-units rec, all picks | Same player | Same position | Position flips |
|---|---|---|---|
| Round 1 | 78% | 5% | 17% |
| Rounds 2-4 | 57% | 3% | 40% |
| Rounds 5-7 | 51% | 5% | 44% |

Flips are almost all one trade: EDGE / WR / TE (single) vs OL (family). Rounds 2-4 recommendations: OL 23% single,
59% family; EDGE 14% vs 0%.

Production recs minus the team's pick where they disagree (each rec once, averaged; titles on the single
yardstick, then on the family one; bootstrap over draft years):

| | Holds under both | Position flips |
|---|---|---|
| Round 1 | +0.83 (+0.13 to +1.39) / +0.73, 35 recs | +0.04 (-1.28 to +0.91) / -0.36, 15 recs |
| Rounds 2-4 | +0.67 same player, +0.74 same position (wide) / +0.87, +0.56, 56 recs | +0.47 (-0.01 to +0.95) / +0.34, 38 recs |
| Rounds 5-7 | about -0.1 | about -0.1 |

- Round 1: the edge over teams is in the recommendations that hold under both pricings; the flips are worth
  about nothing.
- Rounds 2-4: both kinds beat teams by similar amounts, so the edge isn't a pricing artifact.
- Where the two pricings disagree in rounds 2-4, the family (OL) recs did better on every outside yardstick
  (seasons started +0.70, AV +4.5, second-contract pay +0.55% of cap) while titles tie (single yardstick +0.10,
  family -0.26, both crossing zero). That reverses the Nov 5 pay result that favored single units; not enough to
  switch back, but "you undervalue OL" is a fair point.

Adopted (simulator only, no model change): the simulator also carries the family-units tables
(`OUTCOME_UNITS=family python src/simulator.py build`) and labels the model's top choice "firm" or "depends on
position pricing" (naming the family-units choice), with the past-drafts result for that round. Tables are now
stored differenced (`export_simulator_artifact._delta`), so both sets take less room than one did.

## Late rounds: value a player where he'd really land (`scripts/landing_shade.py`, Nov 14 2026)

The complaint a team would make: "your rounds 5-7 picks are worse than ours." Not a thin candidate pool (deeper boards
didn't help: "Deeper rankings: Brugler grades"); the model's late recs deliver about half their predicted titles and half
went undrafted. Fall shading values a player still there at pick k as if he'll turn out like a pick-k player, but it
was fit on where fallers really went: still there at k means landing at k *or later*, often undrafted. Test: average
the shaded talent over his landing spot given he's still there (draftday's availability model: log(L+10) normal around
log(rank+10) + drift, spread by rank, truncated at L >= k, undrafted at 300); same tables, only the lookup changes.
"fallen": only players already past their rank; "all": every player. Replayed choices ("now" = the production
review at 100% of picks), fair yardstick, 2014-2022:

| Model minus team (each rec once) | Now | Fallen | All |
|---|---|---|---|
| Round 1 | +0.60 (-0.07 to +1.23) | +0.58 | +0.85 (-0.13 to +1.71) |
| Rounds 2-4 | +0.59 (+0.21 to +0.98) | +0.13 | +0.23 (-0.08 to +0.65) |
| Rounds 5-7 | -0.09 (-0.22 to +0.11) | -0.10 | -0.06 (-0.17 to +0.10) |
| Rounds 5-7 delivered / predicted | 0.55 | 0.40 | 0.63 |

"All" minus now, each distinct swap once: round 1 +2.26 (+0.29 to +4.51, 19 swaps, starts +1.9; top 3 carry 47%:
Wentz over Lynch, M. Thomas over Cook, Watson over M. Williams; losses Bridgewater over Bortles, Burrow over Herbert),
rounds 2-4 -0.14 (-0.66 to +0.39, starts +0.5), rounds 5-7 +0.12 (+0.01 to +0.24, starts +0.4; moves toward CB / S).
"Fallen" is worse (it pushes the model toward players just about to fall).

Verdict: not adopted. "All" wins head to head but rounds 2-4 vs teams drop (+0.59 -> +0.23) while the paired test ties,
the same split that kept the QB haircut out. Recheck when the 2023-24 classes have four seasons. Neither version makes
rounds 5-7 beat teams: the answer to the complaint stays "don't use the model's late-round picks" (the simulator says so).

## After the rookie deal: seasons 5-7 for the drafting team (`scripts/second_contract.py`, Nov 14 2026)

The complaint a team would make: "you only value rookie-contract years; a hit gets extended or tagged." The model's
window is 5 seasons: seasons 1-4 on the rookie deal, season 5 only for round-1 picks kept on the option. Seasons 5-7
priced like the label (`scripts/second_contract_label.py`: same harness and pay curves as the darko label, which it
reproduces for seasons 1-5), counted for the drafting team only if he played that season for it; cost = APY (% of
cap) of the latest contract with the drafting team (cap-hit version as a check). Drafted players 2011-2020 (2,335):

- Still with the drafting team: season 5 20% (top-10 picks 52%), season 6 14%, season 7 11%; mostly extensions
  (9% of all players in season 5), tags 1%.
- No below-market second contracts on average: retained players cost more than their seasons were worth (top-10
  picks -1.5% of cap in season 5 by APY; by the backloaded cap hit +0.6% in season 5, negative from season 6).
- In the model's own currency (0.19 x value - 0.10 x cap cost, QBs 0.34) a kept player still adds, so season 5 would
  add, as a share of seasons 1-4 (+ option): picks 1-10 +13%, 11-32 +11%, 33-64 +4%, 65-100 +6%, 101+ +4-5%;
  by position (picks 1-100) QB +14%, EDGE +15%, WR / DL +8%, OL / TE +5-6%, LB / RB / DB ~0. Seasons 6-7 are outside
  the 5-season title window.

Not adopted: top picks would gain ~8% relative to late picks and QB / EDGE a little, in the direction already found
too high (round-1 QB recs deliver 0.59x predicted, "Round-1 QBs: the yardstick was flattering them"); doing it properly
needs season 5 sampled for every pick plus a matching realized yardstick and a full rebuild. For trades: moving down
out of the top 10 is slightly less attractive than the calculator says.

## Draft-night trade-ups: value the player on the clock (`scripts/trade_target.py`, `review.TRADE_VALUE`, Nov 14 2026)

The complaint a team would make: "we traded up for a player, and you scored the trade as if we'd get a typical pick."
The calculator valued every current pick as a typical player at the slot (a received pick as a QB if one was taken).
A team trading on the clock knows the board. Test: draft-night pick-only trades 2014-2021 (221), each team's draft-day
state, outcomes scored as played (as `scripts/future_pick_realized.py`); the trade's earliest current pick valued as
the player taken there (`trade.Pick(talent=...)`: his board rank shaded toward the pick, his position):

| Corr(predicted, realized) | Slot (old) | On the clock | Every current pick (upper bound) |
|---|---|---|---|
| One side per trade (221) | +0.24 (+0.02 to +0.50) | +0.51 (+0.27 to +0.70) | +0.52 |
| Trade-up side (221) | +0.34 | +0.39 | +0.43 |
| Trade-ups for a QB (19) | +0.12 | +0.16 | +0.19 |
| Trade-ups for a non-QB (202) | -0.01 (-0.24 to +0.20) | +0.26 (+0.03 to +0.45) | +0.37 |

Paired gain on the clock minus slot: one side per trade +0.25 (0.00 to +0.52), non-QB trade-ups +0.27 (+0.02 to
+0.51); calibration slope ~1 (slot 0.60). Mean absolute error barely moves (QB trade-ups dominate it).

Adopted for the team that gets the pick on the clock (`review.score_trades`, `TRADE_VALUE=clock`; `slot` restores):
the team giving it up keeps the typical-player value, since its alternative was its own choice there, not the
buyer's target (valuing it at the buyer's player swung SEA 2014's #32 trade-down from +0.84 to -11.7 for passing on
Bridgewater). Verdicts recomputed (`python src/review.py verdicts`, each draft's leave-one-draft-out engine, as the
review): 35% move by more than 0.05, 46 of 802 change sign, e.g. NO 2014 up to #20 for Brandin Cooks -0.51 -> +0.74,
GB 2020 up to #26 for Love +8.1 -> +3.0. Simulator trade verdicts follow. The app's Trade offer tab still values
picks as typical players (`trade.Pick` now takes an optional `talent` for a known target).

## QB need while the veteran is under contract (`scripts/qb_need_contract.py`, `review.p_need`, Nov 15 2026)

The complaint a team would make: "our veteran is signed through next year; we're not drafting his replacement in
round 1." `review.p_need` (the chance a flagged QB need is real) was one rate per flag. For each flagged team-year
2014-2024 (the review's draft-day states): was the team's main QB the next season (most regular-season snaps) someone
other than the draft-day QB? Split by the draft-day QB's contract (latest deal known on draft day, as
`review.States.qb`):

| Flag | Under contract next year | Expiring | Old rate |
|---|---|---|---|
| Age 36+ | 0.38 (16) | 0.63 (27) | 0.52 |
| Paid veteran after a below-average season | 0.40 (30) | 0.76 (17) | 0.54 |
| Both pooled | 0.39 (46) | 0.68 (44) | |
| Stopgap veteran | 0.83 (12) | 0.80 (25) | 0.79 |
| Later-round rookie | 0.61 (23) | 0.67 (3) | 0.62 |

The flag totals reproduce the old rates; the pooled veteran split is stable leaving out any one draft year (0.36-0.43
vs 0.64-0.72) and the same excluding teams that took a QB in rounds 1-2 (under contract / expiring: age 0.38 / 0.59;
below average 0.38 / 0.69). Stopgaps and later-round rookies don't split.

Choice test (`python scripts/qb_need_contract.py choice`): the review's choices at every pick of the 80 affected
team-drafts 2014-22 (same board, states, tables, need counts; "now" reproduces the production review at 100%) with
the split rates, each draft's own year left out of the rates. Contract rule minus now, titles x100 per draft (95% over
draft years), scored three ways: fair yardstick with the old p_need, with the new one, and "what happened" (the rec
QB counts as played if the incumbent was in fact replaced the next season, else as sat; teams that took a QB in rounds
1-2 use the new p_need, since the pick may have caused the change):

| | Under contract (p down) | Expiring (p up) | All affected |
|---|---|---|---|
| Old p_need yardstick | +3.37 (-1.30 to +9.14) | +1.87 (-4.81 to +10.70) | +5.25 (-0.50 to +12.77) |
| New p_need yardstick | +4.09 (+0.07 to +9.33) | +3.46 (-4.02 to +14.59) | +7.55 (+1.52 to +17.06) |
| What happened | +5.01 (+1.45 to +10.09) | +3.95 (-3.60 to +14.55) | +8.96 (+2.59 to +17.53) |

Under contract: QB recs 26% -> 15% of picks (round 1 68% -> 51%); out go Hundley (x8, 2015), Connor Cook, Paxton
Lynch, Kizer, Drew Lock (MIN 2019); in come mostly WRs (Michael Thomas, DK Metcalf, Allen Robinson, Ja'Marr Chase);
one loss, Watson to CAR 2017. Expiring: QB recs 26% -> 32%; Watson to NYJ 2017 and Hurts to IND 2020 vs Connor Cook
over Michael Thomas (ARI 2016) and Davis Webb over Chris Godwin (NE 2017). The expiring side is noisy.

**Adopted** (`review.p_need(status, qb_from, left)`, `left` = seasons left on his deal after this one, from
`review.States.qb`; veteran flags 0.39 under contract next year / 0.68 expiring; contract not in the data keeps the
old rate; `QB_NEED_CONTRACT=off` restores). Review rerun (2014-2026): 68 recommendations change. Changed picks
2014-22, realized (fair yardstick, new p_need) new minus old: round 1 +5.35 (9 picks), rounds 2-4 +54.5 (43).
Scorecard, model minus team where they disagree, each rec once (old -> new): round 1 averaged +0.60 -> +0.80 (+0.17 to
+1.34), first pick -0.49 -> +0.20 (-0.52 to +0.89); rounds 2-4 averaged +0.59 -> +0.43 (+0.10 to +0.81), first pick
+0.40 -> +0.20 (-0.12 to +0.55). The rounds 2-4 drop is the each-rec-once averaging, not worse picks: Hurts is now
also recommended at IND #34/#41 and PIT #49 (+14 each over the old recs), which lowers his averaged margin from +22.2
to +15.1, and Chris Godwin (+7.9) drops out. Round-1 QB recs 41% -> 40%. Trade verdicts unchanged (no p_need).
2026: CLE (Watson), LA (Stafford), LV (Cousins), PIT (Rodgers) 0.52-0.54 -> 0.68; LAC (Herbert), TB (Mayfield) 0.54
-> 0.39. The simulator shows the contract next to the chance ("39% likely, he's under contract next year").

## Do some positions matter more against strong opponents? (`scripts/opp_strength_absence.py`, Nov 15 2026)

The question: are contracts mispriced compared with titles? Contracts pay for what a player adds over a regular
season against an average schedule, but titles are won in playoff games against good teams. If one unit's worth over
its backup grows with opponent strength faster than others', title value per curve-dollar would differ from pay by
position. Data: the betting-line absence data (carry-forward absences, 9 groups, 2,992 regular-season games 2013-25,
20,781 absent starter-games), plus 155 playoff games (1,024 absent starter-games). Opponent strength = full-strength
team-season rating from the base absence model, centered by season (sd 3.3 pts; playoff teams average +2.9).
Decision rule set before the run: a choice test only if equal interactions across positions is rejected (p < 0.05)
on the spread AND the actual margin points the same way.

**1. Opponent rating as a continuous measure** (extra pts per absent starter per +1 pt of opponent; worth against a
playoff-level opponent / worth against an average one):

| | QB | RB | WR | TE | OL | DL | LB | CB | S |
|---|---|---|---|---|---|---|---|---|---|
| Spread: ratio | 1.00 | 1.02 | 0.99 | 0.74 | 1.15 | 1.02 | 1.21 | 1.12 | 0.91 |
| Margin: ratio | 1.03 | (b ~0) | 0.54 | 0.77 | 0.70 | 2.22 | 0.75 | 0.98 | (b > 0) |

Spread: all interactions 0 p 0.21, equal across groups p 0.39; only OL is individually nonzero (x1.15, interaction
-0.024, -0.042 to -0.005). Margin: equal p 0.57; a common interaction of +0.05 (+0.015 to +0.086), so absences
cost *less* against strong teams in actual results.

**2. Strong opponent = top third of ratings** (+1.3 or better). Spread: equal p 0.026 (OL -0.40 -> -0.60, CB -0.32
-> -0.46 vs strong; TE and S smaller). Margin: equal p 0.17, and OL goes the other way (-0.88 -> -0.13).

**3. Playoffs** (regular-season ratings as an offset). Spread: non-QB absences cost 0.24 *less* in the playoffs
(+0.08 to +0.41), TE +0.99; QB -0.35 (-0.95 to +0.26). Margin: every interval wide (QB +1.6, -0.9 to +4.0; S -2.7).
Carry-forward counts long-gone starters as absent, which the market may already treat as part of the team by
January, so the playoff spread result is partly mechanical.

Verdict: no position's worth reliably rises against good opponents. The one spread signal (OL a little more
against strong teams, x1.15) reverses on actual results, and on the pre-set rule this doesn't go to a choice test.
On this measure, contracts' regular-season pricing isn't missing a title premium by position. Not adopted.

### Depth vs the top player in high-leverage games (`scripts/depth_leverage.py [price|share]`, Nov 15 2026)

User's lead (from a post on WR3s in the playoffs): with more preparation, defenses take away a team's best player, so
the 2nd / 3rd option matters more in big games, and pay (set on regular-season production) would under-price depth.
Within each team-game's regulars at a group, rank 1 = top player, 2+ = depth (WR3 / CB3: rank 3+), ranked by pre-season
price (robustness: recent snap share). Statistic = c_depth - c_top: how much more a missing depth player costs than a
missing top player per +1 pt of opponent rating, in playoff games, or in rematches (2nd+ meeting this season, more
film; 562 games). The lead predicts < 0. Rule set before the run: predicted sign on spread AND margin, p < 0.05 on
one, for WR or pooled non-QB.

| c_depth - c_top (p) | Opp rating, spread | Opp, margin | Playoffs, spread | Playoffs, margin | Rematch, spread | Rematch, margin |
|---|---|---|---|---|---|---|
| WR, by price | +0.04 (0.35) | -0.10 (0.61) | **-1.22 (0.04)** | -2.27 (0.30) | +0.21 (0.49) | +0.15 (0.92) |
| WR, by share | +0.02 (0.60) | -0.31 (0.07) | +0.41 (0.47) | -1.06 (0.58) | **+0.99 (0.002)** | +0.10 (0.95) |
| WR3, by price | -0.03 (0.51) | -0.07 (0.73) | -0.89 (0.39) | -3.50 (0.22) | +0.38 (0.27) | +1.29 (0.41) |
| CB3, by price | -0.03 (0.25) | -0.27 (0.13) | -1.15 (0.06) | -1.51 (0.52) | +0.06 (0.83) | -1.13 (0.33) |
| Non-QB pooled, by price | -0.00 (0.78) | -0.02 (0.73) | -0.21 (0.29) | +1.92 (0.09) | **+0.22 (0.01)** | -0.39 (0.40) |
| Non-QB pooled, by share | -0.02 (0.08) | +0.02 (0.76) | -0.52 (0.08) | -0.20 (0.86) | +0.11 (0.40) | +0.28 (0.57) |

- Against stronger opponents nothing moves: top players' absences cost the same (c_top ~0), and depth's no more.
- Playoffs: WR by price meets the rule (spread -1.22, margin same sign but n.s.), but it rests on 36 depth-WR absences in
  155 games, comes mostly from the market pricing missing top WRs *less* in January (c_top +0.66), and flips sign when
  roles are ranked by snap share (+0.41). One pass in 12 eligible cells, not robust: treated as noise.
- Rematches, the cleanest "more preparation" test, go the wrong way where significant (depth absences cost *less*).
Verdict: no support for depth mattering more in high-leverage games. Note the measure is depth player vs his backup
among regulars (60%+ snaps); a true part-time WR3 is outside it. Not adopted.

## Does the market overpay older veterans? (`scripts/age_pricing.py`, `scripts/age_pricing_tenure.py`, Nov 15 2026)

Lead #3 of the contract-efficiency list. If teams pay veterans for past seasons, a dollar on an older player should buy
fewer points of team strength. That would make rookie-contract production worth more next to the veteran
alternative and would move pick-vs-veteran and trade advice (not which prospect to take). Bridge-style team test,
416 team-seasons 2013-25: season-demeaned rating on roster value (snap share x pre-season price, % of cap: latest
veteran contract signed by then, or slot-expected price on a rookie deal) split by contract type and age at
September 1. Rule set before the run: equal slopes across non-QB veteran age buckets; the lead predicts a decline.

| Pts per 1% of cap | <=26 | 27-29 | 30-32 | 33+ | Equal (p) | Trend per step |
|---|---|---|---|---|---|---|
| Non-QB vets, dollars that played (share x price) | 0.20 (0.13-0.28) | 0.21 (0.13-0.29) | 0.16 (0.07-0.25) | 0.35 (0.11-0.60) | 0.45 | +0.04 (-0.04 to +0.12) |
| Non-QB vets, dollars spent (price if any snap) | 0.11 | 0.11 | 0.08 | 0.26 | 0.30 | +0.04 (-0.01 to +0.10) |

- Pooled: non-QB vets under 30 0.205, 30+ 0.192 (p 0.80; ratio ~0.94, roughly 0.5 to 1.4). QB vets under 30 0.34, 30+
  0.32. By position group 30+ vs under 30: OL 0.22 vs 0.21, DB 0.27 vs 0.29, front 0.13 vs 0.13, skill 0.14 vs 0.25
  (skill the only gap, wide intervals).
- Contract age: deals signed 2+ seasons earlier 0.16 vs newer 0.22 (p 0.13), a weak hint that pay set on old
  information buys a bit less, not specific to age.
- Players cut before playing (invisible to the team test): veteran contracts 2013-21 with APY >= 1% of cap, unplayed
  guaranteed seasons as a share of contract value: non-QB <=26 1.3%, 27-29 2.7%, 30-32 4.4%, 33+ 2.1%. Older players
  get shorter deals (1.6-2.1 years vs 2.7), so the hidden dead-money cost of age is ~3% of contract value at most.
Verdict: the market prices age about right per dollar; no overpay big enough to matter. Nothing changes.

## Outside free agents vs own-team re-signings (`scripts/fa_vs_extension.py`, `scripts/fa_vs_extension_checks.py`, Nov 15 2026)

Lead #4 of the contract-efficiency list: a winner's curse. Outside buyers know less than the player's own team, so
free-agent dollars might buy fewer points than re-signing / extension dollars. Same team test as the age section.
Each veteran's current contract is classed by where he played the season before signing: own team (re-signed,
extended, tagged; 56% of non-QB veteran value), new team (outside free agent or traded; 30%), or no snaps the season
before (14%). Rule set before the run: equal slopes own vs new for non-QB veterans.

| Non-QB vets, pts per 1% of cap | Own team | New team | Own - new (p) |
|---|---|---|---|
| Baseline (dollars that played) | 0.22 (0.16-0.28) | 0.12 (0.05-0.19) | +0.10 (0.005) |
| Dollars spent (price if any snap) | 0.11 | 0.04 | (0.02) |
| Contracts >= 5% of cap | 0.20 | 0.15 | (0.25) |
| Under 29 / 29+ | 0.22 / 0.21 | 0.11 / 0.13 | +0.11 (0.06) / +0.08 (0.24) |
| **Franchise fixed effects** | 0.21 (0.14-0.27) | 0.23 (0.15-0.32) | **-0.03 (0.55)** |

- The pre-set test passes at baseline, and it isn't age (value-weighted ages 28.4 vs 28.9). By group the gap is
  largest at the front seven (0.15 vs 0.01, p 0.03).
- But within a franchise the gap disappears: a year with more outside free-agent money buys as many points per dollar
  as own-team money. The baseline gap is *which teams* lean on free agency (franchises that are persistently weaker
  for other reasons), not free agents being overpaid. Consistent with that, this season's free-agent money predicts a
  *lower* rating next season (-0.07 vs own +0.11): it tracks weak organizations.
- Early exits: unplayed guaranteed seasons, contracts signed 2013-21 (APY >= 1%): non-QB new team 1.9% of contract
  value, own 0.6%: real but small.
Verdict: no winner's curse in prices once franchise quality is held fixed. Nothing changes for the draft model.

## Do teams pay for counting stats, and do those dollars win? (`scripts/stat_premium.py`, Nov 15 2026)

Lead #2 of the contract-efficiency list. Step 1: veteran contracts signed 2014-25 with a prior season of 30%+ snaps
(4,433): APY (% of cap) on prior-season label quality q, snap share, age and signing year, then adding that season's
counting stats (per 17 games, standardized within group). Also: do the stats predict *next* season's q beyond this
season's (information the grade misses) or not (noise the market pays for)?

| Group | R2 quality only -> + stats | Pay per SD of stat, % of cap (95%) | Predicts next season's q? |
|---|---|---|---|
| WR | 0.70 -> 0.75 | rec. yards +1.86 (1.46-2.26); TDs +0.27 (0.05-0.49) | yards yes (+0.15, p < 0.01); TDs no |
| TE | 0.52 -> 0.60 | rec. yards +0.85 (0.56-1.14); TDs +0.11 | yards yes (+0.08, p 0.01); TDs negative (p 0.05) |
| EDGE | 0.60 -> 0.67 | sacks +0.47 (0.11-0.82); QB hits +0.89 (0.54-1.24) | sacks no (p 0.33); hits weak (p 0.16) |
| DL | 0.60 -> 0.68 | sacks +0.52 (0.22-0.82); QB hits +0.60 (0.31-0.90) | sacks no (p 0.94); hits yes (p 0.03) |
| LB | 0.41 -> 0.43 | tackles +0.45 (0.10-0.79); sacks +0.23 | tackles yes (p 0.01) |
| DB | 0.37 -> 0.41 | passes defended +0.49 (0.32-0.67); INT +0.15 (-0.02-0.32) | neither (p 0.13 / 0.50) |
| RB | 0.51 -> 0.58 | rush yards +0.48; rush TDs +0.34; rec. yards +0.48 | none significant |
| QB | 0.66 -> 0.68 | pass TDs +2.28 (0.34-4.22); yards +0.09 | no (163 contracts) |

Step 2: each contract's APY = quality part (q / share / age terms) + stat part + residual; team-seasons 2014-25,
rating on roster sums of snap share x each part while the contract is in force (79% of non-QB veteran value covered).
Pts per 1% of cap: quality part +0.26 (0.16-0.36), **stat part +0.01 (-0.27 to +0.29)**, residual +0.17; stat minus
quality -0.25 (p 0.16); franchise fixed effects -0.29 (p 0.053).

- Teams clearly pay for production beyond graded quality (R2 up 0.02-0.08 in every group with stats). Some of it is
  information the grade misses (WR / TE yards, LB tackles, DL QB hits predict next season's quality), some looks like
  noise (sacks, TDs, passes defended predict nothing beyond the grade).
- The stat-driven dollars buy about nothing in team strength, but the gap from quality dollars misses the pre-set
  0.05 bar (p 0.16; 0.053 within franchise). Suggestive, consistent with the lead.
- The model's own currency is already protected: the label prices seasons on (snap share, quality) curves, so stats
  never enter, and the stat part is centered within group, so it doesn't move position levels. What it does touch is
  the **second-contract pay yardstick** used in choice tests: it credits stat producers (WR yards most) with pay that
  doesn't buy points, so wins on that yardstick for WR- or EDGE-heavy choices are partly stat premium. Titles remain
  the main yardstick. Not adopted (nothing in the engine to change).

### Rescoring past choice tests without the stat premium (`scripts/pay_rescore.py`, Nov 15 2026)

The second-contract pay yardstick (pick_outcomes.best_pay, classes 2014-21) carries the stat premium. Adjusted
yardsticks per drafted player's best second contract: APY minus the stat part of the pay model at signing (all stats),
or minus only the "noise" stats (sacks, INT, passes defended, TDs: no predictive value for next season's quality).
Coverage: 38% of best contracts, 58% APY-weighted (the rest keep their APY, so the adjustment is understated).
Average stat part on best contracts >= 3% of cap: WR 1.68% of cap, IDL 0.76, EDGE 0.73, RB 0.69, TE 0.58, CB 0.41,
QB 0.34, LB 0.24, S 0.11, OL ~0. Recomputed raw pay matches the stored column exactly (correlation 1.0). Per
team-draft, shared scoring (a player on n teams' paths counts 1/n; earlier README tables used raw sums, so some
numbers differ from those sections):

| Comparison | Titles | Pay as stored | Stat-adjusted | Noise-only adjusted |
|---|---|---|---|---|
| Single vs family units, all | +0.77 | +2.16 (1.40-2.98) | +1.77 (1.11-2.47) | +2.19 |
| Single vs family units, round 1 | +0.29 | +0.32 (0.12-0.53) | +0.13 (-0.07-0.33) | +0.28 |
| Model (single) vs team, all | +5.24 | -0.92 (-2.52-0.65) | -1.41 (-2.82-0.00) | -0.58 |
| Market factors vs B | -0.06 | -0.37 | -0.36 | -0.37 |
| Rank tilt vs B | +0.05 | +0.17 | +0.14 | +0.15 |
| WR / OL stacking discount vs B | +0.21 | +0.53 | +0.52 | +0.54 |
| Rollout lookahead vs B, round 1 | +0.08 | +0.46 | +0.36 | +0.43 |
| Position pay factor vs B | -0.00 | +0.05 | +0.05 | +0.05 |

- No past decision changes. Moves are -0.0 to -0.5 % of cap per team-draft, all in the same direction (the model's
  WR-leaning choices carried more stat premium than the alternatives). The largest: single units' round-1 pay edge
  over family units shrinks from +0.32 to +0.13 (interval now touches 0); the units choice rested mainly on titles
  (+0.29) and holds. The model's pay deficit vs teams grows from -0.92 to -1.41: teams' own picks earned slightly
  less stat-driven pay than the model's recommendations.
- Removing only the noise stats changes almost nothing: most of the premium on WRs is yardage, which does carry
  information about future quality.
Verdict: the pay yardstick is mildly tilted toward stat producers but hasn't decided anything. No change.

## Strict follower, step 1: room capacity (`scripts/strict_follower.py`, `scripts/strict_follower_realized.py`, Nov 15 2026)

Question: if a team took the simulator's top choice at every pick, year after year, would its draftees fit on
one roster? The model has no memory across drafts for non-QBs (a WR taken last year doesn't lower this year's
WR value) and takes the QB need from the team's real history, so a follower drafts the same mix every year.
Follower: the top of `simulator.options` at each pick the team really made, 2014-2025, all 32 teams (other
teams draft as they really did). Snap shares from snap counts (season snaps / team offensive or defensive
snaps); projected = average for drafted board players at the same group, career season 1-4 and consensus rank;
"room" = median total snap share of a group per team-season. Own draftees from the last 4 drafts (R1: 5).

| Group | Room | Real teams' draftees (mean / p95) | Follower (mean / p95) | Follower seasons over room |
|---|---|---|---|---|
| WR | 2.56 | 1.16 / 1.82 | **4.10 / 5.63** | **93%** |
| G | 1.97 | 1.08 / 2.26 | 2.17 / 3.79 | 53% |
| QB | 1.00 | 0.43 / 1.10 | 0.92 / 1.82 | 47% |
| C | 1.00 | 0.29 / 0.96 | 0.83 / 1.79 | 29% |
| T | 2.02 | 0.73 / 1.57 | 1.49 / 2.44 | 19% |
| CB / S / LB / RB / IDL | 2.7 / 2.1 / 2.8 / 1.1 / 1.8 | 0.8-1.4 | 0.0-0.3 | 0% |

Follower picks 2014-25 (teams in brackets): WR 827 (377), G 595 (298), ED 418 (307), T 418 (155), C 241 (68),
QB 160 (140), CB 114 (404), IDL 16 (291), LB 0 (298), RB 0 (246). The recommended WRs really played as much as
projected (3.95 per team-season, spread over other teams), so the overflow isn't a projection artifact: one
team couldn't play them all.

Value at stake, drafts 2017-22 (model's own expected titles, % of one, per team-draft): follower 7.41 vs the
team's real picks 3.94, claimed gain 3.47; snaps beyond the room (everyone keeps room / total) cost 1.70, half
the claimed gain (WR 37% of WR value, QB 22%, C 16%, G 14%). Realized (follower picks matching the stored
review recommendation, 1,139 of 2,304, real careers, per team-draft 2014-22):

| Scoring | Follower minus team (95% CI) |
|---|---|
| As scored (each player where he really landed) | +0.87 (+0.29 to +1.44) |
| Overflow removed, best-first on realized snaps (most favorable) | +0.34 (-0.19 to +0.89) |
| Overflow removed, proportional on projected snaps | -0.21 (-0.72 to +0.30) |

Most of the review's realized edge comes from crediting a strict follower with careers that one roster
couldn't host, mostly WR and QB. Caveats: surplus players could be traded (some value back); real rosters also
hold veterans, so the room left for draftees is smaller than the room used here; half the follower's picks
lack stored realized values. Not adopted yet. Next: a cross-year need term (prior drafts' players at the group
lower a pick's value, measured on rooms, not guessed) tested in the whole-draft choice test, then the
cost of filling the positions the follower never drafts.

## Cross-year need term: the room rule (`scripts/cross_year_need.py`, Nov 15 2026)

Follow-up to the strict follower. **Measured first**: on real teams (drafts 2014-22, picks 1-150, seasons 1-4), a
rookie's playing time doesn't fall as the team's own draftees from the previous 3 drafts hold more of the room
(realized / expected 1.10, 1.16, 1.13, 1.13, 1.08 from 0 to 70% of the room; expected x load interaction +0.04,
se 0.09; no group significant). Veterans absorb the squeeze. Real teams almost never pass ~70%, so the data
can't say what happens beyond the room. **Rule ("room")**: draftees displace veterans freely, but a team's own
draftees can't play more than the room. A newcomer's useful snaps each season = min(L + e, R) - min(L, R), with
L the expected snap share of the team's own draftees at the group that season (last 4 drafts incl. this one;
R1 option year), e his expected share, R the median room; m = useful / expected, snap-weighted over his seasons.
Value at m: quadratic in playing time through cost-only (m = 0), the x0.88 and the x1 table values (titles are
convex in playing time; within ~5% of exact). Within-draft step kept. QBs: p_need x m.

**Test**: two sequential strict followers for all 32 teams, drafts 2014-25 (both start from the team's real
2011-13 draftees): base (today's model) vs room. Realized titles (% of one, per team-draft, 2014-22, all
follower picks, recommended QBs by the fair rule), with snaps beyond the room removed as in step 1:

| Follower minus team's real picks | Uncapped | Capped best-first (favorable) | Capped proportional |
|---|---|---|---|
| Base | +0.74 (+0.10 to +1.36) | -0.29 (-0.88 to +0.28) | -1.22 (-1.79 to -0.67) |
| Room | +0.40 (-0.24 to +1.03) | -0.13 (-0.74 to +0.46) | -0.36 (-0.97 to +0.22) |
| Room minus base, paired | -0.34 (-0.71 to +0.02) | +0.16 (-0.16 to +0.48) | +0.86 (+0.59 to +1.12) |

Picks 2014-25, base / room / teams: WR 827 / 576 / 413, G 594 / 465 / 320, CB 113 / 224 / 433, S 66 / 165 / 232,
ED 419 / 486 / 333, TE 221 / 276 / 188, IDL 16 / 35 / 318, RB 0 / 6 / 259, LB 0 / 0 / 322; 60% of choices
unchanged. The room follower still overfills WR (projected 2.97 vs room 2.56, 86% of team-seasons) and QB (49%):
the rule gives earlier draftees first claim and doesn't anticipate the follower's own later picks, and a WR at
m ~0.8 still beats the alternatives. For real teams (their own recent draftees as L) the rule rarely binds:
m < 1 at 2.2% of picks 1-150 (C 36%, the C room may be undercounted as many C draftees play guard; QB 5%; ED 6%).

Reading: once a strict follower's draftees have to share one roster, neither follower beats the teams' real
drafts (best-first CIs straddle 0); the review's per-pick edge holds for single decisions on a real roster, not
for following every recommendation for years. The room rule helps on the capped yardstick (clearly on the
proportional one) but isn't enough on its own. Waivers recover ~nothing: the rookie deal costs at most the
veteran minimum after ~pick 64, and before that the signing bonus is guaranteed. Not adopted yet.

## Which veteran snaps can a rookie displace? (`scripts/displacement.py`, Nov 15 2026)

Follow-up to the room rule, which let draftees push any veteran aside. Incumbents on draft day as in
`scripts/roster_need.py` (last season's starter-equivalents, se, of players still on the team plus veterans added),
with draft-day attributes: quality (DARKO-for-PFF talent after last season, tercile within group; known for 100%
of incumbent snaps), pay (cap % this season, paid >= 3%), capital (own round 1-2 pick from the last 3 drafts), age.
Drafts 2014-24, non-QB, SEs clustered by team-draft.

**A. Rookie side**: rookie's year-1 se per incumbent se of each type (controls: group x log pick, group x log rank,
earlier rookies at the group). Rookie year-1 se averages R1 0.58, R2 0.46, R3 0.32, R4 0.23.

| Incumbent type | Year 1 | Years 1-2 |
|---|---|---|
| High quality, paid | -0.040 (0.010) | -0.029 (0.010) |
| High quality, cheap | -0.038 (0.010) | -0.036 (0.009) |
| Low/mid quality, paid | -0.022 (0.014) | -0.022 (0.015) |
| Low/mid quality, cheap | -0.006 (0.006) | -0.009 (0.006) |
| Own recent R1-2 pick, not high quality | -0.003 (0.015) | -0.011 (0.015) |

By quality alone: high -0.040, mid -0.025, low +0.010 (year 1). E.g. a WR room of two high-quality starters vs two
low-quality ones costs a rookie ~0.10 se in year 1 (a sixth of a first-rounder's, half a fourth-rounder's).

**B. Incumbent side**: change in an incumbent's se per 1.0 predicted rookie se arriving at his group (controls: group,
last season's se, type, age, age squared): low/mid cheap -0.030 (0.008), high cheap -0.029 (0.025), high paid +0.032
(0.026), low/mid paid -0.011 (0.047), own recent R1-2 +0.001 (0.032). Summed over a room, the rookie's snaps come
almost entirely from cheap low/mid-quality incumbents (-0.13 of -0.14 per 1.0 arriving); only 0.14 of the arrival
shows up as identified incumbents' losses (teams draft where they expect incumbents to fade anyway, so B is
descriptive, not causal).

Reading: quality decides who plays; pay adds a little protection to lesser players (not significant); draft capital
and recency don't protect (no sunk-cost effect detected). For the strict follower this means its earlier draftees
who turn out well keep their snaps and the busts lose them, i.e. the best-first capped scoring in the room-rule
test is the realistic one: base -0.29 (-0.88 to +0.28), room -0.13 (-0.74 to +0.46) vs the teams' real drafts.
For real teams it suggests a quality-weighted roster-need term (the count-based one wasn't adopted); untested.

## Quality-weighted roster need (`scripts/quality_need.py`, Nov 15 2026)

Follow-up to the displacement test: discount a rookie by the quality of the incumbents at his position, not by
bodies. Measured on value (rookie's cfdarko value years 1-4, classes 2014-22, Poisson, controls group x log pick,
group x log rank, off-board, earlier rookies at the group): per incumbent starter-equivalent, high quality -0.143
(0.040), mid -0.051 (0.038), low -0.118 (0.040); bodies only -0.104 (0.027). On value, unlike playing time, poor
incumbents cost about as much as good ones, so quality doesn't separate cleanly. Multiplier for a team's room
exp(sum b (inc - group median)), clipped 0.6-1.4, refit leaving out the draft scored: p10 0.85, median 0.97, p90 1.08.

Choice test on today's production model at every pick 2014-22 (team's real roster and earlier picks; current rule
reproduces the stored review at 100% of picks). Realized titles (% of one); "adjusted" scales a recommended
player's real career by m(this team) / m(team that drafted him).

| Rule | Same choice as current | Paired vs current, all picks (adjusted) | R1 | Model minus team where they disagree (adjusted) |
|---|---|---|---|---|
| Current | | | | +0.07 (-0.36 to +0.52) |
| Count (bodies) | 72% | +0.003 (-0.027 to +0.033) | -0.025 | +0.13 (-0.25 to +0.50) |
| Quality | 71% | +0.001 (-0.029 to +0.030) | -0.056 (-0.149 to +0.022) | +0.05 (-0.39 to +0.46) |

Changes 29% of choices, realized value unchanged; the position mix barely moves (R1-4: WR 535 -> 523, QB 302 -> 298,
CB 0 -> 11). Not adopted. Note: on today's scoring (fair QB rule, units, holdout tables), the per-pick edge where
model and team disagree is +0.07 to +0.10, not significant (R5-7 slightly negative), well below the +0.57 of the
earlier roster-need test on older scoring.

## Room check: are team rooms persistent? (`scripts/room_persistence.py`, Nov 16 2026)

The simulator's room check divides a team's occupied snaps by one league-median room per position (2014-25),
the same for every team. Test: predict each team's total starter-equivalents per position in season Y
(2016-25, 320 team-seasons per group) with coefficients fit only on seasons before Y, from what is known on
draft day. Candidates: league median (pooled, or recent seasons only); + b x the team's deviation last
season (lag1); + two seasons (lag12); lag1 with a separate b after a head-coach change.

| Group | sd of room | RMSE, median | RMSE, lag1 | Cut | Carry-over b | b, new HC | b, same HC |
|---|---|---|---|---|---|---|---|
| LB | 0.38 | 0.385 | 0.308 | 20% | 0.59 | 0.34 | 0.66 |
| S | 0.48 | 0.486 | 0.394 | 19% | 0.56 | 0.51 | 0.57 |
| RB | 0.14 | 0.136 | 0.107 | 21% | 0.59 | 0.50 | 0.61 |
| IDL | 0.30 | 0.305 | 0.252 | 17% | 0.57 | 0.43 | 0.62 |
| WR | 0.22 | 0.219 | 0.187 | 15% | 0.54 | 0.43 | 0.56 |
| ED | 0.32 | 0.321 | 0.276 | 14% | 0.49 | 0.40 | 0.53 |
| CB | 0.44 | 0.435 | 0.373 | 14% | 0.51 | 0.42 | 0.53 |
| G / T / C | 0.60 / 0.42 / 0.44 | 0.602 / 0.419 / 0.441 | 0.536 / 0.386 / 0.401 | 11 / 8 / 9% | 0.46 / 0.37 / 0.40 | | |
| TE | 0.19 | 0.187 | 0.169 | 10% | 0.42 | 0.32 | 0.44 |
| OL (T+G+C) | 0.11 | 0.107 | 0.107 | 0% | 0.37 | | |

- lag1 beats the median in every group (team-clustered 95% CI of the MSE difference excludes 0 in all but QB
  and OL). About half of a team's deviation carries over one season; a second season adds little (b2 0-0.16,
  lag12 no better out of sample). A head-coach change (22% of team-seasons) lowers the carry-over (LB 0.66 ->
  0.34) but splitting by it doesn't improve out-of-sample error.
- Rooms are shares of fixed totals: defense 11.00 (sd 0.05), OL 5.02 (0.13), WR+TE+RB 4.97 (0.09). Within
  them, CB vs S correlate -0.78, ED vs IDL -0.74, WR vs TE -0.74: partly scheme/personnel (nickel, 3-4 fronts,
  12 personnel), partly labels (slot CB vs box S, 3-4 DE). The OL total is all but constant, so the T/G/C
  spread is mostly label movement.
- Even with lag1, most of the spread stays unpredictable (errors 80-90% of the median's).

Not applied to the simulator at this point; applied the same day in 'Room check: projected open snaps' below.
Ledger row room_persistence.

## Incumbents' development: age and years in the league (`scripts/incumbent_development.py`, Nov 16 2026)

Draft-day incumbents (as in `scripts/displacement.py`, drafts 2014-24). Target: his starter-equivalents (se)
for the same team this season (0 if cut, retired, hurt all year). Out of sample 2017-24 (fit on earlier seasons):

| Forecast of incumbents' se | RMSE per player | RMSE per team-group |
|---|---|---|
| Last season's se | 0.267 | 0.768 |
| Room check rule (good 1, middling 0.6) read as a forecast | 0.285 | 0.821 |
| Fitted: group x se + tier x se (base) | 0.234 | 0.526 |
| base + age band | 0.229 | 0.508 |
| base + years-in-league band | 0.229 | 0.503 |
| base + both | 0.229 | 0.506 |

- Most of the gain is attrition: incumbents keep ~70% of last season's snaps on average. Age / years add a
  further ~4% (team-group MSE -0.021, 95% -0.027 to -0.015), largest for WR, T, CB, S, ED, G.
- Starters (se >= 0.5) keep 82% of their snaps at age <= 24, 75% at 25-27, 69% at 28-30, 61% at 31-33, 53% at
  34+; gone entirely 6% -> 29%. Part-timers grow when young (age <= 24 x1.14; after year 1 x1.16) and shrink
  with age (31-33 x0.55, 34+ x0.41).
- Quality: next season's percentile vs 25-27: <= 24 +0.035, 28-30 -0.026, 31-33 -0.037, 34+ -0.057 (q carries
  0.86). Middling incumbents in years 1-3 move up to the top tier 22% of the time, old ones 10%.
- Rookie side: rookie year-1 se per incumbent se, good young -0.045, good prime -0.030, good old -0.062
  (young minus old p 0.35); middling young vs old p 0.82. Career stage doesn't change how much an incumbent
  holds a rookie back; age works through the incumbent's own snaps.

Numbers after the signing-team fix below (before it: 0.788 / 0.531 / 0.511; same conclusions). Ledger row
incumbent_development.

**Missed games (injuries).** A player hurt for most of last season has a low se, so he was projected low (Malik
Nabers, 2026 Giants: 0.19 last season -> 0.30). Added: miss = his median snap share in games played x share of
games missed (injuries, also benchings), by tier and age, plus se two seasons ago ("+inj"). Out of sample 2017-24,
players who missed 40%+ of games while playing 60%+ of snaps when active (n 1,329):

| Subset | n | Last se | Actual | base+both | +inj |
|---|---|---|---|---|---|
| Good tier, 70%+ when active | 251 | 0.34 | 0.46 | 0.36 | 0.49 |
| same, age <= 27 | 102 | 0.35 | 0.55 | 0.40 | |
| Middling / weak tier | 1,047 | 0.26 | 0.23 | 0.21 | 0.26 |

Good players win much of the missed time back, weaker ones mostly don't (40% gone entirely). Team-group RMSE 0.5059
-> 0.5071 (not different; the injured are few per room), player RMSE 0.2285 -> 0.2272. Adopted for the room
check because it fixes the bias where it matters (Nabers -> 0.53).

## Room check: projected open snaps (simulator, Nov 16 2026)

The simulator's room check now shows projected open snaps per position, in starters (1 = every snap), replacing
"% of a league-median room held by good + 0.6 x middling incumbents" (`export_simulator_artifact.room_info`):

- Room: mean league median of the 4 seasons before + b x the team's deviation last season (b per group fit
  2014-25, ~0.4-0.6; `scripts/room_persistence.py`). T / G / C share one OL room (total ~5.0 every season).
- Incumbents: each draft-day incumbent's projected se this season from the "+inj" regression
  (`scripts/incumbent_development.py`: group x last se + tier x last se + age band x last se + years band x last se
  + games missed by tier and age + se two seasons ago; fit on 2014-25 seasons, so past drafts are in sample).
- Open = room - projected incumbents - expected season-1 share of picks made in the simulation. Gold under 0.25 open,
  red under 0.05 (full / over). Information only, as before.
- Room column per candidate: his position's room filled now (%), + his expected first-season share, -> total.
  A green bar marks the model's highest-ranked available player at each position with 0.05+ open (OL one room; removed in v36, see "Room column: rookie-deal players vs the room").

Signing-team fix (`roster_need.departures_and_adds`): OTC lists every team a contract passed through ("LAR/DAL/MIA",
not always in order: "PHI/CHI" for a Bears signing later traded to the Eagles). The last listed team was taken as
the signer, so a re-signed player later traded looked like he left before the draft and joined the later team
(Robert Quinn on the 2014 Dolphins). Now: the listed team he played for in the season signed, else the first.
~3% of contracts. Displacement rerun: good -0.040, middling -0.024, weak +0.009 (was -0.040 / -0.025 / +0.010).
Reruns with the fix (old -> new; no conclusion changes):

| Test | Old | New |
|---|---|---|
| Roster need (`roster_need.py`), +1 open spot: year-1 snap share | +2.0 pts (+0.9 to +3.0) | +1.9 (+0.8 to +3.0) |
| same, value years 1-4 (% of cap) | +0.50 (+0.18 to +0.82) | +0.45 (+0.13 to +0.78) |
| Need choice (`need_choice.py`), paired vs current, all picks: roster | +0.008 (-0.025 to +0.040), same 70% | +0.013 (-0.025 to +0.052), same 70% |
| same: both | -0.007 (-0.038 to +0.026) | -0.004 (-0.039 to +0.030) |
| Quality need (`quality_need.py`), paired vs current, adjusted: quality | +0.001 (-0.029 to +0.030) | -0.008 (-0.036 to +0.021) |
| same: count | +0.003 (-0.027 to +0.033) | +0.006 (-0.022 to +0.034) |
| WR stacking A (`wr_stacking.py`), WR incumbents when a R1-2 WR arrives | -0.74 (-1.70 to +0.22) | -0.76 (-1.71 to +0.20) |
| Availability (`availability_test.py`), needs vs truncated scatter, log loss | -0.001 | -0.001 (QB needs +0.012, same) |

`need_choice.py`'s model-minus-team levels (current rule +0.38 now vs +0.57 in its README table) moved because of
later yardstick changes (fair QB scoring, Nov 13), not this fix; its paired comparisons are the check. Not affected:
`strict_follower.py` / `cross_year_need.py` (rooms from snap counts only), `synergy_test.py`, `rb_passgame_test.py`,
`stack_choice.py` (its own version counting every listed team).

## Room check v2: crowding, missed-season players, contract fixes (simulator, Nov 16 2026)

Spot checks of past drafts (`scripts/room_crowding.py` docstring): of 83 round 1-2 picks made into rooms the check
called full (2015-25), many rooms weren't full (retirements and unsigned free agents still counted; signings after
the draft counted as draft-day roster; OTC repeating a contract under the team that later traded for the player;
known suspensions / long injuries); 12 picks were counted in the wrong room (CB who played S, EDGE vs DL). Rookies
drafted into full rooms still played 0.40 of a starter in year 1 vs 0.49 in rooms under 70% (-0.11 per unit of fill,
se 0.04). In rooms left thin after the draft, incumbents played more than projected (1.66 vs 1.32 starters) and in
crowded rooms less (1.72 vs 2.18): each incumbent was projected on his own, but snaps are shared.

Changes:
1. Crowding. Veterans take a share k of the gap (room - projected incumbents - picks' expected share): incumbents
   playing more or less, and players added after the draft. The page shows (1 - k) x gap as open, the space left
   for rookies; the Room column (v33) reads "filled now +his own expected share -> effective total", the total = filled now +
   (1 - k) x his share (veterans give back k of his snaps; hover for the split). v31 showed the net share ("+2%" for a
   top WR), v32 the raw total (often 120-170%). k (fit
   2014-25 on all veterans' snaps): S 0.64, CB 0.70, IDL 0.75, ED 0.77, LB 0.78, TE 0.84, RB 0.88, WR 0.91, OL 0.95,
   QB 0.98. So OL, QB and WR rooms rarely show open space (2026: 141 of 320 team-rooms 0.05+ open before any pick).
2. Roster players with no snaps last season (`displacement.missed`): two seasons ago on the team or signed / traded
   for, none anywhere last season (Derwin James 2021, Deshaun Watson 2022), and last year's draft picks who didn't play
   (J.J. McCarthy 2025). Projected by kind with their share two seasons ago (`incumbent_development.FORMS["+out"]`).
   They rarely come back as regulars: good players with 0.6+ of a starter two seasons ago averaged 0.17.
3. Contracts (`roster_need.signed_by_draft`): signings from OTC's chronological per-player contract history instead
   of its contract rows. Drops trade copies (Stefon Diggs's 2022 Bills extension listed as a Texans signing),
   practice-squad deals, and signings after the draft (snaps for another team first that season: Le'Veon Bell,
   Chiefs 2020, Jaylon Smith, Packers / Giants 2021, who was counted as leaving Dallas; or a second signing that year
   by a team he wasn't with in week 1: Antonio Brown, Patriots 2019). ~30% fewer signings per year.
4. QB starter rule (the QB-need model's draft-day starter at least the designated starters' mean, 0.75): tested,
   not adopted. Crowding already fills the QB room (QB RMSE 0.182 without the rule, 0.186 with).

Out of sample 2017-25 (2,880 team-rooms; regression, starter rule and k fit on earlier seasons), RMSE in starters:

| Projection | Incumbents' total | + crowding | All veterans' snaps | + crowding |
|---|---|---|---|---|
| "+inj", last season's players only (before) | 0.476 | 0.385 | 0.492 | 0.292 |
| + players with no snaps last season (adopted) | 0.476 | 0.358 | 0.474 | 0.291 |
| + QB starter rule | 0.483 | 0.358 | 0.479 | 0.291 |

All veterans' snaps = everyone in the room but that season's draft class. By room, before -> after (veterans'
snaps): CB 0.54 -> 0.39, ED 0.41 -> 0.29, IDL 0.44 -> 0.27, LB 0.46 -> 0.31, OL 0.91 -> 0.34, QB 0.31 -> 0.14,
RB 0.26 -> 0.16, S 0.52 -> 0.42, TE 0.31 -> 0.18, WR 0.44 -> 0.26. With the old contract rows the same comparison
gave 0.488 -> 0.293 (with missed players), so the contract fix is a small gain; crowding is nearly all of it.
Not fixed (needs dated transactions): retirements and releases before the draft without a new contract (Aaron Donald
2024, Joe Thomas 2018 still count), suspensions, fifth-year options; draft position labels (CB/S, EDGE/DL).

Reruns with the contract fix (old -> new; no conclusion changes):

| Test | Old | New |
|---|---|---|
| Roster need, +1 open spot: year-1 snap share | +1.9 pts (+0.8 to +3.0) | +1.7 (+0.5 to +2.8) |
| same, value years 1-4 (% of cap) | +0.45 (+0.13 to +0.78) | +0.44 (+0.08 to +0.79) |
| Need choice, paired vs current, all picks: roster | +0.013 (-0.025 to +0.052) | +0.010 (-0.027 to +0.048), same 70% |
| same: both | -0.004 (-0.039 to +0.030) | -0.007 (-0.041 to +0.024) |
| Quality need, paired vs current, adjusted: quality | -0.008 (-0.036 to +0.021) | -0.008 (-0.036 to +0.021) |
| same: count | +0.006 (-0.022 to +0.034) | +0.006 (-0.022 to +0.034) |
| Displacement, rookie yr-1 per incumbent se: good / middling / weak | -0.040 / -0.024 / +0.009 | -0.038 / -0.024 / +0.009 |
| WR stacking A, WR incumbents when a R1-2 WR arrives | -0.76 (-1.71 to +0.20) | -0.83 (-1.78 to +0.12) |
| Availability, needs vs truncated scatter, log loss | -0.001 | -0.001 |
| Incumbent development A, team-group RMSE: last / base / +both | 0.768 / 0.526 / 0.506 | 0.682 / 0.497 / 0.481 |

## Position labels in the room check: CB / S and EDGE / DL / LB (Nov 16 2026)

`scripts/position_rooms.py`. The room check counts a pick in the room of his consensus-board label; rooms count
veterans by PFF position. Drafted defenders 2014-25 with a PFF position (1,373): board CB -> S 44 of 348, S -> CB 18
of 204; EDGE -> DL 14, EDGE -> LB 15 of 280; DL -> EDGE 24 of 288; LB -> EDGE 24 of 253. Can pre-draft information
place them better? Each draft predicted from the others:

| Family | Model | Accuracy (top room) | Log loss | Share of picks' expected snaps in the wrong room |
|---|---|---|---|---|
| CB / S | board label (current) | 0.888 | 0.776 | 0.097 |
| | P(room given Hasan's label code) | 0.888 | 0.362 | 0.184 |
| | + combine weight, height, 40 (multinomial logit) | 0.884 | 0.330 | 0.183 |
| EDGE / DL / LB | board label (current) | 0.906 | 0.648 | 0.089 |
| | P(room given label code) | 0.906 | 0.392 | 0.146 |
| | + combine size | 0.904 | 0.271 | 0.156 |

No model picks the right room more often than the label; the moves aren't predictable from size, speed or the
label's finer code (NCB, DL5T, OB). Splitting each pick across rooms by probability is better calibrated (log loss)
and cuts the RMSE of picks' expected snaps per team-room 0.106 -> 0.096, but raises the mean absolute error
(0.037 -> 0.056) and is ~0.01 of a starter against a room error of ~0.29. Brugler's guide (2019-23 only) uses the
same kinds of labels (L.J. Collier: EDGE). Not adopted: labels stay as they are.

## Net gain of a pick: room value and the quality of displaced snaps (`scripts/net_room_value.py`, Nov 16 2026)

Question: with crowding (a rookie's snaps mostly come out of veterans'), should a pick be valued by its gain over the
snaps it displaces, and does that depend on how good the incumbents are? Earlier tests used the rookie's own value
(quality need: good and weak incumbents cost him about equally), which a team's selection could bias: teams rarely
draft early into strength unless they know something. Outcome here: the team's whole room value (everyone at the
position, snap share priced at quality on the veteran pay curve, % of cap), team x season x group, drafts 2014-25,
non-QB. Treatment E = expected own value of the team's round 1-3 picks there (slot and consensus rank), interacted with
draft-day incumbents' starter-equivalents by quality tier (centred on the average room that drafted). Controls: tiers,
last season's room value, group x season FE; team-clustered. Fallers = top third of pick vs consensus rank (>= ~7%
later than rank; 388 of 1,159 rookies), taken mostly because they were there.

| Horizon | Room gain per 1.0 of E (all) | x 1 more good incumbent se | Fallers | Fallers x good se | Others | Others x good se |
|---|---|---|---|---|---|---|
| Season 1 | 0.05 (-0.07 to 0.18) | +0.03 (-0.12 to 0.17) | 0.19 (0.02 to 0.36) | +0.09 | -0.05 | -0.03 |
| Seasons 1-2 | 0.13 (0.04 to 0.22) | -0.02 (-0.14 to 0.10) | 0.27 (0.13 to 0.40) | +0.03 | 0.04 | -0.05 |
| Seasons 1-4 | 0.18 (0.11 to 0.26) | +0.04 (-0.08 to 0.15) | 0.24 (0.12 to 0.36) | +0.15 (-0.01 to 0.30) | 0.15 | -0.03 |

Mean E per rookie: 1.7 (season 1), 4.0 (1-2), 8.1 (1-4) % of cap. By group (seasons 1-2) the gain is 0.02 (DB, OL)
to 0.35 (LB); no group's quality interaction is significant.

- Net gain: only ~5-25% of a rookie's own pay-curve value shows up as a better room on the field; the rest is snaps
  veterans would have played at similar priced quality. This is the replacement-currency issue in "Replacement level by
  position" seen from the team side. It ignores cap: displaced veterans' pay can be cut or never spent, which the
  pay-curve currency counts as the rookie's value; this test measures the field only.
- Displaced quality: no sign that a good room shrinks the net gain (interaction ~0, CI about +-0.13 per good
  starter-equivalent vs a gain of 0.13-0.18). Selection: fallers show no larger penalty for good rooms (if anything
  positive, p 0.06-0.07 over 4 seasons); their overall net gain is higher (other picks partly go where rooms were about
  to decline, which biases their gain down). No support for a quality-of-displaced-snaps term. Not adopted.

## Cap side of a pick (`scripts/cap_side.py`, Nov 16 2026)

Follow-up to "Net gain of a pick": does the ~80% of a rookie's own value that doesn't show up on the field come back
as cap money? Same units, treatment E and controls (+ last season's spend at the group); cap hits from OTC season
histories (every player carried, % of the league cap; covers ~80-84% of team caps). Per 1.0 of the rookies' expected
own value, summed over seasons 1-4 (season 1 / 1-2 similar):

| Outcome | All picks | Fallers |
|---|---|---|
| Room value at the group (field) | +0.18 (0.11 to 0.25) | +0.24 (0.11 to 0.37) |
| Veteran spend at the group | -0.54 (-0.61 to -0.47) | -0.61 (-0.71 to -0.50) |
| Rookies' own cap | +0.45 (0.43 to 0.47) | +0.45 (0.41 to 0.49) |
| Spend at other positions | +0.14 (-0.01 to 0.28) | +0.26 (0.01 to 0.51) |
| Room value at other positions (field) | -0.06 (-0.29 to 0.18) | -0.09 (-0.49 to 0.31) |
| Team total spend | +0.04 (-0.08 to 0.16) | +0.10 (-0.10 to 0.30) |

- Teams do cut veteran spending where they draft, by about the rookie's own salary: spending at the position is
  nearly flat (-0.54 + 0.45 = -0.09) while the field improves +0.18. The small saving (and a bit more) goes to other
  positions, where no value gain is detectable (CI +-0.23). Total spend unchanged.
- The model credits a pick with surplus (own value - rookie cost, `src/value.py`): ~0.55 per 1.0 of own value. Realized
  at the position (field + net cap saved): ~0.27, about half; nothing detectable comes back elsewhere.
- By group (seasons 1-4; field + net cap at the group, as a share of the credited surplus): EDGE 0.57, DL 0.54, OL 0.51,
  WR 0.50, RB 0.34, LB 0.30, DB 0.29, TE 0.10. Field CIs are +-0.15-0.25 and veteran-spend CIs +-0.15-0.30, so the
  groups overlap; the ordering also disagrees with the betting-line test in "Replacement level by position" (which
  found EDGE / RB / IDL over-credited). A roughly uniform shortfall doesn't change rankings; position-specific
  rescaling has failed the choice test before. Not adopted.

## Choice test with net-gain scoring (`scripts/netgain_choice.py`, Nov 16 2026)

Net-gain share r per group (field + veteran spend saved - rookies' cap, over the credited surplus; `cap_side.py`,
seasons 1-4, drafts 2014-22): DB 0.29, DL 0.54, EDGE 0.57, LB 0.30, OL 0.51, RB 0.34, TE 0.10, WR 0.50 (mean 0.40).
Net-gain scoring = realized titles x r of the chosen player's group (QB: the mean). Rule "netgain" = non-QB table
values x r / mean r, r refit leaving out the draft chosen. Harness as `quality_need.py`, every pick 2014-22
(current rule = stored review at 100%).

| | Raw scoring (realized titles) | Net-gain scoring |
|---|---|---|
| Model (current) minus team where they disagree | +0.098 (-0.335 to +0.575) | +0.090 (-0.100 to +0.293) |
| netgain minus current, paired, all picks (same choice 66%) | -0.039 (-0.078 to -0.003) | -0.012 (-0.028 to +0.005) |
| same, round 1 | -0.190 (-0.376 to -0.010) | -0.080 (-0.176 to +0.002) |
| same, rounds 5-7 | -0.057 (-0.092 to -0.028) | -0.024 (-0.042 to -0.007) |

- Scoring: crediting only what teams realize (about 40% of the surplus on average) leaves the model's edge over teams
  about where it was (+0.09 vs +0.10; still not significant): its disagreements lean to positions with higher net-gain
  shares than the teams' picks.
- Rule: the netgain rule (R1-4: QB 302 -> 234, EDGE 135 -> 214, T 48 -> 91, LB 16 -> 3, TE 17 -> 0) loses to the
  current rule on raw scoring (significantly, mostly round 1) and doesn't win even on net-gain scoring, which uses the
  same shares and so favors it. Not adopted; the position values stay.

## Hybrid position pricing: family units in rounds 2-4 (`scripts/hybrid_units_choice.py`, Nov 16 2026)

Rule: round 1 and rounds 5-7 as production; rounds 2-4, the family-units (OL-friendly) recommendation where it differs.
Family review rerun with today's code (QB-need contract split), both leave-one-draft-out, 2014-22, 2,206 picks; the
hybrid changes 426 (44% of rounds 2-4). Rounds 2-4 recommendations, production -> hybrid: WR 372 -> 177, G 177 -> 311,
T 48 -> 239, C 1 -> 29, QB 188 -> 216, EDGE 131 -> 0, IDL 23 -> 0, LB 16 -> 0, TE 16 -> 0.

| Hybrid minus production | All picks | Rounds 2-4 | The 426 switched picks |
|---|---|---|---|
| Titles, single-units yardstick | -0.029 (-0.094 to +0.033) | -0.066 | -0.151 (-0.481 to +0.180) |
| Titles, family-units yardstick | +0.029 (-0.034 to +0.096) | +0.066 | +0.150 (-0.193 to +0.499) |
| Seasons started | +0.158 (+0.020 to +0.328) | +0.359 (+0.040 to +0.738) | +0.82 |
| AV | +0.92 (+0.10 to +1.97) | +2.09 (+0.25 to +4.46) | +4.8 |
| Second-contract pay (% of cap) | +0.086 (-0.137 to +0.361) | +0.194 (-0.292 to +0.822) | +0.44 (-0.71 to +1.76) |

Rounds 2-4, recommendation minus the team's pick where they disagree: titles (single) production +0.43 (+0.10 to +0.81),
hybrid +0.44 (-0.02 to +0.92); starts -0.49 -> -0.19; AV -2.7 -> -0.6; pay +0.37 -> +0.43.

- Titles tie (signs depend on the yardstick's own pricing); starts and AV favor the hybrid significantly; pay, the
  most position-neutral outside yardstick, leans its way but within noise. Starts and AV structurally credit OL
  (five starters every snap; AV shares team offense to linemen), so part of that gain is the yardstick.
- The hybrid is its own extreme: no EDGE, DL, LB or TE in rounds 2-4. Fresh classes (2023-25) not checked (no
  family-units review for them yet). Not adopted (Nov 17: see the follow-ups below; current rule kept).

## Hybrid pricing follow-ups: softer versions and the 2023-25 classes (`scripts/hybrid_variants.py`, Nov 17 2026)

The hybrid above swung rounds 2-4 from WR-heavy to OL-heavy with no EDGE / DL / LB / TE. Two checks, from a rerun of the
review loop under both pricings that saves every available player's value at every pick 2014-25 (the single-pricing
top choice matches the stored production recommendation at 100% of picks). Rounds 2-4 only; round 1 and 5-7 unchanged.

| Rule (rounds 2-4) | Picks changed | EDGE / DL / LB / TE recs | Titles single | Titles family | Starts | AV | Pay |
|---|---|---|---|---|---|---|---|
| production | - | 186 / 38 / 16 / 25 | | | | | |
| hybrid (family rec wherever it differs) | 43% | 0 / 0 / 0 / 0 | -0.029 | +0.029 | **+0.16** | **+0.92** | +0.09 |
| blend (50/50 of the two values) | 29% | 20 / 1 / 0 / 0 | +0.006 | +0.039 | +0.12 | +0.83 | +0.12 |
| near80 (family rec if production values it within 20%) | 30% | 22 / 1 / 0 / 2 | -0.003 | +0.030 | +0.11 | +0.75 | +0.12 |
| near90 (within 10%) | 20% | 71 / 28 / 3 / 8 | **-0.020** | -0.002 | +0.04 | +0.30 | +0.05 |

Paired rule minus production over all picks 2014-22; bold = interval excludes 0 (resampling draft years).

- **Every rule that keeps the gains drops EDGE and DL.** Family pricing values EDGE / IDL far below production, so any
  mix of the two leaves almost none in rounds 2-4. The one version that keeps some (near90) loses on titles under
  production's yardstick and gains nothing elsewhere. The starts / AV gains are the OL swap itself, and both yardsticks
  favor OL by construction.
- **2023-25 classes** (starts and AV to date, 1-3 seasons): differences are tiny (hybrid starts -0.007, AV +0.03 per
  pick; blend AV +0.11). No reversal and no confirmation; too early.

Verdict: not adopted. No version improves on titles; the outside-yardstick gains come with a rounds 2-4 board with
almost no front-seven picks. Recheck with the 2023-24 classes at four seasons.

## Audit fixes: off-board values, OL snaps, QB-need wording (Nov 17 2026)

Three items from a full read of the project. Value tables (both pricings), the review 2014-26 and the simulator
artifact (v34) were rebuilt. Pre-fix tables and review outputs: `data/derived/before_nov17_fixes/`.

**1. OL snaps missing in 2020 and 2025** (`scripts/strict_follower.py`, `SNAP_POS`, `season_shares`). The snap file
labels most linemen plain "OL" in those two seasons (946 and 1,457 player-games, 186 players), plus a few combined
labels ("G/T", "DE/L", ...), and the position map dropped them. Median OL room: 3.92 starters in 2020 and 4.26 in
2025, vs 5.01-5.05 in every other season. Fix: combined labels mapped; a plain "OL" takes his PFF position, else his
draft position, else guard. Now 5.02 and 5.04. Affects the simulator's room check (OL incumbents' 2020 and 2025
snaps feed the 2021 and 2026 projections) and every script built on `season_shares`. Those scripts' recorded numbers
(strict_follower, cross_year_need, room_crowding, room_persistence, quality_need, position_rooms) predate the fix and
were not rerun; the room-check coefficients are refit each time the artifact is built, so the page is current.

**2. QB need when the veteran's deal has already ended** (`review.p_need`). No number changes. The docstring said
"expiring" meant one season left; the code also counts a veteran whose latest deal in the data ended before the
season (seasons left < 0, 12 team-years). Checked whether those belong with "contract not in data" (0.52-0.54)
instead: no. Their team changed its main QB the next season in 10 of 12, vs 20 of 32 with a known expiring deal
(pooled 30 of 44 = 0.68, the adopted rate) and 18 of 46 under contract (0.39). Kept as is; docstring and
`scripts/qb_need_contract.py` now say so.

**3. Talent clipped at rank 262** (`trade.py`, `draftday.py`). The engine values players out to rank 330 (off the
board) and gives off-board drafted players their own value (see "Board-less draft classes and monotone value":
they did better than board players ranked 251-300, 0.93x of actual). But `trade.pick_effect` and
`DraftEngine._sample` clipped talent to 262 before calling it, so every rank past 262, on or off the board, was
valued as rank 262. Fix: clip removed (the engine caps at 330; the v1 engine keeps its 262 cap). Effect on the
review, 3,334 picks 2014-26:

- Model value of the team's own pick changed at 697 picks, all late-ranked or off-board players (mean +0.004 titles
  %, largest +0.36: off-board centers and guards drafted in rounds 6-7, about 0.3 -> 0.6).
- Recommendations changed at 6 picks, all in round 7 (one in 2014-22: 2017 #245). No off-board player is a top
  recommendation in any draft, as before.
- Realized values unchanged, so the scorecard (round 1 and rounds 2-4) is unchanged; mean modeled gap in rounds 5-7
  0.166 -> 0.156.
- Trades: 25 of 802 trade sides moved (largest 0.29), 2 changed sign; the model's choice changed in 5 of 860
  trade decisions (one from "don't trade" to "same trade": CLE 2021 #113).

Also fixed in the same pass (no reruns needed): `public_board` athletic nudge had the wrong sign; `draftday.
trade_down_scan` moved the wrong pick when the team's next pick wasn't the current one; dead code and stale
docstrings / captions.

## Team-level room test: thin vs full rooms (`scripts/room_fullness_value.py`, Nov 17 2026)

Question: at equal pick quality, does a pick improve its team's position group more when the room is thin than when
it is full? "Net gain of a pick" asked this about the quality of incumbents (no effect); this asks about the quantity
of returning snaps. Same units, outcome (team room value on the field, % of cap), treatment E (expected own value of
the team's round 1-3 picks at the group) and controls. Returning snaps = draft-day incumbents' starter-equivalents
(last season's snaps by players still on the team plus veterans added before the draft) over the group's mean
(1 = typical; sd 0.29). 3,072 team x season x group rooms 2014-25, 1,058 received a round 1-3 rookie. Teams do draft
for need: 44% of thin-third rooms got a round 1-3 pick, 32% of typical, 27% of full.

Room gain per 1.0 of E, by third of returning snaps within the group (thin 2.2 starter-equivalents returning on
average, typical 3.1, full 4.1):

| Horizon | Thin | Typical | Full | Thin - full | E x returning (per 1.0 = a whole typical room) |
|---|---|---|---|---|---|
| Season 1 | 0.05 | -0.03 | 0.14 | -0.09 (-0.41 to +0.23) | +0.18 (-0.35 to +0.71) |
| Seasons 1-2 | 0.12 | 0.13 | 0.14 | -0.03 (-0.29 to +0.24) | +0.10 (-0.38 to +0.59) |
| Seasons 1-4 | 0.17 | 0.18 | 0.19 | -0.02 (-0.25 to +0.21) | +0.15 (-0.22 to +0.53) |

- No difference: a pick into a thin room adds the same to the room as a pick into a full one (0.17 vs 0.19 over four
  seasons). Fallers (taken mostly because they were there): thin - full +0.02 (-0.43 to +0.47) over four seasons, -0.04
  over two. No group's interaction is significant (seasons 1-2).
- The rookies' own realized value per 1.0 of E: thin 0.95, typical 1.00, full 1.04 over four seasons (thin - full
  -0.08, -0.27 to +0.10); season 1 thin 1.03 vs full 0.98 (+0.05, -0.13 to +0.23). A full room doesn't cut what the
  rookie himself plays and earns either.
- Power: the thin - full interval is about +-0.25 on a gain of ~0.18, so thin rooms gaining twice as much can't be
  ruled out; but the estimate is zero or the wrong sign at every horizon and in both splits.

Reading: consistent with the room check (veterans absorb most of any gap, in either direction) and with the earlier
nulls (roster need, quality need, net gain). No support for a need term from the team side. Not adopted. The room
matters only as a cap across several picks at one position ("Strict follower").

## Room column: rookie-deal players vs the room (simulator, Nov 18 2026)

The tests agree on one practical rule: take the best player, and use the room only as a cap against stacking a
position ("Team-level room test", "Strict follower", "Cross-year need term"). The Room column (v33: room filled now,
his season-1 share, effective total after veterans give back) didn't show that: veterans give back 91-98% of a WR,
OL or QB pick's snaps, so a second or third pick at the positions a follower overfills barely moved it; it counted
veterans, who yield; and it covered season 1 only.

v35: the column is "Young / room", the quantity in the room rule (`scripts/cross_year_need.py`):

- Load = expected snap share, in starters, of the roster's players still on rookie deals at the candidate's position
  (drafted in the last 3 drafts, round 1: 4; on the draft-day roster per the room check's incumbents, whoever
  drafted them) plus the picks made so far in the simulation. Each by consensus rank and career season
  (`strict_follower.expected_curve`; classes before 2014 have no board, so pick number stands in for rank).
- Room = league median total snap share at the position, 2014-25 (T 2.03, G 1.99, C 1.00 separately; WR 2.56,
  QB 1.00, CB 2.67, ED 1.84, ...).
- Shown as "1.2 → 1.8 of 2.6": load before and after taking him, in the fullest of his first four seasons. Red when
  part of his expected snaps over the four seasons doesn't fit (the rule's m < 1), gold from 85% of the room; hover
  for the players and the share that fits.

The ranking ignores it (the room rule wasn't adopted). On draft day the load alone is over the room in 63 of 4,992
team x year x position rooms 2014-26 (22 of them C: center draftees often play guard) and over 85% in 157, so red
mostly appears when the user stacks a position: NYG 2026 at WR reads 1.2 → 1.8 of 2.6 at the first pick and
2.7 → 3.1 (49% of his snaps fit) for a fourth WR after taking three. The open-snaps room check stays in the note
under the tiles (`export_simulator_artifact.room_info` exports both).

v36: the green bar on the top player at each "open" position is removed. It came from the open-snaps check, which
the tests say doesn't help a single pick, and it could sit on a row the stacking check marks red. Rows are now marked
only for the team's real pick (gold).

## Trade log corrections (Nov 18 2026)

The simulator showed the Rams' 2025 trade with the Colts as "we give nothing, we get #127, #127, #190". The
public trade log (`data/raw/trades.parquet`) has errors in 2022-25: when a team sent two picks the same way, the
first often carries the second's number (23 trades, 21 on draft night). `trade.trade_log()` now corrects them and
is the loader for the review, the real-trade scoring and the two future-pick scripts:

- 17 trades: the row names the player later drafted with the pick, so his real pick number replaces the listed one.
- 5 trades with no names (ids 1649, 1656, 1660, 1759, 1773): fixed by hand.
- 2025 Rams-Colts (1938): the Rams gave #127 and #190 for #117; the log had all three going to the Rams.
- 2025 Eagles-Chiefs (1922): the Eagles gave #32 and #164 for #31; the log had only #31.
- Left as is: a 2024 Eagles-Steelers trade listing two 2025 seventh-rounders at one number, six one-sided 2005 trades.

`review.draft_night` also handles a pick that went round in a loop on draft night (A -> B -> A -> C): the owner at
the start is the only giver on the first day it moved (2023 #230).

Effect: 20 picks had the wrong owner at the start of the draft (2022: 7, 2024: 3, 2025: 10). Pick recommendations
and values are unchanged. Of 802 trade sides, 38 verdicts changed (mean absolute change 0.04% of a title, 3 changed
sign) and 4 were added (the two repaired 2025 trades); the total over all sides went from +52.6 to +50.4. The market
chart refit on 593 trades instead of 591: lam 0.081 -> 0.078, beta 0.747 -> 0.756; the discount on picks two years
out moved 0.02 -> 0.10 (five trades, an extrapolation as before). The simulator test now reproduces 410 of 416 real
drafts (398 before). Artifact version 37. Old outputs in `data/derived/before_tradelog_fix/`.

Reruns with the corrected log (same day):
- `future_pick_test.py` (full): market chart exp(-0.078 (k-1)^0.756), premium x1.10, next 1st 0.48 (0.39-0.54),
  rounds 2-7 0.63 (0.56-0.74), two years out 0.10 (0.02-0.27). A middle team's next 1st still trades like #43
  (#37-#53), next 2nd #68, next 3rd #103. The model table is identical (it doesn't use the log).
- `trade.py --real-only`: 396 trades (was 394; the two repaired 2025 trades). The 42 sides of corrected trades
  moved 0.04% of a title on average (max 0.21; two changed up/down role); the other 746 moved 0.004 (simulation
  and Nov 17 model changes). Non-QB trade-ups: up -0.36 -> -0.37, down +0.45 -> +0.45.
- `future_pick_realized.py`, both modes: every printed number unchanged (its trades are 2006-21; the errors are
  2022-25).

## Titles non-increasing in rank (`draftday.TITLE_MONOTONE`, Nov 18 2026)

Drafting as Miami in 2020, the simulator ranked Justin Herbert (consensus #15) above Tua Tagovailoa (#5) at pick 5.
At a fixed pick the contract is the same, so a worse-ranked player should never be worth more. The cause
(`scripts/qb_rank_bump.py`; a team starting at -5 with a QB need, QB taken at #5, % of a title):

| QB rank | 5 | 10 | 15 | 20 | 25 | 32 |
|---|---|---|---|---|---|---|
| All classes | 5.6 | 5.4 | 6.0 | 6.4 | 4.7 | 3.2 |
| Without 2020 | 4.7 | 4.6 | 5.8 | 5.7 | 5.1 | 3.3 |

- Not Herbert's own career: the bump is larger without the 2020 class (the review and simulator already leave
  each draft's class out).
- Not the mean: the engine already caps mean 4-season value to be non-increasing in rank (0.29 of cap from rank 5
  to 22). It is the spread: comparison careers at rank 5 are 32% big hits (over 0.40 of cap) and 10% near zero; at
  ranks 15-20, 37-45% and about 21%. Titles are convex in QB value, so the same mean with more spread scores higher.
  Since 2014, 8 of 20 QBs ranked 11-30 returned over 0.47 of cap in four seasons, against 2 of 19 ranked 1-10.

Fix: `TITLE_MONOTONE = "min"`: each value is capped at the lowest value of any better rank at the same pick (board
ranks only; off-board keeps its own value, as in the engine). Applied to the review's and simulator's tables when
built or loaded (`review.monotone_titles`; the cached simulator tables stay raw, no rebuild) and to the draft-day
app's `value_matrix`. Along pick = rank the cheaper contract still raises value (rank-15 QB at #15 above rank-5 QB
at #5). Not applied: a named player in a trade valuation (`trade.package_points`). Largest cut in any table cell:
QB 5.8% of a title, other positions 0.1-0.4.

Test (`scripts/title_monotone_compare.py`; full review 2014-26, off vs min):

| Scorecard, model minus team, each rec once | off | min |
|---|---|---|
| R1 | +0.80 (-0.21 to +1.98) n=49 | +0.98 (-0.09 to +2.13) n=45 |
| R2-4 | +0.43 (+0.02 to +0.97) n=93 | +0.44 (-0.02 to +1.03) n=86 |
| R5-7 | -0.09 (-0.25 to +0.07) n=59 | -0.04 (-0.19 to +0.10) n=55 |
| All | +0.37 (+0.03 to +0.73) n=201 | +0.43 (+0.07 to +0.80) n=186 |

- 268 of 3,334 recommendations changed (R1 23; most in rounds 4 and 6, small non-QB bumps). Where both outcomes
  are known, new minus old realized +0.48 (+0.08 to +0.92; 73 distinct swaps).
- QB recs 391 -> 387. The model no longer takes Bortles / Manziel (#14-15) over Bridgewater (#6) in 2014, Lynch over
  Wentz in 2016, Lance over Fields in 2021, Sanders over Ward in 2025; it loses Herbert over Burrow / Tua in 2020
  (-3.6, -4.7), the one case where the worse-ranked QB was right.
- Adopted. Review outputs before / after in `data/derived/before_title_monotone/` and `title_monotone_min/`.
  Artifact version 38.
- Tie-break (Nov 19 2026): the cap holds players exactly level (Manziel #14 capped at Bridgewater's value, with
  Bridgewater shaded to a #9 talent at pick 9), and interpolating the table at different points left ~1e-15 noise
  that could still put the worse-ranked player first: the simulator showed Manziel over Bridgewater at #9 for
  MIN 2014 once the BUF/CLE trade was OK'd. Values equal to 12 decimals now go to the better consensus rank
  (`draftday.tiebreak`, used by the review, the draft-day app and the simulator page). 7 of 3,334 review recs
  changed, all exact ties (e.g. Sutton #31 over Kirk #44 at MIN #30 / IND #36 in 2018); scorecard all +0.43 ->
  +0.43, R1 +0.98 -> +0.96 only because Sutton and Kirk were two recommended players and are now one (once per
  draft). Review outputs before in `data/derived/before_tiebreak/`. Artifact version 39.

## Class-size test: late picks in big draft classes (`scripts/class_size_test.py`, Nov 19 2026)

Question: the model values each pick on its own, but a team keeps only so many rookies (90 in camp, 53 on the
roster, about 8-10 rookies kept). Does a late pick return less when the team makes more picks in that draft?
Team-drafts 2014-22 had 3 to 15 picks (60 of 288 with 10 or more; class size vs prior strength corr -0.15).

| Rounds 5-7, by class size | <=7 picks | 8-9 | 10+ |
|---|---|---|---|
| Model's predicted titles per pick (% of one) | 0.25 | 0.24 | 0.22 |
| Realized | 0.26 | 0.33 | 0.21 |
| Realized / predicted | 1.08 | 1.38 | 0.97 |
| Played as a rookie | 73% | 71% | 67% |
| Seasons 1-4 value vs slot (% of cap) | +0.06 | +0.11 | -0.21 |

- Realized minus predicted titles per extra pick in the class: -0.002 (-0.018 to +0.016; 947 picks), +0.002 with
  prior strength controlled; per extra round 5-7 pick +0.009 (-0.017 to +0.045). Rounds 4-7: -0.003 (-0.019 to +0.013).
- Played as a rookie, vs slot: -1.7 points per extra pick (-3.3 to 0.0; rounds 4-7 -1.5, -2.7 to -0.2). Seasons 1-4
  value vs slot: -0.02% of cap per extra pick (-0.13 to +0.09), not different from zero.
- Whole class per team-draft: predicted 3.6 / 3.9 / 4.7 titles (% of one) for <=7 / 8-9 / 10+ picks, realized
  4.6 / 5.3 / 5.5: bigger classes still delivered more in total.
- Long history, drafts 2000-2021 (2,616 round 5-7 picks): career AV vs slot per extra pick +0.02 (-0.22 to +0.28);
  ever played a game -0.2 points (-0.9 to +0.5). By his order among the team's round 5-7 picks, AV vs slot
  +0.1 / -0.1 / -0.3 / +0.3 / +0.2 for the 1st to 5th+: no decline.

Late picks in big classes are a little less likely to play as rookies, but their four-season value, titles and
career AV are not lower. Not adopted: no class-size term. Limit: real classes rarely go past 12 picks (2 of 288),
so this says nothing about a follower who trades down into 13 or more.

## Revealed team opinion: trade-ups and undrafted guarantees (`scripts/conviction_test.py`, Nov 20 2026)

Two ways teams show what they think of a player with money or picks rather than words (the press-conference
route had too few prospect mentions: ledger `presser_feasibility`).

**A. Draft-night trade-ups.** Does a player a team traded up for beat his draft slot? Drafted players 2006-2022,
value seasons 1-4 vs an out-of-sample slot model (leave-one-class-out 2006-15, rolling 2016-22; each class
rescaled to actual). Trade-up = the team got the pick on draft night and gave no earlier pick of that draft in the
deal (481 picks; 92 in round 1, 36 QBs).

| Test | n | Beat slot | Value vs other picks |
|---|---|---|---|
| A1 all trade-ups | 481 | 39% vs 35% (p 0.13) | x1.01 (0.91-1.12) |
| A2 round 1 | 92 | 45% vs 38% | x1.09 (0.88-1.32) |
| A3 picks 33+ | 389 | 37% vs 34% | x0.96 (0.85-1.08) |
| A4 non-QB | 445 | 39% vs 35% | x1.00 (0.90-1.10) |
| A5 per sd of overpay (chart value paid / got) | 436 | | x0.97 (0.85-1.10) |
| A6 2018-22 vs consensus board | 164 | 46% vs 41% (odds x1.52, p 0.02) | x0.97 (0.81-1.16) |

A trade-up says nothing about value beyond the slot it buys, and paying more doesn't mean a better player. The
slightly higher beat rates come with no more total value (one A6 p 0.02 among nine comparisons). Not adopted;
the trade calculator keeps valuing the pick, not the team's conviction.

**B. Undrafted free-agent guarantees.** Do undrafted rookies whose teams guaranteed more money do better?
OverTheCap contracts with no draft round signed in the draft year, matched by name + class to the prospect
universe: 1,860 signers 2018-2022 (350 on the consensus board). 84% get some guarantee; median $12k, 90th
percentile $85k. Value seasons 1-4, 0 if he never played.

| Test | n | Value per sd of log guarantee |
|---|---|---|
| B1 controls: position, class, on board, log board rank | 1,860 | x1.77 (1.54-2.05) |
| B3 top quartile vs rest (B1 controls) | 1,860 | x2.22 (1.61-3.05) |
| B2 board signers vs consensus-board model | 350 | x1.08 (0.84-1.38) |
| B4 vs the public pre-draft model (board + stats + combine + recruiting), offset | 1,860 | x1.25 (1.08-1.44) |
| same, public model as a free covariate (+ board rank) | 1,860 | x1.51 (1.31-1.75) |
| played at all (odds), same controls | 1,860 | x1.47 |
| value given he played | 681 | x1.23 (p 0.001) |

By class (free-slope version): 2018 x0.87, 2019-2022 x1.49-1.61. Teams know a good deal about the undrafted pool
that public data doesn't: the guarantee ranks those players better than board rank, college stats and combine
together. Within the 350 board players the board model's own comparison (B2) is null, so most of the edge is in
the ~80% of signers the board doesn't rank. Upper bound: guaranteed money (and the roster promise it stands for)
can buy playing time; the money itself is small, and the effect holds for value among players who did play.

Use: guarantees are signed after the draft, so they can't feed a draft-day choice. What they show is that teams'
late-round information is real and goes beyond public data, consistent with the model's lack of an edge in
rounds 5-7. Not adopted as an input; a possible later use is pricing undrafted signings, or checking whether any
pre-draft proxy (visits, workouts) picks up the same interest for players off the board.

### Follow-up: do pre-draft visits pick up the same interest? (`scripts/visits_udfa_test.py`, Nov 20 2026)

Off-board undrafted signers 2018-2022 (1,510), matched to reported pre-draft meetings (top-30 visits, combine /
pro-day meetings, private workouts). Only 19% have any reported meeting (2021: 1%, COVID); the count leaves out
the team that signed him, since a meeting with the signer can be reported after the draft.

| Test (controls: position, class, public pre-draft model) | n | Effect |
|---|---|---|
| V1 guarantee ~ log1p(teams met) | 1,510 | +0.65 log points (+0.51 to +0.79); median $5k / $8k / $13k for 0 / 1 / 2+ teams |
| V2 value ~ log1p(teams met) | 1,510 | x1.03 (0.79-1.35) |
| V3 guarantee per sd, given teams met | 1,510 | x1.48 (1.24-1.76); teams met x0.89 |
| V4 value ~ top-30 private visits (log1p) | 1,510 | x1.88 (1.32-2.68) |
| V4 value ~ private workouts | 1,510 | x0.70 (0.34-1.43) |

Visits track the guarantee market (teams that show interest before the draft bid after it), but the count of
teams met says nothing about value: the information in the guarantee isn't in the visit count. One sub-result
stands out: an off-board player brought in for a top-30 visit by a team other than his signer (94 players) did
x2.4 better than the public model expects (x2.1 given his guarantee), in 3 of 4 classes (2018 x3.8 / 1.2,
2019 x1.4 / 0.6, 2020 x0.25 / 0.50, 2022 x2.1 / 0.8; actual / public, visited vs not), with 30% of the visited
group's value from three players (Charvarius Ward, Tony Adams, Jaylen Warren). It's one of several sub-tests and
the workout count points the other way, so it's a lead, not a finding. Fresh check: classes 2016-17 and 2023.
Not adopted.

**Fresh replication** (`scripts/visits_udfa_replication.py`, Nov 20 2026): classes 2016, 2017 (4 seasons) and 2023
(3 seasons), never in the original test; same definitions, except the public model is fit leave-one-class-out on
2014-2022 (board features start in 2014, so the rolling fit can't reach 2016-17). 894 off-board signers, 61 with a
top-30 visit by another team.

| Test | Replication | Original 2018-22 |
|---|---|---|
| R1 top-30 visit flag (primary) | x1.83 (1.07-3.13), one-sided p 0.013 | x2.43 |
| R2 same, given the guarantee | x1.26 (0.77-2.06) | x2.13 |
| R3 per log1p(top-30 visits) | x1.64 (1.01-2.66) | x1.88 |
| R4 by class | 2016 x2.95, 2017 x1.47, 2023 x1.25 | 3 of 4 classes |
| R5 teams met, any meeting | x0.87 (0.59-1.29) | x1.03 |

The primary test replicated: off-board undrafted players brought in for a top-30 visit by another team did about
x1.8 better than the public model expected, all three classes in the same direction. Smaller than the original
and concentrated (Bourne, J. Wilson, E. Wilson: 47% of the visited group's value). Once the guarantee is known
most of it is in the guarantee, but the visit is known before the draft and the guarantee isn't. Any meeting,
counted by teams, still says nothing: it's specifically the top-30 invitation. Not yet used: the model's
candidates are the board plus drafted players, so using it needs off-board visited players in the late-round pool
and a choice test against the model's round 6-7 picks.

**Late-round choice test** (`scripts/visits_late_choice.py`, Nov 20 2026): would taking an off-board player with a
top-30 visit (by a team other than the one that got him) beat the model's round 5-7 pick? 2016-2022 (outcomes
complete), 801 picks; 222 such players in the pool (5 in 2021), a median of 30 still available at each pick.
Realized titles at the pick's cost for the team on the clock (review.RealizedV2, the review's yardstick).

| Rule (realized, % of a title) | Rule | Model's pick | Team's pick | Rule minus model |
|---|---|---|---|---|
| Visited, random among those available (801 picks) | 0.073 | 0.281 | 0.276 | -0.198 (-0.236 to -0.160) |
| Visited, best public-model expectation (once per draft) | 0.191 | 0.276 | | -0.085 (-0.291 to +0.131), n=12 |
| Placebo: unvisited, best public model (once per draft) | 0.102 | 0.145 | | -0.043 (-0.229 to +0.104), n=10 |

No. Visited off-board players beat what the public model expects of them, but from a very low base: on average
they deliver about a quarter of what a round 5-7 pick does, the model's or the team's (minus the team's pick
-0.20, -0.29 to -0.14). By year they beat the model's pick in 2016-18 and 2020, when its late picks happened to
miss, and lose badly in 2021-22. The "best by public model" rules collapse to about one player per draft (the
same top undrafted player stays available all draft), too few to read. The visit marks undrafted players worth
signing, not players worth a pick. Not adopted.

## Free agency and trades, step 1: a veteran on a contract, in titles (`src/veteran.py`, `scripts/vet_value.py`, Nov 20 2026)

Goal: value one veteran on one contract on the draft model's yardstick, so free-agent signings, player trades and
picks can be compared in the same currency (step 2 = a Free agency tab; step 3 = players in the trade calculator).

**Season value** (`scripts/vet_season_values.py`): every player-season 2013-25 priced like the draft label's seasons
(snap share x veteran pay curve(share, darko quality) x games fraction, /1.10 to this project's units). Same harness
and pay curves as `scripts/darko_label.py`; 24.9k player-seasons.

**Currency check** (`scripts/vet_currency_check.py`): team rating ~ curve value, season means removed (416 team-seasons).
Veterans and rookie-deal players buy the same strength per 1% of curve value: non-QB 0.249 vs 0.259 (equal p 0.70),
QB 0.51 vs 0.58. So a veteran's seasons can be credited exactly like a pick's. (Absolute slopes, 0.26 non-QB / 0.51 QB,
are inflated by same-season quality; the model keeps the bridge's 0.19 / 0.34, as for picks.)

**Forecast** (`veteran.Comps`): the 60 nearest past "anchors" in the same pricing group by age, value last season,
value the season before and games played last season; each comp's next five seasons (zeros count: cut, retired,
hurt), shifted by a per-group linear slope times the feature gap. Out of sample (other players, other seasons, 4000
anchors): actual / forecast 1.00 in each of seasons 1-5 (plain nearest neighbours ran 6-15% low; fixed by the shift);
RMSE beats carrying last season forward at every horizon (1.12 vs 1.27 in season 1, 1.34 vs 2.15 in season 5, % of
cap); by group 0.97-1.07 in season 1. Weak spot: age 31+ over-forecast by ~10-20% from season 3 (34+: 0.81, n 180).
Aging, players worth >= 2% of cap: value kept after 3 seasons 97% (<25), 72% (25-27), 48% (28-30), 35% (31-33).

**Contract** (`veteran.team_seasons`): flat APY, guarantees paid first; before each later season the team cuts him if
last season's value < theta x the non-guaranteed part of next season's pay, then charges the guarantees left as dead
money. theta = 0.5 fit to UFA signees' retention (still on team in contract season 2-5: actual 0.64 / 0.44 / 0.35 /
0.25, modeled 0.62 / 0.38 / 0.27 / 0.21). Extensions stay longer than the rule says (season 3: 0.74 actual vs 0.49).
Points per season as `pick_value.pick_effect`: rate x (value - share x vet min) - COST_RATE (0.10) x (cap charge - vet min).

**Past signings** (OTC contract_history 'UFA', 2015-23, APY >= 1% of cap, finished by 2025: 1510): points over the
contract, predicted (comps, leave-player-out) vs realized (value with the signing team, APY while there, unearned
guarantees as dead money): **+0.149 vs +0.137, realized - predicted -0.012 (-0.049 to +0.025)**; per-signing corr 0.54;
by predicted tercile -0.20 / +0.06 / +0.59 predicted vs -0.13 / +0.10 / +0.44 realized (ranks right, a little
overconfident). Contracts >= 5% of cap: predicted +0.26, realized +0.03 (diff -0.23, -0.42 to -0.06; non-QB -0.18,
-0.30 to -0.06): the value forecast holds for them (~5% high), the gap is mostly cost timing (more leave in year 1,
fewer cut in years 3-4 than the rule; the realized cost is a proxy, real deals are backloaded) - read big-contract
verdicts as ~0.2 points optimistic. QBs: +0.86 predicted vs +0.61 (n 107, ns). Extensions (488): +0.74 vs +0.68;
franchise tags (76): +0.40 vs +0.32.

**Example contracts**, titles added over 5 seasons (% of one Super Bowl; contender with veteran QB / rebuilding),
with the model's cap charge 0.10 and, for comparison, the bridge's non-QB rate 0.19:

| player | APY (% cap) x yrs | @0.10 contender | @0.10 rebuild | @0.19 contender | @0.19 rebuild |
|---|---|---|---|---|---|
| EDGE 27, star (6% value) | 10 x 4 | 0.0 | +0.1 | -4.5 | -0.6 |
| EDGE 27, star | 7 x 4 | +1.7 | +0.3 | -1.9 | -0.3 |
| WR 30, good (4%) | 6 x 3 | -0.8 | -0.1 | -2.8 | -0.3 |
| OL 26, starter (3%) | 4 x 4 | +0.4 | +0.1 | -1.4 | -0.2 |
| QB 29, starter (12%) | 17 x 4 | +11.5 | +2.6 | +1.6 | +0.6 |

**The cap charge decides the sign of most non-QB free-agent verdicts.** At 0.10 a fairly priced veteran adds a little
(value ~0.6 of APY x 0.19 > 0.10); at 0.19 nearly every market deal subtracts. The realized check above can't settle
it (prediction and realization share the charge). Earlier evidence points both ways: teams turn QB savings into only
~a quarter of their value (`scripts/qb_savings.py`, the basis of 0.10), but money spent on veterans buys ~0.2 points
per 1% (`scripts/fa_vs_extension.py`, with franchise effects). For choosing between free agents, or between a free
agent and a pick, at the same cap cost, the charge cancels; for "sign or keep the space", it doesn't. Not adopted
anywhere yet (no app change); `veteran.evaluate(situation, player, years, apy, gtd)` is the entry point for step 2.

### Follow-up: a cap price that depends on season and situation, applied to picks (`scripts/cap_shadow_picks.py`, Nov 20 2026)

Idea (user): manage the cap to maximize titles. Then a cap dollar costs what its best use buys, and unused cap rolls
over. Measured best use: market free agency buys ~0.12 points per 1% of cap spent (UFA 2015-23, non-QB, value with
the team / actual cost x 0.19; 0.16 for 1-2% deals, 0.13 for 2-5%, 0.10 for 5%+). Titles per point by season
(contender vet QB 2.78 / 1.83 / 1.39 / 1.15 / 1.04; middle 0.52 -> 0.85; rebuilding 0.06 / 0.32 / 0.52 / 0.64 / 0.74,
% of one SB). Shadow charge for 1% of cap in season t = 0.12 x the best titles-per-point from t on (within 5 seasons).
Value side unchanged; ranking of players at one slot unchanged (same rookie cost). Typical non-QB pick, titles %
old -> new: contenders -0.1 to -0.3 at the top, ~0 after #32; middle #1 1.33 -> 1.06; **rebuilding #1 0.84 -> 0.20,
#5 0.80 -> 0.23, #10 0.95 -> 0.62, #48 0.51 -> 0.45, #100+ unchanged**. QB picks move < 0.4. Rebuilding #5 vs #20 + #48:
old 0.80 vs 1.25, new 0.23 vs 0.98. Caveats: 5-season window drops seasons 6+ (value of top picks' second contracts
and the rebuild's good years); assumes rolled-over cap is later spent at 0.12, while `scripts/qb_savings.py` found real
teams redeploy QB savings at ~0.05 (the reason for 0.10). Measurement only; not adopted, no app change.

**7-season window** (`scripts/cap_shadow_picks.py 7`): seasons 6-7 (and season 5 for picks after round 1) added from
what drafted players really gave the drafting team then (`second_contract_seasons.parquet`, expected by pick), baseline
paths and title curve run 7 seasons. It doesn't rescue top picks for rebuilders: by season 6-7 a kept top pick is on a
market-rate extension, so it adds little. Typical non-QB, old -> new: rebuilding #1 0.90 -> 0.01, #5 0.82 -> 0.04,
#10 0.94 -> 0.45, #48 0.54 -> 0.41; contender vet QB #5 2.55 -> 2.18; middle #5 1.31 -> 0.98. Rebuilding #5 vs #20 + #48:
0.82 vs 1.32 old, 0.04 vs 0.86 new.

**Does unspent cap turn into strength later?** (`scripts/cap_rollover_test.py`; the evidence that decides the price -
scoring real trades can't, since a slot's rookie cost is fixed in advance and shifts prediction and realization alike.)
Team-seasons 2014-22 (288): unused cap = 100% - cap used by the team's players (OTC season_history; mean 16.5%, sd 8.4).
Rating in t+1..t+3 ~ unused + rating t, t-1 + roster curve value (veteran, rookie-deal) + season FE, clustered by team.
Pre-set reading: summed coefficient ~0.12 per 1% supports the shadow price, ~0.05 the current charge.
- Without franchise FE: +0.23 (+0.01 to +0.45). But the money isn't respent: cap used next season *falls* 0.30% per 1%
  unused (-0.44 to -0.16) - frugal teams stay frugal, so the later gain isn't rolled-over cap at work.
- With franchise FE: **+0.02 (-0.16 to +0.21)**, used next season -0.07 (ns). Same season: -0.09 per 1% (p 0.007).
Read: no sign that a team's unspent cap becomes later strength once franchise is held fixed (the confound found in
`scripts/fa_vs_extension.py` again); too wide to rule out 0.12 or 0. The shadow price describes a team that really
does redeploy the money, not the typical team. **Kept the current charge (0.10) in the draft model; not adopted.**

## Free agency page (step 2; `src/pages/4_Free_agency.py`, Nov 20 2026)

`uv run streamlit run src/app.py`, then "Free agency" in the page list. Same team inputs as the draft room (situation,
QB contract season, QB need). Pick any player with value in 2024-25 (or enter a custom one), set years / APY /
guarantees in $M and the cap, and it shows:
- Super Bowls added over 5 seasons (`veteran.evaluate`), and the highest APY that still adds titles at the same
  guarantee share (`veteran.break_even`).
- Value delivered per $1 of cap charged, against the break-even ratio (0.10 / 0.187 = $0.53 non-QB, $0.30 QB) and past
  free agents at that price ($0.88 / $0.72 / $0.53 for 1-2% / 2-5% / 5%+ of cap).
- Per contract season: chance he's still on the roster, expected value with its 10-90% range, cap charged (with dead
  money), points of team strength.
- A comparison list (same-cost comparisons hold whatever the cap price).
- Scenario switch: rolled-over cap spent well later (`cap_price="rollover"`; not the default, see the rollover test).
Data ends with 2025, so for a 2027 signing the forecast skips 2026 (comps' season 2 = contract season 1; `veteran.H`
is now 6). Warnings: player older than his comps by 1.5+ years (e.g. Aaron Rodgers at 43), QB rate caveat, QB signed by
a team that has its QB. Spot checks (3 yrs, contender break-even): Ja'Marr Chase $50M, Penei Sewell $38M, Myles
Garrett $49M, Brian Branch $23M, Daron Payne $12M. Test: `.venv/bin/python scripts/test_fa_page.py`.
Not modeled: backloaded cap hits, void years, restructures; player trades (step 3).

### Ranked free agents and predicted contracts (`src/fa_market.py`, Nov 20 2026)

The Free agency page opens on **Ranked free agents**: every player out of contract after 2026 (no real 2027 salary in
OTC season_history; void years don't count; `years` isn't used, OTC counts extensions inconsistently): 869 with value
in 2024-25, 201 likely restricted (3 or fewer seasons). Each is priced at his predicted market contract and scored for
the chosen team: titles added, the most he's worth per year to us at the predicted years / guarantee share, and the
gap. Filters: positions, price, restricted, players who cost titles. Selecting a row opens it in "One signing", with
the predicted contract filled in. About 30 s per team setup (cached).

**Contract model** (`scripts/contract_predict.py`): LightGBM on 3,276 UFA signings 2015-26 (APY >= 0.5% of cap); log
APY, years, guaranteed share; features: group, age, value in the last two seasons of data, games, previous contract
APY, previous deal a rookie deal. Trained with the gap it's used with (lag 2: data through 2025, signing 2027).
Leave-one-year-out:

| | R2 log APY | median miss | within 25% |
|---|---|---|---|
| lag 1 (normal March) | 0.60 | 31% | 40% |
| lag 2 (a year ahead) | 0.44 | 38% | 30% |
| previous contract only | 0.36 | 39% | 28% |
| value only (lag 2) | 0.35 | 46% | 25% |

exp(predicted log) ran 13-19% under the mean; scaled to the mean (`SMEAR`), calibrated in every price quintile.
Years within one 89%; guaranteed share miss 0.21. Stars off rookie deals rarely reach the market (4 UFAs worth 8%+
of cap): they signed ~1.25x the prediction; adding extensions and tags to training didn't fix it (1.27), so they're
flagged ("likely extended/tagged") instead.

**Do the ranked bargains pay off?** (`scripts/fa_bargain_check.py`) Past UFA contracts 2015-23 (1,509), bargain score =
forecast value per season / out-of-sample predicted APY. Realized value with the team per $1 of actual cost by score
tercile: **0.48 / 0.66 / 0.85; high - low +0.37 (+0.30 to +0.44)**, same in 2015-18 (0.51 / 0.68 / 0.85) and 2019-23
(0.44 / 0.65 / 0.85). The market didn't bid the bargains up (actual / predicted APY 1.03 high vs 1.32 low tercile).
Caveat: forecast and realized value use the same pay-curve yardstick, so a persistent quirk of the yardstick (a kind of
player it rates above what wins) would show here as a "bargain" too; the currency check (veterans vs rookies) supports
the yardstick on average, not within it.
Example (KC, contender): Puka Nacua, George Pickens, Moro Ojomo, Ronnie Hickman Jr., Kobie Turner top the list.

### Team needs in free agency (`scripts/fa_room_test.py`, Nov 20 2026)

Draft-test design (`scripts/room_fullness_value.py`) with outside free agents as the treatment: 1,707 outside UFA
signings 2015-25 (no snaps for the team the season before), E = their forecast own value (`veteran.Comps`), rooms =
team x season x group (non-QB), fullness = returning starter-equivalents after departures, before signings (1 =
typical). Controls: fullness, last season's room value, the team's round 1-3 rookies' expected value, group x season
FE; team-clustered. Teams sign for need: 56% of thin rooms got an outside signing, 43% typical, 34% full.

| per 1.0 of E | typical room | x fullness | thin / typical / full | thin - full |
|---|---|---|---|---|
| Room value, season 1 | 0.12 (0.01 to 0.23) | -0.20 (-0.52 to +0.12) | 0.14 / 0.17 / 0.02 | +0.12 (-0.11 to +0.35) |
| Room value, seasons 1-2 | 0.16 (0.06 to 0.27) | -0.06 | 0.17 / 0.21 / 0.10 | +0.07 (-0.16 to +0.30) |
| Signees' own value, season 1 | 0.86 (0.81 to 0.92) | -0.22 (-0.40 to -0.03) | 0.93 / 0.86 / 0.77 | +0.15 (+0.00 to +0.31) |
| Signees' own value, seasons 1-2 | 0.80 (0.75 to 0.85) | -0.18 (-0.34 to -0.02) | 0.84 / 0.81 / 0.70 | +0.15 (-0.00 to +0.29) |

By group (season 1 room value), fullness interaction: OL -2.15 (p < 0.001), LB -0.64 (p 0.003), DB -0.38 (p 0.19),
others ~0. Cap side (OTC cap hits, + last season's group spend): per 1.0 of E the signees cost 0.76 (season 1) / 0.85
(1-2); the group's other players' cap falls 0.49 (-0.61 to -0.37) both horizons; team total +0.18 (ns) / +0.32
(+0.01 to +0.63). Draft picks: room gain 0.18, veteran spend at the group -0.54.
Reading: free agents work like picks - ~15% of their value is net field gain, about half comes back as less spending on
the position's other players - so the model's own-value accounting is as consistent for them as for picks. Need shows
up mainly in the signee's own playing time: a full room cuts it ~15% vs thin (borderline), strongly at OL and LB.
Not yet in the tool.

**Adoption check for a depth factor** (`scripts/fa_need_check.py`). Factor from the room test (signees' own value,
seasons 1-2): value x (1 + slope x (fullness - 0.93)), slope OL -0.57, LB -0.37, other -0.17 per 1.0 of fullness,
clipped 0.6-1.25, fit leaving the signing year out; applied to the bargain check's 1,402 contracts with a known room
(re-signings: fullness excludes the player). Pre-set: default only if the value-per-$ spread and correlation improve and
the full-room bias shrinks. Result - all three worse:
- realized / forecast by fullness: thin 1.07 -> 0.97, typical 1.01 -> 1.00, full **1.15 -> 1.24** (outside signings
  alone: 1.11 / 0.99 / 1.06 without the factor, no fullness pattern; re-signings into full rooms 1.50)
- value per $1, high - low tercile +0.353 -> +0.331; corr(score, value per $) 0.325 -> 0.317
- forecast MSE +0.035e-4 (+0.008e-4 to +0.064e-4) worse
Reading: the room regression's own-value penalty doesn't survive into per-contract forecasts; the comps forecast
already carries it, and teams sign into full rooms when they expect the player to play. **Not adopted.** The page shows
the team's depth by group for 2027 (players under contract, by 2026 snaps per game so far, vs the other teams today;
`fa_market.rooms`; fixed Nov 20 after NE showed 0.00 at EDGE: positions now from the active OTC contract - snap-count
labels file many edge rushers as LB/DL - and contract status from the ACTIVE contract row via `fa_market.under_contract`,
since older rows carry stale season histories (Buckner's 2027 salary only on his active row; this also kept some
signed players on the free-agent list); 2026 rookies without an OTC season history use year signed + years; players
with < 3 games in 2026 keep their 2025 snaps (ATL's Pearce, Walker)) as information and in an "Our depth" column; the factor is an off-by-default "experimental" sidebar
switch; the comparison list warns when two options share a position.

**Rerun with OTC position labels** (Nov 20 2026; `scripts/fa_room_test.py`, `--snap-labels` reproduces the first run).
The first run grouped returning snaps by draft / snap-count labels (edge rushers filed as LB or DL), while room values
and signees used the OTC pricing group. Relabeled: signees' own value thin / typical / full 0.91 / 0.88 / 0.74
(thin - full +0.17, +0.01 to +0.34; seasons 1-2 +0.14, -0.03 to +0.30); room value thin - full +0.13 (ns); by group
OL -2.10 (p < 0.001), LB -1.01 (p < 0.001; stronger, so not a labeling artifact), DB -0.56 (p 0.07); cap side unchanged
(other players' cap -0.46 / -0.47). Adoption check rerun (`scripts/fa_need_check.py`; slopes OL -0.57, LB -0.41, other
-0.29): realized / forecast by fullness 1.12 / 1.04 / 1.07 without the factor, 0.99 / 1.02 / 1.18 with; value per $1
spread +0.353 -> +0.329; MSE +0.030e-4 (-0.006 to +0.065). Same verdict: not adopted.

## Free agency and trades, step 3: players in the trade calculator (`src/vet_trade.py`, Nov 20 2026)

Free agency page, **Trades** view: an offseason trade before 2027 with any mix of players (under contract for 2027,
real remaining contracts; our own listed first) and picks (2027 draft numbers; future picks "2028 R1", with "middle" /
"weak" / "strong" for a partner's). Everything goes onto one 5-season strength path, as `trade.evaluate` does for picks.
Shows the change in titles, each part alone, contract years and cap hits for the team holding each player, and dead
money for players we send away.
- Contract (`vet_trade.contract`): the player's ACTIVE OTC row, season by season from 2027 (seasons with a base salary
  and a cap number; void / placeholder years only carry proration). Acquiring team: cap number minus signing-bonus
  proration, guaranteed salary as dead money if cut (`veteran.team_seasons_sched`, same cut rule). Our player: full cap
  number, cut -> remaining guarantees + proration; trading him -> all remaining proration accelerates into 2027.
  Cap $ -> share: the 2027 cap, growing 7% a year (assumption; 2023-26 averaged ~10%).
- Forecast: `veteran.Comps`, contract seasons = comps' seasons 2-6 (data ends 2025).
- **Lemons check** (`scripts/traded_player_check.py`): 340 veterans traded in the offseason (Feb-Aug) 2014-24, actual /
  forecast value: season 1 any team 0.94 (0.86 to 1.03), with the acquirer 0.93; forecast >= 3% of cap 0.90 (0.80 to
  1.01); age 30+ 0.97; season 2 with the acquirer 0.65 (many leave; the contract / cut rule handles that). Not enough to
  haircut traded players; shown as a caution.
- Examples (cap $320M): contender (vet QB) gets A.J. Brown for #28: +1.6% of a title (Brown +3.7, #28 -2.1). Myles Garrett
  for #28 + own 2028 1st: contender +2.4; rebuilding team selling him for #20 + a contender's 2028 1st: +1.1 (both win:
  a strength point is worth far more to a contender). KC (its 2027 inputs): Garrett for #28 + 2028 R1 roughly even
  (+0.08: his 2029-31 cap hits $33-48M).
- Not modeled: post-June 1 designations, restructures, conditional picks; the partner's verdict (run it with their
  situation). Test: `scripts/test_fa_page.py` part 4.

## Free agency and trades, step 4: offseason planner (`src/offseason.py`, Nov 20 2026)

Free agency page, **Offseason plan** view (a team must be chosen). Composes tested parts; no new model:
- **Budget** = 2027 cap + rollover - commitments - dead money already on the books - rookie pool - filling to 51 -
  reserve (default 3%). Commitments: active OTC contracts with a 2027 row for the team (cap number), plus active
  contracts with no season history yet (mostly 2026 rookies) at APY. Dead money and rollover aren't in the data: inputs.
  Rookie pool: `value.rookie_cost_curve` for the team's picks (projected slots, editable). Fill: 0.35% of cap per open
  spot. Teams over their 2027 commitments (e.g. KC, $66M over at a $299M cap) get a warning: cuts / restructures first.
- **Plan**: free agents from the ranked list (`fa_market.rank`) at predicted contracts with positive titles; greedy by
  titles per $ within the budget; at most 2 at OL / DB / WR / DL, 1 elsewhere (each signing is valued as if he alone
  fills a minimum-salary spot); likely-restricted and likely-extended players skipped by default; QBs only with a QB
  need. Scored together on one strength path (`offseason.joint_titles`; joint > sum for most teams, the title curve is
  convex). Example (MIA, $299M cap, $87M budget): 8 signings, $81M, +3.1% of a title.
- **Draft or free agency by position**: a typical player at our first pick at each position (`trade.package_points`)
  next to the best free agent there (titles at predicted price), planned signings and our depth. MIA at #2: WR 2.35,
  EDGE 1.72, DL 1.20 ... LB -0.13; best free agents WR Pickens 0.71, DB Hickman 0.55, OL Ekwonu 0.47.
- Not validated as a whole (no way to replay counterfactual offseasons); its parts are (forecast, contract model,
  bargain ranking, pick values). Not modeled: re-signing decisions as such, trades, cap moves, June 1 cuts, the market
  reacting, a plan that also reserves the first pick's position. Test: `scripts/test_fa_page.py` part 5.

### CB vs S (and T / G / C) in the free-agency tools (`scripts/cb_s_split_check.py`, Nov 20 2026)

The tools use one DB group (and one OL group), like the draft model's pricing. Checked where a split could matter:
1. **Contract model**: pooled DB under-predicted CBs and over-predicted safeties, both significantly (lag 1: CB 1.13,
   1.06-1.19; S 0.87, 0.81-0.93; lag 2 CB 1.12, S 0.91); OL the same way, smaller (T 1.08-1.11, G 0.95-0.96, C 0.90);
   EDGE / LB / DL fine. **Adopted**: fine position (CB, S, T, G, C; most common OTC contract position) as a feature
   (`fa_market.fine_positions`). R2 0.620 -> 0.626 (lag 1), 0.481 -> 0.490 (lag 2); now CB 1.01 / S 0.99 / T 1.02 /
   G 1.00 / C 0.99 (lag 2: 1.04 / 0.98 / 1.03 / 0.97 / 0.98); scale factors unchanged; bargain check +0.375 (was +0.372).
2. **Forecast** (comps from all DBs): CB 1.00 (0.95-1.05), S 1.06 (1.01-1.11) in season 1; season 3 0.94 / 1.04. Small;
   not split (would halve the comp pools).
3. **Currency**: team strength per 1% of curve value CB 0.70 vs S 0.64, equal p 0.57 - the pooled pay curve is fine.
Depth stays pooled (CB / S rooms trade off, r ~ -0.75, "Room check: are team rooms persistent?"). Draft side unchanged
(CB x1.11 / S x0.89 pay factors failed their choice test, "Pay within OL and DB").
Nov 20 2026: the page's cap input now defaults to a 2027 projection (the 2026 cap from the contract data x 1.07 =
$320M) instead of the 2026 cap ($299M); budgets and the trade view's cap shares use it. NE at $320M: -$16M (was -$35M).

### Offseason workbench: cuts, restructures and signings, toggled together (`offseason.roster_moves`, Nov 20 2026)

The Offseason plan view is now a workbench: three tables (cuts, restructures, signings) with a "Do it" checkbox each,
pre-checked with suggestions; the summary at the top (budget before moves, room freed, signings, money left, Super Bowls
added) updates on every toggle, scoring all checked moves together on one strength path (`offseason.evaluate_moves`).
- **Cut** (before June 1): dead money = remaining guaranteed salary + all remaining proration (void years too), all in
  2027; saved = 2027 cap number - dead; later seasons save his whole cap number; titles = losing his forecast value
  (cut rule, `veteran.team_seasons_sched`) vs taking the dead money now. Suggested when cutting adds >= 0.02% of a
  title on its own (forecast value well below his pay). Each cut opens a roster spot (fill to 51 recomputed).
- **Restructure**: 2027 base salary above the minimum becomes a bonus prorated over his remaining playing seasons
  (2+, no void years added): saves amount x (1 - 1/n) now, adds amount / n each later season and to his dead money if
  cut later. Suggested only while the budget is still negative after suggested cuts (largest savings first, under 32).
  Direction is situational: restructuring Christian Gonzalez (saves $13.6M) is +0.49% of a title for a contender,
  -0.12% for a rebuilding team (cap pushed into the seasons a rebuilder hopes to be good).
- **Signings**: pre-checked = the greedy plan for the budget after the suggested moves; "Re-plan" redoes it for the
  current toggles; "Reset to suggestions" restores everything.
- Examples ($320M cap): NE -$15.6M before moves; suggested cuts Carlton Davis, Milton Williams, Robert Spillane,
  Rhamondre Stevenson free $50.3M net; 6 signings $34.4M; together +2.4% of a title. KC -$47.9M: 1 cut + 3
  restructures free $53.7M. MIA $102M: 1 cut, 8 signings, +2.7%.
- Not validated as a whole (the suggestions follow the tested forecast and cut rule; past offseasons can't be replayed
  with other moves). Not modeled: post-June 1 cuts, void years in restructures, extensions, pay cuts.
- **Placeholder years fixed** (`fa_market.real_season`): OTC marks void / placeholder seasons with a base salary above
  the cap number (99.999999, or 50.0 with a small cap number); they were being read as real seasons (A.J. Brown "2030"
  at $53.5M; Arnold Ebiketie, Julius Brents, Marquise Brown, Davante Adams "under contract" for 2027). Now a real
  season needs base > 0 and cap number >= base: contracts, the free-agent list (869 -> 894), depth and roster counts.

**Default picks from the trade log** (Nov 20 2026; `teams.pick_ownership`, `team_inputs()["picks"]`). The draft room's
"Our picks" box and the offseason planner assumed one pick per round (first-round slot + 32 per round), ignoring traded
picks. Now: replay every trade of a 2027 pick in the nflverse trade log in date order (the log names giver, receiver and
round, not the original team: a team gives its own pick of that round if it still has it, else the first it acquired;
conditional picks count as conveyed), then project each pick from its original team's simulated first-round slot + 32
per round + compensatory picks ahead (`trade.COMP_OFFSET`, rounds 4-7). All 2027 pick trades resolve; the hand-coded
first-round trades (GB -> DAL, IND -> NYJ, DAL -> NYJ) are reproduced; 224 picks league-wide. NE: 22, 54, 86, 125, 162,
165 (CHI), 186 (NO), 193 (HOU), 200 (its 7th went to HOU). The trade log ends April 17 2026: later trades and
compensatory picks (awarded in March) still need adding by hand.

**Dead money already on the books** (`inputs/dead_money_2027.csv`, `offseason.load_dead` / `save_dead`, Nov 20 2026).
The contract data has no dead-money records: OTC rewrites a player's season history once he signs elsewhere, so the
2027 charge from June 1 cuts and post-June 1 trades in 2026 can't be rebuilt (void-year acceleration is already in
'Committed': placeholder 2027 rows carry it, $87.0M league-wide, matching the remaining void proration). OverTheCap's
terms (overthecap.com/terms, checked Nov 20 2026) prohibit automated or systematic retrieval without written
permission, so there is no fetcher; ask support@overthecap.com for a data license to automate it. Instead the planner
reads each team's figure from `inputs/dead_money_2027.csv` (team, dead_money_m, checked, source; tracked in git, all
blank to start) as the box's default, shows when it was checked, and a "Save" button writes what's typed back to the file.

### Spotrac's 2027 free-agent list as a check (Nov 20 2026)

Spotrac's 2027 free-agent page (saved by the user as a PDF; parsed locally, not stored in the repo): 748 players
(589 UFA, 78 RFA, 81 ERFA), Spotrac's projected market value (APY) for the top 106.
- **Who's a free agent.** First pass: 598 on our list, 36 we had under contract for 2027. Those were void years OTC
  fills with a minimum-level base salary that it also counts in the cap number (Travis Kelce, Kirk Cousins, Dallas
  Goedert, Tariq Woolen on one-year 2026 deals; Deshaun Watson's 2022 deal ends 2026; Daron Payne). Fixed
  (`fa_market.fresh_deal_end` / `real_season(x, end)`): seasons after the end of a fresh deal (any type but extension or
  rookie deal: year signed + years - 1) are void; their remaining proration counts as 2027 dead money in the planner
  (Watson: CLE $86.2M; Goedert: PHI $18.3M; Kelce: KC $7.1M). Now 623 of 748 on our list (922 in all), 9 still
  disputed (mostly ERFAs), 116 not on ours because they have no 2024-25 value to forecast from.
- **Prices** (102 matched): corr 0.84 (log); Spotrac 5% higher in total, median player 0.90x ours; typically 35%
  apart - about the same as our out-of-sample miss on real contracts (37%). Spotrac is far higher on young players
  coming off rookie deals (Joey Porter Jr. $27.4M vs our $10.8M, Tyrique Stevenson $20.1M vs $7.2M, Puka Nacua $47.6M vs
  $25.9M) and lower on older veterans (31+: 0.82x ours) and TEs (0.66x).
- **Which is right, by real signings** (our lag-2 predictions, out of sample, actual / predicted):

| | 2015-19 | 2020-23 | 2024-26 |
|---|---|---|---|
| off rookie deal, starter | 1.07 | 1.05 | 1.02 (0.92-1.13) |
| off rookie deal, other | 1.23 | 0.94 | 0.97 |
| veterans | 1.04 | 0.94 | 0.94 (0.88-0.98) |
| age 31+ | 0.98 | 0.98 | **0.89 (0.83-0.95)** |

  Real signings don't support Spotrac's much higher prices for rookie-deal players as a group (stars excepted, already
  flagged), nor a TE gap (0.91-0.96, ns). They do support Spotrac's lower prices for older veterans: the market for them
  has drifted down, and a model trained on 2015-26 equally prices them ~10% high. Candidate fix: weight recent signings
  more, tested forward (fit through year Y-1, predict Y).

**Recency weighting** (`scripts/contract_recency.py`, `fa_market.fit(half_life=)`). Forward-chained as the tool is used:
for 2020-26, fit on earlier UFA signings only (lag 2), predict that year; mean scaling also re-estimated forward.
Equal weights vs half-lives of 2 / 4 / 6 years, scored on 2022-26 (1,545 signings):

| | R2 log | median miss | all | 31+ | 28-30 | <=27 | off rookie deal, starter |
|---|---|---|---|---|---|---|---|
| equal | 0.480 | 35.0% | 1.02 | 0.98 (0.91-1.07) | 0.99 | 1.06 | 1.10 |
| h2 | 0.474 | 35.6% | 1.04 | 0.97 | 1.03 | 1.07 | 1.09 |
| h4 | 0.482 | 35.6% | 1.03 | 0.97 | 1.01 | 1.06 | 1.09 |
| h6 | 0.478 | 35.1% | 1.03 | 0.98 | 1.00 | 1.06 | 1.09 |

Not adopted (pre-set rule: no accuracy gain, no change in the 31+ bias). Forward, there is no 31+ overpricing (0.98);
the 0.89 in the leave-one-year-out table came from a single pooled scale factor. What the forward run shows instead:
2025 and 2026 signings ran ~10% above prediction overall (1.09, 1.12; 31+ 0.93, 1.02 those years) - prices rising as a
share of the cap, mostly for younger players, which recency weights don't capture (trees don't extrapolate a trend).
Worth rechecking after the 2027 signings; Spotrac's higher prices for young players point the same way.

**Counting stats in the contract model** (`scripts/contract_stats.py`; `fa_market.FEATS_STATS1/2`, `_stats`). Last
season's counting stats per 17 games (passing yds / TDs, rushing yds / TDs, receptions, receiving yds / TDs, sacks, QB
hits, tackles, INTs, passes defended; nflverse player_stats 2012-25), and the season before too:

| | R2 base -> stats1 -> stats2 | median miss |
|---|---|---|
| leave-year-out, lag 1 (normal March) | 0.626 -> **0.654** -> 0.650 | 30.9% -> **29.4%** -> 29.4% |
| leave-year-out, lag 2 (a year ahead) | 0.490 -> 0.490 -> 0.487 | 37.4% -> 37.4% -> 37.3% |
| forward 2022-26, lag 2 | 0.480 -> 0.476 -> 0.475 | 35.5% -> 35.8% -> 35.9% |

The market pays for the stats of the season just played; two seasons back they add nothing. Pre-set rule (better in
the forward check and at both lags) not met: default stays base. Planning 2027 now (data through 2025) is lag 2, so
nothing changes today. Once 2026 snaps, grades and stats are added (lag 1, next March), stats1 would cut the median miss
~1.5 points - a separate decision, recheck then with a forward lag-1 test.

### The season in progress: weeks 1-11 of 2026 (`scripts/partial_season_test.py`, Nov 20 2026)

Data added: 2026 snap counts through week 11 (`data/raw/snap_counts_2026.parquet`, was through week 8) and nflverse
weekly player stats 2012-2026 (`data/raw/player_stats_week.parquet`). No 2026 PFF grades (cfdarko's PFF files end with
2025). Features (`fa_market.partial`, `veteran.partial_snaps`): weeks 1-11 snap share per team game, share of team
games played, counting stats per 17 team games. Every past year replicated the same way (full data through Y-2,
weeks 1-11 of Y-1).

**A. Contract model** (planning a year ahead, lag 2):

| | leave-year-out R2 / median miss | forward 2022-26 R2 / median miss |
|---|---|---|
| base | 0.490 / 37.4% | 0.480 / 35.5% |
| **+ weeks 1-11** | **0.596 / 32.4%** | **0.605 / 30.8%** |
| full season (lag 1, next March) | 0.626 / 30.9% | 0.635 / 31.0% |

Adopted (`fa_market.feats_for(2)` = FEATS_PARTIAL; scale factor re-estimated 1.185 -> 1.153; calibrated by quintile).

**B. Value forecast** (seasons after the one in progress; 4000 anchors 2014-24): season-2 RMSE 1.344 -> 1.264 (% of
cap; MSE change -0.21e-4, -0.29 to -0.14), season 3 1.389 -> 1.335. Actual / forecast, season 2: players who played most
of weeks 1-11 base 1.12 -> 0.99; players who missed half or more base 0.44 -> 0.66; overall 0.99 -> 0.95 (season 3 0.97
-> 0.92, slightly optimistic). Adopted (pre-set rule on season 2): `veteran.Comps` adds p_share, p_gp to the linear shift
when the player has them (real players on every page; custom players forecast without).
Effects: KC trading #28 + 2028 R1 for Myles Garrett (30% of 2026 games so far) went from about even to -0.90% of a title;
Spotrac comparison corr 0.84 -> 0.81 (we now discount players hurt in 2026, e.g. Joey Porter Jr., Ethan Pocic, Mekhi
Becton; Spotrac doesn't appear to).

### Extensions in the workbench, and a cut-rule fix (Nov 20 2026)

**Extensions** (`fa_market.predict_extension`, `offseason.roster_moves`): a fourth table. Predicted new years, APY and
guaranteed share from a model trained on UFA signings + extensions (flag ufa = 0 for extensions; lag 2 + weeks 1-11).
Out of sample on 704 past extensions (leave-year-out): R2 log APY 0.80, median miss 20%, calibrated by quintile after a
scale of 1.039 (`EXT_SMEAR`); new years within one 90% of the time; guaranteed share miss 0.15. The extended contract:
his current seasons, then the new ones at the predicted APY (guarantees on the first new seasons), with his 2027 base
above the minimum turned into a bonus over all seasons in the window (how extensions usually lower the current hit).
Not suggested by the model (nothing pre-checked). Conflicts warn: cut wins over restructure / extension; extension wins
over restructure. Keep, cut, restructure and extend now use one set of comp futures per player, and each player's draws
in `evaluate_moves` are seeded by player, so options compare like for like whatever else is toggled.
Examples (NE): Drake Maye 4 new years at ~$71.7M: +3.7% of a title; Christian Barmore 3 years at $22.8M: -0.16% (cut
+0.05%, restructure +0.09%: his forecast is below that price).

**Cut rule** (`veteran.team_seasons_sched`, used by trades and the workbench; the flat-contract `team_seasons` used for
free agents and its validation is unchanged): a player still playing is now cut only when it saves cap that season (dead
money below his cap hit). Before, the rule ignored dead money and "cut" players right after big guarantees (an
extension's $39M of new guarantees landed as 2028 dead money in the bad futures). Workbench suggestions moved (KC: 5
restructures instead of a cut + restructures; MIA: no suggested cut; NE unchanged). Untested against history.
**Renegotiation eligibility** (Nov 20 2026): the workbench offered extensions and restructures to rookies who can't sign
them yet. A drafted player's rookie contract can't be renegotiated until after his third regular season, so for 2027
only draft classes through 2024 qualify (Drake Maye, 2024, is eligible this offseason; NE's 2025 draftees Reed, Woodson,
Henderson, Kyle Williams, Farmer are not). Enforced in `offseason.roster_moves` (no extension or restructure option;
listed under the Extensions table). Undrafted players aren't restricted here: the rule is written for drafted rookie
contracts, not verified for undrafted deals.

### Double count fixed; optimizer prototype (Nov 20 2026)

**Bug: players already on the roster were counted twice.** The workbench (per-player "Titles" columns, the summary
total) and the trade view's "players we give" scored a move as strength(baseline + his new points) - strength(baseline +
his current points). The team's baseline strength already includes everyone on its roster, so his current points were
added a second time, raising the team's strength level and, through the convex title curve, every move's value. Now a
move adds only its change: strength(baseline + new - current) - strength(baseline). Signings and picks (new players)
were right and are unchanged. NE's suggested plan: +1.9% -> +3.6% of a title, suggested cuts now Landry, Davis, Barmore,
Milton Williams; trades sending our players away score lower than before.

**Optimizer prototype** (`offseason.optimize`, not on the page). Local search over one action per player (keep / cut /
restructure / extend) and signings (add / drop / swap; an action plus the best signing it pays for), within the budget,
position caps and eligibility; every option drawn once per player, so a combination scores in ~1 ms; ~3 s per team.
NE: search score +12.3%, re-scored on fresh draws +12.1% (suggested plan +3.6%). The plan cuts 4, restructures 12,
extends 7 (including Morgan Moses at 36) and signs 8 - it leans on three model weak spots rather than finding a better
offseason: (1) value is credited at 0.19 points per 1% of cap and cap charged at 0.10, so any fairly priced deal adds
titles and extending everyone looks good; (2) restructures look nearly free (later seasons weigh less for NE; no cost
for lost flexibility or player consent); (3) a QB extension is double counted against the situation's "QB second
contract from season 2" shift, and "let his deal run out" means losing him for nothing while the baseline assumes he
stays. Not shipped; needs constraints or model fixes first.

### Extensions judged against re-signing later; optimizer on the page (Nov 20 2026)

**Extension comparison** (`offseason.roster_moves`, arrays["ext_alt"]): an extension was credited with the player's
whole value over a minimum-salary player in the new seasons, while "keep" meant losing him for nothing when his deal
ends - but the team's baseline strength already assumes a player of his level stays. Now an extension is judged
against keeping his deal and re-signing him for the same new seasons at the same predicted price and guarantees,
without restructuring his current year: it's credited with what doing it now changes (his current salary spread over
more seasons, more guaranteed money at risk). Same price on purpose - the free-agent model underprices young stars
(Drake Maye: $35.0M open market vs $71.7M extension), so a price gap between the two models would be an artifact; the
open-market price is kept (`ext_fa_apy`, `fa_market.predict_fa_price`) for information. A QB's second contract (already in
the situation's shift) is now paid in both versions. NE: Milton Williams +0.20%, Barmore +0.18% (now his best option),
Maye ~0 (almost no rookie salary to spread).

**Optimizer** (`offseason.optimize`, "Find the best combination" on the Offseason plan view): local search over one
action per player and signings from the top 60 candidates, within budget (signings count toward the 51), position caps
and eligibility; each action must add >= 0.02% of a title alone; at most 3 restructures and 2 extensions by default
(editable). Its result fills the tables; toggle from there; "Reset to suggestions" clears it. NE: +3.6% (suggestions) ->
+6.9% (cut Landry, Davis; extend Barmore, Williams; restructure Gonzalez, Doubs, Dre'Mont Jones; 8 signings; $0.4M left);
search score 7.15 vs fresh draws 7.02 (small optimizer's curse). With no limits it restructures / extends nearly
everyone (+8.0%): later seasons weigh less for NE and any fairly priced deal adds titles (value credited at 0.19, cap
charged at 0.10) - the limits are judgment, not fitted. Untested against past offseasons.
**Title odds on the plan** (Nov 20 2026): the workbench summary now shows the team's title odds before and after the
checked moves, from the same simulation that scores them (`offseason.evaluate_moves(detail=True)`): expected titles over
2027-31 (the five seasons' chances added up), chance of at least one title, and each season's chance. NE, suggested
moves: expected titles 6.9% -> 10.4% of one; at least one title 6.5% -> 9.7%; 2027 2.0% -> 4.6% (the gain lands where
the team is strongest and the signings are best). For an optimized plan the gain is an upper end.

**Void-year restructures** (Nov 20 2026; `offseason.roster_moves` arrays["restr_v"], `evaluate_moves(voids=)`): a second
restructure option. The 2027 salary above the minimum is spread over 5 seasons (real seasons left + void years): more cap
now (amount x 4/5); when the deal ends, the proration on the void years lands as dead money the next season (already in
his dead money if he's cut earlier). Final-year players can be restructured only this way. Restructures table: "Void
years" checkbox; both versions' 2027 savings and titles, and the dead money when it voids. Suggestions stay plain; the
optimizer may use either (counted toward its restructure limit). NE: Milton Williams $9.9M -> $15.9M freed (+0.12% ->
+0.22%, $11.9M dead when it voids); optimizer +6.9% -> +7.4% with 2 void-year restructures. The void version always
scores better (later seasons weigh less), and carrying dead money forward has no cost beyond its cap charge: read the
gain as the model's timing view.
**Void-year extensions** (Nov 20 2026; arrays["ext_v"], `evaluate_moves(ext_voids=)`): when the extended contract
(current + new seasons) is shorter than 5 seasons, the converted 2027 salary can be spread over 5, void years filling
the gap; the proration on the void years lands as dead money the season after the real contract ends. Extensions table:
"Void years" checkbox and both versions' numbers (blank when the contract already reaches 5). Optimizer: counted toward
its extension limit. NE: Milton Williams $14.9M -> $15.9M freed ($4.0M dead when it voids), Harold Landry $7.9M -> $9.5M;
small gains, since an extension already spreads most of the salary. Optimizer +7.3% (local search: adding options can
move where it stops by a few hundredths).

**Off-limits free agents** (Nov 20 2026; `inputs/off_limits_2027.csv`, `offseason.load_off_limits` / `save_off_limits`):
on the Offseason plan view, pick free agents who won't sign with the team (e.g. a player unlikely to return); they're
left out of the suggested signings, the signings table and the optimizer, and flagged "off limits for us" in the ranked
list. Saved per team when the selection changes (team, pfr_id, player, date added; tracked in git, empty to start).
Open issue noted: a team's own pending free agents (e.g. Mike Onwenu for NE) are treated like any outside signing -
the "before moves" baseline doesn't drop them if they leave - so re-signing them is undervalued relative to losing them.
