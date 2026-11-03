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
- [ ] M3: simulator: team strength -> playoff and title odds, checked against history
- [ ] M4: first chart - pick value in title odds for the 4 team situations
