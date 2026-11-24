# NFL draft decisions valued in Super Bowl odds

How much does a draft pick, a trade up or down, or a future pick change a team's chance of winning a Super Bowl over the next five seasons? This project answers that question for general team situations (a contender with a veteran QB, a contender with a rookie-deal QB, a middle-of-the-pack team, a rebuilding team) and for each real team's 2027 picks.

## The chain

1. **Player outlook.** Chance of bust, starter or star by pick slot and position, plus season-by-season value sampled from careers of similar drafted players.
2. **Player → team strength.** A player's value is priced from second-contract pay (share of the salary cap) minus rookie-contract cost, over the years he is cheap. This is then converted into points of team strength.
3. **Team strength → titles.** A season and playoff simulator using the real schedule, seeding and bracket. It is calibrated on 1999–2026 games, and its playoff-game Brier score is 0.226 against 0.219 for the betting line.
4. **Decisions.** Compare picks, trades and future picks per team situation, scored in five-year title odds.

## Tools

The Streamlit app (`uv run streamlit run src/app.py`) has these pages:
- **Draft day:** live pick-by-pick advice from your own board: value of each available player, chance he's still there at your next pick, and what a trade down must return.
- **Trade offers:** any mix of current and future picks, from the draft-day page. `src/trade.py` scores the 396 real pick-only trades from 2013–2025.
- **Past drafts:** replays the 2014–2024 drafts pick by pick against the model's recommendation, with a scorecard on realized value.
- **Draft simulator** and **Free agency** pages.

## How changes get adopted

Each modeling idea is tested out of sample (rolling or leave-one-draft-out) against a stated primary metric before it's adopted. Many ideas failed and were not adopted, including roster-need adjustments, lookahead planning, scouting-text features, coaching-staff connections and a QB reach penalty. A holdout check tests whether the backtest leaks. [`NOTES.md`](NOTES.md) is the full research log, with every test, its numbers and its decision. `reports/ledger.csv` lists the decisions in one table.

## Setup

```bash
uv sync
uv run python scripts/check_data.py     # confirms the public nflverse data loads
uv run streamlit run src/app.py
```

Data is free via [nflreadpy](https://github.com/nflverse/nflreadpy): draft picks 1980–2026, OverTheCap contracts, schedules, snap counts. Some parts (outcome engine v2, the prospect model) read labels built by a companion college-to-NFL pipeline, [cfdarko-public](https://github.com/natecdavis/cfdarko-public). Third-party inputs used in research (consensus big boards, draft guides, paid grades) are kept local and not included here.
