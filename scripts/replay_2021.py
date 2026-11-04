"""Replay the 2021 draft from the 49ers' seat, with ESPN's board as the team's board."""
import sys, time
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from draftday import DraftEngine
from simulate import league_params, load_schedules
from trade import Pick

b = pd.read_csv("data/raw/lich/nfl_draft_prospects.csv", low_memory=False)
b = b[(b.draft_year == 2021) & b.ovr_rk.notna()].sort_values("ovr_rk")
board = pd.DataFrame({"name": b.player_name, "pos": b.pos_abbr, "rank": b.ovr_rk.astype(int), "adp": b.ovr_rk})
print("positions on board:", sorted(board.pos.unique()))
rating_2020 = league_params(load_schedules())["ratings"][2020]["SF"]
sit = {"start": 0.6 * 0.73 * rating_2020, "shift": {}, "qb_from": 2}
print(f"SF 2020 rating {rating_2020:+.1f} -> expected 2021 strength {sit['start']:+.1f}")
t = time.time()
eng = DraftEngine(sit)
board = eng.prepare(board)
picks = [12, 43, 102, 172]
print(f"engine ready in {time.time() - t:.0f}s")

# Live use: players drafted before our pick are marked taken. Here: the real picks 1-11.
taken_before_12 = set(b[b.overall < 12].player_name)
print("Taken before #12:", ", ".join(sorted(taken_before_12, key=lambda n: b.set_index("player_name").overall[n])))
live = board[~board.name.isin(taken_before_12)].reset_index(drop=True)
t = time.time()
ranked = eng.on_the_clock(live, picks, now=12)
print(f"\nOn the clock at #12 ({time.time() - t:.1f}s): top 10 by titles added (% of one title)")
print(ranked.head(10).round(3).to_string(index=False))

# The actual trade: #12 + 2022 1st + 2023 1st + 2022 3rd -> #3 (they took Trey Lance)
t = time.time()
r = eng.evaluate_trade(board, picks, now=1, give_now=[12], get_now=[3],
                       give_future=[Pick(rnd=1, year=1), Pick(rnd=1, year=2), Pick(rnd=3, year=1)],
                       future_owned=[Pick(rnd=1, year=1), Pick(rnd=1, year=2), Pick(rnd=3, year=1)])
print(f"\nActual trade, judged in March before the draft ({time.time() - t:.1f}s): "
      f"titles before {r['before']:.1f}%, after {r['after']:.1f}%, change {r['change']:+.1f}% of a title")
sim = eng.simulate(board, picks, now=1)
i = board.index[board.name == "Trey Lance"][0]
print(f"Chance Trey Lance would still be there at #12 (using ESPN rank as expected draft spot): {sim['p_available'][i, 0]:.0%}")
for name in ["Justin Fields", "Mac Jones"]:
    j = board.index[board.name == name][0]
    print(f"   {name}: {sim['p_available'][j, 0]:.0%}")
lance = board[board.name.str.contains("Lance")]
print("Trey Lance on ESPN's board:", lance[["name", "pos", "rank"]].to_dict("records"))
