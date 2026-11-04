"""Headless check of the draft-day app with Streamlit's test runner."""
import pandas as pd
from streamlit.testing.v1 import AppTest

def fresh():
    at = AppTest.from_file("../src/app.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at

# 1. Demo board, team switch, players taken, a trade
at = fresh()
at.radio[0].set_value("Demo: 2021 draft").run()
print("demo:", [s.value for s in at.success][:1])
at.selectbox[0].set_value("NYJ").run()
at.multiselect[0].set_value(["Trevor Lawrence", "Zach Wilson"]).run()
print("NYJ after 2 taken:", [s.value for s in at.success][:1])
at.text_input(key=None) if False else None
inputs = {w.label: w for w in at.text_input}
inputs["We give: picks this draft"].set_value("5")
inputs["We get: picks this draft"].set_value("10, 40")
at.button[0].click().run()
assert not at.exception, at.exception
print("trade:", [(m.value, m.delta) for m in at.metric])

# 2. Pasted public boards (built from ESPN 2021 in two copy formats), combine nudge for 2021
b = pd.read_csv("data/raw/lich/nfl_draft_prospects.csv", low_memory=False)
b = b[(b.draft_year == 2021) & b.ovr_rk.notna()].sort_values("ovr_rk").head(150)
board1 = "\n".join(f"{int(r.ovr_rk)}. {r.player_name}, {r.pos_abbr}, {r.school}" for r in b.itertuples())
shuffled = b.assign(r2=b.ovr_rk + (b.ovr_rk % 7) - 3).sort_values("r2")
board2 = "Rank\tPlayer\tPos\tSchool\n" + "\n".join(
    f"{i}\t{r.player_name}\t{r.pos_abbr}\t{r.school}" for i, r in enumerate(shuffled.itertuples(), 1))
at = fresh()
at.sidebar.number_input[0].set_value(2021).run()
at.text_area(key="board0").set_value(board1).run()
at.text_area(key="board1").set_value(board2).run()
assert not at.exception, at.exception
print("captions:", [c.value for c in at.caption if "Read" in c.value or "ombine" in c.value])
print("pasted boards:", [s.value for s in at.success][:1])
