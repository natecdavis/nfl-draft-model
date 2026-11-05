"""Draft simulator page: KC 2017 end to end (team's pick, model's #1, any player, undo)."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("../src/pages/2_Draft_simulator.py", default_timeout=180)
at.run(); assert not at.exception, at.exception
at.selectbox[0].set_value(2017).run(); at.selectbox[1].set_value("KC").run()
assert not at.exception, at.exception
btn = lambda label: next(b for b in at.button if b.label.startswith(label))
print("on the clock:", [s.value for s in at.subheader][0])
print(at.dataframe[0].value.head(5).to_string(index=False))

first = at.dataframe[0].value.Player.iloc[0]
btn("Take the model's #1").click().run(); assert not at.exception, at.exception   # #10: model's choice
print("after 1:", [s.value for s in at.subheader][0])
board = at.dataframe[0].value
assert first not in board.Player.values, "a player you took must be gone"
btn("Take KC's actual pick").click().run(); assert not at.exception, at.exception  # #59
at.selectbox[2].set_value(5).run(); btn("Draft selected player").click().run()      # #86: 6th-best option
assert not at.exception, at.exception
btn("Undo last pick").click().run(); at.selectbox[2].set_value(2).run(); btn("Draft selected player").click().run()
for _ in range(3):
    btn("Take the model's #1").click().run(); assert not at.exception, at.exception
assert any("Draft complete" in s.value for s in at.success)
H = at.dataframe[0].value
print(H.to_string(index=False))
assert len(H) == 6 and "Kareem Hunt" not in H["You took"].iloc[2]
print("metrics:", [(m.label, m.value) for m in at.metric])
for y, t in [(2026, "NYG"), (2014, "HOU")]:
    at.selectbox[0].set_value(y).run(); at.selectbox[1].set_value(t).run(); assert not at.exception, at.exception
print("2026 NYG / 2014 HOU load")
