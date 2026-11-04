from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../src/pages/1_Past_drafts.py", default_timeout=120)
at.run(); assert not at.exception, at.exception
at.selectbox[0].set_value(2017).run(); at.selectbox[1].set_value("KC").run()
assert not at.exception, at.exception
print("KC 2017 trades:", [m.value for m in at.markdown if "gave" in m.value])
at.radio[0].set_value("Round 1").run(); assert not at.exception, at.exception
print("captions:", [c.value[:120] for c in at.caption][-1])
df = at.dataframe[0].value
tr = df[df["Owner at start → used by"] == "trade"]
print(f"KC 2017 table: {len(df)} rows, {len(tr)} trade rows")
print(tr[["Pick", "Team took", "Titles (team's pick)", "Model would take", "Titles (model's pick)"]].to_string(index=False))
assert len(tr) > 0
for y in (2025, 2026):
    at.selectbox[0].set_value(y).run(); assert not at.exception, at.exception
print("2025/2026 pages load")
