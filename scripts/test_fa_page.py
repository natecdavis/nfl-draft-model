"""Headless check of the Free agency page (src/pages/4_Free_agency.py). Run from the project root:
.venv/bin/python scripts/test_fa_page.py"""
from streamlit.testing.v1 import AppTest


def fresh():
    at = AppTest.from_file("../src/pages/4_Free_agency.py", default_timeout=600)
    at.run()
    assert not at.exception, at.exception
    return at


# 1. Ranked list (default view), custom team, then a contender
at = fresh()
df = at.dataframe[0].value
print("ranked rows:", len(df), "| top 5:", list(df.Player[:5]))
assert len(df) > 100
at.sidebar.selectbox[0].set_value("KC").run()
assert not at.exception, at.exception
df = at.dataframe[0].value
print("KC top 5:", list(zip(df.Player[:5], df["Titles added (%)"][:5].round(2))))
print("KC depth:", [(m.label, m.value) for m in at.metric][:9])
at.sidebar.checkbox[2].check().run()  # experimental depth factor
assert not at.exception, at.exception
df = at.dataframe[0].value
print("KC with depth factor, top 5:", list(zip(df.Player[:5], df["Our depth"][:5], df["Titles added (%)"][:5].round(2))))
at.sidebar.checkbox[2].uncheck().run()
at.segmented_control[1].set_value("Value gap ($)").run()
assert not at.exception, at.exception
print("KC by gap:", list(at.dataframe[0].value.Player[:3]))

# 2. One signing: search, team switch, rollover, comparison
at.segmented_control[0].set_value("One signing").run()
assert not at.exception, at.exception
search = [s for s in at.selectbox if s.label == "Search"][0]
search.set_value([o for o in search.options if o.startswith("George Pickens")][0]).run()
assert not at.exception, at.exception
print("Pickens info:", [i.value[:90] for i in at.info])
print("depth caption:", [c.value[:120] for c in at.caption if "depth" in c.value])
print("signing:", [(m.label, m.value, m.delta) for m in at.metric])
at.sidebar.checkbox[1].check().run()
assert not at.exception, at.exception
print("rollover:", [(m.value, m.delta) for m in at.metric][:2])
at.button[0].click().run()
assert not at.exception, at.exception
print("comparison rows:", len(at.dataframe[-1].value))

# 3. An old QB: age warning shows
at = fresh()
at.segmented_control[0].set_value("One signing").run()
search = [s for s in at.selectbox if s.label == "Search"][0]
search.set_value([o for o in search.options if o.startswith("Aaron Rodgers")][0]).run()
assert not at.exception, at.exception
print("Rodgers warnings:", [w.value[:40] for w in at.warning], [i.value[:30] for i in at.info])

# 4. Trades: contender gets a star EDGE for #28 + next year's 1st
at = fresh()
at.sidebar.selectbox[0].set_value("KC").run()
at.segmented_control[0].set_value("Trades").run()
assert not at.exception, at.exception
get = [m for m in at.multiselect if m.label == "Players"][1]
get.set_value([o for o in get.options if o.startswith("Myles Garrett")])
inputs = {(w.key): w for w in at.text_input}
inputs["tr_give_now"].set_value("28")
inputs["tr_give_fut"].set_value("2028 R1")
at.button[0].click().run()
assert not at.exception, at.exception
print("trade:", [(m.label, m.value, m.delta) for m in at.metric])
print("assets:", at.dataframe[0].value[["side", "asset", "under contract", "titles (alone)"]].round(2).values.tolist())

# 5. Offseason plan: a team with room (MIA) and one over the 2027 cap (KC)
at = fresh()
at.sidebar.selectbox[0].set_value("MIA").run()
at.segmented_control[0].set_value("Offseason plan").run()
assert not at.exception, at.exception
ms = {m.label: (m.value, m.delta) for m in at.metric}
print("MIA:", {k: ms[k] for k in ["Free-agency budget", "Cap moves free", "Signings (APY)", "Left",
                                   "Super Bowls added over 5 seasons"]})
draft = at.dataframe[-1].value
print("MIA draft vs FA (top 3):", draft.iloc[:3, :3].round(2).values.tolist())
at.sidebar.selectbox[0].set_value("KC").run()
assert not at.exception, at.exception
ms = {m.label: (m.value, m.delta) for m in at.metric}
print("KC:", {k: ms[k] for k in ["Budget before moves", "Cap moves free", "Left"]})

# 6. Workbench: NE over the cap -> suggested cuts fund signings; totals shown
at = fresh()
at.sidebar.selectbox[0].set_value("NE").run()
at.segmented_control[0].set_value("Offseason plan").run()
assert not at.exception, at.exception
ms = {m.label: (m.value, m.delta) for m in at.metric}
print("NE workbench:", {k: ms[k] for k in ["Budget before moves", "Cap moves free", "Signings (APY)", "Left",
                                            "Super Bowls added over 5 seasons"]})
at.button(key=None) if False else None
[b for b in at.button if b.label.startswith("Re-plan")][0].click().run()
assert not at.exception, at.exception
print("after re-plan:", {m.label: m.value for m in at.metric if m.label in ("Signings (APY)", "Left")})
