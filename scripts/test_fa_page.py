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

# 7. Optimizer: NE, find the best combination with default limits
at = fresh()
at.sidebar.selectbox[0].set_value("NE").run()
at.segmented_control[0].set_value("Offseason plan").run()
before = {m.label: m.value for m in at.metric}["Super Bowls added over 5 seasons"]
[b for b in at.button if b.label.startswith("Find the best")][0].click().run()
assert not at.exception, at.exception
ms = {m.label: (m.value, m.delta) for m in at.metric}
print("NE optimizer:", before, "->", ms["Super Bowls added over 5 seasons"][0], "|", ms["Cap moves free"], "|",
      ms["Signings (APY)"], "| left", ms["Left"][0])

# 7. Cap price robustness (scripts/fa_cap_price_robust.py): one sidebar "Firm signings only" switch for the ranked list,
# the plan and the optimizer; applies at once (no Build plan click), also after an optimizer run
at = fresh()
at.sidebar.selectbox[0].set_value("MIA").run()
n_all = len(at.dataframe[0].value)
firm = lambda: [t for t in at.sidebar.toggle if t.label == "Firm signings only"][0]  # noqa: E731
firm().set_value(True).run()
assert not at.exception, at.exception
df = at.dataframe[0].value
assert (df["Cap price"] == "firm").all()
print(f"MIA ranked: {n_all} rows, firm only {len(df)}; top 5 firm:", list(df.Player[:5]))
at.segmented_control[0].set_value("Offseason plan").run()
sig = lambda: {m.label: (m.value, m.delta) for m in at.metric}["Signings (APY)"]  # noqa: E731
print("MIA plan, firm only:", sig())
assert "depend" not in sig()[1]
[b for b in at.button if b.label == "Find the best combination"][0].click().run()
assert not at.exception, at.exception
print("MIA optimizer, firm only:", sig())
assert "depend" not in sig()[1]
firm().set_value(False).run()
[b for b in at.button if b.label == "Find the best combination"][0].click().run()
print("MIA optimizer, all:", sig())
firm().set_value(True).run()
print("after switching back on (stale optimizer plan dropped):", sig())
assert "depend" not in sig()[1]

# 8. Structured contract in One signing (veteran.structure): presets, void year, vs the same money flat
at = fresh()
at.sidebar.selectbox[0].set_value("KC").run()
at.segmented_control[0].set_value("One signing").run()
search = [s for s in at.selectbox if s.label == "Search"][0]
search.set_value([o for o in search.options if o.startswith("George Pickens")][0]).run()
flat = {m.label: m.value for m in at.metric}["Super Bowls added over 5 seasons"]
[t for t in at.toggle if t.label.startswith("Set the structure")][0].set_value(True).run()
assert not at.exception, at.exception
for preset in ["Signing bonus + rising salaries", "Front-loaded", "Flat"]:
    [s for s in at.selectbox if s.label == "Start from"][0].set_value(preset).run()
    assert not at.exception, at.exception
    ms = {m.label: (m.value, m.delta) for m in at.metric}
    print(f"structure {preset}: {ms['Super Bowls added over 5 seasons'][0]} (flat model {flat}); "
          f"vs same money flat {ms['This structure vs the same money paid flat']}")
[s for s in at.selectbox if s.label == "Start from"][0].set_value("Signing bonus + rising salaries").run()
[n for n in at.number_input if n.label == "Void years"][0].set_value(1).run()
assert not at.exception, at.exception
ms = {m.label: (m.value, m.delta) for m in at.metric}
print("rising + 1 void:", ms["Super Bowls added over 5 seasons"][0], ms["This structure vs the same money paid flat"])
print("cap hits:", [c.value[:200] for c in at.caption if c.value.startswith("Total $")])
print("season rows:", list(at.dataframe[-1].value.index) if len(at.dataframe) else None)

# 9. Workbench tables list the rows checked when they open first (suggestions, then an optimizer plan)
at = fresh()
at.sidebar.selectbox[0].set_value("NE").run()
at.segmented_control[0].set_value("Offseason plan").run()


def checked_first(tag):
    for df in [d.value for d in at.dataframe if "Do it" in d.value.columns]:
        d = df["Do it"].astype(bool).to_numpy()
        assert not (~d[:-1] & d[1:]).any(), f"{tag}: checked row below an unchecked one"
        print(f"{tag}: {len(df)} rows, {d.sum()} checked first: {list(df.Player[d])[:5]}")


checked_first("suggestions")
[b for b in at.button if b.label == "Find the best combination"][0].click().run()
assert not at.exception, at.exception
checked_first("optimizer")

# 10. Teams starting over the cap (KC, DAL): the suggestions and the optimizer both end under it (the optimizer used
# to drop suggested restructures that cost a little on their own, start from nothing and stay over); the page credits
# each signing with the minimum-salary roster spot it fills, as the optimizer does
for team in ["KC", "DAL", "MIA"]:
    at = fresh()
    at.sidebar.selectbox[0].set_value(team).run()
    at.segmented_control[0].set_value("Offseason plan").run()
    over = lambda: [w.value for w in at.warning if w.value.startswith("Over the")]  # noqa: E731
    left = lambda: {m.label: m.value for m in at.metric}["Left"]  # noqa: E731
    assert not over(), (team, "suggested", over())
    [b for b in at.button if b.label == "Find the best combination"][0].click().run()
    assert not at.exception, at.exception
    print(f"{team} over-cap check: optimizer left {left()}")
    assert not over(), (team, "optimizer", over())
