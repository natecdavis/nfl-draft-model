from streamlit.testing.v1 import AppTest
at = AppTest.from_file("../src/pages/1_Past_drafts.py", default_timeout=120)
at.run(); assert not at.exception, at.exception
at.selectbox[0].set_value(2017).run(); at.selectbox[1].set_value("KC").run()
assert not at.exception, at.exception
print("KC 2017 trades:", [m.value for m in at.markdown if "gave" in m.value])
at.radio[0].set_value("Round 1").run(); assert not at.exception, at.exception
print("captions:", [c.value[:120] for c in at.caption][-1])
