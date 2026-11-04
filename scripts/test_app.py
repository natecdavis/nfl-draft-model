"""Headless check of the draft-day app with Streamlit's test runner."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("../src/app.py", default_timeout=180)
at.run()
assert not at.exception, at.exception
print("loaded; success box:", [s.value for s in at.success])
at.selectbox[0].set_value("NYJ").run()
assert not at.exception, at.exception
print("NYJ:", [s.value for s in at.success][:1])
at.multiselect[0].set_value(["Trevor Lawrence", "Zach Wilson"]).run()
print("after 2 taken:", [s.value for s in at.success][:1])
ti = at.text_input
ti[2].set_value("5")
ti[4].set_value("10, 40")
at.button[0].click().run()
assert not at.exception, at.exception
print("trade metric:", [(m.label, m.value, m.delta) for m in at.metric])
