"""Main draft simulator page (HTML in the app): builds and embeds without errors."""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("../src/pages/2_Draft_simulator.py", default_timeout=300)
at.run()
assert not at.exception, at.exception
assert not at.error, [e.value for e in at.error]
print("main simulator page builds")
