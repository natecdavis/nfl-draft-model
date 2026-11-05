"""Draft simulator: replay a team's past draft at its real picks, choosing your own player each time.

The page is the same self-contained HTML as the Claude artifact (scripts/simulator_artifact.html,
built by scripts/export_simulator_artifact.build_html), built here from the current review data and
simulator tables, so it follows any rebuild. The classic Streamlit version (with what your own picks
actually delivered) is the "Draft simulator classic" page.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from outlook import OUT  # noqa: E402

st.set_page_config(page_title="Draft simulator", layout="wide")


@st.cache_data(show_spinner="Packing the boards and the model for the simulator...")
def page(stamp, theme):
    from export_simulator_artifact import build_html
    return build_html(theme=theme)


tables = OUT / "sim_tables"
if not (OUT / "review_picks.parquet").exists() or not tables.exists():
    st.error("No review data or simulator tables yet. Run:  uv run python scripts/review_parallel.py  then  "
             "uv run python src/simulator.py build")
    st.stop()
# rebuild the page when the review, the tables or the template change
stamp = (max(f.stat().st_mtime for f in OUT.glob("review_*.parquet")), tables.stat().st_mtime,
         (ROOT / "scripts" / "simulator_artifact.html").stat().st_mtime)
theme = getattr(getattr(st.context, "theme", None), "type", None)
st.iframe(page(stamp, theme), height=2300)  # our own generated page, not user input
