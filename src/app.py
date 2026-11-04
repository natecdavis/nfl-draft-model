"""Draft-day app. Run:  uv run streamlit run src/app.py

Everything runs locally. The team's board never leaves the machine.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from draftday import NEED_EARLY, DraftEngine  # noqa: E402
from outlook import GROUPS  # noqa: E402
from trade import Pick  # noqa: E402

st.set_page_config(page_title="Draft room: Super Bowl odds", layout="wide")


@st.cache_data(show_spinner="Loading team situations...")
def load_teams():
    from teams import team_inputs
    return team_inputs()


@st.cache_resource(show_spinner="Preparing the model for this team (once per setup)...")
def get_engine(start, shift_key, qb_from, need_key):
    sit = {"start": start, "shift": dict(shift_key), "qb_from": qb_from}
    return DraftEngine(sit, need=dict(need_key))


def demo_board():
    b = pd.read_csv("data/raw/lich/nfl_draft_prospects.csv", low_memory=False)
    b = b[(b.draft_year == 2021) & b.ovr_rk.notna()].sort_values("ovr_rk")
    return pd.DataFrame({"name": b.player_name, "pos": b.pos_abbr, "rank": b.ovr_rk.astype(int),
                         "adp": b.ovr_rk.astype(float)})


def parse_picks(text):
    return sorted({int(x) for x in text.replace(" ", "").split(",") if x})


def parse_future(text):
    """'2028 R1, 2028 R3 from middle' -> Picks. Year is relative to this draft."""
    out = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        parts = item.upper().split()
        yr, rnd = int(parts[0]), int(parts[1].lstrip("R"))
        src = "Middle of the pack" if "MIDDLE" in parts else ("Rebuilding" if "WEAK" in parts else
              ("Contender, veteran QB" if "STRONG" in parts else "own"))
        out.append(Pick(rnd=rnd, year=yr - st.session_state.get("draft_year", 2027), src=src))
    return out


# ---------------- sidebar: who we are ----------------
st.sidebar.header("Our team")
teams = load_teams()
team = st.sidebar.selectbox("Team", ["Custom"] + list(teams.index))
if team != "Custom":
    row = teams.loc[team]
    sit0 = row.situation
    st.sidebar.caption(f"Market strength now {row.strength_now:+.1f}. QB: {row.qb} ({row.qb_status}).")
    default_picks = ", ".join(str(p) for p in row.firsts + [row.firsts[0] + 32 * r for r in range(1, 7)]) \
        if row.firsts else "40, 72, 104, 136, 168, 200"
else:
    sit0 = {"start": 0.0, "shift": {}, "qb_from": None}
    default_picks = "16, 48, 80, 112, 144, 176, 208"

start = st.sidebar.slider("Expected strength next season (pts/game vs average)", -8.0, 8.0,
                          float(round(sit0["start"], 1)), 0.5)
qb_opts = {"No QB need": None, "Need a starter now": 1, "Starter needed from year 2": 2,
           "Starter needed from year 3": 3}
qb_default = [k for k, v in qb_opts.items() if v == sit0["qb_from"]][0] if sit0["qb_from"] in (None, 1, 2, 3) else "Need a starter now"
qb_choice = st.sidebar.selectbox("QB situation", list(qb_opts), index=list(qb_opts).index(qb_default))
cap_hit = st.sidebar.selectbox("Our QB's big second contract starts in season",
                               ["none", 1, 2, 3, 4, 5],
                               index=(["none", 1, 2, 3, 4, 5].index(list(sit0["shift"])[0]) if sit0["shift"] else 0))
shift = {} if cap_hit == "none" else {int(cap_hit): -3.0}

with st.sidebar.expander("Position needs (how much a rookie would play in years 1-2)"):
    need = {g: st.selectbox(g, list(NEED_EARLY), key=f"need_{g}") for g in GROUPS if g != "QB"}

st.sidebar.header("Our picks")
picks_text = st.sidebar.text_input("This draft (overall pick numbers)", default_picks)
future_text = st.sidebar.text_input("Future picks we own (e.g. '2028 R1, 2028 R2')", "2028 R1, 2028 R2")
st.session_state["draft_year"] = st.sidebar.number_input("Draft year", 2020, 2035, 2027)

engine = get_engine(start, tuple(sorted(shift.items())), qb_opts[qb_choice],
                    tuple(sorted(need.items())))

# ---------------- board ----------------
st.title("Draft room: what each move does to our Super Bowl odds")
st.caption("Values = change in expected Super Bowls over the next 5 seasons, in % of one title. "
           "Your board supplies the scouting; the model supplies position value, contract cost, "
           "your title window, and who is likely to be left.")
up = st.file_uploader("Our board (CSV with columns: name, pos, rank, adp). "
                      "rank = our talent order; adp = where the market expects him to go.", type="csv")
if up is not None:
    board = pd.read_csv(io.BytesIO(up.getvalue()))
    st.session_state.pop("demo", None)
else:
    board = demo_board()
    st.info("No board loaded: showing a demo with ESPN's 2021 board (rank = adp). Load your own CSV above.")
board = engine.prepare(board)

taken = st.multiselect("Players already drafted (mark as they go)", board.name.tolist(), key="taken")
live = board[~board.name.isin(taken)].reset_index(drop=True)
our_picks = parse_picks(picks_text)
now = st.number_input("Current pick number", 1, 300, len(taken) + 1)
upcoming = [p for p in our_picks if p >= now]

tab1, tab2, tab3 = st.tabs(["On the clock", "Trade offer", "Trade down?"])

with tab1:
    if not upcoming:
        st.warning("We have no picks left at or after the current pick.")
    else:
        ours = upcoming[0]
        if ours != now:
            st.write(f"Our next pick is **#{ours}** ({ours - now} picks away). "
                     "Values assume we pick there; availability accounts for the picks in between.")
        res = engine.on_the_clock(live, upcoming, now)
        show = res.head(25).copy()
        show = show.rename(columns={"titles_if_taken_now": f"titles if taken at #{ours}"})
        for c in show.columns:
            if c.startswith("p_avail_at_"):
                show = show.rename(columns={c: f"chance still there at #{c.split('_')[-1]}"})
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in show.columns
                                    if c.startswith("titles")} |
                                   {c: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0f%%")
                                    for c in show.columns if c.startswith("chance")})
        best = show.iloc[0]
        st.success(f"Best value at #{ours}: **{best['name']}** ({best.pos}), "
                   f"+{best[f'titles if taken at #{ours}']:.2f}% of a title.")

with tab2:
    st.write("Enter what we give and what we get. Picks this draft by number; future picks like '2028 R1'.")
    c1, c2 = st.columns(2)
    give_now = parse_picks(c1.text_input("We give: picks this draft", ""))
    give_fut = parse_future(c1.text_input("We give: future picks", ""))
    get_now = parse_picks(c2.text_input("We get: picks this draft", ""))
    get_fut = parse_future(c2.text_input("We get: future picks (add 'middle'/'weak'/'strong' for another team's)", ""))
    if st.button("Evaluate trade", type="primary") and (give_now or get_now or give_fut or get_fut):
        owned_fut = parse_future(future_text)
        r = engine.evaluate_trade(live, upcoming, now, give_now, get_now, give_fut, get_fut, owned_fut)
        verdict = "ACCEPT" if r["change"] > 0.1 else ("DECLINE" if r["change"] < -0.1 else "ROUGHLY EVEN")
        st.metric("Change in Super Bowls over 5 seasons", f"{r['change']:+.2f}% of a title", verdict)
        st.caption(f"Before: {r['before']:.2f}%, after: {r['after']:.2f}% (titles added by these picks). "
                   "Simulation noise is about ±0.2.")

with tab3:
    if upcoming:
        ours = upcoming[0]
        targets = [ours + d for d in (3, 6, 10, 15, 20, 32)]
        if st.button(f"Scan moving down from #{ours}"):
            scan = engine.trade_down_scan(live, upcoming, now, targets)
            scan["to break even, an extra pick worth at least"] = scan.titles_lost.map(lambda x: f"{x:.2f}% of a title")
            st.dataframe(scan.rename(columns={"move_to": "move down to", "titles_lost": "titles lost by moving"}),
                         hide_index=True)
            st.caption("Compare with the 'Trade offer' tab to price the extra pick a partner offers.")
