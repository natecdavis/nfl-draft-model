"""Draft-day app. Run:  uv run streamlit run src/app.py

Everything runs locally. The team's board never leaves the machine.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from draftday import NEED_EARLY, DraftEngine, qb_reach_note  # noqa: E402
from public_board import athletic_nudge, consensus, parse_board  # noqa: E402
from outlook import GROUPS  # noqa: E402
from trade import Pick  # noqa: E402

st.set_page_config(page_title="Draft room: Super Bowl odds", layout="wide")


@st.cache_data(show_spinner="Loading team situations...")
def load_teams():
    from teams import team_inputs
    return team_inputs()


@st.cache_resource(show_spinner="Preparing the model for this team (once per setup)...")
def get_engine(start, shift_key, qb_from, need_key, fall_weight):
    sit = {"start": start, "shift": dict(shift_key), "qb_from": qb_from}
    return DraftEngine(sit, need=dict(need_key), fall_weight=fall_weight)


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

with st.sidebar.expander("Advanced"):
    fall_weight = st.slider("When a player falls past his board rank, how much to believe the draft "
                            "instead of the board", 0.0, 1.0, 0.85, 0.05,
                            help="History says about 0.85: players who fall usually turn out like the pick "
                                 "where they went. Lower it only if you trust your board over 31 teams.")

st.sidebar.header("Our picks")
picks_text = st.sidebar.text_input("This draft (overall pick numbers)", default_picks)
future_text = st.sidebar.text_input("Future picks we own (e.g. '2028 R1, 2028 R2')", "2028 R1, 2028 R2")
st.session_state["draft_year"] = st.sidebar.number_input("Draft year", 2020, 2035, 2027)

engine = get_engine(start, tuple(sorted(shift.items())), qb_opts[qb_choice],
                    tuple(sorted(need.items())), fall_weight)

# ---------------- board ----------------
st.title("Draft room: what each move does to our Super Bowl odds")
st.caption("Values = change in expected Super Bowls over the next 5 seasons, in % of one title. "
           "Your board supplies the scouting; the model supplies position value, contract cost, "
           "your title window, and who is likely to be left.")
source = st.radio("Where do the player rankings come from?",
                  ["Public big boards (paste)", "Our own board (CSV)", "Demo: 2021 draft"], horizontal=True)
if source == "Public big boards (paste)":
    st.markdown("Open any published big board (e.g. ESPN, PFF, The Athletic, Bleacher Report), "
                "select the list, copy, and paste it below: one player per line with name, position and "
                "school. Paste 2-3 different outlets and the app averages them into a consensus.")
    cols = st.columns(3)
    pasted = {}
    for i, c in enumerate(cols):
        src = c.text_input(f"Board {i + 1} name", f"Board {i + 1}", key=f"src{i}")
        txt = c.text_area("Paste here", height=180, key=f"board{i}")
        if txt.strip():
            parsed, skipped = parse_board(txt)
            c.caption(f"Read {len(parsed)} players" + (f"; skipped {len(skipped)} lines" if skipped else ""))
            if skipped:
                with c.expander("Skipped lines"):
                    st.write(skipped[:20])
            if len(parsed):
                pasted[src] = parsed
    if not pasted:
        st.stop()
    cons = consensus(pasted)
    board = engine.prepare(cons)
    board, note = athletic_nudge(board, int(st.session_state["draft_year"]))
    st.caption(note)
    board["rank"] = board.talent_rank.round().astype(int)
    with st.expander(f"Consensus board ({len(board)} players from {len(pasted)} boards)"):
        st.dataframe(board[["rank", "name", "pos", "school", "adp", "n_boards", "best", "worst", "nudge"]]
                     .rename(columns={"adp": "expected pick", "n_boards": "boards listing him",
                                      "best": "best rank", "worst": "worst rank",
                                      "nudge": "combine nudge"}).round(1),
                     hide_index=True, width="stretch")
elif source == "Our own board (CSV)":
    up = st.file_uploader("CSV with columns: name, pos, rank, adp. rank = our talent order; "
                          "adp = where the market expects him to go.", type="csv")
    if up is None:
        st.stop()
    board = engine.prepare(pd.read_csv(io.BytesIO(up.getvalue())))
else:
    board = engine.prepare(demo_board())
    st.info("Demo: ESPN's 2021 board (talent rank = expected pick).")

taken = st.multiselect("Players already drafted (mark as they go)", board.name.tolist(), key="taken")
live = board[~board.name.isin(taken)].reset_index(drop=True)
our_picks = parse_picks(picks_text)
now = st.number_input("Current pick number", 1, 300, len(taken) + 1)
upcoming = [p for p in our_picks if p >= now]

with st.expander("How to read this (plain English)"):
    st.markdown("""
- **Titles added** = how much a move raises our chance of winning a Super Bowl over the next five
  seasons. +1.0 means one extra percentage point, so 100 moves like that would be worth one title.
  Most single picks are worth 0.5-3; a quarterback we truly need can be worth 10+.
- **Chance still there** = how often, in simulations of the other teams' picks, the player is still
  available at our next pick. If it's high, we can wait or trade down and still get him.
- Rankings come from the boards you paste. The model doesn't claim to scout better than the experts
  (public data can't, we tested it). It adds what the boards don't: how much each position helps win,
  what the rookie contract costs, our team's title window, and what trades are really worth.
""")

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
        note = qb_reach_note(res, ours, upcoming[1] if len(upcoming) > 1 else None)
        if note:
            (st.warning if note[0] == "warning" else st.info)(note[1])
        if len(upcoming) > 1:
            st.subheader("Take him, then the best available at our later picks")
            look = engine.lookahead(live, upcoming, now)
            look = look.rename(columns={"titles_now": f"titles at #{ours}", "titles_later_picks": "titles at our later picks",
                                        "titles_total": "titles, whole draft", "p_there_at_our_pick": f"chance there at #{ours}",
                                        "vs_best_now": "vs taking the best-value player"})
            st.dataframe(look, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in look.columns
                                        if c.startswith(("titles", "vs"))} |
                                       {f"chance there at #{ours}": st.column_config.ProgressColumn(
                                           min_value=0, max_value=1, format="%.0f%%")})
            st.caption("Each row: we take this player, then the best player for our roster at each later pick, over "
                       "simulated drafts (the same simulations for every row). In past drafts (2014-22) choosing by this "
                       "column tied or slightly trailed plain best-value-now on title value (scripts/lookahead_recheck.py), "
                       "so the ranking above stays the recommendation; use this to see what a choice costs later.")
            st.subheader(f"Position drop-off: #{ours} vs #{upcoming[1]}")
            drop = engine.position_dropoff(live, upcoming, now)
            st.dataframe(drop, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.2f") for c in drop.columns
                                        if c not in ("position", "chance of a big drop (>half)")} |
                                       {"chance of a big drop (>half)": st.column_config.ProgressColumn(
                                           min_value=0, max_value=1, format="%.0f%%")})
            st.caption("Best value at each position now vs the expected best at that position at our next pick if we "
                       "pass on it. Read it for cliffs, not as a ranking: choosing by largest drop-off did worse in past "
                       "drafts (it chases positions whose talent falls fastest even when they're worth little).")

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
                   "Simulation noise is about ±0.1.")

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
