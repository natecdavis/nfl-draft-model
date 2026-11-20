"""Free agency: what signing a veteran to a given contract does to our Super Bowl odds (engine: src/veteran.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402

st.set_page_config(page_title="Free agency: Super Bowl odds", layout="wide")
SIGN_SEASON = vt.LAST + 2  # data runs through the 2025 season; next signing period is March 2027
SKIP = SIGN_SEASON - vt.LAST - 1
GROUPS = ["QB", "RB", "WR", "TE", "OL", "DL", "EDGE", "LB", "DB"]


@st.cache_data(show_spinner="Loading team situations...")
def load_teams():
    from teams import team_inputs
    return team_inputs()


@st.cache_data(show_spinner="Loading players...")
def load_players():
    d = vt.current_players()
    d["age"] = (pd.Timestamp(f"{SIGN_SEASON}-09-01") - d.born).dt.days / 365.25
    d["label"] = (d.display_name + " (" + d.grp + ", " + d.latest_team.fillna("FA") + ", "
                  + (d.v1 * 100).round(1).astype(str) + "% value in " + str(vt.LAST) + ")")
    return d


@st.cache_data(show_spinner="Predicting contracts for this year's free agents...")
def load_free_agents():
    return fm.free_agents(SIGN_SEASON)


@st.cache_data(show_spinner="Ranking every free agent for this team (about 30 seconds, once per team setup)...",
               max_entries=20)
def ranked(start, shift_key, qb_from, cap_price):
    s = {"start": start, "shift": dict(shift_key), "qb_from": qb_from}
    R = fm.rank(load_free_agents(), s, cap_price=cap_price)
    R["note"] = [fm.notes(r) for r in R.itertuples()]
    return R


def open_signing():
    """Row picked in the ranked list: show that player in the one-signing view."""
    rows = st.session_state.fa_table.selection.rows
    if rows:
        st.session_state.fa_search = st.session_state.fa_shown.iloc[rows[0]].label
        st.session_state.fa_view = "One signing"


@st.cache_resource(show_spinner="Loading the forecast model...")
def warm():
    vt.comps(); vt._curve()
    return True


# ---------------- sidebar: who we are (same inputs as the draft room) ----------------
st.sidebar.header("Our team")
teams = load_teams()
team = st.sidebar.selectbox("Team", ["Custom"] + list(teams.index))
if team != "Custom":
    row = teams.loc[team]
    sit0 = row.situation
    st.sidebar.caption(f"Market strength now {row.strength_now:+.1f}. QB: {row.qb} ({row.qb_status}).")
else:
    sit0 = {"start": 0.0, "shift": {}, "qb_from": None}
start = st.sidebar.slider("Expected strength next season (pts/game vs average)", -8.0, 8.0,
                          float(round(sit0["start"], 1)), 0.5)
cap_hit = st.sidebar.selectbox("Our QB's big second contract starts in season", ["none", 1, 2, 3, 4, 5],
                               index=(["none", 1, 2, 3, 4, 5].index(list(sit0["shift"])[0]) if sit0["shift"] else 0))
qb_need = st.sidebar.checkbox("We need a starting QB", value=sit0["qb_from"] is not None)
sit = {"start": start, "shift": {} if cap_hit == "none" else {int(cap_hit): -3.0}, "qb_from": 1 if qb_need else None}
cap_m = st.sidebar.number_input(f"Salary cap in {SIGN_SEASON} ($M)", 200.0, 500.0, 299.0, 1.0,
                                help="Defaults to the 2026 cap. Contract dollars are turned into shares of this.")
rollover = st.sidebar.checkbox("Scenario: cap we don't spend rolls over and is spent well later", value=False,
                               help="Charges cap at what it would buy in our best later season. Real teams on "
                                    "average don't re-spend saved cap (scripts/cap_rollover_test.py), so this is a "
                                    "plan you'd have to carry out, not the default.")
cap_price = "rollover" if rollover else "model"

warm()
st.title("Free agency: what a signing does to our Super Bowl odds")
st.caption(f"Contract seasons {SIGN_SEASON}-{SIGN_SEASON + 4}. Value = playing time x quality at veteran market "
           f"prices (the same yardstick as draft picks). Data through the {vt.LAST} season; the forecast runs from "
           f"there, so {vt.LAST + 1} counts as a season of aging.")

view = st.segmented_control("View", ["Ranked free agents", "One signing"], default="Ranked free agents",
                            key="fa_view", required=True)
fa = load_free_agents()

# ---------------- ranked free agents ----------------
if view == "Ranked free agents":
    R = ranked(start, tuple(sorted(sit["shift"].items())), sit["qb_from"], cap_price)
    f1, f2, f3 = st.columns([3, 2, 2], vertical_alignment="bottom")
    pos = f1.pills("Positions", GROUPS, selection_mode="multi", default=[g for g in GROUPS if g != "QB" or qb_need])
    sort = f2.segmented_control("Rank by", ["Titles added", "Value gap ($)"], default="Titles added", required=True)
    max_apy = f3.number_input("Predicted cost up to ($M / yr)", 1.0, 80.0, 80.0, 1.0)
    g1_, g2_ = st.columns(2)
    show_rfa = g1_.toggle("Include likely restricted free agents", value=False,
                          help="3 or fewer seasons since he entered the league: his team can match offers.")
    show_neg = g2_.toggle("Include players who cost titles at their predicted price", value=True)
    D = R[R.grp.isin(pos) & (R.apy_pred * cap_m <= max_apy) & (show_rfa | ~R.restricted) & (show_neg | (R.titles > 0))]
    D = D.sort_values("titles" if sort == "Titles added" else "gap", ascending=False).reset_index(drop=True)
    st.session_state.fa_shown = D
    T = pd.DataFrame({
        "Player": D.display_name, "Pos": D.grp, "Age": D.age.round(1), "2026 team": D.otc_team,
        f"Value {vt.LAST} ($M)": D.v1 * cap_m,
        "Predicted APY ($M)": D.apy_pred * cap_m, "Yrs": D.yrs_pred,
        "Worth to us up to ($M / yr)": D.break_even * cap_m, "Gap ($M / yr)": D.gap * cap_m,
        "Titles added (%)": D.titles * 100, "Notes": D.note})
    st.caption(f"{len(D)} of {len(R)} players out of contract after {SIGN_SEASON - 1} (no {SIGN_SEASON} salary in the "
               "contract data). Each is priced at his predicted market contract (years and guarantees too). "
               "Select a row to open that signing.")
    st.dataframe(T, hide_index=True, key="fa_table", on_select=open_signing, selection_mode="single-row",
                 alt="Free agents ranked by Super Bowl odds added at their predicted contract",
                 column_config={
                     f"Value {vt.LAST} ($M)": st.column_config.NumberColumn(format="$%.1f",
                         help="What his playing time and quality were worth at veteran market prices"),
                     "Predicted APY ($M)": st.column_config.NumberColumn(format="$%.1f",
                         help="Average contract for players like him; typical miss ~37% either way"),
                     "Worth to us up to ($M / yr)": st.column_config.NumberColumn(format="$%.1f"),
                     "Gap ($M / yr)": st.column_config.NumberColumn(format="$%+.1f",
                         help="Worth to us minus predicted price: room to outbid the market"),
                     "Titles added (%)": st.column_config.NumberColumn(format="%+.2f",
                         help="Super Bowls added over 5 seasons at the predicted contract, % of one title"),
                     "Age": st.column_config.NumberColumn(format="%.1f")})
    with st.expander("How the contract predictions work"):
        st.markdown(f"""
- Trained on 3,276 past unrestricted free-agent signings (2015-2026): position, age, value in his last two seasons of
  data, games played, his previous contract's APY, and whether that was a rookie deal. Built with the same one-season gap
  as now (data through {vt.LAST}, signing in March {SIGN_SEASON}).
- Checked year by year out of sample: calibrated in every price band; the typical miss is ~37% either way
  (previous contract alone: 39%; on-field value alone: 46%). Years within one 89% of the time.
- Stars coming off rookie deals rarely reach free agency, so there are few examples: those who did signed for ~1.25x
  the prediction. Most get extended or tagged first.
- "Worth to us up to" uses the predicted years and guarantee share. Team needs aren't in it: every player is assumed
  to take a minimum-salary player's snaps. Two signings at one position don't both get that.
""")
    st.stop()

# ---------------- player ----------------
players = load_players()
c1, c2 = st.columns([3, 2])
with c1:
    mode = st.segmented_control("Player", ["Find a player", "Custom player"], default="Find a player", required=True)
    if mode == "Find a player":
        if "fa_search" in st.session_state and st.session_state.fa_search not in set(players.label):
            del st.session_state["fa_search"]
        pick = st.selectbox("Search", players.label, index=None, placeholder="Type a name...", key="fa_search")
        p = players[players.label == pick].iloc[0] if pick else None
    else:
        p = None
with c2:
    if p is not None:
        st.markdown(f"**{p.display_name}**: {p.grp}, age {p.age:.1f} in {SIGN_SEASON}")
        st.dataframe(pd.DataFrame({"season": [vt.LAST - 2, vt.LAST - 1, vt.LAST],
                                   "value (% of cap)": [p.v3 * 100, p.v2 * 100, p.v1 * 100]}).set_index("season").T
                     .round(2), width="stretch")
        st.caption(f"{vt.LAST}: {p.share1:.0%} of snaps in games played, played {p.g1:.0%} of games.")

with st.expander("Player inputs (edit to override)", expanded=(mode == "Custom player")):
    d1, d2, d3, d4, d5 = st.columns(5)
    grp = d1.selectbox("Position group", GROUPS, index=GROUPS.index(p.grp) if p is not None else 2)
    age = d2.number_input(f"Age in {SIGN_SEASON}", 21.0, 50.0, float(round(p.age, 1)) if p is not None else 28.0, 0.5)
    v1 = d3.number_input(f"Value {vt.LAST} (% of cap)", 0.0, 25.0, float(round(p.v1 * 100, 2)) if p is not None else 3.0, 0.1)
    v2 = d4.number_input(f"Value {vt.LAST - 1} (% of cap)", 0.0, 25.0, float(round(p.v2 * 100, 2)) if p is not None else 3.0, 0.1)
    g1 = d5.slider(f"Share of games played {vt.LAST}", 0.0, 1.0, float(p.g1) if p is not None else 1.0, 0.05)
    st.caption("Value guide: a good starter is ~3-4% of cap at most positions, a star EDGE/WR 6-9%, a top QB 15-19%.")
player = {"grp": grp, "age": age - SKIP, "v1": v1 / 100, "v2": v2 / 100, "g1": g1,
          "pfr_id": p.pfr_id if p is not None else None}
# age is fed as of the first forecast season (the comps' features), the contract starts SKIP seasons later

# ---------------- contract ----------------
st.subheader("Contract")
pred = fa[fa.pfr_id == p.pfr_id].iloc[0] if p is not None and (fa.pfr_id == p.pfr_id).any() else None
if pred is not None:
    st.info(f"Free agent in {SIGN_SEASON}. Predicted market contract: {pred.yrs_pred} yrs, "
            f"${pred.apy_pred * cap_m:.1f}M / yr, {pred.gshare_pred:.0%} guaranteed (typical miss ~37%). "
            "The contract below starts there.")
elif p is not None:
    st.caption(f"Under contract for {SIGN_SEASON} (a trade or extension); the contract below starts from his value.")
k1, k2, k3 = st.columns(3)
years = k1.number_input("Years", 1, 5, int(pred.yrs_pred) if pred is not None else 3)
apy0 = pred.apy_pred * cap_m if pred is not None else max(v1, 0.5) / 100 * cap_m / 0.7
apy_m = k2.number_input("Average per year ($M)", 0.5, 80.0, round(float(apy0), 1), 0.5,
                        help="Starts at his predicted market contract (free agents) or recent value / 0.7.")
g0 = pred.gshare_pred * apy_m * years if pred is not None else apy_m * min(years, 2)
gtd_m = k3.number_input("Guaranteed ($M, assumed paid first)", 0.0, 400.0, round(float(g0), 1), 0.5)
apy, gtd = apy_m / cap_m, gtd_m / cap_m
comp_age = vt.comps().comp_age(player, need=int(years) + SKIP)
if age - SKIP - comp_age > 1.5:
    st.warning(f"Few past players this old: his comparable players averaged {comp_age + SKIP:.0f}. The forecast "
               "likely runs high (players 31+ were already over-forecast by ~10-20% from the third season on).")
if grp == "QB":
    st.info("QB values use the model's QB rate (0.34 points per 1% of cap), which ran high for drafted QBs "
            "(round 1 QB recommendations delivered ~0.57x predicted). Past free-agent QBs: +0.86 predicted vs +0.61 "
            "realized points (not significant). Treat QB break-evens as an upper end.")
if grp == "QB" and not qb_need:
    st.warning("We have our QB: the model assumes a signed QB plays, so this overstates a backup's value.")

paths = vt.comps().paths(player, int(years), exclude=False, skip=SKIP)
r = vt.evaluate(sit, player, int(years), apy, gtd, cap_price=cap_price, paths=paths)
be = vt.break_even(sit, player, int(years), gtd / max(apy * years, 1e-9), skip=SKIP, cap_price=cap_price)

m1, m2, m3 = st.columns(3)
m1.metric("Super Bowls added over 5 seasons", f"{r['titles_added'] * 100:+.2f}% of one title")
gap = (be - apy) * cap_m
m2.metric("Worth signing up to (same guarantee share)", f"${be * cap_m:.1f}M / yr",
          f"${abs(gap):.1f}M {'of room under' if gap >= 0 else 'too much in'} this offer",
          delta_color="normal" if gap >= 0 else "inverse")
# value per dollar of pay, against past free agents at the same price (scripts/vet_value.py: value with the team /
# what the team paid, UFA 2015-23, non-QB 1-2% / 2-5% / 5%+ of cap; QBs similar on average)
typical = 0.88 if apy < 0.02 else 0.72 if apy < 0.05 else 0.53
paid = (r["cost"].sum() / max(int(years), 1))
ratio = r["value"].mean() / paid if paid > 0 else float("nan")
even = 0.10 / (0.336 if grp == "QB" else 0.187)  # cap charge / value rate: the ratio that adds nothing
m3.metric("Value delivered per $1 of cap charged", f"${ratio:.2f}",
          f"breaks even at ${even:.2f}; typical free agent at this price ${typical:.2f}", delta_color="off")

seasons = list(range(SIGN_SEASON, SIGN_SEASON + int(years)))
lo, med, hi = r["value_pct"]
T = pd.DataFrame({"season": seasons, "still on roster": r["on_team"], "expected value ($M)": r["value"] * cap_m,
                  "range 10-90% ($M)": [f"{a * cap_m:.1f} - {b * cap_m:.1f}" for a, b in zip(lo, hi)],
                  "cap charged ($M)": r["cost"] * cap_m, "team strength (pts/game)": r["pts_per_season"]})
st.dataframe(T.set_index("season").style.format({"still on roster": "{:.0%}", "expected value ($M)": "{:.1f}",
                                                 "cap charged ($M)": "{:.1f}", "team strength (pts/game)": "{:+.2f}"}),
             width="stretch")
st.caption("Cap charged includes dead money when he's cut (the team cuts him once last season's value falls below "
           "half of next season's non-guaranteed pay, fit to past free agents).")

# ---------------- comparison ----------------
st.subheader("Compare options")
if "fa_compare" not in st.session_state:
    st.session_state.fa_compare = []
name = p.display_name if p is not None else f"Custom {grp}, {age:.0f}"
if st.button("Add this signing to the comparison"):
    st.session_state.fa_compare.append({
        "player": name, "pos": grp, "years": int(years), "APY $M": apy_m, "guaranteed $M": gtd_m,
        "titles added %": r["titles_added"] * 100, "per $10M of APY": r["titles_added"] * 100 / apy_m * 10,
        "worth up to $M/yr": be * cap_m, "cap price": cap_price})
if st.session_state.fa_compare:
    C = pd.DataFrame(st.session_state.fa_compare)
    st.dataframe(C.round(2), width="stretch", hide_index=True)
    st.caption("Options at the same cost compare cleanly whatever the cap price; 'per $10M' ranks uses of the same "
               "money. Clear the list by reloading the page.")

with st.expander("How this works and how far to trust it"):
    st.markdown(f"""
- **Forecast**: the 60 most similar past players at his position (age, value in each of his last two seasons, games
  played), with their next seasons adjusted to his level. Checked out of sample: actual / forecast 1.00 in each of
  the next five seasons. Players 31+ are over-forecast by ~10-20% from the third season on.
- **Team strength**: value is turned into points per game at the draft model's rates (QB 0.34, others 0.19 per 1% of
  cap of value over a minimum-salary player); cap used costs 0.10 points per 1% (the draft model's charge). Assumes he
  takes snaps that would otherwise go to a minimum-salary player (fills a need).
- **Checked on 1,510 past free-agent contracts (2015-23)**: predicted +0.150 vs realized +0.137 points over the
  contract. Contracts of 5%+ of cap (about ${0.05 * cap_m:.0f}M+ / yr) came out ~0.2 points worse than predicted: read big
  deals as somewhat optimistic.
- **Not modeled**: backloaded cap hits, void years, restructures (the cap charge is the average per year); trade costs.
""")
