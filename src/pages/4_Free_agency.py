"""Free agency: what signing a veteran to a given contract does to our Super Bowl odds (engine: src/veteran.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fa_market as fm  # noqa: E402
import offseason as off  # noqa: E402
import trade as tr  # noqa: E402
import vet_trade as vtr  # noqa: E402
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
def ranked(start, shift_key, qb_from, cap_price, team, need):
    s = {"start": start, "shift": dict(shift_key), "qb_from": qb_from}
    R = fm.rank(load_free_agents(), s, cap_price=cap_price, team=team, need=need)
    R["note"] = [fm.notes(r) for r in R.itertuples()]
    return R


def open_signing():
    """Row picked in the ranked list: show that player in the one-signing view."""
    rows = st.session_state.fa_table.selection.rows
    if rows:
        st.session_state.fa_search = st.session_state.fa_shown.iloc[rows[0]].label
        st.session_state.fa_view = "One signing"


@st.cache_data
def cap_projection():
    c = pd.read_parquet(vt.RAW / "contracts.parquet", columns=["year_signed", "apy", "apy_cap_pct"])
    cap_last = float((c.apy / c.apy_cap_pct).replace([np.inf], np.nan).groupby(c.year_signed).median().loc[SIGN_SEASON - 1])
    return float(round(cap_last * vtr.CAP_GROWTH))


@st.cache_resource(show_spinner="Loading the trade calculator...")
def trade_ctx():
    return tr.Context()


@st.cache_data(show_spinner="Loading rosters...")
def trade_roster():
    return vtr.roster(SIGN_SEASON)


def parse_now(text):
    return [tr.Pick(int(x)) for x in text.replace(" ", "").split(",") if x.strip().isdigit()]


def parse_future(text):
    """'2028 R1, 2029 R2 strong' -> Picks; 'middle' / 'weak' / 'strong' = another team's pick (else ours)."""
    out = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        parts = item.upper().split()
        try:
            yr, rnd = int(parts[0]), int(parts[1].lstrip("R"))
        except (ValueError, IndexError):
            continue
        src = ("Middle of the pack" if "MIDDLE" in parts else "Rebuilding" if "WEAK" in parts
               else "Contender, veteran QB" if "STRONG" in parts else "own")
        out.append(tr.Pick(rnd=rnd, year=yr - SIGN_SEASON, src=src))
    return out


def pick_label(pk):
    if pk.year == 0:
        return f"#{pk.number} ({SIGN_SEASON} draft)"
    src = "" if pk.src == "own" else {"Middle of the pack": " (middle team's)", "Rebuilding": " (weak team's)",
                                      "Contender, veteran QB": " (strong team's)"}[pk.src]
    return f"{SIGN_SEASON + pk.year} round {pk.rnd}{src}"


@st.cache_data(show_spinner="Pricing cuts and restructures for our roster...", max_entries=20)
def moves_for(team, start, shift_key, qb_from, cap):
    s = {"start": start, "shift": dict(shift_key), "qb_from": qb_from}
    return off.roster_moves(team, s, cap)


@st.cache_data(show_spinner="Valuing our first pick at each position...", max_entries=20)
def draft_values(start, shift_key, qb_from, pick):
    s = {"start": start, "shift": dict(shift_key), "qb_from": qb_from}
    return off.draft_by_group(s, pick, trade_ctx(), qb_need=qb_from is not None)


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
cap_m = st.sidebar.number_input(f"Salary cap in {SIGN_SEASON} ($M, projected)", 200.0, 500.0, cap_projection(), 1.0,
                                help=f"Default: the {SIGN_SEASON - 1} cap x {vtr.CAP_GROWTH:.2f} (the league sets the real "
                                     f"figure in late February; 2023-26 grew ~10% a year). Contract dollars are turned "
                                     f"into shares of this.")
rollover = st.sidebar.checkbox("Scenario: cap we don't spend rolls over and is spent well later", value=False,
                               help="Charges cap at what it would buy in our best later season. Real teams on "
                                    "average don't re-spend saved cap (scripts/cap_rollover_test.py), so this is a "
                                    "plan you'd have to carry out, not the default.")
cap_price = "rollover" if rollover else "model"
need_on = st.sidebar.checkbox("Experimental: adjust for our depth at his position", value=False,
                              disabled=(team == "Custom"),
                              help="Scales a signee's value by how full our room at his position is (strongest at "
                                   "OL and LB). It did NOT improve forecasts of past signings "
                                   "(scripts/fa_need_check.py), so it's off by default.")
rooms_full, rooms_players = fm.rooms(SIGN_SEASON)
room_row = rooms_full.loc[team] if team in rooms_full.index else None

warm()
st.title("Free agency: what a signing does to our Super Bowl odds")
st.caption(f"Contract seasons {SIGN_SEASON}-{SIGN_SEASON + 4}. Value = playing time x quality at veteran market "
           f"prices (the same yardstick as draft picks). Data through the {vt.LAST} season; the forecast runs from "
           f"there, so {vt.LAST + 1} counts as a season of aging.")

view = st.segmented_control("View", ["Ranked free agents", "One signing", "Trades", "Offseason plan"],
                            default="Ranked free agents",
                            key="fa_view", required=True)
fa = load_free_agents()

# ---------------- ranked free agents ----------------
if view == "Ranked free agents":
    if room_row is not None:
        with st.container(border=True):
            st.markdown(f"**{team}'s depth for {SIGN_SEASON}**: players under contract, by their {SIGN_SEASON - 1} "
                        "snaps per game so far, vs the other teams (1.00 = typical today).")
            cols = st.columns(len(room_row))
            for col, (g, v) in zip(cols, room_row.items()):
                col.metric(g, f"{v:.2f}", "open" if v < 0.75 else ("full" if v > 1.25 else None),
                           delta_color="off", border=False)
            st.caption("Information. Past free agents didn't deliver less in full rooms once the forecast is "
                       "accounted for, except that OL and LB rooms have fixed starting spots.")
    R = ranked(start, tuple(sorted(sit["shift"].items())), sit["qb_from"], cap_price,
               team if team != "Custom" else None, need_on and team != "Custom")
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
        "Our depth": [None if room_row is None or g not in room_row else round(float(room_row[g]), 2) for g in D.grp],
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

# ---------------- offseason plan: free-agency budget, best set of signings, draft vs free agency ----------------
if view == "Offseason plan":
    if team == "Custom":
        st.info("Pick a team in the sidebar: the plan needs its contracts and picks.")
        st.stop()
    trow = teams.loc[team]
    pk0 = list(trow.picks) if trow.picks else [40 + 32 * r for r in range(6)]
    with st.form("plan_form", border=True):
        a1, a2, a3, a4 = st.columns(4)
        picks_txt = a1.text_input(f"Our {SIGN_SEASON} picks", ", ".join(str(x) for x in pk0),
                                  help="Every pick we hold per the trade log (ends April 2026), slots projected from the "
                                       "2026 season simulation, compensatory picks ahead counted in rounds 4-7. Add "
                                       "later trades and our own compensatory picks (awarded in March).")
        rollover = a2.number_input("Cap rolled over from 2026 ($M)", 0.0, 150.0, 0.0, 1.0)
        dead = a3.number_input(f"Dead money already on {SIGN_SEASON} ($M)", 0.0, 150.0, 0.0, 1.0,
                               help="Not in the contract data; check OverTheCap.")
        reserve = a4.number_input("Reserve (% of cap)", 0.0, 10.0, 3.0, 0.5, help="In-season signings, injuries.")
        b1, b2 = st.columns(2)
        max_n = b1.slider("Most signings", 1, 15, 8)
        flagged = b2.toggle("Include players likely to be extended or tagged first", value=False)
        st.form_submit_button("Build plan", type="primary")
    picks_ = sorted({int(x) for x in picks_txt.replace(" ", "").split(",") if x.isdigit()})
    B = off.budget(team, cap_m, picks_, rollover, dead, reserve / 100)
    m = st.columns(5)
    m[0].metric(f"{SIGN_SEASON} cap", f"${B['cap'] + B['rollover']:.0f}M")
    m[1].metric("Committed", f"${B['commitments']:.0f}M", f"{B['n_signed']} players", delta_color="off")
    m[2].metric("Rookie pool", f"${B['rookies']:.0f}M", f"{len(picks_)} picks", delta_color="off")
    m[3].metric("Fill to 51 + reserve", f"${B['fill'] + B['reserve']:.0f}M")
    m[4].metric("Free-agency budget", f"{'-' if B['budget'] < 0 else ''}${abs(B['budget']):.0f}M")
    with st.expander(f"Contracts counted for {SIGN_SEASON} ({B['n_signed']})"):
        st.dataframe(B["committed_players"].round(1), hide_index=True,
                     column_config={"cap_m": st.column_config.NumberColumn("cap hit ($M)", format="$%.1f")})
    R = ranked(start, tuple(sorted(sit["shift"].items())), sit["qb_from"], cap_price,
               team, need_on)
    T, arrays = moves_for(team, start, tuple(sorted(sit["shift"].items())), sit["qb_from"], cap_m)
    sug_c, sug_r = off.suggest(T, B["budget"])
    ver = st.session_state.setdefault(f"plan_ver_{team}", 0)
    summary = st.container(border=True)
    money = lambda x: f"{'-' if x < 0 else ''}${abs(x):.1f}M"  # noqa: E731

    st.subheader("Cuts")
    Tc = T.assign(sug=T.pfr_id.isin(sug_c)).sort_values(["sug", "titles_cut"], ascending=False).reset_index(drop=True)
    cut_ed = st.data_editor(pd.DataFrame({
        "Do it": Tc.sug, "Player": Tc.player, "Pos": Tc.grp, "Age": Tc.age.round(1), "Years left": Tc.years_left,
        f"{SIGN_SEASON} cap hit": Tc.cap_now, "Dead money": Tc.dead_cut, f"{SIGN_SEASON} saved": Tc.saved_cut,
        "Titles (%)": Tc.titles_cut * 100, "Suggested": np.where(Tc.sug, "yes", "")}),
        key=f"cuts_{team}_{ver}", hide_index=True, disabled=[c for c in ["Player", "Pos", "Age", "Years left",
        f"{SIGN_SEASON} cap hit", "Dead money", f"{SIGN_SEASON} saved", "Titles (%)", "Suggested"]],
        column_config={"Do it": st.column_config.CheckboxColumn(width="small")} |
                      {c: st.column_config.NumberColumn(format="$%.1f") for c in
                       [f"{SIGN_SEASON} cap hit", "Dead money", f"{SIGN_SEASON} saved"]} |
                      {"Titles (%)": st.column_config.NumberColumn(format="%+.2f",
                       help="Super Bowls over 5 seasons from cutting him alone: losing his forecast value, taking the "
                            "dead money now, freeing his cap (charged at the model's 0.10 points per 1% of cap)")})
    st.caption("Before June 1: all remaining signing-bonus proration and guaranteed salary become dead money in "
               f"{SIGN_SEASON}. Suggested when cutting him adds titles on its own (forecast value well below his pay). "
               "Players with no 2024-25 snaps (most 2026 rookies) aren't listed.")
    cuts = set(Tc.pfr_id[cut_ed["Do it"].to_numpy()])

    st.subheader("Restructures")
    Tr = T[(T.years_left >= 2) & (T.saved_restr > 0.05)].assign(sug=lambda d: d.pfr_id.isin(sug_r))
    Tr = Tr.sort_values(["sug", "saved_restr"], ascending=False).reset_index(drop=True)
    restr_ed = st.data_editor(pd.DataFrame({
        "Do it": Tr.sug, "Player": Tr.player, "Pos": Tr.grp, "Age": Tr.age.round(1), "Years left": Tr.years_left,
        "Salary converted": Tr.restr_amount, f"{SIGN_SEASON} saved": Tr.saved_restr,
        "Added each later year": Tr.later_restr, "Titles (%)": Tr.titles_restr * 100,
        "Suggested": np.where(Tr.sug, "yes", "")}),
        key=f"restr_{team}_{ver}", hide_index=True, disabled=["Player", "Pos", "Age", "Years left", "Salary converted",
        f"{SIGN_SEASON} saved", "Added each later year", "Titles (%)", "Suggested"],
        column_config={"Do it": st.column_config.CheckboxColumn(width="small")} |
                      {c: st.column_config.NumberColumn(format="$%.1f") for c in
                       ["Salary converted", f"{SIGN_SEASON} saved", "Added each later year"]} |
                      {"Titles (%)": st.column_config.NumberColumn(format="%+.2f",
                       help="Moving cap from now to later seasons, and more dead money if he's cut later")})
    st.caption(f"His {SIGN_SEASON} salary above the minimum becomes a bonus spread over his remaining seasons (no void "
               "years added). Suggested only when the budget is still negative after the suggested cuts.")
    restrs = set(Tr.pfr_id[restr_ed["Do it"].to_numpy()]) - cuts

    saved = T[T.pfr_id.isin(cuts)].saved_cut.sum() + T[T.pfr_id.isin(restrs)].saved_restr.sum()
    fill_after = max(off.ROSTER - (B["n_signed"] - len(cuts)) - len(picks_), 0) * off.FILL_SHARE * cap_m
    budget_after = B["budget"] + saved - (fill_after - B["fill"])

    st.subheader("Signings")
    sver = st.session_state.setdefault(f"sign_ver_{team}", 0)
    if st.button("Re-plan signings for the current budget", icon=":material/refresh:"):
        st.session_state[f"sign_ver_{team}"] = sver = sver + 1
    plan_now = off.plan(R, budget_after, cap_m, max_n, qb_need=qb_need, include_flagged=flagged) if budget_after > 0 \
        else R.iloc[0:0]
    Cand = R[(R.titles > 0) & ~R.restricted & ((R.grp != "QB") | qb_need) &
             (flagged | ~R.note.str.contains("likely extended"))].copy()
    Cand["cost_m"] = Cand.apy_pred * cap_m
    Cand["per_10m"] = Cand.titles * 100 / Cand.cost_m * 10
    Cand = Cand.sort_values("per_10m", ascending=False)
    Cand = pd.concat([Cand[Cand.pfr_id.isin(plan_now.pfr_id)], Cand[~Cand.pfr_id.isin(plan_now.pfr_id)].head(60)])
    sign_ed = st.data_editor(pd.DataFrame({
        "Do it": Cand.pfr_id.isin(plan_now.pfr_id).to_numpy(), "Player": Cand.display_name.to_numpy(),
        "Pos": Cand.grp.to_numpy(), "Age": Cand.age.round(1).to_numpy(), "Predicted APY": Cand.cost_m.to_numpy(),
        "Yrs": Cand.yrs_pred.to_numpy(), "Titles (%)": (Cand.titles * 100).to_numpy(),
        "Per $10M": Cand.per_10m.to_numpy(), "Notes": Cand.note.to_numpy()}),
        key=f"sign_{team}_{ver}_{sver}", hide_index=True,
        disabled=["Player", "Pos", "Age", "Predicted APY", "Yrs", "Titles (%)", "Per $10M", "Notes"],
        column_config={"Do it": st.column_config.CheckboxColumn(width="small"),
                       "Predicted APY": st.column_config.NumberColumn(format="$%.1f"),
                       "Titles (%)": st.column_config.NumberColumn(format="%+.2f"),
                       "Per $10M": st.column_config.NumberColumn(format="%.2f", help="% of a title per $10M of APY")})
    st.caption("Checked at first: the greedy plan for the budget after the suggested moves (best titles per dollar, at "
               "most 2 at OL / DB / WR / DL and 1 elsewhere). Toggle anything; 'Re-plan' redoes it for the budget after "
               "your current cuts and restructures. Predicted prices miss by ~37% either way.")
    P = Cand[sign_ed["Do it"].to_numpy()]
    spend = P.cost_m.sum()
    total, alone = off.evaluate_moves(sit, arrays, cuts, restrs, P)
    with summary:
        st.markdown("**This offseason, as toggled below**")
        k = st.columns(5)
        k[0].metric("Budget before moves", money(B["budget"]))
        k[1].metric("Cuts + restructures free", money(saved - (fill_after - B["fill"])),
                    f"{len(cuts)} cuts, {len(restrs)} restructures", delta_color="off")
        k[2].metric("Signings (APY)", money(spend), f"{len(P)} players", delta_color="off")
        k[3].metric("Left", money(budget_after - spend))
        k[4].metric("Super Bowls added over 5 seasons", f"{total * 100:+.2f}%")
        st.caption(f"Each kind alone: cuts {alone['cuts'] * 100:+.2f}%, restructures {alone['restructures'] * 100:+.2f}%, "
                   f"signings {alone['signings'] * 100:+.2f}% of a title; the total scores everything together on one "
                   "strength path. Not modeled: post-June 1 cuts, void years in restructures, extensions, trades "
                   "(Trades view), re-signing our own free agents as such (they're in the signings list).")
        if budget_after - spend < 0:
            st.warning(f"Over the {SIGN_SEASON} cap by {money(spend - budget_after)}: free more room or drop a signing.")
        if st.button("Reset to suggestions", icon=":material/restart_alt:"):
            st.session_state[f"plan_ver_{team}"] = ver + 1
            st.rerun()
    st.subheader(f"Draft or free agency, by position (our first pick: #{picks_[0] if picks_ else '-'})")
    if picks_:
        dv = draft_values(start, tuple(sorted(sit["shift"].items())), sit["qb_from"], picks_[0])
        rows = []
        for g, v in sorted(dv.items(), key=lambda kv: -kv[1]):
            C_ = R[(R.grp == g) & ~R.restricted & (R.titles > 0) & (flagged | ~R.note.str.contains("likely extended"))]
            best = C_.sort_values("titles", ascending=False).iloc[0] if len(C_) else None
            planned = ", ".join(P[P.grp == g].display_name)
            rows.append({"Position": g, f"Typical player at #{picks_[0]} (titles %)": v * 100,
                         "Best free agent": best.display_name if best is not None else "",
                         "His titles at predicted price (%)": best.titles * 100 if best is not None else np.nan,
                         "His predicted APY ($M)": best.apy_pred * cap_m if best is not None else np.nan,
                         "In the plan": planned,
                         "Our depth": round(float(room_row[g]), 2) if room_row is not None and g in room_row else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Draft pick value vs best free agent by position",
                     column_config={c: st.column_config.NumberColumn(format="%+.2f") for c in
                                    [f"Typical player at #{picks_[0]} (titles %)", "His titles at predicted price (%)"]} |
                                   {"His predicted APY ($M)": st.column_config.NumberColumn(format="$%.1f")})
        st.caption("The first pick goes to one position: positions near the top are where it does most. A position "
                   "whose best free agent beats the pick there is a good one to fill in free agency and leave the pick "
                   "for elsewhere. Depth is information only (it didn't improve past forecasts). On draft day, take the "
                   "best player for us whatever this table says: the board decides (Draft room page).")
    st.stop()

# ---------------- trades: players and picks ----------------
if view == "Trades":
    roster_tbl = trade_roster()
    ours = roster_tbl[roster_tbl.team == team] if team != "Custom" else roster_tbl.iloc[0:0]
    give_opts = list(ours.label) + [x for x in roster_tbl.label if x not in set(ours.label)]
    st.caption(f"An offseason trade before the {SIGN_SEASON} season: players under contract for {SIGN_SEASON} (their "
               f"real remaining contracts) and picks. Our own players are listed first.")
    with st.form("trade_form", border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**We give**")
            give_lab = st.multiselect("Players", give_opts, key="tr_give_p", placeholder="Our players...")
            give_now = st.text_input(f"Picks in the {SIGN_SEASON} draft (numbers)", key="tr_give_now")
            give_fut = st.text_input("Future picks (e.g. '2028 R1, 2029 R2')", key="tr_give_fut")
        with c2:
            st.markdown("**We get**")
            get_lab = st.multiselect("Players", list(roster_tbl.label), key="tr_get_p", placeholder="Their players...")
            get_now = st.text_input(f"Picks in the {SIGN_SEASON} draft (numbers)", key="tr_get_now")
            get_fut = st.text_input("Future picks (add 'middle' / 'weak' / 'strong' for the partner's)", key="tr_get_fut")
        go = st.form_submit_button("Evaluate trade", type="primary")
    if go:
        lab2id = dict(zip(roster_tbl.label, roster_tbl.pfr_id))
        id2lab = {v: k for k, v in lab2id.items()}
        gp, gt = [lab2id[x] for x in give_lab], [lab2id[x] for x in get_lab]
        gpk, gtk = parse_now(give_now) + parse_future(give_fut), parse_now(get_now) + parse_future(get_fut)
        if not (gp or gt or gpk or gtk):
            st.info("Add at least one player or pick.")
            st.stop()
        with st.spinner("Simulating the trade..."):
            r = vtr.evaluate(sit, gp, gt, gpk, gtk, cap0=cap_m, ctx=trade_ctx())
        ch = r["change"] * 100
        verdict = "accept" if ch > 0.1 else ("decline" if ch < -0.1 else "roughly even")
        st.metric("Change in Super Bowls over 5 seasons", f"{ch:+.2f}% of a title", verdict,
                  delta_color="normal" if ch > 0.1 else ("inverse" if ch < -0.1 else "off"))
        st.caption(f"Simulation noise about ±{max(r['se'] * 100 * 2, 0.05):.2f}. Each part alone (below) doesn't add up "
                   "exactly to the total: gains combine on the title curve.")
        rows = []
        for a in r["assets"]:
            if isinstance(a["asset"], str):
                c = r["contracts"].get(("keep" if a["side"] == "give" else "get", a["asset"]))
                yrs = f"{c['years'][0]}-{c['years'][-1]}" if c and c["years"] else "none"
                sh = (c["acq_cost"] if a["side"] == "get" else c["keep_cost"]) if c else []
                hits = ", ".join(f"${x * cap_m * vtr.CAP_GROWTH ** i:.1f}M" for i, x in enumerate(sh))
                note = a["note"] or (f"we'd carry ${c['dead_if_traded'] * cap_m:.1f}M dead money in {SIGN_SEASON}"
                                     if a["side"] == "give" and c and c["dead_if_traded"] > 0.001 else "")
                rows.append({"side": a["side"], "asset": id2lab.get(a["asset"], a["asset"]), "under contract": yrs,
                             "cap hits for the team holding him": hits, "titles (alone)": a["titles"] * 100, "note": note})
            else:
                rows.append({"side": a["side"], "asset": pick_label(a["asset"]), "under contract": "",
                             "cap hits for the team holding him": "", "titles (alone)": a["titles"] * 100, "note": ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Each part of the trade and what it's worth alone",
                     column_config={"titles (alone)": st.column_config.NumberColumn(format="%+.2f",
                                    help="Super Bowls over 5 seasons, % of one title: gained (get) or given up (give)")})
        if gt:
            st.info("Veterans traded in the offseason (2014-24) delivered 0.94x their forecast in their first season with "
                    "the new team (0.86 to 1.03); starters 0.90x (0.80 to 1.01). Sellers may know a little more; "
                    "not applied.")
        if any(vtr.player_features(x) and vtr.player_features(x)["grp"] == "QB" for x in gp + gt):
            st.warning("QB values use the model's QB rate, which ran high for drafted QBs: treat QB trades' "
                       "verdicts as an upper end.")
        st.caption(f"Cap: the acquiring team pays his cap number minus his signing-bonus proration (the old team keeps "
                   f"that, all of it accelerating into {SIGN_SEASON}); cap grows {vtr.CAP_GROWTH - 1:.0%} a year from "
                   f"${cap_m:.0f}M. Future picks: slot drawn from the team's likely finish. Not modeled: post-June 1 "
                   "splits, restructures, the partner's view (each team's verdict depends on its own situation).")
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
if room_row is not None and grp in room_row:
    depth = float(room_row[grp])
    who = rooms_players[(rooms_players.team == team) & (rooms_players.g == grp)].sort_values("se", ascending=False)
    names = players.set_index("pfr_id").display_name
    listed = ", ".join(f"{names.get(p_, p_)} {s_:.2f}" for p_, s_ in zip(who.pfr_id[:5], who.se[:5]))
    nf = fm.need_factor(grp, depth) if need_on else 1.0
    st.caption(f"Our {grp} depth for {SIGN_SEASON}: {depth:.2f} x typical (under contract, {SIGN_SEASON - 1} starter "
               f"share so far: {listed or 'nobody'})." + (f" Experimental depth factor on his value: x{nf:.2f}." if need_on else ""))
    if nf != 1.0:
        paths = (paths[0] * nf, paths[1])
r = vt.evaluate(sit, player, int(years), apy, gtd, cap_price=cap_price, paths=paths)
be = vt.break_even(sit, player, int(years), gtd / max(apy * years, 1e-9), skip=SKIP, cap_price=cap_price, paths=paths)

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
    dup = C.pos.value_counts()
    if (dup > 1).any():
        st.warning(f"More than one option at {', '.join(dup[dup > 1].index)}: each is valued as if he alone fills the "
                   "position. Signing two would give each fewer snaps than shown.")
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
