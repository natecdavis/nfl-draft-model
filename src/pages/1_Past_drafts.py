"""Past drafts: what teams did on draft night vs what the model would have recommended."""
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[2] / "data" / "derived"
st.set_page_config(page_title="Past drafts", layout="wide")


@st.cache_data
def load(stamp):
    p = pd.read_parquet(OUT / "review_picks.parquet")
    s = pd.read_parquet(OUT / "review_states.parquet")
    t = pd.read_parquet(OUT / "review_trades.parquet")
    f = OUT / "review_trade_decisions.parquet"
    td = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["year", "team"])
    return p, s, t, td


if not (OUT / "review_picks.parquet").exists():
    st.error("No review data yet. Run:  uv run python src/review.py")
    st.stop()
FILES = ["review_picks", "review_states", "review_trades", "review_trade_decisions"]
picks, states, trades, trade_dec = load(tuple((OUT / f"{n}.parquet").stat().st_mtime if (OUT / f"{n}.parquet").exists()
                                              else 0 for n in FILES))  # reload when the files change

st.title("Past drafts: what teams did vs what the model would have done")
st.caption("Values = change in expected Super Bowls over the 5 seasons after the draft, in % of one title, "
           "judged with what was known on draft night: Arif Hasan's consensus board, each team's strength "
           "coming off the prior season, and its QB situation. 'Actually delivered' uses what really happened "
           "(drafts 2014-2022, first four seasons): each player's real season-by-season value, at the cost "
           "of that pick, turned into title odds for that team. Above zero = he was worth more than the pick cost.")

years = sorted(picks.year.unique())
c1, c2 = st.columns([1, 2])
year = c1.selectbox("Draft", years[::-1])
team_list = sorted(picks[picks.year == year].team.unique())
tm = c2.selectbox("Team (optional)", ["All teams"] + team_list)

P = picks[picks.year == year].copy()
S = states[states.year == year].copy()
T = trades[trades.year == year].copy()
if tm != "All teams":
    P = P[P.team == tm]  # only this team's decisions: picks it gave away become trade rows below
    S = S[S.team == tm]
    T = T[T.trade_id.isin(T[T.team == tm].trade_id)]

if year == 2026:
    st.info("2026: the public trade log (nflverse) doesn't include draft-night trades yet, so each pick shows "
            "the team that used it, and the trades tab is empty. Too soon to score how anyone turned out.")
elif year > 2022:
    st.info(f"{year}: too soon to score how the players turned out (that needs four seasons), so the "
            "'actually delivered' columns are blank. The draft-night comparison is complete.")

tab1, tab2, tab3, tab4 = st.tabs(["Pick by pick", "Teams going in", "Draft-night trades", "Did the model's picks pan out?"])

with tab1:
    show = P.assign(
        pick_owner=lambda d: [f"{o} → {t}" if tr else t for o, t, tr in zip(d.original_team, d.team, d.traded_on_night)],
        took=lambda d: [f"{n} ({g}, board #{int(r)})" if pd.notna(r) else f"{n} ({g}, off board)"
                        for n, g, r in zip(d.player, d.grp, d.board_rank)],
        model=lambda d: [("same" if n == p else f"{n} ({g}, #{int(r)})") if isinstance(n, str) else ""
                         for n, g, r, p in zip(d.rec_player, d.rec_grp, d.rec_rank.fillna(0), d.player)],
    )
    if tm != "All teams":
        D = trade_dec[(trade_dec.year == year) & (trade_dec.team == tm)]
        if trade_dec.empty and year < 2026:
            st.warning("Trade rows are still being built (python src/review.py trades); this team's draft-night "
                       "trades are in the 'Draft-night trades' tab meanwhile.")
        if len(D):
            tr_rows = pd.DataFrame({
                "pick": D.pick, "order": 0, "pick_owner": "trade",
                "took": [f"TRADE: gave {g}; got {o}" + (f" → {d}" if isinstance(d, str) else "")
                         for g, o, d in zip(D.gave, D["got"], D.get("team_did", pd.Series(index=D.index)))],
                "value_actual": D.get("value_team"),
                "model": [n if isinstance(n, str) and n else (c or "") for c, n in
                          zip(D.get("model_choice", pd.Series(index=D.index)), D.note)],
                "value_rec": D.get("value_model"),
            })
            tr_rows["value_gap"] = tr_rows.value_rec - tr_rows.value_actual
            show = pd.concat([show.assign(order=1), tr_rows], ignore_index=True).sort_values(["pick", "order"])
    cols = {"pick": "Pick", "pick_owner": "Owner at start → used by", "took": "Team took",
            "value_actual": "Titles (team's pick)", "model": "Model would take", "value_rec": "Titles (model's pick)",
            "value_gap": "Titles left on table", "real_team": "Team's pick actually delivered",
            "real_model": "Model's pick actually delivered"}
    st.dataframe(show[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=560,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                ["Titles (team's pick)", "Titles (model's pick)", "Titles left on table",
                                 "Team's pick actually delivered", "Model's pick actually delivered"]})
    agree = (P.player == P.rec_player).mean()
    st.caption(f"The model agreed with the team on {agree:.0%} of these picks. 'Titles left on table' is how much "
               "more the model's choice was worth to that team, in its situation, at that moment. The 'actually "
               "delivered' columns are blank for drafts after 2022 (too soon to know) and for players who never "
               "reached the NFL before 2016 (not tracked).")
    if tm != "All teams":
        st.caption("Trade rows (one team selected): the trade is the team's decision. The model's alternative is the "
                   "better of 'don't trade' (keep the pick(s) and take the best players left on the board) or "
                   "'same trade' (then its own choices with the picks received). The team's value uses the players "
                   "it actually took with those picks; picks in later drafts are valued like the trade calculator. "
                   "A pick received in a trade also keeps its own row: given the trade, was it the right player? "
                   "Trades that include players aren't scored, and 'actually delivered' isn't scored for trades yet.")

with tab2:
    cols = {"team": "Team", "strength_prior": "Last season (pts/game vs avg)", "start": "Expected next season",
            "qb": "QB on draft day", "status": "QB status", "qb_from": "Needs a starting QB from season"}
    st.dataframe(S[list(cols)].rename(columns=cols).sort_values("Expected next season", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"Last season (pts/game vs avg)": st.column_config.NumberColumn(format="%+.1f"),
                                "Expected next season": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("QB status rules: a first-round QB on his rookie deal = no need; a later-round rookie = maybe by "
               "year 2; a veteran 36+ = successor needed soon; a cheap veteran = need now; a paid veteran who "
               "played below the league median last season = upgrade wanted by year 2.")

with tab3:
    if T.empty:
        st.write("No pick-for-pick trades on draft night" + ("" if tm == "All teams" else f" involving {tm}") + ".")
    else:
        for tid, g in T.groupby("trade_id"):
            a, b = g.iloc[0], g.iloc[-1]
            st.markdown(f"**{a.team}** gave {a.gave} and got {a.got} → **{a.titles_change:+.2f}** · "
                        f"**{b.team}** → **{b.titles_change:+.2f}**")
        st.caption("Trades of players are not included. Values judged with each team's own situation.")

with tab4:
    st.write("Every pick in the 2014-2022 drafts (four seasons played), scored by what each player actually delivered: "
             "his real playing time and market value, at the cost of that pick, turned into Super Bowl odds for "
             "the team that held it. Three candidates per pick: who the team took, who the model would have "
             "taken, and the top-ranked player left on the consensus board.")
    d = picks[(picks.year <= 2022) & picks.real_team.notna()].copy()
    rnd = st.radio("Rounds", ["All", "Round 1", "Rounds 2-3", "Rounds 4-7"], horizontal=True)
    if rnd == "Round 1":
        d = d[d["round"] == 1]
    elif rnd == "Rounds 2-3":
        d = d[d["round"].isin([2, 3])]
    elif rnd == "Rounds 4-7":
        d = d[d["round"] >= 4]
    real = pd.DataFrame({
        "Avg titles delivered per pick": [d.real_team.mean(), d.real_model.mean(), d.real_consensus.mean()],
        "Worth more than the pick cost": [(d.real_team > 0).mean(), (d.real_model > 0).mean(),
                                         (d.real_consensus > 0).mean()],
    }, index=["Team's pick", "Model's pick", "Best left on consensus board"])
    st.dataframe(real.style.format({"Avg titles delivered per pick": "{:+.2f}", "Worth more than the pick cost": "{:.0%}"}),
                 width="content")
    dis = d[d.player != d.rec_player].sort_values(["year", "pick"])
    once = dis.drop_duplicates(["year", "rec_player"])
    st.markdown(f"**Where the model disagreed, counting each recommended player once per draft** "
                f"({len(once)} picks): team's pick {once.real_team.mean():+.2f}, model's pick "
                f"{once.real_model.mean():+.2f}, difference **{(once.real_model - once.real_team).mean():+.2f}**.")
    st.caption("Why 'once': the model can recommend the same available player to many teams in a row (e.g. a "
               "late-round QB who later started), but only one team could have drafted him, so repeated "
               "recommendations would count his career many times. Other caveats: a player drafted by another "
               "team had that team's role and coaching; undrafted players count only if they reached the NFL "
               "(labeled from 2016).")
