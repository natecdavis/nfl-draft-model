"""Past drafts: what teams did on draft night vs what the model would have recommended."""
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path(__file__).resolve().parents[2] / "data" / "derived"
st.set_page_config(page_title="Past drafts", layout="wide")


@st.cache_data
def load():
    p = pd.read_parquet(OUT / "review_picks.parquet")
    s = pd.read_parquet(OUT / "review_states.parquet")
    t = pd.read_parquet(OUT / "review_trades.parquet")
    return p, s, t


if not (OUT / "review_picks.parquet").exists():
    st.error("No review data yet. Run:  uv run python src/review.py")
    st.stop()
picks, states, trades = load()

st.title("Past drafts: what teams did vs what the model would have done")
st.caption("Values = change in expected Super Bowls over the 5 seasons after the draft, in % of one title, "
           "judged with what was known on draft night: Arif Hasan's consensus board, each team's strength "
           "coming off the prior season, and its QB situation. 'Turned out' uses what actually happened "
           "(drafts through 2021): star = paid like a top player at his position, starter = played half "
           "the snaps for 2+ seasons, bust = neither.")

years = sorted(picks.year.unique())
c1, c2 = st.columns([1, 2])
year = c1.selectbox("Draft", years[::-1])
team_list = sorted(picks[picks.year == year].team.unique())
tm = c2.selectbox("Team (optional)", ["All teams"] + team_list)

P = picks[picks.year == year].copy()
S = states[states.year == year].copy()
T = trades[trades.year == year].copy()
if tm != "All teams":
    P = P[(P.team == tm) | (P.original_team == tm)]
    S = S[S.team == tm]
    T = T[T.trade_id.isin(T[T.team == tm].trade_id)]

tab1, tab2, tab3, tab4 = st.tabs(["Pick by pick", "Teams going in", "Draft-night trades", "Did the model's picks pan out?"])

with tab1:
    show = P.assign(
        pick_owner=lambda d: [f"{o} → {t}" if tr else t for o, t, tr in zip(d.original_team, d.team, d.traded_on_night)],
        took=lambda d: [f"{n} ({g}, board #{int(r)})" if pd.notna(r) else f"{n} ({g}, off board)"
                        for n, g, r in zip(d.player, d.grp, d.board_rank)],
        model=lambda d: [("same" if n == p else f"{n} ({g}, #{int(r)})") if isinstance(n, str) else ""
                         for n, g, r, p in zip(d.rec_player, d.rec_grp, d.rec_rank.fillna(0), d.player)],
    )
    cols = {"pick": "Pick", "pick_owner": "Owner at start → used by", "took": "Team took",
            "value_actual": "Titles (team's pick)", "model": "Model would take", "value_rec": "Titles (model's pick)",
            "value_gap": "Titles left on table", "outcome_actual": "Team's pick turned out",
            "outcome_rec": "Model's pick turned out"}
    st.dataframe(show[list(cols)].rename(columns=cols), hide_index=True, width="stretch", height=560,
                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                                ["Titles (team's pick)", "Titles (model's pick)", "Titles left on table"]})
    agree = (P.player == P.rec_player).mean()
    st.caption(f"The model agreed with the team on {agree:.0%} of these picks. 'Titles left on table' is how much "
               "more the model's choice was worth to that team, in its situation, at that moment.")

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
        "Became a star": [(d.outcome_actual == "star").mean(), (d.outcome_rec == "star").mean(), float("nan")],
    }, index=["Team's pick", "Model's pick", "Best left on consensus board"])
    st.dataframe(real.style.format({"Avg titles delivered per pick": "{:+.2f}", "Became a star": "{:.0%}"}),
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
