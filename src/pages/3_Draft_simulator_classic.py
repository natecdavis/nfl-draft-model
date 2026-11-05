"""Draft simulator, classic Streamlit version: same model as the main simulator page, plus what your
own picks actually delivered (drafts through 2022)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from positions import weights  # noqa: E402
from simulator import GROUPS, OUT, get_table, options, pool, situation, value_of  # noqa: E402

st.set_page_config(page_title="Draft simulator (classic)", layout="wide")
TOP_N = 15
LABEL = {"T": "OT", "G": "OG", "ED": "EDGE", "IDL": "DL", "SPEC": "K/P"}
pos = lambda g: LABEL.get(g, g)


@st.cache_data
def load_review(stamp):
    return (pd.read_parquet(OUT / "review_picks.parquet"), pd.read_parquet(OUT / "review_states.parquet"))


@st.cache_data(show_spinner="Loading the board for this draft...")
def load_pool(year):
    return pool(year)


@st.cache_resource(show_spinner="Preparing the model for this team's situation (first time only, ~20s)...")
def load_table(key, year):
    return get_table({"start": key[0], "qb_from": key[1], "shift": dict(key[2])}, year=year)


@st.cache_resource(show_spinner=False)
def load_realized():
    try:
        from review import RealizedV2
        from trade import Context
        return RealizedV2(Context())
    except Exception:  # cfdarko labels not on this machine
        return None


@st.cache_data(show_spinner=False)
def realized(key, grp, year, drafted_at, pick, name):
    R = load_realized()
    if R is None:
        return np.nan
    return R.titles({"start": key[0], "qb_from": key[1], "shift": dict(key[2])}, grp, year, drafted_at, pick, name)


def rank_txt(r):
    return f"#{int(r)}" if pd.notna(r) else "off board"


def went_txt(row, tm):
    if pd.isna(row.drafted_at):
        return "undrafted"
    return f"#{int(row.drafted_at)} {row.drafted_by}" + (" (this team)" if row.drafted_by == tm else "")


if not (OUT / "review_picks.parquet").exists():
    st.error("No review data yet. Run:  uv run python src/review.py")
    st.stop()
picks, states = load_review((OUT / "review_picks.parquet").stat().st_mtime)

st.title("Draft simulator (classic)")
st.caption("Pick a team and a past draft, then make your own choice at each pick the team actually made. "
           "Other teams draft exactly as they really did: at each pick, everyone really taken before it is "
           "gone (including this team's own earlier choices, even if you passed on them), and anyone you took "
           "is gone at your later picks. Values = change in expected Super Bowls over the 5 seasons after the "
           "draft, in % of one title, for this team's situation on draft night (same model and consensus board "
           "as the Past drafts page). Draft-night trades aren't simulated: you use the picks the team ended up with.")

c1, c2, c3 = st.columns([1, 1, 2])
year = c1.selectbox("Draft", sorted(picks.year.unique())[::-1])
tm = c2.selectbox("Team", sorted(picks[picks.year == year].team.unique()))
srow = states[(states.year == year) & (states.team == tm)].iloc[0]
sit = situation(srow)
pn = float(srow.p_need) if "p_need" in srow and pd.notna(srow.p_need) else 1.0
key = (sit["start"], sit["qb_from"], tuple(sorted(sit["shift"].items())))
qb_need = ("no QB need in the window" if sit["qb_from"] is None
           else f"needs a starting QB from season {sit['qb_from']}" + (f" ({pn:.0%} likely)" if pn < 1 else ""))
c3.markdown(f"**{tm} going into {year}:** last season {srow.strength_prior:+.1f} pts/game vs average · "
            f"QB {srow.qb} ({srow.status}) → {qb_need}")

P = load_pool(year)
W = weights(int(year))
table = load_table(key, int(year))
mine = picks[(picks.year == year) & (picks.team == tm)].sort_values("pick").reset_index(drop=True)
skey = f"sim_{year}_{tm}"
choices = st.session_state.setdefault(skey, [])
show_real = year <= 2022 and load_realized() is not None

# ---- replay the choices made so far ----
taken, counts, hist = set(), {}, []
by_pid = P.set_index("pid")
for i, pid in enumerate(choices):
    r = mine.iloc[i]
    k = int(r.pick)
    o = options(P, table, k, taken, counts, pn, W)
    me = by_pid.loc[pid]
    v = value_of(table, me.grp, me["rank"], k, counts, pn, me.label, W)
    h = {"Pick": k, "Round": int(r["round"]), "You took": f"{me['name']} ({pos(me.grp)}, {rank_txt(me['rank'])})",
         "Your titles": v, "Model's #1 then": f"{o.iloc[0]['name']} ({pos(o.iloc[0].grp)})" if len(o) else "",
         "Model's #1 titles": o.iloc[0].titles if len(o) else np.nan,
         "Team took": f"{r.player} ({pos(r.grp)}, {rank_txt(r.board_rank)})", "Team's titles": r.value_actual}
    if show_real:
        h["You: actually delivered"] = realized(key, me.grp_record, year, me.drafted_at, k, me["name"])
        h["Team: actually delivered"] = r.real_team
    hist.append(h)
    taken.add(pid)
    counts[me.grp] = counts.get(me.grp, 0) + 1

# ---- the decision on the clock ----
if len(choices) < len(mine):
    r = mine.iloc[len(choices)]
    k = int(r.pick)
    via = f" · acquired from {r.original_team} on draft night" if r.traded_on_night else ""
    st.subheader(f"On the clock: pick #{k} (round {int(r['round'])}) · decision {len(choices) + 1} of {len(mine)}{via}")
    o = options(P, table, k, taken, counts, pn, W)
    team_row = o[o.drafted_at == k]

    a, b = st.columns(2)
    if len(team_row):
        t_ = team_row.iloc[0]
        a.metric(f"{tm} actually took", f"{t_['name']} ({pos(t_.grp)})", f"{t_.titles:+.2f}% of a title · model #{t_.model_rank}",
                 delta_color="off")
    elif r.grp not in GROUPS:
        a.metric(f"{tm} actually took", f"{r.player} ({pos(r.grp)})", "specialist: not valued", delta_color="off")
    else:
        a.metric(f"{tm} actually took", f"{r.player} ({pos(r.grp)})", "you already drafted him", delta_color="off")
    if len(o):
        b.metric("Model's top choice for you", f"{o.iloc[0]['name']} ({pos(o.iloc[0].grp)})",
                 f"{o.iloc[0].titles:+.2f}% of a title", delta_color="off")

    top = o.head(TOP_N).assign(
        pos=lambda d: d.grp.map(pos),
        cons=lambda d: d["rank"].map(rank_txt),
        went=lambda d: [went_txt(x, tm) + (" ← team's pick" if x.drafted_at == k else "") for x in d.itertuples()])
    st.dataframe(top[["model_rank", "name", "pos", "school", "cons", "went", "titles"]].rename(columns={
        "model_rank": "Model rank", "name": "Player", "pos": "Pos", "school": "School", "cons": "Consensus rank",
        "went": "Actually went", "titles": "Titles (% of one)"}), hide_index=True, width="stretch",
        column_config={"Titles (% of one)": st.column_config.NumberColumn(format="%+.2f")})

    labels = [f"{x['name']} · {pos(x.grp)} · {x.school} · consensus {rank_txt(x['rank'])} · model #{x.model_rank} · "
              f"{x.titles:+.2f} · went {went_txt(x, tm)}" for _, x in o.iterrows()]
    sel = st.selectbox(f"Any available player ({len(o)}; type to search)", range(len(o)),
                       format_func=lambda i: labels[i], key=f"{skey}_sel_{len(choices)}")
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("Draft selected player", type="primary", disabled=not len(o)):
        choices.append(int(o.iloc[sel].pid)); st.rerun()
    if b2.button("Take the model's #1", disabled=not len(o)):
        choices.append(int(o.iloc[0].pid)); st.rerun()
    if b3.button(f"Take {tm}'s actual pick", disabled=not len(team_row)):
        choices.append(int(team_row.iloc[0].pid)); st.rerun()
    st.caption("Model rank = order by titles added for this team at this pick, given the players you've "
               "already taken (a second player at a position counts less; after one QB the QB need is filled). "
               "Players drafted but not on the consensus board are valued as ranked just past it (#330). The "
               "team's pick here is valued with your draft so far, so it can differ from the team's own draft "
               "below once your choices differ.")
else:
    st.success(f"Draft complete: {len(mine)} picks.")

u1, u2, _ = st.columns([1, 1, 4])
if u1.button("Undo last pick", disabled=not choices):
    choices.pop(); st.rerun()
if u2.button("Start over", disabled=not choices):
    choices.clear(); st.rerun()

# ---- your draft so far vs the team's ----
if hist:
    st.subheader("Your draft vs the team's")
    H = pd.DataFrame(hist)
    num = [c for c in H.columns if c in ("Your titles", "Model's #1 titles", "Team's titles",
                                         "You: actually delivered", "Team: actually delivered")]
    st.dataframe(H, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%+.2f") for c in num})
    m = st.columns(4 if show_real else 2)
    m[0].metric("Your draft (titles, % of one)", f"{H['Your titles'].sum():+.2f}")
    team_total = H["Team's titles"].fillna(0).sum()
    m[1].metric(f"{tm}'s picks at the same slots", f"{team_total:+.2f}")
    if show_real:
        m[2].metric("Yours: actually delivered", f"{H['You: actually delivered'].fillna(0).sum():+.2f}")
        m[3].metric(f"{tm}'s: actually delivered", f"{H['Team: actually delivered'].fillna(0).sum():+.2f}")
    st.caption("Titles = the model's draft-night value. The team's values follow its own draft (its needs set by "
               "its own earlier picks), as on the Past drafts page. "
               + ("'Actually delivered' = each player's real first four seasons (with the team that really drafted "
                  "him), at the cost of this pick, as titles for this team; players who never reached the NFL count "
                  "zero (undrafted players are tracked from 2016)." if show_real else
                  ("Too soon to score how these players turned out (needs four seasons; drafts through 2022)."
                   if year > 2022 else "")))
