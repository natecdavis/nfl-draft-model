"""Do teams do better with players their own coaches coached in college?

Connection (src/coaching.py): someone on the drafting team's staff in the draft year coached at the
player's final college in the 5 seasons before the draft ("recent": in the last 2).
Comparisons (expectations out of sample, scripts/multisport_test.expectations):
  1. Selection: are teams more likely to draft players from schools their staff coached at?
  2. Reach: do connected teams take players earlier than the consensus board?
  3. Value vs draft slot, drafted players 2016-2022:
       a. connected picks vs all other picks
       b. fair version: only players whose school had a connection to SOME NFL staff that year,
          drafted by a connected team vs by an unconnected one (removes school / big-program effects)
  4. The same against the consensus board (board players 2018-2022).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import nflreadpy as nr
import prospect_model as pm
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT


def picks_with_connections():
    car, college = co.college_keys(pd.read_parquet(OUT / "coach_careers.parquet"))
    cn = co.connections()
    d = nr.load_draft_picks().to_pandas()
    d = d[d.season.between(2012, 2024)].copy()
    d["fr"] = d.team.map(co.franchise)
    d["key"] = d.college.map(college)
    team = cn.groupby(["fr", "year", "key"]).agg(n_coaches=("page", "nunique"), last=("last", "max"),
                                                  coaches=("name", lambda s: ", ".join(sorted(set(s)))),
                                                  roles=("role", lambda s: "; ".join(sorted(set(s))))).reset_index()
    d = d.merge(team, left_on=["fr", "season", "key"], right_on=["fr", "year", "key"], how="left").drop(columns="year")
    d["conn"] = d.n_coaches.notna()
    d["conn_recent"] = d.conn & (d["last"] >= d.season - 2)
    d["conn_lead"] = d.conn & d.roles.fillna("").str.contains(r"Head coach|coordinator", case=False)
    any_team = cn.groupby(["year", "key"]).fr.nunique().rename("n_teams_conn").reset_index()
    d = d.merge(any_team, left_on=["season", "key"], right_on=["year", "key"], how="left").drop(columns="year")
    d["n_teams_conn"] = d.n_teams_conn.fillna(0).astype(int)
    return d, cn


def main():
    d, cn = picks_with_connections()
    print(f"picks 2012-2024: {len(d)}; connected to the drafting team's staff: {d.conn.sum()} ({d.conn.mean():.1%}); "
          f"recent (last 2 seasons): {d.conn_recent.sum()}; via head coach / coordinator: {d.conn_lead.sum()}")

    # 1. selection: share of picks connected vs the share expected if teams ignored connections
    exp = (d.n_teams_conn / 32).mean()  # chance a random team has a staff link to that player's school
    print(f"\n1. SELECTION: connected picks {d.conn.mean():.1%} vs {exp:.1%} expected if connections didn't matter "
          f"(x{d.conn.mean() / exp:.2f})")
    for r, g in d.groupby(d["round"].clip(upper=5)):
        e = (g.n_teams_conn / 32).mean()
        print(f"   round {r}{'+' if r == 5 else ' '}: {g.conn.mean():.1%} vs {e:.1%} (x{g.conn.mean() / e:.2f})")

    # 2. reach vs consensus
    uni = pm.load()
    u = uni[uni.drafted.astype(bool) & uni.pfr_id.notna()][["pfr_id", "cls", "hasan_rank", "player_id"]]
    d = d.merge(u, left_on=["pfr_player_id", "season"], right_on=["pfr_id", "cls"], how="left")
    b = d[d.hasan_rank.notna()].copy()
    b["reach"] = np.log((b.hasan_rank + 10) / (b.pick + 10))  # >0 = taken earlier than his board rank
    print(f"\n2. REACH (picks 2014-2024 on the board, {len(b)}): taken earlier than board rank by "
          f"x{np.exp(b[b.conn].reach.mean()):.2f} when connected vs x{np.exp(b[~b.conn].reach.mean()):.2f} otherwise; "
          f"same-school fair group: x{np.exp(b[b.conn].reach.mean()):.2f} vs "
          f"x{np.exp(b[~b.conn & (b.n_teams_conn > 0)].reach.mean()):.2f}")

    # 3-4. value
    R = expectations(uni, SLOT_CLASSES)
    v = d.merge(R[["player_id", "actual", "exp_slot", "exp_cons", "on_board"]], on="player_id", how="inner")
    v["any_conn"] = v.n_teams_conn > 0
    print(f"\n3. VALUE vs DRAFT SLOT, picks 2016-2022 matched to outcomes: {len(v)}")
    for flag, lab in [("conn", "connected (any staff)"), ("conn_recent", "connected, last 2 seasons"),
                      ("conn_lead", "connected via HC / coordinator")]:
        print(f"  {lab}")
        compare(v, flag, "exp_slot", label="all picks")
        compare(v[v.any_conn], flag, "exp_slot", label="school linked to some NFL staff")
        compare(v[v.any_conn & (v["round"] <= 3)], flag, "exp_slot", label="  same, rounds 1-3")
        compare(v[v.any_conn & (v["round"] >= 4)], flag, "exp_slot", label="  same, rounds 4-7")
    print("\n4. VALUE vs CONSENSUS (board players 2018-2022)")
    bb = v[(v.on_board == 1) & v.exp_cons.notna()]
    compare(bb, "conn", "exp_cons", label="all board picks")
    compare(bb[bb.any_conn], "conn", "exp_cons", label="school linked to some NFL staff")
    d.to_parquet(OUT / "coaching_picks.parquet")
    v.to_parquet(OUT / "coaching_value.parquet")


if __name__ == "__main__":
    main()
