"""Strict follower, step 1: room capacity. If a team took the simulator's top choice at every one of its
picks, 2014-2025, how many snaps would its own draftees need at each position, and do they fit?

The model has no memory across drafts (non-QB): a WR drafted last year doesn't lower this year's WR value,
and the QB need comes from the team's real history. Each recommended player is valued like a typical
player at his position and consensus rank, and those players mostly landed where they could play. If a
follower's draftees need more snaps than a position room has, the model counts some value twice.

Follower: at each pick the team really made (incl. draft-night trades), the top of simulator.options
(same tables, QB need probability, within-draft step, position blending as the app). Other teams draft as
they really did; players the follower took are gone for its later picks. Specialists left out.

Snap shares: a player's season snaps / his team's offensive or defensive snaps (regular + playoffs),
summed over teams. Projected = average for real drafted board players 2013-2022 at the same group,
season of career (1-4) and consensus rank (Gaussian kernel in log(rank + 10), bw 0.35); R1 picks also
season 5 (option year, = season 4). Realized = what the follower's players really played (anywhere).
Room = the group's total snap share in a team-season (league median). Baseline = the same projection for
the real team's own draftees.

Run:  python scripts/strict_follower.py draft      (follower picks -> data/derived/strict_follower.parquet)
      python scripts/strict_follower.py capacity
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import GROUPS, OUT, RAW  # noqa: E402

YEARS = list(range(2014, 2026))
FOLLOW = OUT / "strict_follower.parquet"
SNAP_POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "T", "OT": "T", "G": "G", "OG": "G",
            "C": "C", "DE": "ED", "OLB": "ED", "DT": "IDL", "NT": "IDL", "DL": "IDL", "LB": "LB", "ILB": "LB",
            "MLB": "LB", "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S", "SAF": "S",
            # combined labels (as roster_need.SNAP_POS). nflverse lists about a third of 2020 and 2025 line snaps
            # as "OL" or "G/T" etc.; these were dropped until Nov 17 2026.
            "HB": "RB", "FB/D": "RB", "FB/R": "RB", "FB/T": "RB", "RB/F": "RB", "RB/W": "RB", "WR/R": "WR",
            "G/T": "T", "T/G": "T", "G/OT": "T", "G/C": "G", "C/G": "C", "DE/L": "ED", "DE/D": "ED", "DT/D": "IDL",
            "LB/F": "LB", "LB/S": "LB", "CB/R": "CB", "DB/L": "CB"}
PFF_OL = {"T": "T", "G": "G", "C": "C"}
OFFENSE = {"QB", "RB", "WR", "TE", "T", "G", "C"}


def draft():
    from positions import weights
    from review import draft_night
    from simulator import get_table, options, pool, situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    rows = []
    for y in YEARS:
        picks, _ = draft_night(y)
        P = pool(y)
        W = weights(y)
        for _, s in S[S.year == y].iterrows():
            tab = get_table(situation(s), year=y)
            taken, counts = set(), {}
            for _, p in picks[picks.team == s.team].sort_values("pick").iterrows():
                o = options(P, tab, int(p.pick), taken, counts, float(s.p_need), W)
                if not len(o):
                    continue
                b = o.iloc[0]
                taken.add(b.pid)
                counts[b.grp] = counts.get(b.grp, 0) + 1
                rows.append({"year": y, "team": s.team, "pick": int(p.pick), "round": int(p["round"]),
                             "name": b["name"], "grp": b.grp, "rank": b["rank"], "drafted_at": b.drafted_at,
                             "drafted_by": b.drafted_by, "titles": b.titles,
                             "actual_grp": p.grp, "actual_id": p.pfr_player_id})
        print(y, len(rows), flush=True)
    pd.DataFrame(rows).to_parquet(FOLLOW)


def season_shares() -> pd.DataFrame:
    """pfr_player_id, season -> snap share (summed over teams), team, group from snap position."""
    s = pd.read_parquet(RAW / "snap_counts.parquet")
    s["grp"] = s.position.map(SNAP_POS)
    # plain "OL": his PFF position, else his draft position, else guard
    ol = s.position == "OL"
    pl = pd.read_parquet(RAW / "players.parquet").dropna(subset=["pfr_id"])
    ppos = dict(zip(pl.pfr_id, pl.pff_position.map(PFF_OL)))
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    dpos = dict(zip(dp.pfr_player_id, dp.position.map({"T": "T", "OT": "T", "G": "G", "OG": "G", "OL": "G", "C": "C"})))
    s.loc[ol, "grp"] = [ppos.get(p) if isinstance(ppos.get(p), str) else dpos.get(p) if isinstance(dpos.get(p), str) else "G"
                        for p in s.pfr_player_id[ol]]
    s = s[s.grp.notna()]
    off = s.grp.isin(OFFENSE)
    s["snaps"] = np.where(off, s.offense_snaps, s.defense_snaps)
    # team snaps per game = max offensive / defensive snaps any player had
    tg = s.groupby(["season", "team", "game_id"]).agg(o=("offense_snaps", "max"), d=("defense_snaps", "max")).reset_index()
    tt = tg.groupby(["season", "team"])[["o", "d"]].sum()
    s = s.join(tt, on=["season", "team"])
    s["share"] = s.snaps / np.where(off, s.o, s.d)
    return s.groupby(["pfr_player_id", "season", "team", "grp"], as_index=False).share.sum()


def rooms(sh: pd.DataFrame) -> pd.DataFrame:
    """Total snap share by team-season-group."""
    return sh.groupby(["season", "team", "grp"], as_index=False).share.sum()


def board_players() -> pd.DataFrame:
    """Drafted board players 2013-2025: year, pick, rank, grp (board label group), pfr id."""
    from review import board, draft_night
    out = []
    for y in range(2013, 2026):
        try:
            picks, _ = draft_night(y)
            b = board(y, picks)
        except Exception as e:  # no board that year
            print("skip", y, e)
            continue
        b = b[b.drafted_at.notna()].merge(picks[["pick", "pfr_player_id", "team"]], left_on="drafted_at", right_on="pick")
        out.append(pd.DataFrame({"year": y, "pick": b.pick, "rank": b["rank"].astype(float), "grp": b.grp_label,
                                 "pfr_player_id": b.pfr_player_id, "team": b.team}))
    return pd.concat(out, ignore_index=True)


def expected_curve(bp: pd.DataFrame, pshare: pd.Series):
    """f(grp, rank) -> (4,) expected snap share in seasons 1-4 (fit on 2013-2022 classes)."""
    d = bp[bp.year <= 2022].copy()
    for t in range(1, 5):
        d[f"s{t}"] = [pshare.get((pid, y + t - 1), 0.0) for pid, y in zip(d.pfr_player_id, d.year)]
    cache = {}

    def f(g, r):
        r = 300.0 if pd.isna(r) else float(r)
        key = (g, round(r))
        if key not in cache:
            x = d[d.grp == g]
            w = np.exp(-0.5 * ((np.log(x["rank"] + 10) - np.log(r + 10)) / 0.35) ** 2)
            cache[key] = (x[[f"s{t}" for t in range(1, 5)]].to_numpy() * w.to_numpy()[:, None]).sum(0) / w.sum()
        return cache[key]
    return f


def stack(df: pd.DataFrame, f, idcol=None, pshare=None) -> pd.DataFrame:
    """Per team-season-group: projected (and realized) snap share of this team's draftees."""
    rows = []
    for r in df.itertuples():
        e = f(r.grp, r.rank)
        n = 5 if r.pick <= 32 else 4
        for t in range(1, n + 1):
            row = {"team": r.team, "season": r.year + t - 1, "grp": r.grp, "draft": r.year, "proj": e[min(t, 4) - 1]}
            if idcol:
                pid = getattr(r, idcol)
                row["real"] = pshare.get((pid, r.year + t - 1), 0.0) if isinstance(pid, str) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def capacity():
    F = pd.read_parquet(FOLLOW)
    F = F[F.grp.isin(GROUPS)]
    sh = season_shares()
    pshare = sh.groupby(["pfr_player_id", "season"]).share.sum()
    R = rooms(sh)
    room = R[R.season.between(2014, 2025)].groupby("grp").share.median()
    bp = board_players()
    f = expected_curve(bp, pshare)

    # realized for follower picks: the player's pfr id via the pick he was really drafted at
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    pid = {(int(s), int(k)): i for s, k, i in zip(dp.season, dp.pick, dp.pfr_player_id)}
    F["pfr_id"] = [pid.get((y, int(k))) if pd.notna(k) else None for y, k in zip(F.year, F.drafted_at)]
    FS = stack(F, f, "pfr_id", pshare)

    # baseline: real teams' own drafted board players, same projection
    picks = pd.concat([__import__("review").draft_night(y)[0].assign(year=y) for y in YEARS])
    A = picks[picks.grp.isin(GROUPS)].merge(bp[["year", "pick", "rank"]], on=["year", "pick"], how="left")
    A = A[["year", "team", "pick", "grp", "rank", "pfr_player_id"]]
    AS = stack(A, f, "pfr_player_id", pshare)

    seasons = range(2017, 2026)  # follower has 4 drafts' worth from 2017
    def agg(X):
        X = X[X.season.isin(seasons)].groupby(["team", "season", "grp"])[["proj", "real"]].sum()
        idx = pd.MultiIndex.from_product([sorted(F.team.unique()), seasons, GROUPS], names=X.index.names)
        X = X.reindex(idx, fill_value=0.0).reset_index()
        X["room"] = X.grp.map(room)
        X["over"] = (X.proj - X.room).clip(lower=0)
        return X
    FX, AX = agg(FS), agg(AS)

    print("Room = median total snap share of a position group per team-season, 2014-25 "
          "(e.g. WR 3.0 = three full-time WRs)\n")
    out = []
    for g in GROUPS:
        a, b = FX[FX.grp == g], AX[AX.grp == g]
        out.append({"grp": g, "room": room[g],
                    "team_proj": b.proj.mean(), "team_p95": b.proj.quantile(.95),
                    "fol_proj": a.proj.mean(), "fol_p95": a.proj.quantile(.95), "fol_max": a.proj.max(),
                    "fol_real": a.real.mean(),
                    "fol_%room": a.proj.mean() / room[g] * 100,
                    "%seasons_over_team_max": (a.proj > b.proj.max()).mean() * 100,
                    "%seasons_over_room": (a.proj > room[g]).mean() * 100})
    T = pd.DataFrame(out).set_index("grp").round(2)
    print("Draftees' snap share per team-season, 2017-25 (own draftees from the last 4 drafts, R1 5):")
    print(T.to_string(), "\n")

    # value at risk: share of the follower's projected snaps beyond the most any real team's draftees took
    tmax = AX.groupby("grp").proj.quantile(.99)
    FX["excess"] = (FX.proj - FX.grp.map(tmax)).clip(lower=0)
    tot = FX.groupby("grp")[["proj", "excess"]].sum()
    tot["%excess"] = tot.excess / tot.proj * 100
    print("Follower projected snaps beyond the 99th pct of real teams' draftees (out of sample):")
    print(tot.round(1).to_string(), "\n")

    print("Picks by group, follower vs teams (2014-25, non-specialists):")
    print(pd.DataFrame({"follower": F.grp.value_counts(), "teams": A.grp.value_counts()}).fillna(0).astype(int).to_string(), "\n")

    ne = FX[(FX.team == "NE")].pivot_table(index="season", columns="grp", values="proj").round(1)
    print("NE follower: projected snap share of own draftees by season (room in header row below)")
    print(room.round(1).to_frame().T.to_string())
    print(ne.to_string())
    print("\nNE follower picks R1-4:")
    print(F[(F.team == "NE") & (F["round"] <= 4)][["year", "pick", "name", "grp", "rank"]].to_string(index=False))
    FX.to_parquet(OUT / "strict_follower_rooms.parquet")
    AX.to_parquet(OUT / "strict_follower_rooms_teams.parquet")


if __name__ == "__main__":
    if sys.argv[1:2] != ["haircut"]:
        {"draft": draft, "capacity": capacity}[sys.argv[1] if len(sys.argv) > 1 else "capacity"]()


def haircut():
    """Model titles the follower's picks lose if snaps beyond the room can't be played, drafts 2017-2022.
    Each team-season-group over its room: every draftee there keeps room / projected of his snaps
    (proportional; giving the snaps to the best players instead would lose a bit less). A player's loss =
    his projected-snap-weighted share lost over his seasons. Compared with the model's own expected gain
    over the team's real picks (review: value_rec - value_actual summed per team-draft is per-pick and
    not additive across a follower's draft, so here: follower titles minus the model's value of the team's
    real picks at the same need step)."""
    F = pd.read_parquet(FOLLOW)
    F = F[F.grp.isin(GROUPS)].reset_index(drop=True)
    sh = season_shares()
    pshare = sh.groupby(["pfr_player_id", "season"]).share.sum()
    room = rooms(sh)
    room = room[room.season.between(2014, 2025)].groupby("grp").share.median()
    f = expected_curve(board_players(), pshare)
    F["pid_"] = F.index
    rows = []
    for r in F.itertuples():
        e = f(r.grp, r.rank)
        for t in range(1, (5 if r.pick <= 32 else 4) + 1):
            rows.append({"pid_": r.pid_, "team": r.team, "season": r.year + t - 1, "grp": r.grp, "proj": e[min(t, 4) - 1]})
    X = pd.DataFrame(rows)
    tot = X.groupby(["team", "season", "grp"]).proj.transform("sum")
    X["keep"] = np.minimum(1.0, X.grp.map(room) / tot.where(tot > 0, 1))
    pl = X.groupby("pid_").apply(lambda d: 1 - (d.proj * d.keep).sum() / max(d.proj.sum(), 1e-9), include_groups=False)
    F["lost"] = F.pid_.map(pl).fillna(0)
    F["titles_lost"] = F.titles * F.lost
    G = F[F.year.between(2017, 2022)]
    rp = pd.read_parquet(OUT / "review_picks.parquet")
    rp = rp[rp.year.between(2017, 2022) & rp.grp.ne("SPEC")]
    team_val = rp.groupby(["year", "team"]).value_actual.sum()
    fol = G.groupby(["year", "team"])[["titles", "titles_lost"]].sum()
    J = fol.join(team_val).dropna()
    J["gap"] = J.titles - J.value_actual
    print("Drafts 2017-22, per team-draft, % of one title (model's own expected values):")
    print(f"  follower titles {J.titles.mean():.2f}, team's real picks {J.value_actual.mean():.2f}, "
          f"model's claimed gain {J.gap.mean():.2f}")
    print(f"  lost to room overflow {J.titles_lost.mean():.2f} "
          f"({J.titles_lost.mean() / J.gap.mean() * 100:.0f}% of the claimed gain); "
          f"gain after overflow {(J.gap - J.titles_lost).mean():.2f}")
    print(f"  team-drafts where the overflow wipes out the gain: {((J.gap - J.titles_lost) < 0).mean() * 100:.0f}%")
    by = G.groupby("grp")[["titles", "titles_lost"]].sum()
    by["%lost"] = by.titles_lost / by.titles * 100
    print("\nBy group (sum over 2017-22 follower picks):")
    print(by.round(1).sort_values("titles", ascending=False).to_string())
    by_r = G.assign(rb=pd.cut(G["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])).groupby("rb", observed=True)[["titles", "titles_lost"]].sum()
    by_r["%lost"] = by_r.titles_lost / by_r.titles * 100
    print("\nBy round:")
    print(by_r.round(1).to_string())
    ne = J.loc[(slice(None), "NE"), :]
    print("\nNE:")
    print(ne.round(2).to_string())


if __name__ == "__main__" and sys.argv[1:2] == ["haircut"]:
    haircut()
