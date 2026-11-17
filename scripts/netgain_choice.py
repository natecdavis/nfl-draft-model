"""Choice test with net-gain scoring (Nov 16 2026).

scripts/net_room_value.py and scripts/cap_side.py: per 1.0 of a rookie's expected own value, his team realizes field
gain at the position plus veteran spending saved net of his own cap, ~0.27 (seasons 1-4), against the ~0.55 surplus
the model credits; the redeployed cap buys nothing detectable elsewhere. The share realized differs by group
(EDGE 0.57 ... TE 0.10; overlapping CIs). Two questions:

1. Scoring. Score every choice (the model's, the team's) by realized titles x its group's net-gain share r_g (QB and
   specialists: the mean share, so they are neither favored nor penalized). Does the model's edge over teams survive
   when it is credited only what teams realize?
2. Rule. "netgain": non-QB table values x r_g / mean r (only the relative position values change), r_g refit leaving
   out the draft being chosen (cap_side.parquet regressions, seasons 1-4, drafts <= 2022 minus that year). Does it beat
   the current rule, paired, under raw and net-gain scoring? (Net-gain scoring uses full-sample shares, so it partly
   favors the netgain rule.)
Harness as scripts/quality_need.py (today's production model, the team's real roster and earlier picks, at every pick
2014-22).

Usage: python scripts/netgain_choice.py          (choices + scoring -> data/derived/netgain_choice.parquet)
       python scripts/netgain_choice.py report
"""
import sys

sys.path[:0] = ["src", "scripts"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from outlook import OUT  # noqa: E402

YEARS = range(2014, 2023)
OUTF = OUT / "netgain_choice.parquet"
G_OF = {"RB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "IDL": "DL", "ED": "EDGE", "LB": "LB",
        "CB": "DB", "S": "DB"}
CLIP = (0.05, 1.0)


def shares(leave_out=None) -> dict:
    """Net-gain share per cap_side group: (field + veteran spend saved - rookies' cap) / (1 - rookies' cap), seasons 1-4."""
    U = pd.read_parquet(OUT / "cap_side.parquet")
    U = U[U.vet_4.notna() & (U.year != leave_out)]
    out = {}
    for g, d in U.groupby("g"):
        b = {}
        for y in ["y", "vet", "class"]:
            m = smf.ols(f"{y}_4 ~ Eall_4 + high + mid + low + unk + v_last + spend_last + C(year)", d).fit()
            b[y] = m.params["Eall_4"]
        out[g] = float(np.clip((b["y"] - (b["vet"] + b["class"])) / (1 - b["class"]), *CLIP))
    return out


def main():
    from positions import blended, weights
    from review import RealizedV2, board, draft_night
    from simulator import get_table, situation
    S = pd.read_parquet(OUT / "review_states.parquet")
    full = shares()
    rbar = float(np.mean(list(full.values())))
    print("net-gain shares (all drafts 2014-22):", {k: round(v, 2) for k, v in full.items()}, "mean", round(rbar, 2))
    loo = {y: shares(y) for y in YEARS}
    rows = []
    for y in YEARS:
        picks, _ = draft_night(y)
        b = board(y, picks)
        W = weights(y)
        St = S[S.year == y].set_index("team")
        r_y = loo[y]
        mean_y = float(np.mean(list(r_y.values())))
        tabs, counts = {}, {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            s = St.loc[tm]
            if tm not in tabs:
                tabs[tm] = get_table(situation(s), year=y)
            tab = tabs[tm]
            cnt = counts.setdefault(tm, {})
            a = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
            pn = float(s.p_need)
            res = {"year": y, "pick": k, "round": int(p["round"]), "team": tm, "player": p.pfr_player_name, "grp": p.grp}
            for rule in ("current", "netgain"):
                def lk(g, level, r, pk, p_need=1.0, rule=rule):
                    v = tab.lookup(g, level, r, pk, p_need)
                    if rule == "netgain" and g in G_OF:
                        v *= r_y[G_OF[g]] / mean_y
                    return v
                vals = np.array([blended(lk, lab, g, min(cnt.get(g, 0), 2), r, k, pn, W)
                                 for lab, g, r in zip(a.pos, a.grp_label, a["rank"])])
                best = a.iloc[int(np.argmax(vals))]
                res.update({f"{rule}_player": best["name"], f"{rule}_grp": best.grp, f"{rule}_at": best.drafted_at})
            rows.append(res)
            if p.grp in G_OF or p.grp == "QB":
                cnt[p.grp] = cnt.get(p.grp, 0) + 1
        print(y, "choices", flush=True)
    D = pd.DataFrame(rows)
    rp = pd.read_parquet(OUT / "review_picks.parquet")[["year", "pick", "real_team", "rec_player"]]
    D = D.merge(rp, on=["year", "pick"], how="left")
    from trade import Context
    real = RealizedV2(Context())
    pneed = {(r.year, r.team): float(r.p_need) for r in S.itertuples()}
    sits = {(r.year, r.team): situation(r) for _, r in S.iterrows()}

    def score(r, g, at, nm):
        sit = sits[(r.year, r.team)]
        x = real.titles(sit, g, r.year, at, r.pick, nm)
        pn = pneed[(r.year, r.team)]
        if g == "QB" and pn < 1:
            sat = real.titles({**sit, "qb_from": None}, g, r.year, at, r.pick, nm, qb="rule")
            x = pn * x + (1 - pn) * sat
        return x
    for rule in ("current", "netgain"):
        D[f"real_{rule}"] = [score(r, getattr(r, f"{rule}_grp"), getattr(r, f"{rule}_at"), getattr(r, f"{rule}_player"))
                             for r in D.itertuples()]
        print("scored", rule, flush=True)
    D["share_rbar"] = rbar
    for who, gcol in [("team", "grp"), ("current", "current_grp"), ("netgain", "netgain_grp")]:
        D[f"r_{who}"] = [full.get(G_OF.get(g), rbar) for g in D[gcol]]
    D.to_parquet(OUTF)
    report(D)


def report(D=None):
    D = pd.read_parquet(OUTF) if D is None else D
    rng = np.random.default_rng(0)

    def boot(x, cl):
        x, cl = np.asarray(x, float), np.asarray(cl)
        ids, inv = np.unique(cl, return_inverse=True)
        sums, ns = np.bincount(inv, x), np.bincount(inv)
        bs = []
        for _ in range(2000):
            j = rng.integers(0, len(ids), len(ids))
            bs.append(sums[j].sum() / ns[j].sum())
        return f"{x.mean():+.3f} ({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f})"
    D = D[D.real_team.notna() & D.grp.ne("SPEC")].copy()
    D["rnd"] = pd.cut(D["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    D["cl"] = D.year.astype(str) + D.team
    print(f"\nCheck: current rule = stored review recommendation at {(D.current_player == D.rec_player).mean():.0%} of picks")
    for sc in ("raw", "net"):
        f = (lambda d, who: d[f"real_{who}"] if who != "team" else d.real_team) if sc == "raw" else \
            (lambda d, who: (d[f"real_{who}"] if who != "team" else d.real_team) * d[f"r_{who}"])
        print(f"\n{sc} scoring" + (" (titles x group net-gain share)" if sc == "net" else " (realized titles)") + ":")
        for rule in ("current", "netgain"):
            dis = D[D[f"{rule}_player"] != D.player].drop_duplicates(["year", f"{rule}_player"])
            parts = [f"{k}: {boot(f(g, rule) - f(g, 'team'), g.cl)}" for k, g in [("all", dis)] + list(dis.groupby("rnd", observed=True))]
            print(f"  model minus team where they disagree, {rule:8s} " + " | ".join(parts))
        same = (D.netgain_player == D.current_player).mean()
        parts = [f"{k}: {boot(f(g, 'netgain') - f(g, 'current'), g.cl)}" for k, g in [("all", D)] + list(D.groupby("rnd", observed=True))]
        print(f"  netgain minus current, paired, all picks (same choice {same:.0%}): " + " | ".join(parts))
    for rule in ("current", "netgain"):
        print(f"\n{rule} picks by group, R1-4:", D[D["round"] <= 4][f"{rule}_grp"].value_counts().to_dict())


if __name__ == "__main__":
    report() if sys.argv[1:2] == ["report"] else main()
