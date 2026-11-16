"""Quality-weighted roster need: discount a rookie by the good incumbents at his position, not by bodies.

From scripts/displacement.py: rookies lose playing time to high-quality incumbents (DARKO-for-PFF talent tercile
within group), somewhat to middling ones, not to poor ones; pay and draft capital add little. Here the effect is
measured on value: rookie's cfdarko value, years 1-4 (classes 2014-22) ~ Poisson(controls + incumbent
starter-equivalents by quality tier), controls group x log pick, group x log rank, off-board, earlier rookies at
the group this draft. Multiplier for a team's room: m = exp(sum_t b_t (inc_t - median_t[group])), so a typical room
is 1 (the model's outcome tables come from rookies landing in typical rooms); clipped to 0.6-1.4. Coefficients are
refit leaving out the draft being scored. "count" = the same with one total (bodies only), for comparison.

Choice test, as scripts/need_choice.py but on today's production model (simulator tables, QB need probability,
within-draft step, position blending): at every pick 2014-22, on the team's real roster and its real earlier picks,
the best available player under "current" (no roster term), "count" and "quality". Non-QB value at multiplier x:
quadratic in playing time through cost-only (0), x0.88 and x1 table values (scripts/cross_year_need.py). QBs unchanged.
Scoring: realized titles (review.RealizedV2; recommended QBs by the fair rule), raw, and adjusted for landing spot:
a recommended player's real career is scaled by m(this team) / m(team that drafted him) with the quality
coefficients (his career happened in another room; favors the quality rule somewhat: circular). Model minus team
where they disagree (each recommended player once per draft), and each rule vs current, paired, all picks.

Run:  python scripts/quality_need.py          (choices + scoring -> data/derived/quality_need.parquet)
      python scripts/quality_need.py report
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from outlook import GROUPS, OUT, RAW  # noqa: E402

YEARS = range(2014, 2023)
TIERS = ["high", "mid", "low", "unk"]
WD = (1.0, 0.95, 0.88)
CLIP = (0.6, 1.4)
OUTF = OUT / "quality_need.parquet"


def rooms():
    I = pd.read_parquet(OUT / "displacement_incumbents.parquet")
    X = I.pivot_table(index=["year", "team", "grp"], columns="qt", values="se", aggfunc="sum", fill_value=0.0)
    X = X.reindex(columns=TIERS, fill_value=0.0).add_prefix("inc_")
    X["inc_all"] = X.sum(axis=1)
    return X.reset_index()


def fit_data():
    R = pd.read_parquet(OUT / "roster_need_rookies.parquet")
    R = R[R.year.between(2014, 2022) & R.v14.notna()].copy()
    R["lp"], R["lr"] = np.log(R.pick + 10), np.log(R.board_rank.fillna(330) + 10)
    R["off"] = R.board_rank.isna().astype(int)
    R = R.sort_values("pick")
    R["earlier"] = R.groupby(["year", "team", "grp"]).cumcount()
    R = R.merge(rooms(), on=["year", "team", "grp"], how="left").fillna({f"inc_{t}": 0.0 for t in TIERS + ["all"]})
    R["cl"] = R.year.astype(str) + R.team
    return R


CTRL = "C(grp) + C(grp):lp + C(grp):lr + off + earlier"
TERMS = {"count": ["inc_all"], "quality": ["inc_high", "inc_mid", "inc_low"]}


def coefs(R, leave_out=None):
    d = R if leave_out is None else R[R.year != leave_out]
    out = {}
    for rule, xs in TERMS.items():
        m = smf.glm(f"v14 ~ {CTRL} + " + " + ".join(xs), d, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.cl)[0]})
        out[rule] = (m.params[xs].to_dict(), m.bse[xs].to_dict())
    return out


def multipliers(B, C, med):
    """m by (year, team, grp) for each rule. B: rooms(); C: {year: coefs}; med: group medians of each inc col."""
    out = {}
    for rule, xs in TERMS.items():
        for r in B.itertuples():
            if r.year not in C:
                continue
            b = C[r.year][rule][0]
            z = sum(b[x] * (getattr(r, x) - med.loc[r.grp, x]) for x in xs)
            out[(rule, r.year, r.team, r.grp)] = float(np.clip(np.exp(z), *CLIP))
    return out


def main():
    from cross_year_need import cost_only, quad
    from positions import blended, weights
    from review import OFF_BOARD_RANK, RealizedV2, board, draft_night
    from simulator import get_table, situation
    R = fit_data()
    full = coefs(R)
    print("Poisson coefficients on rookie value y1-4 per incumbent starter-equivalent (all drafts 2014-22):")
    for rule, (b, se) in full.items():
        print(f"  {rule:8s} " + "  ".join(f"{k} {b[k]:+.3f} ({se[k]:.3f})" for k in b))
    C = {y: coefs(R, y) for y in YEARS}
    B = rooms()
    med = B[B.year.between(2014, 2022)].groupby("grp")[[f"inc_{t}" for t in TIERS + ["all"]]].median()
    M = multipliers(B, C, med)
    mq = pd.Series({k[1:]: v for k, v in M.items() if k[0] == "quality"})
    print(f"\nquality multipliers 2014-22: p10 {mq.quantile(.1):.2f}, median {mq.median():.2f}, p90 {mq.quantile(.9):.2f}")

    S = pd.read_parquet(OUT / "review_states.parquet")
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    import roster_need as rn
    DR = {(y, p): rn.team(t) for y, p, t in zip(dp.season, dp.pick, dp.team)}
    rows = []
    for y in YEARS:
        picks, _ = draft_night(y)
        b = board(y, picks)
        W = weights(y)
        St = S[S.year == y].set_index("team")
        tabs, c0s, counts = {}, {}, {}
        for _, p in picks.sort_values("pick").iterrows():
            tm, k = p.team, int(p.pick)
            s = St.loc[tm]
            sit = situation(s)
            if tm not in tabs:
                tabs[tm] = get_table(sit, year=y)
                c0s[tm] = cost_only(sit, sorted(int(x) for x in picks[picks.team == tm].pick))
            tab, c0 = tabs[tm], c0s[tm]
            cnt = counts.setdefault(tm, {})
            a = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
            pn = float(s.p_need)
            res = {"year": y, "pick": k, "round": int(p["round"]), "team": tm, "player": p.pfr_player_name, "grp": p.grp}
            for rule in ("current", "count", "quality"):
                def lk(g, level, r, pk, p_need=1.0, rule=rule):
                    m = 1.0 if rule == "current" or g == "QB" else M.get((rule, y, tm, g), 1.0)
                    if m == 1.0:
                        return tab.lookup(g, level, r, pk, p_need)
                    return quad(c0[pk], tab.lookup(g, 2, r, pk), tab.lookup(g, 0, r, pk), WD[level] * m)
                vals = np.array([blended(lk, lab, g, min(cnt.get(g, 0), 2), r, k, pn, W)
                                 for lab, g, r in zip(a.pos, a.grp_label, a["rank"])])
                i = int(np.argmax(vals))
                best = a.iloc[i]
                res.update({f"{rule}_player": best["name"], f"{rule}_grp": best.grp, f"{rule}_at": best.drafted_at,
                            f"{rule}_rank": best["rank"], f"{rule}_m": M.get((rule, y, tm, best.grp), 1.0)})
            rows.append(res)
            if p.grp in GROUPS:
                cnt[p.grp] = cnt.get(p.grp, 0) + 1
        print(y, "choices", flush=True)
    D = pd.DataFrame(rows)
    rp = pd.read_parquet(OUT / "review_picks.parquet")[["year", "pick", "real_team", "rec_player"]]
    D = D.merge(rp, on=["year", "pick"], how="left")

    from trade import Context
    real = RealizedV2(Context())
    pneed = {(r.year, r.team): float(r.p_need) for r in S.itertuples()}
    sits = {(r.year, r.team): situation(r) for _, r in S.iterrows()}

    def score(r, g, at, nm, sc):
        sit = sits[(r.year, r.team)]
        x = real.titles(sit, g, r.year, at, r.pick, nm, scale=sc)
        pn = pneed[(r.year, r.team)]
        if g == "QB" and pn < 1:
            sat = real.titles({**sit, "qb_from": None}, g, r.year, at, r.pick, nm, scale=sc, qb="rule")
            x = pn * x + (1 - pn) * sat
        return x
    for rule in ("current", "count", "quality"):
        for adj in (False, True):
            out = []
            for r in D.itertuples():
                g, at, nm = getattr(r, f"{rule}_grp"), getattr(r, f"{rule}_at"), getattr(r, f"{rule}_player")
                sc = 1.0
                if adj and pd.notna(at) and g != "QB":
                    sc = M.get(("quality", r.year, r.team, g), 1.0) / M.get(("quality", r.year, DR.get((r.year, int(at))), g), 1.0)
                out.append(score(r, g, at, nm, sc))
            D[f"real_{rule}" + ("_adj" if adj else "")] = out
        print("scored", rule, flush=True)
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
    print("\nModel minus team where they disagree (titles, % of one; each rec once per draft):")
    for rule in ("current", "count", "quality"):
        for adj in ("", "_adj"):
            dis = D[D[f"{rule}_player"] != D.player].drop_duplicates(["year", f"{rule}_player"])
            parts = [f"{rn_}: {boot(g[f'real_{rule}{adj}'] - g.real_team, g.cl)}"
                     for rn_, g in [("all", dis)] + list(dis.groupby("rnd", observed=True))]
            print(f"  {rule:8s} {'adjusted' if adj else 'raw':8s} " + " | ".join(parts))
    print("\nPaired vs current, all picks:")
    for rule in ("count", "quality"):
        for adj in ("", "_adj"):
            same = (D[f"{rule}_player"] == D.current_player).mean()
            parts = [f"{rn_}: {boot(g[f'real_{rule}{adj}'] - g[f'real_current{adj}'], g.cl)}"
                     for rn_, g in [("all", D)] + list(D.groupby("rnd", observed=True))]
            print(f"  {rule:8s} {'adjusted' if adj else 'raw':8s} same choice {same:.0%} | " + " | ".join(parts))
    for rule in ("current", "count", "quality"):
        print(f"\n{rule} picks by group, R1-4:", D[D["round"] <= 4][f"{rule}_grp"].value_counts().to_dict())


if __name__ == "__main__":
    report() if sys.argv[1:2] == ["report"] else main()
