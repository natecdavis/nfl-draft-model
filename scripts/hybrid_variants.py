"""Hybrid position pricing, follow-ups (Nov 17 2026): softer hybrids, and the 2023-25 classes.

scripts/hybrid_units_choice.py: taking the family-units (OL-friendly) recommendation in rounds 2-4 wherever it
differs from production tied on titles and won on starts / AV, but swung rounds 2-4 from WR-heavy to OL-heavy with no
EDGE / DL / LB / TE at all. Two checks:

1. Softer versions, rounds 2-4 only (round 1 and rounds 5-7 stay production):
     hybrid        the family recommendation wherever it differs (as before)
     blend         the best player on 0.5 x production value + 0.5 x family value
     near90/near80 the family recommendation only where production values it within 10% / 20% of its own top choice
2. Fresh classes: 2023-25 scored on seasons started and AV to date (titles need four seasons; second-contract pay
   mostly hasn't happened).

Harness: the review's own loop (review.review_year: leave-one-draft-out engine, the team's real roster, QB need and
earlier picks), run once per pricing (OUTCOME_UNITS single / family), saving every available player's value at every
pick 2014-2025 (data/derived/hybrid_variants/vals_<units>.parquet). Scoring as hybrid_units_choice: paired rule minus
production over all picks; recommendation minus the team where they disagree, each rec once per draft (mean); intervals
resample draft years.

Usage: OUTCOME_UNITS=single python scripts/hybrid_variants.py dump
       OUTCOME_UNITS=family python scripts/hybrid_variants.py dump
       python scripts/hybrid_variants.py report
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path[:0] = ["src", "scripts"]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402

YEARS = range(2014, 2026)
D = OUT / "hybrid_variants"
RULES = ["production", "hybrid", "blend", "near90", "near80"]


def one_year(year):
    import review as R
    from positions import blended, weights
    from trade import Context
    ctx, states, tables = Context(), R.States(), {}
    R.engine_for(year, tables)
    picks, _ = R.draft_night(year)
    b = R.board(year, picks)
    teams_st = {tm: states.state(tm, year) for tm in sorted(set(picks.team) | set(picks.original_team))}

    def table_for(tm):
        s = teams_st[tm]
        key = (s["start"], s["qb_from"], tuple(sorted(s["shift"].items())))
        if key not in tables:
            tables[key] = R.Table({"start": s["start"], "shift": dict(key[2]), "qb_from": key[1]}, ctx, fall_weight=R.FALL_W)
        return tables[key]

    W = weights(year)
    taken, rows = {}, []
    for _, p in picks.sort_values("pick").iterrows():
        tm, k = p.team, int(p.pick)
        tab = table_for(tm)
        cnt = taken.setdefault(tm, {})
        avail = b[(b.drafted_at.isna() | (b.drafted_at >= k)) & (b.grp_label != "SPEC")]
        pn = teams_st[tm]["p_need"]
        vals = [blended(tab.lookup, lab, g, min(cnt.get(g, 0), 2), r, k, pn, W)
                for lab, g, r in zip(avail.pos, avail.grp_label, avail["rank"])]
        rows.append(pd.DataFrame({"year": year, "pick": k, "cand": avail.index.to_numpy(), "name": avail["name"].to_numpy(),
                                  "cgrp": avail.grp.to_numpy(), "at": avail.drafted_at.to_numpy(), "v": np.array(vals) * 100}))
        if p.grp != "SPEC":
            cnt[p.grp] = cnt.get(p.grp, 0) + 1
    return year, pd.concat(rows, ignore_index=True)


def dump(workers=6):
    import outcomes_v2 as o
    D.mkdir(exist_ok=True)
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for y, df in ex.map(one_year, YEARS):
            out.append(df)
            print(f"{o.UNITS} {y}: {df.pick.nunique()} picks ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(out, ignore_index=True).to_parquet(D / f"vals_{o.UNITS}.parquet")


def choices() -> pd.DataFrame:
    S = pd.read_parquet(D / "vals_single.parquet")
    F = pd.read_parquet(D / "vals_family.parquet")[["year", "pick", "cand", "v"]]
    V = S.merge(F, on=["year", "pick", "cand"], suffixes=("_s", "_f"))
    assert len(V) == len(S) == len(F), "candidate sets differ between pricings"
    V["v_b"] = 0.5 * V.v_s + 0.5 * V.v_f
    g = V.groupby(["year", "pick"])
    top = {c: V.loc[g[c].idxmax()].set_index(["year", "pick"]) for c in ["v_s", "v_f", "v_b"]}
    smax = top["v_s"].v_s
    rp = pd.read_parquet(OUT / "review_picks.parquet").query("year in @YEARS")
    rp = rp[["year", "pick", "round", "team", "player", "grp", "rec_player"]].set_index(["year", "pick"])
    out = rp.copy()
    r24 = out["round"].between(2, 4)
    f_s = top["v_f"].v_s  # production's value of the family rec
    for rule in RULES:
        src = {"production": "v_s", "blend": "v_b"}.get(rule, "v_f")
        t = top[src]
        use = r24 & (t.cand != top["v_s"].cand)
        if rule.startswith("near"):
            use &= f_s >= (int(rule[4:]) / 100) * smax
        if rule == "production":
            use[:] = False
        for c, sc in [("player", "name"), ("grp", "cgrp"), ("at", "at")]:
            out[f"{rule}_{c}"] = np.where(use, t[sc], top["v_s"][sc])
        out[f"{rule}_sw"] = use
    out = out.reset_index()
    print(f"check: single-pricing top choice = stored production recommendation at "
          f"{(out.production_player == out.rec_player).mean():.1%} of picks")
    return out


def score(C: pd.DataFrame) -> pd.DataFrame:
    import review as R
    from trade import Context
    cache = OUT / "hybrid_variants" / "scored.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    st = pd.read_parquet(OUT / "review_states.parquet")
    ctx, car = Context(), R.Careers()
    for k in ("gs", "av", "pay"):
        C[f"{k}_team"] = [car.get(k, y, p, n) for y, p, n in zip(C.year, C.pick, C.player)]
    old = C[C.year <= 2022]
    for rule in RULES:
        for k in ("gs", "av", "pay"):
            C[f"{k}_{rule}"] = [car.get(k, y, a, n) for y, a, n in zip(C.year, C[f"{rule}_at"], C[f"{rule}_player"])]
        for u in ("single", "family"):
            x = R.add_realized_v2(old.assign(rec_player=old[f"{rule}_player"], rec_grp=old[f"{rule}_grp"],
                                             rec_drafted_at=old[f"{rule}_at"], best_available_grp=None,
                                             best_available_drafted_at=np.nan, best_available=None).copy(), st, ctx, units=u)
            C.loc[old.index, f"t{u[0]}_{rule}"] = x.real_model.to_numpy()
            if rule == "production":
                C.loc[old.index, f"t{u[0]}_team"] = x.real_team.to_numpy()
        print("scored", rule, flush=True)
    C.to_parquet(cache)
    return C


def boot(x, years, rng, reps=2000, fmt="+.3f"):
    x, years = np.asarray(x, float), np.asarray(years)
    ok = ~np.isnan(x)
    x, years = x[ok], years[ok]
    if len(x) == 0:
        return "n/a"
    ys = np.unique(years)
    by = {y: x[years == y] for y in ys}
    b = [np.concatenate([by[y] for y in rng.choice(ys, len(ys))]).mean() for _ in range(reps)]
    return f"{x.mean():{fmt}} ({np.percentile(b, 2.5):{fmt}} to {np.percentile(b, 97.5):{fmt}})"


def report():
    rng = np.random.default_rng(0)
    C = score(choices())
    C = C[C.grp.ne("SPEC")].copy()
    C["band"] = pd.cut(C["round"], [0, 1, 4, 7], labels=["R1", "R2-4", "R5-7"])
    pd.set_option("display.width", 220)
    r24 = C[C.band == "R2-4"]
    print("\nrounds 2-4 recommendations by group, 2014-25:")
    print(pd.DataFrame({r: r24[f"{r}_grp"].value_counts() for r in RULES}).fillna(0).astype(int).T.to_string())
    print("\nshare of rounds 2-4 picks changed:", {r: f"{r24[f'{r}_sw'].mean():.0%}" for r in RULES[1:]})
    Y = {"titles (single)": "ts", "titles (family)": "tf", "starts": "gs", "AV": "av", "2nd-contract pay": "pay"}
    for label, yrs in [("2014-22", range(2014, 2023)), ("2023-25 (starts / AV to date)", range(2023, 2026))]:
        d = C[C.year.isin(yrs)]
        print(f"\n== {label}: rule minus production, paired over all picks (rounds 2-4 picks in brackets) ==")
        for r in RULES[1:]:
            parts = []
            for k, c in Y.items():
                if d[f"{c}_{r}"].notna().sum() == 0 or (yrs.start >= 2023 and k in ("2nd-contract pay",)):
                    continue
                dd = d[f"{c}_{r}"] - d[f"{c}_production"]
                b = d.band == "R2-4"
                parts.append(f"{k} {boot(dd, d.year, rng)} [{boot(dd[b], d.year[b], rng)}]")
            print(f"  {r:7s} " + " | ".join(parts))
        print(f"  rec minus team where they disagree, rounds 2-4, each rec once per draft:")
        for r in RULES:
            parts = []
            for k, c in Y.items():
                if d[f"{c}_{r}"].notna().sum() == 0 or (yrs.start >= 2023 and k in ("2nd-contract pay",)):
                    continue
                x = d[(d.band == "R2-4") & (d[f"{r}_player"] != d.player)].assign(v=d[f"{c}_{r}"] - d[f"{c}_team"])
                x = x.groupby(["year", f"{r}_player"]).v.mean().reset_index()
                parts.append(f"{k} {boot(x.v, x.year, rng)}")
            print(f"  {r:10s} " + " | ".join(parts))


if __name__ == "__main__":
    dump() if sys.argv[1:2] == ["dump"] else report()
