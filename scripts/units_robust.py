"""Complaint #2: is the model's position mix an artifact of the units choice?

Production review (OUTCOME_UNITS=single, leave-one-draft-out) vs the same review under family units
(scripts/holdout_review.py with OUTCOME_UNITS=family -> data/derived/holdout/review_picks_lodo_unitsfamily.parquet).
At each pick of 2014-2022 the production recommendation is
  same player   both units recommend the same player
  same position different player, same position (board label)
  flips         the position changes with the units choice
Then production's recommendations are scored against teams' picks within each class (each rec once per
draft and band, averaged over his picks: review.once_per_draft "mean"), on both title yardsticks (single and
family units; fair QB scoring) and on starts, AV and second-contract pay. Intervals: bootstrap over draft years.

Run:  .venv/bin/python scripts/units_robust.py        (rescoring cached in data/derived/units_robust.parquet)
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from outlook import OUT  # noqa: E402

YEARS = range(2014, 2023)
CACHE = OUT / "units_robust.parquet"


def load():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import review as R
    from trade import Context
    st = pd.read_parquet(OUT / "review_states.parquet")
    S = pd.read_parquet(OUT / "review_picks.parquet").query("year in @YEARS").reset_index(drop=True)
    F = pd.read_parquet(OUT / "holdout" / "review_picks_lodo_unitsfamily.parquet").query("year in @YEARS")
    ctx = Context()
    for d in (S, F):  # both title yardsticks for both versions
        for u in ("single", "family"):
            x = R.add_realized_v2(d.copy(), st, ctx, units=u)
            for c in ("real_team", "real_model"):
                d[f"{c}_{u}"] = x[c]
    keep = ["year", "pick", "rec_player", "rec_grp", "rec_drafted_at", "value_rec", "real_model_single", "real_model_family"]
    m = S.merge(F[keep], on=["year", "pick"], suffixes=("", "_f"))
    car = R.Careers()
    for k in ("gs", "av", "pay"):
        for sfx in ("", "_f"):
            m[f"{k}_model{sfx}"] = [car.get(k, y, a, n) for y, a, n in zip(m.year, m["rec_drafted_at" + sfx], m["rec_player" + sfx])]
        m[f"{k}_team"] = [car.get(k, y, p, n) for y, p, n in zip(m.year, m.pick, m.player)]
    m["cls"] = np.select([m.rec_player == m.rec_player_f, m.rec_grp == m.rec_grp_f], ["same player", "same position"], "flips")
    m.to_parquet(CACHE)
    return m


def boot(d, col, rng, reps=2000):
    """Mean of col with a 95% interval, resampling draft years."""
    if len(d) == 0:
        return "n/a"
    ys = d.year.unique()
    by = {y: d.loc[d.year == y, col].to_numpy(float) for y in ys}
    b = []
    for _ in range(reps):
        x = np.concatenate([by[y] for y in rng.choice(ys, len(ys))])
        b.append(x.mean())
    return f"{d[col].mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


def main():
    import review as R
    m = load()
    m["band"] = R.band_of(m["round"])
    rng = np.random.default_rng(0)
    print("Production recommendation vs family-units recommendation, 2014-2022, all picks:")
    print(pd.crosstab(m.band, m.cls, normalize="index").round(2).to_string(), "\n")
    for b in ("R1", "R2-4", "R5-7"):
        x = m[m.band == b]
        mix = pd.DataFrame({"single": x.rec_grp.value_counts(normalize=True), "family": x.rec_grp_f.value_counts(normalize=True)}).fillna(0)
        print(f"{b} recommendation mix:", " ".join(f"{g} {r.single:.0%}/{r.family:.0%}" for g, r in mix.sort_values("single", ascending=False).iterrows()))
    flips = m[m.cls == "flips"]
    print("\nMost common flips (single -> family):")
    print((flips.rec_grp + " -> " + flips.rec_grp_f).value_counts().head(10).to_string(), "\n")

    # production recs scored against teams, by class
    for u in ("single", "family"):
        m[f"d_t_{u}"] = m[f"real_model_{u}"] - m[f"real_team_{u}"]
    for k in ("gs", "av", "pay"):
        m[f"d_{k}"] = m[f"{k}_model"] - m[f"{k}_team"]
    m = m[m.real_team_single.notna()]
    once = R.once_per_draft(m)  # each rec once per draft and band (production rec)
    print("Production recommendations minus team's pick where they disagree (each rec once), by class:")
    for b in ("R1", "R2-4", "R5-7"):
        for c in ("same player", "same position", "flips", "all"):
            o = once[(once.band == b) & ((once.cls == c) | (c == "all"))]
            print(f"  {b:5s} {c:13s} n={len(o):4d} | titles single {boot(o, 'd_t_single', rng)} | family {boot(o, 'd_t_family', rng)}"
                  f" | starts {o.d_gs.mean():+.2f} AV {o.d_av.mean():+.1f} pay {o.d_pay.mean():+.2f}")
    # where they flip: which of the two recommendations did better?
    f = m[m.cls == "flips"].copy()
    f["band"] = R.band_of(f["round"])
    f = f.drop_duplicates(["year", "rec_player", "rec_player_f", "band"])
    print("\nWhere the position flips: single rec minus family rec (each pair once):")
    for b in ("R1", "R2-4", "R5-7"):
        x = f[f.band == b].copy()
        x["dd_s"] = x.real_model_single - x.real_model_single_f
        x["dd_f"] = x.real_model_family - x.real_model_family_f
        for k in ("gs", "av", "pay"):
            x[f"dd_{k}"] = x[f"{k}_model"] - x[f"{k}_model_f"]
        print(f"  {b:5s} n={len(x):4d} | titles single {boot(x, 'dd_s', rng)} | family {boot(x, 'dd_f', rng)}"
              f" | starts {x.dd_gs.mean():+.2f} AV {x.dd_av.mean():+.1f} pay {x.dd_pay.mean():+.2f}")


if __name__ == "__main__":
    main()
