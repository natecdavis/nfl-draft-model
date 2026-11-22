"""Contract model: position market and class supply (#5), draft pedigree and talent for rookie-deal players (#6).

Setting = the tool's (lag 2 + weeks 1-11 of the season in progress, fa_market.FEATS_PARTIAL), inputs past-only
(fa_market.VALUE_COL = value_ours_f; scripts/forecast_benchmark.py). Two models:
  UFA - free-agent model, UFA signings 2015-26 (fa_market.models(2))
  EXT - extension model, fit on UFA + extensions (ufa flag), scored on extensions (fa_market.ext_models())
New inputs, all known by November of the season before the signing:
  #5 market:   mkt_top - mean APY (share of cap) of the 5 biggest UFA / extension deals at his position signed in the
                         two prior league years (Y-2, Y-1); fine position for DB / OL (CB, S, T, G, C)
               mkt_med - median APY of all such deals >= 0.5% of cap at his position
               class_n - number of UFA signings in his group in year Y (the class as it turned out: slightly leaky,
                         live use would take fa_market.free_agents)
               class_rank - his rank in that class by last full season's value (1 = best), class_pct the percentile
  #6 pedigree: pick (draft pick, 300 undrafted), q1 (past-only talent percentile, last full season), vmax (career peak
               value before the signing), nseas (seasons with snaps)
Checks: leave-one-signing-year-out and forward-chained (fit on earlier years, predict 2020-26, score 2022-26).
R2 of log APY, median miss after scaling to the mean, and bias (actual / predicted, after the same scaling) by
segment: rookie deal before (prev_rookie), young stars (prev_rookie and top-20% value), age bands, groups.
Pre-set rule: adopt a variant for a model if it beats base on R2 AND median miss in BOTH checks, and no segment's
|bias - 1| grows by more than 0.03 in the forward check (segments with n >= 30).
Run: .venv/bin/python scripts/contract_market.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import fa_market as fm  # noqa: E402
import veteran as vt  # noqa: E402

fm.VALUE_COL = "value_ours_f"
MKT = ["mkt_top", "mkt_med", "class_n", "class_rank", "class_pct"]
PED = ["pick", "q1", "vmax", "nseas"]
VARIANTS = {"base": fm.FEATS_PARTIAL, "+market (#5)": fm.FEATS_PARTIAL + MKT, "+pedigree (#6)": fm.FEATS_PARTIAL + PED,
            "+both": fm.FEATS_PARTIAL + MKT + PED}


def poskey(p, grp, fp):
    return fp.get(p, grp) if grp in ("DB", "OL") else grp


def add_features(D):
    H = fm.history()
    V, _, grp, _ = fm._values("value_ours")
    fp = fm.fine_positions()
    # deals by (position, year)
    deals = {}
    for p, h in H.items():
        g = grp.get(p)
        if g is None:
            continue
        for (y, kind, apy, *_r) in h:
            if kind in ("UFA", "Extension", "Franchise") and apy >= fm.MIN_APY:
                deals.setdefault((poskey(p, g, fp), y), []).append(apy)
    def mkt(pk, Y):
        a = sorted(sum((deals.get((pk, y), []) for y in (Y - 2, Y - 1)), []), reverse=True)
        return (np.mean(a[:5]) if a else np.nan), (np.median(a) if a else np.nan)
    m = [mkt(poskey(p, g, fp), Y) for p, g, Y in zip(D.pfr_id, D.grp, D.Y)]
    D["mkt_top"], D["mkt_med"] = [x[0] for x in m], [x[1] for x in m]
    # class: UFA signers in Y, ranked within group by last full season's value (lag 2: season Y-2, past-only)
    Vf = fm._values("value_ours_f")[0]
    cls = pd.DataFrame([(p, y, grp.get(p)) for p, h in H.items() for (y, kind, apy, *_r) in h
                        if kind == "UFA" and apy >= fm.MIN_APY], columns=["pfr_id", "Y", "g"]).drop_duplicates(["pfr_id", "Y"])
    cls["v"] = [Vf.get((p, y - 2), 0.0) for p, y in zip(cls.pfr_id, cls.Y)]
    n = cls.groupby(["Y", "g"]).size().to_dict()
    ranked = {(y, g): np.sort(G.v.to_numpy())[::-1] for (y, g), G in cls.groupby(["Y", "g"])}
    D["class_n"] = [n.get((Y, g), 0) for Y, g in zip(D.Y, D.grp)]
    D["class_rank"] = [1 + int((ranked.get((Y, g), np.array([])) > v).sum()) for Y, g, v in zip(D.Y, D.grp, D.v1)]
    D["class_pct"] = D.class_rank / D.class_n.clip(lower=1)
    # pedigree / talent
    pl = pd.read_parquet(vt.RAW / "players.parquet").dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")
    pick = dict(zip(pl.pfr_id, pl.draft_pick))
    st = vt.season_table()
    Q = st.set_index(["pfr_id", "season"]).q_f.to_dict()
    seas = st.groupby("pfr_id").season.apply(sorted).to_dict()
    D["pick"] = [pick.get(p) if pd.notna(pick.get(p)) else 300.0 for p in D.pfr_id]
    D["q1"] = [Q.get((p, Y - 2), np.nan) for p, Y in zip(D.pfr_id, D.Y)]
    D["vmax"] = [max([Vf.get((p, s), 0.0) for s in seas.get(p, []) if s <= Y - 2], default=0.0) for p, Y in zip(D.pfr_id, D.Y)]
    D["nseas"] = [sum(1 for s in seas.get(p, []) if s <= Y - 2) for p, Y in zip(D.pfr_id, D.Y)]
    return D


def score(P, col):
    e = np.log(P.apy) - np.log(P[col])
    r2 = 1 - (e ** 2).sum() / ((np.log(P.apy) - np.log(P.apy).mean()) ** 2).sum()
    rel = np.abs(P.apy / (P[col] * (P.apy.sum() / P[col].sum())) - 1)
    return r2, np.median(rel)


def segments(P):
    P = P.assign(sc=P.pred * P.apy.sum() / P.pred.sum())
    top = P.v1 >= P.groupby("grp").v1.transform(lambda x: x.quantile(0.8))
    segs = {"rookie deal before": P.prev_rookie == 1, "young stars": (P.prev_rookie == 1) & top,
            "age <26": P.age < 26, "age 26-28": P.age.between(26, 29, inclusive="left"),
            "age 29-30": P.age.between(29, 31, inclusive="left"), "age 31+": P.age >= 31}
    segs.update({f"grp {g}": P.grp == g for g in fm.GROUPS})
    return {k: (int(m.sum()), P.apy[m].sum() / P.sc[m].sum()) for k, m in segs.items()}


def run(D, feats, check, score_rows):
    out = []
    years = sorted(D.Y.unique()) if check == "leave-year-out" else range(2020, 2027)
    for Y in years:
        tr = D[D.Y != Y] if check == "leave-year-out" else D[D.Y < Y]
        te = D[D.Y == Y].copy()
        te["pred"] = np.exp(fm.fit(tr, feats=feats)["log_apy"].predict(te[feats]))
        out.append(te)
    P = pd.concat(out)
    P = P[score_rows(P)]
    if check != "leave-year-out":
        P = P[P.Y >= 2022]
    return P


def main():
    pd.set_option("display.width", 220)
    models = {"UFA": (fm.training(2), lambda P: P.ufa == 1),
              "EXT": (fm.training(2, kinds=("UFA", "Extension")), lambda P: P.ufa == 0)}
    for mname, (D, sel) in models.items():
        D = add_features(D)
        rows, segs = [], {}
        for vname, feats in VARIANTS.items():
            for check in ("leave-year-out", "forward 2022-26"):
                P = run(D, feats, check, sel)
                r2, med = score(P, "pred")
                rows.append({"variant": vname, "check": check, "n": len(P), "R2": r2, "median miss": med})
                if check.startswith("forward"):
                    segs[vname] = segments(P)
        R = pd.DataFrame(rows)
        print(f"\n{mname} model")
        print(R.round(3).to_string(index=False))
        S = pd.DataFrame({v: {k: f"{b:.2f} ({n})" for k, (n, b) in s.items()} for v, s in segs.items()})
        print("  bias actual/predicted by segment, forward check (n)")
        print(S.to_string())
        base = R[R.variant == "base"].set_index("check")
        for vname in list(VARIANTS)[1:]:
            v = R[R.variant == vname].set_index("check")
            better = all((v.loc[c, "R2"] > base.loc[c, "R2"]) and (v.loc[c, "median miss"] < base.loc[c, "median miss"])
                         for c in base.index)
            worse = [k for k, (n, b) in segs[vname].items()
                     if n >= 30 and abs(b - 1) > abs(segs["base"][k][1] - 1) + 0.03]
            print(f"  {vname}: beats base both checks {better}; segments worse >0.03: {worse or 'none'} -> "
                  f"{'ADOPT' if better and not worse else 'not adopted'}")


if __name__ == "__main__":
    main()
