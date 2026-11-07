"""Do earlier decisions hold on the new yardsticks? (Nov 2026)

Two scoring fixes came after most tests: realized titles for 2014-2015 undrafted players (review.udfa_pre2016_labels;
before, all zero, e.g. La'el Collins) and starts / AV / second-contract pay for undrafted players in every year
(review.Careers; before, zero). This rescores the saved leave-one-draft-out reviews behind each decision on both
yardsticks, without rerunning any engine:

  realized titles  recomputed only where the player is a 2014-2015 undrafted player now labeled, at the cost /
                   WR settings the file was built with (detected by reproducing its stored values on other rows)
  starts, AV, pay  old: draft records by (year, pick); new: review.Careers

Scorecard as everywhere: model minus team where they disagree, each recommendation once per draft, by round.
Board deference (scripts/defer_test.py) and the QB reach penalty (scripts/qb_reach.py) are rerun on rescored copies
of their candidate files (their saved outputs are restored afterwards).

Run:  python scripts/rescore_check.py
"""
import io
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import pick_value as pv  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from review import Careers, RealizedV2, _norm, band_of, once_per_draft, udfa_pre2016_labels  # noqa: E402
from trade import Context  # noqa: E402

pd.set_option("display.width", 250)
HD = OUT / "holdout"
DECISIONS = [
    ("Fall shading (not adopted)", ["review_picks_lodo.parquet", "review_picks_lodo_fw1.0.parquet", "review_picks_lodo_fw1.2.parquet"]),
    ("Cap cost (0.10 adopted)", ["review_picks_lodo_cost0.19.parquet", "review_picks_lodo_cost0.1.parquet", "review_picks_lodo_cost0.parquet"]),
    ("WR x0.75 (not adopted)", ["review_picks_lodo_cost0.1.parquet", "review_picks_lodo_wr0.75.parquet"]),
    ("Comparison-career correction (adopted)", ["review_picks_lodo_cost0.1.parquet", "review_picks_lodo_ccll.parquet"]),
    ("Board-less classes / monotone (drop + min adopted)", ["review_picks_lodo_ccll.parquet", "review_picks_lodo_nbdrop.parquet",
                                                           "review_picks_lodo_nbdrop_monoiso.parquet", "review_picks_lodo_nbdrop_monomin.parquet"]),
]
SETTINGS = [(0.10, 1.0), (0.19, 1.0), (0.0, 1.0), (0.10, 0.75)]
TRIPLES = [("real_team", "grp", "pick", "player"), ("real_model", "rec_grp", "rec_drafted_at", "rec_player"),
           ("real_consensus", "best_available_grp", "best_available_drafted_at", "best_available")]
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return f"{x.mean():+.2f} ({np.percentile(b, 2.5):+.2f} to {np.percentile(b, 97.5):+.2f})"


class Scorer:
    def __init__(self):
        self.car = Careers()
        dp = pd.read_parquet(RAW / "draft_picks.parquet")
        po = pd.read_parquet(OUT / "pick_outcomes.parquet")
        self.old = {"gs": dict(zip(zip(dp.season, dp.pick), dp.seasons_started)),
                    "av": dict(zip(zip(dp.season, dp.pick), dp.w_av)),
                    "pay": {k: v * 100 for k, v in zip(zip(po.draft_year, po.pick), po.best_pay)}}
        pre = udfa_pre2016_labels()
        self.affected = {(int(y), _norm(n)) for y, n in zip(pre.draft_year, pre.player_name)}
        S = pd.read_parquet(OUT / "review_states.parquet")
        self.sits = {(r.year, r.team): {"start": r.start, "shift": r.shift if isinstance(r.shift, dict) else eval(r.shift),
                                        "qb_from": None if pd.isna(r.qb_from) else int(r.qb_from)} for r in S.itertuples()}
        self.ctx, self.RV = Context(), None

    def realized(self, settings):
        pv.COST_RATE, pv.WR_FACTOR = settings
        if self.RV is None:
            self.RV = RealizedV2(self.ctx)
        return self.RV

    def detect(self, d):
        """The (cost, WR) settings that reproduce the file's stored realized titles on unaffected rows."""
        s = d[d.year.between(2016, 2022) & d.real_model.notna() & d.rec_drafted_at.notna()].sample(25, random_state=0)
        err = {}
        for st in SETTINGS:
            RV = self.realized(st)
            got = np.array([RV.titles(self.sits[(r.year, r.team)], r.rec_grp, r.year, r.rec_drafted_at, r.pick, r.rec_player)
                            for r in s.itertuples()])
            err[st] = np.abs(got - s.real_model.to_numpy()).mean()
        best = min(err, key=err.get)
        self.last_err = err[best]
        return best if err[best] < 0.05 else None

    def rescore_titles(self, d):
        d = d.copy()
        st = self.detect(d)
        n = 0
        if st is None:
            return d, None, 0
        RV = self.realized(st)
        for col, g, at, nm in TRIPLES:
            m = d.year.isin([2014, 2015]) & d[at].isna() & np.array([(y, _norm(x or "")) in self.affected for y, x in zip(d.year, d[nm])])
            for i in d.index[m]:
                r = d.loc[i]
                d.at[i, col] = RV.titles(self.sits[(r.year, r.team)], r[g], r.year, r[at], r.pick, r[nm])
                n += 1
        pv.COST_RATE, pv.WR_FACTOR = SETTINGS[0]
        return d, st, n

    def card(self, d, new):
        d = d[(d.year <= 2022) & d.real_team.notna()].copy()
        d["t"] = d.real_model - d.real_team
        for m in ("gs", "av", "pay"):
            if new:
                d[m] = [self.car.get(m, y, a, rn) - self.car.get(m, y, k, tn)
                        for y, a, rn, k, tn in zip(d.year, d.rec_drafted_at, d.rec_player, d.pick, d.player)]
            else:
                g = lambda y, a: 0.0 if pd.isna(a) else float(np.nan_to_num(self.old[m].get((y, int(a)), 0.0)))
                d[m] = [g(y, a) - g(y, k) for y, a, k in zip(d.year, d.rec_drafted_at, d.pick)]
        # each rec once per draft and band: "new" averages over his picks; "old" at his first pick (as before)
        o = (once_per_draft(d) if new else
             d[d.player != d.rec_player].sort_values(["year", "pick"]).drop_duplicates(["year", "rec_player"])
             .assign(band=lambda x: band_of(x["round"])))
        rows = {}
        for b in ["R1", "R2-4", "R5-7"]:
            x = o[o.band == b]
            rows[b] = {"titles": x.t.mean(), "starts": x.gs.mean(), "AV": x.av.mean(), "pay": x.pay.mean(), "n": len(x)}
        return pd.DataFrame(rows).T


def reviews():
    sc, cache, out = Scorer(), {}, []
    for label, files in DECISIONS:
        print(f"\n=== {label} ===")
        for f in files:
            if f not in cache:
                d0 = pd.read_parquet(HD / f)
                d1, st, n = sc.rescore_titles(d0)
                cache[f] = (sc.card(d0, False), sc.card(d1, True), st, n, sc.last_err)
            old, new, st, n, _ = cache[f]
            print(f"-- {f}  (settings cost {st[0] if st else '?'}, WR {st[1] if st else '?'}, mean gap to stored "
                  f"{cache[f][4]:.3f}; {n} realized cells rescored)")
            t = pd.concat({"old": old, "new": new}, axis=1)
            print(t.round(2).to_string())
            for b in ["R1", "R2-4", "R5-7"]:
                out.append({"decision": label, "file": f, "band": b, **{f"{k}_old": old.at[b, k] for k in old.columns},
                            **{f"{k}_new": new.at[b, k] for k in new.columns}})
    pd.DataFrame(out).to_parquet(HD / "rescore_check.parquet")


def candidates_tests():
    """Board deference and QB reach on rescored candidates (gs / av via Careers; realized for 2014-15 undrafted)."""
    import defer_test
    import qb_reach
    sc = Scorer()
    S = pd.read_parquet(OUT / "review_states.parquet")
    for mod, f, outf in [(defer_test, "defer_candidates.parquet", "defer_test.parquet"),
                         (qb_reach, "qb_reach_candidates.parquet", "qb_reach.parquet")]:
        C = pd.read_parquet(HD / f)
        # settings: reproduce stored 'real' on a sample of drafted 2016-2022 rows
        s = C[C.year.between(2016, 2022) & C.drafted_at.notna() & C.real.notna()].sample(25, random_state=0)
        sit = {(r.year, r.team): sc.sits[(r.year, r.team)] for r in S.itertuples()}
        st, best_e = None, 1e9
        for cand in SETTINGS:
            RV = sc.realized(cand)
            got = np.array([RV.titles(sit[(r.year, r.team)], r.grp, r.year, r.drafted_at, r.pick, r["name"]) for _, r in s.iterrows()])
            e = np.abs(got - s.real.to_numpy()).mean()
            if e < 0.05 and (st is None or e < best_e):
                st, best_e = cand, e
        n = 0
        if st is not None:
            RV = sc.realized(st)
            m = C.year.isin([2014, 2015]) & C.drafted_at.isna() & np.array([(y, _norm(x or "")) in sc.affected for y, x in zip(C.year, C["name"])])
            for i in C.index[m]:
                r = C.loc[i]
                C.at[i, "real"] = RV.titles(sit[(r.year, r.team)], r.grp, r.year, r.drafted_at, r.pick, r["name"])
                n += 1
            pv.COST_RATE, pv.WR_FACTOR = SETTINGS[0]
        gs_old, av_old = C.gs.copy(), C.av.copy()
        C["gs"] = [sc.car.get("gs", y, a, nm) for y, a, nm in zip(C.year, C.drafted_at, C["name"])]
        C["av"] = [sc.car.get("av", y, a, nm) for y, a, nm in zip(C.year, C.drafted_at, C["name"])]
        print(f"\n=== {mod.__name__} on rescored {f} (settings {st}; {n} realized cells, "
              f"{(C.gs != gs_old).sum()} starts and {(C.av != av_old).sum()} AV values changed) ===")
        tmp = HD / f"rescored_{f}"
        C.to_parquet(tmp)
        keep = HD / outf
        bak = HD / f"{outf}.bak"
        if keep.exists():
            shutil.copy(keep, bak)
        mod.CACHE = tmp
        if mod is qb_reach:
            qb_reach.defer_test.CACHE = tmp
        buf = io.StringIO()
        with redirect_stdout(buf):
            mod.main()
        if bak.exists():
            shutil.move(bak, keep)
        print(buf.getvalue())


if __name__ == "__main__":
    reviews()
    candidates_tests()
