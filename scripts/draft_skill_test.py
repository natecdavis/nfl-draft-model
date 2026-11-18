"""Team drafting skill: do some franchises, general managers or head coaches consistently beat draft slot?

Why it matters here: if skill exists and persists, a skilled team's reach is information (shade toward its pick
more), a trade partner's "reach" can be judged, and draft-day fall shading could depend on who passed. If it doesn't
persist, the app's uniform treatment stands.

Players: drafted 2006-2022 with seasons 1-4 labels (pm.LABEL_FILE, default darko; never in the NFL = 0).
Expected value from slot: the project's slot model (position group + log pick) fit leave-one-class-out over
2006-22 (scripts/old_class_replication.expected_slot with the class range widened). Residual per player
= actual - expected (in % of cap, summed seasons 1-4), demeaned by class. Team-draft score = sum of its players'
residuals (the drafting team = who made the pick). Note: the label counts value wherever he played, but most
of seasons 1-4 are with the drafting team, so "skill" here includes development and opportunity.

GM / head coach for draft year Y: the Wikipedia infobox of the team's season-Y page (data/raw/wiki/seasons, cached
for 2006-2024); first name listed.

Pre-specified tests (null = no skill):
  1. Franchise spread: SD of franchise means over 17 drafts vs a placebo shuffling team labels among
     team-drafts within each year (2000 draws). Estimated true SD = sqrt(var(means) - placebo var).
  2. Persistence: corr(score, mean of the same franchise's previous 3 drafts), vs the same placebo.
  3. Decision-makers: split-half within GM tenures with 4+ drafts (odd vs even drafts); same for head coaches.
     Across a GM change vs within a tenure: corr of consecutive drafts.
  4. GMs who drafted for 2+ franchises: corr of their mean at the first vs the later franchise.
  5. Usefulness: predict each draft from the shrunk (empirical-Bayes) mean of the franchise's / GM's earlier
     drafts; out-of-sample corr and R^2.
Also by round group (1-2 vs 3-7), and seasons-started / AV as outside yardsticks for test 1-2.

Run:  python scripts/draft_skill_test.py
"""
import json
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import old_class_replication as ocr  # noqa: E402
from outlook import OUT, RAW  # noqa: E402
from review import team as norm_team  # noqa: E402

CLASSES = range(2006, 2023)
WIKI = RAW / "wiki/seasons"
NAME = {"Arizona_Cardinals": "ARI", "Atlanta_Falcons": "ATL", "Baltimore_Ravens": "BAL", "Buffalo_Bills": "BUF",
        "Carolina_Panthers": "CAR", "Chicago_Bears": "CHI", "Cincinnati_Bengals": "CIN", "Cleveland_Browns": "CLE",
        "Dallas_Cowboys": "DAL", "Denver_Broncos": "DEN", "Detroit_Lions": "DET", "Green_Bay_Packers": "GB",
        "Houston_Texans": "HOU", "Indianapolis_Colts": "IND", "Jacksonville_Jaguars": "JAX", "Kansas_City_Chiefs": "KC",
        "Los_Angeles_Rams": "LA", "St._Louis_Rams": "LA", "Los_Angeles_Chargers": "LAC", "San_Diego_Chargers": "LAC",
        "Las_Vegas_Raiders": "LV", "Oakland_Raiders": "LV", "Miami_Dolphins": "MIA", "Minnesota_Vikings": "MIN",
        "New_England_Patriots": "NE", "New_Orleans_Saints": "NO", "New_York_Giants": "NYG", "New_York_Jets": "NYJ",
        "Philadelphia_Eagles": "PHI", "Pittsburgh_Steelers": "PIT", "Seattle_Seahawks": "SEA",
        "San_Francisco_49ers": "SF", "Tampa_Bay_Buccaneers": "TB", "Tennessee_Titans": "TEN",
        "Washington_Redskins": "WAS", "Washington_Football_Team": "WAS", "Washington_Commanders": "WAS"}


def clean(v):
    v = re.sub(r"<!--.*?-->|<ref[^>]*/>|<ref.*?</ref>|\{\{.*?\}\}|\(.*?\)|'''?", "", v, flags=re.S)
    v = re.sub(r"^\s*Vacant\s*\|", "", v)  # "Vacant |Bob Quinn]] fired ..." -> the fired GM ran the draft
    v = re.split(r"<br\s*/?>|\n|/|;|,(?![^\[]*\])| and | fired", v)[0]
    v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", v)
    v = re.sub(r"[\[\]{}]", "", v)
    return re.sub(r"\s+", " ", v).strip(" .*-")


def deciders():
    rows = []
    for f in sorted(WIKI.glob("*.json")):
        y, rest = f.stem.split("_", 1)
        tm = NAME.get(rest.removesuffix("_season"))
        if tm is None:
            continue
        t = json.load(open(f))["text"]
        rec = {"year": int(y), "team": tm}
        for fld, key in [("general_manager", "gm"), ("coach", "hc")]:
            m = re.search(rf"^\|\s*{fld}\s*=([^\n]*)$", t, re.M)
            v = clean(m.group(1)) if m else ""
            rec[key] = v if v and v.lower() not in ("none", "vacant") else None
        rows.append(rec)
    D = pd.DataFrame(rows).sort_values(["team", "year"]).reset_index(drop=True)
    D.loc[(D.team == "CIN") & D.gm.isna(), "gm"] = "Mike Brown"  # owner, no GM title; runs personnel
    D["gm_source"] = np.where(D.gm.notna(), "infobox", None)
    # Missing GM years (e.g. Belichick as de facto GM): fill a gap when the known GMs on both sides agree,
    # or from the nearest known GM when the head coach is the same across the gap.
    for tm, g in D.groupby("team"):
        idx = g.index.to_list()
        for k, i in enumerate(idx):
            if pd.notna(D.at[i, "gm"]):
                continue
            before = [j for j in idx[:k] if D.at[j, "gm_source"] == "infobox"]
            after = [j for j in idx[k + 1:] if D.at[j, "gm_source"] == "infobox"]
            b, a = (before[-1] if before else None), (after[0] if after else None)
            fill = None
            if b is not None and a is not None and D.at[b, "gm"] == D.at[a, "gm"]:
                fill = D.at[b, "gm"]
            else:
                for j in [x for x in (b, a) if x is not None]:
                    span = D.loc[min(i, j):max(i, j)]
                    span = span[span.team == tm]
                    if span.hc.notna().all() and span.hc.nunique() == 1:
                        fill = D.at[j, "gm"]
                        break
            if fill:
                D.at[i, "gm"], D.at[i, "gm_source"] = fill, "filled"
    return D


def players():
    ocr.CLASSES = CLASSES
    d = ocr.expected_slot()
    dp = pd.read_parquet(RAW / "draft_picks.parquet")
    tm = {(y, p): norm_team(t) for y, p, t in zip(dp.season, dp.pick, dp.team)}
    gs = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.seasons_started)}
    av = {(y, p): a for y, p, a in zip(dp.season, dp.pick, dp.w_av)}
    d["team"] = [tm.get((c, p)) for c, p in zip(d.cls, d.pick)]
    d["gs"] = [gs.get((c, p)) for c, p in zip(d.cls, d.pick)]
    d["av"] = [av.get((c, p)) for c, p in zip(d.cls, d.pick)]
    d = d.dropna(subset=["team"])
    d["resid"] = (d.actual - d.exp_slot) * 100
    d["resid"] -= d.groupby("cls").resid.transform("mean")
    # outside yardsticks: residual vs pick within class (quadratic in log pick), demeaned
    for y in ("gs", "av"):
        d[y] = d[y].fillna(0.0)
        r = []
        for c, g in d.groupby("cls"):
            b = np.polyfit(np.log(g.pick + 10), g[y], 2)
            r.append(g[y] - np.polyval(b, np.log(g.pick + 10)))
        d[f"{y}_resid"] = pd.concat(r)
    return d


def team_drafts(d, col="resid", mask=None):
    x = d if mask is None else d[mask]
    T = x.groupby(["cls", "team"])[col].sum().rename("score").reset_index()
    full = pd.MultiIndex.from_product([sorted(d.cls.unique()), sorted(d.team.unique())], names=["cls", "team"])
    return T.set_index(["cls", "team"]).reindex(full, fill_value=0.0).reset_index()


def shuffle(T, rng):
    return T.assign(team=T.groupby("cls").team.transform(lambda s: rng.permutation(s.to_numpy())))


def spread(T):
    return T.groupby("team").score.mean().std()


def persistence(T, k=3):
    T = T.sort_values(["team", "cls"])
    prev = T.groupby("team").score.transform(lambda s: s.shift(1).rolling(k, min_periods=k).mean())
    m = prev.notna()
    return np.corrcoef(T.score[m], prev[m])[0, 1]


def test_franchise(T, label, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s0, p0 = spread(T), persistence(T)
    S, P = [], []
    for _ in range(n):
        Tp = shuffle(T, rng)
        S.append(spread(Tp)); P.append(persistence(Tp))
    S, P = np.array(S), np.array(P)
    true_sd = np.sqrt(max(0.0, s0 ** 2 - np.mean(S ** 2)))
    print(f"  {label:34s} franchise-mean SD {s0:.2f} (placebo {np.mean(S):.2f}, p {np.mean(S >= s0):.3f}); est. true SD "
          f"{true_sd:.2f} per draft; persistence r {p0:+.3f} (placebo {np.mean(P):+.3f}, p {np.mean(P >= p0):.3f})")
    return true_sd


def split_half(T, who, D, min_n=4):
    X = T.merge(D[["year", "team", who]].rename(columns={"year": "cls"}), on=["cls", "team"], how="left")
    X = X.dropna(subset=[who])
    X["tenure"] = X.team + ":" + X[who]
    rows = []
    for k, g in X.sort_values("cls").groupby("tenure"):
        if len(g) >= min_n:
            rows.append((g.score.iloc[0::2].mean(), g.score.iloc[1::2].mean(), len(g)))
    a = np.array(rows)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    rng = np.random.default_rng(1)
    bs = [np.corrcoef(*a[rng.integers(0, len(a), len(a))][:, :2].T)[0, 1] for _ in range(2000)]
    return r, np.percentile(bs, [2.5, 97.5]), len(a), X


def consecutive(X, who):
    X = X.sort_values(["team", "cls"]).copy()
    X["prev_score"] = X.groupby("team").score.shift(1)
    X["prev_who"] = X.groupby("team")[who].shift(1)
    X["prev_cls"] = X.groupby("team").cls.shift(1)
    X = X[X.prev_cls == X.cls - 1]
    same = X[X[who] == X.prev_who]
    chg = X[X[who] != X.prev_who]
    r = lambda g: np.corrcoef(g.score, g.prev_score)[0, 1]
    return r(same), len(same), r(chg), len(chg)


def movers(X, who):
    X = X.sort_values("cls")
    rows = []
    for name, g in X.groupby(who):
        teams = list(dict.fromkeys(g.team))
        if len(teams) >= 2:
            a = g[g.team == teams[0]].score.mean()
            b = g[g.team.isin(teams[1:])].score.mean()
            rows.append((name, teams, len(g[g.team == teams[0]]), len(g[g.team.isin(teams[1:])]), a, b))
    return pd.DataFrame(rows, columns=["who", "teams", "n_first", "n_later", "first", "later"])


def predict(T, X, who, tau2):
    """Out-of-sample: each draft predicted by the shrunk mean of earlier drafts (franchise or tenure)."""
    key = "team" if who == "team" else "tenure"
    Z = (X if who != "team" else T).sort_values("cls").copy()
    noise = T.score.var()
    pred = []
    for _, g in Z.groupby(key):
        s = g.score.to_numpy()
        cum = np.concatenate([[0], np.cumsum(s)[:-1]])
        n = np.arange(len(s))
        shrink = n * tau2 / (n * tau2 + noise)
        pred.append(pd.Series(np.where(n > 0, shrink * cum / np.maximum(n, 1), np.nan), index=g.index))
    Z["pred"] = pd.concat(pred)
    m = Z.pred.notna() & (Z.cls >= 2009)
    r = np.corrcoef(Z.score[m], Z.pred[m])[0, 1]
    r2 = 1 - ((Z.score[m] - Z.pred[m]) ** 2).sum() / ((Z.score[m] - Z.score[m].mean()) ** 2).sum()
    return r, r2, int(m.sum())


def main():
    d = players()
    D = deciders()
    print(f"\nDrafted players 2006-22: {len(d)}; team-drafts {d.groupby(['cls', 'team']).ngroups}; "
          f"GM known for {D.gm.notna().mean():.0%} of team-seasons, head coach {D.hc.notna().mean():.0%}")
    print(f"Team-draft score (sum of residuals, % of cap over seasons 1-4): SD {team_drafts(d).score.std():.2f}")

    print("\n1-2. FRANCHISE spread and persistence (placebo: team labels shuffled within year)")
    tau = {}
    for lab, mask, col in [("all rounds, value vs slot", None, "resid"),
                           ("rounds 1-2", d.pick <= 64, "resid"),
                           ("rounds 3-7", d.pick > 64, "resid"),
                           ("seasons started vs pick (outside)", None, "gs_resid"),
                           ("AV vs pick (outside)", None, "av_resid")]:
        tau[lab] = test_franchise(team_drafts(d, col, mask), lab)
    T = team_drafts(d)
    print("\n   Franchise means (value vs slot, per draft), top and bottom 5:")
    fm = T.groupby("team").score.mean().sort_values()
    print("   " + ", ".join(f"{k} {v:+.2f}" for k, v in fm.tail(5)[::-1].items()) + " | ... | "
          + ", ".join(f"{k} {v:+.2f}" for k, v in fm.head(5).items()))
    T2 = T.copy()
    T2["half"] = np.where(T2.cls <= 2014, "a", "b")
    h = T2.groupby(["team", "half"]).score.mean().unstack()
    print(f"   2006-14 vs 2015-22 franchise means: r {np.corrcoef(h.a, h.b)[0, 1]:+.2f} (32 franchises)")

    print("\n3. DECISION-MAKERS: split-half within tenures with 4+ drafts (odd vs even drafts)")
    Xs = {}
    for who, lab in [("gm", "general manager"), ("hc", "head coach")]:
        r, ci, n, X = split_half(T, who, D)
        Xs[who] = X
        rs, ns, rc, nc = consecutive(X, who)
        print(f"  {lab:16s} tenures {n}: split-half r {r:+.2f} (95% {ci[0]:+.2f} to {ci[1]:+.2f}) | consecutive drafts: "
              f"same {who.upper()} r {rs:+.2f} (n {ns}), after a change r {rc:+.2f} (n {nc})")

    print("\n4. GMs who drafted for 2+ franchises: mean score at the first vs later franchise(s)")
    M = movers(Xs["gm"], "gm")
    M = M[(M.n_first >= 2) & (M.n_later >= 2)]
    if len(M) >= 4:
        print(f"  {len(M)} GMs: r {np.corrcoef(M['first'], M['later'])[0, 1]:+.2f}")
        print(M.round(2).to_string(index=False))
    Mh = movers(Xs["hc"], "hc")
    Mh = Mh[(Mh.n_first >= 2) & (Mh.n_later >= 2)]
    if len(Mh) >= 4:
        print(f"  head coaches who drafted for 2+ franchises: {len(Mh)}, r {np.corrcoef(Mh['first'], Mh['later'])[0, 1]:+.2f}")

    print("\n5. USEFULNESS: predict each draft (2009-22) from the shrunk mean of earlier drafts")
    t2 = tau["all rounds, value vs slot"] ** 2
    for who in ("team", "gm", "hc"):
        r, r2, n = predict(T, Xs.get(who), who, max(t2, 1e-6))
        print(f"  by {who:4s} (true SD {np.sqrt(t2):.2f}): corr {r:+.3f}, R^2 {r2:+.4f}, n {n}")
    d.to_parquet(OUT / "draft_skill_players.parquet")
    D.to_parquet(OUT / "draft_deciders.parquet")


if __name__ == "__main__":
    main()
