"""Do the ways college players answer press-conference questions predict NFL outcomes beyond the board?

Transcripts: ASAP Sports bowl / title-game / kickoff press conferences 2011-2022 (scripts/asap_collect.py).
Features (specified before looking at outcomes), from all of a player's answers pooled (>= 200 words):
  words_per_answer  log mean words per answer (expansiveness)
  tech              football-technical vocabulary per 1,000 words (coverage, leverage, technique, film, ...)
  we_share          we / us / our as a share of first-person pronouns (team orientation)
  cliche            stock phrases per 1,000 words ("one game at a time", "blessed", "at the end of the day", ...)
  word_len          mean letters per word (vocabulary sophistication; crude)
Each is standardized and tested alone: drafted players vs draft slot (classes 2016-22), board players vs consensus
(2018-22), out of sample (scripts/multisport_test.expectations); controls: log total words, class fixed effects.
10 primary tests, so one p < 0.05 is expected by chance. Secondary: QBs only (the 'cerebral' Beast hint).
Secondary features (added after the literature review, before outcomes were seen; reported separately):
  cdi    categorical-dynamic index (Pennebaker et al. 2014, admissions essays -> college GPA): % articles +
         % prepositions - % personal and impersonal pronouns - % auxiliary verbs - % conjunctions - % adverbs
         - % negations (+30); higher = categorical / analytic
  hedge  hedges and fillers per 1,000 words ("you know", "i mean", "kind of", "i guess", "probably", ...;
         cf. Larcker & Zakolyukina 2012 on earnings calls)
"""
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, expectations
from outlook import OUT

TECH = (r"\b(coverages?|leverage|technique|footwork|film|tape|scheme|blitz(es|ing)?|protections?|reads?|reading|"
        r"assignments?|gap|gaps|fronts?|alignment|keys|tendenc(y|ies)|zone|man coverage|press|route|routes|"
        r"pre-?snap|post-?snap|hips|hands|leverage|pad level|angles?|playbook|fundamentals|details?|"
        r"rotation|safety|safeties|linebackers?|corners?|edge|pocket|mesh|motion|formations?|personnel|"
        r"rpo|play-?action|check|checks|audible|disguise)\b")
CLICHE = (r"(one game at a time|one day at a time|day by day|at the end of the day|blessed|blessing|it is what it is|"
          r"control what we can control|control what i can control|take care of business|give (it )?(our|my) all|"
          r"(110|a hundred and ten) percent|stay humble|next man up|trust the process|we just gotta|just go out there|"
          r"go out there and (play|have fun|compete|execute)|whatever it takes|grind)")
WE = r"\b(we|us|our|ours|we're|we've|we'll)\b"
I_ = r"\b(i|me|my|mine|i'm|i've|i'll|i'd)\b"
FEATS = ["words_per_answer", "tech", "we_share", "cliche", "word_len"]
SECONDARY = ["cdi", "hedge"]
W = lambda s: set(s.split())
ARTICLES = W("a an the")
PREPS = W("about above across after against along among around at before behind below beneath beside between beyond "
          "by down during except for from in inside into near of off on onto out outside over past since through "
          "throughout to toward towards under until up upon with within without")
PERS = W("i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his himself she her "
         "hers herself they them their theirs themselves i'm i've i'll i'd we're we've we'll we'd you're you've you'll "
         "he's she's they're they've they'll")
IMPERS = W("it its itself it's this that these those something anything everything nothing someone anyone everyone "
           "somebody anybody everybody nobody thing things stuff")
AUX = W("am is are was were be been being have has had having do does did will would shall should can could may might "
        "must gonna gotta")
CONJ = W("and but or so because although though while whereas if unless yet nor")
ADV = W("just really very so too also always never often sometimes still even pretty definitely probably actually "
        "obviously certainly maybe basically literally truly quite again already here there now then")
NEG = W("no not never none nobody nothing neither nor don't doesn't didn't can't couldn't won't wouldn't isn't aren't "
        "wasn't weren't shouldn't haven't hasn't hadn't ain't")
HEDGE = (r"\b(you know|i mean|kind of|kinda|sort of|i guess|i feel like|probably|maybe|somewhat|pretty much|"
         r"or something|stuff like that|things like that|i think)\b")


def cdi(words):
    if not words:
        return np.nan
    n = len(words)
    pct = lambda s: 100 * sum(w in s for w in words) / n
    return 30 + pct(ARTICLES) + pct(PREPS) - pct(PERS) - pct(IMPERS) - pct(AUX) - pct(CONJ) - pct(ADV) - pct(NEG)


def features(a):
    a = a.copy()
    # transcriber notes such as "(Inaudible.)", "(Laughter.)", "(Indiscernible)" are not the player's words
    a["a"] = a.a.fillna("").str.replace(r"\([^)]*\)", " ", regex=True).str.lower()
    g = a.groupby(["player_id", "cls"])
    t = g.a.apply(" ".join)
    words = t.str.findall(r"[a-z']+")
    n = words.map(len)
    f = pd.DataFrame({"n_words": n, "n_answers": g.size(), "n_interviews": g.id.nunique()})
    f["words_per_answer"] = np.log(n / f.n_answers)
    f["tech"] = t.str.count(TECH) / n * 1000
    we, i = t.str.count(WE), t.str.count(I_)
    f["we_share"] = we / (we + i).clip(lower=1)
    f["cliche"] = t.str.count(CLICHE) / n * 1000
    f["word_len"] = words.map(lambda w: np.mean([len(x.replace("'", "")) for x in w]) if w else np.nan)
    f["cdi"] = words.map(cdi)
    f["hedge"] = t.str.count(HEDGE) / n * 1000
    return f.reset_index()


def cont(d, e, x, label):
    d = d.dropna(subset=[e, x]).copy()
    d["z"] = (d[x] - d[x].mean()) / d[x].std()
    d["log_words"] = np.log(d.n_words)
    fe = pd.get_dummies(d.cls, prefix="c", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[["z", "log_words"]], fe], axis=1).astype(float))
    m = sm.GLM(d.actual, X, family=sm.families.Poisson(), offset=np.log(d[e].clip(lower=1e-4))).fit(cov_type="HC1")
    lo, hi = np.exp(m.conf_int().loc["z"])
    print(f"    {label:30s} n={len(d):4d}  per 1 sd: value x{np.exp(m.params.z):.2f} (95% {lo:.2f}-{hi:.2f}, p {m.pvalues.z:.2f})")


def reach(d, x, label):
    """Pricing half of a bias test: is the feature tied to going earlier (> 1) or later than the board rank?"""
    b = d[d.drafted.astype(bool) & d.hasan_rank.notna() & d.pick.notna()].dropna(subset=[x]).copy()
    if len(b) < 30:
        return
    b["z"] = (b[x] - b[x].mean()) / b[x].std()
    y = np.log((b.hasan_rank + 10) / (b.pick + 10))
    fe = pd.get_dummies(b.grp, prefix="g", drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([b[["z"]], np.log(b.n_words).rename("log_words"), fe], axis=1).astype(float))
    m = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"    {label:30s} n={len(b):4d}  per 1 sd: taken x{np.exp(m.params.z):.2f} earlier than board (p {m.pvalues.z:.2f})")


def main():
    a = pd.read_parquet(OUT / "asap_answers.parquet")
    a["player_id"] = a.player_id.astype(str)
    f = features(a)
    f = f[f.n_words >= 200]
    uni = pm.load()
    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R.merge(f, on=["player_id", "cls"], how="inner")
    dr = d[d.drafted.astype(bool) & d.exp_slot.notna()]
    bd = d[(d.on_board == 1) & d.exp_cons.notna()]
    print(f"players with >= 200 words: {len(f)}; drafted 2016-22 {len(dr)}, board 2018-22 {len(bd)}; "
          f"median words {int(f.n_words.median())}, interviews {f.n_interviews.median():.0f}")
    print(d.groupby("grp")[FEATS + SECONDARY].mean().round(2).to_string())
    print("\nfeature correlations:\n" + d[FEATS + SECONDARY].corr().round(2).to_string())
    print("\nPRICING: drafted board players (bias = taken earlier AND then below slot, or the reverse)")
    for x in FEATS + SECONDARY:
        reach(d, x, x)
    for e, s, nm in [("exp_slot", dr, "VS DRAFT SLOT"), ("exp_cons", bd, "VS CONSENSUS")]:
        print(f"\n{nm}")
        for x in FEATS:
            cont(s, e, x, x)
        print("  secondary (literature-based)")
        for x in SECONDARY:
            cont(s, e, x, x)
        print("  QBs only")
        for x in FEATS + SECONDARY:
            cont(s[s.grp == "QB"], e, x, "  " + x)
    d.to_parquet(OUT / "transcript_features.parquet")


if __name__ == "__main__":
    main()
