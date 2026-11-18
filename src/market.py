"""What the trade market would charge: a trade priced on the chart fitted to real draft-window trades.

Chart (scripts/future_pick_test.py, README 'Future picks'): pick-only trades 2002-2025, pick k this year worth
v(k) = exp(-lam (k-1)^beta) (v(1) = 1); next year's pick worth delta x E[v(its slot)], delta 0.48 for a 1st and
0.64 for rounds 2-7, the slot drawn from the original owner's likely finish (same rule as the model). Picks two or
more years out: almost no trades to fit (5), so delta^years, an extrapolation. The team moving up (getting the best
pick this year) typically pays ~11% over the chart.

This is the going rate, not the model's value: the model's verdict (titles) is what to act on; the market price
says whether a partner is likely to accept and how far an offer sits from what teams usually pay.
"""
import json
from functools import lru_cache

import numpy as np

from outlook import OUT
from pick_value import baseline_paths
from trade import _slots


@lru_cache(maxsize=1)
def chart() -> dict:
    return json.load(open(OUT / "market_chart.json"))


def available() -> bool:
    return (OUT / "market_chart.json").exists()


def v(k, p):
    return np.exp(-p["lam"] * np.maximum(np.asarray(k, float) - 1, 0) ** p["beta"])


def equivalent(value, p) -> float:
    """The single pick this year worth `value` on the chart (above 1 = more than the #1 pick)."""
    if value >= 1:
        return 1.0
    k = np.arange(1, 400)
    return float(np.interp(-value, -v(k, p), k))


def _side(now, future, slots, p) -> float:
    total = float(v(np.asarray(now, float), p).sum()) if now else 0.0
    for f, s in zip(future, slots):
        d = p["d1r1"] if f.rnd == 1 else p["d1l"]
        total += d ** max(f.year, 1) * float(v(s, p).mean())
    return total


def price(sit: dict, give_now=(), get_now=(), give_future=(), get_future=(), n=4000, seed=0) -> dict:
    """Market value of each side (in #1-pick units), get / give with a 95% range from refitting the chart on
    resampled trades, each side as an equivalent single pick this year, and who is moving up."""
    c = chart()
    rng = np.random.default_rng(seed)
    base = baseline_paths(sit, rng, n)
    slots = {id(f): _slots(f, sit, base, rng) for f in list(give_future) + list(get_future)}
    gs = [slots[id(f)] for f in give_future]
    ts = [slots[id(f)] for f in get_future]

    def both(p):
        return _side(list(give_now), give_future, gs, p), _side(list(get_now), get_future, ts, p)

    give, get = both(c["fit"])
    ratios = np.array([(lambda a, b: b / a if a > 0 else np.nan)(*both(p)) for p in c["boot"]])
    best_give, best_get = min(give_now, default=999), min(get_now, default=999)
    up = "us" if best_get < best_give else ("them" if best_give < best_get else None)
    return {"give": give, "get": get, "ratio": get / give if give > 0 else np.nan,
            "ratio_lo": float(np.nanpercentile(ratios, 2.5)), "ratio_hi": float(np.nanpercentile(ratios, 97.5)),
            "give_eq": equivalent(give, c["fit"]), "get_eq": equivalent(get, c["fit"]), "moving_up": up,
            "premium": float(np.exp(c["fit"]["c"])), "far_future": any(f.year >= 2 for f in list(give_future) + list(get_future)),
            "n_trades": c["n_trades"]}


def describe(r: dict) -> str:
    """One plain-English paragraph for the app."""
    pct = (r["ratio"] - 1) * 100
    lead = (f"At market prices we get **{abs(pct):.0f}% {'more' if pct >= 0 else 'less'}** than we give "
            f"(95% range {(r['ratio_lo'] - 1) * 100:+.0f}% to {(r['ratio_hi'] - 1) * 100:+.0f}%). "
            f"We give the equivalent of pick #{r['give_eq']:.0f} this year, we get #{r['get_eq']:.0f}.")
    prem = (r["premium"] - 1) * 100
    if r["moving_up"] == "us":
        norm = (f" We're moving up: teams moving up usually pay ~{prem:.0f}% over the chart, so a typical deal "
                f"would have us getting about {100 / r['premium']:.0f}% of what we give.")
    elif r["moving_up"] == "them":
        norm = (f" They're moving up: teams moving up usually pay ~{prem:.0f}% over the chart, so a typical deal "
                f"would have us getting about {r['premium'] * 100:.0f}% of what we give.")
    else:
        norm = ""
    far = " Picks two or more years out are extrapolated (almost no such trades to fit)." if r["far_future"] else ""
    return lead + norm + far
