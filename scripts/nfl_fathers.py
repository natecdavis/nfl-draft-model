"""Does having a father who played in the NFL predict outcomes beyond consensus rank, or
beyond draft slot? (A Yahoo writer reported this kind of signal for the NBA draft.)

Fathers coded by hand from the BACKGROUND / SUMMARY text of Brugler's guides (2019-2023),
after reading every sentence linking a father to the NFL:
  PLAYED  - the guide says he played NFL games/seasons (or he clearly did, e.g. Troy Vincent)
  DRAFTED - drafted by an NFL team, but the guide doesn't say he played
Not counted: coaches, executives, agents, uncles/brothers, camp/tryout/practice-squad only.
Only fathers the guide mentions are caught, so a few real cases may be missing.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import prospect_model as pm
from outcomes_v2 import REG
from outlook import OUT

PLAYED = {
    2019: ["Clayton Thorson", "Benny Snell Jr.", "D.K. Metcalf", "Jon’Vea Johnson", "Bunchy Stallings", "Nick Bosa",
           "Christian Miller", "Jamell Garcia-Williams", "Tre Watson", "Irv Smith Jr.", "Chris Lindstrom",
           "Marquez Tucker", "Greg Roberts", "Daniel Wise", "Kaden Elliss", "Mark Fields"],
    2020: ["Steven Montez", "Bryce Perkins", "Jalen Reagor", "Van Jefferson", "Isaiah Hodgins", "Cody White",
           "Thaddeus Moss", "Charlie Heck", "Jon Runyan", "Luke Juriga", "Chris Orr", "Michael Pittman Jr.",
           "Brycen Hopkins", "Darnay Holmes", "Antoine Winfield", "J.R. Reed", "Myles Dorn", "Nigel Warrior"],
    2021: ["Tommy Tremble", "Pete Werner", "Tuf Borland", "Jaycee Horn", "Jevon Holland", "Amari Rodgers",
           "Blake Proehl", "Darius Stills", "Patrick Surtain Ii", "Asante Samuel Jr.", "Elijah Molden",
           "Zech Mcphearson", "Brendon White"],
    2022: ["Ronnie Rivers", "Christian Watson", "Armani Rogers", "Clint Ratkovich", "Tyler Vrabel", "Jalen Mckenzie",
           "Noah Elliss", "Kaiir Elam", "Nolan Turner", "Russ Yeast", "Connor Heyward", "Alec Lindstrom",
           "Chris Hinton", "Quentin Lake"],
    2023: ["Keaton Mitchell", "Taron Vincent", "Joey Porter Jr.", "Kyle Patterson", "Dante Stills", "Kyu Blu Kelly"],
}
DRAFTED = {
    2019: ["Tommy Sweeney", "Greg Little", "Will Harris"],
    2020: ["Joe Burrow", "Jacob Breeland", "Cole Kmet"],
    2021: ["Brevin Jordan", "Kary Vincent Jr."],
    2022: [],
    2023: ["Cedric Tillman", "Paris Johnson Jr."],
}


def main():
    import statsmodels.api as sm
    t = pd.read_parquet(OUT / "beast_test.parquet")  # matched 2019-22 profiles with out-of-sample expectations
    key = lambda y, n: (y, n.lower())
    played = {key(y, n) for y, ns in PLAYED.items() for n in ns}
    drafted = {key(y, n) for y, ns in DRAFTED.items() for n in ns}
    t["father_played"] = [key(y, n) in played for y, n in zip(t.year, t.name)]
    t["father_any"] = t.father_played | np.array([key(y, n) in drafted for y, n in zip(t.year, t.name)])
    found = t.father_played.sum()
    print(f"2019-22 fathers who played, coded: {sum(len(PLAYED[y]) for y in range(2019, 2023))}, matched to outcomes: {found}")

    # slot-based expectation, same out-of-sample setup (each draft predicted by models of earlier drafts)
    uni = pm.load()
    saved = dict(pm.PARAMS); pm.PARAMS.update(REG)
    try:
        R = pm.rolling(uni, versions=("slot",), test_classes=(2019, 2020, 2021, 2022))
    finally:
        pm.PARAMS.clear(); pm.PARAMS.update(saved)
    R["exp_slot"] = R[[f"slot_p{s}" for s in pm.SEASONS]].sum(axis=1)
    t = t.merge(R[["player_id", "exp_slot"]], on="player_id", how="left")

    def compare(d, exp_col, label):
        f = d[d.father_played]; o = d[~d.father_played]
        ratio = lambda x: x.actual.sum() / x[exp_col].sum()
        rng = np.random.default_rng(0)
        bs = [ratio(f.sample(len(f), replace=True, random_state=int(rng.integers(1e9)))) /
              ratio(o.sample(len(o), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(2000)]
        X = sm.add_constant(pd.DataFrame({"father": d.father_played.astype(float)}))
        m = sm.GLM(d.actual, X, family=sm.families.Poisson(), offset=np.log(d[exp_col].clip(lower=1e-4))).fit(cov_type="HC1")
        # simpler, robust to big stars: share who beat their expectation
        beat_f = (f.actual > f[exp_col]).mean(); beat_o = (o.actual > o[exp_col]).mean()
        print(f"  {label}: n={len(f)} vs {len(o)} | actual / expected: {ratio(f):.2f} vs {ratio(o):.2f} -> "
              f"father x{ratio(f) / ratio(o):.2f} (95% {np.percentile(bs, 2.5):.2f} to {np.percentile(bs, 97.5):.2f}); "
              f"Poisson x{np.exp(m.params['father']):.2f} (z {m.tvalues['father']:+.1f}) | beat expectation {beat_f:.0%} vs {beat_o:.0%}")

    print("\nFather played in the NFL, 2019-22 drafts, 4-season value:")
    compare(t, "expected", "vs consensus rank + position (all matched)")
    compare(t[t.on_board == 1], "expected", "vs consensus, board players only")
    compare(t[t.drafted.astype(bool) & t.exp_slot.notna()], "exp_slot", "vs draft slot + position (drafted)")
    print("\nFathers:")
    f = t[t.father_played].sort_values(["year", "pick"])
    print(f[["year", "name", "grp", "hasan_rank", "pick", "expected", "actual"]].assign(
        expected=lambda x: (x.expected * 100).round(1), actual=lambda x: (x.actual * 100).round(1)).to_string(index=False))
    t.to_parquet(OUT / "nfl_fathers.parquet")


if __name__ == "__main__":
    main()
