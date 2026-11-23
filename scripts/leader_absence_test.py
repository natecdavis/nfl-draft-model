"""Leadership through absences: does losing a long-time veteran starter cost his team more than his pay predicts?

Specified before looking (Nov 22 2026). Uses the betting-line absence design (scripts/absence_test.py: team-season
fixed effects, absent regular starters home minus away, pay-scaled QB / non-QB terms) on its saved starter-games
(data/derived/absence_starters.parquet), seasons 2016-2025 (team tenure needs rosters from 2013).
  Leader proxy = absent starter in his 4th+ season with this team (weekly rosters) AND 6th+ NFL season.
  PRIMARY: extra points per absent leader beyond pay (non-QB and QB terms) on the closing spread and on the
  actual margin, with an age-30+ absent count as control (old vets are overpaid for current quality, which
  would push the leader term down for a reason that has nothing to do with leadership).
  SECONDARY: margin - spread (does the market miss it); team tenure 5+ alone; NFL season 8+ alone; no age control.
No NFL captain data yet: this is the stand-in.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from absence_test import OUT, RAW, RELOC, design, fit

TEAM = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SL": "LA", "SD": "LAC", "OAK": "LV"}


def tenure():
    r = pd.read_parquet(RAW / "rosters_weekly.parquet")
    r = r[r.status.isin(["ACT", "RES", "INA"])]
    r["team"] = r.team.replace(TEAM).replace(RELOC)
    p = pd.read_parquet(RAW / "players.parquet")[["gsis_id", "pfr_id", "rookie_season", "draft_year", "birth_date"]]
    ts = r[["gsis_id", "team", "season"]].drop_duplicates().sort_values("season")
    ts["team_yrs"] = ts.groupby(["gsis_id", "team"]).cumcount() + 1
    ts = ts.merge(p, on="gsis_id").dropna(subset=["pfr_id"])
    first = ts.groupby("gsis_id").season.transform("min")
    start = ts.rookie_season.fillna(ts.draft_year).fillna(first)
    ts["nfl_yrs"] = ts.season - np.minimum(start, first) + 1
    ts["age"] = ts.season + 0.7 - pd.to_datetime(ts.birth_date).dt.year - pd.to_datetime(ts.birth_date).dt.dayofyear / 365
    return ts[["pfr_id", "team", "season", "team_yrs", "nfl_yrs", "age"]].drop_duplicates(["pfr_id", "team", "season"])


def main():
    A = pd.read_parquet(OUT / "absence_starters.parquet")
    A = A[A.season >= 2016].copy()
    A = A.merge(tenure().rename(columns={"pfr_id": "pfr"}), on=["pfr", "team", "season"], how="left")
    print(f"starter-games 2016-25: {len(A)}; tenure linked {A.team_yrs.notna().mean():.1%}; absent {int(A.absent.sum())}")
    A[["team_yrs", "nfl_yrs"]] = A[["team_yrs", "nfl_yrs"]].fillna(1)
    A["age"] = A.age.fillna(A.age.median())
    qb = A.grp == "QB"
    defs = {"lead": (A.team_yrs >= 4) & (A.nfl_yrs >= 6), "team5": A.team_yrs >= 5, "nfl8": A.nfl_yrs >= 8}
    for k, f in defs.items():
        A[f"abs_{k}_nonQB"] = (A.absent & f & ~qb).astype(float)
        A[f"abs_{k}_QB"] = (A.absent & f & qb).astype(float)
    A["abs_old"] = (A.absent & (A.age >= 30)).astype(float)
    ab = A[A.absent]
    print(f"absent leaders (team 4+ & NFL 6+): non-QB {int(A.abs_lead_nonQB.sum())} of {int((~qb & A.absent).sum())}, "
          f"QB {int(A.abs_lead_QB.sum())} of {int((qb & A.absent).sum())}; leaders' mean price "
          f"{ab[(ab.team_yrs >= 4) & (ab.nfl_yrs >= 6) & (ab.grp != 'QB')].price_f.mean():.2f}% vs others "
          f"{ab[~((ab.team_yrs >= 4) & (ab.nfl_yrs >= 6)) & (ab.grp != 'QB')].price_f.mean():.2f}%")

    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[(s.game_type == "REG") & s.season.between(2016, 2025) & s.spread_line.notna() & s.result.notna()].copy()
    for c in ["home_team", "away_team"]:
        s[c] = s[c].replace(RELOC)
    s = s[s.game_id.isin(set(A.game_id))]
    print(f"games: {len(s)}")
    base = ["abs_QB", "abs_pay_QB", "abs_nonQB", "abs_pay_nonQB"]
    pd.set_option("display.width", 200)
    runs = [("PRIMARY: leader (team 4+ & NFL 6+), age-30 control", "lead", True),
            ("SECONDARY: leader, no age control", "lead", False),
            ("SECONDARY: team tenure 5+", "team5", True),
            ("SECONDARY: NFL season 8+", "nfl8", True)]
    out = []
    for lab, k, age in runs:
        cols = base + [f"abs_{k}_nonQB", f"abs_{k}_QB"] + (["abs_old"] if age else [])
        M, names, g = design(s, A, cols)
        cl = g.home_team.to_numpy()
        print(f"\n{lab}")
        ys = [("spread_line", "spread"), ("result", "margin")] + ([("mdiff", "margin - spread")] if k == "lead" else [])
        g["mdiff"] = g.result - g.spread_line
        for y, yl in ys:
            T = fit(g[y].to_numpy(float), M, names, cl)
            sel = [n for n in names if n.startswith(f"abs_{k}") or n == "abs_old"]
            print(f"  {yl:16s} " + " | ".join(f"{n.replace('abs_', '')} {T.b[n]:+.2f} ({T.lo[n]:+.2f} to {T.hi[n]:+.2f})"
                                               for n in sel))
            for n in sel:
                out.append(dict(run=lab, y=yl, term=n, b=T.b[n], lo=T.lo[n], hi=T.hi[n]))
            if k == "lead" and age and y == "spread_line":
                print(f"  {'':16s} (base: nonQB {T.b['abs_nonQB']:+.2f} + {T.b['abs_pay_nonQB']:+.3f}/1% cap; "
                      f"QB {T.b['abs_QB']:+.2f} + {T.b['abs_pay_QB']:+.3f}/1%)")
    pd.DataFrame(out).to_csv(ROOT / "reports/leader_absence.csv", index=False)


if __name__ == "__main__":
    main()
