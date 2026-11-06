"""Does team strength turn into playoff runs and titles at the rate the model says?

Pick values are (points of team strength a player adds) x (how fast title odds rise with strength), the
second from the season simulator (src/simulate.py: game noise 12.6, home edge, real schedules, seeding,
bracket) applied to a generic league (pick_value.strength_curve). Checked on real seasons 2003-2025:

  A. pre-season: each team's strength predicted from last season's luck-adjusted rating with the model's
     own carryover (true strength ~ N(carry x reliability x last rating, sd from the same parameters)),
     each season simulated on its real schedule -> P(playoffs), P(conference title), P(Super Bowl);
  B. in-season: the season's own luck-adjusted rating (playoff games aren't in the fit, so conference
     titles and Super Bowls are out of sample; making the playoffs isn't).
Then actual vs predicted by bins, and calibration slopes (logistic regression of the outcome on the
model's log-odds: slope 1 = title odds rise with strength as fast as the model says; < 1 = too fast).
Plus C: carryover over 1-4 seasons vs the model's 0.60 per season (pick values run 5 seasons).

Run:  python scripts/title_check.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from simulate import SEASONS, league_params, load_schedules, simulate  # noqa: E402
from outlook import OUT, RAW  # noqa: E402

N = 4000


def actual(sched, yr):
    po = sched[(sched.season == yr) & (sched.game_type != "REG") & sched.result.notna()]
    made = set(po.home_team) | set(po.away_team)
    sb = po[po.game_type == "SB"].iloc[0]
    conf = {sb.home_team, sb.away_team}
    champ = sb.home_team if sb.result > 0 else sb.away_team
    wins = {}
    for h, a, r in zip(po.home_team, po.away_team, po.result):
        w = h if r > 0 else a
        wins[w] = wins.get(w, 0) + 1
    return made, conf, champ, wins


def run(sched, params, teams_df):
    conf_of = dict(zip(teams_df.team_abbr, teams_df.team_conf))
    div_of = dict(zip(teams_df.team_abbr, teams_df.team_division))
    rel, carry, tsd = params["reliability"], params["carryover"], params["true_sd"]
    pre_sd = tsd * np.sqrt(1 - carry ** 2 * rel)  # true strength given last season's rating
    rng = np.random.default_rng(0)
    rows = []
    for yr in SEASONS[1:]:
        g = sched[(sched.season == yr) & (sched.game_type == "REG")]
        now, last = params["ratings"][yr], params["ratings"][yr - 1]
        teams = list(now.index)
        made, conf, champ, wins = actual(sched, yr)
        npo = 6 if yr < 2020 else 7
        hfa = params["hfa_by_season"][yr]
        for mode in ("pre", "in"):
            if mode == "pre":
                m = np.array([carry * rel * last.get(t, 0.0) for t in teams])
                r = m + pre_sd * rng.standard_normal((N, len(teams)))
            else:
                m = (now * rel).to_numpy()
                r = np.tile(m, (N, 1))
            res = simulate(r, teams, conf_of, div_of, g, hfa, params["game_sd"], npo, rng)
            for i, t in enumerate(teams):
                rows.append({"season": yr, "team": t, "mode": mode, "strength": m[i],
                             "p_playoffs": res["playoffs"][i], "p_conf": res["conf"][i], "p_sb": res["sb"][i],
                             "playoffs": t in made, "conf": t in conf, "sb": t == champ, "po_wins": wins.get(t, 0)})
    return pd.DataFrame(rows)


def slope(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    X = sm.add_constant(np.log(p / (1 - p)).to_numpy())
    m = sm.Logit(y.astype(float).to_numpy(), X).fit(disp=0)
    lo, hi = m.conf_int()[1]
    return f"{m.params[1]:.2f} ({lo:.2f} to {hi:.2f})"


def main():
    sched = load_schedules()
    teams_df = pd.read_parquet(RAW / "teams.parquet")
    params = league_params(sched)
    f = OUT / "holdout" / "title_check.parquet"
    df = pd.read_parquet(f) if f.exists() else run(sched, params, teams_df)
    df.to_parquet(f)
    print(f"{df.season.nunique()} seasons, {df[df['mode'] == 'pre'].shape[0]} team-seasons")
    for mode, lab in [("pre", "A. pre-season strength (last season x carryover)"), ("in", "B. in-season luck-adjusted rating")]:
        d = d0 = df[df["mode"] == mode]
        print(f"\n=== {lab} ===")
        print("  calibration slope (1 = right):  playoffs " + slope(d.playoffs, d.p_playoffs)
              + " | conf title " + slope(d.conf, d.p_conf) + " | Super Bowl " + slope(d.sb, d.p_sb))
        bins = [-99, -4, -2, 0, 2, 4, 6, 99]
        d = d.assign(b=pd.cut(d.strength, bins))
        t = d.groupby("b", observed=True).agg(n=("team", "size"), playoffs=("playoffs", "mean"), p_playoffs=("p_playoffs", "mean"),
                                              conf=("conf", "sum"), p_conf=("p_conf", "sum"), sb=("sb", "sum"), p_sb=("p_sb", "sum"),
                                              wins=("po_wins", "mean"))
        t["playoffs"] = (t.playoffs * 100).round(0)
        t["p_playoffs"] = (t.p_playoffs * 100).round(0)
        print("  by strength (pts/game vs average): playoffs % actual vs predicted; conference titles and Super Bowls, count actual vs predicted")
        print(t.round(1).to_string())
        # top teams: the range pick values care about (contenders at +5 to +9)
        for lo in (4, 6):
            x = d0[d0.strength >= lo]
            print(f"  strength >= +{lo}: n={len(x)}, Super Bowls {x.sb.sum()} vs {x.p_sb.sum():.1f} predicted, "
                  f"conf titles {x.conf.sum()} vs {x.p_conf.sum():.1f}")
    # C. carryover
    rat, rel = params["ratings"], params["reliability"]
    print("\n=== C. carryover: correlation of season ratings k seasons apart, actual vs model (reliability x 0.60^k) ===")
    for k in (1, 2, 3, 4):
        a = np.array([(rat[y][t], rat[y + k][t]) for y in SEASONS if y + k in rat for t in rat[y].index])
        print(f"  k={k}: actual {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.2f}, model {rel * 0.60 ** k:.2f} (n={len(a)})")


if __name__ == "__main__":
    main()
