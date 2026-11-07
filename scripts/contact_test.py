"""Pooled 'direct contact' test: did the drafting team have inside contact with the player?

Contact = someone on the drafting team's staff coached at his college in the last 5 seasons (src/coaching.py) OR
the team reported a meeting with him before the draft (visits; scripts/visits_test.py). Each alone ran x1.06 / x1.09
vs draft slot, not significant; pooled once (specified before looking) to see if the shared direction holds.
Primary: drafted players 2016-2022 who had some possible contact (school linked to an NFL staff, or >= 1 reported
meeting), contact drafter vs not, value vs draft slot out of sample, controlling for how many teams could have had
contact. Secondary: all drafted players; by pick range; without 2020-21 (thin visit reporting).
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import prospect_model as pm
from multisport_test import SLOT_CLASSES, compare, expectations
from outlook import OUT
from visits_test import visits


def main():
    uni = pm.load()
    c = pd.read_parquet(OUT / "coaching_picks.parquet").dropna(subset=["player_id"])
    c["player_id"] = c.player_id.astype(str)
    c = c[["player_id", "season", "pick", "fr", "conn", "n_teams_conn"]]
    vk = visits(uni)
    vk["player_id"] = vk.player_id.astype(str)
    met = vk.groupby(["player_id", "cls"]).fr.agg(set).to_dict()
    R = expectations(uni, SLOT_CLASSES)
    R["player_id"] = R.player_id.astype(str)
    d = R.drop(columns=[x for x in ["pick", "fr", "season"] if x in R]).merge(
        c, left_on=["player_id", "cls"], right_on=["player_id", "season"], how="inner")
    d = d[d.exp_slot.notna()].copy()
    m = [met.get((p, y), set()) for p, y in zip(d.player_id, d.cls)]
    d["visit"] = [f in s for f, s in zip(d.fr, m)]
    d["n_met"] = [len(s) for s in m]
    d["conn"] = d.conn.astype(bool)
    d["contact"] = d.conn | d.visit
    d["possible"] = (d.n_teams_conn > 0) | (d.n_met > 0)
    d["log_nmet"] = np.log1p(d.n_met)
    d["log_nconn"] = np.log1p(d.n_teams_conn)
    covs = ("log_nmet", "log_nconn")
    print(f"drafted 2016-22 with outcomes {len(d)}; possible contact {d.possible.sum()}; contact drafter {d.contact.sum()} "
          f"(coaching {d.conn.sum()}, visit {d.visit.sum()}, both {(d.conn & d.visit).sum()})")
    p = d[d.possible]
    print("\nPRIMARY: possible-contact players, drafted by a contact team vs not (vs draft slot)")
    compare(p, "contact", "exp_slot", covs=covs, label="direct contact (coaching or meeting)")
    print("\nSECONDARY")
    compare(d, "contact", "exp_slot", covs=covs, label="all drafted players")
    compare(p[p.pick <= 100], "contact", "exp_slot", covs=covs, label="picks 1-100")
    compare(p[p.pick > 100], "contact", "exp_slot", covs=covs, label="picks 101+")
    compare(p[~p.cls.isin([2020, 2021])], "contact", "exp_slot", covs=covs, label="without 2020-21")
    compare(p[p.conn | ~p.contact], "conn", "exp_slot", covs=covs, label="  coaching only (vs no contact)")
    compare(p[p.visit | ~p.contact], "visit", "exp_slot", covs=covs, label="  meeting only (vs no contact)")
    for y, g in p.groupby("cls"):
        r = lambda x: x.actual.sum() / x.exp_slot.sum()
        print(f"    {y}: contact {g.contact.sum():3d}  value x{r(g[g.contact]) / r(g[~g.contact]):.2f}")
    d.to_parquet(OUT / "contact.parquet")


if __name__ == "__main__":
    main()
