"""Flag 2025 and 2026 draftees whose drafting team's draft-year staff coached at their final college in the 5 prior
seasons (same definition as scripts/contact_value.py), with the coach, his NFL role, whether the link is recent (last
2 seasons) and whether his role matches the player's position.

Writes reports/connected_rookies_2025_26.csv (all positions) and prints the skill-position list for fantasy use.
Staffs from scripts/staffs_2025.py (2026 = staff template as of Nov 2026).
"""
import sys

import nflreadpy as nr
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
from connected_fantasy_test import connections_ext
from contact_value import pos_match, role_kind

GRP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OL", "G": "OL", "C": "OL", "OL": "OL",
       "OT": "OL", "OG": "OL", "DT": "DL", "NT": "DL", "DL": "DL", "DE": "EDGE", "OLB": "LB", "ILB": "LB", "LB": "LB",
       "CB": "DB", "S": "DB", "SAF": "DB", "DB": "DB", "FS": "DB", "SS": "DB"}


def main():
    cn, college, have = connections_ext()
    p = nr.load_draft_picks().to_pandas()
    p = p[p.season.isin([2025, 2026])].copy()
    p["fr"] = p.team.map(co.franchise)
    p["key"] = p.college.map(college)
    p["grp"] = p.position.map(GRP).fillna(p.position)
    nt = cn.groupby(["year", "key"]).fr.nunique().rename("n_teams_linked").reset_index()
    p = p.merge(nt, left_on=["season", "key"], right_on=["year", "key"], how="left").drop(columns="year")
    p["n_teams_linked"] = p.n_teams_linked.fillna(0).astype(int)
    links = p.merge(cn, left_on=["fr", "season", "key"], right_on=["fr", "year", "key"])
    staffs = pd.read_parquet(co.OUT / "nfl_staffs_2025.parquet").drop_duplicates(["fr", "year", "page"])
    links = links.merge(staffs[["fr", "year", "page", "section"]], on=["fr", "year", "page"], how="left")
    links["recent"] = links["last"] >= links.season - 2
    links["match"] = [pos_match(r, s, g) for r, s, g in zip(links.role, links.section, links.grp)]
    links["kind"] = links.role.map(role_kind)
    links["coach"] = links.name + " (" + links.role + ", " + links["first"].astype(str) + "-" + links["last"].astype(str) + ")"
    agg = links.groupby(["season", "pfr_player_name"]).agg(
        coaches=("coach", lambda s: "; ".join(sorted(set(s)))), recent=("recent", "any"), role_match=("match", "any"),
        lead=("kind", lambda s: (s == "lead").any())).reset_index()
    out = p.merge(agg, on=["season", "pfr_player_name"], how="inner")
    cols = ["season", "round", "pick", "team", "pfr_player_name", "position", "college", "coaches", "recent",
            "role_match", "lead", "n_teams_linked"]
    out = out[cols].sort_values(["season", "pick"])
    out.to_csv("reports/connected_rookies_2025_26.csv", index=False)
    for y in (2025, 2026):
        n = (p.season == y).sum()
        print(f"{y}: {n} picks, {(out.season == y).sum()} connected ({(out.season == y).sum() / n:.1%}); "
              f"rounds 1-2 {((out.season == y) & (out['round'] <= 2)).sum()}; "
              f"skill {((out.season == y) & out.position.isin(['QB', 'RB', 'WR', 'TE'])).sum()}")
    sk = out[out.position.isin(["QB", "RB", "WR", "TE"])]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print("\nSKILL POSITIONS (fantasy):")
    print(sk.to_string(index=False))
    print("\nALL ROUNDS 1-2:")
    print(out[out["round"] <= 2][["season", "pick", "team", "pfr_player_name", "position", "college", "recent",
                                   "role_match"]].to_string(index=False))


if __name__ == "__main__":
    main()
