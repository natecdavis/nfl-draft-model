"""What is inside contact with a draft prospect worth, through which coaches, and why?

Base: drafted players 2006-2022 (labels, seasons 1-4), expected value from the slot model fit leave-one-class-out
over 2006-22 (per season). Coaching connection as in src/coaching.py (drafting team's draft-year staff coached at
the player's final college in the 5 prior seasons); staffs 2006-24 (2006-11 from scripts/staffs_pre2012.py; picks
by team-seasons without a parsed staff dropped). "Fair group" = players whose school had a link to some NFL staff.

Specified before looking:
  1. Pooled effect (fair group, class FE, log # connected teams) and a dollar value: extra cap-years per connected
     pick = (ratio - 1) x mean expected value of connected picks, priced at the 2025 cap ($279.2M); connected picks
     per coach-season and per team-season.
  2. Role (NFL staff role of the connected coach, each vs unconnected in the fair group):
       head coach / coordinator; position coach; assistant / quality control / other.
     Position match: the connected coach's role names the player's position group (DL coach -> DL / EDGE, LB coach
     -> LB / EDGE, secondary -> DB, ...), or coordinator / head coach on the player's side of the ball.
  3. Mechanism. (a) Usage vs quality, player-seasons 1-4 in the NFL (PFF NFL, scripts/darko_pff.py): snaps at the same
     pre-season talent estimate (favoritism would raise them) and PFF grade at >= 200 snaps (quality), controls
     log pick, group, career season, class. (b) Coach departure: connected picks' season values vs slot expectation
     in seasons when the connected coach is still on the drafting team's staff vs gone. Information at the draft
     predicts the edge persists; favoritism / familiarity predicts it fades.
  4. Other contact: Senior Bowl (NFL staffs coached the two teams 2007-2023; Wikipedia "{year} Senior Bowl"):
     drafted Senior Bowl players taken by the team that coached them vs by others; selection rate vs chance.
     Pre-draft visits were tested in scripts/visits_test.py (x1.09 vs slot, 2015-22).
"""
import re
import sys

import nflreadpy as nr
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import coaching as co
import prospect_model as pm
from multisport_test import compare
from old_class_replication import connections_all
from old_class_replication3 import base

OUT = pm.OUT
CLASSES = range(2006, 2023)
CAP_2025 = 279.2  # $M
POS_ROLE = [(r"quarterback", {"QB"}), (r"running back", {"RB"}), (r"wide receiver|receivers", {"WR"}),
            (r"tight end", {"TE"}), (r"offensive line|tackles|guards|centers", {"OL"}),
            (r"defensive line|defensive tackle|defensive end|pass rush", {"DL", "EDGE"}),
            (r"linebacker", {"LB", "EDGE"}), (r"secondary|defensive back|safet|cornerback", {"DB"})]
OFF = {"QB", "RB", "WR", "TE", "OL"}


def role_kind(role):
    r = str(role).lower()
    if re.search(r"quality control|assistant|intern|fellow|analyst|consultant|senior", r) and "head coach" not in r.split("/")[0]:
        return "assistant"
    if r.startswith("head coach") or "coordinator" in r:
        return "lead"
    if any(re.search(p, r) for p, _ in POS_ROLE):
        return "position"
    return "assistant"


def pos_match(role, section, grp):
    r = str(role).lower()
    for p, gs in POS_ROLE:
        if re.search(p, r) and grp in gs:
            return True
    if r.startswith("head coach"):
        return True
    if "coordinator" in r:
        side = "off" if "offens" in r or "offens" in str(section).lower() else "def" if "defens" in r or "defens" in str(section).lower() else None
        return side == ("off" if grp in OFF else "def")
    return False


def base_frame():
    L = pd.read_parquet(pm.LABEL_FILE)
    d = L[L.draft_year.between(min(CLASSES), max(CLASSES)) & L.drafted.astype(bool)].copy()
    d = d.dropna(subset=["overall_pick", "group"]).drop_duplicates("pfr_id")
    d["cls"] = d.draft_year.astype(int)
    d["grp"] = d.group
    d["pick"] = d.overall_pick.astype(int)
    d["log_pick"] = np.log(d.pick + 10)
    d["grp_code"] = d.grp.astype("category").cat.codes
    for t in pm.SEASONS:
        d[f"y{t}"] = d[f"value_y{t}"].fillna(0.0).clip(lower=0)
        d[f"e{t}"] = 0.0
    for c in CLASSES:
        te = d.cls == c
        for t in pm.SEASONS:
            m, feats = pm.fit(d[~te], "slot", t)
            d.loc[te, f"e{t}"] = m.predict(d.loc[te, feats])
    d["actual"] = d[[f"y{t}" for t in pm.SEASONS]].sum(axis=1)
    d["exp_slot"] = d[[f"e{t}" for t in pm.SEASONS]].sum(axis=1)
    return d


def picks(d):
    cn, college, have = connections_all()
    staffs = pd.concat([pd.read_parquet(OUT / "nfl_staffs_2006.parquet"), pd.read_parquet(OUT / "nfl_staffs.parquet")])
    roles = staffs.dropna(subset=["page"]).groupby(["fr", "year", "page"]).agg(section=("section", "first"))
    p = nr.load_draft_picks().to_pandas()
    p = p[p.season.between(min(CLASSES), max(CLASSES))].copy()
    p["fr"] = p.team.map(co.franchise)
    p["key"] = p.college.map(college)
    have_set = set(zip(have.fr, have.year))
    p = p[[(f, y) in have_set for f, y in zip(p.fr, p.season)]]
    v = d.merge(p[["pfr_player_name", "pfr_player_id", "fr", "key", "college"]],
                left_on="pfr_id", right_on="pfr_player_id", how="inner")
    c = cn.merge(roles.reset_index(), on=["fr", "year", "page"], how="left")
    links = v[["pfr_id", "fr", "cls", "key", "grp"]].merge(c, left_on=["fr", "cls", "key"], right_on=["fr", "year", "key"])
    links["kind"] = links.role.map(role_kind)
    links["match"] = [pos_match(r, s, g) for r, s, g in zip(links.role, links.section, links.grp)]
    links["recent"] = links["last"] >= links.cls - 2
    agg = links.groupby("pfr_id").agg(conn=("page", "size"), lead=("kind", lambda s: (s == "lead").any()),
                                      position=("kind", lambda s: (s == "position").any()),
                                      assistant=("kind", lambda s: (s == "assistant").all()),
                                      match=("match", "any"), recent=("recent", "any"),
                                      pages=("page", lambda s: sorted(set(s))))
    v = v.merge(agg, on="pfr_id", how="left")
    v["conn"] = v.conn.fillna(0) > 0
    for f in ["lead", "position", "assistant", "match", "recent"]:
        v[f] = v[f].fillna(False).astype(bool)
    nt = cn.groupby(["year", "key"]).fr.nunique().rename("n_teams_conn").reset_index()
    v = v.merge(nt, left_on=["cls", "key"], right_on=["year", "key"], how="left").drop(columns="year")
    v["n_teams_conn"] = v.n_teams_conn.fillna(0)
    v["log_nconn"] = np.log1p(v.n_teams_conn)
    return v, links, c, staffs, have


def with_fe(s):
    fe = pd.get_dummies(s.cls, prefix="c", drop_first=True, dtype=float)
    return pd.concat([s.reset_index(drop=True), fe.reset_index(drop=True)], axis=1), ("log_nconn", *fe.columns)


def part1(v, c, have):
    s, covs = with_fe(v[v.n_teams_conn > 0])
    print(f"\n1. POOLED 2006-22: drafted with staff data {len(v)}; connected {v.conn.sum()} ({v.conn.mean():.1%}); "
          f"fair group {len(s)}")
    compare(s, "conn", "exp_slot", covs=covs, label="connected vs not (fair group)")
    r = lambda x: x.actual.sum() / x.exp_slot.sum()
    ratio = r(s[s.conn]) / r(s[~s.conn])
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        a, b = s[s.conn].sample(frac=1, replace=True, random_state=int(rng.integers(1e9))), \
            s[~s.conn].sample(frac=1, replace=True, random_state=int(rng.integers(1e9)))
        bs.append((r(a) / r(b) - 1) * a.exp_slot.mean())
    e = s[s.conn].exp_slot.mean()
    extra = (ratio - 1) * e
    lo, hi = np.percentile(bs, [5, 95])
    print(f"   connected picks: mean expected {e:.4f} cap-years (= ${e * CAP_2025:.1f}M of 2025 cap over 4 seasons)")
    print(f"   extra per connected pick: {extra:.4f} cap-years = ${extra * CAP_2025:.2f}M (90% ${lo * CAP_2025:.2f}M to "
          f"${hi * CAP_2025:.2f}M)")
    for lab, g in [("rounds 1-2", s[s["round"] <= 2]), ("rounds 3-7", s[s["round"] >= 3])]:
        rr = r(g[g.conn]) / r(g[~g.conn])
        print(f"   {lab}: {g.conn.sum()} connected, ratio x{rr:.2f}, mean expected {g[g.conn].exp_slot.mean():.4f}, "
              f"extra per pick ${(rr - 1) * g[g.conn].exp_slot.mean() * CAP_2025:.2f}M")
    ts = have[have.year.isin(CLASSES)]
    per_team = v.conn.sum() / len(ts)
    cs = c[c.year.isin(CLASSES) & c.key.isin(set(v.key.dropna()))].groupby(["fr", "year", "page"]).size()  # college programs only (careers also key NFL jobs)
    staff_coaches = pd.concat([pd.read_parquet(OUT / "nfl_staffs_2006.parquet"), pd.read_parquet(OUT / "nfl_staffs.parquet")])
    staff_coaches = staff_coaches[staff_coaches.year.isin(CLASSES)].dropna(subset=["page"]).drop_duplicates(["fr", "year", "page"])
    per_linked_coach = v.conn.sum() / len(cs)
    print(f"   connected picks per team-season {per_team:.2f}; coach-seasons with any college link in the 5-year window "
          f"{len(cs)} of {len(staff_coaches)} ({len(cs) / len(staff_coaches):.0%}); connected picks per linked coach-season "
          f"{per_linked_coach:.2f} (shared when 2+ coaches link)")
    return extra, per_team


def part2(v):
    s, covs = with_fe(v[v.n_teams_conn > 0])
    print("\n2. ROLE of the connected coach (each vs unconnected, fair group)")
    for f, lab in [("lead", "head coach / coordinator"), ("position", "position coach"),
                   ("assistant", "assistant / QC only"), ("match", "role matches player's position / side"),
                   ("recent", "coached there in last 2 seasons")]:
        g = s[s[f] | ~s.conn]
        compare(g, f, "exp_slot", covs=covs, label=lab)
    g = s[s.conn]
    g = g.assign(nomatch=~g.match)
    compare(pd.concat([g[~g.match].assign(flag=True), s[~s.conn].assign(flag=False)]), "flag", "exp_slot", covs=covs,
            label="connected, role does NOT match")


def part3(v, links, staffs):
    print("\n3. MECHANISM")
    pl = nr.load_players().to_pandas()[["pfr_id", "pff_id"]].dropna().drop_duplicates("pfr_id")
    pl["pff_id"] = pl.pff_id.astype(int)
    dk = pd.read_parquet(OUT / "darko_pff.parquet")[["player_id", "season", "snaps", "grade", "talent_pre"]]
    s = v[v.n_teams_conn > 0].merge(pl, left_on="pfr_id", right_on="pfr_id", how="left")
    rows = []
    for t in pm.SEASONS:
        x = s[["pfr_id", "pff_id", "cls", "grp", "log_pick", "conn", "fr", "pages", "log_nconn"]].copy()
        x["t"], x["season"] = t, x.cls + t - 1
        x[f"y"] = s[f"y{t}"].values
        x[f"e"] = s[f"e{t}"].values
        rows.append(x)
    ps = pd.concat(rows, ignore_index=True)
    ps = ps.merge(dk.rename(columns={"player_id": "pff_id"}), on=["pff_id", "season"], how="left")
    ps["snaps"] = ps.snaps.fillna(0)
    q = ps[ps.talent_pre.notna()].copy()
    X = lambda df, extra: sm.add_constant(pd.concat([df[["conn", "log_pick", *extra]].astype(float),
                                                     pd.get_dummies(df.grp, prefix="g", drop_first=True, dtype=float),
                                                     pd.get_dummies(df.t, prefix="t", drop_first=True, dtype=float),
                                                     pd.get_dummies(df.cls, prefix="c", drop_first=True, dtype=float)], axis=1))
    m = sm.OLS(np.log1p(q.snaps), X(q, ["talent_pre"])).fit(cov_type="cluster", cov_kwds={"groups": q.pfr_id.astype("category").cat.codes})
    print(f"   (a) usage: snaps at the same pre-season talent estimate, connected x{np.exp(m.params.conn):.2f} "
          f"(p {m.pvalues.conn:.2f}; {len(q)} player-seasons in PFF)")
    g = ps[ps.snaps >= 200].dropna(subset=["grade"])
    m2 = sm.OLS(g.grade, X(g, [])).fit(cov_type="cluster", cov_kwds={"groups": g.pfr_id.astype("category").cat.codes})
    print(f"       quality: PFF grade at >= 200 snaps, connected {m2.params.conn:+.2f} points (p {m2.pvalues.conn:.2f}; "
          f"{len(g)} player-seasons; grade sd {g.grade.std():.1f})")
    # (b) coach departure
    on = staffs.dropna(subset=["page"]).groupby(["fr", "year"]).page.agg(set).to_dict()
    known = lambda f, y: (f, y) in on
    c = ps[ps.conn].copy()
    c = c[[known(f, y) for f, y in zip(c.fr, c.season)]]
    c["present"] = [bool(set(p) & on[(f, y)]) for p, f, y in zip(c.pages, c.fr, c.season)]
    u = ps[~ps.conn]
    rr = lambda x: x.y.sum() / x.e.sum()
    print(f"   (b) coach departure, connected player-seasons with staff data {len(c)}: connected coach still on staff "
          f"{c.present.mean():.0%}")
    for t in pm.SEASONS:
        a, b, n = c[(c.t == t) & c.present], c[(c.t == t) & ~c.present], u[u.t == t]
        print(f"       season {t}: coach present x{rr(a) / rr(n):.2f} (n {len(a)}), coach gone x{rr(b) / rr(n):.2f} "
              f"(n {len(b)})  [vs unconnected same season]")
    a, b, n = c[c.present & (c.t >= 2)], c[~c.present & (c.t >= 2)], u[u.t >= 2]
    print(f"       seasons 2-4: present x{rr(a) / rr(n):.2f} (n {len(a)}), gone x{rr(b) / rr(n):.2f} (n {len(b)})")
    z = c[c.t >= 2].copy()
    z["gone"] = (~z.present).astype(float)
    Z = sm.add_constant(pd.concat([z[["gone"]], pd.get_dummies(z.t, prefix="t", drop_first=True, dtype=float)], axis=1))
    mm = sm.GLM(z.y, Z, family=sm.families.Poisson(), offset=np.log(z.e.clip(lower=1e-4))).fit(
        cov_type="cluster", cov_kwds={"groups": z.pfr_id.astype("category").cat.codes})
    print(f"       Poisson, connected seasons 2-4: coach gone x{np.exp(mm.params.gone):.2f} vs present "
          f"(p {mm.pvalues.gone:.2f})")


def senior_bowl(v):
    t = co.fetch([f"{y} Senior Bowl" for y in range(2007, 2024)], "seniorbowl")
    name_to_team = {re.sub(r"\s+", " ", n.lower()): a for a, n in co.TEAMS.items()}
    name_to_team["washington redskins"] = "WAS"
    rows = []
    for y in range(2007, 2024):
        x = t.get(f"{y} Senior Bowl") or ""
        side = {}
        for who in ("visitor", "home"):
            m = re.search(rf"\|\s*{who}_coach\s*=([^\n]*)", x)
            if m:
                teams = [l.group(1) for l in co.LINK.finditer(m.group(1))]
                tm = [name_to_team.get(tt.lower()) for tt in teams if tt.lower() in name_to_team]
                if tm:
                    side[who] = co.franchise(tm[-1])
        m = re.search(r"\|\s*home\s*=\s*([^\n|]*)", x)
        home = (m.group(1).strip().lower() if m else "")
        sec_home = "american" if y >= 2018 else "south"
        sec_vis = "national" if y >= 2018 else "north"
        if "american" in home or "south" in home:
            sec_home = "american" if "american" in home else "south"
            sec_vis = "national" if sec_home == "american" else "north"
        for sec, who in [(sec_home, "home"), (sec_vis, "visitor")]:
            mm = re.search(rf"(?im)^===\s*{sec}\s+team\s*===\s*$", x)
            if not mm or who not in side:
                continue
            nxt = re.search(r"(?m)^==[^=]", x[mm.end():])
            nx2 = re.search(r"(?m)^===[^=]", x[mm.end():])
            end = min([q.start() for q in (nxt, nx2) if q] or [len(x) - mm.end()])
            block = x[mm.end(): mm.end() + end]
            for l in co.LINK.finditer(block):
                tgt = l.group(1)
                if re.search(r"football|university|college|state|NFL|\bdraft\b", tgt, re.I) and "|" not in l.group(0):
                    continue
                nm = l.group(2) or re.sub(r"\s*\(.*\)$", "", tgt)
                if "," in nm:  # "Last, First"
                    a, b = nm.split(",", 1)
                    nm = f"{b.strip()} {a.strip()}"
                rows.append({"year": y, "name": base(nm)[0], "coach_fr": side[who]})
    sb = pd.DataFrame(rows).drop_duplicates(["year", "name"])
    print(f"\n4. SENIOR BOWL: rostered players parsed {len(sb)}; years {sorted(sb.year.unique())}; per year "
          f"{sb.groupby('year').size().median():.0f} (median)")
    w = v.copy()
    w["name"] = [base(n)[0] for n in w.pfr_player_name]
    w = w.merge(sb, left_on=["cls", "name"], right_on=["year", "name"], how="inner")
    w["sb_contact"] = w.fr == w.coach_fr
    print(f"   drafted Senior Bowl players matched {len(w)}; drafted by the team that coached them {w.sb_contact.sum()} "
          f"({w.sb_contact.mean():.1%}; chance ~{1 / 32:.1%} per player x picks share)")
    teams_picks = v.groupby(["cls", "fr"]).size()
    exp = np.mean([teams_picks.get((y, f), 0) / teams_picks.loc[y].sum() for y, f in zip(w.cls, w.coach_fr)])
    print(f"   expected if coaching team drafted at its normal rate: {exp:.1%} -> x{w.sb_contact.mean() / exp:.2f}")
    s = w.copy()
    s["log_nconn"] = 0.0
    s, covs = with_fe(s)
    compare(s, "sb_contact", "exp_slot", covs=covs[1:], label="coached at Senior Bowl, drafted by that team", min_n=5)
    compare(s[s.pick <= 100], "sb_contact", "exp_slot", covs=covs[1:], label="  picks 1-100", min_n=5)
    print("   cases: " + ", ".join(f"{n} ({fr} {y} #{p})" for n, fr, y, p in
                                   w[w.sb_contact].sort_values("pick")[["pfr_player_name", "fr", "cls", "pick"]].head(25).values))
    return w[["pfr_id", "sb_contact"]]


def main():
    d = base_frame()
    print(f"drafted 2006-22: {len(d)}; actual / slot x{d.actual.sum() / d.exp_slot.sum():.2f}")
    v, links, c, staffs, have = picks(d)
    part1(v, c, have)
    part2(v)
    part3(v, links, staffs)
    sb = senior_bowl(v)
    v = v.merge(sb, on="pfr_id", how="left")
    v["sb_contact"] = v.sb_contact.fillna(False).astype(bool)
    v["any_contact"] = v.conn | v.sb_contact
    s = v[(v.n_teams_conn > 0) | v.sb_contact].copy()
    s, covs = with_fe(s)
    print("\n5. ANY CONTACT (coaching or Senior Bowl staff), players with some possible contact")
    compare(s, "any_contact", "exp_slot", covs=covs, label="coaching or Senior Bowl")
    v.drop(columns=["pages"]).to_parquet(OUT / "contact_value.parquet")


if __name__ == "__main__":
    main()
