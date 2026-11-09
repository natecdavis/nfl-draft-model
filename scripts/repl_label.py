"""Label with position-specific replacement levels (scripts/replacement_market.py, spread fit) subtracted.

value_t' = max(value_t - r_g x snap share_t x games fraction_t x 1.10, 0) for seasons 1-5 (r_g in this
project's units, % of cap per full season; x1.10 converts to cfdarko's). Snap seasons 2013+ use the real
share (cfdarko load_snaps, as the label does); earlier seasons (depth-chart values) use the share implied by
the value: isotonic fit of share x games fraction on value within group, 2013+ seasons.
Writes data/derived/labels_pickadj_narrow_repl.parquet (prospect_model LABEL="repl").
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import replacement_level as rl  # noqa: E402  (cfdarko import set-up)

rv = rl.rv
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "raw"  # raw | eb (shrunk) | pooled (one non-QB level)
_R = pd.read_csv(ROOT / "data/derived/replacement_market.csv")
_R = _R[_R.outcome == "spread_line"].set_index("group")


def levels(variant):
    """r_g in % of cap. eb: non-QB groups shrunk toward their precision-weighted mean (DerSimonian-Laird
    between-group variance); pooled: every non-QB group at that mean. QB keeps its own estimate."""
    r, se = _R.r.copy(), (_R.hi - _R.lo) / 3.92
    if variant == "raw":
        return r
    n = r.index != "QB"
    w = 1 / se[n] ** 2
    mu = (w * r[n]).sum() / w.sum()
    Q = (w * (r[n] - mu) ** 2).sum()
    tau2 = max(0.0, (Q - (n.sum() - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    r[n] = mu + (tau2 / (tau2 + se[n] ** 2)) * (r[n] - mu) if variant == "eb" else mu
    return r


R = levels(VARIANT).clip(lower=0) / 100  # share of cap per full season


def main():
    L = pd.read_parquet(ROOT / "data/derived/labels_pickadj_narrow.parquet")
    s = rv.load_snaps()
    s["share"] = s[["offense_pct", "defense_pct"]].max(axis=1)
    s["gfrac"] = (s["n_games"] / s["season"].map(rv.reg_season_games)).clip(upper=1.0)
    s = s.sort_values("share").groupby(["pfr_player_id", "season"])[["share", "gfrac"]].last()
    sg = (s.share * s.gfrac).to_dict()
    # isotonic share ~ value per group for pre-snap seasons
    rows = []
    for r in L.itertuples():
        for k in range(1, 6):
            y = r.draft_year + k - 1
            v = getattr(r, f"value_y{k}")
            if pd.notna(v) and y >= 2013:
                rows.append((r.group, v, sg.get((r.pfr_id, y), 0.0)))
    D = pd.DataFrame(rows, columns=["g", "v", "sg"])
    iso = {g: IsotonicRegression(increasing=True, out_of_bounds="clip").fit(G.v, G.sg) for g, G in D.groupby("g")}
    L2 = L.copy()
    cut = []
    for k in range(1, 6):
        col = f"value_y{k}"
        yr = L.draft_year + k - 1
        share = np.array([sg.get((p, y), 0.0) if y >= 2013 else np.nan for p, y in zip(L.pfr_id, yr)])
        pre = np.isnan(share) & L[col].notna()
        for g in iso:
            m = pre & (L.group == g).to_numpy()
            if m.any():
                share[m] = iso[g].predict(L.loc[m, col].to_numpy())
        r_g = L.group.map(R).fillna(0).to_numpy()
        new = np.where(L[col].notna(), np.maximum(L[col] - r_g * np.nan_to_num(share) * 1.10, 0), np.nan)
        cut.append(np.nansum(L[col]) - np.nansum(new))
        L2[col] = new
    vy = [f"value_y{k}" for k in range(1, 5)]
    L2["value_y14"] = L2[vy].sum(axis=1, min_count=4)
    L2["surplus_y14"] = L2.value_y14 - L2.cost_y14
    L2["surplus_y15"] = L2.surplus_y14 + L2.value_y5 - L2.cost_y5
    out = ROOT / f"data/derived/labels_pickadj_narrow_repl{'' if VARIANT == 'raw' else '_' + VARIANT}.parquet"
    L2.to_parquet(out)
    print("r_g (% cap / full season):", (R * 100).round(2).to_dict())
    d = L[L.drafted & L.draft_year.between(2014, 2022)]
    d2 = L2.loc[d.index]
    print("4-season value per drafted player 2014-22, old -> new (% cap), by group:")
    print(pd.DataFrame({"old": d.groupby("group").value_y14.mean() * 100,
                        "new": d2.groupby(d.group).value_y14.mean() * 100}).assign(
        ratio=lambda x: x.new / x.old).round(2).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
