"""A correction for players who sat out healthy (lead from scripts/availability_split.py: actual / forecast 0.77 for
players who missed 25%+ of their team's games healthy - benched, scratched, practice squad).

Production forecast (veteran.comps(), skip 1, with the season in progress p_share / p_gp), 4000 random anchors
2014-24 (same sample as the earlier tests). h = max(healthy-absence share in the anchor's last full season, in weeks 1-11
of the anchor season) (veteran.availability).
Variants, each factor fit leaving the anchor's season out:
  group   forecast x f_k for h >= 0.25 (f_k = actual / forecast of that group in the other seasons, per horizon k)
  slope   forecast x (1 - b_k x h), b_k fit by least squares on the ratio in the other seasons, clipped to [0.3, 1]
Pre-set: adopt a variant if season-2 RMSE falls with a 95% interval below 0, the group's actual / forecast moves into
0.9-1.1, and the others' calibration doesn't move by more than 0.02.
Run: .venv/bin/python scripts/healthy_absence_test.py
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = ["src"]
import veteran as vt  # noqa: E402


def main():
    pd.set_option("display.width", 220)
    C = vt.comps()
    A = C.A
    S = A[A.Y.between(2014, 2024)].sample(4000, random_state=5).copy()
    h1 = {Y: vt.availability(int(Y) - 1).a_healthy for Y in S.Y.unique()}
    hp = {Y: vt.availability(int(Y), vt.PARTIAL_W).a_healthy for Y in S.Y.unique()}
    S["h"] = [max(float(h1[y].get(p, 0.0)), float(hp[y].get(p, 0.0))) for p, y in zip(S.pfr_id, S.Y)]
    S["grp_h"] = S.h >= 0.25
    for k in (2, 3):
        S[f"f{k}"] = [C.outcomes(r, k)[0].mean() if pd.notna(r[f"y{k}"]) else np.nan for r in S.to_dict("records")]
    print(f"anchors {len(S)}; healthy absence >= 25%: {S.grp_h.sum()} ({S.grp_h.mean():.0%}); "
          f"mean season-2 forecast in that group {S[S.grp_h].f2.mean() * 100:.2f}% of cap vs {S[~S.grp_h].f2.mean() * 100:.2f}%")
    rows = []
    for k in (2, 3):
        d = S[S[f"y{k}"].notna()].copy()
        a, f = d[f"y{k}"].to_numpy(), d[f"f{k}"].to_numpy()
        corr = {"group": np.empty(len(d)), "slope": np.empty(len(d))}
        for y in d.Y.unique():
            tr, te = d.Y.to_numpy() != y, d.Y.to_numpy() == y
            g = d.grp_h.to_numpy()
            fg = a[tr & g].sum() / f[tr & g].sum()
            corr["group"][te] = np.where(g[te], f[te] * fg, f[te])
            # slope: ratio ~ 1 - b h on the training seasons (forecast-weighted least squares)
            hh, rr, ww = d.h.to_numpy()[tr], (a[tr] / np.maximum(f[tr], 1e-6)).clip(0, 5), f[tr]
            b = np.sum(ww * hh * (1 - rr)) / max(np.sum(ww * hh * hh), 1e-9)
            corr["slope"][te] = f[te] * np.clip(1 - b * d.h.to_numpy()[te], 0.3, 1.0)
        e0 = (a - f) ** 2
        g = d.grp_h.to_numpy()
        for name, fc in corr.items():
            e1 = (a - fc) ** 2
            rng = np.random.default_rng(0)
            bs = [(e1 - e0)[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
            rows.append({"season": k, "variant": name, "RMSE current": np.sqrt(e0.mean()) * 100,
                         "RMSE corrected": np.sqrt(e1.mean()) * 100,
                         "MSE change (x1e4)": f"{(e1 - e0).mean() * 1e4:+.4f} ({np.percentile(bs, 2.5) * 1e4:+.4f} to "
                                              f"{np.percentile(bs, 97.5) * 1e4:+.4f})",
                         "group a/f current": a[g].sum() / f[g].sum(), "group a/f corrected": a[g].sum() / fc[g].sum(),
                         "others a/f current": a[~g].sum() / f[~g].sum(), "others a/f corrected": a[~g].sum() / fc[~g].sum()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # what the group looks like, and the all-data factors (for the tool if adopted)
    d = S[S.y2.notna()]
    print("\nall-seasons factors: group season 2 %.3f, season 3 %.3f" % (
        d[d.grp_h].y2.sum() / d[d.grp_h].f2.sum(), S[S.grp_h & S.y3.notna()].y3.sum() / S[S.grp_h & S.y3.notna()].f3.sum()))
    d = d.assign(hb=pd.cut(d.h, [-0.01, 0, 0.25, 0.5, 0.75, 1.0]))
    print("actual / forecast season 2 by healthy-absence share:",
          d.groupby("hb", observed=True).apply(lambda x: f"{x.y2.sum() / x.f2.sum():.2f} (n {len(x)})",
                                               include_groups=False).to_dict())


if __name__ == "__main__":
    main()
