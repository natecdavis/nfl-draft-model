"""Captain own value, 2023 class holdout (3 seasons; lowess slot curve on log pick fit to 2016-22 seasons 1-3)."""
import sys; sys.path[:0]=['src','scripts']
import numpy as np, pandas as pd, prospect_model as pm
from statsmodels.nonparametric.smoothers_lowess import lowess
from team_leadership_test import captains
cap=captains(); cap['player_id']=cap.player_id.astype(str)
u=pm.load(); u['player_id']=u.player_id.astype(str)
u=u[u.drafted.astype(bool) & u.overall_pick.notna()].copy()
u['v3']=u[['value_y1','value_y2','value_y3']].sum(axis=1,min_count=3)
tr=u[u.cls.between(2016,2022) & u.v3.notna()]
fit=lowess(tr.v3, np.log(tr.overall_pick), frac=0.3, return_sorted=True)
exp=lambda p: np.interp(np.log(p), fit[:,0], fit[:,1])
def show(d,lab):
    d=d.copy(); d['e']=exp(d.overall_pick).clip(1e-4)
    f,o=d[d.captain],d[~d.captain]; r=lambda x: x.v3.sum()/x.e.sum()
    rng=np.random.default_rng(0)
    bs=[r(f.sample(len(f),replace=True,random_state=int(rng.integers(1e9))))/r(o.sample(len(o),replace=True,random_state=int(rng.integers(1e9)))) for _ in range(2000)]
    print(f'{lab:32s} n={len(f)} vs {len(o)} beat {(f.v3>f.e).mean():.0%} vs {(o.v3>o.e).mean():.0%} value x{r(f)/r(o):.2f} (95% {np.percentile(bs,2.5):.2f}-{np.percentile(bs,97.5):.2f})')
for yrs,lab in [((2019,2022),'2019-22 seasons 1-3 (check)'),((2023,2023),'2023 seasons 1-3 (holdout)')]:
    d=u[u.cls.between(*yrs)].merge(cap[['player_id','captain']],on='player_id'); d=d[d.v3.notna()]; d['captain']=d.captain.astype(bool)
    show(d,lab); show(d[d.overall_pick>100],'  picks 101+'); show(d[d.overall_pick<=100],'  picks 1-100')
