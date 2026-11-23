"""Captain own-value robustness (team_leadership_test secondary): age, no QBs, pick range, position, class. The 2023 tail errors (rolling needs 4 seasons); see captain_value_2023.py."""
import sys; sys.path[:0]=['src','scripts']
import numpy as np, pandas as pd, prospect_model as pm
from multisport_test import compare, expectations
from team_leadership_test import captains
cap=captains(); cap['player_id']=cap.player_id.astype(str)
cap=cap[['player_id','captain','log_bg','background','strengths','summary']]
uni=pm.load()
E=expectations(uni,(2019,2020,2021,2022)); E['player_id']=E.player_id.astype(str)
P=E.merge(cap,on='player_id'); P=P[P.drafted.astype(bool)].copy(); P['captain']=P.captain.astype(bool)
P['age']=P.age_at_draft.fillna(P.age_at_draft.median())
txt=(P.background.fillna('')+' '+P.strengths.fillna('')+' '+P.summary.fillna('')).str.lower()
P['leader']=txt.str.contains('leader')
print(pd.crosstab(P.grp,P.captain)); print('age', P.groupby('captain').age.mean().round(2).to_dict(), 'pick', P.groupby('captain').pick.median().to_dict())
c=('log_bg',)
compare(P,'captain','exp_slot',covs=c+('age',),label='age control')
compare(P[P.grp!='QB'],'captain','exp_slot',covs=c,label='no QBs')
compare(P[P.pick<=100],'captain','exp_slot',covs=c,label='picks 1-100')
compare(P[P.pick>100],'captain','exp_slot',covs=c,label='picks 101+')
compare(P,'leader','exp_slot',covs=c,label='leader word (not captain)')
for g in ['QB','RB','WR','TE','OL','DL','LB','DB','EDGE','IDL','CB','S','ED']:
    q=P[P.grp==g]
    if q.captain.sum()>=15: compare(q,'captain','exp_slot',covs=c,label=f'grp {g}')
for y,g in P.groupby('cls'):
    r=lambda x: x.actual.sum()/x.exp_slot.sum(); print(y, int(g.captain.sum()), round(r(g[g.captain])/r(g[~g.captain]),2))
# 2023 class, 3 seasons, partial replication (slot model fit on earlier classes)
import prospect_model as pm2
saved=pm.SEASONS
try:
    pm.SEASONS=(1,2,3)
    E3=expectations(uni,(2023,))
finally:
    pm.SEASONS=saved
E3['player_id']=E3.player_id.astype(str)
Q=E3.merge(cap,on='player_id'); Q=Q[Q.drafted.astype(bool)].copy(); Q['captain']=Q.captain.astype(bool)
print('2023 cols', [c for c in E3.columns if c.startswith('y')], Q[['y1','y2','y3']].notna().mean().round(2).to_dict())
compare(Q,'captain','exp_slot',covs=c,label='2023 class, seasons 1-3 (replication)')
