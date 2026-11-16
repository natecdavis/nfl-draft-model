"""Strict follower, step 1b: does the realized edge survive room capacity? (scripts/strict_follower.py)

Follower picks that match the stored review recommendation (2014-22) are scored on their real careers
(review real_model) minus the team's real pick (real_team), then with snaps beyond the room removed:
proportional (projected snaps, everyone keeps room/total) and best-first (realized snaps, the players
who really played most fill the room first; most favorable to the follower).

Run: python scripts/strict_follower_realized.py
"""
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
import numpy as np, pandas as pd
import strict_follower as sf
from outlook import GROUPS, OUT
F = pd.read_parquet(sf.FOLLOW); F = F[F.grp.isin(GROUPS)].reset_index(drop=True)
sh = sf.season_shares(); pshare = sh.groupby(["pfr_player_id","season"]).share.sum()
room = sf.rooms(sh); room = room[room.season.between(2014,2025)].groupby("grp").share.median()
f = sf.expected_curve(sf.board_players(), pshare)
rows=[]
for i,r in F.iterrows():
    e=f(r.grp,r["rank"])
    for t in range(1,(5 if r.pick<=32 else 4)+1):
        rows.append((i,r.team,r.year+t-1,r.grp,e[min(t,4)-1]))
X=pd.DataFrame(rows,columns=["i","team","season","grp","proj"])
tot=X.groupby(["team","season","grp"]).proj.transform("sum")
X["keep"]=np.minimum(1,X.grp.map(room)/tot.where(tot>0,1))
X["pk"]=X.proj*X.keep
g=X.groupby("i")[["proj","pk"]].sum(); F["lost"]=1-g.pk/g.proj.clip(lower=1e-9)
rp=pd.read_parquet(OUT/"review_picks.parquet")
M=F.merge(rp[["year","team","pick","rec_player","real_model","real_team","grp"]].rename(columns={"grp":"agrp"}),
          left_on=["year","team","pick","name"],right_on=["year","team","pick","rec_player"])
M=M[M.year.between(2014,2022)]
print("follower picks 2014-22:",(F.year<=2022).sum(),"matched to review rec:",len(M))
M=M[M.agrp.ne("SPEC")]
d=M.real_model-M.real_team; dh=M.real_model*(1-M.lost)-M.real_team
def boot(x,B=2000):
    # cluster by team-draft
    k=x.groupby([M.year,M.team]).sum(); rng=np.random.default_rng(0)
    s=[k.sample(len(k),replace=True,random_state=rng.integers(1e9)).mean() for _ in range(B)]
    return f"{k.mean():+.2f} ({np.percentile(s,2.5):+.2f} to {np.percentile(s,97.5):+.2f})"
print("realized titles (%), follower pick minus team pick, per team-draft, matched picks, 2014-22:")
print("  as scored (players played where they really landed):", boot(d))
print("  with room overflow removed:                           ", boot(dh))
for lab,m in [("R1",M["round"]==1),("R2-4",M["round"].between(2,4)),("R5-7",M["round"]>=5)]:
    print(f"  {lab}: per pick raw {d[m].mean():+.3f}, after overflow {dh[m].mean():+.3f}, n={m.sum()}")
print("overflow share, matched picks, by group:"); print((M.real_model*M.lost).groupby(M.grp).sum().div(M.real_model.groupby(M.grp).sum()).round(2).to_string())

# favorable rule: realized snap shares, room filled best-first
dp=pd.read_parquet("data/raw/draft_picks.parquet")
pid={(int(s),int(k)):i for s,k,i in zip(dp.season,dp.pick,dp.pfr_player_id)}
F["pfr"]=[pid.get((y,int(k))) if pd.notna(k) else None for y,k in zip(F.year,F.drafted_at)]
rows=[]
for i,r in F.iterrows():
    for t in range(1,(5 if r.pick<=32 else 4)+1):
        s=r.year+t-1
        rows.append((i,r.team,s,r.grp,pshare.get((r.pfr,s),0.0) if isinstance(r.pfr,str) else 0.0))
Y=pd.DataFrame(rows,columns=["i","team","season","grp","real"]).sort_values("real",ascending=False)
Y["cum"]=Y.groupby(["team","season","grp"]).real.cumsum()
Y["kept"]=np.clip(Y.grp.map(room)-(Y.cum-Y.real),0,Y.real)
g=Y.groupby("i")[["real","kept"]].sum(); F["lost_best"]=(1-g.kept/g.real.where(g.real>0)).fillna(0)
M2=M.merge(F[["year","team","pick","lost_best"]],on=["year","team","pick"])
dh2=M2.real_model*(1-M2.lost_best)-M2.real_team
k=dh2.groupby([M2.year,M2.team]).sum(); rng=np.random.default_rng(0)
s=[k.sample(len(k),replace=True,random_state=rng.integers(1e9)).mean() for _ in range(2000)]
print(f"  best-first, realized snaps:  {k.mean():+.2f} ({np.percentile(s,2.5):+.2f} to {np.percentile(s,97.5):+.2f})")
for lab,m in [("R1",M2["round"]==1),("R2-4",M2["round"].between(2,4)),("R5-7",M2["round"]>=5)]:
    print(f"  {lab}: per pick {dh2[m].mean():+.3f}")
print("lost share best-first by group:"); print((M2.real_model*M2.lost_best).groupby(M2.grp).sum().div(M2.real_model.groupby(M2.grp).sum()).round(2).to_string())
