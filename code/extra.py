"""Episode accounting, quarterly import function, robustness of the sector regressions."""
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf, json
from statsmodels.tsa.stattools import coint
D="../data/"; O="../results/"
fx=pd.read_csv(O+"fxpos.csv",index_col=0); fx.index=pd.PeriodIndex(fx.index,freq="M")
res={}
# --- episodes: balance-of-payments financing and sectoral positions
ep={"2012m12--2018m7":("2013-01","2018-07"),"2018m8--2021m11":("2018-08","2021-11"),"2021m12--2023m5":("2021-12","2023-05"),"2023m6--2026m9":("2023-06","2026-09")}
rows=[]
for k,(a,b) in ep.items():
    w=fx.loc[a:b]; a0=(pd.Period(a,freq="M")-1)
    r={"episode":k,"months":len(w),"ca":w.ca.sum(),"neo":w.neo.sum(),"res_bop":w.res_bop.sum()}
    for v in ["nfc_net","hh_fxdep","kkm_usd","net_res_exswap","public_net","gross_res"]: r["d_"+v]=fx[v].loc[b]-fx[v].loc[a0]
    r["dep"]=100*np.log(fx.usdtry.loc[b]/fx.usdtry.loc[a0]); r["dreer"]=100*np.log(fx.reer.loc[b]/fx.reer.loc[a0])
    rows.append(r)
E1=pd.DataFrame(rows); E1.to_csv(O+"episodes.csv",index=False); print(E1.round(1).T.to_string())
lv=fx.loc[["2012-12","2018-07","2021-11","2023-05","2026-09"],["nfc_net","hh_fxdep","kkm_usd","net_res_exswap","public_net","gross_res","usdtry","reer","dollar","dollar_kkm"]]
lv.to_csv(O+"levels.csv"); print(lv.round(1).to_string())
# mirror: correlation of 12-month changes in NFC net position and public net position
d=fx[["nfc_net","public_net","hh_fxdep","kkm_usd"]].diff(12).dropna()
res["corr_nfc_public_12m"]=float(d.nfc_net.corr(d.public_net)); res["corr_nfc_public_12m_2018on"]=float(d.loc["2018-01":].nfc_net.corr(d.loc["2018-01":].public_net))
mod=sm.OLS(d.loc["2018-01":].public_net,sm.add_constant(d.loc["2018-01":].nfc_net)).fit(cov_type="HAC",cov_kwds={"maxlags":12})
res["slope_public_on_nfc"]=[float(mod.params.iloc[1]),float(mod.bse.iloc[1]),int(mod.nobs)]
# --- cointegration tests, annual
a=pd.read_csv(O+"annual.csv",index_col=0).dropna(subset=["q"])
res["eg_imp"]=[float(x) for x in coint(a.m,a[["y","q"]],trend="c",maxlag=1,autolag=None)[:2]]
res["eg_exp"]=[float(x) for x in coint(a.x,a[["z","q"]],trend="c",maxlag=1,autolag=None)[:2]]
# --- quarterly import and export functions, year-on-year growth
q=pd.read_csv(D+"tr_quarterly.csv"); g=q[q.group=="bie_gsyhhrczinc"].pivot(index="date",columns="code",values="value")
g.index=pd.PeriodIndex([s.replace("-Q","Q") for s in g.index],freq="Q"); g=np.log(g.rename(columns={"TP.GSYIH20.CY.B1GQ":"y","TP.GSYIH20.CY.P6":"x","TP.GSYIH20.CY.P7":"m"})[["y","x","m"]])
g["reer"]=np.log(fx.reer.groupby(fx.index.asfreq("Q")).mean())
dq=g.diff(4).dropna(); dq=dq.loc[:"2026Q2"]
for nm,sl in {"all":slice(None),"pre2018":slice(None,"2017Q4"),"post2018":slice("2018Q1",None)}.items():
    w=dq.loc[sl]; f=sm.OLS(w.m,sm.add_constant(w[["y","reer"]])).fit(cov_type="HAC",cov_kwds={"maxlags":4})
    f2=sm.OLS(w.x,sm.add_constant(w[["reer"]])).fit(cov_type="HAC",cov_kwds={"maxlags":4})
    res["q_imp_"+nm]={"y":[float(f.params.y),float(f.bse.y)],"reer":[float(f.params.reer),float(f.bse.reer)],"n":int(f.nobs),"x_reer":[float(f2.params.reer),float(f2.bse.reer)]}
print({k:v for k,v in res.items()})
# --- sector regressions: robustness
s=pd.read_csv(O+"sector_panel.csv",dtype={"sector":str}); old=s[s.vint=="old"].copy()
base=old[old.year.between(2010,2012)].groupby("sector").fxsh.mean().rename("fx0"); old=old.join(base,on="sector")
def W(x): lo,hi=x.quantile([.01,.99]); return x.clip(lo,hi)
r=old.dropna(subset=["dfin_s","fxsh_l","dep","fx0"]).copy(); r["dfin_w"]=W(r.dfin_s); r["dpbt_w"]=W(r.dpbt_s)
rob={}
def est(name,form,df,key):
    m=smf.ols(form,data=df).fit(cov_type="cluster",cov_kwds={"groups":df.sector}); rob[name]=[float(m.params[key]),float(m.bse[key]),float(m.pvalues[key]),int(m.nobs)]
r["x"]=r.fxsh_l*r.dep; r["x0"]=r.fx0*r.dep; r["xa"]=r.fxsh_l*r.dep_avg
est("base","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r,"x")
est("fixed_share","dfin_w ~ x0 + C(sector)+C(year)",r[r.year>=2013],"x0")
est("avg_dep","dfin_w ~ xa + fxsh_l + C(sector)+C(year)",r,"xa")
est("ex2018","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r[r.year!=2018],"x")
est("ex2021_23","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r[r.year<2021],"x")
est("manuf","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r[r.sector.astype(int).between(10,33)],"x")
est("unwins","dfin_s ~ x + fxsh_l + C(sector)+C(year)",r,"x")
est("pbt_fixed","dpbt_w ~ x0 + C(sector)+C(year)",r[r.year>=2013],"x0")
r["lev_l"]=r.groupby("sector").lev.shift(1); r["xl"]=r.lev_l*r.dep/100
est("with_leverage","dfin_w ~ x + xl + fxsh_l + lev_l + C(sector)+C(year)",r.dropna(subset=["lev_l"]),"x")
# weighted by assets
m=smf.wls("dfin_w ~ x + fxsh_l + C(sector)+C(year)",data=r,weights=r.assets/r.groupby("year").assets.transform("sum")).fit(cov_type="cluster",cov_kwds={"groups":r.sector}); rob["weighted"]=[float(m.params["x"]),float(m.bse["x"]),float(m.pvalues["x"]),int(m.nobs)]
for k,v in rob.items(): print(k,[round(x,3) for x in v])
res["rob"]=rob; res["fx_iqr"]=[float(r.fxsh_l.quantile(.25)),float(r.fxsh_l.quantile(.75))]
res["top_fx"]=old[old.year==2017].nlargest(6,"fxsh")[["sector","name","fxsh"]].values.tolist()
json.dump(res,open(O+"extra.json","w"),indent=1,default=str)
# --- further heterogeneity: timing of depreciation, tradable versus non-tradable sectors
r["manuf"]=r.sector.astype(int).between(10,33).astype(int)
est("avg_dep_pre2021","dfin_w ~ xa + fxsh_l + C(sector)+C(year)",r[r.year<2021],"xa")
est("nonmanuf","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r[r.manuf==0],"x")
est("nonmanuf_pre2021","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r[(r.manuf==0)&(r.year<2021)],"x")
est("manuf_pre2021","dfin_w ~ x + fxsh_l + C(sector)+C(year)",r[(r.manuf==1)&(r.year<2021)],"x")
r["post"]=(r.year>=2021).astype(int); r["xpost"]=r.x*r.post
mm=smf.ols("dfin_w ~ x + xpost + fxsh_l + fxsh_l:post + C(sector)+C(year)",data=r).fit(cov_type="cluster",cov_kwds={"groups":r.sector})
rob["post_interaction"]=[float(mm.params["xpost"]),float(mm.bse["xpost"]),float(mm.pvalues["xpost"]),int(mm.nobs)]; rob["pre_main"]=[float(mm.params["x"]),float(mm.bse["x"]),float(mm.pvalues["x"]),int(mm.nobs)]
est("pbt_pre2021","dpbt_w ~ x + fxsh_l + C(sector)+C(year)",r[r.year<2021],"x")
r["ponzi_f"]=r.ponzi
est("ponzi_pre2021","ponzi_f ~ x + fxsh_l + C(sector)+C(year)",r[r.year<2021],"x")
for k in ["avg_dep_pre2021","nonmanuf","nonmanuf_pre2021","manuf_pre2021","post_interaction","pre_main","pbt_pre2021","ponzi_pre2021"]: print(k,[round(x,3) for x in rob[k]])
print(r.groupby("manuf").fxsh_l.mean().round(2).to_dict(), r.groupby("year").dep.first().round(2).to_dict())
res["rob"]=rob; json.dump(res,open(O+"extra.json","w"),indent=1,default=str)
