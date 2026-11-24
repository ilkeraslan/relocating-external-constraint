"""Sector-level Minsky taxonomy and FX-exposure regressions from CBRT company accounts."""
import pandas as pd, numpy as np, statsmodels.formula.api as smf, json
O="../results/"
p=pd.read_csv(O+"sector_raw.csv",dtype={"sector":str})
fx=pd.read_csv(O+"fxpos.csv",index_col=0); fx.index=pd.PeriodIndex(fx.index,freq="M")
e=fx.usdtry.groupby(fx.index.year).mean(); dep=np.log(e).diff()          # annual average depreciation
eoy=fx.usdtry[fx.index.month==12]; eoy.index=eoy.index.year; dep_eoy=np.log(eoy).diff()
p["two"]=p.sector.str.fullmatch(r"\d\d")
d=p[p.two].copy().sort_values(["vint","sector","year"])
d["stbank"]=d.stbank_stl/100*d.stl
d["stbank_l"]=d.groupby(["vint","sector"]).stbank.shift(1)
d["icr"]=d.op/d.fin
d["fxsh"]=d.cr_fx/(d.cr_fx+d.cr_try)
d["fxsh_l"]=d.groupby(["vint","sector"]).fxsh.shift(1)
d["fin_s"]=100*d.fin/d.sales; d["op_s"]=100*d.op/d.sales; d["pbt_s"]=100*d.pbt/d.sales
d["dfin_s"]=d.groupby(["vint","sector"]).fin_s.diff(); d["dpbt_s"]=d.groupby(["vint","sector"]).pbt_s.diff(); d["dop_s"]=d.groupby(["vint","sector"]).op_s.diff()
d["dep"]=d.year.map(dep_eoy); d["dep_avg"]=d.year.map(dep)
def cls(r):
    if r.op<r.fin: return "Ponzi"
    if np.isnan(r.stbank_l): return np.nan
    return "Hedge" if r.op>=r.fin+r.stbank_l else "Speculative"
d["cls"]=d.apply(cls,axis=1)
d["ponzi"]=(d.op<d.fin).astype(float)
d.to_csv(O+"sector_panel.csv",index=False)
old=d[d.vint=="old"]; new=d[d.vint=="new"]
def shares(x,w):
    t=x.dropna(subset=["cls"]).groupby(["year","cls"])[w].sum().unstack().fillna(0); return 100*t.div(t.sum(1),axis=0)
x=old.assign(n=1.0)
tab=pd.concat({"n":shares(x,"n"),"assets":shares(x,"assets")},axis=1)
pz=pd.DataFrame({"ponzi_n":100*old.groupby("year").ponzi.mean(),"ponzi_w":100*old.groupby("year").apply(lambda g: (g.ponzi*g.assets).sum()/g.assets.sum()),"icr_med":old.groupby("year").icr.median(),"N":old.groupby("year").size()})
agg=p[(p.sector=="GENEL")].set_index(["vint","year"]); pz["icr_agg"]=(agg.op/agg.fin).loc["old"]; pz["dep"]=100*dep_eoy
print(tab.round(1).to_string()); print(pz.round(2).to_string())
xn=new.assign(n=1.0); pzn=pd.DataFrame({"ponzi_n":100*new.groupby("year").ponzi.mean(),"ponzi_w":100*new.groupby("year").apply(lambda g:(g.ponzi*g.assets).sum()/g.assets.sum()),"icr_med":new.groupby("year").icr.median(),"icr_agg":(agg.op/agg.fin).loc["new"]})
print(pzn.round(2).to_string())
pz.to_csv(O+"minsky_old.csv"); pzn.to_csv(O+"minsky_new.csv"); tab.to_csv(O+"minsky_shares.csv")
# FX-exposure regressions, old vintage 2010-2023
res={}
r=old.dropna(subset=["dfin_s","fxsh_l","dep"]).copy()
lo,hi=r.dfin_s.quantile([.01,.99]); r["dfin_w"]=r.dfin_s.clip(lo,hi)
lo,hi=r.dpbt_s.quantile([.01,.99]); r["dpbt_w"]=r.dpbt_s.clip(lo,hi)
lo,hi=r.dop_s.quantile([.01,.99]); r["dop_w"]=r.dop_s.clip(lo,hi)
r["x"]=r.fxsh_l*r.dep
for y in ["dfin_w","dop_w","dpbt_w","ponzi"]:
    mod=smf.ols(f"{y} ~ x + fxsh_l + C(sector) + C(year)",data=r).fit(cov_type="cluster",cov_kwds={"groups":r.sector})
    res[y]={"b":mod.params["x"],"se":mod.bse["x"],"p":mod.pvalues["x"],"n":int(mod.nobs),"sectors":r.sector.nunique()}
    print(y, {k:round(v,3) for k,v in res[y].items()})
print(r.fxsh_l.describe().round(2).to_string())
json.dump(res,open(O+"minsky_reg.json","w"),indent=1)
