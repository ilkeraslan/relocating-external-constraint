"""Balance-of-payments-constrained growth: import and export demand functions for Turkey, 1980-2025."""
import pandas as pd, numpy as np, statsmodels.api as sm, json
D="../data/"; O="../results/"
w=pd.concat([pd.read_csv(D+"wb_annual.csv"),pd.read_csv(D+"wb_cpi.csv")]); w=w.pivot_table(index="year",columns=["country","indicator"],values="value")
T=w["TUR"]; a=pd.DataFrame({"y":np.log(T["NY.GDP.MKTP.KD"]),"x":np.log(T["NE.EXP.GNFS.KD"]),"m":np.log(T["NE.IMP.GNFS.KD"]),
   "z":np.log(w["WLD"]["NY.GDP.MKTP.KD"]),"zo":np.log(w["OED"]["NY.GDP.MKTP.KD"]),
   "ca":T["BN.CAB.XOKA.GD.ZS"],"xs":T["NE.EXP.GNFS.ZS"],"ms":T["NE.IMP.GNFS.ZS"]})
# real exchange rate against the US dollar: lira price of the dollar deflated by relative CPI (rise = real depreciation)
a["q"]=np.log(T["PA.NUS.FCRF"])+np.log(w["USA"]["FP.CPI.TOTL"])-np.log(T["FP.CPI.TOTL"])
a=a.dropna(subset=["y","x","m","z"]); a.to_csv(O+"annual.csv"); A=a.copy(); a=a.dropna(subset=["q"])
print(a.index.min(),a.index.max(),len(a))
def dols(dep,reg,df,k=1):
    X=df[reg].copy()
    for r in reg:
        dx=df[r].diff()
        for j in range(-k,k+1): X[f"d{r}_{j}"]=dx.shift(-j)
    X=sm.add_constant(X); d=pd.concat([df[dep],X],axis=1).dropna()
    f=sm.OLS(d[dep],d.drop(columns=dep)).fit(cov_type="HAC",cov_kwds={"maxlags":2})
    return f,len(d)
def ecm(dep,reg,df):
    d=pd.DataFrame({"dy":df[dep].diff(),"l":df[dep].shift(1)})
    for r in reg: d["l_"+r]=df[r].shift(1); d["d_"+r]=df[r].diff()
    d=d.dropna(); f=sm.OLS(d.dy,sm.add_constant(d.drop(columns="dy"))).fit(cov_type="HAC",cov_kwds={"maxlags":2})
    lr={r:-f.params["l_"+r]/f.params["l"] for r in reg}
    return f,lr,len(d)
res={}
for nm,(s0,s1) in {"full":(1987,2025),"pre2002":(1987,2001),"post2002":(2002,2025)}.items():
    df=a.loc[s0:s1]
    out={}
    for eq,dep,reg in [("imp","m",["y","q"]),("exp","x",["z","q"])]:
        if nm=="full":
            f,n=dols(dep,reg,df); out[eq+"_dols"]={r:(f.params[r],f.bse[r]) for r in reg}; out[eq+"_n"]=n
            g,lr,n2=ecm(dep,reg,df); out[eq+"_ecm"]={"lr":lr,"adj":(g.params["l"],g.bse["l"]),"t_adj":g.tvalues["l"],"n":n2}
        # growth-rate regression (first differences)
        d=pd.DataFrame({"dy":df[dep].diff()}); 
        for r in reg: d[r]=df[r].diff()
        d=d.dropna(); f2=sm.OLS(d.dy,sm.add_constant(d[reg])).fit(cov_type="HAC",cov_kwds={"maxlags":2})
        out[eq+"_fd"]={r:(f2.params[r],f2.bse[r]) for r in reg}
    res[nm]=out
pi=res["full"]["imp_dols"]["y"][0]; eps=res["full"]["exp_dols"]["z"][0]
psi=res["full"]["imp_dols"]["q"][0]; eta=res["full"]["exp_dols"]["q"][0]
print("pi",round(pi,2),"eps",round(eps,2),"psi",round(psi,2),"eta",round(eta,2),"ML sum",round(eta-psi,2))
per=[(1988,2001),(2002,2012),(2013,2017),(2018,2025),(2018,2021),(2022,2025),(1988,2025)]
rows=[]
for s0,s1 in per:
    g=lambda v: 100*(A[v].loc[s1]-A[v].loc[s0-1])/(s1-s0+1)
    rows.append({"period":f"{s0}--{s1}","y":g("y"),"x":g("x"),"m":g("m"),"z":g("z"),"q":100*(A.q.loc[min(s1,2024)]-A.q.loc[s0-1])/(min(s1,2024)-s0+1),"yB_weak":g("x")/pi,"yB_strong":eps*g("z")/pi,"ca":A.ca.loc[s0:s1].mean()})
tab=pd.DataFrame(rows); print(tab.round(2).to_string())
tab.to_csv(O+"thirlwall_periods.csv",index=False)
def cv(o):
    if isinstance(o,dict): return {k:cv(v) for k,v in o.items()}
    if isinstance(o,tuple): return [float(x) for x in o]
    return float(o) if isinstance(o,(np.floating,float,int,np.integer)) else o
json.dump(cv(res),open(O+"thirlwall.json","w"),indent=1)
for k in ["imp_dols","exp_dols","imp_ecm","exp_ecm","imp_fd","exp_fd"]: print(k, cv(res["full"][k]))
print("post2002 fd", cv(res["post2002"]["imp_fd"]), cv(res["post2002"]["exp_fd"]))
