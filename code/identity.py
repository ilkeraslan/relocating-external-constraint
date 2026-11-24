"""Checks of the relocation identity: external counterpart from the international investment position,
offset regressions across sectors, and quarterly cointegration tests for the trade equations."""
import pandas as pd, numpy as np, statsmodels.api as sm, json
from statsmodels.tsa.stattools import adfuller
D="../data/"; O="../results/"
fx=pd.read_csv(O+"fxpos.csv",index_col=0); fx.index=pd.PeriodIndex(fx.index,freq="M")
i=pd.read_csv(D+"tr_iip.csv"); q=pd.read_csv(D+"tr_quarterly.csv")
def qs(df,code):
    x=df[df.code==code].set_index("date").value.astype(float); x.index=pd.PeriodIndex([s.replace("-Q","Q") for s in x.index],freq="Q"); return x.sort_index()/1e3
A=qs(i,"TP.UYPUCAY.K2")-qs(i,"TP.UYPUCAY.K20")                     # external assets excluding lira deposits abroad
debt=qs(q,"TP.BRUTDBDOVKOMP.G1"); debt_try=qs(q,"TP.BRUTDBDOVKOMP.G6")
gg=qs(q,"TP.BRUTDBORCLU.G3")-qs(i,"TP.UYPUCAY.K53")                # general government external debt excluding bonds issued at home
Q=pd.DataFrame({"assets":A,"fxdebt":debt-debt_try}).dropna()
Q["nstar"]=Q.assets-Q.fxdebt                                        # residents' net FX claims on non-residents
Q["lstar"]=qs(i,"TP.UYPUCAY.K42")+qs(i,"TP.UYPUCAY.K86")+debt_try  # non-residents' lira claims: equity and lira debt
Q["ggfx"]=-gg
m=fx[["nfc_net","hh_fxdep","kkm_usd","net_res_exswap","public_net","ca","usdtry"]].copy()
e=m[m.index.month.isin([3,6,9,12])].copy(); e.index=e.index.asfreq("Q")
Q=Q.join(e[["nfc_net","hh_fxdep","kkm_usd","net_res_exswap","public_net"]]); Q["ca"]=m.ca.groupby(m.index.asfreq("Q")).sum()
Q["hh_all"]=Q.hh_fxdep+Q.kkm_usd
Q["sum3"]=Q.nfc_net+Q.hh_all+Q.public_net
Q["resid"]=Q.nstar-Q.sum3-Q.ggfx                                    # banks and unmeasured items
Q=Q.loc["2012Q4":"2026Q3"]; Q.to_csv(O+"identity.csv")
print(Q.loc[["2012Q4","2018Q2","2021Q4","2023Q2","2026Q3"],["nstar","lstar","sum3","ggfx","resid","nfc_net","hh_all","public_net"]].round(0).to_string())
print(Q[["nstar","sum3","resid","public_net","nfc_net","lstar"]].describe().loc[["mean","std","min","max"]].round(0).to_string())
res={"corr_nstar_sum":float(Q.nstar.corr(Q.sum3)),"corr_d4":float(Q.nstar.diff(4).corr(Q.sum3.diff(4)))}
# offset regressions, quarterly changes: public position on private positions and the external counterpart
d=Q[["public_net","nfc_net","hh_all","nstar","ca","resid","ggfx"]].diff().dropna()
def reg(y,xs,lab):
    f=sm.OLS(d[y],sm.add_constant(d[xs])).fit(cov_type="HAC",cov_kwds={"maxlags":4})
    res[lab]={k:[float(f.params[k]),float(f.bse[k])] for k in xs}; res[lab]["r2"]=float(f.rsquared); res[lab]["n"]=int(f.nobs)
    w=f.wald_test(" , ".join(f"{k} = -1" for k in xs if k in ("nfc_net","hh_all")),scalar=True); res[lab]["p_minus1"]=float(w.pvalue)
    print(lab,{k:(round(f.params[k],2),round(f.bse[k],2)) for k in xs},"R2",round(f.rsquared,2),"p(-1)",round(float(w.pvalue),3),int(f.nobs))
reg("public_net",["nfc_net","hh_all"],"off_a")
reg("public_net",["nfc_net","hh_all","nstar"],"off_b")
reg("public_net",["nfc_net","hh_all","ca"],"off_c")
d18=d.loc["2018Q1":]
f=sm.OLS(d18.public_net,sm.add_constant(d18[["nfc_net","hh_all","nstar"]])).fit(cov_type="HAC",cov_kwds={"maxlags":4}); res["off_b_2018"]={k:[float(f.params[k]),float(f.bse[k])] for k in ["nfc_net","hh_all","nstar"]}; res["off_b_2018"]["r2"]=float(f.rsquared); res["off_b_2018"]["n"]=int(f.nobs); res["off_b_2018"]["p_minus1"]=float(f.wald_test("nfc_net = -1 , hh_all = -1",scalar=True).pvalue); print("2018on",res["off_b_2018"])
# quarterly trade equations in levels with seasonal dummies: residual-based cointegration test
g=q[q.group=="bie_gsyhhrczinc"].pivot(index="date",columns="code",values="value"); g.index=pd.PeriodIndex([s.replace("-Q","Q") for s in g.index],freq="Q")
g=np.log(g.rename(columns={"TP.GSYIH20.CY.B1GQ":"y","TP.GSYIH20.CY.P6":"x","TP.GSYIH20.CY.P7":"m"})[["y","x","m"]])
mm=pd.concat([pd.read_csv(D+"tr_monthly.csv")]); r=mm[mm.code=="TP.RK.T1.Y"].set_index("date").value; r.index=pd.PeriodIndex(r.index,freq="M")
g["q"]=-np.log(r.groupby(r.index.asfreq("Q")).mean()); g=g.dropna().loc[:"2026Q2"]
for k in (1,2,3): g[f"s{k}"]=(g.index.quarter==k).astype(float)
def dolsq(dep,reg,k=2):
    X=g[reg+["s1","s2","s3"]].copy()
    for v in reg:
        dx=g[v].diff()
        for j in range(-k,k+1): X[f"d{v}_{j}"]=dx.shift(-j)
    dd=pd.concat([g[dep],sm.add_constant(X)],axis=1).dropna(); f=sm.OLS(dd[dep],dd.drop(columns=dep)).fit(cov_type="HAC",cov_kwds={"maxlags":4})
    st=sm.OLS(g[dep],sm.add_constant(g[reg+["s1","s2","s3"]])).fit(); adf=adfuller(st.resid,maxlag=4,autolag=None,regression="n",result_object=False)[0]
    return {v:[float(f.params[v]),float(f.bse[v])] for v in reg},float(adf),int(f.nobs)
res["q_dols_imp"]=dolsq("m",["y","q"]); res["q_dols_exp_q"]=dolsq("x",["q"])
print("quarterly DOLS imports",res["q_dols_imp"]," (Engle-Granger 5% critical value with two regressors about -3.8, 10% about -3.5)"); print(g.index.min(),g.index.max(),len(g))
json.dump(res,open(O+"identity.json","w"),indent=1)
# ---- table bodies
def st(b,se):
    t=abs(b/se); return "^{***}" if t>2.58 else "^{**}" if t>1.96 else "^{*}" if t>1.645 else ""
lab={"2012Q4":"2012Q4","2018Q2":"2018Q2","2021Q4":"2021Q4","2023Q2":"2023Q2","2026Q3":"2026Q3"}
rows=[f"{k} & {Q.nstar.loc[k]:.0f} & {Q.lstar.loc[k]:.0f} & {Q.sum3.loc[k]:.0f} & {Q.ggfx.loc[k]:.0f} & {Q.resid.loc[k]:.0f} \\\\" for k in lab]
rows.append("\\midrule\nStandard deviation, 2012Q4--2026Q3 & "+" & ".join(f"{Q[c].std():.0f}" for c in ["nstar","lstar","sum3","ggfx","resid"])+" \\\\")
open(O+"tab_external.tex","w").write("\n".join(rows)+"\n")
cols=[("off_a",["nfc_net","hh_all"]),("off_c",["nfc_net","hh_all","ca"]),("off_b",["nfc_net","hh_all","nstar"]),("off_b_2018",["nfc_net","hh_all","nstar"])]
names={"nfc_net":"$\\Delta$ firms' net position","hh_all":"$\\Delta$ household FX deposits and KKM","ca":"Current account","nstar":"$\\Delta$ external counterpart $N^{\\ast}$"}
out=[]
for v in ["nfc_net","hh_all","ca","nstar"]:
    b=" & ".join((f"${res[c][v][0]:.2f}{st(*res[c][v])}$" if v in xs else "") for c,xs in cols); s=" & ".join((f"({res[c][v][1]:.2f})" if v in xs else "") for c,xs in cols)
    out+= [f"{names[v]} & {b} \\\\", f" & {s} \\\\"]
out.append("\\midrule\n$R^2$ & "+" & ".join(f"{res[c]['r2']:.2f}" for c,_ in cols)+" \\\\")
out.append("Test: both private coefficients $=-1$, p-value & "+" & ".join((f"{res[c]['p_minus1']:.2f}" if 'p_minus1' in res[c] else "") for c,_ in cols)+" \\\\")
out.append("Quarters & "+" & ".join(str(res[c]['n']) for c,_ in cols)+" \\\\")
open(O+"tab_offset.tex","w").write("\n".join(out)+"\n"); print("\n".join(out)); print(open(O+"tab_external.tex").read())
T=pd.read_csv(O+"thirlwall_periods.csv"); print((T.set_index("period").x/res["q_dols_imp"][0]["y"][0]).round(1).to_dict(), T.set_index("period").y.round(1).to_dict())
