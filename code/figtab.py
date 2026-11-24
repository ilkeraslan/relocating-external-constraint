"""Figures and LaTeX table bodies."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="../data/"; O="../results/"
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fx=pd.read_csv(O+"fxpos.csv",index_col=0); fx.index=pd.PeriodIndex(fx.index,freq="M"); t=fx.index.to_timestamp()
TH=json.load(open(O+"thirlwall.json")); X=json.load(open(O+"extra.json")); MR=json.load(open(O+"minsky_reg.json"))
def st(p): return "^{***}" if p<.01 else "^{**}" if p<.05 else "^{*}" if p<.1 else ""
def pv(b,se):
    from scipy.stats import norm; return 2*(1-norm.cdf(abs(b/se)))
def shade(ax):
    for a,b in [("2018-08","2018-10"),("2021-12","2023-05")]: ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color="0.9",lw=0)
# Figure 1: sectoral FX positions
f,ax=plt.subplots(1,2,figsize=(7.4,3.9))
w=fx.loc["2008-01":]; tt=w.index.to_timestamp()
ax[0].plot(tt,w.nfc_net,"k-",label="Non-financial firms, net"); ax[0].plot(tt,w.public_net,"k--",label="Public sector: net reserves excl. swaps less KKM")
ax[0].plot(tt,w.net_res_exswap,color="0.55",ls=":",label="Central bank net reserves excl. swaps")
ax[0].axhline(0,color="k",lw=.5); shade(ax[0]); ax[0].set_ylabel("USD billion"); ax[0].legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12)); ax[0].set_title("(a) Firms and the public sector",fontsize=9)
w2=fx.loc["2012-12":]; t2=w2.index.to_timestamp()
ax[1].plot(t2,w2.hh_fxdep,"k-",label="FX deposits of households and other residents"); ax[1].plot(t2,w2.hh_fxdep+w2.kkm_usd,"k--",label="FX deposits plus KKM")
ax[1].plot(t2,w2.kkm_usd,color="0.55",ls=":",label="KKM balance"); shade(ax[1]); ax[1].legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12)); ax[1].set_title("(b) Households",fontsize=9); ax[1].set_ylim(0,260)
import matplotlib.dates as mdates
for _a in ax: _a.xaxis.set_major_locator(mdates.YearLocator(3)); _a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout(); plt.savefig(O+"fig_positions.pdf"); plt.close()
# Figure 2: actual growth and BoP-constrained growth
T=pd.read_csv(O+"thirlwall_periods.csv"); pi=TH["full"]["imp_dols"]["y"][0]; eps=TH["full"]["exp_dols"]["z"][0]; psi=TH["full"]["imp_dols"]["q"][0]; eta=TH["full"]["exp_dols"]["q"][0]
T["yB_ext"]=((1+eta+psi)*(-T.q)+eps*T.z)/pi if False else np.nan
a=pd.read_csv(O+"annual.csv",index_col=0)
f,ax=plt.subplots(1,2,figsize=(7.4,3.8))
roll=pd.DataFrame({"y":100*a.y.diff(),"x":100*a.x.diff(),"m":100*a.m.diff()}).rolling(5).mean()
ax[0].plot(roll.index,roll.y,"k-",label="GDP growth"); ax[0].plot(roll.index,roll.x/pi,"k--",label=r"Constrained rate $x/\pi$"); ax[0].legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12)); ax[0].set_ylabel("Percent, five-year moving average"); ax[0].set_title("(a) Actual and constrained growth",fontsize=9)
ax[1].plot(a.index,a.ca,"k-",label="Current account, % of GDP (left)"); ax[1].axhline(0,color="k",lw=.5); ax2=ax[1].twinx(); ax2.spines["top"].set_visible(False)
ax2.plot(a.index,100*(a.q-a.q.loc[2003]),color="0.5",ls="--",label="Real lira price of the dollar, 2003 = 0 (right)"); ax[1].set_title("(b) Current account and real exchange rate",fontsize=9)
h1,l1=ax[1].get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax[1].legend(h1+h2,l1+l2,frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12))
plt.tight_layout(); plt.savefig(O+"fig_thirlwall.pdf"); plt.close()
# Figure 3: Minsky
mo=pd.read_csv(O+"minsky_old.csv",index_col=0); mn=pd.read_csv(O+"minsky_new.csv",index_col=0)
f,ax=plt.subplots(1,2,figsize=(7.4,3.8))
ax[0].bar(mo.index,mo.ponzi_w,color="0.75",label="2009-2023 accounts"); ax[0].bar(mn.index[1:]+0.0,mn.ponzi_w.iloc[1:],color="0.4",label="Inflation-adjusted accounts"); ax[0].plot([2023],[mn.ponzi_w.iloc[0]],"kx")
ax[0].set_ylabel("Percent of total assets"); ax[0].set_title("(a) Ponzi sectors, share of assets",fontsize=9); ax[0].legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12))
ax[1].plot(mo.index,mo.icr_agg,"k-",label="Operating profit / financing expenses (left); bars: depreciation (right)"); ax[1].plot(mn.index,mn.icr_agg,"k--"); ax[1].axhline(1,color="k",lw=.5)
ax3=ax[1].twinx(); ax3.spines["top"].set_visible(False); ax3.bar(mo.index,mo.dep,color="0.85",width=.5); ax3.set_ylabel("Depreciation, log points x 100"); ax[1].set_zorder(ax3.get_zorder()+1); ax[1].patch.set_visible(False)
ax[1].set_title("(b) Coverage ratio and depreciation",fontsize=9); ax[1].legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12))
plt.tight_layout(); plt.savefig(O+"fig_minsky.pdf"); plt.close()
# Figure 4: KKM
k=pd.read_csv(O+"kkm_monthly.csv",index_col=0); k.index=pd.PeriodIndex(k.index,freq="M"); kt=k.index.to_timestamp()
f,ax=plt.subplots(1,2,figsize=(7.4,3.8))
ax[0].plot(kt,k.s_fx/k.usdtry,"k-",label="Converted from FX (central bank)"); ax[0].plot(kt,k.s_try/k.usdtry,"k--",label="Converted from lira (Treasury to July 2023)"); ax[0].set_ylabel("USD billion"); ax[0].legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12)); ax[0].set_title("(a) KKM balances",fontsize=9)
ax[1].bar(kt,k.c_policy_3,width=25,color="0.6",label="Estimated compensation, TRY bn per month"); ax[1].set_title("(b) Estimated compensation",fontsize=9)
ax4=ax[1].twinx(); ax4.spines["top"].set_visible(False); ax4.plot(kt,k.usdtry,"k-",lw=1,label="USD/TRY (right)"); h1,l1=ax[1].get_legend_handles_labels(); h2,l2=ax4.get_legend_handles_labels(); ax[1].legend(h1+h2,l1+l2,frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,-0.12))
plt.tight_layout(); plt.savefig(O+"fig_kkm.pdf"); plt.close()

def w(name,s): open(O+name,"w").write(s+"\n")
c=lambda v,d=2: f"{v:.{d}f}"
cs=lambda b,se,d=2: f"${b:.{d}f}{st(pv(b,se))}$"
# Table: elasticities
F=TH["full"]; P2=TH["post2002"]; Q=X
rows=[]
rows.append(f"Income elasticity & {cs(*F['imp_dols']['y'])} & {cs(*F['imp_fd']['y'])} & {cs(*Q['q_imp_all']['y'])} & {cs(*F['exp_dols']['z'])} & {cs(*F['exp_fd']['z'])} & \\\\")
rows.append(f" & ({c(F['imp_dols']['y'][1])}) & ({c(F['imp_fd']['y'][1])}) & ({c(Q['q_imp_all']['y'][1])}) & ({c(F['exp_dols']['z'][1])}) & ({c(F['exp_fd']['z'][1])}) & \\\\")
rows.append(f"Real depreciation & {cs(*F['imp_dols']['q'])} & {cs(*F['imp_fd']['q'])} & {cs(-Q['q_imp_all']['reer'][0],Q['q_imp_all']['reer'][1])} & {cs(*F['exp_dols']['q'])} & {cs(*F['exp_fd']['q'])} & {cs(-Q['q_imp_all']['x_reer'][0],Q['q_imp_all']['x_reer'][1])} \\\\")
rows.append(f" & ({c(F['imp_dols']['q'][1])}) & ({c(F['imp_fd']['q'][1])}) & ({c(Q['q_imp_all']['reer'][1])}) & ({c(F['exp_dols']['q'][1])}) & ({c(F['exp_fd']['q'][1])}) & ({c(Q['q_imp_all']['x_reer'][1])}) \\\\")
rows.append(f"Observations & {F['imp_n']:.0f} & 37 & {Q['q_imp_all']['n']} & {F['exp_n']:.0f} & 37 & {Q['q_imp_all']['n']} \\\\")
w("tab_elast.tex","\n".join(rows))
# Table: periods
kq=1-TH["full"]["exp_fd"]["q"][0]+TH["full"]["imp_fd"]["q"][0]; print("1+eta+psi",kq)
T=pd.read_csv(O+"thirlwall_periods.csv"); T["ext"]=(kq*(-T.q)+eps*T.z)/pi
w("tab_periods.tex","\n".join(f"{r.period} & {c(r.y,1)} & {c(r.x,1)} & {c(r.m,1)} & {c(r.z,1)} & {c(r.q,1)} & {c(r.yB_weak,1)} & {c(r.ext,1)} & {c(r.ca,1)} \\\\" for r in T.itertuples()))
# Table: positions
L=pd.read_csv(O+"levels.csv",index_col=0); lab={"2012-12":"December 2012","2018-07":"July 2018","2021-11":"November 2021","2023-05":"May 2023","2026-09":"September 2026"}
rows=[f"{lab[i]} & {c(r.nfc_net,0)} & {c(r.hh_fxdep,0)} & {c(r.kkm_usd,0)} & {c(r.gross_res,0)} & {c(r.net_res_exswap,0)} & {c(r.public_net,0)} & {c(r.nfc_net+r.hh_fxdep+r.kkm_usd+r.public_net,0)} & {c(r.usdtry,1)} \\\\" for i,r in L.iterrows()]
E=pd.read_csv(O+"episodes.csv"); en=["2013m1--2018m7","2018m8--2021m11","2021m12--2023m5","2023m6--2026m9"]
rows.append("\\midrule\n\\multicolumn{9}{l}{\\textit{Change over the episode}} \\\\")
for n,r in zip(en,E.itertuples()): rows.append(f"{n.replace('m','M')} & {c(r.d_nfc_net,0)} & {c(r.d_hh_fxdep,0)} & {c(r.d_kkm_usd,0)} & {c(r.d_gross_res,0)} & {c(r.d_net_res_exswap,0)} & {c(r.d_public_net,0)} & {c(r.d_nfc_net+r.d_hh_fxdep+r.d_kkm_usd+r.d_public_net,0)} & {c(r.dep,0)} \\\\")
w("tab_positions.tex","\n".join(rows))
# Table: Minsky shares
S=pd.read_csv(O+"minsky_shares.csv",header=[0,1],index_col=0); S=S.loc[2010:]
rows=[f"{y} & {c(S.loc[y,('n','Hedge')],0)} & {c(S.loc[y,('n','Speculative')],0)} & {c(S.loc[y,('n','Ponzi')],0)} & {c(S.loc[y,('assets','Hedge')],0)} & {c(S.loc[y,('assets','Speculative')],0)} & {c(S.loc[y,('assets','Ponzi')],0)} & {c(mo.icr_agg.loc[y])} & {c(mo.dep.loc[y],0)} \\\\" for y in S.index]
w("tab_minsky.tex","\n".join(rows))
# Table: regressions
R=X["rob"]; f3=lambda k: (f"${R[k][0]:.1f}{st(R[k][2])}$", f"({R[k][1]:.1f})", R[k][3])
cols=[("Financing expenses, 2010--2023","base"),("Financing expenses, 2010--2020","ex2021_23"),("2010--2020, non-manufacturing","nonmanuf_pre2021"),("2010--2020, manufacturing","manuf_pre2021"),("Fixed 2010--2012 FX share, 2013--2023","fixed_share"),("Asset-weighted, 2010--2023","weighted"),("Excluding 2018","ex2018"),("Controlling for leverage $\\times$ depreciation","with_leverage"),("Profit before tax, 2010--2020","pbt_pre2021"),("Profit before tax, fixed share, 2013--2023","pbt_fixed")]
w("tab_reg.tex","\n".join(f"{n} & {f3(k)[0]} & {f3(k)[1]} & {f3(k)[2]} \\\\" for n,k in cols))
# Table: KKM costs
K=pd.read_csv(O+"kkm_cost.csv",index_col=0); e=fx.usdtry.groupby(fx.index.year).mean()
q=pd.read_csv(D+"tr_quarterly.csv"); gdp=q[q.code=="TP.GSYIH20.HY.B1GQ"] if (q.code=="TP.GSYIH20.HY.B1GQ").any() else q[(q.group=="bie_gsyhhrccar")&(q.name.str.contains("Gross Domestic Product"))]
gy=gdp.assign(y=gdp.date.str[:4].astype(int)).groupby("y").value.sum()/1e6   # thousand TRY -> TRY bn
off={2022:(92.5,72.8),2023:(59.5,833.4),2024:(0.0,240.2),2025:(0.0,166.5)}
rows=[]; tot=[0,0,0,0,0]
for y,(tr,cb) in off.items():
    est=K.c_policy_3.get(y,0); lo=K.c_deposit_3.get(y,0); usd=(tr+cb)/e.loc[y]
    rows.append(f"{y} & {c(tr,1)} & {c(cb,1)} & {c(tr+cb,1)} & {c(usd,1)} & {c(100*(tr+cb)/gy.loc[y],1)} & {c(lo,0)}--{c(est,0)} \\\\"); tot=[tot[0]+tr,tot[1]+cb,tot[2]+tr+cb,tot[3]+usd,tot[4]+0]
rows.append(f"\\midrule\nTotal & {c(tot[0],1)} & {c(tot[1],1)} & {c(tot[2],1)} & {c(tot[3],1)} & & {c(K.c_deposit_3.sum(),0)}--{c(K.c_policy_3.sum(),0)} \\\\")
w("tab_kkm.tex","\n".join(rows)); print("\n".join(rows)); print(gy.loc[2021:2025].round(0).to_dict(), e.loc[2021:2025].round(2).to_dict())
for n in ["tab_elast","tab_periods","tab_positions","tab_minsky","tab_reg"]: print("---",n); print(open(O+n+".tex").read())
