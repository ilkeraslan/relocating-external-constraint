"""Replicable estimate of the compensation paid under the FX-protected deposit scheme (KKM)."""
import pandas as pd, numpy as np, json
D="../data/"; O="../results/"
fx=pd.read_csv(O+"fxpos.csv",index_col=0); fx.index=pd.PeriodIndex(fx.index,freq="M")
r=pd.read_csv(D+"tr_fx_monthly.csv"); dep=r[r.code=="TP.TRY.MT02"].set_index("date").value; dep.index=pd.PeriodIndex(dep.index,freq="M")
k=fx.loc["2021-09":"2026-10",["usdtry","kkm_fxconv","kkm_try","policy"]].copy(); k["dep3m"]=dep
k["s_fx"]=k.kkm_fxconv.fillna(0)*k.usdtry        # TRY bn, CBRT-backed (converted from FX)
k["s_try"]=k.kkm_try.fillna(0)                   # TRY bn, Treasury-backed until July 2023
out={}
for h in (3,6):
    for lab,rate in (("policy",k.policy),("deposit",k.dep3m)):
        gain=k.usdtry/k.usdtry.shift(h)-1-rate.shift(h)/100*h/12
        pay=gain.clip(lower=0)
        for part in ("s_fx","s_try"):
            k[f"c_{part}_{lab}_{h}"]=k[part].shift(h)/h*pay
        k[f"c_{lab}_{h}"]=k[f"c_s_fx_{lab}_{h}"]+k[f"c_s_try_{lab}_{h}"]
        k[f"cusd_{lab}_{h}"]=k[f"c_{lab}_{h}"]/k.usdtry
y=k.groupby(k.index.year)[[c for c in k.columns if c.startswith("c_policy") or c.startswith("c_deposit") or c.startswith("cusd_") or c.startswith("c_s_")]].sum()
gdp={2022:15012,2023:26546,2024:43411,2025:None}
print(y[["c_policy_3","c_deposit_3","c_policy_6","cusd_policy_3","cusd_deposit_3","cusd_policy_6","c_s_fx_policy_3","c_s_try_policy_3"]].round(1).to_string())
print("total USD", y[["cusd_policy_3","cusd_deposit_3","cusd_policy_6","cusd_deposit_6"]].sum().round(1).to_dict())
print("peak stock TRY bn", (k.s_fx+k.s_try).max().round(0), (k.s_fx+k.s_try).idxmax(), "USD", fx.kkm_usd.max().round(1), fx.kkm_usd.idxmax())
y.to_csv(O+"kkm_cost.csv"); k.to_csv(O+"kkm_monthly.csv")
