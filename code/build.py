"""Builds analysis datasets from the raw EVDS and World Bank downloads."""
import pandas as pd, numpy as np
D="../data/"; O="../results/"
m=pd.concat([pd.read_csv(D+f) for f in ["tr_monthly.csv","tr_money.csv","tr_fx_monthly.csv"]]).drop_duplicates(["code","date"])
def s(code):
    x=m[m.code==code].set_index("date").value.astype(float); x.index=pd.PeriodIndex(x.index,freq="M"); return x.sort_index()
e=s("TP.DK.USD.A.YTL")                    # USD/TRY, monthly average
fx=pd.DataFrame({"usdtry":e})
# end-month rate implied by CBRT analytical balance sheet is not published; use monthly average
fx["nfc_net"]=s("TP.FDVY37")/1e3            # NFC net FX position, USD bn
fx["nfc_assets"]=s("TP.FDVY01")/1e3
fx["nfc_liab"]=s("TP.FDVY12")/1e3
fx["nfc_dep_dom"]=s("TP.FDVY03")/1e3
fx["nfc_st_net"]=s("TP.FDVY40")/1e3
fx["nfc_fxloans_dom"]=s("TP.FDVY14")/1e3
fx["nfc_ext_loans"]=s("TP.FDVY27")/1e3
fx["kkm_fxconv"]=s("TP.KKM.K1")             # FX-conversion KKM, USD bn (CBRT-backed)
fx["kkm_try"]=s("TP.KKM.K4")                # TRY KKM, TRY bn (Treasury-backed until July 2023)
fx["kkm_usd"]=fx.kkm_fxconv.fillna(0)+(fx.kkm_try/fx.usdtry).fillna(0)
fx["gross_res"]=s("TP.REZVARPD.K1")/1e3
fx["cb_nfa"]=(s("TP.AB.A02")-s("TP.AB.A10"))/1e6/fx.usdtry   # thousand TRY -> USD bn
fx["cb_fwd"]=s("TP.DOVVARNC.K15")/1e3       # short positions in forwards/swaps (negative)
fx["net_res_exswap"]=fx.cb_nfa+fx.cb_fwd
# residents' FX deposits, USD bn (archive to 2025-01, new table from 2024-06)
old=(s("TP.YPMEVD.M05")+s("TP.YPMEVD.M10"))/1e6/fx.usdtry
new=s("TP.HPBITABLO4.2")/1e3
fx["res_fxdep"]=old.combine_first(new)
fx.loc[new.index,"res_fxdep"]=new
fx["hh_fxdep"]=fx.res_fxdep-fx.nfc_dep_dom
fx["hh_fxdep_new"]=s("TP.HPBITABLO4.3")/1e3
fx["m2"]=s("TP.PBD.H09")/1e6; fx["fxdep_try"]=(s("TP.PBD.H06")+s("TP.PBD.H11"))/1e6
fx["dollar"]=fx.fxdep_try/fx.m2
fx["dollar_kkm"]=(fx.fxdep_try+fx.kkm_usd*fx.usdtry)/fx.m2
fx["cpi"]=s("TP.GENENDEKS.T1"); fx["policy"]=s("TP.BISPOLFAIZ.TUR"); fx["reer"]=s("TP.RK.T1.Y")
fx["ca"]=s("TP.ODANA6.Q01")/1e3; fx["neo"]=s("TP.ODANA6.Q31")/1e3; fx["res_bop"]=s("TP.ODANA6.Q33")/1e3
fx["public_net"]=fx.net_res_exswap-fx.kkm_usd
fx=fx.loc["2005-01":"2026-10"]
fx.to_csv(O+"fxpos.csv")
print(fx.loc[["2012-12","2017-12","2018-08","2020-12","2021-11","2022-12","2023-05","2023-08","2024-03","2024-12","2025-06","2026-06","2026-09"],["usdtry","nfc_net","hh_fxdep","kkm_usd","gross_res","net_res_exswap","public_net","dollar","dollar_kkm","policy"]].round(2).to_string())

# ---------- sector panel from CBRT company accounts ----------
c=pd.concat([pd.read_csv(D+"tr_company_accounts.csv"),pd.read_csv(D+"tr_company_ratios.csv")])
c["vint"]=np.where(c.group.str.startswith("bie_sekbila"),"new","old")
c["item"]=c.group.str.replace("bie_sekbila","",regex=False).str.replace("bie_sekbil","",regex=False)
c["sector"]=c.code.str.split(".").str[-1]
names=c.drop_duplicates("sector").set_index("sector").name
p=c.pivot_table(index=["vint","sector","date"],columns="item",values="value").reset_index().rename(columns={"date":"year"})
p=p.rename(columns={"5020":"op","5039":"fin","5009":"sales","5050":"pbt","5015":"gross","1122":"assets","3001":"stl","3051":"ltl","3086":"equity","tpn":"cr_try","ypn":"cr_fx","tga":"npl","b13a":"stbank_stl","b14a":"bank_loans","b01a":"lev","d02f":"fin_sales","d02a":"op_sales"})
p["name"]=p.sector.map(names)
p.to_csv(O+"sector_raw.csv",index=False)
print(p.groupby(["vint","year"]).size().unstack(0).T.to_string())
g=p[p.sector=="GENEL"].set_index(["vint","year"])
print((g[["op","fin","sales","assets","stl","cr_try","cr_fx"]]/1e6).round(0).to_string()); print(g[["stbank_stl","bank_loans","lev","fin_sales","op_sales"]].round(2).to_string())
