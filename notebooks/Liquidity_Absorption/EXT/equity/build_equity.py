import sys; sys.path.insert(0,'/home/claude/deribit-market-data')
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.data_engineering.modeling import liquidity_absorption_extended as ex
def net(df, makercol=None):
    t=ex.deribit_net(df,False)
    if makercol is not None:
        m=df[makercol].astype(str).str.lower().isin(['true','1'])|df[makercol].astype(str).str.contains('MAKER')
        t=t.where(~m, ex.deribit_net(df,True))
    return t
fam={}
a=pd.read_csv('EXT/03/03_execution_events.csv'); a['net']=net(a); fam['A03 (VWAP/prior-day target, A02 stop)']={f'A03_{m}':g for m,g in a.groupby('mode')}
b=pd.read_csv('EXT/03b/03b_1to1_events.csv'); b['net']=net(b)
c=pd.read_csv('EXT/03c/03c_events.csv'); c['net']=net(c,'is_inferred_maker')
f2={'A03b_1to1_taker':b}; f2.update({f'A03c_{m}':g for m,g in c.groupby('mode')}); fam['A03b/A03c (1:1, reverse, maker 3rd touch)']=f2
d=pd.read_csv('EXT/03d/03d_risk_execution.csv',low_memory=False); d['net']=net(d,'entry_kind')
fam['A03d (VD18 engine 1σ60 stop)']={f'{m}_{h}m':g for (m,h),g in d.groupby(['model_mode','hold_minutes'])}
v=pd.read_csv('V2/V2_events.csv'); v['net']=net(v)
fam['V2 (CVD divergence, 18:00 ET profile LVN→HVN)']={f"{k}{' +CVD' if o else ''}":g for (k,o),g in v.groupby(['v2_variant','cvd_ok'])} 
# V2: cvd_ok=False rows are the unfiltered complement; build 'all' = both
fam['V2 (CVD divergence, 18:00 ET profile LVN→HVN)']={**{f'{k} all':g for k,g in v.groupby('v2_variant')},**{f'{k} CVD':g for k,g in v[v.cvd_ok].groupby('v2_variant')}}
rows=[]
fig,ax=plt.subplots(len(fam),2,figsize=(20,6*len(fam)))
for i,(fn,vs) in enumerate(fam.items()):
    for name,g in vs.items():
        g=g.dropna(subset=['gross_return_bp']).copy(); g['t']=pd.to_datetime(g.signal_time_utc,utc=True); g=g.sort_values('t')
        ls='-' if not name.endswith(' all') else '--'
        for j,col in enumerate(['gross_return_bp','net']):
            ax[i,j].plot(g.t,g[col].cumsum(),ls,lw=1.2,label=f'{name} (n={len(g)})')
        rows.append(dict(family=fn,variant=name,n=len(g),gross_total_bp=g.gross_return_bp.sum(),net_total_bp=g.net.sum(),gross_mean=g.gross_return_bp.mean(),net_mean=g.net.mean()))
    for j,lab in enumerate(['GROSS','NET (Deribit 0/5 bp)']):
        A=ax[i,j]; A.axvspan(pd.Timestamp('2026-07-01',tz='UTC'),pd.Timestamp('2026-10-01',tz='UTC'),color='orange',alpha=.15)
        A.axvline(pd.Timestamp('2025-04-01',tz='UTC'),color='grey',ls=':'); A.axhline(0,color='k',lw=.6)
        A.set_title(f'{fn} — {lab}, cumulative bp/trade, BTC+ETH'); A.legend(fontsize=7,ncol=2); A.grid(alpha=.3)
plt.tight_layout(); plt.savefig('EXT/equity/all_variants_equity.png',dpi=110)
S=pd.DataFrame(rows).round(1); S.to_csv('EXT/equity/all_variants_totals.csv',index=False); print(S.to_string())
