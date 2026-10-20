"""LA-A05a: Signal-close anchored, -240..+240m historical event-study context.

Source prices/flow = existing Binance UM 1-minute completed OHLCV candles.
Original A02 signal direction/level, A04 frozen A03d outcome, VD18 forecast stay fixed.
Descriptive, retrospectively outcome-labelled paths; not an ex-ante predictive rule.
Market trajectories continue AFTER a trade exits; annotate risk-set separately.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-A05a-0.1.0"
MIN_MS = 60_000
OUTCOMES = ("TARGET", "STOP", "TIMEOUT")
EXTRA_OUTCOMES = ("NO_STRUCTURAL_TARGET", "CENSORED_TAPE_END")


@dataclass(frozen=True)
class Config:
    before_min: int = 240
    after_min: int = 240
    trailing_minutes: int = 60
    flow_window_minutes: int = 15
    rv_windows: tuple[int, ...] = (15, 60)
    cohort_entry_kind: str = "TAKER_FIRST_POST_SIGNAL"
    cohort_mode: str = "VD18_VOL_FIXED"
    cohort_hold_min: int = 240
    target_sigma_ratio_split: float = 1.0
    source_A04_version: str = "LA-A04-0.1.1"
    # Do not turn these into a selection/optimization grid. Fixed descriptive protocol.


def check_config(cfg: Config) -> None:
    if (cfg.before_min != 240 or cfg.after_min != 240 or cfg.trailing_minutes != 60
            or cfg.flow_window_minutes != 15 or cfg.rv_windows != (15, 60)
            or cfg.target_sigma_ratio_split != 1.0 or cfg.cohort_hold_min != 240):
        raise ValueError("Unexpected A05a preregistered event-study configuration")


def ts_millis(s: pd.Series) -> pd.Series:
    # pandas datetime64[ms/us/ns] never implicitly assumed to mean nanoseconds.
    d = pd.to_datetime(s, utc=True, errors="raise")
    if d.isna().any():
        raise ValueError("Null UTC timestamp")
    return d.dt.as_unit("ms").astype("int64")


def load_event_cohort(repo: Path, cfg: Config = Config()) -> pd.DataFrame:
    check_config(cfg)
    base = Path(repo) / "notebooks" / "Liquidity_Absorption"
    state = json.loads((base / "04" / "04_decision.json").read_text(encoding="utf8"))
    if state.get("version") != cfg.source_A04_version or state.get("original_candidates") != 452:
        raise ValueError("A04 cohort decision contract changed")
    z = pd.read_csv(base / "04" / "04_primary_event_outcomes.csv", low_memory=False)
    needed = {"a02_candidate_row", "symbol", "side", "reason", "signal_time_utc",
              "entry_time_ms", "exit_time_ms", "level_price", "zone_halfwidth",
              "target", "initial_stop", "entry_print", "vol_sigma_bp_60",
              "target_to_sigma60", "entry_kind", "model_mode", "hold_minutes"}
    if missing := needed - set(z):
        raise ValueError(f"Missing A04 primary fields: {sorted(missing)}")
    if (len(z) != 452 or z.a02_candidate_row.duplicated().any()
            or set(z.a02_candidate_row.astype(int)) != set(range(452))):
        raise ValueError("Expected original 452 unique candidate IDs")
    if not (z.entry_kind.eq(cfg.cohort_entry_kind) & z.model_mode.eq(cfg.cohort_mode)
            & pd.to_numeric(z.hold_minutes).eq(cfg.cohort_hold_min)).all():
        raise ValueError("A04 primary sample is no longer the frozen A03d policy")
    if not z.symbol.isin(("BTCUSDT", "ETHUSDT")).all() or not z.side.isin(("long", "short")).all():
        raise ValueError("Unexpected symbol/side in event cohort")
    if not z.reason.isin(OUTCOMES + EXTRA_OUTCOMES).all():
        raise ValueError("Unregistered terminal outcome")
    z["signal_ms"] = ts_millis(z.signal_time_utc)
    # A02b contains the causal context bar close and the two originally detected touches.
    b = pd.read_csv(base / "02b" / "02b_touch_features.csv", usecols=[
        "a02_candidate_row", "symbol", "side", "signal_time_utc", "level_price",
        "bar_close", "first_touch_close", "second_touch_close", "level_source_time",
        "level_type"], low_memory=False)
    if len(b) != 452 or b.a02_candidate_row.duplicated().any():
        raise ValueError("A02b source population changed")
    joined = z.merge(b, on="a02_candidate_row", validate="one_to_one", suffixes=("", "_a02b"))
    if len(joined) != 452:
        raise ValueError("A02b/A04 row drift")
    for col in ("symbol", "side"):
        if not joined[col].eq(joined[col + "_a02b"]).all():
            raise ValueError(f"A02b/A04 identity drift: {col}")
    if not np.array_equal(ts_millis(joined.signal_time_utc).to_numpy(),
                          ts_millis(joined.signal_time_utc_a02b).to_numpy()):
        raise ValueError("A02b/A04 signal-close drift")
    if not np.allclose(pd.to_numeric(joined.level_price),pd.to_numeric(joined.level_price_a02b),rtol=0,atol=1e-7):
        raise ValueError("A02b/A04 level-price drift")
    if not np.array_equal(ts_millis(joined.second_touch_close).to_numpy(), joined.signal_ms.to_numpy()):
        raise ValueError("Second-touch close must be the frozen signal timestamp")
    joined["first_touch_ms"] = ts_millis(joined.first_touch_close)
    joined["level_source_ms"] = ts_millis(joined.level_source_time)
    if (joined.first_touch_ms > joined.signal_ms).any() or (joined.level_source_ms > joined.signal_ms).any():
        raise ValueError("Touch/level source cannot occur after signal")
    joined["eventual_outcome"] = joined.reason
    joined["feasibility_band"] = np.where(
        (joined.reason.isin(OUTCOMES)) & pd.to_numeric(joined.target_to_sigma60).le(1.0),
        "TARGET_LE_1SIGMA60", "TARGET_GT_1SIGMA60")
    joined.loc[~joined.reason.isin(OUTCOMES), "feasibility_band"] = "NO_CLOSED_OUTCOME"
    return joined.sort_values("a02_candidate_row").reset_index(drop=True)


def load_context_minutes(path: Path, cfg: Config = Config()) -> pd.DataFrame:
    """Read 1m candles only once. Rolling stats invalidate windows crossing missing minutes."""
    from src.data_engineering.modeling.liquidity_absorption_stage_a import parse_kline_open_time
    columns = ["open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    raw = pd.read_parquet(path, columns=columns)
    ts = parse_kline_open_time(raw.open_time)
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError(f"Invalid one-minute timestamps in {path}")
    d = raw.drop(columns=["open_time"]).copy()
    d["close_ms"] = ts.dt.as_unit("ms").astype("int64") + MIN_MS
    for col in columns[1:]:
        d[col] = pd.to_numeric(d[col],errors="raise")
    if (not np.isfinite(d[columns[1:]].to_numpy(float)).all()
            or (d[["open","high","low","close"]] <= 0).any().any()
            or (d.low > d.high).any() or (d.high < d[["open", "close"]].max(axis=1)).any()
            or (d.low > d[["open", "close"]].min(axis=1)).any()
            or (d.quote_volume < 0).any() or (d.taker_buy_quote_volume < 0).any()
            or (d.taker_buy_quote_volume > d.quote_volume * (1+1e-7) + 1e-8).any()):
        raise ValueError(f"Invalid one-minute OHLC/quote fields in {path}")
    # Check duplicates explicitly; pandas v4 deprecates set_index verify_integrity.
    if d.close_ms.duplicated().any():
        raise ValueError(f"Duplicate one-minute candle closes in {path}")
    d = d.set_index("close_ms", drop=False).sort_index()
    segment = d.close_ms.diff().ne(MIN_MS).cumsum()
    d["log_return_1m"] = np.log(d.close / d.close.shift()).where(d.close_ms.diff().eq(MIN_MS))
    for mins in cfg.rv_windows:
        d[f"rv_{mins}m_bp"] = d.log_return_1m.pow(2).groupby(segment).transform(
            lambda s: np.sqrt(s.rolling(mins,min_periods=mins).sum())*10_000)
    d["quote_prior60_mean"] = d.quote_volume.groupby(segment).transform(
        lambda s: s.shift(1).rolling(cfg.trailing_minutes,min_periods=cfg.trailing_minutes).mean())
    d["quote_vs_prior60"] = (d.quote_volume / d.quote_prior60_mean.replace(0,np.nan))
    d["net_buy_quote"] = 2*d.taker_buy_quote_volume-d.quote_volume
    flow = d.net_buy_quote.groupby(segment).transform(
        lambda s: s.rolling(cfg.flow_window_minutes,min_periods=cfg.flow_window_minutes).sum())
    quote = d.quote_volume.groupby(segment).transform(
        lambda s: s.rolling(cfg.flow_window_minutes,min_periods=cfg.flow_window_minutes).sum())
    d["flow_15m_buy_signed"] = flow / quote.replace(0,np.nan)
    d["volume_15m_vs_prior60"] = d.quote_volume.groupby(segment).transform(
        lambda s: s.rolling(cfg.flow_window_minutes,min_periods=cfg.flow_window_minutes).mean()) / d.quote_prior60_mean.replace(0,np.nan)
    return d


def event_minute_panel(e: dict, bars: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Return ALL market minutes, even beyond trade exit; no forward-filled prices.

    Outcome, level, target and exit are retrospective metadata. Candidate's liquidity
    reference may not have existed at every pre-signal minute. Never use event label
    or future level as a predictive feature at negative times.
    """
    s = int(e["signal_ms"])
    rel = np.arange(-cfg.before_min,cfg.after_min+1,dtype=int)
    ms = s + rel*MIN_MS
    b = bars.reindex(pd.Index(ms, name="close_ms"))
    d = 1 if e["side"]=="long" else -1
    price = float(e["bar_close"])
    level = float(e["level_price"])
    width = float(e["zone_halfwidth"])
    if not np.isfinite(price) or price <= 0 or not np.isfinite(level) or level <= 0 or not np.isfinite(width) or width <= 0:
        raise ValueError("Invalid signal/level geometry")
    at_zero = b.iloc[cfg.before_min]
    if pd.isna(at_zero.close) or not np.isclose(float(at_zero.close),price,rtol=0,atol=max(1e-6,price*1e-9)):
        raise ValueError(f"Frozen A02b signal close does not match 1m perp candle for event {e['a02_candidate_row']}")
    result = pd.DataFrame({
        "a02_candidate_row":int(e["a02_candidate_row"]), "symbol":e["symbol"],
        "side":e["side"],"eventual_outcome":e["eventual_outcome"],
        "level_type":e.get("level_type", ""),"feasibility_band":e["feasibility_band"],
        "relative_min":rel,"utc_close_ms":ms,
        "has_minute_bar":b.close.notna().to_numpy(),
        "is_presignal":rel<0,"is_signal_bar":rel==0,"is_postsignal":rel>0,
        "first_touch_marker":ms == int(e["first_touch_ms"]),
        "second_touch_marker":ms == s,
        "level_source_marker":ms == int(e["level_source_ms"]),
        "level_source_timestamp_le_bar":ms >= int(e["level_source_ms"]),
        "signal_close_price":price,"liquidity_level_price":level,
        "forecast_sigma60_bp":float(e["vol_sigma_bp_60"]) if pd.notna(e["vol_sigma_bp_60"]) else np.nan,
        "target_distance_over_sigma60":float(e["target_to_sigma60"]) if pd.notna(e["target_to_sigma60"]) else np.nan,
        "signed_close_from_signal_bp":d*(b.close.to_numpy(float)-price)/price*10000,
        "signed_close_from_level_bp":d*(b.close.to_numpy(float)-level)/price*10000,
        "signed_candle_body_bp":d*(b.close.to_numpy(float)-b.open.to_numpy(float))/price*10000,
        "signed_1m_close_return_bp":d*b.log_return_1m.to_numpy(float)*10000,
        "signed_favorable_extreme_bp":d*((b.high if d==1 else b.low).to_numpy(float)-price)/price*10000,
        "signed_adverse_extreme_bp":d*((b.low if d==1 else b.high).to_numpy(float)-price)/price*10000,
        "rv_15m_bp":b.rv_15m_bp.to_numpy(float),"rv_60m_bp":b.rv_60m_bp.to_numpy(float),
        "quote_volume":b.quote_volume.to_numpy(float),
        "quote_vs_prior60":b.quote_vs_prior60.to_numpy(float),
        "quote_15m_vs_prior60":b.volume_15m_vs_prior60.to_numpy(float),
        "flow_alignment_1m":d*b.net_buy_quote.to_numpy(float)/b.quote_volume.replace(0,np.nan).to_numpy(float),
        "flow_alignment_15m":d*b.flow_15m_buy_signed.to_numpy(float),
        "zone_intersect":((b.high>=level-width)&(b.low<=level+width)).where(b.close.notna()).to_numpy(),
        "close_on_favorable_side":((d*(b.close-level))>0).where(b.close.notna()).to_numpy(),
    })
    # strict window-complete rolling zone interaction; NaN propagates across gaps.
    result["zone_overlap_trailing15"] = (pd.to_numeric(result.zone_intersect,errors="coerce")
                                             .rolling(15,min_periods=15).mean())
    ent, ext = e.get("entry_time_ms",np.nan),e.get("exit_time_ms",np.nan)
    ent = float(ent) if pd.notna(ent) else np.nan
    ext = float(ext) if pd.notna(ext) else np.nan
    result["trade_entered_by_t"] = bool(np.isfinite(ent)) & (ms>=ent)
    result["trade_active_at_t"] = (bool(np.isfinite(ent)) & bool(np.isfinite(ext))
                                   & (ms>=ent) & (ms<ext))
    result["trade_exited_by_t"] = bool(np.isfinite(ext)) & (ms>=ext)
    result["entry_marker"] = (np.isfinite(ent) & (ms>=ent) & (ms<ent+MIN_MS))
    result["exit_marker"] = (np.isfinite(ext) & (ms>=ext) & (ms<ext+MIN_MS))
    return result


def build_panel(cohort: pd.DataFrame, bars_by_symbol: dict[str,pd.DataFrame],cfg:Config=Config()) -> pd.DataFrame:
    frames=[event_minute_panel(r,bars_by_symbol[r["symbol"]],cfg) for r in cohort.to_dict("records")]
    result=pd.concat(frames,ignore_index=True)
    if len(result)!=len(cohort)*(cfg.before_min+cfg.after_min+1):
        raise AssertionError("A05a expected one original signal per minute on full event grid")
    return result


def summaries(panel:pd.DataFrame, cohort:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Median market paths include post-exit price tape; NOT an 'active trades' chart."""
    values=["signed_close_from_signal_bp","signed_close_from_level_bp","signed_1m_close_return_bp",
            "flow_alignment_1m","flow_alignment_15m","rv_15m_bp","rv_60m_bp",
            "quote_vs_prior60","quote_15m_vs_prior60","zone_overlap_trailing15"]
    work = panel[panel.eventual_outcome.isin(OUTCOMES)]
    groups=["symbol","eventual_outcome","feasibility_band","relative_min"]
    a=work.groupby(groups,observed=True).agg(
        n_candidates=("a02_candidate_row","size"),
        n_price_present=("signed_close_from_signal_bp","count"),
        n_trade_active=("trade_active_at_t","sum"),
        n_trade_exited=("trade_exited_by_t","sum"),
        **{f"median_{col}":(col,"median") for col in values}
    ).reset_index()
    # Also aggregate by outcome across both feasibility bands, with ALL symbols.
    allx=work.assign(symbol="ALL",feasibility_band="ALL")
    incl=work.assign(feasibility_band="ALL")
    together=pd.concat([a,
      incl.groupby(groups,observed=True).agg(n_candidates=("a02_candidate_row","size"),n_price_present=("signed_close_from_signal_bp","count"),
      n_trade_active=("trade_active_at_t","sum"),n_trade_exited=("trade_exited_by_t","sum"),
      **{f"median_{col}":(col,"median") for col in values}).reset_index(),
      allx.groupby(groups,observed=True).agg(n_candidates=("a02_candidate_row","size"),n_price_present=("signed_close_from_signal_bp","count"),
      n_trade_active=("trade_active_at_t","sum"),n_trade_exited=("trade_exited_by_t","sum"),
      **{f"median_{col}":(col,"median") for col in values}).reset_index()],ignore_index=True)
    # Endpoints and pre/post 5m/15m/60m/240m landmarks; no label leakage in underlying bars.
    selected=[-240,-120,-60,-30,-15,-5,-1,0,1,5,15,30,60,120,240]
    endpoints=together[together.relative_min.isin(selected)].copy()
    # Completeness by candidate: precise missing minutes, early exits, and original cohorts.
    q=panel.groupby("a02_candidate_row").agg(
       symbol=("symbol","first"),eventual_outcome=("eventual_outcome","first"),
       observed_pre=("has_minute_bar",lambda s:int(s.iloc[:240].sum())),
       observed_signal=("has_minute_bar",lambda s:int(s.iloc[240])),
       observed_post=("has_minute_bar",lambda s:int(s.iloc[241:].sum())),
       observed_first_touch=("first_touch_marker","sum"),
       source_available_any=("level_source_timestamp_le_bar","sum"),
       exited_during_post_window=("exit_marker","sum"),
    ).reset_index()
    q["complete_pre_240"] = q.observed_pre.eq(240)
    q["complete_post_240"] = q.observed_post.eq(240)
    return together.sort_values(groups).reset_index(drop=True),endpoints.sort_values(groups).reset_index(drop=True),q



# Inclusive 1m close-relative ranges, designated before examining historical paths.
PHASES = (
    ("context_-240_-61", -240, -61),
    ("approach_-60_-16", -60, -16),
    ("formation_-15_0", -15, 0),
    ("immediate_+1_+15", 1, 15),
    ("early_+16_+60", 16, 60),
    ("late_+61_+240", 61, 240),
)


def phase_panels(panel:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Descriptive fixed-window price/flow/zone summaries, no outcome filtering."""
    output=[]
    for (id_,symbol),g in panel.groupby(["a02_candidate_row","symbol"],sort=True):
        g=g.set_index("relative_min")
        for label,lo,hi in PHASES:
            seg=g.loc[lo:hi]
            need=hi-lo+1
            complete=(len(seg)==need and seg.has_minute_bar.all()
                      and seg["quote_volume"].notna().all())
            volume=seg.quote_volume.to_numpy(dtype=float)
            flow=seg.flow_alignment_1m.to_numpy(dtype=float)
            valid=(np.isfinite(volume)&np.isfinite(flow))
            aligned_quote=float(np.sum(volume[valid]*flow[valid]))
            total_quote=float(np.sum(volume[valid]))
            phase_return=(float(seg.signed_close_from_signal_bp.iloc[-1]
                                - seg.signed_close_from_signal_bp.iloc[0]) if complete else np.nan)
            outcome=g.eventual_outcome.iloc[0]
            output.append({"a02_candidate_row":int(id_),"symbol":symbol,
               "eventual_outcome":outcome,"feasibility_band":g.feasibility_band.iloc[0],
               "phase":label,"from_min":lo,"to_min":hi,"expected_bars":need,
               "observed_bars":int(seg.has_minute_bar.sum()),"complete_window":bool(complete),
               "signed_price_change_bp":phase_return,
               "aligned_flow_fraction":aligned_quote/total_quote if complete and total_quote>0 else np.nan,
               "quote_total":total_quote if complete else np.nan,
               "median_quote_vs_prior60":float(seg.quote_vs_prior60.median()) if complete else np.nan,
               "median_rv_15m_bp":float(seg.rv_15m_bp.median()) if complete else np.nan,
               "zone_overlap_fraction":float(pd.to_numeric(seg.zone_intersect,errors="coerce").mean()) if complete else np.nan,
               "first_touch_in_phase":int(seg.first_touch_marker.sum()),
               "level_source_in_phase":int(seg.level_source_marker.sum()),
               "trade_active_at_phase_end":bool(seg.trade_active_at_t.iloc[-1]),
            })
    phases=pd.DataFrame(output)
    m=["signed_price_change_bp","aligned_flow_fraction","quote_total",
       "median_quote_vs_prior60","median_rv_15m_bp","zone_overlap_fraction"]
    grouping=["symbol","eventual_outcome","feasibility_band","phase"]
    z=phases.groupby(grouping,observed=True).agg(n_events=("a02_candidate_row","size"),
      n_complete=("complete_window","sum"),
      **{f"median_{x}":(x,"median") for x in m}).reset_index()
    together=phases.assign(feasibility_band="ALL").groupby(grouping,observed=True).agg(
      n_events=("a02_candidate_row","size"),n_complete=("complete_window","sum"),
      **{f"median_{x}":(x,"median") for x in m}).reset_index()
    return phases,pd.concat([z,together],ignore_index=True).sort_values(grouping).reset_index(drop=True)


def plot_diagnostics(panel:pd.DataFrame, summary:pd.DataFrame,out:Path) -> None:
    import matplotlib.pyplot as plt
    outcome_color={"TARGET":"tab:green","STOP":"tab:red","TIMEOUT":"tab:orange"}
    metrics=[("median_signed_close_from_signal_bp","Signed price from signal close (bp)","05a_price_paths.png"),
             ("median_flow_alignment_15m","15m trailing aligned taker-flow imbalance","05a_flow_context.png"),
             ("median_quote_15m_vs_prior60","15m mean quote volume / prior 60m mean","05a_activity_context.png"),
             ("median_rv_15m_bp","15m trailing realized volatility (bp)","05a_rv_context.png"),
             ("median_signed_close_from_level_bp","Signed price relative to eventual level (bp)","05a_level_context.png")]
    for metric,ylabel,filename in metrics:
        fig, axes=plt.subplots(1,2,figsize=(14,4.8),sharex=True)
        for ax,sym in zip(axes,("BTCUSDT","ETHUSDT")):
            for outcome in OUTCOMES:
                x=summary[(summary.symbol==sym)&(summary.eventual_outcome==outcome)&(summary.feasibility_band=="ALL")]
                ax.plot(x.relative_min,x[metric],label=f"{outcome} (n={int(x.n_candidates.max())})",color=outcome_color[outcome],lw=1.7)
            ax.axvline(0,color="black",lw=1,linestyle="--",label="signal close")
            ax.axhline(0,color="0.5",lw=.5)
            ax.set(title=sym,xlabel="Minutes relative to confirmed second-touch signal",ylabel=ylabel)
            ax.legend(fontsize=8)
        fig.suptitle("A05a: FULL market path; after exit is market context, NOT continuing trade P&L")
        fig.tight_layout();fig.savefig(out/filename,dpi=145,bbox_inches="tight");plt.close(fig)
    # Raw-event heatmap: sort by outcome and original ID (not optimized thresholds).
    for sym in ("BTCUSDT","ETHUSDT"):
        z=panel[(panel.symbol==sym)&panel.eventual_outcome.isin(OUTCOMES)]
        groups=[];labels=[]
        for o in OUTCOMES:
            g=z[z.eventual_outcome==o].pivot(index="a02_candidate_row",columns="relative_min",values="signed_close_from_signal_bp")
            groups.append(g.sort_index());labels.append((o,len(g)))
        full=pd.concat(groups)
        if full.empty:continue
        fig,ax=plt.subplots(figsize=(14,6));im=ax.imshow(full.to_numpy(dtype=float),aspect="auto",origin="upper",
            extent=[-240.5,240.5,len(full)-.5,-.5],vmin=-160,vmax=160,cmap="RdBu")
        ax.axvline(0,color="black",lw=1.5)
        loc=0
        for _,n in labels[:-1]:
            loc+=n;ax.axhline(loc-.5,color="black",lw=1)
        mids=[];st=0
        for o,n in labels:mids.append(st+(n-1)/2);st+=n
        ax.set_yticks(mids,[f"{o} ({n})" for o,n in labels]);ax.set(xlabel="Minutes relative to second-touch close",title=f"{sym} signed event path (bp), retrospectively grouped by exit")
        fig.colorbar(im,ax=ax,label="Signed close vs signal close, bp (clipped at ±160)")
        fig.tight_layout();fig.savefig(out/f"05a_{sym}_event_heatmap.png",dpi=145,bbox_inches="tight");plt.close(fig)



def plot_individual_examples(panel:pd.DataFrame, out:Path)->None:
    """The smallest complete candidate ID in each outcome/symbol; examples, not exemplars."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(17,9),sharex=True)
    for i,sym in enumerate(("BTCUSDT","ETHUSDT")):
        for j,outcome in enumerate(OUTCOMES):
            ax=axes[i,j]
            sample=panel[(panel.symbol==sym)&(panel.eventual_outcome==outcome)]
            ids=sample.groupby("a02_candidate_row").has_minute_bar.all()
            ids=ids[ids].index
            if not len(ids):
                ax.set_title(f"{sym} / {outcome}: no complete event")
                continue
            p=sample[sample.a02_candidate_row==ids.min()].sort_values("relative_min")
            ax.plot(p.relative_min,p.signed_close_from_signal_bp,lw=1.3,label="Signed close vs signal")
            lev=float(p.signed_close_from_level_bp.iloc[0]-p.signed_close_from_signal_bp.iloc[0])
            ax.axhline(-lev,color="tab:purple",linestyle=":",lw=1,label="Eventual liquidity level")
            ax.axvline(0,color="black",linestyle="--",label="Second touch / signal")
            for flag,name,color in [("first_touch_marker","First touch","tab:green"),
                                      ("level_source_marker","Level source","tab:gray"),
                                      ("entry_marker","Entry","tab:blue"),
                                      ("exit_marker","Exit","tab:red")]:
                x=p.loc[p[flag],"relative_min"]
                if len(x):ax.axvline(float(x.iloc[0]),color=color,alpha=.7,lw=1,label=name)
            ax.set_title(f"{sym} {outcome} · ID {ids.min()} (illustrative)")
            ax.set_xlabel("Minutes from confirmed signal")
            ax.set_ylabel("Signed price (bp)")
            if i==0 and j==0:ax.legend(fontsize=7,loc="upper left")
    fig.suptitle("A05a historical examples: earliest complete candidate IDs in each outcome (not representative)")
    fig.tight_layout();fig.savefig(out/"05a_example_event_timelines.png",dpi=145,bbox_inches="tight");plt.close(fig)


def run(repo:Path,futures_root:Path,out:Path,cfg:Config=Config())->dict:
    check_config(cfg)
    cohort=load_event_cohort(repo,cfg)
    candles={sym:load_context_minutes(Path(futures_root)/f"{sym}_um_1m.parquet",cfg)
             for sym in ("BTCUSDT","ETHUSDT")}
    panel=build_panel(cohort,candles,cfg)
    curves,checkpoints,qc=summaries(panel,cohort)
    phases,phase_summary=phase_panels(panel)
    out.mkdir(parents=True,exist_ok=True)
    panel.to_csv(out/"05a_event_minute_panel.csv.gz",index=False,compression="gzip")
    curves.to_csv(out/"05a_median_context_curves.csv",index=False)
    checkpoints.to_csv(out/"05a_context_checkpoints.csv",index=False)
    qc.to_csv(out/"05a_coverage.csv",index=False)
    phases.to_csv(out/"05a_phase_panel.csv",index=False)
    phase_summary.to_csv(out/"05a_phase_summary.csv",index=False)
    cohort[["a02_candidate_row","symbol","side","signal_time_utc","eventual_outcome",
            "level_type","level_price","first_touch_close","second_touch_close","level_source_time",
            "target_to_sigma60","target","initial_stop","entry_time_ms","exit_time_ms"]].to_csv(out/"05a_cohort_metadata.csv",index=False)
    plot_diagnostics(panel,curves,out)
    plot_individual_examples(panel,out)
    decision={"version":VERSION,"parent":"LA-A04-0.1.1","status":"EXPLORATORY_RETROSPECTIVE_CONTEXT_NOT_PREDICTIVE",
              "config":asdict(cfg),"signals":len(cohort),"minute_rows":len(panel),"phase_rows":len(phases),
              "per_event_expected_rows":cfg.before_min+cfg.after_min+1,
              "complete_pre240":int(qc.complete_pre_240.sum()),"complete_post240":int(qc.complete_post_240.sum()),
              "missing_bar_rows":int((~panel.has_minute_bar).sum()),
              "outcomes":{k:int((cohort.reason==k).sum()) for k in OUTCOMES+EXTRA_OUTCOMES},
              "anchor":"t=0 is completed A02 second-touch signal candle close; actual entry is separate",
              "post_exit":"market trajectory remains displayed even once position has exited; not account P&L",
              "level_reference":"the eventually identified level applied retrospectively; pre-level-formation interpretation is hindsight",
              "forecast":"frozen A03d VD18 60m spot forecast is context only; not computed from post-signal futures bars"}
    (out/"05a_decision.json").write_text(json.dumps(decision,indent=2)+"\n",encoding="utf8")
    return decision


def main(argv=None):
    from src.data_engineering.modeling.liquidity_absorption_stage_a import repo_root,binance_root
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path);p.add_argument("--binance-root",type=Path);p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    r=a.repo or repo_root()
    result=run(r,a.binance_root or binance_root(r),a.out or r/"notebooks"/"Liquidity_Absorption"/"05a")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
