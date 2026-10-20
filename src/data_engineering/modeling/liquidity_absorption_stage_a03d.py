"""LA-A03d: causal VD18 HAR forecasts -> risk controls for Liquidity Absorption.

NO new direction signal. A03/A03c select entries and frozen structural targets.
Volatility risk floors sit beyond signal wick AND defended zone. The source is
TRAIN-fitted VD18 HAR_4, with 5m Binance SPOT data completed at/before signal.
Research-only; maker trade-through cannot identify queue fills. No portfolio PnL.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-A03d-0.1.0"
HORIZONS = (15, 60, 240)
HOLD_WINDOWS = (60, 240)
CLOSED = {"STOP", "TARGET", "TIMEOUT"}
REF_EQUITY = 10_000.0


@dataclass(frozen=True)
class RiskConfig:
    sigma_stop_horizon_min: int = 60
    sigma_stop_multiplier: float = 1.0
    risk_fraction: float = 0.0025  # synthetic risk-budget convention, NOT chosen from P&L
    reference_equity_usdt: float = REF_EQUITY
    max_notional_usdt: float = 10_000.0
    taker_roundtrip_cost_allowance_bp: float = 14.0
    maker_entry_taker_exit_cost_allowance_bp: float = 9.0
    hold_windows: tuple[int, ...] = HOLD_WINDOWS
    spot_max_stale_minutes: int = 0  # only exact most recent completed 5m bar


def _check(cfg: RiskConfig) -> None:
    if (cfg.sigma_stop_horizon_min not in HORIZONS or cfg.sigma_stop_multiplier <= 0
            or cfg.risk_fraction <= 0 or cfg.reference_equity_usdt <= 0
            or cfg.max_notional_usdt <= 0 or cfg.spot_max_stale_minutes != 0
            or not cfg.hold_windows or any(h not in HORIZONS for h in cfg.hold_windows)
            or min(cfg.taker_roundtrip_cost_allowance_bp, cfg.maker_entry_taker_exit_cost_allowance_bp) < 0):
        raise ValueError("Invalid or unregistered LA-A03d risk policy")


def _env_value(repo: Path, key: str) -> str | None:
    if os.environ.get(key):
        return os.environ[key]
    f = repo / ".env"
    if f.exists():
        for raw in f.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    return None


def spot_history_directory(repo: Path, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    data = _env_value(repo, "DERIBIT_DATA_ROOT")
    if data is None:
        raise FileNotFoundError("VD18 spot history requires DERIBIT_DATA_ROOT or --spot-history")
    return Path(data) / "history" / "price_5m"


def load_vd18_fits(repo: Path, horizons=HORIZONS) -> tuple[dict, dict]:
    manifest = repo / "notebooks" / "Volatility_Dynamics" / "18" / "18_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("parameters", {}).get("version") != "VD18-0.1.0":
        raise ValueError("VD18 fitted artifact version drift")
    if data.get("decision", {}).get("status") != "HISTORICAL_EXPLORATORY":
        raise ValueError("Unexpected VD18 training decision")
    fitted = data["parameters"]["fitted_har"]
    fits = {}
    for h in horizons:
        record = fitted[str(h)]
        if data["decision"]["selected_har_by_horizon"][str(h)] != "HAR_4":
            raise ValueError(f"VD18 selected model for {h}m changed; refuse HAR4 substitution")
        fit = record["har_fits"]["HAR_4"]
        if fit["lookbacks"] != [30, 60, 240, 1440]:
            raise ValueError("Unexpected HAR4 lookback contract")
        fits[h] = fit
    provenance = {"model": "VD18-0.1.0", "selection": "HAR_4 historical-VALID chosen",
                  "saved_sample_last": data["sample_last"], "source": str(manifest.relative_to(repo)),
                  "forecast_measure": "E[net terminal 5m SPOT log-return squared], not path max adverse excursion"}
    return fits, provenance


def build_spot_history(path: Path) -> pd.DataFrame:
    """Exactly replicate VD14's causal 5m log return and VD18 rolling HAR features."""
    raw = pd.read_csv(path, usecols=["state_time", "close"])
    time = pd.to_datetime(raw.state_time, utc=True, errors="raise")
    px = pd.to_numeric(raw.close, errors="raise")
    if (time.isna().any() or time.duplicated().any() or not time.is_monotonic_increasing
            or ((time.dt.minute % 5 != 0) | (time.dt.second != 0)).any()
            or not np.isfinite(px).all() or (px <= 0).any()):
        raise ValueError(f"Invalid VD18 5m spot close history: {path}")
    z = pd.DataFrame({"state_time": time, "close": px})
    valid = z.state_time.diff().eq(pd.Timedelta(minutes=5))
    z["return_5m"] = np.log(z.close / z.close.shift()).where(valid)
    sq = z.return_5m.pow(2)
    for horizon in (30, 60, 240, 1440):
        z[f"har_logrv_{horizon}"] = np.log(np.maximum(sq.rolling(horizon // 5, min_periods=horizon // 5).sum(), 1e-16))
    t = z.state_time
    z["slot"] = (t.dt.hour * 60 + t.dt.minute) // 5
    return z.set_index("state_time", drop=False, verify_integrity=True)


def predict_saved_har(fit: dict, frame: pd.DataFrame, asset: str) -> np.ndarray:
    """VD18 markov_har_variance.predict_har algebra from the frozen HAR4 artifact.

    Duplicates only this deterministic inference kernel; no model fitting/refitting,
    and no VD18 output is altered. Synthetic tests check the full design matrix.
    """
    d = fit["design"]
    if asset not in d["asset_names"] or asset not in fit["net_return_scales"]:
        raise ValueError(f"Asset {asset} absent from frozen VD18 fit")
    if list(d["features"]) != ["har_logrv_30", "har_logrv_60", "har_logrv_240", "har_logrv_1440"]:
        raise ValueError("Unexpected VD18 predictor feature contract")
    x = (frame[d["features"]].to_numpy(dtype=float) - np.asarray(d["center"], dtype=float)) / np.asarray(d["std"], dtype=float)
    slot = frame["slot"].to_numpy(dtype=float)
    extras = [np.full(len(frame), float(asset == a)) for a in d["asset_names"][1:]]
    X = np.column_stack((np.ones(len(frame)), x, np.sin(2*np.pi*slot/288), np.cos(2*np.pi*slot/288), *extras))
    beta = np.asarray(fit["coefficients"], dtype=float)
    if X.shape[1] != len(beta) or not np.isfinite(X).all() or not np.isfinite(beta).all():
        raise ValueError("Invalid VD18 predict design / incomplete spot history")
    return np.maximum(np.exp(np.clip(X @ beta, -40, 5)) * float(fit["net_return_scales"][asset]), 1e-12)


def forecast_origins(events: pd.DataFrame, spot_dir: Path, fits: dict) -> pd.DataFrame:
    """Attach only a bar with state_time == floor(signal to 5m); no forward fill."""
    if events.a02_candidate_row.duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    results = []
    for symbol, group in events.groupby("symbol", sort=True):
        if symbol not in ("BTCUSDT", "ETHUSDT"):
            raise ValueError("Unexpected event asset")
        asset = symbol.removesuffix("USDT")
        spot = build_spot_history(Path(spot_dir) / f"{asset}.csv.gz")
        ts = pd.to_datetime(group.signal_time_utc, utc=True, errors="raise")
        origins = ts.dt.floor("5min")
        x = spot.reindex(pd.DatetimeIndex(origins))
        for idx, (row, time) in enumerate(zip(group.to_dict("records"), origins)):
            a = x.iloc[idx]
            p = {"a02_candidate_row": int(row["a02_candidate_row"]), "symbol": symbol,
                 "signal_time_utc": row["signal_time_utc"], "vd18_origin_utc": str(time),
                 "forecast_status": "PRESENT"}
            if pd.isna(a.state_time) or not np.isfinite(a[[f"har_logrv_{h}" for h in (30,60,240,1440)]].to_numpy(dtype=float)).all():
                p["forecast_status"] = "NO_COMPLETE_CAUSAL_5M_HISTORY"
                for h in HORIZONS:
                    p[f"variance_{h}"] = np.nan
                    p[f"sigma_bp_{h}"] = np.nan
            else:
                if pd.Timestamp(a.state_time) > pd.Timestamp(row["signal_time_utc"]):
                    raise AssertionError("Volatility origin after signal close")
                one = x.iloc[[idx]].copy()
                for h in HORIZONS:
                    v = float(predict_saved_har(fits[h], one, asset)[0])
                    p[f"variance_{h}"] = v
                    p[f"sigma_bp_{h}"] = np.sqrt(v) * 10000
            results.append(p)
    return pd.DataFrame(results).sort_values("a02_candidate_row").reset_index(drop=True)


def risk_stop(side: str, entry: float, level: float, halfwidth: float, signal_extreme: float,
              sigma_bp: float, tick: float, cfg: RiskConfig) -> float:
    """Protect confirmed second-touch wick and level; add one sigma-scale price buffer.

    Stop is beyond *the most distant* of zone edge and wick by >= 1 tick,
    or an entry-relative 60m forecast sigma. Vol predicts scale, not direction.
    """
    _check(cfg)
    if side not in ("long", "short") or any(not np.isfinite(float(x)) for x in
                                            (entry, level, halfwidth, signal_extreme, sigma_bp, tick)):
        raise ValueError("Invalid risk-stop input")
    if min(entry, level, signal_extreme, halfwidth, sigma_bp, tick) <= 0:
        raise ValueError("Risk stop inputs must be positive")
    sigma_dist = entry * (sigma_bp / 10000) * cfg.sigma_stop_multiplier
    if side == "long":
        raw = min(level - halfwidth - tick, signal_extreme - tick, entry - sigma_dist)
        stop = np.floor(raw / tick + 1e-9) * tick
        if not 0 < stop < entry:
            raise ValueError("Invalid risk-stop geometry")
        return float(stop)
    raw = max(level + halfwidth + tick, signal_extreme + tick, entry + sigma_dist)
    stop = np.ceil(raw / tick - 1e-9) * tick
    if stop <= entry:
        raise ValueError("Invalid risk-stop geometry")
    return float(stop)


def reference_position_size(entry: float, stop: float, cfg: RiskConfig, maker: bool) -> dict:
    _check(cfg)
    distance_bp = abs(entry-stop)/entry*10000
    expense = cfg.maker_entry_taker_exit_cost_allowance_bp if maker else cfg.taker_roundtrip_cost_allowance_bp
    gross_budget = cfg.reference_equity_usdt*cfg.risk_fraction
    notional = min(cfg.max_notional_usdt, gross_budget*10000/(distance_bp + expense))
    return {"risk_distance_bp": distance_bp, "cost_allowance_bp": expense,
            "risk_budget_usdt": gross_budget, "reference_notional_usdt": notional,
            "reference_quantity": notional/entry}


def _read_event_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Independent frozen target source + causal 1m wick. Do NOT rescore A02.
    from src.data_engineering.modeling.liquidity_absorption_stage_a03 import load_candidates
    from src.data_engineering.modeling.liquidity_absorption_stage_a03c import validate_inputs
    candidates = load_candidates(repo)
    base = validate_inputs(repo)
    for name in ("bar_low", "bar_high"):
        if name not in candidates:
            raise ValueError(f"A02b lacks necessary signal wick {name}")
    cols = ["a02_candidate_row", "bar_low", "bar_high"]
    base = base.merge(candidates[cols], on="a02_candidate_row", validate="one_to_one")
    check = base.merge(candidates[["a02_candidate_row", "symbol", "side", "signal_time_utc", "level_price"]],
                       on="a02_candidate_row", suffixes=("", "_a02"), validate="one_to_one")
    if len(check) != len(base) or len(check) != 452:
        raise ValueError("Candidate population changed")
    for k in ("symbol", "side", "signal_time_utc"):
        if not check[k].astype(str).eq(check[k+"_a02"].astype(str)).all():
            raise ValueError(f"A02/A03b identity mismatch: {k}")
    if not np.allclose(check.level_price.astype(float),check.level_price_a02.astype(float),rtol=0,atol=1e-7):
        raise ValueError("A02/A03b level mismatch")
    return base, candidates


def paired_old_comparison(frame: pd.DataFrame, repo: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Paired bp outcomes on identical candidate IDs when BOTH variants closed.

    Historical tests in this sample are exploratory; NEVER select winners by net outcomes.
    """
    base=repo/"notebooks"/"Liquidity_Absorption"
    src=[]
    f=base/"03"/"03_execution_events.csv"
    if f.exists():
        x=pd.read_csv(f)
        x=x[x["mode"]=="FIXED"]
        x=x[["a02_candidate_row","symbol","reason","net_return_bp_assumed_costs","gross_return_bp"]].copy()
        x["old_label"]="A03_ORIGINAL_TAKER_FIXED"
        src.append(x)
    g=base/"03c"/"03c_events.csv"
    if g.exists():
        x=pd.read_csv(g)
        x=x[x["mode"]=="MAKER_TOUCH_3RD"]
        x=x[["a02_candidate_row","symbol","reason","net_return_bp_assumed_costs","gross_return_bp"]].copy()
        x["old_label"]="A03C_MAKER_TOUCH"
        src.append(x)
    if not src:
        return pd.DataFrame(),pd.DataFrame()
    old=pd.concat(src,ignore_index=True)
    new=frame[(frame.hold_minutes==240)&(
        ((frame.entry_kind=="TAKER_FIRST_POST_SIGNAL")&(frame.model_mode=="VD18_VOL_FIXED"))|
        ((frame.entry_kind=="MAKER_ELIGIBILITY_ONLY")&(frame.model_mode=="VD18_VOL_MAKER_TOUCH_3RD")))]
    new=new[["a02_candidate_row","symbol","reason","net_return_bp_assumed_costs","gross_return_bp","model_mode"]].copy()
    new["old_label"]=np.where(new.model_mode=="VD18_VOL_FIXED","A03_ORIGINAL_TAKER_FIXED","A03C_MAKER_TOUCH")
    pairs=new.merge(old,on=["a02_candidate_row","symbol","old_label"],how="inner",validate="one_to_one",suffixes=("_vol","_old"))
    pairs["both_closed"] = pairs.reason_vol.isin(CLOSED)&pairs.reason_old.isin(CLOSED)
    good=pairs[pairs.both_closed]
    stats=[]
    for (sym,label),z in good.groupby(["symbol","old_label"]):
        stats.append({"symbol":sym,"baseline":label,"paired_closed":len(z),
                      "mean_original_net_bp":z.net_return_bp_assumed_costs_old.mean(),
                      "mean_vol_net_bp":z.net_return_bp_assumed_costs_vol.mean(),
                      "mean_vol_minus_original_bp":(z.net_return_bp_assumed_costs_vol-z.net_return_bp_assumed_costs_old).mean()})
    return pairs,pd.DataFrame(stats)


def plot_diagnostics(trades: pd.DataFrame, origins: pd.DataFrame, output: Path) -> None:
    """Virtual realised-PnL curves; overlapping positions/capital NOT modeled."""
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for ax,asset in zip(axes,("BTCUSDT","ETHUSDT")):
        for key,subset in trades[(trades.symbol==asset)&trades.reason.isin(CLOSED)].groupby(["entry_kind","hold_minutes","model_mode"]):
            entry_kind,hold,mode=key
            if mode not in ("VD18_VOL_FIXED","VD18_VOL_MAKER_TOUCH_3RD"):
                continue
            z=subset.sort_values(["exit_time_ms","a02_candidate_row"])
            label=("Taker" if entry_kind.startswith("TAKER") else "Maker eligibility")+f" {hold}m"
            ax.step(pd.to_datetime(z.exit_time_ms,unit="ms",utc=True),z.virtual_net_usdt.cumsum(),where="post",label=label)
        ax.axhline(0,linewidth=.8)
        ax.set_ylabel(asset+"\nvirtual net USDT")
        ax.legend(fontsize=8)
    fig.suptitle("A03d volatility-risk virtual realised P&L: fixed reference risk budget per hypothetical entry\nNOT a funded account equity curve; spot vol / perp prices / model fills")
    axes[-1].set_xlabel("UTC simulated exit time")
    fig.tight_layout(rect=[0,0,1,.91])
    fig.savefig(output/"03d_virtual_equity.png",dpi=140,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6))
    for sym,z in trades[(trades.model_mode=="VD18_VOL_FIXED")&(trades.hold_minutes==240)&trades.initial_risk_bp.notna()].groupby("symbol"):
        ax.scatter(z.vol_sigma_bp_60,z.initial_risk_bp,alpha=.45,s=15,label=sym)
    if ax.get_legend_handles_labels()[0]:ax.legend()
    ax.set(xlabel="VD18 spot sigma 60m (bp)",ylabel="Perp initial stop distance (bp)",
           title="Protective stop distance vs forecast terminal-return sigma\nWick/zone protection can dominate the volatility floor")
    fig.tight_layout();fig.savefig(output/"03d_forecast_vs_stop.png",dpi=140,bbox_inches="tight");plt.close(fig)


def build_results(repo: Path, data_root: Path, spot_dir: Path, out: Path, cfg: RiskConfig = RiskConfig()) -> dict:
    """Run frozen structural-target volatility risk variants without changing earlier files."""
    _check(cfg)
    from src.data_engineering.modeling.liquidity_absorption_stage_a03 import Config as A03Config, EventSimulation, prepare_bars, scan_event_tapes
    from src.data_engineering.modeling.liquidity_absorption_stage_a03c import Config as A03cConfig, TapeStudy, scan_tapes
    from src.data_engineering.modeling.liquidity_absorption_stage_a03 import tick_for
    a03b, candidates = _read_event_inputs(repo)
    fits, provenance = load_vd18_fits(repo)
    origins = forecast_origins(a03b, spot_dir, fits)
    with_forecast = a03b.merge(origins.drop(columns=["symbol","signal_time_utc"]), on="a02_candidate_row", validate="one_to_one")
    if len(with_forecast)!=452:
        raise AssertionError("Expected 452 original signals")
    out.mkdir(parents=True, exist_ok=True)
    with_forecast.to_csv(out/"03d_event_forecasts.csv", index=False)
    usable=with_forecast[with_forecast.forecast_status=="PRESENT"].copy()
    if usable.empty:
        raise ValueError("No complete causal VD18 5-minute spot forecasts; check VD18 history and bar-end timestamps. See 03d_event_forecasts.csv")
    source_map={int(z.a02_candidate_row):z for z in candidates.itertuples(index=False)}
    
    class VolTaker(EventSimulation):
        def __init__(self, event, bars, a03cfg, source):
            self.vol_source=source
            super().__init__(event,bars,a03cfg)
        def _start(self,tm,px):
            super()._start(tm,px)
            if self.entry_ms is None:
                return
            expected=self.vol_source
            if (not np.isclose(px, float(expected["entry_print"]),rtol=1e-10,atol=1e-7)
                or int(tm)!=int(float(expected["entry_time_ms"]))
                or not np.isclose(self.target, float(expected["target"]),rtol=1e-10,atol=1e-7)):
                raise AssertionError("A03b entry/target drift: cannot compare vol risk")
            vol_stop=risk_stop(self.side, px, self.level, self.width,
                               self.signal_low if self.d==1 else self.signal_high,
                               float(expected["sigma_bp_60"]),self.tick,cfg)
            self.initial_stop=vol_stop
            self.risk=self.d*(px-vol_stop)
            for st in self.states.values():
                st.stop=vol_stop
                st.initial_risk_price=self.risk
        def records(self):
            output=super().records()
            for row in output:
                row["entry_kind"]="TAKER_FIRST_POST_SIGNAL"
                row["target_frozen_a03b"]=True
                row["vol_sigma_bp_60"]=self.vol_source["sigma_bp_60"]
                row["vd18_origin_utc"]=self.vol_source["vd18_origin_utc"]
                row["hold_minutes"]=self.cfg.hold_minutes
                row["model_mode"]="VD18_VOL_"+row["mode"]
                row["initial_risk_bp"] = row["initial_risk_bps"]
                row["maker_eligibility_not_fill"] = False
                _augment_sizing(row,self.cfg.hold_minutes,False)
            return output

    class VolMaker(TapeStudy):
        def __post_init__(self):
            super().__post_init__()
            if self.reason:
                return
            self.stop=risk_stop(self.side,self.level,self.level,float(self.source["zone_halfwidth"]),
                                float(self.source["bar_low"] if self.d==1 else self.source["bar_high"]),
                                float(self.source["sigma_bp_60"]),tick_for(self.symbol,A03Config()),cfg)
        def _entry(self,tm,entry_price,print_price):
            if not (self.d*(entry_price-self.stop)>0 and self.d*(self.target-entry_price)>0):
                raise ValueError("Invalid passive volatility stop/structural target")
            self.entry_time=tm
            self.entry_px=entry_price
            self.qualifying_print=print_price
            self.max_favorable=entry_price
            self.max_adverse=entry_price
        def record(self):
            r=super().record()
            r["entry_kind"]="MAKER_ELIGIBILITY_ONLY"
            r["target_frozen_a03b"]=True
            r["vol_sigma_bp_60"]=self.source["sigma_bp_60"]
            r["vd18_origin_utc"]=self.source["vd18_origin_utc"]
            r["hold_minutes"]=self.cfg.hold_minutes
            r["model_mode"]="VD18_VOL_"+self.mode
            r["maker_eligibility_not_fill"]=True
            _augment_sizing(r,self.cfg.hold_minutes,True)
            return r

    def _augment_sizing(row, h, maker):
        entry=float(row.get("entry_print" if not maker else "entry_price",np.nan))
        stop=float(row.get("initial_stop" if not maker else "stop",np.nan))
        if np.isfinite(entry) and entry>0 and np.isfinite(stop) and stop>0:
            s=reference_position_size(entry,stop,cfg,maker)
            row.update(s)
            ret=float(row.get("net_return_bp_assumed_costs",np.nan))
            row["virtual_net_usdt"]=(ret/10000*s["reference_notional_usdt"] if np.isfinite(ret) else np.nan)
        else:
            row.update(dict(risk_distance_bp=np.nan,cost_allowance_bp=np.nan,risk_budget_usdt=cfg.reference_equity_usdt*cfg.risk_fraction,
                            reference_notional_usdt=np.nan,reference_quantity=np.nan,virtual_net_usdt=np.nan))

    simulations=[]; maker_sims=[]
    for sym,group in usable.groupby("symbol"):
        bars=prepare_bars(data_root,sym)
        for rec in group.to_dict("records"):
            raw=source_map[int(rec["a02_candidate_row"])]._asdict()
            for hold in cfg.hold_windows:
                c=A03Config(hold_minutes=hold)
                simulations.append(VolTaker(raw,bars,c,rec))
                mc=A03cConfig(hold_minutes=hold)
                for mode in ("MAKER_TOUCH_3RD", "MAKER_THROUGH_3RD"):
                    maker_sims.append(VolMaker(rec,mode,mc))
    taker_meta=scan_event_tapes(data_root, simulations) if simulations else {}
    maker_meta=scan_tapes(data_root,maker_sims) if maker_sims else {}
    records=[rec for s in simulations for rec in s.records()]+[s.record() for s in maker_sims]
    frame=pd.DataFrame.from_records(records)
    frame.to_csv(out/"03d_risk_execution.csv",index=False)
    keys=["entry_kind","model_mode","symbol","hold_minutes"]
    summary=[]
    for group,z in frame.groupby(keys,dropna=False):
        closed=z[z.reason.isin(CLOSED)]
        summary.append(dict(zip(keys,group))|{
            "n":int(len(z)),"n_closed":int(len(closed)),"n_censored":int(z.reason.astype(str).str.startswith("CENSORED").sum()),
            "n_entered":int(z.entry_time_ms.notna().sum()),
            "mean_net_bp_per_closed":float(closed.net_return_bp_assumed_costs.mean()) if len(closed) else None,
            "mean_net_r_per_closed":float(closed.net_r_assumed_costs.mean()) if len(closed) else None,
            "mean_virtual_usdt_per_closed":float(closed.virtual_net_usdt.mean()) if len(closed) else None,
            "total_virtual_net_usdt":float(closed.virtual_net_usdt.sum()) if len(closed) else None,
            "median_initial_risk_bp":float(closed.initial_risk_bp.median()) if len(closed) else None,
            "mean_net_bp_per_452_signals_partial":float(closed.net_return_bp_assumed_costs.sum()/452),
        })
    pd.DataFrame(summary).to_csv(out/"03d_summary.csv",index=False)
    paired, paired_summary=paired_old_comparison(frame,repo)
    paired.to_csv(out/"03d_paired_baseline.csv",index=False)
    paired_summary.to_csv(out/"03d_paired_summary.csv",index=False)
    if len(frame):
        plot_diagnostics(frame,with_forecast,out)
    contract={"version":VERSION,"risk_config":asdict(cfg),"vd18":provenance,
              "source_events":452,"forecast_present":int(len(usable)),
              "no_forecast_events":int(452-len(usable)),
              "stop_rule":"farthest protective stop: beyond second-touch wick, original zone outer edge, or k x VD18 60m spot net-return sigma; rounded outward to A03 assumed tick",
              "target":"unchanged original A03/A03b selected structural target, asserted at taker entry; maker shares frozen target",
              "holding":"compare predeclared 60 and 240 minutes; no outcome-based hold selection",
              "sizing":"reference position budget, bounded notional, includes assumed round-trip cost allowance; not live account equity",
              "maker":"A03c post-signal contact/through eligibility only; may be same encounter, NEVER proven maker queue fill",
              "caveat":"volatility predicts terminal return scale, NOT direction/MAE/first-passage; maker queue, bid/ask, funding, capital reservation absent",
              "data":"VD18 spot for predictor; Binance UM perps for executable print proxies; no spot-perp calibration validated",
              "status":"EXPLORATORY_FROZEN_MODEL_RISK_BRIDGE_NO_EDGE_CLAIM"}
    (out/"03d_contract.json").write_text(json.dumps(contract,indent=2,default=str)+"\n",encoding="utf-8")
    report={"version":VERSION,"status":contract["status"],"n_events":452,"n_forecasts":len(usable),
            "n_rows":len(frame),"n_taker_sims":len(simulations),"n_maker_sims":len(maker_sims),
            "taker_tape":taker_meta,"maker_tape":maker_meta}
    (out/"03d_decision.json").write_text(json.dumps(report,indent=2,default=str)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    from src.data_engineering.modeling.liquidity_absorption_stage_a import repo_root,binance_root
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path)
    p.add_argument("--binance-root",type=Path)
    p.add_argument("--spot-history",type=Path)
    p.add_argument("--out",type=Path)
    a=p.parse_args(argv)
    repo=a.repo or repo_root()
    root=a.binance_root or binance_root(repo)
    spot=spot_history_directory(repo,a.spot_history)
    out=a.out or repo/"notebooks"/"Liquidity_Absorption"/"03d"
    print(json.dumps(build_results(repo,root,spot,out),indent=2))

if __name__=="__main__":
    main()
