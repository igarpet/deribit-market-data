"""LA-A04: descriptive intratrade 1-minute paths, competing outcomes and causal landmarks.

Signal and target selection are FROZEN in A02/A03d. No outcomes are used as predictors.
One-minute UM futures candles approximate aggressive flow using kline taker-buy quote.
No tick archive duplication, no trades/portfolio optimization, no claim of causation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "LA-A04-0.1.0"
MIN_MS = 60_000
LANDMARKS = (1, 5, 15, 30, 60)
CLOSED = ("TARGET", "STOP", "TIMEOUT")


@dataclass(frozen=True)
class Config:
    hold_minutes: int = 240
    entry_kind: str = "TAKER_FIRST_POST_SIGNAL"
    model_mode: str = "VD18_VOL_FIXED"
    landmarks_min: tuple[int, ...] = LANDMARKS
    history_minutes: int = 30


def validate_primary(events: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    required = ("a02_candidate_row", "symbol", "side", "signal_time_utc", "entry_time_ms",
                "exit_time_ms", "reason", "entry_print", "initial_stop", "target",
                "level_price", "zone_halfwidth", "vol_sigma_bp_60", "entry_kind",
                "model_mode", "hold_minutes")
    missing = sorted(set(required).difference(events.columns))
    if missing:
        raise ValueError(f"A03d result missing columns: {missing}")
    z = events.loc[(events.entry_kind == cfg.entry_kind) &
                   (events.model_mode == cfg.model_mode) &
                   (pd.to_numeric(events.hold_minutes) == cfg.hold_minutes)].copy()
    if len(z) != 452 or z.a02_candidate_row.isna().any() or z.a02_candidate_row.duplicated().any():
        raise ValueError(f"A03d primary cohort identity mismatch: {len(z)} instead of 452 unique events")
    if not z.symbol.isin(("BTCUSDT", "ETHUSDT")).all() or not z.side.isin(("long", "short")).all():
        raise ValueError("Unexpected symbol/direction")
    z["signal_ms"] = (pd.to_datetime(z.signal_time_utc, utc=True, errors="raise")
                        .astype("int64") // 1_000_000)
    z["target_distance_bp"] = np.where(z.side == "long", 1, -1) * (
        pd.to_numeric(z.target) - pd.to_numeric(z.entry_print)) / pd.to_numeric(z.entry_print) * 10000
    z["risk_distance_bp"] = (pd.to_numeric(z.entry_print) - pd.to_numeric(z.initial_stop)).abs() / pd.to_numeric(z.entry_print) * 10000
    z["target_to_sigma60"] = z.target_distance_bp / pd.to_numeric(z.vol_sigma_bp_60)
    z["time_to_exit_min"] = (pd.to_numeric(z.exit_time_ms) - pd.to_numeric(z.entry_time_ms)) / MIN_MS
    entered = z.entry_time_ms.notna()
    if (z.loc[entered, "target_distance_bp"] <= 0).any() or (z.loc[entered, "risk_distance_bp"] <= 0).any():
        raise ValueError("Nonfavorable target / invalid risk geometry")
    if not z.loc[z.reason.isin(CLOSED), "exit_time_ms"].notna().all():
        raise ValueError("Completed event missing exit timestamp")
    return z.sort_values("a02_candidate_row").reset_index(drop=True)


def load_minute_bars(path: Path) -> pd.DataFrame:
    """Read each historical UM 1m file only once. Times are candle OPEN; use complete CLOSE."""
    from src.data_engineering.modeling.liquidity_absorption_stage_a import parse_kline_open_time
    cols = ["open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
    bars = pd.read_parquet(path, columns=cols)
    ts = parse_kline_open_time(bars.open_time)
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError(f"Invalid 1-minute timestamp sequence: {path}")
    z = bars.drop(columns="open_time").copy()
    z["close_ms"] = ts.astype("int64") // 1_000_000 + MIN_MS
    for c in cols[1:]:
        z[c] = pd.to_numeric(z[c], errors="raise")
    if ((z[["open", "high", "low", "close"]] <= 0).any().any() or
            z[["high", "low", "quote_volume", "taker_buy_quote_volume"]].isna().any().any() or
            (z.low > z.high).any() or (z.quote_volume < 0).any() or
            (z.taker_buy_quote_volume < 0).any() or
            (z.taker_buy_quote_volume > z.quote_volume + z.quote_volume.abs() * 1e-7 + 1e-8).any()):
        raise ValueError(f"Invalid 1-minute OHLC / aggressive flow: {path}")
    return z.set_index("close_ms", drop=False, verify_integrity=True).sort_index()


def one_landmark(event: dict, bars: pd.DataFrame, landmark_min: int, history_minutes: int = 30) -> dict:
    """Causal completed-bar features only for still-active trades at signal+h minutes.

    Important: signal is completed 1m candle CLOSE; no future data, no post-exit bar.
    An observation is a *landmark risk set*, not the whole original event population.
    """
    s = int(event["signal_ms"])
    m = s + landmark_min * MIN_MS
    o = {"a02_candidate_row": int(event["a02_candidate_row"]), "symbol": event["symbol"],
         "side": event["side"], "reason_final": event["reason"],
         "landmark_min": landmark_min, "landmark_time_utc": pd.Timestamp(m, unit="ms", tz="UTC").isoformat(),
         "at_risk": False, "observation_status": "NOT_EVALUATED"}
    if event["reason"] not in CLOSED or pd.isna(event["entry_time_ms"]) or pd.isna(event["exit_time_ms"]):
        o["observation_status"] = "NOT_COMPLETED_COHORT"
        return o
    if int(float(event["exit_time_ms"])) <= m:
        o["observation_status"] = "EXITED_BEFORE_LANDMARK"
        return o
    if m < int(float(event["entry_time_ms"])):
        o["observation_status"] = "NOT_YET_ENTERED"
        return o
    # Need exactly h consecutive complete minute bars ending strictly after signal.
    times = np.arange(s + MIN_MS, m + 1, MIN_MS, dtype=np.int64)
    loc = bars.index.get_indexer(times)
    if (loc < 0).any():
        o["observation_status"] = "INCOMPLETE_MINUTE_WINDOW"
        return o
    z = bars.iloc[loc]
    # Use the same completed 30m pre-signal activity reference for all landmark times.
    past_times = np.arange(s - history_minutes * MIN_MS, s, MIN_MS, dtype=np.int64) + MIN_MS
    past_locs = bars.index.get_indexer(past_times)
    baseline = bars.iloc[past_locs] if (past_locs >= 0).all() else None
    d = 1 if event["side"] == "long" else -1
    entry, level = float(event["entry_print"]), float(event["level_price"])
    stop, target = float(event["initial_stop"]), float(event["target"])
    width = float(event["zone_halfwidth"])
    risk = d * (entry - stop)
    target_distance = d * (target - entry)
    if min(entry, risk, target_distance) <= 0:
        raise ValueError("Invalid stop/target geometry at active landmark")
    vol = float(z.quote_volume.sum())
    buy = float(z.taker_buy_quote_volume.sum())
    highs, lows, closes = z.high.to_numpy(float), z.low.to_numpy(float), z.close.to_numpy(float)
    f_hi = float(highs.max()) if d == 1 else float(lows.min())
    a_lo = float(lows.min()) if d == 1 else float(highs.max())
    max_fav = d * (f_hi - entry)
    max_adv = d * (a_lo - entry)
    candle_center = (z.high.to_numpy(float) + z.low.to_numpy(float)) / 2
    zone_intersects = (z.high.to_numpy(float) >= level - width) & (z.low.to_numpy(float) <= level + width)
    price_change = d * (float(closes[-1]) - entry)
    o.update({
        "at_risk": True, "observation_status": "PRESENT", "bars_complete": int(len(z)),
        "signed_return_bp": price_change / entry * 10000,
        "progress_to_target": price_change / target_distance,
        "progress_to_stop_r": price_change / risk,
        "max_favorable_bp": max_fav / entry * 10000,
        "max_adverse_bp": max_adv / entry * 10000,
        "max_favorable_r": max_fav / risk,
        "max_adverse_r": max_adv / risk,
        "flow_alignment": d * (2 * buy - vol) / vol if vol > 0 else np.nan,
        "opposing_aggressor_share": (1 - d * (2 * buy - vol) / vol) / 2 if vol > 0 else np.nan,
        "quote_volume_window": vol,
        "quote_activity_vs_past30": (vol / len(z)) / baseline.quote_volume.mean()
            if baseline is not None and baseline.quote_volume.mean() > 0 else np.nan,
        "level_favorable_close_share": float(np.mean(d * (closes - level) > 0)),
        "zone_overlap_minute_fraction": float(zone_intersects.mean()),
        "zone_center_distance_bp": float(np.mean(np.abs(candle_center - level) / entry * 10000)),
        "target_distance_bp": target_distance / entry * 10000,
        "stop_distance_bp": risk / entry * 10000,
        "target_over_sigma60": float(event["target_to_sigma60"]),
        "sigma_bp_60": float(event["vol_sigma_bp_60"]),
        "target_source": str(event.get("target_source", "")),
        "level_type": str(event.get("level_type", "")),
        "time_to_exit_min": float(event["time_to_exit_min"]),
    })
    return o


def build_landmarks(primary: pd.DataFrame, by_symbol: dict[str, pd.DataFrame], cfg: Config = Config()) -> pd.DataFrame:
    output = []
    for r in primary.to_dict("records"):
        sym = str(r["symbol"])
        if sym not in by_symbol:
            raise ValueError(f"Missing 1-minute candles for {sym}")
        for h in cfg.landmarks_min:
            output.append(one_landmark(r, by_symbol[sym], h, cfg.history_minutes))
    a = pd.DataFrame(output)
    if len(a) != len(primary) * len(cfg.landmarks_min):
        raise AssertionError("Landmark row-count mismatch")
    return a


def summarize_landmarks(landmarks: pd.DataFrame) -> pd.DataFrame:
    """Condition on being alive at landmark. Never pool risk sets across horizons."""
    clean = landmarks.loc[landmarks.observation_status == "PRESENT"].copy()
    metrics = ["signed_return_bp", "progress_to_target", "max_favorable_bp", "max_adverse_bp",
               "flow_alignment", "opposing_aggressor_share", "quote_activity_vs_past30",
               "level_favorable_close_share", "zone_overlap_minute_fraction", "target_over_sigma60"]
    out = []
    for (sym, h, reason), z in clean.groupby(["symbol", "landmark_min", "reason_final"]):
        record = {"symbol": sym, "landmark_min": int(h), "eventual_outcome": reason, "n_at_risk": int(len(z))}
        for k in metrics:
            record[f"median_{k}"] = float(z[k].median())
        out.append(record)
    return pd.DataFrame(out)


def cumulative_competing_exits(primary: pd.DataFrame, horizons=(5, 15, 30, 60, 120, 240)) -> pd.DataFrame:
    """Cumulative exit counts / original 452 signals; no timeouts forced into wins or losses."""
    out = []
    for sym in ("BTCUSDT", "ETHUSDT", "ALL"):
        p = primary if sym == "ALL" else primary[primary.symbol == sym]
        total = len(p)
        for h in horizons:
            # Use exact time from first assumed entry; timeout may fall seconds after 240m.
            done = p.loc[p.reason.isin(CLOSED) & p.time_to_exit_min.le(h)]
            record = {"symbol": sym, "horizon_min": int(h), "all_signals": int(total),
                      "entered": int(p.entry_time_ms.notna().sum()), "completed_by_h": int(len(done))}
            for outcome in CLOSED:
                n = int((done.reason == outcome).sum())
                record[f"n_{outcome.lower()}"] = n
                record[f"fraction_original_{outcome.lower()}"] = n / total if total else np.nan
            out.append(record)
    return pd.DataFrame(out)


def run(repo: Path, futures_root: Path, output: Path, cfg: Config = Config()) -> dict:
    base = repo / "notebooks" / "Liquidity_Absorption"
    contract = json.loads((base / "03d" / "03d_contract.json").read_text(encoding="utf-8"))
    if contract.get("version") != "LA-A03d-0.1.0" or contract.get("source_events") != 452:
        raise ValueError("Unexpected / changed A03d source policy")
    data = pd.read_csv(base / "03d" / "03d_risk_execution.csv", low_memory=False)
    primary = validate_primary(data, cfg)
    minutes = {s: load_minute_bars(Path(futures_root) / f"{s}_um_1m.parquet")
               for s in ("BTCUSDT", "ETHUSDT")}
    landmarks = build_landmarks(primary, minutes, cfg)
    summary = summarize_landmarks(landmarks)
    exits = cumulative_competing_exits(primary)
    output.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output / "04_primary_event_outcomes.csv", index=False)
    landmarks.to_csv(output / "04_landmark_panel.csv", index=False)
    summary.to_csv(output / "04_landmark_summary.csv", index=False)
    exits.to_csv(output / "04_competing_exits.csv", index=False)
    # Fixed design, descriptive only: no crossvalidated classifier, no supervised threshold optimization.
    _plot(summary, exits, output)
    report = {"version": VERSION, "parent": "LA-A03d-0.1.0", "configuration": asdict(cfg),
              "original_candidates": len(primary), "closed_primary": int(primary.reason.isin(CLOSED).sum()),
              "censored_primary": int(primary.reason.astype(str).str.startswith("CENSORED").sum()),
              "landmarks": len(landmarks), "usable_risk_set_landmarks": int((landmarks.observation_status=="PRESENT").sum()),
              "missing_minute_windows": int((landmarks.observation_status=="INCOMPLETE_MINUTE_WINDOW").sum()),
              "status": "EXPLORATORY_OUTCOME_CONDITIONED_NOT_CAUSAL_NOT_PREDICTIVE"}
    (output / "04_decision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _plot(summary: pd.DataFrame, exits: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    all_out = exits[exits.symbol == "ALL"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for reason in CLOSED:
        ax.plot(all_out.horizon_min, all_out[f"fraction_original_{reason.lower()}"] * 100, marker="o", label=reason)
    ax.set(xlabel="Minutes from assumed first entry", ylabel="% of 452 original signals", title="Competing cumulative exit outcomes (no censor adjustment)")
    ax.legend();fig.tight_layout();fig.savefig(out / "04_competing_exits.png", dpi=150);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, metric in enumerate(("median_signed_return_bp", "median_flow_alignment")):
        for reason in CLOSED:
            z = summary.loc[(summary.symbol == "BTCUSDT") & (summary.eventual_outcome == reason)]
            w = summary.loc[(summary.symbol == "ETHUSDT") & (summary.eventual_outcome == reason)]
            # Across symbols, plot distinct lines; cohort sizes change at each checkpoint.
            for sym, sub in (("BTC", z), ("ETH", w)):
                axes[i].plot(sub.landmark_min, sub[metric], marker="o", label=f"{sym} {reason}")
        axes[i].set(xlabel="Minutes after signal close", ylabel=metric.replace("median_", ""),
                    title="Surviving trades only: eventual outcome")
        axes[i].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out / "04_landmark_features.png", dpi=150);plt.close(fig)


def main(argv=None):
    from src.data_engineering.modeling.liquidity_absorption_stage_a import repo_root, binance_root
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path)
    p.add_argument("--futures-root", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    repo = a.repo or repo_root()
    root = a.futures_root or binance_root(repo)
    dst = a.out or repo / "notebooks" / "Liquidity_Absorption" / "04"
    print(json.dumps(run(repo, root, dst), indent=2))


if __name__ == "__main__":
    main()
