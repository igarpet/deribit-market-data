"""LA10: all observable A02 level episodes, including untouched / single-touched controls.

Does not respecify A02 signals or infer actual intraminute touch timestamps.
Replay preserves the exact A02 price-only state machine and fails closed when
reconstructed signals disagree with saved A02. Research-only, no trade execution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_engineering.modeling.liquidity_absorption_stage_a import binance_root, repo_root
from src.data_engineering.modeling import liquidity_absorption_stage_a02 as original

VERSION = "LA-LEVEL-UNIVERSE-10-0.1.0"
STREAMS = (
    ("support", "prior_day", "prior_day_low", "prior_day_source_time"),
    ("resistance", "prior_day", "prior_day_high", "prior_day_source_time"),
    ("support", "pivot_5m", "pivot_support_price", "pivot_support_source_time"),
    ("resistance", "pivot_5m", "pivot_resistance_price", "pivot_resistance_source_time"),
)
MINUTE = pd.Timedelta(minutes=1)
HORIZONS = (5, 15, 30, 60, 90)


@dataclass(frozen=True)
class Config:
    primary_start: str = "2026-07-01"
    primary_end: str = "2026-09-30"
    horizons_min: tuple[int, ...] = HORIZONS


def _stamp(x):
    return pd.Timestamp(x).tz_convert("UTC") if pd.Timestamp(x).tzinfo else pd.Timestamp(x, tz="UTC")


def level_available(source_time, family, cfg: original.Config):
    source = _stamp(source_time)
    if family == "pivot_5m":
        return source + pd.Timedelta(minutes=5 * cfg.pivot_right_5m)
    if family == "prior_day":
        return source + pd.Timedelta(days=1)
    raise ValueError("Unknown A02 reference family")


def _new_episode(symbol, side, family, source, price, available, when, cfg):
    return dict(symbol=symbol, side=side, level_type=family,
                level_source_time=source, level_price=price, available_at=available,
                zone_halfwidth=price * cfg.zone_halfwidth_bp / 10000,
                first_observed_close=when, last_observed_close=when,
                first_valid_touch=None, first_signal=None, last_valid_touch=None,
                first_touch_confirmed_after_available=None,
                n_valid_touch_candles=0, n_second_touch_signals=0,
                n_invalidations=0, n_invalidations_after_first=0,
                n_expired_firsts=0, n_cooldown_touches=0,
                end_reason="UNSET", first_to_signal_minutes=np.nan)


def replay_level_episodes(x: pd.DataFrame, symbol: str, cfg: original.Config):
    """One chronological replay: observation episode accounting + exact A02 events.

    The level universe includes the same last-active reference in each of the
    four A02 streams. Unobserved/non-current historical pivots are not exposed.
    """
    refs = original.get_reference_levels(x, cfg)
    data = x.merge(refs.drop(columns="close_time"), on="open_time", validate="one_to_one")
    data = data.sort_values("close_time").reset_index(drop=True)
    states = {s[:2]: dict(identifier=None, first=None, away=False,
                          last_signal=None, episode=None, spell=None) for s in STREAMS}
    episodes, detections, spells = [], [], []
    start = _stamp(cfg.start)
    end = _stamp(cfg.end) + pd.Timedelta(days=1)

    def close_spell(state, reason, when):
        spell = state["spell"]
        if spell is None:
            return
        spell["end_reason"] = reason
        spell["end_time"] = when
        spell["duration_min"] = (when - spell["first_touch_close"]).total_seconds() / 60
        spells.append(spell)
        state["spell"] = None

    def close_episode(state, reason):
        if state["episode"] is not None:
            close_spell(state, reason, state["episode"]["last_observed_close"])
        ep = state["episode"]
        if ep is not None:
            ep["end_reason"] = reason
            episodes.append(ep)
        state["identifier"] = None
        state["first"] = None
        state["away"] = False
        state["episode"] = None

    for bar in data.to_dict("records"):
        when = bar["close_time"]
        if pd.isna(when):
            continue
        op, high, low, close = (float(bar[k]) for k in ("open", "high", "low", "close"))
        candle_range = max(high - low, 1e-12)
        low_wick = (min(op, close) - low) / candle_range
        high_wick = (high - max(op, close)) / candle_range
        for side, family, column, stamp_col in STREAMS:
            state = states[(side, family)]
            price, source = bar[column], bar[stamp_col]
            if pd.isna(price) or pd.isna(source):
                if state["episode"] is not None:
                    close_episode(state, "REFERENCE_EXPIRED_OR_MISSING")
                continue
            price = float(price)
            width = price * cfg.zone_halfwidth_bp / 10000
            if not np.isfinite(width) or width <= 0:
                if state["episode"] is not None:
                    close_episode(state, "INVALID_LEVEL_WIDTH")
                continue
            identifier = (source, price)
            if state["identifier"] != identifier:
                if state["episode"] is not None:
                    close_episode(state, "REPLACED_BY_NEW_REFERENCE")
                available = level_available(source, family, cfg)
                if when < available:
                    raise ValueError("Reference exposed before its confirmation clock")
                state["episode"] = _new_episode(symbol, side, family, source, price, available, when, cfg)
                state["identifier"] = identifier
            ep = state["episode"]
            ep["last_observed_close"] = when
            if side == "support":
                invalid_close = close < price - 2 * width
                departed = close >= price + cfg.leave_zone_multiple * width
                valid = (price - cfg.max_penetration_zone_widths * width <= low <= price + width
                         and close > price + width and low_wick >= cfg.wick_fraction_min)
            else:
                invalid_close = close > price + 2 * width
                departed = close <= price - cfg.leave_zone_multiple * width
                valid = (price - width <= high <= price + cfg.max_penetration_zone_widths * width
                         and close < price - width and high_wick >= cfg.wick_fraction_min)
            if invalid_close:
                if state["spell"] is not None:
                    close_spell(state, "INVALIDATED", when)
                ep["n_invalidations"] += 1
                if state["first"] is not None:
                    ep["n_invalidations_after_first"] += 1
                state.update(first=None, away=False)
                continue
            first = state["first"]
            if first is not None:
                elapsed = (when - first["close_time"]).total_seconds() / 60
                if elapsed > cfg.max_retest_gap_min:
                    close_spell(state, "RETEST_GAP_EXPIRED", first["close_time"] + pd.Timedelta(minutes=cfg.max_retest_gap_min))
                    ep["n_expired_firsts"] += 1
                    state.update(first=None, away=False)
                    first = None
            cooldown = (state["last_signal"] is not None and
                        (when - state["last_signal"]).total_seconds() / 60 < cfg.cooldown_after_event_min)
            if not valid or cooldown:
                if valid and cooldown:
                    ep["n_cooldown_touches"] += 1
                if first is not None and departed and when > first["close_time"]:
                    state["away"] = True
                continue
            ep["n_valid_touch_candles"] += 1
            ep["last_valid_touch"] = when
            if ep["first_valid_touch"] is None:
                ep["first_valid_touch"] = when
                ep["first_touch_confirmed_after_available"] = bool(when > ep["available_at"])
            if first is not None:
                elapsed = (when - first["close_time"]).total_seconds() / 60
                if elapsed >= cfg.min_touch_separation_min and state["away"] and start <= when < end:
                    ep["n_second_touch_signals"] += 1
                    if ep["first_signal"] is None:
                        ep["first_signal"] = when
                        ep["first_to_signal_minutes"] = elapsed
                    detections.append(dict(symbol=symbol,
                        side="long" if side == "support" else "short",
                        level_type=family, level_source_time=source,
                        level_price=price, first_touch_close=first["close_time"],
                        second_touch_close=when, signal_time_utc=when))
                    close_spell(state, "SECOND_TOUCH_SIGNAL", when)
                    state.update(first=None, away=False, last_signal=when)
                    continue
            if first is None:
                state["spell"] = dict(symbol=symbol, side=side, level_type=family,
                    level_source_time=source, level_price=price, available_at=ep["available_at"],
                    first_touch_close=when, first_touch_after_availability=bool(when > ep["available_at"]),
                    end_time=None, end_reason="UNSET", duration_min=np.nan)
                state.update(first={"close_time": when}, away=False)
            elif departed and when > first["close_time"]:
                state["away"] = True
    for st in states.values():
        if st["episode"] is not None:
            close_episode(st, "RIGHT_CENSORED_END_OF_ARCHIVE")
    e = pd.DataFrame(episodes)
    if e.empty:
        raise ValueError("No observable level episodes were reconstructed")
    e["n_observed_minute_closes"] = (e.last_observed_close - e.first_observed_close).dt.total_seconds() / 60 + 1
    e["observed_from_availability_min"] = (e.first_observed_close - e.available_at).dt.total_seconds() / 60
    e["first_valid_touch"] = pd.to_datetime(e["first_valid_touch"], utc=True)
    e["minutes_until_first_touch"] = (e.first_valid_touch - e.available_at).dt.total_seconds() / 60
    e["full_horizon_at_risk_minutes"] = (e.last_observed_close - e.available_at).dt.total_seconds() / 60
    e["classification"] = np.select([e.n_second_touch_signals.gt(0), e.n_valid_touch_candles.gt(0)],
                                     ["SECOND_TOUCH_SIGNAL", "FIRST_TOUCH_NO_SIGNAL"],
                                     default="NO_VALID_TOUCH")
    e = e.sort_values(["symbol","available_at","level_type","side"]).reset_index(drop=True)
    e.insert(0, "episode_id", [f"{symbol}-{i:06d}" for i in range(len(e))])
    spells_df = pd.DataFrame(spells, columns=["symbol","side","level_type","level_source_time",
        "level_price","available_at","first_touch_close","first_touch_after_availability",
        "end_time","end_reason","duration_min"])
    return e, pd.DataFrame(detections), spells_df


def replay_contract(detections: pd.DataFrame, frozen: pd.DataFrame) -> dict:
    """Require exact A02 signal membership/touch/source identity as release gate."""
    keys = ["symbol", "side", "level_type", "level_source_time", "level_price",
            "first_touch_close", "second_touch_close", "signal_time_utc"]
    if not set(keys).issubset(detections) or not set(keys).issubset(frozen):
        raise ValueError("A02 frozen/replay contract lacks identity fields")
    def rows(data):
        z = data[keys].copy()
        for t in ("level_source_time", "first_touch_close", "second_touch_close", "signal_time_utc"):
            z[t] = pd.to_datetime(z[t], utc=True, errors="raise").dt.as_unit("ms").astype("int64")
        z["level_price"] = pd.to_numeric(z["level_price"], errors="raise").round(8)
        return Counter(tuple(row) for row in z.itertuples(index=False, name=None))
    a, b = rows(detections), rows(frozen)
    missing = b - a
    extra = a - b
    if missing or extra:
        raise ValueError(f"A02 replay mismatch: missing_frozen={sum(missing.values())}, extra_replay={sum(extra.values())}; "
                         "do not interpret non-signal controls before fixing")
    return dict(exact_signal_replay=True, signals=int(sum(a.values())), frozen=int(sum(b.values())))


def incidence(episodes: pd.DataFrame, cfg: Config):
    """Availability-anchored risks; censored horizons are not counted as negatives."""
    p = episodes.copy()
    start = _stamp(cfg.primary_start)
    end = _stamp(cfg.primary_end) + pd.Timedelta(days=1)
    p["primary_newly_observable"] = p.available_at.ge(start) & p.available_at.lt(end)
    # The reference is visible to the bar-close detector only if 0..1 min delay.
    p["fully_observed_from_availability"] = p.observed_from_availability_min.between(0, 1)
    p["strict_first_touch"] = p.first_touch_confirmed_after_available.map(lambda x: bool(x) if pd.notna(x) else False).astype(bool)
    p["first_touch_may_precede_confirmation"] = p.first_valid_touch.eq(p.available_at)
    sample = p[p.primary_newly_observable & p.fully_observed_from_availability].copy()
    rows = []
    for keys, group in sample.groupby(["symbol", "level_type", "side"], dropna=False):
        for horizon in cfg.horizons_min:
            # This metric is for uninterrupted visible reference episodes;
            # exposure before replacement/expiry is right-censored.
            mature = group.full_horizon_at_risk_minutes.ge(horizon)
            first_in_h = group.minutes_until_first_touch.le(horizon)
            # An early first touch counts even if episode ended before horizon;
            # lack of first touch before episode ended is censored, not a failure.
            valid = first_in_h | mature
            eligible = group.loc[valid]
            n_first = int(first_in_h[valid].sum())
            rows.append(dict(symbol=keys[0],level_type=keys[1],side=keys[2],
                 horizon_min=horizon, n_original_new_episodes=len(group),
                 n_first_touch_known=int(valid.sum()), n_first_touch_by_h=n_first,
                 fraction_first_touch_by_h=n_first / valid.sum() if valid.any() else np.nan,
                 n_censored_before_h=int((~valid).sum()),
                 # Signals are outcomes of the detector, potentially after a
                 # prior first touch, not necessarily before this horizon.
                 n_episodes_with_any_signal=int(group.n_second_touch_signals.gt(0).sum())))
    return p, pd.DataFrame(rows)


def second_touch_incidence(spells: pd.DataFrame, cfg: Config):
    """Competing outcomes from each detector-recognized first-touch spell.

    Ending by invalidation / level replacement / expiry is observed as a
    competing terminal event, NOT a censored failure. Only a spell open when
    the archive ends is right-censored. No retrospective second-touch filter.
    """
    if spells.empty:
        return pd.DataFrame(columns=["symbol","level_type","side","horizon_min", "n_spells",
            "n_assessable","n_second_by_h","n_competing_by_h","n_right_censored_before_h",
            "observed_second_incidence_by_h"])
    p = spells.copy()
    start, end = _stamp(cfg.primary_start), _stamp(cfg.primary_end) + pd.Timedelta(days=1)
    p = p[p.first_touch_close.ge(start) & p.first_touch_close.lt(end) & p.first_touch_after_availability].copy()
    rows=[]
    for k, group in p.groupby(["symbol","level_type","side"]):
        for h in cfg.horizons_min:
            by_h=group.duration_min.le(h)
            second=by_h & group.end_reason.eq("SECOND_TOUCH_SIGNAL")
            right=by_h & group.end_reason.eq("RIGHT_CENSORED_END_OF_ARCHIVE")
            competing=by_h & ~group.end_reason.isin(("SECOND_TOUCH_SIGNAL","RIGHT_CENSORED_END_OF_ARCHIVE"))
            assessable=~right
            rows.append(dict(symbol=k[0],level_type=k[1],side=k[2],horizon_min=h,
                n_spells=len(group),n_assessable=int(assessable.sum()),
                n_second_by_h=int(second.sum()),n_competing_by_h=int(competing.sum()),
                n_right_censored_before_h=int(right.sum()),
                observed_second_incidence_by_h=float(second.sum()/assessable.sum()) if assessable.any() else np.nan))
    return pd.DataFrame(rows)


def report_by_episode(episodes: pd.DataFrame):
    p = episodes[episodes.primary_newly_observable & episodes.fully_observed_from_availability].copy()
    return (p.groupby(["symbol","level_type","side","classification"],dropna=False)
            .agg(n=("episode_id","size"),
                 median_observed_minutes=("n_observed_minute_closes","median"),
                 median_time_to_first_touch=("minutes_until_first_touch","median"),
                 median_touch_count=("n_valid_touch_candles","median"),
                 median_invalidations=("n_invalidations","median"))
            .reset_index())


def run(repo: Path, data_root: Path, out: Path, cfg: Config = Config()) -> dict:
    contract = json.loads((repo / "notebooks" / "Liquidity_Absorption" / "02" / "02_feature_contract.json").read_text(encoding="utf-8"))
    if contract.get("version") != "LA-A02-0.1.0":
        raise ValueError("Frozen A02 contract version changed")
    a02 = original.Config(**contract["config"])
    if (cfg.primary_start != a02.start or cfg.primary_end != a02.end
            or cfg.horizons_min != HORIZONS):
        raise ValueError("A10 time/threshold specification does not match frozen A02")
    frozen = pd.read_csv(repo / "notebooks" / "Liquidity_Absorption" / "02" / "02_price_only_candidates.csv")
    if len(frozen) != 452 or not {"symbol","side","level_type","level_source_time", "level_price",
                                          "first_touch_close","second_touch_close","signal_time_utc"}.issubset(frozen):
        raise ValueError("Frozen A02 452-event identity contract missing or changed")
    ep_list, signal_list, spell_list = [], [], []
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = Path(data_root) / f"{symbol}_um_1m.parquet"
        candles = pd.read_parquet(path, columns=["open_time","open","high","low","close","volume","quote_volume","taker_buy_quote_volume"])
        minutes = original.get_minutes(candles, a02)
        episodes, replay, spells = replay_level_episodes(minutes, symbol, a02)
        ep_list.append(episodes)
        signal_list.append(replay)
        spell_list.append(spells)
    episodes = pd.concat(ep_list, ignore_index=True)
    detected = pd.concat(signal_list, ignore_index=True)
    spells = pd.concat(spell_list, ignore_index=True)
    gate = replay_contract(detected, frozen)
    population, hazard = incidence(episodes, cfg)
    table = report_by_episode(population)
    second = second_touch_incidence(spells, cfg)
    out.mkdir(parents=True, exist_ok=True)
    population.to_csv(out / "10_level_episodes.csv", index=False)
    table.to_csv(out / "10_episodes_by_type.csv", index=False)
    hazard.to_csv(out / "10_first_touch_incidence.csv", index=False)
    spells.to_csv(out / "10_first_touch_spells.csv", index=False)
    second.to_csv(out / "10_second_touch_incidence.csv", index=False)
    detected.to_csv(out / "10_replayed_A02_signals.csv", index=False)
    selection = population[population.primary_newly_observable & population.fully_observed_from_availability]
    decision = {"version": VERSION, "status":"EXPLORATORY_SELECTED_VS_UNSELECTED_LEVEL_UNIVERSE_NOT_PREDICTIVE",
                "original_A02_signals":gate["frozen"],
                "replayed_signals":gate["signals"],
                "exact_signal_replay":gate["exact_signal_replay"],
                "level_episode_rows_all_including_warmup":len(population),
                "primary_new_level_episodes":len(selection),
                "primary_classification_counts":{k:int(v) for k,v in selection.classification.value_counts().items()},
                "primary_censored_at_end":int(selection.end_reason.eq("RIGHT_CENSORED_END_OF_ARCHIVE").sum()),
                "first_touch_spells_including_warmup":len(spells),
                "first_touch_same_close_as_availability":int(selection.first_touch_may_precede_confirmation.sum()),
                "flags":"Future second touch is an outcome; unselected control level episodes cannot be called failed trades.",
                "data_source":"Binance USD-M 1m candle, exact frozen A02 reference/selection rules; no tick/L2 or queue",
                "interpretation":"Observed reference episodes; first-touch incidence uses observed/competing censored denominators, not naive all-level probability.",
                "configuration":asdict(cfg)}
    (out / "10_decision.json").write_text(json.dumps(decision, indent=2)+"\n",encoding="utf-8")
    return decision


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path)
    p.add_argument("--binance-root",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args(argv)
    repo=args.repo or repo_root()
    root=args.binance_root or binance_root(repo)
    out=args.out or repo / "notebooks" / "Liquidity_Absorption" / "10"
    print(json.dumps(run(repo, root, out),indent=2))


if __name__ == "__main__":
    main()
