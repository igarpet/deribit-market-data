"""
Breakout Trend branch, BT 01: a third-party hourly breakout strategy
(index-futures origin, described in a YouTube review) re-implemented exactly
and tested on crypto perpetuals and on long ATM options.

Strategy (BT 001 §2; hourly quantities use COMPLETED hours only)
---------------------------------------------------------------
Hourly bar h ends at HH:00 and aggregates the twelve 5-minute bars ending
HH:05 .. (HH+1):00. At a 5-minute decision time t the last usable hour is the
latest h with end(h) <= t.

    M_h  = SMA_100(C)                         anchor
    A_h  = SMA_100(H - L)                      hourly candle size (length: assumption)
    U_h  = M_h + 1.5 A_h ,  L_h = M_h - 1.5 A_h
    S_t  = 0.5 P_t + 0.5 SMA_3(C)_h            P_t = close of the 5-minute bar ending at t
    flat:   S_t > U_h -> long ; S_t < L_h -> short          (fill at P_t)
    stop:   P_t moves 1.5 A_entry against the entry -> close and open the
            opposite side at P_t with fresh levels (stop-and-reverse)
    trail:  once P_t has been >= 1.0 A_entry in favour, exit to flat when P_t
            crosses SMA_14(C)_h against the position
    re-entry after a trailing exit (assumption, configurable): the same side
            needs S_t back inside the band first ('cross'); 'state' re-enters
            immediately.

Perpetual leg: 1 unit of underlying per trade, return on entry notional,
taker fee + slippage per side, funding ignored (BT 001 §3).

Options leg: on every perp entry (including flips) buy the ATM call (long) or
put (short) of the first daily expiry (08:00 UTC) at least 24 h away;
sell it when the perp position closes; if it expires first, settle at
intrinsic and roll into the next one. Size: delta-matched to the perp
(q = 1/|Delta| at purchase) by default.
  * model-priced over the full history: Black-76, sigma = k * EWMA realized
    volatility at purchase, held fixed for the option's life (no vega P&L),
    bought at sigma(1+h), sold at sigma(1-h), h = relative IV half-spread by
    TTE bucket; k and h calibrated on the collector's real quotes.
  * real-quote check on the collector window: ask in, bid out, exchange fee.

Units: everything is a return on 1 unit of underlying notional at entry,
summed (not compounded) -> equity in % of notional.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import option_pricing as op


VERSION = "0.1.0"

# Costs (Deribit published schedule as understood at implementation; Ivan to confirm)
PERP_TAKER_FEE = 0.0005      # 0.05% of notional per side
PERP_SLIPPAGE = 0.0001       # 1 bp per side
OPT_FEE_RATE = 0.0003        # 0.03% of underlying per option
OPT_FEE_CAP = 0.125          # capped at 12.5% of the option price
# Daily-option delivery fee assumed zero (as in VRP 02).

# Windows
DEV_START = pd.Timestamp("2024-09-01", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")

# Options leg
EXPIRY_HOUR_UTC = 8
MIN_OPTION_TTE_H = 24
EWMA_HALFLIFE_BARS = 288      # 24 h of 5-minute bars
BARS_PER_YEAR = 365.0 * 288.0
YEAR_H = 365.0 * 24.0
TTE_BUCKETS_H = (0.0, 6.0, 24.0, 48.0)
TTE_LABELS = ("0-6h", "6-24h", "24-48h")
ATM_LOG_MONEYNESS = 0.03
CALIB_MIN_TTE_H = 12          # k uses ATM quotes with TTE >= 12 h (0DTE grid from 2026-09-19, 1DTE from 2026-10-06)
REAL_ENTRY_WAIT_MIN = 10
REAL_EXIT_WAIT_MIN = 30

# Null and decision
N_NULL = 200
NULL_SEED = 1201
NULL_ALPHA = 0.05


@dataclass(frozen=True)
class Params:
    anchor_h: int = 100
    band_k: float = 1.5
    range_h: int = 100           # ASSUMPTION: length of the mean high-low
    fast_h: int = 3
    w_price: float = 0.5
    stop_k: float = 1.5
    trail_trigger_k: float = 1.0
    trail_h: int = 14
    reentry: str = "cross"       # 'cross' or 'state'
    min_bars_per_hour: int = 10
    max_hour_age_h: float = 2.0  # hourly state older than this at t -> no decisions


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

def ohlc5_from_um_1m(df1: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """5-minute OHLC from Binance USD-M perpetual 1-minute klines (BT 002 source change).
    Input: open_time (bar OPEN, UTC), open, high, low, close[, volume]. Output in the
    download_ohlc_history format: state_time = 5-minute bar END (UTC), source, symbol, OHLC, volume.
    A 5-minute bar is kept only if all five of its minutes are present."""
    x = df1.copy()
    x["open_time"] = pd.to_datetime(x["open_time"], utc=True)
    x = x.drop_duplicates("open_time").set_index("open_time").sort_index()
    if "volume" not in x:
        x["volume"] = np.nan
    g = x.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    n = x["close"].resample("5min", label="left", closed="left").count()
    g = g[n.reindex(g.index).fillna(0).to_numpy() == 5].dropna(subset=["close"])
    out = g.reset_index().rename(columns={"open_time": "state_time"})
    out["state_time"] = out["state_time"] + pd.Timedelta(minutes=5)
    out["source"], out["symbol"] = "binance_um_perp", symbol
    return out[["state_time", "source", "symbol", "open", "high", "low", "close", "volume"]]


def clean_5m(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated 5-minute OHLC with state_time = bar END (UTC)."""
    x = df.copy()
    x["state_time"] = pd.to_datetime(x["state_time"], utc=True)
    for c in ("open", "high", "low", "close"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["high", "low", "close"])
    x = x[(x["close"] > 0) & (x["high"] >= x["low"])]
    x = x.drop_duplicates("state_time", keep="last").sort_values("state_time").reset_index(drop=True)
    return x[["state_time", "open", "high", "low", "close"]]


def hourly_bars(df5: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """Completed hourly bars indexed by hour END, with the strategy indicators.

    A 5-minute bar ending at t belongs to the hour ending ceil(t, 1h): the bar
    ending 10:05 is in the hour ending 11:00, the bar ending 11:00 too. An
    hour is kept only if it has >= min_bars_per_hour bars and its end is not
    after the last available bar (so the trailing partial hour is dropped)."""
    x = df5.set_index("state_time")
    hour_end = x.index.ceil("h")
    g = x.groupby(hour_end).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"),
                                n=("close", "size"))
    g = g[(g["n"] >= p.min_bars_per_hour) & (g.index <= x.index.max())]
    if g.empty:
        return g
    full = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
    g = g.reindex(full)
    g.index.name = "hour_end"

    def sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n, min_periods=max(1, int(math.ceil(0.9 * n)))).mean()

    g["M"] = sma(g["close"], p.anchor_h)
    g["A"] = sma(g["high"] - g["low"], p.range_h)
    g["F3"] = sma(g["close"], p.fast_h)
    g["T14"] = sma(g["close"], p.trail_h)
    g["U"] = g["M"] + p.band_k * g["A"]
    g["L"] = g["M"] - p.band_k * g["A"]
    return g


def decision_frame(df5: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """5-minute decision grid with the last completed hour's indicators."""
    b = clean_5m(df5)
    h = hourly_bars(b, p)
    hh = h[["M", "A", "F3", "T14", "U", "L"]].reset_index()
    hh["hour_end"] = hh["hour_end"].astype(b["state_time"].dtype)
    out = pd.merge_asof(b, hh, left_on="state_time", right_on="hour_end", direction="backward",
                        allow_exact_matches=True)
    age_h = (out["state_time"] - out["hour_end"]).dt.total_seconds() / 3600.0
    out["S"] = p.w_price * out["close"] + (1.0 - p.w_price) * out["F3"]
    out["valid"] = (out[["U", "L", "A", "T14", "S"]].notna().all(axis=1)
                    & (out["A"] > 0) & (age_h <= p.max_hour_age_h))
    return out


# --------------------------------------------------------------------------
# Strategy state machine
# --------------------------------------------------------------------------

TRADE_COLS = ["i_entry", "i_exit", "dir", "entry_px", "exit_px", "exit_reason", "A_entry", "entry_kind"]


def run_strategy(fr: pd.DataFrame, p: Params = Params(), mode: str = "signal",
                 p_entry: float = 0.0, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Walk the 5-minute grid. mode='signal' uses the breakout entry;
    mode='random' replaces the entry from flat by a coin (probability p_entry
    per valid flat bar, side 50/50); exits are identical in both modes.
    The last open trade is closed at the last bar with reason 'end'."""
    P = fr["close"].to_numpy(float).tolist()
    S = fr["S"].to_numpy(float).tolist()
    U = fr["U"].to_numpy(float).tolist()
    L = fr["L"].to_numpy(float).tolist()
    A = fr["A"].to_numpy(float).tolist()
    T14 = fr["T14"].to_numpy(float).tolist()
    V = fr["valid"].to_numpy(bool).tolist()
    n = len(P)
    if mode == "random":
        rng = rng or np.random.default_rng(NULL_SEED)
        draws = rng.random(n).tolist()
        sides = rng.random(n).tolist()
    stop_k, trig_k, state_mode = p.stop_k, p.trail_trigger_k, p.reentry == "state"
    trades = []
    pos, ie, E, Ae, armed, kind = 0, -1, 0.0, 0.0, False, ""
    ok_long, ok_short = True, True
    for i in range(n):
        if not V[i]:
            continue
        px = P[i]
        if pos != 0:
            if (pos == 1 and px <= E - stop_k * Ae) or (pos == -1 and px >= E + stop_k * Ae):
                trades.append((ie, i, pos, E, px, "stop_reverse", Ae, kind))
                pos, ie, E, Ae, armed, kind = -pos, i, px, A[i], False, "reverse"
                continue
            if pos * (px - E) >= trig_k * Ae:
                armed = True
            if armed and ((pos == 1 and px < T14[i]) or (pos == -1 and px > T14[i])):
                trades.append((ie, i, pos, E, px, "trail", Ae, kind))
                if pos == 1:
                    ok_long = False
                else:
                    ok_short = False
                pos = 0
            continue
        if not ok_long and S[i] <= U[i]:
            ok_long = True
        if not ok_short and S[i] >= L[i]:
            ok_short = True
        go = 0
        if mode == "signal":
            if S[i] > U[i] and (ok_long or state_mode):
                go = 1
            elif S[i] < L[i] and (ok_short or state_mode):
                go = -1
        elif draws[i] < p_entry:
            go = 1 if sides[i] < 0.5 else -1
        if go:
            pos, ie, E, Ae, armed, kind = go, i, px, A[i], False, "breakout" if mode == "signal" else "random"
    if pos != 0:
        trades.append((ie, n - 1, pos, E, P[n - 1], "end", Ae, kind))
    t = pd.DataFrame(trades, columns=TRADE_COLS)
    if len(t):
        times = fr["state_time"].to_numpy()
        t["entry_time"] = pd.to_datetime(times[t["i_entry"].to_numpy()], utc=True)
        t["exit_time"] = pd.to_datetime(times[t["i_exit"].to_numpy()], utc=True)
    else:
        t["entry_time"] = pd.Series(dtype="datetime64[ns, UTC]")
        t["exit_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    return t


def flat_entry_rate(fr: pd.DataFrame, trades: pd.DataFrame) -> float:
    """Entries from flat per valid flat bar of the signal run (null p_entry)."""
    n_valid = int(fr["valid"].sum())
    in_pos = np.zeros(len(fr), bool)
    for a, b in zip(trades["i_entry"], trades["i_exit"]):
        in_pos[a:b] = True
    flat_bars = int((fr["valid"].to_numpy() & ~in_pos).sum())
    n_from_flat = int((trades["entry_kind"] != "reverse").sum())
    return n_from_flat / flat_bars if flat_bars and n_valid else 0.0


# --------------------------------------------------------------------------
# Perpetual leg
# --------------------------------------------------------------------------

def perp_cost_side(fee: float = PERP_TAKER_FEE, slip: float = PERP_SLIPPAGE) -> float:
    return fee + slip


def perp_returns(trades: pd.DataFrame, fee: float = PERP_TAKER_FEE, slip: float = PERP_SLIPPAGE) -> pd.DataFrame:
    t = trades.copy()
    c = perp_cost_side(fee, slip)
    t["perp_gross"] = t["dir"] * (t["exit_px"] / t["entry_px"] - 1.0)
    t["perp_net"] = t["perp_gross"] - 2.0 * c
    return t


def perp_equity(fr: pd.DataFrame, trades: pd.DataFrame, fee: float = PERP_TAKER_FEE,
                slip: float = PERP_SLIPPAGE) -> np.ndarray:
    """Mark-to-market equity per 5-minute bar (sum of returns on notional).
    Open trade at bar j: dir (P_j/E - 1) - entry cost; realized at its exit bar."""
    n = len(fr)
    P = fr["close"].to_numpy(float)
    c = perp_cost_side(fee, slip)
    realized = np.zeros(n)
    unreal = np.zeros(n)
    for ie, ix, d, E, X in zip(trades["i_entry"], trades["i_exit"], trades["dir"],
                               trades["entry_px"], trades["exit_px"]):
        realized[ix] += d * (X / E - 1.0) - 2.0 * c
        if ix > ie:
            unreal[ie:ix] = d * (P[ie:ix] / E - 1.0) - c
    return np.cumsum(realized) + unreal


# --------------------------------------------------------------------------
# Options leg
# --------------------------------------------------------------------------

def next_expiry(t: pd.Timestamp, min_h: float = MIN_OPTION_TTE_H) -> pd.Timestamp:
    """First 08:00 UTC at least min_h hours after t."""
    x = t + pd.Timedelta(hours=min_h)
    e = x.normalize() + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
    return e if e >= x else e + pd.Timedelta(days=1)


def ewma_vol(fr: pd.DataFrame, halflife: int = EWMA_HALFLIFE_BARS) -> np.ndarray:
    """Annualized EWMA volatility of 5-minute log returns, using bars up to t."""
    r = np.log(fr["close"]).diff()
    gap = fr["state_time"].diff().dt.total_seconds().fillna(300.0) / 300.0
    r2 = (r ** 2 / gap.clip(lower=1.0)).fillna(0.0)          # per-5m variance across gaps
    v = r2.ewm(halflife=halflife, adjust=False).mean().to_numpy()
    return np.sqrt(v * BARS_PER_YEAR)


@dataclass(frozen=True)
class OptionModel:
    k: float                       # IV / EWMA realized vol
    h: tuple = (0.03, 0.03, 0.03)  # relative IV half-spread by TTE bucket
    sizing: str = "delta"          # 'delta' (q = 1/|Delta|) or 'unit' (q = 1)

    def half_spread(self, tte_h: np.ndarray) -> np.ndarray:
        idx = np.clip(np.searchsorted(TTE_BUCKETS_H[1:-1], np.asarray(tte_h, float), side="right"), 0, 2)
        return np.asarray(self.h, float)[idx]


def option_fee(price_usd: np.ndarray, F: np.ndarray) -> np.ndarray:
    return np.minimum(OPT_FEE_RATE * np.asarray(F, float), OPT_FEE_CAP * np.asarray(price_usd, float))


def option_pieces(trades: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    """Split each perp trade into option holding pieces (expiry rolls)."""
    times = pd.DatetimeIndex(fr["state_time"])
    rows = []
    for tid, tr in enumerate(trades.itertuples(index=False)):
        start_i, start_t, k = tr.i_entry, tr.entry_time, 0
        while True:
            exp = next_expiry(start_t)
            if tr.exit_time <= exp:
                rows.append((tid, k, start_i, tr.i_exit, tr.dir, exp, "close"))
                break
            j = int(times.searchsorted(exp, side="left"))
            if j >= tr.i_exit or j >= len(times):          # expiry bar missing -> close with the trade
                rows.append((tid, k, start_i, tr.i_exit, tr.dir, exp, "close"))
                break
            rows.append((tid, k, start_i, j, tr.dir, exp, "expiry"))
            start_i, start_t, k = j, times[j], k + 1
    return pd.DataFrame(rows, columns=["trade_id", "piece", "i_start", "i_end", "dir", "expiry", "end_kind"])


def model_option_leg(fr: pd.DataFrame, trades: pd.DataFrame, model: OptionModel,
                     vol: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Model-priced option pieces and mark-to-market equity per bar."""
    n = len(fr)
    P = fr["close"].to_numpy(float)
    tt = fr["state_time"]
    tsec = (tt - tt.iloc[0]).dt.total_seconds().to_numpy() if n else np.zeros(0)
    vol = ewma_vol(fr) if vol is None else vol
    pcs = option_pieces(trades, fr)
    realized = np.zeros(n)
    unreal = np.zeros(n)
    out = []
    for pc in pcs.itertuples(index=False):
        a, b = pc.i_start, pc.i_end
        is_call = pc.dir == 1
        F0, K = P[a], P[a]
        exp_s = (pc.expiry - tt.iloc[0]).total_seconds()
        T = np.maximum(exp_s - tsec[a:b + 1], 0.0) / (YEAR_H * 3600.0)
        sig = model.k * vol[a]
        T0h = T[0] * YEAR_H
        buy = float(op.black76_usd(F0, K, T[0], sig * (1 + model.half_spread(T0h)), is_call))
        fee_in = float(option_fee(buy, F0))
        q = 1.0
        if model.sizing == "delta":
            dl = abs(float(op.black76_delta(F0, K, T[0], sig, is_call)))
            q = 1.0 / dl if dl > 1e-3 else 0.0
        if pc.end_kind == "expiry":
            sell = max(P[b] - K, 0.0) if is_call else max(K - P[b], 0.0)
            fee_out = 0.0
        else:
            Tb = T[-1]
            sell = float(op.black76_usd(P[b], K, Tb, sig * (1 - model.half_spread(Tb * YEAR_H)), is_call))
            fee_out = float(option_fee(sell, P[b]))
        r = q * (sell - fee_out - buy - fee_in) / F0
        realized[b] += r
        if b > a:
            mid = op.black76_usd(P[a:b], K, T[:-1], sig, is_call)
            unreal[a:b] = q * (mid - buy - fee_in) / F0
        out.append((pc.trade_id, pc.piece, a, b, pc.dir, pc.expiry, pc.end_kind, F0, K, T0h, sig, q,
                    buy, fee_in, sell, fee_out, r, q * buy / F0))
    cols = ["trade_id", "piece", "i_start", "i_end", "dir", "expiry", "end_kind", "F0", "K", "tte_h",
            "sigma", "q", "buy_usd", "fee_in", "sell_usd", "fee_out", "opt_ret", "premium_share"]
    return pd.DataFrame(out, columns=cols), np.cumsum(realized) + unreal


# --------------------------------------------------------------------------
# Real quotes (collector window)
# --------------------------------------------------------------------------

QUOTE_COLS = ("state_time", "instrument_name", "expiry_time", "strike", "is_call", "F",
              "bid_usd", "ask_usd", "bid_iv", "ask_iv", "delta")


def load_option_quotes(data_root, underlying: str, max_log_moneyness: float = 0.08) -> tuple[pd.DataFrame, dict]:
    """Near-ATM quotes of one underlying from the collector's compacted grids
    (0DTE daily/option_grid and 1DTE daily/option_grid_1dte), USD-converted,
    primary family only (the product_group with the most rows)."""
    import duckdb
    from pathlib import Path
    root = Path(data_root)
    globs = [g for g in (root / "daily" / "option_grid", root / "daily" / "option_grid_1dte")
             if any(g.glob("expiry=*/option_grid.parquet"))]
    if not globs:
        return pd.DataFrame(columns=list(QUOTE_COLS)), {"status": "no_option_files"}
    parts = []
    con = duckdb.connect()
    try:
        for g in globs:
            src = (g / "expiry=*" / "option_grid.parquet").as_posix()
            cols = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet('{src}', union_by_name=true)")
                       .df()["column_name"])
            exp = ("to_timestamp(expiration_timestamp / 1000.0)" if "expiration_timestamp" in cols
                   else "CAST(expiration_time AS TIMESTAMPTZ)")
            t = "scheduled_snapshot_time" if "scheduled_snapshot_time" in cols else "snapshot_time"
            parts.append(con.execute(f"""
                SELECT {t} AS state_time, instrument_name, {exp} AS expiry_time, product_group,
                       instrument_type, CAST(strike AS DOUBLE) AS strike, option_type,
                       CAST(underlying_price AS DOUBLE) AS F,
                       CAST(best_bid_price AS DOUBLE) AS bid, CAST(best_ask_price AS DOUBLE) AS ask,
                       CAST(bid_iv AS DOUBLE) AS bid_iv, CAST(ask_iv AS DOUBLE) AS ask_iv,
                       CAST(delta AS DOUBLE) AS delta, '{g.name}' AS grid
                FROM read_parquet('{src}', union_by_name=true)
                WHERE underlying = '{underlying}' AND underlying_price > 0 AND strike > 0
                  AND abs(ln(strike / underlying_price)) < {max_log_moneyness}
            """).df())
    finally:
        con.close()
    q = pd.concat(parts, ignore_index=True)
    if q.empty:
        return pd.DataFrame(columns=list(QUOTE_COLS)), {"status": "no_rows"}
    fam = q["product_group"].astype(str).value_counts()
    q = q[q["product_group"].astype(str) == fam.index[0]].copy()
    q["state_time"] = pd.to_datetime(q["state_time"], utc=True).dt.round("5min")
    q["expiry_time"] = pd.to_datetime(q["expiry_time"], utc=True).dt.round("5min")
    inv = op.is_inverse(q["instrument_type"])
    q["bid_usd"] = np.where(q["bid"] > 0, op.to_usd(q["bid"].to_numpy(float), q["F"].to_numpy(float), inv), 0.0)
    q["ask_usd"] = np.where(q["ask"] > 0, op.to_usd(q["ask"].to_numpy(float), q["F"].to_numpy(float), inv), 0.0)
    q["is_call"] = q["option_type"].astype(str).str.lower().str[0].eq("c")
    q = q.drop_duplicates(["state_time", "instrument_name"], keep="last")
    info = {"status": "ok", "family": str(fam.index[0]), "families": fam.to_dict(), "inverse": bool(inv.any()),
            "rows": int(len(q)), "first": str(q["state_time"].min()), "last": str(q["state_time"].max()),
            "by_grid": q["grid"].value_counts().to_dict()}
    return q[list(QUOTE_COLS)].sort_values("state_time").reset_index(drop=True), info


def calibrate_option_model(quotes: pd.DataFrame, fr: pd.DataFrame, vol: np.ndarray | None = None,
                           sizing: str = "delta") -> tuple[OptionModel, dict]:
    """k = median(ATM mid IV at TTE >= CALIB_MIN_TTE_H / EWMA vol); h by TTE bucket =
    median((ask_iv - bid_iv) / (ask_iv + bid_iv)) of near-ATM two-sided quotes."""
    q = quotes.copy()
    q = q[(q["bid_iv"] > 0) & (q["ask_iv"] >= q["bid_iv"]) & (q["F"] > 0)]
    q["tte_h"] = (q["expiry_time"] - q["state_time"]).dt.total_seconds() / 3600.0
    q["lm"] = np.abs(np.log(q["strike"] / q["F"]))
    q = q[(q["lm"] < ATM_LOG_MONEYNESS) & (q["tte_h"] > 0) & (q["tte_h"] <= TTE_BUCKETS_H[-1])]
    q["rel"] = (q["ask_iv"] - q["bid_iv"]) / (q["ask_iv"] + q["bid_iv"])
    q["bucket"] = pd.cut(q["tte_h"], TTE_BUCKETS_H, labels=TTE_LABELS, include_lowest=True)
    h = tuple(float(q.loc[q["bucket"] == b, "rel"].median()) if (q["bucket"] == b).any() else np.nan
              for b in TTE_LABELS)
    vol = ewma_vol(fr) if vol is None else vol
    vs = pd.Series(vol, index=pd.DatetimeIndex(fr["state_time"]))
    atm = q[(q["tte_h"] >= CALIB_MIN_TTE_H) & (q["state_time"].dt.minute == 0)]
    atm = atm.sort_values(["lm", "tte_h"], ascending=[True, False]).drop_duplicates("state_time")
    rv = vs.reindex(pd.DatetimeIndex(atm["state_time"])).to_numpy()
    ratio = (0.5 * (atm["bid_iv"] + atm["ask_iv"]) / 100.0).to_numpy() / rv
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    k = float(np.median(ratio)) if len(ratio) else np.nan
    fill = np.nanmedian(h) if np.isfinite(h).any() else np.nan
    h = tuple(fill if not np.isfinite(x) else x for x in h)
    info = {"k": k, "k_iqr": [float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))] if len(ratio) else None,
            "n_k_obs": int(len(ratio)), "h": list(h), "n_spread_obs": int(len(q)),
            "first": str(q["state_time"].min()) if len(q) else None, "last": str(q["state_time"].max()) if len(q) else None}
    return OptionModel(k=k, h=h, sizing=sizing), info


def real_option_leg(fr: pd.DataFrame, trades: pd.DataFrame, quotes: pd.DataFrame,
                    sizing: str = "delta") -> pd.DataFrame:
    """Executable P&L of the option pieces fully inside the quote window.
    Entry: ask of the nearest-forward strike of the target expiry (first
    two-sided snapshot within REAL_ENTRY_WAIT_MIN). Exit: first bid > 0 of the
    same instrument within REAL_EXIT_WAIT_MIN, else intrinsic (flagged).
    Expiry: intrinsic at the 08:00 bar close (delivery proxy)."""
    if quotes.empty or trades.empty:
        return pd.DataFrame()
    q = quotes.sort_values("state_time")
    q0, q1 = q["state_time"].min(), q["state_time"].max()
    P = fr["close"].to_numpy(float)
    times = fr["state_time"].to_numpy()
    by_inst = {k: g.set_index("state_time") for k, g in q.groupby("instrument_name")}
    pcs = option_pieces(trades, fr)
    rows = []
    for pc in pcs.itertuples(index=False):
        t0 = pd.Timestamp(times[pc.i_start])
        t1 = pd.Timestamp(times[pc.i_end])
        if t0 < q0 or t1 > q1:
            continue
        is_call = pc.dir == 1
        w = q[(q["state_time"] >= t0) & (q["state_time"] <= t0 + pd.Timedelta(minutes=REAL_ENTRY_WAIT_MIN))
              & (q["expiry_time"] == pc.expiry) & (q["is_call"] == is_call) & (q["ask_usd"] > 0)]
        if w.empty:
            rows.append((pc.trade_id, pc.piece, pc.i_start, pc.i_end, pc.dir, None, np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, "no_entry_quote"))
            continue
        first_t = w["state_time"].min()
        w = w[w["state_time"] == first_t]
        w = w.assign(lm=np.abs(np.log(w["strike"] / w["F"]))).sort_values(["lm", "strike"])
        e = w.iloc[0]
        F0, buy = float(e["F"]), float(e["ask_usd"])
        fee_in = float(option_fee(buy, F0))
        dl = abs(float(e["delta"])) if np.isfinite(e["delta"]) else np.nan
        qty = (1.0 / dl if dl > 1e-3 else 0.0) if sizing == "delta" else 1.0
        if pc.end_kind == "expiry":
            Fx = P[pc.i_end]
            sell, fee_out, flag = (max(Fx - e["strike"], 0.0) if is_call else max(e["strike"] - Fx, 0.0)), 0.0, "expiry"
        else:
            g = by_inst[e["instrument_name"]]
            g = g[(g.index >= t1) & (g.index <= t1 + pd.Timedelta(minutes=REAL_EXIT_WAIT_MIN)) & (g["bid_usd"] > 0)]
            if len(g):
                sell = float(g["bid_usd"].iloc[0])
                fee_out, flag = float(option_fee(sell, float(g["F"].iloc[0]))), "bid"
            else:
                Fx = P[pc.i_end]
                sell = max(Fx - e["strike"], 0.0) if is_call else max(e["strike"] - Fx, 0.0)
                fee_out, flag = 0.0, "no_bid_intrinsic"
        r = qty * (sell - fee_out - buy - fee_in) / F0
        rows.append((pc.trade_id, pc.piece, pc.i_start, pc.i_end, pc.dir, e["instrument_name"], float(e["strike"]),
                     F0, qty, buy, sell, r, flag))
    return pd.DataFrame(rows, columns=["trade_id", "piece", "i_start", "i_end", "dir", "instrument", "K", "F0",
                                       "q", "buy_usd", "sell_usd", "opt_ret_real", "exit_flag"])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def daily_equity(fr: pd.DataFrame, eq: np.ndarray) -> pd.Series:
    s = pd.Series(eq, index=pd.DatetimeIndex(fr["state_time"]))
    return s.resample("1D").last().ffill()


def period_metrics(eq_daily: pd.Series, start=None, end=None) -> dict:
    s = eq_daily
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index < end]
    if len(s) < 3:
        return {"pnl": np.nan, "sharpe": np.nan, "max_dd": np.nan, "days": int(len(s))}
    base = eq_daily[eq_daily.index < s.index[0]]
    s0 = float(base.iloc[-1]) if len(base) else 0.0
    path = pd.concat([pd.Series([s0]), s.reset_index(drop=True)])
    d = path.diff().dropna()
    sd = float(d.std(ddof=1))
    sharpe = float(d.mean() / sd * math.sqrt(365.0)) if sd > 0 else np.nan
    dd = float((path - path.cummax()).min())
    return {"pnl": float(path.iloc[-1] - s0), "sharpe": sharpe, "max_dd": dd, "days": int(len(s))}


def trade_metrics(t: pd.DataFrame, col: str, start=None, end=None) -> dict:
    x = t
    if start is not None:
        x = x[x["exit_time"] >= start]
    if end is not None:
        x = x[x["exit_time"] < end]
    r = x[col].dropna()
    if not len(r):
        return {"n_trades": 0}
    span_y = max((x["exit_time"].max() - x["entry_time"].min()).total_seconds() / (365.0 * 86400.0), 1e-9)
    return {"n_trades": int(len(r)), "trades_per_year": float(len(r) / span_y),
            "win_rate": float((r > 0).mean()), "avg_win": float(r[r > 0].mean()) if (r > 0).any() else np.nan,
            "avg_loss": float(r[r <= 0].mean()) if (r <= 0).any() else np.nan,
            "pnl_long": float(r[x.loc[r.index, "dir"] == 1].sum()),
            "pnl_short": float(r[x.loc[r.index, "dir"] == -1].sum())}


def buy_hold_daily(fr: pd.DataFrame) -> pd.Series:
    c = pd.Series(fr["close"].to_numpy(float), index=pd.DatetimeIndex(fr["state_time"]))
    c = c.resample("1D").last().ffill()
    return c / c.iloc[0] - 1.0


def time_in_market(fr: pd.DataFrame, trades: pd.DataFrame, start=None, end=None) -> float:
    m = np.zeros(len(fr), bool)
    for a, b in zip(trades["i_entry"], trades["i_exit"]):
        m[a:b] = True
    t = fr["state_time"]
    sel = fr["valid"].to_numpy().copy()
    if start is not None:
        sel &= (t >= start).to_numpy()
    if end is not None:
        sel &= (t < end).to_numpy()
    return float(m[sel].mean()) if sel.any() else np.nan


# --------------------------------------------------------------------------
# Coin-flip null
# --------------------------------------------------------------------------

def null_distribution(fr: pd.DataFrame, p: Params, p_entry: float, windows: dict,
                      n_runs: int = N_NULL, seed: int = NULL_SEED) -> pd.DataFrame:
    """Random entries from flat (probability matched to the signal run), the
    strategy's own exits; perpetual leg net of costs. One row per run x window."""
    ss = np.random.SeedSequence(seed)
    rows = []
    for run, child in enumerate(ss.spawn(n_runs)):
        t = run_strategy(fr, p, mode="random", p_entry=p_entry, rng=np.random.default_rng(child))
        ed = daily_equity(fr, perp_equity(fr, t))
        for name, (a, b) in windows.items():
            m = period_metrics(ed, a, b)
            rows.append({"run": run, "window": name, "pnl": m["pnl"], "sharpe": m["sharpe"]})
    return pd.DataFrame(rows)


def null_share(null: pd.DataFrame, window: str, real_sharpe: float) -> float:
    """Share of null runs with Sharpe >= the real Sharpe (one-sided p-value)."""
    x = null.loc[null["window"] == window, "sharpe"].dropna()
    if not len(x) or not np.isfinite(real_sharpe):
        return np.nan
    return float((x >= real_sharpe).mean())


def verdict(test_pnl: float, p_null: float, alpha: float = NULL_ALPHA) -> str:
    if not (np.isfinite(test_pnl) and np.isfinite(p_null)):
        return "INSUFFICIENT"
    return "EDGE_CANDIDATE" if (test_pnl > 0 and p_null < alpha) else "NO_EDGE"


def registration(p: Params = Params()) -> dict:
    return {"version": VERSION, "params": asdict(p), "perp_taker_fee": PERP_TAKER_FEE,
            "perp_slippage": PERP_SLIPPAGE, "funding": "ignored",
            "opt_fee_rate": OPT_FEE_RATE, "opt_fee_cap": OPT_FEE_CAP,
            "dev": [str(DEV_START), str(TEST_START)], "test_start": str(TEST_START),
            "option_min_tte_h": MIN_OPTION_TTE_H, "calib_min_tte_h": CALIB_MIN_TTE_H, "ewma_halflife_bars": EWMA_HALFLIFE_BARS,
            "n_null": N_NULL, "null_seed": NULL_SEED, "null_alpha": NULL_ALPHA,
            "decision": "per window: EDGE_CANDIDATE iff net perp P&L > 0 and coin-flip null share < alpha; "
                        "primary = pooled equal-weight portfolio of all assets on the TEST window; options reported, no decision"}


# --------------------------------------------------------------------------
# Synthetic data and self-test
# --------------------------------------------------------------------------

def synthetic_5m(n_days: int = 120, seed: int = 0, vol: float = 0.6, trend_blocks: bool = True,
                 start: str = "2025-01-01") -> pd.DataFrame:
    """Random-walk 5-minute OHLC (optionally with persistent drift blocks)."""
    rng = np.random.default_rng(seed)
    n = n_days * 288
    sig = vol / math.sqrt(BARS_PER_YEAR)
    mu = np.zeros(n)
    if trend_blocks:
        blk = 288 * 5
        for s in range(0, n, blk):
            mu[s:s + blk] = rng.choice([-1, 0, 1]) * 0.04 * sig
    sub = 5
    r = rng.normal(mu[:, None] / sub, sig / math.sqrt(sub), size=(n, sub))
    path = 100.0 * np.exp(np.cumsum(r.ravel()))
    path = path.reshape(n, sub)
    opn = np.r_[100.0, path[:-1, -1]]
    hi = np.maximum(path.max(1), opn)
    lo = np.minimum(path.min(1), opn)
    t = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=5), periods=n, freq="5min")
    return pd.DataFrame({"state_time": t, "open": opn, "high": hi, "low": lo, "close": path[:, -1]})


def synthetic_quotes(fr: pd.DataFrame, k: float = 1.1, rel_half: float = 0.04, start: str | None = None,
                     strike_step_rel: float = 0.01) -> pd.DataFrame:
    """Black-76 quotes (IV = k * EWMA vol, held at the snapshot) for the first
    two daily expiries at every bar after `start`."""
    vol = ewma_vol(fr)
    x = fr[["state_time", "close"]].assign(v=vol)
    if start is not None:
        x = x[x["state_time"] >= pd.Timestamp(start, tz="UTC")]
    rows = []
    for t, F, v in x.itertuples(index=False):
        e1 = t.normalize() + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
        if e1 <= t:
            e1 += pd.Timedelta(days=1)
        for exp in (e1, e1 + pd.Timedelta(days=1), e1 + pd.Timedelta(days=2)):
            T = (exp - t).total_seconds() / (YEAR_H * 3600.0)
            if T * YEAR_H > 48.0 + 1e-9:
                continue
            step = float(f"{strike_step_rel * F:.1g}")
            base = round(F / step) * step
            for K in (base - step, base, base + step):
                for c in (True, False):
                    s = k * v
                    rows.append((t, f"X-{exp:%d%b%y}-{K:g}-{'C' if c else 'P'}", exp, K, c, F,
                                 float(op.black76_usd(F, K, T, s * (1 - rel_half), c)),
                                 float(op.black76_usd(F, K, T, s * (1 + rel_half), c)),
                                 100 * s * (1 - rel_half), 100 * s * (1 + rel_half),
                                 float(op.black76_delta(F, K, T, s, c))))
    return pd.DataFrame(rows, columns=list(QUOTE_COLS))


def selftest() -> dict:
    out = {}
    p = Params()
    df = synthetic_5m(60, seed=3)
    fr = decision_frame(df, p)
    # 1. no lookahead: indicators at t do not change when bars after t are removed
    cut = fr["state_time"].iloc[len(fr) // 2]
    fr2 = decision_frame(df[df["state_time"] <= cut], p)
    a = fr.set_index("state_time").loc[fr2["state_time"], ["U", "L", "T14", "S"]].to_numpy()
    b = fr2[["U", "L", "T14", "S"]].to_numpy()
    out["no_lookahead"] = bool(np.allclose(a, b, equal_nan=True))
    # 2. trade list identical up to the cut (decisions are causal)
    t1 = run_strategy(fr, p)
    t2 = run_strategy(fr2, p)
    k = (t2["exit_reason"] != "end").to_numpy()
    out["causal_trades"] = bool(t1.iloc[:k.sum()][TRADE_COLS].equals(t2[k][TRADE_COLS]))
    # 3. accounting: final MTM equity equals the sum of net trade returns
    eq = perp_equity(fr, t1)
    out["equity_matches_trades"] = bool(abs(eq[-1] - perp_returns(t1)["perp_net"].sum()) < 1e-9)
    # 4. flips are consecutive (reverse entry at the stop bar)
    rv = t1.index[t1["exit_reason"] == "stop_reverse"]
    out["flip_consistent"] = bool(all(t1.loc[i + 1, "i_entry"] == t1.loc[i, "i_exit"]
                                      and t1.loc[i + 1, "dir"] == -t1.loc[i, "dir"] for i in rv if i + 1 < len(t1)))
    # 5. option model reproduces quotes when calibrated on its own synthetic book
    qs = synthetic_quotes(fr, k=1.2, rel_half=0.05, start=str(fr["state_time"].iloc[-288 * 6].date()))
    m, info = calibrate_option_model(qs, fr)
    out["calib_k"] = round(m.k, 3)
    out["calib_h"] = [round(x, 3) for x in m.h]
    return out
