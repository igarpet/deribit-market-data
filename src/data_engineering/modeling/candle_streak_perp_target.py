"""CS03c: does Binance USD-M signed taker flow predict the PERPETUAL's own next 5-minute return?

Protocol: discussions/Candle_Streaks/003c_claude_perp_target_preregistration.md.

CS03b found that USD-M futures signed flow improves forecasts of the next SPOT candle. Perpetuals lead spot,
so part of that can be spot catching up to a perp move that has already happened. A trader on the perp
needs the perp's OWN next return. CS03c reruns the frozen CS03b ablation with the perp as the target market
(sign, streak, volatility and target all from perp 1-minute closes), on the same complete days as a
spot-target reference, and adds an economic test net of perpetual trading costs.

Frozen from CS03/CS03b: panel construction (tf.build_origin_panel), complete-day rule, 60/20/20 day split,
features, multinomial logit (newton-cholesky, C=.25), losses and paired day bootstrap (ab.*).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_engineering.modeling import candle_streak_flow_attribution as ab
from data_engineering.modeling import candle_streak_trade_flow as tf

VERSION = "CS03c-0.1.0"
ECON_MODELS = ("price_vol", "price_vol_signed", "price_vol_both")
STEP = tf.STEP
MINUTE = tf.MINUTE


@dataclass(frozen=True)
class Specification:
    version: str = VERSION
    assets: tuple = ("BTC", "ETH", "SOL", "XRP")
    primary_assets: tuple = ("BTC", "ETH")
    end: str = "2026-10-14T00:00:00Z"            # CS03b loader cutoff
    edge_quantile: float = 0.90                  # trade only the top 10% |P(up)-P(down)| (TRAIN cutoff)
    taker_cost_per_side: float = 0.0006          # 0.05% Deribit perp taker + 1 bp slippage (BT 001)
    low_cost_per_side: float = 0.0003            # 0.02% fee tier + 1 bp slippage
    maker_cost_per_side: float = 0.0             # optimistic bound, no adverse selection modelled
    bootstrap_reps: int = 1000
    seed: int = 2606
    status: str = "HISTORICAL_EXPLORATORY"


def cost_scenarios(spec: Specification | None = None) -> dict:
    spec = spec or Specification()
    return {"taker": spec.taker_cost_per_side, "low_fee": spec.low_cost_per_side, "zero": spec.maker_cost_per_side}


def registration() -> dict:
    return {**asdict(Specification()), "cs03b": ab.registration(), "cs03": tf.registration()}


# ---------------------------------------------------------------- perp prices

def perp_returns_5m(frame: pd.DataFrame) -> pd.DataFrame:
    """5-minute perp close-to-close log returns from 1-minute klines.
    state_time = 5m bucket END; the bucket must contain exactly minutes t-5..t-1; its close is the t-1 minute
    close. The return needs the previous bucket complete too; otherwise NaN (no fill)."""
    q = frame[["open_time", "close"]].copy()
    q["open_time"] = pd.to_datetime(q["open_time"], utc=True)
    q = q.drop_duplicates("open_time").sort_values("open_time")
    q["close"] = pd.to_numeric(q["close"], errors="coerce")
    q["state_time"] = q["open_time"].dt.floor("5min") + STEP
    z = q.groupby("state_time", sort=True).agg(n=("open_time", "size"), first=("open_time", "min"),
                                                last=("open_time", "max"), close=("close", "last"))
    good = (z["n"] == 5) & (z["first"] == z.index - STEP) & (z["last"] == z.index - MINUTE) & (z["close"] > 0)
    grid = pd.date_range(z.index.min(), z.index.max(), freq=STEP)
    z = z.reindex(grid)
    good = good.reindex(grid).fillna(False).to_numpy(bool)
    lc = np.where(good, np.log(z["close"].to_numpy(float)), np.nan)
    ret = lc - np.r_[np.nan, lc[:-1]]
    return pd.DataFrame({"state_time": grid, "return_5m": ret})


def read_perp(path: Path, end=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(perp 5m returns, perp 5m flow) from the existing USD-M 1-minute parquet."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Binance USD-M file: {path}")
    q = pd.read_parquet(path, columns=["open_time", "close", "quote_volume", "taker_buy_quote_volume", "trades"])
    if end is not None:
        q = q[pd.to_datetime(q["open_time"], utc=True) < pd.Timestamp(end)]
    return perp_returns_5m(q), tf.one_minute_futures_flow(q)


def build_panel(prices: pd.DataFrame, flow5m: pd.DataFrame, spec: Specification | None = None) -> pd.DataFrame:
    """CS03 origin panel plus the realised next return of the target market (for P&L only)."""
    s = ab.Specification()
    x = tf.build_origin_panel(prices, flow5m, fast_bars=s.fast_bars, slow_bars=s.slow_bars, flow_bars=s.flow_bars)
    r = pd.to_numeric(x["return_5m"], errors="coerce").to_numpy(float)
    x["next_return"] = np.where(x["next_valid"].to_numpy(bool), np.r_[r[1:], np.nan], np.nan)
    return x


# ---------------------------------------------------------------- economics

def edge(prob: np.ndarray) -> np.ndarray:
    """P(UP) - P(DOWN) from a three-class (DOWN, ZERO, UP) forecast."""
    return prob[:, 2] - prob[:, 0]


def edge_threshold(train: pd.DataFrame, fit: dict, model: str, q: float) -> float:
    tr = train.loc[train.scorable]
    e = edge(ab.predict_ablation(tr, fit)[model])
    return float(np.quantile(np.abs(e), q))


def daily_pnl(events: pd.DataFrame, e: np.ndarray, thr: float, cost_side: float) -> pd.DataFrame:
    """Hold sign(edge) for the next 5 minutes when |edge| >= thr, else flat. Positions open and close within
    each UTC day (flat at the first origin of the day and after the last). Returns per-day gross, cost, net
    (log-return units), active bars and turnover."""
    ev = events[["date", "state_time", "next_return"]].copy().reset_index(drop=True)
    ev["pos"] = np.where(np.abs(e) >= thr, np.sign(e), 0.0)
    ev = ev.sort_values(["date", "state_time"]).reset_index(drop=True)
    rows = []
    for d, g in ev.groupby("date", sort=True):
        pos = g["pos"].to_numpy(float)
        t = g["state_time"]
        contiguous = np.r_[False, (t.diff().iloc[1:] == STEP).to_numpy()]
        prev = np.where(contiguous, np.r_[0.0, pos[:-1]], 0.0)          # a gap forces flat in between
        nxt_contig = np.r_[contiguous[1:], False]
        close_out = np.where(nxt_contig, 0.0, np.abs(pos))               # exit before a gap / end of day
        turnover = np.abs(pos - prev).sum() + close_out.sum()
        gross = float(np.nansum(pos * g["next_return"].to_numpy(float)))
        rows.append({"date": d, "gross": gross, "cost": float(turnover * cost_side), "net": gross - turnover * cost_side,
                     "active_bars": int((pos != 0).sum()), "turnover": float(turnover),
                     "hits": int(((pos * g["next_return"].to_numpy(float)) > 0).sum())})
    return pd.DataFrame(rows)


def econ_summary(days: pd.DataFrame, reps: int, seed: int) -> dict:
    net = days["net"].to_numpy(float)
    rng = np.random.default_rng(seed)
    if len(net) >= 20:
        b = net[rng.integers(0, len(net), size=(reps, len(net)))].mean(axis=1)
        lo, hi = np.quantile(b, [0.025, 0.975])
    else:
        lo = hi = np.nan
    act = days["active_bars"].sum()
    return {"days": int(len(days)), "active_bars": int(act),
            "gross_bp_per_active_bar": float(1e4 * days["gross"].sum() / act) if act else np.nan,
            "hit_rate": float(days["hits"].sum() / act) if act else np.nan,
            "turnover_per_day": float(days["turnover"].mean()),
            "net_bp_per_day": float(1e4 * net.mean()), "net_bp_per_day_lo": float(1e4 * lo),
            "net_bp_per_day_hi": float(1e4 * hi), "gross_bp_per_day": float(1e4 * days["gross"].mean()),
            "cost_bp_per_day": float(1e4 * days["cost"].mean())}


def decision(boot: pd.DataFrame, econ: pd.DataFrame, spec: Specification | None = None, *, synthetic=False) -> dict:
    """Frozen gates on the PERP target, BTC and ETH, TEST window."""
    spec = spec or Specification()
    if synthetic:
        return {"verdict": "SYNTHETIC_NO_MARKET_INFERENCE"}
    stat, econ_ok, rows = [], [], []
    for a in spec.primary_assets:
        b = boot[(boot.asset == a) & (boot.period == "test") & (boot.target == "perp")]
        for metric in ("threeclass_logloss", "binary_logloss"):
            r = b[(b.metric == metric) & (b.model == "price_vol_signed") & (b.reference == "price_vol")]
            ok = len(r) == 1 and r.iloc[0].status == "OK"
            stat.append(bool(ok and r.iloc[0].ci975 < 0))
            rows.append({"asset": a, "gate": f"stat_{metric}", "delta": float(r.iloc[0].delta_equal_day) if ok else None,
                         "upper95": float(r.iloc[0].ci975) if ok else None, "pass": stat[-1]})
        e = econ[(econ.asset == a) & (econ.period == "test") & (econ.target == "perp") &
                 (econ.model == "price_vol_signed") & (econ.cost == "taker") & (econ.rule == "top_decile")]
        ok = len(e) == 1 and np.isfinite(e.iloc[0].net_bp_per_day_lo)
        econ_ok.append(bool(ok and e.iloc[0].net_bp_per_day_lo > 0))
        rows.append({"asset": a, "gate": "econ_taker_top_decile", "net_bp_per_day": float(e.iloc[0].net_bp_per_day) if ok else None,
                     "lower95": float(e.iloc[0].net_bp_per_day_lo) if ok else None, "pass": econ_ok[-1]})
    s, c = all(stat), all(econ_ok)
    verdict = "PERP_TRADEABLE_CANDIDATE" if (s and c) else ("PERP_STATISTICAL_ONLY" if s else
              ("ECONOMIC_WITHOUT_STAT_GATE" if c else "NO_PERP_SIGNAL"))
    return {"verdict": verdict, "status": "HISTORICAL_EXPLORATORY", "gates": rows,
            "prospective_confirmation": False, "AM_T_HAR_Blend_modified": False,
            "note": "Primary target = perp's own next 5m return; spot target reported on the same days as a "
                    "lead-lag reference. CS03 TEST dates reused."}


# ---------------------------------------------------------------- synthetic

def synthetic_pair(n_days: int = 130, seed: int = 0, perp_beta: float = 0.0, spot_lag_beta: float = 0.0):
    """1m perp klines and 5m spot returns. perp_beta: perp flow predicts the perp's next 5m return.
    spot_lag_beta: spot follows the CURRENT perp 5m move one bar later (pure lead-lag)."""
    rng = np.random.default_rng(seed)
    n5 = n_days * 288
    t1 = pd.date_range("2026-01-01", periods=n5 * 5, freq="1min", tz="UTC")
    frac = np.clip(0.5 + 0.15 * rng.standard_normal(n5).repeat(5) + 0.03 * rng.standard_normal(n5 * 5), 0.02, 0.98)
    v = rng.uniform(100.0, 800.0, n5 * 5)
    flow5 = (2 * (v * frac).reshape(n5, 5).sum(1) - v.reshape(n5, 5).sum(1)) / v.reshape(n5, 5).sum(1)
    sig = 0.0015
    r5 = sig * rng.standard_normal(n5)
    z = flow5 / flow5.std()
    r5 += 0.5 * sig * z                                              # flow and same-bar perp move co-move
    r5[1:] += perp_beta * sig * z[:-1]                               # flow at t predicts perp return at t+1
    r5[rng.random(n5) < 0.03] = 0.0                                 # exact-zero bars (tick discreteness)
    r1 = (r5 / 5).repeat(5)
    close = np.round(100 * np.exp(np.cumsum(r1)), 10)
    kl = pd.DataFrame({"open_time": t1, "close": close, "quote_volume": v, "taker_buy_quote_volume": v * frac,
                       "trades": rng.integers(3, 40, n5 * 5)})
    spot = 0.002 * rng.standard_normal(n5) * 0.3
    spot[1:] += spot_lag_beta * r5[:-1]
    spot[1:] += 0.5 * r5[1:] * (spot_lag_beta == 0)                 # contemporaneous co-movement otherwise
    spot[rng.random(n5) < 0.03] = 0.0
    sp = pd.DataFrame({"state_time": pd.date_range("2026-01-01 00:05", periods=n5, freq="5min", tz="UTC"),
                       "return_5m": spot})
    return kl, sp
