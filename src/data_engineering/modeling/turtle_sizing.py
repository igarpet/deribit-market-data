"""Turtle Breakout 08: risk per trade from 1% to 100% on BTC, against the 2% of the original rules.

Protocol: discussions/Turtle_Breakout/010_claude_tb08_preregistration.md (fixed before any real-data output).

Strategy: TB 01 daily long-only turtle on the Binance BTCUSDT perpetual, 5m fills, taker 0.05% + 1 bp, funding,
window = all BTC data after a 30-day warm-up (as TB 01's single-asset table). Only risk_frac changes:
units = f x HWM / (2 ATR), f = 1%, 2%, ..., 100%.
    Primary sweep  P   max_pos_lev = 125 (Binance BTCUSDT maximum), sizing on the high-water mark (TB 01)
    Reported       C3  TB 01's 3x notional cap;  EQ  sizing on current equity instead of the high-water mark
Ruin: equity <= 0 at a daily mark; equity is floored at 1e-6 of the start for log returns (ruin = -13.8 log units).

Decision (per f != 2%, P sweep): paired difference of mean daily log return (geometric growth) vs f = 2%,
21-day circular block bootstrap (2,000), two-sided p from the bootstrap standard error, Holm over the 99 comparisons at alpha = 0.05:
HIGHER_GROWTH / LOWER_GROWTH / NO_DIFFERENCE.  Sharpe differences: same bootstrap, Holm, reported.
Kelly (reported): growth-optimal f on the 2% run's trade R (f x R of equity per trade), with a 5,000-run trade
bootstrap of the optimum.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import turtle_breakout as tb

VERSION = "TB08-0.3.0"   # 0.3.0 (TB 10): cross-coin tests; 0.2.0 (TB 09): trade-list equity engine for long + stop-and-reverse; 0.1.0 functions unchanged
FRACS = tuple(round(i / 100, 2) for i in range(1, 101))
BASE_F = 0.02
FLOOR = 1e-6


@dataclass(frozen=True)
class SParams:
    max_lev_primary: float = 125.0
    max_lev_c3: float = 3.0
    block_days: int = 21
    n_boot: int = 2000
    n_kelly_boot: int = 5000
    alpha: float = 0.05
    seed: int = 20261026
    status: str = "PREREGISTERED_TB010"


def registration(sp: SParams | None = None) -> dict:
    d = asdict(sp or SParams())
    d.update(version=VERSION, fracs="0.01..1.00 step 0.01", base_f=BASE_F, asset="BTC",
             sweeps={"P": "cap 125x, HWM sizing (primary)", "C3": "cap 3x (TB 01)", "EQ": "cap 125x, current-equity sizing"},
             decision="per f: mean daily log-return difference vs 2%, 21-day block bootstrap, Holm over 99, alpha 0.05",
             reported=["CAGR, Sharpe, max DD, Calmar, ruin, worst trade as % of equity, mean/max leverage",
                       "Sharpe differences (Holm)", "C3 and EQ sweeps", "Kelly curve and bootstrap of the optimal f",
                       "equity curves for selected f"])
    return d


def run_f(d: tb.AssetData, p: tb.Params, f: float, start, cap: float, size_base: str = "hwm"):
    q = replace(p, risk_frac=f, max_pos_lev=cap, size_base=size_base)
    E, T, st = tb.simulate({d.name: d}, q, start=start)
    return E, T, st


def daily_equity(E: pd.Series, e0: float) -> pd.Series:
    D = tb.daily(E)
    ruined = (D <= 0).cummax()
    D = D.where(~ruined, 0.0)
    return D.clip(lower=FLOOR * e0)


def summarize(D: pd.Series, T: pd.DataFrame, st: dict, e0: float) -> dict:
    lr = np.log(D).diff().dropna()
    yrs = (D.index[-1] - D.index[0]).total_seconds() / (365.25 * 86400)
    tot = D.iloc[-1] / e0
    r = D.pct_change().dropna()
    dd = float((D / D.cummax() - 1).min())
    ruined = bool(D.iloc[-1] <= FLOOR * e0 * 1.0001)
    cagr = float(tot ** (1 / yrs) - 1) if (yrs > 0 and not ruined) else -1.0
    worst = np.nan
    if len(T):
        eq_before = e0 + T.pnl.cumsum().shift(1, fill_value=0.0)
        worst = float((T.pnl / eq_before.where(eq_before > 0)).min())
    return {"final_equity": float(D.iloc[-1]), "multiple": float(tot), "cagr": cagr,
            "log_growth_per_year": float(lr.mean() * 365.25),
            "sharpe": float(r.mean() / r.std() * math.sqrt(365)) if r.std() > 0 else np.nan,
            "max_dd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan, "ruined": ruined,
            "trades": int(len(T)), "worst_trade_pct_equity": worst,
            "mean_gross_lev": float(st["mean_gross_leverage"]), "max_gross_lev": float(st["max_gross_leverage"]),
            "capped_entries": int(sum(st["leverage_capped"].values()))}


def sweep(d: tb.AssetData, p: tb.Params, start, cap: float, size_base: str = "hwm", fracs=FRACS):
    rows, curves = [], {}
    for f in fracs:
        E, T, st = run_f(d, p, f, start, cap, size_base)
        D = daily_equity(E, p.start_equity)
        curves[f] = D
        rows.append({"f": f, **summarize(D, T, st, p.start_equity)})
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ---------------------------------------------------------------- statistics

def block_index(n: int, block: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    st = rng.integers(0, n, size=(n_boot, nb))
    return ((st[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]       # circular


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def _p_norm(est: float, bs: np.ndarray) -> float:
    """Two-sided p from the bootstrap standard error (normal approximation). A percentile p cannot resolve
    the Holm threshold 0.05 / 99 with 2,000 draws."""
    bs = bs[np.isfinite(bs)]
    se = float(np.std(bs, ddof=1)) if len(bs) > 10 else np.nan
    if not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return 1.0
    return float(math.erfc(abs(est) / se / math.sqrt(2)))


def compare(curves: pd.DataFrame, sp: SParams, base: float = BASE_F) -> pd.DataFrame:
    """Growth and Sharpe differences of every f against base, paired, circular block bootstrap, Holm."""
    L = np.log(curves).diff().dropna()
    X = L.to_numpy(); n = len(X)
    b = list(curves.columns).index(base)
    idx = block_index(n, sp.block_days, sp.n_boot, sp.seed)
    xb = X[:, b]
    rows = []
    for i, f in enumerate(curves.columns):
        if f == base:
            continue
        dlt = X[:, i] - xb
        est = dlt.mean() * 365.25
        bs = dlt[idx].mean(1) * 365.25
        pg = _p_norm(est, bs)
        ra = np.expm1(X[:, i]); rb = np.expm1(xb)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = lambda z: z.mean(-1) / z.std(-1) * math.sqrt(365)
            sd = sh(ra) - sh(rb)
            bsd = sh(ra[idx]) - sh(rb[idx])
        ps = _p_norm(sd, bsd)
        rows.append({"f": f, "growth_diff": est, "growth_lo95": float(np.quantile(bs, .025)), "growth_hi95": float(np.quantile(bs, .975)),
                     "p_growth": pg, "sharpe_diff": float(sd), "sharpe_lo95": float(np.nanquantile(bsd, .025)),
                     "sharpe_hi95": float(np.nanquantile(bsd, .975)), "p_sharpe": ps})
    R = pd.DataFrame(rows)
    R["p_growth_holm"] = holm(R.p_growth.to_numpy()); R["p_sharpe_holm"] = holm(R.p_sharpe.to_numpy())
    R["growth_result"] = np.where(R.p_growth_holm < sp.alpha, np.where(R.growth_diff > 0, "HIGHER_GROWTH", "LOWER_GROWTH"), "NO_DIFFERENCE")
    R["sharpe_result"] = np.where(R.p_sharpe_holm < sp.alpha, np.where(R.sharpe_diff > 0, "HIGHER_SHARPE", "LOWER_SHARPE"), "NO_DIFFERENCE")
    return R


def decide(cmp: pd.DataFrame, table: pd.DataFrame) -> dict:
    hi = cmp.loc[cmp.growth_result == "HIGHER_GROWTH", "f"].tolist()
    lo = cmp.loc[cmp.growth_result == "LOWER_GROWTH", "f"].tolist()
    best = table.loc[table.log_growth_per_year.idxmax()]
    return {"higher_growth_than_2pct": hi, "lower_growth_than_2pct": lo,
            "n_higher": len(hi), "n_lower": len(lo), "n_no_difference": int((cmp.growth_result == "NO_DIFFERENCE").sum()),
            "in_sample_best_f": float(best.f), "in_sample_best_cagr": float(best.cagr), "in_sample_best_max_dd": float(best.max_dd),
            "first_ruin_f": float(table.loc[table.ruined, "f"].min()) if table.ruined.any() else None,
            "verdict": "LARGER_RISK_GROWS_FASTER" if hi else ("2PCT_NOT_BEATEN" if not lo else "2PCT_NOT_BEATEN_SOME_WORSE")}


def kelly_curve(R: np.ndarray, fracs=FRACS) -> pd.Series:
    F = np.asarray(fracs)[:, None]
    g = 1 + F * np.asarray(R)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(g > 0, np.log(np.maximum(g, 1e-300)), -np.inf).sum(1)
    return pd.Series(lg, index=list(fracs))


def kelly_boot(R: np.ndarray, sp: SParams, fracs=FRACS) -> np.ndarray:
    rng = np.random.default_rng(sp.seed + 7)
    R = np.asarray(R); F = np.asarray(fracs)
    out = np.empty(sp.n_kelly_boot)
    for i in range(sp.n_kelly_boot):
        x = R[rng.integers(0, len(R), len(R))]
        out[i] = F[int(np.argmax(kelly_curve(x, fracs).to_numpy()))]
    return out


# ---------------------------------------------------------------- TB 09: long + stop-and-reverse on one asset

def trade_list(d: tb.AssetData, p: tb.Params, start, rp=None, with_reverse: bool = True) -> pd.DataFrame:
    """Chronological, non-overlapping trades of one asset: TB 01 longs (entries do not depend on f) and, optionally,
    the TB 07 reverse short at every losing long stop exit. Columns: dir, je, jx, px_in, px_out, atr, cost, kind."""
    from . import turtle_reverse as tr
    from . import turtle_trade_features as tf
    rp = rp or tr.RParams()
    _, TL, _ = tb.simulate({d.name: d}, p, start=start)                    # includes a final open trade ('end')
    k = tb.period_id(pd.DatetimeIndex(TL.entry_time), p.period) - d.pid0
    L = pd.DataFrame({"dir": 1, "je": TL.je.astype(int), "jx": TL.jx.astype(int), "px_in": TL.entry_px, "px_out": TL.exit_px,
                      "atr": d.atr[k], "cost": p.fee + p.slip, "kind": "long",
                      "end": (TL.reason == "end").to_numpy()})
    if not with_reverse:
        return L.sort_values("je").reset_index(drop=True)
    T = tf.run_trades(d, p, start)
    R = tr.reverse_trades(d, T, rp)
    if len(R):
        S = pd.DataFrame({"dir": -1, "je": d.end.get_indexer(pd.DatetimeIndex(R.rev_time)),
                          "jx": d.end.get_indexer(pd.DatetimeIndex(R.rev_exit_time)), "px_in": R.rev_px, "px_out": R.rev_exit_px,
                          "atr": R.atr_rev, "cost": rp.side_cost, "kind": "reverse", "end": False})
        L = pd.concat([L, S], ignore_index=True)
    L = L.sort_values(["je", "dir"]).reset_index(drop=True)
    # positions must not overlap: each trade opens at or after the previous exit
    assert (L.je.to_numpy()[1:] >= L.jx.to_numpy()[:-1]).all(), "overlapping trades"
    return L


def equity_from_trades(d: tb.AssetData, L: pd.DataFrame, p: tb.Params, f: float, start, cap: float,
                       size_base: str = "hwm"):
    """Daily-marked equity of a one-position-at-a-time trade list with TB 01 sizing:
    u = min(f x base / (2 ATR), cap x E_prev / price), base = HWM or E_prev at the last period mark."""
    pid_lo = int(tb.period_id(pd.DatetimeIndex([start + pd.Timedelta(minutes=tb.BAR_MIN)]), p.period)[0])
    k_lo = max(0, pid_lo - d.pid0)
    je_arr = L.je.to_numpy()
    ev = [(int(r.jx), 0, i) for i, r in enumerate(L.itertuples())] + [(int(r.je), 1, i) for i, r in enumerate(L.itertuples())]
    # by bar; at one bar: closes of earlier trades, then opens, then a close of a trade opened at that same bar
    ev.sort(key=lambda x: (x[0], 1 if x[1] == 1 else (2 if je_arr[x[2]] == x[0] else 0)))
    realized, hwm, e_prev = 0.0, p.start_equity, p.start_equity
    units = {}; pnl_rows = []; marks = []; gross = []; capped = 0
    ei = 0
    for k in range(k_lo, len(d.s)):
        if d.s[k] < 0:
            continue
        while ei < len(ev) and ev[ei][0] <= d.e[k]:
            j, typ, i = ev[ei]; ei += 1
            r = L.iloc[i]
            if typ == 1:
                if e_prev <= 0:
                    continue
                base = hwm if size_base == "hwm" else e_prev
                want = f * base / (p.stop_atr * r.atr)
                u = min(want, cap * e_prev / r.px_in)
                capped += int(u < want)
                units[i] = u
            elif i in units and not bool(r.end):            # an 'end' trade stays open until after the last mark
                u = units.pop(i)
                pnl = r.dir * u * (r.px_out - r.px_in) - r.cost * u * (r.px_in + r.px_out) - r.dir * u * (d.fcum[int(r.jx)] - d.fcum[int(r.je)])
                realized += pnl
                pnl_rows.append({"kind": r.kind, "pnl": pnl, "exit_bar": int(r.jx)})
        j = d.e[k]; unreal = 0.0; g = 0.0
        for i, u in units.items():
            r = L.iloc[i]
            unreal += r.dir * u * (d.c[j] - r.px_in) - r.cost * u * r.px_in - r.dir * u * (d.fcum[j] - d.fcum[int(r.je)])
            g += u * d.c[j]
        E = p.start_equity + realized + unreal
        marks.append((d.pid0 + k, E)); gross.append(g / E if E > 0 else np.nan)
        hwm = max(hwm, E); e_prev = E
    # open trade at the end: closed at the last bar, as TB 01
    for i, u in list(units.items()):
        r = L.iloc[i]; j = len(d.c) - 1
        pnl = r.dir * u * (d.c[j] - r.px_in) - r.cost * u * (r.px_in + d.c[j]) - r.dir * u * (d.fcum[j] - d.fcum[int(r.je)])
        realized += pnl; pnl_rows.append({"kind": r.kind, "pnl": pnl, "exit_bar": j})
    pid = np.array([m[0] for m in marks])
    Eser = pd.Series([m[1] for m in marks], index=tb.period_end(pid, p.period))
    st = {"mean_gross_leverage": float(np.nanmean(gross)), "max_gross_leverage": float(np.nanmax(gross)),
          "leverage_capped": {d.name: capped}}
    return Eser, pd.DataFrame(pnl_rows), st


def sweep_trades(d: tb.AssetData, L: pd.DataFrame, p: tb.Params, start, cap: float, size_base: str = "hwm", fracs=FRACS):
    rows, curves = [], {}
    for f in fracs:
        E, T, st = equity_from_trades(d, L, p, f, start, cap, size_base)
        D = daily_equity(E, p.start_equity)
        curves[f] = D
        row = {"f": f, **summarize(D, T, st, p.start_equity)}
        if len(T):
            row["pnl_long"] = float(T.loc[T.kind == "long", "pnl"].sum()); row["pnl_reverse"] = float(T.loc[T.kind == "reverse", "pnl"].sum())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(curves)


# ---------------------------------------------------------------- TB 10: long vs long + reverse across coins

def sign_test(k: int, n: int) -> float:
    """One-sided P(X >= k), X ~ Binomial(n, 1/2)."""
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else np.nan


def pooled_log_returns(curves: dict) -> pd.Series:
    """Equal-weight sleeves: each coin is its own account; the pool's daily log return is the mean over the
    coins trading that day (coins enter when their window starts)."""
    L = pd.DataFrame({a: np.log(c).diff() for a, c in curves.items()})
    return L.mean(axis=1, skipna=True).dropna()


def pooled_compare(curves_a: dict, curves_b: dict, sp: SParams) -> dict:
    """Paired growth difference (b - a) of the pooled sleeves, circular block bootstrap."""
    a = pooled_log_returns(curves_a); b = pooled_log_returns(curves_b)
    i = a.index.intersection(b.index)
    dl = (b[i] - a[i]).to_numpy()
    idx = block_index(len(dl), sp.block_days, sp.n_boot, sp.seed + 3)
    bs = dl[idx].mean(1) * 365.25
    ea = np.exp(a[i].cumsum()); eb = np.exp(b[i].cumsum())
    mdd = lambda e: float((e / e.cummax() - 1).min())
    sh = lambda x: float(np.expm1(x).mean() / np.expm1(x).std() * math.sqrt(365))
    return {"days": int(len(i)), "growth_a": float(a[i].mean() * 365.25), "growth_b": float(b[i].mean() * 365.25),
            "growth_diff": float(dl.mean() * 365.25), "lo95": float(np.quantile(bs, .025)), "hi95": float(np.quantile(bs, .975)),
            "lo_bonf": float(np.quantile(bs, sp.alpha / 3)), "max_dd_a": mdd(ea), "max_dd_b": mdd(eb),
            "sharpe_a": sh(a[i]), "sharpe_b": sh(b[i])}


def decide_all(per_coin: pd.DataFrame, pooled: dict, sp: SParams) -> dict:
    """per_coin: one row per TEST coin with dd_L, dd_LR, g_L, g_LR at f = 2% (P sweep). Bonferroni over 3 at alpha."""
    n = len(per_coin)
    k_dd = int((per_coin.dd_LR > per_coin.dd_L).sum()); k_g = int((per_coin.g_LR > per_coin.g_L).sum())
    p_dd, p_g = sign_test(k_dd, n), sign_test(k_g, n)
    a = sp.alpha / 3
    h1, h2, h3 = p_dd < a, p_g < a, pooled["lo_bonf"] > 0
    return {"coins": n, "shallower_dd": k_dd, "p_dd": p_dd, "higher_growth": k_g, "p_growth": p_g,
            "pooled_growth_diff": pooled["growth_diff"], "pooled_lo_bonf": pooled["lo_bonf"],
            "H1_drawdown": "PASS" if h1 else "FAIL", "H2_growth": "PASS" if h2 else "FAIL", "H3_pooled_growth": "PASS" if h3 else "FAIL",
            "verdict": "REVERSE_HELPS" if (h1 and h2 and h3) else ("HEDGE_ONLY" if h1 else "NO_IMPROVEMENT")}
