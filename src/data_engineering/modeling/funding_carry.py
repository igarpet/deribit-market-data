"""
Funding Carry branch, FC 01: does harvesting Binance USD-M funding with a delta-neutral
long spot / short perpetual position earn more than cash, net of costs?

Pre-registration: discussions/Funding_Carry/001_claude_branch_preregistration.md

Mechanics (per asset, per unit of ENTRY notional; constant units within an episode, no compounding)
---------------------------------------------------------------------------------------------------
Decision / execution boundaries b_d = 00:05 UTC every day. Prices at b_d = close of the 1m bar
that ends at b_d (open_time 00:04), as-of with a 10-minute staleness limit.
Holding interval d = (b_d, b_{d+1}]. A position decided at b_d is held over interval d.
    units u = 1 / S_entry                     long u spot, short u perpetual
    pnl_d   = u * [ (S_{d+1} - S_d) - (F_{d+1} - F_d) + sum_{i in d} f_i * F(t_i) ]
              f_i = funding rate printed at t_i (shorts RECEIVE f_i when f_i > 0, pay when < 0);
              F(t_i) = perpetual close of the minute ending at t_i (mark-price proxy)
    costs   entry 17 bp of entry notional (spot 11 + perp 6), exit 17 bp of current notional
            (u * S at exit); reset (see below) 34 bp of current notional
    reset   margin-safety rule: if F_d / F_ref - 1 >= RESET_MOVE at a boundary while in
            position, both legs are re-balanced (charged a full round trip) and F_ref = F_d.

Strategies
    C (primary)  f7_d = annualized mean funding over the 7 days before b_d (sum * 365/7).
                 flat -> long carry when f7_d >= ENTRY; in -> flat when f7_d < EXIT.
    U (secondary benchmark) long carry on every valid day from first to last.

Capital: each asset slot reserves 1 + PERP_MARGIN = 1.5 units of capital per unit entry notional
(spot paid in full; 2x on the short perpetual). Portfolio returns are summed asset P&L divided by
reserved capital of eligible assets (idle capital credited 0) and, separately, by deployed capital.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

VERSION = "funding_carry 1.0.0"

UNIVERSE = ("BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LINK", "DOT", "LTC", "BCH", "UNI", "NEAR", "APT",
            "ARB", "OP", "SUI", "FIL", "ATOM", "ETC", "INJ", "AAVE", "TON", "WIF", "HBAR", "XLM", "SEI", "TIA",
            "ENA", "WLD")          # = PF02 universe (fixed 2026-10-16); AVAX/TRX/HYPE excluded at Ivan's request

SPOT_COST_BP = 11.0                 # Binance spot taker 0.10% + 1 bp slippage (as PF02)
PERP_COST_BP = 6.0                  # USD-M taker 0.05% + 1 bp slippage
SIDE_COST = (SPOT_COST_BP + PERP_COST_BP) / 1e4        # one side, both legs = 17 bp
LOW_SPOT_COST_BP = 8.5              # sensitivity: BNB-discounted spot taker 0.075% + 1 bp
LOW_PERP_COST_BP = 5.0              # sensitivity: USD-M taker 0.045% (BNB) + ~0.5 bp

ENTRY = 0.10                        # annualized trailing-7d funding to enter
EXIT = 0.03                         # annualized trailing-7d funding below which to exit
LOOKBACK_DAYS = 7
MIN_PRINTS_FRAC = 0.8               # f7 needs >= 80% of the expected funding prints
PERP_MARGIN = 0.5
CAPITAL_PER_NOTIONAL = 1.0 + PERP_MARGIN
RESET_MOVE = 0.40
BOUNDARY = pd.Timedelta(minutes=5)  # 00:05 UTC
STALE = pd.Timedelta(minutes=10)

DEV_END = pd.Timestamp("2026-01-01 00:05", tz="UTC")   # DEV = start .. 2025-12-31, TEST = 2026
MIN_ASSET_DAYS = 180
HURDLE = 0.04                       # cash alternative, annualized
BREADTH = 0.60
BOOT_B = 2000
BOOT_BLOCK = 14
SEED = 20261018

GRID_ENTRY = (0.0, 0.05, 0.10, 0.20, 0.30)
GRID_EXIT = (-0.05, 0.0, 0.03, 0.10)


@dataclass(frozen=True)
class Params:
    entry: float = ENTRY
    exit: float = EXIT
    side_cost: float = SIDE_COST
    reset_move: float = RESET_MOVE


def registration() -> dict:
    return {"version": VERSION, "universe": list(UNIVERSE), "spot_cost_bp": SPOT_COST_BP,
            "perp_cost_bp": PERP_COST_BP, "entry": ENTRY, "exit": EXIT, "lookback_days": LOOKBACK_DAYS,
            "perp_margin": PERP_MARGIN, "reset_move": RESET_MOVE, "boundary_utc": "00:05",
            "dev_end": str(DEV_END), "min_asset_days": MIN_ASSET_DAYS, "hurdle": HURDLE, "breadth": BREADTH,
            "boot_B": BOOT_B, "boot_block_days": BOOT_BLOCK, "grid_entry": list(GRID_ENTRY),
            "grid_exit": list(GRID_EXIT), "venue": "Binance spot + Binance USD-M perpetual (executed there)"}


# ----------------------------------------------------------------------------- data
def load_close(path) -> pd.Series:
    d = pd.read_parquet(path, columns=["open_time", "close"])
    t = pd.to_datetime(d["open_time"], utc=True) + pd.Timedelta(minutes=1)   # minute END
    s = pd.Series(pd.to_numeric(d["close"], errors="coerce").to_numpy(float), index=t)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s[s > 0].dropna()


def load_funding(path) -> pd.Series:
    """Binance fundingRate files: calc_time (ms or us), last_funding_rate. Index = funding time (UTC, minute)."""
    f = pd.read_parquet(path)
    t = pd.to_numeric(f["calc_time"], errors="coerce")
    unit = "us" if t.max() > 10 ** 14 else "ms"
    s = pd.Series(pd.to_numeric(f["last_funding_rate"], errors="coerce").to_numpy(float),
                  index=pd.to_datetime(t, unit=unit, utc=True).dt.floor("min"))
    s = s[s.index.notna()].dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _ns(ix) -> np.ndarray:
    """int64 nanoseconds regardless of the index's datetime resolution (pandas 3 defaults to us)."""
    return pd.DatetimeIndex(ix).as_unit("ns").asi8


def _asof(s: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    """Value of the last observation at or before each time, NaN if older than STALE."""
    if len(s) == 0:
        return np.full(len(times), np.nan)
    ti = _ns(s.index)
    q = _ns(times)
    k = np.searchsorted(ti, q, side="right") - 1
    v = np.where(k >= 0, s.to_numpy()[np.clip(k, 0, None)], np.nan)
    age = q - np.where(k >= 0, ti[np.clip(k, 0, None)], 0)
    return np.where((k >= 0) & (age <= STALE.value), v, np.nan)


def daily_frame(spot: pd.Series, perp: pd.Series, funding: pd.Series) -> pd.DataFrame:
    """One row per boundary b_d with prices, the funding cash flow over (b_d, b_{d+1}] and f7 at b_d."""
    start = max(spot.index.min(), perp.index.min(), funding.index.min() + pd.Timedelta(days=LOOKBACK_DAYS))
    end = min(spot.index.max(), perp.index.max(), funding.index.max())
    b0 = start.floor("D") + BOUNDARY
    if b0 < start:
        b0 += pd.Timedelta(days=1)
    b = pd.date_range(b0, end, freq="D")
    S = _asof(spot, b)
    F = _asof(perp, b)
    ft = funding.index
    fr = funding.to_numpy(float)
    fF = _asof(perp, ft)                      # perp price at each print
    fF = np.where(np.isfinite(fF), fF, np.nan)
    flow = fr * fF
    cs_flow = np.r_[0.0, np.cumsum(np.nan_to_num(flow))]
    cs_rate = np.r_[0.0, np.cumsum(fr)]
    cs_n = np.arange(len(fr) + 1)
    fti = _ns(ft)
    kb = np.searchsorted(fti, _ns(b), side="right")         # prints with t <= b_d
    kn = np.r_[kb[1:], kb[-1]]                               # prints with t <= b_{d+1}
    fund_cash = cs_flow[kn] - cs_flow[kb]                    # (b_d, b_{d+1}]
    fund_rate = cs_rate[kn] - cs_rate[kb]
    k7 = np.searchsorted(fti, _ns(b - pd.Timedelta(days=LOOKBACK_DAYS)), side="right")
    n7 = cs_n[kb] - cs_n[k7]
    s7 = cs_rate[kb] - cs_rate[k7]
    # expected prints from the asset's own interval (median spacing of prints)
    spacing_h = float(np.median(np.diff(fti)) / 3.6e12) if len(ft) > 2 else 8.0
    need = MIN_PRINTS_FRAC * LOOKBACK_DAYS * 24 / max(spacing_h, 1.0)
    f7 = np.where(n7 >= need, s7 * 365.0 / LOOKBACK_DAYS, np.nan)
    d = pd.DataFrame({"b": b, "S": S, "F": F, "fund_cash": fund_cash, "fund_rate": fund_rate, "f7": f7})
    d["S_next"] = d["S"].shift(-1)
    d["F_next"] = d["F"].shift(-1)
    d.loc[d.index[-1], "fund_cash"] = np.nan                 # last row has no complete interval
    d["valid"] = d[["S", "F", "S_next", "F_next", "fund_cash"]].notna().all(axis=1)
    d["valid"] &= np.isfinite(d["f7"]).to_numpy()
    # a print without a perp price inside an interval invalidates that interval
    missing_px = np.isnan(fF)
    if missing_px.any():
        cs_m = np.r_[0, np.cumsum(missing_px)]
        d.loc[(cs_m[kn] - cs_m[kb]) > 0, "valid"] = False
    return d.iloc[:-1].reset_index(drop=True)


# ----------------------------------------------------------------------------- strategy
def positions_conditional(f7: np.ndarray, valid: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    pos = np.zeros(len(f7), bool)
    on = False
    for i in range(len(f7)):
        if not valid[i]:
            on = False                       # forced flat over unusable intervals
            continue
        if not on and f7[i] >= entry:
            on = True
        elif on and f7[i] < exit_:
            on = False
        pos[i] = on
    return pos


def simulate(d: pd.DataFrame, pos: np.ndarray, p: Params = Params()) -> pd.DataFrame:
    """Daily P&L per unit of entry notional, decomposed. pos[i] = held over interval i."""
    n = len(d)
    S, F = d["S"].to_numpy(float), d["F"].to_numpy(float)
    Sn, Fn = d["S_next"].to_numpy(float), d["F_next"].to_numpy(float)
    fc = d["fund_cash"].to_numpy(float)
    basis = np.zeros(n)
    fund = np.zeros(n)
    cost = np.zeros(n)
    notional = np.zeros(n)
    episode = np.full(n, -1)
    resets = np.zeros(n, bool)
    ep = -1
    S_entry = F_ref = np.nan
    for i in range(n):
        prev = pos[i - 1] if i > 0 else False
        if pos[i] and not prev:                               # enter at b_i
            ep += 1
            S_entry, F_ref = S[i], F[i]
            cost[i] += p.side_cost
        elif pos[i] and prev and F[i] / F_ref - 1.0 >= p.reset_move:
            cost[i] += 2 * p.side_cost * S[i] / S_entry
            F_ref = F[i]
            resets[i] = True
        if pos[i]:
            u = 1.0 / S_entry
            basis[i] = u * ((Sn[i] - S[i]) - (Fn[i] - F[i]))
            fund[i] = u * fc[i]
            notional[i] = S[i] / S_entry
            episode[i] = ep
            nxt = pos[i + 1] if i + 1 < n else False
            if not nxt:                                       # exit at b_{i+1}
                cost[i] += p.side_cost * Sn[i] / S_entry
    out = pd.DataFrame({"b": d["b"].to_numpy(), "pos": pos.astype(int), "basis": basis, "funding": fund,
                        "cost": cost, "notional": notional, "episode": episode, "reset": resets})
    out["net"] = out["basis"] + out["funding"] - out["cost"]
    return out


def run_asset(d: pd.DataFrame, strategy: str, p: Params = Params()) -> pd.DataFrame:
    valid = d["valid"].to_numpy()
    if strategy == "U":
        pos = valid.copy()
    else:
        pos = positions_conditional(d["f7"].to_numpy(float), valid, p.entry, p.exit)
    return simulate(d, pos, p)


# ----------------------------------------------------------------------------- portfolio and statistics
def portfolio(sims: dict, frames: dict) -> pd.DataFrame:
    """Daily portfolio: reserved capital = 1.5 per eligible asset (valid that day); deployed = 1.5 x active."""
    rows = []
    for a, s in sims.items():
        x = s[["b", "net", "basis", "funding", "cost", "pos", "notional"]].copy()
        x["eligible"] = frames[a]["valid"].to_numpy().astype(int)
        x["asset"] = a
        rows.append(x)
    z = pd.concat(rows)
    g = z.groupby("b").agg(net=("net", "sum"), basis=("basis", "sum"), funding=("funding", "sum"),
                           cost=("cost", "sum"), active=("pos", "sum"), eligible=("eligible", "sum"),
                           deployed_notional=("notional", "sum"))
    g = g[g["eligible"] > 0]
    g["ret_reserved"] = g["net"] / (CAPITAL_PER_NOTIONAL * g["eligible"])
    g["ret_deployed"] = np.where(g["active"] > 0, g["net"] / (CAPITAL_PER_NOTIONAL * g["active"].clip(lower=1)), 0.0)
    return g.reset_index()


def block_bootstrap_ci(x: np.ndarray, block: int = BOOT_BLOCK, B: int = BOOT_B, seed: int = SEED) -> tuple:
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 * block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    st = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return float("nan")
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float(np.max(peak - cum))


def summarize(port: pd.DataFrame, sims: dict, frames: dict, label: str) -> dict:
    r = port["ret_reserved"].to_numpy()
    lo, hi = block_bootstrap_ci(r)
    dev = port["b"] < DEV_END
    act = port["active"] > 0
    dep_days = float((CAPITAL_PER_NOTIONAL * port["active"]).sum())
    per_asset_net = {a: float(s["net"].sum()) for a, s in sims.items()
                     if frames[a]["valid"].sum() >= MIN_ASSET_DAYS}
    tot = port[["basis", "funding", "cost"]].sum()
    return {
        "strategy": label, "days": int(len(port)), "first": str(port["b"].min().date()),
        "last": str(port["b"].max().date()),
        "ann_reserved": 365 * float(r.mean()), "ann_reserved_lo": 365 * lo, "ann_reserved_hi": 365 * hi,
        "ann_deployed": 365 * float(port["net"].sum() / dep_days) if dep_days > 0 else float("nan"),
        "ann_reserved_dev": 365 * float(port.loc[dev, "ret_reserved"].mean()),
        "ann_reserved_test": 365 * float(port.loc[~dev, "ret_reserved"].mean()),
        "ann_deployed_dev": _dep(port[dev]), "ann_deployed_test": _dep(port[~dev]),
        "share_days_active": float((port["active"] / port["eligible"]).mean()),
        "funding_notional_days": float(tot["funding"]), "basis_notional_days": float(tot["basis"]),
        "cost_notional_days": float(tot["cost"]),
        "max_dd_reserved": max_drawdown(np.cumsum(r)),
        "worst_day_reserved": float(r.min()) if len(r) else float("nan"),
        "assets_scored": len(per_asset_net), "assets_net_positive": int(sum(v > 0 for v in per_asset_net.values())),
        "episodes": int(sum(s["episode"].max() + 1 for s in sims.values())),
        "resets": int(sum(s["reset"].sum() for s in sims.values())),
        "active_days_pct": float(act.mean()),
    }


def _dep(p: pd.DataFrame) -> float:
    dep = float((CAPITAL_PER_NOTIONAL * p["active"]).sum())
    return 365 * float(p["net"].sum() / dep) if dep > 0 else float("nan")


def verdict(s: dict) -> dict:
    g1 = bool(s["ann_reserved_lo"] > 0)
    g2 = bool(s["ann_reserved_dev"] > 0 and s["ann_reserved_test"] > 0)
    g3 = bool(s["assets_scored"] > 0 and s["assets_net_positive"] >= BREADTH * s["assets_scored"])
    g4 = bool(s["ann_deployed"] >= HURDLE)
    ok = g1 and g2 and g3 and g4
    return {"G1_net_ci_positive": g1, "G2_dev_and_test_positive": g2, "G3_breadth": g3,
            "G4_beats_cash_hurdle": g4, "verdict": "CARRY_EDGE" if ok else
            ("CARRY_POSITIVE_BELOW_HURDLE" if (g1 and g2 and g3) else "NO_CARRY_EDGE")}


def episode_table(sims: dict) -> pd.DataFrame:
    rows = []
    for a, s in sims.items():
        for e, g in s[s["episode"] >= 0].groupby("episode"):
            rows.append({"asset": a, "episode": int(e), "start": g["b"].iloc[0], "days": len(g),
                         "funding": g["funding"].sum(), "basis": g["basis"].sum(), "cost": g["cost"].sum(),
                         "net": g["net"].sum(), "worst_cum": float(np.min(np.cumsum(g["net"].to_numpy()))),
                         "resets": int(g["reset"].sum())})
    return pd.DataFrame(rows)


def funding_profile(frames: dict, funding: dict) -> pd.DataFrame:
    rows = []
    for a, d in frames.items():
        f = funding[a]
        f = f[(f.index > d["b"].min()) & (f.index <= d["b"].max())]
        spacing = float(np.median(np.diff(_ns(f.index))) / 3.6e12) if len(f) > 2 else np.nan
        per_year = 365 * 24 / spacing if spacing and np.isfinite(spacing) else np.nan
        rows.append({"asset": a, "first": d["b"].min(), "last": d["b"].max(), "valid_days": int(d["valid"].sum()),
                     "interval_h": spacing, "mean_ann": float(f.mean() * per_year),
                     "median_ann": float(f.median() * per_year), "share_negative": float((f < 0).mean()),
                     "share_at_base_1bp": float((np.abs(f - 1e-4) < 1e-7).mean()),
                     "f7_p10": float(np.nanpercentile(d["f7"], 10)), "f7_p90": float(np.nanpercentile(d["f7"], 90))})
    return pd.DataFrame(rows)


def run_all(frames: dict, funding: dict, p: Params = Params(), label: str = "C") -> tuple:
    sims = {a: run_asset(d, label, p) for a, d in frames.items()}
    port = portfolio(sims, frames)
    return sims, port, summarize(port, sims, frames, label)


# ----------------------------------------------------------------------------- synthetic
def synthetic(n_days: int = 500, seed: int = 0, assets=("AAA", "BBB", "CCC"), base_rate: float = 1e-4,
              regime_amp: float = 2e-4, basis_bp: float = 5.0) -> tuple:
    """Spot 1m random walk; perp = spot * (1 + basis); 8-hourly funding following a persistent regime
    (base_rate + regime_amp * AR(1) in [-1, 1]); basis tracks the funding regime."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    t = pd.date_range("2025-01-01 00:01", periods=n, freq="1min", tz="UTC")
    spot, perp, fund = {}, {}, {}
    nf = n_days * 3
    tf = pd.date_range("2025-01-01 08:00", periods=nf, freq="8h", tz="UTC")
    for j, a in enumerate(assets):
        x = np.zeros(nf)
        for i in range(1, nf):
            x[i] = 0.98 * x[i - 1] + 0.2 * rng.standard_normal()
        x = np.tanh(x)
        f = base_rate + regime_amp * x
        lp = np.log(50.0 * (j + 1)) + np.cumsum(rng.normal(0, 6e-4, n))
        reg_min = np.repeat(x, 480)[:n]
        bas = basis_bp / 1e4 * reg_min + rng.normal(0, 1e-4, n)
        spot[a] = pd.Series(np.exp(lp), index=t)
        perp[a] = pd.Series(np.exp(lp) * (1 + bas), index=t)
        fund[a] = pd.Series(f, index=tf)
    return spot, perp, fund


def selftest() -> str:
    """Constant funding, zero basis: U's net must equal funding cash minus entry/exit costs exactly."""
    t = pd.date_range("2025-01-01 00:01", periods=40 * 1440, freq="1min", tz="UTC")
    rng = np.random.default_rng(1)
    px = pd.Series(np.exp(np.log(100) + np.cumsum(rng.normal(0, 5e-4, len(t)))), index=t)
    f = pd.Series(1e-4, index=pd.date_range("2025-01-01 08:00", periods=120, freq="8h", tz="UTC"))
    d = daily_frame(px, px, f)
    s = run_asset(d, "U")
    expect = s["funding"].sum() - SIDE_COST - SIDE_COST * d["S_next"].iloc[-1] / d["S"].iloc[0]
    assert abs(s["basis"].sum()) < 1e-12
    assert abs(s["net"].sum() - expect) < 1e-12
    return f"{VERSION}: accounting selftest ok"
