"""
Volatility Risk Premium branch, step 4 (VRP 009): conditional short volatility,
straddle and defined-risk iron fly, held to settlement.

Question
--------
VRP 02: unconditionally, selling the ATM straddle at the bid earns about zero
(+1 bp of forward), while at the mid the seller would have earned roughly +14%
of premium. Is the premium concentrated where implied variance is high
relative to a causal physical forecast? If so, a seller who trades only then,
in defined-risk form, may clear the spread.

Forecast (VRP-local; no other branch's model, threshold or output is used)
--------------------------------------------------------------------------
Index 5m log returns r_t (time_series_feature_state.return_5m). Clock
s^2(slot) = mean r^2 by 5-minute UTC slot over DEVELOPMENT sessions only,
smoothed over 7 slots and normalized to mean 1. De-clocked squared returns
x_t = r_t^2 / s^2(slot_t). Level v_t = mean of causal EMAs of x with
half-lives 1h, 4h and 24h (missing bars ignored; at least 2 days of history
required). Forecast variance to expiry:
    V_fc(t) = v_t * sum_{bars j in (t, T]} s^2(slot_j)
Implied variance to expiry: V_iv(t) = sigma_straddle_mid(t)^2 * T, with
sigma_straddle_mid the Black-76 vol that reprices the ATM straddle mid (VRP 02).
Wedge: w(t) = ln(V_iv / V_fc).

Structures (entries: VRP 02's hourly ATM straddles, TTE 2-20 h)
    Short straddle: sell call and put at K.
    Short iron fly: short straddle + buy the call at K_up and the put at K_dn,
    the listed strikes nearest F exp(+/- Z sigma_straddle sqrt(T)), Z = 1.5,
    with K_dn < K < K_up and both wing legs two-sided.
Prices: executable (sell at bid, buy at ask) and mid (bound). Fees per leg
min(0.0003 F, 0.125 premium). Settlement: exchange delivery price.
Units native (coin for inverse families), as VRP 02/03.
    Iron fly P&L  = credit - fees - |S_T - K| + (S_T - K_up)^+ + (K_dn - S_T)^+
    Max risk      = max(K_up - K, K - K_dn) - credit + fees   (in native units at F)
    Return on risk RoR = P&L / max risk.  Straddle: P&L over credit.

Terciles of w: thresholds = 1/3 and 2/3 quantiles of w over DEVELOPMENT entries
(sessions before 2026-11-07), pooled across assets.

Pre-registered decision (CONFIRMATION only)
-------------------------------------------
Freeze: after the last development session (2026-11-06), run with
VRP04_FREEZE=1. This writes 04/04_frozen_spec.json (clock profiles,
thresholds, Z, half-lives) and refuses to overwrite it.
Confirm: sessions >= 2026-11-07, frozen spec, binding at >= 20 sessions with a
top-tercile entry.
Statistic: mean over sessions of the session-mean RoR of the top-tercile short
iron fly at executable prices; 95% session bootstrap (4,000 draws, seed 9004).
    CI > 0 -> CONDITIONAL_PREMIUM_CONFIRMED   CI < 0 -> REJECTED   else INCONCLUSIVE
Development runs report the same statistics by tercile and are descriptive.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_engineering.modeling import vrp_long_straddle as ls
from data_engineering.modeling import vrp_execution as ve


VERSION = "0.1.0"
CONF_START = ls.VRP_CONFIRMATION_START                      # 2026-11-07
HALF_LIVES_BARS = (12, 48, 288)
MIN_HISTORY_BARS = 576
Z_WING = 1.5
N_BOOT = 4000
SEED = 9004
MIN_CONFIRM_SESSIONS = 20


# --------------------------------------------------------------------------
# Forecast
# --------------------------------------------------------------------------

def index_panel(ts: pd.DataFrame) -> dict:
    """{asset: Series of 5m index log returns on a regular grid (bar END times)}."""
    out = {}
    f = ts.copy()
    f["state_time"] = pd.to_datetime(f["state_time"], utc=True)
    for a, g in f.groupby("underlying"):
        s = g.drop_duplicates("state_time").set_index("state_time")["return_5m"].astype(float).sort_index()
        grid = pd.date_range(s.index.min(), s.index.max(), freq="5min")
        out[a] = s.reindex(grid)
    return out


def slot_of(times: pd.DatetimeIndex) -> np.ndarray:
    t0 = times - pd.Timedelta(minutes=5)
    return np.asarray((t0.hour * 60 + t0.minute) // 5, int)


def clock_profile(r: pd.Series, dev_end: pd.Timestamp = CONF_START) -> np.ndarray:
    """s^2(slot), development bars only, 7-slot circular smoothing, mean 1."""
    rr = r[r.index < dev_end]
    sq = pd.Series(rr.to_numpy() ** 2).groupby(slot_of(rr.index)).mean().reindex(range(288)).to_numpy()
    sq = np.where(np.isfinite(sq) & (sq > 0), sq, np.nanmean(sq))
    ext = np.r_[sq[-3:], sq, sq[:3]]
    sm = np.convolve(ext, np.ones(7) / 7, mode="valid")
    return sm / sm.mean()


def level_forecast(r: pd.Series, s2: np.ndarray) -> pd.Series:
    """v_t: mean of causal EMAs of de-clocked squared returns; NaN with < 2 days of history."""
    x = pd.Series(r.to_numpy() ** 2 / s2[slot_of(r.index)], index=r.index)
    emas = [x.ewm(halflife=h, ignore_na=True, adjust=True).mean() for h in HALF_LIVES_BARS]
    v = pd.concat(emas, axis=1).mean(axis=1)
    hist = x.notna().cumsum()
    return v.where(hist >= MIN_HISTORY_BARS)


def remaining_clock(entry_times: pd.Series, minutes: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """sum of s^2 over the bars ending in (t, t + minutes]."""
    cs = np.r_[0.0, np.cumsum(np.tile(s2, 3))]
    start = slot_of(pd.DatetimeIndex(entry_times)) + 1          # slot of the first future bar
    start = start % 288
    n = np.round(np.asarray(minutes, float) / 5.0).astype(int)
    return cs[start + n] - cs[start]


def attach_wedge(st: pd.DataFrame, rets: dict, profiles: dict) -> pd.DataFrame:
    x = st.copy()
    x["V_fc"] = np.nan
    for a, g in x.groupby("underlying"):
        if a not in rets or a not in profiles:
            continue
        v = level_forecast(rets[a], profiles[a])
        vt = v.reindex(pd.DatetimeIndex(g["state_time"])).to_numpy(float)
        x.loc[g.index, "V_fc"] = vt * remaining_clock(g["state_time"], g["minutes_to_expiry"].to_numpy(float),
                                                      profiles[a])
    x["V_iv"] = x["iv_straddle_mid"] ** 2 * x["T"]
    with np.errstate(divide="ignore", invalid="ignore"):
        x["wedge"] = np.log(x["V_iv"] / x["V_fc"])
    return x


def prepare_legs(chain: pd.DataFrame, confirm: bool) -> pd.DataFrame:
    """VRP 02's leg preparation (hourly grid, TTE 2-20h, USD prices, two-sided flag) for the
    development window (sessions < CONF_START) or the confirmation window (sessions >= CONF_START)."""
    q = chain.copy()
    q["state_time"] = pd.to_datetime(q["state_time"], utc=True)
    for c in ("strike", "option_forward_price", "minutes_to_expiry", "best_bid_price", "best_ask_price"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["is_call"] = q["option_type"].astype(str).str.lower().str[0].eq("c")
    q["inverse"] = ls.op.is_inverse(q["instrument_type"])
    F = q["option_forward_price"].to_numpy(float)
    q["bid_usd"] = ls.op.to_usd(q["best_bid_price"].to_numpy(float), F, q["inverse"])
    q["ask_usd"] = ls.op.to_usd(q["best_ask_price"].to_numpy(float), F, q["inverse"])
    mins = q["minutes_to_expiry"]
    keep = ((q["state_time"].dt.minute % ls.ENTRY_GRID_MIN == 0) & (q["state_time"].dt.second == 0)
            & (mins >= ls.MIN_TTE_MIN) & (mins <= ls.MAX_TTE_MIN) & np.isfinite(F) & (F > 0) & np.isfinite(q["strike"]))
    q = q[keep].copy()
    q["two_sided"] = (q["bid_usd"] > 0) & (q["ask_usd"] >= q["bid_usd"])
    q["expiry_time"] = (q["state_time"] + pd.to_timedelta(q["minutes_to_expiry"].round(), unit="min")).dt.round("5min")
    q["session"] = q["expiry_time"].dt.strftime("%Y-%m-%d")
    sd = pd.to_datetime(q["session"], utc=True)
    q = q[(sd >= CONF_START) if confirm else (sd < CONF_START)]
    return q.reset_index(drop=True)


# --------------------------------------------------------------------------
# Wings and P&L
# --------------------------------------------------------------------------

def attach_wings(st: pd.DataFrame, legs: pd.DataFrame, z: float = Z_WING) -> pd.DataFrame:
    """Nearest two-sided strikes to F exp(+/- z sigma sqrt(T)) beyond K, per entry."""
    x = st.copy()
    two = legs[legs["two_sided"]]
    calls = two[two["is_call"]][["state_time", "option_family_id", "strike", "bid_usd", "ask_usd"]]
    puts = two[~two["is_call"]][["state_time", "option_family_id", "strike", "bid_usd", "ask_usd"]]
    key = ["state_time", "option_family_id"]
    width = z * x["iv_straddle_mid"] * np.sqrt(x["T"])
    x["K_up_target"] = x["F"] * np.exp(width)
    x["K_dn_target"] = x["F"] * np.exp(-width)
    for name, book, tgt, cond in (("up", calls, "K_up_target", lambda k, K: k > K), ("dn", puts, "K_dn_target", lambda k, K: k < K)):
        m = x[key + ["K", tgt]].reset_index().merge(book, on=key, how="left")
        m = m[cond(m["strike"], m["K"])]
        m["dist"] = np.abs(np.log(m["strike"] / m[tgt]))
        m = m.sort_values(["index", "dist", "strike"]).drop_duplicates("index").set_index("index")
        x[f"K_{name}"] = m["strike"]
        x[f"bid_usd_{name}"] = m["bid_usd"]
        x[f"ask_usd_{name}"] = m["ask_usd"]
    return x


def _native(x: pd.DataFrame, col: str) -> np.ndarray:
    usd = x[col].to_numpy(float)
    return np.where(x["inverse"].to_numpy(bool), usd / x["F"].to_numpy(float), usd)


def short_pnl(x: pd.DataFrame, prices: str = "exec") -> pd.DataFrame:
    """Short straddle and short iron fly outcomes per entry (native units)."""
    inv, F, ST = x["inverse"].to_numpy(bool), x["F"].to_numpy(float), x["S_T"].to_numpy(float)
    K = x["K"].to_numpy(float)
    unit = np.where(inv, ST, 1.0)                                    # coin payoff = USD payoff / S_T

    def px(leg, side):
        b, a = _native(x, f"bid_usd_{leg}"), _native(x, f"ask_usd_{leg}")
        if prices == "mid":
            return 0.5 * (a + b)
        return b if side == "sell" else a

    c_s, p_s = px("call", "sell"), px("put", "sell")
    fee = lambda p: ve.fee_native(p, F, inv)
    credit_sd = c_s + p_s
    fees_sd = fee(c_s) + fee(p_s)
    pay_sd = np.abs(ST - K) / unit
    out = pd.DataFrame(index=x.index)
    out["straddle_pnl"] = credit_sd - fees_sd - pay_sd
    out["straddle_ret_on_credit"] = out["straddle_pnl"] / credit_sd
    out["straddle_pnl_bps_underlying"] = out["straddle_pnl"] / np.where(inv, 1.0, F) * 1e4
    has_w = x[["K_up", "K_dn", "bid_usd_up", "ask_usd_up", "bid_usd_dn", "ask_usd_dn"]].notna().all(axis=1).to_numpy()
    cu, pdn = px("up", "buy"), px("dn", "buy")
    Ku, Kd = x["K_up"].to_numpy(float), x["K_dn"].to_numpy(float)
    credit_if = credit_sd - cu - pdn
    fees_if = fees_sd + fee(cu) + fee(pdn)
    pay_w = (np.maximum(ST - Ku, 0) + np.maximum(Kd - ST, 0)) / unit
    pnl_if = credit_if - fees_if - pay_sd + pay_w
    width = np.maximum(Ku - K, K - Kd) / np.where(inv, F, 1.0)
    risk = width - credit_if + fees_if
    out["ironfly_pnl"] = np.where(has_w, pnl_if, np.nan)
    out["ironfly_max_risk"] = np.where(has_w, risk, np.nan)
    out["ironfly_ror"] = np.where(has_w & (risk > 0), pnl_if / risk, np.nan)
    out["ironfly_pnl_bps_underlying"] = np.where(has_w, pnl_if / np.where(inv, 1.0, F) * 1e4, np.nan)
    out["ironfly_credit_share"] = np.where(has_w, credit_if / credit_sd, np.nan)
    return out


# --------------------------------------------------------------------------
# Terciles, statistics, freeze
# --------------------------------------------------------------------------

def tercile_thresholds(w: pd.Series) -> list[float]:
    w = w[np.isfinite(w)]
    return [float(w.quantile(1 / 3)), float(w.quantile(2 / 3))]


def tercile(w: np.ndarray, thr: list[float]) -> np.ndarray:
    w = np.asarray(w, float)
    t = np.where(w <= thr[0], "low", np.where(w <= thr[1], "mid", "high"))
    return np.where(np.isfinite(w), t, "na")


def tercile_table(x: pd.DataFrame, cols: list[str], n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for t, g in x[x["tercile"] != "na"].groupby("tercile"):
        for c in cols:
            rows.append({"tercile": t, "metric": c, "n_entries": int(g[c].notna().sum()),
                         **ve.session_boot(g[c].to_numpy(float), g["session"].to_numpy(), n_boot)})
    return pd.DataFrame(rows)


def top_minus_bottom(x: pd.DataFrame, col: str, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    s = x[x["tercile"].isin(["high", "low"])].groupby(["session", "tercile"])[col].mean().unstack()
    s = s.dropna()
    if len(s) < 3:
        return {"n_sessions": int(len(s))}
    return ls.bootstrap_mean((s["high"] - s["low"]).to_numpy(), n_boot=n_boot, seed=seed)


def decide(x: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    top = x[(x["tercile"] == "high") & x["ironfly_ror"].notna()]
    st = ve.session_boot(top["ironfly_ror"].to_numpy(float), top["session"].to_numpy(), n_boot)
    if st.get("n_sessions", 0) < MIN_CONFIRM_SESSIONS or "ci_low" not in st:
        v = "INSUFFICIENT_SESSIONS"
    elif st["ci_low"] > 0:
        v = "CONDITIONAL_PREMIUM_CONFIRMED"
    elif st["ci_high"] < 0:
        v = "REJECTED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, **st}


def freeze(path: Path, profiles: dict, thresholds: list[float], dev_last_session: str) -> dict:
    if Path(path).exists():
        raise FileExistsError(f"{path} exists: the frozen spec is never overwritten")
    spec = {"version": VERSION, "frozen_at_utc": str(pd.Timestamp.now(tz="UTC")), "dev_last_session": dev_last_session,
            "confirmation_start": str(CONF_START.date()), "thresholds": thresholds, "z_wing": Z_WING,
            "half_lives_bars": list(HALF_LIVES_BARS), "min_history_bars": MIN_HISTORY_BARS,
            "profiles": {a: [float(v) for v in p] for a, p in profiles.items()}}
    Path(path).write_text(json.dumps(spec, indent=2))
    return spec


def load_frozen(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if spec["version"] != VERSION or spec["z_wing"] != Z_WING or tuple(spec["half_lives_bars"]) != HALF_LIVES_BARS:
        raise ValueError("Frozen spec does not match the module constants: a change needs a new version and window")
    spec["profiles"] = {a: np.asarray(p, float) for a, p in spec["profiles"].items()}
    return spec


def registration() -> dict:
    return {"version": VERSION, "confirmation_start": str(CONF_START.date()), "half_lives_bars": list(HALF_LIVES_BARS),
            "min_history_bars": MIN_HISTORY_BARS, "z_wing": Z_WING, "n_boot": N_BOOT, "seed": SEED,
            "min_confirm_sessions": MIN_CONFIRM_SESSIONS,
            "decision": "confirmation only: top-tercile short iron fly at executable prices, session-bootstrap CI of mean RoR"}


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def selftest() -> dict:
    # Remaining clock: flat profile -> number of bars
    s2 = np.ones(288)
    t = pd.Series(pd.to_datetime(["2026-10-01 20:00"], utc=True))
    assert np.isclose(remaining_clock(t, np.array([720.0]), s2)[0], 144.0)
    # Level forecast recovers a constant variance
    idx = pd.date_range("2026-01-01 00:05", periods=288 * 5, freq="5min", tz="UTC")
    r = pd.Series(np.random.default_rng(1).standard_normal(len(idx)) * 0.002, index=idx)
    v = level_forecast(r, s2)
    assert v.iloc[:575].isna().all() and abs(v.iloc[-1] / 0.002 ** 2 - 1) < 0.35
    # Iron fly arithmetic (linear): credit 10 - 2 = 8; S_T at K -> P&L 8 - fees; S_T = K_up + 50 -> 8 - fees - 150 + 50 = -max risk
    x = pd.DataFrame({"inverse": [False, False], "F": [1000.0, 1000.0], "S_T": [1000.0, 1150.0], "K": [1000.0, 1000.0],
                      "bid_usd_call": [5.0, 5.0], "ask_usd_call": [5.0, 5.0], "bid_usd_put": [5.0, 5.0], "ask_usd_put": [5.0, 5.0],
                      "K_up": [1100.0, 1100.0], "K_dn": [900.0, 900.0], "bid_usd_up": [1.0, 1.0], "ask_usd_up": [1.0, 1.0],
                      "bid_usd_dn": [1.0, 1.0], "ask_usd_dn": [1.0, 1.0]})
    o = short_pnl(x)
    fees = 2 * min(0.0003 * 1000, 0.125 * 5) + 2 * min(0.0003 * 1000, 0.125 * 1)    # 0.85
    assert np.allclose(o["ironfly_pnl"], [8 - fees, 8 - fees - 150 + 50])
    assert np.isclose(o["ironfly_max_risk"].iloc[0], 100 - 8 + fees)
    assert np.isclose(o["ironfly_pnl"].iloc[1], -o["ironfly_max_risk"].iloc[1])
    # Terciles
    assert list(tercile(np.array([0.0, 0.5, 1.0, np.nan]), [0.3, 0.6])) == ["low", "mid", "high", "na"]
    return {"version": VERSION, "ok": True}
