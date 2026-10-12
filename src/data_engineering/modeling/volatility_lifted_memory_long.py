"""
Volatility Dynamics 12: Module 11 re-run on the two-year 5-minute history, with
two added benchmarks (VD 026).

Why: Module 11 (research DB, 21 days) returned INSUFFICIENT_DATA (5 test days,
15 unique onset times). Its descriptive results also showed (a) the rough
kernel C tied with two plain EMAs E, and (b) H was not identified by
validation log loss. Module 12 uses the same pipeline and rules on about two
years of Binance-spot 5m closes (tools/download_price_history.py), and adds:

  U_unrestricted   current + every factor EMA of the N-grid as a separate
                   regressor (N chosen on validation). Any fixed kernel
                   weighting (C, F) is a restricted version of U, so U is the
                   ceiling for "shape of the memory kernel".
  K_climatology    asset fixed effects + time-of-day harmonics only (no
                   volatility input). Reference for skill.

Everything else is Module 11 unchanged (volatility_lifted_memory v0.1.1):
de-clocking, train-only thresholds, causal observed-weight EMAs, 80% / 15 min
history readiness, censored labels, 60/20/20 chronological UTC-day split,
validation-only selection, paired UTC-day bootstrap.

Sample (pre-registered): sessions strictly before 2026-09-19 00:00 UTC, so the
Module 11 window (seen when designing this module) is excluded. Assets with
fewer than MIN_DAYS of history are excluded (HYPE has no long history).

Decision at 60 minutes (pre-registered; q975 = upper 97.5% bootstrap bound of
the paired mean log-loss difference, negative = left model better):
  Adequacy  >= 20 unique test onset times and >= 10 test days, else
            INSUFFICIENT_DATA.
  Flags     memory_vs_lags   : best of {E, F, C, U} - B, q975 < 0
            rough_specific   : C - B, C - E and C - F all q975 < 0
            beyond_two_ema   : U - E q975 < 0
            skill            : best model - K q975 < 0
  Status (first that applies)
            PASS_ROUGH_SPECIFIC                  rough_specific
            MULTISCALE_MEMORY_NOT_ROUGH_SPECIFIC beyond_two_ema
            TWO_EMA_MEMORY_SUFFICIENT            E - B q975 < 0
            SKILL_WITHOUT_LONG_MEMORY            skill
            NO_SKILL                             otherwise
  The Module 11 rule (C beats B and F) is also reported for continuity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from data_engineering.modeling import volatility_lifted_memory as lm


VERSION = "0.1.0"
SAMPLE_END = pd.Timestamp("2026-09-19 00:00", tz="UTC")     # exclusive; Module 11 window excluded
MIN_DAYS = 365
BOOT = 1000
MODELS = ("K_climatology", "A_current", "B_fixed", "E_exponential", "F_equal_multiscale", "C_lifted",
          "U_unrestricted")


def load_returns(directory: Path, assets: Sequence[str], end: pd.Timestamp = SAMPLE_END,
                 min_days: int = MIN_DAYS) -> tuple[pd.DataFrame, dict]:
    """state_time (bar end, 5m grid), underlying, return_5m from <ASSET>.csv.gz closes.
    Returns are log differences of consecutive 5m bars only; a gap gives NaN."""
    frames, excluded = [], {}
    for a in assets:
        p = Path(directory) / f"{a}.csv.gz"
        if not p.exists():
            excluded[a] = "missing file"
            continue
        h = pd.read_csv(p, usecols=lambda c: c in ("state_time", "close"))
        h["state_time"] = pd.to_datetime(h["state_time"], utc=True)
        h = h[h["state_time"] < end].drop_duplicates("state_time").sort_values("state_time")
        h = h[(h["state_time"].dt.minute % 5 == 0) & (h["state_time"].dt.second == 0)]
        c = pd.to_numeric(h["close"], errors="coerce")
        consec = h["state_time"].diff().eq(pd.Timedelta(minutes=5))
        r = np.log(c / c.shift(1)).where(consec & (c > 0) & (c.shift(1) > 0))
        days = int(h["state_time"].dt.floor("D").nunique())
        if days < min_days:
            excluded[a] = f"{days} days"
            continue
        frames.append(pd.DataFrame({"state_time": h["state_time"].to_numpy(), "underlying": a,
                                    "return_5m": r.to_numpy(float)}))
    if not frames:
        raise FileNotFoundError(f"No usable history in {directory}")
    return pd.concat(frames, ignore_index=True), excluded


def unrestricted_columns(N: int) -> list[str]:
    return [f"u_n{N}_m{j}" for j in range(N)]


def add_unrestricted(frame: pd.DataFrame, em: np.ndarray, N: int) -> list[str]:
    cols = unrestricted_columns(N)
    for j, c in enumerate(cols):
        frame[c] = em[:, j].astype(np.float32)
    return cols


def decide(boot: pd.DataFrame, n_unique_onsets: int, n_test_days: int, horizon: int = lm.PRIMARY_HORIZON) -> dict:
    """boot: rows with columns horizon_min, comparison ('X - Y'), delta_logloss, bootstrap_q975."""
    b = boot[boot["horizon_min"] == horizon].set_index("comparison")

    def ok(comp):
        return bool(comp in b.index and b.loc[comp, "delta_logloss"] < 0 and b.loc[comp, "bootstrap_q975"] < 0)

    best = str(b.attrs.get("best_model", ""))
    flags = {
        "memory_vs_lags": any(ok(f"{m} - B_fixed") for m in ("E_exponential", "F_equal_multiscale", "C_lifted",
                                                            "U_unrestricted")),
        "rough_specific": ok("C_lifted - B_fixed") and ok("C_lifted - E_exponential") and ok("C_lifted - F_equal_multiscale"),
        "beyond_two_ema": ok("U_unrestricted - E_exponential"),
        "two_ema_vs_lags": ok("E_exponential - B_fixed"),
        "skill": any(ok(f"{m} - K_climatology") for m in MODELS if m != "K_climatology"),
        "vd11_rule_pass": ok("C_lifted - B_fixed") and ok("C_lifted - F_equal_multiscale"),
    }
    if n_unique_onsets < 20 or n_test_days < 10:
        status = "INSUFFICIENT_DATA"
    elif flags["rough_specific"]:
        status = "PASS_ROUGH_SPECIFIC"
    elif flags["beyond_two_ema"]:
        status = "MULTISCALE_MEMORY_NOT_ROUGH_SPECIFIC"
    elif flags["two_ema_vs_lags"]:
        status = "TWO_EMA_MEMORY_SUFFICIENT"
    elif flags["skill"]:
        status = "SKILL_WITHOUT_LONG_MEMORY"
    else:
        status = "NO_SKILL"
    return {"status": status, "primary_horizon_min": horizon, "n_unique_test_onset_times": int(n_unique_onsets),
            "n_test_days": int(n_test_days), **flags}


COMPARISONS = (
    ("C_lifted", "B_fixed"), ("C_lifted", "E_exponential"), ("C_lifted", "F_equal_multiscale"),
    ("C_lifted", "U_unrestricted"), ("U_unrestricted", "E_exponential"), ("E_exponential", "B_fixed"),
    ("F_equal_multiscale", "B_fixed"), ("U_unrestricted", "B_fixed"), ("B_fixed", "A_current"),
    ("A_current", "K_climatology"), ("B_fixed", "K_climatology"), ("E_exponential", "K_climatology"),
    ("F_equal_multiscale", "K_climatology"), ("C_lifted", "K_climatology"), ("U_unrestricted", "K_climatology"),
)


def registration() -> dict:
    return {"version": VERSION, "base_module": "volatility_lifted_memory", "sample_end_exclusive": str(SAMPLE_END),
            "min_days": MIN_DAYS, "boot": BOOT, "models": list(MODELS), "h_grid": list(lm.H_GRID),
            "n_grid": list(lm.N_GRID), "ridge_grid": list(lm.RIDGE_GRID), "horizons": list(lm.HORIZONS),
            "primary_horizon": lm.PRIMARY_HORIZON, "history_coverage": lm.MIN_HISTORY_COVERAGE,
            "status_order": ["INSUFFICIENT_DATA", "PASS_ROUGH_SPECIFIC", "MULTISCALE_MEMORY_NOT_ROUGH_SPECIFIC",
                             "TWO_EMA_MEMORY_SUFFICIENT", "SKILL_WITHOUT_LONG_MEMORY", "NO_SKILL"]}


def selftest() -> dict:
    rows = []
    for comp in COMPARISONS:
        rows.append({"horizon_min": 60, "comparison": f"{comp[0]} - {comp[1]}", "delta_logloss": 0.001,
                     "bootstrap_q975": 0.002})
    b = pd.DataFrame(rows)
    assert decide(b, 100, 100)["status"] == "NO_SKILL"
    b.loc[b.comparison == "E_exponential - K_climatology", ["delta_logloss", "bootstrap_q975"]] = (-0.01, -0.005)
    assert decide(b, 100, 100)["status"] == "SKILL_WITHOUT_LONG_MEMORY"
    b.loc[b.comparison == "E_exponential - B_fixed", ["delta_logloss", "bootstrap_q975"]] = (-0.01, -0.005)
    assert decide(b, 100, 100)["status"] == "TWO_EMA_MEMORY_SUFFICIENT"
    b.loc[b.comparison == "U_unrestricted - E_exponential", ["delta_logloss", "bootstrap_q975"]] = (-0.01, -0.005)
    assert decide(b, 100, 100)["status"] == "MULTISCALE_MEMORY_NOT_ROUGH_SPECIFIC"
    for c in ("C_lifted - B_fixed", "C_lifted - E_exponential", "C_lifted - F_equal_multiscale"):
        b.loc[b.comparison == c, ["delta_logloss", "bootstrap_q975"]] = (-0.01, -0.005)
    d = decide(b, 100, 100)
    assert d["status"] == "PASS_ROUGH_SPECIFIC" and d["vd11_rule_pass"]
    assert decide(b, 10, 100)["status"] == "INSUFFICIENT_DATA"
    return {"version": VERSION, "ok": True}
