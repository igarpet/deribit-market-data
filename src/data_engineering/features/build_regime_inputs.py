"""
Build regime_input_state: anchors, displacement and persistence statistics
(regime_inputs.py) from time_series_feature_state.

    python src/data_engineering/features/build_regime_inputs.py

Run after build_features.py. One row per state; all inputs are causal.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data_engineering.catalog.database import (
    initialize_metadata,
    open_research_database,
)

from data_engineering.features.market_time_series import (
    drop_persistent_relation,
)

from data_engineering.features.regime_inputs import (
    EWMA_HALF_LIVES,
    REGIME_INPUT_VERSION,
    ROLLING_VWAP_WINDOWS,
    compute_regime_inputs,
)


SOURCE_TABLE = "time_series_feature_state"
TARGET_TABLE = "regime_input_state"

SOURCE_COLUMNS = [
    "state_id", "state_time", "underlying", "session_id",
    "futures_index_price", "return_5m", "realized_vol_1h_annualized",
    "index_return_input_eligible", "perp_integrity_ok",
    "vwap", "volume_usd", "futures_tradable_mid_price",
]


def load_source(connection) -> pd.DataFrame:
    columns = set(connection.execute(f"DESCRIBE {SOURCE_TABLE}").df()["column_name"])
    present = [c for c in SOURCE_COLUMNS if c in columns]
    absent = [c for c in SOURCE_COLUMNS if c not in columns]
    if absent:
        print(f"[REGIME INPUTS] Not in {SOURCE_TABLE}: {absent}")
    frame = connection.execute(
        f"SELECT {', '.join(present)} FROM {SOURCE_TABLE} ORDER BY underlying, state_time"
    ).df()
    frame["state_time"] = pd.to_datetime(frame["state_time"], utc=True)
    return frame


def validate(source: pd.DataFrame, out: pd.DataFrame) -> None:
    problems = []
    if len(out) != len(source):
        problems.append(f"row count {len(out)} != source {len(source)}")
    if out["state_id"].duplicated().any():
        problems.append("duplicate state_id")
    for c in out.columns:
        if out[c].dtype.kind == "f" and np.isinf(out[c]).any():
            problems.append(f"infinite values in {c}")
    for c in [c for c in out.columns if c.startswith("vr_q")]:
        if (out[c].dropna() < 0).any():
            problems.append(f"negative variance ratio in {c}")
    for c in [c for c in out.columns if c.startswith("er_n")]:
        v = out[c].dropna()
        if ((v < 0) | (v > 1 + 1e-12)).any():
            problems.append(f"efficiency ratio outside [0, 1] in {c}")
    if problems:
        raise RuntimeError("Regime input validation failed:\n  " + "\n  ".join(problems))
    print("[REGIME INPUTS] Validation passed.")


def print_summary(out: pd.DataFrame) -> None:
    ready = [f"D_ewma_hl{H}_ready" for H in EWMA_HALF_LIVES]
    ready += [c for c in ["D_session_vwap_ready"] if c in out]
    ready += [f"D_rvwap_{W}_ready" for W in ROLLING_VWAP_WINDOWS if f"D_rvwap_{W}_ready" in out]
    cov = out.groupby("underlying")[ready].mean().mul(100).round(1)
    cov.columns = [c.replace("_ready", "") for c in cov.columns]
    print("\nREGIME INPUT COVERAGE (% of states with a ready displacement)")
    print("=" * 110)
    print(cov.to_string())

    stats = [c for c in out.columns if c.startswith(("vr_q", "er_n"))]
    med = out.groupby("underlying")[stats].median().round(3)
    print("\nPERSISTENCE STATISTICS (median; VR = 1 and mean ER = 1/sqrt(N) under a random walk)")
    print("=" * 110)
    print(med.to_string())

    d = [c for c in out.columns if c.startswith("D_") and not c.endswith("_ready")]
    sd = out.groupby("underlying")[d].std().round(2)
    print("\nDISPLACEMENT STANDARD DEVIATION (raw scale; ~1 if sigma and h are consistent)")
    print("=" * 110)
    print(sd.to_string())


def main() -> None:
    connection = open_research_database()
    try:
        started = time.perf_counter()
        source = load_source(connection)
        out = compute_regime_inputs(source)
        validate(source, out)

        drop_persistent_relation(connection, TARGET_TABLE)
        connection.register("_regime_inputs_df", out)
        connection.execute(f"CREATE TABLE {TARGET_TABLE} AS SELECT * FROM _regime_inputs_df")
        connection.unregister("_regime_inputs_df")

        initialize_metadata(connection)
        connection.execute(
            "INSERT OR REPLACE INTO catalog_metadata VALUES (?, ?)",
            ["regime_input_version", REGIME_INPUT_VERSION],
        )
        print(f"[REGIME INPUTS] {TARGET_TABLE}: {len(out):,} rows in {time.perf_counter() - started:.1f}s")
        print_summary(out)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
