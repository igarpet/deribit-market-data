"""
Fit the deterministic clock baseline on a training window and write the
clock-adjusted feature table.

    python src/data_engineering/modeling/build_clock_baseline.py
    python src/data_engineering/modeling/build_clock_baseline.py --train-end 2026-09-28

Training window: sessions (expiry dates) from --train-start to --train-end,
inclusive. Default: the first half of the sessions that have returns, so
the second half is out of sample for the diagnostics.

Outputs
  * <DERIBIT_DATA_ROOT>/models/clock_baseline/<baseline_id>.json
  * table clock_baseline_registry       (one row per fitted baseline)
  * table clock_adjusted_feature_state  (one row per state, all states;
    clock_baseline_in_sample marks rows inside the training window, which
    must not be used for out-of-sample evaluation)

Diagnostics: out-of-sample fit of each location model, calibration of the
scale models, |r| autocorrelation before/after adjustment, and the book
update level by collector version (open item from Notebook 06).
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data_engineering.catalog.database import (
    DATA_ROOT,
    initialize_metadata,
    open_research_database,
)

from data_engineering.features.market_time_series import (
    drop_persistent_relation,
)

from data_engineering.modeling.clock_baseline import (
    CLOCK_BASELINE_VERSION,
    DEFAULT_FEATURE_SPECS,
    ClockBaselineSet,
    transformed_values,
    add_clock_coordinates,
)


SOURCE_TABLE = "time_series_feature_state"
TARGET_TABLE = "clock_adjusted_feature_state"
REGISTRY_TABLE = "clock_baseline_registry"
MODEL_DIR = DATA_ROOT / "models" / "clock_baseline"

BASE_COLUMNS = [
    "state_id", "state_time", "underlying", "session_id", "expiry_date",
    "minutes_to_expiry", "is_weekend", "us_dst", "eu_dst", "expiry_type",
    "perp_integrity_ok", "fit_ok", "in_settlement_window",
]

V03_ROLLOUT = pd.Timestamp("2026-10-01 10:05", tz="UTC")


# --------------------------------------------------
# Load
# --------------------------------------------------

def load_feature_frame(connection) -> pd.DataFrame:

    columns = set(
        connection.execute(f"DESCRIBE {SOURCE_TABLE}").df()["column_name"]
    )
    wanted = list(dict.fromkeys(
        BASE_COLUMNS + [s.column for s in DEFAULT_FEATURE_SPECS]
    ))
    required = ["state_id", "state_time", "underlying", "expiry_date", "is_weekend", "us_dst"]
    missing_required = [c for c in required if c not in columns]
    if missing_required:
        raise RuntimeError(
            f"{SOURCE_TABLE} lacks {missing_required}: rebuild with "
            "build_features.py (schema >= 0.5.0)."
        )

    present = [c for c in wanted if c in columns]
    absent = [c for c in wanted if c not in columns]
    if absent:
        print(f"[CLOCK BASELINE] Columns not in {SOURCE_TABLE} (features skipped): {absent}")

    frame = connection.execute(
        f"SELECT {', '.join(present)} FROM {SOURCE_TABLE} ORDER BY underlying, state_time"
    ).df()
    frame["state_time"] = pd.to_datetime(frame["state_time"], utc=True)
    frame["session_key"] = pd.to_datetime(frame["expiry_date"]).dt.strftime("%Y-%m-%d")
    return frame


def training_mask(frame: pd.DataFrame, train_start: str | None, train_end: str | None) -> pd.Series:

    with_returns = sorted(
        frame.loc[pd.to_numeric(frame.get("return_5m"), errors="coerce").notna(), "session_key"].unique()
    )
    if not with_returns:
        raise RuntimeError("No session has index returns.")

    start = train_start or with_returns[0]
    end = train_end or with_returns[max(len(with_returns) // 2 - 1, 0)]
    if end < start:
        raise ValueError(f"--train-end {end} is before --train-start {start}.")

    mask = (frame["session_key"] >= start) & (frame["session_key"] <= end)
    sessions = sorted(frame.loc[mask, "session_key"].unique())
    later = [s for s in with_returns if s > end]
    print(f"[CLOCK BASELINE] Training sessions: {start} → {end} ({len(sessions)} sessions, "
          f"{sum(s in with_returns for s in sessions)} with returns)")
    print(f"[CLOCK BASELINE] Out-of-sample sessions with returns: {len(later)}"
          + (f" ({later[0]} → {later[-1]})" if later else ""))
    return mask


# --------------------------------------------------
# Write
# --------------------------------------------------

def write_outputs(connection, baseline: ClockBaselineSet, adjusted: pd.DataFrame) -> Path:

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{baseline.baseline_id}.json"
    path.write_text(baseline.to_json(), encoding="utf-8")

    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
            baseline_id VARCHAR PRIMARY KEY,
            version VARCHAR,
            train_start TIMESTAMPTZ,
            train_end TIMESTAMPTZ,
            created_at TIMESTAMPTZ,
            is_current BOOLEAN,
            model_json VARCHAR
        )
        """
    )
    connection.execute(f"UPDATE {REGISTRY_TABLE} SET is_current = FALSE")
    connection.execute(
        f"INSERT OR REPLACE INTO {REGISTRY_TABLE} VALUES (?, ?, ?, ?, ?, TRUE, ?)",
        [baseline.baseline_id, baseline.version, baseline.train_start,
         baseline.train_end, baseline.created_at, baseline.to_json()],
    )

    drop_persistent_relation(connection, TARGET_TABLE)
    connection.register("_clock_adjusted_df", adjusted)
    connection.execute(f"CREATE TABLE {TARGET_TABLE} AS SELECT * FROM _clock_adjusted_df")
    connection.unregister("_clock_adjusted_df")

    initialize_metadata(connection)
    for key, value in [
        ("clock_baseline_version", CLOCK_BASELINE_VERSION),
        ("clock_baseline_current_id", baseline.baseline_id),
    ]:
        connection.execute(
            "INSERT OR REPLACE INTO catalog_metadata VALUES (?, ?)", [key, value]
        )
    return path


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate(frame: pd.DataFrame, baseline: ClockBaselineSet, adjusted: pd.DataFrame) -> None:

    problems = []
    if len(adjusted) != len(frame):
        problems.append(f"row count {len(adjusted)} != input {len(frame)}")
    if adjusted["state_id"].duplicated().any():
        problems.append("duplicate state_id")

    data = add_clock_coordinates(frame)
    for fb in baseline.fitted:
        s = fb.feature_spec
        rows = adjusted["underlying"] == fb.underlying
        adj = adjusted.loc[rows, f"{s.name}__clock_adj"]
        if s.transform == "log_abs":
            f = adjusted.loc[rows, f"{s.name}__clock_vol_factor"]
            if not (np.isfinite(f).all() and (f > 0).all()):
                problems.append(f"{fb.underlying}/{s.name}: non-positive or non-finite volatility factor")
            raw = pd.to_numeric(frame.loc[rows, s.column], errors="coerce")
            ins = adjusted.loc[rows, "clock_baseline_in_sample"] & raw.notna()
            mean_f2 = float((f[ins] ** 2).mean())
            if abs(mean_f2 - 1.0) > 1e-6:
                problems.append(f"{fb.underlying}/{s.name}: mean f^2 over training = {mean_f2:.6f}")
            source = raw
        else:
            source = transformed_values(data.loc[rows], s)
        bad = source.notna() & ~np.isfinite(adj)
        if bad.any():
            problems.append(f"{fb.underlying}/{s.name}: {int(bad.sum())} defined values without a finite adjustment")
        leaked = source.isna() & adj.notna()
        if leaked.any():
            problems.append(f"{fb.underlying}/{s.name}: {int(leaked.sum())} adjustments where the source is undefined")

    if problems:
        raise RuntimeError("Clock baseline validation failed:\n  " + "\n  ".join(problems))
    print("[CLOCK BASELINE] Validation passed.")


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_fit_diagnostics(frame: pd.DataFrame, baseline: ClockBaselineSet, adjusted: pd.DataFrame) -> None:

    data = add_clock_coordinates(frame)
    rows = []
    for fb in baseline.fitted:
        s = fb.feature_spec
        g = data[data["underlying"] == fb.underlying]
        a = adjusted.loc[g.index]
        oos = ~a["clock_baseline_in_sample"]
        row = {"underlying": fb.underlying, "feature": s.name, "clock": s.clock,
               "weekend": fb.weekend_terms_used, "n_train": fb.n_train,
               "train_r2": fb.train_r2_location}
        if s.transform == "log_abs":
            y = transformed_values(g, s)
            m = np.log(a[f"{s.name}__clock_vol_factor"]) + fb.vol_factor_const
            r = pd.to_numeric(g[s.column], errors="coerce")
            row["oos_mean_adj2_over_raw2"] = float(
                (a.loc[oos, f"{s.name}__clock_adj"] ** 2).mean() / (r[oos] ** 2).mean()
            ) if oos.any() else np.nan
        else:
            y = transformed_values(g, s)
            m = a[f"{s.name}__clock_mean"]
            row["oos_mean_adj2"] = float((a.loc[oos, f"{s.name}__clock_adj"] ** 2).mean()) if oos.any() else np.nan
        tr = y[~oos].dropna()
        te = (y[oos].notna()) & oos
        if te.sum() > 50 and len(tr) > 0:
            sst = float(((y[te] - tr.mean()) ** 2).sum())
            row["oos_r2"] = 1.0 - float(((y[te] - m[te]) ** 2).sum()) / sst if sst > 0 else np.nan
            row["n_oos"] = int(te.sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\nCLOCK BASELINE FIT")
    print("=" * 125)
    print("train_r2 / oos_r2: location model vs the training mean. oos_mean_adj2: mean squared adjusted value out of")
    print("sample (≈1 if the scale model is calibrated AND the day-level mean is unchanged; > 1 mostly reflects")
    print("day-level drift, which the clock baseline is not meant to remove). oos_mean_adj2_over_raw2: returns,")
    print("ratio of mean squares adjusted/raw out of sample.")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(table.round(3).to_string(index=False))

    if baseline.skipped:
        print("\nSkipped:")
        print(pd.DataFrame(baseline.skipped).to_string(index=False))


def print_acf_diagnostics(frame: pd.DataFrame, adjusted: pd.DataFrame) -> None:

    if "return_5m__clock_adj" not in adjusted:
        return
    rows = []
    for asset, g in frame.groupby("underlying"):
        a = adjusted.loc[g.index]
        grid = pd.date_range(g["state_time"].min(), g["state_time"].max(), freq="5min")
        for label, part in [("in-sample", a["clock_baseline_in_sample"]), ("out-of-sample", ~a["clock_baseline_in_sample"])]:
            if part.sum() < 600:
                continue
            raw = pd.Series(pd.to_numeric(g["return_5m"], errors="coerce").abs().where(part).to_numpy(),
                            index=g["state_time"]).reindex(grid)
            adj = pd.Series(a["return_5m__clock_adj"].abs().where(part).to_numpy(),
                            index=g["state_time"]).reindex(grid)
            row = {"underlying": asset, "window": label}
            for lag in [1, 12, 288]:
                row[f"raw_{lag}"] = raw.autocorr(lag)
                row[f"adj_{lag}"] = adj.autocorr(lag)
            rows.append(row)
    if rows:
        print("\n|r| AUTOCORRELATION BEFORE / AFTER CLOCK ADJUSTMENT")
        print("=" * 100)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def print_book_update_diagnostic(connection, frame: pd.DataFrame) -> None:
    """Notebook 06 open item: BTC/ETH/SOL book-update level shifted between
    the two halves of the sample. Compare by collector version."""

    if "book_update_count" not in frame:
        return
    canon_cols = set(connection.execute("DESCRIBE canonical_market_state").df()["column_name"]) \
        if connection.execute(
            "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'canonical_market_state'"
        ).fetchone()[0] + connection.execute(
            "SELECT COUNT(*) FROM duckdb_views() WHERE view_name = 'canonical_market_state'"
        ).fetchone()[0] else set()

    g = frame.copy()
    g["perp_ok"] = g["perp_integrity_ok"].fillna(False).astype(bool) if "perp_integrity_ok" in g else True
    g = g[g["perp_ok"]]
    if "collector_version" in canon_cols:
        cv = connection.execute(
            "SELECT state_id, COALESCE(collector_version, 'pre-v0.3') AS collector_version FROM canonical_market_state"
        ).df()
        g = g.merge(cv, on="state_id", how="left")
        key = "collector_version"
    else:
        g["period"] = np.where(g["state_time"] >= V03_ROLLOUT, "after 2026-10-01 10:05", "before")
        key = "period"
    g["log_updates"] = np.log1p(pd.to_numeric(g["book_update_count"], errors="coerce"))
    t = g.groupby(["underlying", key]).agg(
        states=("log_updates", "size"),
        median_book_updates=("book_update_count", "median"),
        median_log_updates=("log_updates", "median"),
        first=("state_time", "min"),
        last=("state_time", "max"),
    )
    print("\nBOOK UPDATE LEVEL BY COLLECTOR VERSION (integrity-gated states)")
    print("=" * 110)
    with pd.option_context("display.width", 200):
        print(t.round({"median_book_updates": 1, "median_log_updates": 3}).to_string())


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", default=None, help="First training session (expiry date, YYYY-MM-DD).")
    parser.add_argument("--train-end", default=None,
                        help="Last training session (expiry date, inclusive). Default: first half of sessions with returns.")
    parser.add_argument("--harmonics", type=int, default=None,
                        help="Override the number of Fourier harmonics K of every location model (default 4).")
    args = parser.parse_args()
    specs = DEFAULT_FEATURE_SPECS
    if args.harmonics is not None:
        if args.harmonics < 1:
            raise ValueError("--harmonics must be >= 1.")
        specs = tuple(dataclasses.replace(s, harmonics=args.harmonics) for s in specs)

    connection = open_research_database()
    try:
        frame = load_feature_frame(connection)
        mask = training_mask(frame, args.train_start, args.train_end)

        baseline = ClockBaselineSet.fit(frame, mask, specs=specs)
        adjusted = baseline.transform(frame)
        adjusted.insert(3, "session_key", frame["session_key"].to_numpy())

        validate(frame, baseline, adjusted)
        path = write_outputs(connection, baseline, adjusted)

        print(f"[CLOCK BASELINE] {baseline.baseline_id}: {len(baseline.fitted)} fitted feature baselines")
        print(f"[CLOCK BASELINE] Model: {path}")
        print(f"[CLOCK BASELINE] Table: {TARGET_TABLE} ({len(adjusted):,} rows; "
              f"{int(adjusted['clock_baseline_in_sample'].sum()):,} in sample; "
              f"{int(adjusted['clock_dst_unseen'].sum()):,} with an unseen US DST regime)")

        print_fit_diagnostics(frame, baseline, adjusted)
        print_acf_diagnostics(frame, adjusted)
        print_book_update_diagnostic(connection, frame)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
