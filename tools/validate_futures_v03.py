"""
Post-deployment validation for futures collector v0.3.

Usage (from the repository root, collector environment active):

    python tools/validate_futures_v03.py            # today (UTC)
    python tools/validate_futures_v03.py 2026-10-05 # explicit UTC date

Reads futures/5m/date=YYYY-MM-DD/futures_5m.parquet under
DERIBIT_DATA_ROOT and checks only the rows written by v0.3.
Read-only; it never modifies data.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

data_root = os.getenv("DERIBIT_DATA_ROOT")

if not data_root:
    raise RuntimeError("DERIBIT_DATA_ROOT is not defined.")

utc_date = (
    sys.argv[1]
    if len(sys.argv) > 1
    else datetime.now(timezone.utc).date().isoformat()
)

path = (
    Path(data_root).expanduser().resolve()
    / "futures" / "5m" / f"date={utc_date}" / "futures_5m.parquet"
)

print(f"File: {path}")

df = pd.read_parquet(path)

if "collector_version" not in df.columns:
    raise SystemExit("No v0.3 rows yet (collector_version column absent).")

v3 = df[df["collector_version"].notna()].copy()

print(f"Rows total: {len(df):,}   v0.3 rows: {len(v3):,}")

if v3.empty:
    raise SystemExit("No v0.3 rows yet.")

print("\nVersion / source hash:")
print(v3.groupby(["collector_version", "collector_code_sha256"]).size().to_string())

print("\n1. Stream completeness and reasons")
print(f"   complete: {int(v3['stream_complete'].fillna(False).sum()):,} / {len(v3):,}")
print(v3["stream_incomplete_reason"].fillna("(complete)").value_counts().to_string())

print("\n2. Feed message gap (ms) per bar - calibrates FEED_MAX_GAP_FOR_COMPLETE_MS")
print(v3["feed_max_message_gap_ms"].describe(percentiles=[0.5, 0.9, 0.99]).to_string())

print("\n3. L2 invariants (must be zero mismatches)")
hist_sum = v3["book_obi_20_abs_histogram_ms"].apply(
    lambda values: float(sum(values)) if values is not None else None
)
duration = v3["book_obi_20_duration_ms"]
mask = hist_sum.notna() & duration.notna()
abs_mismatch = int(((hist_sum[mask] - duration[mask]).abs() > 1e-6).sum())
rev_sum = v3["book_obi_20_reversal_strength_histogram"].apply(
    lambda values: int(sum(values)) if values is not None else None
)
mask2 = rev_sum.notna() & v3["book_obi_20_raw_zero_cross_count"].notna()
rev_mismatch = int((rev_sum[mask2] != v3.loc[mask2, "book_obi_20_raw_zero_cross_count"]).sum())
print(f"   |OBI| histogram vs duration mismatches: {abs_mismatch} / {int(mask.sum())}")
print(f"   reversal histogram vs zero-cross mismatches: {rev_mismatch} / {int(mask2.sum())}")
print(f"   out-of-order L2 updates (total): {int(v3['book_out_of_order_update_count'].fillna(0).sum()):,}")

print("\n4. Per-instrument integrity evidence (medians / maxima)")
summary = v3.groupby("instrument_name").agg(
    bars=("bar_end", "count"),
    complete=("stream_complete", "sum"),
    med_book_gap_ms=("book_max_update_gap_ms", "median"),
    max_book_gap_ms=("book_max_update_gap_ms", "max"),
    med_bbo_age_ms=("ticker_bbo_age_ms", "median"),
    max_bbo_age_ms=("ticker_bbo_age_ms", "max"),
    bbo_match_rate=("ticker_book_bbo_match", "mean"),
    trades_checked=("trades_checked_vs_book", "sum"),
    trades_outside=("trades_outside_book_bbo", "sum"),
    max_outside_bps=("max_trade_outside_book_bbo_bps", "max"),
    rest_checks=("rest_verify_count", "sum"),
    rest_mismatch=("rest_verify_mismatch_count", "sum"),
    rest_stale=("rest_stale_confirmed", "sum"),
)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(summary.to_string())

print("\n5. Mid-path statistics availability")
print(v3.groupby("instrument_name").agg(
    med_valid_points=("mid_path_valid_points", "median"),
    med_increments=("mid_path_increment_count", "median"),
    med_zero_increments=("mid_path_zero_increments", "median"),
    med_sign_flips=("mid_path_return_sign_flips", "median"),
).to_string())

print("\n6. Funding fields populated")
for column in ["current_funding", "funding_8h"]:
    print(f"   {column}: {int(v3[column].notna().sum()):,} / {len(v3):,}")

with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 90):

    print("\n7. Incident timeline (bar_end UTC, any instrument incomplete)")
    complete_flag = v3["stream_complete"].fillna(False).astype(bool)
    incidents = v3[~complete_flag]
    if incidents.empty:
        print("   none")
    else:
        timeline = incidents.groupby("bar_end").agg(
            instruments=("instrument_name", "nunique"),
            reasons=(
                "stream_incomplete_reason",
                lambda s: " | ".join(sorted(set(s.dropna())))
            ),
            feed_gap_ms=("feed_max_message_gap_ms", "max"),
            max_book_gap_ms=("book_max_update_gap_ms", "max"),
            connection_count=("connection_count", "max"),
            rest_stale=("rest_stale_confirmed", "sum"),
        )
        print(timeline.to_string())

    print("\n   Reconnects (connection_count increases between consecutive bars):")
    connections = v3.groupby("bar_end")["connection_count"].max().sort_index()
    increases = connections[connections.diff() > 0]
    if increases.empty:
        print("   none")
    else:
        print(increases.to_string())

    print("\n   REST-confirmed stale streams:")
    stale = v3[v3["rest_stale_confirmed"].fillna(False).astype(bool)]
    if stale.empty:
        print("   none")
    else:
        print(stale[[
            "bar_end", "instrument_name", "rest_verify_count",
            "rest_verify_mismatch_count", "book_max_update_gap_ms",
            "ticker_bbo_age_ms", "trade_count"
        ]].to_string(index=False))

    print("\n8. Trades outside book BBO, measured in boundary spreads")
    print("   (count is dominated by 1-tick channel timing; distance is the staleness signal)")
    work = v3.copy()
    bid = pd.to_numeric(work["best_bid_price"], errors="coerce")
    ask = pd.to_numeric(work["best_ask_price"], errors="coerce")
    mid = (bid + ask) / 2.0
    work["boundary_spread_bps"] = (ask - bid) / mid * 10000.0
    work["outside_share"] = (
        work["trades_outside_book_bbo"]
        / work["trades_checked_vs_book"].where(work["trades_checked_vs_book"] > 0)
    )
    has_outside = (work["trades_outside_book_bbo"] > 0) & (work["boundary_spread_bps"] > 0)
    work.loc[has_outside, "max_outside_in_spreads"] = (
        work.loc[has_outside, "max_trade_outside_book_bbo_bps"]
        / work.loc[has_outside, "boundary_spread_bps"]
    )
    per_instrument = work.groupby("instrument_name").agg(
        med_spread_bps=("boundary_spread_bps", "median"),
        med_outside_share=("outside_share", "median"),
        bars_with_outside=("max_outside_in_spreads", "count"),
        p50_spreads=("max_outside_in_spreads", "median"),
        p90_spreads=("max_outside_in_spreads", lambda s: s.quantile(0.90)),
        p99_spreads=("max_outside_in_spreads", lambda s: s.quantile(0.99)),
        max_spreads=("max_outside_in_spreads", "max"),
        max_bps=("max_trade_outside_book_bbo_bps", "max"),
    )
    print(per_instrument.to_string())

    print("\n   Ten largest outside-BBO distances (bps):")
    print(work.nlargest(10, "max_trade_outside_book_bbo_bps")[[
        "bar_end", "instrument_name", "max_trade_outside_book_bbo_bps",
        "max_outside_in_spreads", "boundary_spread_bps", "trades_outside_book_bbo",
        "trades_checked_vs_book", "trade_count", "book_max_update_gap_ms",
        "stream_complete"
    ]].to_string(index=False))

    print("\n9. Out-of-order L2 rejections per instrument")
    l2 = v3.groupby("instrument_name").agg(
        out_of_order=("book_out_of_order_update_count", "sum"),
        accepted_updates=("book_update_count", "sum"),
    )
    l2["rejected_pct"] = 100.0 * l2["out_of_order"] / (l2["out_of_order"] + l2["accepted_updates"])
    print(l2.to_string())
