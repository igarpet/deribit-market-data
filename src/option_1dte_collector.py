"""
1DTE option-grid collector (next daily expiry).

Purpose: constant-maturity implied volatility needs two expiries that
bracket the target maturity. The 0DTE collector covers the expiring
contracts only; this collector covers the NEXT daily expiry during the
same session, on the same 5-minute grid:

    collection window W_D = [08:00 UTC day d, 08:00 UTC day d+1)
    target expiry       E   = 08:00 UTC day d+2
    time to expiry          = 1440 .. 2875 minutes

At the end of W_D the target contracts become the 0DTE contracts of the
next session and are collected by the 0DTE orchestrator from then on.

Design
  * Separate process and systemd service. The 0DTE orchestrator is not
    touched, so a failure here cannot affect 0DTE collection.
  * Reuses OptionStateCache: full universe of the target expiry, the
    instrument-creation channel and REST reconciliation.
  * The next window's cache is started PREWARM_LEAD_SECONDS before the
    boundary, so the first snapshot of a window is not lost to WebSocket
    initialization. There is no "opening observation" semantics here:
    every row carries the cache readiness at capture time instead.
  * After a window ends, its raw snapshots are compacted (in a worker
    thread, so the live feed is never blocked) to
        daily/option_grid_1dte/expiry=YYYY-MM-DD/option_grid.parquet

Outputs
  raw/option_snapshots_1dte/expiry=YYYY-MM-DD/snapshot_YYYYmmddTHHMMSSZ.parquet
  daily/option_grid_1dte/expiry=YYYY-MM-DD/option_grid.parquet

Usage
  python src/option_1dte_collector.py               # run continuously
  python src/option_1dte_collector.py --once        # one test snapshot
  python src/option_1dte_collector.py --compact-pending
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from option_state_cache import OptionStateCache
from session_manager import (
    SessionWindow,
    calculate_session_window,
    get_session_for_expiry_date,
    select_session_instruments,
    session_window_for_expiry_date,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

data_root_env = os.getenv("DERIBIT_DATA_ROOT")

if not data_root_env:
    raise RuntimeError(
        "DERIBIT_DATA_ROOT is not defined in the .env file."
    )

DATA_ROOT = Path(data_root_env).expanduser().resolve()

SNAPSHOT_ROOT = DATA_ROOT / "raw" / "option_snapshots_1dte"
TEST_SNAPSHOT_ROOT = DATA_ROOT / "raw" / "test_snapshots_1dte"
DAILY_ROOT = DATA_ROOT / "daily" / "option_grid_1dte"

SNAPSHOT_INTERVAL = timedelta(minutes=5)
EXPECTED_SNAPSHOTS = 288

# Days between the collection window's own 0DTE expiry and the target.
TARGET_OFFSET_DAYS = 1

# The next window's cache starts this long before the boundary.
PREWARM_LEAD_SECONDS = 120

# How long to wait for full coverage before writing anyway. Rows always
# carry the readiness state, so partial coverage is visible downstream.
INITIALIZATION_WAIT_SECONDS = 120

# Retry delay when the target expiry has no listed instruments yet.
UNIVERSE_RETRY_SECONDS = 5 * 60

COLLECTION_PHASE = "1dte"


# --------------------------------------------------
# Collector version stamp
# --------------------------------------------------

OPTION_1DTE_COLLECTOR_VERSION = "option_grid_1dte_v0.1.0"


def _source_sha256() -> str | None:

    try:
        digest = hashlib.sha256()
        source_dir = Path(__file__).resolve().parent
        for name in ["option_1dte_collector.py", "option_state_cache.py"]:
            digest.update((source_dir / name).read_bytes())
        return digest.hexdigest()
    except OSError:
        return None


OPTION_1DTE_COLLECTOR_CODE_SHA256 = _source_sha256()


# --------------------------------------------------
# Windows
# --------------------------------------------------

@dataclass(frozen=True)
class CollectionWindow:
    """One 1DTE collection window and its target expiry."""

    window_start: datetime
    window_end: datetime
    target: SessionWindow

    @property
    def label(self) -> str:
        return f"{self.target.session_id} (window {self.window_start:%Y-%m-%d %H:%M}Z)"


def window_for_time(now: datetime) -> CollectionWindow:
    """The window containing `now` and its target expiry."""

    current = calculate_session_window(now)
    target_expiry = current.session_expiry + timedelta(days=TARGET_OFFSET_DAYS)
    target = session_window_for_expiry_date(target_expiry.strftime("%Y-%m-%d"))
    return CollectionWindow(
        window_start=current.session_start,
        window_end=current.session_expiry,
        target=target,
    )


def window_for_target_expiry(expiry_date: str) -> CollectionWindow:
    target = session_window_for_expiry_date(expiry_date)
    start = target.session_expiry - timedelta(days=TARGET_OFFSET_DAYS + 1)
    return CollectionWindow(
        window_start=start,
        window_end=start + timedelta(days=1),
        target=target,
    )


def snapshot_grid(window: CollectionWindow) -> list[datetime]:
    times, t = [], window.window_start
    while t < window.window_end:
        times.append(t)
        t += SNAPSHOT_INTERVAL
    if len(times) != EXPECTED_SNAPSHOTS:
        raise RuntimeError(f"Internal grid error: {len(times)} timestamps.")
    return times


# --------------------------------------------------
# Universe
# --------------------------------------------------

def load_target_instruments(window: CollectionWindow) -> pd.DataFrame:
    """
    Instruments of the target expiry from the latest instrument master.
    If the master predates their listing, rebuild the master in memory
    (it is NOT saved: the orchestrator owns the daily master file).
    """

    try:
        _, instruments = get_session_for_expiry_date(window.target.expiry_date)
        return instruments
    except RuntimeError as exc:
        print(f"\nInstrument master has no target instruments: {exc}")

    from instrument_master import build_instrument_master

    master = build_instrument_master()
    return select_session_instruments(master, window.target)


def summarize_universe(instruments: pd.DataFrame) -> str:
    if "underlying" not in instruments:
        return f"{len(instruments)} options"
    counts = instruments.groupby("underlying").size().sort_index()
    return ", ".join(f"{k} {v}" for k, v in counts.items())


# --------------------------------------------------
# Snapshot writing
# --------------------------------------------------

def save_snapshot(
    cache: OptionStateCache,
    window: CollectionWindow,
    scheduled_time: datetime,
    test_mode: bool = False,
) -> Path:

    actual = datetime.now(timezone.utc)
    grid = cache.to_dataframe(snapshot_time=actual)
    readiness = cache.readiness_snapshot()

    grid["scheduled_snapshot_time"] = scheduled_time
    grid["capture_delay_ms"] = (actual - scheduled_time).total_seconds() * 1000
    grid["session_id"] = window.target.session_id
    grid["expiry_date"] = window.target.expiry_date
    grid["collection_phase"] = COLLECTION_PHASE
    grid["collection_window_start"] = window.window_start
    grid["collection_window_end"] = window.window_end
    grid["scheduled_minutes_to_expiry"] = (
        (window.target.session_expiry - scheduled_time).total_seconds() / 60.0
    )
    grid["has_ticker_state"] = grid["exchange_timestamp_ms"].notna()
    grid["cache_connection_count"] = readiness["connection_count"]
    grid["cache_fully_initialized"] = readiness["fully_initialized"]
    grid["cache_fully_initialized_at_ms"] = readiness["fully_initialized_at_ms"]
    grid["cache_coverage_pct"] = readiness["coverage_pct"]
    grid["feed_last_message_received_at_ms"] = readiness["last_message_received_at_ms"]
    grid["feed_last_message_age_ms"] = readiness["last_message_age_ms"]
    grid["initial_instrument_count"] = readiness["initial_instrument_count"]
    grid["dynamic_instruments_added"] = readiness["dynamic_instruments_added"]
    grid["universe_size_at_capture"] = readiness["expected_instruments"]
    grid["last_universe_update_ms"] = readiness["last_universe_update_ms"]
    grid["collector_version"] = OPTION_1DTE_COLLECTOR_VERSION
    grid["collector_code_sha256"] = OPTION_1DTE_COLLECTOR_CODE_SHA256

    if len(grid) != len(cache.instrument_names):
        raise RuntimeError(
            f"Grid row mismatch: expected {len(cache.instrument_names)}, got {len(grid)}."
        )
    if not grid["instrument_name"].is_unique:
        raise RuntimeError("Duplicate instrument names in 1DTE snapshot.")

    expiry_ms = pd.to_numeric(grid["expiration_timestamp"], errors="coerce")
    wrong = expiry_ms.notna() & (expiry_ms != window.target.expiry_timestamp_ms)
    if wrong.any():
        raise RuntimeError(f"{int(wrong.sum())} rows do not belong to the target expiry.")

    root = TEST_SNAPSHOT_ROOT if test_mode else SNAPSHOT_ROOT
    folder = root / window.target.session_id
    folder.mkdir(parents=True, exist_ok=True)
    stamp = scheduled_time.strftime("%Y%m%dT%H%M%SZ")
    prefix = "test_snapshot" if test_mode else "snapshot"
    output = folder / f"{prefix}_{stamp}.parquet"
    temporary = folder / f".{prefix}_{stamp}.tmp.parquet"

    if output.exists() and not test_mode:
        print(f"\n[1DTE] Snapshot already exists: {output}")
        return output
    if temporary.exists():
        temporary.unlink()

    grid.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output)

    live = int(grid["has_ticker_state"].sum())
    print(
        f"[1DTE] {scheduled_time:%Y-%m-%d %H:%M}Z  {window.target.session_id}  "
        f"rows {len(grid)}  with ticker {live}  "
        f"coverage {readiness['coverage_pct']:.1f}%  "
        f"delay {grid['capture_delay_ms'].iloc[0]:.0f} ms"
    )
    return output


# --------------------------------------------------
# Compaction
# --------------------------------------------------

def compact_window(expiry_date: str) -> dict | None:
    """
    Compact the raw snapshots of a finished window into one daily file.
    Keeps the latest capture per (scheduled time, instrument). Raw files
    are not deleted. Returns diagnostics, or None if there is nothing.
    """

    window = window_for_target_expiry(expiry_date)
    if datetime.now(timezone.utc) < window.window_end:
        raise RuntimeError(f"Refusing to compact an unfinished window: {window.label}")

    folder = SNAPSHOT_ROOT / f"expiry={expiry_date}"
    files = sorted(folder.glob("snapshot_*.parquet"))
    if not files:
        return None

    source = (folder / "snapshot_*.parquet").as_posix()
    daily = DAILY_ROOT / f"expiry={expiry_date}"
    daily.mkdir(parents=True, exist_ok=True)
    final = daily / "option_grid.parquet"
    temporary = daily / ".option_grid.tmp.parquet"
    if temporary.exists():
        temporary.unlink()

    connection = duckdb.connect()
    try:
        connection.execute("SET TimeZone = 'UTC'")
        connection.execute(f"""
            COPY (
                SELECT * EXCLUDE (duplicate_rank)
                FROM (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY scheduled_snapshot_time, instrument_name
                            ORDER BY snapshot_time DESC
                        ) AS duplicate_rank
                    FROM read_parquet('{source}', union_by_name = true)
                )
                WHERE duplicate_rank = 1
                ORDER BY scheduled_snapshot_time, underlying, product_group, strike, option_type
            )
            TO '{temporary.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        rows, snapshots, instruments = connection.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT scheduled_snapshot_time), COUNT(DISTINCT instrument_name)
            FROM read_parquet('{temporary.as_posix()}')
        """).fetchone()
        times = {
            int(r[0])
            for r in connection.execute(f"""
                SELECT DISTINCT epoch_ms(scheduled_snapshot_time)
                FROM read_parquet('{temporary.as_posix()}')
            """).fetchall()
        }
    finally:
        connection.close()

    expected = {int(t.timestamp() * 1000) for t in snapshot_grid(window)}
    temporary.replace(final)
    diagnostics = {
        "expiry_date": expiry_date,
        "rows": rows,
        "snapshots": snapshots,
        "instruments": instruments,
        "missing_timestamps": len(expected - times),
        "unexpected_timestamps": len(times - expected),
        "file": str(final),
    }
    print(
        f"\n[1DTE] Compacted {window.label}: {snapshots}/{EXPECTED_SNAPSHOTS} snapshots, "
        f"{instruments} instruments, {rows:,} rows, missing {diagnostics['missing_timestamps']}, "
        f"unexpected {diagnostics['unexpected_timestamps']}\n       -> {final}"
    )
    return diagnostics


def pending_expiries() -> list[str]:
    """Finished windows with raw snapshots but no daily file."""

    if not SNAPSHOT_ROOT.exists():
        return []
    now = datetime.now(timezone.utc)
    pending = []
    for folder in sorted(SNAPSHOT_ROOT.glob("expiry=*")):
        expiry = folder.name.split("=", 1)[-1]
        try:
            window = window_for_target_expiry(expiry)
        except ValueError:
            continue
        if now < window.window_end:
            continue
        if (DAILY_ROOT / folder.name / "option_grid.parquet").exists():
            continue
        if any(folder.glob("snapshot_*.parquet")):
            pending.append(expiry)
    return pending


async def compact_in_background(expiry_date: str) -> None:
    try:
        await asyncio.to_thread(compact_window, expiry_date)
    except Exception as exc:
        print(f"\n[1DTE] Compaction failed for {expiry_date}: {exc}")


# --------------------------------------------------
# Window runner
# --------------------------------------------------

async def sleep_until(target: datetime, stop: asyncio.Event) -> bool:
    """True when the target time is reached, False if stopped first."""

    delay = max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
        return False
    except asyncio.TimeoutError:
        return True


async def wait_for_coverage(cache: OptionStateCache, stop: asyncio.Event) -> None:
    started = time.monotonic()
    while not stop.is_set():
        if cache.is_fully_initialized():
            print(f"[1DTE] Cache initialized: {cache.initialized_count}/{cache.expected_count}")
            return
        if time.monotonic() - started >= INITIALIZATION_WAIT_SECONDS:
            print(
                f"[1DTE] Initialization incomplete after {INITIALIZATION_WAIT_SECONDS}s: "
                f"{cache.initialized_count}/{cache.expected_count} "
                f"({cache.coverage_pct:.1f}%). Writing anyway; rows carry the coverage."
            )
            return
        await asyncio.sleep(0.5)


async def run_window(window: CollectionWindow, shutdown: asyncio.Event) -> None:
    """Collect one window: start the cache now (pre-warm if early), write
    every grid snapshot inside the window, stop, compact."""

    while not shutdown.is_set():
        try:
            instruments = await asyncio.to_thread(load_target_instruments, window)
            break
        except Exception as exc:
            print(f"\n[1DTE] No instruments for {window.label}: {exc}")
            print(f"[1DTE] Retrying in {UNIVERSE_RETRY_SECONDS}s.")
            if not await sleep_until(
                datetime.now(timezone.utc) + timedelta(seconds=UNIVERSE_RETRY_SECONDS), shutdown
            ):
                return
            if datetime.now(timezone.utc) >= window.window_end:
                print(f"[1DTE] Window ended without a universe: {window.label}")
                return
    else:
        return

    print(f"\n[1DTE] Starting {window.label}")
    print(f"[1DTE] Target expiry {window.target.session_expiry.isoformat()} | {summarize_universe(instruments)}")

    stop = asyncio.Event()
    cache = OptionStateCache(session=window.target, instruments=instruments, stop_event=stop)
    cache_task = asyncio.create_task(cache.run_forever())

    async def relay_shutdown() -> None:
        await shutdown.wait()
        stop.set()

    relay = asyncio.create_task(relay_shutdown())

    try:
        await wait_for_coverage(cache, stop)
        now = datetime.now(timezone.utc)
        for scheduled in snapshot_grid(window):
            if stop.is_set():
                break
            if scheduled < now - SNAPSHOT_INTERVAL / 2:
                continue  # started late: skip times already passed
            if not await sleep_until(scheduled, stop):
                break
            try:
                await asyncio.to_thread(save_snapshot, cache, window, scheduled)
            except Exception as exc:
                print(f"\n[1DTE] Snapshot error at {scheduled.isoformat()}: {exc}")
        await sleep_until(window.window_end, stop)
    finally:
        stop.set()
        relay.cancel()
        try:
            await cache_task
        except asyncio.CancelledError:
            pass

    if not shutdown.is_set() and datetime.now(timezone.utc) >= window.window_end:
        await compact_in_background(window.target.expiry_date)


async def run(shutdown: asyncio.Event) -> None:

    for expiry in pending_expiries():
        print(f"[1DTE] Compacting pending window for {expiry}")
        await compact_in_background(expiry)

    window = window_for_time(datetime.now(timezone.utc))
    tasks = {window.target.expiry_date: asyncio.create_task(run_window(window, shutdown))}

    while not shutdown.is_set():
        prewarm_at = window.window_end - timedelta(seconds=PREWARM_LEAD_SECONDS)
        if not await sleep_until(prewarm_at, shutdown):
            break
        window = window_for_time(window.window_end + timedelta(seconds=1))
        print(f"\n[1DTE] Pre-warming next window: {window.label}")
        tasks[window.target.expiry_date] = asyncio.create_task(run_window(window, shutdown))
        for key in [k for k, t in tasks.items() if t.done()]:
            tasks.pop(key)

    await asyncio.gather(*tasks.values(), return_exceptions=True)
    print("\n[1DTE] Collector stopped cleanly.")


async def run_once() -> None:
    window = window_for_time(datetime.now(timezone.utc))
    instruments = load_target_instruments(window)
    print(f"[1DTE] Test snapshot for {window.label} | {summarize_universe(instruments)}")
    stop = asyncio.Event()
    cache = OptionStateCache(session=window.target, instruments=instruments, stop_event=stop)
    task = asyncio.create_task(cache.run_forever())
    await wait_for_coverage(cache, stop)
    path = save_snapshot(cache, window, datetime.now(timezone.utc), test_mode=True)
    print(f"[1DTE] Test snapshot written: {path}")
    stop.set()
    await task


def main() -> None:

    parser = argparse.ArgumentParser(description="Deribit 1DTE option-grid collector")
    parser.add_argument("--once", action="store_true", help="Write one test snapshot and exit.")
    parser.add_argument("--compact-pending", action="store_true",
                        help="Compact finished windows without a daily file and exit.")
    args = parser.parse_args()

    print("\n1DTE GRID COLLECTOR")
    print("=" * 60)
    print(f"Version: {OPTION_1DTE_COLLECTOR_VERSION}  sha256: {OPTION_1DTE_COLLECTOR_CODE_SHA256}")
    print(f"Raw:     {SNAPSHOT_ROOT}")
    print(f"Daily:   {DAILY_ROOT}")

    if args.compact_pending:
        for expiry in pending_expiries():
            compact_window(expiry)
        return

    if args.once:
        asyncio.run(run_once())
        return

    async def runner() -> None:
        shutdown = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown.set)
            except NotImplementedError:
                signal.signal(sig, lambda *_: shutdown.set())
        await run(shutdown)

    asyncio.run(runner())


if __name__ == "__main__":
    main()
