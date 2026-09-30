from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from option_state_cache import OptionStateCache
from session_manager import (
    STATE_ROOT,
    get_current_session,
    get_session_for_expiry_date,
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

DATA_ROOT = Path(
    data_root_env
).expanduser().resolve()

SNAPSHOT_ROOT = (
    DATA_ROOT
    / "raw"
    / "option_snapshots"
)

TEST_SNAPSHOT_ROOT = (
    DATA_ROOT
    / "raw"
    / "test_snapshots"
)

SNAPSHOT_INTERVAL_SECONDS = 5 * 60

INITIALIZATION_TIMEOUT_SECONDS = 120
MINIMUM_INITIALIZATION_PCT = 99.0

# Only normal scheduler jitter is accepted for the true
# 08:00 opening observation.
OPENING_CAPTURE_MAX_DELAY_SECONDS = 1.0


# --------------------------------------------------
# Timing
# --------------------------------------------------

def next_snapshot_time(
    now: datetime | None = None
) -> datetime:

    if now is None:
        now = datetime.now(
            timezone.utc
        )

    now = now.astimezone(
        timezone.utc
    )

    timestamp = now.timestamp()

    next_timestamp = (
        int(
            timestamp
            // SNAPSHOT_INTERVAL_SECONDS
        )
        + 1
    ) * SNAPSHOT_INTERVAL_SECONDS

    return datetime.fromtimestamp(
        next_timestamp,
        tz=timezone.utc
    )


async def sleep_until(
    target: datetime,
    stop_event: asyncio.Event
) -> bool:

    delay = max(
        0,
        (
            target
            - datetime.now(
                timezone.utc
            )
        ).total_seconds()
    )

    try:
        await asyncio.wait_for(
            stop_event.wait(),
            timeout=delay
        )

        return False

    except asyncio.TimeoutError:
        return True


# --------------------------------------------------
# Cache initialization
# --------------------------------------------------

async def wait_for_cache(
    cache: OptionStateCache,
    stop_event: asyncio.Event
) -> datetime | None:

    expected = len(
        cache.instrument_names
    )

    print(
        "\nWaiting for WebSocket cache "
        "initialization..."
    )

    start = time.monotonic()

    while not stop_event.is_set():

        initialized = len(
            cache.state
        )

        pct = (
            initialized
            / expected
            * 100
            if expected
            else 0
        )

        if pct >= MINIMUM_INITIALIZATION_PCT:

            ready_at = datetime.now(
                timezone.utc
            )

            print(
                f"Cache initialized: "
                f"{initialized}/"
                f"{expected} "
                f"({pct:.1f}%)"
            )

            print(
                f"Cache ready at: "
                f"{ready_at.isoformat()}"
            )

            return ready_at

        elapsed = (
            time.monotonic()
            - start
        )

        if elapsed >= INITIALIZATION_TIMEOUT_SECONDS:

            print(
                "\nInitialization timeout."
            )

            print(
                f"Initialized: "
                f"{initialized}/"
                f"{expected} "
                f"({pct:.1f}%)"
            )

            return None

        await asyncio.sleep(1)

    return None


def warmup_ready_file(
    session
) -> Path:

    return (
        STATE_ROOT
        / "warmup"
        / session.session_id
        / "ready.json"
    )


def write_warmup_ready_state(
    cache: OptionStateCache,
    ready_at: datetime
) -> Path:

    output_file = warmup_ready_file(
        cache.session
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    expected = len(
        cache.instrument_names
    )

    initialized = len(
        cache.state
    )

    coverage_pct = (
        initialized
        / expected
        * 100
        if expected
        else 0.0
    )

    payload = {
        "session_id": cache.session.session_id,
        "expiry_date": cache.session.expiry_date,
        "session_start_utc": (
            cache.session.session_start.isoformat()
        ),
        "session_expiry_utc": (
            cache.session.session_expiry.isoformat()
        ),
        "expected_instruments": expected,
        "initialized_instruments": initialized,
        "coverage_pct": coverage_pct,
        "ready_at_utc": ready_at.isoformat(),
        "ready_before_session_start": (
            ready_at
            <= cache.session.session_start
        )
    }

    temporary_file = output_file.with_suffix(
        ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            payload,
            file,
            indent=4
        )

    temporary_file.replace(
        output_file
    )

    print(
        "\nWarm-up readiness state written:"
    )

    print(
        output_file
    )

    return output_file


# --------------------------------------------------
# Snapshot creation
# --------------------------------------------------

def save_grid_snapshot(
    cache: OptionStateCache,
    scheduled_time: datetime,
    cache_ready_at: datetime | None = None,
    test_mode: bool = False
) -> Path:

    actual_capture_time = datetime.now(
        timezone.utc
    )

    grid = cache.to_dataframe(
        snapshot_time=actual_capture_time
    )

    capture_delay_ms = (
        actual_capture_time
        - scheduled_time
    ).total_seconds() * 1000

    is_opening_snapshot = (
        scheduled_time
        == cache.session.session_start
    )

    cache_initialized_before_boundary = (
        cache_ready_at is not None
        and cache_ready_at <= scheduled_time
    )

    opening_snapshot_valid = (
        is_opening_snapshot
        and cache_initialized_before_boundary
        and capture_delay_ms
        <= (
            OPENING_CAPTURE_MAX_DELAY_SECONDS
            * 1000
        )
    )

    grid[
        "scheduled_snapshot_time"
    ] = scheduled_time

    grid[
        "capture_delay_ms"
    ] = capture_delay_ms

    grid[
        "session_id"
    ] = cache.session.session_id

    grid[
        "expiry_date"
    ] = cache.session.expiry_date

    grid[
        "has_ticker_state"
    ] = (
        grid[
            "exchange_timestamp_ms"
        ].notna()
    )

    grid[
        "is_opening_snapshot"
    ] = is_opening_snapshot

    grid[
        "cache_initialized_before_boundary"
    ] = cache_initialized_before_boundary

    grid[
        "opening_snapshot_valid"
    ] = opening_snapshot_valid

    expected_rows = len(
        cache.instrument_names
    )

    if len(grid) != expected_rows:
        raise RuntimeError(
            f"Grid row mismatch: "
            f"expected {expected_rows}, "
            f"got {len(grid)}."
        )

    if not grid[
        "instrument_name"
    ].is_unique:
        raise RuntimeError(
            "Duplicate instrument names "
            "detected in grid snapshot."
        )

    root = (
        TEST_SNAPSHOT_ROOT
        if test_mode
        else SNAPSHOT_ROOT
    )

    output_folder = (
        root
        / cache.session.session_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp_string = (
        scheduled_time.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    prefix = (
        "test_snapshot"
        if test_mode
        else "snapshot"
    )

    output_file = (
        output_folder
        / (
            f"{prefix}_"
            f"{timestamp_string}.parquet"
        )
    )

    temporary_file = (
        output_folder
        / (
            f".{prefix}_"
            f"{timestamp_string}"
            f".tmp.parquet"
        )
    )

    if (
        output_file.exists()
        and not test_mode
    ):
        print(
            "\nSnapshot already exists:"
        )

        print(
            output_file
        )

        return output_file

    if temporary_file.exists():
        temporary_file.unlink()

    grid.to_parquet(
        temporary_file,
        index=False,
        compression="zstd"
    )

    if not temporary_file.exists():
        raise RuntimeError(
            "Temporary Parquet file "
            "was not created."
        )

    temporary_file.replace(
        output_file
    )

    print_snapshot_summary(
        grid=grid,
        output_file=output_file
    )

    return output_file


# --------------------------------------------------
# Snapshot diagnostics
# --------------------------------------------------

def print_snapshot_summary(
    grid,
    output_file: Path
) -> None:

    rows = len(
        grid
    )

    initialized = int(
        grid[
            "has_ticker_state"
        ].sum()
    )

    coverage = (
        initialized
        / rows
        * 100
        if rows
        else 0
    )

    ages = (
        grid[
            "quote_age_ms"
        ]
        .dropna()
        .tolist()
    )

    file_size_mb = (
        output_file.stat().st_size
        / 1024
        / 1024
    )

    print(
        "\nGRID SNAPSHOT SAVED"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows:               "
        f"{rows}"
    )

    print(
        f"Ticker coverage:    "
        f"{initialized}/"
        f"{rows} "
        f"({coverage:.2f}%)"
    )

    print(
        f"Capture delay:      "
        f"{grid['capture_delay_ms'].iloc[0]:,.0f} ms"
    )

    print(
        f"Opening snapshot:   "
        f"{grid['is_opening_snapshot'].iloc[0]}"
    )

    if bool(
        grid[
            "is_opening_snapshot"
        ].iloc[0]
    ):
        print(
            f"Opening valid:      "
            f"{grid['opening_snapshot_valid'].iloc[0]}"
        )

    if ages:

        print(
            f"Median quote age:   "
            f"{statistics.median(ages):,.0f} ms"
        )

        print(
            f"Maximum quote age:  "
            f"{max(ages):,.0f} ms"
        )

    print(
        f"File size:          "
        f"{file_size_mb:.3f} MB"
    )

    print(
        "File:"
    )

    print(
        output_file
    )


# --------------------------------------------------
# Continuous writer
# --------------------------------------------------

async def snapshot_writer(
    cache: OptionStateCache,
    stop_event: asyncio.Event,
    cache_ready_at: datetime
) -> None:

    while not stop_event.is_set():

        now = datetime.now(
            timezone.utc
        )

        # Future-session warm-up collectors can never
        # write before their true 08:00 start.
        if now < cache.session.session_start:
            scheduled_time = (
                cache.session.session_start
            )
        else:
            scheduled_time = (
                next_snapshot_time(now)
            )

        if (
            scheduled_time
            >= cache.session.session_expiry
        ):

            print(
                "\nFinal 5-minute snapshot "
                "completed."
            )

            print(
                f"Waiting until session expiry: "
                f"{cache.session.session_expiry.isoformat()}"
            )

            continue_running = (
                await sleep_until(
                    cache.session.session_expiry,
                    stop_event
                )
            )

            if not continue_running:
                return

            print(
                "\nSession expiry reached."
            )

            print(
                f"Locked expiry: "
                f"{cache.session.session_expiry.isoformat()}"
            )

            print(
                "Stopping collector "
                "for rollover."
            )

            stop_event.set()

            return

        print(
            f"\nNext snapshot: "
            f"{scheduled_time.isoformat()}"
        )

        continue_running = (
            await sleep_until(
                scheduled_time,
                stop_event
            )
        )

        if not continue_running:
            return

        # True 08:00 boundary rule.
        if (
            scheduled_time
            == cache.session.session_start
        ):

            if (
                cache_ready_at
                > cache.session.session_start
            ):
                print(
                    "\nOPENING SNAPSHOT MISSED"
                )

                print(
                    "Reason: cache was not ready "
                    "before the 08:00 boundary."
                )

                print(
                    f"Cache ready at: "
                    f"{cache_ready_at.isoformat()}"
                )

                continue

            opening_delay = (
                datetime.now(
                    timezone.utc
                )
                - scheduled_time
            ).total_seconds()

            if (
                opening_delay
                > OPENING_CAPTURE_MAX_DELAY_SECONDS
            ):
                print(
                    "\nOPENING SNAPSHOT MISSED"
                )

                print(
                    "Reason: boundary capture "
                    "was too late."
                )

                print(
                    f"Delay: "
                    f"{opening_delay:.3f} seconds"
                )

                continue

        try:
            save_grid_snapshot(
                cache=cache,
                scheduled_time=scheduled_time,
                cache_ready_at=cache_ready_at,
                test_mode=False
            )

        except Exception as exc:
            print(
                f"\nSnapshot error: "
                f"{exc}"
            )


# --------------------------------------------------
# Shutdown handling
# --------------------------------------------------

def configure_signals(
    stop_event: asyncio.Event
) -> None:

    loop = (
        asyncio.get_running_loop()
    )

    def request_shutdown(
        signum,
        frame
    ):

        print(
            f"\nReceived shutdown "
            f"signal ({signum})."
        )

        loop.call_soon_threadsafe(
            stop_event.set
        )

    signal.signal(
        signal.SIGINT,
        request_shutdown
    )

    if hasattr(
        signal,
        "SIGTERM"
    ):
        signal.signal(
            signal.SIGTERM,
            request_shutdown
        )

    if hasattr(
        signal,
        "SIGBREAK"
    ):
        signal.signal(
            signal.SIGBREAK,
            request_shutdown
        )


# --------------------------------------------------
# Application
# --------------------------------------------------

async def run(
    once: bool = False,
    expiry_date: str | None = None,
    warmup: bool = False
) -> None:

    stop_event = asyncio.Event()

    configure_signals(
        stop_event
    )

    if expiry_date is None:
        session, instruments = (
            get_current_session()
        )
    else:
        session, instruments = (
            get_session_for_expiry_date(
                expiry_date
            )
        )

    now = datetime.now(
        timezone.utc
    )

    if now >= session.session_expiry:
        raise RuntimeError(
            "Refusing to start collector for "
            f"expired session {session.session_id}."
        )

    print(
        "\n0DTE GRID COLLECTOR"
    )

    print(
        "=" * 60
    )

    print(
        f"Session: "
        f"{session.session_id}"
    )

    print(
        f"Start:   "
        f"{session.session_start.isoformat()}"
    )

    print(
        f"Expiry:  "
        f"{session.session_expiry.isoformat()}"
    )

    print(
        f"Options: "
        f"{len(instruments)}"
    )

    print(
        f"Warm-up: "
        f"{warmup}"
    )

    cache = OptionStateCache(
        session=session,
        instruments=instruments,
        stop_event=stop_event
    )

    cache_task = asyncio.create_task(
        cache.run_forever()
    )

    cache_ready_at = (
        await wait_for_cache(
            cache=cache,
            stop_event=stop_event
        )
    )

    if cache_ready_at is None:

        print(
            "\nCache did not reach "
            "minimum initialization "
            "coverage."
        )

        stop_event.set()

        await cache_task

        return

    if warmup:
        write_warmup_ready_state(
            cache=cache,
            ready_at=cache_ready_at
        )

    if once:

        test_time = datetime.now(
            timezone.utc
        )

        save_grid_snapshot(
            cache=cache,
            scheduled_time=test_time,
            cache_ready_at=cache_ready_at,
            test_mode=True
        )

        stop_event.set()

        await cache_task

        return

    writer_task = asyncio.create_task(
        snapshot_writer(
            cache=cache,
            stop_event=stop_event,
            cache_ready_at=cache_ready_at
        )
    )

    await stop_event.wait()

    for task in [
        writer_task,
        cache_task
    ]:

        if not task.done():

            try:
                await task

            except asyncio.CancelledError:
                pass

    print(
        "\nGrid collector "
        "stopped cleanly."
    )


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Deribit 0DTE option-grid collector"
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Write one immediate "
            "test snapshot and exit."
        )
    )

    parser.add_argument(
        "--expiry-date",
        default=None,
        help=(
            "Explicit target expiry in YYYY-MM-DD. "
            "Used by the orchestrator for warm handover. "
            "If omitted, use the wall-clock current session."
        )
    )

    parser.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "Mark this process as a pre-warmed next-session "
            "collector. No snapshot is written before the "
            "session's true 08:00 UTC start."
        )
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            once=args.once,
            expiry_date=args.expiry_date,
            warmup=args.warmup
        )
    )


if __name__ == "__main__":
    main()