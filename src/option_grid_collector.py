from __future__ import annotations

import argparse
import asyncio
import os
import signal
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from option_state_cache import OptionStateCache
from session_manager import get_current_session


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


# --------------------------------------------------
# Timing
# --------------------------------------------------

def next_snapshot_time(
    now: datetime | None = None
) -> datetime:
    """
    Return the next exact 5-minute UTC boundary.

    Examples:
        14:03 -> 14:05
        14:05:02 -> 14:10
        14:10:01 -> 15:15
    """

    if now is None:
        now = datetime.now(timezone.utc)

    now = now.astimezone(timezone.utc)

    timestamp = now.timestamp()

    next_timestamp = (
        int(timestamp // SNAPSHOT_INTERVAL_SECONDS)
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
    """
    Sleep until target time.

    Returns False if shutdown was requested first.
    """

    now = datetime.now(timezone.utc)

    delay = max(
        0,
        (
            target - now
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
) -> bool:

    expected = len(
        cache.instrument_names
    )

    print(
        f"\nWaiting for WebSocket cache "
        f"initialization..."
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

            print(
                f"Cache initialized: "
                f"{initialized}/{expected} "
                f"({pct:.1f}%)"
            )

            return True

        elapsed = (
            time.monotonic()
            - start
        )

        if (
            elapsed
            >= INITIALIZATION_TIMEOUT_SECONDS
        ):

            print(
                f"\nInitialization timeout."
            )

            print(
                f"Initialized: "
                f"{initialized}/{expected} "
                f"({pct:.1f}%)"
            )

            return False

        await asyncio.sleep(1)

    return False


# --------------------------------------------------
# Snapshot creation
# --------------------------------------------------

def save_grid_snapshot(
    cache: OptionStateCache,
    scheduled_time: datetime,
    test_mode: bool = False
) -> Path:
    """
    Freeze the complete cache into one Parquet file.

    Every expected instrument is represented by one row,
    even if a ticker is missing.
    """

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
        grid["exchange_timestamp_ms"]
        .notna()
    )

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

    if test_mode:

        root = TEST_SNAPSHOT_ROOT

    else:

        root = SNAPSHOT_ROOT

    output_folder = (
        root
        / cache.session.session_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp_string = (
        scheduled_time
        .strftime(
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
        / f"{prefix}_{timestamp_string}.parquet"
    )

    temporary_file = (
        output_folder
        / f".{prefix}_{timestamp_string}.tmp.parquet"
    )

    # Prevent accidental duplicate production snapshots.
    if (
        output_file.exists()
        and not test_mode
    ):

        print(
            f"\nSnapshot already exists:"
        )

        print(
            output_file
        )

        return output_file

    grid.to_parquet(
        temporary_file,
        index=False,
        compression="zstd"
    )

    # Verify it was actually written.
    if not temporary_file.exists():
        raise RuntimeError(
            "Temporary Parquet file "
            "was not created."
        )

    # Atomic finalization.
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

    rows = len(grid)

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
        f"{initialized}/{rows} "
        f"({coverage:.2f}%)"
    )

    print(
        f"Capture delay:      "
        f"{grid['capture_delay_ms'].iloc[0]:,.0f} ms"
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
        f"File:"
    )

    print(
        output_file
    )


# --------------------------------------------------
# Continuous 15-minute writer
# --------------------------------------------------

async def snapshot_writer(
    cache: OptionStateCache,
    stop_event: asyncio.Event
) -> None:

    while not stop_event.is_set():

        scheduled_time = (
            next_snapshot_time()
        )

        # Do not write the old session at/after expiry.
        if (
            scheduled_time
            >= cache.session.session_expiry
        ):

            print(
                "\nSession expiry reached."
            )

            print(
                f"Locked expiry: "
                f"{cache.session.session_expiry.isoformat()}"
            )

            print(
                "Stopping collector for rollover."
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

        try:

            save_grid_snapshot(
                cache=cache,
                scheduled_time=scheduled_time,
                test_mode=False
            )

        except Exception as exc:

            print(
                f"\nSnapshot error: {exc}"
            )


# --------------------------------------------------
# Shutdown handling
# --------------------------------------------------

def configure_signals(
    stop_event: asyncio.Event
) -> None:

    loop = asyncio.get_running_loop()

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
    once: bool = False
) -> None:

    stop_event = asyncio.Event()

    configure_signals(
        stop_event
    )

    session, instruments = (
        get_current_session()
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

    cache = OptionStateCache(
        session=session,
        instruments=instruments,
        stop_event=stop_event
    )

    cache_task = asyncio.create_task(
        cache.run_forever()
    )

    initialized = await wait_for_cache(
        cache=cache,
        stop_event=stop_event
    )

    if not initialized:

        print(
            "\nCache did not reach "
            "minimum initialization coverage."
        )

        stop_event.set()

        await cache_task

        return

    # ----------------------------------------------
    # One-off test mode
    # ----------------------------------------------

    if once:

        now = datetime.now(
            timezone.utc
        )

        save_grid_snapshot(
            cache=cache,
            scheduled_time=now,
            test_mode=True
        )

        stop_event.set()

        await cache_task

        return

    # ----------------------------------------------
    # Continuous mode
    # ----------------------------------------------

    writer_task = asyncio.create_task(
        snapshot_writer(
            cache=cache,
            stop_event=stop_event
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
        "\nGrid collector stopped cleanly."
    )


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
            "Write one immediate test snapshot "
            "and exit."
        )
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            once=args.once
        )
    )


if __name__ == "__main__":
    main()