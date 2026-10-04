from __future__ import annotations

import argparse
import asyncio
import hashlib
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

# Opening-state validity requires the COMPLETE expected
# universe on the current WebSocket generation.
MINIMUM_INITIALIZATION_PCT = 100.0

# Only normal scheduler jitter is accepted for the
# true 08:00 opening observation.
OPENING_CAPTURE_MAX_DELAY_SECONDS = 1.0

# Overall-feed continuity threshold. Individual illiquid
# option quotes may legitimately have older exchange
# timestamps while still being the latest known state.
OPENING_MAX_FEED_SILENCE_MS = 10_000

WARMUP_STATUS_INTERVAL_SECONDS = 1.0


# --------------------------------------------------
# Collector version stamp
# --------------------------------------------------
#
# Every snapshot row carries the collector version and a
# SHA256 over the two source files that determine option
# row content (this file + option_state_cache.py).

OPTION_COLLECTOR_VERSION = "option_grid_v0.3.0"


def _option_source_sha256() -> str | None:

    try:

        digest = hashlib.sha256()

        source_dir = Path(__file__).resolve().parent

        for name in [
            "option_grid_collector.py",
            "option_state_cache.py",
        ]:

            digest.update(
                (source_dir / name).read_bytes()
            )

        return digest.hexdigest()

    except OSError:

        return None


OPTION_COLLECTOR_CODE_SHA256 = _option_source_sha256()


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
) -> bool:

    print(
        "\nWaiting for WebSocket cache "
        "initialization..."
    )

    start = time.monotonic()

    while not stop_event.is_set():

        if cache.is_fully_initialized():

            ready_at = datetime.fromtimestamp(
                cache.fully_initialized_at_ms / 1000,
                tz=timezone.utc
            )

            print(
                f"Cache initialized: "
                f"{cache.initialized_count}/"
                f"{cache.expected_count} "
                f"({cache.coverage_pct:.1f}%)"
            )

            print(
                f"Cache ready at: "
                f"{ready_at.isoformat()}"
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
                "\nInitialization timeout."
            )

            print(
                f"Initialized: "
                f"{cache.initialized_count}/"
                f"{cache.expected_count} "
                f"({cache.coverage_pct:.1f}%)"
            )

            return False

        await asyncio.sleep(
            0.25
        )

    return False


# --------------------------------------------------
# Warm-up state
# --------------------------------------------------

def warmup_ready_file(
    session
) -> Path:

    return (
        STATE_ROOT
        / "warmup"
        / session.session_id
        / "ready.json"
    )


def opening_boundary_status(
    cache: OptionStateCache,
    scheduled_time: datetime
) -> dict:

    scheduled_ms = int(
        scheduled_time.timestamp()
        * 1000
    )

    readiness = (
        cache.readiness_snapshot()
    )

    initialized_at_ms = (
        readiness[
            "fully_initialized_at_ms"
        ]
    )

    initialized_before_boundary = (
        initialized_at_ms is not None
        and initialized_at_ms
        <= scheduled_ms
    )

    feed_recent = (
        readiness[
            "last_message_age_ms"
        ] is not None
        and readiness[
            "last_message_age_ms"
        ]
        <= OPENING_MAX_FEED_SILENCE_MS
    )

    valid = (
        readiness[
            "connected"
        ]
        and readiness[
            "fully_initialized"
        ]
        and initialized_before_boundary
        and feed_recent
    )

    return {
        **readiness,

        "cache_initialized_before_boundary":
            initialized_before_boundary,

        "feed_recent":
            feed_recent,

        "ready_for_opening_boundary":
            valid
    }


def write_warmup_ready_state(
    cache: OptionStateCache
) -> Path:

    output_file = warmup_ready_file(
        cache.session
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    now = datetime.now(
        timezone.utc
    )

    boundary = opening_boundary_status(
        cache,
        cache.session.session_start
    )

    payload = {
        "session_id":
            cache.session.session_id,

        "expiry_date":
            cache.session.expiry_date,

        "session_start_utc":
            cache.session.session_start.isoformat(),

        "session_expiry_utc":
            cache.session.session_expiry.isoformat(),

        "updated_at_utc":
            now.isoformat(),

        "connected":
            boundary[
                "connected"
            ],

        "connection_count":
            boundary[
                "connection_count"
            ],

        "expected_instruments":
            boundary[
                "expected_instruments"
            ],

        "initialized_instruments":
            boundary[
                "initialized_instruments"
            ],

        "coverage_pct":
            boundary[
                "coverage_pct"
            ],

        "fully_initialized":
            boundary[
                "fully_initialized"
            ],

        "fully_initialized_at_ms":
            boundary[
                "fully_initialized_at_ms"
            ],

        "last_message_received_at_ms":
            boundary[
                "last_message_received_at_ms"
            ],

        "last_message_age_ms":
            boundary[
                "last_message_age_ms"
            ],

        "cache_initialized_before_boundary":
            boundary[
                "cache_initialized_before_boundary"
            ],

        "feed_recent":
            boundary[
                "feed_recent"
            ],

        "ready_for_opening_boundary":
            boundary[
                "ready_for_opening_boundary"
            ]
    }

    temporary_file = (
        output_file
        .with_name(
            ".ready.tmp.json"
        )
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

    return output_file


async def warmup_status_reporter(
    cache: OptionStateCache,
    stop_event: asyncio.Event
) -> None:

    last_signature = None

    while not stop_event.is_set():

        try:

            path = (
                write_warmup_ready_state(
                    cache
                )
            )

            status = (
                opening_boundary_status(
                    cache,
                    cache.session.session_start
                )
            )

            signature = (
                status[
                    "connected"
                ],
                status[
                    "connection_count"
                ],
                status[
                    "initialized_instruments"
                ],
                status[
                    "expected_instruments"
                ],
                status[
                    "ready_for_opening_boundary"
                ]
            )

            if signature != last_signature:

                print(
                    "\n[WARMUP STATUS]"
                )

                print(
                    f"Connected: "
                    f"{status['connected']}"
                )

                print(
                    f"Connection generation: "
                    f"{status['connection_count']}"
                )

                print(
                    f"Initialized: "
                    f"{status['initialized_instruments']}/"
                    f"{status['expected_instruments']}"
                )

                print(
                    f"Ready for 08:00: "
                    f"{status['ready_for_opening_boundary']}"
                )

                print(
                    f"Status file: "
                    f"{path}"
                )

                last_signature = (
                    signature
                )

        except Exception as exc:

            print(
                f"\nWarm-up status "
                f"write error: {exc}"
            )

        try:

            await asyncio.wait_for(
                stop_event.wait(),
                timeout=(
                    WARMUP_STATUS_INTERVAL_SECONDS
                )
            )

        except asyncio.TimeoutError:
            pass


# --------------------------------------------------
# Snapshot creation
# --------------------------------------------------

def save_grid_snapshot(
    cache: OptionStateCache,
    scheduled_time: datetime,
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

    readiness = (
        cache.readiness_snapshot()
    )

    boundary = (
        opening_boundary_status(
            cache,
            scheduled_time
        )
    )

    opening_snapshot_valid_global = (
        is_opening_snapshot
        and boundary[
            "ready_for_opening_boundary"
        ]
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
        "cache_connection_count"
    ] = readiness[
        "connection_count"
    ]

    grid[
        "cache_fully_initialized"
    ] = readiness[
        "fully_initialized"
    ]

    grid[
        "cache_fully_initialized_at_ms"
    ] = readiness[
        "fully_initialized_at_ms"
    ]

    grid[
        "feed_last_message_received_at_ms"
    ] = readiness[
        "last_message_received_at_ms"
    ]

    grid[
        "feed_last_message_age_ms"
    ] = readiness[
        "last_message_age_ms"
    ]

    grid[
        "initial_instrument_count"
    ] = readiness[
        "initial_instrument_count"
    ]

    grid[
        "dynamic_instruments_added"
    ] = readiness[
        "dynamic_instruments_added"
    ]

    grid[
        "universe_size_at_capture"
    ] = readiness[
        "expected_instruments"
    ]

    grid[
        "last_universe_update_ms"
    ] = readiness[
        "last_universe_update_ms"
    ]

    grid[
        "is_opening_snapshot"
    ] = is_opening_snapshot

    grid[
        "collector_version"
    ] = OPTION_COLLECTOR_VERSION

    grid[
        "collector_code_sha256"
    ] = OPTION_COLLECTOR_CODE_SHA256

    grid[
        "cache_initialized_before_boundary"
    ] = boundary[
        "cache_initialized_before_boundary"
    ]

    # Row-level validity. An opening row cannot
    # be valid if that row has no ticker state.
    grid[
        "opening_snapshot_valid"
    ] = (
        opening_snapshot_valid_global
        & grid[
            "has_ticker_state"
        ]
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

        valid_rows = int(
            grid[
                "opening_snapshot_valid"
            ].sum()
        )

        print(
            f"Opening valid rows: "
            f"{valid_rows}/"
            f"{rows}"
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
    stop_event: asyncio.Event
) -> None:

    while not stop_event.is_set():

        now = datetime.now(
            timezone.utc
        )

        # Future-session warm-up collectors can
        # never write before their true 08:00 start.
        if now < cache.session.session_start:

            scheduled_time = (
                cache.session.session_start
            )

        else:

            scheduled_time = (
                next_snapshot_time(
                    now
                )
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

        if (
            scheduled_time
            == cache.session.session_start
        ):

            status = (
                opening_boundary_status(
                    cache,
                    scheduled_time
                )
            )

            opening_delay = (
                datetime.now(
                    timezone.utc
                )
                - scheduled_time
            ).total_seconds()

            if not status[
                "ready_for_opening_boundary"
            ]:

                print(
                    "\nOPENING SNAPSHOT MISSED"
                )

                print(
                    "Reason: current WebSocket "
                    "generation was not fully "
                    "ready/live at 08:00."
                )

                print(
                    f"Connected: "
                    f"{status['connected']}"
                )

                print(
                    f"Initialized: "
                    f"{status['initialized_instruments']}/"
                    f"{status['expected_instruments']}"
                )

                print(
                    f"Connection generation: "
                    f"{status['connection_count']}"
                )

                print(
                    f"Last message age: "
                    f"{status['last_message_age_ms']} ms"
                )

                continue

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

    stop_event = (
        asyncio.Event()
    )

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

    warmup_status_task = None

    if warmup:

        warmup_status_task = (
            asyncio.create_task(
                warmup_status_reporter(
                    cache,
                    stop_event
                )
            )
        )

    initialized = (
        await wait_for_cache(
            cache=cache,
            stop_event=stop_event
        )
    )

    if not initialized:

        print(
            "\nCache did not reach "
            "100% initialization "
            "coverage."
        )

        stop_event.set()

        await cache_task

        if (
            warmup_status_task
            is not None
            and not warmup_status_task.done()
        ):

            await warmup_status_task

        return

    if warmup:

        write_warmup_ready_state(
            cache
        )

    if once:

        test_time = datetime.now(
            timezone.utc
        )

        save_grid_snapshot(
            cache=cache,
            scheduled_time=test_time,
            test_mode=True
        )

        stop_event.set()

        await cache_task

        if (
            warmup_status_task
            is not None
            and not warmup_status_task.done()
        ):

            await warmup_status_task

        return

    writer_task = asyncio.create_task(
        snapshot_writer(
            cache=cache,
            stop_event=stop_event
        )
    )

    await stop_event.wait()

    for task in [
        writer_task,
        cache_task,
        warmup_status_task
    ]:

        if task is None:
            continue

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
            "Deribit 0DTE "
            "option-grid collector"
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
            "Pre-warm the next session. "
            "The process continuously publishes "
            "its current readiness state and cannot "
            "write before the true session start."
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
