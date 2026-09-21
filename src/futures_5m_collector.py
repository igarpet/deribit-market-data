from __future__ import annotations

import asyncio
import os
import signal
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from futures_state_cache import (
    FuturesStateCache,
    discover_perpetuals
)

from session_manager import (
    get_current_session
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

data_root_env = os.getenv(
    "DERIBIT_DATA_ROOT"
)

if not data_root_env:

    raise RuntimeError(
        "DERIBIT_DATA_ROOT is not "
        "defined in the .env file."
    )

DATA_ROOT = Path(
    data_root_env
).expanduser().resolve()

# Completely separate from options.
FUTURES_ROOT = (
    DATA_ROOT
    / "futures"
    / "5m"
)

SNAPSHOT_INTERVAL_SECONDS = (
    5 * 60
)

TICKER_INITIALIZATION_TIMEOUT_SECONDS = (
    60
)

# Allows the 100 ms trade feed a small amount
# of time to deliver trades executed immediately
# before the exact 5-minute boundary.
TRADE_FINALIZATION_DELAY_SECONDS = (
    1.0
)


# --------------------------------------------------
# Timing
# --------------------------------------------------

def next_bar_end(
    now: datetime | None = None
) -> datetime:

    if now is None:

        now = datetime.now(
            timezone.utc
        )

    timestamp = (
        now.timestamp()
    )

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
# Wait for initial ticker snapshots
# --------------------------------------------------

async def wait_for_tickers(
    cache: FuturesStateCache,
    stop_event: asyncio.Event
) -> bool:

    expected = len(
        cache.instrument_names
    )

    start = (
        asyncio.get_running_loop()
        .time()
    )

    print(
        "\nWaiting for futures "
        "ticker initialization..."
    )

    while not stop_event.is_set():

        initialized = len(
            cache.ticker_state
        )

        if initialized >= expected:

            print(
                f"Futures tickers initialized: "
                f"{initialized}/{expected}"
            )

            return True

        elapsed = (
            asyncio.get_running_loop()
            .time()
            - start
        )

        if (
            elapsed
            >= TICKER_INITIALIZATION_TIMEOUT_SECONDS
        ):

            print(
                "\nFutures ticker "
                "initialization timeout."
            )

            print(
                f"Initialized: "
                f"{initialized}/{expected}"
            )

            return False

        await asyncio.sleep(
            0.5
        )

    return False


# --------------------------------------------------
# Daily Parquet storage
# --------------------------------------------------

def save_bar_rows(
    rows: pd.DataFrame
) -> Path:

    if rows.empty:

        raise RuntimeError(
            "No futures rows "
            "were produced."
        )

    bar_start = pd.Timestamp(
        rows[
            "bar_start"
        ].iloc[0]
    )

    utc_date = (
        bar_start
        .tz_convert("UTC")
        .date()
        .isoformat()
    )

    output_folder = (
        FUTURES_ROOT
        / f"date={utc_date}"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_folder
        / "futures_5m.parquet"
    )

    temporary_file = (
        output_folder
        / ".futures_5m.tmp.parquet"
    )

    if output_file.exists():

        existing = pd.read_parquet(
            output_file
        )

        combined = pd.concat(
            [
                existing,
                rows
            ],
            ignore_index=True
        )

    else:

        combined = (
            rows.copy()
        )

    # Makes writes idempotent.
    combined = (
        combined
        .sort_values(
            [
                "bar_end",
                "underlying",
                "instrument_name"
            ]
        )
        .drop_duplicates(
            subset=[
                "bar_end",
                "instrument_name"
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    if temporary_file.exists():

        temporary_file.unlink()

    combined.to_parquet(
        temporary_file,
        index=False,
        compression="zstd"
    )

    if not temporary_file.exists():

        raise RuntimeError(
            "Temporary futures "
            "Parquet was not created."
        )

    temporary_file.replace(
        output_file
    )

    return output_file


# --------------------------------------------------
# Console summary
# --------------------------------------------------

def print_bar_summary(
    rows: pd.DataFrame,
    output_file: Path
) -> None:

    bar_start = rows[
        "bar_start"
    ].iloc[0]

    bar_end = rows[
        "bar_end"
    ].iloc[0]

    print(
        "\nFUTURES 5-MINUTE BAR SAVED"
    )

    print(
        "=" * 60
    )

    print(
        f"Window: "
        f"{bar_start} -> "
        f"{bar_end}"
    )

    print(
        f"Perpetuals: "
        f"{len(rows)}"
    )

    print(
        f"Trades: "
        f"{int(rows['trade_count'].sum()):,}"
    )

    print(
        f"USD volume: "
        f"{rows['volume_usd'].sum():,.2f}"
    )

    incomplete = int(
        (~rows[
            "stream_complete"
        ]).sum()
    )

    print(
        f"Incomplete streams: "
        f"{incomplete}"
    )

    print(
        "File:"
    )

    print(
        output_file
    )


# --------------------------------------------------
# 5-minute writer
# --------------------------------------------------

async def bar_writer(
    cache: FuturesStateCache,
    stop_event: asyncio.Event
) -> None:

    while not stop_event.is_set():

        scheduled_end = (
            next_bar_end()
        )

        print(
            f"\nNext futures bar: "
            f"{scheduled_end.isoformat()}"
        )

        reached_boundary = (
            await sleep_until(
                scheduled_end,
                stop_event
            )
        )

        if not reached_boundary:
            return

        # Freeze the futures quote/index state
        # at the same exact 5-minute boundary.
        boundary_state = (
            cache.capture_boundary_state(
                scheduled_end
            )
        )

        # Allow batched 100 ms trade messages
        # immediately before the boundary to arrive.
        try:

            await asyncio.wait_for(
                stop_event.wait(),
                timeout=(
                    TRADE_FINALIZATION_DELAY_SECONDS
                )
            )

            return

        except asyncio.TimeoutError:

            pass

        rows = cache.finalize_bar(
            bar_end=scheduled_end,
            boundary_state=boundary_state
        )

        try:

            output_file = (
                save_bar_rows(
                    rows
                )
            )

            print_bar_summary(
                rows,
                output_file
            )

        except Exception as exc:

            print(
                f"\nFutures bar "
                f"write error: {exc}"
            )


# --------------------------------------------------
# Shutdown
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
            f"\nFutures collector "
            f"received signal "
            f"({signum})."
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
# Main
# --------------------------------------------------

async def run() -> None:

    stop_event = (
        asyncio.Event()
    )

    configure_signals(
        stop_event
    )

    # Use the exact current 0DTE option session
    # to determine which underlying coins matter.
    session, option_instruments = (
        get_current_session()
    )

    print(
        "\nDERIBIT FUTURES 5M COLLECTOR"
    )

    print(
        "=" * 60
    )

    print(
        f"Option session: "
        f"{session.session_id}"
    )

    perpetuals = (
        discover_perpetuals(
            option_instruments
        )
    )

    print(
        "\nSelected perpetuals:"
    )

    display_columns = [
        column
        for column in [
            "base_currency",
            "instrument_name",
            "instrument_type",
            "settlement_currency",
            "api_currency"
        ]
        if column
        in perpetuals.columns
    ]

    print(
        perpetuals[
            display_columns
        ].to_string(
            index=False
        )
    )

    cache = FuturesStateCache(
        instruments=perpetuals,
        stop_event=stop_event
    )

    cache_task = (
        asyncio.create_task(
            cache.run_forever()
        )
    )

    initialized = (
        await wait_for_tickers(
            cache,
            stop_event
        )
    )

    if not initialized:

        stop_event.set()

        await cache_task

        raise RuntimeError(
            "Futures ticker cache "
            "failed to initialize."
        )

    writer_task = (
        asyncio.create_task(
            bar_writer(
                cache,
                stop_event
            )
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
        "\nFutures 5-minute "
        "collector stopped cleanly."
    )


def main():

    asyncio.run(
        run()
    )


if __name__ == "__main__":

    main()