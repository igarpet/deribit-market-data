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

FUTURES_ROOT = (
    DATA_ROOT
    / "futures"
    / "5m"
)

SNAPSHOT_INTERVAL_SECONDS = (
    5 * 60
)

MARKET_INITIALIZATION_TIMEOUT_SECONDS = (
    60
)

# The public trade/book channels are aggregated
# at 100 ms. Give the final messages preceding
# the boundary a short opportunity to arrive.
FINALIZATION_DELAY_SECONDS = (
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
# Initial state
# --------------------------------------------------

async def wait_for_market_state(
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
        "ticker + L2 initialization..."
    )

    while not stop_event.is_set():

        tickers = len(
            cache.ticker_state
        )

        books = (
            cache.valid_book_count()
        )

        if (
            tickers >= expected
            and books >= expected
        ):

            print(
                f"Futures market state "
                f"initialized: "
                f"{expected}/{expected}"
            )

            return True

        elapsed = (
            asyncio.get_running_loop()
            .time()
            - start
        )

        if (
            elapsed
            >= MARKET_INITIALIZATION_TIMEOUT_SECONDS
        ):

            print(
                "\nFutures market-state "
                "initialization timeout."
            )

            print(
                f"Tickers: "
                f"{tickers}/{expected}"
            )

            print(
                f"L2 books: "
                f"{books}/{expected}"
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
    """
    Append one finalized five-minute row per perpetual.

    Schema evolution is explicit because L2 instability v0.1
    introduces scalar columns plus two fixed-length list
    distributions. If the collector is upgraded during an
    existing UTC day, pre-upgrade rows receive NULL values for
    the new columns and post-upgrade rows carry the new schema.
    """

    if rows.empty:

        raise RuntimeError(
            "No futures rows "
            "were produced."
        )

    rows = (
        rows.copy()
    )

    bar_start = pd.Timestamp(
        rows[
            "bar_start"
        ].iloc[0]
    )

    if (
        bar_start.tzinfo
        is None
    ):

        bar_start = (
            bar_start.tz_localize(
                "UTC"
            )
        )

    else:

        bar_start = (
            bar_start.tz_convert(
                "UTC"
            )
        )

    utc_date = (
        bar_start
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

        existing = (
            pd.read_parquet(
                output_file
            )
        )

        # Add missing columns explicitly before concat.
        # This is especially important for the two
        # list-valued histogram columns.
        for column in rows.columns:

            if column not in existing.columns:

                existing[
                    column
                ] = None

        for column in existing.columns:

            if column not in rows.columns:

                rows[
                    column
                ] = None

        column_order = list(
            dict.fromkeys(
                [
                    *existing.columns.tolist(),
                    *rows.columns.tolist()
                ]
            )
        )

        existing = existing[
            column_order
        ]

        rows = rows[
            column_order
        ]

        combined = pd.concat(
            [
                existing,
                rows
            ],
            ignore_index=True,
            sort=False
        )

    else:

        combined = (
            rows.copy()
        )

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

    bar_start = (
        rows[
            "bar_start"
        ].iloc[0]
    )

    bar_end = (
        rows[
            "bar_end"
        ].iloc[0]
    )

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

    incomplete_streams = int(
        (
            ~rows[
                "stream_complete"
            ].fillna(
                False
            )
        ).sum()
    )

    print(
        f"Incomplete streams: "
        f"{incomplete_streams}"
    )

    if (
        "book_valid"
        in rows.columns
    ):

        valid_books = int(
            rows[
                "book_valid"
            ]
            .fillna(
                False
            )
            .sum()
        )

        print(
            f"Valid L2 books: "
            f"{valid_books}/"
            f"{len(rows)}"
        )

    if (
        "book_stats_complete"
        in rows.columns
    ):

        complete_book_stats = int(
            rows[
                "book_stats_complete"
            ]
            .fillna(
                False
            )
            .sum()
        )

        print(
            f"Complete L2 stats: "
            f"{complete_book_stats}/"
            f"{len(rows)}"
        )

    if (
        "book_obi_20_std"
        in rows.columns
    ):

        valid_std = (
            rows[
                "book_obi_20_std"
            ]
            .dropna()
        )

        if not valid_std.empty:

            print(
                f"Median OBI20 std: "
                f"{valid_std.median():.6f}"
            )

    if (
        "book_obi_20_coherence_noise"
        in rows.columns
    ):

        valid_noise = (
            rows[
                "book_obi_20_coherence_noise"
            ]
            .dropna()
        )

        if not valid_noise.empty:

            print(
                f"Median OBI20 noise: "
                f"{valid_noise.median():.6f}"
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

        # Freeze ticker + boundary L2 state as close
        # as possible to the exact five-minute boundary.
        boundary_state = (
            cache.capture_boundary_state(
                scheduled_end
            )
        )

        # Allow the 100 ms aggregated trade/book
        # stream to flush exchange events from just
        # before the boundary.
        try:

            await asyncio.wait_for(
                stop_event.wait(),
                timeout=(
                    FINALIZATION_DELAY_SECONDS
                )
            )

            return

        except asyncio.TimeoutError:

            pass

        rows = (
            cache.finalize_bar(
                bar_end=scheduled_end,
                boundary_state=boundary_state
            )
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

    # The current exact 0DTE option universe
    # determines which underlyings need a
    # canonical perpetual reference.
    session, option_instruments = (
        get_current_session()
    )

    print(
        "\nDERIBIT FUTURES 5M + L2 COLLECTOR"
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
        await wait_for_market_state(
            cache,
            stop_event
        )
    )

    if not initialized:

        stop_event.set()

        await cache_task

        raise RuntimeError(
            "Futures ticker/L2 cache "
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
