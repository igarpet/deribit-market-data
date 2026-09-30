from __future__ import annotations

import sys
from pathlib import Path


# --------------------------------------------------
# Project imports
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

SRC_ROOT = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT)
    )


from data_engineering.catalog.database import (
    open_research_database,
    refresh_raw_views,
)


# --------------------------------------------------
# Futures diagnostics
# --------------------------------------------------

def print_futures_failure_counts(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,

            SUM(
                CASE
                    WHEN stream_complete IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            ) AS bad_stream,

            SUM(
                CASE
                    WHEN book_valid IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            ) AS bad_book,

            SUM(
                CASE
                    WHEN book_stats_complete IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            ) AS bad_book_stats

        FROM raw_futures_5m
        """
    ).fetchone()

    print("\nFUTURES FAILURE COUNTS")
    print("=" * 80)

    print(f"Rows:                {row[0]:,}")
    print(f"Incomplete stream:   {row[1]:,}")
    print(f"Invalid book:        {row[2]:,}")
    print(f"Incomplete L2 stats: {row[3]:,}")


def print_futures_percentiles(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            quantile_cont(
                book_age_ms,
                0.50
            ),

            quantile_cont(
                book_age_ms,
                0.90
            ),

            quantile_cont(
                book_age_ms,
                0.95
            ),

            quantile_cont(
                book_age_ms,
                0.99
            ),

            MAX(book_age_ms),

            quantile_cont(
                book_stats_coverage_ms,
                0.50
            ),

            quantile_cont(
                book_stats_coverage_ms,
                0.95
            ),

            quantile_cont(
                book_stats_coverage_ms,
                0.99
            )

        FROM raw_futures_5m

        WHERE
            book_valid
        """
    ).fetchone()

    print("\nFUTURES / L2 DISTRIBUTIONS")
    print("=" * 80)

    print(f"Book age p50:        {row[0]}")
    print(f"Book age p90:        {row[1]}")
    print(f"Book age p95:        {row[2]}")
    print(f"Book age p99:        {row[3]}")
    print(f"Book age max:        {row[4]}")

    print()
    print(f"Coverage p50 ms:     {row[5]}")
    print(f"Coverage p95 ms:     {row[6]}")
    print(f"Coverage p99 ms:     {row[7]}")


# --------------------------------------------------
# Option snapshot timing
# --------------------------------------------------

def print_capture_delay_distribution(
    connection
) -> None:

    row = connection.execute(
        """
        WITH states AS (

            SELECT
                expiry,
                underlying,
                scheduled_snapshot_time,

                MAX(capture_delay_ms)
                    AS capture_delay_ms

            FROM raw_option_grid

            GROUP BY
                expiry,
                underlying,
                scheduled_snapshot_time
        )

        SELECT
            COUNT(*),

            quantile_cont(
                capture_delay_ms,
                0.50
            ),

            quantile_cont(
                capture_delay_ms,
                0.90
            ),

            quantile_cont(
                capture_delay_ms,
                0.95
            ),

            quantile_cont(
                capture_delay_ms,
                0.99
            ),

            quantile_cont(
                capture_delay_ms,
                0.999
            ),

            MAX(capture_delay_ms),

            SUM(
                CASE
                    WHEN capture_delay_ms > 1000
                    THEN 1 ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN capture_delay_ms > 2000
                    THEN 1 ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN capture_delay_ms > 5000
                    THEN 1 ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN capture_delay_ms > 30000
                    THEN 1 ELSE 0
                END
            )

        FROM states
        """
    ).fetchone()

    print("\nOPTION CAPTURE DELAY")
    print("=" * 80)

    print(f"States:       {row[0]:,}")
    print(f"p50:          {row[1]}")
    print(f"p90:          {row[2]}")
    print(f"p95:          {row[3]}")
    print(f"p99:          {row[4]}")
    print(f"p99.9:        {row[5]}")
    print(f"Maximum:      {row[6]}")
    print()
    print(f"> 1 second:   {row[7]:,}")
    print(f"> 2 seconds:  {row[8]:,}")
    print(f"> 5 seconds:  {row[9]:,}")
    print(f"> 30 seconds: {row[10]:,}")


# --------------------------------------------------
# Option field completeness
# --------------------------------------------------

def print_option_field_completeness(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            COUNT(*),

            SUM(
                CASE WHEN best_bid_price IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN best_ask_price IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN mark_price IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN bid_iv IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN ask_iv IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN mark_iv IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN delta IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN gamma IS NULL
                THEN 1 ELSE 0 END
            ),

            SUM(
                CASE WHEN open_interest IS NULL
                THEN 1 ELSE 0 END
            )

        FROM raw_option_grid
        """
    ).fetchone()

    total = row[0]

    names = [
        "best_bid_price",
        "best_ask_price",
        "mark_price",
        "bid_iv",
        "ask_iv",
        "mark_iv",
        "delta",
        "gamma",
        "open_interest",
    ]

    print("\nOPTION FIELD COMPLETENESS")
    print("=" * 80)

    print(f"Total contract rows: {total:,}")
    print()

    for name, missing in zip(
        names,
        row[1:]
    ):

        pct = (
            missing
            / total
            * 100
            if total
            else 0
        )

        print(
            f"{name:<20}"
            f" missing={missing:>10,}"
            f" ({pct:>7.3f}%)"
        )


# --------------------------------------------------
# Quote-age distribution
# --------------------------------------------------

def print_quote_age_distribution(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            quantile_cont(
                quote_age_ms,
                0.50
            ),

            quantile_cont(
                quote_age_ms,
                0.90
            ),

            quantile_cont(
                quote_age_ms,
                0.95
            ),

            quantile_cont(
                quote_age_ms,
                0.99
            ),

            quantile_cont(
                quote_age_ms,
                0.999
            ),

            MAX(
                quote_age_ms
            )

        FROM raw_option_grid

        WHERE
            quote_age_ms IS NOT NULL
        """
    ).fetchone()

    print("\nOPTION QUOTE AGE")
    print("=" * 80)

    print(f"p50:   {row[0]}")
    print(f"p90:   {row[1]}")
    print(f"p95:   {row[2]}")
    print(f"p99:   {row[3]}")
    print(f"p99.9: {row[4]}")
    print(f"max:   {row[5]}")


# --------------------------------------------------
# Near-ATM quote quality
# --------------------------------------------------

def print_near_atm_quality(
    connection
) -> None:

    row = connection.execute(
        """
        WITH ranked AS (

            SELECT
                *,

                ABS(
                    strike
                    / NULLIF(
                        underlying_price,
                        0
                    )
                    - 1
                )
                    AS abs_moneyness

            FROM raw_option_grid

            WHERE
                underlying_price IS NOT NULL

                AND
                underlying_price > 0
        )

        SELECT
            COUNT(*),

            quantile_cont(
                quote_age_ms,
                0.50
            ),

            quantile_cont(
                quote_age_ms,
                0.95
            ),

            quantile_cont(
                quote_age_ms,
                0.99
            ),

            SUM(
                CASE
                    WHEN best_bid_price IS NULL
                    THEN 1 ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN best_ask_price IS NULL
                    THEN 1 ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN mark_iv IS NULL
                    THEN 1 ELSE 0
                END
            )

        FROM ranked

        WHERE
            abs_moneyness <= 0.05
        """
    ).fetchone()

    print("\nNEAR-ATM OPTION QUALITY (+/-5% STRIKE Moneyness)")
    print("=" * 80)

    print(f"Rows:              {row[0]:,}")
    print(f"Quote age p50:     {row[1]}")
    print(f"Quote age p95:     {row[2]}")
    print(f"Quote age p99:     {row[3]}")
    print(f"Missing bid:       {row[4]:,}")
    print(f"Missing ask:       {row[5]:,}")
    print(f"Missing mark IV:   {row[6]:,}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    connection = (
        open_research_database()
    )

    try:

        refresh_raw_views(
            connection
        )

        print_futures_failure_counts(
            connection
        )

        print_futures_percentiles(
            connection
        )

        print_capture_delay_distribution(
            connection
        )

        print_option_field_completeness(
            connection
        )

        print_quote_age_distribution(
            connection
        )

        print_near_atm_quality(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()