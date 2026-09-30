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
# Late snapshot audit
# --------------------------------------------------

def print_late_snapshots(
    connection
) -> None:

    rows = connection.execute(
        """
        WITH snapshots AS (

            SELECT
                expiry,

                scheduled_snapshot_time,

                MIN(snapshot_time)
                    AS snapshot_time,

                MAX(capture_delay_ms)
                    AS capture_delay_ms,

                COUNT(
                    DISTINCT underlying
                )
                    AS underlyings,

                COUNT(*)
                    AS contract_rows

            FROM raw_option_grid

            GROUP BY
                expiry,
                scheduled_snapshot_time
        )

        SELECT
            expiry,
            scheduled_snapshot_time,
            snapshot_time,
            capture_delay_ms,
            underlyings,
            contract_rows

        FROM snapshots

        WHERE
            capture_delay_ms > 1000

        ORDER BY
            scheduled_snapshot_time
        """
    ).fetchall()


    print(
        "\nLATE OPTION SNAPSHOTS"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Expiry':<15}"
        f"{'Scheduled':<30}"
        f"{'Actual capture':<30}"
        f"{'Delay ms':>12}"
        f"{'Assets':>8}"
    )


    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{str(row[1]):<30}"
            f"{str(row[2]):<30}"
            f"{row[3]:>12.2f}"
            f"{row[4]:>8}"
        )


# --------------------------------------------------
# Quote availability by moneyness
# --------------------------------------------------

def print_quote_availability(
    connection
) -> None:

    rows = connection.execute(
        """
        WITH options AS (

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
                underlying_price > 0
        ),

        bucketed AS (

            SELECT
                *,

                CASE

                    WHEN abs_moneyness <= 0.02
                        THEN '0-2%'

                    WHEN abs_moneyness <= 0.05
                        THEN '2-5%'

                    WHEN abs_moneyness <= 0.10
                        THEN '5-10%'

                    WHEN abs_moneyness <= 0.20
                        THEN '10-20%'

                    ELSE '>20%'

                END
                    AS moneyness_bucket

            FROM options
        )


        SELECT
            moneyness_bucket,

            COUNT(*)
                AS rows,

            SUM(
                CASE
                    WHEN best_bid_price > 0
                    THEN 1 ELSE 0
                END
            )
                AS positive_bid,

            SUM(
                CASE
                    WHEN best_ask_price > 0
                    THEN 1 ELSE 0
                END
            )
                AS positive_ask,

            SUM(
                CASE
                    WHEN
                        best_bid_price > 0
                        AND best_ask_price > 0
                    THEN 1 ELSE 0
                END
            )
                AS two_sided_quote,

            SUM(
                CASE
                    WHEN mark_iv > 0
                    THEN 1 ELSE 0
                END
            )
                AS valid_mark_iv

        FROM bucketed

        GROUP BY
            moneyness_bucket

        ORDER BY

            CASE moneyness_bucket
                WHEN '0-2%' THEN 1
                WHEN '2-5%' THEN 2
                WHEN '5-10%' THEN 3
                WHEN '10-20%' THEN 4
                ELSE 5
            END
        """
    ).fetchall()


    print(
        "\nQUOTE AVAILABILITY BY MONEYNESS"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Bucket':<12}"
        f"{'Rows':>15}"
        f"{'Bid > 0':>15}"
        f"{'Ask > 0':>15}"
        f"{'Two-sided':>15}"
        f"{'Mark IV':>15}"
    )


    for row in rows:

        print(
            f"{row[0]:<12}"
            f"{row[1]:>15,}"
            f"{row[2]:>15,}"
            f"{row[3]:>15,}"
            f"{row[4]:>15,}"
            f"{row[5]:>15,}"
        )


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

        print_late_snapshots(
            connection
        )

        print_quote_availability(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":
    main()