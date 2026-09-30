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
# Latest completed option session
# --------------------------------------------------

def print_latest_session(
    connection
) -> None:

    latest_expiry = connection.execute(
        """
        SELECT MAX(expiry)
        FROM raw_option_grid
        """
    ).fetchone()[0]

    print(
        "\nLATEST COMPLETED SESSION"
    )

    print(
        "=" * 100
    )

    print(
        f"Expiry: {latest_expiry}"
    )


    # --------------------------------------------------
    # Opening snapshot
    # --------------------------------------------------

    opening = connection.execute(
        """
        SELECT
            expiry,

            scheduled_snapshot_time,

            MIN(snapshot_time)
                AS actual_capture,

            MAX(capture_delay_ms)
                AS capture_delay_ms,

            COUNT(
                DISTINCT underlying
            )
                AS underlyings,

            COUNT(*)
                AS contract_rows

        FROM raw_option_grid

        WHERE
            expiry = ?

            AND

            EXTRACT(
                HOUR
                FROM scheduled_snapshot_time
            ) = 8

            AND

            EXTRACT(
                MINUTE
                FROM scheduled_snapshot_time
            ) = 0

        GROUP BY
            expiry,
            scheduled_snapshot_time

        ORDER BY
            scheduled_snapshot_time

        LIMIT 1
        """,
        [latest_expiry]
    ).fetchone()


    print(
        "\n08:00 OPTION SNAPSHOT"
    )

    print(
        "=" * 100
    )

    if opening is None:

        print(
            "No 08:00 option snapshot found."
        )

    else:

        print(
            f"Scheduled:      "
            f"{opening[1]}"
        )

        print(
            f"Actual capture: "
            f"{opening[2]}"
        )

        print(
            f"Delay ms:       "
            f"{opening[3]:,.3f}"
        )

        print(
            f"Underlyings:    "
            f"{opening[4]}"
        )

        print(
            f"Contract rows:  "
            f"{opening[5]:,}"
        )


    # --------------------------------------------------
    # Exact option/futures alignment
    # --------------------------------------------------

    alignment = connection.execute(
        """
        WITH opening_options AS (

            SELECT DISTINCT
                underlying,
                scheduled_snapshot_time
                    AS state_time

            FROM raw_option_grid

            WHERE
                expiry = ?

                AND

                EXTRACT(
                    HOUR
                    FROM scheduled_snapshot_time
                ) = 8

                AND

                EXTRACT(
                    MINUTE
                    FROM scheduled_snapshot_time
                ) = 0
        )

        SELECT
            o.underlying,

            o.state_time,

            f.bar_end IS NOT NULL
                AS has_futures,

            f.stream_complete,

            f.book_valid,

            f.book_stats_complete

        FROM opening_options o

        LEFT JOIN raw_futures_5m f

            ON
                o.underlying
                = f.underlying

            AND
                o.state_time
                = f.bar_end

        ORDER BY
            o.underlying
        """,
        [latest_expiry]
    ).fetchall()


    print(
        "\n08:00 FUTURES ALIGNMENT"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Underlying':<12}"
        f"{'Futures':>12}"
        f"{'Stream':>12}"
        f"{'Book':>12}"
        f"{'Book stats':>15}"
    )


    for row in alignment:

        print(
            f"{row[0]:<12}"
            f"{str(row[2]):>12}"
            f"{str(row[3]):>12}"
            f"{str(row[4]):>12}"
            f"{str(row[5]):>15}"
        )


    # --------------------------------------------------
    # Session completeness
    # --------------------------------------------------

    summary = connection.execute(
        """
        WITH option_states AS (

            SELECT DISTINCT
                underlying,
                scheduled_snapshot_time

            FROM raw_option_grid

            WHERE
                expiry = ?
        )

        SELECT
            COUNT(*),

            COUNT(
                DISTINCT scheduled_snapshot_time
            ),

            COUNT(
                DISTINCT underlying
            )

        FROM option_states
        """,
        [latest_expiry]
    ).fetchone()


    print(
        "\nSESSION COVERAGE"
    )

    print(
        "=" * 100
    )

    print(
        f"Underlying × timestamp states: "
        f"{summary[0]:,}"
    )

    print(
        f"Unique timestamps:             "
        f"{summary[1]:,}"
    )

    print(
        f"Underlyings:                   "
        f"{summary[2]:,}"
    )

    expected = (
        288
        * summary[2]
    )

    print(
        f"Expected states:               "
        f"{expected:,}"
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

        print_latest_session(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()