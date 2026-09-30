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
# Canonical state index
# --------------------------------------------------

def create_state_index(
    connection
) -> None:

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        canonical_state_index AS

        WITH session_universe AS (

            SELECT
                expiry,

                underlying,

                MIN(session_id)
                    AS session_id,

                MIN(expiration_time)
                    AS session_end,

                MIN(expiration_time)
                    - INTERVAL '1 day'
                    AS session_start

            FROM raw_option_grid

            GROUP BY
                expiry,
                underlying
        ),


        expected_states AS (

            SELECT
                s.expiry
                    AS expiry_date,

                s.session_id,

                s.underlying,

                s.session_start,

                s.session_end,

                grid.state_time,

                date_diff(
                    'minute',
                    grid.state_time,
                    s.session_end
                )
                    AS minutes_to_expiry

            FROM session_universe s

            CROSS JOIN LATERAL

                generate_series(
                    s.session_start,
                    s.session_end
                        - INTERVAL '5 minutes',
                    INTERVAL '5 minutes'
                )

                AS grid(state_time)
        ),


        futures_states AS (

            SELECT
                underlying,

                bar_end
                    AS state_time,

                COUNT(*)
                    AS futures_rows,

                MAX(
                    CASE
                        WHEN stream_complete
                        THEN 1
                        ELSE 0
                    END
                ) = 1
                    AS futures_stream_complete,

                MAX(
                    CASE
                        WHEN book_valid
                        THEN 1
                        ELSE 0
                    END
                ) = 1
                    AS book_valid,

                MAX(
                    CASE
                        WHEN book_stats_complete
                        THEN 1
                        ELSE 0
                    END
                ) = 1
                    AS book_stats_complete

            FROM raw_futures_5m

            GROUP BY
                underlying,
                bar_end
        ),


        option_states AS (

            SELECT
                expiry,

                underlying,

                scheduled_snapshot_time
                    AS state_time,

                COUNT(*)
                    AS option_contract_count,

                SUM(
                    CASE
                        WHEN has_ticker_state
                        THEN 1
                        ELSE 0
                    END
                )
                    AS option_ticker_count,

                100.0
                * SUM(
                    CASE
                        WHEN has_ticker_state
                        THEN 1
                        ELSE 0
                    END
                )
                / COUNT(*)
                    AS option_coverage_pct,

                MAX(capture_delay_ms)
                    AS capture_delay_ms,

                MEDIAN(quote_age_ms)
                    AS median_quote_age_ms,

                MAX(quote_age_ms)
                    AS max_quote_age_ms

            FROM raw_option_grid

            GROUP BY
                expiry,
                underlying,
                scheduled_snapshot_time
        )


        SELECT

            e.underlying
                || '|'
                || strftime(
                    e.state_time,
                    '%Y-%m-%dT%H:%M:%SZ'
                )
                    AS state_id,

            e.state_time,

            e.underlying,

            e.session_id,

            e.expiry_date,

            e.session_start,

            e.session_end,

            e.minutes_to_expiry,


            COALESCE(
                f.futures_rows,
                0
            ) > 0
                AS has_futures,


            COALESCE(
                o.option_contract_count,
                0
            ) > 0
                AS has_options,


            f.futures_stream_complete,

            f.book_valid,

            f.book_stats_complete,


            COALESCE(
                o.option_contract_count,
                0
            )
                AS option_contract_count,


            COALESCE(
                o.option_ticker_count,
                0
            )
                AS option_ticker_count,


            o.option_coverage_pct,

            o.capture_delay_ms,

            o.median_quote_age_ms,

            o.max_quote_age_ms,


            (
                COALESCE(
                    f.futures_rows,
                    0
                ) > 0

                AND

                COALESCE(
                    o.option_contract_count,
                    0
                ) > 0
            )
                AS state_complete


        FROM expected_states e


        LEFT JOIN futures_states f

            ON
                e.underlying
                = f.underlying

            AND
                e.state_time
                = f.state_time


        LEFT JOIN option_states o

            ON
                e.expiry_date
                = o.expiry

            AND
                e.underlying
                = o.underlying

            AND
                e.state_time
                = o.state_time
        """
    )


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_session_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            expiry_date,

            COUNT(
                DISTINCT underlying
            )
                AS underlyings,

            COUNT(*)
                AS expected_states,

            SUM(
                CAST(
                    has_futures
                    AS INTEGER
                )
            )
                AS futures_states,

            SUM(
                CAST(
                    has_options
                    AS INTEGER
                )
            )
                AS option_states,

            SUM(
                CAST(
                    state_complete
                    AS INTEGER
                )
            )
                AS complete_states

        FROM canonical_state_index

        GROUP BY
            expiry_date

        ORDER BY
            expiry_date
        """
    ).fetchall()


    print(
        "\nCANONICAL STATE INDEX"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Expiry':<15}"
        f"{'Assets':>10}"
        f"{'Expected':>15}"
        f"{'Futures':>15}"
        f"{'Options':>15}"
        f"{'Complete':>15}"
    )


    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>10,}"
            f"{row[2]:>15,}"
            f"{row[3]:>15,}"
            f"{row[4]:>15,}"
            f"{row[5]:>15,}"
        )


# --------------------------------------------------
# Missing timestamps
# --------------------------------------------------

def print_collection_gaps(
    connection
) -> None:

    futures_bounds = (
        connection.execute(
            """
            SELECT
                MIN(bar_end),
                MAX(bar_end)

            FROM raw_futures_5m
            """
        )
        .fetchone()
    )

    first_futures = futures_bounds[0]
    last_futures = futures_bounds[1]


    rows = connection.execute(
        """
        SELECT
            state_time,

            COUNT(*) AS assets,

            SUM(
                CASE
                    WHEN NOT has_futures
                    THEN 1
                    ELSE 0
                END
            )
                AS missing_futures,

            SUM(
                CASE
                    WHEN NOT has_options
                    THEN 1
                    ELSE 0
                END
            )
                AS missing_options

        FROM canonical_state_index

        WHERE
            state_time BETWEEN ? AND ?

            AND (
                NOT has_futures
                OR
                NOT has_options
            )

        GROUP BY
            state_time

        ORDER BY
            state_time
        """,
        [
            first_futures,
            last_futures
        ]
    ).fetchall()


    print(
        "\nCOLLECTION GAPS INSIDE "
        "FUTURES COVERAGE"
    )

    print(
        "=" * 100
    )

    if not rows:

        print(
            "No gaps found."
        )

        return


    print(
        f"{'Timestamp':<32}"
        f"{'Assets':>10}"
        f"{'Missing futures':>20}"
        f"{'Missing options':>20}"
    )


    for row in rows:

        print(
            f"{str(row[0]):<32}"
            f"{row[1]:>10}"
            f"{row[2]:>20}"
            f"{row[3]:>20}"
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

        create_state_index(
            connection
        )

        print_session_summary(
            connection
        )

        print_collection_gaps(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()