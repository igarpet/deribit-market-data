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
# Schema helpers
# --------------------------------------------------

def relation_columns(
    connection,
    relation_name: str
) -> set[str]:

    rows = connection.execute(
        f"DESCRIBE {relation_name}"
    ).fetchall()

    return {
        row[0]
        for row in rows
    }


def optional_option_aggregates(
    option_columns: set[str]
) -> str:
    """
    Build provenance aggregates only when the underlying
    Parquet schema exposes the corresponding collector fields.

    This lets the research layer work with:
      - legacy option sessions,
      - warm-handover sessions,
      - reconnect-hardened sessions,
      - dynamic-universe sessions,

    all through one union_by_name raw view.
    """

    expressions = []

    # --------------------------------------------------
    # Opening provenance
    # --------------------------------------------------

    if (
        "is_opening_snapshot"
        in option_columns
        or "opening_snapshot_valid"
        in option_columns
        or "cache_initialized_before_boundary"
        in option_columns
    ):

        presence_terms = []

        for column in [
            "is_opening_snapshot",
            "opening_snapshot_valid",
            "cache_initialized_before_boundary",
        ]:

            if column in option_columns:

                presence_terms.append(
                    f"{column} IS NOT NULL"
                )

        expressions.append(
            """
            MAX(
                CASE
                    WHEN
                        """
            + " OR ".join(
                presence_terms
            )
            + """
                    THEN 1
                    ELSE 0
                END
            ) = 1
                AS opening_provenance_available
            """
        )

    else:

        expressions.append(
            """
            FALSE
                AS opening_provenance_available
            """
        )

    if "is_opening_snapshot" in option_columns:

        expressions.append(
            """
            MAX(
                CASE
                    WHEN is_opening_snapshot IS TRUE
                    THEN 1
                    ELSE 0
                END
            ) = 1
                AS collector_marks_opening
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS BOOLEAN)
                AS collector_marks_opening
            """
        )

    if "opening_snapshot_valid" in option_columns:

        expressions.append(
            """
            100.0
            * SUM(
                CASE
                    WHEN opening_snapshot_valid IS TRUE
                    THEN 1
                    ELSE 0
                END
            )
            / COUNT(*)
                AS opening_snapshot_valid_pct
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS DOUBLE)
                AS opening_snapshot_valid_pct
            """
        )

    if (
        "cache_initialized_before_boundary"
        in option_columns
    ):

        expressions.append(
            """
            MIN(
                CASE
                    WHEN
                        cache_initialized_before_boundary
                            IS TRUE
                    THEN 1
                    ELSE 0
                END
            ) = 1
                AS cache_initialized_before_boundary
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS BOOLEAN)
                AS cache_initialized_before_boundary
            """
        )

    # --------------------------------------------------
    # Connection/feed provenance
    # --------------------------------------------------

    if (
        "cache_fully_initialized"
        in option_columns
        or "feed_last_message_age_ms"
        in option_columns
        or "cache_connection_count"
        in option_columns
    ):

        presence_terms = []

        for column in [
            "cache_fully_initialized",
            "feed_last_message_age_ms",
            "cache_connection_count",
        ]:

            if column in option_columns:

                presence_terms.append(
                    f"{column} IS NOT NULL"
                )

        expressions.append(
            """
            MAX(
                CASE
                    WHEN
                        """
            + " OR ".join(
                presence_terms
            )
            + """
                    THEN 1
                    ELSE 0
                END
            ) = 1
                AS feed_provenance_available
            """
        )

    else:

        expressions.append(
            """
            FALSE
                AS feed_provenance_available
            """
        )

    if (
        "cache_fully_initialized"
        in option_columns
    ):

        expressions.append(
            """
            MIN(
                CASE
                    WHEN cache_fully_initialized IS TRUE
                    THEN 1
                    ELSE 0
                END
            ) = 1
                AS cache_fully_initialized
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS BOOLEAN)
                AS cache_fully_initialized
            """
        )

    if (
        "cache_fully_initialized_at_ms"
        in option_columns
    ):

        expressions.append(
            """
            MAX(
                cache_fully_initialized_at_ms
            )
                AS cache_fully_initialized_at_ms
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS BIGINT)
                AS cache_fully_initialized_at_ms
            """
        )

    if "cache_connection_count" in option_columns:

        expressions.append(
            """
            MAX(
                cache_connection_count
            )
                AS cache_connection_count
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS BIGINT)
                AS cache_connection_count
            """
        )

    if (
        "feed_last_message_received_at_ms"
        in option_columns
    ):

        expressions.append(
            """
            MAX(
                feed_last_message_received_at_ms
            )
                AS feed_last_message_received_at_ms
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS BIGINT)
                AS feed_last_message_received_at_ms
            """
        )

    if (
        "feed_last_message_age_ms"
        in option_columns
    ):

        expressions.append(
            """
            MAX(
                feed_last_message_age_ms
            )
                AS feed_last_message_age_ms
            """
        )

    else:

        expressions.append(
            """
            CAST(NULL AS DOUBLE)
                AS feed_last_message_age_ms
            """
        )

    # --------------------------------------------------
    # Dynamic-universe provenance
    # --------------------------------------------------

    if (
        "initial_instrument_count"
        in option_columns
        or "dynamic_instruments_added"
        in option_columns
        or "universe_size_at_capture"
        in option_columns
    ):

        presence_terms = []

        for column in [
            "initial_instrument_count",
            "dynamic_instruments_added",
            "universe_size_at_capture",
        ]:

            if column in option_columns:

                presence_terms.append(
                    f"{column} IS NOT NULL"
                )

        expressions.append(
            """
            MAX(
                CASE
                    WHEN
                        """
            + " OR ".join(
                presence_terms
            )
            + """
                    THEN 1
                    ELSE 0
                END
            ) = 1
                AS universe_provenance_available
            """
        )

    else:

        expressions.append(
            """
            FALSE
                AS universe_provenance_available
            """
        )

    for source_column, alias in [
        (
            "initial_instrument_count",
            "initial_instrument_count",
        ),
        (
            "dynamic_instruments_added",
            "dynamic_instruments_added",
        ),
        (
            "universe_size_at_capture",
            "universe_size_at_capture",
        ),
        (
            "last_universe_update_ms",
            "last_universe_update_ms",
        ),
    ]:

        if source_column in option_columns:

            expressions.append(
                f"""
                MAX(
                    {source_column}
                )
                    AS {alias}
                """
            )

        else:

            expressions.append(
                f"""
                CAST(NULL AS BIGINT)
                    AS {alias}
                """
            )

    return ",\n".join(
        expressions
    )


# --------------------------------------------------
# Canonical state index
# --------------------------------------------------

def create_state_index(
    connection
) -> None:

    option_columns = relation_columns(
        connection,
        "raw_option_grid"
    )

    provenance_sql = (
        optional_option_aggregates(
            option_columns
        )
    )

    connection.execute(
        f"""
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

                COUNT(
                    DISTINCT instrument_name
                )
                    AS futures_instrument_count,

                MIN(
                    CASE
                        WHEN stream_complete IS TRUE
                        THEN 1
                        ELSE 0
                    END
                ) = 1
                    AS futures_stream_complete,

                MIN(
                    CASE
                        WHEN book_valid IS TRUE
                        THEN 1
                        ELSE 0
                    END
                ) = 1
                    AS book_valid,

                MIN(
                    CASE
                        WHEN book_stats_complete IS TRUE
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
                        WHEN has_ticker_state IS TRUE
                        THEN 1
                        ELSE 0
                    END
                )
                    AS option_ticker_count,

                100.0
                * SUM(
                    CASE
                        WHEN has_ticker_state IS TRUE
                        THEN 1
                        ELSE 0
                    END
                )
                / COUNT(*)
                    AS option_coverage_pct,

                MIN(snapshot_time)
                    AS option_capture_time_min,

                MAX(snapshot_time)
                    AS option_capture_time_max,

                MAX(capture_delay_ms)
                    AS capture_delay_ms,

                MEDIAN(quote_age_ms)
                    AS median_quote_age_ms,

                MAX(quote_age_ms)
                    AS max_quote_age_ms,

                {provenance_sql}

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

            (
                e.state_time
                = e.session_start
            )
                AS is_opening_state,


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


            COALESCE(
                f.futures_rows,
                0
            )
                AS futures_rows,

            COALESCE(
                f.futures_instrument_count,
                0
            )
                AS futures_instrument_count,

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

            o.option_capture_time_min,

            o.option_capture_time_max,

            o.capture_delay_ms,

            o.median_quote_age_ms,

            o.max_quote_age_ms,


            o.opening_provenance_available,

            o.collector_marks_opening,

            o.opening_snapshot_valid_pct,

            o.cache_initialized_before_boundary,


            o.feed_provenance_available,

            o.cache_fully_initialized,

            o.cache_fully_initialized_at_ms,

            o.cache_connection_count,

            o.feed_last_message_received_at_ms,

            o.feed_last_message_age_ms,


            o.universe_provenance_available,

            o.initial_instrument_count,

            o.dynamic_instruments_added,

            o.universe_size_at_capture,

            o.last_universe_update_ms,


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
                AS complete_states,

            SUM(
                CAST(
                    opening_provenance_available
                    AS INTEGER
                )
            )
                AS opening_provenance_states,

            SUM(
                CAST(
                    feed_provenance_available
                    AS INTEGER
                )
            )
                AS feed_provenance_states,

            SUM(
                CAST(
                    universe_provenance_available
                    AS INTEGER
                )
            )
                AS universe_provenance_states

        FROM canonical_state_index

        GROUP BY
            expiry_date

        ORDER BY
            expiry_date
        """
    ).fetchall()


    print(
        "\nCANONICAL STATE INDEX v0.2"
    )

    print(
        "=" * 150
    )

    print(
        f"{'Expiry':<15}"
        f"{'Assets':>8}"
        f"{'Expected':>12}"
        f"{'Futures':>12}"
        f"{'Options':>12}"
        f"{'Complete':>12}"
        f"{'Open prov':>12}"
        f"{'Feed prov':>12}"
        f"{'Univ prov':>12}"
    )


    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>8,}"
            f"{row[2]:>12,}"
            f"{row[3]:>12,}"
            f"{row[4]:>12,}"
            f"{row[5]:>12,}"
            f"{row[6]:>12,}"
            f"{row[7]:>12,}"
            f"{row[8]:>12,}"
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
