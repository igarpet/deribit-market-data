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

from data_engineering.alignment.state_index import (
    create_state_index,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OPTION_CAPTURE_MAX_DELAY_MS = 1000.0


# --------------------------------------------------
# Quality contract
# --------------------------------------------------

def create_quality_view(
    connection
) -> None:

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW
        state_quality AS


        WITH option_semantics AS (

            SELECT
                expiry,

                underlying,

                scheduled_snapshot_time
                    AS state_time,


                SUM(
                    CASE
                        WHEN best_bid_price < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_bid_count,


                SUM(
                    CASE
                        WHEN best_ask_price < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_ask_count,


                SUM(
                    CASE
                        WHEN best_bid_price = 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS zero_bid_count,


                SUM(
                    CASE
                        WHEN best_ask_price = 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS zero_ask_count,


                SUM(
                    CASE
                        WHEN
                            best_bid_price > 0
                            AND best_ask_price > 0
                            AND best_bid_price
                                > best_ask_price
                        THEN 1
                        ELSE 0
                    END
                )
                    AS crossed_market_count,


                SUM(
                    CASE
                        WHEN mark_iv <= 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS invalid_mark_iv_count,


                SUM(
                    CASE
                        WHEN delta < -1
                            OR delta > 1
                        THEN 1
                        ELSE 0
                    END
                )
                    AS invalid_delta_count,


                SUM(
                    CASE
                        WHEN gamma < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_gamma_count,


                SUM(
                    CASE
                        WHEN open_interest < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_open_interest_count


            FROM raw_option_grid


            GROUP BY
                expiry,
                underlying,
                scheduled_snapshot_time
        )


        SELECT

            s.*,


            -- ---------------------------------------
            -- Alignment
            -- ---------------------------------------

            (
                s.has_futures
                AND s.has_options
            )
                AS alignment_ready,


            -- ---------------------------------------
            -- Futures / trades
            -- ---------------------------------------

            (
                s.has_futures
                AND
                s.futures_stream_complete
                    IS TRUE
            )
                AS price_features_ready,


            -- ---------------------------------------
            -- Boundary L2
            -- ---------------------------------------

            (
                s.has_futures
                AND
                s.book_valid
                    IS TRUE
            )
                AS l2_boundary_ready,


            -- ---------------------------------------
            -- Intrawindow L2
            -- ---------------------------------------

            (
                s.has_futures
                AND
                s.futures_stream_complete
                    IS TRUE
                AND
                s.book_stats_complete
                    IS TRUE
            )
                AS l2_intrawindow_ready,


            -- ---------------------------------------
            -- Option timing
            -- ---------------------------------------

            (
                s.has_options

                AND

                s.capture_delay_ms
                    <= {OPTION_CAPTURE_MAX_DELAY_MS}
            )
                AS option_capture_on_time,


            -- ---------------------------------------
            -- Option semantics
            -- ---------------------------------------

            COALESCE(
                o.negative_bid_count,
                0
            )
                AS negative_bid_count,

            COALESCE(
                o.negative_ask_count,
                0
            )
                AS negative_ask_count,

            COALESCE(
                o.zero_bid_count,
                0
            )
                AS zero_bid_count,

            COALESCE(
                o.zero_ask_count,
                0
            )
                AS zero_ask_count,

            COALESCE(
                o.crossed_market_count,
                0
            )
                AS crossed_market_count,

            COALESCE(
                o.invalid_mark_iv_count,
                0
            )
                AS invalid_mark_iv_count,

            COALESCE(
                o.invalid_delta_count,
                0
            )
                AS invalid_delta_count,

            COALESCE(
                o.negative_gamma_count,
                0
            )
                AS negative_gamma_count,

            COALESCE(
                o.negative_open_interest_count,
                0
            )
                AS negative_open_interest_count,


            (
                s.has_options

                AND

                COALESCE(
                    o.negative_bid_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.negative_ask_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.crossed_market_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.invalid_mark_iv_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.invalid_delta_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.negative_gamma_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.negative_open_interest_count,
                    0
                ) = 0
            )
                AS option_semantics_valid,


            (
                s.has_options

                AND

                s.capture_delay_ms
                    <= {OPTION_CAPTURE_MAX_DELAY_MS}

                AND

                COALESCE(
                    o.negative_bid_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.negative_ask_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.crossed_market_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.invalid_mark_iv_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.invalid_delta_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.negative_gamma_count,
                    0
                ) = 0

                AND

                COALESCE(
                    o.negative_open_interest_count,
                    0
                ) = 0
            )
                AS option_snapshot_ready


        FROM canonical_state_index s


        LEFT JOIN option_semantics o

            ON
                s.expiry_date
                = o.expiry

            AND
                s.underlying
                = o.underlying

            AND
                s.state_time
                = o.state_time
        """
    )


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_quality_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            expiry_date,

            COUNT(*)
                AS expected,

            SUM(
                CAST(
                    alignment_ready
                    AS INTEGER
                )
            )
                AS aligned,

            SUM(
                CAST(
                    price_features_ready
                    AS INTEGER
                )
            )
                AS price_ready,

            SUM(
                CAST(
                    l2_boundary_ready
                    AS INTEGER
                )
            )
                AS l2_boundary,

            SUM(
                CAST(
                    l2_intrawindow_ready
                    AS INTEGER
                )
            )
                AS l2_intrawindow,

            SUM(
                CAST(
                    option_snapshot_ready
                    AS INTEGER
                )
            )
                AS options_ready

        FROM state_quality

        GROUP BY
            expiry_date

        ORDER BY
            expiry_date
        """
    ).fetchall()


    print(
        "\nQUALITY CONTRACT v0.1"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Expiry':<15}"
        f"{'Expected':>12}"
        f"{'Aligned':>12}"
        f"{'Price':>12}"
        f"{'L2 bound':>12}"
        f"{'L2 intra':>12}"
        f"{'Options':>12}"
    )


    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>12,}"
            f"{row[2]:>12,}"
            f"{row[3]:>12,}"
            f"{row[4]:>12,}"
            f"{row[5]:>12,}"
            f"{row[6]:>12,}"
        )


def print_option_semantics(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            SUM(negative_bid_count),
            SUM(negative_ask_count),

            SUM(zero_bid_count),
            SUM(zero_ask_count),

            SUM(crossed_market_count),

            SUM(invalid_mark_iv_count),
            SUM(invalid_delta_count),
            SUM(negative_gamma_count),
            SUM(negative_open_interest_count),

            SUM(
                CASE
                    WHEN has_options
                    AND option_capture_on_time
                        IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            )

        FROM state_quality
        """
    ).fetchone()


    print(
        "\nOPTION SEMANTIC CHECKS"
    )

    print(
        "=" * 80
    )

    print(
        f"Negative bids:          "
        f"{row[0]:,}"
    )

    print(
        f"Negative asks:          "
        f"{row[1]:,}"
    )

    print(
        f"Zero bids:              "
        f"{row[2]:,}"
    )

    print(
        f"Zero asks:              "
        f"{row[3]:,}"
    )

    print(
        f"Crossed markets:        "
        f"{row[4]:,}"
    )

    print(
        f"Invalid mark IV:        "
        f"{row[5]:,}"
    )

    print(
        f"Invalid delta:          "
        f"{row[6]:,}"
    )

    print(
        f"Negative gamma:         "
        f"{row[7]:,}"
    )

    print(
        f"Negative open interest: "
        f"{row[8]:,}"
    )

    print(
        f"Late option states:     "
        f"{row[9]:,}"
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

        create_quality_view(
            connection
        )

        print_quality_summary(
            connection
        )

        print_option_semantics(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()