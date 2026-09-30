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

QUALITY_CONTRACT_VERSION = "0.2"

OPTION_CAPTURE_MAX_DELAY_MS = 1000.0

OPTION_FEED_MAX_AGE_MS = 10_000.0


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


                COUNT(*)
                    AS option_rows,


                SUM(
                    CASE
                        WHEN has_ticker_state IS TRUE
                        THEN 1
                        ELSE 0
                    END
                )
                    AS ticker_state_count,


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
                        WHEN
                            mark_iv IS NOT NULL
                            AND mark_iv <= 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS invalid_mark_iv_count,


                SUM(
                    CASE
                        WHEN mark_iv > 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS valid_mark_iv_count,


                SUM(
                    CASE
                        WHEN
                            delta IS NOT NULL
                            AND (
                                delta < -1
                                OR delta > 1
                            )
                        THEN 1
                        ELSE 0
                    END
                )
                    AS invalid_delta_count,


                SUM(
                    CASE
                        WHEN
                            gamma IS NOT NULL
                            AND gamma < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_gamma_count,


                SUM(
                    CASE
                        WHEN
                            open_interest IS NOT NULL
                            AND open_interest < 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS negative_open_interest_count,


                SUM(
                    CASE
                        WHEN best_bid_price > 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS positive_bid_count,


                SUM(
                    CASE
                        WHEN best_ask_price > 0
                        THEN 1
                        ELSE 0
                    END
                )
                    AS positive_ask_count,


                SUM(
                    CASE
                        WHEN
                            best_bid_price > 0
                            AND best_ask_price > 0
                            AND best_bid_price
                                <= best_ask_price
                        THEN 1
                        ELSE 0
                    END
                )
                    AS two_sided_quote_count


            FROM raw_option_grid


            GROUP BY
                expiry,
                underlying,
                scheduled_snapshot_time
        )


        SELECT

            s.*,


            -- ---------------------------------------
            -- Collection / alignment
            -- ---------------------------------------

            (
                s.has_futures
                AND
                s.has_options
            )
                AS collection_complete,


            (
                s.has_futures
                AND
                s.has_options
            )
                AS alignment_ready,


            (
                s.futures_rows = 1
                AND
                s.futures_instrument_count = 1
            )
                AS futures_unique,


            -- ---------------------------------------
            -- Futures / trades
            -- ---------------------------------------

            (
                s.has_futures

                AND

                s.futures_rows = 1

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

                s.futures_rows = 1

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

                s.futures_rows = 1

                AND

                s.futures_stream_complete
                    IS TRUE

                AND

                s.book_stats_complete
                    IS TRUE
            )
                AS l2_intrawindow_ready,


            -- ---------------------------------------
            -- Option coverage / timing
            -- ---------------------------------------

            (
                s.has_options

                AND

                s.option_contract_count > 0

                AND

                s.option_ticker_count
                    = s.option_contract_count

                AND

                s.option_coverage_pct
                    >= 100.0
            )
                AS option_coverage_complete,


            (
                s.has_options

                AND

                s.capture_delay_ms
                    >= 0

                AND

                s.capture_delay_ms
                    <= {OPTION_CAPTURE_MAX_DELAY_MS}
            )
                AS option_capture_on_time,


            -- ---------------------------------------
            -- Opening provenance
            -- ---------------------------------------

            CASE

                WHEN NOT s.is_opening_state
                THEN TRUE

                WHEN NOT s.has_options
                THEN FALSE

                WHEN NOT s.opening_provenance_available
                THEN NULL

                ELSE (
                    s.collector_marks_opening
                        IS TRUE

                    AND

                    s.opening_snapshot_valid_pct
                        >= 100.0

                    AND

                    s.cache_initialized_before_boundary
                        IS TRUE
                )

            END
                AS option_opening_provenance_valid,


            CASE

                WHEN NOT s.is_opening_state
                THEN TRUE

                WHEN NOT s.has_options
                THEN FALSE

                WHEN s.opening_provenance_available
                THEN (
                    s.collector_marks_opening
                        IS TRUE

                    AND

                    s.opening_snapshot_valid_pct
                        >= 100.0

                    AND

                    s.cache_initialized_before_boundary
                        IS TRUE
                )

                -- Legacy sessions cannot prove pre-boundary
                -- initialization. We therefore accept only
                -- genuinely on-time captures, while exposing
                -- the missing provenance separately.
                ELSE (
                    s.capture_delay_ms
                        >= 0

                    AND

                    s.capture_delay_ms
                        <= {OPTION_CAPTURE_MAX_DELAY_MS}
                )

            END
                AS option_opening_gate,


            -- ---------------------------------------
            -- Feed provenance
            -- ---------------------------------------

            CASE

                WHEN NOT s.has_options
                THEN FALSE

                WHEN NOT s.feed_provenance_available
                THEN NULL

                ELSE (
                    s.cache_fully_initialized
                        IS TRUE

                    AND

                    s.feed_last_message_age_ms
                        IS NOT NULL

                    AND

                    s.feed_last_message_age_ms
                        <= {OPTION_FEED_MAX_AGE_MS}
                )

            END
                AS option_feed_provenance_valid,


            CASE

                WHEN NOT s.has_options
                THEN FALSE

                WHEN s.feed_provenance_available
                THEN (
                    s.cache_fully_initialized
                        IS TRUE

                    AND

                    s.feed_last_message_age_ms
                        IS NOT NULL

                    AND

                    s.feed_last_message_age_ms
                        <= {OPTION_FEED_MAX_AGE_MS}
                )

                -- Legacy rows are not rejected solely because
                -- the collector did not yet emit feed provenance.
                ELSE TRUE

            END
                AS option_feed_gate,


            -- ---------------------------------------
            -- Dynamic-universe provenance
            -- ---------------------------------------

            CASE

                WHEN NOT s.has_options
                THEN FALSE

                WHEN NOT s.universe_provenance_available
                THEN NULL

                ELSE (
                    s.initial_instrument_count
                        IS NOT NULL

                    AND

                    s.dynamic_instruments_added
                        IS NOT NULL

                    AND

                    s.universe_size_at_capture
                        IS NOT NULL

                    AND

                    s.universe_size_at_capture
                        >= s.initial_instrument_count

                    AND

                    s.dynamic_instruments_added
                        =
                        (
                            s.universe_size_at_capture
                            - s.initial_instrument_count
                        )
                )

            END
                AS option_universe_metadata_valid,


            CASE

                WHEN NOT s.has_options
                THEN FALSE

                WHEN s.universe_provenance_available
                THEN (
                    s.initial_instrument_count
                        IS NOT NULL

                    AND

                    s.dynamic_instruments_added
                        IS NOT NULL

                    AND

                    s.universe_size_at_capture
                        IS NOT NULL

                    AND

                    s.universe_size_at_capture
                        >= s.initial_instrument_count

                    AND

                    s.dynamic_instruments_added
                        =
                        (
                            s.universe_size_at_capture
                            - s.initial_instrument_count
                        )
                )

                ELSE TRUE

            END
                AS option_universe_gate,


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
                o.valid_mark_iv_count,
                0
            )
                AS valid_mark_iv_count,

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

            COALESCE(
                o.positive_bid_count,
                0
            )
                AS positive_bid_count,

            COALESCE(
                o.positive_ask_count,
                0
            )
                AS positive_ask_count,

            COALESCE(
                o.two_sided_quote_count,
                0
            )
                AS two_sided_quote_count,


            CASE

                WHEN COALESCE(
                    o.ticker_state_count,
                    0
                ) > 0

                THEN (
                    100.0
                    * COALESCE(
                        o.valid_mark_iv_count,
                        0
                    )
                    / o.ticker_state_count
                )

                ELSE NULL

            END
                AS option_mark_iv_coverage_pct,


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
                AS option_structural_semantics_valid,


            -- Compatibility alias from v0.1.
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


            -- ---------------------------------------
            -- Model-ready option state
            -- ---------------------------------------

            (
                s.has_options

                AND

                s.option_contract_count > 0

                AND

                s.option_ticker_count
                    = s.option_contract_count

                AND

                s.option_coverage_pct
                    >= 100.0

                AND

                s.capture_delay_ms
                    >= 0

                AND

                s.capture_delay_ms
                    <= {OPTION_CAPTURE_MAX_DELAY_MS}

                AND

                (
                    CASE

                        WHEN NOT s.is_opening_state
                        THEN TRUE

                        WHEN s.opening_provenance_available
                        THEN (
                            s.collector_marks_opening
                                IS TRUE

                            AND

                            s.opening_snapshot_valid_pct
                                >= 100.0

                            AND

                            s.cache_initialized_before_boundary
                                IS TRUE
                        )

                        ELSE (
                            s.capture_delay_ms
                                >= 0

                            AND

                            s.capture_delay_ms
                                <= {OPTION_CAPTURE_MAX_DELAY_MS}
                        )

                    END
                )

                AND

                (
                    CASE

                        WHEN s.feed_provenance_available
                        THEN (
                            s.cache_fully_initialized
                                IS TRUE

                            AND

                            s.feed_last_message_age_ms
                                IS NOT NULL

                            AND

                            s.feed_last_message_age_ms
                                <= {OPTION_FEED_MAX_AGE_MS}
                        )

                        ELSE TRUE

                    END
                )

                AND

                (
                    CASE

                        WHEN s.universe_provenance_available
                        THEN (
                            s.initial_instrument_count
                                IS NOT NULL

                            AND

                            s.dynamic_instruments_added
                                IS NOT NULL

                            AND

                            s.universe_size_at_capture
                                IS NOT NULL

                            AND

                            s.universe_size_at_capture
                                >= s.initial_instrument_count

                            AND

                            s.dynamic_instruments_added
                                =
                                (
                                    s.universe_size_at_capture
                                    - s.initial_instrument_count
                                )
                        )

                        ELSE TRUE

                    END
                )

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
                AS option_snapshot_ready,


            -- ---------------------------------------
            -- Research readiness tiers
            -- ---------------------------------------

            (
                s.has_futures

                AND

                s.futures_rows = 1

                AND

                s.futures_stream_complete
                    IS TRUE

                AND

                s.has_options

                AND

                s.option_contract_count > 0

                AND

                s.option_ticker_count
                    = s.option_contract_count

                AND

                s.option_coverage_pct
                    >= 100.0

                AND

                s.capture_delay_ms
                    >= 0

                AND

                s.capture_delay_ms
                    <= {OPTION_CAPTURE_MAX_DELAY_MS}

                AND

                (
                    CASE

                        WHEN NOT s.is_opening_state
                        THEN TRUE

                        WHEN s.opening_provenance_available
                        THEN (
                            s.collector_marks_opening
                                IS TRUE

                            AND

                            s.opening_snapshot_valid_pct
                                >= 100.0

                            AND

                            s.cache_initialized_before_boundary
                                IS TRUE
                        )

                        ELSE (
                            s.capture_delay_ms
                                >= 0

                            AND

                            s.capture_delay_ms
                                <= {OPTION_CAPTURE_MAX_DELAY_MS}
                        )

                    END
                )

                AND

                (
                    CASE

                        WHEN s.feed_provenance_available
                        THEN (
                            s.cache_fully_initialized
                                IS TRUE

                            AND

                            s.feed_last_message_age_ms
                                IS NOT NULL

                            AND

                            s.feed_last_message_age_ms
                                <= {OPTION_FEED_MAX_AGE_MS}
                        )

                        ELSE TRUE

                    END
                )

                AND

                (
                    CASE

                        WHEN s.universe_provenance_available
                        THEN (
                            s.initial_instrument_count
                                IS NOT NULL

                            AND

                            s.dynamic_instruments_added
                                IS NOT NULL

                            AND

                            s.universe_size_at_capture
                                IS NOT NULL

                            AND

                            s.universe_size_at_capture
                                >= s.initial_instrument_count

                            AND

                            s.dynamic_instruments_added
                                =
                                (
                                    s.universe_size_at_capture
                                    - s.initial_instrument_count
                                )
                        )

                        ELSE TRUE

                    END
                )

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
                AS state_research_ready,


            (
                (
                    s.has_futures
                    AND
                    s.futures_rows = 1
                    AND
                    s.futures_stream_complete
                        IS TRUE
                )

                AND

                (
                    s.has_options
                    AND
                    s.option_contract_count > 0
                    AND
                    s.option_ticker_count
                        = s.option_contract_count
                    AND
                    s.option_coverage_pct
                        >= 100.0
                    AND
                    s.capture_delay_ms
                        >= 0
                    AND
                    s.capture_delay_ms
                        <= {OPTION_CAPTURE_MAX_DELAY_MS}
                )

                AND

                s.book_valid
                    IS TRUE
            )
                AS state_l2_boundary_candidate,


            (
                (
                    s.has_futures
                    AND
                    s.futures_rows = 1
                    AND
                    s.futures_stream_complete
                        IS TRUE
                )

                AND

                (
                    s.has_options
                    AND
                    s.option_contract_count > 0
                    AND
                    s.option_ticker_count
                        = s.option_contract_count
                    AND
                    s.option_coverage_pct
                        >= 100.0
                    AND
                    s.capture_delay_ms
                        >= 0
                    AND
                    s.capture_delay_ms
                        <= {OPTION_CAPTURE_MAX_DELAY_MS}
                )

                AND

                s.book_valid
                    IS TRUE

                AND

                s.book_stats_complete
                    IS TRUE
            )
                AS state_l2_intrawindow_candidate


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
                    collection_complete
                    AS INTEGER
                )
            )
                AS collected,

            SUM(
                CAST(
                    price_features_ready
                    AS INTEGER
                )
            )
                AS price_ready,

            SUM(
                CAST(
                    option_snapshot_ready
                    AS INTEGER
                )
            )
                AS options_ready,

            SUM(
                CAST(
                    state_research_ready
                    AS INTEGER
                )
            )
                AS research_ready,

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
                AS l2_intrawindow

        FROM state_quality

        GROUP BY
            expiry_date

        ORDER BY
            expiry_date
        """
    ).fetchall()


    print(
        f"\nQUALITY CONTRACT "
        f"v{QUALITY_CONTRACT_VERSION}"
    )

    print(
        "=" * 140
    )

    print(
        f"{'Expiry':<15}"
        f"{'Expected':>12}"
        f"{'Collected':>12}"
        f"{'Price':>12}"
        f"{'Options':>12}"
        f"{'Research':>12}"
        f"{'L2 bound':>12}"
        f"{'L2 intra':>12}"
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
            f"{row[7]:>12,}"
        )


def print_provenance_summary(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            SUM(
                CASE
                    WHEN is_opening_state
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN
                        is_opening_state
                        AND opening_provenance_available
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN
                        is_opening_state
                        AND option_opening_gate
                            IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN feed_provenance_available
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN
                        feed_provenance_available
                        AND option_feed_provenance_valid
                            IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN universe_provenance_available
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN
                        universe_provenance_available
                        AND option_universe_metadata_valid
                            IS NOT TRUE
                    THEN 1
                    ELSE 0
                END
            )

        FROM state_quality
        """
    ).fetchone()


    print(
        "\nPROVENANCE COVERAGE"
    )

    print(
        "=" * 80
    )

    print(
        f"Opening states:                 "
        f"{row[0]:,}"
    )

    print(
        f"Opening provenance available:  "
        f"{row[1]:,}"
    )

    print(
        f"Opening states rejected:       "
        f"{row[2]:,}"
    )

    print(
        f"Feed provenance states:        "
        f"{row[3]:,}"
    )

    print(
        f"Feed provenance failures:      "
        f"{row[4]:,}"
    )

    print(
        f"Universe provenance states:    "
        f"{row[5]:,}"
    )

    print(
        f"Universe metadata failures:    "
        f"{row[6]:,}"
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

            ROUND(
                MEDIAN(
                    option_mark_iv_coverage_pct
                ),
                3
            ),

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
        "\nOPTION QUALITY DIAGNOSTICS"
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
        f"Invalid mark IV rows:   "
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
        f"Median mark-IV cov %:   "
        f"{row[9]}"
    )

    print(
        f"Late option states:     "
        f"{row[10]:,}"
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

        print_provenance_summary(
            connection
        )

        print_option_semantics(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
