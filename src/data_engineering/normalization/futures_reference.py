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
)

from data_engineering.canonical.build_canonical import (
    build_canonical,
)


REFERENCE_LAYER_VERSION = "0.1"


# --------------------------------------------------
# Futures / perpetual reference state
# --------------------------------------------------

def create_futures_reference_state(
    connection
) -> None:
    """
    Create one reference-price row per canonical market state.

    Price hierarchy
    ---------------
    1. Valid two-sided perpetual ticker midpoint.
    2. Positive perpetual mark price.
    3. Positive perpetual underlying price.
    4. Positive perpetual index price.

    The fallback hierarchy is deliberately explicit and the
    selected source is persisted. Downstream return features
    should normally require market_reference_is_tradable,
    while diagnostics/visualization may choose to use the
    broader fallback reference.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        futures_reference_state AS

        WITH base AS (

            SELECT
                *,

                (
                    best_bid_price > 0
                    AND
                    best_ask_price > 0
                    AND
                    best_bid_price
                        <= best_ask_price
                )
                    AS has_valid_two_sided_ticker,

                CASE

                    WHEN
                        best_bid_price > 0
                        AND
                        best_ask_price > 0
                        AND
                        best_bid_price
                            <= best_ask_price

                    THEN
                        (
                            best_bid_price
                            + best_ask_price
                        ) / 2.0

                    ELSE NULL

                END
                    AS tradable_mid_price

            FROM canonical_market_state
        )


        SELECT
            state_id,

            state_time,

            underlying,

            session_id,

            expiry_date,

            session_start,

            session_end,

            minutes_to_expiry,

            is_opening_state,


            collection_complete,

            price_features_ready,

            state_research_ready,

            l2_boundary_ready,

            l2_intrawindow_ready,


            instrument_name
                AS futures_instrument_name,

            instrument_type
                AS futures_instrument_type,

            settlement_currency
                AS futures_settlement_currency,

            quote_currency
                AS futures_quote_currency,


            best_bid_price
                AS futures_bid_price,

            best_ask_price
                AS futures_ask_price,

            tradable_mid_price
                AS futures_tradable_mid_price,

            mark_price
                AS futures_mark_price,

            index_price
                AS futures_index_price,

            underlying_price
                AS futures_underlying_price,

            microprice
                AS futures_microprice,


            has_valid_two_sided_ticker,


            CASE

                WHEN tradable_mid_price > 0
                THEN tradable_mid_price

                WHEN mark_price > 0
                THEN mark_price

                WHEN underlying_price > 0
                THEN underlying_price

                WHEN index_price > 0
                THEN index_price

                ELSE NULL

            END
                AS market_reference_price,


            CASE

                WHEN tradable_mid_price > 0
                THEN 'perpetual_mid'

                WHEN mark_price > 0
                THEN 'perpetual_mark'

                WHEN underlying_price > 0
                THEN 'perpetual_underlying'

                WHEN index_price > 0
                THEN 'index'

                ELSE NULL

            END
                AS market_reference_source,


            (
                tradable_mid_price > 0
            )
                AS market_reference_is_tradable,


            (
                CASE

                    WHEN tradable_mid_price > 0
                    THEN tradable_mid_price

                    WHEN mark_price > 0
                    THEN mark_price

                    WHEN underlying_price > 0
                    THEN underlying_price

                    WHEN index_price > 0
                    THEN index_price

                    ELSE NULL

                END

            ) > 0
                AS market_reference_available,


            CASE

                WHEN
                    tradable_mid_price > 0
                    AND
                    best_ask_price
                        >= best_bid_price

                THEN
                    (
                        best_ask_price
                        - best_bid_price
                    )
                    / tradable_mid_price
                    * 10000.0

                ELSE NULL

            END
                AS futures_spread_bps,


            CASE

                WHEN
                    tradable_mid_price > 0
                    AND
                    index_price > 0

                THEN
                    (
                        tradable_mid_price
                        / index_price
                        - 1.0
                    )
                    * 10000.0

                ELSE NULL

            END
                AS perpetual_mid_vs_index_bps,


            CASE

                WHEN
                    mark_price > 0
                    AND
                    index_price > 0

                THEN
                    (
                        mark_price
                        / index_price
                        - 1.0
                    )
                    * 10000.0

                ELSE NULL

            END
                AS perpetual_mark_vs_index_bps,


            CASE

                WHEN
                    tradable_mid_price > 0
                    AND
                    microprice > 0

                THEN
                    (
                        microprice
                        / tradable_mid_price
                        - 1.0
                    )
                    * 10000.0

                ELSE NULL

            END
                AS microprice_vs_mid_bps,


            ticker_age_ms
                AS futures_ticker_age_ms,

            book_age_ms
                AS futures_book_age_ms,

            stream_complete
                AS futures_stream_complete,

            book_valid
                AS futures_book_valid,

            book_stats_complete
                AS futures_book_stats_complete,


            (
                price_features_ready
                AND
                tradable_mid_price > 0
            )
                AS return_input_eligible


        FROM base
        """
    )


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_reference_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            market_reference_source,

            COUNT(*) AS states,

            SUM(
                CASE
                    WHEN state_research_ready
                    THEN 1
                    ELSE 0
                END
            )
                AS research_ready,

            ROUND(
                MEDIAN(
                    futures_spread_bps
                ),
                4
            )
                AS median_spread_bps,

            ROUND(
                MEDIAN(
                    perpetual_mid_vs_index_bps
                ),
                4
            )
                AS median_basis_bps

        FROM futures_reference_state

        GROUP BY
            market_reference_source

        ORDER BY
            states DESC
        """
    ).fetchall()


    print(
        "\nFUTURES REFERENCE STATE"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Source':<24}"
        f"{'States':>12}"
        f"{'Research':>12}"
        f"{'Med spread':>16}"
        f"{'Med basis':>16}"
    )


    for row in rows:

        source = (
            row[0]
            if row[0] is not None
            else "missing"
        )

        print(
            f"{source:<24}"
            f"{row[1]:>12,}"
            f"{row[2]:>12,}"
            f"{str(row[3]):>16}"
            f"{str(row[4]):>16}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    connection = (
        open_research_database()
    )

    try:

        build_canonical(
            connection
        )

        create_futures_reference_state(
            connection
        )

        print_reference_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
