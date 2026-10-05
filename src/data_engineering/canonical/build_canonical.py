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
    initialize_metadata,
    open_research_database,
    refresh_raw_views,
)

from data_engineering.alignment.state_index import (
    create_state_index,
)

from data_engineering.quality_checks.quality_contract import (
    QUALITY_CONTRACT_VERSION,
    create_quality_view,
)


CANONICAL_SCHEMA_VERSION = "0.2"


# --------------------------------------------------
# Canonical market-state view
# --------------------------------------------------

def create_canonical_market_state(
    connection
) -> None:
    """
    One row per:

        underlying x 5-minute state

    No bad states are dropped. Quality/readiness fields from
    state_quality travel with every row.

    If multiple futures rows somehow exist for the same
    underlying/state, one deterministic row is attached for
    inspection, while futures_unique/price_features_ready stay
    false in the quality layer.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        canonical_market_state AS

        WITH futures_ranked AS (

            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        underlying,
                        bar_end

                    ORDER BY
                        instrument_name
                )
                    AS _canonical_rank

            FROM raw_futures_5m
        )

        SELECT
            q.*,

            f.* EXCLUDE (
                underlying,
                bar_end,
                _canonical_rank
            )

        FROM state_quality q

        LEFT JOIN futures_ranked f

            ON
                q.underlying
                = f.underlying

            AND
                q.state_time
                = f.bar_end

            AND
                f._canonical_rank = 1
        """
    )


# --------------------------------------------------
# Canonical option-chain view
# --------------------------------------------------

def create_canonical_option_chain(
    connection
) -> None:
    """
    One row per:

        state_id x option instrument

    The view retains all raw option rows and annotates them with:
      - state/session identity,
      - state-level quality gates,
      - synchronized futures reference prices,
      - contract-level structural/IV/quote flags.

    This is the direct parent dataset for smile/surface building.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        canonical_option_chain AS

        WITH futures_ranked AS (

            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        underlying,
                        bar_end

                    ORDER BY
                        instrument_name
                )
                    AS _canonical_rank

            FROM raw_futures_5m
        )

        SELECT
            q.state_id,

            q.state_time,

            q.underlying,

            q.session_id,

            q.expiry_date,

            q.session_start,

            q.session_end,

            q.minutes_to_expiry,

            q.is_opening_state,


            q.collection_complete,

            q.price_features_ready,

            q.option_snapshot_ready,

            q.state_research_ready,

            q.l2_boundary_ready,

            q.l2_intrawindow_ready,

            q.opening_provenance_available,

            q.feed_provenance_available,

            q.universe_provenance_available,


            f.instrument_name
                AS futures_instrument_name,

            f.mid_price
                AS futures_mid_price,

            f.mark_price
                AS futures_mark_price,

            f.index_price
                AS futures_index_price,

            f.underlying_price
                AS futures_underlying_price,

            f.open_interest
                AS futures_open_interest,

            f.ticker_age_ms
                AS futures_ticker_age_ms,

            f.stream_complete
                AS futures_stream_complete,

            f.book_valid
                AS futures_book_valid,


            o.* EXCLUDE (
                scheduled_snapshot_time,
                underlying,
                session_id,
                expiry_date,
                expiry,
                minutes_to_expiry
            ),


            (
                o.has_ticker_state
                    IS TRUE
            )
                AS contract_ticker_ready,


            (
                (
                    o.best_bid_price IS NULL
                    OR o.best_bid_price >= 0
                )

                AND

                (
                    o.best_ask_price IS NULL
                    OR o.best_ask_price >= 0
                )
            )
                AS contract_quotes_nonnegative,


            (
                o.best_bid_price IS NULL

                OR

                o.best_ask_price IS NULL

                OR

                o.best_bid_price <= 0

                OR

                o.best_ask_price <= 0

                OR

                o.best_bid_price
                    <= o.best_ask_price
            )
                AS contract_market_not_crossed,


            (
                (
                    o.delta IS NULL
                    OR (
                        o.delta >= -1
                        AND o.delta <= 1
                    )
                )

                AND

                (
                    o.gamma IS NULL
                    OR o.gamma >= 0
                )

                AND

                (
                    o.open_interest IS NULL
                    OR o.open_interest >= 0
                )
            )
                AS contract_structural_values_valid,


            (
                o.mark_iv > 0
            )
                AS contract_mark_iv_valid,


            (
                o.best_bid_price > 0
                AND
                o.best_ask_price > 0
                AND
                o.best_bid_price
                    <= o.best_ask_price
            )
                AS contract_two_sided_quote,


            (
                o.has_ticker_state
                    IS TRUE

                AND

                (
                    o.best_bid_price IS NULL
                    OR o.best_bid_price >= 0
                )

                AND

                (
                    o.best_ask_price IS NULL
                    OR o.best_ask_price >= 0
                )

                AND

                (
                    o.best_bid_price IS NULL

                    OR

                    o.best_ask_price IS NULL

                    OR

                    o.best_bid_price <= 0

                    OR

                    o.best_ask_price <= 0

                    OR

                    o.best_bid_price
                        <= o.best_ask_price
                )

                AND

                (
                    o.delta IS NULL
                    OR (
                        o.delta >= -1
                        AND o.delta <= 1
                    )
                )

                AND

                (
                    o.gamma IS NULL
                    OR o.gamma >= 0
                )

                AND

                (
                    o.open_interest IS NULL
                    OR o.open_interest >= 0
                )
            )
                AS contract_structurally_valid,


            -- v0.2 (audit B1): option-side readiness only. The option
            -- chain does not depend on the futures stream; the futures
            -- and option blocks are joined only in the final readiness
            -- flags of time_series_feature_state.
            (
                COALESCE(
                    q.option_snapshot_ready,
                    FALSE
                )

                AND

                o.has_ticker_state
                    IS TRUE

                AND

                o.mark_iv > 0

                AND

                (
                    o.best_bid_price IS NULL
                    OR o.best_bid_price >= 0
                )

                AND

                (
                    o.best_ask_price IS NULL
                    OR o.best_ask_price >= 0
                )

                AND

                (
                    o.best_bid_price IS NULL

                    OR

                    o.best_ask_price IS NULL

                    OR

                    o.best_bid_price <= 0

                    OR

                    o.best_ask_price <= 0

                    OR

                    o.best_bid_price
                        <= o.best_ask_price
                )

                AND

                (
                    o.delta IS NULL
                    OR (
                        o.delta >= -1
                        AND o.delta <= 1
                    )
                )

                AND

                (
                    o.gamma IS NULL
                    OR o.gamma >= 0
                )

                AND

                (
                    o.open_interest IS NULL
                    OR o.open_interest >= 0
                )
            )
                AS iv_input_eligible


        FROM raw_option_grid o


        INNER JOIN state_quality q

            ON
                q.expiry_date
                = o.expiry

            AND
                q.underlying
                = o.underlying

            AND
                q.state_time
                = o.scheduled_snapshot_time


        LEFT JOIN futures_ranked f

            ON
                q.underlying
                = f.underlying

            AND
                q.state_time
                = f.bar_end

            AND
                f._canonical_rank = 1
        """
    )


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def register_canonical_metadata(
    connection
) -> None:

    initialize_metadata(
        connection
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO
        catalog_metadata
        VALUES (
            'quality_contract_version',
            ?
        )
        """,
        [
            QUALITY_CONTRACT_VERSION
        ]
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO
        catalog_metadata
        VALUES (
            'canonical_schema_version',
            ?
        )
        """,
        [
            CANONICAL_SCHEMA_VERSION
        ]
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_canonical_views(
    connection
) -> None:

    market = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,

            COUNT(
                DISTINCT state_id
            )
                AS unique_states,

            SUM(
                CASE
                    WHEN state_research_ready
                    THEN 1
                    ELSE 0
                END
            )
                AS research_ready,

            SUM(
                CASE
                    WHEN collection_complete
                    THEN 1
                    ELSE 0
                END
            )
                AS collected

        FROM canonical_market_state
        """
    ).fetchone()


    market_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    COUNT(*) AS n

                FROM canonical_market_state

                GROUP BY
                    state_id

                HAVING
                    COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )


    options = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,

            COUNT(
                DISTINCT state_id
            )
                AS states,

            SUM(
                CASE
                    WHEN iv_input_eligible
                    THEN 1
                    ELSE 0
                END
            )
                AS iv_eligible

        FROM canonical_option_chain
        """
    ).fetchone()


    option_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    instrument_name,
                    COUNT(*) AS n

                FROM canonical_option_chain

                GROUP BY
                    state_id,
                    instrument_name

                HAVING
                    COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )


    print(
        "\nCANONICAL DATASET"
    )

    print(
        "=" * 80
    )

    print(
        "\ncanonical_market_state"
    )

    print(
        f"Rows:             "
        f"{market[0]:,}"
    )

    print(
        f"Unique states:    "
        f"{market[1]:,}"
    )

    print(
        f"Collected states: "
        f"{market[3]:,}"
    )

    print(
        f"Research ready:   "
        f"{market[2]:,}"
    )

    print(
        f"Duplicate IDs:    "
        f"{market_duplicates:,}"
    )


    print(
        "\ncanonical_option_chain"
    )

    print(
        f"Rows:             "
        f"{options[0]:,}"
    )

    print(
        f"States:           "
        f"{options[1]:,}"
    )

    print(
        f"IV eligible rows: "
        f"{options[2]:,}"
    )

    print(
        f"Duplicate keys:   "
        f"{option_duplicates:,}"
    )


    if market_duplicates:

        raise RuntimeError(
            "canonical_market_state "
            "contains duplicate state_id values."
        )

    if option_duplicates:

        raise RuntimeError(
            "canonical_option_chain "
            "contains duplicate "
            "(state_id, instrument_name) keys."
        )


# --------------------------------------------------
# Build
# --------------------------------------------------

def build_canonical(
    connection
) -> None:

    refresh_raw_views(
        connection
    )

    create_state_index(
        connection
    )

    create_quality_view(
        connection
    )

    create_canonical_market_state(
        connection
    )

    create_canonical_option_chain(
        connection
    )

    register_canonical_metadata(
        connection
    )

    validate_canonical_views(
        connection
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

    finally:

        connection.close()


if __name__ == "__main__":

    main()
