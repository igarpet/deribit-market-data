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

from data_engineering.normalization.futures_reference import (
    create_futures_reference_state,
)


# --------------------------------------------------
# Option-family identity
# --------------------------------------------------

def create_option_family_chain(
    connection
) -> None:
    """
    Add an explicit pricing-family identifier to every option row.

    Important:
    Deribit's product_group is a gateway/tier classification,
    not a pricing-family identifier. BTC/ETH may have both
    inverse and linear USDC option families. They must not share
    one forward estimate.

    Family key:
        underlying
        + instrument_type
        + settlement_currency
        + api_currency
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        option_family_chain AS

        SELECT
            c.*,

            c.underlying
                || '|'
                || COALESCE(
                    LOWER(
                        CAST(
                            c.instrument_type
                            AS VARCHAR
                        )
                    ),
                    'unknown_type'
                )
                || '|'
                || COALESCE(
                    UPPER(
                        CAST(
                            c.settlement_currency
                            AS VARCHAR
                        )
                    ),
                    'unknown_settlement'
                )
                || '|'
                || COALESCE(
                    UPPER(
                        CAST(
                            c.api_currency
                            AS VARCHAR
                        )
                    ),
                    'unknown_api'
                )
                    AS option_family_id

        FROM canonical_option_chain c
        """
    )


# --------------------------------------------------
# Family-specific forward state
# --------------------------------------------------

def create_option_forward_state(
    connection
) -> None:
    """
    Build one forward reference per:

        state_id x option_family_id

    The native option ticker underlying_price is used as the
    expiry-forward observation. We aggregate it cross-sectionally
    with a median rather than mixing it with the perpetual.

    The perpetual reference is attached only as a diagnostic
    comparison. It is never silently substituted for the option
    expiry forward.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        option_forward_state AS

        WITH valid_forward_rows AS (

            SELECT
                state_id,

                state_time,

                underlying,

                session_id,

                expiry_date,

                session_start,

                session_end,

                minutes_to_expiry,

                option_family_id,

                instrument_type,

                settlement_currency,

                api_currency,

                underlying_price,

                index_price,

                interest_rate

            FROM option_family_chain

            WHERE
                underlying_price > 0
        ),


        aggregated AS (

            SELECT
                state_id,

                state_time,

                underlying,

                session_id,

                expiry_date,

                session_start,

                session_end,

                minutes_to_expiry,

                option_family_id,

                MIN(
                    instrument_type
                )
                    AS instrument_type,

                MIN(
                    settlement_currency
                )
                    AS settlement_currency,

                MIN(
                    api_currency
                )
                    AS api_currency,


                COUNT(*)
                    AS forward_observation_count,


                MEDIAN(
                    underlying_price
                )
                    AS option_forward_price,


                MIN(
                    underlying_price
                )
                    AS option_forward_min,


                MAX(
                    underlying_price
                )
                    AS option_forward_max,


                quantile_cont(
                    underlying_price,
                    0.05
                )
                    AS option_forward_p05,


                quantile_cont(
                    underlying_price,
                    0.95
                )
                    AS option_forward_p95,


                MEDIAN(
                    CASE
                        WHEN index_price > 0
                        THEN index_price
                        ELSE NULL
                    END
                )
                    AS option_index_price,


                MEDIAN(
                    interest_rate
                )
                    AS option_interest_rate


            FROM valid_forward_rows

            GROUP BY
                state_id,
                state_time,
                underlying,
                session_id,
                expiry_date,
                session_start,
                session_end,
                minutes_to_expiry,
                option_family_id
        )


        SELECT
            a.*,


            CASE

                WHEN a.option_forward_price > 0
                THEN
                    (
                        a.option_forward_p95
                        - a.option_forward_p05
                    )
                    / a.option_forward_price
                    * 10000.0

                ELSE NULL

            END
                AS option_forward_cross_section_bps,


            CASE

                WHEN
                    a.option_forward_price > 0
                    AND
                    a.option_index_price > 0

                THEN
                    (
                        a.option_forward_price
                        / a.option_index_price
                        - 1.0
                    )
                    * 10000.0

                ELSE NULL

            END
                AS option_forward_vs_index_bps,


            r.market_reference_price
                AS perpetual_reference_price,

            r.market_reference_source
                AS perpetual_reference_source,

            r.market_reference_is_tradable
                AS perpetual_reference_is_tradable,


            CASE

                WHEN
                    a.option_forward_price > 0
                    AND
                    r.market_reference_price > 0

                THEN
                    (
                        a.option_forward_price
                        / r.market_reference_price
                        - 1.0
                    )
                    * 10000.0

                ELSE NULL

            END
                AS option_forward_vs_perpetual_bps,


            (
                a.option_forward_price > 0
                AND
                a.forward_observation_count > 0
            )
                AS option_forward_available


        FROM aggregated a

        LEFT JOIN futures_reference_state r

            ON
                a.state_id
                = r.state_id
        """
    )


# --------------------------------------------------
# Normalized option chain
# --------------------------------------------------

def create_normalized_option_chain(
    connection
) -> None:
    """
    Normalize every canonical option row into family-specific
    forward moneyness and transparent quote/IV measures.

    Convention:
        log_moneyness = ln(K / F)

    Therefore:
        log_moneyness > 0  -> strike above forward
        log_moneyness < 0  -> strike below forward

    Surface-base eligibility intentionally does NOT impose a
    final liquidity threshold. Liquidity filters belong in the
    later smile/surface construction layer.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        normalized_option_chain AS

        WITH joined AS (

            SELECT
                c.*,

                f.option_forward_price,

                f.option_index_price,

                f.option_interest_rate,

                f.forward_observation_count,

                f.option_forward_cross_section_bps,

                f.option_forward_vs_index_bps,

                f.perpetual_reference_price,

                f.perpetual_reference_source,

                f.perpetual_reference_is_tradable,

                f.option_forward_vs_perpetual_bps,

                f.option_forward_available


            FROM option_family_chain c


            LEFT JOIN option_forward_state f

                ON
                    c.state_id
                    = f.state_id

                AND
                    c.option_family_id
                    = f.option_family_id
        ),


        normalized AS (

            SELECT
                *,

                date_diff(
                    'millisecond',
                    state_time,
                    session_end
                )
                / 1000.0
                    AS time_to_expiry_seconds,


                date_diff(
                    'millisecond',
                    state_time,
                    session_end
                )
                / (
                    365.0
                    * 24.0
                    * 60.0
                    * 60.0
                    * 1000.0
                )
                    AS time_to_expiry_years_365,


                CASE

                    WHEN
                        strike > 0
                        AND
                        option_forward_price > 0

                    THEN
                        strike
                        / option_forward_price

                    ELSE NULL

                END
                    AS strike_over_forward,


                CASE

                    WHEN
                        strike > 0
                        AND
                        option_forward_price > 0

                    THEN
                        LN(
                            strike
                            / option_forward_price
                        )

                    ELSE NULL

                END
                    AS log_moneyness,


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
                    AS option_mid_price_native,


                CASE

                    WHEN
                        best_bid_price > 0
                        AND
                        best_ask_price > 0
                        AND
                        best_bid_price
                            <= best_ask_price

                    THEN
                        best_ask_price
                        - best_bid_price

                    ELSE NULL

                END
                    AS option_spread_native,


                CASE

                    WHEN
                        bid_iv > 0
                        AND
                        ask_iv > 0
                        AND
                        ask_iv >= bid_iv

                    THEN
                        (
                            bid_iv
                            + ask_iv
                        ) / 2.0

                    ELSE NULL

                END
                    AS iv_mid,


                CASE

                    WHEN
                        bid_iv > 0
                        AND
                        ask_iv > 0
                        AND
                        ask_iv >= bid_iv

                    THEN
                        ask_iv
                        - bid_iv

                    ELSE NULL

                END
                    AS iv_spread

            FROM joined
        )


        SELECT
            *,


            ABS(
                log_moneyness
            )
                AS abs_log_moneyness,


            CASE

                WHEN
                    option_mid_price_native > 0
                    AND
                    option_spread_native >= 0

                THEN
                    option_spread_native
                    / option_mid_price_native
                    * 10000.0

                ELSE NULL

            END
                AS option_relative_spread_bps,


            CASE

                WHEN
                    option_mid_price_native > 0
                    AND
                    mark_price > 0

                THEN
                    (
                        mark_price
                        / option_mid_price_native
                        - 1.0
                    )
                    * 10000.0

                ELSE NULL

            END
                AS option_mark_vs_mid_bps,


            CASE

                WHEN
                    best_bid_price > 0
                    AND
                    best_ask_price > 0
                    AND
                    best_bid_price
                        <= best_ask_price

                THEN 'two_sided'

                WHEN
                    best_bid_price > 0
                    AND
                    (
                        best_ask_price IS NULL
                        OR
                        best_ask_price <= 0
                    )

                THEN 'bid_only'

                WHEN
                    best_ask_price > 0
                    AND
                    (
                        best_bid_price IS NULL
                        OR
                        best_bid_price <= 0
                    )

                THEN 'ask_only'

                ELSE 'no_positive_two_sided_quote'

            END
                AS quote_state,


            CASE

                WHEN
                    LOWER(
                        CAST(
                            option_type
                            AS VARCHAR
                        )
                    )
                    IN (
                        'call',
                        'c'
                    )

                    AND

                    strike >= option_forward_price

                THEN TRUE


                WHEN
                    LOWER(
                        CAST(
                            option_type
                            AS VARCHAR
                        )
                    )
                    IN (
                        'put',
                        'p'
                    )

                    AND

                    strike <= option_forward_price

                THEN TRUE


                WHEN
                    option_forward_price IS NULL

                THEN NULL


                ELSE FALSE

            END
                AS is_otm_or_atm,


            CASE

                WHEN
                    LOWER(
                        CAST(
                            option_type
                            AS VARCHAR
                        )
                    )
                    IN (
                        'call',
                        'c'
                    )

                    AND

                    strike > option_forward_price

                THEN 'OTM'

                WHEN
                    LOWER(
                        CAST(
                            option_type
                            AS VARCHAR
                        )
                    )
                    IN (
                        'put',
                        'p'
                    )

                    AND

                    strike < option_forward_price

                THEN 'OTM'

                WHEN
                    strike = option_forward_price

                THEN 'ATM'

                WHEN
                    option_forward_price IS NULL

                THEN NULL

                ELSE 'ITM'

            END
                AS forward_moneyness_region,


            CASE

                WHEN iv_mid > 0
                THEN iv_mid

                WHEN mark_iv > 0
                THEN mark_iv

                ELSE NULL

            END
                AS preferred_iv,


            CASE

                WHEN iv_mid > 0
                THEN 'bid_ask_iv_mid'

                WHEN mark_iv > 0
                THEN 'mark_iv'

                ELSE NULL

            END
                AS preferred_iv_source,


            -- v0.2 (audit B1): option-side readiness, not the
            -- futures-dependent state_research_ready.
            (
                COALESCE(
                    option_snapshot_ready,
                    FALSE
                )

                AND

                contract_structurally_valid

                AND

                contract_mark_iv_valid

                AND

                option_forward_available

                AND

                option_forward_price > 0

                AND

                strike > 0

                AND

                time_to_expiry_seconds > 0

                AND

                (
                    (
                        LOWER(
                            CAST(
                                option_type
                                AS VARCHAR
                            )
                        )
                        IN (
                            'call',
                            'c'
                        )

                        AND

                        strike
                            >= option_forward_price
                    )

                    OR

                    (
                        LOWER(
                            CAST(
                                option_type
                                AS VARCHAR
                            )
                        )
                        IN (
                            'put',
                            'p'
                        )

                        AND

                        strike
                            <= option_forward_price
                    )
                )
            )
                AS surface_base_eligible,


            (
                COALESCE(
                    option_snapshot_ready,
                    FALSE
                )

                AND

                contract_structurally_valid

                AND

                option_forward_available

                AND

                option_forward_price > 0

                AND

                strike > 0

                AND

                time_to_expiry_seconds > 0

                AND

                best_bid_price > 0

                AND

                best_ask_price > 0

                AND

                best_bid_price
                    <= best_ask_price

                AND

                bid_iv > 0

                AND

                ask_iv > 0

                AND

                ask_iv >= bid_iv

                AND

                (
                    (
                        LOWER(
                            CAST(
                                option_type
                                AS VARCHAR
                            )
                        )
                        IN (
                            'call',
                            'c'
                        )

                        AND

                        strike
                            >= option_forward_price
                    )

                    OR

                    (
                        LOWER(
                            CAST(
                                option_type
                                AS VARCHAR
                            )
                        )
                        IN (
                            'put',
                            'p'
                        )

                        AND

                        strike
                            <= option_forward_price
                    )
                )
            )
                AS surface_two_sided_eligible


        FROM normalized
        """
    )


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_forward_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            option_family_id,

            COUNT(*) AS states,

            ROUND(
                MEDIAN(
                    forward_observation_count
                ),
                2
            )
                AS median_observations,

            ROUND(
                MEDIAN(
                    option_forward_cross_section_bps
                ),
                6
            )
                AS median_dispersion_bps,

            ROUND(
                quantile_cont(
                    option_forward_cross_section_bps,
                    0.99
                ),
                6
            )
                AS p99_dispersion_bps,

            ROUND(
                MEDIAN(
                    option_forward_vs_perpetual_bps
                ),
                4
            )
                AS median_vs_perp_bps

        FROM option_forward_state

        GROUP BY
            option_family_id

        ORDER BY
            option_family_id
        """
    ).fetchall()


    print(
        "\nOPTION FORWARD STATE"
    )

    print(
        "=" * 120
    )

    print(
        f"{'Family':<42}"
        f"{'States':>10}"
        f"{'Med obs':>12}"
        f"{'Med disp':>14}"
        f"{'p99 disp':>14}"
        f"{'Med vs perp':>16}"
    )


    for row in rows:

        print(
            f"{row[0]:<42}"
            f"{row[1]:>10,}"
            f"{str(row[2]):>12}"
            f"{str(row[3]):>14}"
            f"{str(row[4]):>14}"
            f"{str(row[5]):>16}"
        )


def print_normalization_summary(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,

            SUM(
                CASE
                    WHEN option_forward_available
                    THEN 1
                    ELSE 0
                END
            )
                AS forward_rows,

            SUM(
                CASE
                    WHEN is_otm_or_atm IS TRUE
                    THEN 1
                    ELSE 0
                END
            )
                AS otm_rows,

            SUM(
                CASE
                    WHEN surface_base_eligible
                    THEN 1
                    ELSE 0
                END
            )
                AS surface_base,

            SUM(
                CASE
                    WHEN surface_two_sided_eligible
                    THEN 1
                    ELSE 0
                END
            )
                AS surface_two_sided,

            SUM(
                CASE
                    WHEN quote_state = 'two_sided'
                    THEN 1
                    ELSE 0
                END
            )
                AS two_sided_quotes

        FROM normalized_option_chain
        """
    ).fetchone()


    print(
        "\nNORMALIZED OPTION CHAIN"
    )

    print(
        "=" * 80
    )

    print(
        f"Rows:                       "
        f"{row[0]:,}"
    )

    print(
        f"Rows with native forward:   "
        f"{row[1]:,}"
    )

    print(
        f"OTM/ATM rows:               "
        f"{row[2]:,}"
    )

    print(
        f"Surface base eligible:      "
        f"{row[3]:,}"
    )

    print(
        f"Two-sided surface eligible: "
        f"{row[4]:,}"
    )

    print(
        f"Two-sided quotes:           "
        f"{row[5]:,}"
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

        create_option_family_chain(
            connection
        )

        create_option_forward_state(
            connection
        )

        create_normalized_option_chain(
            connection
        )

        print_forward_summary(
            connection
        )

        print_normalization_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
