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

from data_engineering.normalization.build_normalized import (
    build_normalized,
)


SMILE_GRID_VERSION = "0.1"


# --------------------------------------------------
# Standard log-moneyness grid
# --------------------------------------------------

SMILE_NODES = (
    -0.10,
    -0.05,
    -0.025,
    0.00,
    0.025,
    0.05,
    0.10,
)


def smile_nodes_sql() -> str:

    rows = []

    for node in SMILE_NODES:

        rows.append(
            f"({node:.6f})"
        )

    return ",\n".join(
        rows
    )


# --------------------------------------------------
# Surface input diagnostics
# --------------------------------------------------

def create_surface_input_state(
    connection
) -> None:
    """
    One row per:

        state_id x option_family_id

    summarizing how much normalized option information is
    available before interpolation.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        surface_input_state AS

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


            MAX(
                option_forward_price
            )
                AS option_forward_price,


            MAX(
                option_forward_vs_perpetual_bps
            )
                AS option_forward_vs_perpetual_bps,


            COUNT(*)
                AS contract_rows,


            SUM(
                CASE
                    WHEN is_otm_or_atm IS TRUE
                    THEN 1
                    ELSE 0
                END
            )
                AS otm_atm_rows,


            SUM(
                CASE
                    WHEN surface_base_eligible
                    THEN 1
                    ELSE 0
                END
            )
                AS base_eligible_rows,


            SUM(
                CASE
                    WHEN surface_two_sided_eligible
                    THEN 1
                    ELSE 0
                END
            )
                AS two_sided_eligible_rows,


            SUM(
                CASE
                    WHEN
                        surface_base_eligible
                        AND log_moneyness < 0
                    THEN 1
                    ELSE 0
                END
            )
                AS base_put_wing_rows,


            SUM(
                CASE
                    WHEN
                        surface_base_eligible
                        AND log_moneyness > 0
                    THEN 1
                    ELSE 0
                END
            )
                AS base_call_wing_rows,


            MIN(
                CASE
                    WHEN surface_base_eligible
                    THEN log_moneyness
                    ELSE NULL
                END
            )
                AS base_min_log_moneyness,


            MAX(
                CASE
                    WHEN surface_base_eligible
                    THEN log_moneyness
                    ELSE NULL
                END
            )
                AS base_max_log_moneyness,


            MEDIAN(
                CASE
                    WHEN
                        surface_two_sided_eligible
                        AND option_relative_spread_bps
                            IS NOT NULL
                    THEN option_relative_spread_bps
                    ELSE NULL
                END
            )
                AS median_two_sided_spread_bps,


            quantile_cont(
                CASE
                    WHEN
                        surface_two_sided_eligible
                        AND option_relative_spread_bps
                            IS NOT NULL
                    THEN option_relative_spread_bps
                    ELSE NULL
                END,
                0.90
            )
                AS p90_two_sided_spread_bps,


            MAX(
                state_research_ready
            )
                AS state_research_ready


        FROM normalized_option_chain

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
        """
    )


# --------------------------------------------------
# Long-form interpolated smile nodes
# --------------------------------------------------

def create_smile_node_state(
    connection
) -> None:
    """
    Interpolate the observed OTM smile onto a fixed grid:

        k = ln(K/F)

        -10%, -5%, -2.5%, 0, +2.5%, +5%, +10%

    Two source tiers are produced:

        preferred:
            preferred_iv, which uses bid/ask IV midpoint when
            available and otherwise mark IV.

        two_sided:
            only rows with positive, non-crossed bid/ask prices
            and valid bid/ask IV midpoint.

    There is NO extrapolation.

    For k < 0 only puts are eligible.
    For k > 0 only calls are eligible.
    At k = 0 both sides may bracket the node.

    The exact interpolation span is persisted so later models
    can impose tighter quality thresholds without rebuilding
    the normalized chain.
    """

    nodes = smile_nodes_sql()

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW
        smile_node_state AS

        WITH targets(target_log_moneyness) AS (

            VALUES
                {nodes}
        ),


        source_definitions AS (

            SELECT
                'preferred'
                    AS surface_source

            UNION ALL

            SELECT
                'two_sided'
                    AS surface_source
        ),


        candidates AS (

            SELECT
                n.state_id,

                n.state_time,

                n.underlying,

                n.session_id,

                n.expiry_date,

                n.session_start,

                n.session_end,

                n.minutes_to_expiry,

                n.option_family_id,

                n.option_forward_price,

                n.instrument_name,

                LOWER(
                    CAST(
                        n.option_type
                        AS VARCHAR
                    )
                )
                    AS option_type_normalized,

                n.strike,

                n.log_moneyness,

                n.delta,

                n.quote_age_ms,

                n.option_relative_spread_bps,

                n.preferred_iv,

                n.iv_mid,

                s.surface_source,

                t.target_log_moneyness,


                CASE

                    WHEN s.surface_source
                        = 'preferred'
                    THEN n.preferred_iv

                    WHEN s.surface_source
                        = 'two_sided'
                    THEN n.iv_mid

                    ELSE NULL

                END
                    AS source_iv


            FROM normalized_option_chain n


            CROSS JOIN targets t


            CROSS JOIN source_definitions s


            WHERE
                (
                    (
                        s.surface_source
                            = 'preferred'

                        AND

                        n.surface_base_eligible
                    )

                    OR

                    (
                        s.surface_source
                            = 'two_sided'

                        AND

                        n.surface_two_sided_eligible
                    )
                )


                AND


                (
                    (
                        t.target_log_moneyness < 0

                        AND

                        LOWER(
                            CAST(
                                n.option_type
                                AS VARCHAR
                            )
                        )
                        IN (
                            'put',
                            'p'
                        )
                    )

                    OR

                    (
                        t.target_log_moneyness > 0

                        AND

                        LOWER(
                            CAST(
                                n.option_type
                                AS VARCHAR
                            )
                        )
                        IN (
                            'call',
                            'c'
                        )
                    )

                    OR

                    t.target_log_moneyness = 0
                )


                AND


                (
                    CASE

                        WHEN s.surface_source
                            = 'preferred'
                        THEN n.preferred_iv

                        WHEN s.surface_source
                            = 'two_sided'
                        THEN n.iv_mid

                        ELSE NULL

                    END
                ) > 0
        ),


        lower_ranked AS (

            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        state_id,
                        option_family_id,
                        surface_source,
                        target_log_moneyness

                    ORDER BY
                        log_moneyness DESC,
                        strike DESC,
                        instrument_name
                )
                    AS rn

            FROM candidates

            WHERE
                log_moneyness
                    <= target_log_moneyness
        ),


        upper_ranked AS (

            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        state_id,
                        option_family_id,
                        surface_source,
                        target_log_moneyness

                    ORDER BY
                        log_moneyness ASC,
                        strike ASC,
                        instrument_name
                )
                    AS rn

            FROM candidates

            WHERE
                log_moneyness
                    >= target_log_moneyness
        ),


        lower_points AS (

            SELECT *
            EXCLUDE (rn)

            FROM lower_ranked

            WHERE rn = 1
        ),


        upper_points AS (

            SELECT *
            EXCLUDE (rn)

            FROM upper_ranked

            WHERE rn = 1
        ),


        state_family_source_grid AS (

            SELECT
                i.state_id,

                i.state_time,

                i.underlying,

                i.session_id,

                i.expiry_date,

                i.session_start,

                i.session_end,

                i.minutes_to_expiry,

                i.option_family_id,

                i.option_forward_price,

                i.state_research_ready,

                s.surface_source,

                t.target_log_moneyness

            FROM surface_input_state i

            CROSS JOIN source_definitions s

            CROSS JOIN targets t
        )


        SELECT
            g.state_id,

            g.state_time,

            g.underlying,

            g.session_id,

            g.expiry_date,

            g.session_start,

            g.session_end,

            g.minutes_to_expiry,

            g.option_family_id,

            g.option_forward_price,

            g.state_research_ready,

            g.surface_source,

            g.target_log_moneyness,


            l.instrument_name
                AS left_instrument_name,

            l.log_moneyness
                AS left_log_moneyness,

            l.source_iv
                AS left_iv,

            l.delta
                AS left_delta,

            l.quote_age_ms
                AS left_quote_age_ms,

            l.option_relative_spread_bps
                AS left_spread_bps,


            u.instrument_name
                AS right_instrument_name,

            u.log_moneyness
                AS right_log_moneyness,

            u.source_iv
                AS right_iv,

            u.delta
                AS right_delta,

            u.quote_age_ms
                AS right_quote_age_ms,

            u.option_relative_spread_bps
                AS right_spread_bps,


            CASE

                WHEN
                    l.source_iv IS NULL
                    OR
                    u.source_iv IS NULL

                THEN NULL


                WHEN
                    l.log_moneyness
                        = u.log_moneyness

                THEN l.source_iv


                ELSE
                    l.source_iv

                    +

                    (
                        g.target_log_moneyness
                        - l.log_moneyness
                    )

                    /

                    NULLIF(
                        u.log_moneyness
                        - l.log_moneyness,
                        0
                    )

                    *

                    (
                        u.source_iv
                        - l.source_iv
                    )

            END
                AS interpolated_iv,


            CASE

                WHEN
                    l.log_moneyness IS NULL
                    OR
                    u.log_moneyness IS NULL

                THEN NULL

                ELSE
                    u.log_moneyness
                    - l.log_moneyness

            END
                AS interpolation_span_log_moneyness,


            GREATEST(
                l.quote_age_ms,
                u.quote_age_ms
            )
                AS interpolation_max_quote_age_ms,


            GREATEST(
                l.option_relative_spread_bps,
                u.option_relative_spread_bps
            )
                AS interpolation_max_spread_bps,


            (
                l.source_iv IS NOT NULL

                AND

                u.source_iv IS NOT NULL
            )
                AS node_bracketed,


            (
                l.source_iv IS NOT NULL

                AND

                u.source_iv IS NOT NULL

                AND

                l.log_moneyness
                    <= g.target_log_moneyness

                AND

                u.log_moneyness
                    >= g.target_log_moneyness
            )
                AS node_interpolated_without_extrapolation


        FROM state_family_source_grid g


        LEFT JOIN lower_points l

            ON
                g.state_id
                = l.state_id

            AND
                g.option_family_id
                = l.option_family_id

            AND
                g.surface_source
                = l.surface_source

            AND
                g.target_log_moneyness
                = l.target_log_moneyness


        LEFT JOIN upper_points u

            ON
                g.state_id
                = u.state_id

            AND
                g.option_family_id
                = u.option_family_id

            AND
                g.surface_source
                = u.surface_source

            AND
                g.target_log_moneyness
                = u.target_log_moneyness
        """
    )


# --------------------------------------------------
# Wide smile-state feature view
# --------------------------------------------------

def create_option_smile_state(
    connection
) -> None:
    """
    One row per:

        state_id x option_family_id x surface_source

    This is the first model-facing option-surface state.

    "rr" labels below describe simple symmetric log-moneyness
    wing differences, NOT delta-risk-reversal conventions.
    """

    connection.execute(
        """
        CREATE OR REPLACE VIEW
        option_smile_state AS

        WITH pivoted AS (

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

                MAX(
                    option_forward_price
                )
                    AS option_forward_price,

                MAX(
                    state_research_ready
                )
                    AS state_research_ready,

                surface_source,


                MAX(
                    CASE
                        WHEN target_log_moneyness = -0.10
                        THEN interpolated_iv
                    END
                )
                    AS iv_k_m100,


                MAX(
                    CASE
                        WHEN target_log_moneyness = -0.05
                        THEN interpolated_iv
                    END
                )
                    AS iv_k_m050,


                MAX(
                    CASE
                        WHEN target_log_moneyness = -0.025
                        THEN interpolated_iv
                    END
                )
                    AS iv_k_m025,


                MAX(
                    CASE
                        WHEN target_log_moneyness = 0.00
                        THEN interpolated_iv
                    END
                )
                    AS iv_atm_forward,


                MAX(
                    CASE
                        WHEN target_log_moneyness = 0.025
                        THEN interpolated_iv
                    END
                )
                    AS iv_k_p025,


                MAX(
                    CASE
                        WHEN target_log_moneyness = 0.05
                        THEN interpolated_iv
                    END
                )
                    AS iv_k_p050,


                MAX(
                    CASE
                        WHEN target_log_moneyness = 0.10
                        THEN interpolated_iv
                    END
                )
                    AS iv_k_p100,


                SUM(
                    CASE
                        WHEN interpolated_iv IS NOT NULL
                        THEN 1
                        ELSE 0
                    END
                )
                    AS smile_node_count,


                MAX(
                    interpolation_span_log_moneyness
                )
                    AS max_interpolation_span,


                MEDIAN(
                    interpolation_span_log_moneyness
                )
                    AS median_interpolation_span,


                MAX(
                    interpolation_max_quote_age_ms
                )
                    AS max_node_quote_age_ms,


                MAX(
                    interpolation_max_spread_bps
                )
                    AS max_node_spread_bps


            FROM smile_node_state

            GROUP BY
                state_id,
                state_time,
                underlying,
                session_id,
                expiry_date,
                session_start,
                session_end,
                minutes_to_expiry,
                option_family_id,
                surface_source
        )


        SELECT
            p.*,


            i.contract_rows,

            i.otm_atm_rows,

            i.base_eligible_rows,

            i.two_sided_eligible_rows,

            i.base_put_wing_rows,

            i.base_call_wing_rows,

            i.base_min_log_moneyness,

            i.base_max_log_moneyness,

            i.median_two_sided_spread_bps,

            i.p90_two_sided_spread_bps,


            (
                iv_k_p025
                - iv_k_m025
            )
                AS symmetric_025_iv_difference,


            (
                iv_k_p050
                - iv_k_m050
            )
                AS symmetric_050_iv_difference,


            (
                iv_k_p100
                - iv_k_m100
            )
                AS symmetric_100_iv_difference,


            (
                (
                    iv_k_p025
                    + iv_k_m025
                ) / 2.0
                - iv_atm_forward
            )
                AS curvature_025,


            (
                (
                    iv_k_p050
                    + iv_k_m050
                ) / 2.0
                - iv_atm_forward
            )
                AS curvature_050,


            (
                (
                    iv_k_p100
                    + iv_k_m100
                ) / 2.0
                - iv_atm_forward
            )
                AS curvature_100,


            CASE

                WHEN
                    iv_k_p025 IS NOT NULL
                    AND
                    iv_k_m025 IS NOT NULL

                THEN
                    (
                        iv_k_p025
                        - iv_k_m025
                    ) / 0.05

                ELSE NULL

            END
                AS local_skew_slope_025,


            CASE

                WHEN
                    iv_k_p050 IS NOT NULL
                    AND
                    iv_k_m050 IS NOT NULL

                THEN
                    (
                        iv_k_p050
                        - iv_k_m050
                    ) / 0.10

                ELSE NULL

            END
                AS local_skew_slope_050,


            (
                smile_node_count = 7
            )
                AS full_smile_grid_available,


            (
                iv_atm_forward IS NOT NULL

                AND

                iv_k_m025 IS NOT NULL

                AND

                iv_k_p025 IS NOT NULL
            )
                AS core_smile_available


        FROM pivoted p


        LEFT JOIN surface_input_state i

            ON
                p.state_id
                = i.state_id

            AND
                p.option_family_id
                = i.option_family_id
        """
    )


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_smile_coverage(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            option_family_id,

            surface_source,

            COUNT(*)
                AS states,

            SUM(
                CASE
                    WHEN iv_atm_forward IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            )
                AS atm_states,

            SUM(
                CASE
                    WHEN core_smile_available
                    THEN 1
                    ELSE 0
                END
            )
                AS core_smiles,

            SUM(
                CASE
                    WHEN full_smile_grid_available
                    THEN 1
                    ELSE 0
                END
            )
                AS full_smiles,

            ROUND(
                MEDIAN(
                    max_interpolation_span
                ),
                6
            )
                AS median_max_span,

            ROUND(
                quantile_cont(
                    max_interpolation_span,
                    0.95
                ),
                6
            )
                AS p95_max_span

        FROM option_smile_state

        GROUP BY
            option_family_id,
            surface_source

        ORDER BY
            option_family_id,
            surface_source
        """
    ).fetchall()


    print(
        "\nSMILE GRID COVERAGE"
    )

    print(
        "=" * 145
    )

    print(
        f"{'Family':<42}"
        f"{'Source':<12}"
        f"{'States':>10}"
        f"{'ATM':>10}"
        f"{'Core':>10}"
        f"{'Full':>10}"
        f"{'Med span':>14}"
        f"{'p95 span':>14}"
    )


    for row in rows:

        print(
            f"{row[0]:<42}"
            f"{row[1]:<12}"
            f"{row[2]:>10,}"
            f"{row[3]:>10,}"
            f"{row[4]:>10,}"
            f"{row[5]:>10,}"
            f"{str(row[6]):>14}"
            f"{str(row[7]):>14}"
        )


def print_smile_feature_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            surface_source,

            COUNT(*)
                AS states,

            ROUND(
                MEDIAN(
                    iv_atm_forward
                ),
                4
            )
                AS median_atm_iv,

            ROUND(
                MEDIAN(
                    symmetric_050_iv_difference
                ),
                4
            )
                AS median_sym_050,

            ROUND(
                MEDIAN(
                    curvature_050
                ),
                4
            )
                AS median_curvature_050

        FROM option_smile_state

        WHERE
            core_smile_available

        GROUP BY
            surface_source

        ORDER BY
            surface_source
        """
    ).fetchall()


    print(
        "\nSMILE FEATURE SUMMARY"
    )

    print(
        "=" * 90
    )

    print(
        f"{'Source':<16}"
        f"{'States':>12}"
        f"{'Med ATM IV':>16}"
        f"{'Med +/-5 diff':>18}"
        f"{'Med curvature':>18}"
    )


    for row in rows:

        print(
            f"{row[0]:<16}"
            f"{row[1]:>12,}"
            f"{str(row[2]):>16}"
            f"{str(row[3]):>18}"
            f"{str(row[4]):>18}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    connection = (
        open_research_database()
    )

    try:

        build_normalized(
            connection
        )

        create_surface_input_state(
            connection
        )

        create_smile_node_state(
            connection
        )

        create_option_smile_state(
            connection
        )

        print_smile_coverage(
            connection
        )

        print_smile_feature_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
