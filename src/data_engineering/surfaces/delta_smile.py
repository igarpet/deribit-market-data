from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.catalog.database import open_research_database
from data_engineering.surfaces.build_surfaces import build_surfaces
from data_engineering.surfaces.surface_quality import (
    create_surface_quality_state,
)

DELTA_SMILE_VERSION = "0.1"


def create_delta_smile_node_state(connection) -> None:

    connection.execute(
        """
        CREATE OR REPLACE VIEW delta_smile_node_state AS

        WITH targets(
            delta_node,
            target_delta,
            option_side
        ) AS (

            VALUES
                ('put_10d',  -0.10, 'put'),
                ('put_25d',  -0.25, 'put'),
                ('call_25d',  0.25, 'call'),
                ('call_10d',  0.10, 'call')
        ),

        source_definitions AS (

            SELECT 'preferred' AS surface_source
            UNION ALL
            SELECT 'two_sided' AS surface_source
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
                    CAST(n.option_type AS VARCHAR)
                ) AS option_type_normalized,

                n.strike,
                n.log_moneyness,
                n.delta,
                n.quote_age_ms,
                n.option_relative_spread_bps,
                n.preferred_iv,
                n.iv_mid,

                s.surface_source,
                t.delta_node,
                t.target_delta,
                t.option_side,

                CASE
                    WHEN s.surface_source = 'preferred'
                        THEN n.preferred_iv
                    WHEN s.surface_source = 'two_sided'
                        THEN n.iv_mid
                END AS source_iv

            FROM normalized_option_chain n

            CROSS JOIN targets t

            CROSS JOIN source_definitions s

            WHERE
                n.delta IS NOT NULL

                AND (
                    (
                        t.option_side = 'put'
                        AND LOWER(CAST(n.option_type AS VARCHAR))
                            IN ('put', 'p')
                    )
                    OR
                    (
                        t.option_side = 'call'
                        AND LOWER(CAST(n.option_type AS VARCHAR))
                            IN ('call', 'c')
                    )
                )

                AND (
                    (
                        s.surface_source = 'preferred'
                        AND n.surface_base_eligible
                    )
                    OR
                    (
                        s.surface_source = 'two_sided'
                        AND n.surface_two_sided_eligible
                    )
                )

                AND (
                    CASE
                        WHEN s.surface_source = 'preferred'
                            THEN n.preferred_iv
                        WHEN s.surface_source = 'two_sided'
                            THEN n.iv_mid
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
                        delta_node
                    ORDER BY
                        delta DESC,
                        strike,
                        instrument_name
                ) AS rn

            FROM candidates

            WHERE delta <= target_delta
        ),

        upper_ranked AS (

            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        state_id,
                        option_family_id,
                        surface_source,
                        delta_node
                    ORDER BY
                        delta ASC,
                        strike,
                        instrument_name
                ) AS rn

            FROM candidates

            WHERE delta >= target_delta
        ),

        lower_points AS (

            SELECT * EXCLUDE (rn)
            FROM lower_ranked
            WHERE rn = 1
        ),

        upper_points AS (

            SELECT * EXCLUDE (rn)
            FROM upper_ranked
            WHERE rn = 1
        ),

        family_source_grid AS (

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

                t.delta_node,
                t.target_delta,
                t.option_side

            FROM surface_input_state i

            CROSS JOIN source_definitions s

            CROSS JOIN targets t
        )

        SELECT
            g.*,

            l.instrument_name
                AS left_instrument_name,
            l.delta
                AS left_delta,
            l.source_iv
                AS left_iv,
            l.log_moneyness
                AS left_log_moneyness,
            l.quote_age_ms
                AS left_quote_age_ms,
            l.option_relative_spread_bps
                AS left_spread_bps,

            u.instrument_name
                AS right_instrument_name,
            u.delta
                AS right_delta,
            u.source_iv
                AS right_iv,
            u.log_moneyness
                AS right_log_moneyness,
            u.quote_age_ms
                AS right_quote_age_ms,
            u.option_relative_spread_bps
                AS right_spread_bps,

            CASE
                WHEN
                    l.source_iv IS NULL
                    OR u.source_iv IS NULL
                    THEN NULL

                WHEN l.delta = u.delta
                    THEN l.source_iv

                ELSE
                    l.source_iv
                    +
                    (
                        g.target_delta
                        - l.delta
                    )
                    /
                    NULLIF(
                        u.delta
                        - l.delta,
                        0
                    )
                    *
                    (
                        u.source_iv
                        - l.source_iv
                    )
            END AS interpolated_iv,

            CASE
                WHEN
                    l.delta IS NULL
                    OR u.delta IS NULL
                    THEN NULL
                ELSE
                    u.delta
                    - l.delta
            END AS interpolation_span_delta,

            GREATEST(
                l.quote_age_ms,
                u.quote_age_ms
            ) AS interpolation_max_quote_age_ms,

            GREATEST(
                l.option_relative_spread_bps,
                u.option_relative_spread_bps
            ) AS interpolation_max_spread_bps,

            (
                l.source_iv IS NOT NULL
                AND u.source_iv IS NOT NULL
                AND l.delta <= g.target_delta
                AND u.delta >= g.target_delta
            ) AS node_interpolated_without_extrapolation

        FROM family_source_grid g

        LEFT JOIN lower_points l
            ON g.state_id = l.state_id
            AND g.option_family_id = l.option_family_id
            AND g.surface_source = l.surface_source
            AND g.delta_node = l.delta_node

        LEFT JOIN upper_points u
            ON g.state_id = u.state_id
            AND g.option_family_id = u.option_family_id
            AND g.surface_source = u.surface_source
            AND g.delta_node = u.delta_node
        """
    )


def create_delta_smile_state(connection) -> None:

    connection.execute(
        """
        CREATE OR REPLACE VIEW delta_smile_state AS

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

                MAX(option_forward_price)
                    AS option_forward_price,

                MAX(state_research_ready)
                    AS state_research_ready,

                surface_source,

                MAX(
                    CASE
                        WHEN delta_node = 'put_10d'
                        THEN interpolated_iv
                    END
                ) AS iv_put_10d,

                MAX(
                    CASE
                        WHEN delta_node = 'put_25d'
                        THEN interpolated_iv
                    END
                ) AS iv_put_25d,

                MAX(
                    CASE
                        WHEN delta_node = 'call_25d'
                        THEN interpolated_iv
                    END
                ) AS iv_call_25d,

                MAX(
                    CASE
                        WHEN delta_node = 'call_10d'
                        THEN interpolated_iv
                    END
                ) AS iv_call_10d,

                SUM(
                    CASE
                        WHEN interpolated_iv IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS delta_node_count,

                MAX(interpolation_span_delta)
                    AS max_delta_interpolation_span,

                MEDIAN(interpolation_span_delta)
                    AS median_delta_interpolation_span,

                MAX(interpolation_max_quote_age_ms)
                    AS max_delta_node_quote_age_ms,

                MAX(interpolation_max_spread_bps)
                    AS max_delta_node_spread_bps

            FROM delta_smile_node_state

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

            s.iv_atm_forward,

            (
                iv_call_25d
                - iv_put_25d
            ) AS rr_25d,

            (
                (
                    iv_call_25d
                    + iv_put_25d
                ) / 2.0
                - iv_atm_forward
            ) AS bf_25d,

            (
                iv_call_10d
                - iv_put_10d
            ) AS rr_10d,

            (
                (
                    iv_call_10d
                    + iv_put_10d
                ) / 2.0
                - iv_atm_forward
            ) AS bf_10d,

            (
                iv_atm_forward IS NOT NULL
                AND iv_put_25d IS NOT NULL
                AND iv_call_25d IS NOT NULL
            ) AS delta_25d_smile_available,

            (
                iv_atm_forward IS NOT NULL
                AND iv_put_10d IS NOT NULL
                AND iv_put_25d IS NOT NULL
                AND iv_call_25d IS NOT NULL
                AND iv_call_10d IS NOT NULL
            ) AS full_delta_smile_available

        FROM pivoted p

        LEFT JOIN option_smile_state s
            ON p.state_id = s.state_id
            AND p.option_family_id = s.option_family_id
            AND p.surface_source = s.surface_source
        """
    )


def create_surface_source_comparison(connection) -> None:

    connection.execute(
        """
        CREATE OR REPLACE VIEW surface_source_comparison AS

        WITH preferred AS (

            SELECT *
            FROM delta_smile_state
            WHERE surface_source = 'preferred'
        ),

        two_sided AS (

            SELECT *
            FROM delta_smile_state
            WHERE surface_source = 'two_sided'
        )

        SELECT
            p.state_id,
            p.state_time,
            p.underlying,
            p.session_id,
            p.expiry_date,
            p.minutes_to_expiry,
            p.option_family_id,

            p.iv_atm_forward
                AS preferred_atm_iv,

            t.iv_atm_forward
                AS two_sided_atm_iv,

            t.iv_atm_forward
                - p.iv_atm_forward
                AS two_sided_minus_preferred_atm_iv,

            p.rr_25d
                AS preferred_rr_25d,

            t.rr_25d
                AS two_sided_rr_25d,

            t.rr_25d
                - p.rr_25d
                AS two_sided_minus_preferred_rr_25d,

            p.bf_25d
                AS preferred_bf_25d,

            t.bf_25d
                AS two_sided_bf_25d,

            t.bf_25d
                - p.bf_25d
                AS two_sided_minus_preferred_bf_25d,

            (
                p.iv_atm_forward IS NOT NULL
                AND t.iv_atm_forward IS NOT NULL
            ) AS atm_overlap,

            (
                p.delta_25d_smile_available
                AND t.delta_25d_smile_available
            ) AS delta_25d_overlap

        FROM preferred p

        INNER JOIN two_sided t
            ON p.state_id = t.state_id
            AND p.option_family_id = t.option_family_id
        """
    )


def print_delta_smile_coverage(connection) -> None:

    rows = connection.execute(
        """
        SELECT
            option_family_id,
            surface_source,
            COUNT(*) AS states,

            SUM(
                CASE
                    WHEN delta_25d_smile_available
                    THEN 1 ELSE 0
                END
            ) AS smile_25d,

            SUM(
                CASE
                    WHEN full_delta_smile_available
                    THEN 1 ELSE 0
                END
            ) AS full_delta,

            ROUND(
                MEDIAN(max_delta_interpolation_span),
                6
            ) AS median_max_span,

            ROUND(
                quantile_cont(
                    max_delta_interpolation_span,
                    0.95
                ),
                6
            ) AS p95_max_span

        FROM delta_smile_state

        GROUP BY
            option_family_id,
            surface_source

        ORDER BY
            option_family_id,
            surface_source
        """
    ).fetchall()

    print("\nDELTA SMILE COVERAGE")
    print("=" * 130)

    print(
        f"{'Family':<42}"
        f"{'Source':<12}"
        f"{'States':>10}"
        f"{'25d smile':>12}"
        f"{'Full delta':>12}"
        f"{'Med span':>14}"
        f"{'p95 span':>14}"
    )

    for row in rows:
        print(
            f"{row[0]:<42}"
            f"{row[1]:<12}"
            f"{row[2]:>10,}"
            f"{row[3]:>12,}"
            f"{row[4]:>12,}"
            f"{str(row[5]):>14}"
            f"{str(row[6]):>14}"
        )


def print_source_overlap_summary(connection) -> None:

    row = connection.execute(
        """
        SELECT
            SUM(
                CASE
                    WHEN atm_overlap
                    THEN 1 ELSE 0
                END
            ) AS atm_overlap,

            ROUND(
                MEDIAN(
                    CASE
                        WHEN atm_overlap
                        THEN two_sided_minus_preferred_atm_iv
                    END
                ),
                6
            ) AS median_atm_difference,

            SUM(
                CASE
                    WHEN delta_25d_overlap
                    THEN 1 ELSE 0
                END
            ) AS rr_overlap,

            ROUND(
                MEDIAN(
                    CASE
                        WHEN delta_25d_overlap
                        THEN two_sided_minus_preferred_rr_25d
                    END
                ),
                6
            ) AS median_rr_difference,

            ROUND(
                MEDIAN(
                    CASE
                        WHEN delta_25d_overlap
                        THEN two_sided_minus_preferred_bf_25d
                    END
                ),
                6
            ) AS median_bf_difference

        FROM surface_source_comparison
        """
    ).fetchone()

    print("\nPREFERRED VS TWO-SIDED OVERLAP")
    print("=" * 80)

    print(
        f"ATM overlapping states:       "
        f"{row[0]:,}"
    )

    print(
        f"Median ATM IV difference:     "
        f"{row[1]}"
    )

    print(
        f"25d overlapping states:       "
        f"{row[2]:,}"
    )

    print(
        f"Median RR25 difference:       "
        f"{row[3]}"
    )

    print(
        f"Median BF25 difference:       "
        f"{row[4]}"
    )


def main() -> None:

    connection = open_research_database()

    try:
        build_surfaces(connection)
        create_surface_quality_state(connection)
        create_delta_smile_node_state(connection)
        create_delta_smile_state(connection)
        create_surface_source_comparison(connection)

        print_delta_smile_coverage(connection)
        print_source_overlap_summary(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
