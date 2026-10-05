from __future__ import annotations

import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data_engineering.catalog.database import (
    open_research_database,
)

from data_engineering.features.market_time_series import (
    drop_persistent_relation,
)


OPTION_FEATURE_VERSION = "0.2.0"

# v0.2.0 (audit B2): surface changes are computed only when every node
# involved has the same IV source class (QQ / QM / MQ / MM) at both
# endpoints. A change that crosses a quote <-> mark switch mixes a market
# move with a model switch; Notebook 03 showed such changes have 3-4x the
# tail of same-source changes. Blocked changes are counted by the
# *_source_switch_5m diagnostics instead of being silently dropped.
# Mark-share columns expose how much of each surface is Deribit's model.


def _timed_materialize(
    connection,
    label: str,
    table_name: str,
    query: str
) -> None:

    print(
        f"\n[FEATURE BUILD] {label}..."
    )

    started = time.perf_counter()

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE
        {table_name} AS

        {query}
        """
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    count = connection.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    print(
        f"[FEATURE BUILD] {label}: "
        f"{count:,} rows in "
        f"{elapsed:,.2f}s"
    )


def materialize_option_feature_inputs(
    connection
) -> None:
    """
    Materialize compact surface outputs one at a time.

    This is the core performance fix versus v0.1.

    The old implementation joined several lazy surface views in
    one query. Each of those views expanded independently back
    into normalized_option_chain, causing a very large DuckDB
    query plan over ~1.75M option rows.

    Here the expensive parents are each evaluated once and then
    joined only after they have collapsed to state/family grain.
    """

    _timed_materialize(
        connection,
        "Materialize surface input states",
        "_feature_surface_input",
        """
        SELECT *
        FROM surface_input_state
        """
    )

    _timed_materialize(
        connection,
        "Materialize preferred smile states",
        "_feature_option_smile_preferred",
        """
        SELECT *
        FROM option_smile_state
        WHERE surface_source = 'preferred'
        """
    )

    _timed_materialize(
        connection,
        "Materialize preferred surface quality",
        "_feature_surface_quality_preferred",
        """
        SELECT *
        FROM surface_quality_state
        WHERE surface_source = 'preferred'
        """
    )

    _timed_materialize(
        connection,
        "Materialize preferred delta smiles",
        "_feature_delta_smile_preferred",
        """
        SELECT *
        FROM delta_smile_state
        WHERE surface_source = 'preferred'
        """
    )


def create_option_surface_feature_state(
    connection
) -> None:

    print(
        "\n[FEATURE BUILD] "
        "Build option surface time-series table..."
    )

    started = time.perf_counter()

    drop_persistent_relation(
        connection,
        "option_surface_feature_state"
    )

    connection.execute(
        """
        CREATE TABLE
        option_surface_feature_state AS

        WITH base AS (

            SELECT
                s.state_id,
                s.state_time,
                s.underlying,
                s.session_id,
                s.expiry_date,
                s.session_start,
                s.session_end,
                s.minutes_to_expiry,
                s.option_family_id,

                i.instrument_type,
                i.settlement_currency,
                i.api_currency,

                s.option_forward_price,
                s.iv_atm_forward,
                s.iv_k_m025,
                s.iv_k_p025,
                s.iv_k_m050,
                s.iv_k_p050,

                s.symmetric_025_iv_difference,
                s.symmetric_050_iv_difference,

                s.curvature_025,
                s.curvature_050,

                s.local_skew_slope_025,
                s.local_skew_slope_050,

                q.core_smile_quality_tier,
                q.core_smile_tight,
                q.core_smile_usable,
                q.wing_050_usable,
                q.full_grid_usable,

                q.core_max_interpolation_span,
                q.wing_050_max_interpolation_span,
                q.full_grid_max_interpolation_span,

                s.base_eligible_rows,
                s.two_sided_eligible_rows,
                s.median_two_sided_spread_bps,
                s.p90_two_sided_spread_bps,
                s.option_snapshot_ready,
                s.src_k_m050,
                s.src_k_m025,
                s.src_k_000,
                s.src_k_p025,
                s.src_k_p050,
                s.core_iv_source_signature,
                s.wing_050_iv_source_signature,
                s.core_nodes_with_mark,
                s.node_mark_share,
                s.base_mark_iv_rows,
                s.base_mark_iv_row_share,

                d.iv_put_25d,
                d.iv_call_25d,
                d.rr_25d,
                d.bf_25d,

                d.iv_put_10d,
                d.iv_call_10d,
                d.rr_10d,
                d.bf_10d,

                d.delta_25d_smile_available,
                d.full_delta_smile_available,

                d.max_delta_interpolation_span,
                d.median_delta_interpolation_span,
                d.rr_25d_iv_source_signature,
                d.bf_25d_iv_source_signature

            FROM _feature_option_smile_preferred s

            LEFT JOIN _feature_surface_quality_preferred q
                ON
                    s.state_id = q.state_id
                    AND
                    s.option_family_id = q.option_family_id

            LEFT JOIN _feature_delta_smile_preferred d
                ON
                    s.state_id = d.state_id
                    AND
                    s.option_family_id = d.option_family_id

            LEFT JOIN _feature_surface_input i
                ON
                    s.state_id = i.state_id
                    AND
                    s.option_family_id = i.option_family_id
        ),

        lagged AS (

            SELECT
                *,

                LAG(state_time, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_state_time_1,

                LAG(state_time, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_state_time_3,

                LAG(state_time, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_state_time_12,

                LAG(session_id, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_session_id_1,

                LAG(session_id, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_session_id_3,

                LAG(session_id, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_session_id_12,

                LAG(core_smile_usable, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_core_usable_1,

                LAG(core_smile_usable, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_core_usable_3,

                LAG(core_smile_usable, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_core_usable_12,

                LAG(wing_050_usable, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_wing_050_usable_1,

                LAG(
                    delta_25d_smile_available,
                    1
                ) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_delta_25d_available_1,

                LAG(iv_atm_forward, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_atm_iv_1,

                LAG(iv_atm_forward, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_atm_iv_3,

                LAG(iv_atm_forward, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_atm_iv_12,

                LAG(
                    symmetric_025_iv_difference,
                    1
                ) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_skew_025_1,

                LAG(
                    symmetric_025_iv_difference,
                    3
                ) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_skew_025_3,

                LAG(
                    symmetric_025_iv_difference,
                    12
                ) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_skew_025_12,

                LAG(curvature_025, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_curvature_025_1,

                LAG(curvature_025, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_curvature_025_3,

                LAG(curvature_025, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_curvature_025_12,

                LAG(
                    symmetric_050_iv_difference,
                    1
                ) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_skew_050_1,

                LAG(rr_25d, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_rr_25d_1,

                LAG(bf_25d, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_bf_25d_1,
                LAG(src_k_000, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_src_atm_1,
                LAG(src_k_000, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_src_atm_3,
                LAG(src_k_000, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_src_atm_12,
                LAG(core_iv_source_signature, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_core_sig_1,
                LAG(core_iv_source_signature, 3) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_core_sig_3,
                LAG(core_iv_source_signature, 12) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_core_sig_12,
                LAG(wing_050_iv_source_signature, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_wing_sig_1,
                LAG(rr_25d_iv_source_signature, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_rr_sig_1,
                LAG(bf_25d_iv_source_signature, 1) OVER (
                    PARTITION BY option_family_id
                    ORDER BY state_time
                ) AS lag_bf_sig_1

            FROM base
        )

        SELECT
            * EXCLUDE (
                lag_state_time_1,
                lag_state_time_3,
                lag_state_time_12,

                lag_session_id_1,
                lag_session_id_3,
                lag_session_id_12,

                lag_core_usable_1,
                lag_core_usable_3,
                lag_core_usable_12,

                lag_wing_050_usable_1,
                lag_delta_25d_available_1,

                lag_atm_iv_1,
                lag_atm_iv_3,
                lag_atm_iv_12,

                lag_skew_025_1,
                lag_skew_025_3,
                lag_skew_025_12,

                lag_curvature_025_1,
                lag_curvature_025_3,
                lag_curvature_025_12,

                lag_skew_050_1,
                lag_rr_25d_1,
                lag_bf_25d_1,
                lag_src_atm_1,
                lag_src_atm_3,
                lag_src_atm_12,
                lag_core_sig_1,
                lag_core_sig_3,
                lag_core_sig_12,
                lag_wing_sig_1,
                lag_rr_sig_1,
                lag_bf_sig_1
            ),

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_1
                    AND session_id = lag_session_id_1
                    AND date_diff(
                        'minute',
                        lag_state_time_1,
                        state_time
                    ) = 5
                    AND src_k_000 = lag_src_atm_1
                THEN
                    iv_atm_forward
                    - lag_atm_iv_1
            END
                AS atm_iv_change_5m,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_3
                    AND session_id = lag_session_id_3
                    AND date_diff(
                        'minute',
                        lag_state_time_3,
                        state_time
                    ) = 15
                    AND src_k_000 = lag_src_atm_3
                THEN
                    iv_atm_forward
                    - lag_atm_iv_3
            END
                AS atm_iv_change_15m,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_12
                    AND session_id = lag_session_id_12
                    AND date_diff(
                        'minute',
                        lag_state_time_12,
                        state_time
                    ) = 60
                    AND src_k_000 = lag_src_atm_12
                THEN
                    iv_atm_forward
                    - lag_atm_iv_12
            END
                AS atm_iv_change_1h,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_1
                    AND session_id = lag_session_id_1
                    AND date_diff(
                        'minute',
                        lag_state_time_1,
                        state_time
                    ) = 5
                    AND core_iv_source_signature = lag_core_sig_1
                THEN
                    symmetric_025_iv_difference
                    - lag_skew_025_1
            END
                AS skew_025_change_5m,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_3
                    AND session_id = lag_session_id_3
                    AND date_diff(
                        'minute',
                        lag_state_time_3,
                        state_time
                    ) = 15
                    AND core_iv_source_signature = lag_core_sig_3
                THEN
                    symmetric_025_iv_difference
                    - lag_skew_025_3
            END
                AS skew_025_change_15m,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_12
                    AND session_id = lag_session_id_12
                    AND date_diff(
                        'minute',
                        lag_state_time_12,
                        state_time
                    ) = 60
                    AND core_iv_source_signature = lag_core_sig_12
                THEN
                    symmetric_025_iv_difference
                    - lag_skew_025_12
            END
                AS skew_025_change_1h,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_1
                    AND session_id = lag_session_id_1
                    AND date_diff(
                        'minute',
                        lag_state_time_1,
                        state_time
                    ) = 5
                    AND core_iv_source_signature = lag_core_sig_1
                THEN
                    curvature_025
                    - lag_curvature_025_1
            END
                AS curvature_025_change_5m,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_3
                    AND session_id = lag_session_id_3
                    AND date_diff(
                        'minute',
                        lag_state_time_3,
                        state_time
                    ) = 15
                    AND core_iv_source_signature = lag_core_sig_3
                THEN
                    curvature_025
                    - lag_curvature_025_3
            END
                AS curvature_025_change_15m,

            CASE
                WHEN
                    core_smile_usable
                    AND lag_core_usable_12
                    AND session_id = lag_session_id_12
                    AND date_diff(
                        'minute',
                        lag_state_time_12,
                        state_time
                    ) = 60
                    AND core_iv_source_signature = lag_core_sig_12
                THEN
                    curvature_025
                    - lag_curvature_025_12
            END
                AS curvature_025_change_1h,

            CASE
                WHEN
                    wing_050_usable
                    AND lag_wing_050_usable_1
                    AND session_id = lag_session_id_1
                    AND date_diff(
                        'minute',
                        lag_state_time_1,
                        state_time
                    ) = 5
                    AND wing_050_iv_source_signature = lag_wing_sig_1
                THEN
                    symmetric_050_iv_difference
                    - lag_skew_050_1
            END
                AS skew_050_change_5m,

            CASE
                WHEN
                    delta_25d_smile_available
                    AND lag_delta_25d_available_1
                    AND session_id = lag_session_id_1
                    AND date_diff(
                        'minute',
                        lag_state_time_1,
                        state_time
                    ) = 5
                    AND rr_25d_iv_source_signature = lag_rr_sig_1
                THEN
                    rr_25d
                    - lag_rr_25d_1
            END
                AS rr_25d_change_5m,

            CASE
                WHEN
                    delta_25d_smile_available
                    AND lag_delta_25d_available_1
                    AND session_id = lag_session_id_1
                    AND date_diff(
                        'minute',
                        lag_state_time_1,
                        state_time
                    ) = 5
                    AND bf_25d_iv_source_signature = lag_bf_sig_1
                THEN
                    bf_25d
                    - lag_bf_25d_1
            END
                AS bf_25d_change_5m,

            (
                core_smile_usable
                AND lag_core_usable_1
                AND session_id = lag_session_id_1
                AND date_diff(
                    'minute',
                    lag_state_time_1,
                    state_time
                ) = 5
                AND core_iv_source_signature
                    = lag_core_sig_1
            )
                AS core_surface_change_5m_ready,

            (
                core_smile_usable
                AND lag_core_usable_1
                AND session_id = lag_session_id_1
                AND date_diff(
                    'minute',
                    lag_state_time_1,
                    state_time
                ) = 5
                AND core_iv_source_signature
                    IS DISTINCT FROM lag_core_sig_1
            )
                AS core_source_switch_5m,

            (
                core_smile_usable
                AND lag_core_usable_1
                AND session_id = lag_session_id_1
                AND date_diff(
                    'minute',
                    lag_state_time_1,
                    state_time
                ) = 5
                AND src_k_000
                    IS DISTINCT FROM lag_src_atm_1
            )
                AS atm_source_switch_5m,

            (
                delta_25d_smile_available
                AND lag_delta_25d_available_1
                AND session_id = lag_session_id_1
                AND date_diff(
                    'minute',
                    lag_state_time_1,
                    state_time
                ) = 5
                AND bf_25d_iv_source_signature
                    = lag_bf_sig_1
            )
                AS delta_25d_change_5m_ready

        FROM lagged
        """
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM option_surface_feature_state
        """
    ).fetchone()[0]

    print(
        "[FEATURE BUILD] "
        "Build option surface time-series table: "
        f"{count:,} rows in {elapsed:,.2f}s"
    )


def create_primary_option_surface_feature_state(
    connection
) -> None:

    print(
        "\n[FEATURE BUILD] "
        "Build stable primary option-family table..."
    )

    started = time.perf_counter()

    drop_persistent_relation(
        connection,
        "primary_option_surface_feature_state"
    )

    connection.execute(
        """
        CREATE TABLE
        primary_option_surface_feature_state AS

        WITH families AS (

            SELECT DISTINCT
                underlying,
                option_family_id,
                instrument_type,
                settlement_currency,
                api_currency,

                CASE
                    WHEN
                        LOWER(
                            CAST(
                                instrument_type
                                AS VARCHAR
                            )
                        ) = 'linear'
                        AND
                        UPPER(
                            CAST(
                                settlement_currency
                                AS VARCHAR
                            )
                        ) = 'USDC'
                    THEN 0

                    WHEN
                        LOWER(
                            CAST(
                                instrument_type
                                AS VARCHAR
                            )
                        ) = 'reversed'
                    THEN 1

                    ELSE 2
                END
                    AS option_family_selection_rank

            FROM option_surface_feature_state
        ),

        ranked_families AS (

            SELECT
                *,

                ROW_NUMBER() OVER (
                    PARTITION BY underlying
                    ORDER BY
                        option_family_selection_rank,
                        option_family_id
                )
                    AS family_rank

            FROM families
        ),

        selected_family AS (

            SELECT
                * EXCLUDE (
                    family_rank
                )

            FROM ranked_families

            WHERE family_rank = 1
        )

        SELECT
            f.*,

            p.option_family_selection_rank,

            CASE
                WHEN
                    p.option_family_selection_rank = 0
                THEN 'linear_usdc'

                WHEN
                    p.option_family_selection_rank = 1
                THEN 'reversed'

                ELSE 'fallback'
            END
                AS option_family_selection_reason

        FROM option_surface_feature_state f

        INNER JOIN selected_family p
            ON
                f.underlying = p.underlying
                AND
                f.option_family_id = p.option_family_id
        """
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM primary_option_surface_feature_state
        """
    ).fetchone()[0]

    print(
        "[FEATURE BUILD] "
        "Build stable primary option-family table: "
        f"{count:,} rows in {elapsed:,.2f}s"
    )


def print_primary_family_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            underlying,
            option_family_id,
            option_family_selection_reason,

            COUNT(*)
                AS states,

            SUM(
                CASE
                    WHEN core_smile_usable
                    THEN 1 ELSE 0
                END
            )
                AS core_states,

            SUM(
                CASE
                    WHEN core_surface_change_5m_ready
                    THEN 1 ELSE 0
                END
            )
                AS dynamic_states,

            SUM(
                CASE
                    WHEN delta_25d_smile_available
                    THEN 1 ELSE 0
                END
            )
                AS delta_states

        FROM primary_option_surface_feature_state

        GROUP BY
            underlying,
            option_family_id,
            option_family_selection_reason

        ORDER BY
            underlying,
            option_family_id
        """
    ).fetchall()

    print(
        "\nPRIMARY OPTION FEATURE FAMILY"
    )

    print(
        "=" * 135
    )

    print(
        f"{'Asset':<10}"
        f"{'Family':<40}"
        f"{'Reason':<16}"
        f"{'States':>10}"
        f"{'Core':>10}"
        f"{'Dynamic':>12}"
        f"{'25d':>10}"
    )

    for row in rows:

        print(
            f"{row[0]:<10}"
            f"{row[1]:<40}"
            f"{row[2]:<16}"
            f"{row[3]:>10,}"
            f"{row[4]:>10,}"
            f"{row[5]:>12,}"
            f"{row[6]:>10,}"
        )


def print_iv_source_summary(
    connection
) -> None:
    """
    How much of each primary surface is Deribit's mark model, and how
    many 5-minute changes were withheld because the IV source switched.
    """

    rows = connection.execute(
        """
        SELECT
            underlying,

            COUNT(*)
                AS states,

            ROUND(
                100.0 * AVG(node_mark_share),
                1
            )
                AS node_mark_pct,

            ROUND(
                100.0 * AVG(
                    CASE
                        WHEN core_nodes_with_mark > 0
                        THEN 1.0 ELSE 0.0
                    END
                ),
                1
            )
                AS core_any_mark_pct,

            ROUND(
                100.0 * AVG(base_mark_iv_row_share),
                1
            )
                AS row_mark_pct,

            SUM(
                CASE
                    WHEN core_surface_change_5m_ready
                    THEN 1 ELSE 0
                END
            )
                AS core_changes,

            SUM(
                CASE
                    WHEN core_source_switch_5m
                    THEN 1 ELSE 0
                END
            )
                AS core_switch_blocked

        FROM primary_option_surface_feature_state

        WHERE core_smile_usable

        GROUP BY underlying

        ORDER BY underlying
        """
    ).fetchall()

    print(
        "\nPRIMARY SURFACE IV SOURCE"
    )

    print(
        "=" * 95
    )

    print(
        f"{'Asset':<8}"
        f"{'States':>9}"
        f"{'Node mark %':>13}"
        f"{'Core any mark %':>17}"
        f"{'Row mark %':>12}"
        f"{'Core chg 5m':>13}"
        f"{'Blocked (switch)':>18}"
    )

    for row in rows:

        print(
            f"{row[0]:<8}"
            f"{row[1]:>9,}"
            f"{str(row[2]):>13}"
            f"{str(row[3]):>17}"
            f"{str(row[4]):>12}"
            f"{int(row[5] or 0):>13,}"
            f"{int(row[6] or 0):>18,}"
        )


def main() -> None:

    connection = open_research_database()

    try:

        materialize_option_feature_inputs(
            connection
        )

        create_option_surface_feature_state(
            connection
        )

        create_primary_option_surface_feature_state(
            connection
        )

        print_primary_family_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
