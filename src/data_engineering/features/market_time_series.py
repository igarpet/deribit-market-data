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

from data_engineering.alignment.state_index import (
    relation_columns,
)

from data_engineering.quality_checks.futures_integrity import (
    create_futures_integrity_state,
    create_known_exchange_events_table,
    validate_futures_integrity,
)


MARKET_FEATURE_VERSION = "0.4.0"

# v0.4.0:
#   - perpetual integrity gate (quality_checks/futures_integrity.py):
#     perp_integrity_ok and per-test flags; microstructure and L2
#     instability readiness now also require perp_integrity_ok.
#     Index-based returns and RV are unaffected.
#   - known_exchange_event flag (table known_exchange_events).
#   - clock conditioning variables for the deterministic baseline
#     (framework v2.1, Step 2): UTC hour/minute of day, day of week,
#     weekend, expiry type, US and EU daylight-saving indicators.

# v0.2.0: structural returns and realized volatility are computed from the
# Deribit index (the option settlement reference), not the perpetual mid.
# The perpetual mid is exposed to frozen and one-sided books; Notebook 02
# (Section 30) showed TRX 5m-return kurtosis of 1,508 on the mid against
# 9.8 on the index, with identical median realized volatility.
# The perpetual mid return is kept as `perp_mid_return_5m` for basis and
# diagnostic work only.
RETURN_SOURCE = "deribit_index"

# The index arrives on the perpetual ticker channel, so the ticker age at
# the state boundary bounds how old the captured index value can be.
#
# v0.2.1 rule, from the 2026-10-05 gate audit: of 76 states gated at a
# 30 s age (70 of them TRX), the index had changed since the previous
# state in 75. A late ticker on a quiet perpetual therefore means a
# slightly old index value, not a frozen one.
#   age <= INDEX_FRESH_TICKER_AGE_MS                       -> eligible
#   age <= INDEX_MAX_TICKER_AGE_MS and index changed since
#   the previous five-minute state                         -> eligible
#   otherwise                                              -> not eligible
# The hard cap keeps the timing error of an eligible index value below
# 30% of one five-minute bar.
INDEX_FRESH_TICKER_AGE_MS = 30_000

# v0.3.0 (audit B4): Deribit daily options settle to the 30-minute TWAP
# of the index from 07:30 to 08:00 UTC. States with
# minutes_to_expiry <= SETTLEMENT_WINDOW_MINUTES (07:30-07:55) are
# flagged; settlement_locked_fraction is the share of the averaging
# window already elapsed at the state (0 at 07:30, 25/30 at 07:55).
# Nothing is dropped: models decide how to treat the window.
SETTLEMENT_WINDOW_MINUTES = 30
INDEX_MAX_TICKER_AGE_MS = 90_000

FIVE_MINUTE_PERIODS_PER_YEAR = (
    365.0
    * 24.0
    * 12.0
)


L2_SCALAR_FIELDS = {
    "book_twa_abs_obi_20":
        "DOUBLE",

    "book_obi_20_std":
        "DOUBLE",

    "book_obi_20_path_roughness_per_min":
        "DOUBLE",

    "book_obi_20_range":
        "DOUBLE",

    "book_obi_20_coherence_noise":
        "DOUBLE",

    "book_obi_20_transition_count":
        "BIGINT",

    "book_obi_20_raw_zero_cross_count":
        "BIGINT",

    "book_obi_20_raw_zero_cross_rate":
        "DOUBLE",

    "book_obi_20_reversal_strength_mean":
        "DOUBLE",

    "book_obi_20_reversal_strength_max":
        "DOUBLE",
}


def optional_column_expression(
    available_columns: set[str],
    column_name: str,
    sql_type: str,
    relation_alias: str = "m"
) -> str:

    if column_name in available_columns:
        return (
            f"{relation_alias}."
            f"{column_name}"
        )

    return (
        f"CAST(NULL AS {sql_type})"
    )


def _timed_execute(
    connection,
    label: str,
    sql: str
) -> None:

    print(
        f"\n[FEATURE BUILD] {label}..."
    )

    started = time.perf_counter()

    connection.execute(
        sql
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"[FEATURE BUILD] {label}: "
        f"{elapsed:,.2f}s"
    )



def drop_persistent_relation(
    connection,
    relation_name: str
) -> None:
    """
    Remove an existing persistent view/table before changing
    its object type.

    DuckDB does not allow:
        CREATE OR REPLACE TABLE name ...
    when `name` currently exists as a VIEW, or vice versa.
    """

    view_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM duckdb_views()
        WHERE
            schema_name = current_schema()
            AND
            view_name = ?
        """,
        [relation_name]
    ).fetchone()[0]

    if view_exists:

        connection.execute(
            f"""
            DROP VIEW {relation_name}
            """
        )

    table_exists = connection.execute(
        """
        SELECT COUNT(*)
        FROM duckdb_tables()
        WHERE
            schema_name = current_schema()
            AND
            table_name = ?
        """,
        [relation_name]
    ).fetchone()[0]

    if table_exists:

        connection.execute(
            f"""
            DROP TABLE {relation_name}
            """
        )


def materialize_market_feature_inputs(
    connection
) -> None:
    """
    Freeze the compact state-level parents once for this build.

    This avoids repeatedly expanding canonical/raw Parquet views
    during validation and feature diagnostics.
    """

    _timed_execute(
        connection,
        "Materialize canonical market states",
        """
        CREATE OR REPLACE TEMP TABLE
        _feature_market_state_input AS

        SELECT *
        FROM canonical_market_state
        """
    )

    create_known_exchange_events_table(
        connection
    )

    _timed_execute(
        connection,
        "Materialize futures reference states",
        """
        CREATE OR REPLACE TEMP TABLE
        _feature_futures_reference_input AS

        SELECT *
        FROM futures_reference_state
        """
    )


def create_futures_integrity_inputs(
    connection
) -> None:
    """
    Build the perpetual integrity gate from the frozen market inputs,
    persist it as futures_integrity_state (for notebooks) and keep a
    compact temp copy for the market feature join.
    """

    started = time.perf_counter()

    create_futures_integrity_state(
        connection,
        source_relation="_feature_market_state_input",
        target_table="futures_integrity_state",
        temporary=False,
    )

    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE
        _feature_futures_integrity AS

        SELECT
            state_id,
            perp_integrity_ok,
            integrity_flag_count,
            flag_known_event,
            flag_no_two_sided,
            flag_wide_spread,
            flag_mid_off_index,
            flag_frozen_with_trades,
            flag_no_l2_updates,
            flag_rest_stale,
            flag_trades_outside_bbo
        FROM futures_integrity_state
        """
    )

    print(
        "[FEATURE BUILD] Build futures integrity gate: "
        f"{time.perf_counter() - started:,.2f}s"
    )

    validate_futures_integrity(
        connection
    )


def create_l2_instability_distribution_state(
    connection
) -> None:

    market_columns = relation_columns(
        connection,
        "_feature_market_state_input"
    )

    schema_expr = optional_column_expression(
        market_columns,
        "book_obi_20_histogram_schema",
        "VARCHAR"
    )

    bin_count_expr = optional_column_expression(
        market_columns,
        "book_obi_20_histogram_bin_count",
        "BIGINT"
    )

    abs_hist_expr = optional_column_expression(
        market_columns,
        "book_obi_20_abs_histogram_ms",
        "DOUBLE[]"
    )

    reversal_hist_expr = optional_column_expression(
        market_columns,
        "book_obi_20_reversal_strength_histogram",
        "BIGINT[]"
    )

    transition_expr = optional_column_expression(
        market_columns,
        "book_obi_20_transition_count",
        "BIGINT"
    )

    zero_cross_expr = optional_column_expression(
        market_columns,
        "book_obi_20_raw_zero_cross_count",
        "BIGINT"
    )

    drop_persistent_relation(
        connection,
        "l2_instability_distribution_state"
    )

    _timed_execute(
        connection,
        "Build L2 distribution state",
        f"""
        CREATE TABLE
        l2_instability_distribution_state AS

        SELECT
            m.state_id,
            m.state_time,
            m.underlying,
            m.session_id,
            m.expiry_date,
            m.minutes_to_expiry,
            m.l2_intrawindow_ready,

            {schema_expr}
                AS histogram_schema,

            {bin_count_expr}
                AS histogram_bin_count,

            {abs_hist_expr}
                AS abs_obi_20_histogram_ms,

            {reversal_hist_expr}
                AS reversal_strength_histogram,

            {transition_expr}
                AS transition_count,

            {zero_cross_expr}
                AS raw_zero_cross_count,

            (
                {schema_expr} IS NOT NULL
                AND
                {bin_count_expr} IS NOT NULL
                AND
                {abs_hist_expr} IS NOT NULL
                AND
                {reversal_hist_expr} IS NOT NULL
            )
                AS l2_distribution_available

        FROM _feature_market_state_input m
        """
    )


def create_market_time_series_state(
    connection
) -> None:

    market_columns = relation_columns(
        connection,
        "_feature_market_state_input"
    )

    optional_l2_selects = []

    for (
        column_name,
        sql_type
    ) in L2_SCALAR_FIELDS.items():

        expression = optional_column_expression(
            market_columns,
            column_name,
            sql_type
        )

        optional_l2_selects.append(
            f"""
            {expression}
                AS {column_name}
            """
        )

    optional_l2_sql = (
        ",\n".join(
            optional_l2_selects
        )
    )

    drop_persistent_relation(
        connection,
        "market_time_series_state"
    )

    _timed_execute(
        connection,
        "Build market time-series feature table",
        f"""
        CREATE TABLE
        market_time_series_state AS

        WITH base AS (

            SELECT
                m.state_id,
                m.state_time,
                m.underlying,
                m.session_id,
                m.expiry_date,
                m.session_start,
                m.session_end,
                m.minutes_to_expiry,
                m.is_opening_state,

                -- Clock conditioning (framework v2.1, Step 2). Every field is
                -- computed in UTC explicitly, independent of the session time zone.
                EXTRACT('hour' FROM (m.state_time AT TIME ZONE 'UTC'))
                    AS utc_hour,

                EXTRACT('hour' FROM (m.state_time AT TIME ZONE 'UTC')) * 60
                + EXTRACT('minute' FROM (m.state_time AT TIME ZONE 'UTC'))
                    AS utc_minute_of_day,

                -- ISO day of week of the state: 1 = Monday ... 7 = Sunday.
                ISODOW(m.state_time AT TIME ZONE 'UTC')
                    AS utc_day_of_week,

                (ISODOW(m.state_time AT TIME ZONE 'UTC') >= 6)
                    AS is_weekend,

                -- Deribit expiries are 08:00 UTC. Friday expiries are also
                -- the weekly; the last Friday of a month is the monthly;
                -- a monthly in Mar/Jun/Sep/Dec is the quarterly.
                CASE
                    WHEN
                        ISODOW(CAST(m.expiry_date AS DATE)) = 5
                        AND MONTH(CAST(m.expiry_date AS DATE) + 7)
                            <> MONTH(CAST(m.expiry_date AS DATE))
                        AND MONTH(CAST(m.expiry_date AS DATE))
                            IN (3, 6, 9, 12)
                    THEN 'quarterly'
                    WHEN
                        ISODOW(CAST(m.expiry_date AS DATE)) = 5
                        AND MONTH(CAST(m.expiry_date AS DATE) + 7)
                            <> MONTH(CAST(m.expiry_date AS DATE))
                    THEN 'monthly'
                    WHEN
                        ISODOW(CAST(m.expiry_date AS DATE)) = 5
                    THEN 'weekly'
                    ELSE 'daily'
                END
                    AS expiry_type,

                (
                    date_diff(
                        'minute',
                        (m.state_time AT TIME ZONE 'UTC'),
                        (m.state_time AT TIME ZONE 'America/New_York')
                    ) = -240
                )
                    AS us_dst,

                (
                    date_diff(
                        'minute',
                        (m.state_time AT TIME ZONE 'UTC'),
                        (m.state_time AT TIME ZONE 'Europe/London')
                    ) = 60
                )
                    AS eu_dst,

                COALESCE(gi.perp_integrity_ok, FALSE)
                    AS perp_integrity_ok,

                gi.integrity_flag_count,

                COALESCE(gi.flag_known_event, FALSE)
                    AS known_exchange_event,

                gi.flag_no_two_sided AS integrity_flag_no_two_sided,
                gi.flag_wide_spread AS integrity_flag_wide_spread,
                gi.flag_mid_off_index AS integrity_flag_mid_off_index,
                gi.flag_frozen_with_trades AS integrity_flag_frozen_with_trades,
                gi.flag_no_l2_updates AS integrity_flag_no_l2_updates,
                gi.flag_rest_stale AS integrity_flag_rest_stale,
                gi.flag_trades_outside_bbo AS integrity_flag_trades_outside_bbo,

                (
                    m.minutes_to_expiry
                        <= {SETTLEMENT_WINDOW_MINUTES}
                )
                    AS in_settlement_window,

                GREATEST(
                    0,
                    {SETTLEMENT_WINDOW_MINUTES}
                    - m.minutes_to_expiry
                )
                / {float(SETTLEMENT_WINDOW_MINUTES)}
                    AS settlement_locked_fraction,

                (
                    1440
                    - m.minutes_to_expiry
                )
                    AS minutes_since_session_start,

                (
                    1440
                    - m.minutes_to_expiry
                )
                / 1440.0
                    AS session_progress_fraction,

                m.collection_complete,
                m.price_features_ready,
                m.state_research_ready,
                m.l2_boundary_ready,
                m.l2_intrawindow_ready,

                r.futures_instrument_name,
                r.futures_tradable_mid_price,
                r.futures_mark_price,
                r.futures_index_price,
                r.futures_underlying_price,
                r.market_reference_price,
                r.market_reference_source,
                r.market_reference_is_tradable,
                r.futures_ticker_age_ms,
                r.futures_spread_bps,
                r.perpetual_mid_vs_index_bps,
                r.perpetual_mark_vs_index_bps,
                r.microprice_vs_mid_bps,
                r.return_input_eligible,

                COALESCE(
                    m.price_features_ready
                    AND
                    r.futures_index_price > 0
                    AND
                    r.futures_ticker_age_ms IS NOT NULL,
                    FALSE
                )
                    AS index_base_eligible,

                m.open,
                m.high,
                m.low,
                m.close,
                m.vwap,

                m.volume_usd,
                m.buy_volume_usd,
                m.sell_volume_usd,
                m.trade_count,
                m.open_interest,

                m.book_stats_coverage_ms,
                m.book_stats_complete,
                m.book_update_count,

                m.book_twa_spread,
                m.book_twa_obi_5,
                m.book_twa_obi_10,
                m.book_twa_obi_20,
                m.book_twa_bid_depth_20,
                m.book_twa_ask_depth_20,
                m.book_twa_microprice,

                m.book_min_obi_20,
                m.book_max_obi_20,

                {optional_l2_sql}

            FROM _feature_market_state_input m

            LEFT JOIN _feature_futures_reference_input r
                ON m.state_id = r.state_id

            LEFT JOIN _feature_futures_integrity gi
                ON m.state_id = gi.state_id
        ),

        index_checked AS (

            SELECT
                *,

                (
                    index_base_eligible
                    AND (
                        futures_ticker_age_ms
                            <= {INDEX_FRESH_TICKER_AGE_MS}
                        OR (
                            futures_ticker_age_ms
                                <= {INDEX_MAX_TICKER_AGE_MS}
                            AND
                            futures_index_price
                                IS DISTINCT FROM LAG(
                                    futures_index_price,
                                    1
                                ) OVER (
                                    PARTITION BY underlying
                                    ORDER BY state_time
                                )
                        )
                    )
                )
                    AS index_return_input_eligible

            FROM base
        ),

        lagged AS (

            SELECT
                *,

                LAG(state_time, 1) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_state_time_1,

                LAG(state_time, 3) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_state_time_3,

                LAG(state_time, 6) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_state_time_6,

                LAG(state_time, 12) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_state_time_12,

                LAG(
                    futures_index_price,
                    1
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_1,

                LAG(
                    index_return_input_eligible,
                    1
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_ready_1,

                LAG(
                    futures_index_price,
                    3
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_3,

                LAG(
                    index_return_input_eligible,
                    3
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_ready_3,

                LAG(
                    futures_index_price,
                    6
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_6,

                LAG(
                    index_return_input_eligible,
                    6
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_ready_6,

                LAG(
                    futures_index_price,
                    12
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_12,

                LAG(
                    index_return_input_eligible,
                    12
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_index_ready_12,

                LAG(
                    futures_tradable_mid_price,
                    1
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_mid_1,

                LAG(
                    return_input_eligible,
                    1
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_return_ready_1,

                LAG(
                    open_interest,
                    1
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_open_interest_1,

                LAG(
                    book_twa_obi_20,
                    1
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                ) AS lag_book_twa_obi_20_1

            FROM index_checked
        ),

        point_features AS (

            SELECT
                *,

                CASE
                    WHEN
                        index_return_input_eligible
                        AND lag_index_ready_1
                        AND futures_index_price > 0
                        AND lag_index_1 > 0
                        AND date_diff(
                            'minute',
                            lag_state_time_1,
                            state_time
                        ) = 5
                    THEN
                        LN(
                            futures_index_price
                            / lag_index_1
                        )
                END
                    AS return_5m,

                CASE
                    WHEN
                        index_return_input_eligible
                        AND lag_index_ready_3
                        AND futures_index_price > 0
                        AND lag_index_3 > 0
                        AND date_diff(
                            'minute',
                            lag_state_time_3,
                            state_time
                        ) = 15
                    THEN
                        LN(
                            futures_index_price
                            / lag_index_3
                        )
                END
                    AS return_15m,

                CASE
                    WHEN
                        index_return_input_eligible
                        AND lag_index_ready_6
                        AND futures_index_price > 0
                        AND lag_index_6 > 0
                        AND date_diff(
                            'minute',
                            lag_state_time_6,
                            state_time
                        ) = 30
                    THEN
                        LN(
                            futures_index_price
                            / lag_index_6
                        )
                END
                    AS return_30m,

                CASE
                    WHEN
                        index_return_input_eligible
                        AND lag_index_ready_12
                        AND futures_index_price > 0
                        AND lag_index_12 > 0
                        AND date_diff(
                            'minute',
                            lag_state_time_12,
                            state_time
                        ) = 60
                    THEN
                        LN(
                            futures_index_price
                            / lag_index_12
                        )
                END
                    AS return_1h,

                -- Diagnostic only: perpetual-mid return (pre-v0.2.0
                -- definition of return_5m). Not a structural input.
                CASE
                    WHEN
                        return_input_eligible
                        AND lag_return_ready_1
                        AND futures_tradable_mid_price > 0
                        AND lag_mid_1 > 0
                        AND date_diff(
                            'minute',
                            lag_state_time_1,
                            state_time
                        ) = 5
                    THEN
                        LN(
                            futures_tradable_mid_price
                            / lag_mid_1
                        )
                END
                    AS perp_mid_return_5m,

                '{RETURN_SOURCE}'
                    AS return_source,

                CASE
                    WHEN volume_usd >= 0
                    THEN LN(1.0 + volume_usd)
                END
                    AS log1p_volume_usd,

                (
                    buy_volume_usd
                    - sell_volume_usd
                )
                    AS signed_volume_usd,

                CASE
                    WHEN
                        (
                            buy_volume_usd
                            + sell_volume_usd
                        ) > 0
                    THEN
                        (
                            buy_volume_usd
                            - sell_volume_usd
                        )
                        /
                        (
                            buy_volume_usd
                            + sell_volume_usd
                        )
                END
                    AS trade_flow_imbalance,

                CASE
                    WHEN
                        (
                            buy_volume_usd
                            + sell_volume_usd
                        ) > 0
                    THEN
                        buy_volume_usd
                        /
                        (
                            buy_volume_usd
                            + sell_volume_usd
                        )
                END
                    AS buy_volume_share,

                CASE
                    WHEN trade_count > 0
                    THEN
                        volume_usd
                        / trade_count
                END
                    AS average_trade_size_usd,

                CASE
                    WHEN
                        high IS NOT NULL
                        AND low IS NOT NULL
                        AND futures_tradable_mid_price > 0
                    THEN
                        (
                            high
                            - low
                        )
                        /
                        futures_tradable_mid_price
                        * 10000.0
                END
                    AS intrabar_range_bps,

                CASE
                    WHEN
                        close > 0
                        AND vwap > 0
                    THEN
                        (
                            close
                            / vwap
                            - 1.0
                        )
                        * 10000.0
                END
                    AS close_vs_vwap_bps,

                CASE
                    WHEN
                        open_interest IS NOT NULL
                        AND lag_open_interest_1 IS NOT NULL
                        AND date_diff(
                            'minute',
                            lag_state_time_1,
                            state_time
                        ) = 5
                    THEN
                        open_interest
                        - lag_open_interest_1
                END
                    AS open_interest_change_5m,

                CASE
                    WHEN
                        open_interest IS NOT NULL
                        AND lag_open_interest_1 > 0
                        AND date_diff(
                            'minute',
                            lag_state_time_1,
                            state_time
                        ) = 5
                    THEN
                        (
                            open_interest
                            / lag_open_interest_1
                            - 1.0
                        )
                END
                    AS open_interest_return_5m,

                CASE
                    WHEN
                        book_twa_spread IS NOT NULL
                        AND market_reference_price > 0
                    THEN
                        book_twa_spread
                        /
                        market_reference_price
                        * 10000.0
                END
                    AS book_twa_spread_bps,

                CASE
                    WHEN
                        book_twa_microprice > 0
                        AND market_reference_price > 0
                    THEN
                        (
                            book_twa_microprice
                            / market_reference_price
                            - 1.0
                        )
                        * 10000.0
                END
                    AS book_twa_microprice_vs_reference_bps,

                CASE
                    WHEN
                        book_twa_obi_20 IS NOT NULL
                        AND lag_book_twa_obi_20_1
                            IS NOT NULL
                        AND date_diff(
                            'minute',
                            lag_state_time_1,
                            state_time
                        ) = 5
                    THEN
                        book_twa_obi_20
                        - lag_book_twa_obi_20_1
                END
                    AS book_twa_obi_20_change_5m,

                (
                    book_twa_abs_obi_20 IS NOT NULL
                    AND
                    book_obi_20_std IS NOT NULL
                    AND
                    book_obi_20_path_roughness_per_min
                        IS NOT NULL
                    AND
                    book_obi_20_range IS NOT NULL
                )
                    AS l2_instability_scalar_available,

                (
                    book_obi_20_coherence_noise
                        IS NOT NULL
                )
                    AS l2_coherence_available

            FROM lagged
        ),

        rolling AS (

            SELECT
                *,

                COUNT(return_5m) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN
                        5 PRECEDING
                        AND CURRENT ROW
                )
                    AS rv_30m_return_count,

                SUM(
                    return_5m
                    * return_5m
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN
                        5 PRECEDING
                        AND CURRENT ROW
                )
                    AS rv_30m_sum_squared_returns,

                COUNT(return_5m) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN
                        11 PRECEDING
                        AND CURRENT ROW
                )
                    AS rv_1h_return_count,

                SUM(
                    return_5m
                    * return_5m
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN
                        11 PRECEDING
                        AND CURRENT ROW
                )
                    AS rv_1h_sum_squared_returns,

                COUNT(return_5m) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN
                        47 PRECEDING
                        AND CURRENT ROW
                )
                    AS rv_4h_return_count,

                SUM(
                    return_5m
                    * return_5m
                ) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN
                        47 PRECEDING
                        AND CURRENT ROW
                )
                    AS rv_4h_sum_squared_returns

            FROM point_features
        )

        SELECT
            * EXCLUDE (
                lag_state_time_1,
                lag_state_time_3,
                lag_state_time_6,
                lag_state_time_12,

                lag_index_1,
                lag_index_3,
                lag_index_6,
                lag_index_12,

                lag_index_ready_1,
                lag_index_ready_3,
                lag_index_ready_6,
                lag_index_ready_12,

                lag_mid_1,
                lag_return_ready_1,

                lag_open_interest_1,
                lag_book_twa_obi_20_1,

                rv_30m_sum_squared_returns,
                rv_1h_sum_squared_returns,
                rv_4h_sum_squared_returns
            ),

            CASE
                WHEN rv_30m_return_count = 6
                THEN SQRT(
                    rv_30m_sum_squared_returns
                )
            END
                AS realized_vol_30m,

            CASE
                WHEN rv_30m_return_count = 6
                THEN SQRT(
                    (
                        rv_30m_sum_squared_returns
                        / 6.0
                    )
                    * {FIVE_MINUTE_PERIODS_PER_YEAR}
                )
            END
                AS realized_vol_30m_annualized,

            CASE
                WHEN rv_1h_return_count = 12
                THEN SQRT(
                    rv_1h_sum_squared_returns
                )
            END
                AS realized_vol_1h,

            CASE
                WHEN rv_1h_return_count = 12
                THEN SQRT(
                    (
                        rv_1h_sum_squared_returns
                        / 12.0
                    )
                    * {FIVE_MINUTE_PERIODS_PER_YEAR}
                )
            END
                AS realized_vol_1h_annualized,

            CASE
                WHEN rv_4h_return_count = 48
                THEN SQRT(
                    rv_4h_sum_squared_returns
                )
            END
                AS realized_vol_4h,

            CASE
                WHEN rv_4h_return_count = 48
                THEN SQRT(
                    (
                        rv_4h_sum_squared_returns
                        / 48.0
                    )
                    * {FIVE_MINUTE_PERIODS_PER_YEAR}
                )
            END
                AS realized_vol_4h_annualized,

            (
                rv_30m_return_count = 6
            )
                AS realized_vol_30m_ready,

            (
                rv_1h_return_count = 12
            )
                AS realized_vol_1h_ready,

            (
                rv_4h_return_count = 48
            )
                AS realized_vol_4h_ready,

            (
                l2_intrawindow_ready
                AND
                perp_integrity_ok
                AND
                book_twa_obi_20 IS NOT NULL
                AND
                book_twa_spread IS NOT NULL
                AND
                book_twa_microprice IS NOT NULL
            )
                AS microstructure_core_ready,

            (
                l2_intrawindow_ready
                AND
                perp_integrity_ok
                AND
                l2_instability_scalar_available
            )
                AS l2_instability_features_ready

        FROM rolling
        """
    )


def validate_market_returns(
    connection
) -> None:
    """
    Return-source checks for market_time_series_state.

    Hard invariants (raise):
        - every structural return comes from an eligible index state;
        - return_source is the declared source on every row.

    Diagnostic (print only): how often the perpetual-mid return departs
    from the index return by more than 100 bps in one bar.
    """

    violations = connection.execute(
        """
        SELECT
            SUM(
                CASE
                    WHEN
                        return_5m IS NOT NULL
                        AND NOT index_return_input_eligible
                    THEN 1 ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN
                        return_source IS DISTINCT FROM ?
                    THEN 1 ELSE 0
                END
            )

        FROM market_time_series_state
        """,
        [RETURN_SOURCE]
    ).fetchone()

    rows = connection.execute(
        """
        SELECT
            underlying,

            COUNT(return_5m)
                AS index_returns,

            COUNT(perp_mid_return_5m)
                AS perp_mid_returns,

            SUM(
                CASE
                    WHEN
                        ABS(
                            perp_mid_return_5m
                            - return_5m
                        ) > 0.01
                    THEN 1 ELSE 0
                END
            )
                AS divergent_bars,

            ROUND(
                CORR(
                    perp_mid_return_5m,
                    return_5m
                ),
                4
            )
                AS corr_perp_index

        FROM market_time_series_state

        GROUP BY underlying

        ORDER BY underlying
        """
    ).fetchall()

    print(
        "\nMARKET RETURN SOURCE"
    )

    print(
        "=" * 85
    )

    print(
        f"Structural return source:        {RETURN_SOURCE}"
    )

    print(
        f"Returns on ineligible index:     {int(violations[0] or 0):,}"
    )

    print(
        f"Rows with wrong return_source:   {int(violations[1] or 0):,}"
    )

    print(
        f"\n{'Asset':<10}"
        f"{'Index ret':>12}"
        f"{'Perp ret':>12}"
        f"{'|perp-idx|>100bp':>18}"
        f"{'Corr':>10}"
    )

    for row in rows:

        print(
            f"{row[0]:<10}"
            f"{row[1]:>12,}"
            f"{row[2]:>12,}"
            f"{int(row[3] or 0):>18,}"
            f"{str(row[4]):>10}"
        )

    if violations[0]:
        raise RuntimeError(
            "market_time_series_state has structural returns "
            "on states whose index input is not eligible."
        )

    if violations[1]:
        raise RuntimeError(
            "market_time_series_state has rows whose "
            "return_source differs from RETURN_SOURCE."
        )


def print_market_feature_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            expiry_date,

            COUNT(*)
                AS states,

            SUM(
                CASE
                    WHEN return_5m IS NOT NULL
                    THEN 1 ELSE 0
                END
            )
                AS return_states,

            SUM(
                CASE
                    WHEN realized_vol_1h_ready
                    THEN 1 ELSE 0
                END
            )
                AS rv_1h_states,

            SUM(
                CASE
                    WHEN microstructure_core_ready
                    THEN 1 ELSE 0
                END
            )
                AS microstructure_states,

            SUM(
                CASE
                    WHEN l2_instability_features_ready
                    THEN 1 ELSE 0
                END
            )
                AS instability_states

        FROM market_time_series_state

        GROUP BY expiry_date

        ORDER BY expiry_date
        """
    ).fetchall()

    print(
        "\nMARKET TIME-SERIES FEATURES"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Expiry':<15}"
        f"{'States':>12}"
        f"{'5m return':>14}"
        f"{'RV 1h':>12}"
        f"{'Microstruct':>16}"
        f"{'Instability':>16}"
    )

    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>12,}"
            f"{row[2]:>14,}"
            f"{row[3]:>12,}"
            f"{row[4]:>16,}"
            f"{row[5]:>16,}"
        )


def print_l2_instability_summary(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            underlying,

            SUM(
                CASE
                    WHEN l2_instability_features_ready
                    THEN 1 ELSE 0
                END
            )
                AS states,

            MIN(
                CASE
                    WHEN l2_instability_features_ready
                    THEN state_time
                END
            )
                AS first_state,

            MAX(
                CASE
                    WHEN l2_instability_features_ready
                    THEN state_time
                END
            )
                AS last_state,

            ROUND(
                MEDIAN(
                    CASE
                        WHEN l2_instability_features_ready
                        THEN book_obi_20_std
                    END
                ),
                6
            )
                AS median_std,

            ROUND(
                MEDIAN(
                    CASE
                        WHEN l2_instability_features_ready
                        THEN
                            book_obi_20_path_roughness_per_min
                    END
                ),
                6
            )
                AS median_roughness,

            ROUND(
                MEDIAN(
                    CASE
                        WHEN l2_instability_features_ready
                        THEN
                            book_obi_20_coherence_noise
                    END
                ),
                6
            )
                AS median_noise

        FROM market_time_series_state

        GROUP BY underlying

        ORDER BY underlying
        """
    ).fetchall()

    print(
        "\nL2 INSTABILITY FEATURE COVERAGE"
    )

    print(
        "=" * 125
    )

    print(
        f"{'Asset':<12}"
        f"{'States':>10}"
        f"{'First':>28}"
        f"{'Last':>28}"
        f"{'Med std':>12}"
        f"{'Med rough':>14}"
        f"{'Med noise':>12}"
    )

    for row in rows:

        print(
            f"{row[0]:<12}"
            f"{row[1]:>10,}"
            f"{str(row[2]):>28}"
            f"{str(row[3]):>28}"
            f"{str(row[4]):>12}"
            f"{str(row[5]):>14}"
            f"{str(row[6]):>12}"
        )


def main() -> None:

    connection = open_research_database()

    try:

        materialize_market_feature_inputs(
            connection
        )

        create_futures_integrity_inputs(
            connection
        )

        create_l2_instability_distribution_state(
            connection
        )

        create_market_time_series_state(
            connection
        )

        validate_market_returns(
            connection
        )

        print_market_feature_summary(
            connection
        )

        print_l2_instability_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
