from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from data_engineering.catalog.database import (
    initialize_metadata,
    open_research_database,
)

from data_engineering.surfaces.build_surface_features import (
    build_surface_features,
)

from data_engineering.features.market_time_series import (
    MARKET_FEATURE_VERSION,
    create_l2_instability_distribution_state,
    create_market_time_series_state,
    drop_persistent_relation,
    materialize_market_feature_inputs,
    print_l2_instability_summary,
    print_market_feature_summary,
)

from data_engineering.features.option_time_series import (
    OPTION_FEATURE_VERSION,
    create_option_surface_feature_state,
    create_primary_option_surface_feature_state,
    materialize_option_feature_inputs,
    print_primary_family_summary,
)


TIME_SERIES_FEATURE_SCHEMA_VERSION = "0.1.2"


REQUIRED_UPSTREAM_RELATIONS = (
    "canonical_market_state",
    "futures_reference_state",
    "surface_input_state",
    "option_smile_state",
    "surface_quality_state",
    "delta_smile_state",
)


def relation_exists(
    connection,
    relation_name: str
) -> bool:

    try:

        connection.execute(
            f"""
            SELECT *
            FROM {relation_name}
            LIMIT 0
            """
        )

        return True

    except Exception:

        return False


def ensure_upstream_relations(
    connection,
    refresh_upstream: bool
) -> None:

    missing = [
        relation
        for relation
        in REQUIRED_UPSTREAM_RELATIONS
        if not relation_exists(
            connection,
            relation
        )
    ]

    if (
        refresh_upstream
        or missing
    ):

        if missing:

            print(
                "\nMissing upstream relations:"
            )

            for relation in missing:
                print(
                    f"  {relation}"
                )

        print(
            "\n[FEATURE BUILD] "
            "Refreshing upstream surface pipeline..."
        )

        started = time.perf_counter()

        build_surface_features(
            connection
        )

        print(
            "[FEATURE BUILD] "
            "Upstream refresh complete in "
            f"{time.perf_counter() - started:,.2f}s"
        )

    else:

        print(
            "\n[FEATURE BUILD] "
            "Using existing upstream surface relations."
        )

        print(
            "[FEATURE BUILD] "
            "Use --refresh-upstream when you explicitly "
            "want to rebuild canonical/normalization/surfaces."
        )


def create_time_series_feature_state(
    connection
) -> None:

    print(
        "\n[FEATURE BUILD] "
        "Build final time-series feature table..."
    )

    started = time.perf_counter()

    drop_persistent_relation(
        connection,
        "time_series_feature_state"
    )

    connection.execute(
        """
        CREATE TABLE
        time_series_feature_state AS

        SELECT
            m.*,

            p.option_family_id
                AS primary_option_family_id,

            p.instrument_type
                AS primary_option_instrument_type,

            p.settlement_currency
                AS primary_option_settlement_currency,

            p.api_currency
                AS primary_option_api_currency,

            p.option_family_selection_reason,

            p.option_forward_price,

            p.iv_atm_forward
                AS option_iv_atm_forward,

            p.symmetric_025_iv_difference
                AS option_skew_025,

            p.symmetric_050_iv_difference
                AS option_skew_050,

            p.curvature_025
                AS option_curvature_025,

            p.curvature_050
                AS option_curvature_050,

            p.local_skew_slope_025
                AS option_local_skew_slope_025,

            p.local_skew_slope_050
                AS option_local_skew_slope_050,

            p.core_smile_quality_tier,
            p.core_smile_tight,
            p.core_smile_usable,
            p.wing_050_usable,
            p.full_grid_usable,

            p.core_max_interpolation_span,
            p.wing_050_max_interpolation_span,
            p.full_grid_max_interpolation_span,

            p.base_eligible_rows
                AS option_base_eligible_rows,

            p.two_sided_eligible_rows
                AS option_two_sided_eligible_rows,

            p.median_two_sided_spread_bps
                AS option_median_two_sided_spread_bps,

            p.p90_two_sided_spread_bps
                AS option_p90_two_sided_spread_bps,

            p.atm_iv_change_5m,
            p.atm_iv_change_15m,
            p.atm_iv_change_1h,

            p.skew_025_change_5m,
            p.skew_025_change_15m,
            p.skew_025_change_1h,

            p.curvature_025_change_5m,
            p.curvature_025_change_15m,
            p.curvature_025_change_1h,

            p.skew_050_change_5m,

            p.core_surface_change_5m_ready,

            p.iv_put_25d,
            p.iv_call_25d,
            p.rr_25d,
            p.bf_25d,

            p.iv_put_10d,
            p.iv_call_10d,
            p.rr_10d,
            p.bf_10d,

            p.delta_25d_smile_available,
            p.full_delta_smile_available,

            p.max_delta_interpolation_span,
            p.median_delta_interpolation_span,

            p.rr_25d_change_5m,
            p.bf_25d_change_5m,

            p.delta_25d_change_5m_ready,

            (
                m.state_research_ready
                AND
                m.return_5m IS NOT NULL
                AND
                p.core_smile_usable
            )
                AS structural_level_features_ready,

            (
                m.state_research_ready
                AND
                m.return_5m IS NOT NULL
                AND
                m.realized_vol_1h_ready
                AND
                p.core_surface_change_5m_ready
            )
                AS structural_dynamic_features_ready,

            m.microstructure_core_ready
                AS microstructure_features_ready,

            m.l2_instability_features_ready
                AS instability_features_ready

        FROM market_time_series_state m

        LEFT JOIN primary_option_surface_feature_state p
            ON m.state_id = p.state_id
        """
    )

    count = connection.execute(
        """
        SELECT COUNT(*)
        FROM time_series_feature_state
        """
    ).fetchone()[0]

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        "[FEATURE BUILD] "
        "Build final time-series feature table: "
        f"{count:,} rows in {elapsed:,.2f}s"
    )


def register_feature_metadata(
    connection
) -> None:

    initialize_metadata(
        connection
    )

    for (
        key,
        value
    ) in [
        (
            "market_feature_version",
            MARKET_FEATURE_VERSION,
        ),
        (
            "option_feature_version",
            OPTION_FEATURE_VERSION,
        ),
        (
            "time_series_feature_schema_version",
            TIME_SERIES_FEATURE_SCHEMA_VERSION,
        ),
    ]:

        connection.execute(
            """
            INSERT OR REPLACE INTO
            catalog_metadata
            VALUES (?, ?)
            """,
            [
                key,
                value,
            ]
        )


def validate_feature_tables(
    connection
) -> None:

    print(
        "\n[FEATURE BUILD] "
        "Validate materialized feature tables..."
    )

    started = time.perf_counter()

    market_duplicates = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT state_id
            FROM market_time_series_state
            GROUP BY state_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    family_duplicates = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                state_id,
                option_family_id

            FROM option_surface_feature_state

            GROUP BY
                state_id,
                option_family_id

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    primary_duplicates = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT state_id

            FROM primary_option_surface_feature_state

            GROUP BY state_id

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    unstable_primary_families = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                underlying

            FROM primary_option_surface_feature_state

            GROUP BY underlying

            HAVING
                COUNT(
                    DISTINCT option_family_id
                ) > 1
        )
        """
    ).fetchone()[0]

    feature_duplicates = connection.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT state_id

            FROM time_series_feature_state

            GROUP BY state_id

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    opening_option_changes = connection.execute(
        """
        SELECT COUNT(*)
        FROM time_series_feature_state

        WHERE
            is_opening_state

            AND (
                atm_iv_change_5m IS NOT NULL
                OR
                skew_025_change_5m IS NOT NULL
                OR
                curvature_025_change_5m IS NOT NULL
                OR
                rr_25d_change_5m IS NOT NULL
                OR
                bf_25d_change_5m IS NOT NULL
            )
        """
    ).fetchone()[0]

    invalid_progress = connection.execute(
        """
        SELECT COUNT(*)
        FROM time_series_feature_state

        WHERE
            session_progress_fraction < 0
            OR
            session_progress_fraction >= 1
        """
    ).fetchone()[0]

    print(
        "\nTIME-SERIES FEATURE VALIDATION"
    )

    print(
        "=" * 85
    )

    print(
        f"Duplicate market states:       "
        f"{market_duplicates:,}"
    )

    print(
        f"Duplicate option-family states:"
        f" {family_duplicates:,}"
    )

    print(
        f"Duplicate primary-option rows: "
        f"{primary_duplicates:,}"
    )

    print(
        f"Unstable primary families:     "
        f"{unstable_primary_families:,}"
    )

    print(
        f"Duplicate feature states:      "
        f"{feature_duplicates:,}"
    )

    print(
        f"Opening option-change rows:    "
        f"{opening_option_changes:,}"
    )

    print(
        f"Invalid session progress rows: "
        f"{invalid_progress:,}"
    )

    if market_duplicates:
        raise RuntimeError(
            "market_time_series_state contains "
            "duplicate state_id values."
        )

    if family_duplicates:
        raise RuntimeError(
            "option_surface_feature_state contains "
            "duplicate state/family rows."
        )

    if primary_duplicates:
        raise RuntimeError(
            "primary_option_surface_feature_state "
            "contains duplicate state_id values."
        )

    if unstable_primary_families:
        raise RuntimeError(
            "Primary option-family selection changed "
            "within an underlying."
        )

    if feature_duplicates:
        raise RuntimeError(
            "time_series_feature_state contains "
            "duplicate state_id values."
        )

    if opening_option_changes:
        raise RuntimeError(
            "Option-surface changes crossed an "
            "08:00 UTC session boundary."
        )

    if invalid_progress:
        raise RuntimeError(
            "session_progress_fraction is outside "
            "[0, 1)."
        )

    print(
        "[FEATURE BUILD] Validation complete in "
        f"{time.perf_counter() - started:,.2f}s"
    )


def print_feature_state_summary(
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
                    WHEN state_research_ready
                    THEN 1 ELSE 0
                END
            )
                AS research_ready,

            SUM(
                CASE
                    WHEN structural_level_features_ready
                    THEN 1 ELSE 0
                END
            )
                AS structural_level,

            SUM(
                CASE
                    WHEN structural_dynamic_features_ready
                    THEN 1 ELSE 0
                END
            )
                AS structural_dynamic,

            SUM(
                CASE
                    WHEN microstructure_features_ready
                    THEN 1 ELSE 0
                END
            )
                AS microstructure,

            SUM(
                CASE
                    WHEN instability_features_ready
                    THEN 1 ELSE 0
                END
            )
                AS instability

        FROM time_series_feature_state

        GROUP BY expiry_date

        ORDER BY expiry_date
        """
    ).fetchall()

    print(
        "\nTIME-SERIES FEATURE STATE"
    )

    print(
        "=" * 125
    )

    print(
        f"{'Expiry':<15}"
        f"{'States':>12}"
        f"{'Research':>12}"
        f"{'Struct lvl':>14}"
        f"{'Struct dyn':>14}"
        f"{'Microstruct':>16}"
        f"{'Instability':>16}"
    )

    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>12,}"
            f"{row[2]:>12,}"
            f"{row[3]:>14,}"
            f"{row[4]:>14,}"
            f"{row[5]:>16,}"
            f"{row[6]:>16,}"
        )


def build_features(
    connection,
    refresh_upstream: bool = False
) -> None:

    ensure_upstream_relations(
        connection,
        refresh_upstream=refresh_upstream
    )

    materialize_market_feature_inputs(
        connection
    )

    create_l2_instability_distribution_state(
        connection
    )

    create_market_time_series_state(
        connection
    )

    materialize_option_feature_inputs(
        connection
    )

    create_option_surface_feature_state(
        connection
    )

    create_primary_option_surface_feature_state(
        connection
    )

    create_time_series_feature_state(
        connection
    )

    register_feature_metadata(
        connection
    )

    validate_feature_tables(
        connection
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refresh-upstream",
        action="store_true",
        help=(
            "Rebuild canonical, normalization, smile and "
            "surface views before materializing features."
        )
    )

    args = parser.parse_args()

    connection = open_research_database()

    try:

        build_features(
            connection,
            refresh_upstream=(
                args.refresh_upstream
            )
        )

        print_market_feature_summary(
            connection
        )

        print_primary_family_summary(
            connection
        )

        print_l2_instability_summary(
            connection
        )

        print_feature_state_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
