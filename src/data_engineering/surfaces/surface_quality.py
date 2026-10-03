from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.catalog.database import open_research_database
from data_engineering.surfaces.build_surfaces import build_surfaces

SURFACE_QUALITY_VERSION = "0.1"

CORE_TIGHT_MAX_SPAN = 0.025
CORE_USABLE_MAX_SPAN = 0.050
WING_050_MAX_SPAN = 0.050
FULL_GRID_MAX_SPAN = 0.100


def create_surface_quality_state(connection) -> None:

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW surface_quality_state AS

        WITH aggregated AS (

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
                surface_source,

                MAX(state_research_ready)
                    AS state_research_ready,

                SUM(
                    CASE
                        WHEN target_log_moneyness
                            IN (-0.025, 0.000, 0.025)
                        AND interpolated_iv IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS core_node_count,

                MAX(
                    CASE
                        WHEN target_log_moneyness
                            IN (-0.025, 0.000, 0.025)
                        THEN interpolation_span_log_moneyness
                    END
                ) AS core_max_interpolation_span,

                SUM(
                    CASE
                        WHEN target_log_moneyness
                            IN (-0.050, 0.000, 0.050)
                        AND interpolated_iv IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS wing_050_node_count,

                MAX(
                    CASE
                        WHEN target_log_moneyness
                            IN (-0.050, 0.000, 0.050)
                        THEN interpolation_span_log_moneyness
                    END
                ) AS wing_050_max_interpolation_span,

                SUM(
                    CASE
                        WHEN interpolated_iv IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS full_grid_node_count,

                MAX(interpolation_span_log_moneyness)
                    AS full_grid_max_interpolation_span,

                MAX(
                    CASE
                        WHEN target_log_moneyness = 0
                        THEN interpolated_iv
                    END
                ) AS iv_atm_forward,

                MAX(interpolation_max_quote_age_ms)
                    AS max_node_quote_age_ms,

                MAX(interpolation_max_spread_bps)
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
            *,

            (
                state_research_ready
                AND core_node_count = 3
                AND core_max_interpolation_span
                    <= {CORE_TIGHT_MAX_SPAN}
            ) AS core_smile_tight,

            (
                state_research_ready
                AND core_node_count = 3
                AND core_max_interpolation_span
                    <= {CORE_USABLE_MAX_SPAN}
            ) AS core_smile_usable,

            (
                state_research_ready
                AND wing_050_node_count = 3
                AND wing_050_max_interpolation_span
                    <= {WING_050_MAX_SPAN}
            ) AS wing_050_usable,

            (
                state_research_ready
                AND full_grid_node_count = 7
                AND full_grid_max_interpolation_span
                    <= {FULL_GRID_MAX_SPAN}
            ) AS full_grid_usable,

            CASE
                WHEN state_research_ready
                    AND core_node_count = 3
                    AND core_max_interpolation_span
                        <= {CORE_TIGHT_MAX_SPAN}
                    THEN 'A'

                WHEN state_research_ready
                    AND core_node_count = 3
                    AND core_max_interpolation_span
                        <= {CORE_USABLE_MAX_SPAN}
                    THEN 'B'

                WHEN state_research_ready
                    AND core_node_count = 3
                    THEN 'C'

                ELSE 'D'
            END AS core_smile_quality_tier

        FROM aggregated
        """
    )


def print_surface_quality_summary(connection) -> None:

    rows = connection.execute(
        """
        SELECT
            option_family_id,
            surface_source,
            COUNT(*) AS states,
            SUM(CASE WHEN core_smile_tight THEN 1 ELSE 0 END)
                AS tight_core,
            SUM(CASE WHEN core_smile_usable THEN 1 ELSE 0 END)
                AS usable_core,
            SUM(CASE WHEN wing_050_usable THEN 1 ELSE 0 END)
                AS usable_050,
            SUM(CASE WHEN full_grid_usable THEN 1 ELSE 0 END)
                AS usable_full

        FROM surface_quality_state

        GROUP BY
            option_family_id,
            surface_source

        ORDER BY
            option_family_id,
            surface_source
        """
    ).fetchall()

    print("\nSURFACE QUALITY CONTRACT")
    print("=" * 130)
    print(
        f"{'Family':<42}"
        f"{'Source':<12}"
        f"{'States':>10}"
        f"{'Core A':>10}"
        f"{'Core A/B':>12}"
        f"{'+/-5 usable':>14}"
        f"{'Full usable':>14}"
    )

    for row in rows:
        print(
            f"{row[0]:<42}"
            f"{row[1]:<12}"
            f"{row[2]:>10,}"
            f"{row[3]:>10,}"
            f"{row[4]:>12,}"
            f"{row[5]:>14,}"
            f"{row[6]:>14,}"
        )


def main() -> None:

    connection = open_research_database()

    try:
        build_surfaces(connection)
        create_surface_quality_state(connection)
        print_surface_quality_summary(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
