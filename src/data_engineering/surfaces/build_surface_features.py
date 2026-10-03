from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.catalog.database import (
    initialize_metadata,
    open_research_database,
)

from data_engineering.surfaces.build_surfaces import (
    build_surfaces,
)

from data_engineering.surfaces.surface_quality import (
    SURFACE_QUALITY_VERSION,
    create_surface_quality_state,
    print_surface_quality_summary,
)

from data_engineering.surfaces.delta_smile import (
    DELTA_SMILE_VERSION,
    create_delta_smile_node_state,
    create_delta_smile_state,
    create_surface_source_comparison,
    print_delta_smile_coverage,
    print_source_overlap_summary,
)


SURFACE_FEATURE_SCHEMA_VERSION = "0.1"


def register_surface_feature_metadata(connection) -> None:

    initialize_metadata(connection)

    for key, value in [
        (
            "surface_quality_version",
            SURFACE_QUALITY_VERSION,
        ),
        (
            "delta_smile_version",
            DELTA_SMILE_VERSION,
        ),
        (
            "surface_feature_schema_version",
            SURFACE_FEATURE_SCHEMA_VERSION,
        ),
    ]:
        connection.execute(
            """
            INSERT OR REPLACE INTO catalog_metadata
            VALUES (?, ?)
            """,
            [key, value],
        )


def validate_surface_features(connection) -> None:

    delta_node_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    state_id,
                    option_family_id,
                    surface_source,
                    delta_node,
                    COUNT(*) AS n

                FROM delta_smile_node_state

                GROUP BY
                    state_id,
                    option_family_id,
                    surface_source,
                    delta_node

                HAVING COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )

    delta_state_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT
                    state_id,
                    option_family_id,
                    surface_source,
                    COUNT(*) AS n

                FROM delta_smile_state

                GROUP BY
                    state_id,
                    option_family_id,
                    surface_source

                HAVING COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )

    extrapolated_delta_nodes = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM delta_smile_node_state
            WHERE
                interpolated_iv IS NOT NULL
                AND node_interpolated_without_extrapolation
                    IS NOT TRUE
            """
        )
        .fetchone()[0]
    )

    invalid_delta_node_iv = (
        connection.execute(
            """
            SELECT COUNT(*)
            FROM delta_smile_node_state
            WHERE
                interpolated_iv IS NOT NULL
                AND interpolated_iv <= 0
            """
        )
        .fetchone()[0]
    )

    print("\nSURFACE FEATURE VALIDATION")
    print("=" * 80)

    print(
        f"Duplicate delta nodes:         "
        f"{delta_node_duplicates:,}"
    )
    print(
        f"Duplicate delta states:        "
        f"{delta_state_duplicates:,}"
    )
    print(
        f"Extrapolated delta nodes:      "
        f"{extrapolated_delta_nodes:,}"
    )
    print(
        f"Non-positive delta-node IVs:   "
        f"{invalid_delta_node_iv:,}"
    )

    if delta_node_duplicates:
        raise RuntimeError(
            "delta_smile_node_state contains "
            "duplicate node keys."
        )

    if delta_state_duplicates:
        raise RuntimeError(
            "delta_smile_state contains "
            "duplicate state keys."
        )

    if extrapolated_delta_nodes:
        raise RuntimeError(
            "delta_smile_node_state contains "
            "populated extrapolated nodes."
        )

    if invalid_delta_node_iv:
        raise RuntimeError(
            "delta_smile_node_state contains "
            "non-positive interpolated IV."
        )


def build_surface_features(connection) -> None:

    build_surfaces(connection)

    create_surface_quality_state(connection)

    create_delta_smile_node_state(connection)

    create_delta_smile_state(connection)

    create_surface_source_comparison(connection)

    register_surface_feature_metadata(connection)

    validate_surface_features(connection)


def main() -> None:

    connection = open_research_database()

    try:
        build_surface_features(connection)

        print_surface_quality_summary(connection)

        print_delta_smile_coverage(connection)

        print_source_overlap_summary(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
