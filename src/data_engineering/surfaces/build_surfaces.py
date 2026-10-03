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
)

from data_engineering.normalization.build_normalized import (
    build_normalized,
)

from data_engineering.surfaces.smile_grid import (
    SMILE_GRID_VERSION,
    create_option_smile_state,
    create_smile_node_state,
    create_surface_input_state,
    print_smile_coverage,
    print_smile_feature_summary,
)


SURFACE_SCHEMA_VERSION = "0.1"


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def register_surface_metadata(
    connection
) -> None:

    initialize_metadata(
        connection
    )

    for key, value in [
        (
            "smile_grid_version",
            SMILE_GRID_VERSION,
        ),
        (
            "surface_schema_version",
            SURFACE_SCHEMA_VERSION,
        ),
    ]:

        connection.execute(
            """
            INSERT OR REPLACE INTO
            catalog_metadata
            VALUES (
                ?,
                ?
            )
            """,
            [
                key,
                value,
            ]
        )


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_surface_views(
    connection
) -> None:

    node_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    option_family_id,
                    surface_source,
                    target_log_moneyness,
                    COUNT(*) AS n

                FROM smile_node_state

                GROUP BY
                    state_id,
                    option_family_id,
                    surface_source,
                    target_log_moneyness

                HAVING
                    COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )


    smile_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    option_family_id,
                    surface_source,
                    COUNT(*) AS n

                FROM option_smile_state

                GROUP BY
                    state_id,
                    option_family_id,
                    surface_source

                HAVING
                    COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )


    extrapolated_nodes = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM smile_node_state

            WHERE
                interpolated_iv IS NOT NULL

                AND

                node_interpolated_without_extrapolation
                    IS NOT TRUE
            """
        )
        .fetchone()[0]
    )


    invalid_node_iv = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM smile_node_state

            WHERE
                interpolated_iv IS NOT NULL

                AND

                interpolated_iv <= 0
            """
        )
        .fetchone()[0]
    )


    print(
        "\nSURFACE VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Duplicate smile nodes:       "
        f"{node_duplicates:,}"
    )

    print(
        f"Duplicate smile states:      "
        f"{smile_duplicates:,}"
    )

    print(
        f"Extrapolated populated nodes:"
        f" {extrapolated_nodes:,}"
    )

    print(
        f"Non-positive populated IVs:  "
        f"{invalid_node_iv:,}"
    )


    if node_duplicates:

        raise RuntimeError(
            "smile_node_state contains "
            "duplicate node keys."
        )

    if smile_duplicates:

        raise RuntimeError(
            "option_smile_state contains "
            "duplicate smile-state keys."
        )

    if extrapolated_nodes:

        raise RuntimeError(
            "smile_node_state contains "
            "populated extrapolated nodes."
        )

    if invalid_node_iv:

        raise RuntimeError(
            "smile_node_state contains "
            "non-positive interpolated IV."
        )


# --------------------------------------------------
# Build
# --------------------------------------------------

def build_surfaces(
    connection
) -> None:

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

    register_surface_metadata(
        connection
    )

    validate_surface_views(
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

        build_surfaces(
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
