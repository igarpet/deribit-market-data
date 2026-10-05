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

from data_engineering.canonical.build_canonical import (
    build_canonical,
)

from data_engineering.normalization.futures_reference import (
    REFERENCE_LAYER_VERSION,
    create_futures_reference_state,
    print_reference_summary,
)

from data_engineering.normalization.option_normalization import (
    create_normalized_option_chain,
    create_option_family_chain,
    create_option_forward_state,
    print_forward_summary,
    print_normalization_summary,
)


NORMALIZATION_SCHEMA_VERSION = "0.2"


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def register_normalization_metadata(
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
            'reference_layer_version',
            ?
        )
        """,
        [
            REFERENCE_LAYER_VERSION
        ]
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO
        catalog_metadata
        VALUES (
            'normalization_schema_version',
            ?
        )
        """,
        [
            NORMALIZATION_SCHEMA_VERSION
        ]
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

def validate_normalized_views(
    connection
) -> None:

    reference_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    COUNT(*) AS n

                FROM futures_reference_state

                GROUP BY
                    state_id

                HAVING
                    COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )


    family_forward_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    option_family_id,
                    COUNT(*) AS n

                FROM option_forward_state

                GROUP BY
                    state_id,
                    option_family_id

                HAVING
                    COUNT(*) > 1
            )
            """
        )
        .fetchone()[0]
    )


    normalized_duplicates = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM (
                SELECT
                    state_id,
                    instrument_name,
                    COUNT(*) AS n

                FROM normalized_option_chain

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


    bad_log_moneyness = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM normalized_option_chain

            WHERE
                log_moneyness IS NOT NULL

                AND

                (
                    strike <= 0
                    OR
                    option_forward_price <= 0
                )
            """
        )
        .fetchone()[0]
    )


    bad_expiry_time = (
        connection.execute(
            """
            SELECT COUNT(*)

            FROM normalized_option_chain

            WHERE
                time_to_expiry_seconds < 0
            """
        )
        .fetchone()[0]
    )


    print(
        "\nNORMALIZATION VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Reference duplicate states:      "
        f"{reference_duplicates:,}"
    )

    print(
        f"Forward duplicate family states: "
        f"{family_forward_duplicates:,}"
    )

    print(
        f"Normalized option duplicates:    "
        f"{normalized_duplicates:,}"
    )

    print(
        f"Invalid log-moneyness rows:       "
        f"{bad_log_moneyness:,}"
    )

    print(
        f"Negative time-to-expiry rows:     "
        f"{bad_expiry_time:,}"
    )


    if reference_duplicates:

        raise RuntimeError(
            "futures_reference_state "
            "contains duplicate state_id values."
        )

    if family_forward_duplicates:

        raise RuntimeError(
            "option_forward_state contains duplicate "
            "(state_id, option_family_id) keys."
        )

    if normalized_duplicates:

        raise RuntimeError(
            "normalized_option_chain contains duplicate "
            "(state_id, instrument_name) keys."
        )

    if bad_log_moneyness:

        raise RuntimeError(
            "normalized_option_chain contains invalid "
            "log-moneyness rows."
        )

    if bad_expiry_time:

        raise RuntimeError(
            "normalized_option_chain contains negative "
            "time-to-expiry rows."
        )


# --------------------------------------------------
# Build
# --------------------------------------------------

def build_normalized(
    connection
) -> None:

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

    register_normalization_metadata(
        connection
    )

    validate_normalized_views(
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

        build_normalized(
            connection
        )

        print_reference_summary(
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
