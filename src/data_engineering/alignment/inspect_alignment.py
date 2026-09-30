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
    refresh_raw_views,
)


# --------------------------------------------------
# Coverage windows
# --------------------------------------------------

def print_coverage_windows(
    connection
) -> None:

    rows = connection.execute(
        """
        WITH futures AS (

            SELECT
                underlying,

                MIN(bar_end)
                    AS first_state,

                MAX(bar_end)
                    AS last_state,

                COUNT(
                    DISTINCT bar_end
                )
                    AS states

            FROM raw_futures_5m

            GROUP BY
                underlying
        ),

        options AS (

            SELECT
                underlying,

                MIN(
                    scheduled_snapshot_time
                )
                    AS first_state,

                MAX(
                    scheduled_snapshot_time
                )
                    AS last_state,

                COUNT(
                    DISTINCT
                    scheduled_snapshot_time
                )
                    AS states

            FROM raw_option_grid

            GROUP BY
                underlying
        )

        SELECT
            f.underlying,

            f.first_state,
            f.last_state,
            f.states,

            o.first_state,
            o.last_state,
            o.states,

            GREATEST(
                f.first_state,
                o.first_state
            )
                AS common_start,

            LEAST(
                f.last_state,
                o.last_state
            )
                AS common_end

        FROM futures f

        INNER JOIN options o

            ON
                f.underlying
                = o.underlying

        ORDER BY
            f.underlying
        """
    ).fetchall()

    print(
        "\nCOVERAGE WINDOWS"
    )

    print(
        "=" * 100
    )

    for row in rows:

        print(
            f"\n{row[0]}"
        )

        print(
            f"  Futures: "
            f"{row[1]} -> {row[2]} "
            f"({row[3]:,} states)"
        )

        print(
            f"  Options: "
            f"{row[4]} -> {row[5]} "
            f"({row[6]:,} states)"
        )

        print(
            f"  Common:  "
            f"{row[7]} -> {row[8]}"
        )


# --------------------------------------------------
# Exact matching inside common window
# --------------------------------------------------

def print_common_window_alignment(
    connection
) -> None:

    rows = connection.execute(
        """
        WITH coverage AS (

            SELECT
                f.underlying,

                GREATEST(
                    MIN(f.bar_end),
                    MIN(o.scheduled_snapshot_time)
                )
                    AS common_start,

                LEAST(
                    MAX(f.bar_end),
                    MAX(o.scheduled_snapshot_time)
                )
                    AS common_end

            FROM raw_futures_5m f

            INNER JOIN raw_option_grid o

                ON
                    f.underlying
                    = o.underlying

            GROUP BY
                f.underlying
        ),

        futures_states AS (

            SELECT DISTINCT
                f.underlying,
                f.bar_end
                    AS state_time

            FROM raw_futures_5m f

            INNER JOIN coverage c

                ON
                    f.underlying
                    = c.underlying

            WHERE
                f.bar_end
                BETWEEN
                    c.common_start
                    AND c.common_end
        ),

        option_states AS (

            SELECT DISTINCT
                o.underlying,
                o.scheduled_snapshot_time
                    AS state_time

            FROM raw_option_grid o

            INNER JOIN coverage c

                ON
                    o.underlying
                    = c.underlying

            WHERE
                o.scheduled_snapshot_time
                BETWEEN
                    c.common_start
                    AND c.common_end
        )

        SELECT
            o.underlying,

            COUNT(*)
                AS option_states,

            COUNT(f.state_time)
                AS matched_states,

            COUNT(*)
                - COUNT(f.state_time)
                AS missing_futures

        FROM option_states o

        LEFT JOIN futures_states f

            ON
                o.underlying
                = f.underlying

            AND
                o.state_time
                = f.state_time

        GROUP BY
            o.underlying

        ORDER BY
            o.underlying
        """
    ).fetchall()

    print(
        "\nALIGNMENT INSIDE COMMON WINDOW"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Underlying':<15}"
        f"{'Option states':>15}"
        f"{'Matched':>15}"
        f"{'Missing':>15}"
        f"{'Coverage':>15}"
    )

    for row in rows:

        coverage = (
            row[2]
            / row[1]
            * 100
            if row[1]
            else 0
        )

        print(
            f"{row[0]:<15}"
            f"{row[1]:>15,}"
            f"{row[2]:>15,}"
            f"{row[3]:>15,}"
            f"{coverage:>14.2f}%"
        )


# --------------------------------------------------
# Five-minute grid test
# --------------------------------------------------

def print_grid_errors(
    connection
) -> None:

    futures_bad = connection.execute(
        """
        SELECT COUNT(*)

        FROM (

            SELECT DISTINCT
                underlying,
                bar_end

            FROM raw_futures_5m
        )

        WHERE
            EXTRACT(
                MINUTE
                FROM bar_end
            ) % 5 != 0

            OR

            EXTRACT(
                SECOND
                FROM bar_end
            ) != 0
        """
    ).fetchone()[0]

    options_bad = connection.execute(
        """
        SELECT COUNT(*)

        FROM (

            SELECT DISTINCT
                underlying,
                scheduled_snapshot_time

            FROM raw_option_grid
        )

        WHERE
            EXTRACT(
                MINUTE
                FROM scheduled_snapshot_time
            ) % 5 != 0

            OR

            EXTRACT(
                SECOND
                FROM scheduled_snapshot_time
            ) != 0
        """
    ).fetchone()[0]

    print(
        "\nFIVE-MINUTE GRID CHECK"
    )

    print(
        "=" * 100
    )

    print(
        f"Invalid futures timestamps: "
        f"{futures_bad:,}"
    )

    print(
        f"Invalid option timestamps:  "
        f"{options_bad:,}"
    )


# --------------------------------------------------
# Missing timestamp examples
# --------------------------------------------------

def print_missing_examples(
    connection
) -> None:

    rows = connection.execute(
        """
        WITH futures_states AS (

            SELECT DISTINCT
                underlying,
                bar_end
                    AS state_time

            FROM raw_futures_5m
        ),

        option_states AS (

            SELECT DISTINCT
                underlying,
                scheduled_snapshot_time
                    AS state_time

            FROM raw_option_grid
        )

        SELECT
            o.underlying,
            o.state_time

        FROM option_states o

        LEFT JOIN futures_states f

            ON
                o.underlying
                = f.underlying

            AND
                o.state_time
                = f.state_time

        WHERE
            f.state_time IS NULL

        ORDER BY
            o.state_time,
            o.underlying

        LIMIT 30
        """
    ).fetchall()

    print(
        "\nFIRST 30 OPTION STATES "
        "WITHOUT FUTURES MATCH"
    )

    print(
        "=" * 100
    )

    for row in rows:

        print(
            f"{row[0]:<10} "
            f"{row[1]}"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    connection = (
        open_research_database()
    )

    try:

        refresh_raw_views(
            connection
        )

        print_coverage_windows(
            connection
        )

        print_common_window_alignment(
            connection
        )

        print_grid_errors(
            connection
        )

        print_missing_examples(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()