from __future__ import annotations

from database import (
    open_research_database,
    refresh_raw_views,
)


def print_schema(
    connection,
    view_name: str
) -> None:

    rows = connection.execute(
        f"""
        DESCRIBE {view_name}
        """
    ).fetchall()

    print(
        f"\n{view_name.upper()} SCHEMA"
    )

    print(
        "=" * 80
    )

    for row in rows:

        column_name = row[0]
        column_type = row[1]

        print(
            f"{column_name:<40} "
            f"{column_type}"
        )


def print_underlyings(
    connection
) -> None:

    print(
        "\nUNDERLYINGS"
    )

    print(
        "=" * 80
    )

    futures = connection.execute(
        """
        SELECT DISTINCT underlying
        FROM raw_futures_5m
        ORDER BY underlying
        """
    ).fetchall()

    options = connection.execute(
        """
        SELECT DISTINCT underlying
        FROM raw_option_grid
        ORDER BY underlying
        """
    ).fetchall()

    print(
        "Futures:"
    )

    print(
        [
            row[0]
            for row in futures
        ]
    )

    print(
        "Options:"
    )

    print(
        [
            row[0]
            for row in options
        ]
    )


def print_state_alignment(
    connection
) -> None:

    result = connection.execute(
        """
        WITH futures_states AS (

            SELECT DISTINCT
                underlying,
                bar_end AS state_time

            FROM raw_futures_5m

        ),

        option_states AS (

            SELECT DISTINCT
                underlying,
                scheduled_snapshot_time
                    AS state_time

            FROM raw_option_grid

        ),

        matched AS (

            SELECT
                f.underlying,
                f.state_time

            FROM futures_states f

            INNER JOIN option_states o

                ON
                    f.underlying
                    = o.underlying

                AND
                    f.state_time
                    = o.state_time
        )

        SELECT
            (
                SELECT COUNT(*)
                FROM futures_states
            )
                AS futures_states,

            (
                SELECT COUNT(*)
                FROM option_states
            )
                AS option_states,

            (
                SELECT COUNT(*)
                FROM matched
            )
                AS matched_states
        """
    ).fetchone()

    futures_states = result[0]
    option_states = result[1]
    matched_states = result[2]

    print(
        "\nSTATE ALIGNMENT"
    )

    print(
        "=" * 80
    )

    print(
        f"Unique futures states: "
        f"{futures_states:,}"
    )

    print(
        f"Unique option states:  "
        f"{option_states:,}"
    )

    print(
        f"Exact matches:         "
        f"{matched_states:,}"
    )

    if option_states:

        coverage = (
            matched_states
            / option_states
            * 100
        )

        print(
            f"Option-state coverage: "
            f"{coverage:.2f}%"
        )


def print_alignment_by_underlying(
    connection
) -> None:

    rows = connection.execute(
        """
        WITH futures_states AS (

            SELECT DISTINCT
                underlying,
                bar_end AS state_time

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

            COUNT(*)
                AS option_states,

            COUNT(f.state_time)
                AS matched_states

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
        "\nALIGNMENT BY UNDERLYING"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Underlying':<15}"
        f"{'Option states':>15}"
        f"{'Matched':>15}"
        f"{'Coverage':>15}"
    )

    for row in rows:

        underlying = row[0]
        option_states = row[1]
        matched_states = row[2]

        coverage = (
            matched_states
            / option_states
            * 100
            if option_states
            else 0
        )

        print(
            f"{underlying:<15}"
            f"{option_states:>15,}"
            f"{matched_states:>15,}"
            f"{coverage:>14.2f}%"
        )


def main() -> None:

    connection = (
        open_research_database()
    )

    try:

        refresh_raw_views(
            connection
        )

        print_schema(
            connection,
            "raw_futures_5m"
        )

        print_schema(
            connection,
            "raw_option_grid"
        )

        print_underlyings(
            connection
        )

        print_state_alignment(
            connection
        )

        print_alignment_by_underlying(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()