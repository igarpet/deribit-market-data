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

from data_engineering.alignment.state_index import (
    create_state_index,
)


# --------------------------------------------------
# Futures quality
# --------------------------------------------------

def print_futures_quality(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            expiry_date,

            SUM(
                CASE
                    WHEN has_futures
                    THEN 1
                    ELSE 0
                END
            )
                AS states_present,

            SUM(
                CASE
                    WHEN has_futures
                    AND futures_stream_complete
                    THEN 1
                    ELSE 0
                END
            )
                AS stream_complete,

            SUM(
                CASE
                    WHEN has_futures
                    AND book_valid
                    THEN 1
                    ELSE 0
                END
            )
                AS book_valid,

            SUM(
                CASE
                    WHEN has_futures
                    AND book_stats_complete
                    THEN 1
                    ELSE 0
                END
            )
                AS book_stats_complete

        FROM canonical_state_index

        GROUP BY
            expiry_date

        ORDER BY
            expiry_date
        """
    ).fetchall()

    print(
        "\nFUTURES / L2 QUALITY"
    )

    print(
        "=" * 110
    )

    print(
        f"{'Expiry':<15}"
        f"{'Present':>15}"
        f"{'Stream OK':>15}"
        f"{'Book valid':>15}"
        f"{'Book stats OK':>18}"
    )

    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>15,}"
            f"{row[2]:>15,}"
            f"{row[3]:>15,}"
            f"{row[4]:>18,}"
        )


# --------------------------------------------------
# Option quality
# --------------------------------------------------

def print_option_quality(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            expiry_date,

            SUM(
                CASE
                    WHEN has_options
                    THEN 1
                    ELSE 0
                END
            )
                AS states_present,

            ROUND(
                AVG(option_coverage_pct),
                4
            )
                AS avg_coverage,

            MIN(option_coverage_pct)
                AS min_coverage,

            ROUND(
                MEDIAN(option_coverage_pct),
                4
            )
                AS median_coverage,

            ROUND(
                MEDIAN(capture_delay_ms),
                2
            )
                AS median_capture_delay_ms,

            ROUND(
                MAX(capture_delay_ms),
                2
            )
                AS max_capture_delay_ms,

            ROUND(
                MEDIAN(median_quote_age_ms),
                2
            )
                AS median_quote_age_ms,

            ROUND(
                MAX(max_quote_age_ms),
                2
            )
                AS worst_quote_age_ms

        FROM canonical_state_index

        WHERE
            has_options

        GROUP BY
            expiry_date

        ORDER BY
            expiry_date
        """
    ).fetchall()

    print(
        "\nOPTION QUALITY"
    )

    print(
        "=" * 140
    )

    print(
        f"{'Expiry':<15}"
        f"{'Present':>10}"
        f"{'Avg cov %':>12}"
        f"{'Min cov %':>12}"
        f"{'Med cov %':>12}"
        f"{'Med delay':>14}"
        f"{'Max delay':>14}"
        f"{'Med q age':>14}"
        f"{'Worst q age':>16}"
    )

    for row in rows:

        print(
            f"{str(row[0]):<15}"
            f"{row[1]:>10,}"
            f"{row[2]:>12}"
            f"{row[3]:>12.4f}"
            f"{row[4]:>12}"
            f"{row[5]:>14}"
            f"{row[6]:>14}"
            f"{row[7]:>14}"
            f"{row[8]:>16}"
        )


# --------------------------------------------------
# Distribution of option coverage
# --------------------------------------------------

def print_option_coverage_distribution(
    connection
) -> None:

    row = connection.execute(
        """
        SELECT

            COUNT(*),

            SUM(
                CASE
                    WHEN option_coverage_pct = 100
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN option_coverage_pct >= 99
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN option_coverage_pct >= 95
                    THEN 1
                    ELSE 0
                END
            ),

            SUM(
                CASE
                    WHEN option_coverage_pct < 95
                    THEN 1
                    ELSE 0
                END
            )

        FROM canonical_state_index

        WHERE
            has_options
        """
    ).fetchone()

    print(
        "\nOPTION COVERAGE DISTRIBUTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Option states:       {row[0]:,}"
    )

    print(
        f"100% coverage:       {row[1]:,}"
    )

    print(
        f">= 99% coverage:     {row[2]:,}"
    )

    print(
        f">= 95% coverage:     {row[3]:,}"
    )

    print(
        f"< 95% coverage:      {row[4]:,}"
    )


# --------------------------------------------------
# Bad futures examples
# --------------------------------------------------

def print_bad_futures_examples(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            state_time,
            underlying,
            expiry_date,

            futures_stream_complete,
            book_valid,
            book_stats_complete

        FROM canonical_state_index

        WHERE
            has_futures

            AND (
                futures_stream_complete IS NOT TRUE
                OR
                book_valid IS NOT TRUE
                OR
                book_stats_complete IS NOT TRUE
            )

        ORDER BY
            state_time,
            underlying

        LIMIT 50
        """
    ).fetchall()

    print(
        "\nFIRST 50 FUTURES/L2 QUALITY FAILURES"
    )

    print(
        "=" * 110
    )

    if not rows:

        print(
            "No futures/L2 quality failures."
        )

        return

    for row in rows:

        print(
            row
        )


# --------------------------------------------------
# Poor option examples
# --------------------------------------------------

def print_poor_option_examples(
    connection
) -> None:

    rows = connection.execute(
        """
        SELECT
            state_time,
            underlying,
            expiry_date,

            option_contract_count,
            option_ticker_count,
            option_coverage_pct,

            capture_delay_ms,
            median_quote_age_ms,
            max_quote_age_ms

        FROM canonical_state_index

        WHERE
            has_options

            AND (
                option_coverage_pct < 99

                OR

                capture_delay_ms > 2000

                OR

                max_quote_age_ms > 30000
            )

        ORDER BY
            state_time,
            underlying

        LIMIT 50
        """
    ).fetchall()

    print(
        "\nFIRST 50 OPTION QUALITY OUTLIERS"
    )

    print(
        "=" * 130
    )

    if not rows:

        print(
            "No option quality outliers "
            "under diagnostic thresholds."
        )

        return

    for row in rows:

        print(
            row
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

        create_state_index(
            connection
        )

        print_futures_quality(
            connection
        )

        print_option_quality(
            connection
        )

        print_option_coverage_distribution(
            connection
        )

        print_bad_futures_examples(
            connection
        )

        print_poor_option_examples(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()