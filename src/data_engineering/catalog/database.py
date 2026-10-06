from __future__ import annotations

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)

load_dotenv(
    dotenv_path=ENV_FILE
)


# --------------------------------------------------
# Data root
# --------------------------------------------------

data_root_env = os.getenv(
    "DERIBIT_DATA_ROOT"
)

if not data_root_env:

    raise RuntimeError(
        "DERIBIT_DATA_ROOT is not defined "
        "in the project .env file."
    )

DATA_ROOT = (
    Path(data_root_env)
    .expanduser()
    .resolve()
)


# --------------------------------------------------
# Research database
# --------------------------------------------------

DATABASE_ROOT = (
    DATA_ROOT
    / "database"
)

RESEARCH_DATABASE = (
    DATABASE_ROOT
    / "research.duckdb"
)


# --------------------------------------------------
# Source datasets
# --------------------------------------------------

FUTURES_ROOT = (
    DATA_ROOT
    / "futures"
    / "5m"
)

OPTION_GRID_ROOT = (
    DATA_ROOT
    / "daily"
    / "option_grid"
)

# Next daily expiry (1DTE), collected from 6 Oct 2026 by
# option_1dte_collector.py. Optional: absent before that date.
OPTION_GRID_1DTE_ROOT = (
    DATA_ROOT
    / "daily"
    / "option_grid_1dte"
)


# --------------------------------------------------
# Database connection
# --------------------------------------------------

def open_research_database(
) -> duckdb.DuckDBPyConnection:

    DATABASE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = duckdb.connect(
        str(RESEARCH_DATABASE)
    )

    connection.execute(
        "SET TimeZone='UTC'"
    )

    return connection

# --------------------------------------------------
# Metadata
# --------------------------------------------------

def initialize_metadata(
    connection: duckdb.DuckDBPyConnection
) -> None:

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
        catalog_metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
        )
        """
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO
        catalog_metadata
        VALUES (
            'database_type',
            'deribit_research'
        )
        """
    )


# --------------------------------------------------
# Raw Parquet views
# --------------------------------------------------

def refresh_raw_views(
    connection: duckdb.DuckDBPyConnection
) -> None:

    futures_files = list(
        FUTURES_ROOT.glob(
            "date=*/futures_5m.parquet"
        )
    )

    option_files = list(
        OPTION_GRID_ROOT.glob(
            "expiry=*/option_grid.parquet"
        )
    )

    if not futures_files:

        raise RuntimeError(
            "No futures Parquet files found under:\n"
            f"{FUTURES_ROOT}"
        )

    if not option_files:

        raise RuntimeError(
            "No completed option-grid Parquet files "
            "found under:\n"
            f"{OPTION_GRID_ROOT}"
        )

    futures_glob = (
        FUTURES_ROOT
        / "date=*"
        / "futures_5m.parquet"
    ).as_posix()

    options_glob = (
        OPTION_GRID_ROOT
        / "expiry=*"
        / "option_grid.parquet"
    ).as_posix()

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW
        raw_futures_5m AS

        SELECT *
        FROM read_parquet(
            '{futures_glob}',
            hive_partitioning = true,
            union_by_name = true
        )
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW
        raw_option_grid AS

        SELECT *
        FROM read_parquet(
            '{options_glob}',
            hive_partitioning = true,
            union_by_name = true
        )
        """
    )

    option_1dte_files = list(
        OPTION_GRID_1DTE_ROOT.glob(
            "expiry=*/option_grid.parquet"
        )
    )

    if option_1dte_files:

        options_1dte_glob = (
            OPTION_GRID_1DTE_ROOT
            / "expiry=*"
            / "option_grid.parquet"
        ).as_posix()

        connection.execute(
            f"""
            CREATE OR REPLACE VIEW
            raw_option_grid_1dte AS

            SELECT *
            FROM read_parquet(
                '{options_1dte_glob}',
                hive_partitioning = true,
                union_by_name = true
            )
            """
        )

    print(
        f"Futures files found: "
        f"{len(futures_files)}"
    )

    print(
        f"1DTE option window files found: "
        f"{len(option_1dte_files)}"
    )

    print(
        f"Option session files found: "
        f"{len(option_files)}"
    )


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

def print_catalog_summary(
    connection: duckdb.DuckDBPyConnection
) -> None:

    futures = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            MIN(bar_end) AS first_bar,
            MAX(bar_end) AS last_bar,
            COUNT(DISTINCT underlying)
                AS underlyings
        FROM raw_futures_5m
        """
    ).fetchone()

    options = connection.execute(
        """
        SELECT
            COUNT(*) AS rows,
            MIN(scheduled_snapshot_time)
                AS first_snapshot,
            MAX(scheduled_snapshot_time)
                AS last_snapshot,
            COUNT(DISTINCT underlying)
                AS underlyings,
            COUNT(DISTINCT expiry_date)
                AS expiries
        FROM raw_option_grid
        """
    ).fetchone()

    print(
        "\nRESEARCH CATALOG"
    )

    print(
        "=" * 60
    )

    print(
        f"Database:\n"
        f"{RESEARCH_DATABASE}"
    )

    print(
        "\nFUTURES"
    )

    print(
        f"Rows:        {futures[0]:,}"
    )

    print(
        f"First bar:   {futures[1]}"
    )

    print(
        f"Last bar:    {futures[2]}"
    )

    print(
        f"Underlyings: {futures[3]}"
    )

    print(
        "\nOPTIONS"
    )

    print(
        f"Rows:        {options[0]:,}"
    )

    print(
        f"First snap:  {options[1]}"
    )

    print(
        f"Last snap:   {options[2]}"
    )

    print(
        f"Underlyings: {options[3]}"
    )

    print(
        f"Expiries:    {options[4]}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    connection = (
        open_research_database()
    )

    try:

        initialize_metadata(
            connection
        )

        refresh_raw_views(
            connection
        )

        print_catalog_summary(
            connection
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()