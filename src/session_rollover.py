from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os

import duckdb
from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

data_root_env = os.getenv("DERIBIT_DATA_ROOT")

if not data_root_env:
    raise RuntimeError(
        "DERIBIT_DATA_ROOT is not defined in the .env file."
    )

DATA_ROOT = Path(
    data_root_env
).expanduser().resolve()

RAW_SNAPSHOT_ROOT = (
    DATA_ROOT
    / "raw"
    / "option_snapshots"
)

DAILY_ROOT = (
    DATA_ROOT
    / "daily"
    / "option_grid"
)

DATABASE_ROOT = (
    DATA_ROOT
    / "database"
)

DATABASE_FILE = (
    DATABASE_ROOT
    / "deribit.duckdb"
)

EXPECTED_SNAPSHOTS = 288


# --------------------------------------------------
# Database
# --------------------------------------------------

def initialize_database(
    connection: duckdb.DuckDBPyConnection
) -> None:

    connection.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            expiry_date DATE PRIMARY KEY,
            session_id VARCHAR,
            session_start TIMESTAMPTZ,
            session_end TIMESTAMPTZ,

            snapshot_count INTEGER,
            expected_snapshots INTEGER,
            missing_snapshots INTEGER,
            is_complete BOOLEAN,

            contract_rows BIGINT,
            unique_instruments INTEGER,

            source_file_count INTEGER,
            parquet_path VARCHAR,

            created_at TIMESTAMPTZ
        )
    """)


def refresh_option_grid_view(
    connection: duckdb.DuckDBPyConnection
) -> None:

    parquet_glob = (
        DAILY_ROOT
        / "**"
        / "*.parquet"
    ).as_posix()

    connection.execute(f"""
        CREATE OR REPLACE VIEW option_grid AS

        SELECT *
        FROM read_parquet(
            '{parquet_glob}',
            hive_partitioning = true,
            union_by_name = true
        )
    """)


# --------------------------------------------------
# Session paths
# --------------------------------------------------

def get_session_folder(
    expiry_date: str
) -> Path:

    return (
        RAW_SNAPSHOT_ROOT
        / f"expiry={expiry_date}"
    )


def get_daily_folder(
    expiry_date: str
) -> Path:

    return (
        DAILY_ROOT
        / f"expiry={expiry_date}"
    )


# --------------------------------------------------
# Validation
# --------------------------------------------------

def inspect_session(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str
) -> dict:

    result = connection.execute(f"""
        SELECT
            COUNT(*) AS total_rows,

            COUNT(
                DISTINCT scheduled_snapshot_time
            ) AS snapshot_count,

            COUNT(
                DISTINCT instrument_name
            ) AS unique_instruments,

            MIN(
                scheduled_snapshot_time
            ) AS first_snapshot,

            MAX(
                scheduled_snapshot_time
            ) AS last_snapshot

        FROM read_parquet(
            '{source_glob}',
            union_by_name = true
        )
    """).fetchone()

    return {
        "total_rows": result[0],
        "snapshot_count": result[1],
        "unique_instruments": result[2],
        "first_snapshot": result[3],
        "last_snapshot": result[4]
    }


def count_duplicate_rows(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str
) -> int:

    result = connection.execute(f"""
        SELECT COUNT(*)

        FROM (
            SELECT
                scheduled_snapshot_time,
                instrument_name,
                COUNT(*) AS n

            FROM read_parquet(
                '{source_glob}',
                union_by_name = true
            )

            GROUP BY
                scheduled_snapshot_time,
                instrument_name

            HAVING COUNT(*) > 1
        )
    """).fetchone()

    return result[0]


# --------------------------------------------------
# Daily compaction
# --------------------------------------------------

def compact_session(
    connection: duckdb.DuckDBPyConnection,
    expiry_date: str
) -> tuple[Path, dict]:

    session_folder = get_session_folder(
        expiry_date
    )

    if not session_folder.exists():

        raise RuntimeError(
            f"Session folder does not exist:\n"
            f"{session_folder}"
        )

    files = sorted(
        session_folder.glob(
            "snapshot_*.parquet"
        )
    )

    if not files:

        raise RuntimeError(
            f"No production snapshots found in:\n"
            f"{session_folder}"
        )

    source_glob = (
        session_folder
        / "snapshot_*.parquet"
    ).as_posix()

    print(
        f"\nFound {len(files)} "
        f"snapshot files."
    )

    diagnostics = inspect_session(
        connection,
        source_glob
    )

    duplicates = count_duplicate_rows(
        connection,
        source_glob
    )

    diagnostics[
        "duplicate_groups"
    ] = duplicates

    daily_folder = get_daily_folder(
        expiry_date
    )

    daily_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    final_file = (
        daily_folder
        / "option_grid.parquet"
    )

    temporary_file = (
        daily_folder
        / ".option_grid.tmp.parquet"
    )

    if temporary_file.exists():
        temporary_file.unlink()

    # --------------------------------------------------
    # Merge + deduplicate + sort using DuckDB itself.
    # --------------------------------------------------

    connection.execute(f"""
        COPY (

            SELECT *
            EXCLUDE (duplicate_rank)

            FROM (

                SELECT
                    *,

                    ROW_NUMBER() OVER (
                        PARTITION BY
                            scheduled_snapshot_time,
                            instrument_name

                        ORDER BY
                            snapshot_time DESC
                    ) AS duplicate_rank

                FROM read_parquet(
                    '{source_glob}',
                    union_by_name = true
                )
            )

            WHERE duplicate_rank = 1

            ORDER BY
                scheduled_snapshot_time,
                underlying,
                product_group,
                strike,
                option_type

        )

        TO '{temporary_file.as_posix()}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
    """)

    if not temporary_file.exists():

        raise RuntimeError(
            "DuckDB did not create "
            "the temporary daily Parquet."
        )

    # --------------------------------------------------
    # Verify newly-created daily file.
    # --------------------------------------------------

    final_validation = connection.execute(f"""
        SELECT
            COUNT(*) AS rows,

            COUNT(
                DISTINCT scheduled_snapshot_time
            ) AS snapshots,

            COUNT(
                DISTINCT instrument_name
            ) AS instruments

        FROM read_parquet(
            '{temporary_file.as_posix()}'
        )
    """).fetchone()

    diagnostics[
        "final_rows"
    ] = final_validation[0]

    diagnostics[
        "final_snapshot_count"
    ] = final_validation[1]

    diagnostics[
        "final_unique_instruments"
    ] = final_validation[2]

    # Atomic finalization.
    temporary_file.replace(
        final_file
    )

    return final_file, diagnostics


# --------------------------------------------------
# Register completed session
# --------------------------------------------------

def register_session(
    connection: duckdb.DuckDBPyConnection,
    expiry_date: str,
    daily_file: Path,
    diagnostics: dict,
    source_file_count: int
) -> None:

    expiry = datetime.strptime(
        expiry_date,
        "%Y-%m-%d"
    ).replace(
        hour=8,
        tzinfo=timezone.utc
    )

    session_start = (
        expiry.replace()
    )

    from datetime import timedelta

    session_start = (
        expiry
        - timedelta(days=1)
    )

    snapshot_count = diagnostics[
        "final_snapshot_count"
    ]

    missing_snapshots = max(
        0,
        EXPECTED_SNAPSHOTS
        - snapshot_count
    )

    is_complete = (
        snapshot_count
        == EXPECTED_SNAPSHOTS
    )

    # Makes rerunning rollover idempotent.
    connection.execute(
        """
        DELETE FROM sessions
        WHERE expiry_date = ?
        """,
        [expiry_date]
    )

    connection.execute(
        """
        INSERT INTO sessions (
            expiry_date,
            session_id,
            session_start,
            session_end,

            snapshot_count,
            expected_snapshots,
            missing_snapshots,
            is_complete,

            contract_rows,
            unique_instruments,

            source_file_count,
            parquet_path,

            created_at
        )

        VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?
        )
        """,
        [
            expiry_date,
            f"expiry={expiry_date}",
            session_start,
            expiry,

            snapshot_count,
            EXPECTED_SNAPSHOTS,
            missing_snapshots,
            is_complete,

            diagnostics["final_rows"],
            diagnostics[
                "final_unique_instruments"
            ],

            source_file_count,
            str(daily_file),

            datetime.now(
                timezone.utc
            )
        ]
    )


# --------------------------------------------------
# Console summary
# --------------------------------------------------

def print_rollover_summary(
    expiry_date: str,
    daily_file: Path,
    diagnostics: dict
) -> None:

    snapshot_count = diagnostics[
        "final_snapshot_count"
    ]

    print(
        "\nSESSION FINALIZED"
    )

    print(
        "=" * 60
    )

    print(
        f"Expiry:              "
        f"{expiry_date}"
    )

    print(
        f"Snapshots:           "
        f"{snapshot_count}/"
        f"{EXPECTED_SNAPSHOTS}"
    )

    print(
        f"Missing snapshots:   "
        f"{max(0, EXPECTED_SNAPSHOTS - snapshot_count)}"
    )

    print(
        f"Rows before dedup:   "
        f"{diagnostics['total_rows']:,}"
    )

    print(
        f"Rows final:          "
        f"{diagnostics['final_rows']:,}"
    )

    print(
        f"Unique instruments:  "
        f"{diagnostics['final_unique_instruments']:,}"
    )

    print(
        f"Duplicate groups:    "
        f"{diagnostics['duplicate_groups']:,}"
    )

    print(
        f"First snapshot:      "
        f"{diagnostics['first_snapshot']}"
    )

    print(
        f"Last snapshot:       "
        f"{diagnostics['last_snapshot']}"
    )

    print(
        "\nDaily Parquet:"
    )

    print(
        daily_file
    )


# --------------------------------------------------
# Main rollover function
# --------------------------------------------------

def finalize_session(
    expiry_date: str
) -> Path:

    DATABASE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = duckdb.connect(
        str(DATABASE_FILE)
    )

    try:

        initialize_database(
            connection
        )

        session_folder = (
            get_session_folder(
                expiry_date
            )
        )

        source_files = list(
            session_folder.glob(
                "snapshot_*.parquet"
            )
        )

        daily_file, diagnostics = (
            compact_session(
                connection,
                expiry_date
            )
        )

        register_session(
            connection=connection,
            expiry_date=expiry_date,
            daily_file=daily_file,
            diagnostics=diagnostics,
            source_file_count=len(
                source_files
            )
        )

        refresh_option_grid_view(
            connection
        )

        print_rollover_summary(
            expiry_date,
            daily_file,
            diagnostics
        )

        return daily_file

    finally:

        connection.close()


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Finalize a Deribit 0DTE session"
        )
    )

    parser.add_argument(
        "expiry_date",
        help=(
            "Expiry date in YYYY-MM-DD format"
        )
    )

    args = parser.parse_args()

    finalize_session(
        args.expiry_date
    )