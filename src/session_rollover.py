from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

SNAPSHOT_INTERVAL = timedelta(
    minutes=5
)

SETTLEMENT_HOUR_UTC = 8


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


def session_is_registered(
    connection: duckdb.DuckDBPyConnection,
    expiry_date: str
) -> bool:

    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM sessions
        WHERE expiry_date = ?
        """,
        [
            expiry_date
        ]
    ).fetchone()

    return result[0] > 0


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
# Session-date helpers
# --------------------------------------------------

def expiry_datetime(
    expiry_date: str
) -> datetime:

    return datetime.strptime(
        expiry_date,
        "%Y-%m-%d"
    ).replace(
        hour=SETTLEMENT_HOUR_UTC,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=timezone.utc
    )


def session_start_datetime(
    expiry_date: str
) -> datetime:

    return (
        expiry_datetime(
            expiry_date
        )
        - timedelta(
            days=1
        )
    )


def parse_expiry_folder(
    folder: Path
) -> str | None:

    name = folder.name

    if not name.startswith(
        "expiry="
    ):

        return None

    expiry_date = name.replace(
        "expiry=",
        ""
    )

    try:

        datetime.strptime(
            expiry_date,
            "%Y-%m-%d"
        )

    except ValueError:

        return None

    return expiry_date


# --------------------------------------------------
# Exact expected timestamp grid
# --------------------------------------------------

def expected_timestamp_grid(
    expiry_date: str
) -> set[datetime]:

    start = (
        session_start_datetime(
            expiry_date
        )
    )

    expiry = (
        expiry_datetime(
            expiry_date
        )
    )

    result = set()

    timestamp = start

    while timestamp < expiry:

        result.add(
            timestamp
        )

        timestamp += (
            SNAPSHOT_INTERVAL
        )

    if (
        len(result)
        != EXPECTED_SNAPSHOTS
    ):

        raise RuntimeError(
            "Internal expected-grid "
            f"error: {len(result)} "
            "timestamps."
        )

    return result


def normalize_utc_datetime(
    value
) -> datetime:

    if isinstance(
        value,
        datetime
    ):

        result = value

    else:

        result = (
            datetime.fromisoformat(
                str(value)
            )
        )

    if result.tzinfo is None:

        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def inspect_timestamp_grid(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
    expiry_date: str
) -> dict:

    rows = connection.execute(f"""
        SELECT DISTINCT
            scheduled_snapshot_time

        FROM read_parquet(
            '{source_glob}',
            union_by_name = true
        )

        ORDER BY
            scheduled_snapshot_time
    """).fetchall()

    actual = {
        normalize_utc_datetime(
            row[0]
        )
        for row in rows
    }

    expected = (
        expected_timestamp_grid(
            expiry_date
        )
    )

    missing = sorted(
        expected
        - actual
    )

    unexpected = sorted(
        actual
        - expected
    )

    return {
        "missing_expected_timestamps":
            missing,

        "unexpected_timestamps":
            unexpected,

        "exact_grid_complete":
            (
                len(missing) == 0
                and len(unexpected) == 0
                and len(actual)
                == EXPECTED_SNAPSHOTS
            )
    }


# --------------------------------------------------
# Automatic session discovery
# --------------------------------------------------

def find_latest_completed_session(
    connection: duckdb.DuckDBPyConnection
) -> str:

    now = datetime.now(
        timezone.utc
    )

    candidates = []

    for folder in RAW_SNAPSHOT_ROOT.glob(
        "expiry=*"
    ):

        if not folder.is_dir():
            continue

        expiry_date = parse_expiry_folder(
            folder
        )

        if expiry_date is None:
            continue

        expiry_time = expiry_datetime(
            expiry_date
        )

        if expiry_time > now:
            continue

        snapshot_files = list(
            folder.glob(
                "snapshot_*.parquet"
            )
        )

        if not snapshot_files:
            continue

        if session_is_registered(
            connection,
            expiry_date
        ):

            continue

        candidates.append(
            (
                expiry_time,
                expiry_date
            )
        )

    if not candidates:

        raise RuntimeError(
            "No completed, unregistered "
            "0DTE sessions were found."
        )

    candidates.sort()

    return candidates[
        -1
    ][1]


# --------------------------------------------------
# Paths
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
# Session validation
# --------------------------------------------------

def validate_completed_expiry(
    expiry_date: str
) -> None:

    expiry_time = (
        expiry_datetime(
            expiry_date
        )
    )

    now = datetime.now(
        timezone.utc
    )

    if expiry_time > now:

        raise RuntimeError(
            "\nRefusing to finalize "
            "an active or future session.\n"
            f"Requested expiry: "
            f"{expiry_time.isoformat()}\n"
            f"Current UTC time: "
            f"{now.isoformat()}"
        )


def inspect_session(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str
) -> dict:

    result = connection.execute(f"""
        SELECT
            COUNT(*) AS total_rows,

            COUNT(
                DISTINCT
                scheduled_snapshot_time
            ) AS snapshot_count,

            COUNT(
                DISTINCT
                instrument_name
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
        "total_rows":
            result[0],

        "snapshot_count":
            result[1],

        "unique_instruments":
            result[2],

        "first_snapshot":
            result[3],

        "last_snapshot":
            result[4]
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

            HAVING
                COUNT(*) > 1
        )
    """).fetchone()

    return result[0]


# --------------------------------------------------
# Daily compaction
# --------------------------------------------------

def compact_session(
    connection: duckdb.DuckDBPyConnection,
    expiry_date: str
) -> tuple[
    Path,
    dict
]:

    session_folder = (
        get_session_folder(
            expiry_date
        )
    )

    if not session_folder.exists():

        raise RuntimeError(
            "Session folder does not "
            "exist:\n"
            f"{session_folder}"
        )

    files = sorted(
        session_folder.glob(
            "snapshot_*.parquet"
        )
    )

    if not files:

        raise RuntimeError(
            "No production snapshots "
            "found in:\n"
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

    diagnostics = (
        inspect_session(
            connection,
            source_glob
        )
    )

    diagnostics[
        "duplicate_groups"
    ] = (
        count_duplicate_rows(
            connection,
            source_glob
        )
    )

    diagnostics.update(
        inspect_timestamp_grid(
            connection,
            source_glob,
            expiry_date
        )
    )

    daily_folder = (
        get_daily_folder(
            expiry_date
        )
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

    connection.execute(f"""
        COPY (

            SELECT *
            EXCLUDE (
                duplicate_rank
            )

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

            WHERE
                duplicate_rank = 1

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
            "Temporary daily Parquet "
            "was not created."
        )

    validation = connection.execute(f"""
        SELECT
            COUNT(*),

            COUNT(
                DISTINCT
                scheduled_snapshot_time
            ),

            COUNT(
                DISTINCT
                instrument_name
            )

        FROM read_parquet(
            '{temporary_file.as_posix()}'
        )
    """).fetchone()

    diagnostics[
        "final_rows"
    ] = validation[0]

    diagnostics[
        "final_snapshot_count"
    ] = validation[1]

    diagnostics[
        "final_unique_instruments"
    ] = validation[2]

    temporary_file.replace(
        final_file
    )

    return (
        final_file,
        diagnostics
    )


# --------------------------------------------------
# Session registration
# --------------------------------------------------

def register_session(
    connection: duckdb.DuckDBPyConnection,
    expiry_date: str,
    daily_file: Path,
    diagnostics: dict,
    source_file_count: int
) -> None:

    expiry = (
        expiry_datetime(
            expiry_date
        )
    )

    session_start = (
        expiry
        - timedelta(
            days=1
        )
    )

    snapshot_count = (
        diagnostics[
            "final_snapshot_count"
        ]
    )

    missing_snapshots = len(
        diagnostics[
            "missing_expected_timestamps"
        ]
    )

    is_complete = (
        diagnostics[
            "exact_grid_complete"
        ]
        and snapshot_count
        == EXPECTED_SNAPSHOTS
    )

    connection.execute(
        """
        DELETE FROM sessions
        WHERE expiry_date = ?
        """,
        [
            expiry_date
        ]
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
            ?, ?,
            ?, ?,
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

            diagnostics[
                "final_rows"
            ],

            diagnostics[
                "final_unique_instruments"
            ],

            source_file_count,
            str(
                daily_file
            ),

            datetime.now(
                timezone.utc
            )
        ]
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

def print_rollover_summary(
    expiry_date: str,
    daily_file: Path,
    diagnostics: dict
) -> None:

    missing = (
        diagnostics[
            "missing_expected_timestamps"
        ]
    )

    unexpected = (
        diagnostics[
            "unexpected_timestamps"
        ]
    )

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
        f"{diagnostics['final_snapshot_count']}/"
        f"{EXPECTED_SNAPSHOTS}"
    )

    print(
        f"Missing grid times:  "
        f"{len(missing)}"
    )

    print(
        f"Unexpected times:    "
        f"{len(unexpected)}"
    )

    print(
        f"Exact grid complete: "
        f"{diagnostics['exact_grid_complete']}"
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

    if missing:

        print(
            "\nMissing expected timestamps:"
        )

        for timestamp in missing[
            :20
        ]:

            print(
                f"  {timestamp.isoformat()}"
            )

        if len(missing) > 20:

            print(
                f"  ... and "
                f"{len(missing) - 20} "
                f"more"
            )

    if unexpected:

        print(
            "\nUnexpected timestamps:"
        )

        for timestamp in unexpected[
            :20
        ]:

            print(
                f"  {timestamp.isoformat()}"
            )

        if len(unexpected) > 20:

            print(
                f"  ... and "
                f"{len(unexpected) - 20} "
                f"more"
            )

    print(
        "\nDaily Parquet:"
    )

    print(
        daily_file
    )


# --------------------------------------------------
# Main finalization
# --------------------------------------------------

def finalize_session(
    connection: duckdb.DuckDBPyConnection,
    expiry_date: str,
    force: bool = False
) -> Path:

    validate_completed_expiry(
        expiry_date
    )

    if (
        session_is_registered(
            connection,
            expiry_date
        )
        and not force
    ):

        raise RuntimeError(
            "\nSession is already "
            "registered in DuckDB:\n"
            f"{expiry_date}\n\n"
            "Use --force only if you "
            "intentionally want to "
            "rebuild it."
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


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Finalize completed "
            "Deribit 0DTE sessions."
        )
    )

    parser.add_argument(
        "expiry_date",
        nargs="?",
        default=None,
        help=(
            "Optional expiry date in "
            "YYYY-MM-DD format. "
            "If omitted, the latest "
            "completed unregistered "
            "session is selected."
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow deliberate rebuilding "
            "of an already-registered "
            "session."
        )
    )

    args = parser.parse_args()

    DATABASE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = duckdb.connect(
        str(
            DATABASE_FILE
        )
    )

    try:

        initialize_database(
            connection
        )

        expiry_date = (
            args.expiry_date
        )

        if expiry_date is None:

            expiry_date = (
                find_latest_completed_session(
                    connection
                )
            )

            print(
                "\nAutomatically selected "
                f"completed session: "
                f"{expiry_date}"
            )

        else:

            print(
                "\nManually selected session: "
                f"{expiry_date}"
            )

        finalize_session(
            connection=connection,
            expiry_date=expiry_date,
            force=args.force
        )

    finally:

        connection.close()


if __name__ == "__main__":

    main()
