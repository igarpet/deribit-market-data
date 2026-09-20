from __future__ import annotations

import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from session_manager import calculate_session_window
from session_rollover import (
    DATABASE_FILE,
    DATABASE_ROOT,
    RAW_SNAPSHOT_ROOT,
    expiry_datetime,
    initialize_database,
    parse_expiry_folder,
    session_is_registered,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

SRC_ROOT = (
    PROJECT_ROOT
    / "src"
)

RESTART_DELAY_SECONDS = 5

# Must match the collector's maximum permitted
# delay for an opening 08:00 snapshot.
OPENING_SNAPSHOT_WINDOW_SECONDS = 120


# --------------------------------------------------
# Global process state
# --------------------------------------------------

stop_requested = False

current_process = None


# --------------------------------------------------
# Shutdown handling
# --------------------------------------------------

def request_shutdown(
    signum,
    frame
):

    global stop_requested
    global current_process

    print(
        f"\nOrchestrator received "
        f"shutdown signal ({signum})."
    )

    stop_requested = True

    if (
        current_process is not None
        and current_process.poll() is None
    ):

        print(
            "Stopping child process..."
        )

        try:

            current_process.terminate()

        except Exception as exc:

            print(
                f"Could not terminate "
                f"child process cleanly: "
                f"{exc}"
            )


def configure_signals():

    signal.signal(
        signal.SIGINT,
        request_shutdown
    )

    if hasattr(
        signal,
        "SIGTERM"
    ):

        signal.signal(
            signal.SIGTERM,
            request_shutdown
        )

    if hasattr(
        signal,
        "SIGBREAK"
    ):

        signal.signal(
            signal.SIGBREAK,
            request_shutdown
        )


# --------------------------------------------------
# Generic project-script runner
# --------------------------------------------------

def run_script(
    script_name: str,
    arguments: list[str] | None = None
) -> int:

    global current_process

    if arguments is None:
        arguments = []

    script = (
        SRC_ROOT
        / script_name
    )

    command = [
        sys.executable,
        str(script),
        *arguments
    ]

    print(
        "\nRunning:"
    )

    print(
        " ".join(command)
    )

    current_process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT
    )

    return_code = (
        current_process.wait()
    )

    current_process = None

    return return_code


# --------------------------------------------------
# Instrument master
# --------------------------------------------------

def refresh_instruments() -> bool:

    global stop_requested

    if stop_requested:
        return False

    print(
        "\nRefreshing Deribit "
        "instrument master..."
    )

    return_code = run_script(
        "instrument_master.py"
    )

    if stop_requested:
        return False

    if return_code != 0:

        print(
            "\nInstrument master "
            "refresh failed."
        )

        return False

    print(
        "Instrument master refreshed."
    )

    return True


# --------------------------------------------------
# DuckDB helpers
# --------------------------------------------------

def open_database():

    DATABASE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = duckdb.connect(
        str(DATABASE_FILE)
    )

    initialize_database(
        connection
    )

    return connection


def is_session_registered(
    expiry_date: str
) -> bool:

    connection = open_database()

    try:

        return session_is_registered(
            connection,
            expiry_date
        )

    finally:

        connection.close()


# --------------------------------------------------
# Find completed sessions requiring recovery
# --------------------------------------------------

def find_completed_unregistered_sessions() -> list[str]:
    """
    Return all raw sessions that:

        1. contain production snapshots,
        2. have already expired,
        3. are not registered in DuckDB.

    Results are returned oldest-first.

    This allows recovery after the VM has been offline
    across one or more 08:00 UTC boundaries.
    """

    now = datetime.now(
        timezone.utc
    )

    if not RAW_SNAPSHOT_ROOT.exists():

        return []

    connection = open_database()

    try:

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

            # The current live session must never
            # be finalized.
            if expiry_time > now:
                continue

            # There must actually be production data.
            snapshot_files = list(
                folder.glob(
                    "snapshot_*.parquet"
                )
            )

            if not snapshot_files:
                continue

            # Ignore sessions already successfully
            # registered in DuckDB.
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

        # Oldest first.
        candidates.sort(
            key=lambda item: item[0]
        )

        return [
            expiry_date
            for _, expiry_date
            in candidates
        ]

    finally:

        connection.close()


# --------------------------------------------------
# Session rollover
# --------------------------------------------------

def rollover_session(
    expiry_date: str
) -> bool:
    """
    Finalize one specific expired session.

    Using the explicit expiry date prevents the
    rollover script from accidentally selecting
    another session.
    """

    global stop_requested

    if stop_requested:
        return False

    # Idempotency:
    # if this session has already been finalized,
    # treat that as success.
    if is_session_registered(
        expiry_date
    ):

        print(
            f"\nSession {expiry_date} "
            f"is already registered."
        )

        print(
            "Skipping duplicate rollover."
        )

        return True

    print(
        f"\nFinalizing expired session: "
        f"{expiry_date}"
    )

    return_code = run_script(
        "session_rollover.py",
        [
            expiry_date
        ]
    )

    if stop_requested:
        return False

    if return_code != 0:

        print(
            f"\nSession rollover failed "
            f"for {expiry_date}."
        )

        return False

    print(
        f"Session {expiry_date} "
        f"finalized successfully."
    )

    return True


# --------------------------------------------------
# Startup recovery
# --------------------------------------------------

def catch_up_completed_sessions() -> bool:
    """
    Recover any sessions that expired while the
    orchestrator/VM was offline.

    The scan is repeated after each pass. This handles
    the edge case where the clock crosses 08:00 UTC
    while startup recovery itself is running.
    """

    global stop_requested

    print(
        "\nSTARTUP SESSION RECOVERY"
    )

    print(
        "=" * 60
    )

    recovered_count = 0

    while not stop_requested:

        sessions = (
            find_completed_unregistered_sessions()
        )

        if not sessions:

            if recovered_count == 0:

                print(
                    "No completed unregistered "
                    "sessions found."
                )

            else:

                print(
                    f"\nStartup recovery complete."
                )

                print(
                    f"Recovered sessions: "
                    f"{recovered_count}"
                )

            return True

        print(
            f"\nFound {len(sessions)} "
            f"completed unregistered "
            f"session(s):"
        )

        for expiry_date in sessions:

            print(
                f"  {expiry_date}"
            )

        for expiry_date in sessions:

            if stop_requested:
                return False

            success = rollover_session(
                expiry_date
            )

            if not success:

                print(
                    "\nStartup recovery stopped "
                    "because a session could not "
                    "be finalized."
                )

                return False

            recovered_count += 1

        # Deliberately loop and scan again.
        #
        # Example:
        # startup begins at 07:59:59 and recovery
        # crosses 08:00. The session that just expired
        # will be detected on this second pass.

    return False


# --------------------------------------------------
# Opening-snapshot decision
# --------------------------------------------------

def should_request_opening_snapshot(
    session
) -> bool:
    """
    If the orchestrator starts shortly after 08:00,
    allow the collector to create the opening snapshot.

    The collector itself performs the final safety
    check before writing it.
    """

    now = datetime.now(
        timezone.utc
    )

    delay = (
        now
        - session.session_start
    ).total_seconds()

    return (
        0
        <= delay
        <= OPENING_SNAPSHOT_WINDOW_SECONDS
    )


# --------------------------------------------------
# Collector
# --------------------------------------------------

def run_collector(
    opening_snapshot: bool
) -> int:

    arguments = []

    if opening_snapshot:

        arguments.append(
            "--opening-snapshot"
        )

        print(
            "\nOpening snapshot requested."
        )

    return run_script(
        "option_grid_collector.py",
        arguments
    )


# --------------------------------------------------
# Main orchestration loop
# --------------------------------------------------

def main():

    global stop_requested

    configure_signals()

    print(
        "\nDERIBIT 0DTE SESSION ORCHESTRATOR"
    )

    print(
        "=" * 60
    )

    print(
        f"Started at: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )

    # --------------------------------------------------
    # STEP 1
    # Recover any completed sessions left behind
    # by a VM outage/reboot.
    # --------------------------------------------------

    recovery_ok = (
        catch_up_completed_sessions()
    )

    if stop_requested:
        return

    if not recovery_ok:

        raise RuntimeError(
            "Startup session recovery failed. "
            "Stopping to avoid silently "
            "skipping historical data."
        )

    # --------------------------------------------------
    # STEP 2
    # Refresh instruments for the CURRENT session.
    # --------------------------------------------------

    if not refresh_instruments():

        if stop_requested:
            return

        raise RuntimeError(
            "Initial instrument master "
            "refresh failed."
        )

    # First launch can request an opening snapshot
    # if systemd happened to start very close to 08:00.
    first_collector_launch = True

    opening_snapshot = False

    # --------------------------------------------------
    # Main lifecycle
    # --------------------------------------------------

    while not stop_requested:

        session = (
            calculate_session_window()
        )

        if first_collector_launch:

            opening_snapshot = (
                should_request_opening_snapshot(
                    session
                )
            )

            first_collector_launch = False

        print(
            "\nACTIVE SESSION"
        )

        print(
            "=" * 60
        )

        print(
            f"Session ID: "
            f"{session.session_id}"
        )

        print(
            f"Start:      "
            f"{session.session_start.isoformat()}"
        )

        print(
            f"Expiry:     "
            f"{session.session_expiry.isoformat()}"
        )

        collector_return_code = (
            run_collector(
                opening_snapshot=opening_snapshot
            )
        )

        # Opening mode is only used once.
        # An unexpected restart during the same session
        # must not create another opening snapshot.
        opening_snapshot = False

        if stop_requested:
            break

        now = datetime.now(
            timezone.utc
        )

        # --------------------------------------------------
        # Normal 08:00 session completion
        # --------------------------------------------------

        if (
            now
            >= session.session_expiry
        ):

            print(
                "\nSession expiry reached."
            )

            print(
                f"Expired session: "
                f"{session.expiry_date}"
            )

            rollover_ok = (
                rollover_session(
                    session.expiry_date
                )
            )

            if not rollover_ok:

                raise RuntimeError(
                    "Session rollover failed. "
                    "Stopping orchestrator to "
                    "avoid silently losing data."
                )

            if stop_requested:
                break

            # Get the latest instrument universe
            # before starting the new session.
            if not refresh_instruments():

                if stop_requested:
                    break

                raise RuntimeError(
                    "Instrument master refresh "
                    "failed after rollover."
                )

            # The next collector belongs to the
            # newly-created 08:00 session.
            opening_snapshot = True

            continue

        # --------------------------------------------------
        # Unexpected collector exit
        # --------------------------------------------------

        print(
            "\nCollector exited before "
            "session expiry."
        )

        print(
            f"Return code: "
            f"{collector_return_code}"
        )

        print(
            f"Restarting in "
            f"{RESTART_DELAY_SECONDS} "
            f"seconds..."
        )

        # Do not create an artificial opening snapshot
        # after an intraday collector restart.
        opening_snapshot = False

        for _ in range(
            RESTART_DELAY_SECONDS
        ):

            if stop_requested:
                break

            time.sleep(1)

    print(
        "\nSession orchestrator "
        "stopped cleanly."
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":

    main()