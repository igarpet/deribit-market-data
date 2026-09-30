from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from session_manager import (
    STATE_ROOT,
    calculate_next_session_window,
    calculate_session_window,
)
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

# Shorter overlap than the previous 3 minutes.
# This still leaves ample time for instrument refresh
# and WebSocket initialization on the current VM.
PREWARM_LEAD_SECONDS = 90

PREWARM_RETRY_SECONDS = 15

OLD_COLLECTOR_EXIT_TIMEOUT_SECONDS = 30

ROLLOVER_RETRY_SECONDS = 5 * 60

WARMUP_STATUS_MAX_AGE_SECONDS = 5

WARMUP_MAX_FEED_SILENCE_MS = 10_000


# --------------------------------------------------
# Global process state
# --------------------------------------------------

stop_requested = False

utility_process: subprocess.Popen | None = None

collector_processes: dict[
    str,
    subprocess.Popen
] = {}


# --------------------------------------------------
# Process helpers
# --------------------------------------------------

def process_is_running(
    process: subprocess.Popen | None
) -> bool:

    return (
        process is not None
        and process.poll() is None
    )


def terminate_process(
    process: subprocess.Popen | None,
    label: str
) -> None:

    if not process_is_running(
        process
    ):
        return

    print(
        f"Stopping {label}..."
    )

    try:

        process.terminate()

    except Exception as exc:

        print(
            f"Could not terminate "
            f"{label} cleanly: "
            f"{exc}"
        )


def sleep_interruptibly(
    seconds: float
) -> None:

    deadline = (
        time.monotonic()
        + seconds
    )

    while (
        not stop_requested
        and time.monotonic() < deadline
    ):

        time.sleep(
            min(
                0.5,
                max(
                    0.0,
                    deadline
                    - time.monotonic()
                )
            )
        )


# --------------------------------------------------
# Shutdown handling
# --------------------------------------------------

def request_shutdown(
    signum,
    frame
):

    global stop_requested
    global utility_process

    print(
        f"\nOrchestrator received "
        f"shutdown signal ({signum})."
    )

    stop_requested = True

    terminate_process(
        utility_process,
        "utility process"
    )

    for expiry_date, process in list(
        collector_processes.items()
    ):

        terminate_process(
            process,
            f"collector {expiry_date}"
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

    global utility_process

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
        " ".join(
            command
        )
    )

    utility_process = (
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT
        )
    )

    return_code = (
        utility_process.wait()
    )

    utility_process = None

    return return_code


# --------------------------------------------------
# Instrument master
# --------------------------------------------------

def refresh_instruments() -> bool:

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
        str(
            DATABASE_FILE
        )
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
# Completed-session discovery
# --------------------------------------------------

def find_completed_unregistered_sessions() -> list[str]:

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

        candidates.sort(
            key=lambda item:
                item[0]
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

    if stop_requested:
        return False

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
# Warm-up status
# --------------------------------------------------

def warmup_ready_path(
    session
) -> Path:

    return (
        STATE_ROOT
        / "warmup"
        / session.session_id
        / "ready.json"
    )


def clear_warmup_ready_state(
    session
) -> None:

    path = warmup_ready_path(
        session
    )

    if path.exists():

        try:

            path.unlink()

        except OSError as exc:

            print(
                f"Could not remove stale "
                f"warm-up marker: {exc}"
            )


def read_warmup_status(
    session
) -> dict | None:

    path = warmup_ready_path(
        session
    )

    if not path.exists():
        return None

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            payload = json.load(
                file
            )

    except Exception as exc:

        print(
            f"\nCould not read warm-up "
            f"status: {exc}"
        )

        return None

    if (
        payload.get(
            "session_id"
        )
        != session.session_id
    ):

        return None

    return payload


def warmup_status_is_ready(
    session
) -> tuple[
    bool,
    dict | None,
    str
]:

    payload = read_warmup_status(
        session
    )

    if payload is None:

        return (
            False,
            None,
            "no warm-up status file"
        )

    try:

        updated_at = (
            datetime.fromisoformat(
                payload[
                    "updated_at_utc"
                ]
            )
        )

        if updated_at.tzinfo is None:

            updated_at = (
                updated_at.replace(
                    tzinfo=timezone.utc
                )
            )

        updated_at = (
            updated_at.astimezone(
                timezone.utc
            )
        )

    except Exception:

        return (
            False,
            payload,
            "invalid status timestamp"
        )

    age_seconds = (
        datetime.now(
            timezone.utc
        )
        - updated_at
    ).total_seconds()

    if (
        age_seconds
        > WARMUP_STATUS_MAX_AGE_SECONDS
    ):

        return (
            False,
            payload,
            (
                "warm-up status is stale "
                f"({age_seconds:.1f}s)"
            )
        )

    if not payload.get(
        "connected",
        False
    ):

        return (
            False,
            payload,
            "warm-up WebSocket is not connected"
        )

    expected = int(
        payload.get(
            "expected_instruments",
            0
        )
    )

    initialized = int(
        payload.get(
            "initialized_instruments",
            0
        )
    )

    if (
        expected <= 0
        or initialized != expected
    ):

        return (
            False,
            payload,
            (
                "warm-up cache is not fully "
                f"initialized "
                f"({initialized}/{expected})"
            )
        )

    if not payload.get(
        "fully_initialized",
        False
    ):

        return (
            False,
            payload,
            "fully_initialized is false"
        )

    if not payload.get(
        "cache_initialized_before_boundary",
        False
    ):

        return (
            False,
            payload,
            "cache was not initialized before boundary"
        )

    last_message_age_ms = (
        payload.get(
            "last_message_age_ms"
        )
    )

    if (
        last_message_age_ms is None
        or float(
            last_message_age_ms
        )
        > WARMUP_MAX_FEED_SILENCE_MS
    ):

        return (
            False,
            payload,
            "warm-up feed is not recent"
        )

    if not payload.get(
        "ready_for_opening_boundary",
        False
    ):

        return (
            False,
            payload,
            "collector does not report "
            "opening readiness"
        )

    return (
        True,
        payload,
        "ready"
    )


# --------------------------------------------------
# Collector lifecycle
# --------------------------------------------------

def start_collector(
    session,
    warmup: bool
) -> subprocess.Popen:

    script = (
        SRC_ROOT
        / "option_grid_collector.py"
    )

    command = [
        sys.executable,
        str(script),
        "--expiry-date",
        session.expiry_date,
    ]

    if warmup:

        command.append(
            "--warmup"
        )

        clear_warmup_ready_state(
            session
        )

    print(
        "\nStarting option collector:"
    )

    print(
        " ".join(
            command
        )
    )

    print(
        f"Target session: "
        f"{session.session_id}"
    )

    print(
        f"Warm-up mode:  "
        f"{warmup}"
    )

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT
    )

    collector_processes[
        session.expiry_date
    ] = process

    return process


def forget_collector(
    session,
    process: subprocess.Popen | None
) -> None:

    registered = collector_processes.get(
        session.expiry_date
    )

    if registered is process:

        collector_processes.pop(
            session.expiry_date,
            None
        )


def wait_for_old_collector(
    session,
    process: subprocess.Popen | None
) -> None:

    if process is None:
        return

    if process.poll() is not None:

        forget_collector(
            session,
            process
        )

        return

    print(
        f"\nWaiting for expired "
        f"collector "
        f"{session.expiry_date} "
        f"to exit..."
    )

    try:

        process.wait(
            timeout=(
                OLD_COLLECTOR_EXIT_TIMEOUT_SECONDS
            )
        )

    except subprocess.TimeoutExpired:

        print(
            "Expired collector did not "
            "exit within timeout; "
            "terminating it."
        )

        terminate_process(
            process,
            (
                "expired collector "
                f"{session.expiry_date}"
            )
        )

        try:

            process.wait(
                timeout=10
            )

        except subprocess.TimeoutExpired:

            print(
                "Expired collector still "
                "did not exit; killing it."
            )

            process.kill()
            process.wait()

    forget_collector(
        session,
        process
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    configure_signals()

    print(
        "\nDERIBIT 0DTE "
        "SESSION ORCHESTRATOR"
    )

    print(
        "=" * 60
    )

    print(
        f"Started at: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )

    refresh_ok = (
        refresh_instruments()
    )

    if stop_requested:
        return

    if not refresh_ok:

        print(
            "\nWARNING: initial instrument "
            "refresh failed."
        )

        print(
            "Attempting to start from the "
            "latest existing instrument master."
        )

    current_session = (
        calculate_session_window()
    )

    print(
        "\nACTIVE SESSION"
    )

    print(
        "=" * 60
    )

    print(
        f"Session ID: "
        f"{current_session.session_id}"
    )

    print(
        f"Start:      "
        f"{current_session.session_start.isoformat()}"
    )

    print(
        f"Expiry:     "
        f"{current_session.session_expiry.isoformat()}"
    )

    current_collector = start_collector(
        current_session,
        warmup=False
    )

    # Historical compaction is deferred so startup
    # never blocks live orchestration near 08:00.
    pending_rollovers = (
        find_completed_unregistered_sessions()
    )

    if pending_rollovers:

        print(
            "\nDeferred completed sessions:"
        )

        for expiry_date in (
            pending_rollovers
        ):

            print(
                f"  {expiry_date}"
            )

    else:

        print(
            "\nNo completed unregistered "
            "sessions found."
        )

    next_session = None
    next_collector = None

    prewarm_retry_at = (
        datetime.min.replace(
            tzinfo=timezone.utc
        )
    )

    next_ready_logged = False

    next_rollover_retry_at = (
        datetime.min.replace(
            tzinfo=timezone.utc
        )
    )

    while not stop_requested:

        now = datetime.now(
            timezone.utc
        )

        # --------------------------------------------------
        # A. 08:00 warm handover
        # --------------------------------------------------

        if (
            now
            >= current_session.session_expiry
        ):

            expired_session = (
                current_session
            )

            expired_collector = (
                current_collector
            )

            print(
                "\nSESSION HANDOVER"
            )

            print(
                "=" * 60
            )

            print(
                f"Expired session: "
                f"{expired_session.session_id}"
            )

            if next_session is None:

                next_session = (
                    calculate_next_session_window(
                        expired_session
                    )
                )

            warm_process_alive = (
                process_is_running(
                    next_collector
                )
            )

            warm_ready, _, reason = (
                warmup_status_is_ready(
                    next_session
                )
            )

            if warm_process_alive:

                print(
                    "Warm collector process: "
                    "alive"
                )

                print(
                    f"Warm collector boundary "
                    f"readiness: {warm_ready}"
                )

                print(
                    f"Readiness detail: "
                    f"{reason}"
                )

            else:

                print(
                    "Warm collector process: "
                    "not available"
                )

            if not warm_process_alive:

                print(
                    "\nStarting new session now."
                )

                print(
                    "The true 08:00 opening "
                    "state will remain missing."
                )

                refresh_instruments()

                if stop_requested:
                    break

                next_collector = (
                    start_collector(
                        next_session,
                        warmup=False
                    )
                )

            elif not warm_ready:

                # Keep this child. It already targets
                # the correct expiry and can continue
                # initializing/reconnecting. Its own
                # boundary logic refuses to fabricate
                # an invalid 08:00 row.
                print(
                    "\nWarm child is alive but "
                    "not boundary-ready."
                )

                print(
                    "Promoting it for continuity; "
                    "08:00 may be missing."
                )

            current_session = (
                next_session
            )

            current_collector = (
                next_collector
            )

            next_session = None
            next_collector = None
            next_ready_logged = False

            prewarm_retry_at = (
                datetime.min.replace(
                    tzinfo=timezone.utc
                )
            )

            print(
                f"\nNew active session: "
                f"{current_session.session_id}"
            )

            wait_for_old_collector(
                expired_session,
                expired_collector
            )

            if (
                expired_session.expiry_date
                not in pending_rollovers
            ):

                pending_rollovers.append(
                    expired_session.expiry_date
                )

            # Return immediately to monitoring the
            # new live child. Rollover is deferred.
            continue

        # --------------------------------------------------
        # B. Current-session child unexpectedly stopped
        # --------------------------------------------------

        if not process_is_running(
            current_collector
        ):

            return_code = (
                current_collector.poll()
                if current_collector
                is not None
                else None
            )

            forget_collector(
                current_session,
                current_collector
            )

            print(
                "\nCurrent-session collector "
                "exited before session expiry."
            )

            print(
                f"Return code: "
                f"{return_code}"
            )

            print(
                f"Restarting in "
                f"{RESTART_DELAY_SECONDS} "
                f"seconds..."
            )

            sleep_interruptibly(
                RESTART_DELAY_SECONDS
            )

            if stop_requested:
                break

            current_collector = (
                start_collector(
                    current_session,
                    warmup=False
                )
            )

            continue

        # --------------------------------------------------
        # C. Pre-warm next expiry
        # --------------------------------------------------

        prewarm_start = (
            current_session.session_expiry
            - timedelta(
                seconds=(
                    PREWARM_LEAD_SECONDS
                )
            )
        )

        if (
            now >= prewarm_start
            and now
            < current_session.session_expiry
        ):

            if next_session is None:

                next_session = (
                    calculate_next_session_window(
                        current_session
                    )
                )

            if (
                next_collector is not None
                and not process_is_running(
                    next_collector
                )
            ):

                return_code = (
                    next_collector.poll()
                )

                forget_collector(
                    next_session,
                    next_collector
                )

                print(
                    "\nPre-warm collector "
                    "exited before handover."
                )

                print(
                    f"Return code: "
                    f"{return_code}"
                )

                next_collector = None
                next_ready_logged = False

                prewarm_retry_at = (
                    now
                    + timedelta(
                        seconds=(
                            PREWARM_RETRY_SECONDS
                        )
                    )
                )

            if (
                next_collector is None
                and now >= prewarm_retry_at
            ):

                print(
                    "\nPRE-WARMING NEXT SESSION"
                )

                print(
                    "=" * 60
                )

                print(
                    f"Current: "
                    f"{current_session.session_id}"
                )

                print(
                    f"Next:    "
                    f"{next_session.session_id}"
                )

                refresh_ok = (
                    refresh_instruments()
                )

                if stop_requested:
                    break

                if refresh_ok:

                    try:

                        next_collector = (
                            start_collector(
                                next_session,
                                warmup=True
                            )
                        )

                        next_ready_logged = (
                            False
                        )

                    except Exception as exc:

                        print(
                            "Could not launch "
                            "next-session collector: "
                            f"{exc}"
                        )

                        next_collector = None

                else:

                    print(
                        "Pre-warm instrument "
                        "refresh failed."
                    )

                if next_collector is None:

                    prewarm_retry_at = (
                        datetime.now(
                            timezone.utc
                        )
                        + timedelta(
                            seconds=(
                                PREWARM_RETRY_SECONDS
                            )
                        )
                    )

            if (
                next_collector is not None
                and process_is_running(
                    next_collector
                )
            ):

                ready, payload, reason = (
                    warmup_status_is_ready(
                        next_session
                    )
                )

                if (
                    ready
                    and not next_ready_logged
                ):

                    print(
                        "\nNext-session collector "
                        "reports CURRENTLY READY "
                        "for the 08:00 boundary."
                    )

                    if payload is not None:

                        print(
                            f"Connection generation: "
                            f"{payload.get('connection_count')}"
                        )

                        print(
                            f"Initialized: "
                            f"{payload.get('initialized_instruments')}/"
                            f"{payload.get('expected_instruments')}"
                        )

                    next_ready_logged = True

                elif (
                    not ready
                    and next_ready_logged
                ):

                    print(
                        "\nNext-session collector "
                        "lost readiness."
                    )

                    print(
                        f"Reason: {reason}"
                    )

                    next_ready_logged = False

        # --------------------------------------------------
        # D. Deferred rollover / startup recovery
        # --------------------------------------------------

        seconds_to_expiry = (
            current_session.session_expiry
            - now
        ).total_seconds()

        if (
            pending_rollovers
            and now >= next_rollover_retry_at
            and seconds_to_expiry
            > PREWARM_LEAD_SECONDS + 60
        ):

            expiry_date = (
                pending_rollovers[0]
            )

            print(
                f"\nRunning deferred rollover: "
                f"{expiry_date}"
            )

            if rollover_session(
                expiry_date
            ):

                pending_rollovers.pop(
                    0
                )

                next_rollover_retry_at = (
                    datetime.now(
                        timezone.utc
                    )
                    + timedelta(
                        seconds=5
                    )
                )

            else:

                next_rollover_retry_at = (
                    datetime.now(
                        timezone.utc
                    )
                    + timedelta(
                        seconds=(
                            ROLLOVER_RETRY_SECONDS
                        )
                    )
                )

        time.sleep(
            0.5
        )

    # --------------------------------------------------
    # Graceful shutdown
    # --------------------------------------------------

    for expiry_date, process in list(
        collector_processes.items()
    ):

        terminate_process(
            process,
            f"collector {expiry_date}"
        )

    for expiry_date, process in list(
        collector_processes.items()
    ):

        if process_is_running(
            process
        ):

            try:

                process.wait(
                    timeout=15
                )

            except subprocess.TimeoutExpired:

                process.kill()
                process.wait()

    collector_processes.clear()

    print(
        "\nSession orchestrator "
        "stopped cleanly."
    )


if __name__ == "__main__":

    main()
