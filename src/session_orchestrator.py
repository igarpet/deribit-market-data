from __future__ import annotations

import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from session_manager import calculate_session_window


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

stop_requested = False
current_process = None


# --------------------------------------------------
# Shutdown
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

        current_process.terminate()


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
# Run project scripts
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

    print(
        "\nRefreshing Deribit "
        "instrument master..."
    )

    return_code = run_script(
        "instrument_master.py"
    )

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
# Session rollover
# --------------------------------------------------

def finalize_expired_session() -> bool:

    print(
        "\nFinalizing expired session..."
    )

    return_code = run_script(
        "session_rollover.py"
    )

    if return_code != 0:

        print(
            "\nSession rollover failed."
        )

        return False

    print(
        "Expired session finalized."
    )

    return True


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

    # Refresh once whenever the service starts.
    if not refresh_instruments():

        raise RuntimeError(
            "Initial instrument master "
            "refresh failed."
        )

    opening_snapshot = False

    while not stop_requested:

        session = calculate_session_window()

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

        if stop_requested:
            break

        now = datetime.now(
            timezone.utc
        )

        # --------------------------------------------------
        # Normal 08:00 session completion
        # --------------------------------------------------

        if now >= session.session_expiry:

            print(
                "\nSession expiry reached."
            )

            rollover_ok = (
                finalize_expired_session()
            )

            if not rollover_ok:

                raise RuntimeError(
                    "Session rollover failed. "
                    "Stopping orchestrator to "
                    "avoid silently losing data."
                )

            if not refresh_instruments():

                raise RuntimeError(
                    "Instrument master refresh "
                    "failed after rollover."
                )

            # The next collector is the new
            # 08:00 session.
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
            f"{RESTART_DELAY_SECONDS} seconds..."
        )

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


if __name__ == "__main__":
    main()