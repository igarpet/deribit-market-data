from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os

import pandas as pd
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

DATA_ROOT = Path(data_root_env).expanduser().resolve()

INSTRUMENTS_ROOT = DATA_ROOT / "raw" / "instruments"
STATE_ROOT = DATA_ROOT / "state"

SETTLEMENT_HOUR_UTC = 8


# --------------------------------------------------
# Session definition
# --------------------------------------------------

@dataclass(frozen=True)
class SessionWindow:
    session_start: datetime
    session_expiry: datetime
    expiry_timestamp_ms: int
    expiry_date: str
    session_id: str


def build_session_window(
    session_start: datetime,
    session_expiry: datetime
) -> SessionWindow:
    """Build a normalized UTC SessionWindow."""

    if (
        session_start.tzinfo is None
        or session_expiry.tzinfo is None
    ):
        raise ValueError(
            "Session datetimes must be timezone-aware."
        )

    session_start = session_start.astimezone(
        timezone.utc
    )

    session_expiry = session_expiry.astimezone(
        timezone.utc
    )

    expiry_timestamp_ms = int(
        session_expiry.timestamp() * 1000
    )

    expiry_date = session_expiry.strftime(
        "%Y-%m-%d"
    )

    return SessionWindow(
        session_start=session_start,
        session_expiry=session_expiry,
        expiry_timestamp_ms=expiry_timestamp_ms,
        expiry_date=expiry_date,
        session_id=f"expiry={expiry_date}"
    )


def calculate_session_window(
    now: datetime | None = None
) -> SessionWindow:
    """
    Determine the active Deribit 0DTE session.

    A session runs:

        08:00 UTC -> 08:00 UTC next day

    The expiry timestamp remains fixed for the entire session.
    """

    if now is None:
        now = datetime.now(timezone.utc)

    if now.tzinfo is None:
        raise ValueError(
            "The supplied datetime must be timezone-aware."
        )

    now = now.astimezone(timezone.utc)

    today_settlement = now.replace(
        hour=SETTLEMENT_HOUR_UTC,
        minute=0,
        second=0,
        microsecond=0
    )

    if now >= today_settlement:
        session_start = today_settlement
    else:
        session_start = (
            today_settlement
            - timedelta(days=1)
        )

    session_expiry = (
        session_start
        + timedelta(days=1)
    )

    return build_session_window(
        session_start=session_start,
        session_expiry=session_expiry
    )


def session_window_for_expiry_date(
    expiry_date: str
) -> SessionWindow:
    """
    Build the exact 0DTE session ending at 08:00 UTC on
    the supplied YYYY-MM-DD expiry date.

    This is the key helper used by warm handover: a collector
    can target tomorrow's expiry before tomorrow's session is
    the wall-clock current session.
    """

    try:
        expiry = datetime.strptime(
            expiry_date,
            "%Y-%m-%d"
        ).replace(
            hour=SETTLEMENT_HOUR_UTC,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=timezone.utc
        )

    except ValueError as exc:
        raise ValueError(
            "expiry_date must be YYYY-MM-DD."
        ) from exc

    return build_session_window(
        session_start=(
            expiry
            - timedelta(days=1)
        ),
        session_expiry=expiry
    )


def calculate_next_session_window(
    session: SessionWindow | None = None
) -> SessionWindow:
    """Return the session immediately after the supplied/current one."""

    if session is None:
        session = calculate_session_window()

    return build_session_window(
        session_start=session.session_expiry,
        session_expiry=(
            session.session_expiry
            + timedelta(days=1)
        )
    )


# --------------------------------------------------
# Instrument master
# --------------------------------------------------

def load_latest_instrument_master() -> tuple[pd.DataFrame, Path]:
    """Load the most recent instrument-master Parquet."""

    files = sorted(
        INSTRUMENTS_ROOT.glob(
            "date=*/instruments.parquet"
        )
    )

    if not files:
        raise RuntimeError(
            "No instrument master found. "
            "Run instrument_master.py first."
        )

    latest_file = files[-1]

    df = pd.read_parquet(
        latest_file
    )

    return df, latest_file


def select_session_instruments(
    master: pd.DataFrame,
    session: SessionWindow
) -> pd.DataFrame:
    """
    Select every active option expiring at the locked
    session expiry timestamp.

    Important:
    We intentionally DO NOT filter settlement_period == 'day'.

    On dates where a daily expiry coincides with a weekly,
    monthly, or quarterly expiry, Deribit may not create a
    separate daily contract.
    """

    if "expiration_timestamp" not in master.columns:
        raise KeyError(
            "Instrument master does not contain "
            "'expiration_timestamp'."
        )

    expiration = pd.to_numeric(
        master["expiration_timestamp"],
        errors="coerce"
    )

    mask = (
        expiration
        == session.expiry_timestamp_ms
    )

    session_instruments = (
        master.loc[mask]
        .copy()
    )

    if "is_active" in session_instruments.columns:
        session_instruments = (
            session_instruments[
                session_instruments["is_active"]
                == True
            ]
            .copy()
        )

    if session_instruments.empty:
        raise RuntimeError(
            "No active options found for session expiry "
            f"{session.session_expiry.isoformat()}.\n"
            "Refresh the instrument master and try again."
        )

    sort_columns = [
        column
        for column in [
            "underlying",
            "product_group",
            "strike",
            "option_type"
        ]
        if column in session_instruments.columns
    ]

    if sort_columns:
        session_instruments = (
            session_instruments
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )

    return session_instruments


def get_session_for_expiry_date(
    expiry_date: str
) -> tuple[
    SessionWindow,
    pd.DataFrame
]:
    """
    Resolve an explicit session and its instruments from the
    latest instrument master.
    """

    session = session_window_for_expiry_date(
        expiry_date
    )

    master, _ = (
        load_latest_instrument_master()
    )

    instruments = (
        select_session_instruments(
            master,
            session
        )
    )

    return session, instruments


# --------------------------------------------------
# State persistence
# --------------------------------------------------

def save_session_state(
    session: SessionWindow,
    instruments: pd.DataFrame
) -> Path:
    """
    Save the current session definition.

    This state locks the EXPIRY, not the instrument list.
    The actual list of strikes can therefore be refreshed
    later during the session.
    """

    STATE_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    current_state_file = (
        STATE_ROOT
        / "current_session.json"
    )

    session_folder = (
        STATE_ROOT
        / "sessions"
        / session.session_id
    )

    session_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    historical_state_file = (
        session_folder
        / "session.json"
    )

    underlyings = []

    if "underlying" in instruments.columns:
        underlyings = sorted(
            instruments["underlying"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    product_groups = []

    if "product_group" in instruments.columns:
        product_groups = sorted(
            instruments["product_group"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    payload = {
        "session_id": session.session_id,
        "session_start_utc": (
            session.session_start.isoformat()
        ),
        "session_expiry_utc": (
            session.session_expiry.isoformat()
        ),
        "expiry_timestamp_ms": (
            session.expiry_timestamp_ms
        ),
        "expiry_date": (
            session.expiry_date
        ),
        "instrument_count_at_check": (
            len(instruments)
        ),
        "underlyings_at_check": (
            underlyings
        ),
        "product_groups_at_check": (
            product_groups
        ),
        "last_checked_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        )
    }

    for output_file in [
        current_state_file,
        historical_state_file
    ]:
        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                payload,
                file,
                indent=4
            )

    return current_state_file


# --------------------------------------------------
# Summary
# --------------------------------------------------

def print_session_summary(
    session: SessionWindow,
    instruments: pd.DataFrame,
    master_file: Path
) -> None:

    print("\n0DTE SESSION")
    print("=" * 60)

    print(
        f"Instrument master: "
        f"{master_file}"
    )

    print(
        f"Session start:      "
        f"{session.session_start.isoformat()}"
    )

    print(
        f"Session expiry:     "
        f"{session.session_expiry.isoformat()}"
    )

    print(
        f"Expiry timestamp:   "
        f"{session.expiry_timestamp_ms}"
    )

    print(
        f"Session ID:         "
        f"{session.session_id}"
    )

    print(
        f"Contracts:          "
        f"{len(instruments)}"
    )

    if "underlying" in instruments.columns:

        print(
            "\nContracts by underlying:"
        )

        print(
            instruments
            .groupby("underlying")
            .size()
            .sort_values(
                ascending=False
            )
        )

    if (
        "underlying" in instruments.columns
        and "product_group" in instruments.columns
    ):

        print(
            "\nContracts by product group:"
        )

        print(
            instruments
            .groupby(
                [
                    "underlying",
                    "product_group"
                ]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

    if "settlement_period" in instruments.columns:

        print(
            "\nDeribit settlement-period labels:"
        )

        print(
            instruments
            .groupby("settlement_period")
            .size()
            .sort_values(
                ascending=False
            )
        )


# --------------------------------------------------
# Public collector helpers
# --------------------------------------------------

def get_current_session() -> tuple[
    SessionWindow,
    pd.DataFrame
]:
    """
    Return the wall-clock current session and its instrument
    universe. Existing callers, including the futures collector,
    keep the same interface.
    """

    session = calculate_session_window()

    master, _ = (
        load_latest_instrument_master()
    )

    instruments = (
        select_session_instruments(
            master,
            session
        )
    )

    return session, instruments


if __name__ == "__main__":

    session = calculate_session_window()

    master, master_file = (
        load_latest_instrument_master()
    )

    instruments = (
        select_session_instruments(
            master,
            session
        )
    )

    print_session_summary(
        session,
        instruments,
        master_file
    )

    state_file = save_session_state(
        session,
        instruments
    )

    print(
        f"\nSession state saved to:\n"
        f"{state_file}"
    )