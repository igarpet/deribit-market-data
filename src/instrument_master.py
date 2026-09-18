from pathlib import Path
from datetime import datetime, timezone
import os
import time

import pandas as pd
from dotenv import load_dotenv

from deribit_client import DeribitClient


# Load environment variables from .env
load_dotenv()

data_root_env = os.getenv("DERIBIT_DATA_ROOT")

if not data_root_env:
    raise RuntimeError(
        "DERIBIT_DATA_ROOT is not defined in the .env file."
    )

DATA_ROOT = Path(data_root_env).expanduser().resolve()

INSTRUMENTS_ROOT = DATA_ROOT / "raw" / "instruments"


def build_instrument_master() -> pd.DataFrame:
    """
    Download all currently active Deribit option instruments
    across every API currency bucket.
    """

    client = DeribitClient()

    currencies = client.get_currencies()

    option_frames = []

    print(f"Checking {len(currencies)} Deribit currencies...\n")

    for item in currencies:
        currency = item["currency"]

        try:
            df = client.get_option_instruments(currency)

            if not df.empty:
                # Currency bucket used in Deribit's API request.
                # Examples: BTC, ETH, USDC
                df["api_currency"] = currency

                # Actual underlying asset.
                # Examples: BTC, ETH, SOL, XRP
                df["underlying"] = df["base_currency"]

                option_frames.append(df)

                print(
                    f"{currency}: "
                    f"{len(df)} active options"
                )

        except Exception as exc:
            print(
                f"{currency}: error -> {exc}"
            )

        # Small delay between API requests
        time.sleep(0.05)

    if not option_frames:
        raise RuntimeError(
            "No active option instruments found."
        )

    instruments = pd.concat(
        option_frames,
        ignore_index=True
    )

    # Convert Deribit millisecond timestamps to UTC
    instruments["creation_time"] = pd.to_datetime(
        instruments["creation_timestamp"],
        unit="ms",
        utc=True
    )

    instruments["expiration_time"] = pd.to_datetime(
        instruments["expiration_timestamp"],
        unit="ms",
        utc=True
    )

    # Time at which this instrument master was collected
    instruments["snapshot_time"] = datetime.now(timezone.utc)

    return instruments


def save_instrument_master(df: pd.DataFrame) -> Path:
    """
    Save the current instrument master as a daily Parquet file.
    """

    snapshot_date = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    output_folder = (
        INSTRUMENTS_ROOT
        / f"date={snapshot_date}"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_folder
        / "instruments.parquet"
    )

    df.to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )

    print("\nSaved to:")
    print(output_file)

    return output_file


def print_summary(
    instruments: pd.DataFrame
) -> None:
    """
    Print a simple summary of the active option universe.
    """

    print("\nSummary")
    print("-" * 40)

    print(
        f"Total active options: "
        f"{len(instruments)}"
    )

    print("\nOptions by API currency:")
    print(
        instruments
        .groupby("api_currency")
        .size()
        .sort_values(ascending=False)
    )

    print("\nOptions by underlying:")
    print(
        instruments
        .groupby("underlying")
        .size()
        .sort_values(ascending=False)
    )

    print("\nOptions by underlying and product group:")
    print(
        instruments
        .groupby(
            [
                "underlying",
                "product_group"
            ]
        )
        .size()
        .sort_values(ascending=False)
    )


if __name__ == "__main__":
    instruments = build_instrument_master()

    print_summary(instruments)

    save_instrument_master(instruments)