from pathlib import Path
from datetime import datetime, timezone
import os
import time

import pandas as pd
from dotenv import load_dotenv

from deribit_client import DeribitClient


load_dotenv()

DATA_ROOT = Path(os.getenv("DERIBIT_DATA_ROOT"))
INSTRUMENTS_ROOT = DATA_ROOT / "raw" / "instruments"


def build_instrument_master() -> pd.DataFrame:

    client = DeribitClient()

    currencies = client.get_currencies()

    option_frames = []

    print(f"Checking {len(currencies)} Deribit currencies...\n")

    for item in currencies:

        currency = item["currency"]

        try:
            df = client.get_option_instruments(currency)

            if not df.empty:

                df["api_currency"] = currency
                option_frames.append(df)

                print(f"{currency}: {len(df)} active options")

        except Exception as exc:
            print(f"{currency}: error -> {exc}")

        time.sleep(0.05)

    if not option_frames:
        raise RuntimeError("No active option instruments found.")

    instruments = pd.concat(
        option_frames,
        ignore_index=True
    )

    # Convert Deribit millisecond timestamps to UTC datetimes
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

    instruments["snapshot_time"] = datetime.now(timezone.utc)

    return instruments


def save_instrument_master(df: pd.DataFrame):

    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    output_folder = (
        INSTRUMENTS_ROOT /
        f"date={snapshot_date}"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = output_folder / "instruments.parquet"

    df.to_parquet(
        output_file,
        index=False,
        compression="zstd"
    )

    print(f"\nSaved to:")
    print(output_file)


if __name__ == "__main__":

    instruments = build_instrument_master()

    print("\nSummary")
    print("-" * 40)

    print(f"Total active options: {len(instruments)}")

    print("\nOptions by API currency:")
    print(
        instruments
        .groupby("api_currency")
        .size()
        .sort_values(ascending=False)
    )

    save_instrument_master(instruments)