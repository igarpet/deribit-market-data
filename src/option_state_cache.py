from __future__ import annotations

import asyncio
import copy
import json
import signal
import statistics
import time
from datetime import datetime, timezone

import pandas as pd
import websockets

from session_manager import get_current_session


WS_URL = "wss://www.deribit.com/ws/api/v2"

SUBSCRIPTION_BATCH_SIZE = 400

HEALTH_INTERVAL_SECONDS = 30
RECONNECT_DELAY_SECONDS = 5


def deep_merge(original: dict, update: dict) -> dict:
    """
    Recursively merge an incremental ticker update
    into the existing cached ticker state.
    """

    for key, value in update.items():

        if (
            isinstance(value, dict)
            and isinstance(original.get(key), dict)
        ):
            deep_merge(
                original[key],
                value
            )

        else:
            original[key] = value

    return original


class OptionStateCache:

    def __init__(
        self,
        session,
        instruments: pd.DataFrame,
        stop_event: asyncio.Event
    ):

        self.session = session

        self.stop_event = stop_event

        self.instruments = (
            instruments
            .drop_duplicates(
                subset=["instrument_name"]
            )
            .copy()
        )

        self.instrument_names = (
            self.instruments[
                "instrument_name"
            ]
            .tolist()
        )

        self.metadata = (
            self.instruments
            .set_index("instrument_name")
            .to_dict(orient="index")
        )

        self.state: dict[str, dict] = {}

        self.messages_received = 0

        self.connection_count = 0

        self.last_message_received_at = None

        self.websocket = None

    # --------------------------------------------------
    # Subscription handling
    # --------------------------------------------------

    def build_channels(self) -> list[str]:

        return [
            f"incremental_ticker.{name}"
            for name in self.instrument_names
        ]

    async def subscribe(
        self,
        websocket
    ) -> None:

        channels = self.build_channels()

        print(
            f"\nSubscribing to "
            f"{len(channels)} option channels..."
        )

        batches = [
            channels[i:i + SUBSCRIPTION_BATCH_SIZE]
            for i in range(
                0,
                len(channels),
                SUBSCRIPTION_BATCH_SIZE
            )
        ]

        for batch_number, batch in enumerate(
            batches,
            start=1
        ):

            if self.stop_event.is_set():
                return

            request = {
                "jsonrpc": "2.0",
                "id": 1000 + batch_number,
                "method": "public/subscribe",
                "params": {
                    "channels": batch
                }
            }

            await websocket.send(
                json.dumps(request)
            )

            print(
                f"Subscription batch "
                f"{batch_number}/{len(batches)}: "
                f"{len(batch)} channels"
            )

        print(
            "Subscription requests sent."
        )

    # --------------------------------------------------
    # Ticker processing
    # --------------------------------------------------

    def process_ticker(
        self,
        data: dict
    ) -> None:

        instrument_name = data.get(
            "instrument_name"
        )

        if not instrument_name:
            return

        received_timestamp_ms = int(
            time.time() * 1000
        )

        update_type = data.get(
            "type"
        )

        if (
            update_type == "snapshot"
            or instrument_name not in self.state
        ):

            ticker_state = copy.deepcopy(
                data
            )

        else:

            ticker_state = self.state[
                instrument_name
            ]

            deep_merge(
                ticker_state,
                data
            )

        ticker_state[
            "_received_timestamp_ms"
        ] = received_timestamp_ms

        self.state[
            instrument_name
        ] = ticker_state

        self.messages_received += 1

        self.last_message_received_at = (
            received_timestamp_ms
        )

    async def receive_messages(
        self,
        websocket
    ) -> None:

        async for raw_message in websocket:

            if self.stop_event.is_set():
                break

            message = json.loads(
                raw_message
            )

            if "error" in message:

                print(
                    "\nDeribit error:"
                )

                print(
                    json.dumps(
                        message["error"],
                        indent=2
                    )
                )

                continue

            # Subscription confirmation
            if "id" in message:

                if "result" in message:

                    result = message[
                        "result"
                    ]

                    if isinstance(
                        result,
                        list
                    ):

                        print(
                            f"Confirmed "
                            f"{len(result)} "
                            f"subscriptions."
                        )

                continue

            params = message.get(
                "params"
            )

            if not params:
                continue

            channel = params.get(
                "channel",
                ""
            )

            if not channel.startswith(
                "incremental_ticker."
            ):
                continue

            data = params.get(
                "data"
            )

            if not isinstance(
                data,
                dict
            ):
                continue

            self.process_ticker(
                data
            )

    # --------------------------------------------------
    # Health monitoring
    # --------------------------------------------------

    async def health_reporter(
        self
    ) -> None:

        while not self.stop_event.is_set():

            try:

                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=HEALTH_INTERVAL_SECONDS
                )

                break

            except asyncio.TimeoutError:
                pass

            now_ms = int(
                time.time() * 1000
            )

            expected = len(
                self.instrument_names
            )

            initialized = len(
                self.state
            )

            ages = []

            for ticker in self.state.values():

                exchange_timestamp = (
                    ticker.get(
                        "timestamp"
                    )
                )

                if exchange_timestamp:

                    ages.append(
                        now_ms
                        - exchange_timestamp
                    )

            if ages:

                median_age = statistics.median(
                    ages
                )

                max_age = max(
                    ages
                )

            else:

                median_age = None
                max_age = None

            initialized_pct = (
                initialized
                / expected
                * 100
                if expected
                else 0
            )

            print(
                "\n[CACHE HEALTH]"
            )

            print(
                f"Expected instruments: "
                f"{expected}"
            )

            print(
                f"Initialized: "
                f"{initialized} "
                f"({initialized_pct:.1f}%)"
            )

            print(
                f"Messages received: "
                f"{self.messages_received:,}"
            )

            if median_age is not None:

                print(
                    f"Median exchange age: "
                    f"{median_age:,.0f} ms"
                )

                print(
                    f"Maximum exchange age: "
                    f"{max_age:,.0f} ms"
                )

    # --------------------------------------------------
    # Convert cache to DataFrame
    # --------------------------------------------------

    def to_dataframe(
        self,
        snapshot_time: datetime | None = None
    ) -> pd.DataFrame:

        if snapshot_time is None:

            snapshot_time = datetime.now(
                timezone.utc
            )

        snapshot_timestamp_ms = int(
            snapshot_time.timestamp()
            * 1000
        )

        rows = []

        for instrument_name in self.instrument_names:

            ticker = self.state.get(
                instrument_name,
                {}
            )

            metadata = self.metadata.get(
                instrument_name,
                {}
            )

            greeks = ticker.get(
                "greeks"
            ) or {}

            stats = ticker.get(
                "stats"
            ) or {}

            exchange_timestamp_ms = (
                ticker.get(
                    "timestamp"
                )
            )

            received_timestamp_ms = (
                ticker.get(
                    "_received_timestamp_ms"
                )
            )

            quote_age_ms = None

            if exchange_timestamp_ms is not None:

                quote_age_ms = (
                    snapshot_timestamp_ms
                    - exchange_timestamp_ms
                )

            row = {

                "snapshot_time":
                    snapshot_time,

                "exchange_timestamp_ms":
                    exchange_timestamp_ms,

                "received_timestamp_ms":
                    received_timestamp_ms,

                "quote_age_ms":
                    quote_age_ms,

                "instrument_id":
                    metadata.get(
                        "instrument_id"
                    ),

                "instrument_name":
                    instrument_name,

                "api_currency":
                    metadata.get(
                        "api_currency"
                    ),

                "underlying":
                    metadata.get(
                        "underlying"
                    ),

                "product_group":
                    metadata.get(
                        "product_group"
                    ),

                "instrument_type":
                    metadata.get(
                        "instrument_type"
                    ),

                "settlement_currency":
                    metadata.get(
                        "settlement_currency"
                    ),

                "settlement_period":
                    metadata.get(
                        "settlement_period"
                    ),

                "expiration_timestamp":
                    metadata.get(
                        "expiration_timestamp"
                    ),

                "expiration_time":
                    metadata.get(
                        "expiration_time"
                    ),

                "strike":
                    metadata.get(
                        "strike"
                    ),

                "option_type":
                    metadata.get(
                        "option_type"
                    ),

                "contract_size":
                    metadata.get(
                        "contract_size"
                    ),

                "state":
                    ticker.get(
                        "state"
                    ),

                "index_price":
                    ticker.get(
                        "index_price"
                    ),

                "underlying_price":
                    ticker.get(
                        "underlying_price"
                    ),

                "underlying_index":
                    ticker.get(
                        "underlying_index"
                    ),

                "interest_rate":
                    ticker.get(
                        "interest_rate"
                    ),

                "best_bid_price":
                    ticker.get(
                        "best_bid_price"
                    ),

                "best_bid_amount":
                    ticker.get(
                        "best_bid_amount"
                    ),

                "best_ask_price":
                    ticker.get(
                        "best_ask_price"
                    ),

                "best_ask_amount":
                    ticker.get(
                        "best_ask_amount"
                    ),

                "last_price":
                    ticker.get(
                        "last_price"
                    ),

                "mark_price":
                    ticker.get(
                        "mark_price"
                    ),

                "bid_iv":
                    ticker.get(
                        "bid_iv"
                    ),

                "ask_iv":
                    ticker.get(
                        "ask_iv"
                    ),

                "mark_iv":
                    ticker.get(
                        "mark_iv"
                    ),

                "delta":
                    greeks.get(
                        "delta"
                    ),

                "gamma":
                    greeks.get(
                        "gamma"
                    ),

                "theta":
                    greeks.get(
                        "theta"
                    ),

                "vega":
                    greeks.get(
                        "vega"
                    ),

                "rho":
                    greeks.get(
                        "rho"
                    ),

                "open_interest":
                    ticker.get(
                        "open_interest"
                    ),

                "volume_24h":
                    stats.get(
                        "volume"
                    ),

                "high_24h":
                    stats.get(
                        "high"
                    ),

                "low_24h":
                    stats.get(
                        "low"
                    ),

                "price_change_24h":
                    stats.get(
                        "price_change"
                    )
            }

            rows.append(
                row
            )

        df = pd.DataFrame(
            rows
        )

        if not df.empty:

            df[
                "minutes_to_expiry"
            ] = (
                pd.to_datetime(
                    df["expiration_time"],
                    utc=True
                )
                - pd.Timestamp(
                    snapshot_time
                )
            ).dt.total_seconds() / 60

        return df

    # --------------------------------------------------
    # Connection lifecycle
    # --------------------------------------------------

    async def run_connection(
        self
    ) -> None:

        self.connection_count += 1

        print(
            "\nConnecting to Deribit WebSocket..."
        )

        print(
            f"Connection attempt: "
            f"{self.connection_count}"
        )

        async with websockets.connect(
            WS_URL,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=None
        ) as websocket:

            self.websocket = websocket

            print(
                "WebSocket connected."
            )

            await self.subscribe(
                websocket
            )

            health_task = asyncio.create_task(
                self.health_reporter()
            )

            receive_task = asyncio.create_task(
                self.receive_messages(
                    websocket
                )
            )

            shutdown_task = asyncio.create_task(
                self.stop_event.wait()
            )

            try:

                done, pending = await asyncio.wait(
                    {
                        receive_task,
                        shutdown_task
                    },
                    return_when=asyncio.FIRST_COMPLETED
                )

                if (
                    shutdown_task in done
                    and self.stop_event.is_set()
                ):

                    print(
                        "\nShutdown requested."
                    )

                    print(
                        "Closing WebSocket..."
                    )

                    await websocket.close(
                        code=1000,
                        reason="Collector shutdown"
                    )

                for task in pending:

                    task.cancel()

                for task in pending:

                    try:
                        await task

                    except asyncio.CancelledError:
                        pass

            finally:

                health_task.cancel()

                try:
                    await health_task

                except asyncio.CancelledError:
                    pass

                if not receive_task.done():

                    receive_task.cancel()

                    try:
                        await receive_task

                    except asyncio.CancelledError:
                        pass

                if not shutdown_task.done():

                    shutdown_task.cancel()

                    try:
                        await shutdown_task

                    except asyncio.CancelledError:
                        pass

                self.websocket = None

                print(
                    "WebSocket closed."
                )

    async def run_forever(
        self
    ) -> None:

        while not self.stop_event.is_set():

            try:

                await self.run_connection()

            except asyncio.CancelledError:
                break

            except Exception as exc:

                if self.stop_event.is_set():
                    break

                print(
                    f"\nWebSocket connection "
                    f"error: {exc}"
                )

                print(
                    f"Reconnecting in "
                    f"{RECONNECT_DELAY_SECONDS} "
                    f"seconds..."
                )

                try:

                    await asyncio.wait_for(
                        self.stop_event.wait(),
                        timeout=RECONNECT_DELAY_SECONDS
                    )

                except asyncio.TimeoutError:
                    pass

        print(
            "\nOption-state cache stopped cleanly."
        )


async def main():

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def request_shutdown(
        signum,
        frame
    ):

        print(
            f"\nReceived shutdown signal "
            f"({signum})."
        )

        loop.call_soon_threadsafe(
            stop_event.set
        )

    # Ctrl+C
    signal.signal(
        signal.SIGINT,
        request_shutdown
    )

    # Standard termination signal
    if hasattr(
        signal,
        "SIGTERM"
    ):

        signal.signal(
            signal.SIGTERM,
            request_shutdown
        )

    # Windows Ctrl+Break
    if hasattr(
        signal,
        "SIGBREAK"
    ):

        signal.signal(
            signal.SIGBREAK,
            request_shutdown
        )

    session, instruments = (
        get_current_session()
    )

    print(
        "\n0DTE OPTION STATE CACHE"
    )

    print(
        "=" * 60
    )

    print(
        f"Session expiry: "
        f"{session.session_expiry.isoformat()}"
    )

    print(
        f"Contracts: "
        f"{len(instruments)}"
    )

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

    cache = OptionStateCache(
        session=session,
        instruments=instruments,
        stop_event=stop_event
    )

    await cache.run_forever()


if __name__ == "__main__":

    asyncio.run(
        main()
    )