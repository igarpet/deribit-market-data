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

from deribit_client import DeribitClient
from session_manager import get_current_session


WS_URL = "wss://www.deribit.com/ws/api/v2"

SUBSCRIPTION_BATCH_SIZE = 400

HEALTH_INTERVAL_SECONDS = 30
RECONNECT_DELAY_SECONDS = 5

FEED_STALE_TIMEOUT_SECONDS = 30
INITIAL_MESSAGE_GRACE_SECONDS = 15

# Deribit publishes full instrument metadata here whenever
# a new option instrument is created.
INSTRUMENT_CREATION_CHANNEL = (
    "instrument.creation.option.any"
)

# Safety-net reconciliation. The creation channel provides
# near-real-time discovery; REST reconciliation recovers any
# creation notifications missed during a WebSocket outage.
UNIVERSE_RECONCILE_INTERVAL_SECONDS = 5 * 60

# Created instruments in terminal/non-tradable states are not
# added to the expected live universe. REST reconciliation can
# pick them up later if they become active.
NON_COLLECTIBLE_STATES = {
    "inactive",
    "delivered",
    "archivized",
}


def deep_merge(
    original: dict,
    update: dict
) -> dict:
    """
    Recursively merge an incremental ticker update
    into the existing cached ticker state.
    """

    for key, value in update.items():

        if (
            isinstance(value, dict)
            and isinstance(
                original.get(key),
                dict
            )
        ):

            deep_merge(
                original[key],
                value
            )

        else:

            original[key] = value

    return original


class FeedStaleError(
    RuntimeError
):
    pass


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
                subset=[
                    "instrument_name"
                ]
            )
            .copy()
        )

        self.instrument_names = (
            self.instruments[
                "instrument_name"
            ]
            .astype(str)
            .tolist()
        )

        self.metadata = (
            self.instruments
            .set_index(
                "instrument_name"
            )
            .to_dict(
                orient="index"
            )
        )

        # Ticker state belongs to one WebSocket connection
        # generation only. It is cleared before reconnects.
        self.state: dict[
            str,
            dict
        ] = {}

        self.messages_received = 0
        self.connection_count = 0

        self.connected = False

        self.connection_started_at_ms: (
            int | None
        ) = None

        self.last_message_received_at: (
            int | None
        ) = None

        self.fully_initialized_at_ms: (
            int | None
        ) = None

        self.websocket = None

        # Dynamic-universe diagnostics.
        self.initial_instrument_count = len(
            self.instrument_names
        )

        self.dynamic_instruments_added = 0

        self.creation_notifications_received = 0
        self.creation_notifications_matched = 0

        self.universe_reconciliations = 0
        self.universe_reconciliation_errors = 0

        self.last_universe_update_ms: (
            int | None
        ) = None

        # IDs for subscriptions sent after startup.
        self.next_subscription_request_id = (
            10_000
        )

        # Creation notifications and REST reconciliation can
        # both discover instruments at nearly the same time.
        # Serialize WebSocket subscription writes.
        self.subscription_lock = (
            asyncio.Lock()
        )

    # --------------------------------------------------
    # Readiness / health
    # --------------------------------------------------

    @property
    def expected_count(
        self
    ) -> int:

        return len(
            self.instrument_names
        )

    @property
    def initialized_count(
        self
    ) -> int:

        return len(
            self.state
        )

    @property
    def coverage_pct(
        self
    ) -> float:

        if not self.expected_count:
            return 0.0

        return (
            self.initialized_count
            / self.expected_count
            * 100.0
        )

    def last_message_age_ms(
        self,
        now_ms: int | None = None
    ) -> int | None:

        if (
            self.last_message_received_at
            is None
        ):

            return None

        if now_ms is None:

            now_ms = int(
                time.time()
                * 1000
            )

        return max(
            0,
            now_ms
            - self.last_message_received_at
        )

    def is_fully_initialized(
        self
    ) -> bool:

        return (
            self.expected_count > 0
            and self.initialized_count
            == self.expected_count
            and self.fully_initialized_at_ms
            is not None
        )

    def readiness_snapshot(
        self
    ) -> dict:

        now_ms = int(
            time.time()
            * 1000
        )

        return {
            "connected":
                self.connected,

            "connection_count":
                self.connection_count,

            "connection_started_at_ms":
                self.connection_started_at_ms,

            "expected_instruments":
                self.expected_count,

            "initialized_instruments":
                self.initialized_count,

            "coverage_pct":
                self.coverage_pct,

            "fully_initialized":
                self.is_fully_initialized(),

            "fully_initialized_at_ms":
                self.fully_initialized_at_ms,

            "last_message_received_at_ms":
                self.last_message_received_at,

            "last_message_age_ms":
                self.last_message_age_ms(
                    now_ms
                ),

            "initial_instrument_count":
                self.initial_instrument_count,

            "dynamic_instruments_added":
                self.dynamic_instruments_added,

            "last_universe_update_ms":
                self.last_universe_update_ms
        }

    def reset_connection_state(
        self
    ) -> None:
        """
        Invalidate all connection-dependent ticker state.

        The instrument universe itself is intentionally
        preserved across reconnects. This means any strikes
        discovered intraday remain known and are resubscribed.
        """

        self.state.clear()

        self.connected = False

        self.connection_started_at_ms = None
        self.last_message_received_at = None
        self.fully_initialized_at_ms = None

    # --------------------------------------------------
    # Instrument normalization / dynamic universe
    # --------------------------------------------------

    def normalize_instrument_record(
        self,
        record: dict
    ) -> dict | None:
        """
        Convert one Deribit instrument object into the metadata
        format used by the collector, but only if it belongs to
        this collector's locked session expiry.
        """

        if not isinstance(
            record,
            dict
        ):

            return None

        instrument_name = (
            record.get(
                "instrument_name"
            )
        )

        if not instrument_name:
            return None

        expiration_timestamp = (
            record.get(
                "expiration_timestamp"
            )
        )

        try:

            expiration_timestamp = int(
                expiration_timestamp
            )

        except (
            TypeError,
            ValueError
        ):

            return None

        if (
            expiration_timestamp
            != self.session.expiry_timestamp_ms
        ):

            return None

        state = str(
            record.get(
                "state",
                ""
            )
        ).lower()

        if (
            state
            in NON_COLLECTIBLE_STATES
        ):

            return None

        # public/get_instruments returns active instruments
        # with expired=false. For creation notifications,
        # is_active may be absent or may reflect initial state.
        if (
            record.get(
                "is_active"
            )
            is False
        ):

            return None

        metadata = copy.deepcopy(
            record
        )

        # These are custom columns used throughout this project.
        metadata[
            "underlying"
        ] = (
            metadata.get(
                "underlying"
            )
            or metadata.get(
                "base_currency"
            )
        )

        # When querying currency=any or consuming creation events
        # there is no request-bucket label. For options, the
        # settlement currency is the appropriate API bucket
        # representation used by the existing collector.
        metadata[
            "api_currency"
        ] = (
            metadata.get(
                "api_currency"
            )
            or metadata.get(
                "settlement_currency"
            )
        )

        creation_timestamp = (
            metadata.get(
                "creation_timestamp"
            )
        )

        if creation_timestamp is not None:

            try:

                metadata[
                    "creation_time"
                ] = pd.to_datetime(
                    int(
                        creation_timestamp
                    ),
                    unit="ms",
                    utc=True
                )

            except (
                TypeError,
                ValueError
            ):

                pass

        metadata[
            "expiration_time"
        ] = pd.to_datetime(
            expiration_timestamp,
            unit="ms",
            utc=True
        )

        metadata[
            "snapshot_time"
        ] = datetime.now(
            timezone.utc
        )

        return metadata

    def _append_instrument_metadata(
        self,
        metadata: dict,
        source: str
    ) -> bool:
        """
        Add one previously unseen instrument to the live
        session universe. No await occurs here, so membership
        mutation is atomic inside the asyncio event loop.
        """

        instrument_name = str(
            metadata[
                "instrument_name"
            ]
        )

        if (
            instrument_name
            in self.metadata
        ):

            return False

        self.instrument_names.append(
            instrument_name
        )

        self.metadata[
            instrument_name
        ] = metadata

        self.instruments = pd.concat(
            [
                self.instruments,
                pd.DataFrame(
                    [
                        metadata
                    ]
                )
            ],
            ignore_index=True,
            sort=False
        )

        self.dynamic_instruments_added += 1

        self.last_universe_update_ms = int(
            time.time()
            * 1000
        )

        # The previously recorded "fully initialized" instant
        # no longer describes the expanded universe.
        self.fully_initialized_at_ms = None

        print(
            "\nNEW SESSION INSTRUMENT DISCOVERED"
        )

        print(
            f"Instrument: "
            f"{instrument_name}"
        )

        print(
            f"Source:     "
            f"{source}"
        )

        print(
            f"Universe:   "
            f"{self.expected_count} instruments"
        )

        return True

    def _next_request_id(
        self
    ) -> int:

        request_id = (
            self.next_subscription_request_id
        )

        self.next_subscription_request_id += 1

        return request_id

    async def subscribe_channel_batch(
        self,
        websocket,
        channels: list[str],
        label: str
    ) -> None:

        if not channels:
            return

        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "public/subscribe",
            "params": {
                "channels": channels
            }
        }

        async with self.subscription_lock:

            await websocket.send(
                json.dumps(
                    request
                )
            )

        print(
            f"{label}: "
            f"{len(channels)} channel(s)"
        )

    async def add_instrument_records(
        self,
        records: list[dict],
        source: str
    ) -> int:
        """
        Add all newly discovered exact-expiry instruments,
        then subscribe them in batches on the existing
        WebSocket connection.
        """

        new_names = []

        for record in records:

            metadata = (
                self.normalize_instrument_record(
                    record
                )
            )

            if metadata is None:
                continue

            instrument_name = str(
                metadata[
                    "instrument_name"
                ]
            )

            if (
                instrument_name
                in self.metadata
            ):

                continue

            added = (
                self._append_instrument_metadata(
                    metadata,
                    source=source
                )
            )

            if added:

                new_names.append(
                    instrument_name
                )

        if (
            new_names
            and self.connected
            and self.websocket is not None
        ):

            channels = [
                f"incremental_ticker.{name}"
                for name in new_names
            ]

            batches = [
                channels[
                    i:i + SUBSCRIPTION_BATCH_SIZE
                ]
                for i in range(
                    0,
                    len(channels),
                    SUBSCRIPTION_BATCH_SIZE
                )
            ]

            for index, batch in enumerate(
                batches,
                start=1
            ):

                await self.subscribe_channel_batch(
                    self.websocket,
                    batch,
                    (
                        "Dynamic ticker subscription "
                        f"{index}/{len(batches)}"
                    )
                )

        return len(
            new_names
        )

    def fetch_session_universe_from_rest(
        self
    ) -> list[dict]:
        """
        Fetch the currently active option universe in one REST
        request using currency=any, then return only this
        collector's exact locked expiry.
        """

        client = DeribitClient()

        frame = (
            client.get_option_instruments(
                "any",
                expired=False
            )
        )

        if frame.empty:
            return []

        expiration = pd.to_numeric(
            frame[
                "expiration_timestamp"
            ],
            errors="coerce"
        )

        frame = frame.loc[
            expiration
            == self.session.expiry_timestamp_ms
        ].copy()

        if (
            "is_active"
            in frame.columns
        ):

            frame = frame.loc[
                frame[
                    "is_active"
                ]
                == True
            ].copy()

        return frame.to_dict(
            orient="records"
        )

    async def reconcile_session_universe(
        self
    ) -> int:

        self.universe_reconciliations += 1

        try:

            records = await asyncio.to_thread(
                self.fetch_session_universe_from_rest
            )

            added = (
                await self.add_instrument_records(
                    records,
                    source="rest_reconciliation"
                )
            )

            if added:

                print(
                    "\nUNIVERSE RECONCILIATION"
                )

                print(
                    f"Added: "
                    f"{added}"
                )

                print(
                    f"Current universe: "
                    f"{self.expected_count}"
                )

            return added

        except Exception as exc:

            self.universe_reconciliation_errors += 1

            print(
                "\nUniverse reconciliation "
                f"error: {exc}"
            )

            return 0

    async def universe_reconciler(
        self
    ) -> None:

        # Immediate reconciliation closes any discovery gap
        # that occurred while this process/WebSocket was down.
        await self.reconcile_session_universe()

        while not self.stop_event.is_set():

            try:

                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=(
                        UNIVERSE_RECONCILE_INTERVAL_SECONDS
                    )
                )

                break

            except asyncio.TimeoutError:
                pass

            await self.reconcile_session_universe()

    async def handle_instrument_creation(
        self,
        data: dict
    ) -> None:

        self.creation_notifications_received += 1

        metadata = (
            self.normalize_instrument_record(
                data
            )
        )

        if metadata is None:
            return

        self.creation_notifications_matched += 1

        await self.add_instrument_records(
            [
                data
            ],
            source="instrument.creation"
        )

    # --------------------------------------------------
    # Subscription handling
    # --------------------------------------------------

    async def subscribe(
        self,
        websocket
    ) -> None:

        # Subscribe to creation notifications first so the
        # discovery gap is as small as possible.
        await self.subscribe_channel_batch(
            websocket,
            [
                INSTRUMENT_CREATION_CHANNEL
            ],
            "Instrument-creation subscription"
        )

        channels = [
            f"incremental_ticker.{name}"
            for name in self.instrument_names
        ]

        print(
            f"\nSubscribing to "
            f"{len(channels)} option "
            f"ticker channels..."
        )

        batches = [
            channels[
                i:i + SUBSCRIPTION_BATCH_SIZE
            ]
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

            await self.subscribe_channel_batch(
                websocket,
                batch,
                (
                    "Ticker subscription batch "
                    f"{batch_number}/{len(batches)}"
                )
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

        instrument_name = (
            data.get(
                "instrument_name"
            )
        )

        if not instrument_name:
            return

        # Ignore a ticker for an instrument we did not
        # explicitly admit into this locked-expiry universe.
        if (
            instrument_name
            not in self.metadata
        ):

            return

        received_timestamp_ms = int(
            time.time()
            * 1000
        )

        update_type = (
            data.get(
                "type"
            )
        )

        if (
            update_type == "snapshot"
            or instrument_name
            not in self.state
        ):

            ticker_state = (
                copy.deepcopy(
                    data
                )
            )

        else:

            ticker_state = (
                self.state[
                    instrument_name
                ]
            )

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

        if (
            self.fully_initialized_at_ms
            is None
            and self.expected_count > 0
            and self.initialized_count
            == self.expected_count
        ):

            self.fully_initialized_at_ms = (
                received_timestamp_ms
            )

            ready_time = datetime.fromtimestamp(
                received_timestamp_ms / 1000,
                tz=timezone.utc
            )

            print(
                "\nCURRENT CONNECTION "
                "FULLY INITIALIZED"
            )

            print(
                f"Ready at: "
                f"{ready_time.isoformat()}"
            )

            print(
                f"Instruments: "
                f"{self.initialized_count}/"
                f"{self.expected_count}"
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

                raise RuntimeError(
                    "Deribit subscription error: "
                    + json.dumps(
                        message[
                            "error"
                        ]
                    )
                )

            # Subscription confirmation.
            if "id" in message:

                if "result" in message:

                    result = (
                        message[
                            "result"
                        ]
                    )

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

            params = (
                message.get(
                    "params"
                )
            )

            if not params:
                continue

            channel = str(
                params.get(
                    "channel",
                    ""
                )
            )

            data = (
                params.get(
                    "data"
                )
            )

            if channel.startswith(
                "incremental_ticker."
            ):

                if isinstance(
                    data,
                    dict
                ):

                    self.process_ticker(
                        data
                    )

                continue

            if channel.startswith(
                "instrument.creation.option."
            ):

                if isinstance(
                    data,
                    dict
                ):

                    await self.handle_instrument_creation(
                        data
                    )

                continue

    # --------------------------------------------------
    # Feed watchdog
    # --------------------------------------------------

    async def feed_watchdog(
        self
    ) -> None:

        while not self.stop_event.is_set():

            await asyncio.sleep(
                1
            )

            if not self.connected:
                continue

            now_ms = int(
                time.time()
                * 1000
            )

            if (
                self.last_message_received_at
                is None
            ):

                if (
                    self.connection_started_at_ms
                    is not None
                    and now_ms
                    - self.connection_started_at_ms
                    > (
                        INITIAL_MESSAGE_GRACE_SECONDS
                        * 1000
                    )
                ):

                    raise FeedStaleError(
                        "No ticker messages "
                        "received during "
                        "initialization grace period."
                    )

                continue

            age_ms = (
                now_ms
                - self.last_message_received_at
            )

            if (
                age_ms
                > FEED_STALE_TIMEOUT_SECONDS
                * 1000
            ):

                raise FeedStaleError(
                    "Option ticker feed silent "
                    f"for "
                    f"{age_ms / 1000:.1f} "
                    f"seconds."
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
                    timeout=(
                        HEALTH_INTERVAL_SECONDS
                    )
                )

                break

            except asyncio.TimeoutError:
                pass

            now_ms = int(
                time.time()
                * 1000
            )

            ages = []

            for ticker in (
                self.state.values()
            ):

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

                median_age = (
                    statistics.median(
                        ages
                    )
                )

                max_age = max(
                    ages
                )

            else:

                median_age = None
                max_age = None

            print(
                "\n[CACHE HEALTH]"
            )

            print(
                f"Connected:            "
                f"{self.connected}"
            )

            print(
                f"Connection generation:"
                f" {self.connection_count}"
            )

            print(
                f"Expected instruments: "
                f"{self.expected_count}"
            )

            print(
                f"Initialized: "
                f"{self.initialized_count} "
                f"({self.coverage_pct:.1f}%)"
            )

            print(
                f"Fully initialized:    "
                f"{self.is_fully_initialized()}"
            )

            print(
                f"Dynamic instruments: "
                f"{self.dynamic_instruments_added}"
            )

            print(
                f"Creation notifications:"
                f" {self.creation_notifications_received}"
            )

            print(
                f"Universe reconciles:  "
                f"{self.universe_reconciliations}"
            )

            print(
                f"Reconcile errors:     "
                f"{self.universe_reconciliation_errors}"
            )

            print(
                f"Messages received: "
                f"{self.messages_received:,}"
            )

            last_age = (
                self.last_message_age_ms(
                    now_ms
                )
            )

            if last_age is not None:

                print(
                    f"Last message age:     "
                    f"{last_age:,.0f} ms"
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

        # Copy the list so a dynamically discovered strike
        # cannot change iteration membership mid-conversion.
        instrument_names = list(
            self.instrument_names
        )

        rows = []

        for instrument_name in (
            instrument_names
        ):

            ticker = (
                self.state.get(
                    instrument_name,
                    {}
                )
            )

            metadata = (
                self.metadata.get(
                    instrument_name,
                    {}
                )
            )

            greeks = (
                ticker.get(
                    "greeks"
                )
                or {}
            )

            stats = (
                ticker.get(
                    "stats"
                )
                or {}
            )

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

            if (
                exchange_timestamp_ms
                is not None
            ):

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
                    df[
                        "expiration_time"
                    ],
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

        self.reset_connection_state()

        print(
            "\nConnecting to Deribit "
            "WebSocket..."
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

            self.websocket = (
                websocket
            )

            self.connected = True

            self.connection_started_at_ms = int(
                time.time()
                * 1000
            )

            print(
                "WebSocket connected."
            )

            await self.subscribe(
                websocket
            )

            health_task = (
                asyncio.create_task(
                    self.health_reporter()
                )
            )

            receive_task = (
                asyncio.create_task(
                    self.receive_messages(
                        websocket
                    )
                )
            )

            watchdog_task = (
                asyncio.create_task(
                    self.feed_watchdog()
                )
            )

            universe_task = (
                asyncio.create_task(
                    self.universe_reconciler()
                )
            )

            shutdown_task = (
                asyncio.create_task(
                    self.stop_event.wait()
                )
            )

            try:

                done, pending = (
                    await asyncio.wait(
                        {
                            receive_task,
                            watchdog_task,
                            universe_task,
                            shutdown_task
                        },
                        return_when=(
                            asyncio.FIRST_COMPLETED
                        )
                    )
                )

                if (
                    shutdown_task in done
                    and self.stop_event.is_set()
                ):

                    print(
                        "\nShutdown requested."
                    )

                    await websocket.close(
                        code=1000,
                        reason=(
                            "Collector shutdown"
                        )
                    )

                if receive_task in done:

                    receive_task.result()

                    if not self.stop_event.is_set():

                        raise ConnectionError(
                            "Option WebSocket "
                            "receive loop ended "
                            "unexpectedly."
                        )

                if watchdog_task in done:

                    watchdog_task.result()

                if universe_task in done:

                    universe_task.result()

                    if not self.stop_event.is_set():

                        raise RuntimeError(
                            "Universe reconciliation "
                            "task ended unexpectedly."
                        )

            finally:

                tasks = [
                    receive_task,
                    watchdog_task,
                    universe_task,
                    shutdown_task,
                    health_task
                ]

                for task in tasks:

                    if not task.done():

                        task.cancel()

                for task in tasks:

                    try:

                        await task

                    except asyncio.CancelledError:
                        pass

                    except Exception:
                        # Primary completed-task exception is
                        # propagated above.
                        pass

                self.connected = False
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

                # Ticker state from a failed generation is
                # never retained. The dynamic instrument
                # universe itself remains intact.
                self.reset_connection_state()

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
                        timeout=(
                            RECONNECT_DELAY_SECONDS
                        )
                    )

                except asyncio.TimeoutError:
                    pass

        self.connected = False

        print(
            "\nOption-state cache "
            "stopped cleanly."
        )


async def main():

    stop_event = (
        asyncio.Event()
    )

    loop = (
        asyncio.get_running_loop()
    )

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
