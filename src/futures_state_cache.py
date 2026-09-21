from __future__ import annotations

import asyncio
import copy
import json
import statistics
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import websockets


REST_URL = "https://www.deribit.com/api/v2"
WS_URL = "wss://www.deribit.com/ws/api/v2"

BAR_INTERVAL_MS = 5 * 60 * 1000

HEALTH_INTERVAL_SECONDS = 30
RECONNECT_DELAY_SECONDS = 5


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def deep_merge(
    original: dict,
    update: dict
) -> dict:

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


# --------------------------------------------------
# Perpetual discovery
# --------------------------------------------------

def get_future_instruments(
    api_currency: str
) -> pd.DataFrame:

    response = requests.get(
        f"{REST_URL}/public/get_instruments",
        params={
            "currency": api_currency,
            "kind": "future",
            "expired": "false"
        },
        timeout=30
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):

        raise RuntimeError(
            f"Deribit instrument error: "
            f"{payload['error']}"
        )

    result = payload.get(
        "result",
        []
    )

    df = pd.DataFrame(
        result
    )

    if not df.empty:

        df[
            "api_currency"
        ] = api_currency

    return df


def discover_perpetuals(
    option_instruments: pd.DataFrame
) -> pd.DataFrame:
    """
    Select one perpetual future for each underlying
    represented in the current 0DTE option session.

    Preference:
        1. Linear USDC perpetual
        2. Inverse perpetual
        3. Any other perpetual
    """

    if "underlying" not in option_instruments.columns:

        raise RuntimeError(
            "Option instruments do not contain "
            "an 'underlying' column."
        )

    underlyings = sorted(
        option_instruments[
            "underlying"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    api_currencies = sorted(
        option_instruments[
            "api_currency"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    # These are currently the important
    # Deribit futures buckets for our universe.
    for required in [
        "USDC",
        "BTC",
        "ETH"
    ]:

        if required not in api_currencies:
            api_currencies.append(
                required
            )

    frames = []

    for currency in api_currencies:

        try:

            df = get_future_instruments(
                currency
            )

        except Exception as exc:

            print(
                f"Could not load futures "
                f"for {currency}: {exc}"
            )

            continue

        if not df.empty:

            frames.append(
                df
            )

    if not frames:

        raise RuntimeError(
            "No active futures were returned "
            "by Deribit."
        )

    futures = pd.concat(
        frames,
        ignore_index=True
    )

    if "is_active" in futures.columns:

        futures = futures[
            futures["is_active"] == True
        ].copy()

    perpetual_mask = (
        futures[
            "settlement_period"
        ]
        .astype(str)
        .str.lower()
        .eq("perpetual")
    )

    perpetual_mask |= (
        futures[
            "instrument_name"
        ]
        .astype(str)
        .str.contains(
            "PERPETUAL",
            case=False,
            na=False
        )
    )

    futures = futures[
        perpetual_mask
    ].copy()

    futures = futures[
        futures[
            "base_currency"
        ].isin(
            underlyings
        )
    ].copy()

    if futures.empty:

        raise RuntimeError(
            "No perpetual futures matched "
            "the current option underlyings."
        )

    instrument_type = (
        futures[
            "instrument_type"
        ]
        .astype(str)
        .str.lower()
    )

    settlement_currency = (
        futures[
            "settlement_currency"
        ]
        .astype(str)
        .str.upper()
    )

    # Prefer linear USDC perpetuals.
    futures[
        "_selection_rank"
    ] = 2

    futures.loc[
        (
            instrument_type.eq("linear")
            & settlement_currency.eq("USDC")
        ),
        "_selection_rank"
    ] = 0

    futures.loc[
        instrument_type.eq("reversed"),
        "_selection_rank"
    ] = 1

    futures = futures.sort_values(
        [
            "base_currency",
            "_selection_rank",
            "instrument_name"
        ]
    )

    selected = (
        futures
        .drop_duplicates(
            subset=[
                "base_currency"
            ],
            keep="first"
        )
        .copy()
    )

    selected[
        "underlying"
    ] = selected[
        "base_currency"
    ]

    selected = selected.drop(
        columns=[
            "_selection_rank"
        ]
    )

    selected_underlyings = set(
        selected[
            "underlying"
        ]
    )

    missing = [
        underlying
        for underlying in underlyings
        if underlying
        not in selected_underlyings
    ]

    if missing:

        print(
            "\nWARNING: No perpetual found "
            "for these option underlyings:"
        )

        for underlying in missing:

            print(
                f"  {underlying}"
            )

    return selected.reset_index(
        drop=True
    )


# --------------------------------------------------
# Futures state/cache
# --------------------------------------------------

class FuturesStateCache:

    def __init__(
        self,
        instruments: pd.DataFrame,
        stop_event: asyncio.Event
    ):

        self.instruments = (
            instruments
            .drop_duplicates(
                subset=[
                    "instrument_name"
                ]
            )
            .copy()
        )

        self.stop_event = (
            stop_event
        )

        self.instrument_names = (
            self.instruments[
                "instrument_name"
            ]
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

        self.ticker_state: dict[
            str,
            dict
        ] = {}

        self.bars: dict[
            tuple[str, int],
            dict
        ] = {}

        self.seen_trade_ids: dict[
            tuple[str, int],
            set
        ] = {}

        self.messages_received = 0
        self.trades_received = 0
        self.ignored_late_trades = 0

        self.connection_count = 0

        self.collector_started_ms = int(
            time.time() * 1000
        )

        self.last_finalized_end_ms = 0

        self.websocket = None

    # --------------------------------------------------
    # Channels
    # --------------------------------------------------

    def build_channels(
        self
    ) -> list[str]:

        channels = []

        for instrument in self.instrument_names:

            channels.append(
                f"incremental_ticker."
                f"{instrument}"
            )

            channels.append(
                f"trades."
                f"{instrument}."
                f"100ms"
            )

        return channels

    async def subscribe(
        self,
        websocket
    ) -> None:

        channels = (
            self.build_channels()
        )

        request = {
            "jsonrpc": "2.0",
            "id": 2001,
            "method": "public/subscribe",
            "params": {
                "channels": channels
            }
        }

        print(
            f"\nSubscribing to "
            f"{len(channels)} futures "
            f"channels..."
        )

        await websocket.send(
            json.dumps(
                request
            )
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

        update_type = (
            data.get(
                "type"
            )
        )

        if (
            update_type == "snapshot"
            or instrument_name
            not in self.ticker_state
        ):

            state = copy.deepcopy(
                data
            )

        else:

            state = (
                self.ticker_state[
                    instrument_name
                ]
            )

            deep_merge(
                state,
                data
            )

        state[
            "_received_timestamp_ms"
        ] = int(
            time.time() * 1000
        )

        self.ticker_state[
            instrument_name
        ] = state

    # --------------------------------------------------
    # Trade processing
    # --------------------------------------------------

    def process_trade(
        self,
        trade: dict
    ) -> None:

        instrument_name = (
            trade.get(
                "instrument_name"
            )
        )

        if not instrument_name:

            return

        timestamp_ms = trade.get(
            "timestamp"
        )

        price = trade.get(
            "price"
        )

        amount = trade.get(
            "amount"
        )

        if (
            timestamp_ms is None
            or price is None
            or amount is None
        ):
            return

        timestamp_ms = int(
            timestamp_ms
        )

        price = float(
            price
        )

        amount = float(
            amount
        )

        if (
            price <= 0
            or amount < 0
        ):
            return

        # Trades are assigned by DERIBIT exchange
        # timestamp, not VM receipt time.
        bucket_start_ms = (
            timestamp_ms
            - (
                timestamp_ms
                % BAR_INTERVAL_MS
            )
        )

        bucket_end_ms = (
            bucket_start_ms
            + BAR_INTERVAL_MS
        )

        # If a very late notification arrives
        # after we already finalized that bar,
        # do not silently place it into another bar.
        if (
            bucket_end_ms
            <= self.last_finalized_end_ms
        ):

            self.ignored_late_trades += 1
            return

        key = (
            instrument_name,
            bucket_start_ms
        )

        trade_id = (
            trade.get(
                "trade_id"
            )
        )

        trade_seq = (
            trade.get(
                "trade_seq"
            )
        )

        if trade_id is not None:

            dedup_key = (
                "id",
                str(trade_id)
            )

        elif trade_seq is not None:

            dedup_key = (
                "seq",
                int(trade_seq)
            )

        else:

            dedup_key = (
                timestamp_ms,
                price,
                amount,
                trade.get(
                    "direction"
                )
            )

        seen = (
            self.seen_trade_ids
            .setdefault(
                key,
                set()
            )
        )

        if dedup_key in seen:
            return

        seen.add(
            dedup_key
        )

        metadata = (
            self.metadata.get(
                instrument_name,
                {}
            )
        )

        instrument_type = str(
            metadata.get(
                "instrument_type",
                ""
            )
        ).lower()

        # Deribit linear futures:
        # amount is in underlying coin.
        #
        # Deribit inverse futures:
        # amount is in USD units.
        if instrument_type == "linear":

            volume_base = (
                amount
            )

            volume_usd = (
                amount
                * price
            )

            native_unit = (
                metadata.get(
                    "base_currency"
                )
            )

        else:

            volume_usd = (
                amount
            )

            volume_base = (
                amount
                / price
            )

            native_unit = (
                "USD"
            )

        bar = self.bars.get(
            key
        )

        if bar is None:

            bar = {
                "open": price,
                "high": price,
                "low": price,
                "close": price,

                "first_trade_timestamp_ms":
                    timestamp_ms,

                "last_trade_timestamp_ms":
                    timestamp_ms,

                "volume_native": 0.0,
                "volume_base": 0.0,
                "volume_usd": 0.0,

                "buy_volume_native": 0.0,
                "sell_volume_native": 0.0,

                "buy_volume_base": 0.0,
                "sell_volume_base": 0.0,

                "buy_volume_usd": 0.0,
                "sell_volume_usd": 0.0,

                "trade_count": 0,

                "vwap_numerator": 0.0,

                "native_volume_unit":
                    native_unit
            }

            self.bars[
                key
            ] = bar

        # Open/close determined by exchange
        # timestamps rather than arrival order.
        if (
            timestamp_ms
            < bar[
                "first_trade_timestamp_ms"
            ]
        ):

            bar[
                "first_trade_timestamp_ms"
            ] = timestamp_ms

            bar[
                "open"
            ] = price

        if (
            timestamp_ms
            >= bar[
                "last_trade_timestamp_ms"
            ]
        ):

            bar[
                "last_trade_timestamp_ms"
            ] = timestamp_ms

            bar[
                "close"
            ] = price

        bar[
            "high"
        ] = max(
            bar["high"],
            price
        )

        bar[
            "low"
        ] = min(
            bar["low"],
            price
        )

        bar[
            "volume_native"
        ] += amount

        bar[
            "volume_base"
        ] += volume_base

        bar[
            "volume_usd"
        ] += volume_usd

        # VWAP uses standardized
        # base-coin-equivalent volume.
        bar[
            "vwap_numerator"
        ] += (
            price
            * volume_base
        )

        direction = str(
            trade.get(
                "direction",
                ""
            )
        ).lower()

        if direction == "buy":

            bar[
                "buy_volume_native"
            ] += amount

            bar[
                "buy_volume_base"
            ] += volume_base

            bar[
                "buy_volume_usd"
            ] += volume_usd

        elif direction == "sell":

            bar[
                "sell_volume_native"
            ] += amount

            bar[
                "sell_volume_base"
            ] += volume_base

            bar[
                "sell_volume_usd"
            ] += volume_usd

        bar[
            "trade_count"
        ] += 1

        self.trades_received += 1

    # --------------------------------------------------
    # Boundary ticker state
    # --------------------------------------------------

    def capture_boundary_state(
        self,
        snapshot_time: datetime
    ) -> dict[str, dict]:

        snapshot_ms = int(
            snapshot_time.timestamp()
            * 1000
        )

        result = {}

        for instrument in self.instrument_names:

            ticker = copy.deepcopy(
                self.ticker_state.get(
                    instrument,
                    {}
                )
            )

            bid = ticker.get(
                "best_bid_price"
            )

            ask = ticker.get(
                "best_ask_price"
            )

            mid = None

            if (
                bid is not None
                and ask is not None
                and bid > 0
                and ask > 0
            ):

                mid = (
                    bid + ask
                ) / 2

            exchange_timestamp_ms = (
                ticker.get(
                    "timestamp"
                )
            )

            ticker_age_ms = None

            if (
                exchange_timestamp_ms
                is not None
            ):

                ticker_age_ms = (
                    snapshot_ms
                    - exchange_timestamp_ms
                )

            result[
                instrument
            ] = {
                "best_bid_price":
                    bid,

                "best_bid_amount":
                    ticker.get(
                        "best_bid_amount"
                    ),

                "best_ask_price":
                    ask,

                "best_ask_amount":
                    ticker.get(
                        "best_ask_amount"
                    ),

                "mid_price":
                    mid,

                "last_price":
                    ticker.get(
                        "last_price"
                    ),

                "mark_price":
                    ticker.get(
                        "mark_price"
                    ),

                "index_price":
                    ticker.get(
                        "index_price"
                    ),

                "underlying_price":
                    ticker.get(
                        "underlying_price"
                    ),

                "open_interest":
                    ticker.get(
                        "open_interest"
                    ),

                "ticker_exchange_timestamp_ms":
                    exchange_timestamp_ms,

                "ticker_received_timestamp_ms":
                    ticker.get(
                        "_received_timestamp_ms"
                    ),

                "ticker_age_ms":
                    ticker_age_ms
            }

        return result

    # --------------------------------------------------
    # Finalize 5-minute bar
    # --------------------------------------------------

    def finalize_bar(
        self,
        bar_end: datetime,
        boundary_state: dict[str, dict]
    ) -> pd.DataFrame:

        bar_end = bar_end.astimezone(
            timezone.utc
        )

        bar_end_ms = int(
            bar_end.timestamp()
            * 1000
        )

        bar_start_ms = (
            bar_end_ms
            - BAR_INTERVAL_MS
        )

        bar_start = datetime.fromtimestamp(
            bar_start_ms / 1000,
            tz=timezone.utc
        )

        rows = []

        for instrument in self.instrument_names:

            key = (
                instrument,
                bar_start_ms
            )

            bar = self.bars.pop(
                key,
                None
            )

            self.seen_trade_ids.pop(
                key,
                None
            )

            metadata = (
                self.metadata.get(
                    instrument,
                    {}
                )
            )

            boundary = (
                boundary_state.get(
                    instrument,
                    {}
                )
            )

            if bar is None:

                volume_base = 0.0
                vwap = None

                open_price = None
                high_price = None
                low_price = None
                close_price = None

                volume_native = 0.0
                volume_usd = 0.0

                buy_volume_native = 0.0
                sell_volume_native = 0.0

                buy_volume_base = 0.0
                sell_volume_base = 0.0

                buy_volume_usd = 0.0
                sell_volume_usd = 0.0

                trade_count = 0

                first_trade_timestamp_ms = None
                last_trade_timestamp_ms = None

                instrument_type = str(
                    metadata.get(
                        "instrument_type",
                        ""
                    )
                ).lower()

                native_volume_unit = (
                    metadata.get(
                        "base_currency"
                    )
                    if instrument_type
                    == "linear"
                    else "USD"
                )

            else:

                volume_base = (
                    bar[
                        "volume_base"
                    ]
                )

                vwap = (
                    bar[
                        "vwap_numerator"
                    ]
                    / volume_base
                    if volume_base > 0
                    else None
                )

                open_price = bar[
                    "open"
                ]

                high_price = bar[
                    "high"
                ]

                low_price = bar[
                    "low"
                ]

                close_price = bar[
                    "close"
                ]

                volume_native = bar[
                    "volume_native"
                ]

                volume_usd = bar[
                    "volume_usd"
                ]

                buy_volume_native = bar[
                    "buy_volume_native"
                ]

                sell_volume_native = bar[
                    "sell_volume_native"
                ]

                buy_volume_base = bar[
                    "buy_volume_base"
                ]

                sell_volume_base = bar[
                    "sell_volume_base"
                ]

                buy_volume_usd = bar[
                    "buy_volume_usd"
                ]

                sell_volume_usd = bar[
                    "sell_volume_usd"
                ]

                trade_count = bar[
                    "trade_count"
                ]

                first_trade_timestamp_ms = (
                    bar[
                        "first_trade_timestamp_ms"
                    ]
                )

                last_trade_timestamp_ms = (
                    bar[
                        "last_trade_timestamp_ms"
                    ]
                )

                native_volume_unit = (
                    bar[
                        "native_volume_unit"
                    ]
                )

            stream_complete = (
                self.collector_started_ms
                <= bar_start_ms
            )

            rows.append(
                {
                    "bar_start":
                        bar_start,

                    "bar_end":
                        bar_end,

                    "underlying":
                        metadata.get(
                            "base_currency"
                        ),

                    "instrument_name":
                        instrument,

                    "api_currency":
                        metadata.get(
                            "api_currency"
                        ),

                    "instrument_type":
                        metadata.get(
                            "instrument_type"
                        ),

                    "settlement_currency":
                        metadata.get(
                            "settlement_currency"
                        ),

                    "quote_currency":
                        metadata.get(
                            "quote_currency"
                        ),

                    "contract_size":
                        metadata.get(
                            "contract_size"
                        ),

                    "native_volume_unit":
                        native_volume_unit,

                    "open":
                        open_price,

                    "high":
                        high_price,

                    "low":
                        low_price,

                    "close":
                        close_price,

                    "vwap":
                        vwap,

                    "volume_native":
                        volume_native,

                    "volume_base":
                        volume_base,

                    "volume_usd":
                        volume_usd,

                    "buy_volume_native":
                        buy_volume_native,

                    "sell_volume_native":
                        sell_volume_native,

                    "buy_volume_base":
                        buy_volume_base,

                    "sell_volume_base":
                        sell_volume_base,

                    "buy_volume_usd":
                        buy_volume_usd,

                    "sell_volume_usd":
                        sell_volume_usd,

                    "trade_count":
                        trade_count,

                    "first_trade_timestamp_ms":
                        first_trade_timestamp_ms,

                    "last_trade_timestamp_ms":
                        last_trade_timestamp_ms,

                    "best_bid_price":
                        boundary.get(
                            "best_bid_price"
                        ),

                    "best_bid_amount":
                        boundary.get(
                            "best_bid_amount"
                        ),

                    "best_ask_price":
                        boundary.get(
                            "best_ask_price"
                        ),

                    "best_ask_amount":
                        boundary.get(
                            "best_ask_amount"
                        ),

                    "mid_price":
                        boundary.get(
                            "mid_price"
                        ),

                    "last_price":
                        boundary.get(
                            "last_price"
                        ),

                    "mark_price":
                        boundary.get(
                            "mark_price"
                        ),

                    "index_price":
                        boundary.get(
                            "index_price"
                        ),

                    "underlying_price":
                        boundary.get(
                            "underlying_price"
                        ),

                    "open_interest":
                        boundary.get(
                            "open_interest"
                        ),

                    "ticker_exchange_timestamp_ms":
                        boundary.get(
                            "ticker_exchange_timestamp_ms"
                        ),

                    "ticker_received_timestamp_ms":
                        boundary.get(
                            "ticker_received_timestamp_ms"
                        ),

                    "ticker_age_ms":
                        boundary.get(
                            "ticker_age_ms"
                        ),

                    # False for the first partial bar
                    # after collector startup.
                    "stream_complete":
                        stream_complete,

                    "source":
                        "deribit_ws_100ms"
                }
            )

        self.last_finalized_end_ms = max(
            self.last_finalized_end_ms,
            bar_end_ms
        )

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------
    # Receive WebSocket messages
    # --------------------------------------------------

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
                    "\nDeribit futures error:"
                )

                print(
                    json.dumps(
                        message[
                            "error"
                        ],
                        indent=2
                    )
                )

                continue

            if "id" in message:

                if "result" in message:

                    print(
                        "Futures subscriptions "
                        "confirmed."
                    )

                continue

            params = message.get(
                "params"
            )

            if not params:
                continue

            channel = str(
                params.get(
                    "channel",
                    ""
                )
            )

            data = params.get(
                "data"
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

            elif channel.startswith(
                "trades."
            ):

                if isinstance(
                    data,
                    dict
                ):

                    trades = [
                        data
                    ]

                elif isinstance(
                    data,
                    list
                ):

                    trades = data

                else:

                    trades = []

                for trade in trades:

                    if isinstance(
                        trade,
                        dict
                    ):

                        self.process_trade(
                            trade
                        )

            self.messages_received += 1

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

            expected = len(
                self.instrument_names
            )

            initialized = len(
                self.ticker_state
            )

            initialized_pct = (
                initialized
                / expected
                * 100
                if expected
                else 0
            )

            now_ms = int(
                time.time()
                * 1000
            )

            ages = []

            for ticker in (
                self.ticker_state.values()
            ):

                timestamp_ms = ticker.get(
                    "timestamp"
                )

                if timestamp_ms is not None:

                    ages.append(
                        now_ms
                        - timestamp_ms
                    )

            print(
                "\n[FUTURES CACHE HEALTH]"
            )

            print(
                f"Perpetuals:          "
                f"{expected}"
            )

            print(
                f"Tickers initialized: "
                f"{initialized}/{expected} "
                f"({initialized_pct:.1f}%)"
            )

            print(
                f"Trades received:     "
                f"{self.trades_received:,}"
            )

            print(
                f"Late trades ignored: "
                f"{self.ignored_late_trades:,}"
            )

            if ages:

                print(
                    f"Median ticker age:   "
                    f"{statistics.median(ages):,.0f} ms"
                )

    # --------------------------------------------------
    # Connection lifecycle
    # --------------------------------------------------

    async def run_connection(
        self
    ) -> None:

        self.connection_count += 1

        print(
            "\nConnecting futures "
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

            print(
                "Futures WebSocket "
                "connected."
            )

            await self.subscribe(
                websocket
            )

            receive_task = (
                asyncio.create_task(
                    self.receive_messages(
                        websocket
                    )
                )
            )

            health_task = (
                asyncio.create_task(
                    self.health_reporter()
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

                    await websocket.close(
                        code=1000,
                        reason=(
                            "Futures collector shutdown"
                        )
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
                    f"\nFutures WebSocket "
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

        print(
            "\nFutures state cache "
            "stopped cleanly."
        )