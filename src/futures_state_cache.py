from __future__ import annotations

import asyncio
import copy
import json
import statistics
import time

import pandas as pd
import requests
import websockets


# --------------------------------------------------
# Configuration
# --------------------------------------------------

REST_URL = "https://www.deribit.com/api/v2"
WS_URL = "wss://www.deribit.com/ws/api/v2"

BAR_INTERVAL_MS = 5 * 60 * 1000

BOOK_DEPTH = 20
BOOK_GROUP = "none"
BOOK_INTERVAL = "100ms"

HEALTH_INTERVAL_SECONDS = 30
RECONNECT_DELAY_SECONDS = 5


BOOK_METRIC_NAMES = (
    "spread",
    "bid_depth_5",
    "ask_depth_5",
    "obi_5",
    "bid_depth_10",
    "ask_depth_10",
    "obi_10",
    "bid_depth_20",
    "ask_depth_20",
    "obi_20",
    "microprice",
)


# --------------------------------------------------
# Exceptions
# --------------------------------------------------

class BookSequenceGap(RuntimeError):
    """
    Raised when Deribit order-book sequencing
    indicates that an update was missed.

    The connection is deliberately restarted so
    all books receive fresh snapshots.
    """
    pass


# --------------------------------------------------
# Generic helpers
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
# Futures discovery
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
        3. Any remaining perpetual
    """

    if (
        "underlying"
        not in option_instruments.columns
    ):

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

    for required in [
        "USDC",
        "BTC",
        "ETH"
    ]:

        if (
            required
            not in api_currencies
        ):

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
            "No active futures were "
            "returned by Deribit."
        )

    futures = pd.concat(
        frames,
        ignore_index=True
    )

    if (
        "is_active"
        in futures.columns
    ):

        futures = futures[
            futures[
                "is_active"
            ] == True
        ].copy()

    perpetual_mask = (
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

    if (
        "settlement_period"
        in futures.columns
    ):

        perpetual_mask |= (
            futures[
                "settlement_period"
            ]
            .astype(str)
            .str.lower()
            .eq("perpetual")
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

    futures[
        "_selection_rank"
    ] = 2

    # Prefer linear USDC.
    futures.loc[
        (
            instrument_type.eq(
                "linear"
            )
            & settlement_currency.eq(
                "USDC"
            )
        ),
        "_selection_rank"
    ] = 0

    # Second preference: inverse/reversed.
    futures.loc[
        instrument_type.eq(
            "reversed"
        ),
        "_selection_rank"
    ] = 1

    selected = (
        futures
        .sort_values(
            [
                "base_currency",
                "_selection_rank",
                "instrument_name"
            ]
        )
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
        ].tolist()
    )

    missing = [
        underlying
        for underlying
        in underlyings
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
# Futures state cache
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

        # Latest ticker state only.
        self.ticker_state: dict[
            str,
            dict
        ] = {}

        # Latest L2 top-20 state only.
        self.order_book_state: dict[
            str,
            dict
        ] = {}

        # Current OHLCV/trade accumulators.
        self.bars: dict[
            tuple[str, int],
            dict
        ] = {}

        self.seen_trade_ids: dict[
            tuple[str, int],
            set
        ] = {}

        # Time-weighted L2 statistics.
        self.book_metric_state: dict[
            str,
            dict
        ] = {}

        self.book_bar_accumulators: dict[
            tuple[str, int],
            dict
        ] = {}

        # Bars affected by a WebSocket outage /
        # sequence-loss event.
        self.incomplete_bar_keys: set[
            tuple[str, int]
        ] = set()

        self.messages_received = 0
        self.trades_received = 0
        self.book_messages_received = 0

        self.book_sequence_gaps = 0
        self.ignored_late_trades = 0

        self.connection_count = 0

        self.collector_started_ms = int(
            time.time()
            * 1000
        )

        self.last_finalized_end_ms = 0

        self.websocket = None

    # --------------------------------------------------
    # Subscription channels
    # --------------------------------------------------

    def book_channel(
        self,
        instrument: str
    ) -> str:

        return (
            f"book."
            f"{instrument}."
            f"{BOOK_GROUP}."
            f"{BOOK_DEPTH}."
            f"{BOOK_INTERVAL}"
        )

    def build_channels(
        self
    ) -> list[str]:

        channels = []

        for instrument in (
            self.instrument_names
        ):

            channels.append(
                f"incremental_ticker."
                f"{instrument}"
            )

            channels.append(
                f"trades."
                f"{instrument}."
                f"100ms"
            )

            channels.append(
                self.book_channel(
                    instrument
                )
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
            f"market-data channels..."
        )

        await websocket.send(
            json.dumps(
                request
            )
        )

    # --------------------------------------------------
    # Reconnect handling
    # --------------------------------------------------

    def current_bucket_start_ms(
        self
    ) -> int:

        now_ms = int(
            time.time()
            * 1000
        )

        return (
            now_ms
            - (
                now_ms
                % BAR_INTERVAL_MS
            )
        )

    def mark_current_bar_incomplete(
        self
    ) -> None:

        bucket_start_ms = (
            self.current_bucket_start_ms()
        )

        for instrument in (
            self.instrument_names
        ):

            self.incomplete_bar_keys.add(
                (
                    instrument,
                    bucket_start_ms
                )
            )

    def reset_realtime_state(
        self
    ) -> None:
        """
        Preserve trade/bar accumulators but force
        fresh ticker/book initialization after a
        connection restart.
        """

        self.ticker_state.clear()
        self.order_book_state.clear()
        self.book_metric_state.clear()

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
            time.time()
            * 1000
        )

        self.ticker_state[
            instrument_name
        ] = state

    # --------------------------------------------------
    # L2 order-book helpers
    # --------------------------------------------------

    def _new_book_state(
        self
    ) -> dict:

        return {
            "bids": {},
            "asks": {},
            "change_id": None,
            "timestamp": None,
            "_received_timestamp_ms": None,
            "valid": False
        }

    def _parse_book_level(
        self,
        level
    ) -> tuple[str, float, float] | None:
        """
        Supports both:

            ["new", price, amount]

        and defensive support for:

            [price, amount]
        """

        if (
            not isinstance(
                level,
                (list, tuple)
            )
        ):

            return None

        if len(level) >= 3:

            action = str(
                level[0]
            ).lower()

            try:

                price = float(
                    level[1]
                )

                amount = float(
                    level[2]
                )

            except (
                TypeError,
                ValueError
            ):

                return None

            return (
                action,
                price,
                amount
            )

        if len(level) == 2:

            try:

                price = float(
                    level[0]
                )

                amount = float(
                    level[1]
                )

            except (
                TypeError,
                ValueError
            ):

                return None

            return (
                "new",
                price,
                amount
            )

        return None

    def _apply_book_levels(
        self,
        levels: list,
        side: dict
    ) -> None:

        for raw_level in levels:

            parsed = (
                self._parse_book_level(
                    raw_level
                )
            )

            if parsed is None:
                continue

            action, price, amount = (
                parsed
            )

            if (
                action == "delete"
                or amount <= 0
            ):

                side.pop(
                    price,
                    None
                )

            else:

                side[
                    price
                ] = amount

    def _trim_book(
        self,
        state: dict
    ) -> None:

        bid_prices = sorted(
            state[
                "bids"
            ].keys(),
            reverse=True
        )[
            :BOOK_DEPTH
        ]

        ask_prices = sorted(
            state[
                "asks"
            ].keys()
        )[
            :BOOK_DEPTH
        ]

        state[
            "bids"
        ] = {
            price:
                state[
                    "bids"
                ][price]

            for price
            in bid_prices
        }

        state[
            "asks"
        ] = {
            price:
                state[
                    "asks"
                ][price]

            for price
            in ask_prices
        }

    def _sorted_book_levels(
        self,
        instrument: str
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[float, float]]
    ]:

        state = (
            self.order_book_state.get(
                instrument
            )
        )

        if (
            not state
            or not state.get(
                "valid",
                False
            )
        ):

            return (
                [],
                []
            )

        bids = sorted(
            state[
                "bids"
            ].items(),
            key=lambda item:
                item[0],
            reverse=True
        )[
            :BOOK_DEPTH
        ]

        asks = sorted(
            state[
                "asks"
            ].items(),
            key=lambda item:
                item[0]
        )[
            :BOOK_DEPTH
        ]

        return (
            bids,
            asks
        )

    def _calculate_book_metrics(
        self,
        instrument: str
    ) -> dict | None:

        bids, asks = (
            self._sorted_book_levels(
                instrument
            )
        )

        if (
            not bids
            or not asks
        ):

            return None

        best_bid = bids[0][0]
        best_ask = asks[0][0]

        bid_size_1 = bids[0][1]
        ask_size_1 = asks[0][1]

        spread = (
            best_ask
            - best_bid
        )

        denominator = (
            bid_size_1
            + ask_size_1
        )

        microprice = None

        if denominator > 0:

            microprice = (
                (
                    best_ask
                    * bid_size_1
                )
                +
                (
                    best_bid
                    * ask_size_1
                )
            ) / denominator

        result = {
            "spread":
                spread,

            "microprice":
                microprice
        }

        for depth in [
            5,
            10,
            20
        ]:

            bid_depth = sum(
                amount
                for _, amount
                in bids[
                    :depth
                ]
            )

            ask_depth = sum(
                amount
                for _, amount
                in asks[
                    :depth
                ]
            )

            total_depth = (
                bid_depth
                + ask_depth
            )

            obi = None

            if total_depth > 0:

                obi = (
                    bid_depth
                    - ask_depth
                ) / total_depth

            result[
                f"bid_depth_{depth}"
            ] = bid_depth

            result[
                f"ask_depth_{depth}"
            ] = ask_depth

            result[
                f"obi_{depth}"
            ] = obi

        return result

    # --------------------------------------------------
    # Time-weighted L2 statistics
    # --------------------------------------------------

    def _new_book_accumulator(
        self
    ) -> dict:

        return {
            "valid_duration_ms": 0.0,

            "weighted_sum": {
                metric: 0.0
                for metric
                in BOOK_METRIC_NAMES
            },

            "weighted_duration": {
                metric: 0.0
                for metric
                in BOOK_METRIC_NAMES
            },

            "update_count": 0,

            "min_obi_20": None,
            "max_obi_20": None
        }

    def _get_book_accumulator(
        self,
        instrument: str,
        bucket_start_ms: int
    ) -> dict:

        key = (
            instrument,
            bucket_start_ms
        )

        if (
            key
            not in self.book_bar_accumulators
        ):

            self.book_bar_accumulators[
                key
            ] = (
                self._new_book_accumulator()
            )

        return (
            self.book_bar_accumulators[
                key
            ]
        )

    def _accumulate_book_metrics(
        self,
        instrument: str,
        start_ms: int,
        end_ms: int,
        metrics: dict
    ) -> None:

        if end_ms <= start_ms:
            return

        cursor = (
            start_ms
        )

        while cursor < end_ms:

            bucket_start_ms = (
                cursor
                - (
                    cursor
                    % BAR_INTERVAL_MS
                )
            )

            bucket_end_ms = (
                bucket_start_ms
                + BAR_INTERVAL_MS
            )

            segment_end = min(
                end_ms,
                bucket_end_ms
            )

            duration = (
                segment_end
                - cursor
            )

            accumulator = (
                self._get_book_accumulator(
                    instrument,
                    bucket_start_ms
                )
            )

            accumulator[
                "valid_duration_ms"
            ] += duration

            for metric_name in (
                BOOK_METRIC_NAMES
            ):

                value = (
                    metrics.get(
                        metric_name
                    )
                )

                if value is None:
                    continue

                accumulator[
                    "weighted_sum"
                ][
                    metric_name
                ] += (
                    value
                    * duration
                )

                accumulator[
                    "weighted_duration"
                ][
                    metric_name
                ] += duration

            cursor = (
                segment_end
            )

    def _advance_book_stats(
        self,
        instrument: str,
        target_timestamp_ms: int
    ) -> None:

        state = (
            self.book_metric_state.get(
                instrument
            )
        )

        if state is None:
            return

        previous_timestamp_ms = (
            state[
                "last_timestamp_ms"
            ]
        )

        if (
            target_timestamp_ms
            <= previous_timestamp_ms
        ):

            return

        metrics = (
            state.get(
                "last_metrics"
            )
        )

        if metrics is not None:

            self._accumulate_book_metrics(
                instrument,
                previous_timestamp_ms,
                target_timestamp_ms,
                metrics
            )

        state[
            "last_timestamp_ms"
        ] = target_timestamp_ms

    def _record_book_observation(
        self,
        instrument: str,
        timestamp_ms: int,
        metrics: dict
    ) -> None:

        bucket_start_ms = (
            timestamp_ms
            - (
                timestamp_ms
                % BAR_INTERVAL_MS
            )
        )

        accumulator = (
            self._get_book_accumulator(
                instrument,
                bucket_start_ms
            )
        )

        accumulator[
            "update_count"
        ] += 1

        obi_20 = (
            metrics.get(
                "obi_20"
            )
        )

        if obi_20 is not None:

            current_min = (
                accumulator[
                    "min_obi_20"
                ]
            )

            current_max = (
                accumulator[
                    "max_obi_20"
                ]
            )

            if (
                current_min is None
                or obi_20 < current_min
            ):

                accumulator[
                    "min_obi_20"
                ] = obi_20

            if (
                current_max is None
                or obi_20 > current_max
            ):

                accumulator[
                    "max_obi_20"
                ] = obi_20

    # --------------------------------------------------
    # L2 message processing
    # --------------------------------------------------

    def process_book(
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

        received_timestamp_ms = int(
            time.time()
            * 1000
        )

        timestamp_ms = int(
            data.get(
                "timestamp"
            )
            or received_timestamp_ms
        )

        update_type = str(
            data.get(
                "type",
                ""
            )
        ).lower()

        state = (
            self.order_book_state.get(
                instrument_name
            )
        )

        if state is None:

            state = (
                self._new_book_state()
            )

            self.order_book_state[
                instrument_name
            ] = state

        # Previous valid book remained in effect
        # up to this exchange timestamp.
        self._advance_book_stats(
            instrument_name,
            timestamp_ms
        )

        if update_type == "snapshot":

            state[
                "bids"
            ] = {}

            state[
                "asks"
            ] = {}

            self._apply_book_levels(
                data.get(
                    "bids",
                    []
                ),
                state[
                    "bids"
                ]
            )

            self._apply_book_levels(
                data.get(
                    "asks",
                    []
                ),
                state[
                    "asks"
                ]
            )

            state[
                "valid"
            ] = True

        elif update_type == "change":

            if not state.get(
                "valid",
                False
            ):

                self.book_sequence_gaps += 1

                raise BookSequenceGap(
                    f"No valid L2 snapshot "
                    f"for {instrument_name}"
                )

            previous_change_id = (
                data.get(
                    "prev_change_id"
                )
            )

            last_change_id = (
                state.get(
                    "change_id"
                )
            )

            if (
                previous_change_id
                is not None
                and last_change_id
                is not None
                and previous_change_id
                != last_change_id
            ):

                state[
                    "valid"
                ] = False

                self.book_metric_state.pop(
                    instrument_name,
                    None
                )

                self.book_sequence_gaps += 1

                raise BookSequenceGap(
                    f"L2 sequence gap "
                    f"for {instrument_name}: "
                    f"expected prev_change_id "
                    f"{last_change_id}, got "
                    f"{previous_change_id}"
                )

            self._apply_book_levels(
                data.get(
                    "bids",
                    []
                ),
                state[
                    "bids"
                ]
            )

            self._apply_book_levels(
                data.get(
                    "asks",
                    []
                ),
                state[
                    "asks"
                ]
            )

        else:

            return

        self._trim_book(
            state
        )

        state[
            "change_id"
        ] = data.get(
            "change_id"
        )

        state[
            "timestamp"
        ] = timestamp_ms

        state[
            "_received_timestamp_ms"
        ] = received_timestamp_ms

        metrics = (
            self._calculate_book_metrics(
                instrument_name
            )
        )

        if metrics is None:

            state[
                "valid"
            ] = False

            self.book_metric_state.pop(
                instrument_name,
                None
            )

            return

        state[
            "valid"
        ] = True

        self.book_metric_state[
            instrument_name
        ] = {
            "last_timestamp_ms":
                timestamp_ms,

            "last_metrics":
                metrics
        }

        self._record_book_observation(
            instrument_name,
            timestamp_ms,
            metrics
        )

        self.book_messages_received += 1

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

        timestamp_ms = (
            trade.get(
                "timestamp"
            )
        )

        price = (
            trade.get(
                "price"
            )
        )

        amount = (
            trade.get(
                "amount"
            )
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
                str(
                    trade_id
                )
            )

        elif trade_seq is not None:

            dedup_key = (
                "seq",
                int(
                    trade_seq
                )
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

        # Linear futures:
        # amount is base-coin quantity.
        #
        # Reversed/inverse futures:
        # amount is USD quantity.
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

        bar = (
            self.bars.get(
                key
            )
        )

        if bar is None:

            bar = {
                "open":
                    price,

                "high":
                    price,

                "low":
                    price,

                "close":
                    price,

                "first_trade_timestamp_ms":
                    timestamp_ms,

                "last_trade_timestamp_ms":
                    timestamp_ms,

                "volume_native":
                    0.0,

                "volume_base":
                    0.0,

                "volume_usd":
                    0.0,

                "buy_volume_native":
                    0.0,

                "sell_volume_native":
                    0.0,

                "buy_volume_base":
                    0.0,

                "sell_volume_base":
                    0.0,

                "buy_volume_usd":
                    0.0,

                "sell_volume_usd":
                    0.0,

                "trade_count":
                    0,

                "vwap_numerator":
                    0.0,

                "native_volume_unit":
                    native_unit
            }

            self.bars[
                key
            ] = bar

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
            bar[
                "high"
            ],
            price
        )

        bar[
            "low"
        ] = min(
            bar[
                "low"
            ],
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
    # L2 boundary snapshot
    # --------------------------------------------------

    def _book_boundary_fields(
        self,
        instrument: str,
        snapshot_ms: int
    ) -> dict:

        state = (
            self.order_book_state.get(
                instrument
            )
        )

        fields = {
            "book_valid":
                False,

            "book_timestamp_ms":
                None,

            "book_received_timestamp_ms":
                None,

            "book_age_ms":
                None,

            "book_change_id":
                None
        }

        for level in range(
            1,
            BOOK_DEPTH + 1
        ):

            suffix = (
                f"{level:02d}"
            )

            fields[
                f"bid_px_{suffix}"
            ] = None

            fields[
                f"bid_sz_{suffix}"
            ] = None

            fields[
                f"ask_px_{suffix}"
            ] = None

            fields[
                f"ask_sz_{suffix}"
            ] = None

        for name in [
            "spread",
            "microprice",
            "bid_depth_5",
            "ask_depth_5",
            "obi_5",
            "bid_depth_10",
            "ask_depth_10",
            "obi_10",
            "bid_depth_20",
            "ask_depth_20",
            "obi_20"
        ]:

            fields[
                name
            ] = None

        if (
            not state
            or not state.get(
                "valid",
                False
            )
        ):

            return fields

        bids, asks = (
            self._sorted_book_levels(
                instrument
            )
        )

        metrics = (
            self._calculate_book_metrics(
                instrument
            )
        )

        if metrics is None:

            return fields

        book_timestamp_ms = (
            state.get(
                "timestamp"
            )
        )

        fields[
            "book_valid"
        ] = True

        fields[
            "book_timestamp_ms"
        ] = book_timestamp_ms

        fields[
            "book_received_timestamp_ms"
        ] = state.get(
            "_received_timestamp_ms"
        )

        fields[
            "book_change_id"
        ] = state.get(
            "change_id"
        )

        if (
            book_timestamp_ms
            is not None
        ):

            fields[
                "book_age_ms"
            ] = (
                snapshot_ms
                - book_timestamp_ms
            )

        for index, (
            price,
            amount
        ) in enumerate(
            bids[
                :BOOK_DEPTH
            ],
            start=1
        ):

            suffix = (
                f"{index:02d}"
            )

            fields[
                f"bid_px_{suffix}"
            ] = price

            fields[
                f"bid_sz_{suffix}"
            ] = amount

        for index, (
            price,
            amount
        ) in enumerate(
            asks[
                :BOOK_DEPTH
            ],
            start=1
        ):

            suffix = (
                f"{index:02d}"
            )

            fields[
                f"ask_px_{suffix}"
            ] = price

            fields[
                f"ask_sz_{suffix}"
            ] = amount

        fields.update(
            metrics
        )

        return fields

    # --------------------------------------------------
    # Ticker + L2 boundary state
    # --------------------------------------------------

    def capture_boundary_state(
        self,
        snapshot_time
    ) -> dict[str, dict]:

        snapshot_ms = int(
            snapshot_time.timestamp()
            * 1000
        )

        result = {}

        for instrument in (
            self.instrument_names
        ):

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
                    bid
                    + ask
                ) / 2

            ticker_timestamp_ms = (
                ticker.get(
                    "timestamp"
                )
            )

            ticker_age_ms = None

            if (
                ticker_timestamp_ms
                is not None
            ):

                ticker_age_ms = (
                    snapshot_ms
                    - ticker_timestamp_ms
                )

            state = {
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
                    ticker_timestamp_ms,

                "ticker_received_timestamp_ms":
                    ticker.get(
                        "_received_timestamp_ms"
                    ),

                "ticker_age_ms":
                    ticker_age_ms
            }

            state.update(
                self._book_boundary_fields(
                    instrument,
                    snapshot_ms
                )
            )

            result[
                instrument
            ] = state

        return result

    # --------------------------------------------------
    # Finalize within-bar L2 statistics
    # --------------------------------------------------

    def finalize_book_stats(
        self,
        instrument: str,
        bar_start_ms: int,
        bar_end_ms: int
    ) -> dict:

        # If no post-boundary update has yet
        # advanced the metric state, extend the
        # latest valid book to the bar boundary.
        self._advance_book_stats(
            instrument,
            bar_end_ms
        )

        key = (
            instrument,
            bar_start_ms
        )

        accumulator = (
            self.book_bar_accumulators.pop(
                key,
                None
            )
        )

        result = {
            "book_stats_coverage_ms":
                0.0,

            "book_stats_complete":
                False,

            "book_update_count":
                0,

            "book_min_obi_20":
                None,

            "book_max_obi_20":
                None
        }

        for metric_name in (
            BOOK_METRIC_NAMES
        ):

            result[
                f"book_twa_{metric_name}"
            ] = None

        if accumulator is None:

            return result

        coverage_ms = min(
            float(
                accumulator[
                    "valid_duration_ms"
                ]
            ),
            float(
                BAR_INTERVAL_MS
            )
        )

        result[
            "book_stats_coverage_ms"
        ] = coverage_ms

        # Allow a tiny timestamp rounding tolerance.
        result[
            "book_stats_complete"
        ] = (
            coverage_ms
            >= (
                BAR_INTERVAL_MS
                - 5
            )
        )

        result[
            "book_update_count"
        ] = int(
            accumulator[
                "update_count"
            ]
        )

        result[
            "book_min_obi_20"
        ] = accumulator[
            "min_obi_20"
        ]

        result[
            "book_max_obi_20"
        ] = accumulator[
            "max_obi_20"
        ]

        for metric_name in (
            BOOK_METRIC_NAMES
        ):

            duration = (
                accumulator[
                    "weighted_duration"
                ][
                    metric_name
                ]
            )

            if duration <= 0:
                continue

            result[
                f"book_twa_{metric_name}"
            ] = (
                accumulator[
                    "weighted_sum"
                ][
                    metric_name
                ]
                / duration
            )

        return result

    # --------------------------------------------------
    # Finalize 5-minute futures bar
    # --------------------------------------------------

    def finalize_bar(
        self,
        bar_end,
        boundary_state: dict[str, dict]
    ) -> pd.DataFrame:

        bar_end_ms = int(
            bar_end.timestamp()
            * 1000
        )

        bar_start_ms = (
            bar_end_ms
            - BAR_INTERVAL_MS
        )

        bar_start = (
            pd.Timestamp(
                bar_start_ms,
                unit="ms",
                tz="UTC"
            )
            .to_pydatetime()
        )

        rows = []

        for instrument in (
            self.instrument_names
        ):

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

                open_price = None
                high_price = None
                low_price = None
                close_price = None

                vwap = None

                volume_native = 0.0
                volume_base = 0.0
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

                open_price = (
                    bar[
                        "open"
                    ]
                )

                high_price = (
                    bar[
                        "high"
                    ]
                )

                low_price = (
                    bar[
                        "low"
                    ]
                )

                close_price = (
                    bar[
                        "close"
                    ]
                )

                volume_native = (
                    bar[
                        "volume_native"
                    ]
                )

                volume_usd = (
                    bar[
                        "volume_usd"
                    ]
                )

                buy_volume_native = (
                    bar[
                        "buy_volume_native"
                    ]
                )

                sell_volume_native = (
                    bar[
                        "sell_volume_native"
                    ]
                )

                buy_volume_base = (
                    bar[
                        "buy_volume_base"
                    ]
                )

                sell_volume_base = (
                    bar[
                        "sell_volume_base"
                    ]
                )

                buy_volume_usd = (
                    bar[
                        "buy_volume_usd"
                    ]
                )

                sell_volume_usd = (
                    bar[
                        "sell_volume_usd"
                    ]
                )

                trade_count = (
                    bar[
                        "trade_count"
                    ]
                )

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
                and key
                not in self.incomplete_bar_keys
            )

            self.incomplete_bar_keys.discard(
                key
            )

            book_stats = (
                self.finalize_book_stats(
                    instrument,
                    bar_start_ms,
                    bar_end_ms
                )
            )

            row = {
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

                "stream_complete":
                    stream_complete,

                "source":
                    "deribit_ws_100ms_l2"
            }

            row.update(
                boundary
            )

            row.update(
                book_stats
            )

            rows.append(
                row
            )

        self.last_finalized_end_ms = max(
            self.last_finalized_end_ms,
            bar_end_ms
        )

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------
    # WebSocket message handling
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

            # Subscription confirmations.
            if "id" in message:

                if "result" in message:

                    print(
                        "Futures market-data "
                        "subscriptions confirmed."
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

            elif channel.startswith(
                "book."
            ):

                if isinstance(
                    data,
                    dict
                ):

                    self.process_book(
                        data
                    )

            self.messages_received += 1

    # --------------------------------------------------
    # Health
    # --------------------------------------------------

    def valid_book_count(
        self
    ) -> int:

        return sum(
            1
            for state
            in self.order_book_state.values()
            if state.get(
                "valid",
                False
            )
        )

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

            tickers_initialized = len(
                self.ticker_state
            )

            books_initialized = (
                self.valid_book_count()
            )

            now_ms = int(
                time.time()
                * 1000
            )

            ticker_ages = []

            for ticker in (
                self.ticker_state.values()
            ):

                timestamp_ms = (
                    ticker.get(
                        "timestamp"
                    )
                )

                if (
                    timestamp_ms
                    is not None
                ):

                    ticker_ages.append(
                        now_ms
                        - timestamp_ms
                    )

            book_ages = []

            for state in (
                self.order_book_state.values()
            ):

                if not state.get(
                    "valid",
                    False
                ):

                    continue

                timestamp_ms = (
                    state.get(
                        "timestamp"
                    )
                )

                if (
                    timestamp_ms
                    is not None
                ):

                    book_ages.append(
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
                f"{tickers_initialized}/"
                f"{expected}"
            )

            print(
                f"L2 books initialized: "
                f"{books_initialized}/"
                f"{expected}"
            )

            print(
                f"Trades received:     "
                f"{self.trades_received:,}"
            )

            print(
                f"L2 messages:         "
                f"{self.book_messages_received:,}"
            )

            print(
                f"L2 sequence gaps:    "
                f"{self.book_sequence_gaps:,}"
            )

            print(
                f"Late trades ignored: "
                f"{self.ignored_late_trades:,}"
            )

            if ticker_ages:

                print(
                    f"Median ticker age:   "
                    f"{statistics.median(ticker_ages):,.0f} ms"
                )

            if book_ages:

                print(
                    f"Median L2 age:       "
                    f"{statistics.median(book_ages):,.0f} ms"
                )

    # --------------------------------------------------
    # WebSocket lifecycle
    # --------------------------------------------------

    async def run_connection(
        self
    ) -> None:

        self.connection_count += 1

        if (
            self.connection_count
            > 1
        ):

            self.reset_realtime_state()

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
                            "Futures collector "
                            "shutdown"
                        )
                    )

                if (
                    receive_task in done
                ):

                    exception = (
                        receive_task.exception()
                    )

                    if exception is not None:

                        raise exception

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

                if (
                    not self.stop_event.is_set()
                ):

                    self.mark_current_bar_incomplete()

                    print(
                        "\nFutures WebSocket "
                        "closed unexpectedly."
                    )

            except asyncio.CancelledError:

                break

            except Exception as exc:

                if self.stop_event.is_set():
                    break

                self.mark_current_bar_incomplete()

                print(
                    f"\nFutures WebSocket "
                    f"error: {exc}"
                )

            if self.stop_event.is_set():
                break

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