from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

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


# --------------------------------------------------
# Collector version stamp
# --------------------------------------------------
#
# Every finalized row carries the collector version and
# the SHA256 of this exact source file. Feature-version
# boundaries (for example the L2 event-time fix) are
# therefore identifiable from the data itself.

COLLECTOR_VERSION = "futures_5m_l2_v0.3.1"


def _source_sha256() -> str | None:

    try:

        return hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()

    except OSError:

        return None


COLLECTOR_CODE_SHA256 = _source_sha256()


# --------------------------------------------------
# Feed-integrity configuration (v0.3)
# --------------------------------------------------
#
# These are collection-integrity rules, not research
# thresholds. Raw evidence (gaps, ages, REST checks) is
# always written alongside the resulting flags.

# Any WebSocket message at all. With seven perpetual
# incremental tickers (mark/index update roughly every
# second), silence this long means the connection is not
# delivering data even if ping/pong still succeeds.
FEED_SILENCE_RECONNECT_SECONDS = 15.0

# A bar whose largest gap between consecutive received
# WebSocket messages exceeds this value is not marked
# stream_complete. The raw gap is always stored.
FEED_MAX_GAP_FOR_COMPLETE_MS = 10_000

# Subscription confirmation must arrive within this time.
SUBSCRIPTION_CONFIRM_TIMEOUT_SECONDS = 15.0

# REST cross-check of quiet streams. If an instrument's
# L2 book (or ticker BBO) has not changed on the stream
# for BOOK_QUIET_VERIFY_SECONDS, its REST top of book is
# compared with the cached stream state. A mismatch that
# persists after STALE_CONFIRM_DELAY_SECONDS, while the
# stream still shows no update, is a confirmed stale feed.
STREAM_VERIFY_INTERVAL_SECONDS = 20.0
BOOK_QUIET_VERIFY_SECONDS = 20.0
STALE_CONFIRM_DELAY_SECONDS = 3.0
REST_TIMEOUT_SECONDS = 10

# v0.3.1: a stale channel is resubscribed at most this often
# while its stale state remains open.
STALE_RESUBSCRIBE_MIN_SECONDS = 60.0

# Clock grid for intrawindow mid-price path statistics.
PATH_GRID_MS = 10_000
PATH_GRID_POINTS = BAR_INTERVAL_MS // PATH_GRID_MS


class FeedStaleError(
    RuntimeError
):
    """Raised to force a reconnect after a feed fault."""


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
# L2 instability distribution representation
# --------------------------------------------------
#
# The collector deliberately does NOT select epsilon.
#
# Instead it retains:
#
#   1. epsilon-independent five-minute path summaries;
#   2. a time-weighted distribution of |OBI-20|;
#   3. a distribution of zero-crossing reversal strength.
#
# For a direct sign-changing transition:
#
#     previous_obi * current_obi < 0
#
# define:
#
#     reversal_strength =
#         min(abs(previous_obi), abs(current_obi))
#
# For any later epsilon, that transition is a robust
# +1 <-> -1 flip exactly when:
#
#     reversal_strength > epsilon
#
# Therefore epsilon can be modelled later from observed
# distributions without retaining the 100 ms L2 sequence.
#
# The histogram is only a storage representation, not an
# epsilon choice. Quadratic edges give finer resolution near
# zero, where dead-band inference is most sensitive.

OBI_HISTOGRAM_BIN_COUNT = 64

OBI_HISTOGRAM_SCHEMA = (
    "quadratic_0_1_64_v1"
)

OBI_HISTOGRAM_EDGES = tuple(
    (
        index
        / OBI_HISTOGRAM_BIN_COUNT
    ) ** 2
    for index
    in range(
        OBI_HISTOGRAM_BIN_COUNT + 1
    )
)


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


def obi_histogram_bin_index(
    value: float
) -> int:
    """
    Return the fixed quadratic-histogram bin for an
    OBI magnitude in [0, 1].

    OBI is theoretically bounded by [-1, 1]. Values
    are defensively clipped before binning.
    """

    magnitude = min(
        max(
            abs(
                float(
                    value
                )
            ),
            0.0
        ),
        1.0
    )

    if magnitude >= 1.0:

        return (
            OBI_HISTOGRAM_BIN_COUNT
            - 1
        )

    # Inverse of:
    #
    #     edge_j = (j / N)^2
    #
    index = int(
        math.sqrt(
            magnitude
        )
        * OBI_HISTOGRAM_BIN_COUNT
    )

    return min(
        max(
            index,
            0
        ),
        OBI_HISTOGRAM_BIN_COUNT
        - 1
    )


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


def fetch_rest_top_of_book(
    instrument_name: str
) -> tuple[float, float] | None:
    """
    v0.3: independent REST top of book used to verify that
    a quiet WebSocket stream is genuinely quiet rather than
    stale. Returns (best_bid, best_ask) or None on error.
    """

    try:

        response = requests.get(
            f"{REST_URL}/public/get_order_book",
            params={
                "instrument_name": instrument_name,
                "depth": 1
            },
            timeout=REST_TIMEOUT_SECONDS
        )

        response.raise_for_status()

        result = (
            response.json()
            .get("result")
            or {}
        )

        bid = result.get(
            "best_bid_price"
        )

        ask = result.get(
            "best_ask_price"
        )

        if (
            bid is None
            or ask is None
            or bid <= 0
            or ask <= 0
        ):
            return None

        return (
            float(bid),
            float(ask)
        )

    except Exception:

        return None


def discover_perpetuals(
    option_instruments: pd.DataFrame
) -> pd.DataFrame:
    """
    Select one perpetual future for every underlying
    represented in the current 0DTE option session.

    Preference:
        1. Linear USDC perpetual
        2. Inverse/reversed perpetual
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

    # Prefer linear USDC perpetuals.
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

        # Latest complete top-20 L2 snapshot only.
        self.order_book_state: dict[
            str,
            dict
        ] = {}

        # Current trade/OHLCV accumulators.
        self.bars: dict[
            tuple[str, int],
            dict
        ] = {}

        self.seen_trade_ids: dict[
            tuple[str, int],
            set
        ] = {}

        # Latest valid L2 metric state used for
        # time-weighted carry-forward.
        self.book_metric_state: dict[
            str,
            dict
        ] = {}

        # One active constant-memory accumulator
        # per instrument x five-minute bar.
        self.book_bar_accumulators: dict[
            tuple[str, int],
            dict
        ] = {}

        # Bars crossed by a WebSocket interruption.
        self.incomplete_bar_keys: set[
            tuple[str, int]
        ] = set()

        self.messages_received = 0
        self.trades_received = 0
        self.book_messages_received = 0

        self.book_parse_errors = 0
        self.ignored_late_trades = 0

        # L2 snapshots whose exchange timestamp is older
        # than the currently active book timestamp. These
        # are rejected so event time can never move backward
        # and clock time cannot be counted twice.
        self.book_out_of_order_updates = 0

        self.connection_count = 0

        self.collector_started_ms = int(
            time.time()
            * 1000
        )

        self.last_finalized_end_ms = 0

        self.websocket = None

        # --------------------------------------------------
        # v0.3 feed-integrity state
        # --------------------------------------------------

        # Local receive time of the latest WebSocket
        # message of any kind.
        self.last_message_received_ms: (
            int | None
        ) = None

        # Largest gap between consecutive received
        # WebSocket messages, keyed by bar start
        # (local receive clock).
        self.feed_gap_by_bucket: dict[
            int,
            float
        ] = {}

        # Start of an ongoing outage. While not None,
        # every bar that ends after this instant is
        # finalized as incomplete. Cleared only after the
        # new connection has confirmed its subscriptions
        # and re-seeded every ticker and L2 book, at which
        # point the whole outage interval is marked.
        self.degraded_since_ms: (
            int | None
        ) = None

        self.connection_started_ms: (
            int | None
        ) = None

        self.subscribe_request_id = 2001

        self.requested_channels: set[
            str
        ] = set()

        # None until the subscription response arrives
        # on the current connection.
        self.confirmed_channels: (
            set[str] | None
        ) = None

        self.subscription_sent_ms: (
            int | None
        ) = None

        # Instruments with at least one requested channel
        # that the exchange did not confirm.
        self.unconfirmed_instruments: set[
            str
        ] = set()

        # Exchange / local time of the latest change of
        # ticker best bid/ask PRICES, per instrument.
        self.ticker_bbo_last_change: dict[
            str,
            dict
        ] = {}

        # Local receive time of the latest accepted L2
        # update per instrument (for quiet-stream checks).
        self.book_last_update_received_ms: dict[
            str,
            int
        ] = {}

        # REST cross-check evidence per (instrument, bar).
        self.verify_stats: dict[
            tuple[str, int],
            dict
        ] = {}

        self.rest_verifications = 0
        self.rest_verification_errors = 0
        self.stale_feeds_detected = 0
        self.feed_silence_reconnects = 0
        self.subscription_failures = 0
        self.unconfirmed_channel_count = 0

        # v0.3.1 per-instrument stale-stream state:
        # (instrument, "book"|"ticker") -> {since_ms,
        # last_resubscribe_ms}. Survives reconnects; only a
        # REST match clears it.
        self.stale_streams: dict[
            tuple[str, str],
            dict
        ] = {}

        # Monotonic per-instrument counters of stream
        # changes (accepted L2 updates; ticker BBO price
        # changes). Used to tell a live stream from a
        # single resubscribe snapshot of a frozen feed.
        self.stream_change_counter: dict[
            tuple[str, str],
            int
        ] = {}

        self.resubscribe_request_id = 3000
        self.channel_resubscriptions = 0

    # --------------------------------------------------
    # Subscription channels
    # --------------------------------------------------

    def book_channel(
        self,
        instrument: str
    ) -> str:
        """
        Public depth-limited aggregated order book.

        Every notification is a complete depth-20
        snapshot, not an incremental raw-book delta.
        """

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
            "id": self.subscribe_request_id,
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

        # v0.3: remember exactly what was requested so the
        # confirmation can be checked channel by channel.
        self.requested_channels = set(
            channels
        )

        self.confirmed_channels = None

        self.subscription_sent_ms = int(
            time.time()
            * 1000
        )

        await websocket.send(
            json.dumps(
                request
            )
        )

    def handle_subscription_confirmation(
        self,
        result
    ) -> None:
        """
        Compare the confirmed channel list with the request.

        Missing channels are not retried in a loop: the
        affected instruments are recorded and their rows are
        finalized with subscription_confirmed = False and
        stream_complete = False until a later connection
        confirms them.
        """

        if not isinstance(
            result,
            list
        ):
            result = []

        self.confirmed_channels = set(
            str(channel)
            for channel in result
        )

        missing = sorted(
            self.requested_channels
            - self.confirmed_channels
        )

        self.unconfirmed_instruments = set()

        for instrument in (
            self.instrument_names
        ):

            instrument_channels = {
                channel
                for channel in missing
                if f".{instrument}" in channel
            }

            if instrument_channels:

                self.unconfirmed_instruments.add(
                    instrument
                )

        self.unconfirmed_channel_count = len(
            missing
        )

        print(
            "Futures market-data "
            "subscriptions confirmed: "
            f"{len(self.confirmed_channels)}/"
            f"{len(self.requested_channels)}"
        )

        if missing:

            self.subscription_failures += 1

            print(
                "WARNING: unconfirmed "
                "futures channels:"
            )

            for channel in missing:

                print(
                    f"  {channel}"
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
        """
        Backward-compatible helper. v0.3 callers use
        begin_outage() / mark_interval_incomplete().
        """

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

    def mark_interval_incomplete(
        self,
        start_ms: int,
        end_ms: int,
        instruments: list[str] | None = None
    ) -> None:
        """
        Mark every five-minute bar overlapping
        [start_ms, end_ms] as incomplete for the given
        instruments (default: all).
        """

        if instruments is None:
            instruments = self.instrument_names

        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms

        first_bucket = (
            start_ms
            - (
                start_ms
                % BAR_INTERVAL_MS
            )
        )

        bucket = first_bucket

        while bucket <= end_ms:

            for instrument in instruments:

                self.incomplete_bar_keys.add(
                    (
                        instrument,
                        bucket
                    )
                )

            bucket += BAR_INTERVAL_MS

    def begin_outage(
        self
    ) -> None:
        """
        Start (or extend) an outage at the last instant data
        was known to be flowing. The outage stays open until
        the next connection is fully re-synchronized.
        """

        now_ms = int(
            time.time()
            * 1000
        )

        outage_start_ms = (
            self.last_message_received_ms
            if self.last_message_received_ms
            is not None
            else now_ms
        )

        if (
            self.degraded_since_ms is None
            or outage_start_ms
            < self.degraded_since_ms
        ):

            self.degraded_since_ms = (
                outage_start_ms
            )

        # Bars from the last good message up to now are
        # incomplete regardless of how the outage ends.
        self.mark_interval_incomplete(
            self.degraded_since_ms,
            now_ms
        )

    def is_resynchronized(
        self
    ) -> bool:

        if self.confirmed_channels is None:
            return False

        for instrument in (
            self.instrument_names
        ):

            if instrument not in self.ticker_state:
                return False

            book = self.order_book_state.get(
                instrument
            )

            if not book or not book.get(
                "valid",
                False
            ):
                return False

        return True

    def check_resynchronized(
        self
    ) -> None:
        """
        Close an open outage once the current connection has
        confirmed subscriptions and re-seeded every ticker and
        L2 book. The full outage interval is then marked.
        """

        if self.degraded_since_ms is None:
            return

        if not self.is_resynchronized():
            return

        now_ms = int(
            time.time()
            * 1000
        )

        self.mark_interval_incomplete(
            self.degraded_since_ms,
            now_ms
        )

        print(
            "\nFutures feed re-synchronized. "
            "Outage interval marked incomplete: "
            f"{self.degraded_since_ms} -> {now_ms} "
            f"({(now_ms - self.degraded_since_ms) / 1000:.1f}s)"
        )

        self.degraded_since_ms = None

    def reset_realtime_state(
        self
    ) -> None:
        """
        Trade/bar accumulators are preserved.

        Ticker and current L2 state are discarded so
        the new WebSocket connection seeds them again
        with fresh snapshots.

        The active five-minute accumulator is preserved,
        but the crossed bar is marked incomplete by the
        caller. No transition is measured across the
        disconnected interval because book_metric_state
        is cleared.
        """

        self.ticker_state.clear()
        self.order_book_state.clear()
        self.book_metric_state.clear()

        # v0.3: change-tracking restarts from the fresh
        # snapshots of the new connection. Nothing is known
        # about BBO changes during the disconnected interval.
        self.ticker_bbo_last_change.clear()
        self.book_last_update_received_ms.clear()

        self.confirmed_channels = None
        self.unconfirmed_instruments = set()

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

        previous_state = (
            self.ticker_state.get(
                instrument_name
            )
        )

        previous_bbo = (
            (
                previous_state.get(
                    "best_bid_price"
                ),
                previous_state.get(
                    "best_ask_price"
                )
            )
            if previous_state is not None
            else None
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

        received_ms = int(
            time.time()
            * 1000
        )

        state[
            "_received_timestamp_ms"
        ] = received_ms

        self.ticker_state[
            instrument_name
        ] = state

        # v0.3: the ticker "timestamp" moves whenever ANY
        # field changes (mark/index tick every ~1s), so it
        # cannot reveal a frozen BBO. Track the last time
        # the best bid/ask PRICES actually changed.
        current_bbo = (
            state.get(
                "best_bid_price"
            ),
            state.get(
                "best_ask_price"
            )
        )

        if (
            previous_bbo is None
            or current_bbo != previous_bbo
            or instrument_name
            not in self.ticker_bbo_last_change
        ):

            exchange_ms = state.get(
                "timestamp"
            )

            counter_key = (
                instrument_name,
                "ticker"
            )

            self.stream_change_counter[
                counter_key
            ] = self.stream_change_counter.get(
                counter_key,
                0
            ) + 1

            self.ticker_bbo_last_change[
                instrument_name
            ] = {
                "exchange_ms":
                    int(exchange_ms)
                    if exchange_ms is not None
                    else received_ms,

                "received_ms":
                    received_ms
            }

    # --------------------------------------------------
    # L2 parsing
    # --------------------------------------------------

    def _parse_grouped_level(
        self,
        level
    ) -> tuple[
        float,
        float
    ] | None:
        """
        Grouped depth-limited Deribit book feeds
        normally provide:

            [price, amount]

        We defensively also accept:

            [action, price, amount]
        """

        if not isinstance(
            level,
            (list, tuple)
        ):

            return None

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

        elif len(level) >= 3:

            try:

                price = float(
                    level[-2]
                )

                amount = float(
                    level[-1]
                )

            except (
                TypeError,
                ValueError
            ):

                return None

        else:

            return None

        if (
            price <= 0
            or amount <= 0
        ):

            return None

        return (
            price,
            amount
        )

    def _parse_grouped_side(
        self,
        levels
    ) -> list[
        tuple[
            float,
            float
        ]
    ]:

        result = []

        if not isinstance(
            levels,
            list
        ):

            return result

        for level in levels:

            parsed = (
                self._parse_grouped_level(
                    level
                )
            )

            if parsed is None:

                continue

            result.append(
                parsed
            )

        return result

    def _calculate_metrics_from_levels(
        self,
        bids: list[
            tuple[float, float]
        ],
        asks: list[
            tuple[float, float]
        ]
    ) -> dict | None:

        if (
            not bids
            or not asks
        ):

            return None

        best_bid = (
            bids[0][0]
        )

        best_ask = (
            asks[0][0]
        )

        bid_size_1 = (
            bids[0][1]
        )

        ask_size_1 = (
            asks[0][1]
        )

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
                microprice,

            # v0.3: explicit L2 top of book. Written at the
            # boundary so ticker BBO and book BBO can be
            # compared, and used for trade-vs-book checks
            # and the intrawindow mid-price path.
            "book_best_bid_price":
                best_bid,

            "book_best_ask_price":
                best_ask,

            "book_mid_price":
                (
                    best_bid
                    + best_ask
                ) / 2.0
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
    # Time-weighted L2 + instability statistics
    # --------------------------------------------------

    def _new_book_accumulator(
        self
    ) -> dict:

        return {
            "valid_duration_ms":
                0.0,

            "weighted_sum": {
                metric:
                    0.0

                for metric
                in BOOK_METRIC_NAMES
            },

            "weighted_duration": {
                metric:
                    0.0

                for metric
                in BOOK_METRIC_NAMES
            },

            # Additional OBI-20 time-weighted moments.
            "weighted_sq_sum_obi_20":
                0.0,

            "weighted_abs_sum_obi_20":
                0.0,

            # Clock-time distribution of |OBI-20|.
            "obi_20_abs_histogram_ms":
                [
                    0.0
                    for _
                    in range(
                        OBI_HISTOGRAM_BIN_COUNT
                    )
                ],

            # Incoming valid book updates in this bar.
            "update_count":
                0,

            # Incoming L2 snapshots rejected because their
            # exchange timestamp was older than the active
            # event-time state. Rejected snapshots never
            # alter path statistics or the active book.
            "out_of_order_update_count":
                0,

            # Event-path state. This may be seeded from
            # the book carried across the five-minute
            # boundary before the first new update.
            "previous_event_obi_20":
                None,

            "transition_count_obi_20":
                0,

            "abs_delta_sum_obi_20":
                0.0,

            # Epsilon-free raw sign-crossing data.
            "zero_cross_count_obi_20":
                0,

            "reversal_strength_sum_obi_20":
                0.0,

            "reversal_strength_max_obi_20":
                None,

            "reversal_strength_histogram":
                [
                    0
                    for _
                    in range(
                        OBI_HISTOGRAM_BIN_COUNT
                    )
                ],

            "min_obi_20":
                None,

            "max_obi_20":
                None,

            # --------------------------------------
            # v0.3 feed-integrity evidence
            # --------------------------------------

            # Exchange time of the latest accepted L2
            # update that fell inside this bar.
            "last_update_in_bar_ms":
                None,

            # Largest interval inside the bar with no
            # accepted L2 update (exchange time).
            "max_update_gap_ms":
                0.0,

            # Mid price of the active book at each
            # PATH_GRID_MS clock point of the bar
            # (right-continuous: the state in force at
            # the grid instant). None = no valid book.
            "path_grid_mid":
                [
                    None
                    for _
                    in range(
                        PATH_GRID_POINTS
                    )
                ]
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
        """
        Carry the previous complete L2 state forward
        through clock time.

        No L2 message history is retained. Only the
        active metrics and constant-memory accumulator
        are updated.
        """

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

            # v0.3: sample the active mid price at every
            # clock-grid instant g with cursor <= g < end.
            mid_price = (
                metrics.get(
                    "book_mid_price"
                )
            )

            if mid_price is not None:

                first_index = -(
                    -(
                        cursor
                        - bucket_start_ms
                    )
                    // PATH_GRID_MS
                )

                end_index = -(
                    -(
                        segment_end
                        - bucket_start_ms
                    )
                    // PATH_GRID_MS
                )

                for grid_index in range(
                    max(first_index, 0),
                    min(
                        end_index,
                        PATH_GRID_POINTS
                    )
                ):

                    accumulator[
                        "path_grid_mid"
                    ][
                        grid_index
                    ] = float(
                        mid_price
                    )

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

            obi_20 = (
                metrics.get(
                    "obi_20"
                )
            )

            if obi_20 is not None:

                obi_20 = float(
                    obi_20
                )

                accumulator[
                    "weighted_sq_sum_obi_20"
                ] += (
                    obi_20
                    * obi_20
                    * duration
                )

                accumulator[
                    "weighted_abs_sum_obi_20"
                ] += (
                    abs(
                        obi_20
                    )
                    * duration
                )

                histogram_index = (
                    obi_histogram_bin_index(
                        obi_20
                    )
                )

                accumulator[
                    "obi_20_abs_histogram_ms"
                ][
                    histogram_index
                ] += duration

                # If this is the first activity in a
                # new five-minute bucket, seed the
                # event path from the state that was
                # already active at the boundary.
                if (
                    accumulator[
                        "previous_event_obi_20"
                    ]
                    is None
                ):

                    accumulator[
                        "previous_event_obi_20"
                    ] = obi_20

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
        """
        Update event-path statistics from one new
        valid depth-20 snapshot.

        The incoming snapshot itself is not retained
        historically after current state is replaced.
        """

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

        # v0.3: largest exchange-time interval without
        # an accepted L2 update inside this bar.
        gap_reference_ms = (
            accumulator[
                "last_update_in_bar_ms"
            ]
            if accumulator[
                "last_update_in_bar_ms"
            ] is not None
            else bucket_start_ms
        )

        accumulator[
            "max_update_gap_ms"
        ] = max(
            accumulator[
                "max_update_gap_ms"
            ],
            float(
                timestamp_ms
                - gap_reference_ms
            )
        )

        accumulator[
            "last_update_in_bar_ms"
        ] = timestamp_ms

        obi_20 = (
            metrics.get(
                "obi_20"
            )
        )

        if obi_20 is None:

            return

        obi_20 = float(
            obi_20
        )

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

        previous_obi_20 = (
            accumulator[
                "previous_event_obi_20"
            ]
        )

        if previous_obi_20 is not None:

            accumulator[
                "transition_count_obi_20"
            ] += 1

            delta_obi_20 = (
                obi_20
                - previous_obi_20
            )

            accumulator[
                "abs_delta_sum_obi_20"
            ] += abs(
                delta_obi_20
            )

            # Raw direct sign crossing.
            #
            # No epsilon is selected here.
            if (
                previous_obi_20
                * obi_20
                < 0
            ):

                reversal_strength = min(
                    abs(
                        previous_obi_20
                    ),
                    abs(
                        obi_20
                    )
                )

                accumulator[
                    "zero_cross_count_obi_20"
                ] += 1

                accumulator[
                    "reversal_strength_sum_obi_20"
                ] += (
                    reversal_strength
                )

                current_max_strength = (
                    accumulator[
                        "reversal_strength_max_obi_20"
                    ]
                )

                if (
                    current_max_strength is None
                    or reversal_strength
                    > current_max_strength
                ):

                    accumulator[
                        "reversal_strength_max_obi_20"
                    ] = (
                        reversal_strength
                    )

                histogram_index = (
                    obi_histogram_bin_index(
                        reversal_strength
                    )
                )

                accumulator[
                    "reversal_strength_histogram"
                ][
                    histogram_index
                ] += 1

        accumulator[
            "previous_event_obi_20"
        ] = obi_20

    # --------------------------------------------------
    # Grouped L2 processing
    # --------------------------------------------------

    def process_book(
        self,
        data: dict
    ) -> None:
        """
        Process one complete grouped depth-20 snapshot.

        IMPORTANT:
        book.<instrument>.none.20.100ms is treated as
        a sequence of complete top-20 snapshots.

        Each update is used once for streaming
        aggregation. Historical snapshots are not
        persisted.
        """

        instrument_name = (
            data.get(
                "instrument_name"
            )
        )

        if not instrument_name:

            self.book_parse_errors += 1

            return

        received_timestamp_ms = int(
            time.time()
            * 1000
        )

        timestamp_raw = (
            data.get(
                "timestamp"
            )
        )

        if timestamp_raw is None:

            timestamp_ms = (
                received_timestamp_ms
            )

        else:

            try:

                timestamp_ms = int(
                    timestamp_raw
                )

            except (
                TypeError,
                ValueError
            ):

                timestamp_ms = (
                    received_timestamp_ms
                )

        # Enforce monotonic L2 event time.
        #
        # Deribit book messages can occasionally arrive with
        # an exchange timestamp a few milliseconds older than
        # the currently active state. If that older timestamp
        # were installed as last_timestamp_ms, the next normal
        # update would carry book time forward across an interval
        # that had already been accumulated, double-counting
        # clock time and potentially creating artificial path
        # transitions.
        #
        # A strictly older snapshot is therefore rejected in
        # full. Equal timestamps remain valid: they contribute
        # zero additional clock duration but may represent a real
        # same-millisecond state transition.
        active_metric_state = (
            self.book_metric_state.get(
                instrument_name
            )
        )

        if active_metric_state is not None:

            active_timestamp_ms = int(
                active_metric_state[
                    "last_timestamp_ms"
                ]
            )

            if timestamp_ms < active_timestamp_ms:

                self.book_out_of_order_updates += 1

                audit_bucket_start_ms = (
                    active_timestamp_ms
                    - (
                        active_timestamp_ms
                        % BAR_INTERVAL_MS
                    )
                )

                audit_accumulator = (
                    self._get_book_accumulator(
                        instrument_name,
                        audit_bucket_start_ms
                    )
                )

                audit_accumulator[
                    "out_of_order_update_count"
                ] += 1

                return

        # The previous complete book remained active
        # until this new exchange timestamp.
        self._advance_book_stats(
            instrument_name,
            timestamp_ms
        )

        bids = (
            self._parse_grouped_side(
                data.get(
                    "bids",
                    []
                )
            )
        )

        asks = (
            self._parse_grouped_side(
                data.get(
                    "asks",
                    []
                )
            )
        )

        bids = sorted(
            bids,
            key=lambda item:
                item[0],
            reverse=True
        )[
            :BOOK_DEPTH
        ]

        asks = sorted(
            asks,
            key=lambda item:
                item[0]
        )[
            :BOOK_DEPTH
        ]

        metrics = (
            self._calculate_metrics_from_levels(
                bids,
                asks
            )
        )

        if metrics is None:

            self.book_parse_errors += 1

            self.order_book_state[
                instrument_name
            ] = {
                "bids":
                    [],

                "asks":
                    [],

                "change_id":
                    data.get(
                        "change_id"
                    ),

                "timestamp":
                    timestamp_ms,

                "_received_timestamp_ms":
                    received_timestamp_ms,

                "valid":
                    False
            }

            # Do not carry an invalid book state
            # forward through time.
            self.book_metric_state.pop(
                instrument_name,
                None
            )

            return

        self.order_book_state[
            instrument_name
        ] = {
            "bids":
                bids,

            "asks":
                asks,

            "change_id":
                data.get(
                    "change_id"
                ),

            "timestamp":
                timestamp_ms,

            "_received_timestamp_ms":
                received_timestamp_ms,

            "valid":
                True
        }

        # _advance_book_stats() above has already
        # carried the previous state forward and,
        # when needed, seeded the current bar's
        # previous_event_obi_20.
        self._record_book_observation(
            instrument_name,
            timestamp_ms,
            metrics
        )

        self.book_metric_state[
            instrument_name
        ] = {
            "last_timestamp_ms":
                timestamp_ms,

            "last_metrics":
                metrics
        }

        self.book_last_update_received_ms[
            instrument_name
        ] = received_timestamp_ms

        counter_key = (
            instrument_name,
            "book"
        )

        self.stream_change_counter[
            counter_key
        ] = self.stream_change_counter.get(
            counter_key,
            0
        ) + 1

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
                    native_unit,

                # v0.3 trade-vs-book evidence.
                "trades_checked_vs_book":
                    0,

                "trades_outside_book_bbo":
                    0,

                "max_trade_outside_book_bbo_bps":
                    0.0
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

        # v0.3: compare the execution price with the L2 top
        # of book known when the trade is processed. Trades
        # far outside a long-frozen cached BBO are the most
        # direct evidence of a stale book stream. No
        # threshold is applied here; legitimate sweeps and
        # ~100 ms channel skew also produce small values.
        book = self.order_book_state.get(
            instrument_name
        )

        if (
            book
            and book.get("valid", False)
            and book.get("bids")
            and book.get("asks")
        ):

            book_bid = float(
                book["bids"][0][0]
            )

            book_ask = float(
                book["asks"][0][0]
            )

            book_mid = (
                book_bid
                + book_ask
            ) / 2.0

            bar[
                "trades_checked_vs_book"
            ] += 1

            outside_distance = 0.0

            if price > book_ask:

                outside_distance = (
                    price
                    - book_ask
                )

            elif price < book_bid:

                outside_distance = (
                    book_bid
                    - price
                )

            if (
                outside_distance > 0
                and book_mid > 0
            ):

                bar[
                    "trades_outside_book_bbo"
                ] += 1

                bar[
                    "max_trade_outside_book_bbo_bps"
                ] = max(
                    bar[
                        "max_trade_outside_book_bbo_bps"
                    ],
                    outside_distance
                    / book_mid
                    * 10000.0
                )

        self.trades_received += 1

    # --------------------------------------------------
    # L2 boundary state
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
            "obi_20",
            "book_best_bid_price",
            "book_best_ask_price",
            "book_mid_price"
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

        bids = (
            state[
                "bids"
            ]
        )

        asks = (
            state[
                "asks"
            ]
        )

        metrics = (
            self._calculate_metrics_from_levels(
                bids,
                asks
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
                    ticker_age_ms,

                # v0.3: perpetual funding (already in the
                # incremental ticker; previously discarded).
                "current_funding":
                    ticker.get(
                        "current_funding"
                    ),

                "funding_8h":
                    ticker.get(
                        "funding_8h"
                    ),

                "estimated_delivery_price":
                    ticker.get(
                        "estimated_delivery_price"
                    )
            }

            # v0.3: age of the ticker best bid/ask PRICES,
            # independent of mark/index updates.
            bbo_change = (
                self.ticker_bbo_last_change.get(
                    instrument
                )
            )

            if bbo_change is not None:

                state[
                    "ticker_bbo_last_change_ms"
                ] = bbo_change[
                    "exchange_ms"
                ]

                state[
                    "ticker_bbo_age_ms"
                ] = (
                    snapshot_ms
                    - bbo_change[
                        "exchange_ms"
                    ]
                )

            else:

                state[
                    "ticker_bbo_last_change_ms"
                ] = None

                state[
                    "ticker_bbo_age_ms"
                ] = None

            state.update(
                self._book_boundary_fields(
                    instrument,
                    snapshot_ms
                )
            )

            # v0.3: do the two independent top-of-book
            # sources agree at the boundary?
            book_bid = state.get(
                "book_best_bid_price"
            )

            book_ask = state.get(
                "book_best_ask_price"
            )

            if (
                bid is not None
                and ask is not None
                and book_bid is not None
                and book_ask is not None
            ):

                state[
                    "ticker_book_bbo_match"
                ] = (
                    float(bid) == float(book_bid)
                    and float(ask) == float(book_ask)
                )

            else:

                state[
                    "ticker_book_bbo_match"
                ] = None

            result[
                instrument
            ] = state

        return result

    # --------------------------------------------------
    # Finalize L2 statistics
    # --------------------------------------------------

    def finalize_book_stats(
        self,
        instrument: str,
        bar_start_ms: int,
        bar_end_ms: int
    ) -> dict:

        # Carry the most recent valid book state to
        # the exact five-minute boundary.
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

            # Event-time monotonicity audit. A strictly
            # older snapshot is rejected before it can affect
            # the active book or event-path statistics.
            "book_out_of_order_update_count":
                0,

            "book_event_time_monotonic":
                True,

            # Exact valid clock time represented by OBI-20
            # moments and the |OBI-20| duration histogram.
            "book_obi_20_duration_ms":
                0.0,

            "book_min_obi_20":
                None,

            "book_max_obi_20":
                None,

            # --------------------------------------
            # L2 instability v0.1
            # --------------------------------------

            "book_twa_abs_obi_20":
                None,

            "book_obi_20_std":
                None,

            "book_obi_20_path_roughness_per_min":
                None,

            "book_obi_20_range":
                None,

            "book_obi_20_coherence_noise":
                None,

            # Exact number of consecutive OBI event
            # transitions observed in the bar.
            "book_obi_20_transition_count":
                0,

            # Epsilon-free raw sign crossing.
            "book_obi_20_raw_zero_cross_count":
                0,

            "book_obi_20_raw_zero_cross_rate":
                0.0,

            "book_obi_20_reversal_strength_mean":
                None,

            "book_obi_20_reversal_strength_max":
                None,

            # Distribution metadata and observations.
            "book_obi_20_histogram_schema":
                OBI_HISTOGRAM_SCHEMA,

            "book_obi_20_histogram_bin_count":
                OBI_HISTOGRAM_BIN_COUNT,

            "book_obi_20_abs_histogram_ms":
                [
                    0.0
                    for _
                    in range(
                        OBI_HISTOGRAM_BIN_COUNT
                    )
                ],

            "book_obi_20_reversal_strength_histogram":
                [
                    0
                    for _
                    in range(
                        OBI_HISTOGRAM_BIN_COUNT
                    )
                ],

            # --------------------------------------
            # v0.3 feed-integrity evidence
            # --------------------------------------

            # Largest exchange-time interval inside the bar
            # with no accepted L2 update. 300000 means the
            # book never changed during the bar.
            "book_max_update_gap_ms":
                None,

            # --------------------------------------
            # v0.3 intrawindow mid-price path
            # --------------------------------------
            # Book mid sampled on a PATH_GRID_MS clock grid
            # (30 points at 10 s). Increments are computed
            # only between consecutive valid grid points.
            "mid_path_grid_ms":
                PATH_GRID_MS,

            "mid_path_valid_points":
                0,

            "mid_path_increment_count":
                0,

            # Sum of squared log increments.
            "mid_path_realized_variance":
                None,

            "mid_path_zero_increments":
                0,

            "mid_path_nonzero_increments":
                0,

            # Sign changes between consecutive NON-ZERO
            # increments (zero increments are skipped).
            "mid_path_return_sign_flips":
                0,

            "mid_path_max_abs_log_return":
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

        result[
            "book_stats_complete"
        ] = (
            coverage_ms
            >= (
                BAR_INTERVAL_MS
                - 5
            )
        )

        update_count = int(
            accumulator[
                "update_count"
            ]
        )

        transition_count = int(
            accumulator[
                "transition_count_obi_20"
            ]
        )

        result[
            "book_update_count"
        ] = update_count

        out_of_order_update_count = int(
            accumulator[
                "out_of_order_update_count"
            ]
        )

        result[
            "book_out_of_order_update_count"
        ] = out_of_order_update_count

        result[
            "book_event_time_monotonic"
        ] = (
            out_of_order_update_count == 0
        )

        result[
            "book_obi_20_transition_count"
        ] = transition_count

        min_obi_20 = (
            accumulator[
                "min_obi_20"
            ]
        )

        max_obi_20 = (
            accumulator[
                "max_obi_20"
            ]
        )

        result[
            "book_min_obi_20"
        ] = min_obi_20

        result[
            "book_max_obi_20"
        ] = max_obi_20

        if (
            min_obi_20 is not None
            and max_obi_20 is not None
        ):

            result[
                "book_obi_20_range"
            ] = (
                max_obi_20
                - min_obi_20
            )

        # Existing time-weighted L2 metrics.
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

        # ------------------------------------------
        # OBI-20 time-weighted dispersion/coherence
        # ------------------------------------------

        obi_duration_ms = (
            accumulator[
                "weighted_duration"
            ][
                "obi_20"
            ]
        )

        result[
            "book_obi_20_duration_ms"
        ] = float(
            obi_duration_ms
        )

        if obi_duration_ms > 0:

            obi_mean = (
                accumulator[
                    "weighted_sum"
                ][
                    "obi_20"
                ]
                / obi_duration_ms
            )

            obi_second_moment = (
                accumulator[
                    "weighted_sq_sum_obi_20"
                ]
                / obi_duration_ms
            )

            obi_variance = max(
                (
                    obi_second_moment
                    - (
                        obi_mean
                        * obi_mean
                    )
                ),
                0.0
            )

            result[
                "book_obi_20_std"
            ] = math.sqrt(
                obi_variance
            )

            result[
                "book_twa_abs_obi_20"
            ] = (
                accumulator[
                    "weighted_abs_sum_obi_20"
                ]
                / obi_duration_ms
            )

            weighted_abs_sum = (
                accumulator[
                    "weighted_abs_sum_obi_20"
                ]
            )

            if weighted_abs_sum > 0:

                coherence = (
                    abs(
                        accumulator[
                            "weighted_sum"
                        ][
                            "obi_20"
                        ]
                    )
                    / weighted_abs_sum
                )

                # Numerical safety only.
                coherence = min(
                    max(
                        coherence,
                        0.0
                    ),
                    1.0
                )

                result[
                    "book_obi_20_coherence_noise"
                ] = (
                    1.0
                    - coherence
                )

            # Total variation normalized by valid
            # OBI clock time.
            result[
                "book_obi_20_path_roughness_per_min"
            ] = (
                accumulator[
                    "abs_delta_sum_obi_20"
                ]
                / (
                    obi_duration_ms
                    / 60000.0
                )
            )

        # ------------------------------------------
        # Distributional observations for epsilon
        # ------------------------------------------

        zero_cross_count = int(
            accumulator[
                "zero_cross_count_obi_20"
            ]
        )

        result[
            "book_obi_20_raw_zero_cross_count"
        ] = zero_cross_count

        result[
            "book_obi_20_raw_zero_cross_rate"
        ] = (
            zero_cross_count
            / max(
                transition_count,
                1
            )
        )

        if zero_cross_count > 0:

            result[
                "book_obi_20_reversal_strength_mean"
            ] = (
                accumulator[
                    "reversal_strength_sum_obi_20"
                ]
                / zero_cross_count
            )

            result[
                "book_obi_20_reversal_strength_max"
            ] = (
                accumulator[
                    "reversal_strength_max_obi_20"
                ]
            )

        result[
            "book_obi_20_abs_histogram_ms"
        ] = list(
            accumulator[
                "obi_20_abs_histogram_ms"
            ]
        )

        result[
            "book_obi_20_reversal_strength_histogram"
        ] = list(
            accumulator[
                "reversal_strength_histogram"
            ]
        )

        # ------------------------------------------
        # v0.3: largest update gap, including the tail
        # from the last update to the bar end.
        # ------------------------------------------

        last_update_in_bar = (
            accumulator[
                "last_update_in_bar_ms"
            ]
        )

        tail_gap_ms = float(
            bar_end_ms
            - (
                last_update_in_bar
                if last_update_in_bar
                is not None
                else bar_start_ms
            )
        )

        result[
            "book_max_update_gap_ms"
        ] = max(
            float(
                accumulator[
                    "max_update_gap_ms"
                ]
            ),
            tail_gap_ms
        )

        # ------------------------------------------
        # v0.3: intrawindow mid-price path statistics
        # ------------------------------------------

        grid = accumulator[
            "path_grid_mid"
        ]

        result[
            "mid_path_valid_points"
        ] = sum(
            1
            for value in grid
            if value is not None
            and value > 0
        )

        increments = []

        for left, right in zip(
            grid[:-1],
            grid[1:]
        ):

            if (
                left is None
                or right is None
                or left <= 0
                or right <= 0
            ):
                continue

            increments.append(
                math.log(
                    right
                    / left
                )
            )

        result[
            "mid_path_increment_count"
        ] = len(
            increments
        )

        if increments:

            result[
                "mid_path_realized_variance"
            ] = sum(
                value * value
                for value in increments
            )

            result[
                "mid_path_max_abs_log_return"
            ] = max(
                abs(value)
                for value in increments
            )

            nonzero = [
                value
                for value in increments
                if value != 0.0
            ]

            result[
                "mid_path_zero_increments"
            ] = (
                len(increments)
                - len(nonzero)
            )

            result[
                "mid_path_nonzero_increments"
            ] = len(
                nonzero
            )

            result[
                "mid_path_return_sign_flips"
            ] = sum(
                1
                for left, right in zip(
                    nonzero[:-1],
                    nonzero[1:]
                )
                if left * right < 0
            )

        return result

    # --------------------------------------------------
    # Finalize 5-minute bar
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

            # ------------------------------------------
            # v0.3 stream completeness
            # ------------------------------------------
            #
            # A bar is complete only if no evidence of a
            # collection fault touches it. Each failed
            # condition is written as an explicit reason.

            feed_max_gap_ms = (
                self.feed_gap_for_bar(
                    bar_start_ms,
                    bar_end_ms
                )
            )

            incomplete_reasons = []

            if (
                self.collector_started_ms
                > bar_start_ms
            ):
                incomplete_reasons.append(
                    "collector_start"
                )

            if key in self.incomplete_bar_keys:
                incomplete_reasons.append(
                    "outage_or_stale_interval"
                )

            if (
                self.degraded_since_ms
                is not None
                and self.degraded_since_ms
                < bar_end_ms
            ):
                incomplete_reasons.append(
                    "outage_open_at_finalize"
                )

            open_stale_sources = sorted(
                source
                for (
                    stale_instrument,
                    source
                ), stale_state in (
                    self.stale_streams.items()
                )
                if stale_instrument == instrument
                and stale_state["since_ms"]
                < bar_end_ms
            )

            if open_stale_sources:
                incomplete_reasons.append(
                    "stale_stream_open:"
                    + "+".join(
                        open_stale_sources
                    )
                )

            subscription_confirmed = (
                self.confirmed_channels
                is not None
                and instrument
                not in self.unconfirmed_instruments
            )

            if not subscription_confirmed:
                incomplete_reasons.append(
                    "subscription_unconfirmed"
                )

            if (
                feed_max_gap_ms is None
                or feed_max_gap_ms
                > FEED_MAX_GAP_FOR_COMPLETE_MS
            ):
                incomplete_reasons.append(
                    "feed_message_gap"
                )

            stream_complete = (
                len(incomplete_reasons) == 0
            )

            self.incomplete_bar_keys.discard(
                key
            )

            verify = self.verify_stats.pop(
                key,
                {}
            )

            trade_evidence = (
                bar
                if bar is not None
                else {}
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
                    "deribit_ws_100ms_l2",

                # ------------------------------------------
                # v0.3 provenance / integrity evidence
                # ------------------------------------------

                "collector_version":
                    COLLECTOR_VERSION,

                "collector_code_sha256":
                    COLLECTOR_CODE_SHA256,

                "connection_count":
                    self.connection_count,

                "stream_incomplete_reason":
                    (
                        ",".join(
                            incomplete_reasons
                        )
                        if incomplete_reasons
                        else None
                    ),

                "subscription_confirmed":
                    subscription_confirmed,

                "feed_max_message_gap_ms":
                    feed_max_gap_ms,

                "trades_checked_vs_book":
                    int(
                        trade_evidence.get(
                            "trades_checked_vs_book",
                            0
                        )
                    ),

                "trades_outside_book_bbo":
                    int(
                        trade_evidence.get(
                            "trades_outside_book_bbo",
                            0
                        )
                    ),

                "max_trade_outside_book_bbo_bps":
                    float(
                        trade_evidence.get(
                            "max_trade_outside_book_bbo_bps",
                            0.0
                        )
                    ),

                "rest_verify_count":
                    int(
                        verify.get(
                            "checks",
                            0
                        )
                    ),

                "rest_verify_mismatch_count":
                    int(
                        verify.get(
                            "mismatches",
                            0
                        )
                    ),

                "rest_stale_confirmed":
                    bool(
                        verify.get(
                            "stale_confirmed",
                            False
                        )
                    )
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

        # v0.3: release per-bar integrity state for every
        # bucket that can no longer be finalized.
        for bucket in [
            bucket
            for bucket in self.feed_gap_by_bucket
            if bucket < bar_end_ms
        ]:
            self.feed_gap_by_bucket.pop(
                bucket,
                None
            )

        for stale_key in [
            stale_key
            for stale_key in self.verify_stats
            if stale_key[1] < bar_end_ms
        ]:
            self.verify_stats.pop(
                stale_key,
                None
            )

        for stale_key in [
            stale_key
            for stale_key in self.book_bar_accumulators
            if stale_key[1] < bar_start_ms
        ]:
            # Orphan accumulators (for example an
            # out-of-order audit count that landed on an
            # already-finalized bucket) are released.
            self.book_bar_accumulators.pop(
                stale_key,
                None
            )

        return pd.DataFrame(
            rows
        )

    # --------------------------------------------------
    # v0.3 feed-gap accounting
    # --------------------------------------------------

    def record_message_received(
        self,
        received_ms: int
    ) -> None:
        """
        Track the largest interval without any WebSocket
        message, split across the five-minute buckets it
        overlaps (local receive clock).
        """

        previous_ms = (
            self.last_message_received_ms
        )

        if (
            previous_ms is not None
            and received_ms > previous_ms
        ):

            cursor = previous_ms

            while cursor < received_ms:

                bucket_start = (
                    cursor
                    - (
                        cursor
                        % BAR_INTERVAL_MS
                    )
                )

                segment_end = min(
                    received_ms,
                    bucket_start
                    + BAR_INTERVAL_MS
                )

                self.feed_gap_by_bucket[
                    bucket_start
                ] = max(
                    self.feed_gap_by_bucket.get(
                        bucket_start,
                        0.0
                    ),
                    float(
                        segment_end
                        - cursor
                    )
                )

                cursor = segment_end

        if (
            previous_ms is None
            or received_ms > previous_ms
        ):

            self.last_message_received_ms = (
                received_ms
            )

    def feed_gap_for_bar(
        self,
        bar_start_ms: int,
        bar_end_ms: int
    ) -> float | None:
        """
        Largest message-free interval inside the bar,
        including any silence still open at bar end.
        """

        if self.last_message_received_ms is None:
            return None

        gap_ms = self.feed_gap_by_bucket.get(
            bar_start_ms,
            0.0
        )

        if (
            self.last_message_received_ms
            < bar_end_ms
        ):

            gap_ms = max(
                gap_ms,
                float(
                    bar_end_ms
                    - max(
                        self.last_message_received_ms,
                        bar_start_ms
                    )
                )
            )

        return gap_ms

    # --------------------------------------------------
    # v0.3 silence / subscription watchdog
    # --------------------------------------------------

    async def feed_watchdog(
        self
    ) -> None:
        """
        Force a reconnect when the connection stops
        delivering data or never confirms its subscription.
        Ping/pong alone cannot detect either condition.
        """

        while not self.stop_event.is_set():

            await asyncio.sleep(
                1.0
            )

            now_ms = int(
                time.time()
                * 1000
            )

            reference_ms = max(
                self.last_message_received_ms
                or 0,
                self.connection_started_ms
                or 0
            )

            if (
                reference_ms > 0
                and now_ms - reference_ms
                > FEED_SILENCE_RECONNECT_SECONDS
                * 1000
            ):

                self.feed_silence_reconnects += 1

                raise FeedStaleError(
                    "Futures feed silent for "
                    f"{(now_ms - reference_ms) / 1000:.1f}s."
                )

            if (
                self.confirmed_channels is None
                and self.subscription_sent_ms
                is not None
                and now_ms
                - self.subscription_sent_ms
                > SUBSCRIPTION_CONFIRM_TIMEOUT_SECONDS
                * 1000
            ):

                self.subscription_failures += 1

                raise FeedStaleError(
                    "Futures subscription was not "
                    "confirmed within "
                    f"{SUBSCRIPTION_CONFIRM_TIMEOUT_SECONDS:.0f}s."
                )

    # --------------------------------------------------
    # v0.3 REST cross-check of quiet streams
    # --------------------------------------------------

    def _verify_bucket_stats(
        self,
        instrument: str
    ) -> dict:

        now_ms = int(
            time.time()
            * 1000
        )

        bucket = (
            now_ms
            - (
                now_ms
                % BAR_INTERVAL_MS
            )
        )

        return self.verify_stats.setdefault(
            (
                instrument,
                bucket
            ),
            {
                "checks": 0,
                "mismatches": 0,
                "stale_confirmed": False
            }
        )

    def _cached_tops(
        self,
        instrument: str
    ) -> tuple[
        tuple | None,
        tuple | None
    ]:
        """
        Return (book_bbo, ticker_bbo) from the stream cache.
        """

        book_bbo = None

        book = self.order_book_state.get(
            instrument
        )

        if (
            book
            and book.get("valid", False)
            and book.get("bids")
            and book.get("asks")
        ):

            book_bbo = (
                float(book["bids"][0][0]),
                float(book["asks"][0][0])
            )

        ticker_bbo = None

        ticker = self.ticker_state.get(
            instrument
        )

        if ticker:

            bid = ticker.get(
                "best_bid_price"
            )

            ask = ticker.get(
                "best_ask_price"
            )

            if (
                bid is not None
                and ask is not None
                and bid > 0
                and ask > 0
            ):

                ticker_bbo = (
                    float(bid),
                    float(ask)
                )

        return book_bbo, ticker_bbo

    async def _stale_sources(
        self,
        instrument: str,
        check_book: bool,
        check_ticker: bool
    ) -> list[str]:
        """
        Two-step confirmation. A source is stale only if its
        cached top of book differs from REST on two checks
        STALE_CONFIRM_DELAY_SECONDS apart AND the stream did
        not deliver any change in between.
        """

        stats = self._verify_bucket_stats(
            instrument
        )

        rest_bbo = await asyncio.to_thread(
            fetch_rest_top_of_book,
            instrument
        )

        self.rest_verifications += 1
        stats["checks"] += 1

        if rest_bbo is None:

            self.rest_verification_errors += 1

            return []

        book_bbo, ticker_bbo = (
            self._cached_tops(
                instrument
            )
        )

        suspects = []

        if (
            check_book
            and book_bbo is not None
            and book_bbo != rest_bbo
        ):
            suspects.append(
                "book"
            )

        if (
            check_ticker
            and ticker_bbo is not None
            and ticker_bbo != rest_bbo
        ):
            suspects.append(
                "ticker"
            )

        if not suspects:
            return []

        stats["mismatches"] += 1

        book_marker = (
            self.book_last_update_received_ms.get(
                instrument
            )
        )

        ticker_marker = (
            self.ticker_bbo_last_change
            .get(instrument, {})
            .get("received_ms")
        )

        await asyncio.sleep(
            STALE_CONFIRM_DELAY_SECONDS
        )

        rest_bbo_2 = await asyncio.to_thread(
            fetch_rest_top_of_book,
            instrument
        )

        self.rest_verifications += 1
        stats["checks"] += 1

        if rest_bbo_2 is None:

            self.rest_verification_errors += 1

            return []

        book_bbo_2, ticker_bbo_2 = (
            self._cached_tops(
                instrument
            )
        )

        confirmed = []

        if (
            "book" in suspects
            and self.book_last_update_received_ms.get(
                instrument
            ) == book_marker
            and book_bbo_2 is not None
            and book_bbo_2 != rest_bbo_2
        ):
            confirmed.append(
                "book"
            )

        if (
            "ticker" in suspects
            and self.ticker_bbo_last_change
            .get(instrument, {})
            .get("received_ms") == ticker_marker
            and ticker_bbo_2 is not None
            and ticker_bbo_2 != rest_bbo_2
        ):
            confirmed.append(
                "ticker"
            )

        return confirmed

    async def resubscribe_channels(
        self,
        channels: list[str],
        label: str
    ) -> None:
        """
        v0.3.1: refresh individual channels on the existing
        connection (unsubscribe + subscribe). Other
        instruments and channels keep streaming, so no data
        is lost elsewhere. A fresh subscription delivers a
        new full snapshot for book and ticker channels.
        """

        websocket = self.websocket

        if websocket is None or not channels:
            return

        for method in [
            "public/unsubscribe",
            "public/subscribe"
        ]:

            self.resubscribe_request_id += 1

            await websocket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": self.resubscribe_request_id,
                        "method": method,
                        "params": {
                            "channels": channels
                        }
                    }
                )
            )

        self.channel_resubscriptions += 1

        print(
            f"\nResubscribed {label}: "
            f"{', '.join(channels)}"
        )

    def _stale_source_channel(
        self,
        instrument: str,
        source: str
    ) -> str:

        if source == "book":
            return self.book_channel(
                instrument
            )

        return (
            f"incremental_ticker."
            f"{instrument}"
        )

    def _close_stale_state(
        self,
        instrument: str,
        source: str
    ) -> None:

        stale_key = (
            instrument,
            source
        )

        state = self.stale_streams.pop(
            stale_key,
            None
        )

        if state is None:
            return

        end_ms = int(
            time.time()
            * 1000
        )

        self.mark_interval_incomplete(
            state["since_ms"],
            end_ms,
            [instrument]
        )

        print(
            f"\nStale {source} stream for {instrument} "
            "resolved (stream live / matches REST). "
            "Interval marked incomplete: "
            f"{(end_ms - state['since_ms']) / 1000:.1f}s."
        )

    async def _recheck_stale_streams(
        self
    ) -> None:
        """
        For every open stale state: clear it when the cached
        stream top of book matches REST again; otherwise
        keep the instrument marked and resubscribe the
        channel at most once per STALE_RESUBSCRIBE_MIN_SECONDS.
        """

        for (
            instrument,
            source
        ) in list(
            self.stale_streams
        ):

            state = self.stale_streams.get(
                (instrument, source)
            )

            if state is None:
                continue

            # A live stream delivers more than the single
            # snapshot that follows a resubscription. Two or
            # more changes since the last resubscription mean
            # the channel is updating again; REST equality is
            # not required (a busy book rarely matches a REST
            # snapshot taken milliseconds apart).
            changes_since = (
                self.stream_change_counter.get(
                    (instrument, source),
                    0
                )
                - state["counter_at_resubscribe"]
            )

            if changes_since >= 2:

                self._close_stale_state(
                    instrument,
                    source
                )

                continue

            rest_bbo = await asyncio.to_thread(
                fetch_rest_top_of_book,
                instrument
            )

            self.rest_verifications += 1

            self._verify_bucket_stats(
                instrument
            )["checks"] += 1

            if rest_bbo is None:

                self.rest_verification_errors += 1

                continue

            book_bbo, ticker_bbo = (
                self._cached_tops(
                    instrument
                )
            )

            cached = (
                book_bbo
                if source == "book"
                else ticker_bbo
            )

            if (
                cached is not None
                and cached == rest_bbo
            ):

                self._close_stale_state(
                    instrument,
                    source
                )

                continue

            self._verify_bucket_stats(
                instrument
            )["mismatches"] += 1

            now_ms = int(
                time.time()
                * 1000
            )

            # Keep the ongoing interval marked so bars that
            # close while the stream is still stale are
            # incomplete for this instrument.
            self.mark_interval_incomplete(
                state["since_ms"],
                now_ms,
                [instrument]
            )

            if (
                now_ms
                - state["last_resubscribe_ms"]
                >= STALE_RESUBSCRIBE_MIN_SECONDS
                * 1000
            ):

                state["last_resubscribe_ms"] = now_ms

                state["counter_at_resubscribe"] = (
                    self.stream_change_counter.get(
                        (instrument, source),
                        0
                    )
                )

                await self.resubscribe_channels(
                    [
                        self._stale_source_channel(
                            instrument,
                            source
                        )
                    ],
                    f"stale {source} stream "
                    f"({instrument}, still stale)"
                )

    async def stream_verifier(
        self
    ) -> None:
        """
        Periodically cross-check quiet streams against REST.

        v0.3.1: a confirmed stale source no longer forces a
        full reconnect. Live data (2026-10-05 14:35-14:40 UTC)
        showed exchange-side book freezes that survive a
        reconnect, so reconnecting every ~45 s only destroyed
        trades and tickers for every other instrument.

        Instead the affected instrument/source enters a stale
        state: its bars are finalized incomplete with reason
        'stale_stream', the single channel is resubscribed
        (rate-limited), and the state is cleared when the
        stream again matches REST.
        """

        while not self.stop_event.is_set():

            try:

                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=(
                        STREAM_VERIFY_INTERVAL_SECONDS
                    )
                )

                return

            except asyncio.TimeoutError:

                pass

            if self.confirmed_channels is None:
                continue

            await self._recheck_stale_streams()

            for instrument in list(
                self.instrument_names
            ):

                now_ms = int(
                    time.time()
                    * 1000
                )

                book_received = (
                    self.book_last_update_received_ms.get(
                        instrument
                    )
                )

                ticker_received = (
                    self.ticker_bbo_last_change
                    .get(instrument, {})
                    .get("received_ms")
                )

                quiet_ms = (
                    BOOK_QUIET_VERIFY_SECONDS
                    * 1000
                )

                check_book = (
                    book_received is not None
                    and now_ms - book_received
                    > quiet_ms
                    and (instrument, "book")
                    not in self.stale_streams
                )

                check_ticker = (
                    ticker_received is not None
                    and now_ms - ticker_received
                    > quiet_ms
                    and (instrument, "ticker")
                    not in self.stale_streams
                )

                if not (
                    check_book
                    or check_ticker
                ):
                    continue

                confirmed = await self._stale_sources(
                    instrument,
                    check_book,
                    check_ticker
                )

                if not confirmed:
                    continue

                end_ms = int(
                    time.time()
                    * 1000
                )

                self._verify_bucket_stats(
                    instrument
                )[
                    "stale_confirmed"
                ] = True

                for source in confirmed:

                    since_ms = (
                        book_received
                        if source == "book"
                        else ticker_received
                    )

                    self.stale_feeds_detected += 1

                    self.stale_streams[
                        (instrument, source)
                    ] = {
                        "since_ms":
                            since_ms,

                        "last_resubscribe_ms":
                            end_ms,

                        "counter_at_resubscribe":
                            self.stream_change_counter.get(
                                (instrument, source),
                                0
                            )
                    }

                    self.mark_interval_incomplete(
                        since_ms,
                        end_ms,
                        [instrument]
                    )

                    print(
                        f"\nStale {source} stream "
                        f"confirmed by REST for {instrument}; "
                        "no stream change for "
                        f"{(end_ms - since_ms) / 1000:.1f}s. "
                        "Instrument marked; other "
                        "instruments unaffected."
                    )

                    await self.resubscribe_channels(
                        [
                            self._stale_source_channel(
                                instrument,
                                source
                            )
                        ],
                        f"stale {source} stream "
                        f"({instrument})"
                    )

    # --------------------------------------------------
    # WebSocket receive loop
    # --------------------------------------------------

    async def receive_messages(
        self,
        websocket
    ) -> None:

        async for raw_message in websocket:

            if self.stop_event.is_set():

                break

            # v0.3: every received message advances the
            # feed-gap clock (local receive time).
            self.record_message_received(
                int(
                    time.time()
                    * 1000
                )
            )

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

                # A rejected subscription request means this
                # connection is not collecting what it should.
                if (
                    message.get("id")
                    == self.subscribe_request_id
                ):

                    self.subscription_failures += 1

                    raise FeedStaleError(
                        "Futures subscription "
                        "request was rejected."
                    )

                continue

            if "id" in message:

                if (
                    "result" in message
                    and message.get("id")
                    == self.subscribe_request_id
                ):

                    self.handle_subscription_confirmation(
                        message[
                            "result"
                        ]
                    )

                    self.check_resynchronized()

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

                else:

                    self.book_parse_errors += 1

            self.messages_received += 1

            # v0.3: close an open outage as soon as the new
            # connection is fully re-synchronized.
            if self.degraded_since_ms is not None:

                self.check_resynchronized()

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

                if timestamp_ms is not None:

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

                if timestamp_ms is not None:

                    book_ages.append(
                        now_ms
                        - timestamp_ms
                    )

            print(
                "\n[FUTURES CACHE HEALTH]"
            )

            print(
                f"Perpetuals:           "
                f"{expected}"
            )

            print(
                f"Tickers initialized:  "
                f"{tickers_initialized}/"
                f"{expected}"
            )

            print(
                f"L2 books initialized: "
                f"{books_initialized}/"
                f"{expected}"
            )

            print(
                f"Trades received:      "
                f"{self.trades_received:,}"
            )

            print(
                f"L2 messages:          "
                f"{self.book_messages_received:,}"
            )

            print(
                f"L2 parse errors:      "
                f"{self.book_parse_errors:,}"
            )

            print(
                f"Out-of-order L2:      "
                f"{self.book_out_of_order_updates:,}"
            )

            print(
                f"Late trades ignored:  "
                f"{self.ignored_late_trades:,}"
            )

            # v0.3 integrity counters.
            print(
                f"Subscriptions:        "
                f"{len(self.confirmed_channels or [])}/"
                f"{len(self.requested_channels)} confirmed"
            )

            print(
                f"REST verifications:   "
                f"{self.rest_verifications:,} "
                f"(errors {self.rest_verification_errors:,})"
            )

            print(
                f"Stale feeds detected: "
                f"{self.stale_feeds_detected:,}"
            )

            print(
                f"Silence reconnects:   "
                f"{self.feed_silence_reconnects:,}"
            )

            print(
                f"Outage open:          "
                f"{self.degraded_since_ms is not None}"
            )

            print(
                f"Stale streams open:   "
                f"{len(self.stale_streams)} "
                f"(resubscriptions {self.channel_resubscriptions:,})"
            )

            if ticker_ages:

                print(
                    f"Median ticker age:    "
                    f"{statistics.median(ticker_ages):,.0f} ms"
                )

            if book_ages:

                print(
                    f"Median L2 age:        "
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

            self.connection_started_ms = int(
                time.time()
                * 1000
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

            # v0.3: silence/subscription watchdog and REST
            # cross-check of quiet streams. Either task ends
            # this connection by raising FeedStaleError.
            watchdog_task = (
                asyncio.create_task(
                    self.feed_watchdog()
                )
            )

            verifier_task = (
                asyncio.create_task(
                    self.stream_verifier()
                )
            )

            shutdown_task = (
                asyncio.create_task(
                    self.stop_event.wait()
                )
            )

            monitored = {
                receive_task,
                watchdog_task,
                verifier_task,
                shutdown_task
            }

            try:

                done, pending = (
                    await asyncio.wait(
                        monitored,
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

                for task in (
                    receive_task,
                    watchdog_task,
                    verifier_task
                ):

                    if task in done:

                        exception = (
                            task.exception()
                        )

                        if exception is not None:

                            raise exception

            finally:

                for task in (
                    receive_task,
                    health_task,
                    watchdog_task,
                    verifier_task,
                    shutdown_task
                ):

                    if not task.done():

                        task.cancel()

                    try:

                        await task

                    except asyncio.CancelledError:

                        pass

                    except Exception:

                        # The primary exception (if any) was
                        # already raised above.
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

                    # v0.3: open an outage interval that
                    # stays open until the next connection
                    # is fully re-synchronized.
                    self.begin_outage()

                    print(
                        "\nFutures WebSocket "
                        "closed unexpectedly."
                    )

            except asyncio.CancelledError:

                break

            except Exception as exc:

                if self.stop_event.is_set():

                    break

                self.begin_outage()

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
