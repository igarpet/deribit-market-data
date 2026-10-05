"""
Deterministic offline tests for futures_state_cache v0.3.

No network: WebSocket messages are fed directly into the
cache and REST is replaced by a fake. Wall-clock dependent
pieces are driven by patching time.time().
"""

import asyncio
import math
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, "src")

import futures_state_cache as fsc  # noqa: E402

BAR = fsc.BAR_INTERVAL_MS
T0 = 1_790_000_100_000 - (1_790_000_100_000 % BAR)  # a bar start
INSTR = "BTC_USDC-PERPETUAL"

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


class Clock:
    def __init__(self, ms):
        self.ms = ms

    def __call__(self):
        return self.ms / 1000.0


def new_cache(clock):
    fsc.time.time = clock
    instruments = pd.DataFrame([{
        "instrument_name": INSTR,
        "base_currency": "BTC",
        "instrument_type": "linear",
        "settlement_currency": "USDC",
        "quote_currency": "USDC",
        "api_currency": "USDC",
        "contract_size": 0.001,
    }])
    cache = fsc.FuturesStateCache(instruments, asyncio.Event())
    cache.collector_started_ms = T0 - BAR  # started before test bar
    return cache


def book(ts, bid, ask, bid_sz=1.0, ask_sz=1.0):
    return {
        "instrument_name": INSTR,
        "timestamp": ts,
        "bids": [[bid - i * 1.0, bid_sz] for i in range(20)],
        "asks": [[ask + i * 1.0, ask_sz] for i in range(20)],
    }


def confirm(cache):
    cache.requested_channels = set(cache.build_channels())
    cache.handle_subscription_confirmation(sorted(cache.requested_channels))


def feed_messages(cache, clock, start_ms, end_ms, every_ms):
    t = start_ms
    while t <= end_ms:
        clock.ms = t
        cache.record_message_received(t)
        t += every_ms


# --------------------------------------------------
# 1. Book path: invariants, max gap, mid path stats
# --------------------------------------------------
clock = Clock(T0)
c = new_cache(clock)
confirm(c)

# Book before bar start, then updates at +25s (up), +55s (down), +95s (up)
c.process_book(book(T0 - 1000, 100000.0, 100001.0))
c.process_book(book(T0 + 25_000, 100010.0, 100011.0, bid_sz=3.0))
c.process_book(book(T0 + 55_000, 100005.0, 100006.0, ask_sz=3.0))
c.process_book(book(T0 + 95_000, 100020.0, 100021.0))
feed_messages(c, clock, T0, T0 + BAR + 1000, 500)
stats = c.finalize_book_stats(INSTR, T0, T0 + BAR)

check("histogram ms sums to obi duration",
      abs(sum(stats["book_obi_20_abs_histogram_ms"]) - stats["book_obi_20_duration_ms"]) < 1e-6,
      f'{sum(stats["book_obi_20_abs_histogram_ms"])} vs {stats["book_obi_20_duration_ms"]}')
check("reversal histogram sums to zero-cross count",
      sum(stats["book_obi_20_reversal_strength_histogram"]) == stats["book_obi_20_raw_zero_cross_count"])
check("coverage complete", stats["book_stats_complete"] is True)
check("max update gap = tail 205s", stats["book_max_update_gap_ms"] == 205_000.0,
      stats["book_max_update_gap_ms"])
check("30 valid grid points", stats["mid_path_valid_points"] == 30, stats["mid_path_valid_points"])
check("29 increments", stats["mid_path_increment_count"] == 29)
# grid points: 0,10,20 -> 100000.5; 30,40,50 -> 100010.5; 60..90 -> 100005.5; 100.. -> 100020.5
check("3 nonzero increments", stats["mid_path_nonzero_increments"] == 3, stats["mid_path_nonzero_increments"])
check("2 sign flips (up,down,up)", stats["mid_path_return_sign_flips"] == 2, stats["mid_path_return_sign_flips"])
expected_rv = (math.log(100010.5 / 100000.5) ** 2 + math.log(100005.5 / 100010.5) ** 2
               + math.log(100020.5 / 100005.5) ** 2)
check("realized variance exact", abs(stats["mid_path_realized_variance"] - expected_rv) < 1e-18)

# Frozen book: no update in bar at all
clock2 = Clock(T0)
c2 = new_cache(clock2)
confirm(c2)
c2.process_book(book(T0 - 5000, 100000.0, 100001.0))
stats2 = c2.finalize_book_stats(INSTR, T0, T0 + BAR)
check("frozen book: gap = full bar", stats2["book_max_update_gap_ms"] == float(BAR))
check("frozen book: all 29 increments zero", stats2["mid_path_zero_increments"] == 29)
check("frozen book still 'coverage complete' (evidence now separate)", stats2["book_stats_complete"] is True)

# --------------------------------------------------
# 2. Trades vs book BBO
# --------------------------------------------------
clock.ms = T0 + 100_000
c.process_trade({"instrument_name": INSTR, "timestamp": T0 + 100_000, "price": 100020.0,
                 "amount": 0.1, "direction": "sell", "trade_id": "a"})
c.process_trade({"instrument_name": INSTR, "timestamp": T0 + 100_100, "price": 100500.0,
                 "amount": 0.1, "direction": "buy", "trade_id": "b"})
bar = c.bars[(INSTR, T0)]
check("2 trades checked vs book", bar["trades_checked_vs_book"] == 2)
check("1 trade outside BBO", bar["trades_outside_book_bbo"] == 1)
expected_bps = (100500.0 - 100021.0) / 100020.5 * 1e4
check("outside distance bps", abs(bar["max_trade_outside_book_bbo_bps"] - expected_bps) < 1e-9,
      round(bar["max_trade_outside_book_bbo_bps"], 3))

# --------------------------------------------------
# 3. Ticker BBO change tracking
# --------------------------------------------------
clock.ms = T0 + 1000
c.process_ticker({"instrument_name": INSTR, "type": "snapshot", "timestamp": T0 + 1000,
                  "best_bid_price": 100.0, "best_ask_price": 101.0, "mark_price": 100.5})
clock.ms = T0 + 5000
c.process_ticker({"instrument_name": INSTR, "type": "change", "timestamp": T0 + 5000,
                  "mark_price": 100.6})
check("mark-only update does not move BBO change time",
      c.ticker_bbo_last_change[INSTR]["exchange_ms"] == T0 + 1000)
clock.ms = T0 + 9000
c.process_ticker({"instrument_name": INSTR, "type": "change", "timestamp": T0 + 9000,
                  "best_ask_price": 101.5})
check("ask change moves BBO change time", c.ticker_bbo_last_change[INSTR]["exchange_ms"] == T0 + 9000)
boundary = c.capture_boundary_state(datetime.fromtimestamp((T0 + 30_000) / 1000, tz=timezone.utc))[INSTR]
check("ticker_bbo_age_ms at boundary", boundary["ticker_bbo_age_ms"] == 21_000, boundary["ticker_bbo_age_ms"])
check("ticker ticker_age_ms uses last any-field update", boundary["ticker_age_ms"] == 21_000)
check("book top fields present", boundary["book_best_bid_price"] == 100020.0)
check("ticker/book BBO mismatch flagged", boundary["ticker_book_bbo_match"] is False)

# --------------------------------------------------
# 4. Feed gap -> stream_complete False with reason
# --------------------------------------------------
clock3 = Clock(T0)
c3 = new_cache(clock3)
confirm(c3)
c3.process_book(book(T0 - 1000, 100.0, 101.0))
feed_messages(c3, clock3, T0, T0 + 60_000, 500)
feed_messages(c3, clock3, T0 + 80_000, T0 + BAR + 1000, 500)  # 20 s silence
clock3.ms = T0 + BAR + 1000
bnd = c3.capture_boundary_state(datetime.fromtimestamp((T0 + BAR) / 1000, tz=timezone.utc))
rows = c3.finalize_bar(datetime.fromtimestamp((T0 + BAR) / 1000, tz=timezone.utc), bnd)
r = rows.iloc[0]
check("gap 20s recorded", r["feed_max_message_gap_ms"] == 20_000.0, r["feed_max_message_gap_ms"])
check("gap -> incomplete", bool(r["stream_complete"]) is False)
check("reason feed_message_gap", r["stream_incomplete_reason"] == "feed_message_gap",
      r["stream_incomplete_reason"])
check("version stamped", r["collector_version"] == fsc.COLLECTOR_VERSION and isinstance(r["collector_code_sha256"], str))

# Clean bar afterwards
feed_messages(c3, clock3, T0 + BAR + 1500, T0 + 2 * BAR + 1000, 500)
c3.process_book(book(T0 + BAR + 2000, 100.0, 101.0))
bnd = c3.capture_boundary_state(datetime.fromtimestamp((T0 + 2 * BAR) / 1000, tz=timezone.utc))
r2 = c3.finalize_bar(datetime.fromtimestamp((T0 + 2 * BAR) / 1000, tz=timezone.utc), bnd).iloc[0]
check("clean bar complete", bool(r2["stream_complete"]) is True, r2["stream_incomplete_reason"])

# --------------------------------------------------
# 5. Outage spanning three bars, closed only on resync
# --------------------------------------------------
clock4 = Clock(T0)
c4 = new_cache(clock4)
confirm(c4)
c4.ticker_state[INSTR] = {"best_bid_price": 1.0, "best_ask_price": 2.0}
c4.process_book(book(T0 - 1000, 100.0, 101.0))
feed_messages(c4, clock4, T0, T0 + BAR - 20_000, 500)  # last message 20 s before bar end
clock4.ms = T0 + BAR + 10_000                           # detected 30 s later (ping timeout)
c4.begin_outage()
check("outage start = last message", c4.degraded_since_ms == T0 + BAR - 20_000)
check("bar 0 marked", (INSTR, T0) in c4.incomplete_bar_keys)
check("bar 1 marked", (INSTR, T0 + BAR) in c4.incomplete_bar_keys)
c4.reset_realtime_state()
# finalize bar 1 while outage still open
clock4.ms = T0 + 2 * BAR + 1000
bnd = c4.capture_boundary_state(datetime.fromtimestamp((T0 + 2 * BAR) / 1000, tz=timezone.utc))
r = c4.finalize_bar(datetime.fromtimestamp((T0 + 2 * BAR) / 1000, tz=timezone.utc), bnd).iloc[0]
check("bar during open outage incomplete", bool(r["stream_complete"]) is False, r["stream_incomplete_reason"])
check("reason includes outage_open_at_finalize", "outage_open_at_finalize" in r["stream_incomplete_reason"])
# resync in bar 2
clock4.ms = T0 + 2 * BAR + 40_000
c4.record_message_received(clock4.ms)
confirm(c4)
c4.process_ticker({"instrument_name": INSTR, "type": "snapshot", "timestamp": clock4.ms,
                   "best_bid_price": 100.0, "best_ask_price": 101.0})
c4.process_book(book(clock4.ms, 100.0, 101.0))
c4.check_resynchronized()
check("outage closed after resync", c4.degraded_since_ms is None)
check("resync bar marked", (INSTR, T0 + 2 * BAR) in c4.incomplete_bar_keys)

# --------------------------------------------------
# 6. Subscription confirmation with a missing channel
# --------------------------------------------------
clock5 = Clock(T0)
c5 = new_cache(clock5)
c5.requested_channels = set(c5.build_channels())
c5.handle_subscription_confirmation(
    [ch for ch in c5.requested_channels if not ch.startswith("book.")])
check("missing book channel -> instrument unconfirmed", INSTR in c5.unconfirmed_instruments)

# --------------------------------------------------
# 7. REST stale verification (fake REST) - v0.3.1 behaviour
# --------------------------------------------------
class FakeWebSocket:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        import json as _json
        self.sent.append(_json.loads(payload))


REST = {"bbo": (100.0, 101.0)}
fsc.fetch_rest_top_of_book = lambda name: REST["bbo"]
fsc.STALE_CONFIRM_DELAY_SECONDS = 0.0
fsc.STREAM_VERIFY_INTERVAL_SECONDS = 0.01


def make_verifier_cache():
    clock6 = Clock(T0)
    c6 = new_cache(clock6)
    confirm(c6)
    c6.websocket = FakeWebSocket()
    c6.ticker_state[INSTR] = {"best_bid_price": 100.0, "best_ask_price": 101.0}
    c6.process_book(book(T0, 100.0, 101.0))
    c6.ticker_bbo_last_change[INSTR] = {"exchange_ms": T0, "received_ms": T0}
    c6.book_last_update_received_ms[INSTR] = T0
    return clock6, c6


def run_verifier(c6, seconds=0.2):
    async def go():
        c6.stop_event = asyncio.Event()  # bind to this run's event loop
        try:
            await asyncio.wait_for(c6.stream_verifier(), timeout=seconds)
        except asyncio.TimeoutError:
            return None
        except fsc.FeedStaleError as exc:
            return exc
    return asyncio.run(go())


# 7a. quiet but matching REST
REST["bbo"] = (100.0, 101.0)
clock6, c6 = make_verifier_cache()
clock6.ms = T0 + 30_000
outcome = run_verifier(c6)
check("quiet but matching REST -> no action", outcome is None and not c6.stale_streams)
check("REST checks counted", c6.rest_verifications >= 1, c6.rest_verifications)

# 7b. stale confirmed -> no reconnect, instrument flagged, one channel resubscribed
REST["bbo"] = (150.0, 151.0)
clock7, c7 = make_verifier_cache()
c7.ticker_bbo_last_change[INSTR] = {"exchange_ms": T0 + 29_000, "received_ms": T0 + 29_000}
clock7.ms = T0 + 30_000
outcome = run_verifier(c7)
check("stale confirmed -> no reconnect raised", outcome is None, str(outcome))
check("book stale state open", (INSTR, "book") in c7.stale_streams)
check("ticker not flagged (not quiet)", (INSTR, "ticker") not in c7.stale_streams)
check("stale interval marked", (INSTR, T0) in c7.incomplete_bar_keys)
check("stale counter", c7.stale_feeds_detected == 1, c7.stale_feeds_detected)
methods = [(m["method"], tuple(m["params"]["channels"])) for m in c7.websocket.sent]
book_ch = c7.book_channel(INSTR)
check("unsubscribe+subscribe of book channel only",
      methods == [("public/unsubscribe", (book_ch,)), ("public/subscribe", (book_ch,))], methods)

# 7c. frozen after resubscribe (one snapshot, REST still differs): stays stale, rate-limited
c7.process_book(book(T0 + 30_500, 100.0, 101.0))          # resubscribe snapshot (frozen)
clock7.ms = T0 + 50_000
run_verifier(c7)
check("frozen snapshot -> still stale", (INSTR, "book") in c7.stale_streams)
def book_sends(cache):
    channel = cache.book_channel(INSTR)
    return sum(1 for m in cache.websocket.sent if m["params"]["channels"] == [channel])


check("no second book resubscribe within 60 s", book_sends(c7) == 2, book_sends(c7))
check("frozen ticker also flagged independently", (INSTR, "ticker") in c7.stale_streams)
clock7.ms = T0 + 95_000
run_verifier(c7)
check("book resubscribed again after 60 s", book_sends(c7) == 4, book_sends(c7))
bnd = c7.capture_boundary_state(datetime.fromtimestamp((T0 + BAR) / 1000, tz=timezone.utc))
clock7.ms = T0 + BAR + 1000
feed_messages(c7, clock7, T0, T0 + BAR + 1000, 500)
r7 = c7.finalize_bar(datetime.fromtimestamp((T0 + BAR) / 1000, tz=timezone.utc), bnd).iloc[0]
check("bar reason stale_stream_open:book", "stale_stream_open:book" in str(r7["stream_incomplete_reason"]),
      r7["stream_incomplete_reason"])

# 7d. live stream again (>=2 changes since resubscribe) -> cleared without REST match
c7.process_book(book(T0 + BAR + 2000, 150.0, 151.0))
c7.process_book(book(T0 + BAR + 2500, 150.5, 151.5))
REST["bbo"] = (999.0, 1000.0)                                # REST disagrees: must not matter
clock7.ms = T0 + BAR + 3000
run_verifier(c7)
check("live stream clears stale state", (INSTR, "book") not in c7.stale_streams)
check("cleared interval marked", (INSTR, T0 + BAR) in c7.incomplete_bar_keys)

# 7e. REST match clears a stale state
REST["bbo"] = (150.0, 151.0)
clock8b, c8b = make_verifier_cache()
c8b.ticker_bbo_last_change[INSTR] = {"exchange_ms": T0 + 29_000, "received_ms": T0 + 29_000}
clock8b.ms = T0 + 30_000
run_verifier(c8b)
REST["bbo"] = (100.0, 101.0)                                 # market came back to cached top
clock8b.ms = T0 + 40_000
run_verifier(c8b)
check("REST match clears stale state", not c8b.stale_streams)

# 7f. stale state survives a reconnect reset
REST["bbo"] = (150.0, 151.0)
clock9b, c9b = make_verifier_cache()
c9b.ticker_bbo_last_change[INSTR] = {"exchange_ms": T0 + 29_000, "received_ms": T0 + 29_000}
clock9b.ms = T0 + 30_000
run_verifier(c9b)
c9b.reset_realtime_state()
check("stale state survives reconnect reset", (INSTR, "book") in c9b.stale_streams)

# --------------------------------------------------
# 8. Silence watchdog
# --------------------------------------------------
clock8 = Clock(T0)
c8 = new_cache(clock8)
c8.connection_started_ms = T0
c8.last_message_received_ms = T0
clock8.ms = T0 + 20_000
fsc.FEED_SILENCE_RECONNECT_SECONDS = 15.0


async def wd():
    try:
        await asyncio.wait_for(c8.feed_watchdog(), timeout=3)
    except fsc.FeedStaleError as exc:
        return exc
    return None
check("silence watchdog raises", isinstance(asyncio.run(wd()), fsc.FeedStaleError))

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
