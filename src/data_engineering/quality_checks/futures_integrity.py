"""
Futures (perpetual) integrity gate and known exchange events.

Evidence: Notebook 02 (2026-10-05). Integrity failures in the perpetual feed
are almost all system-wide events: the same timestamps on all seven
perpetuals (21 Sep 17:10, 28 Sep 09:05-09:30, the 1 Oct Deribit migration).
A robust z-score alone is unusable on one-tick books (BTC median spread
~0.01 bps, zero MAD), so every level test also has an absolute floor.

Scope (framework v2.1, section 3.1):
  - The gate qualifies PERPETUAL-based features: flow, basis and L2.
  - Structural returns come from the Deribit index and are NOT gated here
    (the index kept moving through frozen perpetual books).
  - Nothing is deleted. The gate produces flags and one boolean,
    perp_integrity_ok, that downstream readiness flags combine with.

Thresholds are never estimated on the full sample. Relative tests compare
each state with the asset's own TRAILING history (previous
TRAILING_WINDOW_STATES states, excluding the current one), so the rule is
causal and can be applied identically in backtests and live.

Flags (TRUE = suspect):
  flag_no_two_sided        no valid two-sided ticker quote at the boundary
  flag_wide_spread         spread >= max(WIDE_SPREAD_MIN_BPS,
                                         WIDE_SPREAD_MIN_RATIO x trailing median)
                           or spread >= HARD_SPREAD_BPS
  flag_mid_off_index       |basis - trailing median basis| >=
                           max(BASIS_MIN_DEVIATION_BPS, BASIS_ROBUST_Z x
                               trailing IQR / 1.349)
                           or |basis| >= HARD_BASIS_BPS
  flag_frozen_with_trades  identical best bid and ask over FROZEN_MIN_BARS
                           consecutive contiguous states, each with trades
  flag_no_l2_updates       no order-book update in the bar while trades occurred
  flag_known_event         state inside a known exchange event
  flag_rest_stale          collector v0.3 REST cross-check confirmed a stale feed
  flag_trades_outside_bbo  collector v0.3: a trade printed more than
                           TRADE_OUTSIDE_BBO_MIN_BPS outside the live book BBO
                           while the book had no update for at least
                           STALE_BOOK_GAP_MS in the bar, or more than
                           HARD_TRADE_OUTSIDE_BBO_BPS outside it

The two collector-v0.3 flags are FALSE on earlier rows, where the evidence
does not exist (not "passed").
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


FUTURES_INTEGRITY_VERSION = "0.2"

# v0.2 (first real run, 6 Oct 2026): the trades-outside-BBO test fired on
# 7-21% of collector v0.3 bars (SOL 99, ETH/HYPE 66). A market order that
# sweeps several levels legitimately prints beyond the pre-trade top of
# book, and the trade and book channels are ~100 ms apart. A print outside
# the BBO is evidence of a STALE book only when the book was also silent.
# The flag now requires a long book-update gap in the same bar, or an
# extreme distance.

# Trailing history used by the relative tests: 7 days of 5-minute states.
TRAILING_WINDOW_STATES = 2016
# Minimum trailing observations before the relative tests are evaluated.
TRAILING_MIN_STATES = 288

WIDE_SPREAD_MIN_BPS = 5.0
WIDE_SPREAD_MIN_RATIO = 10.0
HARD_SPREAD_BPS = 50.0

BASIS_MIN_DEVIATION_BPS = 10.0
BASIS_ROBUST_Z = 8.0
HARD_BASIS_BPS = 50.0

FROZEN_MIN_BARS = 3
TRADE_OUTSIDE_BBO_MIN_BPS = 5.0
STALE_BOOK_GAP_MS = 30_000
HARD_TRADE_OUTSIDE_BBO_BPS = 50.0


# --------------------------------------------------
# Known exchange events
# --------------------------------------------------
# One row per event. underlying NULL = all underlyings. Times are UTC,
# both ends inclusive, on the five-minute state grid (state_time).
# Add new events here; never edit or delete a past entry silently.

KNOWN_EXCHANGE_EVENTS = (
    (
        "2026-10-01 09:00:00",
        "2026-10-01 10:05:00",
        None,
        "deribit_migration_coinbase_international",
        "Deribit planned platform migration for the Coinbase International "
        "consolidation (announced 09:00 UTC start; completion reported "
        "10:02 UTC). Perpetual BBO frozen on all assets; migrated "
        "positions changed open-interest levels.",
    ),
)


def create_known_exchange_events_table(connection) -> None:
    """
    Persist KNOWN_EXCHANGE_EVENTS as the table known_exchange_events.

    Used by the futures integrity gate and, later, by episode censoring
    in the regime model.
    """

    connection.execute(
        """
        CREATE OR REPLACE TABLE known_exchange_events (
            event_start   TIMESTAMPTZ NOT NULL,
            event_end     TIMESTAMPTZ NOT NULL,
            underlying    VARCHAR,
            event_id      VARCHAR NOT NULL,
            description   VARCHAR
        )
        """
    )

    for start, end, underlying, event_id, description in KNOWN_EXCHANGE_EVENTS:

        connection.execute(
            """
            INSERT INTO known_exchange_events
            VALUES (
                CAST(? || '+00' AS TIMESTAMPTZ),
                CAST(? || '+00' AS TIMESTAMPTZ),
                ?, ?, ?
            )
            """,
            [start, end, underlying, event_id, description],
        )


def _optional(columns: set[str], name: str, sql_type: str) -> str:

    return name if name in columns else f"CAST(NULL AS {sql_type})"


def create_futures_integrity_state(
    connection,
    source_relation: str = "canonical_market_state",
    target_table: str = "futures_integrity_state",
    temporary: bool = False,
) -> None:
    """
    One row per state_id with integrity metrics, flags and perp_integrity_ok.
    """

    columns = set(
        connection.execute(
            f"DESCRIBE SELECT * FROM {source_relation}"
        ).df()["column_name"]
    )

    rest_stale = _optional(columns, "rest_stale_confirmed", "BOOLEAN")
    outside_bps = _optional(columns, "max_trade_outside_book_bbo_bps", "DOUBLE")
    book_updates = _optional(columns, "book_update_count", "BIGINT")
    book_gap = _optional(columns, "book_max_update_gap_ms", "DOUBLE")
    trades = _optional(columns, "trade_count", "BIGINT")

    kind = "TEMP TABLE" if temporary else "TABLE"

    if not temporary:
        connection.execute(f"DROP VIEW IF EXISTS {target_table}")
        connection.execute(f"DROP TABLE IF EXISTS {target_table}")

    connection.execute(
        f"""
        CREATE OR REPLACE {kind} {target_table} AS

        WITH base AS (

            SELECT
                s.state_id,
                s.state_time,
                s.underlying,
                s.price_features_ready,

                s.best_bid_price,
                s.best_ask_price,
                s.index_price,

                {trades} AS trade_count,
                {book_updates} AS book_update_count,
                {rest_stale} AS rest_stale_confirmed,
                {outside_bps} AS max_trade_outside_book_bbo_bps,
                {book_gap} AS book_max_update_gap_ms,

                (
                    s.best_bid_price > 0
                    AND s.best_ask_price > 0
                    AND s.best_bid_price <= s.best_ask_price
                )
                    AS two_sided,

                CASE
                    WHEN
                        s.best_bid_price > 0
                        AND s.best_ask_price > 0
                        AND s.best_bid_price <= s.best_ask_price
                    THEN
                        1e4 * (s.best_ask_price - s.best_bid_price)
                        / ((s.best_ask_price + s.best_bid_price) / 2.0)
                END
                    AS spread_bps,

                CASE
                    WHEN
                        s.best_bid_price > 0
                        AND s.best_ask_price > 0
                        AND s.best_bid_price <= s.best_ask_price
                        AND s.index_price > 0
                    THEN
                        1e4 * LN(
                            ((s.best_ask_price + s.best_bid_price) / 2.0)
                            / s.index_price
                        )
                END
                    AS basis_bps,

                EXISTS (
                    SELECT 1
                    FROM known_exchange_events e
                    WHERE
                        s.state_time BETWEEN e.event_start AND e.event_end
                        AND (
                            e.underlying IS NULL
                            OR e.underlying = s.underlying
                        )
                )
                    AS flag_known_event

            FROM {source_relation} s
        ),

        history AS (

            SELECT
                *,

                COUNT(spread_bps) OVER w AS trailing_spread_n,
                MEDIAN(spread_bps) OVER w AS trailing_spread_median,

                COUNT(basis_bps) OVER w AS trailing_basis_n,
                MEDIAN(basis_bps) OVER w AS trailing_basis_median,
                QUANTILE_CONT(basis_bps, 0.75) OVER w
                - QUANTILE_CONT(basis_bps, 0.25) OVER w
                    AS trailing_basis_iqr,

                LAG(state_time, 1) OVER g AS lag_time_1,
                LAG(best_bid_price, 1) OVER g AS lag_bid_1,
                LAG(best_ask_price, 1) OVER g AS lag_ask_1

            FROM base

            WINDOW
                w AS (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN {TRAILING_WINDOW_STATES} PRECEDING
                        AND 1 PRECEDING
                ),
                g AS (
                    PARTITION BY underlying
                    ORDER BY state_time
                )
        ),

        steps AS (

            SELECT
                *,

                COALESCE(
                    two_sided
                    AND date_diff('minute', lag_time_1, state_time) = 5
                    AND best_bid_price = lag_bid_1
                    AND best_ask_price = lag_ask_1
                    AND trade_count > 0,
                    FALSE
                )
                    AS frozen_step

            FROM history
        ),

        runs AS (

            SELECT
                *,

                -- FROZEN_MIN_BARS consecutive states with an identical
                -- BBO means FROZEN_MIN_BARS - 1 consecutive frozen steps.
                MIN(CASE WHEN frozen_step THEN 1 ELSE 0 END) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN {FROZEN_MIN_BARS - 2} PRECEDING
                        AND CURRENT ROW
                ) = 1
                AND COUNT(*) OVER (
                    PARTITION BY underlying
                    ORDER BY state_time
                    ROWS BETWEEN {FROZEN_MIN_BARS - 2} PRECEDING
                        AND CURRENT ROW
                ) = {FROZEN_MIN_BARS - 1}
                    AS flag_frozen_with_trades

            FROM steps
        ),

        flagged AS (

            SELECT
                state_id,
                state_time,
                underlying,
                price_features_ready,

                spread_bps,
                basis_bps,
                trailing_spread_median,
                trailing_basis_median,
                trailing_basis_iqr,
                (trailing_spread_n >= {TRAILING_MIN_STATES})
                    AS trailing_history_ok,

                NOT COALESCE(two_sided, FALSE)
                    AS flag_no_two_sided,

                COALESCE(
                    spread_bps >= {HARD_SPREAD_BPS}
                    OR (
                        trailing_spread_n >= {TRAILING_MIN_STATES}
                        AND spread_bps >= {WIDE_SPREAD_MIN_BPS}
                        AND spread_bps
                            >= {WIDE_SPREAD_MIN_RATIO} * trailing_spread_median
                    ),
                    FALSE
                )
                    AS flag_wide_spread,

                COALESCE(
                    ABS(basis_bps) >= {HARD_BASIS_BPS}
                    OR (
                        trailing_basis_n >= {TRAILING_MIN_STATES}
                        AND ABS(basis_bps - trailing_basis_median)
                            >= GREATEST(
                                {BASIS_MIN_DEVIATION_BPS},
                                {BASIS_ROBUST_Z} * trailing_basis_iqr / 1.349
                            )
                    ),
                    FALSE
                )
                    AS flag_mid_off_index,

                COALESCE(flag_frozen_with_trades, FALSE)
                    AS flag_frozen_with_trades,

                COALESCE(
                    book_update_count = 0 AND trade_count > 0,
                    FALSE
                )
                    AS flag_no_l2_updates,

                flag_known_event,

                COALESCE(rest_stale_confirmed, FALSE)
                    AS flag_rest_stale,

                COALESCE(
                    max_trade_outside_book_bbo_bps
                        >= {HARD_TRADE_OUTSIDE_BBO_BPS}
                    OR (
                        max_trade_outside_book_bbo_bps
                            > {TRADE_OUTSIDE_BBO_MIN_BPS}
                        AND book_max_update_gap_ms
                            >= {STALE_BOOK_GAP_MS}
                    ),
                    FALSE
                )
                    AS flag_trades_outside_bbo,

                max_trade_outside_book_bbo_bps,
                book_max_update_gap_ms,

                (max_trade_outside_book_bbo_bps IS NOT NULL
                 OR rest_stale_confirmed IS NOT NULL)
                    AS v03_evidence_available

            FROM runs
        )

        SELECT
            *,

            (
                CAST(flag_no_two_sided AS INTEGER)
                + CAST(flag_wide_spread AS INTEGER)
                + CAST(flag_mid_off_index AS INTEGER)
                + CAST(flag_frozen_with_trades AS INTEGER)
                + CAST(flag_no_l2_updates AS INTEGER)
                + CAST(flag_known_event AS INTEGER)
                + CAST(flag_rest_stale AS INTEGER)
                + CAST(flag_trades_outside_bbo AS INTEGER)
            )
                AS integrity_flag_count,

            (
                COALESCE(price_features_ready, FALSE)
                AND NOT flag_no_two_sided
                AND NOT flag_wide_spread
                AND NOT flag_mid_off_index
                AND NOT flag_frozen_with_trades
                AND NOT flag_no_l2_updates
                AND NOT flag_known_event
                AND NOT flag_rest_stale
                AND NOT flag_trades_outside_bbo
            )
                AS perp_integrity_ok

        FROM flagged
        """
    )


def validate_futures_integrity(
    connection,
    table: str = "futures_integrity_state",
) -> None:

    row = connection.execute(
        f"""
        SELECT
            COUNT(*),
            COUNT(*) - COUNT(DISTINCT state_id),
            SUM(CASE WHEN perp_integrity_ok IS NULL THEN 1 ELSE 0 END),
            SUM(
                CASE
                    WHEN perp_integrity_ok AND integrity_flag_count > 0
                    THEN 1 ELSE 0
                END
            )
        FROM {table}
        """
    ).fetchone()

    rows = connection.execute(
        f"""
        SELECT
            underlying,
            COUNT(*) AS states,
            SUM(CASE WHEN COALESCE(price_features_ready, FALSE) THEN 1 ELSE 0 END) AS price_ready,
            SUM(CASE WHEN perp_integrity_ok THEN 1 ELSE 0 END) AS integrity_ok,
            SUM(CASE WHEN flag_no_two_sided AND price_features_ready THEN 1 ELSE 0 END) AS no_two_sided,
            SUM(CASE WHEN flag_wide_spread THEN 1 ELSE 0 END) AS wide_spread,
            SUM(CASE WHEN flag_mid_off_index THEN 1 ELSE 0 END) AS mid_off_index,
            SUM(CASE WHEN flag_frozen_with_trades THEN 1 ELSE 0 END) AS frozen,
            SUM(CASE WHEN flag_no_l2_updates THEN 1 ELSE 0 END) AS no_l2,
            SUM(CASE WHEN flag_known_event THEN 1 ELSE 0 END) AS known_event,
            SUM(CASE WHEN flag_rest_stale THEN 1 ELSE 0 END) AS rest_stale,
            SUM(CASE WHEN flag_trades_outside_bbo THEN 1 ELSE 0 END) AS outside_bbo
        FROM {table}
        GROUP BY underlying
        ORDER BY underlying
        """
    ).fetchall()

    print("\nFUTURES INTEGRITY GATE")
    print("=" * 125)
    print(f"States:                         {row[0]:,}")
    print(f"Duplicate state_id:             {row[1]:,}")
    print(f"NULL perp_integrity_ok:         {int(row[2] or 0):,}")
    print(f"OK states with flags raised:    {int(row[3] or 0):,}")
    print(
        f"\n{'Asset':<7}{'States':>8}{'Price ok':>10}{'Gate ok':>9}"
        f"{'No 2-side':>11}{'Wide':>7}{'Off idx':>9}{'Frozen':>8}"
        f"{'No L2':>7}{'Event':>7}{'REST':>6}{'Out BBO':>9}"
    )
    for r in rows:
        print(
            f"{r[0]:<7}{r[1]:>8,}{int(r[2]):>10,}{int(r[3]):>9,}"
            f"{int(r[4]):>11,}{int(r[5]):>7,}{int(r[6]):>9,}{int(r[7]):>8,}"
            f"{int(r[8]):>7,}{int(r[9]):>7,}{int(r[10]):>6,}{int(r[11]):>9,}"
        )

    if row[1]:
        raise RuntimeError(f"{table} has duplicate state_id values.")
    if row[2]:
        raise RuntimeError(f"{table} has NULL perp_integrity_ok values.")
    if row[3]:
        raise RuntimeError(f"{table} marks flagged states as integrity_ok.")
