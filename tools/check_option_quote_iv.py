"""
Diagnostic: are near-expiry options really unquoted, or does Deribit just
stop publishing bid/ask IV for them?

Notebook 03 found that in the last hours of each session almost every
surface node is priced by Deribit's mark IV, because the contracts lack a
valid bid/ask IV midpoint. Two very different situations produce that:

  A. no two-sided price exists (no bid or no ask)  -> genuinely unquoted;
  B. a two-sided price exists, but bid_iv / ask_iv is zero or missing
     -> Deribit's IV field is missing, and IV could be recovered by
        inverting the quoted prices ourselves.

This script measures the split, per asset and time to expiry, for
near-the-money OTM contracts (|ln K/F| <= 3%) of the primary linear USDC
families. Read-only.

PC:
    python tools\\check_option_quote_iv.py
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.environ["DERIBIT_DATA_ROOT"]) / "database" / "research.duckdb"

SQL = """
WITH base AS (
    SELECT
        underlying,
        CASE
            WHEN minutes_to_expiry <= 30 THEN '1 <=30m'
            WHEN minutes_to_expiry <= 60 THEN '2 30m-1h'
            WHEN minutes_to_expiry <= 180 THEN '3 1-3h'
            WHEN minutes_to_expiry <= 360 THEN '4 3-6h'
            WHEN minutes_to_expiry <= 720 THEN '5 6-12h'
            ELSE '6 12-24h'
        END AS tte_bucket,
        (best_bid_price > 0 AND best_ask_price > 0 AND best_bid_price <= best_ask_price)
            AS two_sided_price,
        (bid_iv > 0 AND ask_iv > 0 AND ask_iv >= bid_iv)
            AS two_sided_iv,
        (best_bid_price > 0) AS has_bid,
        (best_ask_price > 0) AS has_ask
    FROM normalized_option_chain
    WHERE option_family_id LIKE '%|linear|USDC|USDC'
      AND is_otm_or_atm
      AND ABS(log_moneyness) <= 0.03
      AND option_snapshot_ready
)
SELECT
    underlying,
    tte_bucket,
    COUNT(*) AS contracts,
    ROUND(100 * AVG(CASE WHEN has_bid THEN 1 ELSE 0 END), 1) AS has_bid_pct,
    ROUND(100 * AVG(CASE WHEN has_ask THEN 1 ELSE 0 END), 1) AS has_ask_pct,
    ROUND(100 * AVG(CASE WHEN two_sided_price THEN 1 ELSE 0 END), 1) AS two_sided_price_pct,
    ROUND(100 * AVG(CASE WHEN two_sided_iv THEN 1 ELSE 0 END), 1) AS two_sided_iv_pct,
    ROUND(100 * AVG(CASE WHEN two_sided_price AND NOT COALESCE(two_sided_iv, FALSE)
                         THEN 1 ELSE 0 END), 1) AS price_but_no_iv_pct
FROM base
GROUP BY 1, 2
ORDER BY 1, 2
"""


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(SQL).df()
    finally:
        con.close()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("Near-the-money OTM contracts (|k| <= 3%), primary linear USDC families")
    print("price_but_no_iv_pct = two-sided price exists but Deribit bid/ask IV is missing (case B)\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
