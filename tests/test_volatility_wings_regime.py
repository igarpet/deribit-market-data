"""Synthetic tests for VD 10d (volatility_wings_regime).

Run from repo root: python -m unittest discover -s tests -p "test_volatility_wings_regime.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import volatility_wings_empirical as we
from data_engineering.modeling import volatility_wings_regime as wr


def world(mult_low, mult_high, seed=4):
    """Low tercile: fat tails (t3); high tercile: thin (normal). Quotes priced at mult x fair, +/-10% spread."""
    rng = np.random.default_rng(seed)
    t3 = lambda n: rng.standard_t(3, n) / np.sqrt(3.0)
    nrm = lambda n: rng.standard_normal(n)
    pools, quotes = [], []
    for terc, law, mult in (("low", t3, mult_low), ("high", nrm, mult_high)):
        pools.append(pd.DataFrame({"session": np.repeat([f"h{d:03d}" for d in range(200)], 100), "u": law(20_000),
                                   "tercile": terc, "tte_bucket": "(240.0, 480.0]", "underlying": "X"}))
        q = we._fake_quotes(rng, 14, 40, law, noise=0.03)
        q["mid_usd"] *= mult
        q["bid_usd"], q["ask_usd"] = q["mid_usd"] * 0.9, q["mid_usd"] * 1.1
        q["session"] = q["session"] + terc
        q["tercile"] = terc
        quotes.append(q)
    q = wr.net_prices(pd.concat(quotes, ignore_index=True))
    return q, pd.concat(pools, ignore_index=True)


class WingsRegimeTests(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(wr.selftest()["ok"])

    def test_fair_world_rejects_nothing(self):
        q, pool = world(1.0, 1.0)
        d = wr.decide(q, pool, n_boot=150)
        self.assertEqual(d["D_long_verdict"], "NOT_CHEAP")
        self.assertEqual(d["D_short_verdict"], "NOT_RICH")

    def test_mispriced_world_detected(self):
        q, pool = world(0.5, 2.0)          # net bid = 2.0 x 0.9 x 0.875 = 1.58 x fair
        d = wr.decide(q, pool, n_boot=150)
        self.assertEqual(d["D_long_verdict"], "CHEAP_AT_ASK")
        self.assertEqual(d["D_short_verdict"], "RICH_AT_BID")

    def test_cell_pricing_uses_own_cell(self):
        # A quote in the thin-tail cell priced from the fat-tail cell would look cheap; own-cell pricing must not.
        q, pool = world(1.0, 1.0)
        hi = q[q.tercile == "high"].reset_index(drop=True)
        P_own = wr.price_by_cell(hi, pool)
        wrong = pool.assign(tercile=pool.tercile.map({"low": "high", "high": "low"}))
        P_wrong = wr.price_by_cell(hi, wrong)
        self.assertLess(abs(np.median(np.log(hi.mid_usd / P_own))), abs(np.median(np.log(hi.mid_usd / P_wrong))))


if __name__ == "__main__":
    unittest.main()
