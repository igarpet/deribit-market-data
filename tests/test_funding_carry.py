import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import funding_carry as fc  # noqa: E402


def _px(days=30, seed=0, vol=5e-4, start="2025-01-01 00:01"):
    t = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    return pd.Series(np.exp(np.log(100) + np.cumsum(rng.normal(0, vol, len(t)))), index=t)


def _f(days=30, rate=1e-4, start="2025-01-01 08:00"):
    return pd.Series(rate, index=pd.date_range(start, periods=days * 3, freq="8h", tz="UTC"))


class TestFundingCarry(unittest.TestCase):
    def test_selftest(self):
        self.assertIn("ok", fc.selftest())

    def test_funding_window_and_f7(self):
        px = _px(20)
        f = _f(20)
        d = fc.daily_frame(px, px, f)
        # boundary 00:05: interval (b_d, b_{d+1}] contains the 08:00, 16:00 and next-day 00:00 prints
        self.assertTrue(np.allclose(d["fund_rate"], 3e-4))
        self.assertTrue(np.allclose(d["f7"].dropna(), 1e-4 * 3 * 365))
        self.assertEqual(d["b"].iloc[0].strftime("%H:%M"), "00:05")
        # f7 at b_d uses only prints with t <= b_d (causal): change a print after b_5 -> f7[:6] unchanged
        f2 = f.copy()
        f2[f2.index > d["b"].iloc[5]] = 9e-3
        d2 = fc.daily_frame(px, px, f2)
        np.testing.assert_allclose(d["f7"].iloc[:6], d2["f7"].iloc[:6])

    def test_basis_pnl_short_perp(self):
        px = _px(20, vol=1e-5)
        perp = px * 1.002                              # perp 20 bp rich at entry
        late = perp.index >= pd.Timestamp("2025-01-15", tz="UTC")
        perp[late] = px[late]                           # converges later
        d = fc.daily_frame(px, perp, _f(20, rate=0.0))
        s = fc.run_asset(d, "U")
        self.assertAlmostEqual(s["basis"].sum(), 0.002 * d["F"].iloc[0] / 1.002 / d["S"].iloc[0], places=4)
        self.assertGreater(s["basis"].sum(), 0)

    def test_negative_funding_is_paid(self):
        px = _px(20)
        d = fc.daily_frame(px, px, _f(20, rate=-2e-4))
        s = fc.run_asset(d, "U")
        self.assertLess(s["funding"].sum(), 0)

    def test_conditional_state_machine(self):
        f7 = np.array([0.0, 0.12, 0.05, 0.02, 0.2, np.nan, 0.2])
        valid = np.array([True, True, True, True, True, False, True])
        pos = fc.positions_conditional(f7, valid, 0.10, 0.03)
        np.testing.assert_array_equal(pos, [False, True, True, False, True, False, True])

    def test_costs_and_reset(self):
        t = pd.date_range("2025-01-01 00:01", periods=20 * 1440, freq="1min", tz="UTC")
        px = pd.Series(100.0, index=t)
        px[t >= pd.Timestamp("2025-01-14", tz="UTC")] = 150.0   # +50% jump -> one reset
        d = fc.daily_frame(px, px, _f(20, rate=0.0))
        s = fc.run_asset(d, "U")
        self.assertEqual(int(s["reset"].sum()), 1)
        exp = fc.SIDE_COST + 2 * fc.SIDE_COST * 1.5 + fc.SIDE_COST * 1.5
        self.assertAlmostEqual(s["cost"].sum(), exp, places=12)
        self.assertAlmostEqual(s["basis"].sum(), 0.0, places=12)

    def test_portfolio_and_verdicts(self):
        S, P, F = fc.synthetic(500, seed=3)
        fr = {a: fc.daily_frame(S[a], P[a], F[a]) for a in S}
        _, _, s = fc.run_all(fr, F, label="U")
        self.assertAlmostEqual(s["ann_reserved"], s["ann_deployed"], places=12)   # always fully deployed
        self.assertEqual(fc.verdict(s)["verdict"], "CARRY_EDGE")
        S0, P0, F0 = fc.synthetic(500, seed=4, base_rate=0.0, regime_amp=1e-4)
        fr0 = {a: fc.daily_frame(S0[a], P0[a], F0[a]) for a in S0}
        _, _, s0 = fc.run_all(fr0, F0, label="U")
        self.assertNotEqual(fc.verdict(s0)["verdict"], "CARRY_EDGE")


if __name__ == "__main__":
    unittest.main()
