"""Synthetic tests for Regime Structure 06 (gamma_scalp).

Run from repo root: python -m unittest discover -s tests -p "test_gamma_scalp.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import gamma_scalp as gs

P = gs.Params(n_boot=300)


def run(n, vol, iv, rev=0.0, seed=0):
    x, ts = gs.synthetic_sessions(n, vol, iv, reversion=rev, seed=seed)
    return gs.run(x, gs.price_paths(ts, P), P)


class TestGammaScalp(unittest.TestCase):
    def test_fair_zero_cost_hedge_is_unbiased_and_reduces_variance(self):
        y = run(150, 0.6, 0.6, seed=1)
        h = y["R_h5_zero_cost"]
        self.assertLess(abs(h.mean()), 0.03)
        self.assertLess(h.std(), 0.4 * y["R_net"].std())

    def test_realised_above_implied_pays_when_hedged(self):
        y = run(80, 0.9, 0.6, seed=2)
        self.assertGreater(y["R_h5_zero_cost"].mean(), 0.3)
        self.assertGreater(y["rv_over_iv_h5"].mean(), 1.8)

    def test_costs_reduce_return_by_turnover_cost(self):
        y = run(30, 0.6, 0.6, seed=3)
        gap = (y["R_h5_zero_cost"] - y["R_h5_taker"]).mean()
        self.assertGreater(gap, 0)
        self.assertAlmostEqual(gap, y["cost_h5_share"].mean(), delta=0.02)   # spread 0 in synthetic -> fee only

    def test_5m_reversion_pays_fast_hedger_not_slow(self):
        # negative AR(1) at 5m: 5m realised variance above implied, 1h-sampled variance near implied
        y = run(120, 0.75, 0.6, rev=0.35, seed=4)
        self.assertGreater(y["R_h5_zero_cost"].mean(), y["R_h60_zero_cost"].mean() + 0.1)

    def test_vr_by_sampling(self):
        rng = np.random.default_rng(0)
        e = rng.standard_normal((400, 288))
        r = e.copy(); r[:, 1:] = e[:, 1:] - 0.4 * e[:, :-1]
        v = gs.vr_by_sampling(r, np.array(["dev"] * 200 + ["test"] * 200)).set_index(["window", "sampling_min"])["vr"]
        self.assertLess(v[("test", 5)], 0.6)
        self.assertGreater(v[("test", 60)], v[("test", 5)])


if __name__ == "__main__":
    unittest.main()
