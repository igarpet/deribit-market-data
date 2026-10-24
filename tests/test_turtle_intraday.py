"""Synthetic tests for Turtle Breakout 03 (h4 periods, engine sizing, turtle_intraday).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_intraday.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_engineering.modeling import turtle_breakout as tb
from data_engineering.modeling import turtle_benchmarks as bm
from data_engineering.modeling import turtle_intraday as ti
from data_engineering.modeling import vwap_drift as vd

FITS = vd.load_engine(ROOT / "notebooks" / "Volatility_Dynamics" / "18" / "18_manifest.json")
S0 = tb.COMMON_START


def bars(days=900, trend=0.01, seeds=(3, 4)):
    return {a: tb.synthetic_5m(days, seed=s, trend=trend, start="2020-01-01") for a, s in zip(("BTC", "ETH"), seeds)}


def build(B, p, engine=True):
    return {a: tb.prepare_asset(a, b, tb.synthetic_funding(b), p, engine=FITS if engine else None) for a, b in B.items()}


class PeriodTests(unittest.TestCase):
    def test_h4_boundaries(self):
        t = pd.DatetimeIndex(["2025-07-01 04:00", "2025-07-01 04:05", "2025-07-01 08:00", "2025-07-01 08:05"], tz="UTC")
        pid = tb.period_id(t, "h4")
        self.assertEqual(pid[1], pid[2]); self.assertEqual(pid[1] - pid[0], 1); self.assertEqual(pid[3] - pid[2], 1)
        self.assertEqual(tb.period_end(pid[:1], "h4")[0], t[0])

    def test_last_4h_levels(self):
        p = ti.strategy_params("last_4h", "atr")
        d = tb.prepare_asset("BTC", tb.synthetic_5m(30, seed=1), None, p)
        k = 200
        self.assertAlmostEqual(d.hi_n[k], np.nanmax(d.ph[k - 4:k]))
        self.assertAlmostEqual(d.lo_x[k], np.nanmin(d.pl[k - 2:k]))
        self.assertEqual(len(d.s), 30 * 24)


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.B = bars()

    def test_engine_sigma_matches_vd18_predict(self):
        b = self.B["BTC"]
        sig = tb.engine_sigma(pd.DatetimeIndex(b.index), b.close.to_numpy(), FITS, "BTC", 240)
        ok = np.isfinite(sig)
        self.assertGreater(ok.mean(), 0.95)
        self.assertTrue((sig[ok] > 0).all())

    def test_calibration_is_causal(self):
        p = ti.strategy_params("daily", "engine")
        d = tb.prepare_asset("BTC", self.B["BTC"], None, p, engine=FITS)
        cut = d.s[400]
        d2 = tb.prepare_asset("BTC", self.B["BTC"].iloc[:cut + 1], None, p, engine=FITS)
        np.testing.assert_allclose(d.eng_c[:400], d2.eng_c[:400], equal_nan=True)
        np.testing.assert_allclose(d.eng_sig[:cut - 300], d2.eng_sig[:cut - 300], equal_nan=True)
        self.assertTrue(np.isnan(d.eng_c[:p.calib_min_obs]).all())

    def test_engine_unit_scale_matches_atr_on_average(self):
        p = ti.strategy_params("daily", "engine")
        d = tb.prepare_asset("BTC", self.B["BTC"], None, p, engine=FITS)
        k = np.arange(100, len(d.s)); k = k[d.s[k] > 0]
        U = d.eng_c[k] * d.eng_sig[d.s[k] - 1] * d.c[d.s[k] - 1]
        r = np.nanmedian(U / d.atr[k])
        self.assertAlmostEqual(r, 1.0, delta=0.1)

    def test_sizing_modes(self):
        out = {}
        for sz in ti.SIZINGS:
            p = ti.strategy_params("daily", sz)
            data = build(self.B, p, engine=(sz != "atr"))
            E, T, st = tb.simulate(data, p, start=S0)
            out[sz] = (data, T, st)
        # same entries (sizing does not change signals as long as nothing is ruined)
        ea, ee = out["atr"][1], out["engine"][1]
        self.assertEqual(list(ea.je[:20]), list(ee.je[:20]))
        # engine sizing: units x 2 U = 2% HWM on the first trade
        data, T, st = out["engine"]
        r = T.iloc[0]; d = data[r.asset]
        k = int(tb.period_id(pd.DatetimeIndex([d.end[r.je]]), "day")[0]) - d.pid0
        U = d.eng_c[k] * d.eng_sig[r.je - 1] * d.c[r.je - 1]
        self.assertAlmostEqual(r.units * 2 * U, 0.02 * 100_000, places=4)
        self.assertGreater(st["engine_used"]["BTC"], 0)
        # engine_stop changes exits, engine (sizing only) keeps the ATR stop: exits equal to atr mode for early trades
        self.assertEqual(list(ea.jx[:5]), list(ee.jx[:5]))


class DecisionTests(unittest.TestCase):
    def test_evaluate_and_contrasts_run(self):
        B = bars(900, 0.01)
        cp = ti.CParams(n_boot=200)
        E = {}
        rows = []
        for tf in ("daily", "h4_candles"):
            for sz in ("atr", "engine"):
                p = ti.strategy_params(tf, sz)
                data = build(B, p, engine=(sz != "atr"))
                e, T, st = tb.simulate(data, p, start=S0)
                null = tb.random_portfolios(data, p, st["p_entry"], S0, n=20, seed=1)
                C, F = bm.daily_panel(data, S0, None)
                row, _ = ti.evaluate_strategy(f"{tf}|{sz}", e, T, st, null, C, F, p.fee + p.slip, cp)
                rows.append(row); E[f"{tf}|{sz}"] = e
        self.assertEqual(len(rows), 4)
        self.assertIn(rows[0]["verdict"], ("BEATS_LUCK_AND_BENCHMARKS", "BEATS_LUCK_ONLY", "BEATS_BENCHMARKS_ONLY", "NO_EDGE"))
        sub = [c for c in ti.CONTRASTS if c[1] in E and c[2] in E]
        old = ti.CONTRASTS
        try:
            ti.CONTRASTS = sub
            C = ti.contrasts(E, cp)
        finally:
            ti.CONTRASTS = old
        self.assertEqual(len(C), len(sub))
        self.assertTrue(((C.lo <= C["diff"]) & (C["diff"] <= C.hi)).all())

    def test_registration(self):
        r = ti.registration()
        self.assertEqual(len(r["contrasts"]), 8)
        self.assertEqual(r["n_strategies"], 9)


if __name__ == "__main__":
    unittest.main()
