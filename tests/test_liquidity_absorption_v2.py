import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_engineering.modeling import liquidity_absorption_v2 as v2  # noqa: E402
from src.data_engineering.modeling import liquidity_absorption_extended as ex  # noqa: E402


def ms(s):
    return int(pd.Timestamp(s, tz="UTC").value // 10**6)


class TestV2(unittest.TestCase):
    def test_session_clock_dst(self):
        self.assertEqual(v2.session_start_ms(ms("2026-07-15 23:30")), ms("2026-07-15 22:00"))   # EDT: 18:00 = 22:00Z
        self.assertEqual(v2.session_start_ms(ms("2026-07-15 21:59")), ms("2026-07-14 22:00"))
        self.assertEqual(v2.session_start_ms(ms("2026-01-15 23:30")), ms("2026-01-15 23:00"))   # EST: 23:00Z
        self.assertEqual(v2.session_start_ms(ms("2026-11-02 12:00")), ms("2026-11-01 23:00"))   # day after DST end

    def test_cvd_divergence(self):
        idx = [ms("2026-07-01 00:01") + i * 60_000 for i in range(10)]
        bars = pd.DataFrame({"quote_volume": 100.0, "taker_buy_quote_volume": 30.0}, index=idx)   # net selling
        ev = {"first_touch_close": "2026-07-01 00:02+00:00", "second_touch_close": "2026-07-01 00:06+00:00",
              "side": "long", "zone_halfwidth": 1.0, "bar_low": 99.5, "first_touch_low": 100.0, "bar_high": 0, "first_touch_high": 0}
        ok, flow = v2.cvd_divergence(bars, ev)
        self.assertTrue(ok)
        self.assertAlmostEqual(flow, 4 * (60 - 100))                 # 4 minutes after the first touch, incl. the 2nd
        ev["bar_low"] = 98.0                                         # price broke lower: not absorption
        self.assertFalse(v2.cvd_divergence(bars, ev)[0])
        ev2 = dict(ev, side="short", bar_high=100.2, first_touch_high=100.0)
        self.assertFalse(v2.cvd_divergence(bars, ev2)[0])            # short needs net buying

    def test_lvn_hvn_target(self):
        w = math.log1p(5e-4)
        k0 = int(math.floor(math.log(100.0) / w))
        bins = np.arange(k0 - 20, k0 + 40)
        vol = np.full(len(bins), 100.0)
        vol[(bins >= k0 + 4) & (bins <= k0 + 9)] = 5.0                # LVN above
        tgt, src = v2.lvn_hvn_target(bins, vol, 100.0, "long", v2.Spec(smooth_bins=1))
        self.assertEqual(src, "PROFILE_LVN_TO_HVN")
        self.assertAlmostEqual(tgt, math.exp((k0 + 10) * w))           # near (lower) edge of the next HVN bin
        tgt2, src2 = v2.lvn_hvn_target(bins, vol, 100.0, "short", v2.Spec(smooth_bins=1))
        self.assertEqual(src2, "NO_PROFILE_TARGET")                   # no LVN below inside the profile

    def test_kelly(self):
        r = np.r_[np.full(60, 1.0), np.full(40, -1.0)]               # p = 0.6, b = 1 -> f* = 0.2
        f, g = v2.kelly_growth(r)
        self.assertAlmostEqual(f, 0.2, places=2)
        self.assertEqual(v2.kelly_growth(-np.abs(r))[0], 0.0)

    def test_v2_chain_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            hist, root, shadow = d / "history", d / "root", d / "ext"
            days = [str(x.date()) for x in pd.date_range("2026-06-01", periods=12, freq="D")]
            ex.synthetic_history(hist, days, seed=4)
            ex.prepare_venue_root(hist, root, ex.Spec(start="2026-06-01"), log=None)
            ex.stage_a02(root, shadow, ex.Spec(start="2026-06-04", end="2026-06-12"))
            ex.stage_a02b(root, shadow)
            res = v2.run_v2(shadow, root, ROOT, d / "v2")
            sig = pd.read_csv(d / "v2" / "V2_signals.csv")
            self.assertTrue(sig.v2_target_source.isin(["PROFILE_LVN_TO_HVN", "NO_PROFILE_TARGET", "NO_PROFILE",
                                                        "NO_PROFILE_SESSION_TOO_YOUNG"]).all())
            self.assertEqual(len(res), 6 * len(sig))
            ent = res[res.entry_time_ms.notna()]
            if len(ent):
                d_ = np.where(ent.side == "long", 1, -1)
                self.assertTrue((d_ * (ent.target - ent.entry_print) > 0).all())
                self.assertTrue((d_ * (ent.entry_print - ent.initial_stop) > 0).all())
            T, dec = v2.evaluate(d / "v2")
            self.assertIn(dec["verdict"], ("PASS", "FAIL"))


if __name__ == "__main__":
    unittest.main()
