import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import liquidity_absorption_stage_a06 as a6  # noqa: E402


def _cohort(days, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = days[1 + i % (len(days) - 2)]
        t = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=int(rng.integers(5, 19)), minutes=int(rng.integers(60)),
                                                       seconds=int(rng.integers(60)))
        grp = ("TARGET", "STOP")[i % 2]
        rows.append({"a02_candidate_row": i, "symbol": "BTCUSDT", "side": ("long", "short")[(i // 2) % 2],
                     "reason": grp, "group": grp, "entry_ms": int(t.value // 10**6),
                     "sign": 1.0 if (i // 2) % 2 == 0 else -1.0, "currency": "BTC",
                     "target_to_sigma60": 0.8, "geometry": "near_target", "kind": "SIGNAL", "anchor_id": str(i)})
    return pd.DataFrame(rows)


class TestA06(unittest.TestCase):
    def test_liquidation_sides(self):
        d = pd.Series(["sell", "buy", "buy", "sell", "buy"])
        l = pd.Series(["T", "T", "M", "M", None], dtype="string")
        ll, sl = a6.liquidation_sides(d, l)
        np.testing.assert_array_equal(ll, [True, False, True, False, False])
        np.testing.assert_array_equal(sl, [False, True, False, True, False])

    def test_parse(self):
        p = a6._parse(pd.Series(["BTC-27DEC24-100000-C", "BTC-PERPETUAL", "ETH-28MAR25", "BTC-3JAN25-95000-P"]))
        self.assertEqual(list(p.ikind), ["option", "perpetual", "future", "option"])
        self.assertEqual(p.strike.iloc[3], 95000.0)

    def test_bh(self):
        q = a6.bh(np.array([0.01, 0.04, 0.03, np.nan, 0.5]))
        np.testing.assert_allclose(q[[0, 1, 2, 4]], [0.04, 0.0533333, 0.0533333, 0.5], rtol=1e-4)

    def test_end_to_end_recovers_planted_liquidations(self):
        days = [str(x.date()) for x in pd.date_range("2026-07-01", periods=12, freq="D")]
        coh = _cohort(days)
        tgt = coh[coh.group.eq("TARGET")]
        planted = [(int(r.entry_ms), int(r.sign), 50_000.0) for r in tgt.itertuples()]
        with tempfile.TemporaryDirectory() as d:
            a6.synthetic_history(Path(d), days, planted=planted)
            m, cov = a6.build_minutes(Path(d), "BTC", days)
        self.assertTrue(cov.option_file.all())
        self.assertTrue(np.isfinite(m.atm_iv_lvl).any() and np.isfinite(m.fut_basis_lvl).any())
        panel = a6.event_panel(coh, {"BTC": m}, a6.Config(window_min=60))
        self.assertEqual(panel.anchor_id.nunique(), len(coh))
        z = panel[panel.rel_min.eq(0)]
        self.assertTrue(np.allclose(panel[panel.rel_min.eq(-1)].px_bp, 0.0))   # reference = last close before entry
        old = a6.PHASES
        a6.PHASES = {"context": (-60, -31), "approach": (-30, -16), "formation": (-15, -1), "entry": (0, 5),
                     "early": (6, 30), "late": (31, 60)}
        try:
            ph = a6.phase_panel(panel)
            t = a6.compare(ph, "TARGET", "STOP")
        finally:
            a6.PHASES = old
        hit = t[(t.phase == "formation") & (t.feature == "liq_with")].iloc[0]
        self.assertLess(hit.q_pre, 0.01)
        self.assertGreater(hit.medA, hit.medB)
        other = t[(t.phase == "context") & (t.feature == "liq_with")].iloc[0]
        self.assertTrue(np.isnan(other.p) or other.p > 0.01)

    def test_controls_avoid_signals_and_match_hour(self):
        days = [str(x.date()) for x in pd.date_range("2026-07-01", periods=20, freq="D")]
        coh = _cohort(days, n=20)
        ctl = a6.make_controls(coh, a6.Config(first_day=days[0], last_day=days[-1]))
        self.assertEqual(len(ctl), 3 * len(coh))
        for r in ctl.itertuples():
            src = coh[coh.a02_candidate_row == r.a02_candidate_row].iloc[0]
            self.assertEqual(pd.Timestamp(r.entry_ms, unit="ms").hour, pd.Timestamp(int(src.entry_ms), unit="ms").hour)
            self.assertGreaterEqual(np.min(np.abs(coh.entry_ms.to_numpy() - r.entry_ms)), 240 * 60_000)


if __name__ == "__main__":
    unittest.main()
