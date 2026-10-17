import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import multiscale_hmm as mh  # noqa: E402


def _minutes(n=600, seed=0, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    lc = np.log(100) + np.cumsum(rng.normal(0, 1e-3, n))
    prev = np.r_[lc[0], lc[:-1]]
    hi = np.maximum(lc, prev) + 2e-4
    lo = np.minimum(lc, prev) - 2e-4
    qv = rng.uniform(1, 2, n)
    return pd.DataFrame({"open_time": t, "open": np.exp(prev), "high": np.exp(hi), "low": np.exp(lo),
                         "close": np.exp(lc), "quote_volume": qv, "taker_buy_quote_volume": qv * 0.6})


class TestMultiScaleHMM(unittest.TestCase):
    def test_forward_backward(self):
        self.assertIn("ok", mh.selftest())

    def test_bars_aggregate_minutes(self):
        m = _minutes(600)
        b = mh.build_bars(m, 60)
        self.assertEqual(len(b), 10)
        lc = np.log(m["close"].to_numpy())
        r = np.diff(lc)
        # bar 2 (minutes 60..119): its return and RV come from minute returns 60..119 (first measured from minute 59)
        self.assertAlmostEqual(np.log(b["C"].iat[1] / b["C"].iat[0]), r[59:119].sum(), places=12)
        self.assertAlmostEqual(b["RV"].iat[1], np.sum(r[59:119] ** 2), places=14)
        self.assertEqual(b["bar_end"].iat[0], pd.Timestamp("2025-01-01 01:00", tz="UTC"))
        # coverage: drop 4 minutes of hour 3 -> 56 < 57 -> bar removed
        m2 = m.drop(index=range(130, 134))
        self.assertEqual(len(mh.build_bars(m2, 60)), 9)

    def test_sign_flip(self):
        a, b = _minutes(300, 1), _minutes(300, 2)
        f = mh.sign_flip({"A": a, "B": b}, seed=3)
        for orig, new in ((a, f["A"]), (b, f["B"])):
            ro, rn = np.diff(np.log(orig["close"])), np.diff(np.log(new["close"]))
            np.testing.assert_allclose(np.abs(ro), np.abs(rn), atol=1e-12)
            self.assertTrue((new["high"] >= new["low"]).all())
            self.assertTrue((new["high"] >= np.maximum(new["open"], new["close"]) - 1e-9).all())
            self.assertTrue((new["low"] <= np.minimum(new["open"], new["close"]) + 1e-9).all())
        sa = np.sign(np.diff(np.log(f["A"]["close"])) / np.diff(np.log(a["close"])))
        sb = np.sign(np.diff(np.log(f["B"]["close"])) / np.diff(np.log(b["close"])))
        np.testing.assert_array_equal(sa, sb)                                  # shared across assets
        flipped = np.r_[False, sa < 0]
        tb = f["A"]["taker_buy_quote_volume"].to_numpy()
        np.testing.assert_allclose(tb[flipped], (a["quote_volume"] * 0.4).to_numpy()[flipped])

    def test_features_are_causal(self):
        M = mh.synthetic_minutes(40, seed=5, start="2025-01-01", assets=("A",))
        cut = pd.Timestamp("2025-01-30", tz="UTC")
        M2 = {"A": M["A"].copy()}
        late = M2["A"]["open_time"] >= cut
        M2["A"].loc[late, ["open", "high", "low", "close"]] *= 1.5
        f1 = mh.bar_features(mh.build_bars(M["A"], 60), "1h")
        f2 = mh.bar_features(mh.build_bars(M2["A"], 60), "1h")
        early = (f1["bar_end"] <= cut).to_numpy()
        cols = list(mh.FEATURES) + ["sig"]
        np.testing.assert_allclose(f1.loc[early, cols].to_numpy(float), f2.loc[early, cols].to_numpy(float),
                                   equal_nan=True)

    def test_thmm_recovers_two_states(self):
        rng = np.random.default_rng(0)
        T = 4000
        s = np.zeros(T, int)
        for t in range(1, T):
            s[t] = s[t - 1] if rng.random() < 0.95 else 1 - s[t - 1]
        mus = np.array([[-2.0, 0.0], [2.0, 1.0]])
        X = mus[s] + rng.standard_t(6, size=(T, 2)) * 0.7
        m = mh.fit_thmm(X, np.array([0]), np.array([T]), 2, 7.0, 3, 1)
        order = np.argsort(m.mu[:, 0])
        np.testing.assert_allclose(m.mu[order], mus, atol=0.15)
        self.assertGreater(np.min(np.diag(m.A)), 0.9)

    def test_pnl_costs_and_gaps(self):
        t = pd.date_range("2026-03-01 01:00", periods=5, freq="1h", tz="UTC")
        d = pd.DataFrame({"bar_end": t, "r_next": [0.01, -0.01, 0.02, 0.0, 0.01],
                          "fund_next": [0.0, 1e-4, 0.0, 0.0, 0.0], "seg": [1, 1, 1, 2, 2]})
        m = np.array([0.01, 0.01, -0.01, 0.0, 0.01])
        p = mh.pnl_frame(d, m, np.ones(5, bool))
        np.testing.assert_array_equal(p["pos"].to_numpy(), [1, 1, -1, 0, 1])
        # trades: open 1, hold, flip 2 + close at segment end 1, flat, open 1 + close at data end 1
        np.testing.assert_allclose(p["cost"].to_numpy() / mh.COST_SIDE, [1, 0, 3, 0, 2])
        self.assertAlmostEqual(p["funding"].iat[1], 1e-4)
        self.assertAlmostEqual(p["gross"].iat[2], -np.expm1(0.02))

    def test_funding_window(self):
        t = pd.date_range("2026-03-01 04:00", periods=3, freq="4h", tz="UTC")
        d = pd.DataFrame({"bar_end": t})
        f = pd.Series([1e-4, 2e-4, 3e-4], index=pd.to_datetime(["2026-03-01 08:00", "2026-03-01 16:00",
                                                                 "2026-03-02 00:00"], utc=True))
        out = mh.add_funding(d, f, "4h")
        np.testing.assert_allclose(out["fund_next"].to_numpy(), [1e-4, 0.0, 2e-4])

    def test_pipeline_detects_planted_effect_and_null_does_not(self):
        M = mh.synthetic_minutes(760, seed=1, planted=0.4)
        P = mh.build_panel(M, "1h", mh.synthetic_funding(M))
        r = mh.run_pipeline(P, "1h", 2, 7.0, 2, 1)
        self.assertGreater(r["S_stat_bp"], 5)
        self.assertEqual(r["S_assets_positive"], 4)
        self.assertGreater(r["incr_lo_bp_day"], 0)
        st = mh.state_table(r, P, "1h")
        self.assertEqual(sorted(st["label"].str.split("+").str[0]), ["momentum", "no-autocorr"])
        n = mh.run_null(M, "1h", 2, 7.0, 2, 1, 1)
        self.assertTrue((n["S_stat_bp"] < r["S_stat_bp"] / 3).all())
        M0 = mh.synthetic_minutes(760, seed=2, planted=0.0)
        r0 = mh.run_pipeline(mh.build_panel(M0, "1h", mh.synthetic_funding(M0)), "1h", 2, 7.0, 2, 1)
        self.assertLess(abs(r0["S_stat_bp"]), 2)
        self.assertFalse(r0["net_lo_bp_day"] > 0)


if __name__ == "__main__":
    unittest.main()
