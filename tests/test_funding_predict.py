import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import funding_predict as fp  # noqa: E402
from data_engineering.modeling import markov_har_variance as v18  # noqa: E402


def _const(days=80, rate=1e-4, start="2025-01-01"):
    t = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(minutes=1), periods=days * 1440, freq="1min")
    rng = np.random.default_rng(0)
    px = pd.Series(np.exp(np.log(100) + np.cumsum(rng.normal(0, 4e-4, len(t)))), index=t)
    f = pd.Series(rate, index=pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=8),
                                            periods=days * 3, freq="8h"))
    return px, f


class TestFundingPredict(unittest.TestCase):
    def test_funding_windows(self):
        px, f = _const()
        d = fp.asset_daily(px, px * 1.001, f)
        ann = 3e-4 * 365
        for c in ("f1", "f7", "f30"):
            np.testing.assert_allclose(d[c].dropna(), ann)
        for c in ("y1", "y7", "y30"):
            np.testing.assert_allclose(d[c].dropna(), ann)
        np.testing.assert_allclose(d["prem_now"].dropna(), 1e4 * np.log(1.001), rtol=1e-9)
        self.assertTrue(d["y30"].isna().iloc[-25:].all())         # forward window past data end

    def test_causality(self):
        px, f = _const()
        d1 = fp.asset_daily(px, px, f)
        cut = d1["b"].iloc[20]
        px2, f2 = px.copy(), f.copy()
        px2[px2.index > cut] *= 1.3
        f2[f2.index > cut] = 5e-3
        d2 = fp.asset_daily(px2, px2, f2)
        feats = ["f1", "f7", "f30", "prem_now", "prem_24h", "mom7", "mom30", "lrv1", "lrv7", "lrv30"]
        np.testing.assert_allclose(d1.loc[:20, feats].to_numpy(float), d2.loc[:20, feats].to_numpy(float),
                                   equal_nan=True)
        self.assertFalse(np.allclose(d1.loc[20, "y7"], d2.loc[20, "y7"]))   # target looks forward

    def test_metrics_features(self):
        px, f = _const(40)
        tm = pd.date_range(px.index[0].ceil("5min"), px.index[-1], freq="5min")
        oi = pd.Series(np.exp(np.arange(len(tm)) * 1e-4), index=tm)
        m = pd.DataFrame({"sum_open_interest_value": oi, "count_long_short_ratio": 2.0,
                          "sum_toptrader_long_short_ratio": 1.0, "sum_taker_long_short_vol_ratio": 1.0}, index=tm)
        d = fp.asset_daily(px, px, f, m).dropna(subset=["oi_chg7"])
        np.testing.assert_allclose(d["oi_chg7"], 1e2 * 2016 * 1e-4, rtol=1e-6)
        np.testing.assert_allclose(d["ls_acct"].dropna(), np.log(2.0))

    def test_engine_and_models(self):
        S, P, F, M = fp.synthetic(560, seed=1)
        parts = []
        for a in S:
            d = fp.asset_daily(S[a], P[a], F[a], M[a]); d["asset"] = a; parts.append(d)
        panel = fp.add_breadth(pd.concat(parts, ignore_index=True))
        raw = pd.concat([fp.spot_returns_5m(S[a], a) for a in S], ignore_index=True)
        eng, fits = fp.engine_forecasts(raw, panel.rename(columns={"asset": "underlying"}), v18)
        panel["leng1"], panel["leng7"] = eng["leng1"].to_numpy(), eng["leng7"].to_numpy()
        self.assertGreater(panel["leng1"].notna().mean(), 0.9)
        vc = fp.vol_check(panel)
        mz = vc.dropna(subset=["mz_slope"]).set_index(["horizon_days", "model"])["mz_slope"]
        self.assertTrue(0.7 < mz.loc[(1, "ENGINE")] < 1.3)        # engine calibrated in scale
        res = fp.run_models(panel, "y7", list(fp.ORDER))
        t = res["table"].set_index("model")
        self.assertGreater(t.loc["P", "r2_oos_vs_FE"], 0.5)        # planted persistence recovered
        self.assertTrue(bool(t.loc["P+PR", "adds"]))               # planted premium information
        self.assertLess(res["train"]["b"].max() + pd.Timedelta(days=7), fp.SPLIT + pd.Timedelta(seconds=1))
        ef = fp.engine_vs_naive_for_funding(panel, "y7", ["P", "PR", "M", "OI", "B"])
        dec = fp.decide(res["table"], ef, vc)
        self.assertTrue(dec["Q1_predictable_y7"])


if __name__ == "__main__":
    unittest.main()
