"""Tests for Turtle Breakout 13 (turtle_oos + tools/download_tb13_data.py parsing).

Run from repo root: python -m unittest discover -s tests -p "test_turtle_oos.py" -v
"""
import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from data_engineering.modeling import turtle_oos as to

OP = to.OParams(n_boot=500)


class UniverseTests(unittest.TestCase):
    def test_new_candidate(self):
        self.assertTrue(to.new_candidate("RUNEUSDT"))
        self.assertTrue(to.new_candidate("1000PEPEUSDT"))
        for s in ("BTCUSDT", "1000BTCUSDT", "ETHUSDT", "USDCUSDT", "BTCDOMUSDT", "DEFIUSDT", "BTCUSDT_250627", "ETHBTC", "TONUSDT"):
            self.assertFalse(to.new_candidate(s), s)

    def test_spot_candidate(self):
        for s in ("BTCUSDT", "NEOUSDT", "VENUSDT", "BCCUSDT"):
            self.assertTrue(to.spot_candidate(s), s)
        for s in ("TUSDUSDT", "BTCUPUSDT", "ETHDOWNUSDT", "BNBBULLUSDT", "EOSBEARUSDT", "PAXUSDT", "BTCBUSD"):
            self.assertFalse(to.spot_candidate(s), s)

    def test_core_base(self):
        self.assertEqual(to.core_base("1000PEPE"), "PEPE"); self.assertEqual(to.core_base("1000000MOG"), "MOG")
        self.assertEqual(to.core_base("1MBABYDOGE"), "BABYDOGE"); self.assertEqual(to.core_base("RUNE"), "RUNE")

    def test_cut_at_gap(self):
        days = list(pd.date_range("2021-01-01", periods=50, freq="D", tz="UTC")) + list(pd.date_range("2022-01-01", periods=30, freq="D", tz="UTC"))
        d = pd.DataFrame({"day": days, "close": 1.0, "quote_volume": 1.0})
        c = to.cut_at_gap(d, 7)
        self.assertEqual(len(c), 50); self.assertEqual(c.day.iloc[-1], pd.Timestamp("2021-02-19", tz="UTC"))
        d2 = d.iloc[:50]; self.assertEqual(len(to.cut_at_gap(d2, 7)), 50)

    def test_screen_and_draw(self):
        S = pd.DataFrame({"symbol": [f"S{i}USDT" for i in range(100)], "first": "2022-01-01", "last": "2026-09-30",
                          "days": [500] * 90 + [100] * 10, "med_qvol": [20e6] * 80 + [1e6] * 20})
        e = to.screen(S, OP, "new")
        self.assertEqual(int(e.eligible.sum()), 80)
        a = to.draw(e.loc[e.eligible, "symbol"].tolist(), OP); b = to.draw(list(reversed(e.loc[e.eligible, "symbol"].tolist())), OP)
        self.assertEqual(len(a), 60); self.assertEqual(a, b)                          # order-independent, seeded
        late = S.assign(first="2025-12-01"); self.assertEqual(int(to.screen(late, OP, "new").eligible.sum()), 0)
        sp = to.screen(S.assign(first="2018-01-01"), OP, "spot"); self.assertEqual(int(sp.eligible.sum()), 100)


class StatTests(unittest.TestCase):
    def test_growth_gate(self):
        idx = pd.date_range("2020-01-01", periods=1500, freq="D", tz="UTC")
        rng = np.random.default_rng(0)
        up = pd.Series(1e5 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 1500))), index=idx)
        flat = pd.Series(1e5 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 1500))), index=idx)
        self.assertTrue(to.gate_growth(up, 1e5, OP, 1)["pass"])
        self.assertFalse(to.gate_growth(flat, 1e5, OP, 1)["pass"])

    def test_calmar_gate_identical(self):
        idx = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
        D = pd.Series(1e5 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.001, 0.02, 800))), index=idx)
        r = to.gate_calmar(D, D, 1e5, OP, 3)
        self.assertAlmostEqual(r["calmar_diff"], 0.0); self.assertFalse(r["pass"])

    def test_verdict(self):
        self.assertEqual(to.verdict(True, True), "HOLDS"); self.assertEqual(to.verdict(True, False), "EDGE_ONLY")
        self.assertEqual(to.verdict(False, True), "REVERSAL_ONLY"); self.assertEqual(to.verdict(False, False), "FAILS")

    def test_equal_weight_hold(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
        a = pd.Series(np.exp(np.arange(10) * 0.01), index=idx); b = pd.Series(np.exp(np.arange(5) * 0.03), index=idx[5:])
        L = to.equal_weight_hold({"a": a, "b": b})
        self.assertAlmostEqual(L.iloc[0], 0.01); self.assertAlmostEqual(L.iloc[-1], 0.02)


class ToolParsingTests(unittest.TestCase):
    def test_s3_listing_pagination(self):
        import download_tb13_data as dl
        ns = "http://s3.amazonaws.com/doc/2006-03-01/"
        p1 = f'<ListBucketResult xmlns="{ns}"><IsTruncated>true</IsTruncated><NextMarker>m1</NextMarker><CommonPrefixes><Prefix>data/x/AUSDT/</Prefix></CommonPrefixes></ListBucketResult>'.encode()
        p2 = f'<ListBucketResult xmlns="{ns}"><IsTruncated>false</IsTruncated><CommonPrefixes><Prefix>data/x/BUSDT/</Prefix></CommonPrefixes><Contents><Key>data/x/k.zip</Key></Contents></ListBucketResult>'.encode()
        with mock.patch.object(dl, "fetch", side_effect=[p1, p2]):
            self.assertEqual(dl.s3_list("data/x/", folders=True), ["data/x/AUSDT/", "data/x/BUSDT/"])
        with mock.patch.object(dl, "fetch", side_effect=[p2]):
            self.assertEqual(dl.s3_list("data/x/", folders=False), ["data/x/k.zip"])

    def test_parse_klines_ms_us_header(self):
        import download_tb13_data as dl
        for unit, t0 in (("ms", 1514764800000), ("us", 1514764800000000)):
            for header in (False, True):
                rows = "\n".join(f"{t0 + i * (60000 if unit == 'ms' else 60000000)},1,2,0.5,1.5,10,0,15,3,4,6,0" for i in range(3))
                if header:
                    rows = ",".join(dl.KCOLS) + "\n" + rows
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w") as z:
                    z.writestr("x.csv", rows)
                df = dl.parse_klines(buf.getvalue())
                self.assertEqual(df.open_time.iloc[0], pd.Timestamp("2018-01-01", tz="UTC")); self.assertEqual(len(df), 3)


if __name__ == "__main__":
    unittest.main()
