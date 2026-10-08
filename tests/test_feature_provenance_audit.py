"""Synthetic provenance checks (v0.2.0); no local DuckDB required."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import feature_provenance_audit as p


def fixture():
    t = pd.date_range("2026-09-23 08:00", periods=4, freq="5min", tz="UTC")
    final = pd.DataFrame({
        "underlying": ["BTC"] * 4, "state_time": t,
        "expiry_date": ["2026-09-24"] * 4,
        "book_obi_20_std": [None, None, 1., 2.],
        "book_twa_abs_obi_20": [None, None, .4, .5],
        "instability_features_ready": [False, False, False, True],
        "l2_intrawindow_ready": [False, False, True, True],
        "perp_integrity_ok": [True] * 4,
        "in_settlement_window": [False] * 4,
        "option_atm_quote_ok": [True] * 4,
        "atm_iv_change_1h": [None] * 4,
    })
    raw = pd.DataFrame({
        "underlying": ["BTC"] * 4, "bar_end": t,
        "book_obi_20_std": [None, 1., 1., 2.],
        "book_twa_abs_obi_20": [None, None, .4, .5],
        "book_stats_complete": [False, True, True, True],
        "collector_version": ["v0", "v1", "v1", "v1"],
    })
    canon = pd.DataFrame({
        "underlying": ["BTC"] * 4, "state_time": t,
        "book_obi_20_std": [None, 1., 1., 2.],
        "book_twa_abs_obi_20": [None, None, .4, .5],
    })
    market = pd.DataFrame({
        "underlying": ["BTC"] * 4, "state_time": t,
        "book_obi_20_std": [None, None, 1., 2.],
        "book_twa_abs_obi_20": [None, None, .4, .5],
    })
    return final, raw, canon, market


class FeatureProvenanceAuditTests(unittest.TestCase):
    def test_align_and_report_duplicate_raw_keys(self):
        final, raw, canon, market = fixture()
        duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        stages, diagnostic = p.align_stages(final, {
            "raw": p.normalize_stage(duplicate, "bar_end"),
            "canonical": canon, "market": market})
        self.assertEqual(len(stages["raw"]), 4)
        self.assertEqual(int(diagnostic.loc[diagnostic.stage == "raw",
                                           "duplicate_key_rows"].iloc[0]), 2)
        self.assertEqual(int(diagnostic.loc[diagnostic.stage == "raw",
                                           "matched_model_states"].iloc[0]), 4)

    def test_distinguish_schema_missing_from_filtered_observations(self):
        final, raw, canon, market = fixture()
        stages, _ = p.align_stages(final, {
            "raw": p.normalize_stage(raw, "bar_end"),
            "canonical": canon, "market": market})
        cov = p.stage_coverage(stages,
              ("book_obi_20_std", "book_twa_abs_obi_20", "nonexistent"))
        diag = p.l2_attribution(cov).set_index("field")
        self.assertEqual(diag.loc["nonexistent", "diagnosis"],
                         "raw_field_absent_in_union_schema")
        self.assertEqual(diag.loc["book_obi_20_std", "raw_nonnull"], 3)
        self.assertEqual(diag.loc["book_obi_20_std", "market_nonnull"], 2)
        self.assertEqual(diag.loc["book_obi_20_std", "model_gate_nonnull"], 1)
        self.assertEqual(diag.loc["book_obi_20_std", "diagnosis"],
                         "partial_readiness_exclusion")

    def test_attribute_upstream_drop_not_to_collector(self):
        final, raw, canon, market = fixture()
        canon["book_twa_abs_obi_20"] = None
        market["book_twa_abs_obi_20"] = None
        stages, _ = p.align_stages(final, {
            "raw": p.normalize_stage(raw, "bar_end"),
            "canonical": canon, "market": market})
        diagnosis = p.l2_attribution(p.stage_coverage(
            stages, ("book_twa_abs_obi_20",))).iloc[0].diagnosis
        self.assertEqual(diagnosis, "raw_to_canonical_missing")

    def test_gate_summary_and_option_clock(self):
        final, raw, _, _ = fixture()
        stages, _ = p.align_stages(final, {
            "raw": p.normalize_stage(raw, "bar_end")})
        summary = p.gate_summary(stages["final"], stages["raw"])
        self.assertEqual(int(summary.iloc[0]["raw_book_stats_complete"]), 3)
        self.assertEqual(int(summary.iloc[0]["instability_features_ready"]), 1)
        options = p.option_hour_coverage(stages["final"])
        self.assertEqual(int(options.iloc[0].utc_hour), 8)
        self.assertEqual(int(options.iloc[0].atm_iv_change_1h_native), 0)

    def test_duplicate_model_states_fail_hard(self):
        final, raw, _, _ = fixture()
        final.loc[1, "state_time"] = final.loc[0, "state_time"]
        with self.assertRaisesRegex(ValueError, "Duplicate model states"):
            p.align_stages(final, {"raw": p.normalize_stage(raw, "bar_end")})

    def test_unmatched_states_by_expiry_and_hour(self):
        final, raw, _, _ = fixture()
        out = p.unmatched_states(final, p.normalize_stage(raw.iloc[2:], "bar_end"))
        self.assertEqual(int(out["unmatched"].sum()), 2)
        self.assertEqual(int(out["states"].sum()), 4)
        self.assertEqual(out["first_raw_bar_end"].iloc[0], raw["bar_end"].iloc[2])

    def test_null_return_simultaneity(self):
        t = pd.date_range("2026-09-23 08:00", periods=3, freq="5min", tz="UTC")
        f = pd.DataFrame({
            "underlying": ["BTC"] * 3 + ["ETH"] * 3,
            "state_time": list(t) * 2,
            "return_5m": [None, 0.1, None, None, 0.2, 0.3],
            "index_return_input_eligible": [False, True, True, True, True, True],
        })
        per_t, summary = p.null_return_simultaneity(f)
        s = summary.set_index("underlying")
        self.assertEqual(int(s.loc["BTC", "null_returns"]), 2)
        self.assertAlmostEqual(s.loc["BTC", "null_all_assets_share"], 0.5)  # t0 shared
        self.assertAlmostEqual(s.loc["BTC", "null_isolated_share"], 0.5)    # t2 alone
        self.assertAlmostEqual(s.loc["BTC", "null_with_own_bar_ineligible_share"], 0.5)
        self.assertEqual(int(per_t["all_assets_null"].sum()), 1)

    def test_collector_vintage_from_data(self):
        r = pd.DataFrame({
            "underlying": ["BTC"] * 5,
            "state_time": pd.date_range("2026-10-01", periods=5, freq="5min", tz="UTC"),
            "trade_count": [1., 1., 1., 1., np.nan],
            "book_obi_20_std": [np.nan, 1., 1., 1., np.nan],
            "book_out_of_order_update_count": [np.nan, np.nan, 0., 0., np.nan],
            "collector_version": [None, None, None, "futures_5m_l2_v0.3.0", None],
        })
        v = p.collector_vintage(r).tolist()
        self.assertEqual(v, ["pre_l2_instability", "l2_instability_pre_eventtime_fix",
                             "eventtime_fix_unstamped", "futures_5m_l2_v0.3.0", "<no_raw>"])

    def test_boundary_shift_detects_definitional_break(self):
        t = pd.Series(pd.date_range("2026-10-03", periods=96, freq="5min", tz="UTC"))
        v = pd.Series(["l2_instability_pre_eventtime_fix"] * 48 + ["eventtime_fix_unstamped"] * 48)
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"l2_feat": np.r_[rng.normal(0, 1, 48), rng.normal(3, 1, 48)],
                          "index_feat": rng.normal(0, 1, 96)})
        out = p.boundary_shift(X, v, t, families={"l2_feat": "l2", "index_feat": "volatility"})
        s = out.set_index("feature")["std_mean_shift"]
        self.assertGreater(s["l2_feat"], 2)
        self.assertLess(abs(s["index_feat"]), 1)


if __name__ == "__main__":
    unittest.main()
