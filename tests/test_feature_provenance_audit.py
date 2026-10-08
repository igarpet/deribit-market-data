"""Five synthetic provenance checks; no local DuckDB required."""
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


if __name__ == "__main__":
    unittest.main()
