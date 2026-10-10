"""Synthetic tests for the Volatility Dynamics "b" run (02b-06b), 06c, 08, 09 and 10; no research DuckDB required.

Run from repo root: python -m unittest discover -s tests -p "test_volatility_dynamics_b.py" -v
"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_engineering.modeling import volatility_vd_b_core as vb
from data_engineering.modeling import volatility_q99_precursors as qp
from data_engineering.modeling import volatility_clock_null as cn
from data_engineering.modeling import volatility_q99_precursor_null as pn
from data_engineering.modeling import volatility_q99_matching_b as mb
from data_engineering.modeling import volatility_q99_motif95_prospective as mp
from data_engineering.modeling import volatility_hmm_states_b as hb
from data_engineering.modeling import volatility_hmm_states_c as hc
from data_engineering.modeling import volatility_jump_hawkes as jh
from data_engineering.modeling import volatility_jump_compound as jc
from data_engineering.modeling import volatility_jump_option_wings as ow

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestCoreReproducesModule03(unittest.TestCase):
    """03b/04b rest on the vectorized core matching Module 03 exactly."""

    def test_events_controls_scores_rankings(self):
        f = vb.synthetic_frame(n_days=14, seed=5)
        f = f.drop(index=f.sample(150, random_state=1).index)
        f = f.sort_values(["underlying", "state_time"]).reset_index(drop=True)
        feats = ["return_5m", "intrabar_range_bps", "volume_usd", "noise_feature"]
        rel = list(range(-12, 3))

        ev, _ = qp.identify_q99_onsets(f)
        org = qp.build_origin_index(f, ev)
        ctl = qp.choose_controls(org, ev, max_controls=8)
        sc = qp.percentile_scores(f, feats, ev, ctl, rel, min_controls=4)
        rk = qp.precursor_rankings(sc).set_index("feature")

        g = vb.build_grid(f)
        tail = vb.tail_flags(g, vb.q99_thresholds(g))
        er = np.flatnonzero(vb.onset_flags(g, tail))
        elig = vb.eligible_origins(g, tail)
        self.assertTrue(np.array_equal(np.sort(er), np.sort(ev.event_panel_index.to_numpy())))
        self.assertTrue(np.array_equal(elig, org.eligible_control_origin.to_numpy()))

        cc = vb.clock_controls(g, er, elig, max_controls=8)
        ref = ctl.groupby("event_id").control_panel_index.apply(sorted).to_dict()
        evid = dict(zip(ev.event_panel_index, ev.event_id))
        for k, e in enumerate(er):
            self.assertEqual(ref.get(evid[e], []), sorted(cc[k][cc[k] >= 0].tolist()))

        P = vb.percentile_tensor(g, g.feature_matrix(feats), er, cc, rel, min_controls=4)
        self.assertEqual(int(np.isfinite(P).sum()), len(sc))
        idx = {e: k for k, e in enumerate(er)}
        row = dict(zip(ev.event_id, ev.event_panel_index))
        for r in sc.itertuples():
            self.assertAlmostEqual(
                P[idx[row[r.event_id]], rel.index(r.rel_bar), feats.index(r.feature)],
                r.percentile_vs_controls,
            )
        R = vb.module03_rankings(P, g.time_ns[er], g.asset_code[er], rel, qp.DEFAULT_RANK_LAGS)
        for j, name in enumerate(feats):
            a = rk.discovery_score.get(name, np.nan)
            b = R.loc[j, "discovery_score"]
            self.assertTrue((np.isnan(a) and np.isnan(b)) or np.isclose(a, b))


class TestModuleSelftests(unittest.TestCase):
    def test_core(self):
        self.assertTrue(all(vb.selftest().values()))

    def test_02b_clock_null(self):
        self.assertTrue(all(cn.selftest().values()))

    def test_03b_precursor_null(self):
        self.assertTrue(all(pn.selftest().values()))

    def test_04b_matching(self):
        self.assertTrue(all(mb.selftest().values()))

    def test_05b_prospective(self):
        self.assertTrue(all(mp.selftest().values()))

    def test_06b_hmm(self):
        self.assertTrue(all(hb.selftest().values()))

    def test_06c_student_t_hmm(self):
        self.assertTrue(all(hc.selftest().values()))

    def test_08_jump_hawkes(self):
        r = jh.selftest()
        self.assertTrue(all(v for k, v in r.items() if isinstance(v, bool)))

    def test_09_compound_jumps(self):
        self.assertTrue(all(v for v in jc.selftest().values() if isinstance(v, bool)))

    def test_10_option_wings(self):
        self.assertTrue(all(v for v in ow.selftest().values() if isinstance(v, bool)))


class TestHolm(unittest.TestCase):
    def test_holm_matches_definition(self):
        p = np.array([0.01, 0.04, 0.03, np.nan, 0.5])
        h = mb.holm(p)
        self.assertAlmostEqual(h[0], 0.04)
        self.assertAlmostEqual(h[2], 0.09)
        self.assertAlmostEqual(h[1], 0.09)
        self.assertAlmostEqual(h[4], 0.5)
        self.assertTrue(np.isnan(h[3]))


if __name__ == "__main__":
    unittest.main()
