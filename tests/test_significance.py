"""Verify the significance tests in `forecasting/significance.py`.

The conclusion that the model does not beat `ticker_prior` rests on these tests, so the module checks that
its AUROC matches `sklearn`, that identical systems show no difference, and that a clearly better system
yields a small p-value with an interval excluding zero.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.significance import compare_systems, delong_test, paired_bootstrap  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()


def make_systems(seed: int = 0):
    """Two systems sharing one label vector: a class-separating one and a near-random one."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * 40 + [1] * 24)
    good = np.clip(0.5 + 0.35 * (y - 0.5) * 2 + rng.normal(0, 0.08, y.size), 0, 1)
    weak = np.clip(0.5 + rng.normal(0, 0.2, y.size), 0, 1)
    return y, good, weak


class TestDelong(unittest.TestCase):
    """DeLong (1988): AUROC and the variance of the difference between two correlated ROC curves."""

    def test_auc_matches_sklearn(self):
        y, good, weak = make_systems(1)
        result = delong_test(y, good, weak)
        self.assertTrue(result["auc_matches_sklearn"])
        self.assertAlmostEqual(result["auc_a"], float(roc_auc_score(y, good)), places=9)

    def test_identical_systems_have_no_difference(self):
        y, good, _ = make_systems(2)
        result = delong_test(y, good, good)
        self.assertAlmostEqual(result["delta"], 0.0, places=12)
        self.assertTrue(result["p_value"] is None or result["p_value"] > 0.99)

    def test_clearly_better_system_has_small_p_value(self):
        y, good, weak = make_systems(3)
        result = delong_test(y, good, weak)
        self.assertGreater(result["delta"], 0.2)
        self.assertLess(result["p_value"], 0.01)
        self.assertGreater(result["z"], 2.0)

    def test_too_few_of_one_class_is_reported(self):
        y = np.array([1, 1, 0])
        result = delong_test(y, [0.9, 0.8, 0.2], [0.7, 0.6, 0.4])
        self.assertIsNone(result["p_value"])
        self.assertIn("reason", result)


class TestPairedBootstrap(unittest.TestCase):
    """Paired bootstrap for the delta in average precision and AUROC, with a 95% interval and p-value."""

    def test_delta_and_direction(self):
        y, good, weak = make_systems(4)
        result = paired_bootstrap(y, good, weak, "average_precision", n_boot=400, seed=7)
        self.assertGreater(result["delta"], 0.0)
        self.assertGreater(result["ci95_lower"], 0.0)        # the interval excludes zero
        self.assertLess(result["p_value"], 0.05)
        self.assertTrue(result["significant_5pct"])

    def test_swapping_order_flips_sign(self):
        y, good, weak = make_systems(5)
        forward = paired_bootstrap(y, good, weak, "auroc", n_boot=200, seed=11)
        backward = paired_bootstrap(y, weak, good, "auroc", n_boot=200, seed=11)
        self.assertAlmostEqual(forward["delta"], -backward["delta"], places=12)
        self.assertAlmostEqual(forward["ci95_lower"], -backward["ci95_upper"], places=12)

    def test_identical_systems_give_zero_delta(self):
        y, good, _ = make_systems(6)
        result = paired_bootstrap(y, good, good, "average_precision", n_boot=100, seed=13)
        self.assertAlmostEqual(result["delta"], 0.0, places=12)
        self.assertEqual(result["ci95_lower"], 0.0)
        self.assertEqual(result["ci95_upper"], 0.0)
        self.assertAlmostEqual(result["p_value"], 1.0, places=6)


class TestCompareSystems(unittest.TestCase):
    """Comparison table: per-system AUROC and AP plus DeLong and bootstrap results for every pair."""

    def test_reports_all_pairs_and_metrics(self):
        y, good, weak = make_systems(7)
        result = compare_systems(y, {"good": good, "weak": weak,
                                     "prior": np.full(y.size, y.mean())},
                                 n_boot=100, baseline="prior", seed=3)
        self.assertEqual(result["n_samples"], y.size)
        self.assertEqual(len(result["pairs"]), 3)            # C(3, 2) pairs
        self.assertGreater(result["systems"]["good"]["auroc"], result["systems"]["weak"]["auroc"])
        self.assertEqual(len(result["pairs_vs_baseline"]), 2)
        for pair in result["pairs"]:
            self.assertIn("delong_auroc", pair)
            self.assertIn("bootstrap_ap", pair)
