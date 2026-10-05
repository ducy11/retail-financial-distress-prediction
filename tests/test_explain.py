"""Verify the self-implemented KernelSHAP in `forecasting/explain.py`.

The environment has no `shap` package, so the implementation must be shown correct rather than merely
runnable: it must recover the analytic Shapley values of a linear function, satisfy the efficiency
identity, and rank features in agreement with permutation importance.
"""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.explain import (coalition_weight, kernel_shap_matrix, kernel_shap_values,
                                 linear_shap_exact, mean_abs_shap, rank_agreement,
                                 sample_coalitions)  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()


class TestCoalitionWeights(unittest.TestCase):
    """Shapley kernel weights follow pi(S) = (M - 1) / (C(M, |S|) * |S| * (M - |S|))."""

    def test_matches_formula_and_is_symmetric(self):
        n_features = 6
        for size in (1, 2, 3, 4, 5):
            expected = (n_features - 1) / (math.comb(n_features, size) * size * (n_features - size))
            self.assertAlmostEqual(coalition_weight(n_features, size), expected, places=12)
        # Symmetry between |S| and M - |S|, a defining property of the Shapley kernel.
        self.assertAlmostEqual(coalition_weight(n_features, 1),
                               coalition_weight(n_features, n_features - 1), places=12)

    def test_boundaries_are_excluded(self):
        self.assertEqual(coalition_weight(5, 0), 0.0)
        self.assertEqual(coalition_weight(5, 5), 0.0)

    def test_sampling_returns_expected_shape_and_non_empty_coalitions(self):
        rng = np.random.default_rng(0)
        masks = sample_coalitions(7, 50, rng)
        self.assertEqual(masks.shape, (50, 7))
        sizes = masks.sum(axis=1)
        self.assertTrue(np.all(sizes >= 1) and np.all(sizes <= 6))

    def test_single_feature_returns_no_coalitions(self):
        self.assertEqual(sample_coalitions(1, 10, np.random.default_rng(0)).shape[0], 0)


class TestKernelShapCorrectness(unittest.TestCase):
    """KernelSHAP must recover the Shapley values of a linear function, whose analytic form is known."""

    def setUp(self):
        rng = np.random.default_rng(1)
        self.background = rng.normal(size=(60, 5))
        self.weights = np.array([2.0, -1.5, 0.5, 3.0, 0.0])
        self.bias = 0.7
        self.x = rng.normal(size=5)

        def predict(matrix: np.ndarray) -> np.ndarray:
            return np.asarray(matrix, dtype=float) @ self.weights + self.bias

        self.predict = predict

    def test_recovers_analytic_shapley_values(self):
        base, exact = linear_shap_exact(self.weights, self.x, self.background, bias=self.bias)
        result = kernel_shap_values(self.predict, self.background, self.x, n_coalitions=600,
                                    rng=np.random.default_rng(7))
        self.assertAlmostEqual(result["base_value"], base, places=6)
        # The mean absolute error must be small on the scale of phi, which reaches about 3.
        self.assertLess(float(np.mean(np.abs(result["phi"] - exact))), 0.1)

    def test_efficiency_holds_exactly(self):
        result = kernel_shap_values(self.predict, self.background, self.x, n_coalitions=200,
                                    rng=np.random.default_rng(3))
        self.assertLess(abs(result["efficiency_gap"]), 1e-9)
        self.assertAlmostEqual(result["phi"].sum() + result["base_value"], result["prediction"],
                               places=9)

    def test_matrix_version_reports_self_check(self):
        X_explain = self.background[:6]
        result = kernel_shap_matrix(self.predict, self.background, X_explain, n_coalitions=150,
                                    random_state=5)
        self.assertEqual(result["phi"].shape, (6, 5))
        self.assertLess(result["max_abs_efficiency_gap"], 1e-9)
        self.assertEqual(result["n_explained"], 6)

    def test_zero_weight_feature_gets_zero_contribution(self):
        result = kernel_shap_values(self.predict, self.background, self.x, n_coalitions=400,
                                    rng=np.random.default_rng(11))
        self.assertLess(abs(result["phi"][4]), 0.05)      # the feature with weight zero


class TestRankAgreement(unittest.TestCase):
    """Compare SHAP rankings with permutation importance, which must yield a measurable agreement."""

    def test_identical_rankings_give_spearman_one(self):
        a = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
        result = rank_agreement(a, a, top=3)
        self.assertAlmostEqual(result["spearman"], 1.0, places=6)
        self.assertAlmostEqual(result["top_overlap"], 1.0, places=6)
        self.assertEqual(result["agreed_features"], [0, 1, 2])

    def test_reversed_rankings_give_negative_spearman(self):
        a = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
        result = rank_agreement(a, -a, top=2)          # top-2 of a is {0, 1}, of -a it is {3, 4}
        self.assertAlmostEqual(result["spearman"], -1.0, places=6)
        self.assertAlmostEqual(result["top_overlap"], 0.0, places=6)

    def test_short_inputs_are_handled(self):
        self.assertIsNone(rank_agreement(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["spearman"])


class TestMeanAbsShap(unittest.TestCase):
    """Global importance measured as the mean absolute contribution."""

    def test_averages_absolute_contributions(self):
        phi = np.array([[1.0, -3.0], [3.0, 1.0]])
        np.testing.assert_allclose(mean_abs_shap(phi), [2.0, 2.0])
