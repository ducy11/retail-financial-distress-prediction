"""Verify heavy-tail preprocessing in `forecasting/preprocessing.py` and the model factory.

The focus is that winsorize bounds are learned from train and then applied unchanged elsewhere, since
adapting them per split would leak; that the pipeline keeps its step order; and that the heavy-tail
scalers build and behave differently from `StandardScaler` in the presence of outliers.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.models import DEFAULT_SCALER, MODEL_REGISTRY, make_model  # noqa: E402
from forecasting.preprocessing import (SCALER_KINDS, WINSOR_METHODS, Winsorizer,  # noqa: E402
                                        make_scaler)
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()


class TestWinsorizer(unittest.TestCase):
    """Clip bounds come from the fitted split, which is the main leakage guard here."""

    def setUp(self):
        self.train = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0], [4.0, 16.0],
                               [5.0, 18.0], [100.0, 20.0]])       # outlier in column 0

    def test_iqr_bounds_are_learned_from_fit_data(self):
        winsorizer = Winsorizer(method="iqr").fit(self.train)
        self.assertEqual(winsorizer.lower_.shape, (2,))
        self.assertLess(winsorizer.lower_[0], winsorizer.upper_[0])
        self.assertLess(winsorizer.upper_[0], 100.0)              # the outlier is clipped

    def test_transform_uses_train_bounds_not_new_data(self):
        winsorizer = Winsorizer(method="p1p99").fit(self.train)
        upper = winsorizer.upper_[0]
        # An extreme value in new data must be clipped to the bound learned from train.
        transformed = winsorizer.transform(np.array([[1e6, 11.0]]))
        self.assertAlmostEqual(transformed[0, 0], upper, places=9)
        self.assertAlmostEqual(transformed[0, 1], 11.0, places=9)

    def test_nan_is_preserved_for_imputer(self):
        winsorizer = Winsorizer(method="iqr").fit(self.train)
        transformed = winsorizer.transform(np.array([[np.nan, 12.0]]))
        self.assertTrue(np.isnan(transformed[0, 0]))

    def test_none_method_is_identity(self):
        winsorizer = Winsorizer(method="none").fit(self.train)
        np.testing.assert_allclose(winsorizer.transform(self.train), self.train)

    def test_clip_share_counts_actual_intervention(self):
        winsorizer = Winsorizer(method="iqr").fit(self.train)
        self.assertGreater(winsorizer.clip_share(self.train), 0.0)
        self.assertLessEqual(winsorizer.clip_share(self.train), 1.0)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            Winsorizer(method="khong-ton-tai").fit(self.train)


class TestScalers(unittest.TestCase):
    """Heavy-tail scalers must build and behave differently from `StandardScaler` under outliers."""

    def test_factory_returns_expected_types(self):
        self.assertEqual(make_scaler("none"), "passthrough")
        self.assertIsInstance(make_scaler("standard"), StandardScaler)
        self.assertIsInstance(make_scaler("robust"), RobustScaler)
        self.assertIsInstance(make_scaler("power"), PowerTransformer)
        self.assertIsInstance(make_scaler("quantile"), QuantileTransformer)

    def test_invalid_kind_raises(self):
        with self.assertRaises(ValueError):
            make_scaler("not-a-scaler")

    def test_robust_scaler_keeps_bulk_spread_under_outlier(self):
        # With one extreme outlier the mean and standard deviation are dominated by it, so
        # StandardScaler compresses the bulk while RobustScaler, using median and IQR, does not.
        data = np.concatenate([np.linspace(0, 1, 50), [1e4]]).reshape(-1, 1)
        standard = StandardScaler().fit_transform(data)[:-1, 0]
        robust = make_scaler("robust").fit_transform(data)[:-1, 0]
        self.assertGreater(float(np.std(robust)), float(np.std(standard)))

    def test_all_declared_kinds_are_constructible(self):
        for kind in SCALER_KINDS:
            self.assertIsNotNone(make_scaler(kind))
        for method in WINSOR_METHODS:
            self.assertIsNotNone(Winsorizer(method=method).fit(np.array([[1.0], [2.0], [3.0]])))


class TestPipelineAssembly(unittest.TestCase):
    """`make_model` must assemble the steps as winsorize, impute, optional scale, then model."""

    def test_tree_pipeline_has_no_scaler_by_default(self):
        if "random_forest" not in MODEL_REGISTRY:
            self.skipTest("random_forest is not registered")
        steps = [name for name, _ in make_model("random_forest").steps]
        self.assertEqual(steps, ["impute", "model"])

    def test_linear_pipeline_has_standard_scaler_by_default(self):
        steps = dict(make_model("logistic").steps)
        self.assertIn("scale", steps)
        self.assertIsInstance(steps["scale"], StandardScaler)
        self.assertEqual(DEFAULT_SCALER["logistic"], "standard")

    def test_winsorize_step_is_first_and_configurable(self):
        steps = make_model("logistic", scaler="robust", winsorize="iqr").steps
        self.assertEqual([name for name, _ in steps][:2], ["winsorize", "impute"])
        self.assertIsInstance(dict(steps)["winsorize"], Winsorizer)
        self.assertIsInstance(dict(steps)["scale"], RobustScaler)

    def test_pipeline_clips_outlier_before_model(self):
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [1e6, 0.0]])
        y = np.array([0, 0, 1, 1, 0, 1])
        model = make_model("logistic", winsorize="iqr").fit(X, y)
        winsorizer = dict(model.steps)["winsorize"]
        self.assertLess(winsorizer.upper_[0], 1e6)

    def test_unknown_model_still_raises(self):
        with self.assertRaises(KeyError):
            make_model("not-a-model")


class TestModelRegistry(unittest.TestCase):
    """The project uses exactly four pure scikit-learn families, with no xgboost or lightgbm."""

    def test_registry_has_exactly_four_models(self):
        from forecasting.models import DEFAULT_MODEL_ORDER, HYPERPARAMS

        self.assertEqual(sorted(MODEL_REGISTRY),
                         ["hist_gradient_boosting", "logistic", "mlp", "random_forest"])
        self.assertEqual(sorted(DEFAULT_MODEL_ORDER), sorted(MODEL_REGISTRY))
        self.assertEqual(sorted(HYPERPARAMS), sorted(MODEL_REGISTRY))

    def test_mlp_is_scaled_by_default(self):
        """An MLP learns from feature magnitude, so a scaling step is mandatory."""
        from forecasting.models import DEFAULT_SCALER, NEEDS_SCALING

        self.assertIn("scale", [name for name, _ in make_model("mlp").steps])
        self.assertIn("mlp", NEEDS_SCALING)
        self.assertEqual(DEFAULT_SCALER["mlp"], "standard")

    def test_mlp_returns_probabilities_in_unit_interval(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        proba = make_model("mlp").fit(X, y).predict_proba(X)[:, 1]
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

    def test_no_optional_boosting_library_is_imported(self):
        """The main pipeline must not import lightgbm or xgboost."""
        src = (ROOT / "forecasting" / "models.py").read_text(encoding="utf-8")
        for needle in ("lightgbm", "xgboost"):
            self.assertNotIn(f"import {needle}", src,
                             f"the main pipeline must not import {needle}")

    def test_hist_gradient_boosting_runs_with_winsorize(self):
        steps = make_model("hist_gradient_boosting", winsorize="iqr")
        self.assertEqual([name for name, _ in steps.steps][0], "winsorize")
        X = np.random.default_rng(0).normal(size=(40, 5))
        y = (X[:, 0] > 0).astype(int)
        steps.fit(X, y)
        proba = steps.predict_proba(X)[:, 1]
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))
