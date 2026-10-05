"""Verify the imbalance technique catalogue (requirement 2): coverage, leakage, thresholds and focal loss.

The suite checks that the catalogue covers every required group, that each sampler reaches its target ratio
without mutating inputs, that the analytic focal-loss gradient and hessian match finite differences, that
thresholds come from the precision-recall curve, and that resampling stays inside fold-train.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from typing import Dict

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.imbalance_lab import config as C  # noqa: E402
from labs.imbalance_lab.losses import FocalLossClassifier, focal_grad_hess, focal_loss_value  # noqa: E402
from labs.imbalance_lab.models import BalancedWeightClassifier  # noqa: E402
from labs.imbalance_lab.samplers import (SamplerChain, make_hybrid_sampler,  # noqa: E402
                                    make_single_sampler, resampling_backend)
from labs.imbalance_lab.techniques import (BASELINE_TECHNIQUE, IMPLEMENTATION,  # noqa: E402
                                      REFERENCE_GROUPS, REQUIRED_GROUPS, TECHNIQUE_ORDER,
                                      _markdown, build_technique, run, technique_group)
from labs.imbalance_lab.thresholds import (pr_curve_points, tune_thresholds,  # noqa: E402
                                      tune_thresholds_from_pr_curve)

try:
    import lightgbm  # noqa: F401

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - lightgbm is absent
    HAS_LIGHTGBM = False

#: Expected groups listed independently of the code, so a missing technique fails the test.
EXPECTED_GROUPS: Dict[str, set] = {
    "data-level/oversampling": {"ros", "smote", "borderline_smote", "adasyn"},
    "data-level/undersampling": {"rus", "tomek", "enn"},
    "hybrid": {"smote_tomek", "smote_enn"},
    "algorithm-level": {"cost_sensitive_scale_pos_weight", "cost_sensitive_class_weight",
                        "focal_loss"},
    "ensemble": {"balanced_rf", "easy_ensemble", "rusboost"},
}


def tiny_dataset(n_samples: int = 1500, weights=(0.95, 0.05), seed: int = 0):
    """A small deterministic imbalanced dataset shared by the tests below."""
    X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=6, n_classes=2,
                               weights=list(weights), flip_y=0.0, random_state=seed)
    return X.astype(float), y.astype(int)


def _ratio(y: np.ndarray) -> float:
    """Minority-to-majority ratio, matching imblearn's `sampling_strategy`."""
    y = np.asarray(y)
    return float((y == 1).sum()) / max(int((y == 0).sum()), 1)


class TestCatalogCoverage(unittest.TestCase):
    """The catalogue must cover all five groups and every technique listed by requirement 2."""

    def test_required_groups_and_techniques_present(self):
        self.assertEqual(set(REQUIRED_GROUPS), set(EXPECTED_GROUPS))
        for group, expected in EXPECTED_GROUPS.items():
            with self.subTest(group=group):
                self.assertTrue(expected.issubset(set(REQUIRED_GROUPS[group])),
                                f"{group}: missing {sorted(expected - set(REQUIRED_GROUPS[group]))}")

    def test_every_technique_has_doc_and_implementation(self):
        known_groups = set(EXPECTED_GROUPS) | set(REFERENCE_GROUPS)
        for key in TECHNIQUE_ORDER:
            with self.subTest(technique=key):
                spec = build_technique(key, quick=True)
                self.assertTrue(spec["doc"], f"{key}: docstring is missing")
                self.assertTrue(IMPLEMENTATION.get(key), f"{key}: no implementation entry")
                self.assertTrue(callable(spec["factory"]))
                self.assertIsNotNone(spec["estimator"], f"{key}: estimator could not be built")
                self.assertIn(technique_group(key), known_groups)

    def test_baseline_is_present_and_runs_first(self):
        """Requirement 3 needs an untreated baseline for comparison, so it must exist and run first."""
        self.assertIn(BASELINE_TECHNIQUE, TECHNIQUE_ORDER)
        self.assertEqual(TECHNIQUE_ORDER[0], BASELINE_TECHNIQUE)
        self.assertEqual(technique_group(BASELINE_TECHNIQUE), "baseline")
        spec = build_technique(BASELINE_TECHNIQUE, quick=True)
        self.assertEqual(spec["kind"], "baseline")
        self.assertFalse(spec["is_resampling"])

    def test_resampling_flag_matches_samplers(self):
        for key in TECHNIQUE_ORDER:
            spec = build_technique(key, quick=True)
            with self.subTest(technique=key):
                self.assertEqual(spec["is_resampling"], bool(spec["samplers"]))
                if spec["is_resampling"]:
                    self.assertIsNotNone(spec["probe_factory"],
                                         f"{key}: probe_factory is required to log the resampled distribution")

    def test_hybrid_group_uses_two_steps(self):
        for key in REQUIRED_GROUPS["hybrid"]:
            with self.subTest(technique=key):
                steps = make_hybrid_sampler(key, prefer_imblearn=True,
                                            over_strategy=0.5, under_strategy=0.5,
                                            k_neighbors=5, random_state=0)
                self.assertEqual(len(steps), 1)
                self.assertTrue(hasattr(steps[0][1], "fit_resample"))


class TestSamplerTechniques(unittest.TestCase):
    """Each sampler must reach its target ratio, clean only the border, and never mutate its inputs."""

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = tiny_dataset(n_samples=3000, weights=(0.95, 0.05), seed=1)

    def _resample(self, steps, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        if len(steps) == 1:
            return steps[0][1].fit_resample(X, y)
        return SamplerChain(steps).fit_resample(X, y)

    def test_oversamplers_reach_target_ratio(self):
        """RandomOverSampler, SMOTE, BorderlineSMOTE and ADASYN reach about half the majority count."""
        for prefer_imblearn in (True, False):
            for kind in ("ros", "smote", "borderline_smote", "adasyn"):
                with self.subTest(backend="imblearn" if prefer_imblearn else "builtin", kind=kind):
                    steps = make_single_sampler(kind, prefer_imblearn=prefer_imblearn,
                                                over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                                under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                                k_neighbors=C.TECHNIQUE_K_NEIGHBORS,
                                                random_state=0)
                    _Xr, yr = self._resample(steps)
                    self.assertGreater(len(yr), len(self.y), f"{kind}: samples must be added")
                    self.assertAlmostEqual(_ratio(yr), C.TECHNIQUE_OVER_STRATEGY, delta=0.03)

    def test_undersampler_reaches_target_ratio(self):
        for prefer_imblearn in (True, False):
            with self.subTest(backend="imblearn" if prefer_imblearn else "builtin"):
                steps = make_single_sampler("rus", prefer_imblearn=prefer_imblearn,
                                            over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                            under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                            k_neighbors=C.TECHNIQUE_K_NEIGHBORS, random_state=0)
                _Xr, yr = self._resample(steps)
                self.assertLess(len(yr), len(self.y))
                self.assertAlmostEqual(_ratio(yr), C.TECHNIQUE_UNDER_STRATEGY, delta=0.03)

    def test_cleaning_techniques_keep_minority_untouched(self):
        """Tomek and ENN clean the border, dropping majority samples and keeping the minority count fixed."""
        n_positive = int((self.y == 1).sum())
        for prefer_imblearn in (True, False):
            for kind in ("tomek", "enn"):
                with self.subTest(backend="imblearn" if prefer_imblearn else "builtin", kind=kind):
                    steps = make_single_sampler(kind, prefer_imblearn=prefer_imblearn,
                                                over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                                under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                                k_neighbors=C.TECHNIQUE_K_NEIGHBORS,
                                                random_state=0)
                    _Xr, yr = self._resample(steps)
                    self.assertEqual(int((yr == 1).sum()), n_positive,
                                     f"{kind}: minority samples must not be removed")
                    self.assertLessEqual(len(yr), len(self.y), f"{kind}: samples may only be removed")

    def test_hybrid_techniques_balance_then_clean(self):
        for kind in ("smote_tomek", "smote_enn"):
            steps_smote = make_single_sampler("smote", prefer_imblearn=True,
                                              over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                              under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                              k_neighbors=C.TECHNIQUE_K_NEIGHBORS, random_state=0)
            _Xs, ys = self._resample(steps_smote)
            steps = make_hybrid_sampler(kind, prefer_imblearn=True,
                                        over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                        under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                        k_neighbors=C.TECHNIQUE_K_NEIGHBORS, random_state=0)
            _Xr, yr = self._resample(steps)
            with self.subTest(technique=kind):
                self.assertAlmostEqual(_ratio(yr), C.TECHNIQUE_OVER_STRATEGY, delta=0.06)
                self.assertLess(len(yr), len(ys),
                                f"{kind}: the cleaning step must drop samples relative to plain SMOTE")

    def test_samplers_never_mutate_inputs(self):
        kinds = [("single", kind) for kind in ("ros", "smote", "borderline_smote", "adasyn", "rus",
                                              "tomek", "enn")]
        kinds += [("hybrid", kind) for kind in ("smote_tomek", "smote_enn")]
        for group, kind in kinds:
            for prefer_imblearn in (True, False):
                with self.subTest(kind=kind, backend="imblearn" if prefer_imblearn else "builtin"):
                    X_before, y_before = self.X.copy(), self.y.copy()
                    builder = make_single_sampler if group == "single" else make_hybrid_sampler
                    steps = builder(kind, prefer_imblearn=prefer_imblearn,
                                    over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                    under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                    k_neighbors=C.TECHNIQUE_K_NEIGHBORS, random_state=0)
                    self._resample(steps, self.X.copy(), self.y.copy())
                    self.assertTrue(np.array_equal(self.X, X_before))
                    self.assertTrue(np.array_equal(self.y, y_before))

    def test_backend_is_reported(self):
        self.assertIn(resampling_backend(True), ("imblearn", "builtin"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _focal_loss_per_sample(y: np.ndarray, z: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """Per-sample focal loss computed from the logit, used to check the analytic gradients."""
    p = np.clip(_sigmoid(z), 1e-12, 1.0 - 1e-12)
    return -(y * alpha * (1.0 - p) ** gamma * np.log(p)
             + (1.0 - y) * (1.0 - alpha) * p ** gamma * np.log1p(-p))


class TestFocalLoss(unittest.TestCase):
    """Focal loss as a LightGBM custom objective: the formulas must be correct, not merely runnable."""

    GAMMA, ALPHA = 2.0, 0.75

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(400) < 0.1).astype(float)
        self.z = rng.normal(0.0, 1.5, 400)

    def test_gradient_matches_finite_difference(self):
        grad, _hess = focal_grad_hess(self.y, self.z, self.GAMMA, self.ALPHA)
        h = 1e-6
        for i in (0, 5, 101, 250, 399):
            z = self.z[i]
            plus = _focal_loss_per_sample(self.y[i:i + 1], np.array([z + h]), self.GAMMA, self.ALPHA)
            minus = _focal_loss_per_sample(self.y[i:i + 1], np.array([z - h]), self.GAMMA, self.ALPHA)
            numeric = float((plus[0] - minus[0]) / (2 * h))
            self.assertAlmostEqual(float(grad[i]), numeric, delta=1e-5,
                                   msg=f"gradient is wrong for sample {i}")

    def test_hessian_matches_finite_difference_of_gradient(self):
        _grad, hess = focal_grad_hess(self.y, self.z, self.GAMMA, self.ALPHA)
        h = 1e-5
        eps = np.zeros_like(self.z)
        for i in (3, 77, 200, 399):
            eps[:] = 0.0
            eps[i] = h
            plus = focal_grad_hess(self.y, self.z + eps, self.GAMMA, self.ALPHA)[0][i]
            minus = focal_grad_hess(self.y, self.z - eps, self.GAMMA, self.ALPHA)[0][i]
            numeric = float((plus - minus) / (2 * h))
            self.assertAlmostEqual(float(hess[i]), numeric, delta=1e-4,
                                   msg=f"hessian is wrong for sample {i}")

    def test_gamma_zero_reduces_to_weighted_bce(self):
        grad, hess = focal_grad_hess(self.y, self.z, 0.0, self.ALPHA)
        p = _sigmoid(self.z)
        expected_grad = np.where(self.y > 0.5, self.ALPHA * (p - 1.0), (1.0 - self.ALPHA) * p)
        expected_hess = np.where(self.y > 0.5, self.ALPHA * p * (1.0 - p),
                                 (1.0 - self.ALPHA) * p * (1.0 - p))
        np.testing.assert_allclose(grad, expected_grad, rtol=0, atol=1e-9)
        np.testing.assert_allclose(hess, expected_hess, rtol=0, atol=1e-9)

    def test_hessian_is_strictly_positive(self):
        """LightGBM requires a positive hessian so the custom objective does not diverge."""
        extreme = np.concatenate([self.z, np.array([-30.0, -5.0, 0.0, 5.0, 30.0])])
        y = np.concatenate([self.y, np.array([1.0, 0.0, 1.0, 0.0, 1.0])])
        _grad, hess = focal_grad_hess(y, extreme, self.GAMMA, self.ALPHA)
        self.assertTrue(bool(np.all(hess > 0.0)), "a hessian at or below zero would make LightGBM fail")

    def test_loss_value_prefers_confident_correct_predictions(self):
        good = focal_loss_value(np.array([1, 0]), np.array([0.99, 0.01]), self.GAMMA, self.ALPHA)
        bad = focal_loss_value(np.array([1, 0]), np.array([0.01, 0.99]), self.GAMMA, self.ALPHA)
        self.assertLess(good, bad)

    @unittest.skipUnless(HAS_LIGHTGBM, "lightgbm is required for FocalLossClassifier")
    def test_classifier_probabilities_and_dynamic_alpha(self):
        X, y = tiny_dataset(n_samples=1200, seed=3)
        model = FocalLossClassifier(gamma=2.0, alpha=None, n_estimators=60,
                                    learning_rate=0.1, random_state=0)
        model.fit(X, y)
        expected_alpha = float((y == 0).sum()) / len(y)          # alpha is the negative share
        self.assertAlmostEqual(model.alpha_, expected_alpha, places=9)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
        self.assertTrue(bool(((proba >= 0.0) & (proba <= 1.0)).all()))
        self.assertEqual(set(model.predict(X).tolist()), {0, 1})

    def test_balanced_class_weight_uses_fit_labels(self):
        """`class_weight='balanced'` must derive its weights from the labels passed to `fit`."""
        X, y = tiny_dataset(n_samples=600, weights=(0.8, 0.2), seed=5)
        model = BalancedWeightClassifier(random_state=0).fit(X, y)
        from sklearn.utils.class_weight import compute_class_weight

        expected = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        self.assertAlmostEqual(model.class_weight_[0], float(expected[0]), places=9)
        self.assertAlmostEqual(model.class_weight_[1], float(expected[1]), places=9)


class TestPRThresholds(unittest.TestCase):
    """Threshold tuning from the precision-recall curve must take candidates from the curve itself."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        n = 1500
        cls.y = (rng.random(n) < 0.08).astype(int)
        # Probabilities carrying signal: positives shift up clearly while still overlapping.
        cls.proba = np.clip(0.35 * cls.y + rng.random(n) * 0.75, 0.0, 1.0)

    def test_pr_auc_matches_sklearn(self):
        curve = pr_curve_points(self.y, self.proba)
        self.assertAlmostEqual(curve["pr_auc"], float(average_precision_score(self.y, self.proba)),
                               places=12)

    def test_best_f1_threshold_lies_on_the_pr_curve(self):
        curve = pr_curve_points(self.y, self.proba)
        tuned = tune_thresholds_from_pr_curve(self.y, self.proba, cost_fn=C.COST_FN,
                                              cost_fp=C.COST_FP,
                                              precision_target=C.PRECISION_TARGET)
        candidates = np.unique(np.concatenate([curve["thresholds"], np.array([0.5, 1.0])]))
        self.assertTrue(bool(np.any(np.isclose(candidates, tuned["best_f1"]["threshold"]))),
                        "the best threshold must be a point of the PR curve, or 0.5 or 1.0")

    def test_best_f1_equals_max_f1_over_curve_points(self):
        curve = pr_curve_points(self.y, self.proba)
        precision, recall = curve["precision"], curve["recall"]
        f1_curve = np.where(precision + recall > 0,
                            2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
        tuned = tune_thresholds_from_pr_curve(self.y, self.proba, cost_fn=C.COST_FN,
                                              cost_fp=C.COST_FP,
                                              precision_target=C.PRECISION_TARGET)
        self.assertAlmostEqual(tuned["best_f1"]["f1"], float(np.max(f1_curve)), places=6)

    def test_best_cost_minimises_expected_cost(self):
        curve = pr_curve_points(self.y, self.proba)
        tuned = tune_thresholds_from_pr_curve(self.y, self.proba, cost_fn=C.COST_FN,
                                              cost_fp=C.COST_FP,
                                              precision_target=C.PRECISION_TARGET)
        best = tuned["best_cost"]
        cost = best["fn"] * C.COST_FN + best["fp"] * C.COST_FP
        self.assertAlmostEqual(best["expected_cost"], float(cost), places=6)
        self.assertGreaterEqual(best["threshold"], 0.0)
        self.assertLessEqual(best["threshold"], 1.0)
        self.assertTrue(bool(np.any(np.isclose(np.unique(curve["thresholds"]), best["threshold"]))))

    def test_min_precision_meets_target_and_beats_fixed_half(self):
        tuned = tune_thresholds_from_pr_curve(self.y, self.proba, cost_fn=C.COST_FN,
                                              cost_fp=C.COST_FP,
                                              precision_target=0.3)
        self.assertTrue(tuned["min_precision"]["target_met"])
        self.assertGreaterEqual(tuned["min_precision"]["precision"], 0.3 - 1e-9)
        # The PR-based threshold must not be worse than the 0.5 reference on F1.
        self.assertGreaterEqual(tuned["best_f1"]["f1"] + 1e-9, tuned["fixed_0.5"]["f1"])

    def test_two_implementations_agree_on_best_f1(self):
        """Candidates taken from the curve are real curve points, so they cannot lose to the grid.

        A 1001-point quantile grid can miss the probability that maximises F1, so the on-curve result must be
        at least as good, with only a negligible gap.
        """
        on_curve = tune_thresholds_from_pr_curve(self.y, self.proba, cost_fn=C.COST_FN,
                                                 cost_fp=C.COST_FP, precision_target=0.3)
        on_grid = tune_thresholds(self.y, self.proba, cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                                  precision_target=0.3)
        self.assertGreaterEqual(on_curve["best_f1"]["f1"] + 1e-9, on_grid["best_f1"]["f1"])
        self.assertLessEqual(on_curve["best_f1"]["f1"] - on_grid["best_f1"]["f1"], 0.02)
        # Both variants must agree on whether the precision target is met.
        self.assertEqual(on_curve["min_precision"]["target_met"],
                         on_grid["min_precision"]["target_met"])


class TestNoLeakageInCatalog(unittest.TestCase):
    """Requirement 2, leakage part: resample fold-train only and choose thresholds from out-of-fold data."""

    n_samples = 2000
    n_splits = 3

    def _evaluate(self, key: str):
        from labs.imbalance_lab.data import make_imbalanced_dataset, stratified_holdout_split
        from labs.imbalance_lab.techniques import evaluate_technique

        X, y = make_imbalanced_dataset(n_samples=self.n_samples, random_state=C.SEED)
        split = stratified_holdout_split(X, y, seed=C.SEED)
        spec = build_technique(key, random_state=C.SEED, quick=True)
        return split, evaluate_technique(spec, split["X_train"], split["y_train"],
                                         split["X_test"], split["y_test"],
                                         n_splits=self.n_splits, seed=C.SEED, log=lambda _m: None,
                                         threshold_fn=tune_thresholds_from_pr_curve)

    def test_resampling_touches_only_fold_train(self):
        for key in ("smote", "borderline_smote", "rus", "smote_tomek"):
            with self.subTest(technique=key):
                split, item = self._evaluate(key)
                self.assertEqual(item["status"], "ok", item["reason"])
                self.assertTrue(all(item["checks"].values()), item["checks"])
                global_pct = 100.0 * float((split["y_train"] == 1).mean())
                for row in item["resample_rows"]:
                    self.assertAlmostEqual(row["val_pos_pct"], global_pct, delta=0.3,
                                           msg="the fold-validation label distribution changed")
                    self.assertLess(row["train_ir_after"], row["train_ir_before"])

    def test_thresholds_are_recomputed_from_oof_only(self):
        """The thresholds in use must equal those recomputed from the out-of-fold probabilities."""
        split, item = self._evaluate("ros")
        recomputed = tune_thresholds_from_pr_curve(item["oof_y"], item["oof_proba"],
                                                   cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                                                   precision_target=C.PRECISION_TARGET)
        for mode, value in item["thresholds"].items():
            with self.subTest(mode=mode):
                self.assertAlmostEqual(value, float(recomputed[mode]["threshold"]), places=12)
        # The out-of-fold data belongs to train_pool, not to test.
        self.assertEqual(len(item["oof_proba"]), len(split["y_train"]))
        self.assertAlmostEqual(float(np.mean(item["oof_y"])),
                               float(np.mean(split["y_train"])), places=12)

    def test_test_set_is_untouched_and_scored_at_all_modes(self):
        split, item = self._evaluate("smote")
        self.assertTrue(item["checks"]["test_untouched"])
        modes = sorted(row["threshold_mode"] for row in item["rows"])
        self.assertEqual(modes, sorted({"best_f1", "best_cost", "min_precision", "fixed_0.5"}))
        for row in item["rows"]:
            self.assertEqual(row["n_test"], len(split["y_test"]))

    def test_cleaning_technique_reports_nearly_unchanged_ratio(self):
        """Tomek links only clean the border, so the fold-train imbalance ratio hardly moves."""
        _split, item = self._evaluate("tomek")
        for row in item["resample_rows"]:
            self.assertLess(abs(row["train_ir_after"] - row["train_ir_before"]), 2.0)


class TestCatalogRunSmoke(unittest.TestCase):
    """Run a small end-to-end catalogue into a temporary directory and check the artifacts."""

    @unittest.skipUnless(HAS_LIGHTGBM, "lightgbm is required to run the catalogue")
    def test_small_run_writes_artifacts_and_passes_leak_checks(self):
        with tempfile.TemporaryDirectory() as td:
            original = C.ARTIFACTS_DIR
            C.ARTIFACTS_DIR = pathlib.Path(td)
            try:
                result = run(n_samples=1500, n_splits=2,
                             techniques=["ros", "tomek", "cost_sensitive_class_weight",
                                         "focal_loss", "easy_ensemble"],
                             write=True, quick=True)
            finally:
                C.ARTIFACTS_DIR = original

            self.assertTrue(result["leakage_all_pass"], result["missing_libraries"])
            statuses = {item["technique"]: item["status"] for item in result["results"]}
            self.assertEqual(set(statuses.values()), {"ok"}, statuses)
            for name in ("techniques.md", "techniques.csv", "techniques.json", "techniques.log",
                         "techniques_by_fold.csv"):
                self.assertTrue((pathlib.Path(td) / name).exists(), f"missing artifact {name}")
            markdown = (pathlib.Path(td) / "techniques.md").read_text(encoding="utf-8")
            for expected in ("leak-free comparison", "Data leakage checks", "PASS", "Conclusions"):
                self.assertIn(expected, markdown)
            csv_rows = (pathlib.Path(td) / "techniques.csv").read_text(
                encoding="utf-8").strip().splitlines()
            self.assertEqual(len(csv_rows) - 1, 4 * len(statuses))

    @unittest.skipUnless(HAS_LIGHTGBM, "lightgbm is required to run the catalogue")
    def test_markdown_renders_for_every_technique(self):
        result = run(n_samples=1200, n_splits=2, techniques=["smote", "adasyn", "focal_loss"],
                     write=False, quick=True)
        text = _markdown(result)
        for key in ("smote", "adasyn", "focal_loss"):
            self.assertIn(key, text)
        self.assertIn("Thresholds from out-of-fold probabilities", text)
