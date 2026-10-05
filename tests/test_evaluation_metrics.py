"""Verify the model evaluation section (requirement 3): metric set, accuracy policy and resampling.

The suite checks that the metrics match `sklearn`, that accuracy is never treated as a primary measure, that
resampling runs inside `imblearn.pipeline.Pipeline` rather than a plain scikit-learn pipeline, and that the
baseline-versus-technique comparison reports the right deltas and ordering.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import (average_precision_score, confusion_matrix as sk_confusion,
                             f1_score, fbeta_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labs.imbalance_lab import config as C  # noqa: E402
from labs.imbalance_lab.metrics import (COMPARISON_COLUMNS, DIAGNOSTIC_METRICS, PRIMARY_METRICS,  # noqa: E402
                                   accuracy_diagnostic, bootstrap_ci, confusion_matrix_table,
                                   fbeta, majority_baseline_accuracy, metric_scalar,
                                   metrics_at_threshold)
from labs.imbalance_lab.samplers import (build_sampler_pipeline, make_samplers,  # noqa: E402
                                    make_single_sampler, resampling_backend)
from labs.imbalance_lab.techniques import (BASELINE_TECHNIQUE, build_technique,  # noqa: E402
                                      compare_with_baseline, comparison_markdown, run)

try:
    import imblearn  # noqa: F401

    HAS_IMBLEARN = True
except Exception:  # pragma: no cover - imbalanced-learn is absent
    HAS_IMBLEARN = False

try:
    import lightgbm  # noqa: F401

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - lightgbm is absent
    HAS_LIGHTGBM = False


def make_proba(n_samples: int = 800, positive_rate: float = 0.1, seed: int = 0):
    """Synthetic `(y_true, proba)` pair carrying signal, shared by the metric tests."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n_samples) < positive_rate).astype(int)
    proba = np.clip(0.4 * y + rng.random(n_samples) * 0.7, 0.0, 1.0)
    return y, proba


class TestMetricSet(unittest.TestCase):
    """Metrics must match `sklearn` rather than re-derive formulas, and cover the required indicators."""

    @classmethod
    def setUpClass(cls):
        cls.y, cls.proba = make_proba()
        cls.threshold = 0.45
        cls.y_pred = (cls.proba >= cls.threshold).astype(int)
        cls.metrics = metrics_at_threshold(cls.y, cls.proba, cls.threshold)

    def test_primary_metrics_present(self):
        for name in PRIMARY_METRICS:
            with self.subTest(metric=name):
                self.assertIn(name, self.metrics)
                self.assertTrue(bool(np.isfinite(float(self.metrics[name]))))

    def test_thresholded_metrics_match_sklearn(self):
        expected = {
            "precision": precision_score(self.y, self.y_pred, zero_division=0),
            "recall": recall_score(self.y, self.y_pred, zero_division=0),
            "f1": f1_score(self.y, self.y_pred, zero_division=0),
            "macro_f1": f1_score(self.y, self.y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(self.y, self.y_pred, average="weighted", zero_division=0),
            "fbeta": fbeta_score(self.y, self.y_pred, beta=C.FBETA_BETA, zero_division=0),
            "mcc": matthews_corrcoef(self.y, self.y_pred),
        }
        for name, value in expected.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(self.metrics[name], float(value), places=12)

    def test_threshold_free_metrics_match_sklearn(self):
        self.assertAlmostEqual(self.metrics["pr_auc"],
                               float(average_precision_score(self.y, self.proba)), places=12)
        self.assertAlmostEqual(self.metrics["roc_auc"],
                               float(roc_auc_score(self.y, self.proba)), places=12)

    def test_confusion_matrix_matches_sklearn(self):
        matrix = np.asarray(self.metrics["confusion_matrix"])
        np.testing.assert_array_equal(matrix, sk_confusion(self.y, self.y_pred, labels=[0, 1]))
        tn, fp, fn, tp = matrix.ravel()
        self.assertEqual((self.metrics["tn"], self.metrics["fp"], self.metrics["fn"],
                          self.metrics["tp"]), (int(tn), int(fp), int(fn), int(tp)))
        self.assertEqual(int(matrix.sum()), len(self.y))
        table = confusion_matrix_table(self.y, self.y_pred)
        self.assertEqual(table["labels"], ["negative", "positive"])
        self.assertEqual(table["counts"]["tp"], int(tp))

    def test_fbeta_behaviour(self):
        """At beta 1 the score must equal F1, and `fbeta_beta` must report the coefficient in use."""
        self.assertAlmostEqual(fbeta(self.y, self.y_pred, beta=1.0),
                               float(f1_score(self.y, self.y_pred, zero_division=0)), places=12)
        self.assertAlmostEqual(self.metrics["fbeta_beta"], float(C.FBETA_BETA), places=12)
        self.assertGreater(int(self.y_pred.sum()), 0, "the test needs at least one positive prediction")
        tuned = metrics_at_threshold(self.y, self.proba, self.threshold, beta=3.0)
        self.assertAlmostEqual(tuned["fbeta"], fbeta(self.y, self.y_pred, beta=3.0), places=12)

    def test_metric_scalar_and_bootstrap_cover_primary_metrics(self):
        for metric in ("pr_auc", "f1", "macro_f1", "fbeta", "recall"):
            with self.subTest(metric=metric):
                value = metric_scalar(self.y, self.proba, self.threshold, metric)
                self.assertTrue(bool(np.isfinite(value)))
        self.assertAlmostEqual(metric_scalar(self.y, self.proba, self.threshold, "pr_auc"),
                               self.metrics["pr_auc"], places=12)
        with self.assertRaises(KeyError):
            metric_scalar(self.y, self.proba, self.threshold, "metric_khong_ho_tro")
        ci = bootstrap_ci(self.y, self.proba, self.threshold, metric="macro_f1", n_boot=60)
        self.assertEqual(ci["metric"], "macro_f1")
        self.assertLessEqual(ci["lo95"], ci["point"] + 1e-12)
        self.assertGreaterEqual(ci["hi95"], ci["point"] - 1e-12)


class TestAccuracyPolicy(unittest.TestCase):
    """Requirement 3: accuracy must never serve as a primary measure."""

    def test_accuracy_is_not_a_primary_metric(self):
        self.assertNotIn("accuracy", PRIMARY_METRICS)
        self.assertNotIn("accuracy", COMPARISON_COLUMNS)
        self.assertIn("accuracy", DIAGNOSTIC_METRICS)

    def test_majority_predictor_has_useless_accuracy(self):
        y = np.array([0] * 98 + [1] * 2)
        y_pred = np.zeros(len(y), dtype=int)          # always predict the majority class
        diagnostic = accuracy_diagnostic(y, y_pred)
        self.assertAlmostEqual(diagnostic["accuracy"], 0.98, places=12)
        self.assertAlmostEqual(diagnostic["majority_baseline_accuracy_pct"], 98.0, places=12)
        self.assertFalse(diagnostic["accuracy_better_than_majority"])
        self.assertIn("Accuracy is not the primary measure", diagnostic["note"])
        metrics = metrics_at_threshold(y, (y * 0 + 0.1), 0.5)   # every prediction is negative
        self.assertIn("accuracy", metrics)                      # still reported for diagnosis
        self.assertFalse(metrics["accuracy_better_than_majority"])  # but flagged as no better
        self.assertEqual(metrics["recall"], 0.0)                # the primary metric tells the truth

    def test_majority_baseline_helper(self):
        self.assertAlmostEqual(majority_baseline_accuracy(np.array([1, 1, 0, 0])), 50.0)
        self.assertTrue(bool(np.isnan(majority_baseline_accuracy(np.array([])))))



class RecordingClassifier:
    """A stub classifier that records how many samples and labels it receives in `fit`."""

    def __init__(self) -> None:
        self.n_seen = -1
        self.n_positive_seen = -1
        self.positive_rate_ = 0.5

    def fit(self, X, y):
        self.n_seen = int(len(y))
        self.n_positive_seen = int((np.asarray(y) == 1).sum())
        return self

    def predict_proba(self, X):
        positive = np.full(len(X), float(self.positive_rate_))
        return np.column_stack([1.0 - positive, positive])


class TestImblearnPipeline(unittest.TestCase):
    """Requirement 3: resampling must go through `imblearn.pipeline.Pipeline`, not the plain sklearn one."""

    def test_catalogue_uses_imblearn_pipeline_for_resampling(self):
        spec = build_technique("smote", prefer_imblearn=True, random_state=0, quick=True)
        pipe = spec["estimator"]
        with self.subTest(backend=resampling_backend(True)):
            if HAS_IMBLEARN:
                self.assertEqual(type(pipe).__module__, "imblearn.pipeline",
                                 "must use imblearn.pipeline.Pipeline, not sklearn.pipeline")
            self.assertEqual([name for name, _step in pipe.steps][-1], "classifier")
            self.assertIn("smote", [name for name, _step in pipe.steps])

    def test_baseline_has_no_sampler_and_no_resampling(self):
        spec = build_technique(BASELINE_TECHNIQUE, random_state=0, quick=True)
        self.assertFalse(spec["is_resampling"])
        self.assertIsNone(spec["probe_factory"])
        self.assertFalse(hasattr(spec["estimator"], "fit_resample"))
        self.assertEqual(spec["group"], "baseline")

    def test_sampler_inside_pipeline_only_touches_fold_train(self):
        X, y = make_classification(n_samples=1200, n_features=8, n_informative=5, n_classes=2,
                                   weights=[0.95, 0.05], flip_y=0.0, random_state=1)
        train, val = slice(0, 800), slice(800, 1200)
        X_val_before, y_val_before = X[val].copy(), y[val].copy()
        recorder = RecordingClassifier()
        steps = make_single_sampler("smote", prefer_imblearn=True, over_strategy=0.5,
                                    under_strategy=0.5, k_neighbors=5, random_state=0)
        pipe = build_sampler_pipeline(steps, recorder, prefer_imblearn=True)
        pipe.fit(X[train], y[train])
        pipe.predict_proba(X[val])
        self.assertNotEqual(recorder.n_seen, int(len(y[train])),
                            "the classifier must receive the resampled train set")
        self.assertGreater(recorder.n_seen, int(len(y[train])))
        self.assertTrue(bool(np.array_equal(X[val], X_val_before)))
        self.assertTrue(bool(np.array_equal(y[val], y_val_before)))

    def test_sklearn_pipeline_does_not_resample(self):
        """Shows why imblearn is required: the plain scikit-learn pipeline never calls `fit_resample`.

        Both outcomes prove it, since scikit-learn either raises because the sampler has no `transform`
        or, when it runs, the classifier still sees the original data size.
        """
        from sklearn.pipeline import Pipeline as SkPipeline

        if not HAS_IMBLEARN:  # pragma: no cover - imblearn is absent
            self.skipTest("imbalanced-learn is required to build the comparison sampler")
        from imblearn.over_sampling import SMOTE as ImbSMOTE

        X, y = make_classification(n_samples=600, n_features=8, n_informative=5, n_classes=2,
                                   weights=[0.9, 0.1], flip_y=0.0, random_state=2)
        recorder = RecordingClassifier()
        pipeline = SkPipeline([("smote", ImbSMOTE(sampling_strategy=0.5, random_state=0)),
                               ("clf", recorder)])
        try:
            pipeline.fit(X, y)
        except Exception:                     # the sampler has no transform, so sklearn cannot run it
            return
        self.assertEqual(recorder.n_seen, len(y),
                         "the sklearn pipeline must not resample; use imblearn to resample")


def _row(mode: str, **overrides) -> dict:
    """A fake metric row for one threshold mode, carrying the keys `compare_with_baseline` reads."""
    row = {"technique": "x", "group": "x", "kind": "x", "threshold_mode": mode, "split": "test",
           "n_test": 200, "threshold": 0.4, "precision": 0.6, "recall": 0.7, "f1": 0.6,
           "macro_f1": 0.55, "weighted_f1": 0.58, "fbeta": 0.65, "fbeta_beta": 2.0,
           "pr_auc": 0.5, "roc_auc": 0.9, "balanced_accuracy": 0.7, "mcc": 0.4, "brier": 0.05,
           "accuracy": 0.9, "majority_baseline_accuracy_pct": 98.0,
           "n_predicted_positive": 20, "tn": 170, "fp": 10, "fn": 10, "tp": 10,
           "confusion_matrix": [[170, 10], [10, 10]]}
    row.update(overrides)
    return row


def _item(technique: str, group: str, kind: str, *, is_resampling: bool,
          f1: float, pr_auc: float) -> dict:
    """A fake technique result carrying only the keys the report needs."""
    row = _row("best_f1", technique=technique, group=group, kind=kind, f1=f1, pr_auc=pr_auc,
               macro_f1=f1 - 0.05, weighted_f1=f1 - 0.02, fbeta=f1 - 0.01)
    return {"technique": technique, "group": group, "kind": kind, "doc": technique,
            "implementation": technique, "is_resampling": is_resampling, "status": "ok", "reason": "",
            "rows": [row], "fold_rows": [], "resample_rows": [], "thresholds": {"best_f1": 0.4},
            "oof_pr_auc": pr_auc, "checks": {"test_untouched": True}, "oof_proba": None,
            "oof_y": None, "oof_at_0.5": {}}


def _result(items: list) -> dict:
    """A fake `result` complete enough to run the comparison and the report without training."""
    return {"results": items, "techniques": [item["technique"] for item in items], "n_samples": 100,
            "n_splits": 2, "seed": 42, "quick": True, "backend": "lightgbm",
            "resampling_backend": "imblearn", "missing_libraries": {}, "leakage_all_pass": True,
            "required_groups": {}, "rows": [], "resample_rows": [], "fold_rows": [], "log": "",
            "label_distribution": {"all": {"n": 100, "positive_pct": 2.0, "imbalance_ratio": 49.0},
                                   "train_pool": {"n": 80, "positive_pct": 2.0},
                                   "test": {"n": 20, "positive_pct": 2.0}}}



class TestBaselineComparison(unittest.TestCase):
    """Requirement 3: the comparison of the untreated baseline against each treatment technique."""

    @classmethod
    def setUpClass(cls):
        cls.result = _result([
            _item(BASELINE_TECHNIQUE, "baseline", "baseline", is_resampling=False,
                  f1=0.50, pr_auc=0.70),
            _item("smote", "data-level/oversampling", "data-level", is_resampling=True,
                  f1=0.56, pr_auc=0.75),
            _item("focal_loss", "algorithm-level", "algorithm-level", is_resampling=False,
                  f1=0.40, pr_auc=0.60),
        ])
        cls.table = compare_with_baseline(cls.result)

    def test_baseline_row_is_first_and_deltas_are_zero(self):
        self.assertIsNotNone(self.table["baseline"])
        self.assertEqual(self.table["rows"][0]["technique"], BASELINE_TECHNIQUE)
        for metric in ("pr_auc", "f1", "macro_f1", "fbeta", "recall"):
            self.assertAlmostEqual(self.table["baseline"][f"delta_{metric}"], 0.0, places=12)

    def test_deltas_are_computed_against_baseline(self):
        rows = {row["technique"]: row for row in self.table["rows"]}
        self.assertAlmostEqual(rows["smote"]["delta_pr_auc"], 0.05, places=12)
        self.assertAlmostEqual(rows["smote"]["delta_f1"], 0.06, places=12)
        self.assertAlmostEqual(rows["focal_loss"]["delta_pr_auc"], -0.10, places=12)
        self.assertEqual(self.table["n_better_than_baseline"], 1)
        self.assertEqual(self.table["n_techniques_compared"], 2)

    def test_comparison_excludes_accuracy_and_has_every_primary_metric(self):
        for name in ("accuracy", "n_predicted_positive"):
            with self.subTest(excluded=name):
                self.assertNotIn(name, self.table["metrics"])
                for row in self.table["rows"]:
                    self.assertNotIn(name, row)
        for name in ("precision", "recall", "f1", "macro_f1", "weighted_f1", "fbeta", "pr_auc",
                     "roc_auc", "mcc", "tn", "fp", "fn", "tp"):
            self.assertIn(name, self.table["metrics"])

    def test_best_by_metric_and_markdown(self):
        self.assertEqual(self.table["best_by_metric"]["pr_auc"]["technique"], "smote")
        self.assertEqual(self.table["best_by_metric"]["f1"]["technique"], "smote")
        text = comparison_markdown(self.result)
        for expected in (BASELINE_TECHNIQUE, "dPR-AUC", "`smote`", "accuracy"):
            self.assertIn(expected, text)

    def test_missing_baseline_is_reported_not_crashed(self):
        partial = _result([_item("smote", "data-level/oversampling", "data-level",
                                 is_resampling=True, f1=0.5, pr_auc=0.6)])
        table = compare_with_baseline(partial)
        self.assertIsNone(table["baseline"])
        self.assertEqual(table["n_techniques_compared"], 0)
        self.assertIn("baseline missing", comparison_markdown(partial))

    @unittest.skipUnless(HAS_LIGHTGBM, "lightgbm is required for the real run")
    def test_end_to_end_comparison_and_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            original = C.ARTIFACTS_DIR
            C.ARTIFACTS_DIR = pathlib.Path(td)
            try:
                result = run(n_samples=1500, n_splits=2,
                             techniques=[BASELINE_TECHNIQUE, "smote"], write=True, quick=True)
            finally:
                C.ARTIFACTS_DIR = original
            table = compare_with_baseline(result)
            self.assertIsNotNone(table["baseline"], "the real results must contain a baseline row")
            rows = {row["technique"]: row for row in table["rows"]}
            self.assertIn("smote", rows)
            self.assertEqual(rows[BASELINE_TECHNIQUE]["delta_pr_auc"], 0.0)
            self.assertIsNotNone(rows["smote"]["delta_pr_auc"])
            self.assertTrue((pathlib.Path(td) / "techniques_comparison.csv").exists())
            markdown = (pathlib.Path(td) / "techniques.md").read_text(encoding="utf-8")
            for expected in ("Baseline (untreated) versus handling techniques",
                             "Diagnostic metric, not used to draw conclusions", "F1-macro"):
                self.assertIn(expected, markdown)
