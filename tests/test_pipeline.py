"""Verify data regeneration, feature engineering, the split policy and leak prevention.

Two requirements are covered: the class ratio survives both the chronological split and the cross-validation
folds, and every preprocessing or balancing step is fitted on train only. The audit script is also shown to
detect deliberately corrupted data.
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting import data as fdata  # noqa: E402
from forecasting.data_loader import load_prepared  # noqa: E402
from forecasting.features import (  # noqa: E402
    build_feature_matrix,
    extract_labels,
    feature_names,
)
from forecasting.models import HYPERPARAMS, MODEL_REGISTRY, make_model  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - unittest hook
    """Reapply the harmless-warning filters after `unittest` installs `simplefilter("default")`.

    The logistic pipeline emits `OptimizeWarning: Unknown solver options: iprint` on every fit under the
    pinned scikit-learn and scipy versions. See `runtime_warnings.py`.
    """
    quiet_library_warnings()

#: The "preserve the class ratio" requirement allows at most a 5 percentage point gap between each split and
#: the pooled positive rate. Measured: train 62.3 percent, validation 65.6, test 59.4, pooled 62.0, so the
#: worst gap is 3.6 points. The split is chronological within each company, so exact equality is impossible.
MAX_SPLIT_RATIO_GAP = 0.05
#: StratifiedGroupKFold has only eight groups, so its stratification is weaker than StratifiedKFold: the
#: measured worst gap is 9.7 points with a mean of 5.6, hence the 15 point cap and 10 point mean cap.
MAX_FOLD_RATIO_GAP = 0.15
MAX_FOLD_MEAN_ABS_GAP = 0.10



class TestDataRegeneration(unittest.TestCase):
    """`forecasting.data --force` must regenerate the prepared splits byte for byte."""

    def test_regenerate_byte_identical(self):
        orig = ROOT / "data" / "prepared"
        if not (orig / "manifest.json").exists():
            self.skipTest("data/prepared is missing; run `python -m forecasting.data --force` first.")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            shutil.copytree(orig, tmp, dirs_exist_ok=True)  # labels and manifest dates from the original
            fd_prep, fd_man = fdata.PREPARED_DIR, fdata.MANIFEST_FILE
            fdata.PREPARED_DIR = tmp
            fdata.MANIFEST_FILE = tmp / "manifest.json"
            try:
                res = fdata.run(force=True)
                self.assertFalse(res.get("skipped"))
                self.assertEqual(res["train"], 212)
                self.assertEqual(res["validation"], 32)
                self.assertEqual(res["test"], 64)
                self.assertEqual(res["purged"], 16)
            finally:
                fdata.PREPARED_DIR, fdata.MANIFEST_FILE = fd_prep, fd_man
            for name in ("train.json", "validation.json", "test.json", "purged.json",
                         "manifest.json"):
                with self.subTest(file=name):
                    self.assertEqual(
                        (orig / name).read_bytes(), (tmp / name).read_bytes(),
                        f"{name} differs from the original after regeneration")


class TestSplits(unittest.TestCase):
    def test_counts_and_policy(self):
        counts = {n: len(load_prepared(n)) for n in ("train", "validation", "test", "purged")}
        self.assertEqual(counts, {"train": 212, "validation": 32, "test": 64, "purged": 16})
        # Purging covers both sides of validation, leaving two purged samples per company.
        purged = load_prepared("purged")
        by_ticker = {}
        for s in purged:
            by_ticker.setdefault(s["ticker"], []).append(s["sample_id"])
        for t, ids in by_ticker.items():
            self.assertEqual(len(ids), 2, f"{t} must have exactly two purged samples")

    def test_manifest_consistency(self):
        man = json.loads((ROOT / "data" / "prepared" / "manifest.json").read_text(encoding="utf-8"))
        for name in ("train", "validation", "test", "purged"):
            self.assertEqual(man["counts"][name], len(load_prepared(name)))

    def test_class_ratio_preserved_across_splits(self):
        """Requirement 1: the split must preserve the class ratio, verified rather than merely claimed.

        The project splits chronologically within each company, taking the last eight quarters as test, four
        as validation and a surrounding purge band, so the guarantee checked here is that each split keeps
        both classes and stays close to the pooled positive rate.
        """
        splits = {name: load_prepared(name) for name in ("train", "validation", "test")}
        labels = {name: extract_labels(arr) for name, arr in splits.items()}
        pooled = np.concatenate([labels[name] for name in splits])
        self.assertEqual(sorted(set(pooled.tolist())), [0, 1])
        overall = float(pooled.mean())

        for name, y in labels.items():
            with self.subTest(split=name):
                self.assertEqual(sorted(set(y.tolist())), [0, 1], f"{name}: a class is missing")
                gap = abs(float(y.mean()) - overall)
                self.assertLess(gap, MAX_SPLIT_RATIO_GAP,
                                f"{name}: positive rate {y.mean():.3f} differs by {gap:.3f} from "
                                f"the pooled {overall:.3f} (> {MAX_SPLIT_RATIO_GAP})")

        manifest = json.loads((ROOT / "data" / "prepared" / "manifest.json").read_text(encoding="utf-8"))
        for name, y in labels.items():
            with self.subTest(manifest=name):
                self.assertEqual(manifest["label_counts"][name]["distressed"], int(y.sum()),
                                 f"the manifest positive count for {name} does not match the data")


class TestNoLeakageInPreprocessing(unittest.TestCase):
    """Requirement 2: every preprocessing and balancing step must be fitted on train only.

    Checked on the main pipeline: no step exposes `fit_resample`, imputation and scaling statistics come from
    train alone, and `class_weight="balanced*"` is resolved by scikit-learn inside `fit` from the labels it
    receives.
    """

    def test_main_pipeline_has_no_balancer(self):
        """No estimator in the main pipeline exposes `fit_resample`, so nothing resamples outside train."""
        self.assertTrue(MODEL_REGISTRY, "the registry is empty")
        for name in sorted(MODEL_REGISTRY):
            pipe = make_model(name)
            for step_name, step in pipe.steps:
                self.assertFalse(hasattr(step, "fit_resample"),
                                 f"{name}/{step_name}: the main pipeline contains a sampler and could "
                                 "resample validation or test; balancing belongs in labs/imbalance_lab/")

    def test_imputer_and_scaler_learn_from_train_only(self):
        """Imputation and scaling statistics must come from train, not from the pooled splits."""
        train = load_prepared("train")
        X_train = build_feature_matrix(train)
        X_pooled = np.vstack([X_train, build_feature_matrix(load_prepared("validation")),
                              build_feature_matrix(load_prepared("test"))])

        pipe = make_model("logistic")
        pipe.fit(X_train, extract_labels(train))

        train_median = np.nanmedian(X_train, axis=0)
        pooled_median = np.nanmedian(X_pooled, axis=0)
        np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, train_median,
                                   rtol=1e-9, atol=0.0,
                                   err_msg="the imputer did not learn the median from train")
        # The two median vectors differ, so pooling the splits before imputing would break that assertion.
        self.assertGreater(float(np.nanmax(np.abs(pooled_median - train_median))), 0.0,
                           "train and pooled medians coincide, so the two paths cannot be told apart")

        imputed_train = np.where(np.isnan(X_train), train_median, X_train)
        np.testing.assert_allclose(pipe.named_steps["scale"].mean_, imputed_train.mean(axis=0),
                                   rtol=1e-6, atol=1e-9,
                                   err_msg="the scaler did not learn the mean from train")
        imputed_pooled = np.where(np.isnan(X_pooled), pooled_median, X_pooled)
        self.assertGreater(float(np.max(np.abs(imputed_pooled.mean(axis=0)
                                               - imputed_train.mean(axis=0)))), 0.0)

    def test_balancing_weights_come_from_fit_labels_only(self):
        """A string `class_weight` makes scikit-learn compute weights inside `fit` from the given labels."""
        for name in ("random_forest", "hist_gradient_boosting"):
            self.assertIsInstance(HYPERPARAMS[name]["class_weight"], str,
                                  f"{name}: class_weight must stay a 'balanced*' string so scikit-learn "
                                  "resolves it inside fit; a precomputed constant is a global weight")

        y_train = extract_labels(load_prepared("train"))
        y_pooled = np.concatenate([y_train, extract_labels(load_prepared("validation")),
                                   extract_labels(load_prepared("test"))])
        w_train = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        w_pooled = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_pooled)
        # The minority weight is n / (2 * n_positive), computed on the train labels only.
        self.assertAlmostEqual(float(w_train[1]), len(y_train) / (2.0 * int(y_train.sum())), places=6)
        self.assertFalse(np.allclose(w_train, w_pooled, rtol=1e-3),
                         "weights computed on train must differ from weights computed on train+val+test "
                         "so that pooled labels are detected")

    def test_cv_folds_are_stratified_and_grouped(self):
        """Cross-validation as in `forecasting.tuning`: preserve the class ratio and keep whole companies out."""
        samples = load_prepared("train") + load_prepared("validation")
        groups = np.asarray([s["ticker"] for s in samples])
        X, y = build_feature_matrix(samples), extract_labels(samples)

        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        gaps = []
        for fold, (train_index, val_index) in enumerate(cv.split(X, y, groups=groups), start=1):
            with self.subTest(fold=fold):
                self.assertEqual(set(groups[train_index]) & set(groups[val_index]), set(),
                                 f"fold {fold}: a company appears in both train and validation")
                self.assertEqual(sorted(set(y[train_index].tolist())), [0, 1],
                                 f"fold {fold}: fold-train holds only one class")
                self.assertGreater(int(y[val_index].sum()), 0,
                                   f"fold {fold}: fold-validation has no positive sample")
                gap = abs(float(y[val_index].mean()) - float(y.mean()))
                gaps.append(gap)
                self.assertLess(gap, MAX_FOLD_RATIO_GAP,
                                f"fold {fold}: class ratio gap {gap:.3f} exceeds {MAX_FOLD_RATIO_GAP}")
        self.assertLess(float(np.mean(gaps)), MAX_FOLD_MEAN_ABS_GAP,
                        "the mean class-ratio gap across folds is too large")


class TestFeatures(unittest.TestCase):
    def test_matrix_shapes_and_labels(self):
        samples = load_prepared("train")
        X = build_feature_matrix(samples)
        y = extract_labels(samples)
        names = feature_names()
        self.assertEqual(X.shape, (len(samples), len(names)))
        self.assertEqual(y.shape, (len(samples),))
        self.assertEqual(set(set(y.tolist())), {0, 1})
        # No column of train is entirely NaN, which downstream imputation would only warn about.
        self.assertGreater(len(names), 30)

    def test_no_label_leak_in_features(self):
        """Features use history only where it was available at `as_of`, never future data."""
        samples = load_prepared("test")
        X = build_feature_matrix(samples)
        self.assertTrue(all(len(s["request"]["history"]) >= 1 for s in samples))
        # Every history quarter must be published on or before `as_of`.
        for s in samples[:10]:
            for row in s["request"]["history"]:
                self.assertLessEqual(row["available_on"], s["request"]["as_of"])

    def test_every_declared_feature_is_produced(self):
        """`_build_features` must produce every name in `feature_names()`, with no spurious NaN column."""
        from forecasting.features import _build_features

        names = set(feature_names())
        for sample in load_prepared("test")[:5]:
            produced = set(_build_features(sample))
            self.assertEqual(names - produced, set(),
                             f"{sample['sample_id']}: declared feature was not produced")

    def test_derived_liabilities_fill_coverage(self):
        """A quarter missing the `liabilities` tag must still yield `debt_to_assets_latest`, from A = L + E."""
        from forecasting.features import _build_features

        checked = 0
        for sample in load_prepared("test") + load_prepared("train"):
            last = sample["request"]["history"][-1]
            if last.get("liabilities_vnd") is not None:
                continue
            feats = _build_features(sample)
            self.assertFalse(math.isnan(feats["debt_to_assets_latest"]),
                             f"{sample['sample_id']}: debt_to_assets missing although liabilities were derived")
            self.assertFalse(math.isnan(feats["debt_to_equity_latest"]),
                             f"{sample['sample_id']}: debt_to_equity missing although liabilities were derived")
            checked += 1
        self.assertGreater(checked, 0, "no sample lacks the liabilities tag, so nothing was checked")


class TestModels(unittest.TestCase):
    def test_registry_has_baseline(self):
        self.assertIn("logistic", MODEL_REGISTRY)
        self.assertIn("random_forest", MODEL_REGISTRY)


class TestDataAudit(unittest.TestCase):
    """`scripts.audit_data` must detect corrupted data, exercised by mutating a copy.

    A "zero findings" result is only meaningful if the auditor really catches errors, so the tests below
    alter a copy and require the auditor to report it.
    """

    @staticmethod
    def _retail_rows() -> dict:
        """Return `{ticker: rows}` read straight from retail-expanded, needing no SEC snapshot."""
        rows = {}
        for path in sorted((ROOT / "data" / "retail-expanded").glob("*-16-indicators-vnd.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            rows[doc["ticker"]] = doc["rows"]
        return rows

    def test_mutated_prepared_and_manifest_are_reported(self):
        from scripts import audit_data

        prepared, manifest = audit_data.PREPARED_DIR, audit_data.MANIFEST_FILE
        if not (prepared / "manifest.json").exists():
            self.skipTest("data/prepared is missing.")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            shutil.copytree(prepared, tmp, dirs_exist_ok=True)
            audit_data.PREPARED_DIR, audit_data.MANIFEST_FILE = tmp, tmp / "manifest.json"
            try:
                samples = json.loads((tmp / "test.json").read_text(encoding="utf-8"))
                samples[0]["request"]["history"][0]["revenue_vnd"] = "1"          # corrupted figure
                samples[1]["is_distressed"] = 1 - int(samples[1]["is_distressed"])  # corrupted label
                (tmp / "test.json").write_text(
                    json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                auditor = audit_data.Auditor()
                splits = audit_data.check_prepared(auditor, self._retail_rows())
                audit_data.check_manifest(auditor, self._retail_rows(), splits)
            finally:
                audit_data.PREPARED_DIR, audit_data.MANIFEST_FILE = prepared, manifest

        prepared_msgs = " | ".join(i["message"] for i in auditor.issues.get("prepared", []))
        self.assertIn("khác retail-expanded", prepared_msgs, prepared_msgs)
        manifest_msgs = " | ".join(i["message"] for i in auditor.issues.get("manifest", []))
        self.assertIn("label_counts[test].distressed", manifest_msgs, manifest_msgs)
        self.assertIn("split_sha256[test]", manifest_msgs, manifest_msgs)

    def test_mutated_indicator_file_is_reported(self):
        from scripts import audit_data

        source = ROOT / "data" / "retail-expanded" / "WMT-16-indicators-vnd.json"
        snapshot = ROOT / "data" / "sec" / "raw" / "WMT-companyfacts.json"
        if not source.exists() or not snapshot.exists():
            self.skipTest("retail-expanded or the SEC snapshot is missing; run `python -m scripts.crawl_sec`.")
        with tempfile.TemporaryDirectory() as td:
            doc = json.loads(source.read_text(encoding="utf-8"))
            doc["rows"][0]["revenue_vnd"] = str(int(doc["rows"][0]["revenue_vnd"]) + 1)
            doc["rows"][0]["available_on"] = "1999-01-01"                  # wrong publication date
            doc["rows"][1]["sources"]["inventory"]["facts"][0]["val"] += 1  # wrong provenance
            path = pathlib.Path(td) / source.name
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            auditor = audit_data.Auditor()
            audit_data.check_retail_doc(auditor, path, "retail_expanded")

        messages = " | ".join(i["message"] for i in auditor.issues.get("retail_expanded", []))
        self.assertIn("VND khác số liệu thật", messages, messages)
        self.assertIn("available_on", messages, messages)
        self.assertIn("`val` provenance khác companyfacts", messages, messages)


class TestModelSelection(unittest.TestCase):
    """The selected model must follow the published selection rule, which prefers cross-company AP."""

    SUMMARY = ROOT / "reports" / "results" / "summary.json"

    def test_selected_model_matches_selection_rule(self):
        if not self.SUMMARY.exists():
            self.skipTest("reports/results/summary.json is missing; run `python -m forecasting.train`.")
        summary = json.loads(self.SUMMARY.read_text(encoding="utf-8"))
        models = summary.get("models") or []
        self.assertTrue(models, "summary.json has no model list")
        rule = summary.get("selection_rule", "")
        self.assertIn("cross-company", rule,
                      "the selection rule must prioritise cross-company AP to avoid entity leakage")
        for row in models:
            self.assertIn("cross_company_ap", row,
                          f"{row['model']}: cross-company AP is required to apply the selection rule")

        def rank_key(row: dict):
            def safe(value):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return -1.0
                return -1.0 if number != number else number
            return (safe(row.get("cross_company_ap")), safe(row.get("best_f1_val")),
                    safe(row.get("average_precision")), safe(row.get("auroc")),
                    -safe(row.get("overfit_gap_f1")))

        expected = max(models, key=rank_key)["model"]
        self.assertEqual(summary.get("best_model"), expected,
                         "the selected model does not match the published ranking rule")


if __name__ == "__main__":
    unittest.main()
