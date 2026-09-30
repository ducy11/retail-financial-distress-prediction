"""Kiểm thử `benchmark_imbalanced.py`: dữ liệu 95/5, bảng Markdown và CHỐNG RÒ RỈ dữ liệu.

Chạy: python -m unittest discover -s tests -v

Nhóm test:
1. `TestDataset`            — tỉ lệ mất cân bằng ~95/5 và chia stratified giữ nguyên tỉ lệ.
2. `TestNoLeakage`          — resampling chỉ trên train; test nguyên vẹn; test âm tính (estimator
                             cố tình sửa test) PHẢI bị đánh dấu FAIL.
3. `TestRusBoostFallback`   — bản RUSBoost nội bộ cho xác suất hợp lệ và dùng được qua wrapper.
4. `TestReporting`          — bảng Markdown đủ cột/dòng và thống kê nhóm đúng.
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import benchmark_imbalanced as B  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - `unittest` hook
    """Lọc lại cảnh báo thư viện vô hại sau khi `unittest` đặt `simplefilter("default")`.

    Test fit `LogisticRegression` nhiều lần ⇒ không lọc sẽ in `OptimizeWarning: Unknown solver
    options: iprint` (scikit-learn 1.6 + scipy 1.18) lẫn vào output test. Xem `runtime_warnings.py`.
    """
    quiet_library_warnings()


def _small_split(n_samples: int = 1500, seed: int = B.SEED):
    X, y = B.make_dataset(n_samples=n_samples, seed=seed)
    return train_test_split(X, y, test_size=B.TEST_SIZE, stratify=y, random_state=seed)


def _method(name: str, estimator, samplers=None, group: str = "Non-E Mode"):
    """Tạo đối tượng `method` đúng định dạng mà `benchmark_imbalanced` yêu cầu."""
    if samplers:
        factory = (lambda: B.ImbPipeline([*samplers,
                                          ("classifier", B.RecordingClassifier(estimator))]))
    else:
        factory = (lambda: B.RecordingClassifier(estimator))
    return {"group": group, "name": name, "kind": "data-level", "estimator": estimator,
            "samplers": samplers, "note": "", "factory": factory,
            "is_resampling": bool(samplers)}


class TestDataset(unittest.TestCase):
    def test_dataset_is_about_95_5(self):
        X, y = B.make_dataset(n_samples=3000, seed=B.SEED)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(set(np.unique(y)), {0, 1})
        dist = B._distribution(y)
        self.assertLess(abs(dist["positive_pct"] - 5.0), 1.5)
        self.assertGreater(dist["imbalance_ratio"], 9.0)

    def test_split_is_stratified_and_deterministic(self):
        X_train, X_test, y_train, y_test = _small_split()
        self.assertEqual(len(y_train) + len(y_test), 1500)
        ratio_gap = abs(B._distribution(y_train)["positive_pct"]
                        - B._distribution(y_test)["positive_pct"])
        self.assertLess(ratio_gap, 2.0)
        again = _small_split()
        self.assertTrue(np.array_equal(y_test, again[3]))


class TestNoLeakage(unittest.TestCase):
    def test_resampling_only_changes_train(self):
        """SMOTE trong pipeline: classifier thấy tập train ĐÃ resample, test không bị đụng."""
        X_train, X_test, y_train, y_test = _small_split()
        X_test_before, y_test_before = X_test.copy(), y_test.copy()
        method = _method("SMOTE + Logistic Regression",
                         LogisticRegression(max_iter=500, random_state=B.SEED),
                         samplers=[("smote", SMOTE(sampling_strategy=0.5,
                                                   random_state=B.SEED))])
        row = B.evaluate_method(method, X_train, y_train, X_test, y_test)
        self.assertTrue(row["leakage_ok"], row["leakage_detail"])
        self.assertGreater(row["n_train_after"], row["n_train_before"])
        self.assertLess(row["ir_after"], row["ir_before"])
        self.assertTrue(np.array_equal(X_test, X_test_before))
        self.assertTrue(np.array_equal(y_test, y_test_before))

    def test_baseline_sees_full_train(self):
        X_train, X_test, y_train, y_test = _small_split()
        method = _method("Logistic Regression", LogisticRegression(max_iter=500),
                         group="Baseline")
        row = B.evaluate_method(method, X_train, y_train, X_test, y_test)
        self.assertTrue(row["leakage_ok"])
        self.assertEqual(row["n_train_after"], row["n_train_before"])
        self.assertEqual(row["n_train_before"], len(y_train))

    def test_leaky_estimator_is_flagged(self):
        """Test ÂM TÍNH: estimator cố tình sửa tập test trong `fit` phải bị đánh dấu FAIL."""
        X_train, X_test, y_train, y_test = _small_split()
        target = X_test

        class LeakyEstimator:
            def fit(self, X, y):
                target[:] = 0.0           # hành vi rò rỉ giả lập
                self.positive_rate_ = float(np.mean(y == 1))
                return self

            def predict_proba(self, X):
                positive = np.full(len(X), self.positive_rate_)
                return np.column_stack([1.0 - positive, positive])

        method = _method("Leaky (giả lập)", LeakyEstimator())
        row = B.evaluate_method(method, X_train, y_train, X_test, y_test)
        self.assertFalse(row["leakage_ok"])
        self.assertFalse(row["leakage_detail"]["test_nguyên_vẹn"])


class TestRusBoostFallback(unittest.TestCase):
    def test_internal_rus_boost_probabilities(self):
        X, y = B.make_dataset(n_samples=800, seed=7)
        clf = B.RUSBoostInternal(n_estimators=15, max_depth=3, random_state=7).fit(X, y)
        proba = clf.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
        self.assertGreaterEqual(len(clf.alphas_), 1)
        self.assertLessEqual(len(clf.alphas_), 15)
        self.assertTrue(all(alpha > 0 for alpha in clf.alphas_))
        self.assertEqual(len(clf.alphas_) + clf.n_skipped_, 15)

    def test_robust_wrapper_fits_and_predicts(self):
        X, y = B.make_dataset(n_samples=800, seed=9)
        model = B.RUSBoostRobust(seed=9).fit(X, y)
        self.assertTrue(hasattr(model, "backend_"))
        self.assertEqual(len(model.predict(X)), len(y))


class TestReporting(unittest.TestCase):
    def test_markdown_and_group_summary_contain_expected_content(self):
        X_train, X_test, y_train, y_test = _small_split(n_samples=800, seed=3)
        rows = [
            B.evaluate_method(_method("Logistic Regression",
                                      LogisticRegression(max_iter=400), group="Baseline"),
                              X_train, y_train, X_test, y_test),
            B.evaluate_method(_method("SMOTE + Logistic Regression",
                                      LogisticRegression(max_iter=400),
                                      samplers=[("smote", SMOTE(sampling_strategy=0.5,
                                                                 random_state=3))]),
                              X_train, y_train, X_test, y_test),
        ]
        table = B.render_markdown(rows)
        for column in ("group", "method", "roc_auc", "pr_auc", "f1_minority",
                       "balanced_accuracy", "train_time_s"):
            self.assertIn(column, table)
        for row in rows:
            self.assertIn(row["method"], table)
        summary = B.render_group_summary(rows)
        self.assertIn("Baseline", summary)
        self.assertIn("Non-E Mode", summary)
        self.assertIn("Tốt nhất theo từng chỉ số", summary)


class TestStratifiedCV(unittest.TestCase):
    """Yêu cầu: chia fold giữ tỉ lệ lớp; cân bằng CHỈ trên fold-train, không bao giờ trên fold-val."""

    def test_folds_keep_class_ratio_and_cover_all_samples(self):
        X, y = B.make_dataset(n_samples=1500, seed=B.SEED)
        folds = B.stratified_folds(X, y, n_splits=5, seed=B.SEED)
        self.assertEqual(len(folds), 5)
        global_pct = B._distribution(y)["positive_pct"]
        seen: List[int] = []
        for train_index, val_index in folds:
            self.assertEqual(len(train_index) + len(val_index), len(y))
            self.assertFalse(set(train_index) & set(val_index))       # train ∩ val = ∅
            self.assertLess(abs(B._distribution(y[val_index])["positive_pct"] - global_pct), 1.5)
            seen.extend(val_index.tolist())
        self.assertEqual(sorted(seen), list(range(len(y))))           # phủ đúng 1 lần toàn bộ mẫu

    def test_cv_resamples_only_fold_train(self):
        X, y = B.make_dataset(n_samples=1200, seed=B.SEED)
        method = _method("SMOTE + Logistic Regression",
                         LogisticRegression(max_iter=400, random_state=B.SEED),
                         samplers=[("smote", SMOTE(sampling_strategy=0.5, random_state=B.SEED))])
        summary = B.evaluate_cv(method, X, y, n_splits=3, seed=B.SEED)
        self.assertEqual(summary["n_splits"], 3)
        self.assertTrue(summary["leakage_ok"])
        self.assertEqual(summary["n_folds_pass"], 3)
        for fold in summary["folds"]:
            self.assertGreater(fold["n_train_after"], fold["n_train"])   # SMOTE chỉ trên fold-train
            self.assertLess(fold["ir_after"], fold["ir_before"])
            self.assertTrue(all(fold["checks"].values()), fold["checks"])
        self.assertGreater(summary["pr_auc_mean"], 0.0)

    def test_cv_without_resampling_keeps_fold_train_size(self):
        X, y = B.make_dataset(n_samples=1200, seed=B.SEED)
        method = _method("Logistic Regression", LogisticRegression(max_iter=400), group="Baseline")
        summary = B.evaluate_cv(method, X, y, n_splits=3, seed=B.SEED)
        self.assertTrue(summary["leakage_ok"])
        for fold in summary["folds"]:
            self.assertEqual(fold["n_train_after"], fold["n_train"])


if __name__ == "__main__":
    unittest.main()


