"""Kiểm thử `imbalance_lab`: bảo đảm KHÔNG rò rỉ dữ liệu và các hàm ngưỡng tính đúng.

Chạy: python -m unittest discover -s tests -v

Nhóm test:
1. `TestDataAndSplits`     — tỉ lệ mất cân bằng 98/2 và chia tập stratified.
2. `TestResampling`        — SMOTE/undersample chỉ đổi tập train; không đụng validation.
3. `TestNoLeakage`         — pipeline resampling chỉ train trên dữ liệu đã resample CỦA FOLD;
                             `scale_pos_weight` chỉ tính từ nhãn của fold.
4. `TestThresholds`        — hàm ngưỡng khớp cách tính bằng sklearn/brute-force.
5. `TestRunSmoke`          — chạy end-to-end cỡ nhỏ (bỏ qua nếu thiếu lightgbm).
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbalance_lab import config as C  # noqa: E402
from imbalance_lab.data import (label_distribution, make_imbalanced_dataset,  # noqa: E402
                                stratified_holdout_split)
from imbalance_lab.metrics import metrics_at_threshold  # noqa: E402
from imbalance_lab.models import ScalePosWeightClassifier  # noqa: E402
from imbalance_lab.samplers import (SMOTE, RandomUnderSampler,  # noqa: E402
                                    build_sampler_pipeline, make_samplers, resampling_backend)
from imbalance_lab.thresholds import (best_cost_threshold, best_f1_threshold,  # noqa: E402
                                      scan_counts, threshold_for_precision)

try:
    import lightgbm  # noqa: F401
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - môi trường không có lightgbm
    HAS_LIGHTGBM = False


class RecordingClassifier:
    """Classifier giả: ghi lại số mẫu và phân phối nhãn nó NHẬN ĐƯỢC khi `fit`."""

    def __init__(self) -> None:
        self.n_seen: int = -1
        self.n_positive_seen: int = -1
        self.positive_rate_: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecordingClassifier":
        self.n_seen = int(len(y))
        self.n_positive_seen = int((np.asarray(y) == 1).sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        positive = np.full(len(X), float(self.positive_rate_))
        return np.column_stack([1.0 - positive, positive])


class TestDataAndSplits(unittest.TestCase):
    def test_dataset_has_98_2_imbalance(self):
        X, y = make_imbalanced_dataset(n_samples=5000, random_state=0)
        dist = label_distribution(y)
        self.assertEqual(X.shape, (5000, C.N_FEATURES))
        self.assertAlmostEqual(dist["positive_pct"], 2.0, delta=0.4)
        self.assertGreater(dist["imbalance_ratio"], 20.0)

    def test_stratified_split_preserves_ratio(self):
        X, y = make_imbalanced_dataset(n_samples=5000, random_state=1)
        split = stratified_holdout_split(X, y, test_size=0.2, seed=1)
        train_pct = 100.0 * (split["y_train"] == 1).mean()
        test_pct = 100.0 * (split["y_test"] == 1).mean()
        self.assertAlmostEqual(train_pct, 2.0, delta=0.5)
        self.assertAlmostEqual(test_pct, 2.0, delta=0.5)
        self.assertEqual(len(split["y_train"]) + len(split["y_test"]), 5000)


class TestResampling(unittest.TestCase):
    def test_smote_and_undersampler_reach_target_ratio(self):
        X, y = make_imbalanced_dataset(n_samples=2000, random_state=2)
        X_res, y_res = SMOTE(sampling_strategy=0.1, k_neighbors=5, random_state=0).fit_resample(X, y)
        after_smote = label_distribution(y_res)
        self.assertAlmostEqual(after_smote["n_positive"] / after_smote["n_negative"], 0.1, delta=0.02)
        X_under, y_under = RandomUnderSampler(sampling_strategy=0.5, random_state=0).fit_resample(X_res, y_res)
        after_under = label_distribution(y_under)
        self.assertAlmostEqual(after_under["n_positive"] / after_under["n_negative"], 0.5, delta=0.05)
        self.assertLess(after_under["n"], after_smote["n"])

    def test_samplers_do_not_touch_inputs(self):
        X, y = make_imbalanced_dataset(n_samples=1000, random_state=3)
        X_before, y_before = X.copy(), y.copy()
        SMOTE(sampling_strategy=0.2, k_neighbors=3, random_state=0).fit_resample(X, y)
        self.assertTrue(np.array_equal(X, X_before))
        self.assertTrue(np.array_equal(y, y_before))

    def test_backend_reported(self):
        self.assertIn(resampling_backend(True), ("imblearn", "builtin"))


class TestNoLeakage(unittest.TestCase):
    """Các test cốt lõi: resampling KHÔNG được chạm vào validation/test."""

    def test_pipeline_resamples_only_fold_train_and_val_untouched(self):
        X, y = make_imbalanced_dataset(n_samples=4000, random_state=4)
        split = stratified_holdout_split(X, y, test_size=0.25, seed=4)
        X_tr, y_tr = split["X_train"], split["y_train"]
        X_va, y_va = split["X_test"], split["y_test"]
        X_va_before, y_va_before = X_va.copy(), y_va.copy()

        recorder = RecordingClassifier()
        samplers = make_samplers(True, smote_strategy=0.1, k_neighbors=5, under_strategy=0.5,
                                 random_state=0)
        pipeline = build_sampler_pipeline(samplers, recorder, prefer_imblearn=True)
        pipeline.fit(X_tr, y_tr)
        pipeline.predict_proba(X_va)  # suy luận trên validation

        # (1) classifier chỉ thấy dữ liệu ĐÃ RESAMPLE của fold train
        expected = label_distribution(y_tr)
        self.assertNotEqual(recorder.n_seen, expected["n"])          # đã resample
        self.assertLess(recorder.n_seen, expected["n"])              # và nhỏ hơn (undersample)
        self.assertGreater(recorder.n_positive_seen, expected["n_positive"])  # SMOTE sinh thêm dương
        # (2) validation KHÔNG bị đổi một byte nào
        self.assertTrue(np.array_equal(X_va, X_va_before))
        self.assertTrue(np.array_equal(y_va, y_va_before))
        # (3) tỉ lệ nhãn validation vẫn đúng như dữ liệu gốc (2%)
        self.assertAlmostEqual(100.0 * (y_va == 1).mean(), 2.0, delta=1.0)

    def test_scale_pos_weight_uses_only_given_fold_labels(self):
        """Trọng số phải suy từ nhãn được truyền vào `fit`, không phải từ hằng số toàn cục."""
        model = ScalePosWeightClassifier(random_state=0)
        y_balanced = np.array([0] * 50 + [1] * 50)
        model.fit(np.zeros((100, 3)), y_balanced)
        self.assertAlmostEqual(model.scale_pos_weight_, 1.0, places=6)

        model_b = ScalePosWeightClassifier(random_state=0)
        y_imbalanced = np.array([0] * 90 + [1] * 10)
        model_b.fit(np.zeros((100, 3)), y_imbalanced)
        self.assertAlmostEqual(model_b.scale_pos_weight_, 9.0, places=6)


class TestThresholds(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = (rng.random(400) < 0.05).astype(int)
        self.proba = np.clip(0.2 * self.y + rng.random(400), 0, 1)

    def test_scan_counts_matches_sklearn(self):
        thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        counts = scan_counts(self.y, self.proba, thresholds)
        from sklearn.metrics import confusion_matrix
        for index, threshold in enumerate(thresholds):
            tn, fp, fn, tp = confusion_matrix(self.y, (self.proba >= threshold).astype(int),
                                              labels=[0, 1]).ravel()
            self.assertEqual(int(counts["tp"][index]), tp)
            self.assertEqual(int(counts["fp"][index]), fp)
            self.assertEqual(int(counts["fn"][index]), fn)

    def test_best_f1_threshold_matches_brute_force(self):
        from sklearn.metrics import f1_score
        grid = np.linspace(0.0, 1.0, 201)
        brute = max(grid, key=lambda t: f1_score(self.y, (self.proba >= t).astype(int), zero_division=0))
        tuned = best_f1_threshold(self.y, self.proba)
        brute_f1 = f1_score(self.y, (self.proba >= brute).astype(int), zero_division=0)
        self.assertAlmostEqual(tuned["f1"], brute_f1, places=6)
        self.assertLessEqual(tuned["threshold"], brute + 0.05)

    def test_cost_threshold_favours_recall_when_fn_is_expensive(self):
        cheap = best_cost_threshold(self.y, self.proba, cost_fn=1.0, cost_fp=1.0)
        expensive = best_cost_threshold(self.y, self.proba, cost_fn=50.0, cost_fp=1.0)
        self.assertLessEqual(expensive["threshold"], cheap["threshold"])
        self.assertGreaterEqual(expensive["recall"], cheap["recall"])

    def test_threshold_for_precision_meets_target(self):
        result = threshold_for_precision(self.y, self.proba, target_precision=0.5)
        self.assertTrue(result["target_met"])
        self.assertGreaterEqual(result["precision"], 0.5)

    def test_metrics_at_threshold_basic(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = metrics_at_threshold(y, proba, 0.5)
        self.assertAlmostEqual(metrics["f1"], 1.0, places=6)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (2, 0, 0))
        self.assertAlmostEqual(metrics["pr_auc"], 1.0, places=6)


class TestRunSmoke(unittest.TestCase):
    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm cho test end-to-end")
    def test_run_returns_all_strategies(self):
        from imbalance_lab.run import run
        result = run(n_samples=3000, include_leaky=False, write=False)
        self.assertEqual(result["strategies"],
                         ["baseline", "cost_sensitive", "resampling", "resampling_calibrated"])
        self.assertEqual(len(result["strategy_rows"]), 4 * 4)  # 4 chiến lược × 4 chế độ ngưỡng
        for row in result["strategy_rows"]:
            self.assertTrue(0.0 <= row["pr_auc"] <= 1.0)
            self.assertTrue(0.0 <= row["brier"] <= 1.0)
        resample_rows = [r for r in result["resample_rows"] if r["strategy"] == "resampling"]
        self.assertTrue(resample_rows)
        self.assertGreater(resample_rows[0]["train_ir_before"], resample_rows[0]["train_ir_after"])

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm cho test hiệu chuẩn")
    def test_calibrated_strategy_produces_valid_probabilities(self):
        """Biến thể hiệu chuẩn phải chạy được và cho xác suất hợp lệ (không rò rỉ: test riêng)."""
        from imbalance_lab.data import make_imbalanced_dataset, stratified_holdout_split
        from imbalance_lab.metrics import metrics_at_threshold
        from imbalance_lab.models import build_strategy

        X, y = make_imbalanced_dataset(n_samples=2000, random_state=11)
        split = stratified_holdout_split(X, y, test_size=0.25, seed=11)
        strategy = build_strategy("resampling_calibrated", random_state=11)
        model = strategy["factory"]()
        model.fit(split["X_train"], split["y_train"])
        proba = model.predict_proba(split["X_test"])[:, 1]
        self.assertTrue(((proba >= 0) & (proba <= 1)).all(), "xác suất phải nằm trong [0, 1]")
        metrics = metrics_at_threshold(split["y_test"], proba, 0.5)
        self.assertIn("brier", metrics)
        self.assertTrue(0.0 <= metrics["brier"] <= 1.0)


if __name__ == "__main__":
    unittest.main()

