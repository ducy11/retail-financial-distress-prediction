"""Kiểm thử DANH MỤC KỸ THUẬT mất cân bằng (yêu cầu #2): đủ mục, chống rò rỉ, ngưỡng PR, Focal Loss.

Chạy: python -m unittest discover -s tests -v

Nhóm test:
1. `TestCatalogCoverage`    — danh mục phủ ĐỦ mọi nhóm trong yêu cầu (kể cả khi ai đó sửa code);
                              mọi kỹ thuật dựng được estimator/factory.
2. `TestSamplerTechniques`  — tỉ lệ mục tiêu của từng sampler (cả bản nội bộ lẫn imblearn);
                              Tomek/ENN chỉ LÀM SẠCH biên (không đổi số mẫu thiểu số); không sửa input.
3. `TestFocalLoss`          — grad/hess giải tích khớp sai phân số; `gamma=0` thoái hoá về weighted
                              BCE; hessian > 0; huấn luyện cho xác suất hợp lệ; `alpha=None` tính động.
4. `TestPRThresholds`       — ngưỡng lấy từ ĐƯỜNG PR: là điểm thật của đường cong, PR-AUC khớp
                              `average_precision_score`, F1 ≥ F1@0.5, best_cost tối thiểu chi phí.
5. `TestNoLeakage`          — resampling chỉ trên fold-train; validation/test nguyên vẹn;
                              ngưỡng chọn lại từ OOF cho ĐÚNG giá trị đã dùng (không lấy từ test).
6. `TestCatalogRunSmoke`    — chạy end-to-end cỡ nhỏ + ghi artifact vào thư mục tạm.
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

from imbalance_lab import config as C  # noqa: E402
from imbalance_lab.losses import FocalLossClassifier, focal_grad_hess, focal_loss_value  # noqa: E402
from imbalance_lab.models import BalancedWeightClassifier  # noqa: E402
from imbalance_lab.samplers import (SamplerChain, make_hybrid_sampler,  # noqa: E402
                                    make_single_sampler, resampling_backend)
from imbalance_lab.techniques import (BASELINE_TECHNIQUE, IMPLEMENTATION,  # noqa: E402
                                      REFERENCE_GROUPS, REQUIRED_GROUPS, TECHNIQUE_ORDER,
                                      _markdown, build_technique, run, technique_group)
from imbalance_lab.thresholds import (pr_curve_points, tune_thresholds,  # noqa: E402
                                      tune_thresholds_from_pr_curve)

try:
    import lightgbm  # noqa: F401

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - môi trường thiếu lightgbm
    HAS_LIGHTGBM = False

#: Kỳ vọng ĐÚNG theo danh sách yêu cầu (không đọc từ code ⇒ test bắt được việc bỏ sót kỹ thuật).
EXPECTED_GROUPS: Dict[str, set] = {
    "data-level/oversampling": {"ros", "smote", "borderline_smote", "adasyn"},
    "data-level/undersampling": {"rus", "tomek", "enn"},
    "hybrid": {"smote_tomek", "smote_enn"},
    "algorithm-level": {"cost_sensitive_scale_pos_weight", "cost_sensitive_class_weight",
                        "focal_loss"},
    "ensemble": {"balanced_rf", "easy_ensemble", "rusboost"},
}


def tiny_dataset(n_samples: int = 1500, weights=(0.95, 0.05), seed: int = 0):
    """Dữ liệu nhỏ, mất cân bằng, tất định — dùng cho mọi test dưới đây."""
    X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=6, n_classes=2,
                               weights=list(weights), flip_y=0.0, random_state=seed)
    return X.astype(float), y.astype(int)


def _ratio(y: np.ndarray) -> float:
    """Tỉ lệ thiểu/đa (đúng nghĩa `sampling_strategy` của imblearn)."""
    y = np.asarray(y)
    return float((y == 1).sum()) / max(int((y == 0).sum()), 1)


class TestCatalogCoverage(unittest.TestCase):
    """Danh mục phải phủ ĐỦ 5 nhóm và từng kỹ thuật mà yêu cầu #2 liệt kê."""

    def test_required_groups_and_techniques_present(self):
        self.assertEqual(set(REQUIRED_GROUPS), set(EXPECTED_GROUPS))
        for group, expected in EXPECTED_GROUPS.items():
            with self.subTest(group=group):
                self.assertTrue(expected.issubset(set(REQUIRED_GROUPS[group])),
                                f"{group}: thiếu {sorted(expected - set(REQUIRED_GROUPS[group]))}")

    def test_every_technique_has_doc_and_implementation(self):
        known_groups = set(EXPECTED_GROUPS) | set(REFERENCE_GROUPS)
        for key in TECHNIQUE_ORDER:
            with self.subTest(technique=key):
                spec = build_technique(key, quick=True)
                self.assertTrue(spec["doc"], f"{key}: thiếu mô tả")
                self.assertTrue(IMPLEMENTATION.get(key), f"{key}: thiếu mục 'cài đặt' trong IMPLEMENTATION")
                self.assertTrue(callable(spec["factory"]))
                self.assertIsNotNone(spec["estimator"], f"{key}: không dựng được estimator")
                self.assertIn(technique_group(key), known_groups)

    def test_baseline_is_present_and_runs_first(self):
        """Yêu cầu #3 cần mốc BASELINE (chưa xử lý) để so sánh ⇒ phải có và chạy trước tiên."""
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
                                         f"{key}: thiếu probe_factory để log phân phối sau resample")

    def test_hybrid_group_uses_two_steps(self):
        for key in REQUIRED_GROUPS["hybrid"]:
            with self.subTest(technique=key):
                steps = make_hybrid_sampler(key, prefer_imblearn=True,
                                            over_strategy=0.5, under_strategy=0.5,
                                            k_neighbors=5, random_state=0)
                self.assertEqual(len(steps), 1)
                self.assertTrue(hasattr(steps[0][1], "fit_resample"))



class TestSamplerTechniques(unittest.TestCase):
    """Mỗi sampler phải đạt tỉ lệ mục tiêu, chỉ LÀM SẠCH đúng chỗ, và không sửa dữ liệu đầu vào."""

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
        """RandomOverSampler / SMOTE / BorderlineSMOTE / ADASYN đạt ~50% số mẫu đa số (cả 2 backend)."""
        for prefer_imblearn in (True, False):
            for kind in ("ros", "smote", "borderline_smote", "adasyn"):
                with self.subTest(backend="imblearn" if prefer_imblearn else "builtin", kind=kind):
                    steps = make_single_sampler(kind, prefer_imblearn=prefer_imblearn,
                                                over_strategy=C.TECHNIQUE_OVER_STRATEGY,
                                                under_strategy=C.TECHNIQUE_UNDER_STRATEGY,
                                                k_neighbors=C.TECHNIQUE_K_NEIGHBORS,
                                                random_state=0)
                    _Xr, yr = self._resample(steps)
                    self.assertGreater(len(yr), len(self.y), f"{kind}: phải sinh thêm mẫu")
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
        """Tomek/ENN là 'clean-sampling': bỏ mẫu ĐA SỐ ở biên, KHÔNG đổi số mẫu thiểu số."""
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
                                     f"{kind}: không được bỏ mẫu thiểu số")
                    self.assertLessEqual(len(yr), len(self.y), f"{kind}: chỉ được BỎ mẫu")

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
                                f"{kind}: bước làm sạch phải bỏ bớt mẫu so với SMOTE thuần")

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
    """Focal loss TỪNG MẪU tính từ logit (để kiểm chứng grad/hess bằng sai phân số)."""
    p = np.clip(_sigmoid(z), 1e-12, 1.0 - 1e-12)
    return -(y * alpha * (1.0 - p) ** gamma * np.log(p)
             + (1.0 - y) * (1.0 - alpha) * p ** gamma * np.log1p(-p))


class TestFocalLoss(unittest.TestCase):
    """Focal Loss (custom objective LightGBM): công thức phải ĐÚNG, không chỉ 'chạy được'."""

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
                                   msg=f"gradient sai ở mẫu {i}")

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
                                   msg=f"hessian sai ở mẫu {i}")

    def test_gamma_zero_reduces_to_weighted_bce(self):
        grad, hess = focal_grad_hess(self.y, self.z, 0.0, self.ALPHA)
        p = _sigmoid(self.z)
        expected_grad = np.where(self.y > 0.5, self.ALPHA * (p - 1.0), (1.0 - self.ALPHA) * p)
        expected_hess = np.where(self.y > 0.5, self.ALPHA * p * (1.0 - p),
                                 (1.0 - self.ALPHA) * p * (1.0 - p))
        np.testing.assert_allclose(grad, expected_grad, rtol=0, atol=1e-9)
        np.testing.assert_allclose(hess, expected_hess, rtol=0, atol=1e-9)

    def test_hessian_is_strictly_positive(self):
        """LightGBM yêu cầu hessian > 0 — điều kiện để custom objective không phân kỳ."""
        extreme = np.concatenate([self.z, np.array([-30.0, -5.0, 0.0, 5.0, 30.0])])
        y = np.concatenate([self.y, np.array([1.0, 0.0, 1.0, 0.0, 1.0])])
        _grad, hess = focal_grad_hess(y, extreme, self.GAMMA, self.ALPHA)
        self.assertTrue(bool(np.all(hess > 0.0)), "có hessian ≤ 0 ⇒ LightGBM sẽ lỗi")

    def test_loss_value_prefers_confident_correct_predictions(self):
        good = focal_loss_value(np.array([1, 0]), np.array([0.99, 0.01]), self.GAMMA, self.ALPHA)
        bad = focal_loss_value(np.array([1, 0]), np.array([0.01, 0.99]), self.GAMMA, self.ALPHA)
        self.assertLess(good, bad)

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm cho FocalLossClassifier")
    def test_classifier_probabilities_and_dynamic_alpha(self):
        X, y = tiny_dataset(n_samples=1200, seed=3)
        model = FocalLossClassifier(gamma=2.0, alpha=None, n_estimators=60,
                                    learning_rate=0.1, random_state=0)
        model.fit(X, y)
        expected_alpha = float((y == 0).sum()) / len(y)          # alpha = n_âm/n
        self.assertAlmostEqual(model.alpha_, expected_alpha, places=9)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(y), 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-9)
        self.assertTrue(bool(((proba >= 0.0) & (proba <= 1.0)).all()))
        self.assertEqual(set(model.predict(X).tolist()), {0, 1})

    def test_balanced_class_weight_uses_fit_labels(self):
        """`class_weight='balanced'` phải suy trọng số từ nhãn NHẬN ĐƯỢC khi fit (không rò rỉ)."""
        X, y = tiny_dataset(n_samples=600, weights=(0.8, 0.2), seed=5)
        model = BalancedWeightClassifier(random_state=0).fit(X, y)
        from sklearn.utils.class_weight import compute_class_weight

        expected = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        self.assertAlmostEqual(model.class_weight_[0], float(expected[0]), places=9)
        self.assertAlmostEqual(model.class_weight_[1], float(expected[1]), places=9)



class TestPRThresholds(unittest.TestCase):
    """Threshold tuning theo ĐƯỜNG PR (thay vì 0.5) phải lấy ứng viên từ chính đường cong."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        n = 1500
        cls.y = (rng.random(n) < 0.08).astype(int)
        # Xác suất "có tín hiệu": lớp dương lệch lên rõ rệt nhưng vẫn chồng lấn.
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
                        "ngưỡng tốt nhất phải là một ĐIỂM của đường PR (hoặc 0.5/1.0)")

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
        # Và ngưỡng theo PR phải không tệ hơn mốc 0.5 về F1.
        self.assertGreaterEqual(tuned["best_f1"]["f1"] + 1e-9, tuned["fixed_0.5"]["f1"])

    def test_two_implementations_agree_on_best_f1(self):
        """Ứng viên đường PR là ĐIỂM THẬT của đường cong ⇒ không kém bản lưới lượng tử.

        Lưới lượng tử (1001 điểm) có thể BỎ SÓT đúng giá trị xác suất đạt F1 cao nhất, nên F1 của bản
        trên đường PR phải ≥ bản lưới (và chênh không đáng kể).
        """
        on_curve = tune_thresholds_from_pr_curve(self.y, self.proba, cost_fn=C.COST_FN,
                                                 cost_fp=C.COST_FP, precision_target=0.3)
        on_grid = tune_thresholds(self.y, self.proba, cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                                  precision_target=0.3)
        self.assertGreaterEqual(on_curve["best_f1"]["f1"] + 1e-9, on_grid["best_f1"]["f1"])
        self.assertLessEqual(on_curve["best_f1"]["f1"] - on_grid["best_f1"]["f1"], 0.02)
        # `min_precision` hai bản phải cùng thoả/không thoả mục tiêu precision.
        self.assertEqual(on_curve["min_precision"]["target_met"],
                         on_grid["min_precision"]["target_met"])



class TestNoLeakageInCatalog(unittest.TestCase):
    """Yêu cầu #2 (phần chống rò rỉ): resampling chỉ trên fold-train; ngưỡng chọn trên OOF."""

    n_samples = 2000
    n_splits = 3

    def _evaluate(self, key: str):
        from imbalance_lab.data import make_imbalanced_dataset, stratified_holdout_split
        from imbalance_lab.techniques import evaluate_technique

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
                                           msg="fold-validation bị đổi phân phối nhãn")
                    self.assertLess(row["train_ir_after"], row["train_ir_before"])

    def test_thresholds_are_recomputed_from_oof_only(self):
        """Ngưỡng đã dùng phải bằng ĐÚNG giá trị suy lại từ xác suất out-of-fold (không từ test)."""
        split, item = self._evaluate("ros")
        recomputed = tune_thresholds_from_pr_curve(item["oof_y"], item["oof_proba"],
                                                   cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                                                   precision_target=C.PRECISION_TARGET)
        for mode, value in item["thresholds"].items():
            with self.subTest(mode=mode):
                self.assertAlmostEqual(value, float(recomputed[mode]["threshold"]), places=12)
        # OOF là của train_pool, KHÔNG phải test:
        self.assertEqual(len(item["oof_proba"]), len(split["y_train"]))
        self.assertAlmostEqual(float(np.mean(item["oof_y"])),
                               float(np.mean(split["y_train"])), places=12)

    def test_test_set_is_untouched_and_scored_at_all_modes(self):
        split, item = self._evaluate("smote")
        self.assertTrue(item["checks"]["test_nguyên_vẹn"])
        modes = sorted(row["threshold_mode"] for row in item["rows"])
        self.assertEqual(modes, sorted({"best_f1", "best_cost", "min_precision", "fixed_0.5"}))
        for row in item["rows"]:
            self.assertEqual(row["n_test"], len(split["y_test"]))

    def test_cleaning_technique_reports_nearly_unchanged_ratio(self):
        """Tomek Links là làm sạch biên: IR fold-train gần như không đổi (phát hiện phủ định)."""
        _split, item = self._evaluate("tomek")
        for row in item["resample_rows"]:
            self.assertLess(abs(row["train_ir_after"] - row["train_ir_before"]), 2.0)


class TestCatalogRunSmoke(unittest.TestCase):
    """Chạy end-to-end cỡ nhỏ + ghi artifact tạm: bảo đảm báo cáo sinh được và không rò rỉ."""

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm để chạy danh mục")
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
                self.assertTrue((pathlib.Path(td) / name).exists(), f"thiếu artifact {name}")
            markdown = (pathlib.Path(td) / "techniques.md").read_text(encoding="utf-8")
            for expected in ("yêu cầu #2", "Kiểm chứng chống rò rỉ", "PASS", "Kết luận"):
                self.assertIn(expected, markdown)
            csv_rows = (pathlib.Path(td) / "techniques.csv").read_text(
                encoding="utf-8").strip().splitlines()
            self.assertEqual(len(csv_rows) - 1, 4 * len(statuses))

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm để chạy danh mục")
    def test_markdown_renders_for_every_technique(self):
        result = run(n_samples=1200, n_splits=2, techniques=["smote", "adasyn", "focal_loss"],
                     write=False, quick=True)
        text = _markdown(result)
        for key in ("smote", "adasyn", "focal_loss"):
            self.assertIn(key, text)
        self.assertIn("Ngưỡng chọn trên xác suất out-of-fold", text)
