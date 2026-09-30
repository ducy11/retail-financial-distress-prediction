"""Kiểm thử thực nghiệm `imbalance_experiment`: dữ liệu, danh mục pipeline, CV, chống rò rỉ, insights.

Chạy: python -m unittest discover -s tests -v

Nhóm test:
1. `TestDataLoader`      — dataset 1:50 / 1:100, chia tập stratified, fold giữ tỉ lệ lớp, đọc CSV.
2. `TestPipelineBuilder` — đủ 17 pipeline theo đề bài; mọi kỹ thuật resampling dùng
                            `imblearn.pipeline.Pipeline` và giữ sampler BÊN TRONG pipeline.
3. `TestEvaluation`      — CV mean ± std, metric bắt buộc (PR-AUC, ROC-AUC, F1, Balanced Acc, Recall,
                            FPR), cờ chống rò rỉ PASS, dữ liệu PR curve hợp lệ.
4. `TestInsights`        — phân tích tự động: SMOTE vs dọn biên, chi phí resampling vs cost/weight,
                            hybrid vs đơn lẻ (trên số liệu giả để kiểm tra công thức).
5. `TestEndToEnd`        — chạy thực nghiệm cỡ nhỏ, ghi artifact vào thư mục tạm và kiểm tra nội dung.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from imbalance_experiment import insights  # noqa: E402
from imbalance_experiment.config import ExperimentConfig  # noqa: E402
from imbalance_experiment.data_loader import (load_csv_dataset, load_dataset,  # noqa: E402
                                              make_synthetic_dataset, stratified_folds)
from imbalance_experiment.evaluation import (METRIC_KEYS, FoldResult, TechniqueResult,  # noqa: E402
                                             evaluate_technique, summary_rows)
from imbalance_experiment.main import run  # noqa: E402
from imbalance_experiment.pipeline_builder import (REQUIRED_TECHNIQUES, build_pipelines,  # noqa: E402
                                                   inspect_pipeline, make_estimator,
                                                   missing_requirements)

try:  # pragma: no cover - phụ thuộc môi trường
    import imblearn  # noqa: F401

    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - `unittest` hook
    """Lọc lại cảnh báo thư viện vô hại sau khi `unittest` đặt `simplefilter("default")`.

    Thực nghiệm vẽ PR curve (matplotlib/pyparsing deprecate) và fit mô hình tuyến tính
    (`iprint` scikit-learn 1.6 + scipy 1.18). Xem `runtime_warnings.py`.
    """
    quiet_library_warnings()

try:  # pragma: no cover - phụ thuộc môi trường
    import lightgbm  # noqa: F401

    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

#: Kỳ vọng ĐÚNG theo đề bài (không đọc từ code ⇒ test bắt được việc bỏ sót kỹ thuật).
EXPECTED_TECHNIQUES = {
    "baseline",
    "ros", "smote", "borderline_smote", "adasyn", "rus", "tomek", "enn",
    "class_weight", "focal_loss",
    "balanced_rf", "easy_ensemble", "balanced_bagging",
    "smote_tomek", "smote_enn", "smote_class_weight", "rusboost",
}
#: Kỹ thuật có bước lấy mẫu phải nằm TRONG pipeline.
RESAMPLING_TECHNIQUES = {"ros", "smote", "borderline_smote", "adasyn", "rus", "tomek", "enn",
                         "smote_tomek", "smote_enn", "smote_class_weight"}


def tiny_config(**overrides) -> ExperimentConfig:
    """Cấu hình nhỏ, tất định, dùng cho mọi test (không ghi artifact)."""
    base = dict(n_samples=1500, imbalance_ratio=50, n_splits=2, quick=True, write=False,
                techniques=())
    base.update(overrides)
    return ExperimentConfig(**base)



class TestDataLoader(unittest.TestCase):
    """Dataset mất cân bằng + chia tập/fold phải GIỮ NGUYÊN tỉ lệ lớp."""

    def test_synthetic_imbalance_ratio_and_stratified_split(self):
        cfg = tiny_config(n_samples=5000, imbalance_ratio=50)
        dataset = make_synthetic_dataset(cfg)
        train = dataset.distribution(dataset.y_train)
        test = dataset.distribution(dataset.y_test)
        self.assertEqual(dataset.X_train.shape[1], cfg.n_features)
        self.assertAlmostEqual(train["imbalance_ratio"], 50.0, delta=6.0)
        self.assertAlmostEqual(test["imbalance_ratio"], 50.0, delta=8.0)
        self.assertAlmostEqual(train["positive_pct"], test["positive_pct"], delta=1.0)
        self.assertEqual(train["n"] + test["n"], cfg.n_samples)
        self.assertIn("1:50", dataset.describe())

    def test_ratio_100_is_supported(self):
        cfg = tiny_config(n_samples=6000, imbalance_ratio=100)
        dataset = load_dataset(cfg)
        self.assertAlmostEqual(dataset.distribution(dataset.y_train)["imbalance_ratio"],
                               100.0, delta=12.0)
        self.assertAlmostEqual(cfg.positive_rate, 1 / 101, places=12)

    def test_stratified_folds_keep_class_ratio_and_cover_all_samples(self):
        cfg = tiny_config(n_samples=3000, n_splits=5)
        dataset = make_synthetic_dataset(cfg)
        folds = stratified_folds(dataset.y_train, cfg)
        self.assertEqual(len(folds), 5)
        global_rate = float(dataset.y_train.mean())
        seen: list = []
        for train_index, val_index in folds:
            self.assertFalse(set(train_index.tolist()) & set(val_index.tolist()))
            self.assertEqual(len(train_index) + len(val_index), len(dataset.y_train))
            self.assertAlmostEqual(float(dataset.y_train[val_index].mean()), global_rate, delta=0.01)
            seen.extend(val_index.tolist())
        self.assertEqual(sorted(seen), list(range(len(dataset.y_train))))

    def test_csv_loader_reads_binary_target(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "mini.csv"
            rng = np.random.default_rng(0)
            with open(path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["f1", "f2", "Class"])
                for _ in range(400):
                    label = int(rng.random() < 0.1)
                    writer.writerow([f"{rng.normal():.4f}", f"{rng.normal():.4f}", label])
            dataset = load_csv_dataset(tiny_config(source=str(path), n_samples=400))
            self.assertIn("mini.csv", dataset.name)
            self.assertEqual(set(np.unique(dataset.y_train).tolist()), {0, 1})
            self.assertEqual(dataset.X_train.shape[1], 2)
            self.assertEqual(dataset.meta["source"], str(path))


class TestPipelineBuilder(unittest.TestCase):
    """Danh mục pipeline: đủ kỹ thuật theo đề bài, resampling nằm TRONG imblearn pipeline."""

    def test_catalogue_covers_every_required_technique(self):
        self.assertEqual(set(REQUIRED_TECHNIQUES), EXPECTED_TECHNIQUES)
        specs = build_pipelines(tiny_config())
        self.assertEqual({spec.key for spec in specs}, EXPECTED_TECHNIQUES)
        self.assertEqual(specs[0].key, "baseline")
        groups = {spec.key: spec.group for spec in specs}
        self.assertEqual(groups["baseline"], "baseline")
        for key in RESAMPLING_TECHNIQUES:
            expected = "hybrid" if key in ("smote_tomek", "smote_enn", "smote_class_weight") \
                else "single-data"
            self.assertEqual(groups[key], expected)
        self.assertEqual(groups["class_weight"], "single-algorithm")
        self.assertEqual(groups["focal_loss"], "single-algorithm")
        self.assertEqual(groups["rusboost"], "hybrid")

    def test_resampling_techniques_use_imblearn_pipeline_inside(self):
        cfg = tiny_config()
        for key in sorted(RESAMPLING_TECHNIQUES):
            with self.subTest(technique=key):
                pipeline = make_estimator(cfg, key)
                checks = inspect_pipeline(pipeline, expect_samplers=True)
                if HAS_IMBLEARN:
                    self.assertEqual(type(pipeline).__module__, "imblearn.pipeline")
                self.assertGreaterEqual(checks["n_sampler_steps"], 1)
                self.assertTrue(checks["sampler_nằm_trong_pipeline"])
                self.assertEqual([name for name, _step in pipeline.steps][-1], "classifier")

    def test_non_resampling_techniques_have_no_sampler_step(self):
        cfg = tiny_config()
        for key in ("baseline", "class_weight", "focal_loss", "balanced_rf", "easy_ensemble",
                    "balanced_bagging", "rusboost"):
            with self.subTest(technique=key):
                pipeline = make_estimator(cfg, key)
                checks = inspect_pipeline(pipeline, expect_samplers=False)
                self.assertEqual(checks["n_sampler_steps"], 0)
                self.assertTrue(checks["sampler_nằm_trong_pipeline"])

    def test_focal_loss_requires_lightgbm_base_model(self):
        cfg = tiny_config(base_model="random_forest")
        gaps = missing_requirements(cfg, "focal_loss")
        self.assertTrue(gaps, "Focal Loss phải bị bỏ qua khi base_model không phải lightgbm")
        self.assertTrue(any("lightgbm" in gap for gap in gaps))
        self.assertEqual(missing_requirements(cfg, "baseline"), [])

    def test_unknown_technique_raises(self):
        with self.assertRaises(KeyError):
            build_pipelines(tiny_config(), keys=("khong_ton_tai",))




class TestEvaluation(unittest.TestCase):
    """CV phải cho mean ± std, đủ metric yêu cầu #3, và cờ chống rò rỉ phải PASS."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_config(n_samples=1200, n_splits=2)
        cls.dataset = load_dataset(cls.cfg)
        cls.specs = {spec.key: spec for spec in build_pipelines(cls.cfg)}

    def _evaluate(self, key: str) -> TechniqueResult:
        return evaluate_technique(self.specs[key], self.dataset, self.cfg, log=lambda _m: None)

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm cho baseline")
    def test_baseline_cv_and_holdout_metrics(self):
        result = self._evaluate("baseline")
        self.assertEqual(result.status, "ok", result.reason)
        self.assertEqual(len(result.folds), 2)
        for metric in METRIC_KEYS:
            with self.subTest(metric=metric):
                self.assertTrue(bool(np.isfinite(result.mean[metric])))
                self.assertIn(metric, result.std)
                self.assertIn(metric, result.holdout)
        self.assertGreater(result.fit_seconds_mean, 0.0)
        self.assertTrue(all(result.checks.values()), result.checks)
        self.assertIsNotNone(result.pr_curve)
        self.assertEqual(len(result.pr_curve["recall"]), len(result.pr_curve["precision"]))
        self.assertTrue(0.0 <= result.holdout["fpr"] <= 1.0)
        self.assertFalse(result.is_resampling)

    @unittest.skipUnless(HAS_IMBLEARN and HAS_LIGHTGBM, "Cần imblearn + lightgbm")
    def test_resampling_flags_are_pass_and_size_is_recorded(self):
        result = self._evaluate("smote")
        self.assertEqual(result.status, "ok", result.reason)
        self.assertTrue(result.is_resampling)
        self.assertTrue(all(result.checks.values()), result.checks)
        self.assertIn("resampling dùng imblearn.pipeline", result.checks)
        self.assertGreater(result.n_train_after_mean, result.folds[0].n_train)
        self.assertGreater(result.resample_seconds_mean, 0.0)
        self.assertLess(result.folds[0].ir_after, result.folds[0].ir_before)

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm")
    def test_summary_rows_expose_metrics_for_cv_and_holdout(self):
        results = [self._evaluate("baseline")]
        cv_rows = summary_rows(results, source="cv")
        holdout_rows = summary_rows(results, source="holdout")
        self.assertIn("pr_auc_mean", cv_rows[0])
        self.assertIn("pr_auc_std", cv_rows[0])
        self.assertIn("fpr", holdout_rows[0])
        self.assertIn("group", holdout_rows[0])



class TestInsights(unittest.TestCase):
    """Công thức so sánh của phần phân tích phải đúng (kiểm tra bằng số liệu giả)."""

    @staticmethod
    def _result(key: str, group: str, *, pr_auc: float, f1: float, recall: float, fpr: float,
                seconds: float, n_after: float | None = None) -> TechniqueResult:
        metrics = {"pr_auc": pr_auc, "f1": f1, "recall": recall, "fpr": fpr, "roc_auc": 0.9,
                   "macro_f1": f1 - 0.05, "balanced_accuracy": 0.8, "precision": 0.5, "mcc": 0.4}
        result = TechniqueResult(
            key=key, name=key, group=group, note="", is_resampling=n_after is not None,
            mean=dict(metrics), std={"pr_auc": 0.01}, holdout=dict(metrics),
            fit_seconds_mean=seconds,
            n_train_after_mean=(n_after if n_after is not None else float("nan")),
            resample_seconds_mean=(0.2 if n_after is not None else float("nan")))
        result.folds = [FoldResult(fold=1, n_train=1000, n_val=200, n_train_after=n_after,
                                   ir_before=10.0, ir_after=2.0, metrics=metrics,
                                   fit_seconds=seconds, resample_seconds=0.2,
                                   checks={"fold_val_nguyên_vẹn": True})]
        return result

    @classmethod
    def setUpClass(cls):
        cls.results = [
            cls._result("baseline", "baseline", pr_auc=0.70, f1=0.60, recall=0.55, fpr=0.02,
                        seconds=1.0),
            cls._result("smote", "single-data", pr_auc=0.72, f1=0.65, recall=0.62, fpr=0.03,
                        seconds=2.0, n_after=2000),
            cls._result("smote_tomek", "hybrid", pr_auc=0.74, f1=0.68, recall=0.66, fpr=0.025,
                        seconds=2.4, n_after=1900),
            cls._result("class_weight", "single-algorithm", pr_auc=0.71, f1=0.63, recall=0.60,
                        fpr=0.02, seconds=1.1),
            cls._result("smote_class_weight", "hybrid", pr_auc=0.73, f1=0.67, recall=0.65,
                        fpr=0.021, seconds=2.1, n_after=2000),
        ]
        cls.cfg = tiny_config()
        cls.findings = insights.analyse(cls.results, cls.cfg)

    def test_smote_vs_cleaning_deltas(self):
        cleaning = self.findings["smote_analysis"]["cleaning"]["smote_tomek"]
        self.assertAlmostEqual(cleaning["pr_auc_delta"], 0.02, places=9)
        self.assertAlmostEqual(cleaning["f1_delta"], 0.03, places=9)
        self.assertLess(cleaning["fpr_delta"], 0.0)
        self.assertAlmostEqual(cleaning["n_train_after"], 1900.0, places=6)

    def test_compute_cost_findings(self):
        table = self.findings["compute"]["table"]
        self.assertGreater(table["smote"]["size_factor"], 1.5)
        self.assertAlmostEqual(table["class_weight"]["size_factor"], 1.0, places=9)
        self.assertAlmostEqual(self.findings["compute"]["speedup_smote_over_class_weight"],
                               2.0 / 1.1, places=6)

    def test_hybrid_beats_single_detection(self):
        hybrid = self.findings["hybrid_vs_single"]
        self.assertTrue(hybrid["smote_tomek"]["beats_parent"])
        self.assertAlmostEqual(hybrid["smote_tomek"]["delta_pr_auc"], 0.02, places=9)
        self.assertEqual(hybrid["smote_tomek"]["best_parent"], "smote")
        self.assertTrue(hybrid["smote_class_weight"]["beats_parent"])

    def test_ranking_and_fpr_extremes(self):
        ranking = self.findings["ranking"]
        self.assertEqual(ranking["pr_auc"]["key"], "smote_tomek")
        self.assertAlmostEqual(ranking["pr_auc"]["delta_vs_baseline"], 0.04, places=9)
        self.assertEqual(self.findings["lowest_fpr"][0]["key"], "baseline")
        self.assertEqual(self.findings["highest_fpr"][-1]["key"], "smote")

    def test_markdown_has_three_required_analyses(self):
        text = insights.render_markdown(self.results, self.cfg)
        for expected in ("### 1. SMOTE đơn lẻ vs SMOTE + dọn biên", "### 2. Chi phí tính toán",
                         "### 3. Khi nào kết hợp (hybrid) vượt trội", "Accuracy", "Giới hạn"):
            self.assertIn(expected, text)



class TestEndToEnd(unittest.TestCase):
    """Chạy thực nghiệm cỡ nhỏ có ghi artifact và kiểm tra nội dung báo cáo."""

    @unittest.skipUnless(HAS_LIGHTGBM, "Cần lightgbm để chạy thực nghiệm")
    def test_small_run_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = tiny_config(n_samples=1500, n_splits=2, out_dir=pathlib.Path(td), write=True,
                              techniques=("baseline", "smote", "smote_tomek", "class_weight",
                                          "smote_class_weight"))
            result = run(cfg)
            self.assertEqual(len(result["results"]), 5)
            statuses = {item.key: item.status for item in result["results"]}
            self.assertEqual(set(statuses.values()), {"ok"}, statuses)
            for name in ("summary.md", "summary.csv", "cv_mean_std.csv", "results.json", "run.log"):
                self.assertTrue((pathlib.Path(td) / name).exists(), f"thiếu artifact {name}")
            markdown = (pathlib.Path(td) / "summary.md").read_text(encoding="utf-8")
            for expected in ("Phương pháp đơn lẻ vs Phương pháp kết hợp", "PR-AUC", "FPR",
                             "Kiểm chứng chống rò rỉ", "Phân tích chuyên sâu", "PASS"):
                self.assertIn(expected, markdown)
            findings = result["findings"]
            self.assertIn("pr_auc", findings["ranking"])
            self.assertIn("smote_tomek", findings["hybrid_vs_single"])

    @unittest.skipUnless(HAS_IMBLEARN and HAS_LIGHTGBM, "Cần imblearn + lightgbm")
    def test_all_leak_checks_pass_for_resampling_techniques(self):
        cfg = tiny_config(n_samples=1500, n_splits=2, write=False,
                          techniques=("baseline", "smote", "smote_enn", "class_weight"))
        result = run(cfg)
        for item in result["results"]:
            with self.subTest(technique=item.key):
                self.assertEqual(item.status, "ok", item.reason)
                self.assertTrue(all(item.checks.values()), item.checks)
