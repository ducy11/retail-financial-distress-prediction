"""Kiểm thử EDA chuyên sâu (`forecasting/eda.py`): thống kê nhãn, chất lượng feature, quan hệ, drift.

Chạy: python -m unittest discover -s tests -v
       (riêng: python -m unittest tests.test_eda_deep -v)

Vì sao cần: các chỉ số trong `reports/results/eda_deep.md` được dùng để QUYẾT ĐỊNH (loại feature,
thêm cờ missing, chọn giao thức đánh giá). Nếu công thức entropy/IR/KS/PSI sai thì kết luận sai,
nên mọi hàm đều được kiểm thử trên dữ liệu dựng tay có đáp án biết trước, cộng 1 test chạy thật.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting import eda  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - hook của unittest
    quiet_library_warnings()


def make_sample(sample_id: str, ticker: str, target_end: str, label: int,
                periods=(), available_on: str = "2020-06-01") -> dict:
    """Sample tối thiểu đúng cấu trúc prepared (chỉ các khoá mà EDA đọc)."""
    return {
        "sample_id": sample_id,
        "ticker": ticker,
        "request": {
            "history": [{"period_end": p} for p in periods],
            "as_of": "2020-01-01",
            "target_period_start": target_end,
            "target_period_end": target_end,
        },
        "is_distressed": label,
        "label_available_on": available_on,
    }


class TestDistributionStatistics(unittest.TestCase):
    """Entropy, Gini, Imbalance Ratio, BH-FDR — nền của mọi kết luận về nhãn."""

    def test_balanced_binary_has_entropy_one_bit(self):
        result = eda.distribution_of([1, 1, 0, 0])
        self.assertAlmostEqual(result["entropy_bits"], 1.0, places=6)
        self.assertAlmostEqual(result["gini"], 0.5, places=6)
        self.assertAlmostEqual(result["imbalance_ratio"], 1.0, places=6)
        self.assertEqual(result["level"], "balanced")

    def test_single_class_has_zero_entropy_and_infinite_ratio(self):
        result = eda.distribution_of([1, 1, 1, 1])
        self.assertEqual(result["entropy_bits"], 0.0)
        self.assertEqual(result["gini"], 0.0)
        self.assertIsNone(result["imbalance_ratio"])           # inf → None khi ghi JSON
        self.assertTrue(math.isinf(eda.imbalance_ratio([4, 0])))
        self.assertEqual(result["level"], "severely_imbalanced")
        self.assertAlmostEqual(result["majority_baseline_accuracy_pct"], 100.0, places=6)

    def test_moderate_imbalance_is_slightly_imbalanced(self):
        result = eda.distribution_of([1, 1, 1, 0])            # 75/25 ⇒ thiểu số 25% < 40%
        self.assertEqual(result["level"], "slightly_imbalanced")
        self.assertAlmostEqual(result["imbalance_ratio"], 3.0, places=6)
        expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        self.assertAlmostEqual(result["entropy_bits"], round(expected, 6), places=6)

    def test_normalized_entropy_is_one_for_any_balanced_multiclass(self):
        self.assertAlmostEqual(eda.normalized_entropy([5, 5]), 1.0, places=6)
        self.assertAlmostEqual(eda.normalized_entropy([4, 4, 4, 4]), 1.0, places=6)
        self.assertEqual(eda.normalized_entropy([7, 0, 0, 0]), 0.0)

    def test_effective_number_counts_dominant_entity_as_about_one(self):
        self.assertAlmostEqual(eda.effective_number([25] * 4), 4.0, places=6)
        self.assertAlmostEqual(eda.effective_number([90, 10]), 1.2195, places=3)

    def test_benjamini_hochberg_is_monotone_and_bounded(self):
        p_values = [0.001, 0.02, 0.5]
        q_values = eda.benjamini_hochberg(p_values)
        self.assertEqual(len(q_values), 3)
        self.assertLessEqual(q_values[0], q_values[1])
        self.assertLessEqual(q_values[1], q_values[2])
        for p_value, q_value in zip(p_values, q_values):
            self.assertGreaterEqual(q_value, p_value)
            self.assertLessEqual(q_value, 1.0)

    def test_benjamini_hochberg_keeps_missing_positions(self):
        q_values = eda.benjamini_hochberg([0.01, None])
        self.assertIsNotNone(q_values[0])
        self.assertIsNone(q_values[1])
        self.assertEqual(eda.benjamini_hochberg([]), [])


class TestLabelDeepDive(unittest.TestCase):
    """Nhãn ở cấp thực thể: chuỗi trạng thái, công ty đơn lớp, P(1|1)."""

    def setUp(self):
        self.samples = {
            "train": [
                make_sample("AAA-1", "AAA", "2019-03-31", 1, ["2018-12-31"]),
                make_sample("AAA-2", "AAA", "2019-06-30", 1, ["2018-12-31", "2019-03-31"]),
                make_sample("BBB-1", "BBB", "2019-03-31", 1, ["2018-12-31"]),
                make_sample("BBB-2", "BBB", "2019-06-30", 0, ["2018-12-31", "2019-03-31"]),
                make_sample("BBB-3", "BBB", "2019-09-30", 1, ["2019-03-31", "2019-06-30"]),
            ],
            "validation": [],
            "test": [],
            "purged": [],
        }

    def test_per_ticker_spells_and_switches(self):
        result = eda.label_deep_dive(self.samples)
        self.assertEqual(result["per_ticker"]["AAA"]["n_switches"], 0)
        self.assertEqual(result["per_ticker"]["AAA"]["max_run_one"], 2)
        self.assertEqual(result["per_ticker"]["BBB"]["n_switches"], 2)
        self.assertEqual(result["per_ticker"]["BBB"]["max_run_zero"], 1)
        self.assertEqual(result["per_ticker"]["BBB"]["max_run_one"], 1)

    def test_single_class_entities_are_flagged(self):
        result = eda.label_deep_dive(self.samples)
        self.assertEqual(result["entities"]["tickers_all_one"], ["AAA"])
        self.assertEqual(result["entities"]["tickers_all_zero"], [])
        self.assertEqual(result["entities"]["tickers_needing_entity_level_cv"], ["AAA"])
        self.assertAlmostEqual(
            result["entities"]["share_samples_in_single_class_tickers_pct"], 40.0, places=6)

    def test_transitions_are_counted_chronologically(self):
        result = eda.label_deep_dive(self.samples)
        counts = result["transitions"]["counts"]
        # AAA: (1,1) ⇒ 1→1; BBB: (1,0),(0,1) ⇒ 1→0 và 0→1
        self.assertEqual(counts, {"0->0": 0, "0->1": 1, "1->0": 1, "1->1": 1})
        self.assertAlmostEqual(result["transitions"]["p_one_given_one"], 0.5, places=6)
        self.assertAlmostEqual(result["transitions"]["p_one_given_zero"], 1.0, places=6)
        self.assertAlmostEqual(result["transitions"]["persistence_pct"], 100.0 / 3.0, places=6)


class TestFeatureQuality(unittest.TestCase):
    """Phát hiện cột hằng/gần hằng, thiếu nhiều, đuôi nặng, outlier IQR."""

    def setUp(self):
        constant = np.full(20, 5.0)
        missing = np.array([float(i) for i in range(1, 15)] + [np.nan] * 6)         # 30% thiếu
        near_constant = np.array([0.0] * 19 + [100.0])                              # 95% một giá trị
        outlier = np.array([float(i) for i in range(1, 19)] + [10000.0, 20000.0])   # 10% outlier
        self.X = np.column_stack([constant, missing, near_constant, outlier])
        self.names = ["c_const", "c_missing", "c_near", "c_outlier"]

    def test_missing_rate_and_unique_count(self):
        stats = {r["feature"]: r for r in eda.feature_statistics(self.X, self.names)}
        self.assertEqual(stats["c_missing"]["n_missing"], 6)
        self.assertAlmostEqual(stats["c_missing"]["missing_pct"], 30.0, places=6)
        self.assertEqual(stats["c_missing"]["n_unique"], 14)

    def test_constant_and_near_constant_flags(self):
        stats = {r["feature"]: r for r in eda.feature_statistics(self.X, self.names)}
        self.assertTrue(stats["c_const"]["is_constant"])
        self.assertFalse(stats["c_const"]["is_near_constant"])
        self.assertTrue(stats["c_near"]["is_near_constant"])
        self.assertFalse(stats["c_near"]["is_constant"])
        self.assertAlmostEqual(stats["c_near"]["top_value_share"], 0.95, places=6)

    def test_iqr_outlier_is_detected(self):
        stats = {r["feature"]: r for r in eda.feature_statistics(self.X, self.names)}
        self.assertEqual(stats["c_outlier"]["iqr_outlier_n"], 2)          # 10000 và 20000
        self.assertAlmostEqual(stats["c_outlier"]["iqr_outlier_pct"], 10.0, places=6)

    def test_quality_flags_lists_expected_features(self):
        flags = eda.feature_quality_flags(eda.feature_statistics(self.X, self.names))
        self.assertEqual(flags["constant"], ["c_const"])
        self.assertEqual(flags["near_constant"], ["c_near"])
        self.assertIn("c_missing", flags["missing_gt_20pct"])
        self.assertIn("c_outlier", flags["many_outliers"])


class TestTargetAssociation(unittest.TestCase):
    """Liên hệ feature ↔ nhãn: xếp hạng, hướng, lift, feature không dùng được."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = np.array([0, 1] * 60)
        signal = self.y * 5.0 + rng.normal(0, 0.5, size=120)      # tách lớp rất mạnh
        noise = rng.normal(0, 1.0, size=120)
        constant = np.zeros(120)
        self.X = np.column_stack([signal, noise, constant])
        self.names = ["signal", "noise", "constant"]

    def test_strong_feature_ranked_first_with_high_auc(self):
        result = eda.target_association(self.X, self.y, self.names)
        top = result["ranked_by_effect"][0]
        self.assertEqual(top["feature"], "signal")
        self.assertGreater(top["auc"], 0.95)
        self.assertEqual(top["direction"], "giá trị cao ⇒ nhãn 1")
        self.assertAlmostEqual(top["effect_rank_biserial"], top["auc"] * 2 - 1, places=4)

    def test_constant_feature_has_no_auc_and_is_counted(self):
        result = eda.target_association(self.X, self.y, self.names)
        by_name = {r["feature"]: r for r in result["per_feature"]}
        self.assertIsNone(by_name["constant"]["auc"])
        self.assertEqual(result["n_usable_features"], 2)

    def test_lift_of_separating_feature_is_above_one(self):
        result = eda.target_association(self.X, self.y, self.names)
        by_name = {r["feature"]: r for r in result["per_feature"]}
        self.assertGreater(by_name["signal"]["lift_top_decile"], 1.0)
        self.assertLessEqual(by_name["signal"]["q_value_bh"], 0.05)


class TestCorrelationAndDrift(unittest.TestCase):
    """Đa cộng tuyến, số chiều hiệu dụng, dịch chuyển phân phối train → test."""

    def setUp(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0.0, 1.0, 200)
        self.X = np.column_stack([base, base * 3.0 + 1.0, rng.normal(0.0, 1.0, 200)])
        self.names = ["base", "collinear", "independent"]

    def test_collinear_features_form_one_cluster(self):
        corr = eda.pairwise_correlation(self.X)
        self.assertAlmostEqual(corr[0, 1], 1.0, places=6)
        clusters = eda.correlation_clusters(corr, self.names)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0], ["base", "collinear"])

    def test_effective_dimensionality_drops_below_column_count(self):
        dims = eda.effective_dimensionality(self.X)
        self.assertLess(dims["participation_ratio"], 3.0)
        self.assertLessEqual(dims["n_components_for_95pct_variance"], 3)
        self.assertEqual(dims["n_columns"], 3)

    def test_pairwise_correlation_skips_pairs_without_enough_rows(self):
        corr = eda.pairwise_correlation(np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]]))
        self.assertTrue(math.isnan(corr[0, 1]))
        self.assertEqual(corr[0, 0], 1.0)

    def test_drift_flags_only_the_shifted_feature(self):
        rng = np.random.default_rng(2)
        same_ref = rng.normal(0.0, 1.0, 200)
        same_new = rng.normal(0.0, 1.0, 200)
        X_ref = np.column_stack([rng.normal(0.0, 1.0, 200), same_ref])
        X_new = np.column_stack([rng.normal(5.0, 1.0, 200), same_new])
        result = eda.drift_analysis(X_ref, X_new, ["shifted", "same"])
        self.assertIn("shifted", result["drifted_features"])
        self.assertNotIn("same", result["drifted_features"])
        by_name = {r["feature"]: r for r in result["per_feature"]}
        self.assertGreater(by_name["shifted"]["smd"], 0.5)
        self.assertGreater(by_name["shifted"]["ks_statistic"], 0.9)
        self.assertLess(by_name["same"]["ks_statistic"], 0.3)

    def test_psi_is_zero_for_identical_distribution(self):
        rng = np.random.default_rng(3)
        values = rng.normal(0.0, 1.0, 300)
        self.assertAlmostEqual(eda.population_stability_index(values, values), 0.0, places=6)


class TestMissingnessAndLeakage(unittest.TestCase):
    """Missingness mang thông tin nhãn, đồng-thiếu, dòng trùng, chồng lấn lịch sử."""

    def test_mnar_feature_is_flagged_first(self):
        rng = np.random.default_rng(4)
        labels = np.array([0] * 60 + [1] * 60)
        mnar = rng.normal(0.0, 1.0, 120)
        mnar[labels == 1] = np.nan                     # thiếu đúng bằng lớp dương
        control = rng.normal(0.0, 1.0, 120)
        X = np.column_stack([mnar, control])
        split = {"train": X[:60], "validation": X[60:80], "test": X[80:]}
        y_split = {"train": labels[:60], "validation": labels[60:80], "test": labels[80:]}
        result = eda.missingness_analysis(split, y_split, ["mnar", "control"])
        top = result["label_informativeness"]["ranked_by_abs_delta"][0]
        self.assertEqual(top["feature"], "mnar")
        self.assertAlmostEqual(top["delta_pct_points"], 100.0, places=6)
        self.assertLess(top["q_value_bh"], 0.05)
        self.assertAlmostEqual(result["missing_density_per_sample"]["mean_features_missing"], 0.5,
                               places=6)
        self.assertEqual(result["missing_pct_by_split"]["test"]["mnar"], 100.0)
        self.assertEqual(result["missing_pct_by_split"]["train"]["mnar"], 0.0)

    def test_identical_missing_patterns_are_grouped(self):
        X = np.array([[np.nan, np.nan, 3.0], [np.nan, np.nan, 4.0], [1.0, 2.0, 5.0], [6.0, 7.0, 8.0]])
        result = eda.missingness_analysis({"train": X}, {"train": np.array([1, 0, 1, 0])},
                                          ["a", "b", "complete"])
        self.assertEqual([sorted(group) for group in result["identical_missing_pattern_groups"]],
                         [["a", "b"]])
        self.assertEqual(result["co_missing_top"][0]["jaccard"], 1.0)

    def test_duplicate_feature_rows_are_counted(self):
        X_ref = np.array([[1.0, 2.0], [np.nan, 4.0]])
        X_new = np.array([[1.0, 2.0], [np.nan, 4.0], [9.0, 9.0]])
        result = eda.duplicate_feature_rows(X_ref, X_new)
        self.assertEqual(result["n_new_rows_duplicated_in_ref"], 2)
        self.assertEqual(result["n_duplicate_rows_within_ref"], 0)

    def test_history_overlap_and_cross_split_target_leak(self):
        samples = {
            "train": [make_sample("AAA-1", "AAA", "2019-03-31", 1, ["2018-12-31", "2019-03-31"]),
                      make_sample("AAA-2", "AAA", "2019-06-30", 1, ["2019-03-31", "2019-06-30"])],
            "test": [make_sample("AAA-3", "AAA", "2019-03-31", 0, ["2019-03-31", "2019-06-30"])],
            "validation": [], "purged": [],
        }
        result = eda.history_overlap_analysis(samples)
        self.assertEqual(result["consecutive_samples_same_ticker"]["n_pairs"], 2)
        self.assertAlmostEqual(result["consecutive_samples_same_ticker"]["mean_history_jaccard"],
                               2 / 3, places=6)
        self.assertEqual(result["cross_split_target_leak"]["n_test_targets_seen_in_train_history"], 1)
        key = f"history_periods_{eda.REFERENCE_SPLIT}_vs_{eda.COMPARISON_SPLIT}"
        self.assertAlmostEqual(result[key]["share_of_test_history_periods_in_train_pct"], 100.0,
                               places=6)

    def test_label_availability_detects_impossible_ordering(self):
        samples = {"train": [make_sample("AAA-1", "AAA", "2019-03-31", 1, ["2018-12-31"],
                                        available_on="2019-01-15")],
                   "validation": [], "test": [], "purged": []}
        result = eda.label_availability_analysis(samples)
        self.assertEqual(result["n_label_available_before_target_end"], 1)


class TestEndToEnd(unittest.TestCase):
    """Chạy thật trên `data/prepared` — bỏ qua nếu dữ liệu chưa được tạo."""

    def test_run_writes_artifacts_and_is_json_serialisable(self):
        if not (ROOT / "data" / "prepared" / "train.json").exists():
            self.skipTest("Chưa có data/prepared (chạy `python -m forecasting.data` trước).")
        with tempfile.TemporaryDirectory() as tmp:
            out, figs = pathlib.Path(tmp) / "results", pathlib.Path(tmp) / "figures"
            summary = eda.run(write=True, out_dir=out, fig_dir=figs)
            payload = json.loads((out / "eda_deep.json").read_text(encoding="utf-8"))
            self.assertTrue((out / "eda_deep.md").read_text(encoding="utf-8").startswith("# EDA"))
            self.assertEqual(payload["scope"]["n_features"], len(payload["scope"]["feature_names"]))
            self.assertEqual(payload["label"]["overall"]["n"],
                             sum(payload["scope"]["n_samples"].values()))
            self.assertTrue(payload["auto_conclusions"])
            for key in ("feature_quality", "feature_vs_label", "feature_vs_feature", "drift",
                        "missingness", "leakage"):
                self.assertIn(key, payload)
            self.assertEqual(len(list(figs.glob("*.png"))), 7)
            self.assertEqual(summary["label"]["overall"]["n"],
                             payload["scope"]["n_samples"]["train"]
                             + payload["scope"]["n_samples"]["validation"]
                             + payload["scope"]["n_samples"]["test"]
                             + payload["scope"]["n_samples"]["purged"])
