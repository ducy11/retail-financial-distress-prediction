"""Kiểm thử bản demo dự đoán (`scripts/predict.py`).

Chạy: python -m unittest tests.test_predict -v

Vì sao cần: script này là "sản phẩm chạy được" để demo trong buổi bảo vệ, nên phải chắc chắn:
1. Dựng lại ĐÚNG số feature mà artifact đã huấn luyện (không lệch âm thầm 47 vs 46 cột).
2. Quyết định phải theo **ngưỡng vận hành** trong artifact (không phải 0,5).
3. Tra mẫu phải trả đúng split (để cảnh báo backtest) và báo lỗi rõ khi không có mẫu.
4. Phần KernelSHAP phải thoả ràng buộc efficiency: Σφ + E[f] ≈ f(x).
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.config import MODELS_DIR, RESULTS_DIR  # noqa: E402
from forecasting.features import feature_names  # noqa: E402
from scripts.predict import (find_sample, format_report, load_model,  # noqa: E402
                             operating_threshold, score_sample)
from runtime_warnings import quiet_library_warnings  # noqa: E402

HAS_ARTIFACT = (MODELS_DIR / "best.joblib").exists()
SKIP_REASON = "chưa có reports/models/best.joblib (chạy `python -m forecasting.train` trước)"


def setUpModule() -> None:  # noqa: D103 - hook của unittest
    quiet_library_warnings()


class TestFindSample(unittest.TestCase):
    """Tra mẫu theo sample_id / (ticker, quarter) và theo split."""

    def test_find_by_sample_id_returns_known_split(self):
        sample, split = find_sample(sample_id="HD-2024Q2")
        self.assertEqual(sample["sample_id"], "HD-2024Q2")
        self.assertIn(split, ("train", "validation", "test", "purged"))

    def test_find_by_ticker_and_quarter_matches_sample_id(self):
        by_id, split_id = find_sample(sample_id="FIVE-2024Q3")
        by_pair, split_pair = find_sample(ticker="FIVE", quarter="2024Q3")
        self.assertEqual(by_id["sample_id"], by_pair["sample_id"])
        self.assertEqual(split_id, split_pair)

    def test_unknown_sample_raises_key_error(self):
        with self.assertRaises(KeyError):
            find_sample(sample_id="ZZZ-1900Q1")

    def test_missing_arguments_raise_value_error(self):
        with self.assertRaises(ValueError):
            find_sample(ticker="HD")


class TestScoring(unittest.TestCase):
    """Chấm điểm: số feature khớp artifact, ngưỡng vận hành, xác suất hợp lệ."""

    @classmethod
    def setUpClass(cls):
        if not HAS_ARTIFACT:
            raise unittest.SkipTest(SKIP_REASON)
        cls.artifact = load_model()
        cls.sample, cls.split = find_sample(sample_id="HD-2024Q2")

    def test_feature_width_matches_artifact(self):
        result = score_sample(self.sample, self.artifact)
        self.assertEqual(result["n_features"], len(self.artifact["features"]))
        self.assertEqual(result["n_features"], len(feature_names()))

    def test_threshold_comes_from_artifact_not_half(self):
        result = score_sample(self.sample, self.artifact)
        self.assertAlmostEqual(result["threshold"], float(self.artifact["threshold"]), places=12)
        self.assertNotAlmostEqual(result["threshold"], 0.5, places=6)

    def test_probability_and_decision_are_consistent(self):
        result = score_sample(self.sample, self.artifact)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        expected = int(result["probability"] >= result["threshold"])
        self.assertEqual(result["decision"], expected)

    def test_operating_threshold_prefers_artifact_then_summary(self):
        """Thiếu `threshold` trong artifact ⇒ đọc `summary.json`; cuối cùng mới là 0,5."""
        expected = 0.5
        summary_path = RESULTS_DIR / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected = float(summary["best_threshold"])
        self.assertAlmostEqual(operating_threshold({}), expected, places=12)

    def test_format_report_mentions_backtest_warning(self):
        result = score_sample(self.sample, self.artifact)
        text = format_report(result, self.split)
        self.assertIn(str(result["sample_id"]), text)
        if self.split in ("test", "validation"):
            self.assertIn("backtest", text)


class TestExplain(unittest.TestCase):
    """Ràng buộc efficiency của KernelSHAP khi gọi qua bộ demo."""

    @classmethod
    def setUpClass(cls):
        if not HAS_ARTIFACT:
            raise unittest.SkipTest(SKIP_REASON)
        cls.artifact = load_model()
        cls.sample, _ = find_sample(sample_id="DG-2025Q2")

    def test_efficiency_constraint_holds(self):
        result = score_sample(self.sample, self.artifact, explain=True, top_k=5,
                              n_coalitions=60, n_background=8)
        explain = result["explain"]
        self.assertEqual(len(explain["contributions"]), 5)
        self.assertAlmostEqual(explain["prediction"], result["probability"], places=9)
        self.assertLess(abs(explain["efficiency_gap"]), 1e-6)
        # Σφ trên TOÀN BỘ feature = f(x) − E[f]; top-K chỉ là một phần nên chỉ kiểm tra dấu/hướng
        self.assertTrue(all(c["feature"] in set(feature_names())
                            for c in explain["contributions"]))


if __name__ == "__main__":
    unittest.main()
