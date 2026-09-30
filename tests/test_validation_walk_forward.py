"""Test cho hai phép đo ở `forecasting.validation`: walk-forward theo THỜI GIAN và bootstrap CỤM.

Vì sao cần: mục 6.2 (CI theo cụm công ty) và 6.5 (walk-forward) của báo cáo dựa hoàn toàn vào hai
hàm này — nếu chia fold sai (chồng lấn thời gian, dùng nhãn chưa công bố) hoặc nếu CI theo cụm bị
tính như CI theo từng mẫu thì phần "tổng quát hoá theo thời gian / độ bất định" là vô nghĩa.
"""
from __future__ import annotations

import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.data_loader import load_prepared  # noqa: E402
from forecasting.validation import (cluster_bootstrap_ci, walk_forward_folds,  # noqa: E402
                                    walk_forward_metrics)
from runtime_warnings import quiet_library_warnings  # noqa: E402

quiet_library_warnings()


def _d(text: str) -> date:
    """'YYYY-MM-DD' → date (dùng để so mốc trong test)."""
    year, month, day = (int(part) for part in text.split("-")[:3])
    return date(year, month, day)


def _sample(ticker: str, end: str, available: str) -> dict:
    """Mẫu tối thiểu đủ cho hai hàm cần test (không cần feature)."""
    return {"sample_id": f"{ticker}-{end}", "ticker": ticker, "is_distressed": 1,
            "label_available_on": available,
            "request": {"target_period_end": end, "history": []}}


def _toy_samples() -> list:
    """12 mốc thời gian, nhãn công bố ngay sau kỳ (để test được purge)."""
    return [_sample("A", f"2020-{m:02d}-28", f"2020-{m:02d}-28") for m in range(1, 13)]


class TestWalkForwardFolds(unittest.TestCase):
    def test_train_and_test_never_overlap(self):
        folds = walk_forward_folds(_toy_samples(), n_folds=3, purge_days=0, min_train=1)
        self.assertTrue(folds, "phải sinh được fold")
        for fold in folds:
            self.assertFalse(set(fold["train_idx"]) & set(fold["test_idx"]))

    def test_train_periods_always_before_the_cut(self):
        folds = walk_forward_folds(_toy_samples(), n_folds=3, purge_days=0, min_train=1)
        for fold in folds:
            for i in fold["train_idx"]:
                self.assertLess(_d(_toy_samples()[i]["request"]["target_period_end"]),
                                _d(fold["cut"]))

    def test_purge_drops_labels_not_yet_published(self):
        samples = _toy_samples()
        folds = walk_forward_folds(samples, n_folds=2, purge_days=90, min_train=1)
        for fold in folds:
            limit = _d(fold["cut"]) - timedelta(days=fold["purge_days"])
            for i in fold["train_idx"]:
                self.assertLessEqual(_d(samples[i]["label_available_on"]), limit)

    def test_cuts_move_forward_in_time(self):
        folds = walk_forward_folds(_toy_samples(), n_folds=3, purge_days=0, min_train=1)
        cuts = [_d(fold["cut"]) for fold in folds]
        self.assertEqual(cuts, sorted(cuts), "mốc cắt phải tăng dần theo fold")


class TestWalkForwardMetricsOnRealData(unittest.TestCase):
    """Chạy trên prepared thật (train+validation) — bỏ qua nếu thiếu dữ liệu."""

    def test_summary_covers_every_model(self):
        try:
            samples = load_prepared("train") + load_prepared("validation")
        except FileNotFoundError:  # pragma: no cover
            self.skipTest("chưa có data/prepared — chạy `python -m forecasting.data`")
        result = walk_forward_metrics(["logistic", "mlp"], samples, n_folds=2,
                                      purge_days=60, min_train=40)
        self.assertEqual(sorted(result["summary"]), ["logistic", "mlp"])
        self.assertIn("rows", result)
        for name, block in result["summary"].items():
            self.assertEqual(len(block["per_fold_auroc"]), block["n_folds_evaluated"])
            if block["mean_auroc"] is not None:
                self.assertTrue(0.0 <= block["mean_auroc"] <= 1.0, name)


class TestClusterBootstrap(unittest.TestCase):
    def test_ci_brackets_point_estimate_and_counts_clusters(self):
        rng = np.random.default_rng(0)
        samples, y, p = [], [], []
        for group in range(6):
            for _ in range(5):
                label = int(rng.random() < 0.5)
                samples.append({"ticker": f"C{group}"})
                y.append(label)
                p.append(0.9 if label else 0.1)
        out = cluster_bootstrap_ci(samples, y, p, n_boot=60, seed=1)
        self.assertEqual(out["n_clusters"], 6)
        self.assertGreater(out["auroc"]["n_valid"], 0)
        self.assertLessEqual(out["auroc"]["ci95_low"], out["auroc"]["point"])
        self.assertGreaterEqual(out["auroc"]["ci95_high"], out["auroc"]["point"])

    def test_single_class_input_returns_nan_without_raising(self):
        out = cluster_bootstrap_ci([{"ticker": "A"}, {"ticker": "A"}], [1, 1], [0.9, 0.9],
                                   n_boot=5, seed=0)
        self.assertEqual(out["auroc"]["n_valid"], 0)
        self.assertTrue(math.isnan(out["auroc"]["point"]))


if __name__ == "__main__":
    unittest.main()
