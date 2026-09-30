"""Kiểm thử EDA cơ bản (`scripts/eda.py` + tiện ích chọn dữ liệu trong `forecasting/eda.py`).

Chạy: python -m unittest tests.test_eda_basic -v

Vì sao cần: mục 3.4–3.6 của báo cáo (thống kê mô tả, tương quan, nhận xét) được người chấm đọc
trực tiếp. Nếu công thức quy đổi đơn vị, chọn cột tỷ số, hay phát hiện "tương quan do ngoại lai"
sai thì kết luận trong báo cáo sai — nên mọi hàm đều được kiểm thử bằng dữ liệu dựng tay có đáp án.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.eda as eda_script  # noqa: E402
from forecasting import eda as eda_lib  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - hook của unittest
    quiet_library_warnings()


def make_history_row(revenue=100.0, current_assets=200.0, current_liabilities=100.0,
                     inventory=50.0, cost_of_sales=60.0, total_assets=400.0,
                     stockholders_equity=200.0, **extra) -> dict:
    """Một dòng lịch sử tối thiểu (giá trị VND dạng chuỗi như trong prepared)."""
    row = {
        "period_end": "2019-03-31",
        "revenue_vnd": str(int(revenue * 1e9)),
        "cost_of_sales_vnd": str(int(cost_of_sales * 1e9)),
        "current_assets_vnd": str(int(current_assets * 1e9)),
        "current_liabilities_vnd": str(int(current_liabilities * 1e9)),
        "inventory_vnd": str(int(inventory * 1e9)),
        "total_assets_vnd": str(int(total_assets * 1e9)),
        "stockholders_equity_vnd": str(int(stockholders_equity * 1e9)),
    }
    row.update(extra)
    return row


def make_sample(label=1, history=None, ticker="AAA") -> dict:
    """Sample tối thiểu để tính feature (không cần đầy đủ 16 chỉ tiêu)."""
    return {
        "sample_id": f"{ticker}-2019Q1",
        "ticker": ticker,
        "request": {"history": history or [make_history_row()], "as_of": "2019-04-01",
                    "target_period_start": "2019-04-01", "target_period_end": "2019-06-30"},
        "is_distressed": label,
    }


class TestIndicatorValueMatrix(unittest.TestCase):
    """Chọn dữ liệu gốc cho bảng thống kê mô tả (đơn vị nghìn tỷ VND)."""

    def setUp(self):
        self.rows = {
            "AAA": [{"revenue_vnd": str(int(4722.825 * 1e12)), "net_income_vnd": "-1000000000000",
                     "inventory_vnd": None}],
            "BBB": [{"revenue_vnd": "5000000000000", "net_income_vnd": None}],
        }

    def test_shape_scale_and_missing(self):
        matrix, names = eda_lib.indicator_value_matrix(self.rows, fields=["revenue", "net_income",
                                                                         "inventory"])
        self.assertEqual(names, ["revenue", "net_income", "inventory"])
        self.assertEqual(matrix.shape, (2, 3))
        self.assertAlmostEqual(matrix[0, 0], 4722.825, places=6)      # chia 1e12
        self.assertAlmostEqual(matrix[0, 1], -1.0, places=6)
        self.assertTrue(np.isnan(matrix[0, 2]))                       # None → NaN
        self.assertTrue(np.isnan(matrix[1, 1]))
        self.assertAlmostEqual(matrix[1, 0], 5.0, places=6)

    def test_default_fields_are_16_indicators(self):
        matrix, names = eda_lib.indicator_value_matrix(self.rows)
        self.assertEqual(len(names), 16)
        self.assertEqual(matrix.shape[0], 2)


class TestLatestRatioMatrix(unittest.TestCase):
    """Ma trận 14 tỷ số dùng cho bảng mô tả và tương quan."""

    def test_selects_14_ratio_columns_in_order(self):
        X, names, y = eda_lib.latest_ratio_matrix([make_sample(label=1), make_sample(label=0)])
        self.assertEqual(len(names), 14)
        self.assertEqual(X.shape, (2, 14))
        self.assertTrue(all(name.endswith("_latest") for name in names))
        self.assertEqual(list(y), [1, 0])

    def test_ratio_values_match_formula(self):
        sample = make_sample(history=[make_history_row(current_assets=200.0,
                                                      current_liabilities=100.0)])
        X, names, _ = eda_lib.latest_ratio_matrix([sample])
        current_ratio = X[0, names.index("current_ratio_latest")]
        self.assertAlmostEqual(float(current_ratio), 2.0, places=6)
        debt_to_assets = X[0, names.index("debt_to_assets_latest")]
        self.assertAlmostEqual(float(debt_to_assets), 0.5, places=6)   # (400-200)/400


class TestOutlierDrivenPairs(unittest.TestCase):
    """"Tương quan giả do ngoại lai" phải được phát hiện từ ma trận Pearson/Spearman."""

    def test_detects_pair_where_pearson_and_spearman_disagree(self):
        correlation = {
            "feature_order": ["a", "b", "c"],
            "pearson_matrix": [[1.0, -0.08, 0.9], [-0.08, 1.0, 0.1], [0.9, 0.1, 1.0]],
            "spearman_matrix": [[1.0, 0.45, 0.9], [0.45, 1.0, 0.1], [0.9, 0.1, 1.0]],
        }
        pairs = eda_lib.outlier_driven_pairs(correlation, gap=0.3)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["a"], pairs[0]["b"]}, {"a", "b"})
        self.assertGreater(pairs[0]["gap"], 0.3)
        # cặp (a, c) tương quan cao nhưng Pearson ≈ Spearman ⇒ KHÔNG bị gắn cờ
        self.assertNotIn("c", {pairs[0]["a"], pairs[0]["b"]})

    def test_returns_empty_without_matrices(self):
        self.assertEqual(eda_lib.outlier_driven_pairs({}), [])
        self.assertEqual(eda_lib.outlier_driven_pairs({"feature_order": ["a", "b"]}), [])


class TestBasicConclusions(unittest.TestCase):
    """Nhận xét tự động phải nêu đúng số liệu và đúng việc cần làm."""

    def setUp(self):
        self.ratio_stats = [
            {"feature": "debt_to_equity_latest", "skew": -14.9, "max": 318.9,
             "iqr_outlier_pct": 30.6, "missing_pct": 0.0},
            {"feature": "current_ratio_latest", "skew": 0.9, "max": 2.98,
             "iqr_outlier_pct": 1.2, "missing_pct": 0.0},
        ]
        self.indicator_stats = [{"feature": "revenue", "iqr_outlier_pct": 14.5, "missing_pct": 0.0}]
        self.balance = {"train": {"distress": 132, "total": 212},
                        "test": {"distress": 38, "total": 64}}
        self.label_info = {"companies_all_one": ["HD", "LOW", "WMT"],
                           "label_share_by_ticker": {"HD": 1.0, "ROST": 0.05}}
        self.correlation = {
            "threshold": 0.8, "n_pairs_abs_ge_threshold": 1,
            "clusters_abs_ge_threshold": [["net_margin_latest", "operating_margin_latest"]],
            "feature_order": ["a", "b"],
            "pearson_matrix": [[1.0, -0.08], [-0.08, 1.0]],
            "spearman_matrix": [[1.0, 0.45], [0.45, 1.0]],
        }
        self.association = [{"feature": "debt_to_assets_latest", "auc": 0.88,
                             "effect_rank_biserial": 0.76, "direction": "giá trị cao ⇒ nhãn 1"}]

    def test_mentions_metric_choice_and_outlier_treatment(self):
        text = " ".join(eda_script.basic_conclusions(self.ratio_stats, self.indicator_stats,
                                                     self.balance, self.label_info,
                                                     self.correlation, self.association))
        self.assertIn("KHÔNG dùng Accuracy", text)
        self.assertIn("clip/winsorize", text)
        self.assertIn("62.3%", text)                       # % lớp dương tính từ balance
        self.assertIn("HD, LOW, WMT", text)                # cảnh báo rò rỉ cấp thực thể
        self.assertIn("ngoại lai", text)                   # cảnh báo tương quan giả
        self.assertIn("debt_to_assets", text)              # tỷ số tách lớp tốt nhất
        self.assertNotIn("_latest", text)                  # không lộ hậu tố kỹ thuật trong nhận xét

    def test_handles_empty_inputs_without_crashing(self):
        self.assertEqual(eda_script.basic_conclusions([], [], {}, {}, {}, []), [])


class TestLabelSharesFallback(unittest.TestCase):
    """Bản không vẽ hình vẫn phải tính được tỷ lệ nhãn theo công ty."""

    def test_computes_shares_and_single_class_lists(self):
        splits = {"train": [{"ticker": "AAA", "is_distressed": 1},
                            {"ticker": "AAA", "is_distressed": 1},
                            {"ticker": "BBB", "is_distressed": 0}],
                  "test": [{"ticker": "BBB", "is_distressed": 1}]}
        info = eda_script._label_shares(splits)
        self.assertAlmostEqual(info["label_share_by_ticker"]["AAA"], 1.0, places=6)
        self.assertAlmostEqual(info["label_share_by_ticker"]["BBB"], 0.5, places=6)
        self.assertEqual(info["companies_all_one"], ["AAA"])
        self.assertEqual(info["companies_all_zero"], [])


class TestFigures(unittest.TestCase):
    """3 hình mới phải vẽ được (smoke test) và không lỗi khi có cột toàn NaN."""

    def test_figures_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = pathlib.Path(tmp)
            X = np.column_stack([np.linspace(0, 1, 40), np.full(40, np.nan)])
            names = ["current_ratio_latest", "debt_to_equity_latest"]
            eda_script.fig_ratio_distributions(X, names, target / "d.png",
                                               {"current_ratio_latest": 5.0})
            eda_script.fig_ratio_correlation(np.array([[1.0, -0.7], [-0.7, 1.0]]), names,
                                             target / "c.png")
            eda_script.fig_ratio_vs_label([{"feature": "current_ratio_latest", "auc": 0.1,
                                            "effect_rank_biserial": -0.8}], target / "l.png")
            for name in ("d.png", "c.png", "l.png"):
                self.assertGreater((target / name).stat().st_size, 1000)


class TestRunEndToEnd(unittest.TestCase):
    """Chạy thật `scripts.eda.run` vào thư mục tạm — bỏ qua nếu chưa có data/prepared."""

    def test_run_writes_summary_with_three_new_blocks(self):
        if not (ROOT / "data" / "prepared" / "train.json").exists():
            self.skipTest("Chưa có data/prepared (chạy `python -m forecasting.data` trước).")
        with tempfile.TemporaryDirectory() as tmp:
            out, figs = pathlib.Path(tmp) / "results", pathlib.Path(tmp) / "figures"
            summary = eda_script.run(write=True, out_dir=out, fig_dir=figs)
            payload = json.loads((out / "eda_summary.json").read_text(encoding="utf-8"))
            markdown = (out / "eda.md").read_text(encoding="utf-8")

            # (1) thống kê mô tả: 14 tỷ số + 16 chỉ tiêu, có phân vị và % ngoại lai
            self.assertEqual(len(payload["ratio_stats"]), 14)
            self.assertEqual(len(payload["indicator_stats"]), 16)
            self.assertIn("## Thống kê mô tả 14 tỷ số", markdown)
            self.assertTrue(any(r.get("iqr_outlier_pct") is not None for r in payload["ratio_stats"]))
            # (2) tỉ lệ lớp theo % (không chỉ đếm)
            self.assertIn("% lớp 1", markdown)
            self.assertEqual(payload["splits"]["train"]["total"], 212)
            # (3) tương quan: giữa các tỷ số và với nhãn
            self.assertIn("## Tương quan giữa các tỷ số", markdown)
            self.assertIn("## Tương quan giữa tỷ số và NHÃN", markdown)
            self.assertEqual(len(payload["ratio_vs_label"]), 14)
            self.assertIn("outlier_driven_pairs", payload["ratio_correlation"])
            # nhận xét tự động + danh sách hình
            self.assertTrue(payload["conclusions"])
            self.assertIn("## Nhận xét", markdown)
            self.assertEqual(len(list(figs.glob("*.png"))), 9)
            self.assertEqual(len(payload["figures"]), 9)
            self.assertEqual(summary["corpus"]["n_samples"], 324)

    def test_no_write_and_no_figures_mode(self):
        if not (ROOT / "data" / "prepared" / "train.json").exists():
            self.skipTest("Chưa có data/prepared.")
        with tempfile.TemporaryDirectory() as tmp:
            out, figs = pathlib.Path(tmp) / "results", pathlib.Path(tmp) / "figures"
            summary = eda_script.run(write=False, out_dir=out, fig_dir=figs, figures=False)
            self.assertFalse((out / "eda.md").exists())
            self.assertEqual(list(figs.glob("*.png")), [])
            self.assertEqual(len(summary["ratio_stats"]), 14)          # vẫn tính được số liệu
            self.assertEqual(summary["label"]["companies_all_one"], ["HD", "LOW", "WMT"])
