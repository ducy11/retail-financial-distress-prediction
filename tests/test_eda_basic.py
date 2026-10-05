"""Verify the basic EDA in `scripts/eda.py` and its helpers in `forecasting/eda.py`.

Report sections 3.4 to 3.6 are read directly by a grader, so the unit conversion, the ratio selection and
the outlier-driven-correlation detection are exercised on hand-built data with known answers.
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


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()


def make_history_row(revenue=100.0, current_assets=200.0, current_liabilities=100.0,
                     inventory=50.0, cost_of_sales=60.0, total_assets=400.0,
                     stockholders_equity=200.0, **extra) -> dict:
    """A minimal history row with VND values as strings, matching the prepared layout."""
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
    """A minimal sample carrying enough indicators to build features."""
    return {
        "sample_id": f"{ticker}-2019Q1",
        "ticker": ticker,
        "request": {"history": history or [make_history_row()], "as_of": "2019-04-01",
                    "target_period_start": "2019-04-01", "target_period_end": "2019-06-30"},
        "is_distressed": label,
    }


class TestIndicatorValueMatrix(unittest.TestCase):
    """Source data for the descriptive statistics table, in trillions of VND."""

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
        self.assertAlmostEqual(matrix[0, 0], 4722.825, places=6)      # divided by 1e12
        self.assertAlmostEqual(matrix[0, 1], -1.0, places=6)
        self.assertTrue(np.isnan(matrix[0, 2]))                       # a missing value becomes NaN
        self.assertTrue(np.isnan(matrix[1, 1]))
        self.assertAlmostEqual(matrix[1, 0], 5.0, places=6)

    def test_default_fields_are_16_indicators(self):
        matrix, names = eda_lib.indicator_value_matrix(self.rows)
        self.assertEqual(len(names), 16)
        self.assertEqual(matrix.shape[0], 2)


class TestLatestRatioMatrix(unittest.TestCase):
    """The 14-ratio matrix used by the descriptive and correlation tables."""

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
        self.assertAlmostEqual(float(debt_to_assets), 0.5, places=6)   # (400 - 200) / 400


class TestOutlierDrivenPairs(unittest.TestCase):
    """Outlier-driven correlations must be detected from the Pearson and Spearman matrices."""

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
        # The pair (a, c) correlates highly but Pearson matches Spearman, so it is not flagged.
        self.assertNotIn("c", {pairs[0]["a"], pairs[0]["b"]})

    def test_returns_empty_without_matrices(self):
        self.assertEqual(eda_lib.outlier_driven_pairs({}), [])
        self.assertEqual(eda_lib.outlier_driven_pairs({"feature_order": ["a", "b"]}), [])


class TestBasicConclusions(unittest.TestCase):
    """The automatic commentary must state the right numbers and the right next action."""

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
                             "effect_rank_biserial": 0.76, "direction": "high value implies label 1"}]

    def test_mentions_metric_choice_and_outlier_treatment(self):
        text = " ".join(eda_script.basic_conclusions(self.ratio_stats, self.indicator_stats,
                                                     self.balance, self.label_info,
                                                     self.correlation, self.association))
        self.assertIn("KHÔNG dùng Accuracy", text)
        self.assertIn("clip/winsorize", text)
        self.assertIn("62.3%", text)                       # positive share from the balance table
        self.assertIn("HD, LOW, WMT", text)                # entity-level leakage warning
        self.assertIn("ngoại lai", text)                   # spurious-correlation warning
        self.assertIn("debt_to_assets", text)              # the best class-separating ratio
        self.assertNotIn("_latest", text)                  # the technical suffix must not leak

    def test_handles_empty_inputs_without_crashing(self):
        self.assertEqual(eda_script.basic_conclusions([], [], {}, {}, {}, []), [])


class TestLabelSharesFallback(unittest.TestCase):
    """The figures-free path must still compute the per-company label shares."""

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
    """The three figures must render, including when a column is entirely NaN."""

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
    """Run `scripts.eda.run` into a temporary directory, skipped when data/prepared is absent."""

    def test_run_writes_summary_with_three_new_blocks(self):
        if not (ROOT / "data" / "prepared" / "train.json").exists():
            self.skipTest("data/prepared is missing; run `python -m forecasting.data` first.")
        with tempfile.TemporaryDirectory() as tmp:
            out, figs = pathlib.Path(tmp) / "results", pathlib.Path(tmp) / "figures"
            summary = eda_script.run(write=True, out_dir=out, fig_dir=figs)
            payload = json.loads((out / "eda_summary.json").read_text(encoding="utf-8"))
            markdown = (out / "eda.md").read_text(encoding="utf-8")

            # 1. Descriptive statistics for the 14 ratios and the 16 indicators.
            self.assertEqual(len(payload["ratio_stats"]), 14)
            self.assertEqual(len(payload["indicator_stats"]), 16)
            self.assertIn("## Thống kê mô tả 14 tỷ số", markdown)
            self.assertTrue(any(r.get("iqr_outlier_pct") is not None for r in payload["ratio_stats"]))
            # 2. Class proportions in percent, not only counts.
            self.assertIn("% lớp 1", markdown)
            self.assertEqual(payload["splits"]["train"]["total"], 212)
            # 3. Correlation between ratios and with the label.
            self.assertIn("## Tương quan giữa các tỷ số", markdown)
            self.assertIn("## Tương quan giữa tỷ số và NHÃN", markdown)
            self.assertEqual(len(payload["ratio_vs_label"]), 14)
            self.assertIn("outlier_driven_pairs", payload["ratio_correlation"])
            # Automatic commentary and the figure list.
            self.assertTrue(payload["conclusions"])
            self.assertIn("## Nhận xét", markdown)
            self.assertEqual(len(list(figs.glob("*.png"))), 9)
            self.assertEqual(len(payload["figures"]), 9)
            self.assertEqual(summary["corpus"]["n_samples"], 324)

    def test_no_write_and_no_figures_mode(self):
        if not (ROOT / "data" / "prepared" / "train.json").exists():
            self.skipTest("data/prepared is missing.")
        with tempfile.TemporaryDirectory() as tmp:
            out, figs = pathlib.Path(tmp) / "results", pathlib.Path(tmp) / "figures"
            summary = eda_script.run(write=False, out_dir=out, fig_dir=figs, figures=False)
            self.assertFalse((out / "eda.md").exists())
            self.assertEqual(list(figs.glob("*.png")), [])
            self.assertEqual(len(summary["ratio_stats"]), 14)          # the numbers are still computed
            self.assertEqual(summary["label"]["companies_all_one"], ["HD", "LOW", "WMT"])
