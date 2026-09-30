"""Kiểm thử tìm kiếm siêu tham số (`forecasting/search.py`) và các định nghĩa nhãn mới.

Chạy: python -m unittest tests.test_search_and_labels -v

Vì sao cần: mục 9.5–9.6 của báo cáo dựa vào hai thứ này — (a) sổ thực nghiệm/sampling của random
search, và (b) hai định nghĩa nhãn công khai (Altman Z'' và forward-4Q). Nếu công thức nhãn sai thì
toàn bộ phần kiểm chứng độ nhạy (RQ4) sai theo.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.labels import (ALTMAN_DISTRESS_BELOW, FORWARD_HORIZON_QUARTERS, LABEL_RULES,  # noqa: E402
                                altman_z_double_prime, label_by_rule, label_forward_stress)
from forecasting.search import (SEARCH_SPACES, compare_with_grid, random_search, sample_params,  # noqa: E402
                                write_ledger)
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - hook của unittest
    quiet_library_warnings()


def row(**values) -> dict:
    """Dòng dữ liệu VND dạng chuỗi (như trong retail-expanded)."""
    return {f"{key}_vnd": None if value is None else str(int(value)) for key, value in values.items()}


class TestAltmanZ(unittest.TestCase):
    """Altman Z'' phải tính đúng công thức và trả None khi thiếu thành phần."""

    def test_formula_matches_manual_computation(self):
        record = row(total_assets=1000, current_assets=400, current_liabilities=250,
                     retained_earnings=150, operating_income=120, stockholders_equity=300,
                     liabilities=700)
        expected = (6.56 * 150 / 1000 + 3.26 * 150 / 1000 + 6.72 * 120 / 1000
                    + 1.05 * 300 / 700)
        self.assertAlmostEqual(altman_z_double_prime(record), expected, places=9)

    def test_liabilities_derived_from_accounting_identity(self):
        with_tag = row(total_assets=1000, current_assets=400, current_liabilities=250,
                       retained_earnings=150, operating_income=120, stockholders_equity=300,
                       liabilities=700)
        derived = row(total_assets=1000, current_assets=400, current_liabilities=250,
                      retained_earnings=150, operating_income=120, stockholders_equity=300)
        self.assertAlmostEqual(altman_z_double_prime(with_tag),
                               altman_z_double_prime(derived), places=9)

    def test_missing_component_returns_none(self):
        self.assertIsNone(altman_z_double_prime(row(total_assets=1000)))

    def test_label_uses_documented_threshold(self):
        distress = row(total_assets=1000, current_assets=100, current_liabilities=900,
                       retained_earnings=-500, operating_income=-100, stockholders_equity=100,
                       liabilities=900)
        healthy = row(total_assets=1000, current_assets=600, current_liabilities=200,
                      retained_earnings=400, operating_income=200, stockholders_equity=700,
                      liabilities=300)
        self.assertEqual(label_by_rule("altman_z", distress, [], 0)[0], 1)
        self.assertEqual(label_by_rule("altman_z", healthy, [], 0)[0], 0)
        self.assertLess(ALTMAN_DISTRESS_BELOW, 2.0)


class TestForwardLabel(unittest.TestCase):
    """Nhãn 'distress trong 4 quý tới' phải nhìn ĐÚNG cửa sổ tương lai và xử lý đuôi chuỗi."""

    @staticmethod
    def _series(n: int, bad_from: int | None = None) -> list:
        rows = []
        for i in range(n):
            bad = bad_from is not None and i >= bad_from
            rows.append(row(total_assets=1000, current_assets=500, current_liabilities=100,
                            net_income=-1 if bad else 50, operating_cash_flow=-1 if bad else 50,
                            operating_income=-1 if bad else 50, revenue=1000))
        return rows

    def test_fires_when_a_future_quarter_is_bad(self):
        rows = self._series(12, bad_from=6)
        label, info = label_forward_stress(rows, 3)          # cửa sổ 4..7, quý 6 đã xấu
        self.assertEqual(label, 1)
        self.assertTrue(any(text.startswith("q+") for text in info))

    def test_does_not_fire_and_reports_observed_quarters(self):
        clean = self._series(20, bad_from=15)
        label, _ = label_forward_stress(clean, 3)            # cửa sổ 4..7 còn sạch
        self.assertEqual(label, 0)
        _, info = label_forward_stress(self._series(6), 3)   # chỉ còn 2 quý phía sau idx=3
        self.assertIn("observed_quarters=2", info)

    def test_horizon_is_documented(self):
        self.assertEqual(FORWARD_HORIZON_QUARTERS, 4)
        rows = self._series(12, bad_from=8)     # quý index 8 trở đi mới xấu
        self.assertEqual(label_forward_stress(rows, 3, horizon=4)[0], 0)   # cửa sổ 4..7 ⇒ sạch
        self.assertEqual(label_forward_stress(rows, 4, horizon=4)[0], 1)   # cửa sổ 5..8 ⇒ có quý 8

    def test_all_rules_have_description_and_are_callable(self):
        for name, meta in LABEL_RULES.items():
            self.assertIn("description", meta)
            self.assertTrue(str(meta["description"]))
        self.assertIn("altman_z", LABEL_RULES)
        self.assertIn("forward_4q", LABEL_RULES)
        self.assertEqual(label_by_rule("stress_signals", row(net_income=-5), [], 0)[0], 1)
        with self.assertRaises(KeyError):
            label_by_rule("khong-ton-tai", {}, [], 0)


class TestSearchSampling(unittest.TestCase):
    """Lấy mẫu cấu hình: đúng kiểu, đúng biên, log-uniform cho tham số scale."""

    def test_sampled_values_respect_bounds(self):
        rng = np.random.default_rng(0)
        for _space_name, space in SEARCH_SPACES.items():
            for _ in range(10):
                params = sample_params(space, rng)
                self.assertEqual(sorted(params), sorted(space))
                for key, spec in space.items():
                    value = params[key]
                    if spec[0] in ("loguniform", "float", "int"):
                        self.assertGreaterEqual(float(value), float(spec[1]))
                        self.assertLessEqual(float(value), float(spec[2]))
                    if spec[0] == "int":
                        self.assertIsInstance(value, int)
                    if spec[0] == "choice":
                        self.assertIn(value, list(spec[1]))

    def test_loguniform_covers_orders_of_magnitude(self):
        rng = np.random.default_rng(1)
        values = [sample_params({"C": ("loguniform", 1e-3, 10.0)}, rng)["C"] for _ in range(200)]
        self.assertGreater(max(values) / min(values), 100.0)

    def test_invalid_space_raises(self):
        with self.assertRaises(ValueError):
            sample_params({"x": ("khong-ton-tai", 1)}, np.random.default_rng(0))


class TestRandomSearchAndLedger(unittest.TestCase):
    """Random search phải chạy được trên dữ liệu nhỏ và ghi sổ đúng số dòng."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.groups = np.asarray(np.repeat([f"T{i}" for i in range(6)], 8))
        signal = rng.normal(size=48) + 2.0 * np.repeat([0, 1] * 3, 8)
        self.X = np.column_stack([signal, rng.normal(size=48)])
        self.y = np.tile([0, 1] * 3, 8)
        self.space = {"C": ("loguniform", 0.01, 1.0)}

    def test_search_runs_and_returns_best(self):
        result = random_search("logistic", self.X, self.y, self.groups, n_trials=5, seed=1,
                               n_splits=3, space=self.space)
        self.assertEqual(result["n_trials"], 5)
        self.assertIsNotNone(result["best"])
        self.assertGreater(result["best"]["cv_average_precision"], 0.5)
        self.assertIn("params", result["best"])

    def test_unknown_model_raises(self):
        with self.assertRaises(KeyError):
            random_search("khong-ton-tai", self.X, self.y, self.groups, n_trials=1)

    def test_ledger_rows_match_trials(self):
        result = random_search("logistic", self.X, self.y, self.groups, n_trials=4, seed=2,
                               n_splits=3, space=self.space)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "runs.csv"
            written = write_ledger(path, result["trials"])
            self.assertEqual(written, len(result["trials"]))
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), written + 1)          # + dòng header
            self.assertIn("cv_average_precision", lines[0])

    def test_compare_with_grid_reports_delta(self):
        result = random_search("logistic", self.X, self.y, self.groups, n_trials=3, seed=3,
                               n_splits=3, space=self.space)
        comparison = compare_with_grid(result, 0.9)
        self.assertAlmostEqual(comparison["delta_vs_grid"],
                               result["best"]["cv_average_precision"] - 0.9, places=9)
