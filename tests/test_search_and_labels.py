"""Verify hyperparameter search in `forecasting/search.py` and the alternative label rules.

Report sections 9.5 and 9.6 rest on these: the random-search sampling and its experiment ledger, plus the
two public label definitions, Altman Z'' and the forward four-quarter stress rule. A wrong label formula
would invalidate the whole label-sensitivity analysis.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.baselines import altman_z_probability  # noqa: E402
from forecasting.labels import (ALTMAN_DISTRESS_BELOW, FORWARD_HORIZON_QUARTERS, LABEL_RULES,  # noqa: E402
                                altman_z_double_prime, label_by_rule, label_forward_stress)
from forecasting.search import (SEARCH_SPACES, compare_with_grid, random_search, sample_params,  # noqa: E402
                                write_ledger)
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()


def row(**values) -> dict:
    """A VND data row with string values, matching the retail-expanded layout."""
    return {f"{key}_vnd": None if value is None else str(int(value)) for key, value in values.items()}


class TestAltmanZ(unittest.TestCase):
    """Altman Z'' must follow the formula and return None when a component is missing."""

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
    """The forward distress rule must look at the correct future window and handle a short tail."""

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
        label, info = label_forward_stress(rows, 3)          # window 4 to 7, with quarter 6 already bad
        self.assertEqual(label, 1)
        self.assertTrue(any(text.startswith("q+") for text in info))

    def test_does_not_fire_and_reports_observed_quarters(self):
        clean = self._series(20, bad_from=15)
        label, _ = label_forward_stress(clean, 3)            # window 4 to 7 is still clean
        self.assertEqual(label, 0)
        _, info = label_forward_stress(self._series(6), 3)   # only two quarters follow index 3
        self.assertIn("observed_quarters=2", info)

    def test_horizon_is_documented(self):
        self.assertEqual(FORWARD_HORIZON_QUARTERS, 4)
        rows = self._series(12, bad_from=8)     # quarters from index 8 onward are bad
        self.assertEqual(label_forward_stress(rows, 3, horizon=4)[0], 0)   # window 4 to 7 is clean
        self.assertEqual(label_forward_stress(rows, 4, horizon=4)[0], 1)   # window 5 to 8 has the bad one

    def test_all_rules_have_description_and_are_callable(self):
        for name, meta in LABEL_RULES.items():
            self.assertIn("description", meta)
            self.assertTrue(str(meta["description"]))
        self.assertIn("altman_z", LABEL_RULES)
        self.assertIn("forward_4q", LABEL_RULES)
        self.assertEqual(label_by_rule("stress_signals", row(net_income=-5), [], 0)[0], 1)
        with self.assertRaises(KeyError):
            label_by_rule("not-a-rule", {}, [], 0)


class TestSearchSampling(unittest.TestCase):
    """Configuration sampling: correct types, correct bounds and log-uniform scale parameters."""

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
            sample_params({"x": ("not-a-kind", 1)}, np.random.default_rng(0))


class TestRandomSearchAndLedger(unittest.TestCase):
    """Random search must run on small data and write one ledger row per trial."""

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
            random_search("not-a-model", self.X, self.y, self.groups, n_trials=1)

    def test_ledger_rows_match_trials(self):
        result = random_search("logistic", self.X, self.y, self.groups, n_trials=4, seed=2,
                               n_splits=3, space=self.space)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "runs.csv"
            written = write_ledger(path, result["trials"])
            self.assertEqual(written, len(result["trials"]))
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), written + 1)          # plus the header row
            self.assertIn("cv_average_precision", lines[0])

    def test_compare_with_grid_reports_delta(self):
        result = random_search("logistic", self.X, self.y, self.groups, n_trials=3, seed=3,
                               n_splits=3, space=self.space)
        comparison = compare_with_grid(result, 0.9)
        self.assertAlmostEqual(comparison["delta_vs_grid"],
                               result["best"]["cv_average_precision"] - 0.9, places=9)


class TestAltmanRuleBaseline(unittest.TestCase):
    """The Altman Z'' rule baseline from report section 6.1, which fits no parameters."""

    #: A healthy balance sheet in VND; only the ratios matter.
    HEALTHY = {"total_assets_vnd": "1000", "current_assets_vnd": "600",
               "current_liabilities_vnd": "200", "retained_earnings_vnd": "400",
               "operating_income_vnd": "150", "stockholders_equity_vnd": "700"}

    def _prob(self, row):
        sample = {"ticker": "X", "request": {"history": [row]}}
        return float(altman_z_probability([sample])[0])

    def test_weaker_balance_sheet_gives_higher_risk(self):
        weak = {**self.HEALTHY, "current_assets_vnd": "100", "retained_earnings_vnd": "-300"}
        self.assertLess(self._prob(self.HEALTHY), self._prob(weak))

    def test_missing_components_return_neutral_half(self):
        self.assertAlmostEqual(self._prob({}), 0.5, places=9)

    def test_rule_baseline_has_no_fitted_parameter(self):
        """With no fitted parameters, reordering the samples leaves the result unchanged."""
        rows = [self.HEALTHY, {}, {**self.HEALTHY, "current_liabilities_vnd": "900"}]
        forward = [self._prob(row) for row in rows]
        backward = [self._prob(row) for row in reversed(rows)][::-1]
        self.assertEqual(forward, backward)
