"""Verify the Streamlit demo in `app.py`, the presentation front end.

The suite asserts that all three tabs render without exceptions, that the app falls back to the simulation
model when `reports/models/best.joblib` is absent, that the displayed numbers match the artifacts, and that
a manual profile still produces all 47 features. The whole module is skipped when `streamlit` is missing.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime_warnings import quiet_library_warnings  # noqa: E402

try:
    import streamlit  # noqa: F401

    HAS_STREAMLIT = True
except Exception:  # pragma: no cover - CI has no streamlit
    HAS_STREAMLIT = False

SKIP_STREAMLIT = "streamlit is not installed (pip install -r requirements-app.txt)"

if HAS_STREAMLIT:  # pragma: no cover - only runs when streamlit is present
    from streamlit.testing.v1 import AppTest

    import app

    HAS_MODEL = app.MODEL_FILE.exists()
    SKIP_MODEL = "reports/models/best.joblib is missing; run `python -m forecasting.train`"
    APP_FILE = str(ROOT / "app.py")


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()
    if not HAS_STREAMLIT:
        raise unittest.SkipTest(SKIP_STREAMLIT)
    # Calling a Streamlit cache function outside the runtime logs a harmless WARNING, so the two loggers
    # are raised to ERROR to keep the test output readable.
    import logging

    for name in ("streamlit.runtime.caching.cache_data_api",
                 "streamlit.runtime.scriptrunner_utils.script_run_context"):
        logging.getLogger(name).setLevel(logging.ERROR)


@unittest.skipUnless(HAS_STREAMLIT, SKIP_STREAMLIT)
class TestAppRuns(unittest.TestCase):
    """Run the `app.py` script through AppTest, which needs no browser."""

    @classmethod
    def setUpClass(cls):
        cls.at = AppTest.from_file(APP_FILE, default_timeout=600)
        cls.at.run()

    def test_no_exception_and_three_tabs(self):
        self.assertEqual([exc.value for exc in self.at.exception], [])
        self.assertEqual(len(self.at.tabs), 3)

    def test_kpi_metrics_match_artifact(self):
        metrics = {metric.label: metric.value for metric in self.at.metric}
        self.assertEqual(len(metrics), 7)
        threshold = metrics["Ngưỡng vận hành"]
        self.assertNotEqual(threshold, "0,500")           # not the default 0.5 threshold
        self.assertTrue(threshold.startswith("0,78"))
        self.assertEqual(metrics["AUROC cross-company"], "0,933")

    def test_six_mock_cases_offered(self):
        options = self.at.sidebar.selectbox[0].options
        for sample_id in ("HD-2024Q2", "DG-2025Q2", "FIVE-2023Q3", "ROST"):
            self.assertTrue(any(sample_id in option for option in options), sample_id)

    def test_switching_mock_case_keeps_running(self):
        options = self.at.sidebar.selectbox[0].options
        target = next(option for option in options if "FIVE-2023Q3" in option)
        self.at.sidebar.selectbox[0].set_value(target).run()
        self.assertEqual([exc.value for exc in self.at.exception], [])
        badges = [str(item.value) for item in self.at.markdown if "P(suy giảm)" in str(item.value)]
        self.assertTrue(badges)
        self.assertIn("AN TOÀN", badges[0])                # P = 0.1325, below the 0.788 threshold

    def test_slider_mode_runs_and_flags_stress(self):
        at = AppTest.from_file(APP_FILE, default_timeout=600)
        at.run()
        radio = at.sidebar.radio[0]
        at.sidebar.radio[0].set_value(next(o for o in radio.options if str(o).startswith("B"))).run()
        self.assertEqual([exc.value for exc in at.exception], [])
        self.assertEqual(len(at.sidebar.slider), 7)        # six ratios plus the decision threshold
        for slider in at.sidebar.slider:
            if "current_ratio" in slider.label:
                slider.set_value(0.6)
            elif "Vốn lưu động" in slider.label:
                slider.set_value(-0.15)
            elif "Đòn bẩy" in slider.label:
                slider.set_value(1.30)
            elif "gộp" in slider.label:
                slider.set_value(0.15)
            elif "ròng" in slider.label:
                slider.set_value(-0.10)
            elif "Phải thu" in slider.label:
                slider.set_value(0.15)
        at.run()
        self.assertEqual([exc.value for exc in at.exception], [])
        badges = [str(item.value) for item in at.markdown if "P(suy giảm)" in str(item.value)]
        self.assertIn("NGUY CƠ SUY GIẢM TÀI CHÍNH", badges[0])


@unittest.skipUnless(HAS_STREAMLIT, SKIP_STREAMLIT)
class TestScoringAgainstArtifact(unittest.TestCase):
    """Score real samples with the actual model and feature pipeline."""

    @classmethod
    def setUpClass(cls):
        if not HAS_MODEL:
            raise unittest.SkipTest(SKIP_MODEL)

    def test_known_error_case_probability_matches_report(self):
        entry = app.sample_index().get("HD-2024Q2")
        self.assertIsNotNone(entry, "HD-2024Q2 is missing from data/prepared")
        _split, sample = entry
        result = app.score_sample(sample)
        self.assertEqual(result["source"], "model")
        self.assertEqual(result["n_features"], 47)
        self.assertAlmostEqual(result["p"], 0.7835, places=3)   # console log of analysis.md section 5

    def test_confusion_at_operating_threshold_matches_report(self):
        predictions = app.test_predictions()
        self.assertTrue(predictions["available"])
        self.assertEqual(len(predictions["ids"]), 64)
        counts = app.confusion_at(float(app.FALLBACK["threshold"]), predictions)
        self.assertEqual(counts, {"tp": 36, "tn": 25, "fn": 2, "fp": 1})
        self.assertAlmostEqual(app.expected_cost(counts), 11.0, places=6)
        # The best-F1 threshold on test flips HD-2024Q2 from a false negative to a true positive, leaving
        # the expected cost at 6.0 and matching `test_evaluation.json`.
        optimal = float(app.load_results()["threshold_best_f1_test"])
        counts_optimal = app.confusion_at(optimal, predictions)
        self.assertEqual(counts_optimal, {"tp": 37, "tn": 25, "fn": 1, "fp": 1})
        self.assertAlmostEqual(app.expected_cost(counts_optimal), 6.0, places=6)

    def test_kernel_shap_efficiency_holds(self):
        _split, sample = app.sample_index()["HD-2024Q2"]
        outcome = app.score_sample(sample)
        contributions = app.local_contributions(app.MOCK_CASES[0]["profile"], outcome, top_k=6)
        self.assertEqual(contributions["method"], "kernel_shap")
        self.assertAlmostEqual(contributions["base_value"], 0.6391, places=3)  # E[f] from shap.md
        self.assertLess(abs(contributions["gap"]), 1e-9)
        # Direction matches the published explanation: leverage raises risk, liquidity lowers it.
        top = dict(contributions["top"])
        self.assertGreater(top.get("debt_to_assets_latest", 0.0), 0.0)
        self.assertLess(top.get("current_ratio_latest", 0.0), 0.0)


@unittest.skipUnless(HAS_STREAMLIT, SKIP_STREAMLIT)
class TestSimulationFallback(unittest.TestCase):
    """With `best.joblib` absent the app must still run and label the result as simulated."""

    def setUp(self):
        self.original = app.MODEL_FILE

    def tearDown(self):
        app.MODEL_FILE = self.original
        app.load_artifact.clear()

    def test_score_profile_falls_back_to_simulation(self):
        app.MODEL_FILE = ROOT / "reports" / "models" / "definitely-missing.joblib"
        app.load_artifact.clear()
        self.assertIsNone(app.load_artifact())
        result = app.score_profile(dict(current_ratio=1.2, working_capital_to_assets=0.05,
                                        debt_to_assets=0.75, gross_margin=0.32, net_margin=0.08,
                                        receivables_to_sales=0.05))
        self.assertEqual(result["source"], "simulation")
        self.assertGreaterEqual(result["p"], 0.0)
        self.assertLessEqual(result["p"], 1.0)

    def test_simulation_is_monotone_in_leverage_and_liquidity(self):
        base = dict(current_ratio=1.2, working_capital_to_assets=0.05, debt_to_assets=0.75,
                    gross_margin=0.32, net_margin=0.08, receivables_to_sales=0.05)
        healthy = dict(base, current_ratio=2.5, debt_to_assets=0.45)
        stressed = dict(base, current_ratio=0.6, debt_to_assets=1.30)
        self.assertGreater(app.simulate_probability(stressed), app.simulate_probability(base))
        self.assertLess(app.simulate_probability(healthy), app.simulate_probability(base))

    def test_synthetic_profile_keeps_user_ratios(self):
        """A slider profile must reproduce exactly the ratios the user entered."""
        if not app.HAS_FORECASTING:
            self.skipTest("the forecasting package is missing")
        profile = dict(current_ratio=1.75, working_capital_to_assets=0.12, debt_to_assets=0.55,
                       gross_margin=0.40, net_margin=0.09, receivables_to_sales=0.07)
        sample = app.synth_sample(profile)
        self.assertEqual(len(sample["request"]["history"]), app.SYNTH_QUARTERS)
        matrix = app.build_feature_matrix([sample])
        names = [str(name) for name in app.feature_names()]
        for key, feature in app.SLIDER_TO_FEATURE.items():
            self.assertAlmostEqual(matrix[0, names.index(feature)], profile[key], places=6)