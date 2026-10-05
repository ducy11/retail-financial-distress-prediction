"""Streamlit demo for the retail financial distress model.

Every displayed number is read from `reports/results/*.json` and `data/prepared/*.json`; the constants
below are fallbacks only. Without `reports/models/best.joblib` the app still renders and labels the
score as simulated.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

# Make the repository root importable when Streamlit starts this file from another working directory.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - depends on the environment
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:  # pragma: no cover
    HAS_PLOTLY = False

# Import the project package so the app can run the real pipeline; without it only simulation is available.
try:  # pragma: no cover - depends on the environment
    from forecasting.config import COST_FN, COST_FP, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
    from forecasting.data_loader import load_prepared
    from forecasting.explain import (DEFAULT_N_BACKGROUND, DEFAULT_N_COALITIONS,
                                     kernel_shap_values, pipeline_predict_fn)
    from forecasting.features import build_feature_matrix, feature_names, feature_rows

    HAS_FORECASTING = True
except Exception:  # pragma: no cover
    COST_FN, COST_FP, MODELS_DIR = 5.0, 1.0, ROOT / "reports" / "models"
    RESULTS_DIR, RANDOM_SEED = ROOT / "reports" / "results", 42
    DEFAULT_N_BACKGROUND, DEFAULT_N_COALITIONS = 40, 200
    HAS_FORECASTING = False

MODEL_FILE = MODELS_DIR / "best.joblib"
PREPARED_DIR = ROOT / "data" / "prepared"

# Course and author metadata rendered in the sidebar and the page header.
APP_TITLE = "Dự báo suy giảm tài chính (Financial Distress) — doanh nghiệp bán lẻ"
AUTHOR = "Nguyễn Đức Ý · MSSV 25730094 · Nhóm 18"
SUPERVISOR = "GVHD: Cáp Đình La Thăng"
COURSE = "CS114 — Máy học (Machine Learning)"
TICKERS: Tuple[str, ...] = ("WMT", "HD", "LOW", "ORLY", "DG", "FIVE", "DKS", "ROST")

#: Operating threshold: best-F1 of the committed model, selected on validation, never on test.
#: The cost-optimal threshold for 5*FN + FP is 0.783 on test and 0.788 on validation.
OPERATING_THRESHOLD_DEFAULT = 0.788

# Fallback metrics, used only while an artifact is missing; the values match the project report.
FALLBACK: Dict[str, Any] = {
    "model_name": "random_forest",
    "threshold": 0.7879126873178737,
    "test_n": 64,
    "test_positives": 38,
    "test_auroc": 0.9828,
    "test_ap": 0.9908,
    "test_brier": 0.0580,
    #: Operating threshold 0.788 on test gives TP=36, TN=25, FN=2, FP=1.
    "operating": {"tp": 36, "tn": 25, "fn": 2, "fp": 1,
                  "precision": 0.9730, "recall": 0.9474, "f1": 0.9600, "macro_f1": 0.9517},
    "threshold_best_f1_test": 0.7835,
    "in_domain_auroc": 0.9828,
    "in_domain_ap": 0.9908,
    "cross_company_auroc": 0.9335,
    "cross_company_ap": 0.9569,
    "loco_mean_auroc": 0.6280,
    "loco_n_evaluable": 8,
    "delong_p_model_vs_ticker_prior": 0.7546,
    #: Every system is scored on the same 64 test samples.
    "systems": {
        "model[random_forest]": {"auroc": 0.9828, "ap": 0.9908, "source": "mô hình chốt"},
        "ticker_prior": {"auroc": 0.9858, "ap": 0.9846,
                         "source": "baseline thực thể (chỉ biết tên công ty)"},
        "single_feature[debt_to_assets_latest]": {"auroc": 0.8887, "ap": 0.9379,
                                                  "source": "baseline 1 chỉ tiêu"},
        "rule[altman_z_double_prime<1.1]": {"auroc": 0.7581, "ap": 0.8648,
                                            "source": "quy tắc Altman Z'' (không học tham số)"},
        "dummy_most_frequent": {"auroc": 0.5000, "ap": 0.5938, "source": "đoán lớp đa số"},
    },
    #: Positive-label rate per company on TRAIN, which is the ticker_prior baseline.
    "ticker_prior_train": {"WMT": 1.000, "HD": 1.000, "LOW": 1.000, "ORLY": 0.862,
                           "DG": 0.353, "FIVE": 0.176, "DKS": 0.172, "ROST": 0.069},
    "ticker_prior_n_train": {"WMT": 33, "HD": 29, "LOW": 29, "ORLY": 29,
                             "DG": 17, "FIVE": 17, "DKS": 29, "ROST": 29},
    "shap_base_value": 0.6391,
    "shap_top": [("current_ratio_latest", 0.0765), ("current_ratio_min_window", 0.0764),
                 ("debt_to_assets_latest", 0.0583), ("working_capital_to_assets", 0.0561),
                 ("debt_to_equity_latest", 0.0356), ("quick_ratio_latest", 0.0230),
                 ("receivables_to_sales_latest", 0.0218), ("gross_margin_latest", 0.0212)],
}

# The six ratios exposed as sliders in manual mode, each mapped onto a *_latest model feature.
SLIDER_SPECS: Tuple[Dict[str, Any], ...] = (
    {"key": "current_ratio", "label": "Khả năng thanh toán ngắn hạn — current_ratio",
     "min": 0.5, "max": 3.0, "default": 1.2, "step": 0.01,
     "help": "Tài sản ngắn hạn / Nợ ngắn hạn. < 1 nghĩa là nợ ngắn hạn vượt tài sản ngắn hạn."},
    {"key": "working_capital_to_assets", "label": "Vốn lưu động / Tổng tài sản",
     "min": -0.2, "max": 0.4, "default": 0.05, "step": 0.005,
     "help": "(Tài sản ngắn hạn − Nợ ngắn hạn) / Tổng tài sản."},
    {"key": "debt_to_assets", "label": "Đòn bẩy — debt_to_assets",
     "min": 0.3, "max": 1.5, "default": 0.75, "step": 0.01,
     "help": "Nợ phải trả / Tổng tài sản. > 1 nghĩa là vốn chủ sở hữu đã âm."},
    {"key": "gross_margin", "label": "Biên lợi nhuận gộp — gross_margin",
     "min": 0.1, "max": 0.6, "default": 0.32, "step": 0.005,
     "help": "(Doanh thu − Giá vốn) / Doanh thu."},
    {"key": "net_margin", "label": "Biên lợi nhuận ròng — net_margin",
     "min": -0.2, "max": 0.3, "default": 0.08, "step": 0.005,
     "help": "Lợi nhuận sau thuế / Doanh thu. Âm là lỗ ròng."},
    {"key": "receivables_to_sales", "label": "Phải thu / Doanh thu",
     "min": 0.01, "max": 0.2, "default": 0.05, "step": 0.005,
     "help": "Phải thu khách hàng / Doanh thu — phản ánh chất lượng doanh thu."},
)

#: Slider key to model feature name; used by the evidence table and the explanations.
SLIDER_TO_FEATURE: Dict[str, str] = {
    "current_ratio": "current_ratio_latest",
    "working_capital_to_assets": "working_capital_to_assets",
    "debt_to_assets": "debt_to_assets_latest",
    "gross_margin": "gross_margin_latest",
    "net_margin": "net_margin_latest",
    "receivables_to_sales": "receivables_to_sales_latest",
}

#: Display labels for the features that appear most often in the local explanations.
FEATURE_LABELS: Dict[str, str] = {
    "current_ratio_latest": "current_ratio (quý mới nhất)",
    "current_ratio_min_window": "current_ratio (xấu nhất 8 quý)",
    "quick_ratio_latest": "quick_ratio (quý mới nhất)",
    "working_capital_to_assets": "Vốn lưu động / Tổng tài sản",
    "debt_to_assets_latest": "debt_to_assets (quý mới nhất)",
    "debt_to_equity_latest": "debt_to_equity (quý mới nhất)",
    "debt_to_assets_yoy": "Δ debt_to_assets so cùng kỳ",
    "gross_margin_latest": "Biên gộp (quý mới nhất)",
    "net_margin_latest": "Biên ròng (quý mới nhất)",
    "net_margin_min_window": "Biên ròng (xấu nhất 8 quý)",
    "operating_margin_latest": "Biên hoạt động (quý mới nhất)",
    "receivables_to_sales_latest": "Phải thu / Doanh thu",
    "inventory_to_sales_latest": "Tồn kho / Doanh thu",
    "ocf_to_sales_latest": "OCF / Doanh thu",
    "ocf_to_sales_min_window": "OCF / Doanh thu (xấu nhất 8 quý)",
    "retained_to_assets_latest": "Lợi nhuận giữ lại / Tài sản",
    "revenue_drawdown_window": "Mức giảm doanh thu so đỉnh 8 quý",
    "distress_quarters_in_window": "Số quý OCF âm trong 8 quý",
    "negative_ni_streak": "Số quý lỗ ròng liên tiếp",
    "negative_ocf_streak": "Số quý OCF âm liên tiếp",
    "sgna_pct_revenue_latest": "Chi phí bán hàng & QLDN / Doanh thu",
    "revenue_yoy_growth": "Tăng trưởng doanh thu YoY",
}

#: Fallback ratio profiles per mock case, used when `data/prepared/*` cannot be read.
#: The values are copied from the committed artifacts (`reports/results/analysis.md`, section 5).
MOCK_CASES: Tuple[Dict[str, Any], ...] = (
    {"id": "HD-2024Q2", "ticker": "HD", "split": "test", "actual": 1,
     "expected_p": 0.7835, "kind": "FN sát ngưỡng",
     "summary": "Nhãn thật = SUY GIẢM nhưng P = 0,7835 — chỉ thấp hơn ngưỡng vận hành 0,788 đúng "
                "0,44 điểm % (ca lỗi sát ngưỡng nhất của tập test).",
     "profile": {"current_ratio": 1.3392, "working_capital_to_assets": 0.1043,
                 "debt_to_assets": 0.9770, "gross_margin": 0.3414,
                 "net_margin": 0.0989, "receivables_to_sales": 0.1127}},
    {"id": "DG-2025Q2", "ticker": "DG", "split": "test", "actual": 0,
     "expected_p": 0.7975, "kind": "False Positive (báo động giả)",
     "summary": "Nhãn thật = BÌNH THƯỜNG nhưng P = 0,7975 — đòn bẩy 0,75; biên ròng 3,8%; "
                "OCF/doanh thu 8,1% nên hồ sơ *trông giống* doanh nghiệp căng thẳng.",
     "profile": {"current_ratio": 1.2331, "working_capital_to_assets": 0.0482,
                 "debt_to_assets": 0.7514, "gross_margin": 0.3096,
                 "net_margin": 0.0376, "receivables_to_sales": 0.0500}},
    {"id": "FIVE-2023Q3", "ticker": "FIVE", "split": "validation", "actual": 1,
     "expected_p": 0.1325, "kind": "FN rõ rệt (validation)",
     "summary": "Quý 'trông rất khoẻ' (current_ratio 1,71; đòn bẩy 0,59; biên gộp 34,9%) nhưng nhãn "
                "thật = SUY GIẢM ⇒ lỗi đến từ ĐỊNH NGHĨA NHÃN, không phải từ ngưỡng.",
     "profile": {"current_ratio": 1.7121, "working_capital_to_assets": 0.1303,
                 "debt_to_assets": 0.5943, "gross_margin": 0.3486,
                 "net_margin": 0.0617, "receivables_to_sales": 0.0500}},
    {"id": "ROST-2025Q4", "ticker": "ROST", "split": "test", "actual": 0,
     "expected_p": 0.0957, "kind": "AN TOÀN / Bình thường",
     "summary": "Ca an toàn của ROST — công ty có tỉ lệ nhãn dương thấp nhất (6,9% trên train): "
                "P = 0,0957, cách ngưỡng gần 69 điểm %.",
     "profile": {"current_ratio": 1.5199, "working_capital_to_assets": 0.1693,
                 "debt_to_assets": 0.6183, "gross_margin": 0.2800,
                 "net_margin": 0.0914, "receivables_to_sales": 0.0364}},
    {"id": "HD-2025Q4", "ticker": "HD", "split": "test", "actual": 1,
     "expected_p": 0.9805, "kind": "True Positive (điển hình)",
     "summary": "Ca dương tính rõ: current_ratio 1,05 và vốn lưu động/tài sản 0,016 ⇒ P = 0,9805.",
     "profile": {"current_ratio": 1.0509, "working_capital_to_assets": 0.0164,
                 "debt_to_assets": 0.8860, "gross_margin": 0.3341,
                 "net_margin": 0.0871, "receivables_to_sales": 0.1636}},
    {"id": "FIVE-2024Q4", "ticker": "FIVE", "split": "test", "actual": 0,
     "expected_p": 0.2419, "kind": "True Negative",
     "summary": "Biên ròng gần 0 (0,2%) nhưng thanh khoản tốt ⇒ mô hình vẫn kết luận an toàn.",
     "profile": {"current_ratio": 1.3848, "working_capital_to_assets": 0.0805,
                 "debt_to_assets": 0.6139, "gross_margin": 0.3058,
                 "net_margin": 0.0020, "receivables_to_sales": 0.0500}},
)

#: The three misclassified test samples analysed in tab 3; they match `reports/results/analysis.md`.
ERROR_CASES: Tuple[Dict[str, Any], ...] = (
    {"id": "HD-2024Q2", "ticker": "HD", "actual": 1, "probability": 0.7835, "error": "FN",
     "n_history": 37,
     "features": {"current_ratio_latest": 1.3392, "working_capital_to_assets": 0.1043,
                  "net_margin_latest": 0.0989, "debt_to_assets_latest": 0.9770,
                  "ocf_to_sales_latest": 0.1509, "receivables_to_sales_latest": 0.1127},
     "cause": "**Sát ngưỡng + quý 'trông vẫn khoẻ'.** Chỉ số của quý 2024Q2 hoàn toàn dương "
              "(current_ratio 1,34; WC/TA 0,104; biên ròng 9,9%; OCF/doanh thu 15,1%) nên mô hình "
              "chỉ đẩy P lên 0,7835 — **thiếu 0,0044 so với ngưỡng 0,788**. Đây là ca mà **ngưỡng tối "
              "ưu chi phí 0,783** chuyển thành dự đoán đúng (FN đắt gấp 5 lần FP)."},
    {"id": "DG-2025Q2", "ticker": "DG", "actual": 0, "probability": 0.7975, "error": "FP",
     "n_history": 29,
     "features": {"current_ratio_latest": 1.2331, "working_capital_to_assets": 0.0482,
                  "net_margin_latest": 0.0376, "debt_to_assets_latest": 0.7514,
                  "ocf_to_sales_latest": 0.0812, "receivables_to_sales_latest": None},
     "cause": "**Báo động giả do hồ sơ giống doanh nghiệp căng thẳng.** Đòn bẩy 0,751; biên ròng "
              "3,8%; OCF/doanh thu 8,1%; vốn lưu động/tài sản chỉ 0,048 ⇒ P = 0,7975 (hơn ngưỡng "
              "0,96 điểm %). Nhãn DG **biến động mạnh giữa các quý** (35,3% dương trên train nhưng "
              "54,8% trên toàn bộ 31 quý) nên cùng một hồ sơ có thể mang hai nhãn khác nhau."},
    {"id": "FIVE-2024Q3", "ticker": "FIVE", "actual": 1, "probability": 0.1745, "error": "FN",
     "n_history": 26,
     "features": {"current_ratio_latest": 1.6326, "working_capital_to_assets": 0.1080,
                  "net_margin_latest": 0.0398, "debt_to_assets_latest": 0.5994,
                  "ocf_to_sales_latest": 0.0858, "receivables_to_sales_latest": None},
     "cause": "**Lỗi định nghĩa nhãn, không phải lỗi ngưỡng.** Quý 2024Q3 của FIVE có thanh khoản "
              "tốt (current_ratio 1,63; WC/TA 0,108) và đòn bẩy thấp (0,60) ⇒ P = 0,1745, cách ngưỡng "
              "hơn 61 điểm %. Muốn 'đúng' ca này phải hạ ngưỡng xuống dưới 0,17 — khi đó gần như toàn "
              "bộ tập test thành dương tính ⇒ **không thể sửa bằng ngưỡng**. Cùng công ty, FIVE-2023Q3 "
              "cũng bị bỏ sót với P = 0,1325."},
)

def pretty_feature(name: str) -> str:
    """Return a display label for a feature, or the raw name when no label is defined."""
    return FEATURE_LABELS.get(name, str(name))


# Cached artifact loaders; every number the UI shows is produced or fetched through one of these.
def _load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return an empty mapping when it is missing or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource(show_spinner="Đang nạp mô hình đã chốt (best.joblib)…")
def load_artifact() -> Optional[Dict[str, Any]]:
    """Load `reports/models/best.joblib`, or return None when the model is absent or unreadable."""
    if not MODEL_FILE.exists():
        return None
    try:
        import joblib

        return joblib.load(MODEL_FILE)
    except Exception:
        return None


@st.cache_resource(show_spinner="Đang đọc dữ liệu prepared…")
def load_splits() -> Dict[str, List[Dict[str, Any]]]:
    """Load the four prepared splits, returning an empty mapping when they are unavailable."""
    if not HAS_FORECASTING:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name in ("train", "validation", "test", "purged"):
        try:
            out[name] = load_prepared(name)
        except Exception:
            continue
    return out


def _systems_table(significance: Dict[str, Any], baselines: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect the system comparison table over the same 64 test samples, significance file first."""
    rows: List[Dict[str, Any]] = []
    seen = set()
    for key, metric in (significance.get("systems") or {}).items():
        rows.append({"system": key, "auroc": metric.get("auroc"),
                     "ap": metric.get("average_precision"), "source": "kiểm định DeLong + bootstrap"})
        seen.add(key)
    for row in (baselines.get("rows") or []):
        name = str(row.get("baseline"))
        if name in seen:
            continue
        test = row.get("test") or {}
        rows.append({"system": name, "auroc": test.get("auroc"),
                     "ap": test.get("average_precision"), "source": "baseline riêng của đồ án"})
    if not rows:
        rows = [{"system": k, "auroc": v["auroc"], "ap": v["ap"], "source": v["source"]}
                for k, v in FALLBACK["systems"].items()]
    return rows


def _pick(analysis: Dict[str, Any], model: str, marker: str) -> Dict[str, Any]:
    """Look up one `headline` row of `analysis.json` by model name and protocol marker."""
    for row in (analysis.get("headline") or {}).get("rows") or []:
        system = str(row.get("system", ""))
        if model in system and marker in system:
            return row
    return {}


#: `analysis.json` is small and read once at import time for the headline lookups.
ANALYSIS: Dict[str, Any] = _load_json(RESULTS_DIR / "analysis.json")


@st.cache_data(show_spinner=False)
def load_results() -> Dict[str, Any]:
    """Merge `reports/results/*.json` into one normalised mapping, filling gaps from FALLBACK."""
    # 1. Read every published artifact, tolerating missing files.
    summary = _load_json(RESULTS_DIR / "summary.json")
    test_eval = _load_json(RESULTS_DIR / "test_evaluation.json")
    baselines = _load_json(RESULTS_DIR / "baselines.json")
    significance = _load_json(RESULTS_DIR / "significance.json")
    shap = _load_json(RESULTS_DIR / "shap.json")

    # 2. Resolve the committed model and its operating threshold.
    best_name = str(summary.get("best_model") or FALLBACK["model_name"])
    models = {str(m.get("model")): m for m in (summary.get("models") or [])}
    best_row = models.get(best_name) or {}
    threshold = float(summary.get("best_threshold") or FALLBACK["threshold"])

    # 3. Pull the headline in-domain, cross-company and LOCO figures.
    in_domain = _pick(ANALYSIS, best_name, "in-domain") or {}
    cross = _pick(ANALYSIS, best_name, "cross-company") or {}
    loco = (ANALYSIS.get("headline") or {}).get("loco_mean_auroc") or FALLBACK["loco_mean_auroc"]

    # 4. Find the DeLong p-value of the model against the ticker_prior baseline.
    model_key = f"model[{best_name}]"
    p_value = None
    for pair in (significance.get("pairs_vs_baseline") or []):
        if str(pair.get("b")) == "ticker_prior" and model_key in str(pair.get("a")):
            p_value = (pair.get("delong_auroc") or {}).get("p_value")
    # 5. Assemble the normalised mapping, preferring artifacts over fallbacks.
    metrics = (test_eval.get("test_metrics") or {})
    operating = metrics.get("operating") or {}
    n_test = metrics.get("n")

    return {
        "best_name": best_name,
        "threshold": threshold,
        "n_test": int(n_test or FALLBACK["test_n"]),
        "n_positive": int(round((metrics.get("observed_distress") or 0.0) * n_test))
                      if n_test else FALLBACK["test_positives"],
        "test_auroc": float(metrics.get("auroc") or in_domain.get("auroc") or FALLBACK["test_auroc"]),
        "test_ap": float(metrics.get("average_precision") or in_domain.get("average_precision")
                         or FALLBACK["test_ap"]),
        "test_brier": float(metrics.get("brier") or FALLBACK["test_brier"]),
        "threshold_best_f1_test": float((metrics.get("best_f1") or {}).get("threshold")
                                        or FALLBACK["threshold_best_f1_test"]),
        "operating": {k: operating.get(k) for k in
                      ("tp", "tn", "fn", "fp", "precision", "recall", "f1", "macro_f1")},
        "in_domain_auroc": float(in_domain.get("auroc") or FALLBACK["in_domain_auroc"]),
        "in_domain_ap": float(in_domain.get("average_precision") or FALLBACK["in_domain_ap"]),
        "cross_company_auroc": float(cross.get("auroc") or best_row.get("cross_company_auroc")
                                     or FALLBACK["cross_company_auroc"]),
        "cross_company_ap": float(cross.get("average_precision")
                                  or best_row.get("cross_company_ap")
                                  or FALLBACK["cross_company_ap"]),
        "cross_company_ap_select": best_row.get("cross_company_ap"),
        "cross_company_protocol": "GroupKFold giữ TRỌN công ty ra khỏi fold-train (324 mẫu)",
        "loco_mean_auroc": float(loco),
        "loco_n_evaluable": int((ANALYSIS.get("headline") or {}).get("loco_n_evaluable")
                                or FALLBACK["loco_n_evaluable"]),
        "delong_p": float(p_value) if p_value is not None
                    else float(FALLBACK["delong_p_model_vs_ticker_prior"]),
        "systems": _systems_table(significance, baselines),
        "ticker_prior": baselines.get("ticker_prior_train") or FALLBACK["ticker_prior_train"],
        "selection_rule": summary.get("selection_rule") or "",
        "shap_base_value": float(shap.get("base_value") or FALLBACK["shap_base_value"]),
        "shap_importance": [(r["feature"], r["mean_abs_shap"])
                            for r in (shap.get("importance_top") or [])] or FALLBACK["shap_top"],
        "shap_self_check": shap.get("self_check") or {},
        "sources": {"model": MODEL_FILE.exists(), "summary": bool(summary),
                    "test_evaluation": bool(test_eval), "baselines": bool(baselines),
                    "significance": bool(significance), "analysis": bool(ANALYSIS),
                    "shap": bool(shap)},
    }

# Weights of the simulated scorer that takes over when `best.joblib` is absent. This is not the project
# model; it only keeps the demo runnable and preserves the direction of risk.
SIMULATION_COEF: Dict[str, float] = {
    "current_ratio": -1.5,
    "working_capital_to_assets": -4.0,
    "debt_to_assets": 2.0,
    "gross_margin": -1.5,
    "net_margin": -3.0,
    "receivables_to_sales": 1.0,
}

# Build a prepared sample from the six slider ratios and feed it to the 47-column feature matrix.
#: Quarterly revenue of a synthetic profile; only a scaling anchor, because every feature is a ratio.
SYNTH_REVENUE = 1_000_000_000_000.0
#: Revenue growth of a synthetic profile: 2% per quarter, about 8.2% year over year.
SYNTH_GROWTH = 0.02
#: History length, long enough for the year-over-year feature and the eight-quarter extremes.
SYNTH_QUARTERS = 8
#: Reference values of the simulated scorer, close to the corpus medians.
SIMULATION_REFERENCE: Dict[str, float] = {
    "current_ratio": 1.40, "working_capital_to_assets": 0.10, "debt_to_assets": 0.65,
    "gross_margin": 0.33, "net_margin": 0.05, "receivables_to_sales": 0.05,
}
#: Intercept of the simulated scorer, tuned so that a median profile lands near 0.5.
SIMULATION_INTERCEPT = 0.45


def synth_sample(profile: Dict[str, float], sample_id: str = "HOSO-TUY-CHINH",
                 ticker: str = "SYNTH") -> Dict[str, Any]:
    """Rebuild one prepared sample from the six user-entered ratios for manual mode.

    Feature construction relies on ratios alone, so a normalised revenue history reproduces them exactly;
    the ratios the user does not enter come from fixed accounting assumptions flagged in the UI.
    """
    # 1. Unpack the six ratios the user controlled.
    cr = float(profile["current_ratio"])
    wca = float(profile["working_capital_to_assets"])
    dta = float(profile["debt_to_assets"])
    gm = float(profile["gross_margin"])
    nm = float(profile["net_margin"])
    rts = float(profile["receivables_to_sales"])

    # 2. Emit one quarter at a time, deriving every balance-sheet item from the ratios.
    rows: List[Dict[str, Any]] = []
    for t in range(SYNTH_QUARTERS):
        rev = SYNTH_REVENUE * (1.0 + SYNTH_GROWTH) ** (t - (SYNTH_QUARTERS - 1))
        cl = rev * 0.40                            # current liabilities fixed at 40% of revenue
        ca = cr * cl                               # so that current_ratio = ca / cl matches the slider
        ta = (ca - cl) / wca if abs(wca) > 5e-3 else 4.0 * rev
        if ta <= 0:                                # strongly negative working capital: keep assets positive
            ta = 4.0 * rev
        liabilities = dta * ta
        equity = ta - liabilities
        rows.append({
            "fiscal_year": 2023 + t // 4, "fiscal_quarter": 1 + t % 4,
            "period_start": f"p{t}", "period_end": f"q{t}", "available_on": f"a{t}",
            "currency": "VND",
            "revenue_vnd": str(int(rev)),
            "cost_of_sales_vnd": str(int(rev * (1.0 - gm))),
            "inventory_vnd": str(int(ca * 0.35)),
            "selling_general_admin_vnd": str(int(rev * 0.20)),
            "operating_cash_flow_vnd": str(int(rev * (nm + 0.05))),
            "total_assets_vnd": str(int(ta)),
            "cash_and_equivalents_vnd": str(int(ca * 0.20)),
            "operating_income_vnd": str(int(rev * (gm - 0.20))),
            "current_assets_vnd": str(int(ca)),
            "current_liabilities_vnd": str(int(cl)),
            "net_income_vnd": str(int(rev * nm)),
            "stockholders_equity_vnd": str(int(equity)),
            "liabilities_vnd": str(int(liabilities)),
            "retained_earnings_vnd": str(int(equity * 0.55)),
            "receivables_vnd": str(int(rev * rts)),
            "short_term_investments_vnd": None,
        })
    return {"sample_id": sample_id, "ticker": ticker,
            "request": {"history": rows, "as_of": "2025-12-31",
                        "target_period_start": "2026-01-01", "target_period_end": "2026-03-31",
                        "ratios": None},
            "is_distressed": 0, "label_available_on": "2026-05-01"}


def _ordered_matrix(artifact: Dict[str, Any], sample: Dict[str, Any]) -> np.ndarray:
    """Return a 1 x 47 feature matrix ordered exactly like the columns stored in the artifact."""
    X = build_feature_matrix([sample])
    order = artifact.get("features")
    names = [str(n) for n in feature_names()]
    if order and len(order) == X.shape[1] and all(str(n) in names for n in order):
        X = X[:, [names.index(str(n)) for n in order]]
    return X


def simulate_probability(profile: Dict[str, float]) -> float:
    """Score a ratio profile with the simulated logistic model used when `best.joblib` is missing.

    This is not the project model: it keeps the demo runnable and preserves the direction of risk, where
    thin liquidity, negative working capital, high leverage and narrow margins all raise the score.
    """
    z = SIMULATION_INTERCEPT
    for key, weight in SIMULATION_COEF.items():
        z += weight * (float(profile[key]) - SIMULATION_REFERENCE[key])
    return float(1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0))))


def score_profile(profile: Dict[str, float], artifact: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Score one ratio profile and report the probability, its source and the feature evidence."""
    artifact = artifact if artifact is not None else load_artifact()
    if artifact is not None and HAS_FORECASTING:
        try:
            sample = synth_sample(profile)
            matrix = _ordered_matrix(artifact, sample)
            probability = float(np.asarray(artifact["model"].predict_proba(matrix))[0, 1])
            rows = feature_rows([sample])[0]
            return {"p": probability, "source": "model", "sample": sample, "matrix": matrix,
                    "n_features": int(matrix.shape[1]),
                    "evidence": {k: rows.get(f) for k, f in SLIDER_TO_FEATURE.items()}}
        except Exception as exc:  # pragma: no cover - unexpected scoring failure
            return {"p": simulate_probability(profile), "source": "simulation", "sample": None,
                    "matrix": None, "n_features": 6,
                    "evidence": {k: float(profile[k]) for k in SLIDER_TO_FEATURE},
                    "error": f"{type(exc).__name__}: {exc}"}
    return {"p": simulate_probability(profile), "source": "simulation", "sample": None,
            "matrix": None, "n_features": 6,
            "evidence": {k: float(profile[k]) for k in SLIDER_TO_FEATURE},
            "error": None if artifact is None else "thiếu gói forecasting"}


# Local explanations: KernelSHAP against the real model, or a linearised approximation in simulation mode.
def _train_background() -> Optional[np.ndarray]:
    """Draw the training background of 40 samples, matching `scripts/explain_model.py` with seed 42.

    The seed keeps E[f] at the 0.6391 published in `reports/results/shap.md`.
    """
    if not HAS_FORECASTING:
        return None
    train = (load_splits().get("train") or [])
    if not train:
        return None
    try:
        X = build_feature_matrix(train)
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(len(X), size=min(DEFAULT_N_BACKGROUND, len(X)), replace=False))
        return X[idx]
    except Exception:  # pragma: no cover
        return None


def simulated_contributions(profile: Dict[str, float]) -> Dict[str, Any]:
    """Return the contributions of the simulated scorer, as log-odds terms rescaled to probability.

    This is a first-order approximation, so the contributions only sum to approximately P - P_base; the
    method label states that to keep the explanation from being misread.
    """
    z = SIMULATION_INTERCEPT
    terms: Dict[str, float] = {}
    for key, weight in SIMULATION_COEF.items():
        term = weight * (float(profile[key]) - SIMULATION_REFERENCE[key])
        terms[SLIDER_TO_FEATURE[key]] = term
        z += term
    p = float(1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0))))
    base = float(1.0 / (1.0 + math.exp(-SIMULATION_INTERCEPT)))
    scale = p * (1.0 - p)                       # logistic derivative at z (first-order approximation)
    phi = {name: term * scale for name, term in terms.items()}
    return {"phi": phi, "base_value": base, "prediction": p, "method": "logistic_mo_phong",
            "n_coalitions": 0, "gap": (p - base) - sum(phi.values())}


def _shap_vector(values: Tuple[float, ...], n_coalitions: int) -> Dict[str, Any]:
    """Compute KernelSHAP values for a single feature vector; the caller caches the result."""
    artifact = load_artifact()
    background = _train_background()
    if artifact is None or background is None:
        return {"phi": {}, "base_value": float("nan"), "prediction": float("nan"),
                "method": "unavailable", "n_coalitions": 0, "gap": float("nan")}
    vector = np.asarray(values, dtype=float).reshape(1, -1)
    predict_fn = pipeline_predict_fn(artifact["model"])
    result = kernel_shap_values(predict_fn, background, vector[0], n_coalitions=n_coalitions)
    names = [str(n) for n in feature_names()]
    order = artifact.get("features")
    if order and len(order) == len(names) and all(str(n) in names for n in order):
        names = [str(n) for n in order]
    return {"phi": {name: float(value) for name, value in zip(names, result["phi"])},
            "base_value": float(result["base_value"]), "prediction": float(result["prediction"]),
            "method": "kernel_shap", "n_coalitions": int(result["n_coalitions"]),
            "gap": float(result["efficiency_gap"])}


@st.cache_data(show_spinner="Đang tính KernelSHAP cho hồ sơ này…")
def cached_shap(values: Tuple[float, ...],
                n_coalitions: int = DEFAULT_N_COALITIONS) -> Dict[str, Any]:
    """Cache `_shap_vector` per vector, with values rounded so repeated reruns hit the cache."""
    return _shap_vector(values, n_coalitions)


def local_contributions(profile: Dict[str, float], result: Dict[str, Any],
                        top_k: int = 10) -> Dict[str, Any]:
    """Compute local contributions: KernelSHAP for the real model, otherwise the simulated terms."""
    matrix = result.get("matrix")
    if matrix is not None and result.get("source") == "model":
        values = tuple(round(float(v), 6) for v in np.ravel(matrix))
        out = dict(cached_shap(values))
        out["top"] = sorted((out.get("phi") or {}).items(), key=lambda kv: -abs(kv[1]))[:top_k]
        out["method_label"] = (f"KernelSHAP tự cài đặt · {out['n_coalitions']} liên minh · "
                               f"nền 40 mẫu train (E[f] = {out['base_value']:.4f})")
        return out
    out = simulated_contributions(profile)
    out["top"] = sorted(out["phi"].items(), key=lambda kv: -abs(kv[1]))[:top_k]
    out["method_label"] = ("Xấp xỉ tuyến tính hoá của MÔ HÌNH MÔ PHỎNG (không phải KernelSHAP trên "
                           f"mô hình thật) · P_nền = {out['base_value']:.4f}")
    return out


# Mode A: resolve a sample from `data/prepared` and score it with its own 47 stored features.
def sample_index() -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Map every readable `sample_id` to its split name and sample record."""
    index: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for name, rows in load_splits().items():
        for sample in rows:
            index[str(sample.get("sample_id"))] = (name, sample)
    return index


def profile_from_sample(sample: Dict[str, Any]) -> Dict[str, float]:
    """Extract the six ratios of one stored sample, for evidence display and fallback scoring."""
    profile: Dict[str, float] = {}
    if not HAS_FORECASTING:
        return profile
    try:
        rows = feature_rows([sample])[0]
    except Exception:  # pragma: no cover
        return profile
    for key, feature in SLIDER_TO_FEATURE.items():
        value = rows.get(feature)
        if value is None or not np.isfinite(value):
            value = SIMULATION_REFERENCE[key]
        profile[key] = float(value)
    return profile


def score_sample(sample: Dict[str, Any], artifact: Optional[Dict[str, Any]] = None
                 ) -> Dict[str, Any]:
    """Score a stored sample without rebuilding its history, using its 47 prepared features."""
    artifact = artifact if artifact is not None else load_artifact()
    expected = MOCK_CASES[0]["expected_p"]
    if artifact is not None and HAS_FORECASTING:
        try:
            matrix = _ordered_matrix(artifact, sample)
            probability = float(np.asarray(artifact["model"].predict_proba(matrix))[0, 1])
            rows = feature_rows([sample])[0]
            return {"p": probability, "source": "model", "matrix": matrix,
                    "n_features": int(matrix.shape[1]),
                    "evidence": {k: rows.get(f) for k, f in SLIDER_TO_FEATURE.items()},
                    "n_history": int(rows.get("n_history") or 0)}
        except Exception as exc:  # pragma: no cover
            profile = profile_from_sample(sample)
            return {"p": simulate_probability(profile) if profile else expected,
                    "source": "simulation", "matrix": None, "n_features": 6,
                    "evidence": profile, "n_history": len(sample["request"]["history"]),
                    "error": f"{type(exc).__name__}: {exc}"}
    profile = profile_from_sample(sample)
    return {"p": simulate_probability(profile) if profile else expected, "source": "simulation",
            "matrix": None, "n_features": 6, "evidence": profile,
            "n_history": len(sample["request"]["history"]), "error": None}


@st.cache_data(show_spinner="Đang chấm lại toàn bộ tập test bằng mô hình đã chốt…")
def test_predictions() -> Dict[str, Any]:
    """Return the committed model probabilities for the 64 test samples, used to recount confusion."""
    artifact = load_artifact()
    rows = load_splits().get("test") or []
    if artifact is None or not rows or not HAS_FORECASTING:
        return {"available": False, "ids": [], "y": [], "p": []}
    try:
        matrix = build_feature_matrix(rows)
        order = artifact.get("features")
        names = [str(n) for n in feature_names()]
        if order and len(order) == matrix.shape[1] and all(str(n) in names for n in order):
            matrix = matrix[:, [names.index(str(n)) for n in order]]
        p = np.asarray(artifact["model"].predict_proba(matrix))[:, 1]
        return {"available": True, "ids": [str(s["sample_id"]) for s in rows],
                "y": [int(s["is_distressed"]) for s in rows], "p": [float(v) for v in p]}
    except Exception:  # pragma: no cover
        return {"available": False, "ids": [], "y": [], "p": []}


def confusion_at(threshold: float, predictions: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """Recount the test confusion matrix at the threshold currently displayed."""
    data = predictions if predictions is not None else test_predictions()
    counts = {"tp": 0, "tn": 0, "fn": 0, "fp": 0}
    if not data.get("available"):
        return counts
    for y_true, score in zip(data["y"], data["p"]):
        predicted = int(score >= threshold)
        if y_true == 1 and predicted == 1:
            counts["tp"] += 1
        elif y_true == 0 and predicted == 0:
            counts["tn"] += 1
        elif y_true == 1:
            counts["fn"] += 1
        else:
            counts["fp"] += 1
    return counts


def expected_cost(counts: Dict[str, int]) -> float:
    """Return the expected cost, where one false negative costs five false positives."""
    return COST_FN * counts.get("fn", 0) + COST_FP * counts.get("fp", 0)


def labels_by_ticker() -> Dict[str, List[Tuple[str, int]]]:
    """Return the true label of every quarter per company, which exposes label instability."""
    out: Dict[str, List[Tuple[str, int]]] = {}
    for _split, rows in load_splits().items():
        for sample in rows:
            out.setdefault(str(sample["ticker"]), []).append(
                (str(sample["sample_id"]), int(sample["is_distressed"])))
    for ticker in out:
        out[ticker].sort(key=lambda item: item[0])
    return out


# Plotly figures; each builder returns None when plotly is unavailable so callers can fall back.
RED, GREEN, GREY = "#d62728", "#2ca02c", "#7f7f7f"
PLOT_CONFIG = {"displayModeBar": False}


def show_plotly(figure: Any) -> None:
    """Render a Plotly figure full width, supporting both Streamlit width APIs."""
    try:
        st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG)
    except TypeError:  # older Streamlit without the `width` argument
        st.plotly_chart(figure, use_container_width=True, config=PLOT_CONFIG)


def show_table(rows: Any, **kwargs: Any) -> None:
    """Render a dataframe full width, supporting both Streamlit width APIs."""
    try:
        st.dataframe(rows, width="stretch", **kwargs)
    except TypeError:  # pragma: no cover - older Streamlit
        st.dataframe(rows, use_container_width=True, **kwargs)


def show_bar(mapping: Any) -> None:
    """Render the fallback bar chart used when plotly is unavailable."""
    try:
        st.bar_chart(mapping, width="stretch")
    except TypeError:  # pragma: no cover - older Streamlit
        st.bar_chart(mapping)


def gauge_figure(probability: float, threshold: float) -> Optional[Any]:
    """Build the probability gauge, with the active threshold drawn as a marker."""
    if not HAS_PLOTLY:
        return None
    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"P(suy giảm tài chính)<br><span style='font-size:0.8em'>ngưỡng quyết định "
                       f"= {threshold:.3f}</span>", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": RED if probability >= threshold else GREEN},
            "steps": [{"range": [0, threshold * 100], "color": "#e8f5e9"},
                      {"range": [threshold * 100, 100], "color": "#ffebee"}],
            "threshold": {"line": {"color": "black", "width": 3},
                          "thickness": 0.85, "value": threshold * 100},
        }))
    figure.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return figure


def contributions_figure(items: Sequence[Tuple[str, float]], base_value: float,
                         prediction: float) -> Optional[Any]:
    """Build the horizontal contribution chart; red pushes towards distress, green away from it."""
    if not HAS_PLOTLY:
        return None
    labels = [pretty_feature(name) for name, _ in items][::-1]
    values = [value for _, value in items][::-1]
    colors = [RED if value > 0 else GREEN for value in values]
    figure = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                              text=[f"{value:+.4f}" for value in values], textposition="auto"))
    figure.update_layout(
        title=(f"Đóng góp cục bộ (φ) — nền E[f] = {base_value:.4f} → dự báo = {prediction:.4f} "
               f"· đỏ = đẩy về suy giảm, xanh = kéo về an toàn"),
        xaxis_title="φ (đơn vị: xác suất)", yaxis_title=None,
        height=max(320, 36 * len(labels)), margin=dict(l=10, r=10, t=70, b=10), bargap=0.25)
    figure.add_vline(x=0, line_width=1, line_color="black")
    return figure


def systems_figure(rows: Sequence[Dict[str, Any]], model_key: str,
                   n_test: int) -> Optional[Any]:
    """Build the AUROC and average-precision comparison over the same test samples."""
    if not HAS_PLOTLY:
        return None
    labels = [str(r["system"]).replace("model[", "").replace("]", "")
              .replace("single_feature", "1 chỉ tiêu").replace("rule[", "quy tắc ").replace("]", "")
              for r in rows]
    auroc = [r["auroc"] if r["auroc"] is not None else float("nan") for r in rows]
    ap = [r["ap"] if r["ap"] is not None else float("nan") for r in rows]
    colors = [RED if str(r["system"]).startswith("model[") else GREY for r in rows]
    figure = go.Figure()
    figure.add_trace(go.Bar(x=labels, y=auroc, name="AUROC", marker_color=colors,
                            text=[f"{v:.3f}" for v in auroc], textposition="outside"))
    figure.add_trace(go.Bar(x=labels, y=ap, name="Average Precision", marker_color="#1f77b4",
                            text=[f"{v:.3f}" for v in ap], textposition="outside", opacity=0.5))
    figure.update_layout(title=f"So sánh trên CÙNG {n_test} mẫu test · đỏ = mô hình chốt ({model_key})",
                         yaxis=dict(range=[0, 1.1], title="Điểm số"), barmode="group", height=400,
                         margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h"))
    return figure


def leakage_figure(info: Dict[str, Any]) -> Optional[Any]:
    """Build the leakage chart: in-domain, cross-company, LOCO and the ticker-prior baseline."""
    if not HAS_PLOTLY:
        return None
    prior = next((r["auroc"] for r in info["systems"] if str(r["system"]) == "ticker_prior"), None)
    labels = ["Mô hình chốt<br>in-domain (train→test)",
              "Mô hình chốt<br>cross-company (GroupKFold)",
              f"LOCO trung bình<br>({info['loco_n_evaluable']} công ty, mỗi lần 1 công ty ra ngoài)",
              "Baseline ticker-prior<br>(chỉ biết TÊN công ty)"]
    values = [info["in_domain_auroc"], info["cross_company_auroc"], info["loco_mean_auroc"],
              float(prior) if prior is not None else float("nan")]
    figure = go.Figure(go.Bar(x=labels, y=values, marker_color=["#1f77b4", RED, "#ff7f0e", GREY],
                              text=[f"{value:.4f}" for value in values], textposition="outside"))
    figure.update_layout(title="AUROC: điểm in-domain bị thổi phồng bởi rò rỉ cấp thực thể",
                         yaxis=dict(range=[0, 1.1], title="AUROC"), height=400,
                         margin=dict(l=10, r=10, t=60, b=10))
    figure.add_hline(y=0.5, line_dash="dot", line_color="black",
                     annotation_text="đoán ngẫu nhiên (0,5)")
    return figure


def confusion_figure(counts: Dict[str, int], threshold: float,
                     subtitle: str) -> Optional[Any]:
    """Build the confusion-matrix heatmap for the threshold under inspection."""
    if not HAS_PLOTLY:
        return None
    matrix = [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]]
    text = [[f"TN = {counts['tn']}", f"FP = {counts['fp']}<br>(báo động giả)"],
            [f"FN = {counts['fn']}<br>(bỏ sót ⚠)", f"TP = {counts['tp']}"]]
    figure = go.Figure(go.Heatmap(z=matrix, x=["Dự báo: An toàn", "Dự báo: Suy giảm"],
                                  y=["Thực tế: An toàn", "Thực tế: Suy giảm"],
                                  text=text, texttemplate="%{text}", colorscale="Blues",
                                  showscale=False, textfont={"size": 15}))
    figure.update_layout(title=f"Confusion matrix @ ngưỡng {threshold:.3f} — {subtitle}",
                         height=330, margin=dict(l=10, r=10, t=70, b=10),
                         yaxis=dict(autorange="reversed"))
    return figure


def label_strip_figure(items: Sequence[Tuple[str, int]], title: str) -> Optional[Any]:
    """Build the per-quarter label strip of one company, which shows label drift over time."""
    if not HAS_PLOTLY:
        return None
    labels = [sample_id for sample_id, _ in items]
    values = [label for _, label in items]
    figure = go.Figure(go.Bar(x=labels, y=values,
                              marker_color=[RED if value else GREEN for value in values],
                              text=["suy giảm" if value else "bình thường" for value in values],
                              textposition="outside", hovertext=labels))
    figure.update_layout(title=title, yaxis=dict(range=[0, 1.35], tickvals=[0, 1],
                                                 ticktext=["0 = bình thường", "1 = suy giảm"]),
                         height=300, margin=dict(l=10, r=10, t=70, b=10), showlegend=False)
    return figure


# Formatting helpers shared by the tables and the status badge.
def fmt(value: Any, digits: int = 3) -> str:
    """Format a number with a decimal comma, returning an em dash for missing or infinite values."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}".replace(".", ",")


def badge_html(probability: float, threshold: float) -> str:
    """Render the status badge, red for distress and green for safety, with the gap to the threshold."""
    distress = probability >= threshold
    background, border, label = ((("#fdecea", "#b71c1c", "🔴 NGUY CƠ SUY GIẢM TÀI CHÍNH"))
                                 if distress else
                                 ("#e8f5e9", "#1b5e20", "🟢 AN TOÀN / BÌNH THƯỜNG"))
    gap = abs(probability - threshold)
    direction = "trên" if distress else "dưới"
    return (
        f"<div style='background:{background};border-left:10px solid {border};padding:14px 18px;"
        f"border-radius:8px'>"
        f"<div style='font-size:1.35rem;font-weight:700;color:{border}'>{label}</div>"
        f"<div style='font-size:1.02rem;color:#222;margin-top:4px'>"
        f"P(suy giảm) = <b>{fmt(probability)}</b> · ngưỡng = <b>{fmt(threshold)}</b> ⇒ "
        f"<b>{fmt(gap)}</b> điểm {direction} ngưỡng · quyết định: "
        f"<b>{'1 (dự báo suy giảm)' if distress else '0 (dự báo an toàn)'}</b></div></div>")


def entity_prior(ticker: Optional[str], prior: Dict[str, Any]) -> Optional[float]:
    """Return the share of positive labels on TRAIN for a ticker, when one is selected."""
    if not ticker:
        return None
    value = prior.get(str(ticker))
    return float(value) if value is not None else None


def entity_warning(ticker: Optional[str], prior: Dict[str, Any],
                   n_train: Dict[str, Any]) -> None:
    """Warn about entity leakage for companies whose historical label rate is extreme."""
    rate = entity_prior(ticker, prior)
    if rate is None:
        st.info("Chế độ nhập tay: chưa gắn với công ty cụ thể ⇒ chưa xét được cảnh báo rò rỉ thực thể. "
                "Chọn 'công ty tham chiếu' ở sidebar nếu muốn đối chiếu.")
        return
    n = n_train.get(str(ticker))
    if rate >= 0.95:
        st.error(
            f"⚠️ **Cảnh báo rò rỉ cấp thực thể:** `{ticker}` có **{rate:.1%}** quý mang nhãn suy giảm "
            f"trong TRAIN (n = {n}) — gần như **mọi quý của công ty này đều là 'suy giảm'**. Với công ty "
            f"như vậy, một quy tắc 'chỉ cần biết tên công ty' (`ticker_prior`) đã đạt AUROC 0,986, tương "
            f"đương mô hình học máy ⇒ **kết quả dự báo ở đây KHÔNG chứng minh mô hình học được tín hiệu "
            f"tài chính**; nó phản ánh danh tính công ty. Hãy đọc kèm tab *Đối chiếu Baseline & "
            f"Thực nghiệm* (cross-company AUROC = {fmt(0.9335)}).")
    elif rate <= 0.10:
        st.warning(
            f"⚠️ **Cảnh báo rò rỉ cấp thực thể (chiều ngược):** `{ticker}` chỉ có **{rate:.1%}** quý "
            f"nhãn suy giảm trong TRAIN (n = {n}) ⇒ công ty này gần như luôn 'an toàn'; dự báo 'an toàn' "
            f"ở đây có thể chỉ là **học thuộc danh tính công ty**, không phải vì hồ sơ tài chính tốt.")
    else:
        st.info(
            f"`{ticker}` có tỉ lệ nhãn suy giảm **{rate:.1%}** trên train (n = {n}) — nhãn trải trên cả "
            f"hai lớp, nên kết quả dự báo của công ty này *ít bị* chi phối bởi danh tính công ty.")


def render_company_prior_table(prior: Dict[str, Any], n_train: Dict[str, Any]) -> None:
    """Show the per-company positive-label rate as evidence of entity leakage."""
    rows = [{"Công ty": ticker, "Số quý train": n_train.get(ticker, "—"),
             "Tỉ lệ nhãn suy giảm (train)": f"{float(prior[ticker]):.1%}"}
            for ticker in sorted(prior, key=lambda t: -float(prior[t]))]
    show_table(rows, hide_index=True)
    st.caption("Đọc bảng: 3/8 công ty (WMT, HD, LOW) có **100%** quý là suy giảm ⇒ nhãn gần như là "
               "**thuộc tính của công ty**, không phải của quý. Đây chính là cơ chế của rò rỉ thực thể.")


def render_sidebar(info: Dict[str, Any], artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Render the sidebar: project metadata, input mode selection and the decision threshold.

    Returns the resolved UI state: mode, threshold, selected case, ratio profile and reference ticker.
    """
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center'>"
            "<div style='font-size:2.6rem;line-height:1'>🛒&nbsp;📉</div>"
            "<div style='font-weight:800;font-size:1.12rem;margin-top:6px'>Dự báo suy giảm tài chính"
            "<br>doanh nghiệp bán lẻ</div>"
            "<div style='font-size:0.82rem;color:#666'>8 chuỗi bán lẻ Mỹ · dữ liệu SEC XBRL · "
            "dự báo trước 1 quý</div></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"**👤 {AUTHOR}**  \n{SUPERVISOR}  \n{COURSE}")
        if artifact is not None:
            st.success(f"✅ Đã nạp mô hình chốt `{artifact.get('name', 'best')}` "
                       f"({MODEL_FILE.name}, {len(artifact.get('features') or [])} feature).", icon="🧠")
        else:
            st.warning("⚠️ Chưa có `reports/models/best.joblib` ⇒ app chạy **chế độ MÔ PHỎNG** "
                       "(logistic trên 6 tỷ số), không phải mô hình thật. Chạy "
                       "`python -m forecasting.train` để sinh mô hình.", icon="⚠️")
        st.divider()

        mode_label = st.radio(
            "Chế độ nhập liệu",
            ("A · Chọn nhanh ca kiểm thử mẫu", "B · Tự nhập chỉ số tài chính"),
            index=0, key="mode")
        mode = "A" if mode_label.startswith("A") else "B"

        threshold = st.slider(
            "Ngưỡng quyết định (Decision Threshold)", min_value=0.05, max_value=0.95,
            value=float(OPERATING_THRESHOLD_DEFAULT), step=0.001, format="%.3f", key="threshold")
        st.caption(f"Ngưỡng **0,788** tối ưu F1 (chọn trên validation, không dùng test); ngưỡng tối ưu "
                   f"chi phí 5·FN + FP là **0,783**. Ngưỡng 0,5 là mặc định của sklearn — đồ án "
                   f"**không** dùng.")

        case: Dict[str, Any] = {}
        profile: Dict[str, float] = {}
        reference_ticker: Optional[str] = None

        if mode == "A":
            labels = [f"{c['id']} — {c['kind']}" for c in MOCK_CASES]
            picked = st.selectbox("Ca kiểm thử mẫu", labels, index=0, key="mock")
            case = dict(MOCK_CASES[labels.index(picked)])
            profile = dict(case["profile"])
            reference_ticker = case["ticker"]
            st.caption(case["summary"])
            with st.expander("Hoặc chọn bất kỳ mẫu nào trong `data/prepared`"):
                index = sample_index()
                if not index:
                    st.caption("Chưa nạp được `data/prepared/*.json` ⇒ chỉ dùng được các ca mẫu có sẵn.")
                else:
                    tickers = sorted({sample["ticker"] for _split, sample in index.values()})
                    ticker = st.selectbox("Công ty", tickers, key="custom_ticker")
                    ids = sorted([sid for sid, (_sp, s) in index.items()
                                  if s["ticker"] == ticker], reverse=True)
                    sample_id = st.selectbox("Quý / mẫu", ids, key="custom_quarter")
                    use_custom = st.toggle("Dùng mẫu đã chọn thay cho ca mẫu ở trên", key="use_custom")
                    if use_custom:
                        split, sample = index[sample_id]
                        case = {"id": sample_id, "ticker": ticker, "split": split,
                                "actual": int(sample["is_distressed"]),
                                "expected_p": None, "kind": "mẫu tự chọn từ dữ liệu",
                                "summary": f"Mẫu `{sample_id}` lấy từ split **{split}** của "
                                           f"`data/prepared`.",
                                "profile": profile_from_sample(sample) or dict(MOCK_CASES[0]["profile"])}
                        profile = dict(case["profile"])
                        reference_ticker = ticker
        else:
            st.markdown("**Hồ sơ tỷ số tùy chỉnh**")
            profile = {spec["key"]: st.slider(spec["label"], min_value=float(spec["min"]),
                                              max_value=float(spec["max"]),
                                              value=float(spec["default"]),
                                              step=float(spec["step"]), help=spec["help"],
                                              key=f"slider_{spec['key']}")
                       for spec in SLIDER_SPECS}
            reference_ticker = st.selectbox(
                "Công ty tham chiếu (tuỳ chọn — để xét cảnh báo rò rỉ thực thể)",
                ["— không chọn —"] + list(TICKERS), key="ref_ticker")
            reference_ticker = None if reference_ticker.startswith("—") else reference_ticker
            case = {"id": "HỒ SƠ TÙY CHỈNH", "ticker": reference_ticker, "split": None,
                    "actual": None, "expected_p": None, "kind": "hồ sơ nhập tay",
                    "summary": "Hồ sơ 6 tỷ số do người dùng nhập; các tỷ số khác được suy ra từ giả "
                               "định kế toán cố định (xem ghi chú ở Tab 1).",
                    "profile": profile}

        st.divider()
        st.caption("Tái lập số liệu: `python -m forecasting.train` → `python -m forecasting.evaluate` "
                   "→ `python -m scripts.analyze`. Chạy demo: `streamlit run app.py`.")
    return {"mode": mode, "threshold": float(threshold), "case": case, "profile": profile,
            "reference_ticker": reference_ticker}


# Score the selected input: a stored sample in mode A, a slider profile in mode B.
#: Derived features shown in tab 1, which the user does not enter but the model weighs heavily.
EXTRA_DISPLAY: Tuple[str, ...] = (
    "quick_ratio_latest", "debt_to_equity_latest", "ocf_to_sales_latest",
    "retained_to_assets_latest", "current_ratio_min_window", "distress_quarters_in_window",
)


def resolve_and_score(state: Dict[str, Any],
                      artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve a sample from `data/prepared` when it exists, otherwise score the slider profile."""
    case = state["case"]
    entry = sample_index().get(str(case.get("id"))) if sample_index() else None
    if entry is not None:
        split, sample = entry
        outcome = score_sample(sample, artifact)
        outcome.update({"sample_id": case["id"], "ticker": sample["ticker"], "split": split,
                        "actual": int(sample["is_distressed"]), "loaded": True,
                        "expected_p": case.get("expected_p")})
        return outcome
    outcome = score_profile(state["profile"], artifact)
    outcome.update({"sample_id": case.get("id"), "ticker": case.get("ticker"),
                    "split": case.get("split"), "actual": case.get("actual"), "loaded": False,
                    "expected_p": case.get("expected_p")})
    return outcome


def _evidence_rows(outcome: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    """Build the evidence table: the six ratios plus derived features the model weighs heavily."""
    rows: List[Dict[str, Any]] = []
    for key, feature in SLIDER_TO_FEATURE.items():
        rows.append({"Chỉ số (feature)": pretty_feature(feature),
                     "Giá trị": fmt(outcome["evidence"].get(key)),
                     "Nguồn": "người dùng nhập" if mode == "B" else "trích từ hồ sơ BCTC"})
    if mode == "B" and outcome.get("sample") is not None and HAS_FORECASTING:
        try:
            engineered = feature_rows([outcome["sample"]])[0]
            for feature in EXTRA_DISPLAY:
                rows.append({"Chỉ số (feature)": pretty_feature(feature),
                             "Giá trị": fmt(engineered.get(feature)),
                             "Nguồn": "suy ra (giả định)"})
        except Exception:  # pragma: no cover
            pass
    return rows


def tab_inference(info: Dict[str, Any], state: Dict[str, Any],
                  artifact: Optional[Dict[str, Any]]) -> None:
    """Render tab 1: the score, its local explanation and the feature evidence."""
    # 1. Score the selected input.
    outcome = resolve_and_score(state, artifact)
    probability, threshold = float(outcome["p"]), float(state["threshold"])
    case = state["case"]

    # 2. Show the gauge, falling back to a bare metric when plotly is unavailable.
    left, right = st.columns([1.05, 1.35], gap="large")
    with left:
        figure = gauge_figure(probability, threshold)
        if figure is not None:
            show_plotly(figure)
        else:
            st.metric("P(suy giảm tài chính)", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))
        st.markdown(badge_html(probability, threshold), unsafe_allow_html=True)
        if outcome["source"] == "model":
            st.caption(f"Điểm số tính bằng **đúng pipeline đã chốt**: impute median → "
                       f"RandomForest(300 cây, max_depth=6), {outcome['n_features']} feature "
                       f"(`forecasting.features`).")
        else:
            st.caption("⚠️ Điểm số đến từ **mô hình mô phỏng** (thiếu `best.joblib` hoặc thiếu gói "
                       "`forecasting`) — chỉ minh hoạ luồng tính toán, **không** phải kết quả đồ án.")
        if outcome.get("error"):
            st.caption(f"Ghi chú kỹ thuật: {outcome['error']}")

    with right:
        rows = [{"Thuộc tính": "Ca đang xét", "Giá trị": str(outcome.get("sample_id") or "—")},
                {"Thuộc tính": "Công ty", "Giá trị": str(outcome.get("ticker") or "— (nhập tay)")},
                {"Thuộc tính": "Split của mẫu",
                 "Giá trị": str(outcome.get("split") or "— (hồ sơ tự dựng)")},
                {"Thuộc tính": "Số quý lịch sử",
                 "Giá trị": str(outcome.get("n_history") or "—")},
                {"Thuộc tính": "Nhãn THẬT (nếu có)",
                 "Giá trị": "—" if outcome.get("actual") is None else str(int(outcome["actual"]))},
                {"Thuộc tính": "Ngưỡng đang dùng", "Giá trị": fmt(threshold)},
                {"Thuộc tính": "Dự đoán tại ngưỡng",
                 "Giá trị": "1 (suy giảm)" if probability >= threshold else "0 (an toàn)"}]
        if outcome.get("expected_p") is not None:
            rows.append({"Thuộc tính": "P trong báo cáo (đối chiếu)",
                         "Giá trị": fmt(outcome["expected_p"])})
        show_table(rows, hide_index=True)
        st.markdown(f"**Loại ca:** {case.get('kind', '—')}  \n{case.get('summary', '')}")
        if outcome.get("actual") is not None:
            match = int(outcome["actual"]) == int(probability >= threshold)
            st.markdown(f"Đối chiếu nhãn thật: **{'KHỚP' if match else 'LỆCH'}** với quyết định "
                        f"⇒ {'True Positive/Negative' if match else 'một ca dự đoán sai'}.")
        if outcome.get("split") in ("validation", "test"):
            st.warning("⚠️ Mẫu nằm trong split đã dùng để CHỌN mô hình/ngưỡng ⇒ đây là **chạy lại lịch "
                       "sử (backtest)**, không phải dự báo tương lai.", icon="⚠️")

    # 3. Decompose the score into local contributions.
    st.divider()
    st.subheader("🔍 Vì sao mô hình ra con số này? (đóng góp cục bộ)")
    contributions = local_contributions(state["profile"], outcome, top_k=10)
    items = contributions.get("top") or []
    chart = contributions_figure(items, float(contributions["base_value"]), probability)
    if chart is not None:
        show_plotly(chart)
    else:
        show_bar({pretty_feature(name): value for name, value in items})
    st.caption(f"Phương pháp: **{contributions['method_label']}**. "
               f"Đỏ = đẩy về phía suy giảm (φ > 0), xanh = kéo về an toàn (φ < 0). "
               f"Lưu ý: SHAP giải thích **mô hình**, không phải quan hệ nhân quả — khi 33/47 feature có "
               f"VIF > 10 thì 'công' được chia đều cho cả cụm tương quan.")
    top_positive = [(name, value) for name, value in items if value > 0][:3]
    top_negative = [(name, value) for name, value in items if value < 0][:3]
    columns = st.columns(2)
    with columns[0]:
        st.markdown("**Đẩy về phía suy giảm (φ > 0)**")
        st.markdown("\n".join(f"- `{pretty_feature(n)}` ({fmt(v, 4)})" for n, v in top_positive)
                    or "- (không có)")
    with columns[1]:
        st.markdown("**Kéo về phía an toàn (φ < 0)**")
        st.markdown("\n".join(f"- `{pretty_feature(n)}` ({fmt(v, 4)})" for n, v in top_negative)
                    or "- (không có)")
    if contributions.get("method") == "kernel_shap":
        base, gap = float(contributions["base_value"]), float(contributions["gap"])
        st.caption(f"Tự kiểm chứng (efficiency): Σφ + E[f] − f(x) = {gap:.2e} ⇒ xấp xỉ 0, đúng ràng buộc "
                   f"của KernelSHAP (Lundberg & Lee 2017).")

    # 4. List the ratio and feature evidence behind the score.
    st.divider()
    st.subheader("🧾 Bằng chứng feature của ca đang xét")
    show_table(_evidence_rows(outcome, state["mode"]), hide_index=True)
    if state["mode"] == "B":
        st.caption("Chế độ B: 6 tỷ số bạn nhập được dựng thành một lịch sử 8 quý nhất quán "
                   "(doanh thu chuẩn hoá, tăng trưởng 2%/quý), rồi đưa qua **đúng pipeline 47 feature** "
                   "của đồ án. Các tỷ số khác được suy ra theo giả định: tồn kho = 35% tài sản ngắn hạn, "
                   "tiền = 20% tài sản ngắn hạn, OCF/doanh thu = biên ròng + 5%, nợ ngắn hạn = 40% "
                   "doanh thu, lợi nhuận giữ lại = 55% vốn chủ sở hữu. Đây là **what-if minh hoạ**, "
                   "không phải số liệu của một doanh nghiệp có thật.")

    # 5. Flag entity-leakage risk for the reference company.
    st.divider()
    st.subheader("⚠️ Cảnh báo rủi ro thực thể (Entity risk)")
    entity_warning(state["reference_ticker"], info["ticker_prior"], FALLBACK["ticker_prior_n_train"])
    with st.expander("Tỉ lệ nhãn suy giảm theo từng công ty (bằng chứng rò rỉ thực thể)"):
        render_company_prior_table(info["ticker_prior"], FALLBACK["ticker_prior_n_train"])
        st.caption("Nguồn: `reports/results/baselines.json` → `ticker_prior_train` (tính trên TRAIN). "
                   "Baseline `ticker_prior` chỉ dùng cột công ty mà đạt AUROC 0,986 trên test — "
                   "**gần bằng** mô hình học máy (0,983).")


# Tab 2: baseline comparison and the entity-leakage experiment.
def tab_audit(info: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Render tab 2: the baseline table, the leakage story and the confusion matrix per threshold."""
    model_key = f"model[{info['best_name']}]"
    systems = list(info["systems"])
    model_auroc = next((r["auroc"] for r in systems if str(r["system"]) == model_key), None)
    if model_auroc is None:
        model_auroc = info["test_auroc"]

    # 1. Rank every system on the shared test split.
    st.subheader(f"1. Bảng baseline trên CÙNG {info['n_test']} mẫu test")
    st.markdown(
        f"Tất cả hệ thống được chấm trên **đúng cùng {info['n_test']} mẫu test "
        f"({info['n_positive']} dương)** — test KHÔNG tham gia chọn mô hình/ngưỡng, chỉ dùng để báo cáo "
        f"và so sánh. Mô hình chốt: **{info['best_name']}** "
        f"(AUROC {fmt(info['test_auroc'])}, AP {fmt(info['test_ap'])}, Brier {fmt(info['test_brier'])}).")
    table = []
    for row in sorted(systems, key=lambda r: -(r["auroc"] if r["auroc"] is not None else -1)):
        delta = (None if row["auroc"] is None or model_auroc is None
                 else row["auroc"] - model_auroc)
        table.append({"Hệ thống": str(row["system"]),
                      "AUROC": fmt(row["auroc"]), "AP": fmt(row["ap"]),
                      "ΔAUROC so mô hình chốt": "—" if delta is None else
                                                ("+ " + fmt(delta)) if delta > 0 else fmt(delta),
                      "Ghi chú": str(row["source"])})
    show_table(table, hide_index=True)
    chart = systems_figure(systems, model_key, info["n_test"])
    if chart is not None:
        show_plotly(chart)
    st.warning(
        f"**Đọc bảng cho đúng:** baseline `ticker_prior` (chỉ dùng **tên công ty**, không dùng bất kỳ "
        f"chỉ số tài chính nào) đạt AUROC **0,9858**, còn mô hình học máy **0,9828** ⇒ chênh lệch "
        f"**gần bằng 0** (DeLong p = {fmt(info['delong_p'], 4)} ⇒ **không** có ý nghĩa thống kê). "
        f"Đây là phát hiện trung tâm của đồ án: điểm in-domain cao phần lớn đến từ **rò rỉ cấp thực thể "
        f"(entity leakage)**, không phải từ việc 'học' tín hiệu tài chính.", icon="⚠️")

    # 2. Compare in-domain, cross-company and LOCO AUROC.
    st.divider()
    st.subheader("2. In-domain vs Cross-Company — đóng góp trung thực về Entity Leakage")
    columns = st.columns(3)
    columns[0].metric("AUROC in-domain (train→test)", fmt(info["in_domain_auroc"]),
                      help="Chia tập theo THỜI GIAN trong từng công ty ⇒ mô hình đã thấy các quý khác "
                           "của chính công ty đó.")
    columns[1].metric("AUROC cross-company (GroupKFold)", fmt(info["cross_company_auroc"]),
                      delta=fmt(info["cross_company_auroc"] - info["in_domain_auroc"]),
                      delta_color="inverse",
                      help=info["cross_company_protocol"])
    columns[2].metric(f"AUROC LOCO ({info['loco_n_evaluable']} công ty)",
                      fmt(info["loco_mean_auroc"]),
                      help="LOCO: train 7 công ty, test công ty còn lại — câu hỏi 'công ty hoàn toàn "
                           "mới thì sao?'.")
    chart = leakage_figure(info)
    if chart is not None:
        show_plotly(chart)
    st.markdown(
        f"- **Cơ chế:** nhãn gần như là *thuộc tính của công ty* (WMT/HD/LOW có 100% quý = suy giảm) ⇒ "
        f"khi tập test chứa các quý **cùng công ty** với train, mô hình chỉ cần 'nhớ mặt công ty' là đã "
        f"đạt AUROC 0,983.\n"
        f"- **Kiểm chứng bằng cách giữ TRỌN công ty ra ngoài** (`GroupKFold`, {info['cross_company_protocol']}): "
        f"AUROC còn **{fmt(info['cross_company_auroc'])}** (AP {fmt(info['cross_company_ap'])}) — tức mất "
        f"**{fmt(info['in_domain_auroc'] - info['cross_company_auroc'])}** AUROC.\n"
        f"- **LOCO** (mỗi lần giữ 1 công ty): trung bình **{fmt(info['loco_mean_auroc'])}** AUROC trên "
        f"{info['loco_n_evaluable']} công ty — con số trung thực nhất cho câu hỏi 'doanh nghiệp mới'.\n"
        f"- Toàn bộ quy tắc **chọn mô hình** của đồ án đã ưu tiên AP cross-company trước tiên: "
        f"_{info.get('selection_rule') or 'max AP cross-company → best-F1(val) → AP(val) → AUROC(val)'}_.")

    # 3. Show the published confusion matrix next to the live recount.
    st.divider()
    st.subheader(f"3. Confusion matrix tại ngưỡng vận hành {fmt(info['threshold'])}")
    operating = info["operating"]
    counts_report = {key: int(operating.get(key) or 0) for key in ("tn", "fp", "fn", "tp")}
    left, right = st.columns(2, gap="large")
    with left:
        chart = confusion_figure(counts_report, float(info["threshold"]),
                                 "số công bố trong báo cáo (validation → test)")
        if chart is not None:
            show_plotly(chart)
        st.markdown(
            f"| | |\n|---|---|\n"
            f"| Precision | **{fmt(operating.get('precision'), 4)}** |\n"
            f"| Recall (nhạy) | **{fmt(operating.get('recall'), 4)}** |\n"
            f"| F1 / macro-F1 | **{fmt(operating.get('f1'), 4)}** / {fmt(operating.get('macro_f1'), 4)} |\n"
            f"| Chi phí kỳ vọng 5·FN + FP | **{fmt(expected_cost(counts_report), 1)}** |")
        st.caption(f"TP = {counts_report['tp']} · TN = {counts_report['tn']} · FN = {counts_report['fn']} "
                   f"· FP = {counts_report['fp']} trên {info['n_test']} mẫu test — khớp "
                   f"`reports/results/test_evaluation.json`.")
    with right:
        live = confusion_at(float(state["threshold"]), test_predictions())
        if any(live.values()):
            chart = confusion_figure(live, float(state["threshold"]),
                                     "TÍNH LẠI bằng app khi bạn kéo ngưỡng")
            if chart is not None:
                show_plotly(chart)
            delta_cost = expected_cost(live) - expected_cost(counts_report)
            st.markdown(
                f"Ngưỡng đang chọn **{fmt(state['threshold'])}** cho FN = {live['fn']}, FP = {live['fp']} "
                f"⇒ chi phí kỳ vọng **{fmt(expected_cost(live), 1)}** "
                f"({'+' if delta_cost >= 0 else ''}{fmt(delta_cost, 1)} so với ngưỡng vận hành).")
            st.caption("Vì sao FN đắt gấp 5 lần FP: bỏ sót một doanh nghiệp đang suy giảm (FN) tốn kém "
                       "hơn nhiều so với việc rà soát thêm một doanh nghiệp thực ra bình thường (FP). "
                       "Ngưỡng tối ưu chi phí trên test là 0,783 — gần bằng 0,788 đang dùng.")
        else:
            st.info("Không chấm lại được tập test (thiếu `data/prepared/test.json` hoặc `best.joblib`) ⇒ "
                    "chỉ hiển thị số công bố trong báo cáo.")
    st.caption(f"Ngưỡng best-F1 trên test = **{fmt(info['threshold_best_f1_test'])}**; ngưỡng tối ưu chi "
               f"phí = **0,783**; ngưỡng vận hành của đồ án = **{fmt(info['threshold'])}** (chọn trên "
               f"validation). Ngưỡng quyết định không nằm trong tham số học của mô hình nên "
               f"**không gây rò rỉ** — nhưng phải chọn trên validation, không chọn trên test.")


# Tab 3: error analysis of the three misclassified test samples.
@st.cache_data(show_spinner=False)
def published_local_explanations() -> List[Dict[str, Any]]:
    """Load the published local SHAP explanations from `reports/results/shap.json`."""
    return _load_json(RESULTS_DIR / "shap.json").get("local_explanations") or []


def tab_errors(info: Dict[str, Any], state: Dict[str, Any]) -> None:
    """Render tab 3: the three errors, their causes and the label-stability evidence."""
    threshold = float(info["threshold"])
    if state["case"].get("id") in {case["id"] for case in ERROR_CASES}:
        st.info(f"Bạn đang chọn ca **`{state['case']['id']}`** ở sidebar — chính là một trong 3 ca sai "
                f"được phân tích dưới đây.", icon="👉")
    # 1. Table of the three errors at the operating threshold.
    st.subheader(f"1. Ba ca dự đoán sai trên tập test (ngưỡng vận hành {fmt(threshold)})")
    table = []
    for case in ERROR_CASES:
        gap = case["probability"] - threshold
        table.append({"Mẫu": case["id"], "Công ty": case["ticker"],
                      "Nhãn thật": int(case["actual"]),
                      "Dự đoán": int(case["probability"] >= threshold),
                      "Loại lỗi": case["error"] + (" (sát ngưỡng)" if abs(gap) < 0.05 else " (rõ rệt)"),
                      "P(distress)": fmt(case["probability"], 4),
                      "Lệch so ngưỡng": f"{'+' if gap > 0 else '−'}{fmt(abs(gap), 4)}",
                      "Số quý lịch sử": case["n_history"]})
    show_table(table, hide_index=True)
    st.caption("Nguồn: `reports/results/analysis.json` → `error_cases` (`reports/results/analysis.md` §5) — "
               "đây là **toàn bộ** 3 mẫu sai của mô hình chốt trên 64 mẫu test.")

    for case in ERROR_CASES:
        gap = abs(case["probability"] - threshold)
        with st.expander(f"Chi tiết {case['id']} · {case['error']} · P = {fmt(case['probability'], 4)} "
                         f"(lệch {fmt(gap, 4)})", expanded=(case["id"] == "HD-2024Q2")):
            columns = st.columns([1.1, 1])
            with columns[0]:
                st.markdown(case["cause"])
            with columns[1]:
                rows = [{"Chỉ số": pretty_feature(name), "Giá trị": fmt(value)}
                        for name, value in case["features"].items()]
                show_table(rows, hide_index=True)
                st.caption("`—` nghĩa là chỉ tiêu thiếu ở quý đó (được impute median trong pipeline).")

    # 2. Label strips that show one company carrying both labels across quarters.
    st.divider()
    st.subheader("2. Nhãn thật theo từng quý — vì sao 'cùng hồ sơ' có thể mang hai nhãn khác nhau?")
    st.markdown(
        "Nhãn `is_distressed` **không** phải một quy tắc kế toán đơn giản: khi áp lại các quy tắc công khai "
        "đơn giản nhất lên chính dữ liệu, mức khớp cao nhất chỉ **74,7%** (`stress_signals ≥ 1`), còn các "
        "quy tắc kiểu 'lợi nhuận < 0' chỉ khớp **39–65%** (`reports/results/analysis.md` §10). Hệ quả: nhãn "
        "**đổi giữa các quý ngay trong cùng một công ty**, nên mô hình buộc phải học một quy tắc nhiễu và "
        "các ca sát ngưỡng dễ bị đảo dấu.")
    labels = labels_by_ticker()
    for ticker in ("DG", "FIVE", "HD"):
        items = labels.get(ticker) or []
        if not items:
            continue
        shown = items[-16:]
        chart = label_strip_figure(shown, f"Nhãn thật của {ticker} theo quý "
                                          f"({len(shown)}/{len(items)} quý gần nhất)")
        if chart is not None:
            show_plotly(chart)
        else:
            show_table([{"Mẫu": sid, "Nhãn": label} for sid, label in shown], hide_index=True)
    st.caption("DG là ví dụ rõ nhất: 35,3% nhãn dương trên train nhưng 54,8% trên toàn bộ 31 quý ⇒ nhãn "
               "**đảo chiều giữa các quý**, đúng bối cảnh của ca FP `DG-2025Q2`. Nếu `data/prepared` "
               "không nạp được thì mục này bị bỏ qua.")

    # 3. Published SHAP explanations for the error cases.
    st.divider()
    st.subheader("3. Giải thích SHAP đã công bố cho các ca sai (đối chiếu độc lập với app)")
    explanations = published_local_explanations()
    if explanations:
        for item in explanations:
            positives = ", ".join(f"`{i['feature']}` ({i['phi']:+.3f})" for i in item["top_positive"])
            negatives = ", ".join(f"`{i['feature']}` ({i['phi']:+.3f})" for i in item["top_negative"])
            st.markdown(f"- **{item['sample_id']}** — nhãn thật {item['actual']}, "
                        f"P = {fmt(item['probability'], 4)}, dự đoán {item['predicted']}  \n"
                        f"  đẩy về suy giảm: {positives}  \n  kéo về an toàn: {negatives}")
        st.caption("Nguồn: `reports/results/shap.json` → `local_explanations` (KernelSHAP tự cài đặt, 200 "
                   "liên minh/điểm, nền 40 mẫu train). Chọn ca `FIVE-2023Q3` hoặc `HD-2024Q2` ở sidebar để "
                   "app tính lại **đúng mẫu đó** rồi đối chiếu với bảng này.")
    else:
        st.info("Chưa có `reports/results/shap.json` ⇒ chạy `python -m scripts.explain_model` để sinh giải "
                "thích SHAP đã công bố.")

    # 4. Conclusions drawn from the error analysis.
    st.divider()
    st.subheader("4. Kết luận rút ra từ phân tích lỗi")
    st.markdown(
        "1. **Chỉ 3/64 mẫu sai** ở ngưỡng vận hành (TP 36, TN 25, FN 2, FP 1) — nhưng cả 3 đều có "
        "*nguyên nhân có cấu trúc*, không phải lỗi ngẫu nhiên:\n"
        "   - **HD-2024Q2** (FN, P = 0,7835): lệch ngưỡng 0,0044 ⇒ **đảo quyết định khi ngưỡng dịch "
        "0,5 điểm %**; đây là ca mà ngưỡng tối ưu chi phí 0,783 sửa được.\n"
        "   - **DG-2025Q2** (FP, P = 0,7975): chỉ số *trông* căng thẳng nhưng nhãn thật 0 ⇒ giới hạn nằm ở "
        "**định nghĩa nhãn**, không phải ở mô hình.\n"
        "   - **FIVE-2024Q3** (FN, P = 0,1745): cách ngưỡng hơn 61 điểm % ⇒ **không thể sửa bằng ngưỡng**.\n"
        "2. **Bài học phương pháp luận:** khi nhãn gần như là thuộc tính riêng của từng công ty thì 3 ca sai "
        "này là *trần* của dữ liệu hiện có. Muốn cải thiện thật phải **thêm dữ liệu/thực thể và định nghĩa "
        "nhãn có cơ sở học thuật** (walk-forward, survival) — không phải thêm cấu hình mô hình. Đây cũng là "
        "hướng phát triển đã nêu trong báo cáo đồ án.")


def main() -> None:
    """Render the page: header, sidebar and the three tabs."""
    st.set_page_config(page_title="Dự báo suy giảm tài chính — Demo Streamlit",
                       page_icon="🛒", layout="wide", initial_sidebar_state="expanded")

    # 1. Load the artifacts that back every figure shown on the page.
    info = load_results()
    artifact = load_artifact()

    # 2. Page header and the headline metrics.
    st.markdown(f"<h2 style='margin-bottom:0.2rem'>🛒📉 {APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown(f"**{AUTHOR}** &nbsp;·&nbsp; {SUPERVISOR} &nbsp;·&nbsp; {COURSE}")
    st.caption("Dữ liệu: báo cáo tài chính quý SEC XBRL của 8 chuỗi bán lẻ Mỹ — **324 mẫu** dự báo "
               "(212 train / 32 validation / 64 test + 16 purge) · **47 feature** · dự báo trước **1 quý** "
               "(`is_distressed`).")

    kpis = st.columns(4)
    kpis[0].metric("Mô hình chốt", str(info["best_name"]),
                   help="Chọn theo AP cross-company (GroupKFold) → best-F1(val) → AP(val) → AUROC(val).")
    kpis[1].metric("Ngưỡng vận hành", fmt(float(info["threshold"])),
                   help="Ngưỡng best-F1 trên validation; ngưỡng tối ưu chi phí 5·FN + FP là 0,783.")
    kpis[2].metric(f"AUROC test (n = {info['n_test']})", fmt(info["test_auroc"]),
                   help="In-domain: chia tập theo thời gian trong từng công ty.")
    kpis[3].metric("AUROC cross-company", fmt(info["cross_company_auroc"]),
                   delta=fmt(info["cross_company_auroc"] - info["in_domain_auroc"]),
                   delta_color="inverse",
                   help="Giữ TRỌN công ty ra khỏi fold-train — con số trung thực hơn cho công ty mới.")
    missing = [name for name, present in info["sources"].items() if not present]
    if missing:
        st.info(f"Một số artifact chưa có ⇒ app đang dùng số liệu dự phòng cho: **{', '.join(missing)}**. "
                f"Chạy `python -m scripts.run_all` để sinh đầy đủ `reports/results/*`.")

    # 3. Render the sidebar first, because the tabs consume its state.
    state = render_sidebar(info, artifact)

    # 4. Render the three tabs.
    tab_predict, tab_audit_tab, tab_error = st.tabs([
        "1️⃣ Dự báo & Phân tích rủi ro (Inference)",
        "2️⃣ Đối chiếu Baseline & Thực nghiệm (Audit)",
        "3️⃣ Phân tích mẫu sai (Error Analysis)"])
    with tab_predict:
        tab_inference(info, state, artifact)
    with tab_audit_tab:
        tab_audit(info, state)
    with tab_error:
        tab_errors(info, state)

    # 5. Footer: artifact provenance and the commands that rebuild every number.
    st.divider()
    with st.expander("🔁 Tái lập & nguồn số liệu của demo này"):
        st.markdown(
            "**Nguồn số liệu app đang đọc**\n\n"
            "- `reports/models/best.joblib` — mô hình + ngưỡng đã chốt (Pipeline: impute median → "
            "RandomForest).\n"
            "- `reports/results/summary.json` — chọn mô hình, ngưỡng, so sánh 4 họ mô hình.\n"
            "- `reports/results/test_evaluation.json` — metric + confusion matrix trên test.\n"
            "- `reports/results/baselines.json` — dummy · ticker-prior · 1 chỉ tiêu · Altman Z''.\n"
            "- `reports/results/significance.json` — DeLong/bootstrap trên cùng 64 mẫu test.\n"
            "- `reports/results/analysis.json` — in-domain vs cross-company, LOCO, phân tích lỗi.\n"
            "- `reports/results/shap.json` — KernelSHAP toàn cục + cục bộ cho ca sai.\n"
            "- `data/prepared/{train,validation,test,purged}.json` — dữ liệu theo chính sách split "
            "trong `data/prepared/manifest.json`.\n\n"
            "**Lệnh tái lập toàn bộ**\n\n"
            "```powershell\n"
            "python -m forecasting.train        # sinh best.joblib + summary.json\n"
            "python -m forecasting.evaluate     # test_evaluation.json\n"
            "python -m forecasting.baselines    # baselines.json\n"
            "python -m scripts.significance     # significance.json\n"
            "python -m scripts.analyze          # analysis.json\n"
            "python -m scripts.explain_model    # shap.json\n"
            "streamlit run app.py               # demo này\n"
            "```\n"
            "**Ghi chú về tính trung thực:** app không tự tính lại AUROC/AP mà hiển thị *nguyên* artifact "
            "của pipeline (tránh lệch số với báo cáo); riêng **confusion matrix theo ngưỡng kéo-thả** và "
            "**KernelSHAP của ca đang chọn** được tính lại ngay trong app bằng đúng mô hình + đúng "
            "`forecasting.features`.")
    st.caption("Demo mang tính minh hoạ học thuật cho đồ án CS114 — kết quả không dùng để ra quyết định "
               "đầu tư. Đọc kèm `reports/results/analysis.md`.")


if __name__ == "__main__":
    main()
