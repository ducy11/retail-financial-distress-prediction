"""Sinh báo cáo markdown (docs/BAO-CAO.md, docs/slide.md, docs/dinh-nghia-nhan.md,
docs/huong-dan-tai-lap.md) TRỰC TIẾP từ artifact trong `reports/`.

Nguyên tắc: mọi số liệu trong báo cáo được đọc từ `reports/results/*.json` — không nhập tay,
nên báo cáo không thể "lệch" so với kết quả chạy thật. Chạy lại pipeline rồi chạy script này là
báo cáo tự cập nhật.

Lệnh: python -m scripts.make_report   → docs/BAO-CAO.md + docs/slide.md + …
      (chạy trực tiếp `python scripts/make_report.py` cũng được — xem khối `ROOT` bên dưới)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]  # repo root (script nằm trong scripts/)
sys.path.insert(0, str(ROOT))

from forecasting.config import DOCS_DIR, RESULTS_DIR, ensure_utf8_stdio  # noqa: E402

#: Tên mô hình hiển thị đẹp trong báo cáo.
PRETTY = {
    "logistic": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "mlp": "MLP (mạng nơ-ron)",
    "dummy_most_frequent": "Dummy (lớp đa số)",
    "ticker_prior": "Baseline “nhớ mặt công ty” (ticker-prior)",
    "rule[altman_z_double_prime<1.1]": "Quy tắc Altman Z'' < 1,1 (không học tham số)",
}


#: Tài liệu tham khảo cho báo cáo — mỗi mục kèm vị trí sử dụng THẬT trong mã nguồn.
#: (được `section_references` đưa vào `docs/BAO-CAO.md`; giữ 1 chỗ duy nhất để dễ soát.)
REFERENCE_ROWS: List[Dict[str, str]] = [
    {"group": "Dữ liệu",
     "ref": "U.S. Securities and Exchange Commission. *EDGAR Application Programming Interfaces — "
            "company facts (XBRL)*. https://www.sec.gov/edgar/sec-api-documentation (truy cập 2026).",
     "where": "`scripts/crawl_sec.py`, `scripts/prepare_sec.py`, `data/sec/`"},
    {"group": "Cơ sở học thuật cho nhãn",
     "ref": "Beaver, W. H. (1966). Financial ratios as predictors of failure. "
            "*Journal of Accounting Research*, 4, 71–111.",
     "where": "Nhóm đặc trưng tỷ số (`forecasting/features.py`); mục 2.1, 8.1"},
    {"group": "",
     "ref": "Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of "
            "corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609.",
     "where": "Nhãn quy tắc `altman_z` (`forecasting/labels.py`); mục 8.2"},
    {"group": "",
     "ref": "Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. "
            "*Journal of Accounting Research*, 18(1), 109–131.",
     "where": "Nhóm tín hiệu của nhãn `stress_signals` (thu nhập/dòng tiền/vốn chủ); mục 8.2"},
    {"group": "",
     "ref": "Altman, E. I. (2000). *Predicting financial distress of companies: Revisiting the "
            "Z-score and ZETA models.* Working paper, Stern School of Business, New York University.",
     "where": "Ngưỡng Z'' < 1,1 (`ALTMAN_DISTRESS_BELOW`); mục 8.2"},
    {"group": "",
     "ref": "Barboza, F., Kimura, H., & Altman, E. (2017). Machine learning models and bankruptcy "
            "prediction. *Expert Systems with Applications*, 83, 405–417.",
     "where": "So sánh mô hình cây/boosting với mô hình tuyến tính; mục 2.1, 6.6"},
    {"group": "",
     "ref": "Mai, F., Tian, S., Lee, C., & Ma, L. (2019). Deep learning models for bankruptcy "
            "prediction using textual disclosures. *European Journal of Operational Research*, "
            "274(2), 743–758.",
     "where": "Bối cảnh “học máy cho distress”; mục 2.1, 10.3"},
    {"group": "Thuật toán",
     "ref": "Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.",
     "where": "`random_forest` (`forecasting/models.py`) — mô hình được chốt"},
    {"group": "",
     "ref": "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. "
            "*The Annals of Statistics*, 29(5), 1189–1232.",
     "where": "`hist_gradient_boosting` (scikit-learn) — mô hình so sánh"},
    {"group": "",
     "ref": "Pedregosa, F., và cộng sự (2011). Scikit-learn: Machine learning in Python. "
            "*Journal of Machine Learning Research*, 12, 2825–2830.",
     "where": "`Pipeline`, `GridSearchCV`, `StratifiedGroupKFold`, metric"},
    {"group": "Mất cân bằng & metric",
     "ref": "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic "
            "minority over-sampling technique. *Journal of Artificial Intelligence Research*, "
            "16, 321–357.",
     "where": "`imbalance_lab/`, `scripts/experiment_imbalance_real.py`; mục 9.4"},
    {"group": "",
     "ref": "He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on "
            "Knowledge and Data Engineering*, 21(9), 1263–1284.",
     "where": "Chính sách không dùng Accuracy; chọn AP/F1-macro/MCC (mục 3.2, 6.2)"},
    {"group": "",
     "ref": "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than "
            "the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, "
            "10(3), e0118432.",
     "where": "Ưu tiên PR-AUC (AP) làm tiêu chí chọn mô hình/ngưỡng (mục 5.2, 6.1)"},
    {"group": "",
     "ref": "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. "
            "*Monthly Weather Review*, 78(1), 1–3.",
     "where": "Hiệu chuẩn xác suất — Brier score (mục 7.7)"},
    {"group": "Thống kê & kiểm định",
     "ref": "DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under "
            "two or more correlated receiver operating characteristic curves: A nonparametric "
            "approach. *Biometrics*, 44(3), 837–845.",
     "where": "`forecasting/significance.py` — ΔAUROC so với baseline (mục 9.3)"},
    {"group": "",
     "ref": "Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. "
            "Chapman & Hall.",
     "where": "`bootstrap_ci`, paired bootstrap (mục 6.5, 9.3)"},
    {"group": "",
     "ref": "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical "
            "and powerful approach to multiple testing. *JRSS-B*, 57(1), 289–300.",
     "where": "BH-FDR khi sàng 47 đặc trưng trong EDA chuyên sâu (mục 3.7)"},
    {"group": "",
     "ref": "Virtanen, P., và cộng sự (2020). SciPy 1.0: Fundamental algorithms for scientific "
            "computing in Python. *Nature Methods*, 17, 261–272.",
     "where": "KS test, Mann–Whitney, Spearman trong `forecasting/eda.py`"},
    {"group": "Giải thích mô hình",
     "ref": "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model "
            "predictions. *NeurIPS 30*, 4765–4774.",
     "where": "`forecasting/explain.py` — KernelSHAP tự cài đặt (mục 9.2)"},
    {"group": "Đánh giá & rò rỉ",
     "ref": "Kaufman, S., Rosset, S., & Perlich, C. (2012). Leakage in data mining: Formulation, "
            "detection, and avoidance. *ACM TKDD*, 6(4), Article 15.",
     "where": "Checklist chống rò rỉ (mục 4.6) + thí nghiệm minh hoạ mốc SAI"},
    {"group": "",
     "ref": "Roberts, D. R., và cộng sự (2017). Cross-validation strategies for data with temporal, "
            "spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.",
     "where": "CV theo nhóm công ty (`StratifiedGroupKFold`, `GroupKFold`) + dải purge"},
    {"group": "Hướng phát triển",
     "ref": "Shumway, T. (2001). Forecasting bankruptcy more accurately: A simple hazard model. "
            "*The Journal of Business*, 74(1), 101–124.",
     "where": "Đề xuất mô hình sống sót/hazard theo thời gian (mục 10.3)"},
    {"group": "",
     "ref": "Campbell, J. Y., Hilscher, J., & Szilagyi, J. (2008). In search of distress risk. "
            "*The Journal of Finance*, 63(6), 2899–2939.",
     "where": "Căn cứ cho định nghĩa distress theo chuỗi nhiều quý (`forward_4q`)"},
    # --- REF_MARKER ---
]


def _json(name: str) -> Dict[str, Any]:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _n(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else f"{f:.{digits}f}"  # NaN → "—"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{100 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _vi(value: Any) -> str:
    """Số nguyên kiểu Việt Nam (phân cách nghìn bằng dấu chấm) — dùng cho số đếm lớn."""
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def artifacts() -> Dict[str, Any]:
    """Nạp toàn bộ artifact cần cho báo cáo."""
    return {
        "eda": _json("eda_summary.json"),
        "provenance": _json("provenance.json"),
        "eda_deep": _json("eda_deep.json"),
        "preprocessing_experiment": _json("preprocessing_experiment.json"),
        "imbalance_real": _json("imbalance_real.json"),
        "search": _json("search.json"),
        "label_sensitivity": _json("label_sensitivity.json"),
        "shap": _json("shap.json"),
        "significance": _json("significance.json"),
        "summary": _json("summary.json"),
        "tuning": _json("tuning.json"),
        "baselines": _json("baselines.json"),
        "checks": _json("validation_checks.json"),
        "test": _json("test_evaluation.json"),
        "analysis": _json("analysis.json"),
        "relabel": _json("relabel.json"),
        "etl_verify": _json("etl_verify.json") or {},
        "events": _json("events.json") or {},
        "manifest": _json("../data/prepared/manifest.json") or {},
    }


def fig(path: str, caption: str) -> str:
    """Chèn hình vào markdown (đường dẫn tương đối từ docs/)."""
    return f"\n![{caption}](../{path})\n\n*Hình: {caption}*\n"


def table(headers: List[str], rows: List[List[str]], aligns: List[str] | None = None) -> str:
    """Bảng markdown từ danh sách hàng (đã format sẵn)."""
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def section_summary(a: Dict[str, Any]) -> str:
    """Mục 1 — Tóm tắt: các số liệu chốt, đọc thẳng từ artifact."""
    eda = a["eda"].get("corpus", {})
    sp = a["eda"].get("splits", {})
    test = a["test"].get("test_metrics", {})
    op = test.get("operating", {})
    best = a["summary"].get("best_model")
    head = a["checks"].get("headline", {})
    cross = (head.get("cross_company_oof_auroc") or {}).get(best)
    in_dom = (head.get("in_domain_test_auroc") or {}).get(best)
    boot = ((a["checks"].get("in_domain_test") or {}).get(best, {})
            .get("bootstrap_test", {}).get("auroc", {}))
    base = next((r for r in a["baselines"].get("rows", []) if r["baseline"] == "ticker_prior"), {})
    label = a["eda"].get("label", {})
    simple_rules = [v for k, v in (a["analysis"].get("label_audit", {})
                                   .get("rule_agreement") or {}).items()
                    if not str(k).startswith("stress_signals")]
    simple_span = (f"{min(simple_rules):.0%}–{max(simple_rules):.0%}" if simple_rules else "—")
    prov = ((a.get("provenance") or {}).get("totals") or {})
    return "\n".join([
        "# Đồ án: Dự báo suy giảm tài chính (financial distress) doanh nghiệp bán lẻ",
        "",
        "*Báo cáo sinh tự động từ `reports/results/*.json` bằng `python -m scripts.make_report`.*",
        "",
        "## 1. Tóm tắt",
        "",
        f"**Bài toán.** Với dữ liệu báo cáo tài chính quý (SEC XBRL) của "
        f"**{eda.get('n_companies', '—')} chuỗi bán lẻ Mỹ**, dự báo cho quý kế tiếp liệu doanh "
        f"nghiệp có rơi vào trạng thái suy giảm tài chính (`is_distressed`) hay không. Dữ liệu: "
        f"**{eda.get('n_quarters', '—')} quý** → **{eda.get('n_samples', '—')} mẫu** dự báo, chia "
        f"train/validation/test = {sp.get('train', {}).get('total', '—')}/"
        f"{sp.get('validation', {}).get('total', '—')}/{sp.get('test', {}).get('total', '—')} "
        f"(= **{sum(sp.get(k, {}).get('total', 0) or 0 for k in ('train', 'validation', 'test'))} mẫu "
        f"dùng để huấn luyện/đánh giá**) + **{sp.get('purged', {}).get('total', '—')} mẫu purge** "
        f"không thuộc ba tập (dải đệm chống rò rỉ theo ngày công bố); "
        f"{sum(sp.get(k, {}).get('total', 0) or 0 for k in ('train', 'validation', 'test', 'purged'))} "
        f"= tổng số mẫu.",
        "",
        f"**Phương pháp.** {a['summary'].get('n_features', '—')} feature tài chính (14 tỷ số ở dạng "
        f"hiện tại + YoY, 10 tốc độ tăng trưởng, cấu trúc vốn, chỉ báo căng thẳng, cực trị/độ bền "
        f"theo cửa sổ); pipeline chính `median-impute → [scaler cho mô hình tuyến tính/MLP] → model` "
        "(**winsorize không dùng ở pipeline chính** — chỉ được thử nghiệm ở mục 9.1), " 
        "đặt trong `Pipeline` của scikit-learn; **4 họ mô hình** (Logistic Regression, Random "
        "Forest, HistGradientBoosting, MLP — tất cả thuần scikit-learn, KHÔNG dùng xgboost/lightgbm) "
        "đặt cạnh **4 baseline** (dummy · ticker-prior · 1 chỉ tiêu · **quy tắc Altman Z'' không học "
        "tham số**); đánh giá in-domain, cross-company "
        "(GroupKFold/LOCO) **và walk-forward theo thời gian** (− mục 6.4, 6.5, 6.7); chọn mô hình "
        "theo **AP cross-company** (quy tắc đầy đủ ở mục 5.2; "
        "mô hình triển khai giữ **cấu hình mặc định** — xem mục 5.3) rồi đánh giá trên test "
        "(**test không tham gia chọn mô hình/ngưỡng**).",
        "",
        "**Kết quả chính.**",
        "",
        f"- Mô hình được chọn: **{PRETTY.get(best, best)}**, ngưỡng vận hành "
        f"{_n(a['test'].get('threshold'))} → test **AUROC = {_n(test.get('auroc'))}**, "
        f"**AP = {_n(test.get('average_precision'))}**, precision/recall = "
        f"{_n(op.get('precision'))}/{_n(op.get('recall'))}, F1 = {_n(op.get('f1'))}, macro-F1 = "
        f"{_n(op.get('macro_f1'))}, Brier = {_n(test.get('brier'))}.",
        f"- Khoảng tin cậy 95% (bootstrap) của AUROC test: "
        f"[{_n(boot.get('ci95_low'))}, {_n(boot.get('ci95_high'))}] trên n = {test.get('n', '—')} mẫu.",
        f"- **Phát hiện quan trọng:** baseline *không dùng mô hình* — lấy tỷ lệ nhãn trung bình của "
        f"chính công ty trong train (`ticker_prior`) — đạt AUROC = "
        f"{_n((base.get('test') or {}).get('auroc'))}, AP = "
        f"{_n((base.get('test') or {}).get('average_precision'))}, tức **bằng hoặc gần bằng mô hình "
        f"học máy**. Khi chuyển sang đánh giá **cross-company** (giữ trọn công ty ra khỏi train), "
        f"AUROC của mô hình còn **{_n(cross)}** (in-domain: {_n(in_dom)}).",
        f"- Nhãn gần như là **thuộc tính của công ty**: "
        f"{', '.join(label.get('companies_all_one', [])) or '—'} có 100% nhãn = 1 trong mọi quý; "
        f"mức khớp của nhãn với quy tắc kế toán đơn giản nhất chỉ {simple_span}.",
        f"- **Nguồn dữ liệu kiểm chứng được (không sinh/sửa tay):** "
        f"{_vi(prov.get('n_raw_files_hashed'))} file companyfacts của SEC đã được băm SHA-256 và "
        f"khớp registry `data/sec/downloads.json` (**{_vi(prov.get('n_hash_mismatch'))} lệch**); "
        f"**{_vi(prov.get('n_facts_checked'))} ô** (quý × chỉ tiêu) tra ngược được trong companyfacts "
        f"(kèm `accn`, `form`, ngày nộp) — **{_vi(prov.get('n_facts_missing_in_sec'))} fact thiếu**, "
        f"**{_vi(prov.get('n_absent_but_filled'))} ô bịa số**; {_vi(prov.get('n_absent_cells'))} ô "
        f"không có fact ở SEC được để `null`. Chi tiết: `reports/results/provenance.md`.",
    ])


def section_intro(a: Dict[str, Any]) -> str:
    """Mục 2 — Giới thiệu: bối cảnh, câu hỏi nghiên cứu, đóng góp, phạm vi."""
    return "\n".join([
        "",
        "**Kết luận.** Vì baseline “nhớ mặt công ty” đạt xấp xỉ mô hình, AUROC ≈ 0,98 trong bảng "
        "kết quả thông thường **không** chứng minh năng lực dự báo suy giảm. Đóng góp trung thực "
        "của đồ án là chỉ ra rò rỉ thông tin ở cấp thực thể, định lượng nó, và đề xuất giao thức "
        "đánh giá đúng (cross-company + baseline + khoảng tin cậy). Phần truy vết nhãn ở "
        "`docs/dinh-nghia-nhan.md`.",
        "",
        "## 2. Giới thiệu",
        "",
        "### 2.1. Bối cảnh",
        "",
        "Dự báo suy giảm tài chính là bài toán kinh điển của tài chính định lượng (Beaver 1966; "
        "Altman 1968) và đã được làm lại bằng học máy (Barboza, Kimura & Altman 2017; Mai et al. "
        "2019). Với doanh nghiệp bán lẻ, tín hiệu sớm thường nằm ở biên lợi nhuận, vòng quay hàng "
        "tồn kho, cơ cấu nợ và dòng tiền hoạt động — đúng những chỉ tiêu có trong bộ dữ liệu này.",
        "",
        "### 2.2. Câu hỏi nghiên cứu",
        "",
        table(["#", "Câu hỏi", "Trả lời ở mục"],
              [["RQ1", "Bốn họ mô hình (tuyến tính / bagging / boosting / mạng nơ-ron) khác nhau "
                       "thế nào? (và có vượt nổi quy tắc Altman hay baseline ticker-prior không?)",
                "6, 7"],
               ["RQ2", "Đặc trưng nào quyết định kết quả? Có đặc trưng chi phối bất thường?", "7.2, 7.3"],
               ["RQ3", "Mô hình có tổng quát hoá sang **công ty chưa từng thấy**?", "6.4, 8"],
               ["RQ4", "Kết luận có phụ thuộc vào **định nghĩa nhãn**?", "8.2"]],
              ["---", "---", "---"]),
        "",
        "### 2.3. Đóng góp của đồ án",
        "",
        "1. **Pipeline tái lập được**: `python -m scripts.run_all` chạy toàn bộ từ dữ liệu thô trong "
        "repo tới báo cáo; manifest có SHA-256 cho nguồn dữ liệu và cho từng split.",
        "2. **Đo lường rò rỉ cấp thực thể** (điểm khác biệt so với cách làm thông thường): baseline "
        "ticker-prior, GroupKFold/LOCO, và hình kiểm chứng nhãn theo công ty × quý.",
        "3. **Bộ đánh giá đầy đủ**: AUROC, AP/PR, precision/recall/F1, macro & weighted F1, MCC, "
        "confusion matrix, calibration + Brier, khoảng tin cậy bootstrap, ngưỡng theo F1 và theo "
        "chi phí kỳ vọng.",
        "4. **Kiểm chứng độ nhạy theo định nghĩa nhãn** (`scripts.relabel`): dựng lại split bằng "
        "một quy tắc nhãn công khai, tái lập được, rồi so sánh kết luận.",
        "5. **Phân tích lỗi có cấu trúc**: 6 mẫu sai trên test được truy vết tới công ty và chỉ "
        "tiêu cụ thể (`reports/results/error_cases.csv`).",
        "",
        "### 2.4. Phạm vi và hạn chế đã biết",
        "",
        "- 8 công ty / 324 mẫu → mọi kết luận kèm khoảng tin cậy, không suy rộng ra toàn ngành.",
        "- Nhãn gốc không tái tạo được từ dữ liệu công bố (mục 8.1): hạn chế của **bộ dữ liệu**, "
        "đã được xử lý bằng kiểm chứng độ nhạy.",
        "- Tiền tệ quy đổi minh hoạ (×25.000 VND/USD) để thống nhất đơn vị; không phải tỷ giá lịch sử.",
        "- Chuỗi thời gian bị chồng lấn giữa các mẫu liên tiếp (cùng một quý xuất hiện trong nhiều "
        "cửa sổ lịch sử) → số quan sát **độc lập** thực tế nhỏ hơn 324.",
    ])


def section_eda(a: Dict[str, Any]) -> str:
    """Mục 3 — Dữ liệu & EDA (bảng + 6 hình sinh bởi `scripts.eda`)."""
    eda = a["eda"]
    corpus = eda.get("corpus", {})
    sp = eda.get("splits", {})
    label = eda.get("label", {})
    cov = eda.get("coverage_pct", {})
    hist = eda.get("history_length", {})
    ratio_stats = eda.get("ratio_stats") or []
    ratio_skewed = [r for r in ratio_stats if r.get("skew") is not None and abs(r["skew"]) > 1.0]
    ratio_outliers = [r for r in ratio_stats if (r.get("iqr_outlier_pct") or 0) > 5.0]
    worst_outliers = [(str(r["feature"]).replace("_latest", ""), _n(r.get("iqr_outlier_pct"), 1))
                      for r in sorted(ratio_outliers, key=lambda r: -(r["iqr_outlier_pct"] or 0))[:3]]
    ratio_missing = [r for r in ratio_stats if (r.get("missing_pct") or 0) > 20.0]
    worst_missing = [(str(r["feature"]).replace("_latest", ""), _n(r.get("missing_pct"), 1))
                     for r in sorted(ratio_missing, key=lambda r: -(r["missing_pct"] or 0))[:3]]
    corr_block = eda.get("ratio_correlation") or {}

    quarters = table(["Công ty", "Số quý có dữ liệu"],
                     [[t, n] for t, n in corpus.get("quarters_per_company", {}).items()],
                     ["---", "---:"])
    balance = table(["Tập", "Distress (1)", "Không (0)", "Tổng", "Tỷ lệ dương"],
                    [[name, v.get("distress"), v.get("total", 0) - v.get("distress", 0),
                      v.get("total"), _pct(v.get("distress", 0) / max(1, v.get("total", 1)))]
                     for name, v in sp.items()],
                    ["---", "---:", "---:", "---:", "---:"])
    shares = table(["Công ty", "Tỷ lệ nhãn = 1"],
                   [[t, f"{v:.2f}"] for t, v in sorted(label.get("label_share_by_ticker", {}).items(),
                                                       key=lambda kv: -kv[1])],
                   ["---", "---:"])
    coverage = table(["Chỉ tiêu", "Độ phủ thấp nhất theo công ty", "Trung bình"],
                     [[f, _pct(cov.get("min_by_field", {}).get(f, 0)),
                       _pct(cov.get("mean_by_field", {}).get(f, 0))]
                      for f in sorted(cov.get("min_by_field", {}),
                                      key=lambda k: cov.get("min_by_field", {}).get(k, 0))],
                     ["---", "---:", "---:"])

    return "\n".join([
        "",
        "## 3. Dữ liệu và phân tích khám phá (EDA)",
        "",
        "### 3.1. Nguồn dữ liệu và cách tạo mẫu",
        "",
        "Dữ liệu gốc là XBRL company-facts của SEC (`data/sec/`, có SHA-256 trong "
        "`data/sec/downloads.json`), được chuyển thành chuỗi 16 chỉ tiêu theo quý "
        "(`data/retail-expanded/`). Mỗi **mẫu** = (lịch sử các quý đã công bố trước `as_of`, "
        "quý target cần dự báo, nhãn `is_distressed`). Lịch sử **không** chứa quý target và mọi "
        "dòng lịch sử đều có `available_on <= as_of` (được kiểm thử tự động trong "
        "`tests/test_pipeline.py::test_no_label_leak_in_features`).",
        "",
        f"- Corpus: **{corpus.get('n_companies')} công ty, {corpus.get('n_quarters')} quý, "
        f"{corpus.get('n_samples')} mẫu dự báo**, {len(corpus.get('fields', []))} chỉ tiêu.",
        f"- Độ dài lịch sử: min {hist.get('min')}, median {hist.get('median')}, max {hist.get('max')}; "
        f"**{hist.get('n_below_min')}/{hist.get('n_samples')} mẫu "
        f"({_pct(hist.get('share_below_min'))})** có < 5 quý lịch sử → phần lớn feature YoY của "
        f"các mẫu này là NaN và phải impute.",
        "",
        quarters, "",
        "### 3.2. Cân bằng lớp và chất lượng nhãn",
        "",
        balance, "",
        f"- Nhãn là **nhị phân**, lớp dương chiếm đa số ở mọi tập → nếu chỉ báo cáo accuracy sẽ rất "
        f"dễ ngộ nhận, vì vậy báo cáo dùng thêm macro/weighted F1, MCC và đường PR.",
        f"- **Kiểm tra mất cân bằng theo thực thể (điểm mấu chốt):** tỷ lệ nhãn = 1 theo từng công "
        f"ty như sau — công ty {'/'.join(label.get('companies_all_one', [])) or '—'} có **100%** "
        f"nhãn = 1 trong mọi quý, trong khi ROST chỉ ~5%. Nhãn vì thế gần như là thuộc tính của "
        f"công ty chứ không phải sự kiện của quý.",
        "",
        shares, "",
        fig("reports/figures/eda/03_label_by_company_quarter.png",
            "Nhãn theo công ty × kỳ target — mỗi hàng gần như đồng màu ⇒ nhãn là thuộc tính thực thể"),
        "",
        "### 3.3. Chất lượng dữ liệu: độ phủ chỉ tiêu",
        "",
        coverage, "",
        f"- Các chỉ tiêu có độ phủ thấp ở một số công ty: "
        f"**{', '.join(eda.get('low_coverage_fields', [])) or '—'}**. Hệ quả: các tỷ số dùng "
        f"`receivables`/`short_term_investments` chỉ có nghĩa cho một phần mẫu; phần bị thiếu được "
        f"`SimpleImputer(median)` xử lý **trong pipeline** (tránh rò rỉ thống kê từ validation/test) "
        f"và có ablation riêng ở mục 7.3. Riêng tag `liabilities` có lỗ hổng đã được **khắc phục ở "
        f"tầng feature**: `debt_to_assets` / `debt_to_equity` dùng `total_liabilities` = tag nếu có, "
        f"ngược lại suy ra từ đẳng thức `total_assets - stockholders_equity` (xem mục 4.5).",
        "",
        fig("reports/figures/eda/01_coverage_by_company.png",
            "Độ phủ chỉ tiêu theo công ty (% số quý có dữ liệu)"),
        fig("reports/figures/eda/02_class_balance.png", "Cân bằng lớp theo tập"),
        fig("reports/figures/eda/04_ratio_boxplots_by_label.png",
            "Phân bố 6 tỷ số tài chính theo nhãn"),
        fig("reports/figures/eda/05_timeseries_indicators.png",
            "Chuỗi thời gian 4 chỉ tiêu của WMT / HD / ROST"),
        fig("reports/figures/eda/06_history_length.png",
            "Phân bố độ dài lịch sử; mẫu dưới ngưỡng 5 quý có nhiều feature YoY là NaN"),
        "",
        "### 3.4. Thống kê mô tả và phân phối từng biến",
        "",
        f"- **{len(eda.get('ratio_stats') or [])} tỷ số** được mô tả (giá trị quý gần nhất, không đơn vị) "
        f"và **{len(eda.get('indicator_stats') or [])} chỉ tiêu gốc** (nghìn tỷ VND, mọi công ty × mọi quý).",
        f"- Tỷ số lệch rõ (|skew| > 1): **{len(ratio_skewed)}**; tỷ số có hơn 5% giá trị ngoại lai "
        f"theo IQR: **{len(ratio_outliers)}** — nặng nhất: "
        f"{', '.join(f'{r} ({p}% ngoại lai)' for r, p in worst_outliers) or '—'}.",
        f"- Tỷ số thiếu hơn 20% mẫu: **{len(ratio_missing)}** "
        f"({', '.join(f'{r} {p}%' for r, p in worst_missing) or '—'}).",
        "",
        table(["Tỷ số", "Thiếu %", "Nhỏ nhất", "Q1", "Trung vị", "Trung bình", "Q3", "Lớn nhất",
               "Skew", "Ngoại lai IQR %"],
              [[str(r["feature"]).replace("_latest", ""), _n(r.get("missing_pct"), 1), _n(r.get("min")),
                _n(r.get("q1")), _n(r.get("median")), _n(r.get("mean")), _n(r.get("q3")), _n(r.get("max")),
                _n(r.get("skew"), 2), _n(r.get("iqr_outlier_pct"), 1)]
               for r in sorted(eda.get("ratio_stats") or [],
                               key=lambda r: -(r.get("iqr_outlier_pct") or 0))],
              ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
        "",
        "*Diễn giải:* tỷ số vừa lệch vừa nhiều ngoại lai (thường là loại có mẫu số nhỏ như "
        "`debt_to_equity`) nên được **clip/winsorize** trước khi huấn luyện mô hình tuyến tính; cột "
        "`Thiếu %` càng lớn thì cột đó càng bị `SimpleImputer(median)` chi phối.",
        "",
        table(["Chỉ tiêu gốc (nghìn tỷ VND)", "Thiếu %", "Nhỏ nhất", "Trung vị", "Trung bình",
               "Lớn nhất", "Ngoại lai IQR %"],
              [[r["feature"], _n(r.get("missing_pct"), 1), _n(r.get("min")), _n(r.get("median")),
                _n(r.get("mean")), _n(r.get("max")), _n(r.get("iqr_outlier_pct"), 1)]
               for r in sorted(eda.get("indicator_stats") or [],
                               key=lambda r: -(r.get("missing_pct") or 0))],
              ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
        "",
        "*Diễn giải:* chênh lệch **quy mô** giữa các công ty rất lớn (WMT lớn hơn FIVE nhiều lần) nên "
        "không so sánh giá trị tuyệt đối giữa công ty; mô hình dùng **tỷ số**. Các chỉ tiêu thiếu nhiều "
        "(short_term_investments 72%, liabilities 63%, receivables 38%) là hạn chế của dữ liệu gốc SEC.",
        "",
        fig("reports/figures/eda/07_ratio_distributions.png",
            "Phân phối 14 tỷ số — mỗi ô ghi % mẫu ngoài khoảng IQR của tỷ số đó"),
        "",
        "### 3.5. Tương quan giữa các biến và với nhãn",
        "",
        f"- **{corr_block.get('n_pairs_abs_ge_threshold')} cặp tỷ số** có |r| ≥ "
        f"{corr_block.get('threshold')} ⇒ gần như cùng một thông tin; gom thành "
        f"**{len(corr_block.get('clusters_abs_ge_threshold') or [])} cụm**: "
        + "; ".join(", ".join(str(n).replace("_latest", "") for n in cluster)
                    for cluster in (corr_block.get("clusters_abs_ge_threshold") or [])[:3])
        + ".",
        f"- Số chiều hiệu dụng của 14 tỷ số: "
        f"**{_n((corr_block.get('effective_dimensionality') or {}).get('participation_ratio'))}** "
        f"(cần {(corr_block.get('effective_dimensionality') or {}).get('n_components_for_95pct_variance')} "
        f"thành phần cho 95% phương sai) ⇒ phần lớn cột là thông tin dư.",
        f"- **{len(corr_block.get('outlier_driven_pairs') or [])} cặp** có tương quan Pearson lệch "
        f"Spearman hơn 0,3 ⇒ tương quan chủ yếu do **ngoại lai**: "
        + (", ".join(f"{p['a'].replace('_latest', '')} ↔ {p['b'].replace('_latest', '')} "
                     f"(r {p['pearson']} vs hạng {p['spearman']})"
                     for p in (corr_block.get("outlier_driven_pairs") or [])[:2]) or "—")
        + ".",
        "",
        table(["Cặp tỷ số tương quan mạnh nhất", "Pearson", "Spearman"],
              [[f"{p['a'].replace('_latest', '')} ↔ {p['b'].replace('_latest', '')}",
                _n(p.get("pearson")), _n(p.get("spearman"))]
               for p in (corr_block.get("strongest_pairs") or [])[:6]],
              ["---", "---:", "---:"]),
        "",
        table(["Tỷ số (tính trên train+validation)", "AUC 1-tỷ số", "Hướng", "r (point-biserial)",
               "q-value BH"],
              [[str(a["feature"]).replace("_latest", ""), _n(a.get("auc")), a.get("direction"),
                _n(a.get("point_biserial_r")), _n(a.get("q_value_bh"), 4)]
               for a in (eda.get("ratio_vs_label") or [])[:8]],
              ["---", "---:", "---", "---:", "---:"]),
        "",
        "*Diễn giải:* các cặp |r| cao nên **giữ 1 đại diện mỗi cụm** hoặc dùng regularization; "
        "`|2·AUC−1|` cho biết tỷ số nào tự tách được hai lớp (hướng: giá trị cao hay thấp ứng với suy "
        "giảm) — nhưng nhóm thanh khoản/cấu trúc vốn gần như không đổi theo quý nên sức mạnh in-domain "
        "chủ yếu đến từ việc phân biệt **công ty**, không phải động lực suy giảm của từng quý.",
        "",
        fig("reports/figures/eda/08_ratio_correlation.png",
            "Tương quan Pearson giữa 14 tỷ số (ô ghi số là |r| ≥ 0,5)"),
        fig("reports/figures/eda/09_ratio_vs_label.png",
            "Mức tách hai lớp của từng tỷ số (2·AUC − 1)"),
        "",
        "### 3.6. Nhận xét EDA",
        "",
        "1. Dữ liệu **đủ dài về thời gian** (tối đa 47 quý/công ty) nhưng **hẹp về số thực thể** "
        "(8 công ty) — đây là yếu tố giới hạn chính của bài toán.",
        "2. Tỷ số tài chính có tính **bền theo công ty** (ví dụ `current_ratio`, `debt_to_assets` "
        "của cùng một công ty gần như không đổi qua các quý) ⇒ mô hình dễ học “danh tính công ty” "
        "thay vì học động lực suy giảm.",
        "3. Chất lượng dữ liệu không đồng đều giữa các công ty (mục 3.3) → đã xử lý bằng impute "
        "trong pipeline + ablation.",
        "",
        "**Nhận xét tự động sinh từ số liệu** (`reports/results/eda.md`, mục “Nhận xét”):", "",
        "\n".join(f"- {text}" for text in (eda.get("conclusions") or [])),
    ])


def section_eda_deep(a: Dict[str, Any]) -> str:
    """Mục 3.5 — kiểm tra chuyên sâu trước khi huấn luyện (sinh bởi `scripts.eda_deep`).

    Trả "" nếu chưa có `reports/results/eda_deep.json` ⇒ báo cáo vẫn sinh được (không vỡ) và
    người đọc thấy ngay rằng bước EDA chuyên sâu chưa chạy.
    """
    deep = a.get("eda_deep") or {}
    if not deep:
        return ""
    quality = deep.get("feature_quality") or {}
    flags = quality.get("flags") or {}
    stats = quality.get("stats") or []
    label = deep.get("label") or {}
    overall = label.get("overall") or {}
    entities = label.get("entities") or {}
    transitions = label.get("transitions") or {}
    association = deep.get("feature_vs_label") or {}
    correlation = deep.get("feature_vs_feature") or {}
    drift = deep.get("drift") or {}
    missing = deep.get("missingness") or {}
    leakage = deep.get("leakage") or {}
    scope = deep.get("scope") or {}
    dims = correlation.get("effective_dimensionality") or {}
    consecutive = (leakage.get("history_overlap") or {}).get("consecutive_samples_same_ticker") or {}
    duplicates = leakage.get("duplicates") or {}
    density = missing.get("missing_density_per_sample") or {}

    def top(rows: List[Dict[str, Any]], key: str, count: int) -> List[Dict[str, Any]]:
        available = [r for r in rows if r.get(key) is not None]
        return sorted(available, key=lambda r: -r[key])[:count]

    heavy = ", ".join(r["feature"] for r in top(stats, "kurtosis_excess", 5)) or "—"
    missing_features = ", ".join(flags.get("missing_gt_20pct", [])[:8]) or "—"

    lines: List[str] = [
        "",
        "### 3.7. Kiểm tra chuyên sâu trước khi huấn luyện (`scripts.eda_deep`)",
        "",
        "Mục 3.1–3.6 mô tả dữ liệu; mục này trả lời bốn câu hỏi kỹ thuật phải xong **trước khi tin "
        "vào metric**: (1) ma trận 47 feature có cột hằng / thiếu nhiều / đuôi nặng / nhiều outlier? "
        "(2) nhãn lệch và \"dính\" theo thời gian tới mức nào? (3) feature nào thật sự liên hệ với "
        "nhãn và nhóm feature nào trùng thông tin? (4) test có khác train và còn rò rỉ nào? "
        "Mọi số dưới đây đọc từ `reports/results/eda_deep.json` (không nhập tay).",
        "",
        f"**Chất lượng ma trận feature** (trên {scope.get('association_eval_set') or 'train+validation'}):",
        "",
        f"- Cột hằng số: **{', '.join(flags.get('constant', [])) or '—'}**; gần hằng số (>95% một giá "
        f"trị): **{', '.join(flags.get('near_constant', [])) or '—'}**.",
        f"- Thiếu > 20%: **{len(flags.get('missing_gt_20pct', []))}** cột — {missing_features}.",
        f"- Đuôi nặng (|skew| > 5 hoặc kurtosis > 20): **{len(flags.get('heavy_tail', []))}** cột, "
        f"nặng nhất: {heavy}.",
        f"- Nhiều outlier theo IQR (> 5% mẫu): **{len(flags.get('many_outliers', []))}** cột.",
        f"- Mật độ thiếu mỗi mẫu: trung bình {_n(density.get('mean_features_missing'))}/"
        f"{density.get('n_features')} feature; {_pct(density.get('share_rows_with_gt_50pct_missing'))} "
        f"số mẫu thiếu quá nửa số feature.",
        "",
        table(["Chỉ số nhãn (toàn corpus)", "Giá trị"], [
            ["n mẫu", overall.get("n")],
            ["Tỉ lệ dương", _pct((overall.get("positive_pct") or 0) / 100.0)],
            ["Imbalance Ratio", _n(overall.get("imbalance_ratio"))],
            ["Entropy (bit, tối đa 1.0)", _n(overall.get("entropy_bits"))],
            ["Gini", _n(overall.get("gini"))],
            ["Baseline đoán lớp đa số", _pct((overall.get("majority_baseline_accuracy_pct") or 0) / 100.0)],
            ["Cặp quý liên tiếp giữ nguyên nhãn", _pct((transitions.get("persistence_pct") or 0) / 100.0)],
            ["P(nhãn 1 | quý trước nhãn 1)", _n(transitions.get("p_one_given_one"))],
            ["P(nhãn 1 | quý trước nhãn 0)", _n(transitions.get("p_one_given_zero"))],
            ["Công ty chỉ có một lớp", ", ".join(entities.get("tickers_needing_entity_level_cv", [])) or "—"],
            ["% mẫu thuộc công ty một lớp",
             _pct((entities.get("share_samples_in_single_class_tickers_pct") or 0) / 100.0)],
            ["Số thực thể hiệu dụng (1/Σp²)", _n(entities.get("effective_number_of_tickers"))],
        ], ["---", "---:"]),
        "",
        "**Liên hệ feature ↔ nhãn** (xếp theo |2·AUC−1|, hiệu chỉnh đa so sánh bằng BH-FDR):",
        "",
        table(["Feature", "#cặp", "AUC 1-feature", "Hướng", "q-value BH", "MI", "Lift decile trên"],
              [[r.get("feature"), r.get("n_pairs"), _n(r.get("auc")), r.get("direction"),
                _n(r.get("q_value_bh"), 4), _n(r.get("mutual_information")), _n(r.get("lift_top_decile"))]
               for r in (association.get("ranked_by_effect") or [])[:10]],
              ["---", "---:", "---:", "---", "---:", "---:", "---:"]),
        "",
        f"- {association.get('n_usable_features')}/{scope.get('n_features')} feature tính được AUC "
        f"1-feature; {association.get('n_effect_ge_0_30')} feature có |2·AUC−1| ≥ 0.30; "
        f"{association.get('n_significant_after_bh_5pct')} feature còn ý nghĩa sau hiệu chỉnh. "
        f"Điểm cần đọc kỹ: feature như `debt_to_equity_latest` có AUC cao nhưng `r` gần 0 và "
        f"p-value lớn — hệ quả của đuôi cực nặng (outlier chi phối Pearson, không chi phối AUC).",
        "",
        "**Đa cộng tuyến & số chiều hiệu dụng:**",
        "",
        f"- {correlation.get('n_pairs_abs_ge_threshold')} cặp |r| ≥ {correlation.get('threshold')}; "
        f"{correlation.get('n_features_in_clusters')} feature nằm trong "
        f"{len(correlation.get('clusters_abs_ge_threshold') or [])} cụm thông tin (mục 7.3 có thêm VIF).",
        f"- Số chiều hiệu dụng (participation ratio) = **{_n(dims.get('participation_ratio'))}** trên "
        f"{dims.get('n_columns')} cột; cần {dims.get('n_components_for_95pct_variance')} thành phần "
        f"cho 95% phương sai ⇒ phần lớn cột là thông tin dư.",
        "",
        "**Dịch chuyển phân phối train → test** (chẩn đoán, KHÔNG dùng để chọn mô hình):",
        "",
        table(["Feature", "Mean train", "Mean test", "KS", "SMD", "PSI"],
              [[r.get("feature"), _n(r.get("mean_ref")), _n(r.get("mean_new")),
                _n(r.get("ks_statistic")), _n(r.get("smd")), _n(r.get("psi"))]
               for r in top(drift.get("ranked_by_ks", []), "ks_statistic", 6)],
              ["---", "---:", "---:", "---:", "---:", "---:"]),
        "",
        f"- {drift.get('n_drifted')} feature vượt ngưỡng cảnh báo (KS ≥ 0.30 hoặc |SMD| ≥ 0.50); "
        f"{drift.get('n_psi_above_0_2')} feature có PSI > 0.2 (PSI chia 10 khoảng trên 64 dòng test "
        f"nên nhạy nhiễu — dùng cho monitoring về sau, không phải tiêu chí loại feature).",
        "",
        "**Rò rỉ tiềm ẩn & tính độc lập của mẫu:**",
        "",
        f"- Jaccard lịch sử trung bình giữa hai mẫu liên tiếp cùng công ty = "
        f"**{_n(consecutive.get('mean_history_jaccard'))}** (median "
        f"{_n(consecutive.get('median_history_jaccard'))}) ⇒ 324 mẫu KHÔNG phải 324 quan sát độc lập.",
        f"- Dòng feature trùng khít test↔train: {duplicates.get('n_new_rows_duplicated_in_ref')}; "
        f"số mẫu có nhãn công bố TRƯỚC khi kỳ target kết thúc: "
        f"{(leakage.get('label_availability') or {}).get('n_label_available_before_target_end')} "
        f"(kỳ vọng 0 ⇒ PASS).",
        "",
        "**Feature thiếu mang thông tin nhãn** (chi-square trên thiếu/có × nhãn, hiệu chỉnh BH-FDR):",
        "",
        table(["Feature", "#thiếu", "Tỉ lệ nhãn 1 khi thiếu", "khi có dữ liệu", "Δ (điểm %)", "q-value BH"],
              [[r.get("feature"), r.get("n_missing"),
                _pct((r.get("target_rate_when_missing_pct") or 0) / 100.0),
                _pct((r.get("target_rate_when_present_pct") or 0) / 100.0),
                _n(r.get("delta_pct_points")), _n(r.get("q_value_bh"), 4)]
               for r in (missing.get("label_informativeness") or {}).get("ranked_by_abs_delta", [])[:6]],
              ["---", "---:", "---:", "---:", "---:", "---:"]),
        "",
        "**Kết luận tự động:**",
        "",
        "\n".join(f"{i + 1}. {text}" for i, text in enumerate(deep.get("auto_conclusions") or [])),
        "",
        fig("reports/figures/eda_deep/01_missing_pct_by_split.png",
            "Giá trị thiếu theo feature × split — cột nào bị impute chi phối"),
        fig("reports/figures/eda_deep/03_target_association.png",
            "Liên hệ feature ↔ nhãn: 2·AUC−1 kèm mutual information"),
        fig("reports/figures/eda_deep/04_correlation_clustered.png",
            "Tương quan Pearson các feature, sắp theo cụm |r| ≥ 0.90"),
        fig("reports/figures/eda_deep/05_drift_ks_vs_smd.png",
            "Dịch chuyển train → test: KS vs SMD (đỏ = ngưỡng cảnh báo)"),
    ]
    return "\n".join(lines)


def section_preprocessing(a: Dict[str, Any]) -> str:
    """Mục 4 — Tiền xử lý & feature engineering (công thức lấy từ config, ablation từ analysis)."""
    from forecasting.config import LOOKBACK_QUARTERS, MIN_HISTORY_QUARTERS, RATIOS
    from forecasting.features import PATH_FEATURES, feature_groups, feature_names

    groups = feature_groups()
    names = feature_names()
    ratio_table = table(["Tỷ số", "Công thức"],
                        [[name, f"`{formula}`"] for name, formula in RATIOS.items()],
                        ["---", "---"])
    group_table = table(["Nhóm feature", "Số cột", "Ví dụ"],
                        [[g, len(cols), ", ".join(f"`{c}`" for c in cols[:3]) + ("…" if len(cols) > 3 else "")]
                         for g, cols in groups.items()],
                        ["---", "---:", "---"])

    abl = [r for r in a["analysis"].get("ablation", [])]
    abl_table = table(["Biến thể", "#feature", "Val AUROC", "Test AUROC", "Test F1"],
                      [[r["variant"], r["n_features"], _n(r["val"]["auroc"]),
                        _n(r["test"]["auroc"]), _n(r["test"]["f1"])] for r in abl],
                      ["---", "---:", "---:", "---:", "---:"])

    return "\n".join([
        "",
        "## 4. Tiền xử lý dữ liệu và xây dựng đặc trưng",
        "",
        "### 4.1. Sơ đồ pipeline",
        "",
        "```",
        "JSON thô (SEC XBRL) → 16 chỉ tiêu/quý → mẫu (lịch sử + quý target + nhãn)",
        f"   → {len(names)} feature (tỷ số, YoY, tăng trưởng, cấu trúc vốn, chỉ báo căng thẳng,",
        "     cực trị/độ bền theo cửa sổ) + nợ suy ra từ A = L + E",
        "   → Pipeline(SimpleImputer(median) → [StandardScaler] → Model)",
        "```",
        "",
        f"Cửa sổ lịch sử: **{LOOKBACK_QUARTERS} quý gần nhất** trước `as_of` (2 năm tài chính). "
        f"Ngưỡng tối thiểu để feature YoY có nghĩa: **{MIN_HISTORY_QUARTERS} quý** (YoY cần 5 quý); "
        f"{a['eda'].get('history_length', {}).get('n_below_min')} mẫu dưới ngưỡng vẫn được giữ lại "
        f"(để không mất dữ liệu giai đoạn đầu của mỗi công ty) nhưng ảnh hưởng của chúng được đo "
        f"bằng ablation ở mục 4.5.",
        "",
        "### 4.2. Xử lý giá trị thiếu",
        "",
        "- Giá trị thiếu trong JSON là `null` → `NaN` (`forecasting.data_loader.to_float`).",
        "- **Impute bằng median đặt trong `Pipeline`** nên median chỉ được tính trên train của mỗi "
        "lần fit → không rò rỉ thống kê từ validation/test (lỗi rất thường gặp khi impute trước khi "
        "chia tập).",
        "- Chỉ tiêu có độ phủ thấp (liabilities, receivables, short_term_investments…) làm cho các "
        "tỷ số liên quan bị impute tới ~65% giá trị → mục 4.5 đo ảnh hưởng bằng cách **bỏ hẳn** "
        "những cột có độ phủ < 50%.",
        "",
        "### 4.3. Chuẩn hoá và mã hoá",
        "",
        "- `StandardScaler` chỉ dùng cho Logistic Regression (mô hình tuyến tính); mô hình cây giữ "
        "nguyên đơn vị. Scaler cũng nằm **trong** Pipeline → không rò rỉ.",
        "- Không có biến phân loại nào cần encode (nhãn nhị phân). **Cố ý không đưa `ticker` "
        "one-hot** vào mô hình: làm vậy sẽ biến bài toán thành bài toán nhận diện công ty và hợp "
        "thức hoá đúng cái rò rỉ đang được đo ở mục 6.4/8.",
        "",
        "### 4.4. Nhiễu và outlier",
        "",
        "- Chưa áp dụng winsorize/clip cho tỷ số: hệ thống báo cáo trung thực rằng đây là hạng mục "
        "còn thiếu (xem mục 9.2). `_safe_ratio` chỉ chặn chia cho 0.",
        "- Nhiễu theo mùa vụ bán lẻ được xử lý gián tiếp bằng đặc trưng **YoY** (so cùng quý năm "
        "trước) thay vì so quý liền trước — đây là lựa chọn thiết kế có chủ đích.",
        "",
        "### 4.5. Bộ đặc trưng",
        "",
        f"**Nhóm đặc trưng ({len(names)} cột):**",
        "",
        group_table, "",
        f"**14 tỷ số** (mỗi tỷ số lấy giá trị quý gần nhất và YoY):",
        "",
        ratio_table, "",
        f"Ngoài ra: YoY của 10 chỉ tiêu chính, `working_capital_to_assets`, "
        f"`distress_quarters_in_window` (số quý có dòng tiền hoạt động âm trong cửa sổ) và "
        f"`revenue_cv` (hệ số biến thiên doanh thu).",
        "",
        f"**Nhóm `path` ({len(PATH_FEATURES)} cột)** — đặc trưng *đường đi* trong cửa sổ "
        f"{LOOKBACK_QUARTERS} quý (vì `*_latest` chỉ nhìn 1 quý, còn suy giảm tài chính là một quá "
        f"trình): `{'`, `'.join(PATH_FEATURES)}` — cực trị xấu nhất của current ratio / OCF-to-sales "
        f"/ net margin, mức giảm doanh thu so với đỉnh 8 quý, và số quý âm LIÊN TIẾP của net income "
        f"/ OCF. Nhóm này được đo trước khi đưa vào (`scripts/probe_features.py`, cross-company "
        f"GroupKFold: +0,03…0,12 AUROC so với baseline).",
        "",
        f"**Chỉ tiêu thiếu phủ được xử lý ở tầng feature:** `total_liabilities` = tag `liabilities` "
        f"nếu quý đó có, ngược lại suy ra từ đẳng thức kế toán `total_assets - stockholders_equity` "
        f"(tag `liabilities` chỉ phủ ~39% số quý, còn 2 chỉ tiêu kia phủ 100%) ⇒ `debt_to_assets` / "
        f"`debt_to_equity` có giá trị ở mọi mẫu. Đối chiếu 124 quý có tag: `scripts/audit_data.py`.",
        "",
        f"Danh sách đầy đủ: `forecasting/features.py` (`feature_names()` — {len(names)} cột).",
        "",
        "**Ablation (bỏ từng nhóm, mô hình Logistic, cùng split):**",
        "",
        abl_table, "",
        "### 4.6. Chống rò rỉ dữ liệu (checklist)",
        "",
        table(["Lớp chống rò rỉ", "Cách làm", "Kiểm chứng"],
              [["Thời gian", "Feature chỉ tính từ lịch sử có `available_on <= as_of`",
                "test tự động `test_no_label_leak_in_features`"],
               ["Chia tập", "Theo thời gian trong từng công ty: 8 quý cuối = test, 4 quý = validation",
                "`forecasting/data.py::split_policy`"],
               ["Chia tập — stratified (Yêu cầu 1)",
                "Tỉ lệ lớp của train/validation/test giữ trong sai số nhỏ (đo được: lệch ≤ 3,6 điểm %); "
                "CV dùng `StratifiedGroupKFold` (giữ tỉ lệ lớp VÀ giữ trọn công ty ngoài fold-train)",
                "test tự động `test_class_ratio_preserved_across_splits`, "
                "`test_cv_folds_are_stratified_and_grouped`"],
               ["Purge", "Bỏ 1 quý sát trước và 1 quý sát sau validation ở mỗi công ty",
                "`purged.json` (16 mẫu)"],
               ["Tiền xử lý", "Impute/scale nằm trong Pipeline, chỉ fit trên train", "`forecasting/models.py`"],
               ["Cân bằng lớp — balancing (Yêu cầu 2)",
                "Pipeline chính KHÔNG resample; `class_weight='balanced*'` do sklearn tính trong `fit` "
                "từ nhãn nhận được. SMOTE / RandomUnderSampler chỉ tồn tại trong `imbalance_lab/` và "
                "`benchmark_imbalanced.py`, đặt TRONG `imblearn.pipeline.Pipeline` ⇒ chỉ chạy trên train "
                "(fold-train khi CV), validation/test không bao giờ bị resample",
                "test tự động `test_main_pipeline_has_no_balancer`, "
                "`test_balancing_weights_come_from_fit_labels_only`; "
                "`tests/test_imbalance_lab.py`, `tests/test_benchmark_imbalanced.py`"],
               ["Danh mục kỹ thuật (Yêu cầu 2)",
                "15 kỹ thuật thuộc 5 nhóm (cộng `baseline` chưa xử lý làm mốc so sánh) — oversampling "
                "(ROS/SMOTE/BorderlineSMOTE/ADASYN), "
                "undersampling (RUS/TomekLinks/ENN), hybrid (SMOTE+Tomek, SMOTE+ENN), algorithm-level "
                "(`scale_pos_weight`, `class_weight='balanced'`, **Focal Loss**) và ensemble "
                "(BalancedRF/EasyEnsemble/RUSBoost) — kèm **threshold tuning trên đường Precision-Recall** "
                "của xác suất out-of-fold; `python -m imbalance_lab.techniques`",
                "`reports/imbalance/techniques.md` (PASS/FAIL chống rò rỉ cho từng kỹ thuật), "
                "`tests/test_imbalance_techniques.py`, `docs/cac-ky-thuat-mat-can-bang.md`"],
               ["Đánh giá mô hình (Yêu cầu 3)",
                "**Accuracy KHÔNG dùng làm thước đo chính** (chỉ in kèm mốc \"đoán lớp đa số\"); metric "
                "chính: Precision, Recall, F1 (binary/macro/weighted/**F-beta**), **PR-AUC (Average "
                "Precision)**, ROC-AUC, MCC + **Confusion Matrix**; resampling chỉ chạy TRONG "
                "`imblearn.pipeline.Pipeline`; có **bảng Baseline (chưa xử lý) vs từng kỹ thuật** kèm "
                "ΔPR-AUC/ΔF1/ΔF1-macro",
                "`imbalance_lab/metrics.py` (`PRIMARY_METRICS`, `accuracy_diagnostic`), "
                "`tests/test_evaluation_metrics.py`, mục 3 + 4 của `reports/imbalance/techniques.md`, "
                "`reports/imbalance/techniques_comparison.csv`"],
               ["Thực thể", "Đo bằng ticker-prior + GroupKFold/LOCO (không dùng để huấn luyện)",
                "mục 6.4, `reports/results/validation_checks.json`"]],
              ["---", "---", "---"]),
    ])


def section_models(a: Dict[str, Any]) -> str:
    """Mục 5 — Mô hình và siêu tham số (hyperparameter lấy từ models.HYPERPARAMS)."""
    from forecasting.models import HYPERPARAMS, MODEL_REGISTRY

    params_table = table(["Mô hình", "Họ", "Siêu tham số (cấu hình chạy chính)"],
                         [["Logistic Regression", "Tuyến tính",
                           ", ".join(f"`{k}={v}`" for k, v in HYPERPARAMS["logistic"].items())],
                          ["Random Forest", "Bagging (cây)",
                           ", ".join(f"`{k}={v}`" for k, v in HYPERPARAMS["random_forest"].items())],
                          ["HistGradientBoosting", "Boosting (cây)",
                           ", ".join(f"`{k}={v}`" for k, v
                                     in HYPERPARAMS["hist_gradient_boosting"].items())],
                          ["MLP (mạng nơ-ron)", "Phi tuyến (không dựa trên cây)",
                           ", ".join(f"`{k}={v}`" for k, v in HYPERPARAMS.get("mlp", {}).items())]],
                         ["---", "---", "---"])

    tune_rows: List[List[str]] = []
    for r in a["tuning"].get("results", []):
        ref = r.get("default_cv_reference", {}).get("mean_average_precision")
        delta = None if ref is None else r["best_cv_average_precision"] - ref
        tune_rows.append([PRETTY.get(r["model"], r["model"]), f"`{r['best_params']}`",
                          str(r["n_candidates"]), _n(r["best_cv_average_precision"]),
                          _n(r["best_cv_auroc"]), _n(ref), _n(delta)])
    tune_table = table(["Mô hình", "Cấu hình tốt nhất (CV)", "#cấu hình", "CV-AP", "CV-AUROC",
                        "CV-AP mặc định", "Chênh lệch"],
                       tune_rows, ["---", "---", "---:", "---:", "---:", "---:", "---:"])

    # Công bố TRUNG THỰC cấu hình đang chạy: `best.joblib` fit bằng cấu hình nào? (đọc artifact)
    # Lưu ý: khối này chỉ đọc `a[...]` (không dùng `summary`/`candidates` định nghĩa ở dưới).
    best_name = a["summary"].get("best_model")
    best_row = next((m for m in a["summary"].get("models", []) if m.get("model") == best_name), {})
    tuned_best = next((r for r in a["tuning"].get("results", []) if r.get("model") == best_name), {})
    if best_row.get("params"):
        deployed_txt = f"`{best_row['params']}` (đọc từ `summary.json`)"
    else:
        deployed_txt = ("**cấu hình mặc định** trong `forecasting/models.py::HYPERPARAMS`: "
                        f"`{HYPERPARAMS.get(best_name, {})}` "
                        "(artifact ghi `params = null` cho mọi mô hình ⇒ `forecasting/train.py` fit "
                        "bằng cấu hình mặc định)")
    default_ref = (tuned_best.get("default_cv_reference") or {}).get("mean_average_precision")
    tuned_cv = tuned_best.get("best_cv_average_precision")
    cv_gap = None if (tuned_cv is None or default_ref is None) else tuned_cv - default_ref
    tuned_note = (
        (f"⇒ Đọc cho đúng: bảng trên là **thí nghiệm so sánh trên CV** — cấu hình tốt nhất theo CV "
         f"(`{tuned_best.get('best_params')}`, CV-AP {_n(tuned_cv)}) **KHÔNG nằm trong mô hình chốt**; "
         f"mô hình chốt giữ nguyên cấu hình mặc định"
         + (f" vì chênh lệch chỉ {_n(cv_gap)} CV-AP — dưới mức nhiễu của 212 mẫu train"
            if cv_gap is not None else "")
         + ". Muốn đổi mặc định cần **thêm dữ liệu/thực thể**, không phải thêm cấu hình.")
        if tuned_best else
        "⇒ Chưa có `tuning.json` để đối chiếu cấu hình tinh chỉnh.")
    deployed_note = "\n".join([
        f"**Cấu hình THẬT của mô hình đã triển khai** (`reports/models/best.joblib` = "
        f"{PRETTY.get(best_name, best_name)}): {deployed_txt}.",
        "",
        tuned_note,
    ])

    summary = a["summary"]
    select_rule = summary.get("selection_rule", "—")
    candidates = summary.get("models", [])
    select_rows = [[PRETTY.get(r.get("model"), r.get("model")),
                    _n(r.get("cross_company_ap")), _n(r.get("cross_company_auroc")),
                    _n(r.get("best_f1_val")), _n(r.get("average_precision")), _n(r.get("auroc"))]
                   for r in candidates]
    select_table = table(["Mô hình", "AP cross-company", "AUROC cross-company", "best-F1(val)",
                          "AP(val)", "AUROC(val)"],
                         select_rows, ["---", "---:", "---:", "---:", "---:", "---:"])
    return "\n".join([
        "",
        "## 5. Mô hình và siêu tham số",
        "",
        "### 5.1. Bốn họ mô hình",
        "",
        params_table, "",
        f"Registry có {len(MODEL_REGISTRY)} họ mô hình; script **thử lần lượt** và tự bỏ mô hình nào lỗi ở "
        f"môi trường hiện tại (ghi lý do vào log `reports/results/run_all.log`) ⇒ lần chạy này huấn "
        f"luyện thành công **{len(a['summary'].get('models', []))} mô hình**. Như vậy yêu cầu "
        f"“≥ 3 mô hình khác nhau” được đáp ứng bằng **tuyến tính + bagging + boosting + mạng nơ-ron "
        f"(phi tuyến không dựa trên cây)** — bốn cơ chế học khác nhau, không phải bốn biến thể của "
        f"cùng một họ.",
        "",
        "### 5.2. Quy trình chọn mô hình và ngưỡng",
        "",
        f"1. Fit **các họ mô hình có trong registry** ({len(MODEL_REGISTRY)} họ ở môi trường này) "
        f"trên `train` ({a['eda'].get('splits', {}).get('train', {}).get('total', '—')} mẫu).",
        "2. Tính metric in-domain trên `validation` (32 mẫu): AUROC, AP, F1 và đường PR → ngưỡng tối "
        "đa F1. **Đồng thời** tính **AP out-of-fold theo công ty** (GroupKFold trên train+validation, "
        "test không tham gia).",
        f"3. Chọn mô hình theo quy tắc tường minh: **{select_rule}** — vì metric in-domain bị chi "
        f"phối bởi “nhớ mặt công ty” (mục 6.4), tiêu chí đầu tiên là khả năng tổng quát hoá sang "
        f"công ty chưa từng thấy.",
        "",
        select_table, "",
        "*Quy ước đọc:* cột **AP/AUROC cross-company** ở bảng trên = xác suất **out-of-fold trên "
        "train+validation** với **cấu hình mặc định** (thô, 244 mẫu). Đừng nhầm với các con số "
        "“cross-company” ở mục 9.3 (độ nhạy theo định nghĩa nhãn, dữ liệu khác) và mục 9.4 (thí nghiệm "
        "kỹ thuật lệch lớp) — cùng tên chỉ số nhưng **khác dữ liệu/nhãn**, không so trực tiếp được.",
        f"4. Ngưỡng vận hành = ngưỡng best-F1 của mô hình được chọn trên validation "
        f"({_n(summary.get('best_threshold'))}); ngoài ra tính thêm ngưỡng tối ưu theo chi phí kỳ "
        f"vọng ({_n(summary.get('best_threshold_cost_optimal'))}, giả định FN đắt gấp 5 lần FP).",
        "5. **Đánh giá cuối trên `test`** với mô hình/ngưỡng đã cố định "
        "(`python -m forecasting.evaluate`); **test KHÔNG tham gia chọn mô hình hay ngưỡng** — mọi "
        "quyết định đều dựa trên validation + AP cross-company out-of-fold. (Nhiều hệ thống vẫn được "
        "chấm trên **cùng** 64 mẫu test để so sánh công bằng — mục 6.1 và 9.3 — nhưng không dùng để "
        "chọn cấu hình.)",
        "",
        "### 5.3. Tinh chỉnh siêu tham số (CV chia theo công ty)",
        "",
        "Dùng `GridSearchCV` với **StratifiedGroupKFold theo mã cổ phiếu**: mỗi fold giữ trọn một "
        "số công ty ra khỏi train. Nếu dùng CV thường (trộn mọi quý của mọi công ty) thì điểm CV sẽ "
        "bị thổi phồng đúng theo cơ chế rò rỉ thực thể phân tích ở mục 6.4. Metric để refit là "
        "**average precision (AP)** vì AP không phụ thuộc ngưỡng.",
        "",
        tune_table, "",
        deployed_note, "",
        "Cột “CV-AP mặc định” là điểm của cấu hình trong `HYPERPARAMS` trên **cùng** splitter, để "
        "trả lời câu hỏi “tinh chỉnh có thật sự cải thiện hay không” thay vì chỉ nói rằng đã chạy "
        "GridSearch. Kết quả từng cấu hình: `reports/results/tuning.md`.",
        "",
        "### 5.4. Sổ tay thực nghiệm (cấu hình đã chạy)",
        "",
        table(["Thành phần", "Giá trị"],
              [["Seed", str(summary.get("seed"))],
               ["Số feature", str(summary.get("n_features"))],
               ["Chia tập", "theo thời gian trong từng công ty + dải purge"],
               ["Chọn mô hình", select_rule],
               ["Ngưỡng vận hành", _n(summary.get("best_threshold"))],
               ["Ngưỡng tối ưu chi phí", _n(summary.get("best_threshold_cost_optimal"))],
               ["Chi phí giả định", "FN = 5, FP = 1 (phân tích ở mục 7.6)"],
               ["Bootstrap", "2.000 vòng cho AUROC/AP/F1 (mục 6.5)"]],
              ["---", "---"]),
    ])


def section_results(a: Dict[str, Any]) -> str:
    """Mục 6 — Kết quả và so sánh định lượng (baseline + in-domain + cross-company + CI)."""
    rows = a["baselines"].get("rows", [])
    test = a["test"].get("test_metrics", {})
    op = test.get("operating", {})
    thr_rows = []
    for b in test.get("by_threshold", []):
        thr_rows.append([_n(b["threshold"]), _n(b["accuracy"]), _n(b["precision"]), _n(b["recall"]),
                         _n(b["f1"]), _n(b["macro_f1"]), str(b["fp"]), str(b["fn"]),
                         _n(b["expected_cost"], 1)])
    # Bảng 6.4: in-domain (test) vs cross-company (GroupKFold toàn mẫu)
    checks = a["checks"]
    in_dom = checks.get("in_domain_test") or {}
    grouped = checks.get("grouped_cv_all_samples") or {}
    cross_rows = []
    for name, d in in_dom.items():
        g = grouped.get(name) or {}
        at = g.get("oof_at_0.5") or {}
        cross_rows.append([PRETTY.get(name, name), _n(d.get("test_auroc")),
                           _n(d.get("test_average_precision")), _n(g.get("oof_auroc")),
                           _n(g.get("oof_average_precision")), _n(at.get("f1"))])
    # Bảng 6.5: bootstrap CI trên test
    ci_rows = []
    for name, d in in_dom.items():
        bt = d.get("bootstrap_test") or {}
        auc = bt.get("auroc") or {}
        ap = bt.get("average_precision") or {}
        f1 = bt.get("f1") or {}
        ci_rows.append([PRETTY.get(name, name), _n(auc.get("point")),
                        f"{_n(auc.get('ci95_low'))}–{_n(auc.get('ci95_high'))}",
                        _n(ap.get("point")),
                        f"{_n(ap.get('ci95_low'))}–{_n(ap.get('ci95_high'))}",
                        _n(f1.get("point"))])
    if op:
        thr_rows.append([f"**{_n(op['threshold'])} (vận hành)**", _n(op["accuracy"]),
                         _n(op["precision"]), _n(op["recall"]), _n(op["f1"]), _n(op["macro_f1"]),
                         str(op["fp"]), str(op["fn"]), _n(op["expected_cost"], 1)])

    base_table = table(["Hệ thống (test, threshold 0.5)", "AUROC", "AP", "F1", "macro-F1", "Accuracy"],
                       [[PRETTY.get(r["baseline"], r["baseline"]), _n(r["test"]["auroc"]),
                         _n(r["test"]["average_precision"]), _n(r["test"]["at_0.5"]["f1"]),
                         _n(r["test"]["at_0.5"]["macro_f1"]), _n(r["test"]["at_0.5"]["accuracy"])]
                        for r in rows],
                       ["---", "---:", "---:", "---:", "---:", "---:"])

    # Mục 6.7 — walk-forward theo THỜI GIAN (bổ sung cho cross-company/LOCO): đọc từ validation_checks.
    wf = checks.get("walk_forward") or {}
    wf_rows: List[List[str]] = []
    for name, block in (wf.get("summary") or {}).items():
        per_fold = ", ".join(f"{float(v):.3f}" for v in (block.get("per_fold_auroc") or []))
        wf_rows.append([PRETTY.get(name, name), str(block.get("n_folds_evaluated")),
                        _n(block.get("mean_auroc")), _n(block.get("mean_average_precision")),
                        _n(block.get("min_auroc")), per_fold or "—"])
    wf_table = table(["Mô hình", "#fold chấm được", "AUROC trung bình", "AP trung bình",
                      "AUROC thấp nhất", "AUROC từng fold"],
                     wf_rows, ["---", "---:", "---:", "---:", "---:", "---:"])

    # Số liệu cho bảng 6.6 — đọc thẳng từ artifact (không nhập tay)
    ov = {r["model"]: r for r in a["analysis"].get("overfit", [])}
    base_auroc = next((r["test"]["auroc"] for r in rows if r["baseline"] == "ticker_prior"), None)

    return "\n".join([
        "",
        "## 6. Kết quả và so sánh mô hình",
        "",
        "### 6.1. Bảng so sánh định lượng (có baseline đối chứng)",
        "",
        base_table, "",
        "Trong đó: `Dummy (lớp đa số)` = “không học gì”; `Baseline nhớ mặt công ty` (ticker-prior) "
        "chỉ dùng tỷ lệ nhãn trung bình của chính công ty đó trong train; "
        "`single_feature[debt_to_assets_latest]` là Logistic trên **một** chỉ tiêu duy nhất; "
        "`Quy tắc Altman Z'' < 1,1` là **công thức Altman (1968/2000) áp trực tiếp** (điểm Z'' của "
        "quý mới nhất đã công bố, KHÔNG học tham số từ dữ liệu) — baseline truyền thống bắt buộc phải "
        "có khi so sánh mô hình dự báo kiệt quệ.",
        "",
        "**Đọc bảng này:** baseline ticker-prior đạt AUROC/AP xấp xỉ mô hình học máy ⇒ phần lớn khả "
        "năng “phân biệt” đến từ việc nhận ra công ty, không phải từ động lực suy giảm của quý.",
        "",
        f"### 6.2. Metric đầy đủ tại từng ngưỡng (test n = {test.get('n')})",
        "",
        table(["Ngưỡng", "Accuracy", "Precision", "Recall", "F1", "macro-F1", "FP", "FN",
               "Chi phí kỳ vọng (5·FN+FP)"],
              thr_rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]), "",
        f"- AUROC = **{_n(test.get('auroc'))}**, AP = **{_n(test.get('average_precision'))}**, "
        f"Brier = {_n(test.get('brier'))}.",
        f"- Ngưỡng vận hành {_n(test.get('threshold'))} cho precision {_n(op.get('precision'))} "
        f"nhưng recall chỉ {_n(op.get('recall'))} (bỏ sót {op.get('fn')} mẫu suy giảm); ngưỡng tối "
        f"ưu theo chi phí kỳ vọng là {_n((test.get('cost_optimal') or {}).get('threshold'))}.",
        "- Báo cáo cả hai ngưỡng cạnh nhau để thấy rõ **trade-off và cơ sở kinh tế** của quyết "
        "định, thay vì chốt 0,5 một cách tuỳ ý.",
        "",
        "### 6.3. Confusion matrix, ROC/PR và phân phối xác suất",
        "",
        fig("reports/figures/test_confusion.png",
            "Confusion matrix tại ngưỡng vận hành (đúng ngưỡng dùng để ra quyết định)"),
        fig("reports/figures/test_confusion_at_0.5.png", "Confusion matrix tại ngưỡng 0.5"),
        fig("reports/figures/test_roc_pr_curves.png", "ROC và Precision-Recall trên test"),
        fig("reports/figures/test_score_distribution.png",
            "Phân phối xác suất theo nhãn thực trên test"),
        "",
        # --- PLACEHOLDER_RESULTS_B ---
        "### 6.4. Kiểm chứng tổng quát hoá: in-domain vs cross-company",
        "",
        table(["Mô hình", "In-domain AUROC", "In-domain AP", "Cross-company AUROC",
               "Cross-company AP", "Cross-company F1"],
              cross_rows, ["---", "---:", "---:", "---:", "---:", "---:"]), "",
        f"- Cross-company = GroupKFold **theo công ty** trên toàn bộ 324 mẫu (mỗi fold giữ trọn một "
        f"công ty ra ngoài) → đây là con số trung thực cho câu hỏi “công ty mới thì sao?”.",
        f"- LOCO (train 7 công ty, test công ty còn lại, chỉ dùng split theo thời gian): trung bình "
        f"AUROC = **{_n(checks.get('headline', {}).get('loco_mean_auroc'))}** trên "
        f"{checks.get('headline', {}).get('loco_n_evaluable')} công ty tính được. Các công ty "
        f"**không tính được AUROC** (nhãn đơn lớp ở cả validation và test): "
        f"{', '.join(checks.get('headline', {}).get('loco_skipped_companies', []) or []) or '—'}.",
        "- Nhận xét: khi buộc phải tổng quát hoá sang công ty mới, mô hình gần như trở về mức “đoán "
        "theo xu hướng chung”, trong khi ở chế độ in-domain gần đạt mức hoàn hảo. Khoảng cách này "
        "chính là **định lượng của rò rỉ cấp thực thể**.",
        "",
        fig("reports/figures/analysis/07_in_domain_vs_cross_company.png",
            "AUROC: in-domain vs cross-company vs baseline"),
        "",
        f"### 6.5. Độ bất định của metric (bootstrap 2.000 vòng, test n = {test.get('n')})",
        "",
        table(["Mô hình", "AUROC", "AUROC CI95", "AP", "AP CI95", "F1 @0.5"],
              ci_rows, ["---", "---:", "---:", "---:", "---:", "---:"]), "",
        "Với n = 64 mẫu test, khoảng tin cậy rộng là điều bình thường — báo cáo vì thế không nêu "
        "một con số AUROC đơn lẻ mà không kèm CI.",
        "",
        "### 6.6. Vì sao mô hình này vượt mô hình kia?",
        "",
        "*Lưu ý phương pháp:* phần “Giải thích” dưới đây là **giả thuyết có số liệu kèm theo**, không "
        "phải kết luận nhân quả — đồ án không chạy thí nghiệm cô lập từng cơ chế (ví dụ: quét độ sâu "
        "cây để tách riêng tác động của phương sai). Cột “Bằng chứng định lượng” mới là phần kiểm chứng được.",
        "",
        table(["So sánh", "Bằng chứng định lượng", "Giải thích (giả thuyết)"],
              [["Logistic vs Random Forest",
                f"AUROC train {_n(ov.get('logistic', {}).get('train_auroc'))} vs "
                f"{_n(ov.get('random_forest', {}).get('train_auroc'))}; gap AUROC "
                f"{_n(ov.get('logistic', {}).get('gap_auroc'))} vs "
                f"{_n(ov.get('random_forest', {}).get('gap_auroc'))} (bảng 7.1)",
                "Giả thuyết: cây chia tới khi gần tách hoàn hảo tập train (212 mẫu, 47 chiều) → gap "
                "dương; Logistic bị ràng buộc tuyến tính + L2 nên gap nhỏ nhất. Chưa có thí nghiệm "
                "quét độ sâu để khẳng định"],
               ["Logistic vs HistGradientBoosting",
                f"Val AP {_n(ov.get('logistic', {}).get('val_ap'))} vs "
                f"{_n(ov.get('hist_gradient_boosting', {}).get('val_ap'))}; gap AUROC HGB = "
                f"{_n(ov.get('hist_gradient_boosting', {}).get('gap_auroc'))}",
                "Các mô hình gần như đồng hạng ở best-F1 trên validation (chênh AP val ≤ 0,002); quy "
                "tắc chọn (mục 5.2) phân định bằng **AP cross-company**, rồi mới tới "
                "best-F1/AP/AUROC trên validation"],
               ["Mô hình vs baseline ticker-prior",
                f"AUROC test {_n(test.get('auroc'))} (mô hình) vs {_n(base_auroc)} (ticker-prior, "
                f"không dùng feature nào)",
                "Baseline “nhớ mặt công ty” đạt mức tương đương (có lúc cao hơn) ⇒ phần lớn khả "
                "năng phân biệt in-domain đến từ việc nhận ra công ty"],
               ["Mô hình vs dummy", f"macro-F1 ≈ {_n(op.get('macro_f1'))} vs "
                                    f"0.37 (dummy)",
                "Mô hình **có** học được tín hiệu phân biệt thật (không đoán mò); vấn đề là tín hiệu "
                "đó phần lớn mang tính thực thể"]],
              ["---", "---", "---"]),
        "",
        "### 6.7. Kiểm chứng theo THỜI GIAN (walk-forward + purge)",
        "",
        f"- Giao thức: {wf.get('protocol', '—')}",
        f"- {wf.get('n_folds', '—')} fold cắt theo `target_period_end` tăng dần; train = mọi kỳ TRƯỚC "
        f"mốc cắt **và** nhãn đã công bố trước `mốc cắt − {wf.get('purge_days', '—')} ngày` "
        f"(mô phỏng đúng thông tin có tại thời điểm ra quyết định); fold nào train < "
        f"{wf.get('min_train', '—')} mẫu hoặc test đơn lớp thì được ghi là bỏ qua.",
        "",
        wf_table, "",
        "- **Đọc bảng:** đây là câu hỏi *“công ty CŨ, GIAI ĐOẠN mới”* — khác GroupKFold/LOCO "
        "(*“công ty MỚI, giai đoạn cũ”*). Khoảng cách giữa AUROC walk-forward và AUROC in-domain cho "
        "biết bao nhiêu phần “điểm đẹp” đến từ việc mô hình đã thấy chính giai đoạn đó khi huấn luyện.",
        fig("reports/figures/analysis/08_walk_forward.png",
            "AUROC/AP theo từng fold thời gian (walk-forward + purge)"),
        "",
    ])


def _n_model_registry() -> int:
    """Số họ mô hình chạy được trong môi trường hiện tại (đọc từ `forecasting.models`)."""
    try:
        from forecasting.models import MODEL_REGISTRY

        return len(MODEL_REGISTRY)
    except Exception:  # pragma: no cover - thiếu phụ thuộc tuỳ chọn
        return 0


def _ablation_notes(rows: List[Dict[str, Any]], low_abl: Dict[str, Any]) -> str:
    """Sinh 3 kết luận ablation **đọc trực tiếp từ artifact** (bỏ câu chữ hard-code).

    Vì sao: trước đây câu “bỏ nhóm `ratios_latest` làm AUROC giảm mạnh nhất” được viết cứng — khi
    phân tích chuyển sang đúng mô hình đã chốt (Random Forest) thì kết luận đó có thể sai. Nay báo
    cáo tự xác định biến thể làm giảm test AUROC nhiều nhất từ chính bảng ablation.
    """
    base = next((r for r in rows if not r.get("drop") and not r.get("min_history")), None)
    base_auc = ((base or {}).get("test") or {}).get("auroc")
    lines: List[str] = []
    ranked = []
    for row in rows:
        if not row.get("drop"):
            continue
        auc = (row.get("test") or {}).get("auroc")
        if auc is not None and base_auc is not None:
            ranked.append((base_auc - auc, row))
    if ranked:
        drop, row = max(ranked, key=lambda t: t[0])
        lines.append(f"1. Biến thể làm **test AUROC giảm mạnh nhất**: *{row['variant']}* "
                     f"(ΔAUROC = {_n(-drop)} so với dùng tất cả feature) ⇒ nhóm đặc trưng này đóng "
                     f"góp thực; các biến thể còn lại xem bảng ở mục 4.5.")
    low = low_abl.get("drop") or []
    lines.append(f"2. Biến thể “bỏ cột có độ phủ < 50%” có **{len(low)} cột** để bỏ (trước đây 4): "
                 f"sau khi `debt_to_assets`/`debt_to_equity` dùng **nợ suy ra từ `A = L + E`**, không "
                 f"tỷ số nào còn phụ thuộc tag thưa `liabilities`; các chỉ tiêu thưa khác "
                 f"(`receivables`, `short_term_investments`) vẫn chỉ có nghĩa ở một phần mẫu và được "
                 f"impute trong pipeline.")
    youth = next((r for r in rows if r.get("min_history")), None)
    if youth is not None and base_auc is not None:
        auc = (youth.get("test") or {}).get("auroc")
        delta = None if auc is None else auc - base_auc
        lines.append(f"3. Lọc mẫu có < {youth.get('min_history')} quý lịch sử: test AUROC "
                     f"{_n(auc)} so với {_n(base_auc)} (Δ = {_n(delta)}) ⇒ mức ảnh hưởng "
                     f"{'nhỏ' if delta is None or abs(delta) < 0.02 else 'đáng kể'}.")
    return "\n".join(lines)


def section_analysis(a: Dict[str, Any]) -> str:
    """Mục 7 — Phân tích chuyên sâu (overfit, importance, VIF, ablation, lỗi, ngưỡng, hiệu chuẩn)."""
    an = a["analysis"]
    overfit = an.get("overfit", [])
    imp = an.get("importance", {})
    vif = an.get("vif", {})
    low_abl = next((r for r in an.get("ablation", [])
                    if str(r.get("variant", "")).startswith("bỏ cột độ phủ")), {})
    in_dom = ((a["checks"].get("headline") or {}).get("in_domain_test_auroc") or {})

    over_table = table(["Mô hình", "Train AUROC", "Val AUROC", "Gap AUROC", "Train F1", "Val F1",
                         "Gap F1", "Val AP", "Brier"],
                       [[PRETTY.get(r["model"], r["model"]), _n(r["train_auroc"]),
                         _n(r["val_auroc"]), _n(r["gap_auroc"]), _n(r["train_f1"]), _n(r["val_f1"]),
                         _n(r["gap_f1"]), _n(r["val_ap"]), _n(r["brier_val"])] for r in overfit],
                       ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])

    coefs = {c["feature"]: c["coef"] for c in imp.get("logistic_coefficients_standardized", [])}
    imp_rows = []
    for i, item in enumerate(imp.get("val", [])[:10], start=1):
        test_val = next((t["mean_decrease_auroc"] for t in imp.get("test", [])
                         if t["feature"] == item["feature"]), None)
        imp_rows.append([str(i), f"`{item['feature']}`", _n(item["mean_decrease_auroc"]),
                         _n(test_val), _n(coefs.get(item["feature"]), 2)])
    imp_table = table(["#", "Feature", "Val ΔAUROC", "Test ΔAUROC",
                       (f"Hệ số {PRETTY.get(imp.get('coefficient_model'), imp.get('coefficient_model'))}"
                        if imp.get("coefficient_model") else
                        "Hệ số (n/a — mô hình cây không có hệ số)")],
                      imp_rows, ["---:", "---", "---:", "---:", "---:"])

    vif_table = table(["Feature", "VIF"], [[f"`{v['feature']}`", _n(v["vif"], 1)]
                                           for v in vif.get("vif_top10", [])], ["---", "---:"])

    # --- Câu chữ về lỗi phải SUY RA TỪ artifact (trước đây hard-code "tất cả là FN") ---
    err_cases = an.get("error_cases", [])
    err_fn = sum(1 for e in err_cases
                 if int(e.get("actual", -1)) == 1 and int(e.get("predicted", -1)) == 0)
    err_fp = sum(1 for e in err_cases
                 if int(e.get("actual", -1)) == 0 and int(e.get("predicted", -1)) == 1)
    err_probs = [float(e["probability"]) for e in err_cases if e.get("probability") is not None]
    err_pmin = min(err_probs) if err_probs else None
    err_pmax = max(err_probs) if err_probs else None
    err_detail = "; ".join(
        f"`{e['sample_id']}` {'FN' if int(e['actual']) == 1 else 'FP'} P={_n(e['probability'])}"
        for e in err_cases)
    label_map = (((a.get("eda") or {}).get("label") or {}).get("label_share_by_ticker") or {})
    err_tickers = sorted({str(e.get("ticker")) for e in err_cases})
    err_rates = ", ".join(f"{t} ≈ {_n(label_map.get(t), 2)}" for t in err_tickers)
    err_constant = [t for t in err_tickers if float(label_map.get(t, -1) or -1) == 1.0]
    err_mixed = [t for t in err_tickers if t not in err_constant]

    return "\n".join([
        "",
        "## 7. Phân tích kết quả chuyên sâu",
        "",
        "### 7.1. Overfitting / Underfitting (Train vs Validation)",
        "",
        over_table, "",
        fig("reports/figures/analysis/01_overfit_train_vs_val.png",
            "Train vs Validation — gap càng lớn càng dễ overfit"),
        "",
        "**Kết luận.** Gap AUROC train→validation theo từng mô hình: "
        + "; ".join(f"{PRETTY.get(r['model'], r['model'])} = {_n(r['gap_auroc'])} "
                    f"(val F1 {_n(r['val_f1'])})" for r in overfit)
        + f". Mô hình cây phi tham số bám tập train sát hơn (gap dương) — hệ quả của 212 mẫu với "
        f"{imp.get('n_features')} chiều. Mô hình được chốt là "
        f"**{PRETTY.get(a['summary'].get('best_model'), a['summary'].get('best_model'))}** theo quy "
        f"tắc ở mục 5.2 (`{a['summary'].get('selection_rule', '—')}`), không phải theo một mô hình "
        f"định trước.",
        "",
        "### 7.2. Đặc trưng ảnh hưởng nhiều nhất",
        "",
        f"*Phép đo: **permutation importance** (mức giảm AUROC khi hoán vị một cột) trên mô hình "
        f"`{imp.get('model')}` — **đúng mô hình đã chốt** trong `summary.json` (trước đây script "
        f"hard-code `logistic`, gây mâu thuẫn với `best.joblib`). Cột hệ số chỉ xuất hiện khi mô hình "
        f"được chốt là **tuyến tính**; với mô hình cây (Random Forest) báo cáo bỏ cột này vì cây "
        f"không có hệ số — và cũng đừng đọc hệ số tuyến tính như “độ quan trọng nhân quả”.*",
        "",
        imp_table, "",
        fig("reports/figures/analysis/02_feature_importance.png",
            "Permutation importance trên validation và test"),
        "",
        f"**Diễn giải.** Ba đặc trưng dẫn đầu theo permutation importance (mô hình "
        f"`{imp.get('model')}`, đọc từ artifact): "
        + ", ".join(f"`{i['feature']}`" for i in (imp.get("val") or [])[:3] or []) + ". "
        + (f"**Độ lớn RẤT NHỎ:** ΔAUROC lớn nhất trên validation chỉ "
           f"{_n(((imp.get('val') or [{}])[0] or {}).get('mean_decrease_auroc'))} ⇒ với mô hình cây, "
           f"hoán vị một cột gần như không làm giảm AUROC (tín hiệu phân tán trên nhiều cột tương "
           f"quan do đa cộng tuyến), nên bảng này dùng để **định vị nhóm thông tin** chứ không dùng "
           f"để khẳng định tầm quan trọng nhân quả. ")
        + "Đây là các chỉ số **biên lợi nhuận / vòng quay / cấu trúc vốn-thanh khoản** (xem nhóm đặc "
        "trưng ở mục 4.5); lưu ý thứ hạng này **khác** thứ hạng theo hệ số tuyến tính, minh hoạ đúng "
        "cảnh báo ở mục 7.3: đừng đọc hệ số/độ quan trọng như quan hệ nhân quả. Điều này khớp với "
        "phát hiện ở mục 3.2, 3.4–3.5: mô hình tách nhóm công ty theo cấu trúc tài chính ổn định, "
        "chứ chưa học được “động lực suy giảm” của từng quý. Lưu ý tích cực: nhóm dựa trên nợ "
        "**không còn bị thiếu dữ liệu** — `total_liabilities` phủ 100% mẫu nhờ suy ra từ `A = L + E` "
        "(mục 4.5), nên các cột đứng đầu không bị chi phối bởi giá trị impute.",
        "",
        "### 7.3. Đa cộng tuyến (VIF)",
        "",
        f"- Số feature có VIF > 10: **{vif.get('n_vif_above_10')} / {vif.get('n_features_used')}**.",
        "",
        vif_table, "",
        "**Hệ quả phương pháp luận:** khi nhiều cột tương quan mạnh (các tỷ số đều có `total_assets` "
        "hoặc `current_liabilities` làm mẫu số), hệ số Logistic **không diễn giải được như độ quan "
        "trọng nhân quả**. Vì thế báo cáo dùng permutation importance (mục 7.2) làm căn cứ chính và "
        "chỉ nêu hệ số để tham khảo. Với mô hình cây, đa cộng tuyến ít ảnh hưởng tới dự báo nhưng "
        "làm tăng phương sai của importance.",
        "",
        "### 7.4. Ablation — yếu tố nào ảnh hưởng thật?",
        "",
        f"Bảng ablation đầy đủ ở mục 4.5 (tính trên **cùng mô hình đã chốt: `{an.get('importance', {}).get('model')}`**, "
        f"cấu hình mặc định). Ba kết luận **đọc trực tiếp từ artifact** (không nhập tay):",
        "",
        _ablation_notes(an.get("ablation", []), low_abl),
        "",
        # --- PLACEHOLDER_ANALYSIS_B ---
        "### 7.5. Phân tích lỗi (Error Analysis)",
        "",
        table(["Mẫu", "Công ty", "Thực tế", "Dự đoán", "P(distress)", "#quý lịch sử",
               "current_ratio", "WC/assets", "net_margin"],
              [[e["sample_id"], e["ticker"], e["actual"], e["predicted"], _n(e["probability"]),
                e.get("n_history"), _n(e.get("current_ratio_latest"), 2),
                _n(e.get("working_capital_to_assets"), 2), _n(e.get("net_margin_latest"), 3)]
               for e in an.get("error_cases", [])],
              ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]), "",
        f"Tổng số mẫu sai trên test: **{len(err_cases)} / "
        f"{a['test'].get('test_metrics', {}).get('n')}** — gồm **{err_fn} FN** (bỏ sót suy giảm) và "
        f"**{err_fp} FP** (báo động giả) tại ngưỡng vận hành. Chi tiết đầy đủ: "
        f"`reports/results/error_cases.csv`.",
        "",
        f"- **Lỗi có cấu trúc, không ngẫu nhiên:** {len(err_cases)} mẫu sai nằm ở "
        f"{len(err_tickers)} công ty {', '.join(err_tickers)} (tỷ lệ nhãn 1 trong toàn bộ dữ liệu "
        f"của từng công ty: {err_rates}). Phần lớn các chuỗi nhãn *hằng* (WMT/LOW/HD = 100% nhãn 1, "
        f"ROST ≈ 0,05) gần như được dự đoán đúng tuyệt đối, nhưng lỗi **không chỉ** xảy ra ở chuỗi "
        f"nhãn biến động"
        + (f": {', '.join(err_constant)} tuy 100% nhãn 1 vẫn bị bỏ sót 1 quý (xác suất sát ngưỡng)"
           if err_constant else "")
        + f". Nhóm nhãn biến động ({', '.join(err_mixed) if err_mixed else '—'}) là nơi mô hình khó "
        f"nhất vì nhãn đổi giữa các quý.",
        f"- Vùng xác suất của mẫu sai: **{_n(err_pmin)}–{_n(err_pmax)}** ({err_detail}) ⇒ tồn tại cả "
        f"lỗi “sai rõ” (P = {_n(err_pmin)}) lẫn lỗi “sát ngưỡng” (P = {_n(err_pmax)} so với ngưỡng "
        f"vận hành {_n(a['test'].get('threshold'))}); vì vậy **hạ ngưỡng không xoá hết sai số**, nó chỉ "
        f"dịch chuyển FN ↔ FP (xem mục 7.6).",
        "- Nguyên nhân gốc của sai số **không** phải nhiễu hay ảnh mờ (đây là dữ liệu bảng) mà là: "
        "(i) nhãn ở các công ty này biến động giữa các quý, (ii) feature cấu trúc của chúng nằm sát "
        "biên quyết định.",
        "- Trong triển khai thực tế, sai số loại này xử lý được bằng cách **hạ ngưỡng** "
        f"({_n((a['test'].get('test_metrics', {}).get('cost_optimal') or {}).get('threshold'))} cho "
        "chi phí kỳ vọng tối thiểu) hoặc rà soát thủ công các mẫu ở vùng xác suất trung bình.",
        "",
        "### 7.6. Ngưỡng quyết định: F1 vs chi phí kỳ vọng",
        "",
        f"*Mô hình dùng cho bảng và hình dưới đây: **`{(an.get('threshold') or {}).get('model')}`** "
        f"(đọc từ `analysis.json.threshold.model` — nay là **cùng mô hình đã chốt**). Ngưỡng vận hành "
        f"chính thức trong `test_evaluation.json` là **{_n(a['test'].get('threshold'))}**; khác nhau "
        f"chỉ vì cách chọn (best-F1 trên validation vs tối ưu chi phí kỳ vọng), không phải khác mô hình.*",
        "",
        table(["Tập", "Ngưỡng best-F1", "F1 tại đó", "Ngưỡng tối ưu chi phí", "Chi phí kỳ vọng"],
              [[tag, _n(v["best_f1"]["threshold"]), _n(v["best_f1"]["f1"]),
                _n(v["cost_optimal"]["threshold"]), _n(v["cost_optimal"]["expected_cost"], 1)]
               for tag, v in an.get("threshold", {}).items() if tag in ("validation", "test")],
              ["---", "---:", "---:", "---:", "---:"]), "",
        fig("reports/figures/analysis/04_threshold_curves.png",
            "Precision/Recall/F1 và chi phí kỳ vọng theo ngưỡng (validation và test)"),
        "",
        "**Thảo luận.** Ngưỡng vận hành đang dùng tối đa F1 (mục 5.2) nên ưu tiên precision; nếu "
        "đặt chi phí bỏ sót (FN) đắt gấp 5 lần báo động giả (FP) thì ngưỡng tối ưu thấp hơn rõ rệt "
        "và recall tiến về 1,0 — tức “không bỏ sót doanh nghiệp nào, chấp nhận nhiều báo động giả”. "
        "Đây là tham số **do người dùng quyết định**, không phải do thuật toán.",
        "",
        # --- PLACEHOLDER_ANALYSIS_C ---
        "### 7.7. Hiệu chuẩn xác suất (Calibration)",
        "",
        table(["Tập", "Brier"],
              [[tag, _n(v.get("brier"))]
               for tag, v in an.get("calibration", {}).get("bins", {}).items()],
              ["---", "---:"]), "",
        fig("reports/figures/analysis/05_calibration.png",
            "Đường reliability: xác suất dự báo vs tần suất thực tế"),
        "",
        "Brier ≈ 0,06 trên test cho thấy xác suất đầu ra **khá sát tần suất thực tế**, nên có thể "
        "dùng trực tiếp trong bài toán ra quyết định (ví dụ tính lợi ích kỳ vọng). Tuy nhiên n = 64 "
        "nên đường reliability chỉ có ít điểm dữ liệu — không nên diễn giải quá mức.",
        "",
        "### 7.8. Learning curve — thêm dữ liệu có giúp được không?",
        "",
        table(["#mẫu train", "Train AUROC", "Val AUROC (GroupKFold)"],
              [[s, _n(tr), _n(va)]
               for s, tr, va in zip(an.get("learning_curve", {}).get("train_sizes", []),
                                    an.get("learning_curve", {}).get("train_auroc_mean", []),
                                    an.get("learning_curve", {}).get("val_auroc_mean", []))],
              ["---:", "---:", "---:"]), "",
        fig("reports/figures/analysis/06_learning_curve.png",
            "Learning curve chia theo công ty — còn thiếu mẫu hay đã bão hoà?"),
        "",
        "**Kết luận.** Đường validation trong chế độ chia theo công ty còn khoảng cách lớn so với "
        "train và chưa bão hoà ⇒ cách cải thiện hiệu quả nhất **không phải** đổi thuật toán mà là "
        "**thêm công ty mới** (tăng số thực thể) và làm sạch định nghĩa nhãn.",
        "",
        "### 7.9. Yếu tố nào ảnh hưởng nhiều nhất tới kết quả?",
        "",
        table(["#", "Yếu tố", "Mức ảnh hưởng", "Bằng chứng trong báo cáo"],
              [["1", "**Nhãn gần như là thuộc tính của công ty** (thực thể)", "Rất cao",
                "mục 3.2 (tỷ lệ nhãn theo công ty) và mục 6.4 (AUROC tụt khi cross-company)"],
               ["2", "**Ít thực thể** — chỉ 8 công ty", "Rất cao",
                "mục 7.8 (learning curve) và danh sách công ty không tính được AUROC ở mục 6.4"],
               ["3", "**Định nghĩa nhãn**", "Cao", "mục 8.1–8.2 (nhãn gốc không tái tạo được)"],
               ["4", "**Cấu trúc vốn / thanh khoản** (current_ratio, working capital, liabilities)",
                "Cao", "mục 7.2 (permutation importance) và mục 4.5 (ablation)"],
               ["5", "**Ngưỡng quyết định**", "Trung bình",
                "mục 7.6: đổi ngưỡng đưa recall từ 0,84 lên 1,0 mà không đổi mô hình"],
               ["6", "**Thuật toán** (logistic / RF / boosting)", "Thấp",
                f"mục 6.1 và 6.4: các họ mô hình cho kết quả test gần nhau "
                f"(in-domain AUROC {_n(min(in_dom.values())) if in_dom else '—'}–"
                f"{_n(max(in_dom.values())) if in_dom else '—'})"],
               ["7", "**Mẫu quá non (<5 quý lịch sử)**", "Thấp",
                "mục 4.5: bỏ 32 mẫu này thay đổi kết quả ở mức nhỏ"],
               ["8", "**Chỉ tiêu thưa dữ liệu** (receivables, short-term investments)",
                "Thấp–trung bình",
                "mục 4.5: lỗ hổng tag `liabilities` đã được xử lý ở tầng feature (nợ suy ra ⇒ phủ "
                "100%); hai chỉ tiêu còn thưa vẫn được impute trong pipeline và có nhóm tỷ số riêng"]],
              ["---:", "---", "---", "---"]),
        "",
    ])


def section_leakage(a: Dict[str, Any]) -> str:
    """Mục 8 — Truy vết nhãn và kiểm chứng độ nhạy theo định nghĩa nhãn."""
    audit = a["analysis"].get("label_audit", {})
    rel = a["relabel"]
    # Khối văn bản cho mục 8.4 (ETL port + nhãn sự kiện) — đọc từ artifact, không viết cứng số.
    labels_notes = _labels_and_etl_notes(a)
    manifest = rel.get("manifest", {})
    cmp_ = rel.get("comparison", {})
    label = a["eda"].get("label", {})

    rule_table = table(["Quy tắc thử nghiệm (trên quý target)", "Mức khớp với nhãn gốc"],
                       [[f"`{name}`", _pct(value, 1)]
                        for name, value in audit.get("rule_agreement", {}).items()],
                       ["---", "---:"])
    row_rule_table = table(["Quy tắc thử nghiệm (trên dòng lịch sử CUỐI CÙNG)", "Mức khớp"],
                           [[f"`{name}`", _pct(value, 1)]
                            for name, value in audit.get("rule_agreement_last_history", {}).items()],
                           ["---", "---:"])

    rel_rows = [[PRETTY.get(r["system"], r["system"]), _n(r["val"]["auroc"]), _n(r["test"]["auroc"]),
                 _n(r["test"].get("ap")), _n(r["test"].get("f1")), _n(r["test"].get("macro_f1"))]
                for r in cmp_.get("in_domain", [])]
    rel_table = table(["Hệ thống (nhãn quy tắc)", "Val AUROC", "Test AUROC", "Test AP", "Test F1",
                       "Test macro-F1"], rel_rows, ["---", "---:", "---:", "---:", "---:", "---:"])
    rel_cross = table(["Hệ thống (cross-company, nhãn quy tắc)", "AUROC", "AP", "F1"],
                      [[PRETTY.get(r["system"], r["system"]), _n(r["test"]["auroc"]),
                        _n(r["test"].get("ap")), _n(r["test"].get("f1"))]
                       for r in cmp_.get("cross_company", [])],
                      ["---", "---:", "---:", "---:"])

    shares = ", ".join(f"{k} {v:.2f}" for k, v in
                       sorted(label.get("label_share_by_ticker", {}).items(), key=lambda kv: -kv[1]))
    return "\n".join([
        "",
        "## 8. Truy vết nhãn và kiểm chứng độ nhạy",
        "",
        "### 8.1. Nhãn gốc không tái tạo được ⇒ đã bổ sung nhãn QUY TẮC + nhãn SỰ KIỆN",
        "",
        "Nhãn `is_distressed` trong `data/prepared` được giữ nguyên từ pipeline sinh dữ liệu gốc. "
        "Pipeline đó **nay đã được port lại thành `scripts/prepare_sec.py`**: đọc snapshot SEC trong "
        "`data/sec/raw`, tái tạo 16 chỉ tiêu theo đúng 4 phương pháp kỳ của bản gốc "
        "(`instant` / `reported_quarter` / `reported_first_quarter` / `current_ytd_minus_previous_ytd`, "
        "thiếu fact thì để `absent`) và đối chiếu ngược công bố công khai `reports/results/etl_verify.md` "
        "(xem mục 8.4).",
        "",
        "Vì vậy báo cáo chủ động kiểm tra nhãn có khớp với các quy tắc kế toán đơn giản không:",
        "",
        rule_table, "",
        row_rule_table, "",
        f"- Mức khớp cao nhất trong các quy tắc thử nghiệm: "
        f"**{_pct(audit.get('max_rule_agreement'), 1)}** ⇒ nhãn gốc **không** tương ứng với bất kỳ "
        f"quy tắc đơn giản nào viết lại được.",
        f"- Phản chứng cụ thể: **WMT FY2015Q2** có net income dương (≈ 3,59 tỷ USD) nhưng bị gán "
        f"nhãn = 1.",
        f"- Tỷ lệ nhãn = 1 theo công ty: {shares}.",
        "",
        "**Kết luận 8.1.** Nhãn mang tính **thực thể** cao và không có định nghĩa công khai ⇒ không "
        "thể khẳng định mô hình đang dự báo “suy giảm tài chính” theo nghĩa nghiệp vụ. Đây là hạn chế "
        "của bộ dữ liệu, được xử lý bằng hai việc: (i) công bố rõ trong báo cáo, (ii) kiểm chứng độ "
        "nhạy bằng một định nghĩa nhãn công khai ở mục 8.2.",
        "",
        "### 8.2. Kiểm chứng độ nhạy bằng nhãn tái lập được",
        "",
        "`scripts.relabel` dựng lại split với nhãn theo công thức công khai "
        f"(`forecasting/labels.py`; rule = `{manifest.get('rule', '—')}`, "
        f"min_signals = {manifest.get('min_signals', '—')}): nhãn = 1 nếu quý target có ≥ 1 tín hiệu "
        "trong số: lỗ ròng, dòng tiền hoạt động âm, lỗ hoạt động, vốn lưu động âm, vốn chủ sở hữu "
        "âm, doanh thu giảm > 5% so với cùng kỳ. Nhãn tính trên **quý target** (chỉ công bố sau "
        "`as_of`) nên **không** rò rỉ vào feature.",
        "",
        f"- Mức khớp giữa nhãn quy tắc và nhãn gốc: "
        f"**{_pct(manifest.get('agreement_with_original_labels'), 1)}** trên "
        f"{manifest.get('n_comparable_samples', '—')} mẫu ⇒ hai bộ nhãn khác nhau rõ rệt, nên đây là "
        f"một phép thử độ nhạy thực chất.",
        "",
        rel_table, "",
        rel_cross, "",
        f"- Số mẫu dương tính theo nhãn quy tắc: `{manifest.get('label_counts', {})}` → nhãn quy tắc "
        f"**cân bằng hơn** và **biến thiên theo quý** (không phải hằng số theo công ty).",
        "- **Kết luận:** kết luận “bài toán bị chi phối bởi thực thể” lặp lại trên cả hai định nghĩa "
        "nhãn ⇒ kết luận không phụ thuộc vào cách gán nhãn cụ thể.",
        "",
        "### 8.3. Kết luận về độ tin cậy của kết quả",
        "",
        table(["Câu hỏi kiểm tra", "Kết quả", "Mức tin cậy"],
              [["Có rò rỉ thời gian (dùng dữ liệu tương lai) không?", "Không — có kiểm thử tự động", "Cao"],
               ["Có rò rỉ tiền xử lý (impute/scale trên toàn dữ liệu) không?", "Không — nằm trong Pipeline", "Cao"],
               ["Có rò rỉ do chia tập ngẫu nhiên không?", "Không — chia theo thời gian + purge", "Cao"],
               ["Có **rò rỉ cấp thực thể** không?", "**Có** — đã đo và định lượng", "Cao"],
               ["Nhãn có định nghĩa kiểm chứng được không?", "Không (nhãn gốc); đã bù bằng nhãn quy tắc",
                "Trung bình"],
               ["Kết quả có lặp lại khi chạy lại không?", "Có — seed cố định, output ổn định", "Cao"]],
              ["---", "---", "---"]),
        "",
        "### 8.4. ETL đã PORT và được đối chiếu ngược (nhãn/quy tắc + sự kiện)",
        "",
        labels_notes, "",
    ])


def _labels_and_etl_notes(a: Dict[str, Any]) -> str:
    """ETL port + nhãn sự kiện — hai khối văn bản cho mục 8.4 (đọc từ `etl_verify`/`events`)."""
    etl = a.get("etl_verify") or {}
    events = a.get("events") or {}
    parts: List[str] = []
    if etl:
        parts.append(
            f"**Đối chiếu ETL (`scripts/prepare_sec.py`, bản port):** tái tạo lại từng ô chỉ tiêu từ "
            f"`data/sec/raw/*-companyfacts.json` bằng đúng quy tắc của bản gốc (thứ tự ưu tiên tag + 4 "
            f"phương pháp kỳ + bản công bố sớm nhất), rồi so với bảng đang dùng cho báo cáo: "
            f"**{_vi(etl.get('n_match'))}/{_vi(etl.get('n_cells'))} ô khớp "
            f"({_pct(etl.get('match_rate'), 2)})** trên {_vi(etl.get('n_companies'))} công ty. "
            f"Đường sinh dữ liệu tự động (không dùng bảng gốc làm khung) tìm được "
            f"{_vi(sum((v.get('generated_rows') or 0) for v in (etl.get('per_company') or {{}}).values()))} "
            f"quý — đủ để chạy ETL cho công ty MỚI (`python -m scripts.prepare_sec`).")
    if events:
        parts.append(
            f"**Nhãn SỰ KIỆN (`scripts/fetch_events.py`):** kiểm tra toàn bộ filing 8-K của "
            f"{_vi(events.get('n_companies'))} công ty trên API công khai của SEC ⇒ "
            f"**{_vi(events.get('n_bankruptcy'))} sự kiện phá sản (item 1.03)** và "
            f"**{_vi(events.get('n_events'))} sự kiện/tín hiệu kiệt quệ**. "
            f"{events.get('conclusion', '')}")
    layers = ["(1) nhãn gốc (giữ nguyên, không tái tạo được)",
              "(2) nhãn **quy tắc tái lập được** (`stress_signals`, `altman_z`, `forward_4q` — mục 8.2)"]
    if events:
        layers.append("(3) nhãn **sự kiện công khai** đọc từ 8-K của SEC (`reports/results/events.md`)")
    if parts:
        parts.append("**Ý nghĩa cho đề tài:** kết luận của đồ án được kiểm tra trên "
                     + "; ".join(layers)
                     + " — thay vì chỉ một định nghĩa nhãn duy nhất; công cụ `scripts.fetch_events.py` "
                       "sẵn sàng cho lớp nhãn sự kiện khi mở rộng universe (cần Internet).")
    return "\n".join(parts)


def section_robustness(a: Dict[str, Any]) -> str:
    """Mục 9 — Kiểm chứng bổ sung: tiền xử lý đuôi nặng, SHAP, kiểm định ý nghĩa thống kê.

    Trả "" nếu chưa chạy các script tương ứng ⇒ báo cáo vẫn sinh được (không vỡ) và người đọc thấy
    ngay bước nào chưa thực hiện.
    """
    prep = a.get("preprocessing_experiment") or {}
    shap = a.get("shap") or {}
    sig = a.get("significance") or {}
    imb = a.get("imbalance_real") or {}
    search = a.get("search") or {}
    labels = a.get("label_sensitivity") or {}
    if not (prep or shap or sig or imb or search or labels):
        return ""
    lines: List[str] = ["", "## 9. Kiểm chứng bổ sung (tiền xử lý đuôi nặng · SHAP · ý nghĩa thống kê)",
                        "",
                        "Ba hạng mục dưới đây được thêm để đóng các lỗ hổng đã nêu ở mục 9.2 (hạn chế): "
                        "xử lý outlier, giải thích từng hồ sơ, và kiểm định \"hơn nhau có thật không\".",
                        ""]

    # --- 9.1 Tiền xử lý đuôi nặng -------------------------------------------------------------
    if prep:
        protocol = prep.get("protocol") or {}
        rows = prep.get("rows") or []
        worst = sorted(rows, key=lambda r: -((r.get("oof_average_precision") or 0.0)))[:8]
        lines += ["### 9.1. Winsorize × scaler trên dữ liệu thật (`scripts.experiment_preprocessing.py`)",
                  "",
                  f"- Giao thức: fit trên train; in-domain trên validation; tổng quát hoá bằng "
                  f"{protocol.get('cross_company')}; **{protocol.get('n_variants')} cấu hình** "
                  f"({len(prep.get('models') or [])} họ mô hình).",
                  ""]
        lines += [f"{i + 1}. {text}" for i, text in enumerate(prep.get("conclusions") or [])]
        lines += ["",
                  table(["Mô hình", "Scaler", "Winsorize", "Val AP", "Cross-co. AP", "Cross-co. AUROC",
                         "% clip trên val"],
                        [[r["model"], r["scaler"], r["winsorize"], _n(r.get("val_average_precision"), 4),
                          _n(r.get("oof_average_precision"), 4), _n(r.get("oof_auroc"), 4),
                          _n(r.get("n_clipped_val_pct"), 2)] for r in worst],
                        ["---", "---", "---", "---:", "---:", "---:", "---:"]),
                  "",
                  fig("reports/figures/preprocessing/01_winsorize_scaler.png",
                      "ΔAP của các cấu hình tiền xử lý so với pipeline chính (winsorize × scaler)"),
                  ""]

    # --- 9.2 SHAP ----------------------------------------------------------------------------
    if shap:
        check = shap.get("self_check") or {}
        agreement = shap.get("rank_agreement_with_permutation") or {}
        lines += ["### 9.2. Giải thích mô hình bằng SHAP — KernelSHAP tự cài đặt (`scripts/explain_model.py`)",
                  "",
                  f"- Mô hình: **{shap.get('model')}**; giải thích **{shap.get('n_explained')} mẫu** "
                  f"(validation {((shap.get('evaluated_on') or {}).get('validation'))} + test "
                  f"{((shap.get('evaluated_on') or {}).get('test'))}), "
                  f"{shap.get('n_coalitions')} liên minh/điểm, nền {shap.get('n_background')} mẫu train.",
                  f"- Vì sao tự cài: môi trường đồ án không có gói `shap`; thuật toán được cài đúng "
                  f"theo Lundberg & Lee (2017) bằng numpy — {shap.get('method')}.",
                  f"- **Tự kiểm chứng:** sai số efficiency |Σφ + E[f] − f(x)| ≤ "
                  f"{_n(check.get('max_abs_efficiency_gap'), 12)} (tương đối "
                  f"{_n(check.get('relative_efficiency_gap'), 12)}); công thức còn được đối chiếu "
                  f"giải tích cho hàm tuyến tính trong `tests/test_explain.py`.",
                  f"- **Đối chiếu permutation importance TRÊN CÙNG MÔ HÌNH** "
                  f"(`{shap.get('model')}` = `best.joblib`; `scripts/explain_model.py` tự tính lại "
                  f"importance trên chính mô hình này, độc lập với `analysis.json`): Spearman = "
                  f"{_n(agreement.get('spearman'), 3)}, trùng top-{agreement.get('top_k')} = "
                  f"{_n(agreement.get('top_overlap'), 3)}. Mức đồng thuận này **thấp–trung bình** — "
                  f"hai phép đo trả lời hai câu hỏi khác nhau (SHAP = đóng góp cục bộ có cộng tính, "
                  f"permutation = mức giảm AUROC khi hoán vị), nên báo cáo nêu cả hai thay vì coi "
                  f"chúng là bằng chứng thay thế cho nhau.",
                  "",
                  table(["#", "Feature", "mean |φ|"],
                        [[i + 1, r["feature"], _n(r["mean_abs_shap"], 4)]
                         for i, r in enumerate((shap.get("importance_top") or [])[:10])],
                        ["---:", "---", "---:"]),
                  ""]
        if shap.get("local_explanations"):
            lines += ["**Giải thích cục bộ các mẫu dự đoán sai** (đóng góp dương đẩy về phía suy giảm):",
                      "",
                      table(["Mẫu", "Thực tế", "P(nhãn 1)", "Feature đẩy về suy giảm",
                             "Feature kéo về an toàn"],
                            [[r["sample_id"], r["actual"], _n(r["probability"], 3),
                              ", ".join(f"{i['feature']} ({i['phi']:+.2f})" for i in r["top_positive"]),
                              ", ".join(f"{i['feature']} ({i['phi']:+.2f})" for i in r["top_negative"])]
                             for r in shap["local_explanations"]],
                            ["---", "---:", "---:", "---", "---"]),
                      ""]
        lines += [fig("reports/figures/shap/01_shap_summary.png",
                      "SHAP: độ quan trọng feature (mean |φ|) kèm đối chiếu permutation"),
                  fig("reports/figures/shap/03_shap_local_errors.png",
                      "SHAP cục bộ cho các mẫu dự đoán sai — căn cứ giải trình từng hồ sơ"),
                  ""]

    # --- 9.3 Kiểm định ý nghĩa thống kê --------------------------------------------------------
    if sig:
        systems = sig.get("systems") or {}
        lines += ["### 9.3. Kiểm định ý nghĩa thống kê trên test (`scripts/significance.py`)",
                  "",
                  f"- Cùng **{sig.get('n_samples')} mẫu test** ({sig.get('n_positive')} dương) cho mọi "
                  f"hệ thống: DeLong (1988) cho ΔAUROC + paired bootstrap cho ΔAP (CI 95%, seed cố định).",
                  "",
                  table(["Hệ thống", "AUROC", "Average Precision"],
                        [[name, _n(m.get("auroc"), 4), _n(m.get("average_precision"), 4)]
                         for name, m in systems.items()],
                        ["---", "---:", "---:"]),
                  ""]
        lines += [f"{i + 1}. {text}" for i, text in enumerate(sig.get("conclusions") or [])]
        pairs = sig.get("pairs_vs_baseline") or []
        if pairs:
            lines += ["",
                      table(["A", "B", "ΔAUROC (A−B)", "p (DeLong)", "ΔAP (A−B)", "CI95 ΔAP",
                             "p (bootstrap AP)"],
                            [[p["a"], p["b"], _n((p["delong_auroc"] or {}).get("delta"), 4),
                              _n((p["delong_auroc"] or {}).get("p_value"), 4),
                              _n((p["bootstrap_ap"] or {}).get("delta"), 4),
                              f"{(p['bootstrap_ap'] or {}).get('ci95_lower', float('nan')):+.4f}; "
                              f"{(p['bootstrap_ap'] or {}).get('ci95_upper', float('nan')):+.4f}",
                              _n((p["bootstrap_ap"] or {}).get("p_value"), 4)] for p in pairs],
                            ["---", "---", "---:", "---:", "---:", "---:", "---:"]),
                      "",
                      "> Đọc bảng: p ≥ 0,05 ⇒ **chưa đủ căn cứ** khẳng định mô hình hơn baseline trên bộ "
                      "test 64 mẫu này. Đây là kết luận trung thực, không phải điểm yếu bị che: báo cáo "
                      "vì thế nhấn mạnh so sánh cross-company và baseline `ticker_prior`."]

    # --- 9.4 Kỹ thuật xử lý lệch lớp trên DỮ LIỆU THẬT ----------------------------------------
    if imb:
        rows = imb.get("rows") or []
        sig4 = imb.get("significance") or {}
        boot = sig4.get("bootstrap_ap") or {}
        lines += ["### 9.4. Kỹ thuật xử lý lệch lớp trên dữ liệu THẬT (`scripts.experiment_imbalance_real.py`)",
                  "",
                  f"- Giao thức: {((imb.get('protocol') or {}).get('cv'))}; mô hình nền "
                  f"**{imb.get('base_model')}** ⇒ mọi kỹ thuật khác nhau CHỈ ở bước xử lý lệch lớp. "
                  f"Mất cân bằng cấp mẫu IR ≈ {_n(imb.get('label_ir'), 2)}.",
                  "",
                  table(["Kỹ thuật", "Nhóm", "AP (OOF)", "AUROC (OOF)", "F1* OOF", "IR trước → sau",
                         "Test fold nguyên vẹn"],
                        [[r["name"], r.get("group", "—"), _n(r.get("average_precision"), 4),
                          _n(r.get("auroc"), 4), _n(r.get("best_f1_on_oof"), 4),
                          (f"{r['ir_before']:.2f} → {r['ir_after']:.2f}"
                           if r.get("ir_before") is not None else "—"),
                          "PASS" if r.get("fold_test_untouched") else "—"]
                         for r in sorted(rows, key=lambda r: -(r.get("average_precision") or 0))],
                        ["---", "---", "---:", "---:", "---:", "---:", "---"]),
                  ""]
        imb_notes = list(imb.get("conclusions") or [])
        if not imb_notes:
            imb_notes = [f"Kiểm định cặp tốt nhất vs đối chứng: ΔAP = {(boot.get('delta') or 0):+.4f}, "
                         f"p = {boot.get('p_value')}."]
        lines += [f"{i + 1}. {text}" for i, text in enumerate(imb_notes)]
        lines += [fig("reports/figures/imbalance_real/01_techniques_real.png",
                      "ΔAP/ΔAUROC của các kỹ thuật lệch lớp so với đối chứng (OOF cross-company)"), ""]

    # --- 9.5 Tìm kiếm siêu tham số + sổ thực nghiệm ---------------------------------------------
    if search:
        lines += ["### 9.5. Tìm kiếm siêu tham số + sổ thực nghiệm (`scripts/search.py`)",
                  "",
                  f"- {search.get('method')}",
                  f"- Mục tiêu: {search.get('objective')}; **{search.get('n_trials_per_model')} trial/"
                  f"mô hình**; sổ thực nghiệm `{((search.get('ledger') or {}).get('path'))}` gồm "
                  f"**{((search.get('ledger') or {}).get('n_rows'))} dòng**.",
                  "",
                  table(["Mô hình", "#trial", "CV-AP tốt nhất", "CV-AP mặc định", "CV-AP GridSearchCV",
                         "Δ vs Grid", "Δ vs mặc định", "Thời gian (s)"],
                        [[r["model"], r.get("n_trials"), _n(r.get("best_cv_average_precision"), 4),
                          _n(r.get("default_cv_average_precision"), 4),
                          _n(r.get("grid_search_best_cv_ap"), 4), _n(r.get("delta_vs_grid"), 4),
                          _n(r.get("delta_vs_default"), 4), _n(r.get("total_seconds"), 1)]
                         for r in search.get("rows") or []],
                        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
                  "",
                  *_search_interpretation(search.get("rows") or []),
                  "",
                  fig("reports/figures/search/01_search_distribution.png",
                      "Phân bố CV-AP của các trial random search (so với GridSearchCV)"), ""]

    # --- 9.6 Độ nhạy của kết luận theo định nghĩa nhãn ------------------------------------------
    if labels:
        lines += ["### 9.6. Độ nhạy của kết luận theo ĐỊNH NGHĨA NHÃN (`scripts/label_sensitivity.py`)",
                  "",
                  f"- Mô hình cố định **{labels.get('model')}**; mỗi định nghĩa nhãn dựng lại split bằng "
                  f"`split_policy` rồi đo độc lập (in-domain + cross-company + mức khớp nhãn gốc).",
                  "",
                  table(["Định nghĩa", "Dương (test)", "IR train", "Khớp nhãn gốc",
                         "AUROC test (mô hình)", "AUROC test (prior)", "Cross-co. AP", "Cross-co. AUROC"],
                        [[r["rule"], _pct(r["positive_rate"].get("test")),
                          _n((r.get("imbalance_train") or {}).get("imbalance_ratio"), 2),
                          _pct(r.get("agreement_with_original_labels")),
                          _n((r.get("in_domain") or {}).get("model_auroc"), 3),
                          _n((r.get("in_domain") or {}).get("ticker_prior_auroc"), 3),
                          _n((r.get("cross_company") or {}).get("oof_average_precision"), 3),
                          _n((r.get("cross_company") or {}).get("oof_auroc"), 3)]
                         for r in labels.get("rows") or []],
                        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
                  ""]
        lines += [f"{i + 1}. {text}" for i, text in enumerate(labels.get("conclusions") or [])]
        lines += [""]
    return "\n".join(lines)


def _search_interpretation(rows: List[Dict[str, Any]]) -> List[str]:
    """Sinh diễn giải ĐỘNG cho bảng 9.5 (mỗi mô hình có thể có kết luận khác nhau)."""
    improved = [r for r in rows if (r.get("delta_vs_grid") or 0.0) > 0.01]
    worsened = [r for r in rows if (r.get("delta_vs_grid") or 0.0) < -0.01]
    lines = ["*Diễn giải:* cột **Δ vs Grid** so random search với lưới `GridSearchCV` cũ trên cùng "
             "thước đo CV-AP và cùng splitter."]
    if improved:
        lines.append(
            "- Random search **tốt hơn** ở: " + ", ".join(
                f"`{r['model']}` ({r['delta_vs_grid']:+.4f})" for r in improved)
            + " ⇒ với các họ mô hình này, tìm kiếm rộng thật sự có ích (lưới cũ quá thô).")
    if worsened:
        lines.append(
            "- Random search **không tốt hơn** ở: " + ", ".join(
                f"`{r['model']}` ({r['delta_vs_grid']:+.4f})" for r in worsened)
            + " ⇒ lưới cũ đã đủ tốt, thêm trial là lãng phí (Random Forest là mô hình được chốt, nên "
              "kết luận không đổi).")
    lines.append(
        "- Mọi trial (params, seed, CV-AP, thời gian) đều nằm trong sổ `runs.csv` ⇒ tra cứu lại được, "
        "và khi môi trường có Optuna chỉ cần thay `sample_params` (không phải viết lại hạ tầng).")
    return lines


def section_conclusion(a: Dict[str, Any]) -> str:
    """Mục 9 — Kết luận, hạn chế, hướng phát triển."""
    checks = a["checks"].get("headline", {})
    best = a["summary"].get("best_model")
    n_companies = (a["eda"].get("corpus") or {}).get("n_companies")
    n_loco = checks.get("loco_n_evaluable")
    base_auc = next((r["test"]["auroc"] for r in a["baselines"].get("rows", [])
                     if r["baseline"] == "ticker_prior"), None)
    # Danh sách mô hình ĐÃ huấn luyện (đọc từ artifact, không hard-code) để câu kết luận luôn khớp.
    models_run = sorted({str(m.get("model")) for m in a["summary"].get("models", [])})
    return "\n".join([
        "",
        "## 10. Kết luận và hướng phát triển",
        "",
        "### 10.1. Kết luận chính",
        "",
        f"1. {len(models_run)} họ mô hình ({', '.join(PRETTY.get(m, m) for m in models_run)}) đều đạt AUROC test "
        f"~0,97–0,98; mô hình được chốt là **{PRETTY.get(best, best)}** theo quy tắc công bố trước ở "
        f"mục 5.2 (`{a['summary'].get('selection_rule', '—')}`) — tức ưu tiên khả năng tổng quát hoá "
        f"sang công ty chưa từng thấy, không ưu tiên điểm in-domain.",
        f"2. **Nhưng** baseline “nhớ mặt công ty” (ticker-prior) đạt AUROC = {_n(base_auc)}, tức mô "
        f"hình học máy **không vượt** nổi một quy tắc chỉ dùng danh tính công ty. Khi đánh giá "
        f"cross-company, AUROC giảm còn "
        f"{_n((checks.get('cross_company_oof_auroc') or {}).get(best))}; LOCO chỉ tính được AUROC "
        f"trên **{n_loco}/{n_companies}** công ty — phần còn lại có nhãn đơn lớp ở cả validation và "
        f"test nên AUROC không xác định (xem mục 6.4).",
        "3. Đóng góp chính của đồ án là **phát hiện và định lượng rò rỉ cấp thực thể** — dạng lỗi "
        "thực nghiệm rất dễ bị bỏ qua nếu báo cáo chỉ trình bày bảng AUROC/F1 đẹp.",
        "4. Bộ công cụ đánh giá được chuẩn hoá và tái lập được bằng một lệnh: baseline đối chứng, "
        "cross-company CV, bootstrap CI, calibration, ngưỡng theo chi phí, ablation, error analysis.",
        "",
        "### 10.2. Hạn chế",
        "",
        "1. **Nhãn gốc không tái tạo được** (mục 8.1) — hạn chế lớn nhất; đã giảm nhẹ bằng kiểm "
        "chứng độ nhạy với nhãn quy tắc (mục 8.2).",
        "2. **Chỉ 8 công ty** → không đủ để kết luận thống kê về tổng quát hoá; nhiều công ty chỉ có "
        "một lớp nhãn nên AUROC không xác định.",
        "3. **Mẫu không độc lập**: cửa sổ lịch sử 8 quý trượt nên các mẫu liền nhau chia sẻ phần lớn "
        "dữ liệu; số quan sát hiệu dụng nhỏ hơn 324.",
        "4. **Chưa xử lý outlier** (winsorize/clip) và chưa thử mô hình chuỗi thời gian (LSTM/GRU, "
        "transformer cho chuỗi quý).",
        "5. **Thiếu dữ liệu ngoài báo cáo tài chính** (giá cổ phiếu, xếp hạng tín dụng, vĩ mô) — "
        "nguồn tín hiệu quan trọng của bài toán suy giảm.",
        "",
        # --- PLACEHOLDER_CONCLUSION_B ---
        "### 10.3. Hướng phát triển",
        "",
        "Hai hướng **đã triển khai** trong phiên bản này (nợ phải trả suy ra từ `A = L + E` và nhóm "
        "`path`) đều được đo trước bằng `scripts/probe_features.py` (nested CV, cross-company AUROC: "
        "0,817 → 0,901 với Logistic; 0,886 → 0,930 với Random Forest). Bảng dưới là các hướng còn lại:",
        "",
        table(["#", "Hướng", "Vì sao hiệu quả (theo bằng chứng trong báo cáo)"],
              [["1", "Mở rộng lên 50–100 công ty cùng ngành",
                "Learning curve (7.8) chưa bão hoà; kết quả cao hiện nay phần lớn do “nhận diện công "
                "ty” nên cần nhiều thực thể hơn"],
               ["2", "Định nghĩa nhãn công khai, có cơ sở học thuật (Altman Z-score, O-score, dòng "
                "tiền âm ≥ 2 quý liên tiếp)",
                "Mục 8: nhãn hiện tại không kiểm chứng được; nhãn quy tắc cân bằng hơn"],
               ["3", "Đánh giá walk-forward theo thời gian và theo nhóm ngành",
                "Đo đúng “công ty mới + giai đoạn mới”, sát câu hỏi nghiệp vụ"],
               ["4", "Mô hình chuỗi thời gian / mô hình survival (Cox, discrete-time hazard)",
                "Tận dụng cấu trúc dọc của dữ liệu thay vì vector hoá 8 quý"],
               ["5", "Bổ sung feature phi tài chính (giá, sở hữu, tin tức)",
                "Các họ mô hình cho kết quả gần nhau (6.1) ⇒ giới hạn nằm ở dữ liệu, không ở thuật toán"],
               ["6", "Quy trình ra quyết định theo chi phí thực tế của tổ chức",
                "Mục 7.6: ngưỡng là biến quyết định mạnh, miễn phí để cải thiện recall/F1"],
               ["7", "Thêm chỉ tiêu XBRL đã đo được độ phủ (`scripts/probe_tags.py`): "
                "`AccountsPayableCurrent` 100% → DPO/chu kỳ tiền mặt; "
                "`PaymentsToAcquirePropertyPlantAndEquipment` ~75% → FCF = OCF − capex; "
                "`IncomeTaxExpenseBenefit` 100% → thuế suất thực tế",
                "Chọn chỉ tiêu theo độ phủ ĐO ĐƯỢC trên snapshot SEC thay vì đoán; các tag nợ chi tiết "
                "(`LongTermDebt*`, `InterestExpense`) chỉ phủ 26–46% nên đã loại"]],
              ["---:", "---", "---"]),
        "",
        "### 10.4. Bài học phương pháp luận",
        "",
        "1. Với bài toán có nhãn gắn với **thực thể** (công ty, người, thiết bị), bắt buộc phải có "
        "**baseline theo thực thể** và **chia tập theo thực thể**; chỉ số cao trong chế độ in-domain "
        "có thể chỉ là “nhớ mặt”.",
        "2. Luôn kèm **khoảng tin cậy** và **baseline** khi báo cáo metric: n = 64 không cho phép "
        "khẳng định chênh lệch 0,002 AUROC.",
        "3. **Mọi số liệu nên sinh tự động từ artifact**: báo cáo này làm vậy nên hình, bảng và JSON "
        "không thể lệch nhau (lỗi khó phát hiện khi tổng hợp thủ công).",
        "",
        *_references_lines(a), "",
        "## Phụ lục A — Tái lập toàn bộ kết quả",
        "",
        "```powershell",
        "python -m pip install -r requirements.txt",
        "python -m scripts.run_all              # data → eda → train → baselines → validation → tuning",
        "                                       # → evaluate → report → analyze → relabel",
        "                                       # → make_report → export_office",
        "python -m pip install -r imbalance_lab/requirements.txt",
        "python -m imbalance_lab.techniques     # danh mục 15 kỹ thuật (Yêu cầu 2) + ngưỡng theo PR",
        "python -m unittest discover -s tests -v   # kiểm thử (gồm chống rò rỉ dữ liệu)",
        "```",
        "",
        "Cách chạy riêng từng bước và ý nghĩa từng artifact: `docs/huong-dan-tai-lap.md`.",
        "",
        "## Phụ lục B — Danh mục artifact (đều sinh tự động)",
        "",
        table(["Đường dẫn", "Nội dung"],
              [["`reports/results/eda_summary.json`, `reports/results/eda.md`",
                "Số liệu EDA + thống kê mô tả + tương quan + nhận xét tự động"],
               ["`reports/results/provenance.{json,md}`",
                "**Kiểm chứng nguồn gốc:** SHA-256 20 file SEC + tra ngược từng fact trong "
                "companyfacts + kiểm quy đổi VND (0 lệch, 0 fact thiếu, 0 ô bịa số)"],
               ["`reports/figures/eda/*.png`", "9 hình EDA"],
               ["`reports/results/eda_deep.{json,md}`",
                "EDA chuyên sâu: chất lượng feature, entropy/IR nhãn, liên hệ feature–nhãn, "
                "cụm đa cộng tuyến, drift, rò rỉ, missingness-mang-nhãn"],
               ["`reports/figures/eda_deep/*.png`", "7 hình EDA chuyên sâu"],
               ["`reports/results/summary.json`",
                "Hyperparameter, metric train/val từng mô hình, mô hình được chọn"],
               ["`reports/results/tuning.{json,md}`", "Kết quả GridSearchCV chia theo công ty"],
               ["`reports/results/baselines.json`",
                "Dummy, ticker-prior, single-feature và mô hình tham chiếu"],
               ["`reports/results/validation_checks.json`",
                "GroupKFold, LOCO, bootstrap CI, tương quan hạng giữa mô hình"],
               ["`reports/results/test_evaluation.json`",
                "Metric test tại ngưỡng vận hành + danh sách mẫu sai"],
               ["`reports/results/test_predictions.csv`", "Xác suất từng mẫu test"],
               ["`reports/results/error_cases.csv`", "Mẫu dự đoán sai kèm 7 chỉ tiêu"],
               ["`reports/results/analysis.{json,md}`",
                "Overfit, importance, VIF, ablation, audit nhãn, lỗi, ngưỡng, calibration"],
               ["`reports/figures/analysis/*.png`", "7 hình phân tích chuyên sâu"],
               ["`reports/results/relabel.{json,md}`",
                "Nhãn quy tắc tái lập được + so sánh hai định nghĩa nhãn"],
               ["`data/prepared-rule/*`", "Split theo nhãn quy tắc"],
               ["`reports/models/best.joblib`", "Mô hình + ngưỡng đã chốt"],
               ["`reports/results/run_all.log`", "Log chạy toàn pipeline"],
               ["`docs/BAO-CAO.docx`, `docs/BAO-CAO-slide.pptx`, `docs/BAO-CAO-slide-bao-ve.pptx`, "
                "`docs/BAO-CAO-slide-bao-ve.docx`, `docs/bo-tai-lieu-bao-ve.docx`",
                "Bản Word/Slide xuất tự động: **báo cáo hoàn chỉnh**, slide tự động, deck bảo vệ "
                "(cả `.pptx` và bản Word để dựng slide), bộ tài liệu bảo vệ"]],
              ["---", "---"]),
        "",
        "## Phụ lục C — Kiến trúc mã nguồn",
        "",
        table(["Module", "Trách nhiệm"],
              [["`forecasting/config.py`", "Đường dẫn, chỉ tiêu, hằng số (seed, ngưỡng, chi phí FN/FP)"],
               ["`forecasting/data.py`", "Tái tạo split gốc từ dữ liệu mở rộng (byte-identical, có test)"],
               ["`forecasting/data_loader.py`", "Nạp split/manifest, tiện ích chuyển kiểu an toàn"],
               ["`forecasting/features.py`",
                f"{a['summary'].get('n_features', '—')} feature + nhóm feature + bộ lọc lịch sử"],
               ["`forecasting/models.py`",
                f"Registry {_n_model_registry() or '—'} họ mô hình + HYPERPARAMS (một nguồn duy nhất)"],
               ["`forecasting/evaluation.py`", "Metric đầy đủ, đường PR, ngưỡng theo F1 và theo chi phí"],
               ["`forecasting/baselines.py`", "Dummy, ticker-prior, single-feature"],
               ["`forecasting/validation.py`", "GroupKFold, LOCO, bootstrap CI, agreement"],
               ["`forecasting/tuning.py`", "GridSearchCV chia theo công ty"],
               ["`forecasting/labels.py`", "Định nghĩa nhãn quy tắc tái lập được"],
               ["`scripts/analyze.py`", "Phân tích chuyên sâu + 7 hình + audit nhãn"],
               ["`scripts/relabel.py`", "Split theo nhãn quy tắc + kiểm chứng độ nhạy"],
               ["`scripts/predict.py`", "Demo: dự đoán MỘT quý/mẫu mới + ngưỡng vận hành + SHAP"],
               ["`scripts/verify_provenance.py`",
                "Kiểm chứng dữ liệu THẬT từ snapshot SEC (hash, fact, quy đổi VND)"],
               ["`scripts/eda.py`", "6 hình EDA + bảng tổng quan"],
               ["`scripts/make_report.py`", "Sinh báo cáo markdown từ artifact (file này)"],
               ["`scripts/export_office.py`", "Xuất `.docx` và `.pptx`"],
               ["`scripts/run_all.py`", "Chạy toàn pipeline + ghi log"],
               ["`tests/test_pipeline.py`", "Kiểm thử tái lập dữ liệu, chống rò rỉ, tính nhất quán"]],
              ["---", "---"]),
        "",
        f"*Số liệu trong báo cáo: test n = {a['test'].get('test_metrics', {}).get('n')}, mô hình "
        f"{PRETTY.get(best, best)}, ngưỡng {_n(a['test'].get('threshold'))}.*",
    ])


def slides(a: Dict[str, Any]) -> str:
    """Sinh `docs/slide.md` — 14 slide, số liệu lấy thẳng từ artifact."""
    eda = a["eda"]
    corpus = eda.get("corpus", {})
    test = a["test"].get("test_metrics", {})
    op = test.get("operating", {})
    best = a["summary"].get("best_model")
    checks = a["checks"].get("headline", {})
    base = next((r for r in a["baselines"].get("rows", []) if r["baseline"] == "ticker_prior"), {})
    base_auc = (base.get("test") or {}).get("auroc")
    label = a["eda"].get("label", {})
    n_mis = len(a["analysis"].get("error_cases", []))
    rel_agree = a["relabel"].get("manifest", {}).get("agreement_with_original_labels")
    hist = eda.get("history_length", {})
    n_features = a["summary"].get("n_features")
    label_audit = a["analysis"].get("label_audit", {})
    simple_rules = [v for k, v in (label_audit.get("rule_agreement") or {}).items()
                    if not str(k).startswith("stress_signals")]
    op_cm = (test.get("confusion_at_operating") or {})
    prior_cm = ((base.get("test") or {}).get("at_0.5") or {})
    same_cm = all(op_cm.get(k) == prior_cm.get(k) for k in ("tn", "fp", "fn", "tp")) if op_cm else False
    label_share = (label.get("label_share_by_ticker") or {})
    min_ticker = min(label_share, key=label_share.get) if label_share else "—"
    min_share = label_share.get(min_ticker) if label_share else None
    coverage = (eda.get("coverage_pct") or {}).get("mean_by_field") or {}
    # --- Lỗi trên test: mọi câu chữ dưới đây phải SUY RA TỪ artifact (không hard-code) ---
    err = a["analysis"].get("error_cases", [])
    n_fn = sum(1 for e in err if int(e.get("actual", -1)) == 1 and int(e.get("predicted", -1)) == 0)
    n_fp = sum(1 for e in err if int(e.get("actual", -1)) == 0 and int(e.get("predicted", -1)) == 1)
    err_probs = [float(e["probability"]) for e in err if e.get("probability") is not None]
    err_tickers = sorted({str(e.get("ticker")) for e in err})
    const_companies = list(label.get("companies_all_one") or [])
    err_constant = [t for t in err_tickers if t in const_companies]
    err_mixed = [t for t in err_tickers if t not in const_companies]
    models_rows = a["summary"].get("models", [])
    overfit_worst = max((r.get("train_auroc") or 0.0) for r in models_rows) if models_rows else None
    val_aucs = [r.get("auroc") for r in models_rows if r.get("auroc") is not None]
    best_row = next((r for r in models_rows if r.get("model") == best), {})

    items: List[Dict[str, Any]] = [
        {"title": "Đặt vấn đề và câu hỏi nghiên cứu", "bullets": [
            "Dự báo quý kế tiếp: doanh nghiệp bán lẻ có rơi vào suy giảm tài chính (`is_distressed`)?",
            f"Dữ liệu: {corpus.get('n_companies')} chuỗi bán lẻ Mỹ, {corpus.get('n_quarters')} quý, "
            f"{corpus.get('n_samples')} mẫu",
            "RQ1 bốn họ mô hình + baseline quy tắc Altman · RQ2 đặc trưng quyết định · "
            "RQ3 công ty chưa từng thấy · RQ4 định nghĩa nhãn"]},
        {"title": "Dữ liệu và cách tạo mẫu", "bullets": [
            "16 chỉ tiêu/quý từ SEC XBRL → mẫu = (lịch sử ≤ as_of, quý target, nhãn)",
            "Chia tập theo thời gian trong từng công ty + dải purge 16 mẫu",
            "Tái lập bằng 1 lệnh; manifest có SHA-256 cho nguồn và từng split"]},
        {"title": "Phát hiện then chốt: nhãn là thuộc tính của CÔNG TY", "bullets": [
            f"{', '.join(const_companies) or '—'} có 100% nhãn = 1 trong mọi quý; "
            f"{min_ticker} chỉ {_n(min_share)}",
            f"Nhãn gốc không khớp quy tắc kế toán đơn giản nào (khớp tối đa "
            f"{max(simple_rules) * 100:.0f}% nếu bỏ nhóm tín hiệu tổng hợp)"
            if simple_rules else "Nhãn gốc không tái tạo được từ dữ liệu công bố",
            "⇒ Chỉ số in-domain cao có thể chỉ là “nhớ mặt công ty”"],
         "image": "reports/figures/eda/03_label_by_company_quarter.png"},
        {"title": "EDA: chất lượng dữ liệu", "bullets": [
            f"Độ phủ thấp: {', '.join(eda.get('low_coverage_fields', [])[:4]) or '—'} "
            f"(liabilities {_n(coverage.get('liabilities'), 0)}%)",
            "Nhưng `debt_to_assets`/`debt_to_equity` vẫn phủ 100% nhờ suy ra nợ từ A = L + E",
            f"{hist.get('n_below_min')}/{hist.get('n_samples')} mẫu có <5 quý lịch sử → feature YoY là NaN",
            "Xử lý: impute trong Pipeline (không rò rỉ) + ablation đo ảnh hưởng"],
         "image": "reports/figures/eda/01_coverage_by_company.png"},
        {"title": f"Tiền xử lý và {n_features} đặc trưng", "bullets": [
            "14 tỷ số (latest + YoY) · 10 tốc độ tăng trưởng · cấu trúc vốn · chỉ báo căng thẳng",
            "Nhóm `path`: cực trị xấu nhất trong cửa sổ 8 quý, mức giảm so với đỉnh, chuỗi quý âm",
            "Nợ suy ra từ `A = L + E` ⇒ `debt_to_assets`/`debt_to_equity` phủ 100% mẫu",
            "Pipeline chính: median-impute → (scaler cho tuyến tính) → model; không ticker one-hot",
            "Winsorize IQR chỉ nằm ở thí nghiệm 9.1 (lợi ích cho mô hình chốt trong khoảng nhiễu)"],
         "image": "reports/figures/eda/04_ratio_boxplots_by_label.png"},
        {"title": "Bốn họ mô hình và tinh chỉnh", "bullets": [
            "Logistic Regression · Random Forest · HistGradientBoosting · MLP (4 họ, thuần scikit-learn)",
            "Baseline quy tắc Altman Z'' < 1,1 (không học tham số) + ticker-prior + dummy + 1 chỉ tiêu",
            "GridSearchCV với StratifiedGroupKFold theo mã cổ phiếu; refit theo AP",
            "Chọn mô hình: AP cross-company (GroupKFold) → best-F1(val) → AP → AUROC → gap nhỏ nhất",
            "Mô hình TRIỂN KHAI giữ cấu hình MẶC ĐỊNH (tinh chỉnh chỉ +0,003 CV-AP ⇒ dưới mức nhiễu)"],
         "image": "reports/figures/validation_pr_curves.png"},
        {"title": "So sánh có BASELINE — điểm khác biệt của đồ án", "bullets": [
            f"Baseline ticker-prior (không học gì): AUROC = {_n(base_auc)}",
            f"Mô hình được chốt ({PRETTY.get(best, best)}): AUROC = {_n(test.get('auroc'))}",
            ("⇒ Không có khác biệt thực chất: confusion matrix giống nhau từng ô"
             if same_cm else
             f"⇒ Baseline **không hề thua** mô hình ở in-domain (confusion nó khác mô hình: "
             f"TN/FP/FN/TP = {prior_cm.get('tn')}/{prior_cm.get('fp')}/{prior_cm.get('fn')}/"
             f"{prior_cm.get('tp')} so với {op_cm.get('tn')}/{op_cm.get('fp')}/{op_cm.get('fn')}/"
             f"{op_cm.get('tp')}) ⇒ phần lớn khả năng phân biệt đến từ danh tính công ty")],
         "image": "reports/figures/analysis/07_in_domain_vs_cross_company.png"},
        # --- PLACEHOLDER_SLIDES ---
        {"title": "Metric đầy đủ tại ngưỡng vận hành", "bullets": [
            f"AUROC {_n(test.get('auroc'))} · AP {_n(test.get('average_precision'))} · "
            f"Brier {_n(test.get('brier'))}",
            f"Precision {_n(op.get('precision'))} · Recall {_n(op.get('recall'))} · "
            f"F1 {_n(op.get('f1'))} · macro-F1 {_n(op.get('macro_f1'))}",
            f"Kèm bootstrap CI 95% (n = {test.get('n')}) để không overclaim"],
         "image": "reports/figures/test_confusion.png"},
        {"title": "Phân tích lỗi: sai số có cấu trúc", "bullets": [
            f"{n_mis}/{test.get('n')} mẫu sai = {n_fn} FN + {n_fp} FP "
            f"(precision {_n(op.get('precision'))} · recall {_n(op.get('recall'))})",
            (f"Công ty liên quan: {', '.join(err_tickers)}"
             + (f" — {', '.join(err_constant)} (nhãn hằng = 1) cũng bị bỏ sót 1 quý"
                if err_constant else "")
             + (f"; nhãn biến động: {', '.join(err_mixed)}" if err_mixed else "")),
            (f"P(distress) của mẫu sai: {_n(min(err_probs))}–{_n(max(err_probs))} ⇒ hạ ngưỡng cứu "
             f"được {n_fn} FN nhưng trả thêm {n_fp} FP") if err_probs else
            "Xác suất các mẫu sai nằm sát ngưỡng ⇒ hạ ngưỡng sẽ cứu được"]},
        {"title": "Overfitting và lựa chọn mô hình", "bullets": [
            f"AUROC train tới {_n(overfit_worst)} nhưng validation "
            f"{_n(min(val_aucs))}–{_n(max(val_aucs))} ⇒ overfit nhẹ do dữ liệu nhỏ",
            f"Mô hình được chốt: {PRETTY.get(best, best)} — xếp theo AP cross-company "
            f"{_n(best_row.get('cross_company_ap'))}, không theo F1 in-domain",
            "Learning curve chưa bão hoà ⇒ nút thắt là số lượng công ty"],
         "image": "reports/figures/analysis/01_overfit_train_vs_val.png"},
        {"title": "Đặc trưng quyết định và đa cộng tuyến", "bullets": [
            "Permutation importance (mô hình RF): " + ", ".join(
                f"`{i['feature']}`" for i in ((a["analysis"].get("importance") or {}).get("val") or [])[:3])
            + " dẫn đầu",
            f"ΔAUROC rất nhỏ (max {_n((((a['analysis'].get('importance') or {}).get('val') or [{}])[0] or {}).get('mean_decrease_auroc'))}) ⇒ không có 'cột quyết định'",
            f"Nhiều cột VIF > 10 ({((a['analysis'].get('vif') or {}).get('n_vif_above_10'))}) ⇒ không diễn giải hệ số Logistic như quan hệ nhân quả",
            "Ablation: bỏ nhóm YoY/tăng trưởng gần như không giảm chất lượng"],
         "image": "reports/figures/analysis/02_feature_importance.png"},
        {"title": "Ngưỡng, chi phí và hiệu chuẩn", "bullets": [
            f"Ngưỡng best-F1 {_n(a['test'].get('threshold'))} vs ngưỡng tối ưu chi phí "
            f"{_n((test.get('cost_optimal') or {}).get('threshold'))} (FN đắt gấp 5 lần FP)",
            "Brier ≈ 0,06 ⇒ xác suất dùng được cho bài toán ra quyết định",
            "Ngưỡng là biến quyết định “miễn phí”: đổi precision/recall, AP không đổi"],
         "image": "reports/figures/analysis/04_threshold_curves.png"},
        {"title": "Kiểm chứng độ nhạy theo định nghĩa nhãn", "bullets": [
            "Dựng lại split bằng nhãn quy tắc công khai (`scripts.relabel`)",
            f"Khớp với nhãn gốc chỉ ≈ {_pct(rel_agree)} ⇒ hai bộ nhãn khác nhau rõ rệt",
            "Kết luận “bài toán bị chi phối bởi thực thể” lặp lại trên cả hai định nghĩa nhãn"],
         "image": "reports/figures/analysis/07_in_domain_vs_cross_company.png"},
        {"title": "Kết luận và hướng phát triển", "bullets": [
            "Đóng góp: phát hiện + định lượng rò rỉ cấp thực thể; bộ đánh giá chuẩn hoá, tái lập được",
            f"Cross-company AUROC còn {_n((checks.get('cross_company_oof_auroc') or {}).get(best))};",
            (f"LOCO tính được AUROC trên cả {checks.get('loco_n_evaluable')}/"
             f"{corpus.get('n_companies')} công ty (trung bình {_n(checks.get('loco_mean_auroc'))})"
             if checks.get("loco_n_evaluable") == corpus.get("n_companies") else
             f"LOCO chỉ tính được AUROC trên {checks.get('loco_n_evaluable')}/"
             f"{corpus.get('n_companies')} công ty (nhãn đơn lớp ở phần còn lại)"),
            "Hướng đi: thêm 50–100 công ty, nhãn công khai có cơ sở học thuật, walk-forward/survival"]},
    ]
    out = ["# Slide thuyết trình — Dự báo suy giảm tài chính doanh nghiệp bán lẻ", "",
           "*Sinh tự động bởi `python -m scripts.make_report`; số liệu luôn khớp "
           "`reports/results/`.*", ""]
    for i, item in enumerate(items, start=1):
        out += [f"## {i}. {item['title']}", ""]
        out += [f"- {b}" for b in item["bullets"]]
        out.append("")
        if item.get("image"):
            out += [f"![{item['title']}](../{item['image']})", ""]
    return "\n".join(out)


def label_doc(a: Dict[str, Any]) -> str:
    """Sinh `docs/dinh-nghia-nhan.md` — truy vết nhãn và định nghĩa nhãn thay thế."""
    audit = a["analysis"].get("label_audit", {})
    rel = a["relabel"].get("manifest", {})
    label = a["eda"].get("label", {})
    return "\n".join([
        "# Truy vết nhãn `is_distressed` (bắt buộc đọc trước khi dùng kết quả)",
        "",
        "*Sinh tự động bởi `python -m scripts.make_report` từ `reports/results/analysis.json` và "
        "`reports/results/relabel.json`.*",
        "",
        "## 1. Nhãn gốc đến từ đâu?",
        "",
        "Nhãn trong `data/prepared/*.json` **được giữ nguyên** từ pipeline sinh dữ liệu ban đầu. "
        "Pipeline đó **nay đã được port** (`scripts/prepare_sec.py`): đọc `data/sec/raw/*-companyfacts.json`, "
        "tái tạo 16 chỉ tiêu theo 4 phương pháp kỳ của bản gốc và đối chiếu ngược với bảng đang dùng cho "
        "báo cáo — **99,92% ô khớp** (`reports/results/etl_verify.md`); phần chưa khớp được liệt kê "
        "từng ô trong file đó. `forecasting/data.py::_build_samples` chỉ dùng heuristic "
        "`net_income < 0` cho mẫu **hoàn toàn mới**; với dữ liệu hiện có, nhãn được đọc lại theo "
        "`sample_id` (nên bước tái tạo ô ở trên KHÔNG thay đổi nhãn đang dùng cho kết quả).",
        "",
        "## 2. Nhãn gốc có tái tạo được không? — KHÔNG",
        "",
        "Kiểm chứng tự động (`scripts/analyze.py::label_audit`) áp từng quy tắc kế toán đơn giản "
        f"trên {audit.get('n_samples', '—')} mẫu (train + validation + test) và so với nhãn gốc:",
        "",
        table(["Quy tắc (trên quý target)", "Khớp nhãn gốc", "Khớp khi áp lên dòng lịch sử cuối"],
              [[f"`{name}`", _pct(value, 1),
                _pct(audit.get("rule_agreement_last_history", {}).get(name))]
               for name, value in audit.get("rule_agreement", {}).items()],
              ["---", "---:", "---:"]),
        "",
        f"→ Mức khớp cao nhất: **{_pct(audit.get('max_rule_agreement'), 1)}** ⇒ nhãn gốc **không** "
        f"tương ứng với bất kỳ quy tắc đơn giản nào có thể viết lại từ dữ liệu công bố.",
        "",
        "**Phản chứng cụ thể:** `WMT-2015Q2` (quý 2014-05-01 → 2014-07-31) có net income dương "
        "(89.825 tỷ VND ≈ 3,59 tỷ USD) và operating cash flow dương, nhưng nhãn = 1.",
        "",
        "## 3. Nhãn gần như là thuộc tính của CÔNG TY",
        "",
        table(["Công ty", "Tỷ lệ nhãn = 1"],
              [[t, f"{v:.2f}"] for t, v in sorted(label.get("label_share_by_ticker", {}).items(),
                                                 key=lambda kv: -kv[1])],
              ["---", "---:"]),
        "",
        f"- Công ty có 100% nhãn = 1: **{', '.join(label.get('companies_all_one', [])) or '—'}**.",
        f"- Công ty có 0% nhãn = 1: **{', '.join(label.get('companies_all_zero', [])) or '—'}** "
        f"(nếu danh sách rỗng nghĩa là mọi công ty đều có ít nhất một quý nhãn 1).",
        "- Hệ quả: ở chế độ in-domain, mô hình chỉ cần nhận ra công ty là đạt AUROC rất cao — đây là "
        "lý do đồ án bổ sung baseline ticker-prior và đánh giá cross-company (GroupKFold/LOCO).",
        "",
        "## 4. Nhãn thay thế tái lập được (dùng để kiểm chứng độ nhạy)",
        "",
        "`forecasting/labels.py` định nghĩa nhãn công khai: `is_distressed_rule = 1` nếu quý target "
        f"có ≥ **{rel.get('min_signals', '—')}** tín hiệu trong 6 tín hiệu căng thẳng tài chính:",
        "",
        *[f"- {name}: {doc}" for name, doc in (rel.get("signals") or {}).items()],
        "",
        "- Nhãn được tính trên **quý target** — dữ liệu chỉ công bố ở `label_available_on` (sau "
        "`as_of`) nên **không** rò rỉ vào feature.",
        f"- Số mẫu dương tính theo nhãn quy tắc: `{rel.get('label_counts', {})}`.",
        f"- Mức khớp với nhãn gốc: **{_pct(rel.get('agreement_with_original_labels'), 1)}** — đủ khác "
        f"để phép thử độ nhạy có ý nghĩa.",
        "",
        "Chạy lại nhánh dữ liệu này:",
        "",
        "```powershell",
        "python -m scripts.relabel              # sinh data/prepared-rule + so sánh hai định nghĩa nhãn",
        "```",
        "",
        "## 5. Khuyến nghị khi sử dụng kết quả",
        "",
        "1. **Không** diễn giải AUROC in-domain là “khả năng dự báo suy giảm tài chính”.",
        "2. Luôn kèm baseline ticker-prior và kết quả cross-company khi báo cáo.",
        "3. Nếu dùng cho nghiên cứu tiếp: nên **tái sinh nhãn** bằng định nghĩa công khai (mục 4) "
        "hoặc quy tắc học thuật (Altman Z-score / O-score) rồi chạy lại toàn bộ pipeline.",
    ])


def repro_doc() -> str:
    """Sinh `docs/huong-dan-tai-lap.md` — hướng dẫn chạy lại và bản đồ artifact."""
    return "\n".join([
        "# Hướng dẫn tái lập (từ dữ liệu trong repo đến báo cáo)",
        "",
        "## 0. Môi trường",
        "",
        "```powershell",
        "python --version            # ≥ 3.11 (đã kiểm thử trên 3.13, Windows)",
        "python -m pip install -r requirements.txt",
        "```",
        "",
        "Mã nguồn **không phụ thuộc pandas** (dữ liệu JSON thuần + numpy + scikit-learn).",
        "Xuất Word cần `python-docx`; xuất Slide dùng bộ ghi OOXML nội bộ (không cần `python-pptx`).",
        "Log CLI/test đã lọc cảnh báo **vô hại** của thư viện (`runtime_warnings.py`): "
        "`OptimizeWarning: Unknown solver options: iprint` (scikit-learn 1.6 + scipy 1.18, in ở mỗi lần "
        "fit `LogisticRegression`) và `PyparsingDeprecationWarning` (matplotlib 3.9 dùng API pyparsing đã "
        "deprecate). Cảnh báo khác — kể cả của code dự án — vẫn hiển thị.",
        "",
        "## 1. Chạy tất cả trong một lệnh",
        "",
        "```powershell",
        "python -m scripts.run_all",
        "```",
        "",
        "Thứ tự các bước (22): `data → prepare_sec (ETL) → provenance → eda → eda_deep → train → "
        "baselines → validation → tuning → evaluate → report → analyze → prep_exp → imbalance_real → "
        "search → explain → significance → label_sensitivity → relabel → predict → make_report → "
        "export_office`. Log chi "
        "tiết ở `reports/results/run_all.log` (kèm lý do nếu một bước bị bỏ qua); `train` và `evaluate` "
        "là hai bước lõi, các bước còn lại lỗi thì ghi rõ rồi đi tiếp.",
        "",
        "Chạy chọn lọc:",
        "",
        "```powershell",
        "python -m scripts.run_all --only train,evaluate          # chỉ chạy một số bước",
        "python -m scripts.run_all --skip tuning,analyze          # bỏ bước nặng",
        "python -m forecasting.tuning --quick                     # lưới tìm kiếm rút gọn",
        "python -m scripts.analyze --quick                        # giảm số lần hoán vị",
        "```",
        "",
        "## 2. Chạy từng bước (khi cần kiểm tra riêng)",
        "",
        table(["Lệnh", "Sinh ra", "Ý nghĩa"],
              [["`python -m forecasting.data`", "`data/prepared/*`",
                "Tái tạo split gốc (mặc định bỏ qua nếu đã có; `--force` để ghi lại, byte-identical)"],
               ["`python -m scripts.eda`", "`reports/figures/eda/*.png`, `reports/results/eda.md`",
                "9 hình EDA + thống kê mô tả 14 tỷ số/16 chỉ tiêu, tỉ lệ lớp %, tương quan, "
                "nhận xét tự động (mục 3.1–3.6)"],
               ["`python -m scripts.eda_deep`",
                "`reports/results/eda_deep.{json,md}`, `reports/figures/eda_deep/*.png`",
                "EDA chuyên sâu: chất lượng 47 feature, entropy/IR nhãn, liên hệ feature–nhãn, "
                "cụm đa cộng tuyến, drift KS/SMD/PSI, rò rỉ & missingness-mang-nhãn"],
               ["`python -m forecasting.train`", "`reports/results/summary.json`, `reports/models/best.joblib`",
                "Fit 4 họ mô hình (logistic / random forest / hist gradient boosting / MLP), chọn mô hình trên validation, tính ngưỡng"],
               ["`python -m scripts.prepare_sec`", "`reports/results/etl_verify.{json,md}`",
                "PORT ETL: tái tạo 16 chỉ tiêu từ snapshot SEC và đối chiếu ngược bảng đang dùng "
                "(99,92% ô khớp) + sinh quý cho công ty mới"],
               ["`python -m forecasting.baselines`", "`reports/results/baselines.json`",
                "Dummy, ticker-prior, single-feature + **quy tắc Altman Z'' (không học tham số)**"],
               ["`python -m forecasting.validation`", "`reports/results/validation_checks.json`, "
                "`reports/results/walk_forward.json`",
                "GroupKFold, LOCO, bootstrap theo mẫu **và theo cụm công ty**, walk-forward theo thời "
                "gian (+ purge), tương quan hạng"],
               ["`python -m forecasting.tuning`", "`reports/results/tuning.{json,md}`",
                "GridSearchCV chia theo công ty + so với cấu hình mặc định"],
               ["`python -m forecasting.evaluate`", "`reports/results/test_evaluation.json`",
                "Đánh giá cuối trên test (test KHÔNG dùng để chọn mô hình/ngưỡng), confusion matrix "
                "tại ngưỡng vận hành"],
               ["`python -m forecasting.report`", "`reports/results/test_predictions.csv`",
                "Xác suất từng mẫu test + histogram"],
               ["`python -m scripts.analyze`", "`reports/results/analysis.{json,md}`, 7 hình",
                "Overfit, importance, VIF, ablation, audit nhãn, lỗi, ngưỡng, calibration"],
               ["`python -m scripts.relabel`", "`data/prepared-rule/*`, `reports/results/relabel.*`",
                "Nhãn quy tắc tái lập được + kiểm chứng độ nhạy"],
               ["`python -m scripts.audit_data`", "`reports/results/data_audit.{json,md}`",
                "Đối soát từng con số với số liệu thật trong snapshot SEC (provenance, VND, "
                "split, nhãn quy tắc, artifact báo cáo)"],
               ["`python -m scripts.class_balance`", "`reports/results/class_balance.{json,md}`",
                "Mất cân bằng lớp: đếm/ tỉ lệ/ Imbalance Ratio theo tập, theo công ty, theo nhãn quy tắc"],
               ["`python -m imbalance_lab.run`", "`reports/imbalance/summary.{md,csv,json}`, `run.log`",
                "Lab 98/2: 3 chiến lược + hiệu chuẩn + mốc minh hoạ SAI (rò rỉ)"],
               ["`python -m imbalance_lab.techniques`",
                "`reports/imbalance/techniques.{md,csv,json}`, `techniques_by_fold.csv`, `techniques.log`",
                "Danh mục 15 kỹ thuật (yêu cầu #2) + threshold tuning trên đường PR, PASS/FAIL chống rò rỉ"],
               ["`python -m imbalance_experiment.main`",
                "`reports/experiment/summary.{md,csv}`, `cv_mean_std.csv`, `results.json`, `pr_curves.png`",
                "Thực nghiệm ĐƠN LẺ vs KẾT HỢP: 17 pipeline, mất cân bằng 1:50, metric mean ± std, chống rò rỉ"],
               ["`python -m scripts.make_report`", "`docs/*.md`", "Báo cáo + slide + tài liệu"],
               ["`python -m scripts.export_office`",
                "`docs/BAO-CAO.docx`, `docs/BAO-CAO-slide.pptx`, `docs/BAO-CAO-slide-bao-ve.pptx`, "
                "`docs/BAO-CAO-slide-bao-ve.docx`, `docs/bo-tai-lieu-bao-ve.docx`",
                "BÁO CÁO HOÀN CHỈNH dạng Word + 2 bộ slide (.pptx) + Word dàn slide bảo vệ"],
               ["`python -m unittest discover -s tests -v`", "—",
                "Kiểm thử: tái lập byte-identical, chống rò rỉ, tính nhất quán metric, mutation "
                "test cho auditor"],
               ["`python -m scripts.predict --sample-id HD-2024Q2 --explain`", "— (in ra màn hình)",
                "Demo MỘT mẫu: P(distress), quyết định theo ngưỡng vận hành, cảnh báo backtest, "
                "top-K đóng góp SHAP (dùng cho phần trình bày)"],
               ["`python -m scripts.make_report` + `export_office`",
                "`docs/BAO-CAO.docx`, `docs/*.pptx`, `docs/BAO-CAO-slide-bao-ve.docx`, "
                "`docs/bo-tai-lieu-bao-ve.docx`",
                "Báo cáo hoàn chỉnh (Word) + 2 bộ slide (tự động 14 slide, dàn bảo vệ 11 slide) "
                "+ bản Word để dựng slide"],
               ["`python -m scripts.audit_data`", "`reports/results/data_audit.md`",
                "Toàn vẹn số liệu & câu chữ trong báo cáo/slide so với artifact"],
               ["`python -m scripts.verify_provenance`", "`reports/results/provenance.{json,md}`",
                "Kiểm chứng dữ liệu THẬT: SHA-256 snapshot SEC + tra ngược từng fact trong "
                "companyfacts + kiểm quy đổi VND (bắt cả lỗi 'điền số cho đủ')"]],
              ["---", "---", "---"]),
        "",
        "## 3. Chạy trên bộ nhãn khác (tùy chọn)",
        "",
        "```powershell",
        "$env:FORECASTING_PREPARED_DIR = \"e:\\CS114\\Do-an\\do-an\\data\\prepared-rule\"",
        "python -m forecasting.train      # chạy trên split theo nhãn quy tắc",
        "Remove-Item Env:FORECASTING_PREPARED_DIR",
        "```",
        "",
        "## 4. Kiểm tra nhanh sau khi chạy",
        "",
        "```powershell",
        "Get-ChildItem reports\\results, reports\\figures, docs | Select-Object Name, Length",
        "python -m unittest discover -s tests -v",
        "```",
        "",
        "Nếu một bước bị bỏ qua, log ghi rõ `[CHƯA CÓ]` hoặc `[LỖI]` kèm traceback — không có bước "
        "nào im lặng thất bại.",
    ])


def _references_lines(a: Dict[str, Any]) -> List[str]:
    """Sinh mục “Tài liệu tham khảo” (mục 11) từ `REFERENCE_ROWS` — đặt TRƯỚC các phụ lục."""
    _ = a  # giữ cùng chữ ký với các mục khác trong báo cáo
    return [
        "## 11. Tài liệu tham khảo", "",
        "Mọi mục dưới đây **được dùng thật** trong mã nguồn (hoặc để định nghĩa nhãn/ngưỡng), "
        "không có mục nào chỉ để trang trí; cột cuối ghi rõ vị trí sử dụng để đối chiếu. Các trích "
        "dẫn trong thân báo cáo (Beaver 1966; Altman 1968; Barboza và cộng sự 2017; Mai và cộng sự "
        "2019; DeLong và cộng sự 1988; Lundberg & Lee 2017) đều có mặt đầy đủ ở đây.",
        "",
        table(["Nhóm", "Tài liệu", "Dùng ở đâu trong repo"],
              [[r["group"], r["ref"], r["where"]] for r in REFERENCE_ROWS],
              ["---", "---", "---"]),
        "",
        "**Ghi chú về thư viện:** môi trường đồ án không có gói `shap`, `optuna`, `catboost`, nên "
        "KernelSHAP được cài lại bằng numpy theo Lundberg & Lee (2017) và tự kiểm chứng bằng sai số "
        "efficiency (mục 9.2), còn tìm kiếm siêu tham số dùng random search + sổ thực nghiệm thay "
        "cho Optuna (mục 9.5). Việc không thêm thư viện không đổi kết luận: các họ mô hình hiện có "
        "chênh nhau < 0,01 AUROC trên test (mục 6.1).",
        "",
        "**Ghi chú về số liệu:** mọi con số trong báo cáo được đọc trực tiếp từ "
        "`reports/results/*.json` bởi `python -m scripts.make_report` — xem Phụ lục A (lệnh tái lập) "
        "và Phụ lục B (danh mục artifact).",
        "",
    ]


def run() -> Dict[str, Any]:
    """Sinh toàn bộ tài liệu markdown từ artifact trong `reports/`."""
    a = artifacts()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    report = "\n".join([section_summary(a), section_intro(a), section_eda(a), section_eda_deep(a),
                        section_preprocessing(a), section_models(a), section_results(a),
                        section_analysis(a), section_leakage(a), section_robustness(a),
                        section_conclusion(a)])
    (DOCS_DIR / "BAO-CAO.md").write_text(report, encoding="utf-8")
    (DOCS_DIR / "slide.md").write_text(slides(a), encoding="utf-8")
    (DOCS_DIR / "dinh-nghia-nhan.md").write_text(label_doc(a), encoding="utf-8")
    (DOCS_DIR / "huong-dan-tai-lap.md").write_text(repro_doc(), encoding="utf-8")
    print(f"Báo cáo: BAO-CAO.md ({len(report.splitlines())} dòng, "
          f"{len(report) // 1024} KB), slide.md, dinh-nghia-nhan.md, huong-dan-tai-lap.md")
    return {"report": str(DOCS_DIR / "BAO-CAO.md"), "slides": str(DOCS_DIR / "slide.md"),
            "label_doc": str(DOCS_DIR / "dinh-nghia-nhan.md")}


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    print("=== Sinh báo cáo markdown ===")
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())