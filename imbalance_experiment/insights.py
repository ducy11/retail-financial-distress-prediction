"""Phân tích chuyên sâu (yêu cầu #5): mọi kết luận đều TRÍCH SỐ LIỆU đo được, không nhập tay.

Ba câu hỏi bắt buộc:
1. **SMOTE đơn lẻ vs SMOTETomek/SMOTEENN** — làm sạch biên có giúp gì không?
2. **Resampling lớn vs Cost/Weight** — chênh lệch thời gian/chi phí tính toán thế nào?
3. **Khi nào kết hợp (hybrid) vượt trội rõ rệt so với đơn lẻ?**

Hàm `analyse()` trả dict findings (số liệu + kết luận); `render_markdown()` dựng phần báo cáo. Mọi so
sánh dùng metric CHÍNH (PR-AUC, F1, Recall, FPR) — KHÔNG dùng accuracy.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .config import ExperimentConfig
from .evaluation import TechniqueResult

#: Kỹ thuật đơn lẻ dùng để so với bản hybrid tương ứng.
HYBRID_PAIRS: Dict[str, Sequence[str]] = {
    "smote_tomek": ("smote",),
    "smote_enn": ("smote",),
    "smote_class_weight": ("smote", "class_weight"),
    "rusboost": ("balanced_rf", "easy_ensemble", "balanced_bagging"),
}


def _get(results: Sequence[TechniqueResult], key: str) -> Optional[TechniqueResult]:
    """Lấy kết quả theo khoá (None nếu không chạy hoặc lỗi)."""
    for result in results:
        if result.key == key and result.ok:
            return result
    return None


def _metric(result: Optional[TechniqueResult], metric: str, *, source: str = "holdout") -> float:
    """Giá trị metric của một kết quả (`source="holdout"` hoặc `"cv"` = mean qua fold)."""
    if result is None:
        return float("nan")
    table = result.holdout if source == "holdout" else result.mean
    return float(table.get(metric, float("nan")))


def _fmt(value: float, digits: int = 3) -> str:
    """Định dạng số (NaN → '—')."""
    return "—" if value != value else f"{value:.{digits}f}"


def _delta(after: float, before: float, digits: int = 4) -> str:
    """Chuỗi Δ có dấu (NaN → '—')."""
    if after != after or before != before:
        return "—"
    return f"{after - before:+.{digits}f}"


def _group_stats(items: Sequence[TechniqueResult]) -> Dict[str, float]:
    """Thống kê một nhóm: n, PR-AUC trung bình, F1 trung bình, giây/fold trung bình."""
    if not items:
        return {"n": 0.0, "mean_pr_auc": float("nan"), "mean_f1": float("nan"),
                "mean_seconds": float("nan")}
    return {"n": float(len(items)),
            "mean_pr_auc": float(np.mean([_metric(r, "pr_auc") for r in items])),
            "mean_f1": float(np.mean([_metric(r, "f1") for r in items])),
            "mean_seconds": float(np.mean([r.fit_seconds_mean for r in items]))}


def _cleaning_findings(results: Sequence[TechniqueResult]) -> Dict[str, Any]:
    """Số liệu cho câu hỏi 1: SMOTE đơn lẻ vs SMOTE + dọn biên (Tomek/ENN)."""
    smote = _get(results, "smote")
    if smote is None:
        return {}
    smote_stats = {"pr_auc": _metric(smote, "pr_auc"), "f1": _metric(smote, "f1"),
                   "recall": _metric(smote, "recall"), "fpr": _metric(smote, "fpr"),
                   "cv_pr_auc_mean": _metric(smote, "pr_auc", source="cv"),
                   "cv_pr_auc_std": float(smote.std.get("pr_auc", float("nan"))),
                   "n_train_after": smote.n_train_after_mean,
                   "fit_seconds": smote.fit_seconds_mean,
                   "resample_seconds": smote.resample_seconds_mean}
    cleaning: Dict[str, Any] = {}
    for key in ("smote_tomek", "smote_enn"):
        candidate = _get(results, key)
        if candidate is None:
            continue
        cleaning[key] = {
            "pr_auc": _metric(candidate, "pr_auc"),
            "pr_auc_delta": _metric(candidate, "pr_auc") - smote_stats["pr_auc"],
            "f1": _metric(candidate, "f1"),
            "f1_delta": _metric(candidate, "f1") - smote_stats["f1"],
            "recall": _metric(candidate, "recall"), "fpr": _metric(candidate, "fpr"),
            "fpr_delta": _metric(candidate, "fpr") - smote_stats["fpr"],
            "cv_pr_auc_mean": _metric(candidate, "pr_auc", source="cv"),
            "cv_pr_auc_std": float(candidate.std.get("pr_auc", float("nan"))),
            "n_train_after": candidate.n_train_after_mean,
            "fit_seconds": candidate.fit_seconds_mean,
            "resample_seconds": candidate.resample_seconds_mean,
        }
    return {"smote": smote_stats, "cleaning": cleaning}


def _compute_findings(results: Sequence[TechniqueResult]) -> Dict[str, Any]:
    """Số liệu cho câu hỏi 2: chi phí tính toán của resampling lớn vs cost/weight."""
    keys = ("smote", "borderline_smote", "adasyn", "rus", "smote_tomek", "smote_enn",
            "class_weight", "focal_loss", "smote_class_weight")
    table: Dict[str, Any] = {}
    for key in keys:
        result = _get(results, key)
        if result is None:
            continue
        base_size = float(result.folds[0].n_train) if result.folds else float("nan")
        table[key] = {
            "n_train_after": result.n_train_after_mean,
            "size_factor": (float(result.n_train_after_mean / base_size)
                            if base_size == base_size and base_size and
                            result.n_train_after_mean == result.n_train_after_mean else 1.0),
            "fit_seconds": result.fit_seconds_mean,
            "resample_seconds": result.resample_seconds_mean,
            "pr_auc": _metric(result, "pr_auc"), "f1": _metric(result, "f1"),
            "recall": _metric(result, "recall"), "fpr": _metric(result, "fpr")}
    heavy = table.get("smote", {}).get("fit_seconds", float("nan"))
    light = table.get("class_weight", {}).get("fit_seconds", float("nan"))
    speedup = (heavy / light) if (light == light and light and light > 0) else float("nan")
    return {"table": table, "speedup_smote_over_class_weight": float(speedup)}


def _hybrid_findings(results: Sequence[TechniqueResult]) -> Dict[str, Any]:
    """Số liệu cho câu hỏi 3: hybrid so với (các) phương pháp đơn lẻ tốt nhất của nó."""
    findings: Dict[str, Any] = {}
    for key, parents in HYBRID_PAIRS.items():
        child = _get(results, key)
        parent_results = [r for r in (_get(results, parent) for parent in parents) if r is not None]
        if child is None or not parent_results:
            continue
        best_parent = max(parent_results, key=lambda r: _metric(r, "pr_auc"))
        findings[key] = {
            "parents": [r.key for r in parent_results], "best_parent": best_parent.key,
            "pr_auc": _metric(child, "pr_auc"), "parent_pr_auc": _metric(best_parent, "pr_auc"),
            "delta_pr_auc": _metric(child, "pr_auc") - _metric(best_parent, "pr_auc"),
            "f1": _metric(child, "f1"), "parent_f1": _metric(best_parent, "f1"),
            "delta_f1": _metric(child, "f1") - _metric(best_parent, "f1"),
            "recall": _metric(child, "recall"), "parent_recall": _metric(best_parent, "recall"),
            "fpr": _metric(child, "fpr"), "parent_fpr": _metric(best_parent, "fpr"),
            "beats_parent": bool(_metric(child, "pr_auc") > _metric(best_parent, "pr_auc")),
        }
    return findings


def analyse(results: Sequence[TechniqueResult], cfg: ExperimentConfig) -> Dict[str, Any]:
    """Tính toàn bộ số liệu so sánh cho báo cáo (Baseline · Single · Hybrid).

    Args:
        results: kết quả của mọi pipeline.
        cfg: cấu hình thực nghiệm (để ghi lại bối cảnh số liệu).

    Returns:
        dict gồm: thông tin baseline, thống kê theo nhóm, so sánh SMOTE vs dọn biên, chi phí tính toán
        (resampling lớn vs cost/weight), hybrid vs đơn lẻ, xếp hạng theo metric chính, hai đầu FPR.
    """
    baseline = _get(results, "baseline")
    single = [r for r in results if r.ok and r.group.startswith("single")]
    hybrid = [r for r in results if r.ok and r.group == "hybrid"]

    ranking: Dict[str, Any] = {}
    for metric in ("pr_auc", "f1", "recall"):
        candidates = [r for r in results if r.ok and np.isfinite(_metric(r, metric))]
        if not candidates:
            continue
        winner = max(candidates, key=lambda r: _metric(r, metric))
        ranking[metric] = {"key": winner.key, "name": winner.name, "group": winner.group,
                           "value": _metric(winner, metric),
                           "baseline": _metric(baseline, metric),
                           "delta_vs_baseline": _metric(winner, metric) - _metric(baseline, metric)}

    fpr_ranked = sorted((r for r in results if r.ok and np.isfinite(_metric(r, "fpr"))),
                        key=lambda r: _metric(r, "fpr"))
    return {
        "config": {"n_samples": cfg.n_samples, "imbalance_ratio": cfg.imbalance_ratio,
                   "n_splits": cfg.n_splits, "threshold": cfg.threshold,
                   "base_model": cfg.base_model, "seed": cfg.seed},
        "baseline": None if baseline is None else {
            "pr_auc": _metric(baseline, "pr_auc"), "f1": _metric(baseline, "f1"),
            "recall": _metric(baseline, "recall"), "fpr": _metric(baseline, "fpr"),
            "fit_seconds": baseline.fit_seconds_mean,
            "accuracy": float(baseline.holdout.get("accuracy", float("nan"))),
            "majority_baseline_accuracy_pct": float(
                baseline.holdout.get("majority_baseline_accuracy_pct", float("nan")))},
        "group_stats": {"baseline": _group_stats([baseline] if baseline else []),
                        "single": _group_stats(single), "hybrid": _group_stats(hybrid)},
        "smote_analysis": _cleaning_findings(results),
        "compute": _compute_findings(results),
        "hybrid_vs_single": _hybrid_findings(results),
        "ranking": ranking,
        "lowest_fpr": [{"key": r.key, "fpr": _metric(r, "fpr")} for r in fpr_ranked[:3]],
        "highest_fpr": [{"key": r.key, "fpr": _metric(r, "fpr")} for r in fpr_ranked[-3:]],
    }



def render_markdown(results: Sequence[TechniqueResult], cfg: ExperimentConfig) -> str:
    """Dựng phần báo cáo phân tích chuyên sâu (3 câu hỏi bắt buộc) bằng số liệu đo được."""
    findings = analyse(results, cfg)
    base = findings["baseline"] or {}
    lines: List[str] = [
        "## Phân tích chuyên sâu (yêu cầu #5)", "",
        f"Bối cảnh: {cfg.n_samples} mẫu, mất cân bằng 1:{cfg.imbalance_ratio}, {cfg.n_splits} fold "
        f"stratified, mô hình nền `{cfg.base_model}`, ngưỡng {cfg.threshold}. "
        "Metric chính: PR-AUC, F1 (thiểu số), Recall, FPR.", "",
        f"**Mốc Baseline (không xử lý)**: PR-AUC "
        f"{_fmt(base.get('pr_auc', float('nan')), 4)}, F1 {_fmt(base.get('f1', float('nan')))}, "
        f"Recall {_fmt(base.get('recall', float('nan')))}, "
        f"FPR {_fmt(base.get('fpr', float('nan')), 4)}, "
        f"{_fmt(base.get('fit_seconds', float('nan')), 2)}s/fold. "
        f"(Accuracy chẩn đoán {100 * base.get('accuracy', float('nan')):.2f}% trong khi đoán lớp đa số "
        f"đã được {base.get('majority_baseline_accuracy_pct', float('nan')):.2f}% ⇒ accuracy không dùng "
        "để kết luận.)", ""]

    # --- (1) SMOTE vs SMOTE + cleaning -------------------------------------------------
    smote = findings["smote_analysis"].get("smote")
    cleaning = findings["smote_analysis"].get("cleaning", {})
    lines += ["### 1. SMOTE đơn lẻ vs SMOTE + dọn biên (SMOTETomek / SMOTEENN)", ""]
    if smote:
        lines.append(f"- SMOTE đơn lẻ: PR-AUC **{_fmt(smote['pr_auc'], 4)}**, F1 {_fmt(smote['f1'])} "
                     f"(CV {_fmt(smote['cv_pr_auc_mean'], 4)} ± {_fmt(smote['cv_pr_auc_std'])}), "
                     f"Recall {_fmt(smote['recall'])}; train sau resample "
                     f"{_fmt(smote['n_train_after'], 0)} mẫu.")
        lines.append("- SMOTE sinh mẫu bằng NỘI SUY giữa các láng giềng thiểu số nên dễ tạo mẫu nằm "
                     "trong vùng đa số (boundary blur/nhiễu). SMOTETomek/SMOTEENN thêm bước DỌN: bỏ mẫu "
                     "sát biên hoặc bị láng giềng 'phủ nhận'.")
    smote_pr = smote["pr_auc"] if smote else float("nan")
    smote_f1 = smote["f1"] if smote else float("nan")
    smote_fpr = smote["fpr"] if smote else float("nan")
    for key, stats in cleaning.items():
        label = "SMOTETomek" if key == "smote_tomek" else "SMOTEENN"
        verdict = "TỐT HƠN" if stats["pr_auc_delta"] > 0 else "KÉM HƠN"
        lines.append(f"- {label}: PR-AUC **{_fmt(stats['pr_auc'], 4)}** "
                     f"(Δ {_delta(stats['pr_auc'], smote_pr)}), F1 {_fmt(stats['f1'])} "
                     f"(Δ {_delta(stats['f1'], smote_f1)}), FPR {_fmt(stats['fpr'], 4)} "
                     f"(Δ {_delta(stats['fpr'], smote_fpr)}), train sau resample "
                     f"{_fmt(stats['n_train_after'], 0)} mẫu ⇒ {verdict} so với SMOTE thuần.")
    lines += ["",
              "**Cách đọc**: dọn biên chỉ có lợi khi mẫu tổng hợp thực sự rơi vào vùng chồng lấn. Khi "
              "hai lớp tách rõ (`class_sep` lớn) hoặc mẫu thiểu số quá ít, phần lớn mẫu SMOTE vốn đã "
              "đúng vùng ⇒ dọn biên chỉ làm MẤT thông tin và có thể GIẢM Recall (bỏ cả mẫu sát biên "
              "'khó nhưng thật'). Vì vậy phải đọc đồng thời ΔPR-AUC, ΔF1 và ΔFPR.", ""]

    # --- (2) Chi phí: resampling lớn vs cost/weight ------------------------------------
    compute = findings["compute"]["table"]
    speedup = findings["compute"]["speedup_smote_over_class_weight"]
    lines += ["### 2. Chi phí tính toán: Resampling lớn vs Cost/Weight (algorithm-level)", "",
              "| Kỹ thuật | n train sau resample | Hệ số kích thước | giây/fold (fit) | "
              "giây/fold (resample) | PR-AUC | Recall | FPR |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, stats in compute.items():
        lines.append(f"| `{key}` | {_fmt(stats['n_train_after'], 0)} | {_fmt(stats['size_factor'], 2)}× "
                     f"| {_fmt(stats['fit_seconds'], 3)} | {_fmt(stats['resample_seconds'], 4)} | "
                     f"{_fmt(stats['pr_auc'], 4)} | {_fmt(stats['recall'])} | {_fmt(stats['fpr'], 4)} |")
    lines += ["",
              "- Oversampling làm tập train PHÌNH RA (hệ số kích thước > 1×) ⇒ mỗi vòng lặp boosting "
              "phải xử lý nhiều mẫu hơn, cộng thêm chi phí sinh mẫu (cột giây/fold resample).",
              "- `class_weight='balanced'` và Focal Loss KHÔNG đổi kích thước dữ liệu (1,0×, không tốn "
              "thời gian resample) nhưng vẫn đổi điểm vận hành (Recall/F1)."]
    if speedup == speedup:
        lines.append(f"- Trên cấu hình này, fit với SMOTE chậm hơn `class_weight` khoảng "
                     f"**{speedup:.1f}×** ({_fmt(compute['smote']['fit_seconds'], 2)}s so với "
                     f"{_fmt(compute['class_weight']['fit_seconds'], 2)}s mỗi fold); khoảng cách còn "
                     "tăng khi tỉ lệ mất cân bằng cao hơn (1:100) hoặc số fold lớn hơn, vì mỗi fold "
                     "phải resample lại từ đầu (đúng nguyên tắc chống rò rỉ).")
    lines += ["- Chọn theo chi phí: dữ liệu lớn/nhiều fold ⇒ ưu tiên cost/weight (rẻ, không phình dữ "
              "liệu); resampling chỉ đáng dùng khi mô hình không hỗ trợ trọng số lớp hoặc cần 'thấy' "
              "nhiều mẫu dương hơn.", ""]

    # --- (3) Khi nào hybrid vượt trội --------------------------------------------------
    hybrid_findings = findings["hybrid_vs_single"]
    lines += ["### 3. Khi nào kết hợp (hybrid) vượt trội so với đơn lẻ?", "",
              "| Hybrid | Đơn lẻ tốt nhất | ΔPR-AUC | ΔF1 | ΔRecall | ΔFPR | Kết luận |",
              "|---|---|---:|---:|---:|---:|---|"]
    winners: List[str] = []
    for key, stats in hybrid_findings.items():
        verdict = "HYBRID tốt hơn" if stats["beats_parent"] else "đơn lẻ tốt hơn"
        if stats["beats_parent"]:
            winners.append(key)
        lines.append(f"| `{key}` | `{stats['best_parent']}` | "
                     f"{_delta(stats['pr_auc'], stats['parent_pr_auc'])} | "
                     f"{_delta(stats['f1'], stats['parent_f1'])} | "
                     f"{_delta(stats['recall'], stats['parent_recall'])} | "
                     f"{_delta(stats['fpr'], stats['parent_fpr'])} | {verdict} |")
    lines.append("")
    if winners:
        lines.append("- Hybrid thắng trên bộ này: " + ", ".join(f"`{key}`" for key in winners)
                     + ". Kết hợp có lợi nhất khi hai can thiệp BÙ TRỪ nhau: resampling đưa thêm mẫu "
                       "thiểu số vào vùng khó, còn trọng số lớp/dọn biên chỉnh lại đúng chỗ mô hình "
                       "còn yếu thay vì nhân bản nhiễu.")
    else:
        lines.append("- Trên cấu hình này KHÔNG có hybrid nào vượt đơn lẻ tốt nhất ⇒ thêm bước xử lý "
                     "không tự động tốt hơn; phải đo trên từng bộ dữ liệu cụ thể.")
    lines += ["- Điều kiện hybrid thường thắng: (a) overlap giữa hai lớp lớn (mẫu SMOTE hay rơi vào "
              "vùng đa số) — dọn biên giúp; (b) mẫu thiểu số quá ít (1:100) — SMOTE + trọng số lớp ổn "
              "định hơn SMOTE thuần; (c) cần tỉ lệ dương cao trong train nhưng ngân sách tính toán hạn "
              "chế — RUSBoost/BalancedBagging rẻ hơn oversampling lớn.", ""]

    # --- Khuyến nghị & giới hạn --------------------------------------------------------
    lines += ["### Khuyến nghị rút ra", ""]
    for metric, info in findings["ranking"].items():
        lines.append(f"- Tốt nhất theo {metric.upper()}: `{info['key']}` ({info['group']}) = "
                     f"**{_fmt(info['value'], 4)}** (baseline {_fmt(info['baseline'], 4)}, "
                     f"Δ {_delta(info['value'], info['baseline'])})")
    if findings["lowest_fpr"]:
        lines.append("- FPR thấp nhất: " + ", ".join(f"`{row['key']}` ({_fmt(row['fpr'], 4)})"
                                                     for row in findings["lowest_fpr"]))
    if findings["highest_fpr"]:
        lines.append("- FPR cao nhất (cần chú ý báo động giả): "
                     + ", ".join(f"`{row['key']}` ({_fmt(row['fpr'], 4)})"
                                 for row in findings["highest_fpr"]))
    lines += ["- Quy trình khuyến nghị: (1) luôn lấy baseline làm mốc; (2) xếp hạng theo PR-AUC (không "
              "phụ thuộc ngưỡng) rồi kiểm tra F1/Recall và FPR; (3) chọn ngưỡng theo chi phí FN/FP trên "
              "xác suất out-of-fold; (4) chỉ giữ kỹ thuật khi mức cải thiện vượt độ lệch chuẩn giữa "
              "các fold.", "", "### Giới hạn", "",
              "- Dataset mặc định là GIẢ LẬP (`make_classification`): kết luận định tính chuyển được, "
              "con số thì không. Muốn chạy dữ liệu thật (credit card fraud) dùng "
              "`--data path/to/creditcard.csv --target Class`.",
              "- Mỗi kỹ thuật dùng MỘT cấu hình hợp lý (tỉ lệ oversampling 0,5; Focal Loss `gamma=2`, "
              "`alpha=0,75`) chứ chưa grid-search ⇒ đây là so sánh 'cùng ngân sách', không phải 'tối ưu "
              "cho từng kỹ thuật'.",
              "- Ngưỡng báo cáo cố định 0,5 để so sánh trực tiếp; muốn tối ưu ngưỡng hãy dùng "
              "`imbalance_lab.thresholds.tune_thresholds_from_pr_curve` (chọn trên xác suất OOF).", ""]
    return "\n".join(lines) + "\n"
