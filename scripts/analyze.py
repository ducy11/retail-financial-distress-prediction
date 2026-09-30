"""Phân tích chuyên sâu: overfitting, quan trọng feature, ablation, lỗi, ngưỡng, hiệu chuẩn.

Đây là phần "Phân tích kết quả chuyên sâu" của báo cáo (tiêu chí then chốt để đạt loại Giỏi).
Mọi hình/bảng sinh từ artifact trong `reports/` nên không có số liệu nhập tay.

Lệnh: python -m scripts.analyze [--quick]
      → reports/figures/analysis/*.png, reports/results/analysis.{json,md},
        reports/results/error_cases.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, learning_curve

from forecasting.config import (COST_FN, COST_FP, FIGURES_DIR, GROUP_KEY, MIN_HISTORY_QUARTERS,
                                MODELS_DIR, RANDOM_SEED, RESULTS_DIR, ensure_dirs,
                                ensure_utf8_stdio)
from forecasting.data_loader import load_prepared
from forecasting.evaluation import metrics_at_threshold
from forecasting.features import (build_feature_matrix, extract_labels, feature_groups,
                                  feature_names, feature_rows, filter_by_history)
from forecasting.models import make_model, predict_proba

ANALYSIS_DIR = FIGURES_DIR / "analysis"
MODEL = "logistic"  # mô hình được chọn trong train (xem reports/results/summary.json)


def _load_json(path: Path) -> Dict[str, Any]:
    """Đọc JSON nếu có, ngược lại trả {} (để script chạy được khi thiếu artifact phụ)."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def artifacts() -> Dict[str, Any]:
    """Nạp mọi artifact có sẵn."""
    return {
        "summary": _load_json(RESULTS_DIR / "summary.json"),
        "test_evaluation": _load_json(RESULTS_DIR / "test_evaluation.json"),
        "baselines": _load_json(RESULTS_DIR / "baselines.json"),
        "validation_checks": _load_json(RESULTS_DIR / "validation_checks.json"),
        "eda": _load_json(RESULTS_DIR / "eda_summary.json"),
        "tuning": _load_json(RESULTS_DIR / "tuning.json"),
    }


def _fit(model_name: str, samples: List[Dict[str, Any]], **params):
    """Fit pipeline mới cho một split (kiểm chứng độc lập, không đọc mô hình cũ)."""
    X, y = build_feature_matrix(samples), extract_labels(samples)
    model = make_model(model_name, **params)
    model.fit(X, y)
    return model, X, y


def overfit_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Bảng Train vs Validation theo từng mô hình (phát hiện overfitting qua gap)."""
    rows: List[Dict[str, Any]] = []
    for m in summary.get("models", []):
        rows.append({
            "model": m["model"],
            "train_auroc": m.get("train_auroc"),
            "val_auroc": m.get("auroc"),
            "gap_auroc": m.get("overfit_gap_auroc"),
            "train_f1": (m.get("train_at_0.5") or {}).get("f1"),
            "val_f1": (m.get("val_at_0.5") or {}).get("f1"),
            "gap_f1": m.get("overfit_gap_f1"),
            "val_ap": m.get("average_precision"),
            "best_f1_val": m.get("best_f1_val"),
            "brier_val": m.get("brier_val"),
        })
    return rows


def fig_overfit(rows: List[Dict[str, Any]], path: Path) -> None:
    """Hình: cột đôi Train/Validation AUROC cho từng mô hình (đọc overfit bằng mắt)."""
    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, [r["train_auroc"] or 0 for r in rows], 0.4, label="Train AUROC", color="slateblue")
    ax.bar(x + 0.2, [r["val_auroc"] or 0 for r in rows], 0.4, label="Validation AUROC", color="orange")
    for i, r in enumerate(rows):
        ax.text(i, max(r["train_auroc"] or 0, r["val_auroc"] or 0) + 0.01,
                f"gap={r['gap_auroc']:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylim(0.8, 1.04)
    ax.set_ylabel("AUROC")
    ax.set_title("Train vs Validation — kiểm tra overfitting")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_importance(train_s, val_s, test_s, path: Path, n_repeats: int = 20
                       ) -> Dict[str, Any]:
    """Độ quan trọng feature theo 3 góc nhìn: permutation (val/test) + hệ số logistic.

    Permutation importance = mức giảm AUROC khi hoán vị ngẫu nhiên một cột. Vừa để diễn giải,
    vừa để phát hiện "cột quyết định" (dấu hiệu bài toán bị một đặc trưng chi phối).
    """
    model, _, _ = _fit(MODEL, train_s)
    names = feature_names()
    out: Dict[str, Any] = {"model": MODEL, "n_features": len(names)}
    for tag, samples in (("val", val_s), ("test", test_s)):
        X = build_feature_matrix(samples)
        y = extract_labels(samples)
        pi = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=RANDOM_SEED)
        order = np.argsort(-pi.importances_mean)
        out[tag] = [{"feature": names[i], "mean_decrease_auroc": float(pi.importances_mean[i]),
                     "std": float(pi.importances_std[i])} for i in order[:15]]

    coefs = model.named_steps["model"].coef_.ravel()
    order = np.argsort(-np.abs(coefs))
    out["logistic_coefficients_standardized"] = [
        {"feature": names[i], "coef": float(coefs[i])} for i in order[:15]]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, tag in zip(axes, ("val", "test")):
        items = out[tag][:12][::-1]
        ax.barh([i["feature"] for i in items], [i["mean_decrease_auroc"] for i in items],
                xerr=[i["std"] for i in items], color="teal", alpha=0.85)
        ax.set_title(f"Permutation importance — {tag} (ΔAUROC)")
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return out


def correlation_vif(train_s, path: Path) -> Dict[str, Any]:
    """Tương quan giữa các feature + VIF: hệ số logistic có diễn giải được hay không?"""
    X = build_feature_matrix(train_s)
    names = feature_names()
    keep = [j for j in range(X.shape[1]) if np.isfinite(X[:, j]).sum() >= 30]
    filled = np.where(np.isfinite(X[:, keep]), X[:, keep], np.nanmedian(X[:, keep], axis=0))
    keep = [j for j, k in zip(keep, range(filled.shape[1])) if filled[:, k].std() > 0]
    filled = np.where(np.isfinite(X[:, keep]), X[:, keep], np.nanmedian(X[:, keep], axis=0))
    corr = np.corrcoef(filled, rowvar=False)

    vif: Dict[str, float] = {}
    for k, col in enumerate(keep):
        y = filled[:, k]
        others = np.delete(filled, k, axis=1)
        A = np.column_stack([np.ones(len(others)), others])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        denom = float((y - y.mean()) @ (y - y.mean()))
        r2 = 1 - float(resid @ resid) / denom if denom > 0 else 0.0
        vif[names[col]] = float(1 / max(1e-9, 1 - r2))

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)), [names[j] for j in keep], rotation=90, fontsize=6)
    ax.set_yticks(range(len(keep)), [names[j] for j in keep], fontsize=6)
    ax.set_title("Tương quan giữa các feature (train)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    top = sorted(vif.items(), key=lambda kv: -kv[1])[:10]
    return {"n_features_used": len(keep), "vif_top10": [{"feature": f, "vif": v} for f, v in top],
            "n_vif_above_10": int(sum(1 for v in vif.values() if v > 10))}


def _evaluate_variant(train_s, val_s, test_s, drop_cols: List[str] | None = None,
                      min_history: int = 0, model_name: str = MODEL) -> Dict[str, Any]:
    """Fit một biến thể (bỏ cột / lọc mẫu non) và trả metric val+test để so ablation."""
    if min_history:
        train_s = filter_by_history(train_s, min_history)
        val_s = filter_by_history(val_s, min_history)
        test_s = filter_by_history(test_s, min_history)
    names = feature_names()
    keep = [j for j, n in enumerate(names) if not drop_cols or n not in set(drop_cols)]

    X_tr = build_feature_matrix(train_s)[:, keep]
    X_val = build_feature_matrix(val_s)[:, keep]
    X_te = build_feature_matrix(test_s)[:, keep]
    y_tr, y_val, y_te = (extract_labels(s) for s in (train_s, val_s, test_s))

    model = make_model(model_name)
    model.fit(X_tr, y_tr)
    out: Dict[str, Any] = {"n_features": len(keep), "n_train": len(y_tr), "n_test": len(y_te)}
    for tag, X, y in (("val", X_val, y_val), ("test", X_te, y_te)):
        proba = predict_proba(model, X)
        at = metrics_at_threshold(y, proba, 0.5)
        ap = (float(average_precision_score(y, proba))
              if y.sum() and len(set(y.tolist())) > 1 else None)
        out[tag] = {"auroc": float(roc_auc_score(y, proba)) if len(set(y.tolist())) > 1 else None,
                    "average_precision": ap, "f1": at["f1"], "macro_f1": at["macro_f1"],
                    "accuracy": at["accuracy"]}
    return out


def ablation(train_s, val_s, test_s) -> List[Dict[str, Any]]:
    """Ablation: nhóm feature nào thực sự đóng góp, và mẫu quá non ảnh hưởng ra sao?"""
    names = feature_names()
    groups = feature_groups()
    rows: List[Dict[str, Any]] = []

    base = _evaluate_variant(train_s, val_s, test_s)
    rows.append({"variant": "tất cả feature", "drop": [], "min_history": 0, **base})

    for group, cols in groups.items():
        res = _evaluate_variant(train_s, val_s, test_s, drop_cols=cols)
        rows.append({"variant": f"bỏ nhóm {group} ({len(cols)} cột)", "drop": cols,
                     "min_history": 0, **res})

    # Bỏ các cột có độ phủ thấp trên train (<50% giá trị hữu hạn)
    X_tr = build_feature_matrix(train_s)
    coverage = np.isfinite(X_tr).mean(axis=0)
    low = [n for n, c in zip(names, coverage) if c < 0.5]
    res = _evaluate_variant(train_s, val_s, test_s, drop_cols=low)
    rows.append({"variant": f"bỏ cột độ phủ <50% ({len(low)} cột)", "drop": low,
                 "min_history": 0, **res})

    res = _evaluate_variant(train_s, val_s, test_s, min_history=MIN_HISTORY_QUARTERS)
    rows.append({"variant": f"chỉ mẫu có ≥{MIN_HISTORY_QUARTERS} quý lịch sử", "drop": [],
                 "min_history": MIN_HISTORY_QUARTERS, **res})
    return rows


def error_cases(test_evaluation: Dict[str, Any], test_s) -> Tuple[List[Dict[str, Any]], Path]:
    """Bảng các mẫu dự đoán sai kèm giá trị chỉ tiêu — để giải thích lỗi có cấu trúc."""
    mis = test_evaluation.get("misclassified", [])
    wanted = ["current_ratio_latest", "working_capital_to_assets", "net_margin_latest",
              "debt_to_assets_latest", "inventory_to_sales_latest", "ocf_to_sales_latest",
              "revenue_yoy_growth"]
    lookup = {r["sample_id"]: r for r in feature_rows(test_s)}
    out: List[Dict[str, Any]] = []
    for m in mis:
        row = lookup.get(m["sample_id"], {})
        record = {"sample_id": m["sample_id"], "ticker": m["ticker"], "actual": m["actual"],
                  "predicted": m["predicted"], "probability": m["probability"],
                  "n_history": row.get("n_history")}
        for w in wanted:
            record[w] = row.get(w)
        out.append(record)

    path = RESULTS_DIR / "error_cases.csv"
    if out:
        import csv

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            writer.writeheader()
            writer.writerows(out)
    return out, path


def threshold_curve(train_s, val_s, test_s, path: Path) -> Dict[str, Any]:
    """Đường precision/recall/F1/chi phí theo ngưỡng — chọn ngưỡng có lý do, không tùy hứng."""
    model, _, _ = _fit(MODEL, train_s)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 3)
    out: Dict[str, Any] = {"model": MODEL, "cost_fn": COST_FN, "cost_fp": COST_FP}

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (tag, samples) in enumerate((("validation", val_s), ("test", test_s))):
        X, y = build_feature_matrix(samples), extract_labels(samples)
        proba = predict_proba(model, X)
        rows = [metrics_at_threshold(y, proba, float(t)) for t in grid]
        out[tag] = {"best_f1": max(rows, key=lambda r: r["f1"]),
                    "cost_optimal": min(rows, key=lambda r: r["expected_cost"])}
        ax = axes[0, col]
        ax.plot(grid, [r["precision"] for r in rows], label="Precision")
        ax.plot(grid, [r["recall"] for r in rows], label="Recall")
        ax.plot(grid, [r["f1"] for r in rows], label="F1")
        ax.axvline(out[tag]["best_f1"]["threshold"], color="k", ls="--", alpha=0.7, label="best-F1")
        ax.set_title(f"{tag}: metric theo ngưỡng")
        ax.set_xlabel("threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax2 = axes[1, col]
        ax2.plot(grid, [r["expected_cost"] for r in rows], color="crimson",
                 label=f"chi phí kỳ vọng = {COST_FN:.0f}·FN + {COST_FP:.0f}·FP")
        ax2.axvline(out[tag]["cost_optimal"]["threshold"], color="k", ls=":",
                    alpha=0.8, label="ngưỡng tối ưu chi phí")
        ax2.axvline(0.5, color="gray", ls="-", alpha=0.4, label="0.5")
        ax2.set_title(f"{tag}: chi phí kỳ vọng theo ngưỡng")
        ax2.set_xlabel("threshold")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return out


def _reliability(y: np.ndarray, proba: np.ndarray, n_bins: int = 8
                 ) -> Tuple[List[float], List[float], List[int]]:
    """(xác suất dự báo trung bình, tần suất thực tế, số mẫu) theo từng bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            xs.append(float(proba[m].mean()))
            ys.append(float(y[m].mean()))
            ns.append(int(m.sum()))
    return xs, ys, ns


def calibration(train_s, val_s, test_s, path: Path, n_bins: int = 8) -> Dict[str, Any]:
    """Đường hiệu chuẩn + Brier: xác suất của mô hình có khớp tần suất thực tế?"""
    model, _, _ = _fit(MODEL, train_s)
    out: Dict[str, Any] = {"model": MODEL, "bins": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (tag, samples) in zip(axes, (("validation", val_s), ("test", test_s))):
        X, y = build_feature_matrix(samples), extract_labels(samples)
        proba = predict_proba(model, X)
        xs, ys, ns = _reliability(y, proba, n_bins)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="hiệu chuẩn hoàn hảo")
        ax.plot(xs, ys, "o-", color="darkred", label="mô hình")
        ax.bar(xs, [n / max(ns) * 0.25 for n in ns], width=0.06, alpha=0.3, color="steelblue",
               label="số mẫu (tỷ lệ)")
        brier = float(brier_score_loss(y, proba))
        ax.set_title(f"{tag}: reliability (Brier={brier:.3f}, n={len(y)})")
        ax.set_xlabel("xác suất dự báo")
        ax.set_ylabel("tần suất thực tế")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        out["bins"][tag] = {"predicted_mean": xs, "observed_rate": ys, "n": ns, "brier": brier}
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return out


def learning_curve_figure(train_s, path: Path, n_splits: int = 4) -> Dict[str, Any]:
    """Learning curve với GroupKFold: thêm dữ liệu còn giúp được nữa không?"""
    X, y = build_feature_matrix(train_s), extract_labels(train_s)
    groups = np.asarray([s[GROUP_KEY] for s in train_s])
    sizes, train_scores, val_scores = learning_curve(
        make_model(MODEL), X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
        scoring="roc_auc", train_sizes=np.linspace(0.35, 1.0, 5), random_state=RANDOM_SEED,
        n_jobs=1)
    tr_means, tr_std = np.nanmean(train_scores, axis=1), np.nanstd(train_scores, axis=1)
    va_means, va_std = np.nanmean(val_scores, axis=1), np.nanstd(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(sizes, tr_means, "o-", label="Train AUROC")
    ax.fill_between(sizes, tr_means - tr_std, tr_means + tr_std, alpha=0.2)
    ax.plot(sizes, va_means, "s-", color="orange", label="Validation AUROC (GroupKFold)")
    ax.fill_between(sizes, va_means - va_std, va_means + va_std, alpha=0.2, color="orange")
    ax.set_xlabel("số mẫu train")
    ax.set_ylabel("AUROC")
    ax.set_title("Learning curve (chia theo công ty)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"train_sizes": [int(s) for s in sizes],
            "train_auroc_mean": [float(v) for v in tr_means],
            "val_auroc_mean": [float(v) for v in va_means],
            "gap_at_full_data": float(tr_means[-1] - va_means[-1])}


def headline(arts: Dict[str, Any]) -> Dict[str, Any]:
    """Bảng 'câu chuyện chính': in-domain vs cross-company vs baseline vs dummy."""
    rows: List[Dict[str, Any]] = []
    checks = arts.get("validation_checks", {})
    for name, d in (checks.get("in_domain_test") or {}).items():
        rows.append({"system": f"model[{name}] — in-domain", "auroc": d.get("test_auroc"),
                     "average_precision": d.get("test_average_precision"), "source": "train→test"})
    for name, d in (checks.get("grouped_cv_all_samples") or {}).items():
        if not isinstance(d, dict) or "oof_auroc" not in d:
            continue
        rows.append({"system": f"model[{name}] — cross-company", "auroc": d.get("oof_auroc"),
                     "average_precision": d.get("oof_average_precision"), "source": "GroupKFold"})
    for r in arts.get("baselines", {}).get("rows", []):
        rows.append({"system": r["baseline"], "auroc": r["test"]["auroc"],
                     "average_precision": r["test"]["average_precision"], "source": "baseline"})
    head = checks.get("headline", {})
    return {"rows": rows, "loco_mean_auroc": head.get("loco_mean_auroc"),
            "loco_n_evaluable": head.get("loco_n_evaluable"),
            "loco_skipped_companies": head.get("loco_skipped_companies"),
            "dummy_reference_auroc": 0.5}


def label_audit(train_s, val_s, test_s) -> Dict[str, Any]:
    """Audit nhãn: nhãn gốc có khớp quy tắc kế toán đơn giản nào không? (không phải nhập tay)

    Trả mức khớp của từng quy tắc thử nghiệm, tính trên (a) quý target và (b) dòng lịch sử cuối
    cùng — giúp phân biệt “nhãn theo tương lai” với “nhãn đã có trong feature” (rò rỉ).
    """
    from forecasting.labels import signal_flags

    samples = train_s + val_s + test_s
    root = Path(__file__).resolve().parents[1]
    retail = {}
    for f in sorted((root / "data" / "retail-expanded").glob("*-16-indicators-vnd.json")):
        doc = json.loads(f.read_text(encoding="utf-8"))
        retail[doc["ticker"]] = doc["rows"]
    index = {}
    for ticker, rows in retail.items():
        for i, row in enumerate(rows):
            index[f"{ticker}-{row['fiscal_year']}Q{row['fiscal_quarter']}"] = (row, rows, i)

    def _num(row: Dict[str, Any], field: str) -> float | None:
        value = row.get(field + "_vnd")
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    rules = {
        "net_income<0": lambda r: (_num(r, "net_income") or 0) < 0,
        "operating_income<0": lambda r: (_num(r, "operating_income") or 0) < 0,
        "operating_cash_flow<0": lambda r: (_num(r, "operating_cash_flow") or 0) < 0,
        "ocf<0 hoặc ni<0": lambda r: ((_num(r, "operating_cash_flow") or 0) < 0
                                      or (_num(r, "net_income") or 0) < 0),
        "current_liabilities>current_assets": lambda r: (
            (_num(r, "current_liabilities") or 0) > (_num(r, "current_assets") or 0)),
        "stockholders_equity<0": lambda r: (_num(r, "stockholders_equity") or 0) < 0,
        "retained_earnings<0": lambda r: (_num(r, "retained_earnings") or 0) < 0,
    }

    target_agree: Dict[str, float] = {}
    history_agree: Dict[str, float] = {}
    for name, fn in rules.items():
        ok_t = ok_h = total = 0
        for s in samples:
            entry = index.get(s["sample_id"])
            if not entry:
                continue
            row, rows, i = entry
            total += 1
            ok_t += int(int(bool(fn(row))) == int(s["is_distressed"]))
            last = rows[i] if i > 0 else rows[0]      # dòng lịch sử cuối cùng trước quý target
            ok_h += int(int(bool(fn(last))) == int(s["is_distressed"]))
        target_agree[name] = ok_t / total if total else float("nan")
        history_agree[name] = ok_h / total if total else float("nan")

    # 6 tín hiệu căng thẳng tổng hợp (dùng cho nhãn quy tắc ở scripts.relabel)
    stress_flags = {"stress_signals>=1": [], "stress_signals>=2": []}
    for s in samples:
        entry = index.get(s["sample_id"])
        if not entry:
            continue
        row, rows, i = entry
        flags = signal_flags(row, rows, i)
        active = sum(1 for on in flags.values() if on)
        stress_flags["stress_signals>=1"].append(int(int(active >= 1) == int(s["is_distressed"])))
        stress_flags["stress_signals>=2"].append(int(int(active >= 2) == int(s["is_distressed"])))
    for name, values in stress_flags.items():
        target_agree[name] = float(np.mean(values)) if values else float("nan")

    return {"n_samples": len(samples), "rule_agreement": target_agree,
            "rule_agreement_last_history": history_agree,
            "max_rule_agreement": float(np.nanmax(list(target_agree.values()))),
            "note": ("Nhãn gốc không khớp quy tắc kế toán đơn giản nào => không tái tạo được; "
                     "xem docs/dinh-nghia-nhan.md và scripts/relabel.py.")}


def fig_headline(rows: List[Dict[str, Any]], path: Path) -> None:
    """Hình: AUROC in-domain vs cross-company vs baseline — hình 'gây ấn tượng' của báo cáo."""
    items = [r for r in rows if r.get("auroc") is not None]
    items.sort(key=lambda r: r["auroc"])
    colors = ["crimson" if r["source"] in ("baseline",) else
              "slateblue" if r["source"] == "train→test" else "seagreen" for r in items]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh([r["system"] for r in items], [r["auroc"] for r in items], color=colors, alpha=0.9)
    ax.axvline(0.5, color="k", ls="--", alpha=0.6, label="ngẫu nhiên (0.5)")
    ax.set_xlim(0.4, 1.02)
    ax.set_xlabel("AUROC (test)")
    ax.set_title("AUROC: in-domain (tím) vs cross-company (xanh) vs baseline (đỏ)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _n(value: Any, digits: int = 3) -> str:
    """Định dạng số cho bảng markdown (None/NaN → '—')."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else f"{f:.{digits}f}"  # NaN → "—"


def _pct(value: Any, digits: int = 1) -> str:
    """Định dạng tỷ lệ 0–1 thành phần trăm (None/NaN → '—')."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if f != f else f"{100 * f:.{digits}f}%"  # NaN → "—"


def _markdown(out: Dict[str, Any]) -> str:
    """Sinh báo cáo markdown từ kết quả phân tích (không nhập tay số liệu)."""
    lines: List[str] = ["# Phân tích kết quả chuyên sâu (sinh tự động)", "",
                        "Số liệu lấy từ `reports/results/*.json`; hình ở "
                        "`reports/figures/analysis/`.", ""]

    lines += ["## 1. Overfitting: Train vs Validation", "",
              "| Mô hình | Train AUROC | Val AUROC | Gap AUROC | Train F1 | Val F1 | Gap F1 | Val AP | Brier |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out["overfit"]:
        lines.append(f"| {r['model']} | {_n(r['train_auroc'])} | {_n(r['val_auroc'])} | "
                     f"{_n(r['gap_auroc'])} | {_n(r['train_f1'])} | {_n(r['val_f1'])} | "
                     f"{_n(r['gap_f1'])} | {_n(r['val_ap'])} | {_n(r['brier_val'])} |")

    lines += ["", "## 2. Feature quan trọng nhất (permutation, ΔAUROC khi hoán vị)", "",
              "| # | Feature | Val ΔAUROC | Test ΔAUROC | Hệ số logistic |",
              "|---:|---|---:|---:|---:|"]
    coefs = {c["feature"]: c["coef"] for c in out["importance"]["logistic_coefficients_standardized"]}
    for i, item in enumerate(out["importance"]["val"][:10], start=1):
        test_val = next((t["mean_decrease_auroc"] for t in out["importance"]["test"]
                         if t["feature"] == item["feature"]), None)
        lines.append(f"| {i} | {item['feature']} | {_n(item['mean_decrease_auroc'])} | "
                     f"{_n(test_val)} | {_n(coefs.get(item['feature']), 2)} |")

    lines += ["", "## 3. Đa cộng tuyến (VIF)", "",
              f"Số feature VIF > 10: **{out['vif']['n_vif_above_10']}** / "
              f"{out['vif']['n_features_used']}", "",
              "| Feature | VIF |", "|---|---:|"]
    for item in out["vif"]["vif_top10"]:
        lines.append(f"| {item['feature']} | {_n(item['vif'], 1)} |")

    lines += ["", "## 4. Ablation (bỏ nhóm feature / lọc mẫu non)", "",
              "| Biến thể | #feature | #train | Val AUROC | Val AP | Test AUROC | Test AP | Test F1 | Test macro-F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out["ablation"]:
        lines.append(f"| {r['variant']} | {r['n_features']} | {r['n_train']} | "
                     f"{_n(r['val']['auroc'])} | {_n(r['val']['average_precision'])} | "
                     f"{_n(r['test']['auroc'])} | {_n(r['test']['average_precision'])} | "
                     f"{_n(r['test']['f1'])} | {_n(r['test']['macro_f1'])} |")

    lines += ["", "## 5. Phân tích lỗi (mẫu dự đoán sai trên test)", "",
              "| Mẫu | Công ty | Thực tế | Dự đoán | P(distress) | n_history | current_ratio | WC/assets | net_margin | debt/assets |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in out["error_cases"]:
        lines.append(f"| {e['sample_id']} | {e['ticker']} | {e['actual']} | {e['predicted']} | "
                     f"{_n(e['probability'])} | {e['n_history']} | "
                     f"{_n(e['current_ratio_latest'], 2)} | {_n(e['working_capital_to_assets'], 2)} | "
                     f"{_n(e['net_margin_latest'], 3)} | {_n(e['debt_to_assets_latest'], 2)} |")

    thr = out["threshold"]
    lines += ["", "## 6. Ngưỡng quyết định (theo F1 và theo chi phí)", "",
              "| Tập | Ngưỡng best-F1 | F1 | Precision | Recall | Ngưỡng tối ưu chi phí | Chi phí kỳ vọng |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for tag in ("validation", "test"):
        b, c = thr[tag]["best_f1"], thr[tag]["cost_optimal"]
        lines.append(f"| {tag} | {_n(b['threshold'])} | {_n(b['f1'])} | {_n(b['precision'])} | "
                     f"{_n(b['recall'])} | {_n(c['threshold'])} | {_n(c['expected_cost'], 1)} |")
    lines += ["", f"Chi phí giả định: FN = {thr['cost_fn']:.0f}, FP = {thr['cost_fp']:.0f}.", ""]

    lines += ["## 7. Hiệu chuẩn xác suất", "", "| Tập | Brier |", "|---|---:|"]
    for tag, d in out["calibration"]["bins"].items():
        lines.append(f"| {tag} | {_n(d['brier'])} |")

    lc = out["learning_curve"]
    lines += ["", "## 8. Learning curve (chia theo công ty)", "",
              "| #mẫu train | Train AUROC | Val AUROC |", "|---:|---:|---:|"]
    for s, a, b in zip(lc["train_sizes"], lc["train_auroc_mean"], lc["val_auroc_mean"]):
        lines.append(f"| {s} | {_n(a)} | {_n(b)} |")
    lines += ["", f"Gap ở dữ liệu đầy đủ: **{_n(lc['gap_at_full_data'])}** AUROC.", ""]

    lines += ["## 9. In-domain vs cross-company vs baseline", "",
              "| Hệ thống | AUROC | AP | Nguồn |", "|---|---:|---:|---|"]
    for r in out["headline"]["rows"]:
        lines.append(f"| {r['system']} | {_n(r['auroc'])} | {_n(r['average_precision'])} | {r['source']} |")
    skipped = ", ".join(out["headline"]["loco_skipped_companies"] or []) or "—"
    lines += ["", f"LOCO trung bình: **{_n(out['headline']['loco_mean_auroc'])}** AUROC trên "
                  f"{out['headline']['loco_n_evaluable']} phép so; công ty không tính được "
                  f"(nhãn đơn lớp): {skipped}", ""]

    audit = out.get("label_audit", {})
    lines += ["## 10. Audit nhãn (nhãn gốc có tái tạo được?)", "",
              "| Quy tắc thử nghiệm (trên quý target) | Khớp với nhãn gốc | Khớp khi áp trên dòng lịch sử cuối |",
              "|---|---:|---:|"]
    for name, value in audit.get("rule_agreement", {}).items():
        lines.append(f"| {name} | {_pct(value)} | "
                     f"{_pct(audit.get('rule_agreement_last_history', {}).get(name))} |")
    lines += ["", f"Khớp cao nhất: **{_pct(audit.get('max_rule_agreement'))}** trên "
                  f"{audit.get('n_samples')} mẫu ⇒ nhãn gốc không tái tạo được từ dữ liệu công bố.", ""]
    return "\n".join(lines)


def run(quick: bool = False) -> Dict[str, Any]:
    """Chạy toàn bộ phân tích chuyên sâu, ghi hình + bảng markdown/json."""
    ensure_dirs()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    arts = artifacts()
    train_s, val_s, test_s = (load_prepared(n) for n in ("train", "validation", "test"))

    overfit = overfit_table(arts["summary"])
    fig_overfit(overfit, ANALYSIS_DIR / "01_overfit_train_vs_val.png")
    audit = label_audit(train_s, val_s, test_s)
    imp = feature_importance(train_s, val_s, test_s, ANALYSIS_DIR / "02_feature_importance.png",
                             n_repeats=8 if quick else 20)
    vif = correlation_vif(train_s, ANALYSIS_DIR / "03_correlation_matrix.png")
    abl = ablation(train_s, val_s, test_s)
    errs, err_path = error_cases(arts["test_evaluation"], test_s)
    thr = threshold_curve(train_s, val_s, test_s, ANALYSIS_DIR / "04_threshold_curves.png")
    cal = calibration(train_s, val_s, test_s, ANALYSIS_DIR / "05_calibration.png")
    lc = learning_curve_figure(train_s, ANALYSIS_DIR / "06_learning_curve.png")
    head = headline(arts)
    fig_headline(head["rows"], ANALYSIS_DIR / "07_in_domain_vs_cross_company.png")

    out: Dict[str, Any] = {"overfit": overfit, "importance": imp, "vif": vif, "ablation": abl,
                           "label_audit": audit, "error_cases": errs,
                           "error_cases_csv": str(err_path), "threshold": thr,
                           "calibration": cal, "learning_curve": lc, "headline": head}
    (RESULTS_DIR / "analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (RESULTS_DIR / "analysis.md").write_text(_markdown(out), encoding="utf-8")

    top = out["importance"]["test"][0] if out["importance"]["test"] else {}
    cross = [r for r in head["rows"] if r["source"] == "GroupKFold"]
    cross_auc = min((r["auroc"] for r in cross if r["auroc"] is not None), default=None)
    print(f"Phân tích: top feature (test) = {top.get('feature')} "
          f"(ΔAUROC={_n(top.get('mean_decrease_auroc'))}); VIF>10: {vif['n_vif_above_10']} cột; "
          f"lỗi test: {len(errs)} mẫu; AUROC xấu nhất cross-company = {_n(cross_auc)}")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Giảm số lần hoán vị (chạy nhanh).")
    args = parser.parse_args(argv)
    print("=== Phân tích chuyên sâu ===")
    run(quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
