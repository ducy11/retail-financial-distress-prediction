"""Tinh chỉnh hyperparameter bằng GridSearchCV chia theo CÔNG TY (StratifiedGroupKFold).

Vì sao phải chia theo nhóm công ty: bộ test in-domain chứa đúng các công ty trong train, nên
nếu tune bằng CV thường (trộn mọi quý của mọi công ty) thì điểm CV bị thổi phồng — mô hình chỉ
cần nhận ra công ty. Ở đây mọi fold giữ TRỌN một số công ty ra ngoài.

Cách chọn cấu hình: `refit="average_precision"` (AP) vì AP không phụ thuộc ngưỡng; sau khi chọn,
cấu hình tốt nhất được đánh giá lại trên validation (độc lập với CV) để báo cáo.

Lệnh: python -m forecasting.tuning [--quick] [--models logistic,random_forest]
      → reports/results/tuning.json + reports/results/tuning.md
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score

from .config import GROUP_KEY, RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from .data_loader import load_prepared
from .evaluation import evaluate_proba
from .features import build_feature_matrix, extract_labels
from .models import MODEL_REGISTRY, make_model, predict_proba

#: Lưới đầy đủ (20–40 cấu hình/mô hình tuỳ máy).
GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {"model__C": [0.01, 0.1, 1.0, 10.0],
                 "model__class_weight": [None, "balanced"]},
    "random_forest": {"model__max_depth": [3, 6, None],
                      "model__min_samples_leaf": [1, 2, 4],
                      "model__n_estimators": [200, 500]},
    "hist_gradient_boosting": {"model__learning_rate": [0.03, 0.1],
                               "model__max_depth": [2, 3],
                               "model__max_iter": [200, 400]},
    "lightgbm": {"model__learning_rate": [0.03, 0.05],
                 "model__num_leaves": [7, 15, 31],
                 "model__min_child_samples": [5, 10]},
}

#: Lưới rút gọn cho máy yếu (--quick).
QUICK_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {"model__C": [0.1, 1.0]},
    "random_forest": {"model__max_depth": [3, 6], "model__min_samples_leaf": [2]},
    "hist_gradient_boosting": {"model__learning_rate": [0.05], "model__max_depth": [3]},
    "lightgbm": {"model__learning_rate": [0.05], "model__num_leaves": [15]},
}

#: Hai metric chấm điểm CV: AUROC (xếp hạng) và AP (quan trọng khi lớp dương là lớp cần bắt).
SCORING = {"auroc": "roc_auc", "average_precision": "average_precision"}


def tune_one(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             quick: bool = False, n_splits: int = 4) -> Dict[str, Any]:
    """GridSearchCV cho một mô hình; trả dict gồm bảng kết quả CV và cấu hình tốt nhất."""
    grid = QUICK_GRIDS[name] if quick else GRIDS[name]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(make_model(name), grid, scoring=SCORING,
                          refit="average_precision", cv=cv, n_jobs=1, return_train_score=False)
    search.fit(X, y, groups=groups)

    rows: List[Dict[str, Any]] = []
    for params, mean_ap, mean_auroc in zip(search.cv_results_["params"],
                                          search.cv_results_["mean_test_average_precision"],
                                          search.cv_results_["mean_test_auroc"]):
        rows.append({
            "params": {k.replace("model__", ""): v for k, v in params.items()},
            "cv_average_precision": float(mean_ap),
            "cv_auroc": float(mean_auroc) if mean_auroc == mean_auroc else None,
        })
    rows.sort(key=lambda r: -r["cv_average_precision"])

    # Điểm của cấu hình MẶC ĐỊNH (trong models.HYPERPARAMS) trên cùng splitter: để chứng minh
    # tuning cải thiện thật hay không, chứ không chỉ "đã chạy GridSearch".
    default_scores = cross_val_score(make_model(name), X, y, groups=groups, cv=cv,
                                     scoring="average_precision", n_jobs=1)
    best = search.best_params_
    return {
        "model": name,
        "n_candidates": len(rows),
        "cv_folds": n_splits,
        "best_params": {k.replace("model__", ""): v for k, v in best.items()},
        "best_cv_average_precision": float(search.best_score_),
        "best_cv_auroc": float(search.cv_results_["mean_test_auroc"][search.best_index_]),
        "table": rows,
        "default_cv_reference": {"mean_average_precision": float(np.mean(default_scores)),
                                 "scores": [float(s) for s in default_scores]},
    }


def run(models: List[str] | None = None, quick: bool = False,
        n_splits: int = 4) -> Dict[str, Any]:
    """Tinh chỉnh các mô hình trên train bằng CV chia theo công ty, xác nhận lại trên validation."""
    ensure_dirs()
    names = [m for m in (models or ["logistic", "random_forest", "hist_gradient_boosting", "lightgbm"])
             if m in MODEL_REGISTRY]
    train_s, val_s = load_prepared("train"), load_prepared("validation")
    X, y = build_feature_matrix(train_s), extract_labels(train_s)
    groups = np.asarray([s[GROUP_KEY] for s in train_s])
    X_val, y_val = build_feature_matrix(val_s), extract_labels(val_s)

    results: List[Dict[str, Any]] = []
    for name in names:
        item = tune_one(name, X, y, groups, quick=quick, n_splits=n_splits)
        model = make_model(name, **item["best_params"])
        model.fit(X, y)
        val_metrics = evaluate_proba(y_val, predict_proba(model, X_val))
        item["validation"] = {
            "auroc": val_metrics["auroc"],
            "average_precision": val_metrics["average_precision"],
            "best_f1": val_metrics["best_f1"],
            "at_0.5": next(b for b in val_metrics["by_threshold"] if b["threshold"] == 0.5),
        }
        # Cấu hình mặc định đánh giá cùng ngưỡng để so sánh công bằng
        base = make_model(name)
        base.fit(X, y)
        base_metrics = evaluate_proba(y_val, predict_proba(base, X_val))
        item["validation_default"] = {
            "auroc": base_metrics["auroc"],
            "average_precision": base_metrics["average_precision"],
            "best_f1": base_metrics["best_f1"],
        }
        results.append(item)
        print(f"  {name:24s} best={item['best_params']} "
              f"CV-AP={item['best_cv_average_precision']:.3f} "
              f"(mặc định {item['default_cv_reference']['mean_average_precision']:.3f}) "
              f"val AP={item['validation']['average_precision']:.3f}")

    out = {"cv": "StratifiedGroupKFold theo ticker",
           "refit_metric": "average_precision",
           "quick": quick, "results": results}
    (RESULTS_DIR / "tuning.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = ["# Tinh chỉnh hyperparameter (CV chia theo công ty)", ""]
    lines.append("| Mô hình | Cấu hình tốt nhất | CV-AP | CV-AUROC | Val-AP | Val-AUROC | Val-F1* |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        v = r["validation"]
        lines.append(f"| {r['model']} | `{r['best_params']}` | {r['best_cv_average_precision']:.3f} | "
                     f"{r['best_cv_auroc']:.3f} | {v['average_precision']:.3f} | {v['auroc']:.3f} | "
                     f"{(v['best_f1'] or {}).get('f1', float('nan')):.3f} |")
    lines += ["", "*(F1* = F1 tốt nhất trên validation; AP = average precision.)*", ""]
    for r in results:
        lines.append(f"## {r['model']} — {r['n_candidates']} cấu hình")
        lines.append("")
        lines.append("| # | Cấu hình | CV-AP | CV-AUROC |")
        lines.append("|---:|---|---:|---:|")
        for i, row in enumerate(r["table"][:12], start=1):
            lines.append(f"| {i} | `{row['params']}` | {row['cv_average_precision']:.3f} | "
                         f"{(row['cv_auroc'] if row['cv_auroc'] is not None else float('nan')):.3f} |")
        lines += ["", f"Mặc định: CV-AP = {r['default_cv_reference']['mean_average_precision']:.3f} "
                      f"(chênh {r['best_cv_average_precision'] - r['default_cv_reference']['mean_average_precision']:+.3f})",
                  ""]
    (RESULTS_DIR / "tuning.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Lưới rút gọn cho máy yếu.")
    parser.add_argument("--models", default="", help="Danh sách mô hình, phân tách bằng dấu phẩy.")
    args = parser.parse_args(argv)
    print("=== Tinh chỉnh hyperparameter (chia theo công ty) ===")
    run(models=[m for m in args.models.split(",") if m], quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
