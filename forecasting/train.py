"""Huấn luyện các model ứng viên trên train split, chọn mô hình theo AP CROSS-COMPANY.

Lệnh: python -m forecasting.train [--model logistic|random_forest|hist_gradient_boosting]

- Nạp prepared/train.json, validation.json.
- Xây features (bậc 2) từ lịch sử; impute median trong pipeline.
- Fit từng model; đánh giá trên validation (AUROC, AP, F1, threshold best-F1) **và** tính AP
  out-of-fold khi chia theo CÔNG TY (`cross_company_metrics`, GroupKFold trên train+validation).
- Chọn mô hình theo quy tắc: **AP cross-company → best-F1(val) → AP(val) → AUROC(val) → gap nhỏ nhất**
  (vì metric in-domain bị "nhớ mặt công ty" chi phối — xem `forecasting/validation.py`).
- Lưu: reports/results/summary.json (mọi model) + reports/models/best.joblib (mô hình chọn)
  + reports/figures/validation_pr_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import (FIGURES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, ensure_dirs,
                     ensure_utf8_stdio)
from .data_loader import load_prepared
from .evaluation import (best_f1_point, cost_optimal_threshold, evaluate_proba,
                         metrics_at_threshold, threshold_from_validation)
from .features import build_feature_matrix, extract_labels, feature_names
from .models import DEFAULT_MODEL_ORDER, MODEL_REGISTRY, make_model, predict_proba
from .validation import bootstrap_ci

#: Mô hình mặc định = 3 họ mô hình của đồ án (logistic · random forest · hist gradient boosting).
DEFAULT_MODELS = [m for m in DEFAULT_MODEL_ORDER if m in MODEL_REGISTRY]


def _safe_metric(x) -> float:
    """Chuyển metric (có thể None/NaN) về float; None/NaN → -1 để xếp hạng an toàn."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if v != v else v  # v != v khi NaN


def _fmt(value: Any, digits: int = 3) -> str:
    """Định dạng số cho log (None/NaN → '—')."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if number != number else f"{number:.{digits}f}"


def _rank(row: Dict[str, Any]):
    """Khoá xếp hạng mô hình: AP cross-company (GroupKFold) → best-F1(val) → AP(val) → AUROC(val)
    → gap overfit nhỏ nhất.

    Vì sao AP cross-company đứng đầu: metric in-domain bị chi phối bởi "nhớ mặt công ty"
    (baseline ticker-prior = 0.986), nên chọn mô hình theo in-domain là chọn sai câu hỏi.
    Xem `docs/ke-hoach-tiep-theo.md` (P0-#2) và mục 6.4 của báo cáo.
    """
    return (_safe_metric(row.get("cross_company_ap")), _safe_metric(row["best_f1_val"]),
            _safe_metric(row["average_precision"]), _safe_metric(row["auroc"]),
            -_safe_metric(row["overfit_gap_f1"]))


def train_split(model_name: str, X_train, y_train, X_val, y_val, **params):
    """Fit một mô hình; trả (row metric, pipeline).

    Row gồm metric TRAIN và VALIDATION ở cùng threshold 0.5 (để so overfitting), điểm best-F1
    trên validation và ngưỡng tối ưu chi phí trên validation. `params` ghi đè hyperparameter.
    """
    model = make_model(model_name, **params)
    model.fit(X_train, y_train)
    proba_tr = predict_proba(model, X_train)
    proba_val = predict_proba(model, X_val)
    tr = metrics_at_threshold(y_train, proba_tr, 0.5)
    va = metrics_at_threshold(y_val, proba_val, 0.5)
    val_full = evaluate_proba(y_val, proba_val)
    best = best_f1_point(y_val, proba_val) or {"threshold": 0.5, "f1": float("nan"),
                                               "precision": float("nan"), "recall": float("nan")}
    train_auroc = (float(roc_auc_score(y_train, proba_tr))
                   if len(set(np.asarray(y_train).tolist())) > 1 else float("nan"))
    keys_tr = ("accuracy", "precision", "recall", "f1", "macro_f1")
    keys_va = ("accuracy", "precision", "recall", "f1", "macro_f1", "specificity", "mcc")
    return {
        "model": model_name,
        "params": dict(params) if params else None,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "prevalence_val": float(np.mean(y_val)),
        "auroc": val_full["auroc"],
        "average_precision": val_full["average_precision"],
        "brier_val": val_full["brier"],
        "train_auroc": train_auroc,
        "train_at_0.5": {k: tr[k] for k in keys_tr},
        "val_at_0.5": {k: va[k] for k in keys_va},
        "overfit_gap_f1": float(tr["f1"] - va["f1"]),
        "overfit_gap_auroc": float(train_auroc - val_full["auroc"]),
        "best_f1_val": best["f1"],
        "best_threshold": best["threshold"],
        "precision_at_best": best["precision"],
        "recall_at_best": best["recall"],
        "cost_optimal_val": cost_optimal_threshold(y_val, proba_val),
    }, model


def cross_company_metrics(model_name: str, samples: List[Dict[str, Any]],
                          n_splits: int = 4) -> Dict[str, Any]:
    """AP/AUROC **out-of-fold khi chia theo CÔNG TY** (GroupKFold) — tiêu chí chọn mô hình chính.

    Vì sao cần: bộ test in-domain chứa đúng các công ty trong train nên mọi metric in-domain đều bị
    "nhớ mặt công ty" chi phối (baseline `ticker_prior` = 0.986, xem `forecasting/validation.py`).
    Chọn mô hình theo AP cross-company mới đo đúng câu hỏi "công ty chưa từng thấy thì sao".

    Dữ liệu dùng để chọn = train + validation (không chạm test); mỗi fold GroupKFold giữ TRỌN một
    công ty ra ngoài nên không có nhãn của công ty đó trong phần fit.
    """
    from .validation import grouped_cv

    result = grouped_cv([model_name], samples, n_splits=n_splits)
    block = result.get(model_name) or {}
    return {
        "cross_company_ap": block.get("oof_average_precision"),
        "cross_company_auroc": block.get("oof_auroc"),
        "cross_company_n_oof": block.get("n_oof"),
        "cross_company_n_splits": result.get("n_splits"),
    }


def run(model_names: List[str] | None = None) -> Dict[str, Any]:
    """Huấn luyện các mô hình ứng viên trên train, chọn mô hình trên validation."""
    ensure_dirs()
    model_names = list(model_names or DEFAULT_MODELS)
    train_samples = load_prepared("train")
    val_samples = load_prepared("validation")

    X_train = build_feature_matrix(train_samples)
    y_train = extract_labels(train_samples)
    X_val = build_feature_matrix(val_samples)
    y_val = extract_labels(val_samples)

    feats = feature_names()
    print(f"Train {X_train.shape}, Validation {X_val.shape}; n_features={len(feats)}")

    summaries: List[Dict[str, Any]] = []
    best_name, best_row, best_model = None, None, None
    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"  (bỏ qua) model không có trong registry: {name}")
            continue
        print(f"  Fit {name} ...")
        try:
            row, model = train_split(name, X_train, y_train, X_val, y_val)
        except Exception as e:  # noqa: BLE001 - lỗi fit của sklearn (dữ liệu/quá ít mẫu)
            print(f"    LỖI fit {name}: {e}. Bỏ qua model này.")
            continue
        row["n_features"] = len(feats)
        # Tiêu chí chọn mô hình chính: AP out-of-fold theo CÔNG TY (train+validation, không chạm test)
        row.update(cross_company_metrics(name, train_samples + val_samples, n_splits=4))
        summaries.append(row)
        print(f"    train: AUROC={row['train_auroc']:.3f} F1@0.5={row['train_at_0.5']['f1']:.3f}"
              f" | val: AUROC={row['auroc']:.3f} AP={row['average_precision']:.3f} "
              f"F1@0.5={row['val_at_0.5']['f1']:.3f} bestF1@{row['best_threshold']:.2f}="
              f"{row['best_f1_val']:.3f} | gapF1={row['overfit_gap_f1']:+.3f}"
              f" | cross-company AP={_fmt(row.get('cross_company_ap'))}"
              f" AUROC={_fmt(row.get('cross_company_auroc'))}")
        if best_row is None or _rank(row) > _rank(best_row):
            best_name, best_row, best_model = name, row, model

    summary = {
        "seed": RANDOM_SEED,
        "n_features": len(feats),
        "feature_names": feats,
        "models": summaries,
        "best_model": best_name,
        "best_threshold": threshold_from_validation({
            "best_f1": {
                "threshold": best_row["best_threshold"],
                "f1": best_row["best_f1_val"],
            }}),
        "best_threshold_cost_optimal": best_row["cost_optimal_val"]["threshold"],
        "selection_rule": ("max AP cross-company (GroupKFold, train+validation) → best-F1(val) → "
                           "AP(val) → AUROC(val) → gap overfit nhỏ nhất"),
        "bootstrap_val_best": (bootstrap_ci(y_val, predict_proba(best_model, X_val))
                               if best_model is not None else None),
        "trained_at": None,  # không in timestamp để output ổn định/diff được
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    if best_model is not None:
        joblib.dump({"model": best_model, "name": best_name, "features": feats,
                     "threshold": float(summary["best_threshold"])},
                    MODELS_DIR / "best.joblib")

    # Biểu đồ PR validation cho từng model (fit lại từ cùng tham số cố định)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plotted = False
        for row in summaries:
            try:
                pipe = make_model(row["model"])
                if row["model"] == best_name:
                    pipe = best_model
                else:
                    pipe.fit(X_train, y_train)
                proba_val = predict_proba(pipe, X_val)
            except Exception as e:  # noqa: BLE001
                print(f"  (bỏ qua) vẽ {row['model']}: {e}")
                continue
            prec, rec, _ = precision_recall_curve(y_val, proba_val)
            plt.plot(rec, prec, label=row["model"])
            plotted = True

        if plotted:
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall trên validation")
            plt.legend()
            plt.grid(alpha=0.3)
            from .config import FIGURES_DIR
            plt.savefig(FIGURES_DIR / "validation_pr_curves.png", dpi=120)
        plt.close()
    except Exception as e:  # pragma: no cover
        print("Bỏ qua biểu đồ PR:", e)

    return summary


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[],
                        help="Chỉ huấn luyện các model này (mặc định: tất cả).")
    args = parser.parse_args(argv)
    models = args.model or DEFAULT_MODELS
    summary = run(models)
    print(f"\nXong. Best model: {summary['best_model']} — "
          f"dùng threshold {summary['best_threshold']:.2f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
