"""Đánh giá mô hình cho bài toán MẤT CÂN BẰNG: metric đầy đủ, confusion matrix, bootstrap CI.

QUY TẮC (yêu cầu #3): **Accuracy KHÔNG phải thước đo chính**. Với tỉ lệ 98/2, quy tắc "luôn đoán lớp
đa số" đã đạt ~98% accuracy, nên accuracy chỉ là chỉ số CHẨN ĐOÁN và phải in kèm mốc so sánh
(`accuracy_diagnostic` → `majority_baseline_accuracy_pct`).

Bộ metric chính (`PRIMARY_METRICS`): Precision, Recall, F1 (binary / macro / weighted / **F-beta**),
**PR-AUC (Average Precision)**, ROC-AUC, MCC — kèm **Confusion Matrix** (TN/FP/FN/TP).
Bộ metric chẩn đoán (`DIAGNOSTIC_METRICS`): accuracy + mốc đa số, balanced accuracy, Brier.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from . import config as C

#: Metric CHÍNH — dùng để so sánh/xếp hạng kỹ thuật. Accuracy **không** nằm trong danh sách này.
PRIMARY_METRICS: Tuple[str, ...] = ("precision", "recall", "f1", "macro_f1", "weighted_f1",
                                    "fbeta", "pr_auc", "roc_auc", "balanced_accuracy", "fpr", "mcc")
#: Metric CHẨN ĐOÁN — chỉ in kèm để đọc bối cảnh (không dùng để kết luận/khoe kết quả).
DIAGNOSTIC_METRICS: Tuple[str, ...] = ("accuracy", "majority_baseline_accuracy_pct",
                                       "brier", "n_predicted_positive")
#: Thứ tự cột khi in bảng so sánh (đủ cho yêu cầu #3, KHÔNG có accuracy).
COMPARISON_COLUMNS: Tuple[str, ...] = ("precision", "recall", "f1", "macro_f1", "weighted_f1",
                                       "fbeta", "pr_auc", "roc_auc", "balanced_accuracy", "fpr", "mcc",
                                       "tn", "fp", "fn", "tp")

#: Giữ tên cũ để code hiện có không phải sửa.
METRIC_NAMES: Tuple[str, ...] = ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """(tn, fp, fn, tp) cho nhãn 0/1."""
    matrix = confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred, int), labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def confusion_matrix_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Confusion matrix dạng BẢNG (hàng = thực tế, cột = dự đoán) kèm nhãn dòng/cột."""
    matrix = confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred, int), labels=[0, 1])
    return {"labels": ["không dương", "dương"],
            "matrix": [[int(value) for value in row] for row in matrix.tolist()],
            "counts": confusion_counts(y_true, y_pred)}


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """Accuracy của quy tắc "luôn đoán lớp đa số" — mốc cho thấy Accuracy vô dụng ở đây."""
    y = np.asarray(y_true, int)
    return 100.0 * max(int((y == 1).sum()), int((y == 0).sum())) / len(y) if len(y) else float("nan")


def accuracy_diagnostic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Accuracy kèm MỐC ĐA SỐ + kết luận "có mang thông tin hay không" (chỉ để chẩn đoán)."""
    y = np.asarray(y_true, int)
    accuracy = float(accuracy_score(y, np.asarray(y_pred, int))) if len(y) else float("nan")
    baseline = majority_baseline_accuracy(y)
    informative = bool(np.isfinite(baseline) and accuracy > baseline + 1e-12)
    return {"accuracy": accuracy,
            "majority_baseline_accuracy_pct": baseline,
            "accuracy_pct": 100.0 * accuracy if np.isfinite(accuracy) else float("nan"),
            "accuracy_better_than_majority": informative,
            "note": ("Accuracy KHÔNG dùng làm thước đo chính; chỉ đọc kèm mốc "
                     f"{baseline:.2f}% của quy tắc đoán lớp đa số.")}


def fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = C.FBETA_BETA,
          average: str = "binary") -> float:
    """F-beta score (mặc định beta = 2 ⇒ ưu tiên recall); beta = 1 cho đúng F1."""
    return float(fbeta_score(np.asarray(y_true, int), np.asarray(y_pred, int), beta=float(beta),
                             average=average, zero_division=0))



def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray,
                         threshold: float, beta: float = C.FBETA_BETA) -> Dict[str, Any]:
    """Bộ metric ĐẦY ĐỦ tại một ngưỡng quyết định.

    Metric KHÔNG phụ thuộc ngưỡng: `pr_auc` (Average Precision), `roc_auc`, `brier`.
    Metric phụ thuộc ngưỡng: precision/recall/F1 + biến thể `macro`/`weighted`/**`fbeta`**, MCC,
    balanced accuracy, confusion matrix (`tn/fp/fn/tp` + `confusion_matrix` dạng bảng).
    `accuracy` chỉ là chỉ số CHẨN ĐOÁN (in kèm `majority_baseline_accuracy_pct`).

    Args:
        y_true: nhãn 0/1.
        proba: xác suất lớp dương.
        threshold: ngưỡng quyết định (`proba >= threshold` ⇒ dự đoán dương).
        beta: hệ số F-beta (mặc định lấy từ `config.FBETA_BETA` = 2,0).
    """
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    y_pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    has_two_classes = len(np.unique(y_true)) > 1
    diagnostic = accuracy_diagnostic(y_true, y_pred)
    # FPR (báo động giả) và specificity: FN/FP có giá khác nhau nên phải đọc CẢ HAI phía.
    n_negative = counts["tn"] + counts["fp"]
    specificity = (counts["tn"] / n_negative) if n_negative else float("nan")
    false_positive_rate = (counts["fp"] / n_negative) if n_negative else float("nan")
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted",
                                                    zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fbeta": fbeta(y_true, y_pred, beta),
        "fbeta_beta": float(beta),
        "pr_auc": float(average_precision_score(y_true, proba)) if has_two_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)) if has_two_classes else float("nan"),
        "balanced_accuracy": (float(balanced_accuracy_score(y_true, y_pred))
                              if has_two_classes else float("nan")),
        "fpr": float(false_positive_rate),
        "specificity": float(specificity),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_two_classes else float("nan"),
        "brier": float(np.mean((proba - y_true) ** 2)),
        # --- chỉ số CHẨN ĐOÁN (không phải thước đo chính) ---
        "accuracy": diagnostic["accuracy"],
        "majority_baseline_accuracy_pct": diagnostic["majority_baseline_accuracy_pct"],
        "accuracy_better_than_majority": diagnostic["accuracy_better_than_majority"],
        "n_predicted_positive": int(y_pred.sum()),
        "confusion_matrix": confusion_matrix_table(y_true, y_pred)["matrix"],
        **counts,
    }


def metric_scalar(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                  metric: str = "pr_auc", beta: float = C.FBETA_BETA) -> float:
    """Giá trị của MỘT metric (dùng cho bootstrap): `pr_auc`, `f1`, `macro_f1`, `fbeta`, `recall`."""
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    if metric == "pr_auc":
        return float(average_precision_score(y_true, proba))
    y_pred = (proba >= threshold).astype(int)
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "fbeta":
        return fbeta(y_true, y_pred, beta)
    if metric == "recall":
        return float(recall_score(y_true, y_pred, zero_division=0))
    raise KeyError(f"Metric chưa hỗ trợ bootstrap: {metric!r}; có pr_auc/f1/macro_f1/fbeta/recall")


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                 metric: str = "pr_auc", n_boot: int = 500,
                 seed: int = 42, beta: float = C.FBETA_BETA) -> Dict[str, float]:
    """Khoảng tin cậy 95% (bootstrap theo mẫu) cho `pr_auc`, `f1`, `macro_f1`, `fbeta` hoặc `recall`."""
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        y_b, p_b = y_true[idx], proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        values.append(metric_scalar(y_b, p_b, threshold, metric, beta))
    if not values:
        return {"point": float("nan"), "lo95": float("nan"), "hi95": float("nan"), "n_boot_used": 0,
                "metric": metric}
    arr = np.asarray(values, float)
    point = metric_scalar(y_true, proba, threshold, metric, beta)
    return {"point": float(point), "lo95": float(np.percentile(arr, 2.5)),
            "hi95": float(np.percentile(arr, 97.5)), "n_boot_used": int(len(arr)),
            "metric": metric}
