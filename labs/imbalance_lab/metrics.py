"""Model evaluation for imbalanced classification: metrics, confusion matrix and bootstrap CIs.

Accuracy is not the primary measure, so it is reported as a diagnostic next to the majority-class
baseline. Primary metrics are precision, recall, the F1 variants, PR-AUC, ROC-AUC and MCC, plus the
confusion matrix.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from . import config as C

#: Primary metrics used to compare and rank techniques; accuracy is deliberately excluded.
PRIMARY_METRICS: Tuple[str, ...] = ("precision", "recall", "f1", "macro_f1", "weighted_f1",
                                    "fbeta", "pr_auc", "roc_auc", "balanced_accuracy", "fpr", "mcc")
#: Diagnostic metrics, reported only for context and never used to draw conclusions.
DIAGNOSTIC_METRICS: Tuple[str, ...] = ("accuracy", "majority_baseline_accuracy_pct",
                                       "brier", "n_predicted_positive")
#: Column order for the comparison table; accuracy is excluded on purpose.
COMPARISON_COLUMNS: Tuple[str, ...] = ("precision", "recall", "f1", "macro_f1", "weighted_f1",
                                       "fbeta", "pr_auc", "roc_auc", "balanced_accuracy", "fpr", "mcc",
                                       "tn", "fp", "fn", "tp")

#: Retained for backward compatibility with existing callers.
METRIC_NAMES: Tuple[str, ...] = ("precision", "recall", "f1", "pr_auc", "roc_auc", "mcc")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Return (tn, fp, fn, tp) for 0/1 labels."""
    matrix = confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred, int), labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def confusion_matrix_table(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Confusion matrix as a table with rows for actual labels and columns for predicted, plus labels."""
    matrix = confusion_matrix(np.asarray(y_true, int), np.asarray(y_pred, int), labels=[0, 1])
    return {"labels": ["negative", "positive"],
            "matrix": [[int(value) for value in row] for row in matrix.tolist()],
            "counts": confusion_counts(y_true, y_pred)}


def majority_baseline_accuracy(y_true: np.ndarray) -> float:
    """Accuracy of the always-predict-the-majority-class rule, showing how little accuracy is worth here."""
    y = np.asarray(y_true, int)
    return 100.0 * max(int((y == 1).sum()), int((y == 0).sum())) / len(y) if len(y) else float("nan")


def accuracy_diagnostic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Accuracy with the majority baseline and a flag for whether it carries information; diagnostic only."""
    y = np.asarray(y_true, int)
    accuracy = float(accuracy_score(y, np.asarray(y_pred, int))) if len(y) else float("nan")
    baseline = majority_baseline_accuracy(y)
    informative = bool(np.isfinite(baseline) and accuracy > baseline + 1e-12)
    return {"accuracy": accuracy,
            "majority_baseline_accuracy_pct": baseline,
            "accuracy_pct": 100.0 * accuracy if np.isfinite(accuracy) else float("nan"),
            "accuracy_better_than_majority": informative,
            "note": ("Accuracy is not the primary measure; read it only against the majority-class "
                     f"baseline of {baseline:.2f}%.")}


def fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = C.FBETA_BETA,
          average: str = "binary") -> float:
    """F-beta score, with beta = 2 by default to favor recall; beta = 1 gives F1."""
    return float(fbeta_score(np.asarray(y_true, int), np.asarray(y_pred, int), beta=float(beta),
                             average=average, zero_division=0))



def metrics_at_threshold(y_true: np.ndarray, proba: np.ndarray,
                         threshold: float, beta: float = C.FBETA_BETA) -> Dict[str, Any]:
    """Full metric set at one decision threshold.

    Threshold-free metrics: `pr_auc` from average precision, `roc_auc` and `brier`.
    Threshold-dependent metrics: precision, recall, F1 and its macro, weighted and fbeta variants, MCC,
    balanced accuracy and the confusion matrix (tn/fp/fn/tp plus `confusion_matrix` as a table).
    `accuracy` is a diagnostic only, reported alongside `majority_baseline_accuracy_pct`.

    Args:
        y_true: 0/1 labels.
        proba: positive-class probability.
        threshold: decision threshold; `proba >= threshold` predicts positive.
        beta: F-beta coefficient, taken from `config.FBETA_BETA` (2.0) by default.
    """
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    y_pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    has_two_classes = len(np.unique(y_true)) > 1
    diagnostic = accuracy_diagnostic(y_true, y_pred)
    # Report both false-positive rate and specificity because FN and FP carry different costs.
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
        # Diagnostic metrics, not the primary measures.
        "accuracy": diagnostic["accuracy"],
        "majority_baseline_accuracy_pct": diagnostic["majority_baseline_accuracy_pct"],
        "accuracy_better_than_majority": diagnostic["accuracy_better_than_majority"],
        "n_predicted_positive": int(y_pred.sum()),
        "confusion_matrix": confusion_matrix_table(y_true, y_pred)["matrix"],
        **counts,
    }


def metric_scalar(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                  metric: str = "pr_auc", beta: float = C.FBETA_BETA) -> float:
    """Value of a single metric for the bootstrap: pr_auc, f1, macro_f1, fbeta or recall."""
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
    raise KeyError(f"Metric not supported for bootstrap: {metric!r}; have pr_auc/f1/macro_f1/fbeta/recall")


def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, threshold: float,
                 metric: str = "pr_auc", n_boot: int = 500,
                 seed: int = 42, beta: float = C.FBETA_BETA) -> Dict[str, float]:
    """95% confidence interval by sample bootstrap for pr_auc, f1, macro_f1, fbeta or recall."""
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
