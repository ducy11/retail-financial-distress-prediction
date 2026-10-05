"""Metrics for imbalanced distress prediction plus F1-based and cost-based threshold search.

A miss is the expensive error, so thresholds are chosen by maximizing F1 or minimizing expected cost
`COST_FN * FN + COST_FP * FP`. Reported metrics always include macro and weighted F1.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix as cm_fn,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import COST_FN, COST_FP, EVAL_THRESHOLDS


def metrics_at_threshold(y_true: Sequence[int], y_prob: Sequence[float],
                         threshold: float) -> Dict[str, Any]:
    """Full metrics at the operating threshold.

    Returns: accuracy, precision, recall, f1, macro_f1, weighted_f1, specificity, npv, mcc,
    tn/fp/fn/tp, n_predicted, expected_cost.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = cm_fn(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "specificity": float(spec),
        "npv": float(npv),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(set(pred.tolist())) > 1 else 0.0,
        "n_predicted": int(pred.sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "expected_cost": float(COST_FN * fn + COST_FP * fp),
    }


def _pr_curve(y_true: np.ndarray, y_prob: np.ndarray):
    """PR curve with index-aligned F1.

    `sklearn.precision_recall_curve` returns precision/recall of length n+1 but thresholds of
    length n; drop the trailing point so every array shares one index (avoids an off-by-one
    lookup when reading back a threshold).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]
    if len(thr) != len(prec):
        raise AssertionError("precision_recall_curve returned mismatched lengths")
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, thr, f1


def best_f1_point(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, float] | None:
    """Max-F1 point on the PR curve (plus its threshold, index-aligned)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return None
    prec, rec, thr, f1 = _pr_curve(y_true, y_prob)
    if not len(f1):
        return None
    i = int(np.argmax(f1))
    return {"threshold": float(thr[i]), "precision": float(prec[i]),
            "recall": float(rec[i]), "f1": float(f1[i])}


def cost_optimal_threshold(y_true: Sequence[int], y_prob: Sequence[float],
                           cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> Dict[str, float]:
    """Threshold minimizing expected cost `cost_fn * FN + cost_fp * FP` (decision theory)."""
    prec, rec, thr, _ = _pr_curve(np.asarray(y_true), np.asarray(y_prob))
    cands = np.unique(np.concatenate([[0.0, 1.0], thr])) if len(thr) else np.array([0.0, 0.5, 1.0])
    rows = [metrics_at_threshold(y_true, y_prob, float(t)) for t in cands]
    best = min(rows, key=lambda r: r["expected_cost"])
    return {"threshold": float(best["threshold"]), "expected_cost": float(best["expected_cost"]),
            "cost_fn": float(cost_fn), "cost_fp": float(cost_fp),
            "f1": float(best["f1"]), "precision": float(best["precision"]),
            "recall": float(best["recall"])}



def evaluate_proba(y_true: np.ndarray, y_prob: np.ndarray,
                   thresholds: Sequence[float] = EVAL_THRESHOLDS,
                   operating_threshold: float | None = None) -> Dict[str, Any]:
    """Score probabilities: overall metrics plus per-threshold metrics plus the operating point.

    `y_prob`: probability of class 1 (distress). `operating_threshold`: the threshold actually
    used to decide (stored under `operating` + `confusion_at_operating` so figures and JSON agree).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(y_true) == 0:
        return {"n": 0, "observed_distress": 0.0}

    metrics: Dict[str, Any] = {
        "n": int(len(y_true)),
        "observed_distress": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "by_threshold": [metrics_at_threshold(y_true, y_prob, float(t)) for t in thresholds],
        "best_f1": best_f1_point(y_true, y_prob),
        "cost_optimal": cost_optimal_threshold(y_true, y_prob),
    }

    c05 = metrics_at_threshold(y_true, y_prob, 0.5)
    metrics["confusion_at_0.5"] = {"tn": c05["tn"], "fp": c05["fp"], "fn": c05["fn"], "tp": c05["tp"]}

    if operating_threshold is not None:
        op = metrics_at_threshold(y_true, y_prob, float(operating_threshold))
        metrics["operating"] = op
        metrics["confusion_at_operating"] = {"tn": op["tn"], "fp": op["fp"],
                                             "fn": op["fn"], "tp": op["tp"]}
    return metrics


def confusion_matrix(y_true, y_pred, labels=(0, 1)) -> np.ndarray:
    """Confusion matrix (rows: actual, cols: predicted) - for reports."""
    return cm_fn(y_true, y_pred, labels=labels)


def threshold_from_validation(val_metrics: Dict[str, Any]) -> float:
    """Pick the threshold from best-F1 on validation (only when both classes exist)."""
    best = val_metrics.get("best_f1")
    if best and best.get("threshold") is not None:
        return float(best["threshold"])
    return 0.5
