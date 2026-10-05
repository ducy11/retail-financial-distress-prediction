"""Search for an optimal decision threshold instead of fixing it at 0.5.

Three modes are supported: `best_f1`, `best_cost` on expected FN/FP cost, and `min_precision`. Thresholds
are selected only on out-of-fold probabilities from the train split, never on validation or test.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def candidate_thresholds(proba: np.ndarray, n_grid: int = 1001) -> np.ndarray:
    """Candidate threshold grid: quantiles of the probability distribution plus 0.5."""
    proba = np.asarray(proba, float)
    grid = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, n_grid)))
    return np.unique(np.concatenate([grid, np.array([0.0, 0.5, 1.0])]))


def scan_counts(y_true: np.ndarray, proba: np.ndarray,
                thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """Count (tp, fp, fn) for every threshold at once, in O(n log n + m log n).

    `np.searchsorted` requires an ascending array, so the descending probability array is negated to
    make it ascending.
    """
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    order = np.argsort(-proba, kind="mergesort")          # order by descending probability
    cumulative_positive = np.cumsum(y_true[order])        # positive count within the top-k
    neg_ascending = -np.sort(proba)[::-1]                 # -p_desc is ascending for searchsorted
    k = np.searchsorted(neg_ascending, -thresholds, side="right")  # samples with proba >= threshold
    k = np.clip(k, 0, len(y_true))
    tp = np.where(k > 0, cumulative_positive[np.maximum(k - 1, 0)], 0.0).astype(float)
    fp = k - tp
    total_positive = float((y_true == 1).sum())
    return {"tp": tp, "fp": fp, "fn": total_positive - tp}


def _f1(counts: Dict[str, np.ndarray]) -> np.ndarray:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    denominator = 2.0 * tp + fp + fn
    return np.where(denominator > 0, 2.0 * tp / np.maximum(denominator, 1e-12), 0.0)


def _precision(counts: Dict[str, np.ndarray]) -> np.ndarray:
    tp, fp = counts["tp"], counts["fp"]
    return np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)


def _recall(counts: Dict[str, np.ndarray]) -> np.ndarray:
    tp, fn = counts["tp"], counts["fn"]
    return np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)


def _result(thresholds: np.ndarray, counts: Dict[str, np.ndarray], index: int,
            extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    precision = float(_precision(counts)[index])
    recall = float(_recall(counts)[index])
    return {"threshold": float(thresholds[index]), "precision": precision, "recall": recall,
            "f1": float(_f1(counts)[index]),
            "tp": int(counts["tp"][index]), "fp": int(counts["fp"][index]),
            "fn": int(counts["fn"][index]), **(extra or {})}


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray,
                      n_grid: int = 1001) -> Dict[str, Any]:
    """Threshold that maximizes F1, selected on out-of-fold probabilities."""
    thresholds = candidate_thresholds(proba, n_grid)
    counts = scan_counts(y_true, proba, thresholds)
    return _result(thresholds, counts, int(np.argmax(_f1(counts))))


def best_cost_threshold(y_true: np.ndarray, proba: np.ndarray, cost_fn: float,
                        cost_fp: float, n_grid: int = 1001) -> Dict[str, Any]:
    """Threshold that minimizes expected cost `FN*cost_fn + FP*cost_fp`."""
    thresholds = candidate_thresholds(proba, n_grid)
    counts = scan_counts(y_true, proba, thresholds)
    cost = counts["fn"] * float(cost_fn) + counts["fp"] * float(cost_fp)
    index = int(np.argmin(cost))
    return _result(thresholds, counts, index,
                   {"expected_cost": float(cost[index]), "cost_fn": float(cost_fn),
                    "cost_fp": float(cost_fp)})


def threshold_for_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float,
                            n_grid: int = 1001) -> Dict[str, Any]:
    """Lowest threshold that reaches `precision >= target_precision`, prioritizing recall."""
    thresholds = candidate_thresholds(proba, n_grid)
    counts = scan_counts(y_true, proba, thresholds)
    ok = np.flatnonzero(_precision(counts) >= float(target_precision))
    if len(ok) == 0:
        return _result(thresholds, counts, int(np.argmax(_precision(counts))),
                       {"target_precision": float(target_precision), "target_met": False})
    index = int(ok[np.argmax(_recall(counts)[ok])])
    return _result(thresholds, counts, index,
                   {"target_precision": float(target_precision), "target_met": True})


def tune_thresholds(y_true: np.ndarray, proba: np.ndarray, *, cost_fn: float, cost_fp: float,
                    precision_target: float, n_grid: int = 1001) -> Dict[str, Dict[str, Any]]:
    """Run all three modes and return a dict; called once in `cv.py` to pick the threshold."""

    def _at(threshold: float) -> Dict[str, Any]:
        thresholds = np.array([threshold])
        return _result(thresholds, scan_counts(y_true, proba, thresholds), 0)

    tuned = {
        "best_f1": best_f1_threshold(y_true, proba, n_grid),
        "best_cost": best_cost_threshold(y_true, proba, cost_fn, cost_fp, n_grid),
        "min_precision": threshold_for_precision(y_true, proba, precision_target, n_grid),
        "fixed_0.5": _at(0.5),
    }
    return tuned


# Thresholds from the precision-recall curve, as required instead of a fixed 0.5.
def pr_curve_points(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Precision-recall curve, matching `sklearn.metrics.precision_recall_curve`.

    Returns `precision` and `recall` of length n+1, `thresholds` of length n, and `pr_auc` from average
    precision.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y = np.asarray(y_true, int)
    p = np.asarray(proba, float)
    precision, recall, thresholds = precision_recall_curve(y, p)
    return {"precision": precision, "recall": recall, "thresholds": thresholds,
            "pr_auc": float(average_precision_score(y, p))}


def tune_thresholds_from_pr_curve(y_true: np.ndarray, proba: np.ndarray, *,
                                  cost_fn: float, cost_fp: float, precision_target: float,
                                  ) -> Dict[str, Dict[str, Any]]:
    """Pick thresholds from the points on the PR curve instead of a quantile grid, with the same three modes.

    A good operating point lies on the PR curve, so taking candidates from the curve itself means every
    threshold is a real curve point rather than an interpolated one, and the candidate count equals the
    sample count. That is smaller than the 1001-point grid yet still covers every precision/recall jump.

    `y_true` and `proba` must be out-of-fold train probabilities; never pass test data here.
    """
    y = np.asarray(y_true, int)
    p = np.asarray(proba, float)
    curve = pr_curve_points(y, p)
    edges = np.unique(np.concatenate([curve["thresholds"], np.array([0.5, 1.0])]))
    counts = scan_counts(y, p, edges)

    f1 = _f1(counts)
    cost = counts["fn"] * float(cost_fn) + counts["fp"] * float(cost_fp)
    precision = _precision(counts)
    recall = _recall(counts)

    best_f1 = _result(edges, counts, int(np.argmax(f1)), {"pr_auc": curve["pr_auc"]})
    best_cost = _result(edges, counts, int(np.argmin(cost)),
                        {"expected_cost": float(np.min(cost)), "cost_fn": float(cost_fn),
                         "cost_fp": float(cost_fp)})
    feasible = np.flatnonzero(precision >= float(precision_target))
    if len(feasible):
        index = int(feasible[np.argmax(recall[feasible])])
        min_precision = _result(edges, counts, index,
                                {"target_precision": float(precision_target), "target_met": True})
    else:
        min_precision = _result(edges, counts, int(np.argmax(precision)),
                                {"target_precision": float(precision_target), "target_met": False})
    at_half = _result(np.array([0.5]), scan_counts(y, p, np.array([0.5])), 0)
    return {"best_f1": best_f1, "best_cost": best_cost, "min_precision": min_precision,
            "fixed_0.5": at_half, "pr_auc": curve["pr_auc"],
            "n_candidates": int(len(edges))}
