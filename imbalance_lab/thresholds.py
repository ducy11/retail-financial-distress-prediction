"""Tìm NGƯỠNG QUYẾT ĐỊNH tối ưu thay vì cố định 0.5.

Ba chế độ:
- `best_f1`        — ngưỡng đạt F1 cao nhất;
- `best_cost`      — ngưỡng tối thiểu chi phí kỳ vọng `FN·COST_FN + FP·COST_FP`;
- `min_precision`  — ngưỡng nhỏ nhất đạt precision mục tiêu (giữ recall cao nhất có thể).

QUAN TRỌNG (chống rò rỉ): ngưỡng chỉ được chọn trên xác suất **out-of-fold** của tập train; không
bao giờ chọn trên validation/test rồi báo cáo trên chính tập đó.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def candidate_thresholds(proba: np.ndarray, n_grid: int = 1001) -> np.ndarray:
    """Lưới ngưỡng ứng viên: lượng tử của phân phối xác suất + 0.5."""
    proba = np.asarray(proba, float)
    grid = np.unique(np.quantile(proba, np.linspace(0.0, 1.0, n_grid)))
    return np.unique(np.concatenate([grid, np.array([0.0, 0.5, 1.0])]))


def scan_counts(y_true: np.ndarray, proba: np.ndarray,
                thresholds: np.ndarray) -> Dict[str, np.ndarray]:
    """Đếm (tp, fp, fn) cho MỌI ngưỡng cùng lúc (O(n log n + m log n)).

    Lưu ý kỹ thuật: `np.searchsorted` yêu cầu mảng ĐI LÊN, nên phải đảo dấu (mảng xác suất sắp
    giảm dần ⇒ `-p_desc` tăng dần).
    """
    y_true = np.asarray(y_true, int)
    proba = np.asarray(proba, float)
    order = np.argsort(-proba, kind="mergesort")          # thứ tự xác suất giảm dần
    cumulative_positive = np.cumsum(y_true[order])        # số dương trong top-k
    neg_ascending = -np.sort(proba)[::-1]                 # -p_desc là mảng TĂNG để searchsorted
    k = np.searchsorted(neg_ascending, -thresholds, side="right")  # số mẫu có proba >= ngưỡng
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
    """Ngưỡng tối đa hoá F1 (chọn trên xác suất OOF)."""
    thresholds = candidate_thresholds(proba, n_grid)
    counts = scan_counts(y_true, proba, thresholds)
    return _result(thresholds, counts, int(np.argmax(_f1(counts))))


def best_cost_threshold(y_true: np.ndarray, proba: np.ndarray, cost_fn: float,
                        cost_fp: float, n_grid: int = 1001) -> Dict[str, Any]:
    """Ngưỡng tối thiểu chi phí kỳ vọng `FN·cost_fn + FP·cost_fp`."""
    thresholds = candidate_thresholds(proba, n_grid)
    counts = scan_counts(y_true, proba, thresholds)
    cost = counts["fn"] * float(cost_fn) + counts["fp"] * float(cost_fp)
    index = int(np.argmin(cost))
    return _result(thresholds, counts, index,
                   {"expected_cost": float(cost[index]), "cost_fn": float(cost_fn),
                    "cost_fp": float(cost_fp)})


def threshold_for_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float,
                            n_grid: int = 1001) -> Dict[str, Any]:
    """Ngưỡng NHỎ NHẤT đạt `precision >= target_precision` (ưu tiên recall cao)."""
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
    """Gọi cả 3 chế độ và trả dict — dùng một lần trong `cv.py` để chọn ngưỡng."""

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


# ---------------------------------------------------------------------------
# Ngưỡng theo ĐƯỜNG CONG PRECISION-RECALL (yêu cầu #2: "dựa trên PR curve thay vì 0.5")
# ---------------------------------------------------------------------------
def pr_curve_points(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, np.ndarray]:
    """Đường cong Precision-Recall (đúng như `sklearn.metrics.precision_recall_curve`).

    Trả `precision`, `recall` (độ dài n+1) và `thresholds` (độ dài n) + `pr_auc` (Average Precision).
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
    """Chọn ngưỡng trên các ĐIỂM CỦA ĐƯỜNG PR (thay vì lưới lượng tử) — 3 chế độ như `tune_thresholds`.

    Vì sao: điểm vận hành tốt chỉ nằm trên đường PR; lấy ứng viên từ chính đường PR nên mỗi ngưỡng
    đều là một điểm thực của đường cong (không phải điểm nội suy), và số ứng viên = n mẫu (nhỏ hơn lưới
    1001 nhưng vẫn phủ đủ mọi "bước nhảy" precision/recall).

    `y_true/proba` phải là xác suất OUT-OF-FOLD của train (chống rò rỉ): không truyền test vào đây.
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
