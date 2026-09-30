"""Kiểm định ý nghĩa thống kê khi so sánh các hệ thống dự báo trên CÙNG một tập test.

Vì sao cần: test chỉ có 64 mẫu nên chênh lệch AUROC 0,983 (mô hình) vs 0,986 (`ticker_prior`) là
**không thể kết luận bằng mắt**. Ở đây cài hai kiểm định chuẩn của tài liệu:

1. `delong_test` — DeLong et al. (1988): so hai đường ROC **tương quan** (cùng mẫu) bằng thống kê z
   từ ma trận hiệp phương sai của placement values. Không cần bootstrap.
2. `paired_bootstrap` — bootstrap theo cặp (resample mẫu, giữ nguyên cặp dự đoán) cho **ΔAP** (và
   ΔAUROC) kèm khoảng tin cậy 95% và p-value hai phía.

Cả hai đều kiểm chứng được: AUROC tính trong module phải khớp `sklearn.roc_auc_score`, và hai hệ
thống giống hệt nhau phải cho p = 1.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score


def _placement_values(y: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Placement values V10 (mẫu dương) và V01 (mẫu âm) — nền tảng của DeLong."""
    positive, negative = prob[y == 1], prob[y == 0]
    v10 = np.array([np.mean((p > negative) + 0.5 * (p == negative)) for p in positive])
    v01 = np.array([np.mean((p < positive) + 0.5 * (p == positive)) for p in negative])
    return v10, v01, float(v10.mean())


def delong_test(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float]
                ) -> Dict[str, Any]:
    """DeLong: so AUROC của hai hệ thống tương quan; trả AUC, z, p-value hai phía và đối chiếu sklearn."""
    y = np.asarray(y_true, dtype=int)
    a, b = np.asarray(prob_a, dtype=float), np.asarray(prob_b, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return {"auc_a": None, "auc_b": None, "delta": None, "z": None, "p_value": None,
                "reason": "cần ≥ 2 mẫu mỗi lớp để tính phương sai"}

    v10_a, v01_a, auc_a = _placement_values(y, a)
    v10_b, v01_b, auc_b = _placement_values(y, b)
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1) / n_pos
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1) / n_neg
    cov = s10 + s01
    variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    sklearn_a, sklearn_b = float(roc_auc_score(y, a)), float(roc_auc_score(y, b))
    base = {"auc_a": auc_a, "auc_b": auc_b, "delta": float(auc_a - auc_b),
            "sklearn_auc_a": sklearn_a, "sklearn_auc_b": sklearn_b,
            "auc_matches_sklearn": bool(abs(auc_a - sklearn_a) < 1e-9 and abs(auc_b - sklearn_b) < 1e-9)}
    if variance <= 0:
        return {**base, "z": None, "p_value": None,
                "reason": "hai hệ thống cho điểm giống nhau (phương sai hiệu = 0)"}
    z = float((auc_a - auc_b) / np.sqrt(variance))
    return {**base, "z": z, "p_value": float(2 * (1 - sps.norm.cdf(abs(z)))),
            "std_err": float(np.sqrt(variance))}


def paired_bootstrap(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float],
                     metric: str = "average_precision", n_boot: int = 2000,
                     seed: int = 42) -> Dict[str, Any]:
    """Bootstrap theo cặp cho Δmetric = metric(a) − metric(b): CI 95% + p-value hai phía.

    Resample **theo lớp** (stratified) để mỗi vòng luôn có cả hai lớp — với n = 64 mẫu, resample
    thuần có thể sinh vòng chỉ có một lớp và làm metric không xác định.
    """
    y = np.asarray(y_true, dtype=int)
    a, b = np.asarray(prob_a, dtype=float), np.asarray(prob_b, dtype=float)
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = (
        average_precision_score if metric == "average_precision" else roc_auc_score)
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas: List[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos_idx, pos_idx.size, replace=True),
                              rng.choice(neg_idx, neg_idx.size, replace=True)])
        deltas.append(float(metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])))
    arr = np.asarray(deltas)
    observed = float(metric_fn(y, a) - metric_fn(y, b))
    lower, upper = (float(v) for v in np.percentile(arr, [2.5, 97.5]))
    p_value = min(1.0, float(2 * min(np.mean(arr <= 0), np.mean(arr >= 0))))
    return {"metric": metric, "delta": observed, "ci95_lower": lower, "ci95_upper": upper,
            "p_value": p_value, "n_boot": int(n_boot),
            "share_same_sign": float(np.mean(np.sign(arr) == np.sign(observed))) if observed else None,
            "significant_5pct": bool(p_value < 0.05)}


def compare_systems(y_true: Sequence[int], systems: Dict[str, Sequence[float]],
                    n_boot: int = 2000, baseline: Optional[str] = None,
                    seed: int = 42) -> Dict[str, Any]:
    """Bảng so sánh mọi cặp hệ thống: DeLong (AUROC) + paired bootstrap (AP và AUROC)."""
    y = np.asarray(y_true, dtype=int)
    names = list(systems)
    metrics = {name: {"auroc": float(roc_auc_score(y, prob)) if len(set(y.tolist())) > 1 else None,
                      "average_precision": float(average_precision_score(y, prob))}
               for name, prob in systems.items()}
    pairs: List[Dict[str, Any]] = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            pairs.append({
                "a": first, "b": second,
                "delong_auroc": delong_test(y, systems[first], systems[second]),
                "bootstrap_ap": paired_bootstrap(y, systems[first], systems[second],
                                                 "average_precision", n_boot=n_boot, seed=seed),
                "bootstrap_auroc": paired_bootstrap(y, systems[first], systems[second], "auroc",
                                                    n_boot=n_boot, seed=seed),
            })
    return {"n_samples": int(len(y)), "n_positive": int((y == 1).sum()), "systems": metrics,
            "pairs": pairs, "baseline": baseline,
            "pairs_vs_baseline": [p for p in pairs
                                  if baseline and baseline in (p["a"], p["b"])]}
