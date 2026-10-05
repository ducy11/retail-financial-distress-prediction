"""Statistical significance tests for comparing forecasting systems on the same test set.

With only 64 test samples, small AUROC gaps cannot be judged by eye, so `delong_test` compares two
correlated ROC curves and `paired_bootstrap` gives a confidence interval and p-value for the delta in
average precision and AUROC. Two identical systems must yield p = 1.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score


def _placement_values(y: np.ndarray, prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Placement values V10 (positive samples) and V01 (negative samples) - the basis of DeLong."""
    positive, negative = prob[y == 1], prob[y == 0]
    v10 = np.array([np.mean((p > negative) + 0.5 * (p == negative)) for p in positive])
    v01 = np.array([np.mean((p < positive) + 0.5 * (p == positive)) for p in negative])
    return v10, v01, float(v10.mean())


def delong_test(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float]
                ) -> Dict[str, Any]:
    """DeLong: compare the AUROC of two correlated systems; return AUC, z, two-sided p-value and a sklearn check."""
    y = np.asarray(y_true, dtype=int)
    a, b = np.asarray(prob_a, dtype=float), np.asarray(prob_b, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return {"auc_a": None, "auc_b": None, "delta": None, "z": None, "p_value": None,
                "reason": "need >= 2 samples per class to estimate the variance"}

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
                "reason": "both systems give identical scores (variance of the difference = 0)"}
    z = float((auc_a - auc_b) / np.sqrt(variance))
    return {**base, "z": z, "p_value": float(2 * (1 - sps.norm.cdf(abs(z)))),
            "std_err": float(np.sqrt(variance))}


def paired_bootstrap(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float],
                     metric: str = "average_precision", n_boot: int = 2000,
                     seed: int = 42) -> Dict[str, Any]:
    """Paired bootstrap for delta metric = metric(a) - metric(b): 95% CI + two-sided p-value.

    Resampling is stratified by class so every round keeps both classes; with n = 64 samples a plain
    resample can produce a round with a single class and an undefined metric.
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
    """Comparison table across all system pairs: DeLong (AUROC) + paired bootstrap (AP and AUROC)."""
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
