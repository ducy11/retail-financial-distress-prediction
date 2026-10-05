"""Model explanation with SHAP values from a self-implemented KernelSHAP.

Implements the KernelSHAP algorithm of Lundberg and Lee (2017) because the `shap` package is unavailable.
Three self-checks are recorded in the artifact: the efficiency gap, an analytical cross-check on a linear
function, and a ranking cross-check against permutation importance.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: Number of sampled coalitions per explained point (M = 47, so 2^47 cannot be enumerated).
DEFAULT_N_COALITIONS = 200
#: Number of background samples used to approximate E[f | x_S] (more is more stable, slower).
DEFAULT_N_BACKGROUND = 40
#: Weight floor so the design matrix is not singular when C(M, k) is too large.
WEIGHT_FLOOR = 1e-18


def coalition_weight(n_features: int, size: int) -> float:
    """Shapley kernel weight `pi(S) = (M - 1) / (C(M, |S|) * |S| * (M - |S|))`; empty and full coalitions return 0."""
    if size <= 0 or size >= n_features:
        return 0.0
    return (n_features - 1) / (math.comb(n_features, size) * size * (n_features - size))


def sample_coalitions(n_features: int, n_coalitions: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean matrix (n_coalitions x M): each row is a coalition `S` (True = keep x's feature).

    Coalition sizes are drawn uniformly from 1..M-1 then a random subset is taken - the standard
    KernelSHAP sampling when all 2^M coalitions cannot be enumerated.
    """
    if n_features < 2:
        return np.zeros((0, n_features), dtype=bool)
    sizes = rng.integers(1, n_features, size=n_coalitions)
    masks = np.zeros((n_coalitions, n_features), dtype=bool)
    for i, size in enumerate(sizes):
        masks[i, rng.choice(n_features, size=int(size), replace=False)] = True
    return masks


def linear_shap_exact(weights: Sequence[float], x: Sequence[float],
                      background: Sequence[Sequence[float]], bias: float = 0.0
                      ) -> Tuple[float, np.ndarray]:
    """ANALYTICAL Shapley values for a linear function f(x) = w*x + b (used as ground truth).

    phi_0 = w*E[x] + b and phi_j = w_j (x_j - E[x_j]) - exact for any background set and feature count.
    """
    w = np.asarray(weights, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    background = np.asarray(background, dtype=float)
    if background.ndim == 1:
        background = background.reshape(1, -1)
    mean_x = background.mean(axis=0)
    return float(w @ mean_x + bias), w * (x - mean_x)


def pipeline_predict_fn(pipeline: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Positive-class probability prediction function from a fitted sklearn `Pipeline`."""
    def predict(matrix: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(np.asarray(matrix, dtype=float))[:, 1]

    return predict


def _mixed_predictions(predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                       x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """v(S) for every coalition: average f while keeping the features in S, the rest taken from the background."""
    n_masks, n_features = masks.shape
    n_background = background.shape[0]
    mixed = np.where(masks[:, None, :], x[None, None, :], background[None, :, :])
    preds = np.asarray(predict_fn(mixed.reshape(n_masks * n_background, n_features)), dtype=float)
    return preds.reshape(n_masks, n_background).mean(axis=1)


def kernel_shap_values(predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                       x: np.ndarray, n_coalitions: int = DEFAULT_N_COALITIONS,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """SHAP values for ONE point `x`: return `phi` (M,), `base_value`, `prediction`, `efficiency_gap`."""
    background = np.asarray(background, dtype=float)
    if background.ndim == 1:
        background = background.reshape(1, -1)
    x = np.asarray(x, dtype=float).ravel()
    n_features = x.shape[0]
    rng = rng or np.random.default_rng(0)
    base_value = float(np.mean(predict_fn(background)))
    prediction = float(np.asarray(predict_fn(x.reshape(1, -1)), dtype=float).ravel()[0])
    masks = sample_coalitions(n_features, n_coalitions, rng)
    if masks.shape[0] == 0:
        return {"phi": np.zeros(n_features), "base_value": base_value, "prediction": prediction,
                "efficiency_gap": prediction - base_value, "n_coalitions": 0}

    values = _mixed_predictions(predict_fn, background, x, masks)
    weights = np.maximum(np.array([coalition_weight(n_features, int(m.sum())) for m in masks]),
                         WEIGHT_FLOOR)
    sqrt_w = np.sqrt(weights)
    design = masks.astype(float) * sqrt_w[:, None]
    target = (values - base_value) * sqrt_w
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    # KernelSHAP has the efficiency constraint: shift every phi equally so sum(phi) = f(x) - E[f] exactly.
    phi = solution + (prediction - base_value - solution.sum()) / n_features
    return {"phi": phi, "base_value": base_value, "prediction": prediction,
            "efficiency_gap": float(prediction - base_value - phi.sum()),
            "n_coalitions": int(masks.shape[0])}


def kernel_shap_matrix(predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                       X_explain: np.ndarray, n_coalitions: int = DEFAULT_N_COALITIONS,
                       random_state: int = 0) -> Dict[str, Any]:
    """SHAP values for a whole matrix of points to explain (one row per point) + self-check metrics."""
    X_explain = np.asarray(X_explain, dtype=float)
    rng = np.random.default_rng(random_state)
    rows = [kernel_shap_values(predict_fn, background, X_explain[i], n_coalitions=n_coalitions,
                               rng=rng) for i in range(X_explain.shape[0])]
    phi = np.vstack([r["phi"] for r in rows]) if rows else np.zeros((0, X_explain.shape[1]))
    predictions = np.array([r["prediction"] for r in rows])
    gaps = np.array([r["efficiency_gap"] for r in rows])
    return {
        "phi": phi,
        "base_value": float(np.mean([r["base_value"] for r in rows])) if rows else 0.0,
        "predictions": predictions,
        "n_coalitions": int(n_coalitions),
        "max_abs_efficiency_gap": float(np.max(np.abs(gaps))) if gaps.size else 0.0,
        "relative_efficiency_gap": (float(np.max(np.abs(gaps) / np.maximum(np.abs(predictions), 1e-9)))
                                    if gaps.size else 0.0),
        "n_explained": int(phi.shape[0]),
    }


def mean_abs_shap(phi: np.ndarray) -> np.ndarray:
    """Global importance: mean |phi_j| over the explained points (same as the `shap` summary)."""
    return np.mean(np.abs(np.asarray(phi, dtype=float)), axis=0)


def permutation_importance_ranking(pipeline: Any, X: np.ndarray, y: np.ndarray,
                                   n_repeats: int = 5, random_state: int = 0) -> Optional[np.ndarray]:
    """Delta AUROC when permuting each column (independent cross-check with SHAP); `None` if a single class."""
    from sklearn.inspection import permutation_importance

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    result = permutation_importance(pipeline, np.asarray(X, dtype=float), y, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=random_state)
    return np.asarray(result.importances_mean, dtype=float)


def rank_agreement(a: np.ndarray, b: np.ndarray, top: int = 15) -> Dict[str, Any]:
    """Agreement between two feature rankings: Spearman + Jaccard of the top-k."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 3:
        return {"spearman": None, "top_overlap": None}
    spearman = float(spearmanr(a, b).statistic)
    top_a, top_b = set(np.argsort(-a)[:top].tolist()), set(np.argsort(-b)[:top].tolist())
    return {"spearman": spearman,
            "top_overlap": len(top_a & top_b) / max(1, len(top_a | top_b)),
            "top_k": int(min(top, a.size)),
            "agreed_features": sorted(top_a & top_b)}
