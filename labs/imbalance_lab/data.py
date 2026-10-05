"""Generate an imbalanced tabular dataset and split it while preserving class proportions.

The 20% holdout test is used once to report final results and is never resampled; `StratifiedKFold`
runs over the remaining train pool.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from . import config as C


def make_imbalanced_dataset(n_samples: int = C.N_SAMPLES,
                            weights: Tuple[float, float] = C.CLASS_WEIGHTS,
                            n_features: int = C.N_FEATURES,
                            random_state: int = C.SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Build an imbalanced tabular dataset, 98% class 0 and 2% class 1 by default.

    `flip_y=0` leaves labels unflipped so the class ratio matches `weights` exactly, which every
    imbalance-ratio and `scale_pos_weight` computation relies on.

    Returns:
        (X, y): `X` has shape (n_samples, n_features) with dtype float64, `y` holds int 0/1 labels.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=C.N_INFORMATIVE,
        n_redundant=C.N_REDUNDANT,
        n_clusters_per_class=2,
        class_sep=C.CLASS_SEP,
        weights=list(weights),
        flip_y=C.FLIP_Y,
        shuffle=True,
        random_state=random_state,
    )
    return X.astype(np.float64), y.astype(int)


def label_distribution(y: np.ndarray) -> Dict[str, Any]:
    """Label distribution: negative and positive counts, percentage and majority/minority ratio."""
    n = int(len(y))
    positive = int(np.sum(y == 1))
    negative = n - positive
    minority = min(positive, negative)
    majority = max(positive, negative)
    return {
        "n": n,
        "n_negative": negative,
        "n_positive": positive,
        "positive_pct": 100.0 * positive / n if n else float("nan"),
        "imbalance_ratio": (majority / minority) if minority else float("inf"),
    }


def format_distribution(dist: Dict[str, Any]) -> str:
    """Single-line log string: `n=... | negative=... | positive=... (...%) | IR=...`."""
    return (f"n={dist['n']} | negative={dist['n_negative']} | positive={dist['n_positive']} "
            f"({dist['positive_pct']:.2f}%) | IR={dist['imbalance_ratio']:.1f}")


def stratified_holdout_split(X: np.ndarray, y: np.ndarray,
                             test_size: float = C.TEST_SIZE,
                             seed: int = C.SEED) -> Dict[str, np.ndarray]:
    """Split into `train_pool` and `test` stratified by label, keeping class proportions in both."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed, shuffle=True)
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def positive_rate(y: np.ndarray) -> float:
    """Positive-class rate between 0 and 1, used to derive `scale_pos_weight` dynamically."""
    positive = int(np.sum(y == 1))
    return positive / len(y) if len(y) else float("nan")
