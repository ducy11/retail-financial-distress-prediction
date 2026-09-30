"""Sinh dữ liệu bảng mất cân bằng và chia tập GIỮ NGUYÊN tỉ lệ lớp (stratified).

Luồng chuẩn (không rò rỉ):
1. `make_imbalanced_dataset()` — dữ liệu giả lập 98/2 bằng `make_classification`.
2. `stratified_holdout_split()` — tách holdout test 20% (chỉ dùng để CHỐT kết quả, không resample).
3. `StratifiedKFold` trên phần train (xem `cv.py`).
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
    """Sinh dữ liệu tabular mất cân bằng (mặc định 98% lớp 0 / 2% lớp 1).

    `flip_y=0` để không đảo nhãn ⇒ tỉ lệ lớp đúng như `weights` (cần cho mọi tính toán
    imbalance ratio và `scale_pos_weight`).

    Returns:
        (X, y): `X` shape (n_samples, n_features) kiểu float64, `y` kiểu int 0/1.
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
    """Phân phối nhãn: số âm/dương, tỉ lệ %, imbalance ratio (đa số/thiểu số)."""
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
    """Chuỗi một dòng để log: `n=… | âm=… | dương=… (…%) | IR=…`."""
    return (f"n={dist['n']} | âm={dist['n_negative']} | dương={dist['n_positive']} "
            f"({dist['positive_pct']:.2f}%) | IR={dist['imbalance_ratio']:.1f}")


def stratified_holdout_split(X: np.ndarray, y: np.ndarray,
                             test_size: float = C.TEST_SIZE,
                             seed: int = C.SEED) -> Dict[str, np.ndarray]:
    """Chia stratified thành `train_pool` / `test` (giữ nguyên tỉ lệ lớp ở cả hai tập)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed, shuffle=True)
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def positive_rate(y: np.ndarray) -> float:
    """Tỉ lệ lớp dương (0..1) — dùng để tính `scale_pos_weight` động."""
    positive = int(np.sum(y == 1))
    return positive / len(y) if len(y) else float("nan")
