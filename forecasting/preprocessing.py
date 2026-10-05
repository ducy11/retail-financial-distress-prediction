"""Heavy-tail preprocessing that always fits on train.

Provides `Winsorizer`, which clips by IQR or percentile using thresholds learned from train, and
`make_scaler` for standard, robust, power, quantile or no scaling. Both are sklearn transformers, so they
run inside `Pipeline` and in cross-validation without leaking validation or test statistics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer, RobustScaler,
                                   StandardScaler)

#: Supported winsorize modes (with descriptions so reports stay readable).
WINSOR_METHODS: Dict[str, str] = {
    "none": "no clipping (keep as-is, for control runs)",
    "iqr": "clip outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (thresholds learned from train)",
    "p1p99": "clip outside [P1, P99] (thresholds learned from train)",
}

#: Supported scalers.
SCALER_KINDS: Dict[str, str] = {
    "none": "no scaling (tree models)",
    "standard": "StandardScaler (mean/std - sensitive to outliers)",
    "robust": "RobustScaler (median/IQR - heavy-tail friendly)",
    "power": "PowerTransformer (Yeo-Johnson - reduces skew)",
    "quantile": "QuantileTransformer (rank to normal distribution)",
}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values outside thresholds learned from train, taming heavy tails before scaling.

    `method="iqr"` uses [Q1 - k*IQR, Q3 + k*IQR]; `method="p1p99"` uses percentiles;
    `method="none"` is an empty transformer (pass-through) so pipelines/grids share one path.
    """

    def __init__(self, method: str = "iqr", iqr_factor: float = 1.5,
                 lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.method = method
        self.iqr_factor = iqr_factor
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Winsorizer":
        """Learn per-column lower/upper bounds from the given `X` (which must be train only)."""
        if self.method not in WINSOR_METHODS:
            raise ValueError(f"Invalid method: {self.method!r}; have {sorted(WINSOR_METHODS)}")
        arr = np.asarray(X, dtype=float)
        n_features = arr.shape[1]
        lower = np.full(n_features, -np.inf)
        upper = np.full(n_features, np.inf)
        if self.method != "none":
            for j in range(n_features):
                column = arr[:, j]
                finite = column[np.isfinite(column)]
                if finite.size == 0:
                    continue
                if self.method == "iqr":
                    q1, q3 = (float(v) for v in np.percentile(finite, [25, 75]))
                    spread = q3 - q1
                    lower[j] = q1 - self.iqr_factor * spread
                    upper[j] = q3 + self.iqr_factor * spread
                else:  # p1p99
                    lower[j] = float(np.percentile(finite, self.lower_pct))
                    upper[j] = float(np.percentile(finite, self.upper_pct))
        self.n_features_in_ = n_features
        self.lower_ = lower
        self.upper_ = upper
        clipped = np.clip(arr, lower, upper)
        self.n_clipped_ = int(np.sum(np.isfinite(arr) & ((arr < lower) | (arr > upper))))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the same train-learned bounds (NaN passes through for `SimpleImputer` later)."""
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def clip_share(self, X: np.ndarray) -> float:
        """Share of values clipped on any set (reports how much was actually touched)."""
        arr = np.asarray(X, dtype=float)
        finite = np.isfinite(arr)
        touched = finite & ((arr < self.lower_) | (arr > self.upper_))
        return float(touched.sum()) / float(finite.sum()) if finite.sum() else 0.0


def make_scaler(kind: str, random_seed: int = 0) -> Any:
    """Return the scaler transformer by name (`kind` in `SCALER_KINDS`); `none` means `"passthrough"`."""
    if kind not in SCALER_KINDS:
        raise ValueError(f"Invalid scaler: {kind!r}; have {sorted(SCALER_KINDS)}")
    if kind == "none":
        return "passthrough"
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler(quantile_range=(25.0, 75.0))
    if kind == "power":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    return QuantileTransformer(output_distribution="normal", n_quantiles=200,
                               random_state=random_seed)
