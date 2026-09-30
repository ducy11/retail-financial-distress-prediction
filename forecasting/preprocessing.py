"""Tiền xử lý chịu ĐUÔI NẶNG: winsorize (clip) + lựa chọn scaler — luôn fit TRÊN TRAIN.

Vì sao cần: `scripts/eda.py` đo được 6/14 tỷ số có |skew| > 1 và `debt_to_equity` có skew −14,9,
**30,6%** giá trị nằm ngoài khoảng IQR. `StandardScaler` lấy mean/std nên chính các giá trị cực trị
này quyết định tỉ lệ scale ⇒ mô hình tuyến tính bị kéo lệch. Ở đây bổ sung:

- `Winsorizer` — clip theo IQR (mặc định 1,5·IQR) hoặc theo phân vị (P1–P99); **ngưỡng học từ train**
  (đặt trong `Pipeline` nên không rò rỉ sang validation/test).
- `make_scaler` — `standard` | `robust` (median/IQR) | `power` (Yeo-Johnson) | `quantile` (xếp hạng)
  | `none`.

Cả hai đều là transformer sklearn hợp lệ ⇒ dùng được trong `Pipeline`, trong `GridSearchCV`
(`winsorize__method`, `scale__*`) và trong cross-validation theo nhóm công ty.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer, RobustScaler,
                                   StandardScaler)

#: Các chế độ winsorize được hỗ trợ (kèm mô tả để báo cáo đọc được).
WINSOR_METHODS: Dict[str, str] = {
    "none": "không clip (giữ nguyên, để đối chứng)",
    "iqr": "clip ngoài [Q1 − 1,5·IQR, Q3 + 1,5·IQR] (ngưỡng học từ train)",
    "p1p99": "clip ngoài [P1, P99] (ngưỡng học từ train)",
}

#: Các scaler được hỗ trợ.
SCALER_KINDS: Dict[str, str] = {
    "none": "không scale (mô hình cây)",
    "standard": "StandardScaler (mean/std — nhạy với outlier)",
    "robust": "RobustScaler (median/IQR — chịu đuôi nặng)",
    "power": "PowerTransformer (Yeo-Johnson — giảm lệch)",
    "quantile": "QuantileTransformer (xếp hạng về phân phối chuẩn)",
}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip giá trị ngoài ngưỡng học từ TRAIN (chống đuôi nặng trước khi scale).

    `method="iqr"` dùng [Q1 − k·IQR, Q3 + k·IQR]; `method="p1p99"` dùng phân vị; `method="none"`
    là transformer rỗng (giữ nguyên) để pipeline/grid dùng chung một đường.
    """

    def __init__(self, method: str = "iqr", iqr_factor: float = 1.5,
                 lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.method = method
        self.iqr_factor = iqr_factor
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Winsorizer":
        """Học cận dưới/cận trên cho TỪNG cột từ chính `X` truyền vào (chỉ được là train)."""
        if self.method not in WINSOR_METHODS:
            raise ValueError(f"method không hợp lệ: {self.method!r}; có {sorted(WINSOR_METHODS)}")
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
        """Áp cùng ngưỡng đã học từ train (NaN giữ nguyên cho `SimpleImputer` xử lý sau)."""
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)

    def clip_share(self, X: np.ndarray) -> float:
        """Tỉ lệ giá trị bị clip trên một tập bất kỳ (để báo cáo mức can thiệp thực tế)."""
        arr = np.asarray(X, dtype=float)
        finite = np.isfinite(arr)
        touched = finite & ((arr < self.lower_) | (arr > self.upper_))
        return float(touched.sum()) / float(finite.sum()) if finite.sum() else 0.0


def make_scaler(kind: str, random_seed: int = 0) -> Any:
    """Trả transformer scale theo tên (`kind` ∈ `SCALER_KINDS`); `none` ⇒ `"passthrough"`."""
    if kind not in SCALER_KINDS:
        raise ValueError(f"scaler không hợp lệ: {kind!r}; có {sorted(SCALER_KINDS)}")
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
