"""Mô hình dự báo distress.

Mô tả ĐÚNG như code thực thi (tránh docstring sai so với hành vi):
- Split đã được chia theo thời gian trong từng công ty (`forecasting.data`, có dải purge),
  nhưng ở tầng model ta fit **một pipeline pooled** trên toàn bộ train. Validation/test vẫn
  chứa chính các công ty trong train → đây là đánh giá **in-domain**.
- Đánh giá khả năng tổng quát hoá sang công ty CHƯA TỪNG THẤY nằm ở `forecasting.validation`
  (GroupKFold / Leave-One-Company-Out) và baseline `ticker_prior` ở `forecasting.baselines`.

Model registry (4 họ chạy được offline):
- `logistic` — tuyến tính (tuyến tính trên log-odds của 47 feature đã scale);
- `random_forest` — bagging cây quyết định;
- `hist_gradient_boosting` — boosting cây thuần scikit-learn (**luôn chạy được**, không phụ thuộc xgboost);
- `lightgbm` — boosting hiện đại (leaf-wise + GOSS/EFB); chỉ xuất hiện khi môi trường có `lightgbm`
  (xem `requirements-benchmark.txt`), tự động bỏ qua nếu thiếu;
- `xgboost` — tuỳ chọn; bị bỏ qua khi phiên bản xgboost/sklearn xung đột (môi trường này: xgboost
  2.1.3 ↔ scikit-learn 1.6 lỗi `'super' object has no attribute '__sklearn_tags__'`).

Pipeline chuẩn: [`Winsorizer`] → `SimpleImputer(median)` → [`scaler`] → estimator, trong đó scaler
mặc định là `StandardScaler` cho mô hình tuyến tính và `none` cho mô hình cây — có thể đổi sang
`RobustScaler`/`PowerTransformer`/`QuantileTransformer` qua tham số `scaler` (xem
`forecasting/preprocessing.py` và thí nghiệm `scripts/experiment_preprocessing.py`).
Toàn bộ hyperparameter nằm trong `HYPERPARAMS` để báo cáo/tuning dùng lại một nguồn duy nhất.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runtime_warnings import quiet_library_warnings
from .preprocessing import Winsorizer, make_scaler

# Mỗi lần fit `LogisticRegression` in ra `OptimizeWarning: Unknown solver options: iprint`
# (scikit-learn 1.6 → scipy 1.18, không ảnh hưởng kết quả). Lọc ngay khi import để log huấn luyện
# và output test không bị nhiễu; chi tiết ở `runtime_warnings.py`.
quiet_library_warnings()

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover - máy không có xgboost
    HAS_XGB = False

try:  # LightGBM là phụ thuộc TUỲ CHỌN (lab đã dùng); thiếu thì registry bỏ qua
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:  # pragma: no cover - máy không có lightgbm
    HAS_LGBM = False

from .config import RANDOM_SEED

#: Hyperparameter của từng mô hình (nguồn duy nhất; tuning ghi đè qua `params`).
HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "logistic": {"max_iter": 2000, "C": 0.1},
    "random_forest": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2,
                      "class_weight": "balanced_subsample", "n_jobs": 1},
    "hist_gradient_boosting": {"max_iter": 300, "learning_rate": 0.05, "max_depth": 3,
                               "l2_regularization": 1.0, "class_weight": "balanced"},
    "lightgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 15,
                 "min_child_samples": 10, "subsample": 0.9, "subsample_freq": 1,
                 "colsample_bytree": 0.8, "reg_lambda": 1.0, "class_weight": "balanced",
                 "n_jobs": 1, "verbose": -1, "force_col_wise": True},
    "xgboost": {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9,
                "colsample_bytree": 0.8, "scale_pos_weight": 2.0, "eval_metric": "logloss",
                "n_jobs": 1},
}

MODEL_REGISTRY = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

#: Thứ tự thử nghiệm mặc định (các họ mô hình luôn sẵn có + boosting hiện đại khi môi trường cho phép).
DEFAULT_MODEL_ORDER = ["logistic", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost"]

if HAS_XGB:
    MODEL_REGISTRY["xgboost"] = XGBClassifier
if HAS_LGBM:
    MODEL_REGISTRY["lightgbm"] = LGBMClassifier

#: Mô hình cần scale (tuyến tính).
NEEDS_SCALING = {"logistic"}

#: Scaler mặc định theo họ mô hình: tuyến tính cần scale, mô hình cây không cần.
DEFAULT_SCALER = {"logistic": "standard"}


def make_model(name: str, scaler: str | None = None, winsorize: str = "none", **overrides: Any):
    """Tạo pipeline `[winsorize] → impute → [scale] → model`.

    `overrides` ghi đè hyperparameter trong `HYPERPARAMS` (dùng cho tuning).
    `scaler=None` ⇒ dùng `DEFAULT_SCALER` (standard cho logistic, none cho mô hình cây);
    `scaler="robust"` / `"power"` / `"quantile"` / `"none"` để thí nghiệm tiền xử lý.
    `winsorize="iqr"` / `"p1p99"` để clip đuôi nặng **học ngưỡng từ train**.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model chưa đăng ký: {name!r}; có {sorted(MODEL_REGISTRY)}")
    params = {**HYPERPARAMS[name], **overrides, "random_state": RANDOM_SEED}
    imputer = SimpleImputer(strategy="median")
    if name == "logistic":
        estimator = LogisticRegression(**params)
    elif name == "random_forest":
        # random_state áp cho cả rừng; n_jobs=1 để output ổn định giữa các máy
        estimator = RandomForestClassifier(**params)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(**params)
    elif name == "lightgbm":
        estimator = LGBMClassifier(**params)
    else:  # pragma: no cover - chỉ chạy khi có xgboost
        estimator = XGBClassifier(**params)

    steps: list[tuple[str, Any]] = []
    if winsorize != "none":
        steps.append(("winsorize", Winsorizer(method=winsorize)))
    steps.append(("impute", imputer))
    kind = scaler if scaler is not None else DEFAULT_SCALER.get(name, "none")
    scaler_step = make_scaler(kind, random_seed=RANDOM_SEED)
    if scaler_step != "passthrough":
        steps.append(("scale", scaler_step))
    steps.append(("model", estimator))
    return Pipeline(steps)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Xác suất lớp 1 (distress) từ pipeline đã fit."""
    return model.predict_proba(X)[:, 1]
