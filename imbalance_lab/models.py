"""Ba chiến lược xử lý mất cân bằng + bộ phân loại boosting (LightGBM là mặc định).

| Chiến lược | Cách can thiệp | Ghi chú |
|---|---|---|
| `baseline` | không can thiệp | mốc so sánh |
| `cost_sensitive` | `scale_pos_weight = n_âm/n_dương` tính ĐỘNG trong `fit` (chỉ dùng nhãn của fold train) | can thiệp vào HÀM MẤT MÁT |
| `resampling` | `imblearn.pipeline.Pipeline(SMOTE → RandomUnderSampler → classifier)` | can thiệp vào DỮ LIỆU, chỉ trên train của fold |

`build_leaky_reference()` dựng thêm một mốc "❌ SAI" (resample TOÀN BỘ train trước khi chia fold) để
đo — bằng số — mức lạc quan hoá mà thao tác sai này gây ra.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from .samplers import build_sampler_pipeline, make_samplers, resampling_backend

_BACKEND_CACHE: Dict[str, str] = {}


def classifier_backend() -> str:
    """Chọn backend boosting: `lightgbm` → `xgboost` (nếu fit được) → `hist_gradient_boosting`."""
    if "name" in _BACKEND_CACHE:
        return _BACKEND_CACHE["name"]
    name = "hist_gradient_boosting"
    try:  # ưu tiên LightGBM (bộ phân loại chính của lab)
        import lightgbm  # noqa: F401
        name = "lightgbm"
    except Exception:
        try:  # pragma: no cover - phụ thuộc môi trường
            from xgboost import XGBClassifier
            XGBClassifier(n_estimators=5).fit(np.zeros((10, 3)), np.array([0, 1] * 5))
            name = "xgboost"
        except Exception:
            name = "hist_gradient_boosting"
    _BACKEND_CACHE["name"] = name
    return name


#: Tham số tương ứng cho từng backend (LightGBM là gốc).
def _params_for(backend: str, scale_pos_weight: Optional[float]) -> Dict[str, Any]:
    if backend == "lightgbm":
        params = dict(C.LGBM_PARAMS)
        params["scale_pos_weight"] = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
        return params
    if backend == "xgboost":  # pragma: no cover - phụ thuộc môi trường
        return {"n_estimators": C.LGBM_PARAMS["n_estimators"],
                "learning_rate": C.LGBM_PARAMS["learning_rate"],
                "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.8,
                "reg_lambda": C.LGBM_PARAMS["reg_lambda"],
                "scale_pos_weight": 1.0 if scale_pos_weight is None else float(scale_pos_weight)}
    return {"max_iter": C.LGBM_PARAMS["n_estimators"],
            "learning_rate": C.LGBM_PARAMS["learning_rate"],
            "l2_regularization": C.LGBM_PARAMS["reg_lambda"],
            "class_weight": None if scale_pos_weight is None
            else {0: 1.0, 1: float(scale_pos_weight)}}


def make_base_classifier(scale_pos_weight: Optional[float] = None,
                         random_state: int = C.SEED) -> Any:
    """Tạo bộ phân loại boosting (mặc định LightGBM) — nền chung cho cả 3 chiến lược."""
    backend = classifier_backend()
    params = _params_for(backend, scale_pos_weight)
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=random_state, **params)
    if backend == "xgboost":  # pragma: no cover - phụ thuộc môi trường
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state, n_jobs=1, eval_metric="logloss", **params)
    return HistGradientBoostingClassifier(random_state=random_state, **params)


class ScalePosWeightClassifier(BaseEstimator, ClassifierMixin):
    """Can thiệp COST-SENSITIVE: `scale_pos_weight = n_âm / n_dương` tính trong `fit`.

    Vì sao tính trong `fit`: trong cross-validation, `fit` chỉ nhận tập train của fold ⇒ trọng số
    được suy từ **nhãn của fold train**, không dùng nhãn của validation/test (chống rò rỉ). Nếu tính
    một lần trên toàn bộ dữ liệu rồi truyền vào CV thì đó là rò rỉ nhãn.
    """

    def __init__(self, random_state: int = C.SEED) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScalePosWeightClassifier":
        y = np.asarray(y, dtype=int).ravel()
        n_positive = int(np.sum(y == 1))
        n_negative = int(len(y) - n_positive)
        self.scale_pos_weight_ = (n_negative / n_positive) if n_positive else 1.0
        self.estimator_ = make_base_classifier(scale_pos_weight=self.scale_pos_weight_,
                                               random_state=self.random_state)
        self.estimator_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"random_state": self.random_state}


class BalancedWeightClassifier(BaseEstimator, ClassifierMixin):
    """Cost-sensitive bằng `class_weight='balanced'`: trọng số mẫu tính TRONG `fit`.

    Vì sao tự tính: LightGBM / HistGradientBoosting nhận `sample_weight` nhưng không có tham số
    `class_weight='balanced'` dùng chung, nên lab suy trọng số bằng
    `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` trên **nhãn nhận được** rồi
    truyền xuống `fit` — cùng công thức tỉ lệ nghịch tần suất lớp, và trong CV chỉ thấy fold-train
    (không dùng nhãn validation/test ⇒ không rò rỉ).
    """

    def __init__(self, random_state: int = C.SEED) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BalancedWeightClassifier":
        from sklearn.utils.class_weight import compute_class_weight

        y = np.asarray(y, dtype=int).ravel()
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        self.class_weight_ = {int(c): float(w) for c, w in zip(classes, weights)}
        sample_weight = np.array([self.class_weight_[int(v)] for v in y], dtype=float)
        self.estimator_ = make_base_classifier(None, random_state=self.random_state)
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"random_state": self.random_state}


#: Tên các chiến lược (đúng thứ tự so sánh trong báo cáo).
STRATEGIES = ("baseline", "cost_sensitive", "resampling", "resampling_calibrated")

STRATEGY_DOCS: Dict[str, str] = {
    "baseline": "LightGBM mặc định — KHÔNG can thiệp mất cân bằng",
    "cost_sensitive": "LightGBM + `scale_pos_weight = n_âm/n_dương` tính động trong `fit` (cost-sensitive)",
    "resampling": "SMOTE → RandomUnderSampler → LightGBM trong `imblearn.pipeline.Pipeline`",
    "resampling_calibrated": ("Như `resampling` nhưng hiệu chuẩn xác suất bằng "
                              "`CalibratedClassifierCV` (isotonic) — vì resampling làm lệch prior"),
}


def _build_calibrated(samplers_steps: Any, prefer_imblearn: bool, random_state: int) -> Any:
    """`CalibratedClassifierCV` bọc quanh pipeline resampling (cross-fitting 5 fold).

    Vì sao cần: SMOTE + undersample làm đổi prior của tập train nên xác suất đầu ra bị kéo lệch
    (ngưỡng tối ưu ~0,89 thay vì ~0,09). Hiệu chuẩn isotonic đưa xác suất về đúng tần suất thực tế,
    giúp (a) đọc được Brier/reliability, (b) dùng chung một ngưỡng giữa các chiến lược.

    Chống rò rỉ: `CalibratedClassifierCV` tự chia dữ liệu nó NHẬN ĐƯỢC (train của fold ngoài) thành
    các fold nội bộ; resampling vẫn nằm bên trong pipeline ⇒ chỉ chạy trên phần fit của từng fold con.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold

    base = build_sampler_pipeline(samplers_steps, make_base_classifier(None, random_state=random_state),
                                  prefer_imblearn=prefer_imblearn)
    inner_cv = StratifiedKFold(n_splits=C.CALIBRATION_FOLDS, shuffle=True, random_state=random_state)
    return CalibratedClassifierCV(base, method=C.CALIBRATION_METHOD, cv=inner_cv)


def _build_estimator(name: str, prefer_imblearn: bool, random_state: int) -> Any:
    """Tạo estimator mới cho một chiến lược (hàm riêng để `factory` luôn trả đối tượng sạch)."""
    if name == "baseline":
        return make_base_classifier(None, random_state=random_state)
    if name == "cost_sensitive":
        return ScalePosWeightClassifier(random_state=random_state)
    if name in ("resampling", "resampling_calibrated"):
        samplers = make_samplers(prefer_imblearn, smote_strategy=C.SMOTE_SAMPLING_STRATEGY,
                                 k_neighbors=C.SMOTE_K_NEIGHBORS,
                                 under_strategy=C.UNDER_SAMPLING_STRATEGY,
                                 random_state=random_state)
        if name == "resampling":
            return build_sampler_pipeline(samplers, make_base_classifier(None, random_state=random_state),
                                          prefer_imblearn=prefer_imblearn)
        return _build_calibrated(samplers, prefer_imblearn, random_state)
    raise KeyError(f"Chiến lược không hợp lệ: {name!r}; có {STRATEGIES}")


def _probe_factory(name: str, prefer_imblearn: bool, random_state: int):
    """Factory cho pipeline CHỈ có sampler (`pipe[:-1]`) — dùng để log phân phối sau resampling."""
    def _probe() -> Any:
        samplers = make_samplers(prefer_imblearn, smote_strategy=C.SMOTE_SAMPLING_STRATEGY,
                                 k_neighbors=C.SMOTE_K_NEIGHBORS,
                                 under_strategy=C.UNDER_SAMPLING_STRATEGY,
                                 random_state=random_state)
        return build_sampler_pipeline(samplers, make_base_classifier(None, random_state=random_state),
                                      prefer_imblearn=prefer_imblearn)[:-1]
    return _probe


def build_strategy(name: str, *, prefer_imblearn: bool = C.PREFER_IMBLEARN,
                   random_state: int = C.SEED) -> Dict[str, Any]:
    """Dựng estimator cho một chiến lược; trả kèm metadata + factory để tạo bản mới mỗi fold.

    Raises:
        KeyError: nếu `name` không thuộc `STRATEGIES`.
    """
    if name not in STRATEGIES:
        raise KeyError(f"Chiến lược không hợp lệ: {name!r}; có {STRATEGIES}")
    return {
        "name": name,
        "estimator": _build_estimator(name, prefer_imblearn, random_state),
        "factory": lambda: _build_estimator(name, prefer_imblearn, random_state),
        "probe_factory": (_probe_factory(name, prefer_imblearn, random_state)
                          if name.startswith("resampling") else None),
        "doc": STRATEGY_DOCS[name],
        "backend": classifier_backend(),
        "resampling_backend": (resampling_backend(prefer_imblearn)
                              if name.startswith("resampling") else "—"),
    }


def build_leaky_reference(prefer_imblearn: bool = C.PREFER_IMBLEARN,
                          random_state: int = C.SEED) -> Dict[str, Any]:
    """Mốc MINH HOẠ SAI: resample toàn bộ dữ liệu TRƯỚC khi chia train/test (gây rò rỉ).

    Trả về cả `samplers` (pipeline chỉ có sampler, dùng `fit_resample`) và `classifier` riêng —
    vì `imblearn.Pipeline.fit_resample` chỉ tồn tại khi bước CUỐI cũng là sampler.
    """
    samplers = make_samplers(prefer_imblearn, smote_strategy=C.SMOTE_SAMPLING_STRATEGY,
                             k_neighbors=C.SMOTE_K_NEIGHBORS,
                             under_strategy=C.UNDER_SAMPLING_STRATEGY,
                             random_state=random_state)
    classifier = make_base_classifier(None, random_state=random_state)
    sampler_pipeline = build_sampler_pipeline(samplers, classifier, prefer_imblearn=prefer_imblearn)[:-1]
    return {"name": "leaky_resample_before_split", "samplers": sampler_pipeline,
            "classifier": classifier, "sampler_steps": samplers,
            "doc": "❌ SAI (minh hoạ): resample TRƯỚC khi chia fold ⇒ rò rỉ dữ liệu",
            "backend": classifier_backend(), "resampling_backend": resampling_backend(prefer_imblearn)}


def resample_once(leaky: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
    """Resample MỘT LẦN toàn bộ dữ liệu (chỉ dùng cho mốc minh hoạ sai)."""
    return leaky["samplers"].fit_resample(X, y)

