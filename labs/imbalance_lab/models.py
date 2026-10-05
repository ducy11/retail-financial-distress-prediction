"""Three imbalance-handling strategies plus the boosting classifier, LightGBM by default.

Cost-sensitive weights are derived inside `fit` from the fold-train labels, and resampling runs inside
`imblearn.pipeline.Pipeline`, so validation and test never leak. `build_leaky_reference` adds an
intentionally wrong reference that resamples before splitting, to quantify that optimism.
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
    """Select the boosting backend: lightgbm, then xgboost when it fits, then hist_gradient_boosting."""
    if "name" in _BACKEND_CACHE:
        return _BACKEND_CACHE["name"]
    name = "hist_gradient_boosting"
    try:  # LightGBM is the lab's primary classifier
        import lightgbm  # noqa: F401
        name = "lightgbm"
    except Exception:
        try:  # pragma: no cover - environment dependent
            from xgboost import XGBClassifier
            XGBClassifier(n_estimators=5).fit(np.zeros((10, 3)), np.array([0, 1] * 5))
            name = "xgboost"
        except Exception:
            name = "hist_gradient_boosting"
    _BACKEND_CACHE["name"] = name
    return name


#: Matching parameters for each backend, derived from the LightGBM set.
def _params_for(backend: str, scale_pos_weight: Optional[float]) -> Dict[str, Any]:
    if backend == "lightgbm":
        params = dict(C.LGBM_PARAMS)
        params["scale_pos_weight"] = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
        return params
    if backend == "xgboost":  # pragma: no cover - environment dependent
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
    """Create the boosting classifier, LightGBM by default, shared by all three strategies."""
    backend = classifier_backend()
    params = _params_for(backend, scale_pos_weight)
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=random_state, **params)
    if backend == "xgboost":  # pragma: no cover - environment dependent
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state, n_jobs=1, eval_metric="logloss", **params)
    return HistGradientBoostingClassifier(random_state=random_state, **params)


class ScalePosWeightClassifier(BaseEstimator, ClassifierMixin):
    """Cost-sensitive intervention using `scale_pos_weight = n_negative / n_positive` inside fit.

    The weight is computed in `fit` because under cross-validation `fit` only receives the fold-train
    rows, so the weight comes from fold-train labels rather than validation or test labels. Computing
    it once over the full dataset and passing it into CV would leak labels.
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
    """Cost-sensitive intervention using balanced class weights computed inside fit.

    LightGBM and HistGradientBoosting accept `sample_weight` but share no `class_weight='balanced'`
    parameter, so the lab derives weights with
    `sklearn.utils.class_weight.compute_class_weight("balanced", ...)` from the received labels and
    passes them to `fit`. This matches the inverse-frequency formula and, under CV, only sees
    fold-train labels.
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


#: Strategy names in the comparison order used by the report.
STRATEGIES = ("baseline", "cost_sensitive", "resampling", "resampling_calibrated")

STRATEGY_DOCS: Dict[str, str] = {
    "baseline": "LightGBM defaults with no imbalance intervention",
    "cost_sensitive": "LightGBM with cost-sensitive `scale_pos_weight = n_negative/n_positive` computed in fit",
    "resampling": "SMOTE, then RandomUnderSampler, then LightGBM inside `imblearn.pipeline.Pipeline`",
    "resampling_calibrated": ("Same as `resampling` but with `CalibratedClassifierCV` (isotonic) "
                              "probability calibration, because resampling shifts the prior"),
}


def _build_calibrated(samplers_steps: Any, prefer_imblearn: bool, random_state: int) -> Any:
    """Wrap the resampling pipeline in `CalibratedClassifierCV` with 5-fold cross-fitting.

    SMOTE plus undersampling changes the train prior, which shifts the output probabilities so the
    optimal threshold moves from about 0.09 to about 0.89. Isotonic calibration restores probabilities
    to their true frequency, which makes Brier and reliability readable and lets all strategies share
    one threshold.

    CalibratedClassifierCV splits the data it receives, the outer fold-train, into inner folds; the
    resampling steps stay inside the pipeline, so they run only on each inner fit split.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold

    base = build_sampler_pipeline(samplers_steps, make_base_classifier(None, random_state=random_state),
                                  prefer_imblearn=prefer_imblearn)
    inner_cv = StratifiedKFold(n_splits=C.CALIBRATION_FOLDS, shuffle=True, random_state=random_state)
    return CalibratedClassifierCV(base, method=C.CALIBRATION_METHOD, cv=inner_cv)


def _build_estimator(name: str, prefer_imblearn: bool, random_state: int) -> Any:
    """Create a fresh estimator for one strategy; the factory uses this so each fold gets a clean object."""
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
    raise KeyError(f"Unknown strategy: {name!r}; have {STRATEGIES}")


def _probe_factory(name: str, prefer_imblearn: bool, random_state: int):
    """Factory for a sampler-only pipeline (`pipe[:-1]`), used to log the post-resampling distribution."""
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
    """Build the estimator for one strategy and return it with metadata and a per-fold factory.

    Raises:
        KeyError: if `name` is not in `STRATEGIES`.
    """
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy: {name!r}; have {STRATEGIES}")
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
    """Intentionally wrong reference that resamples all data before the train/test split, causing leakage.

    Returns both `samplers` (a sampler-only pipeline used with `fit_resample`) and the `classifier`
    separately, because `imblearn.Pipeline.fit_resample` exists only when the last step is a sampler.
    """
    samplers = make_samplers(prefer_imblearn, smote_strategy=C.SMOTE_SAMPLING_STRATEGY,
                             k_neighbors=C.SMOTE_K_NEIGHBORS,
                             under_strategy=C.UNDER_SAMPLING_STRATEGY,
                             random_state=random_state)
    classifier = make_base_classifier(None, random_state=random_state)
    sampler_pipeline = build_sampler_pipeline(samplers, classifier, prefer_imblearn=prefer_imblearn)[:-1]
    return {"name": "leaky_resample_before_split", "samplers": sampler_pipeline,
            "classifier": classifier, "sampler_steps": samplers,
            "doc": "Incorrect reference: resample before splitting folds, which leaks data",
            "backend": classifier_backend(), "resampling_backend": resampling_backend(prefer_imblearn)}


def resample_once(leaky: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Any:
    """Resample the whole dataset once; used only for the intentionally wrong reference."""
    return leaky["samplers"].fit_resample(X, y)

