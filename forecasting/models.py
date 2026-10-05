"""Distress forecasting models, covering four scikit-learn families.

The model layer fits one pooled pipeline on all of train, so validation and test measure in-domain
performance, while generalization to unseen companies lives in `forecasting.validation`. Every script
and report reads the model list and hyperparameters from this module.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runtime_warnings import quiet_library_warnings
from .preprocessing import Winsorizer, make_scaler

# Filter the known scikit-learn 1.6 to scipy 1.18 `iprint` warning plus two matplotlib/pyparsing
# warnings, so training logs and test output stay clean. See runtime_warnings.py.
quiet_library_warnings()

from .config import RANDOM_SEED

#: Hyperparameters per model (single source of truth; tuning overrides via `params`).
HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "logistic": {"max_iter": 2000, "C": 0.1},
    "random_forest": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 2,
                      "class_weight": "balanced_subsample", "n_jobs": 1},
    "hist_gradient_boosting": {"max_iter": 300, "learning_rate": 0.05, "max_depth": 3,
                               "l2_regularization": 1.0, "class_weight": "balanced"},
    "mlp": {"hidden_layer_sizes": 32, "alpha": 1e-3, "learning_rate_init": 1e-3,
            "max_iter": 3000, "early_stopping": True, "n_iter_no_change": 30},
}

MODEL_REGISTRY = {
    "logistic": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "mlp": MLPClassifier,
}

#: Default trial order - exactly the 4 model families of the project (linear -> bagging -> boosting -> MLP).
DEFAULT_MODEL_ORDER = ["logistic", "random_forest", "hist_gradient_boosting", "mlp"]

#: Models that need scaling (gradient/distance based: logistic and MLP).
NEEDS_SCALING = {"logistic", "mlp"}

#: Default scaler per model family: linear models and MLP need scaling, tree models do not.
DEFAULT_SCALER = {"logistic": "standard", "mlp": "standard"}


def make_model(name: str, scaler: str | None = None, winsorize: str = "none", **overrides: Any):
    """Build a winsorize, impute, optional scale, model pipeline.

    `overrides` replace hyperparameters in `HYPERPARAMS`, used for tuning. `scaler=None` falls back to
    `DEFAULT_SCALER` (standard for logistic and MLP, none for tree models); pass `robust`, `power`,
    `quantile` or `none` for preprocessing experiments. `winsorize="iqr"` or `"p1p99"` clips heavy
    tails using thresholds learned from train.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unregistered model: {name!r}; have {sorted(MODEL_REGISTRY)}")
    params = {**HYPERPARAMS[name], **overrides, "random_state": RANDOM_SEED}
    imputer = SimpleImputer(strategy="median")
    if name == "logistic":
        estimator = LogisticRegression(**params)
    elif name == "random_forest":
        # random_state seeds the forest too; n_jobs=1 keeps output stable across machines
        estimator = RandomForestClassifier(**params)
    elif name == "mlp":
        # Non-tree non-linear family; `early_stopping` guards overfitting on 212 train rows.
        estimator = MLPClassifier(**params)
    else:  # hist_gradient_boosting - the only boosting family used in the project
        estimator = HistGradientBoostingClassifier(**params)

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
    """Class-1 (distress) probability from a fitted pipeline."""
    return model.predict_proba(X)[:, 1]
