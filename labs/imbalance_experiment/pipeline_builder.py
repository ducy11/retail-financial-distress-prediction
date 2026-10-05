"""Build the experiment pipeline catalog: baseline, single and hybrid techniques.

Every resampling step is composed with `imblearn.pipeline.Pipeline` rather than the sklearn pipeline, so
`fit_resample` only runs on fold-train data and cross-validation stays leak-free. Samplers, metrics and
classifiers are reused from `labs.imbalance_lab`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin

from .config import ExperimentConfig

LOGGER = logging.getLogger("labs.imbalance_experiment")

#: Required techniques, mapping each key to its group.
REQUIRED_TECHNIQUES: Dict[str, str] = {
    "baseline": "baseline",
    "ros": "single-data", "smote": "single-data", "borderline_smote": "single-data",
    "adasyn": "single-data", "rus": "single-data", "tomek": "single-data", "enn": "single-data",
    "class_weight": "single-algorithm", "focal_loss": "single-algorithm",
    "balanced_rf": "single-ensemble", "easy_ensemble": "single-ensemble",
    "balanced_bagging": "single-ensemble",
    "smote_tomek": "hybrid", "smote_enn": "hybrid", "smote_class_weight": "hybrid",
    "rusboost": "hybrid",
}

#: Run order, with the baseline first as the reference.
TECHNIQUE_ORDER: Tuple[str, ...] = tuple(REQUIRED_TECHNIQUES)

#: Display name of each pipeline.
NAMES: Dict[str, str] = {
    "baseline": "Baseline (untreated)",
    "ros": "RandomOverSampler",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "rus": "RandomUnderSampler",
    "tomek": "Tomek Links (cleaning)",
    "enn": "Edited Nearest Neighbours (cleaning)",
    "class_weight": "Class Weights (balanced)",
    "focal_loss": "Focal Loss (custom objective)",
    "balanced_rf": "Balanced Random Forest",
    "easy_ensemble": "EasyEnsemble",
    "balanced_bagging": "Balanced Bagging",
    "smote_tomek": "Hybrid SMOTE + Tomek Links",
    "smote_enn": "Hybrid SMOTE + ENN",
    "smote_class_weight": "Hybrid SMOTE + Class Weights",
    "rusboost": "Hybrid RUSBoost (undersampling + boosting)",
}

#: Short description printed in tables and logs.
NOTES: Dict[str, str] = {
    "baseline": "reference: standard model on the original data",
    "ros": "duplicate minority samples to the target ratio",
    "smote": "interpolate minority samples, which can blur the boundary",
    "borderline_smote": "interpolate minority samples only in the DANGER region",
    "adasyn": "generate more samples for harder minority samples",
    "rus": "randomly drop majority samples, losing information",
    "tomek": "boundary cleaning: drop mixed-class neighbor pairs",
    "enn": "boundary cleaning: drop samples whose neighbors vote for another class",
    "class_weight": "`class_weight='balanced'`, weights derived inside fit",
    "focal_loss": "Focal Loss, down-weighting easy samples through gamma and alpha",
    "balanced_rf": "random forest that undersamples the majority for each tree",
    "easy_ensemble": "many AdaBoost models over balanced subsets",
    "balanced_bagging": "bagging over balanced subsets",
    "smote_tomek": "SMOTE then clean noisy pairs with Tomek Links",
    "smote_enn": "SMOTE then clean samples rejected by their neighbors with ENN",
    "smote_class_weight": "moderate SMOTE at ratio 0.5 plus class weights",
    "rusboost": "boosting with undersampling in each round",
}

#: Library or base-model requirements per pipeline; a missing one skips the pipeline with a reason.
REQUIRES: Dict[str, Tuple[str, ...]] = {
    "ros": ("imblearn",), "smote": ("imblearn",), "borderline_smote": ("imblearn",),
    "adasyn": ("imblearn",), "rus": ("imblearn",), "tomek": ("imblearn",), "enn": ("imblearn",),
    "balanced_rf": ("imblearn",), "easy_ensemble": ("imblearn",), "balanced_bagging": ("imblearn",),
    "smote_tomek": ("imblearn",), "smote_enn": ("imblearn",), "smote_class_weight": ("imblearn",),
    "rusboost": ("imblearn",),
    "focal_loss": ("lightgbm",),
}


@dataclass(frozen=True)
class PipelineSpec:
    """One experiment pipeline.

    Attributes:
        key: technique key used in the CLI and artifacts.
        name: display name.
        group: one of baseline, single-data, single-algorithm, single-ensemble, hybrid.
        note: short description.
        factory: argument-free callable returning a fresh estimator for each fold.
        is_resampling: whether the pipeline contains a resampling step.
        sampler_probe: sampler-only pipeline (`pipe[:-1]`) used to log resampling per fold.
        requires: required libraries.
    """

    key: str
    name: str
    group: str
    note: str
    factory: Callable[[], Any]
    is_resampling: bool
    sampler_probe: Optional[Callable[[], Any]]
    requires: Tuple[str, ...] = ()



# Base classifier and the special estimators.
def base_classifier(cfg: ExperimentConfig, *, seed: Optional[int] = None) -> Any:
    """Shared base classifier, used by the baseline and every technique for a fair comparison.

    With `lightgbm` it reuses `labs.imbalance_lab.models.make_base_classifier`, which degrades to
    XGBoost or HistGradientBoosting when LightGBM is missing. With `random_forest` it returns a
    scikit-learn `RandomForestClassifier` without class weights.
    """
    random_state = int(cfg.seed if seed is None else seed)
    if cfg.base_model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=60 if cfg.quick else 300, max_depth=6,
            min_samples_leaf=4 if cfg.quick else 2, n_jobs=1, random_state=random_state)

    from labs.imbalance_lab.models import make_base_classifier

    return make_base_classifier(None, random_state=random_state)


def cost_sensitive_classifier(cfg: ExperimentConfig, *, seed: Optional[int] = None) -> Any:
    """Cost-sensitive classifier using balanced class weights.

    `random_forest` uses scikit-learn's `class_weight='balanced'` directly, while `lightgbm` uses
    `labs.imbalance_lab.models.BalancedWeightClassifier`, which derives the balanced weights with
    `compute_class_weight` inside `fit` and passes them as `sample_weight`. Both use the same formula and
    no global constant, so no labels leak.
    """
    random_state = int(cfg.seed if seed is None else seed)
    if cfg.base_model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=60 if cfg.quick else 300, max_depth=6,
            min_samples_leaf=4 if cfg.quick else 2, class_weight="balanced", n_jobs=1,
            random_state=random_state)

    from labs.imbalance_lab.models import BalancedWeightClassifier

    return BalancedWeightClassifier(random_state=random_state)


class RobustRUSBoost(BaseEstimator, ClassifierMixin):
    """Resilient RUSBoost combining undersampling with boosting, with fallbacks when AdaBoost refuses to fit.

    `imblearn.ensemble.RUSBoostClassifier` uses AdaBoost, which checks that its base estimator is better
    than random on the undersampled subset. At extreme imbalance such as 1:50 or 1:100 that check can
    fail with `ValueError: BaseClassifier in AdaBoostClassifier ensemble is worse than random`.

    Fallbacks are tried in order: RUSBoostClassifier with depth-3 trees, then with stumps, then an
    `imblearn.pipeline.Pipeline` of RandomUnderSampler followed by boosting. The successful path is
    recorded in `self.implementation_` and logged, so artifacts show which variant ran. Every sampling
    step stays inside `fit` over the received train set, so cross-validation remains safe.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustRUSBoost":
        """Fit the fallbacks in priority order and record the working path in `implementation_`."""
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        n_estimators = 30 if self.cfg.quick else 100
        candidates = (
            ("RUSBoost(DecisionTree max_depth=3)",
             lambda: RUSBoostClassifier(
                 estimator=DecisionTreeClassifier(max_depth=3, random_state=int(self.cfg.seed)),
                 n_estimators=n_estimators, learning_rate=0.1, random_state=int(self.cfg.seed))),
            ("RUSBoost(stump max_depth=1)",
             lambda: RUSBoostClassifier(
                 estimator=DecisionTreeClassifier(max_depth=1, random_state=int(self.cfg.seed)),
                 n_estimators=n_estimators, learning_rate=0.1, random_state=int(self.cfg.seed))),
        )
        for label, build in candidates:
            try:
                estimator = build()
                estimator.fit(X, y)
                self.estimator_, self.implementation_ = estimator, label
                LOGGER.info("    rusboost: using %s", label)
                return self
            except Exception as exc:  # noqa: BLE001 - AdaBoost can refuse for data reasons
                LOGGER.warning("    rusboost: %s failed to fit (%s), trying the next fallback",
                               label, type(exc).__name__)

        from labs.imbalance_lab.samplers import build_sampler_pipeline
        from labs.imbalance_lab.samplers import make_single_sampler

        steps = make_single_sampler("rus", prefer_imblearn=True,
                                    over_strategy=float(self.cfg.over_strategy),
                                    under_strategy=float(self.cfg.under_strategy),
                                    k_neighbors=int(self.cfg.k_neighbors),
                                    random_state=int(self.cfg.seed))
        pipeline = build_sampler_pipeline(steps, base_classifier(self.cfg), prefer_imblearn=True)
        pipeline.fit(X, y)
        self.estimator_ = pipeline
        self.implementation_ = "RandomUnderSampler + boosting (imblearn pipeline)"
        LOGGER.info("    rusboost: using %s", self.implementation_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability from the fitted estimator."""
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted labels at a 0.5 threshold."""
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Public parameters, keeping `cfg` so the model can be cloned or pickled."""
        return {"cfg": self.cfg}


def _ensemble(cfg: ExperimentConfig, key: str) -> Any:  # pragma: no cover - imblearn dependent
    """Imbalanced-learn ensembles for imbalanced data.

    All four estimators resample inside `fit` over the train set passed to them, so they are safe under
    cross-validation without an extra imblearn Pipeline around their sampling step.
    """
    from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier,
                                   EasyEnsembleClassifier, RUSBoostClassifier)

    random_state = int(cfg.seed)
    if key == "balanced_rf":
        return BalancedRandomForestClassifier(
            n_estimators=50 if cfg.quick else 100, max_depth=6, sampling_strategy="auto",
            replacement=False, n_jobs=1, random_state=random_state)
    if key == "easy_ensemble":
        return EasyEnsembleClassifier(n_estimators=5 if cfg.quick else 10, sampling_strategy="auto",
                                      n_jobs=1, random_state=random_state)
    if key == "balanced_bagging":
        from sklearn.ensemble import RandomForestClassifier

        return BalancedBaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=10 if cfg.quick else 30, max_depth=6,
                                             n_jobs=1, random_state=random_state),
            n_estimators=5 if cfg.quick else 10, sampling_strategy="auto", replacement=False,
            random_state=random_state)
    # RUSBoost combines boosting with per-round undersampling; the resilient wrapper is used because
    # AdaBoost can refuse to fit on extreme imbalance (see the RobustRUSBoost docstring).
    return RobustRUSBoost(cfg)


def sampler_steps(cfg: ExperimentConfig, key: str) -> List[Tuple[str, Any]]:
    """Sampler steps for one resampling technique."""
    from labs.imbalance_lab.samplers import SINGLE_SAMPLERS, make_hybrid_sampler, make_single_sampler

    kwargs = {"over_strategy": float(cfg.over_strategy), "under_strategy": float(cfg.under_strategy),
              "k_neighbors": int(cfg.k_neighbors), "random_state": int(cfg.seed)}
    if key in ("smote_tomek", "smote_enn"):
        return make_hybrid_sampler(key, prefer_imblearn=True, **kwargs)
    if key == "smote_class_weight":
        return make_single_sampler("smote", prefer_imblearn=True, **kwargs)
    if key in SINGLE_SAMPLERS:
        return make_single_sampler(key, prefer_imblearn=True, **kwargs)
    raise KeyError(f"Technique {key!r} has no sampler step")


# Pipeline and catalog construction.
def make_estimator(cfg: ExperimentConfig, key: str) -> Any:
    """Build a fresh estimator for one technique.

    Baseline, algorithm-level and ensemble techniques return the estimator directly. Resampling and
    hybrid techniques return `imblearn.pipeline.Pipeline([samplers..., classifier])`, so `fit_resample`
    only runs on the data the pipeline is fit on, the fold train.
    """
    if key == "baseline":
        return base_classifier(cfg)
    if key == "class_weight":
        return cost_sensitive_classifier(cfg)
    if key == "focal_loss":
        from labs.imbalance_lab.losses import FocalLossClassifier

        return FocalLossClassifier(random_state=int(cfg.seed))
    if key in ("balanced_rf", "easy_ensemble", "balanced_bagging", "rusboost"):
        return _ensemble(cfg, key)

    from labs.imbalance_lab.samplers import build_sampler_pipeline

    steps = sampler_steps(cfg, key)
    classifier = (cost_sensitive_classifier(cfg) if key == "smote_class_weight"
                  else base_classifier(cfg))
    pipeline = build_sampler_pipeline(steps, classifier, prefer_imblearn=True)
    checks = inspect_pipeline(pipeline, expect_samplers=True)
    if not checks["samplers_inside_pipeline"]:  # pragma: no cover - defensive
        raise AssertionError(f"{key}: sampler is not inside the pipeline, which risks data leakage")
    return pipeline


def inspect_pipeline(pipeline: Any, *, expect_samplers: bool) -> Dict[str, Any]:
    """Inspect the pipeline structure: is it an imblearn pipeline, and how many sampler steps does it have?

    Returns:
        A dict with `imblearn_pipeline` (bool), `n_sampler_steps` (int) and `samplers_inside_pipeline`
        (bool, whether the structure matches `expect_samplers`).
    """
    from labs.imbalance_lab.samplers import resampling_backend

    steps = list(getattr(pipeline, "steps", []))
    module = type(pipeline).__module__
    imblearn_ok = (module.startswith("imblearn.pipeline") if resampling_backend(True) == "imblearn"
                   else module.startswith("labs.imbalance_lab"))
    n_samplers = sum(1 for _name, step in steps if hasattr(step, "fit_resample"))
    return {"imblearn_pipeline": bool(imblearn_ok), "n_sampler_steps": int(n_samplers),
            "samplers_inside_pipeline": bool(n_samplers > 0) if expect_samplers
            else bool(n_samplers == 0)}


def missing_requirements(cfg: ExperimentConfig, key: str) -> List[str]:
    """Libraries or base-model settings still missing for one technique; empty means it can run."""
    from labs.imbalance_lab.losses import FocalLossClassifier
    from labs.imbalance_lab.samplers import resampling_backend

    gaps: List[str] = []
    for requirement in REQUIRES.get(key, ()):
        if requirement == "imblearn" and resampling_backend(True) != "imblearn":
            gaps.append("imbalanced-learn")
        elif requirement == "lightgbm" and not FocalLossClassifier.available():
            gaps.append("lightgbm")
    if key == "focal_loss" and cfg.base_model != "lightgbm":
        gaps.append(f"Focal Loss requires base_model=lightgbm (currently {cfg.base_model})")
    return gaps


def build_pipelines(cfg: ExperimentConfig, *, keys: Tuple[str, ...] = ()) -> List[PipelineSpec]:
    """Build the pipeline catalog, optionally restricted to `keys`.

    Args:
        cfg: experiment configuration.
        keys: only build these techniques; empty means all, in `TECHNIQUE_ORDER`.

    Raises:
        KeyError: if `keys` contains a key outside the catalog.
    """
    selected = list(keys) if keys else list(TECHNIQUE_ORDER)
    unknown = [key for key in selected if key not in REQUIRED_TECHNIQUES]
    if unknown:
        raise KeyError(f"Techniques not in the catalog: {unknown}; have {list(TECHNIQUE_ORDER)}")

    specs: List[PipelineSpec] = []
    for key in selected:
        is_resampling = key in ("ros", "smote", "borderline_smote", "adasyn", "rus", "tomek", "enn",
                                "smote_tomek", "smote_enn", "smote_class_weight")
        if is_resampling:
            steps = sampler_steps(cfg, key)

            def factory(steps=steps, key=key) -> Any:
                """Resampling pipeline plus classifier; each call returns a fresh instance."""
                from labs.imbalance_lab.samplers import build_sampler_pipeline

                classifier = (cost_sensitive_classifier(cfg) if key == "smote_class_weight"
                              else base_classifier(cfg))
                return build_sampler_pipeline(steps, classifier, prefer_imblearn=True)

            def probe(steps=steps) -> Any:
                """Sampler-only pipeline, so `fit_resample` can log the post-resampling distribution."""
                from labs.imbalance_lab.samplers import build_sampler_pipeline

                return build_sampler_pipeline(steps, base_classifier(cfg),
                                              prefer_imblearn=True)[:-1]
        else:
            def factory(key=key) -> Any:  # type: ignore[misc]
                return make_estimator(cfg, key)

            probe = None  # type: ignore[assignment]
        specs.append(PipelineSpec(key=key, name=NAMES[key], group=REQUIRED_TECHNIQUES[key],
                                  note=NOTES[key], factory=factory, is_resampling=is_resampling,
                                  sampler_probe=probe, requires=REQUIRES.get(key, ())))
    return specs


def pipeline_table(specs: List[PipelineSpec]) -> str:
    """Markdown table describing the pipeline catalog: group, technique and approach."""
    lines = ["| Group | Technique | Description | Resampling? |", "|---|---|---|---|"]
    for spec in specs:
        lines.append(f"| `{spec.group}` | `{spec.key}` - {spec.name} | {spec.note} | "
                     f"{'yes (inside the pipeline)' if spec.is_resampling else 'no'} |")
    return "\n".join(lines) + "\n"

