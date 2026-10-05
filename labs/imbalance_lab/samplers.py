"""Leak-free resampling built on `imblearn.pipeline.Pipeline`.

Only imblearn runs `fit_resample` inside a pipeline, so resampling happens on fold-train data alone. The
module provides in-house SMOTE, RandomUnderSampler and a sampler pipeline as fallbacks, used only when
`imbalanced-learn` is unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

MINORITY_LABEL = 1


def resampling_backend(prefer_imblearn: bool = True) -> str:
    """Return "imblearn" when imbalanced-learn is usable, otherwise "builtin"."""
    if not prefer_imblearn:
        return "builtin"
    try:  # pragma: no cover - environment dependent
        import imblearn  # noqa: F401
    except Exception:
        return "builtin"
    return "imblearn"


# In-house implementations, used only when imbalanced-learn is missing, API-compatible with fit_resample.
class SMOTE(BaseEstimator):
    """Minimal SMOTE: interpolate between a minority sample and one of its `k_neighbors` minority neighbors.

    `sampling_strategy` is the desired minority/majority ratio after oversampling, matching imblearn.
    """

    def __init__(self, sampling_strategy: float = 0.1, k_neighbors: int = 5,
                 random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if len(minority) < 2 or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()
        k = max(1, min(self.k_neighbors, len(minority) - 1))
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X[minority])
        _, indices = neighbours.kneighbors(X[minority])
        base = rng.randint(0, len(minority), size=n_new)
        partner = np.array([indices[i, rng.randint(1, k + 1)] for i in base])
        gap = rng.random_sample((n_new, 1))
        synthetic = X[minority][base] + gap * (X[minority][partner] - X[minority][base])
        return (np.vstack([X, synthetic]),
                np.concatenate([y, np.full(n_new, MINORITY_LABEL, dtype=int)]))


class RandomUnderSampler(BaseEstimator):
    """Keep every minority sample and randomly drop majority samples to reach the sampling_strategy ratio."""

    def __init__(self, sampling_strategy: float = 0.5, random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if not len(minority) or not len(majority):
            return X.copy(), y.copy()
        target = int(round(len(minority) / self.sampling_strategy))
        if target >= len(majority):
            return X.copy(), y.copy()
        keep = np.concatenate([minority, rng.choice(majority, size=target, replace=False)])
        keep.sort()
        return X[keep], y[keep]


class RandomOverSampler(BaseEstimator):
    """Random oversampling: duplicate minority samples to reach the sampling_strategy minority/majority ratio."""

    def __init__(self, sampling_strategy: float = 0.5, random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if not len(minority) or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()
        picks = rng.randint(0, len(minority), size=n_new)
        return (np.vstack([X, X[minority][picks]]),
                np.concatenate([y, np.full(n_new, MINORITY_LABEL, dtype=int)]))


class BorderlineSMOTE(BaseEstimator):
    """Borderline-SMOTE, borderline-1 variant: interpolate only from minority samples on the boundary.

    A minority sample is labeled DANGER when, among its `m_neighbors` nearest neighbors over the whole
    dataset, majority samples outnumber minority samples, meaning it sits against the majority region.
    Interpolation runs only between DANGER samples and their minority neighbors, so synthetic samples
    follow the decision boundary.
    """

    def __init__(self, sampling_strategy: float = 0.5, k_neighbors: int = 5,
                 m_neighbors: int = 10, random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state

    def _danger_indices(self, X: np.ndarray, y: np.ndarray,
                        minority: np.ndarray) -> List[int]:
        """Indices within `minority` of the samples that fall in the DANGER region."""
        m = max(1, min(self.m_neighbors, len(X) - 1))
        neighbours = NearestNeighbors(n_neighbors=m + 1).fit(X).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        return [i for i, row in enumerate(neighbours)
                if int((y[row] != MINORITY_LABEL).sum()) > int((y[row] == MINORITY_LABEL).sum())]

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if len(minority) < 2 or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()

        danger = self._danger_indices(X, y, minority)
        base_pool = danger if danger else list(range(len(minority)))
        k = max(1, min(self.k_neighbors, len(minority) - 1))
        minority_neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X[minority]).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        base = np.asarray([base_pool[rng.randint(0, len(base_pool))] for _ in range(n_new)])
        partner = np.asarray([minority_neighbours[i][rng.randint(0, k)] for i in base])
        gap = rng.random_sample((n_new, 1))
        synthetic = X[minority][base] + gap * (X[minority][partner] - X[minority][base])
        return (np.vstack([X, synthetic]),
                np.concatenate([y, np.full(n_new, MINORITY_LABEL, dtype=int)]))


class ADASYN(BaseEstimator):
    """ADASYN: the number of synthetic samples per minority sample is proportional to its majority neighbors.

    The harder a minority sample, meaning the more majority neighbors it has, the more samples are
    generated around it, so the method concentrates on the decision region instead of spreading evenly
    as SMOTE does.
    """

    def __init__(self, sampling_strategy: float = 0.5, n_neighbors: int = 5,
                 random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if len(minority) < 2 or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()

        k = max(1, min(self.n_neighbors, len(X) - 1))
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        difficulty = np.array([np.mean(y[row] != MINORITY_LABEL) for row in neighbours], dtype=float)
        if difficulty.sum() <= 0:            # no majority neighbor, so fall back to plain SMOTE
            difficulty = np.ones(len(minority), dtype=float)
        quotas = difficulty / difficulty.sum() * n_new
        quotas_int = np.floor(quotas).astype(int)
        quotas_int[int(np.argmax(quotas - quotas_int))] += n_new - int(quotas_int.sum())

        k_min = max(1, min(self.n_neighbors, len(minority) - 1))
        minority_neighbours = NearestNeighbors(n_neighbors=k_min + 1).fit(X[minority]).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        synthetic: List[np.ndarray] = []
        for i, count in enumerate(quotas_int):
            for _ in range(int(count)):
                partner = minority_neighbours[i][rng.randint(0, k_min)]
                gap = rng.random_sample()
                synthetic.append(X[minority][i] + gap * (X[minority][partner] - X[minority][i]))
        if not synthetic:
            return X.copy(), y.copy()
        return (np.vstack([X, np.asarray(synthetic)]),
                np.concatenate([y, np.full(len(synthetic), MINORITY_LABEL, dtype=int)]))


class TomekLinks(BaseEstimator):
    """Tomek Links, boundary cleaning: drop the majority sample in a mutually-nearest majority/minority pair.

    This is clean sampling without a target ratio, matching imblearn: it trims the boundary rather than
    balancing a 98/2 set, so balancing requires oversampling alongside it, as in the SMOTE plus Tomek
    Links hybrid.
    """

    def __init__(self, n_neighbors: int = 1) -> None:
        self.n_neighbors = n_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        if len(np.unique(y)) < 2 or len(X) < 3:
            return X.copy(), y.copy()
        nearest = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X, return_distance=False)[:, 1]
        keep = np.ones(len(y), dtype=bool)
        for i, j in enumerate(nearest):
            if y[i] == y[j]:
                continue
            if y[i] != MINORITY_LABEL and nearest[j] == i:      # Tomek pair, so drop the majority sample
                keep[i] = False
        return X[keep], y[keep]


class EditedNearestNeighbours(BaseEstimator):
    """ENN, boundary cleaning: drop a sample when its `n_neighbors` nearest neighbors vote for another class.

    Matches imblearn: neighbors are searched over the whole dataset excluding the sample itself, the
    default `sampling_strategy="auto"` cleans only the majority class, and `kind_sel="all"` drops a
    sample when any neighbor has a different class, corresponding to
    `np.all(nhood_label == target_class)` in imblearn 0.14. `kind_sel="mode"` drops a sample when its
    class differs from the majority class among its neighbors. This is clean sampling with no target
    ratio.
    """

    def __init__(self, n_neighbors: int = 3, kind_sel: str = "all",
                 sampling_strategy: str = "auto") -> None:
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        labels = np.unique(y)
        if len(labels) < 2 or len(X) < 3:
            return X.copy(), y.copy()
        k = max(1, min(self.n_neighbors, len(X) - 1))
        all_neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
            X, return_distance=False)[:, 1:]
        majority = int(labels[np.argmax(np.bincount(y, minlength=2))])
        targets = labels if self.sampling_strategy == "all" else np.array([majority])
        keep = np.ones(len(y), dtype=bool)
        for label in targets:
            for i in np.flatnonzero(y == label):
                votes = y[all_neighbours[i]]
                if self.kind_sel == "all":
                    keep[i] = bool(np.all(votes == label))
                else:                                    # kind_sel == "mode"
                    keep[i] = bool(int(np.argmax(np.bincount(votes, minlength=2))) == label)
        return X[keep], y[keep]


class SamplerChain(BaseEstimator):
    """Chain several samplers for the in-house path, used by the SMOTE+TomekLinks and SMOTE+ENN hybrids."""

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        self.steps = list(steps)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_out, y_out = np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel()
        for _name, sampler in self.steps:
            X_out, y_out = sampler.fit_resample(X_out, y_out)
        return X_out, y_out


class Pipeline(BaseEstimator):
    """Minimal version of `imblearn.pipeline.Pipeline`: samplers run first, the classifier fits after.

    Supports `fit`, `fit_resample` (sampler steps only, used to log the post-resampling distribution),
    `predict`, `predict_proba` and slicing such as `pipe[:-1]`.
    """

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        self.steps = list(steps)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Pipeline(self.steps[index])
        return self.steps[index]

    @property
    def classifier(self):
        return self.steps[-1][1]

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the sampler steps in order over the fold's train data."""
        return self._resample(X, y)

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_out, y_out = np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel()
        for _name, step in self.steps[:-1]:
            if hasattr(step, "fit_resample"):
                X_out, y_out = step.fit_resample(X_out, y_out)
            else:  # a plain transformer such as a scaler, still run in order
                X_out = step.fit_transform(X_out, y_out)
        return X_out, y_out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Pipeline":
        X_res, y_res = self._resample(X, y)
        self.classifier.fit(X_res, y_res)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)


def build_sampler_pipeline(steps: Sequence[Tuple[str, Any]], classifier: Any,
                           prefer_imblearn: bool = True) -> Any:
    """Combine sampler steps and the classifier into one pipeline, imblearn when available, else in-house.

    Args:
        steps: list of `(name, sampler)` pairs; each sampler must expose `fit_resample`.
        classifier: the final estimator, exposing `fit` and `predict_proba`.
        prefer_imblearn: prefer `imblearn.pipeline.Pipeline`, the default and standard path.
    """
    backend = resampling_backend(prefer_imblearn)
    if backend == "imblearn":  # pragma: no cover - environment dependent
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(list(steps) + [("classifier", classifier)])
    return Pipeline(list(steps) + [("classifier", classifier)])


def make_samplers(prefer_imblearn: bool = True, *, smote_strategy: float, k_neighbors: int,
                  under_strategy: float, random_state: Any) -> List[Tuple[str, Any]]:
    """Build the mixed sampler pair: SMOTE for oversampling, then RandomUnderSampler for undersampling."""
    if resampling_backend(prefer_imblearn) == "imblearn":  # pragma: no cover - environment dependent
        from imblearn.over_sampling import SMOTE as ImbSMOTE
        from imblearn.under_sampling import RandomUnderSampler as ImbRUS
        return [("smote", ImbSMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors,
                                   random_state=random_state)),
                ("under", ImbRUS(sampling_strategy=under_strategy, random_state=random_state))]
    return [("smote", SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors,
                            random_state=random_state)),
            ("under", RandomUnderSampler(sampling_strategy=under_strategy,
                                         random_state=random_state))]


# Technique catalog from the rubric, mapping a technique key to its sampler steps.
#: Single techniques in the catalog, mapping the key to a display name.
SINGLE_SAMPLERS: Dict[str, str] = {
    "ros": "RandomOverSampler",
    "smote": "SMOTE",
    "borderline_smote": "BorderlineSMOTE",
    "adasyn": "ADASYN",
    "rus": "RandomUnderSampler",
    "tomek": "TomekLinks (boundary cleaning)",
    "enn": "EditedNearestNeighbours (boundary cleaning)",
}

#: Hybrid techniques in the catalog, mapping the key to a display name.
HYBRID_SAMPLERS: Dict[str, str] = {
    "smote_tomek": "SMOTE + TomekLinks",
    "smote_enn": "SMOTE + EditedNearestNeighbours",
}


def make_single_sampler(kind: str, prefer_imblearn: bool = True, *,
                        over_strategy: float, under_strategy: float, k_neighbors: int,
                        random_state: Any) -> List[Tuple[str, Any]]:
    """Sampler steps for one single technique, using imblearn when available and the in-house path otherwise."""
    if kind not in SINGLE_SAMPLERS:
        raise KeyError(f"Unknown technique: {kind!r}; have {sorted(SINGLE_SAMPLERS)}")
    if resampling_backend(prefer_imblearn) == "imblearn":  # pragma: no cover - environment dependent
        return _imblearn_single(kind, over_strategy, under_strategy, k_neighbors, random_state)
    return _builtin_single(kind, over_strategy, under_strategy, k_neighbors, random_state)


def make_hybrid_sampler(kind: str, prefer_imblearn: bool = True, *,
                        over_strategy: float, under_strategy: float, k_neighbors: int,
                        random_state: Any) -> List[Tuple[str, Any]]:
    """Sampler steps for a hybrid technique: SMOTE for oversampling, then TomekLinks or ENN for cleaning."""
    if kind not in HYBRID_SAMPLERS:
        raise KeyError(f"Unknown hybrid: {kind!r}; have {sorted(HYBRID_SAMPLERS)}")
    if resampling_backend(prefer_imblearn) == "imblearn":  # pragma: no cover - environment dependent
        from imblearn.combine import SMOTEENN as ImbSMOTEENN
        from imblearn.combine import SMOTETomek as ImbSMOTETomek
        from imblearn.over_sampling import SMOTE as ImbSMOTE

        smote = ImbSMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                         random_state=random_state)
        if kind == "smote_tomek":
            return [(kind, ImbSMOTETomek(smote=smote, random_state=random_state))]
        return [(kind, ImbSMOTEENN(smote=smote, random_state=random_state))]
    smote = SMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                  random_state=random_state)
    cleaner = TomekLinks() if kind == "smote_tomek" else EditedNearestNeighbours()
    return [(kind, SamplerChain([("smote", smote), ("clean", cleaner)]))]


def _imblearn_single(kind: str, over_strategy: float, under_strategy: float, k_neighbors: int,
                     random_state: Any) -> List[Tuple[str, Any]]:  # pragma: no cover - environment
    """imblearn branch of `make_single_sampler`, the standard path."""
    from imblearn.over_sampling import ADASYN as ImbADASYN
    from imblearn.over_sampling import BorderlineSMOTE as ImbBorderline
    from imblearn.over_sampling import RandomOverSampler as ImbROS
    from imblearn.over_sampling import SMOTE as ImbSMOTE
    from imblearn.under_sampling import EditedNearestNeighbours as ImbENN
    from imblearn.under_sampling import RandomUnderSampler as ImbRUS
    from imblearn.under_sampling import TomekLinks as ImbTomek

    if kind == "ros":
        return [(kind, ImbROS(sampling_strategy=over_strategy, random_state=random_state))]
    if kind == "smote":
        return [(kind, ImbSMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                                random_state=random_state))]
    if kind == "borderline_smote":
        return [(kind, ImbBorderline(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                                     random_state=random_state))]
    if kind == "adasyn":
        return [(kind, ImbADASYN(sampling_strategy=over_strategy, n_neighbors=k_neighbors,
                                 random_state=random_state))]
    if kind == "rus":
        return [(kind, ImbRUS(sampling_strategy=under_strategy, random_state=random_state))]
    if kind == "tomek":
        return [(kind, ImbTomek())]
    return [(kind, ImbENN())]


def _builtin_single(kind: str, over_strategy: float, under_strategy: float, k_neighbors: int,
                    random_state: Any) -> List[Tuple[str, Any]]:
    """In-house branch of `make_single_sampler`, used when imbalanced-learn is missing."""
    if kind == "ros":
        return [(kind, RandomOverSampler(sampling_strategy=over_strategy,
                                         random_state=random_state))]
    if kind == "smote":
        return [(kind, SMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                             random_state=random_state))]
    if kind == "borderline_smote":
        return [(kind, BorderlineSMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                                       random_state=random_state))]
    if kind == "adasyn":
        return [(kind, ADASYN(sampling_strategy=over_strategy, n_neighbors=k_neighbors,
                              random_state=random_state))]
    if kind == "rus":
        return [(kind, RandomUnderSampler(sampling_strategy=under_strategy,
                                          random_state=random_state))]
    if kind == "tomek":
        return [(kind, TomekLinks())]
    return [(kind, EditedNearestNeighbours())]

