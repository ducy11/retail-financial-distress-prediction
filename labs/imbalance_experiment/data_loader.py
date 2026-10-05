"""Build or load an imbalanced dataset and split it while preserving class proportions.

Two sources are supported: synthetic data from `make_classification` at a 1:50 or 1:100 ratio, or a
binary CSV such as the Credit Card Fraud dataset. The test split is created first and used once.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """Train and test split of one dataset.

    Attributes:
        name: dataset name for logs and artifacts.
        X_train, y_train: train split used by cross-validation.
        X_test, y_test: holdout split scored once.
        meta: extra information such as source and class ratio.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]

    def distribution(self, y: np.ndarray) -> Dict[str, float]:
        """Positive-class rate and majority/minority ratio for one label array."""
        y = np.asarray(y, dtype=int)
        n_positive = int((y == 1).sum())
        n_negative = int(len(y) - n_positive)
        return {"n": int(len(y)), "n_positive": n_positive, "n_negative": n_negative,
                "positive_pct": 100.0 * n_positive / len(y) if len(y) else float("nan"),
                "imbalance_ratio": (n_negative / n_positive) if n_positive else float("inf")}

    def describe(self) -> str:
        """Single-line log string with the train and test sizes and class ratios."""
        train = self.distribution(self.y_train)
        test = self.distribution(self.y_test)
        return (f"{self.name}: train n={train['n']} (positive {train['n_positive']} = "
                f"{train['positive_pct']:.2f}%, IR={train['imbalance_ratio']:.1f}) | "
                f"test n={test['n']} (positive {test['n_positive']} = {test['positive_pct']:.2f}%, "
                f"IR={test['imbalance_ratio']:.1f}) | {self.X_train.shape[1]} features")


def make_synthetic_dataset(cfg: ExperimentConfig) -> Dataset:
    """Build an imbalanced synthetic dataset and split it with stratification."""
    X, y = make_classification(
        n_samples=int(cfg.n_samples),
        n_features=int(cfg.n_features),
        n_informative=int(cfg.n_informative),
        n_redundant=int(cfg.n_redundant),
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=float(cfg.class_sep),
        weights=list(cfg.class_weights),
        flip_y=float(cfg.flip_y),
        shuffle=True,
        random_state=int(cfg.seed),
    )
    X = X.astype(np.float64)
    y = y.astype(int)
    return _split_dataset(
        X, y,
        name=f"synthetic(1:{cfg.imbalance_ratio})",
        meta={"source": "synthetic", "imbalance_ratio_target": int(cfg.imbalance_ratio),
              "n_features": int(cfg.n_features), "n_informative": int(cfg.n_informative),
              "n_redundant": int(cfg.n_redundant), "class_sep": float(cfg.class_sep),
              "flip_y": float(cfg.flip_y), "seed": int(cfg.seed)},
        cfg=cfg)



def load_csv_dataset(cfg: ExperimentConfig) -> Dataset:
    """Read a binary dataset from CSV, such as `creditcard.csv`, then split it with stratification.

    pandas is used when available because it is faster on large files, otherwise the standard `csv`
    module is used. Labels must be 0/1 and the label column is named by `cfg.target_column`.
    """
    path = cfg.source
    meta_extra: Dict[str, Any]
    try:  # pragma: no cover - fast path phụ thuộc môi trường
        import pandas as pd

        frame = pd.read_csv(path)
        if cfg.target_column not in frame.columns:
            raise KeyError(f"CSV has no label column {cfg.target_column!r}; "
                           f"has {list(frame.columns)[:10]}")
        y = frame[cfg.target_column].to_numpy().astype(int)
        X = frame.drop(columns=[cfg.target_column]).to_numpy().astype(np.float64)
        meta_extra = {"reader": "pandas"}
    except ImportError:  # pragma: no cover - environment without pandas
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if cfg.target_column not in header:
                raise KeyError(f"CSV has no label column {cfg.target_column!r}")
            target_index = header.index(cfg.target_column)
            feature_indexes = [i for i in range(len(header)) if i != target_index]
            rows: List[List[float]] = []
            labels: List[int] = []
            for row in reader:
                if not row:
                    continue
                labels.append(int(float(row[target_index])))
                rows.append([float(row[i]) for i in feature_indexes])
        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        meta_extra = {"reader": "csv"}

    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Labels must be 0/1 for a binary classification task")
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return _split_dataset(X, y, name=f"csv:{name}",
                          meta={"source": str(path), **meta_extra}, cfg=cfg)


def _split_dataset(X: np.ndarray, y: np.ndarray, *, name: str, meta: Dict[str, Any],
                   cfg: ExperimentConfig) -> Dataset:
    """Split train and test with stratification and warn when the class ratios diverge."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg.test_size), stratify=y, random_state=int(cfg.seed), shuffle=True)
    dataset = Dataset(name=name, X_train=X_train, y_train=y_train.astype(int), X_test=X_test,
                      y_test=y_test.astype(int), meta=meta)
    train_rate = dataset.distribution(dataset.y_train)["positive_pct"]
    test_rate = dataset.distribution(dataset.y_test)["positive_pct"]
    if abs(train_rate - test_rate) > 1.0:
        LOGGER.warning("Train/test class rates differ by %.2f points, check the stratification",
                       abs(train_rate - test_rate))
    LOGGER.info("Split complete - %s", dataset.describe())
    return dataset


def load_dataset(cfg: ExperimentConfig) -> Dataset:
    """Load the dataset selected by `cfg.source`, either "synthetic" or a CSV path."""
    if str(cfg.source).lower() in ("synthetic", "make", ""):
        return make_synthetic_dataset(cfg)
    return load_csv_dataset(cfg)


def stratified_folds(y: np.ndarray, cfg: ExperimentConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """List of (train_index, val_index) from `StratifiedKFold`, each fold keeping class proportions."""
    splitter = StratifiedKFold(n_splits=int(cfg.n_splits), shuffle=True, random_state=int(cfg.seed))
    folds = [(train_index, val_index)
             for train_index, val_index in splitter.split(np.zeros(len(y)), y)]
    if len(folds) != int(cfg.n_splits):  # pragma: no cover - defensive
        raise AssertionError(f"StratifiedKFold returned {len(folds)} folds, expected {cfg.n_splits}")
    return folds
