"""Immutable experiment configuration dataclass, the single source of every experiment parameter.

`data_loader`, `pipeline_builder`, `evaluation` and `main` share one config, so logs and artifacts
always record the exact configuration that ran.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

#: Repository root, since this file sits in `labs/imbalance_experiment/`.
ROOT = Path(__file__).resolve().parents[1]
#: Directory for experiment artifacts.
DEFAULT_OUT_DIR = ROOT / "reports" / "experiment"

#: Technique groups used by the report and cross-checked in tests.
GROUPS: Tuple[str, ...] = ("baseline", "single-data", "single-algorithm", "single-ensemble", "hybrid")


@dataclass(frozen=True)
class ExperimentConfig:
    """All parameters of one experiment run.

    Attributes:
        source: "synthetic" by default, or a path to a CSV file such as credit card fraud data.
        target_column: label column name when reading CSV; labels must be 0/1.
        n_samples: sample count of the synthetic dataset.
        imbalance_ratio: majority to minority ratio, so 50 means 1:50 with a positive rate near 1.96%.
        n_features, n_informative, n_redundant, class_sep, flip_y: `make_classification` parameters.
        test_size: holdout test fraction, used once.
        seed: shared seed for splitting, samplers and models.
        base_model: "lightgbm" by default, or "random_forest".
        over_strategy, under_strategy, k_neighbors: sampler target ratio and neighbor count.
        n_splits: StratifiedKFold fold count.
        threshold: decision threshold used when reporting metrics; 0.5 by default.
        quick: lighter mode with fewer estimators for a fast run.
        techniques: technique keys to run; empty means all.
        out_dir: directory for artifacts.
        write: whether to write artifacts.
    """

    # Data.
    source: str = "synthetic"
    target_column: str = "Class"
    n_samples: int = 20_000
    imbalance_ratio: int = 50
    n_features: int = 20
    n_informative: int = 8
    n_redundant: int = 4
    class_sep: float = 1.6
    flip_y: float = 0.0
    test_size: float = 0.20
    seed: int = 42

    # Base model and samplers.
    base_model: str = "lightgbm"
    over_strategy: float = 0.50
    under_strategy: float = 0.50
    k_neighbors: int = 5

    # Evaluation.
    n_splits: int = 5
    threshold: float = 0.5
    quick: bool = False
    techniques: Tuple[str, ...] = ()
    out_dir: Path = field(default=DEFAULT_OUT_DIR)
    write: bool = True

    @property
    def positive_rate(self) -> float:
        """Minority, meaning positive, class rate derived from `imbalance_ratio`."""
        return 1.0 / (float(self.imbalance_ratio) + 1.0)

    @property
    def class_weights(self) -> Tuple[float, float]:
        """(negative rate, positive rate) for `make_classification(weights=...)`."""
        return (1.0 - self.positive_rate, self.positive_rate)

    def with_(self, **changes: Any) -> "ExperimentConfig":
        """Copy with a few fields overridden, since the dataclass is immutable."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """Config as a dict for JSON artifacts, with the Path converted to a string."""
        payload = dataclasses.asdict(self)
        payload["out_dir"] = str(self.out_dir)
        payload["class_weights"] = list(self.class_weights)
        payload["positive_rate"] = self.positive_rate
        return payload
