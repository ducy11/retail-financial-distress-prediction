"""Cross-validate each pipeline, aggregate per-fold metrics and score the holdout once.

Every fold keeps class proportions and logs the label distribution before and after resampling. The
fold validation set is compared against a copy taken before fit to detect leakage. Aggregate metrics are
mean and std across folds, while the holdout metrics and PR curve come from one refit on the full train.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve

from .config import ExperimentConfig
from .data_loader import Dataset, stratified_folds
from .pipeline_builder import PipelineSpec, inspect_pipeline, missing_requirements

LOGGER = logging.getLogger(__name__)

#: Primary metrics and the column order when printing tables; accuracy is excluded on purpose.
METRIC_KEYS: Tuple[str, ...] = ("pr_auc", "roc_auc", "f1", "macro_f1", "balanced_accuracy", "recall",
                                "fpr", "precision", "mcc")
#: Display label of each metric, used in the report.
METRIC_LABELS: Dict[str, str] = {
    "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "f1": "F1 (minority)", "macro_f1": "F1-macro",
    "balanced_accuracy": "Balanced Acc", "recall": "Recall", "fpr": "FPR", "precision": "Precision",
    "mcc": "MCC", "n_train_after": "n train after resample", "fit_seconds": "seconds/fold (fit)",
    "resample_seconds": "seconds/fold (resample)",
}


@dataclass
class FoldResult:
    """Result of one fold: metrics, timing, train size after resampling and the anti-leak flag."""

    fold: int
    n_train: int
    n_val: int
    n_train_after: Optional[int]
    ir_before: float
    ir_after: Optional[float]
    metrics: Dict[str, float]
    fit_seconds: float
    resample_seconds: Optional[float]
    checks: Dict[str, bool]


@dataclass
class TechniqueResult:
    """Result of one pipeline: per-fold mean and std, holdout metrics and PR curve data."""

    key: str
    name: str
    group: str
    note: str
    is_resampling: bool
    folds: List[FoldResult] = field(default_factory=list)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    holdout: Dict[str, float] = field(default_factory=dict)
    pr_curve: Optional[Dict[str, np.ndarray]] = None
    fit_seconds_mean: float = float("nan")
    resample_seconds_mean: float = float("nan")
    n_train_after_mean: float = float("nan")
    total_seconds: float = float("nan")
    checks: Dict[str, bool] = field(default_factory=dict)
    status: str = "ok"
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Whether the pipeline completed successfully."""
        return self.status == "ok"


def _imbalance_ratio(y: np.ndarray) -> float:
    """Imbalance ratio as negative over positive count, infinite when there are no positive samples."""
    y = np.asarray(y, dtype=int)
    n_positive = int((y == 1).sum())
    n_negative = int(len(y) - n_positive)
    return (n_negative / n_positive) if n_positive else float("inf")


def _fold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    """Metrics at the operating threshold, including FPR, from the standard `labs.imbalance_lab` set."""
    from labs.imbalance_lab.metrics import metrics_at_threshold

    metrics = metrics_at_threshold(np.asarray(y_true, int), np.asarray(proba, float), threshold)
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def _extra_holdout_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Extra holdout metrics: accuracy as a diagnostic plus the majority-class baseline."""
    from labs.imbalance_lab.metrics import accuracy_diagnostic

    y_pred = (np.asarray(proba, float) >= 0.5).astype(int)
    diagnostic = accuracy_diagnostic(np.asarray(y_true, int), y_pred)
    return {"accuracy": float(diagnostic["accuracy"]),
            "majority_baseline_accuracy_pct": float(diagnostic["majority_baseline_accuracy_pct"])}


def evaluate_technique(spec: PipelineSpec, dataset: Dataset, cfg: ExperimentConfig,
                       log: Any = LOGGER.info) -> TechniqueResult:
    """Run cross-validation and the holdout score for one pipeline; return a `TechniqueResult`.

    Exceptions are captured and recorded in the result rather than raised.

    Args:
        spec: pipeline to evaluate.
        dataset: dataset already split into train and test.
        cfg: experiment configuration.
        log: logging function, `LOGGER.info` by default.
    """
    from labs.imbalance_lab.cv import assert_val_untouched

    result = TechniqueResult(key=spec.key, name=spec.name, group=spec.group, note=spec.note,
                             is_resampling=spec.is_resampling)
    missing = missing_requirements(cfg, spec.key)
    if missing:
        result.status, result.reason = "skipped", f"missing {', '.join(missing)}"
        log(f"    SKIPPED {spec.name}: {result.reason}")
        return result

    fold_rows: List[FoldResult] = []
    started = time.perf_counter()
    try:
        for fold, (train_index, val_index) in enumerate(stratified_folds(dataset.y_train, cfg),
                                                      start=1):
            X_tr, y_tr = dataset.X_train[train_index], dataset.y_train[train_index]
            X_va, y_va = dataset.X_train[val_index], dataset.y_train[val_index]
            X_va_before, y_va_before = X_va.copy(), y_va.copy()

            n_after: Optional[int] = None
            ir_after: Optional[float] = None
            resample_seconds: Optional[float] = None
            if spec.sampler_probe is not None:
                probe = spec.sampler_probe()
                started_resample = time.perf_counter()
                _X_res, y_res = probe.fit_resample(np.array(X_tr, copy=True),
                                                   np.array(y_tr, copy=True))
                resample_seconds = time.perf_counter() - started_resample
                n_after, ir_after = int(len(y_res)), _imbalance_ratio(y_res)

            estimator = spec.factory()
            started_fit = time.perf_counter()
            estimator.fit(X_tr, y_tr)
            proba = estimator.predict_proba(X_va)[:, 1]
            fit_seconds = time.perf_counter() - started_fit

            checks = {"fold_validation_untouched": True}
            try:  # compare the fold validation set against the copy taken before fit
                assert_val_untouched(X_va, y_va, X_va_before, y_va_before)
            except AssertionError as exc:  # pragma: no cover - only on a real leak
                checks["fold_validation_untouched"] = False
                LOGGER.error("Leakage in fold %d of %s: %s", fold, spec.key, exc)

            fold_rows.append(FoldResult(
                fold=fold, n_train=int(len(y_tr)), n_val=int(len(y_va)), n_train_after=n_after,
                ir_before=_imbalance_ratio(y_tr), ir_after=ir_after,
                metrics=_fold_metrics(y_va, proba, cfg.threshold), fit_seconds=fit_seconds,
                resample_seconds=resample_seconds, checks=checks))
            row = fold_rows[-1]
            log(f"    fold {fold}/{cfg.n_splits}: n_train={len(y_tr)}"
                + (f" -> {n_after} after resample" if n_after is not None else "")
                + f" | val n={len(y_va)} | PR-AUC={row.metrics['pr_auc']:.4f} "
                  f"F1={row.metrics['f1']:.3f} Recall={row.metrics['recall']:.3f} "
                  f"FPR={row.metrics['fpr']:.4f} | {fit_seconds:.2f}s")

        # Aggregate mean and std across folds.
        for metric in METRIC_KEYS:
            values = [row.metrics[metric] for row in fold_rows]
            result.mean[metric] = float(np.nanmean(values))
            result.std[metric] = float(np.nanstd(values))
        result.fit_seconds_mean = float(np.mean([row.fit_seconds for row in fold_rows]))
        resample_times = [row.resample_seconds for row in fold_rows
                          if row.resample_seconds is not None]
        result.resample_seconds_mean = (float(np.mean(resample_times)) if resample_times
                                        else float("nan"))
        sizes = [row.n_train_after for row in fold_rows if row.n_train_after is not None]
        result.n_train_after_mean = float(np.mean(sizes)) if sizes else float("nan")
        result.folds = fold_rows
        _score_holdout(result, spec, dataset, cfg)
    except Exception as exc:  # noqa: BLE001 - one failing pipeline must not stop the experiment
        result.status, result.reason = "error", f"{type(exc).__name__}: {exc}"
        LOGGER.error("ERROR %s: %s", spec.name, result.reason)
    result.total_seconds = time.perf_counter() - started
    return result


def _score_holdout(result: TechniqueResult, spec: PipelineSpec, dataset: Dataset,
                   cfg: ExperimentConfig) -> None:
    """Refit on the whole train set, then score the holdout test once, producing metrics and the PR curve."""
    X_test_before, y_test_before = dataset.X_test.copy(), dataset.y_test.copy()
    estimator = spec.factory()
    estimator.fit(dataset.X_train, dataset.y_train)
    proba_test = estimator.predict_proba(dataset.X_test)[:, 1]
    result.holdout = _fold_metrics(dataset.y_test, proba_test, cfg.threshold)
    result.holdout.update(_extra_holdout_metrics(dataset.y_test, proba_test))
    precision, recall, _thresholds = precision_recall_curve(dataset.y_test, proba_test)
    result.pr_curve = {"recall": recall, "precision": precision}
    pipeline_checks = inspect_pipeline(spec.factory(), expect_samplers=spec.is_resampling)
    result.checks = {
        # The imblearn pipeline is required only for pipelines with a sampling step; ensembles sample
        # inside their own `fit`.
        "resampling uses imblearn.pipeline":
            bool(pipeline_checks["imblearn_pipeline"]) if spec.is_resampling else True,
        f"sampler stays inside the pipeline (n={pipeline_checks['n_sampler_steps']})":
            bool(pipeline_checks["samplers_inside_pipeline"]),
        "fold_validation_untouched": all(row.checks["fold_validation_untouched"] for row in result.folds),
        "test_untouched": bool(np.array_equal(dataset.X_test, X_test_before)
                               and np.array_equal(dataset.y_test, y_test_before)),
    }


def evaluate_all(specs: List[PipelineSpec], dataset: Dataset, cfg: ExperimentConfig,
                 log: Any = None) -> List[TechniqueResult]:
    """Run every pipeline in the catalog in order: baseline, then single, then hybrid."""
    logger = log or (lambda message: LOGGER.info(message))
    results: List[TechniqueResult] = []
    for index, spec in enumerate(specs, start=1):
        logger(f"\n[{index}/{len(specs)}] {spec.name} ({spec.group}) - {spec.note}")
        results.append(evaluate_technique(spec, dataset, cfg, log=logger))
    return results


def summary_rows(results: List[TechniqueResult], *, source: str = "cv") -> List[Dict[str, Any]]:
    """Flat table of every pipeline for CSV or DataFrame output.

    Args:
        results: pipeline results.
        source: "cv" for per-fold mean and std, or "holdout" for test metrics.
    """
    rows: List[Dict[str, Any]] = []
    for result in results:
        row: Dict[str, Any] = {"key": result.key, "group": result.group, "name": result.name,
                               "status": result.status, "reason": result.reason,
                               "is_resampling": result.is_resampling,
                               "n_train_after": (None if not np.isfinite(result.n_train_after_mean)
                                                 else round(result.n_train_after_mean, 1)),
                               "fit_seconds_per_fold": round(result.fit_seconds_mean, 3),
                               "resample_seconds_per_fold": (None if
                                                             not np.isfinite(result.resample_seconds_mean)
                                                             else round(result.resample_seconds_mean, 4)),
                               "total_seconds": round(result.total_seconds, 2)}
        for metric in METRIC_KEYS:
            if source == "cv":
                row[f"{metric}_mean"] = (None if metric not in result.mean
                                         else round(result.mean[metric], 6))
                row[f"{metric}_std"] = (None if metric not in result.std
                                        else round(result.std[metric], 6))
            else:
                row[metric] = (None if metric not in result.holdout
                               else round(result.holdout[metric], 6))
        rows.append(row)
    return rows


def _markdown_from_rows(rows: List[Dict[str, Any]], columns: List[str],
                        headers: List[str]) -> str:
    """Markdown table using `DataFrame.to_markdown()` when pandas is available, otherwise built by hand."""
    try:  # pragma: no cover - đường nhanh khi có pandas
        import pandas as pd

        frame = pd.DataFrame(rows, columns=columns)
        frame.columns = headers
        return frame.to_markdown(index=False)
    except Exception:  # noqa: BLE001 - pandas missing or a version issue
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(_fmt_cell(row.get(column)) for column in columns) + " |")
        return "\n".join(lines)


def _fmt_cell(value: Any, digits: int = 4) -> str:
    """Format one Markdown table cell: numbers to the given precision, None and NaN to '-'."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "-" if number != number else f"{number:.{digits}f}"


def format_mean_std(result: TechniqueResult, metric: str = "pr_auc", digits: int = 3) -> str:
    """`mean +/- std` string for one metric, used in summary tables."""
    if metric not in result.mean:
        return "-"
    return f"{result.mean[metric]:.{digits}f} +/- {result.std.get(metric, float('nan')):.{digits}f}"

