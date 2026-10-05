"""Stratified cross-validation with a per-fold log of the label distribution before and after resampling.

Resampling sits inside the pipeline, so it runs on the fold train only, and `assert_val_untouched`
confirms the fold validation set is unchanged. Out-of-fold probabilities drive threshold selection.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import format_distribution, label_distribution
from .metrics import metrics_at_threshold


def assert_val_untouched(X_va: np.ndarray, y_va: np.ndarray,
                         X_va_before: np.ndarray, y_va_before: np.ndarray) -> None:
    """Confirm the validation set is unchanged by resampling or the model."""
    if X_va.shape != X_va_before.shape or y_va.shape != y_va_before.shape:
        raise AssertionError("Validation changed shape, which suggests data leakage")
    if not (np.array_equal(X_va, X_va_before) and np.array_equal(y_va, y_va_before)):
        raise AssertionError("Validation values changed, which suggests data leakage")


def fold_distribution_report(estimator: Any, X_tr: np.ndarray, y_tr: np.ndarray,
                             factory: Callable[[], Any] | None = None
                             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the before and after resampling distributions for one fold.

    `factory` creates a fresh pipeline with the same configuration that contains only samplers, so the
    counting does not affect the model.
    """
    before = label_distribution(y_tr)
    if factory is None:
        return before, before
    probe = factory()
    X_res, y_res = probe.fit_resample(np.array(X_tr, copy=True), np.array(y_tr, copy=True))
    return before, label_distribution(y_res)


def cross_validate_strategy(strategy: Dict[str, Any], X_pool: np.ndarray, y_pool: np.ndarray, *,
                            n_splits: int, seed: int,
                            probe_factory: Callable[[], Any] | None = None,
                            log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Run cross-validation for one strategy and return out-of-fold probabilities, per-fold metrics and logs.

    Returns:
        A dict with `oof_proba`, `fold_metrics` (metrics at 0.5), `folds` (distribution logs) and
        `n_splits`.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y_pool), np.nan)
    fold_rows: List[Dict[str, Any]] = []
    fold_logs: List[Dict[str, Any]] = []

    for fold, (train_index, val_index) in enumerate(splitter.split(X_pool, y_pool), start=1):
        X_tr, y_tr = X_pool[train_index], y_pool[train_index]
        X_va, y_va = X_pool[val_index], y_pool[val_index]
        X_va_before, y_va_before = X_va.copy(), y_va.copy()

        before, after = fold_distribution_report(strategy["estimator"], X_tr, y_tr, probe_factory)
        estimator = strategy["factory"]() if "factory" in strategy else strategy["estimator"]
        estimator.fit(X_tr, y_tr)
        proba = estimator.predict_proba(X_va)[:, 1]

        assert_val_untouched(X_va, y_va, X_va_before, y_va_before)
        oof[val_index] = proba

        metrics = metrics_at_threshold(y_va, proba, 0.5)
        fold_rows.append({"fold": fold, "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                          **{f"val_{k}": v for k, v in metrics.items()}})
        fold_logs.append({"fold": fold, "train_before": before, "train_after": after,
                          "val": label_distribution(y_va)})
        log(f"    fold {fold}/{n_splits}: train {format_distribution(before)}"
            f" -> after resample {format_distribution(after)} | "
            f"val {format_distribution(fold_logs[-1]['val'])} | "
            f"val F1@0.5={metrics['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f}")

    return {"oof_proba": oof, "fold_metrics": fold_rows, "folds": fold_logs,
            "n_splits": n_splits}


def refit_and_score(strategy: Dict[str, Any], X_tr: np.ndarray, y_tr: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Refit on the whole train pool, then score the holdout test at several thresholds."""
    estimator = strategy["factory"]() if "factory" in strategy else strategy["estimator"]
    estimator.fit(X_tr, y_tr)
    proba = estimator.predict_proba(X_test)[:, 1]
    return {"proba": proba,
            "at_threshold": {name: metrics_at_threshold(y_test, proba, value)
                             for name, value in thresholds.items()}}
