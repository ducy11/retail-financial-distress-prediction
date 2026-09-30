"""Cross-validation STRATIFIED theo tỉ lệ lớp + log phân phối nhãn TRƯỚC/SAU resampling mỗi fold.

Quy tắc chống rò rỉ được thực thi trong `cross_validate_strategy()`:
1. Mỗi fold: `X_tr, y_tr` / `X_va, y_va` lấy từ `StratifiedKFold` trên **train_pool**.
2. Resampling (SMOTE/undersample) nằm TRONG pipeline ⇒ chỉ chạy trên `X_tr, y_tr`.
3. Log "sau resampling" lấy từ một pipeline THỨ HAI cùng cấu hình (chỉ chạy `fit_resample`), nên
   không ảnh hưởng mô hình; và có `assert_val_untouched()` xác nhận `X_va/y_va` không bị thay đổi.
4. Xác suất **out-of-fold** dùng để chọn ngưỡng; test holdout không tham gia bước này.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data import format_distribution, label_distribution
from .metrics import metrics_at_threshold


def assert_val_untouched(X_va: np.ndarray, y_va: np.ndarray,
                         X_va_before: np.ndarray, y_va_before: np.ndarray) -> None:
    """Bảo đảm tập validation KHÔNG bị resampling/model thay đổi (kiểm tra cấu trúc)."""
    if X_va.shape != X_va_before.shape or y_va.shape != y_va_before.shape:
        raise AssertionError("Validation bị đổi kích thước ⇒ nghi ngờ rò rỉ dữ liệu")
    if not (np.array_equal(X_va, X_va_before) and np.array_equal(y_va, y_va_before)):
        raise AssertionError("Giá trị validation bị thay đổi ⇒ nghi ngờ rò rỉ dữ liệu")


def fold_distribution_report(estimator: Any, X_tr: np.ndarray, y_tr: np.ndarray,
                             factory: Callable[[], Any] | None = None
                             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(phân phối trước, phân phối sau) resampling của một fold.

    `factory` tạo pipeline MỚI cùng cấu hình (chỉ có sampler) để đếm — tránh ảnh hưởng mô hình.
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
    """Chạy CV cho một chiến lược; trả OOF proba, metric từng fold và log resampling.

    Returns:
        dict gồm `oof_proba`, `fold_metrics` (metric tại 0.5), `folds` (log phân phối) và `n_splits`.
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
            f" → sau resample {format_distribution(after)} | "
            f"val {format_distribution(fold_logs[-1]['val'])} | "
            f"val F1@0.5={metrics['f1']:.3f} PR-AUC={metrics['pr_auc']:.3f}")

    return {"oof_proba": oof, "fold_metrics": fold_rows, "folds": fold_logs,
            "n_splits": n_splits}


def refit_and_score(strategy: Dict[str, Any], X_tr: np.ndarray, y_tr: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Fit lại trên TOÀN BỘ train_pool rồi đánh giá trên holdout test tại nhiều ngưỡng."""
    estimator = strategy["factory"]() if "factory" in strategy else strategy["estimator"]
    estimator.fit(X_tr, y_tr)
    proba = estimator.predict_proba(X_test)[:, 1]
    return {"proba": proba,
            "at_threshold": {name: metrics_at_threshold(y_test, proba, value)
                             for name, value in thresholds.items()}}
