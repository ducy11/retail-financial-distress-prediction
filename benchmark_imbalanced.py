#!/usr/bin/env python
"""Benchmark: **Non-E Mode** (resampling / cost-sensitive) vs **E-Mode** (ensemble) cho bài toán
phân loại MẤT CÂN BẰNG.

Chạy: `python benchmark_imbalanced.py` (tuỳ chọn `--n-samples 20000`, `--quick`)

Thiết kế:
- Dữ liệu: `make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)`,
  chia **Stratified 80/20** — tập test tách một lần và dùng chung cho MỌI phương pháp.
- Ba nhóm phương pháp:
  * `Baseline`    : Logistic Regression (không can thiệp); XGBoost/LightGBM mặc định.
  * `Non-E Mode`  : data-level (SMOTE, RandomUnderSampler, SMOTE+Tomek Links — chạy TRONG
    `imblearn.pipeline.Pipeline`) + cost-sensitive (`class_weight='balanced'`, `scale_pos_weight`).
  * `E-Mode`      : bagging (`BalancedRandomForestClassifier`, `EasyEnsembleClassifier`) và
    boosting (`RUSBoostClassifier`).
- Metric trên test (ngưỡng 0.5): ROC-AUC, PR-AUC (Average Precision), F1 lớp thiểu số,
  Balanced Accuracy, Precision/Recall và **thời gian huấn luyện** (giây).
- Xuất bảng Markdown (DataFrame `pandas` nếu có, ngược lại tự sinh) → in ra terminal và ghi
  `reports/benchmark_imbalanced.md` + `.csv`.

CHỐNG RÒ RỈ DỮ LIỆU (được kiểm chứng tự động, in PASS/FAIL khi chạy):
1. Mọi sampler nằm trong `imblearn.pipeline.Pipeline` ⇒ `fit_resample` chỉ chạy trên TRAIN.
2. Cost-sensitive tính trọng số trong `fit` (chỉ từ nhãn train), không dùng hằng số toàn cục.
3. `RecordingClassifier` ghi lại số mẫu/nhãn mà classifier THỰC SỰ nhận khi fit ⇒ so với kỳ vọng.
4. Tập test được sao chép trước khi huấn luyện và so `np.array_equal` sau khi dự báo.
5. Log phân phối nhãn TRƯỚC/SAU resampling của từng phương pháp (lấy từ pipeline chỉ có sampler).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

# Cảnh báo vô hại của cặp phiên bản thư viện (scipy/scikit-learn `iprint`, matplotlib/pyparsing):
# xem `runtime_warnings.py`. Lọc ngay khi import để CẢ CLI và test đều sạch log (không che cảnh
# báo khác).
from runtime_warnings import quiet_library_warnings

quiet_library_warnings()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from imblearn.combine import SMOTETomek
from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                               RUSBoostClassifier)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

try:  # pandas là tuỳ chọn: nếu thiếu, bảng Markdown được tự sinh bằng tay
    import pandas as pd
    HAS_PANDAS = True
except Exception:  # pragma: no cover - môi trường không có pandas
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 10_000
N_FEATURES = 20
N_INFORMATIVE = 8
N_REDUNDANT = 4
CLASS_WEIGHTS = [0.95, 0.05]      # 95% lớp 0 / 5% lớp 1 (mất cân bằng cao)
TEST_SIZE = 0.20                  # Stratified 80/20
THRESHOLD = 0.5
OUTPUT_MD = "reports/benchmark_imbalanced.md"
OUTPUT_CSV = "reports/benchmark_imbalanced.csv"
OUTPUT_CV_MD = "reports/benchmark_imbalanced_cv.md"
OUTPUT_CV_CSV = "reports/benchmark_imbalanced_cv.csv"

#: (tên cột, hướng tốt) — dùng để in bảng và tìm phương pháp tốt nhất.
METRIC_COLUMNS = [("roc_auc", True), ("pr_auc", True), ("f1_minority", True),
                  ("balanced_accuracy", True), ("precision", True), ("recall", True),
                  ("train_time_s", False)]


def log(message: str = "") -> None:
    """In log ra terminal (và giữ trật tự khi bị redirect)."""
    print(message, flush=True)


def _distribution(y: np.ndarray) -> Dict[str, Any]:
    """Số mẫu mỗi lớp + imbalance ratio (đa số/thiểu số)."""
    n = int(len(y))
    positive = int((np.asarray(y) == 1).sum())
    negative = n - positive
    minority = min(positive, negative)
    return {"n": n, "n_negative": negative, "n_positive": positive,
            "positive_pct": 100.0 * positive / n if n else float("nan"),
            "imbalance_ratio": (max(positive, negative) / minority) if minority else float("inf")}


def _fmt_distribution(dist: Dict[str, Any]) -> str:
    return (f"n={dist['n']} | âm={dist['n_negative']} | dương={dist['n_positive']} "
            f"({dist['positive_pct']:.2f}%) | IR={dist['imbalance_ratio']:.1f}")


def make_dataset(n_samples: int = N_SAMPLES, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Dữ liệu phân loại nhị phân mất cân bằng 95/5 (tham số theo đúng yêu cầu đề bài)."""
    X, y = make_classification(n_samples=n_samples, n_features=N_FEATURES, n_classes=2,
                               n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT,
                               weights=CLASS_WEIGHTS, flip_y=0.01, shuffle=True,
                               random_state=seed)
    return X.astype(np.float64), y.astype(int)


class RecordingClassifier(BaseEstimator, ClassifierMixin):
    """Bọc classifier để GHI LẠI số mẫu/nhãn nó nhận khi `fit` (phục vụ kiểm chứng chống rò rỉ)."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RecordingClassifier":
        self.n_seen_ = int(len(y))
        self.n_positive_seen_ = int((np.asarray(y) == 1).sum())
        self.estimator.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


class PosWeightClassifier(BaseEstimator, ClassifierMixin):
    """Cost-sensitive cho boosting: `scale_pos_weight = n_âm/n_dương` tính TRONG `fit`.

    Vì sao tính trong `fit`: trọng số chỉ được suy từ nhãn của tập được truyền vào (train), không
    lấy từ toàn bộ dữ liệu ⇒ không rò rỉ nhãn của test.
    """

    def __init__(self, backend: str = "lightgbm", seed: int = SEED) -> None:
        self.backend = backend
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PosWeightClassifier":
        y = np.asarray(y, dtype=int).ravel()
        n_positive = int((y == 1).sum())
        n_negative = int(len(y) - n_positive)
        self.scale_pos_weight_ = (n_negative / n_positive) if n_positive else 1.0
        self.estimator_ = _make_boosting(self.backend, self.seed,
                                         scale_pos_weight=self.scale_pos_weight_)
        self.estimator_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"backend": self.backend, "seed": self.seed}


def _xgboost_usable() -> bool:
    """XGBoost 2.1 + scikit-learn 1.6 có lỗi runtime — thử fit nhỏ để biết có dùng được không."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - thiếu thư viện
        log(f"  (bỏ qua XGBoost) không import được: {exc}")
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = np.zeros((20, 3))
            y = np.array([0, 1] * 10)
            XGBClassifier(n_estimators=5).fit(X, y)
        return True
    except Exception as exc:  # pragma: no cover - xung đột phiên bản
        log(f"  (bỏ qua XGBoost) fit thử thất bại: {type(exc).__name__}: {exc}")
        return False


def _lightgbm_usable() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover
        log(f"  (bỏ qua LightGBM) không import được: {exc}")
        return False


def _make_boosting(backend: str, seed: int, scale_pos_weight: Optional[float] = None) -> Any:
    """Bộ phân loại boosting theo backend (`xgboost` | `lightgbm` | `hist_gradient_boosting`)."""
    weight = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
    if backend == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.9,
                             colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=weight,
                             n_jobs=1, random_state=seed, eval_metric="logloss")
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=weight,
                              n_jobs=1, random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=seed,
                                          class_weight=None if scale_pos_weight is None
                                          else {0: 1.0, 1: weight})


class RUSBoostInternal(BaseEstimator, ClassifierMixin):
    """RUSBoost tối giản (AdaBoost + RandomUnderSampler mỗi vòng) — bản DỰ PHÒNG.

    Vì sao cần: `imblearn.RUSBoostClassifier` dựa trên AdaBoost **SAMME.R**; scikit-learn 1.6 đã bỏ
    SAMME.R nên với dữ liệu 95/5 nó báo *“BaseClassifier in AdaBoostClassifier ensemble is worse
    than random, ensemble can not be fit.”*. Bản này cài đúng RUSBoost (Seiffert et al., 2010):
    mỗi vòng lấy mẫu CÂN BẰNG theo trọng số, tính lỗi có trọng số trên mẫu cân bằng rồi cập nhật
    trọng số cho toàn bộ tập train. Weak learner có lỗi ≥ 0,5 bị BỎ QUA (như AdaBoost chuẩn) nên
    mọi `alpha` đều dương ⇒ xác suất luôn nằm trong [0, 1].
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 1.0, max_depth: int = 3,
                 sampling_strategy: float = 0.5, random_state: int = SEED) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RUSBoostInternal":
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, float)
        y = np.asarray(y, int).ravel()
        weights = np.full(len(y), 1.0 / len(y))
        minority = np.flatnonzero(y == 1)
        majority = np.flatnonzero(y == 0)
        n_keep = max(1, int(round(len(minority) / self.sampling_strategy)))
        self.estimators_: List[DecisionTreeClassifier] = []
        self.alphas_: List[float] = []
        self.n_skipped_ = 0
        for _ in range(self.n_estimators):
            probability = weights[majority] / weights[majority].sum()   # trọng số vào RUS
            picked = rng.choice(majority, size=min(n_keep, len(majority)), replace=False,
                                p=probability)
            subset = np.concatenate([minority, picked])
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          random_state=int(rng.randint(1 << 30)))
            tree.fit(X[subset], y[subset])
            wrong = tree.predict(X[subset]) != y[subset]
            error = float(weights[subset][wrong].sum() / weights[subset].sum())
            if error >= 0.5:
                # weak learner tệ hơn ngẫu nhiên → bỏ qua (chuẩn AdaBoost). Nếu KHÔNG bỏ qua,
                # alpha = 0,5·ln((1-e)/e) < 0 sẽ làm xác suất vượt khỏi [0, 1].
                self.n_skipped_ += 1
                continue
            error = max(error, 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1.0 - error) / error)
            signed = (2 * y - 1) * (2 * tree.predict(X) - 1)
            weights = weights * np.exp(-alpha * signed)
            weights /= weights.sum()
            self.estimators_.append(tree)
            self.alphas_.append(float(alpha))
        if not self.estimators_:  # bảo hiểm: luôn có ít nhất một cây để dự báo
            fallback = DecisionTreeClassifier(max_depth=self.max_depth,
                                              random_state=self.random_state)
            fallback.fit(X, y)
            self.estimators_.append(fallback)
            self.alphas_.append(1.0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        alphas = np.asarray(self.alphas_, float)                    # (n_estimators,)
        votes = np.column_stack([(est.predict(X) == 1).astype(float)
                                 for est in self.estimators_])      # (n_samples, n_estimators)
        positive = votes @ alphas / (alphas.sum() or 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RUSBoostRobust(BaseEstimator, ClassifierMixin):
    """Đường chuẩn `imblearn.RUSBoostClassifier`; nếu lỗi thì tự chuyển `RUSBoostInternal`."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RUSBoostRobust":
        attempts = [
            ("imblearn.RUSBoostClassifier", lambda: RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                                 random_state=self.seed),
                n_estimators=100, random_state=self.seed)),
            ("RUSBoostInternal (dự phòng — sklearn 1.6 bỏ SAMME.R)",
             lambda: RUSBoostInternal(n_estimators=100, max_depth=3, random_state=self.seed)),
        ]
        last_error: Optional[Exception] = None
        for backend, factory in attempts:
            try:
                self.estimator_ = factory()
                self.estimator_.fit(X, y)
                self.backend_ = backend
                self.classes_ = np.array([0, 1])
                return self
            except Exception as exc:  # noqa: BLE001 - thử backend kế tiếp
                last_error = exc
                log(f"    (RUSBoost) {backend} không chạy được: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"Không fit được RUSBoost: {last_error}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"seed": self.seed}


BOOST_LABEL = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "hist_gradient_boosting":
               "HistGradientBoosting"}


def build_methods(seed: int, boosting: str) -> List[Dict[str, Any]]:
    """Danh sách phương pháp benchmark theo 3 nhóm (bỏ phương pháp không chạy được ở môi trường)."""
    def logreg() -> LogisticRegression:
        return LogisticRegression(max_iter=2000, random_state=seed)

    def wrap(estimator: Any, samplers: Optional[List[Tuple[str, Any]]] = None) -> Any:
        """Bọc classifier trong `RecordingClassifier` (để kiểm chứng) và (tuỳ chọn) trong pipeline."""
        recorder = RecordingClassifier(estimator)
        if samplers:
            return ImbPipeline([*samplers, ("classifier", recorder)])
        return recorder

    methods: List[Dict[str, Any]] = [
        # --- a) Baseline -------------------------------------------------
        {"group": "Baseline", "kind": "baseline",
         "name": "Logistic Regression (không can thiệp)", "estimator": logreg(),
         "samplers": None, "note": "mốc so sánh tuyến tính"},
        {"group": "Baseline", "kind": "baseline",
         "name": f"{BOOST_LABEL[boosting]} (mặc định)", "estimator": _make_boosting(boosting, seed),
         "samplers": None, "note": "mốc so sánh boosting, không chỉnh gì cho mất cân bằng"},

        # --- b) Non-E Mode: data-level (resampling TRONG pipeline) -------
        {"group": "Non-E Mode", "kind": "data-level",
         "name": "SMOTE + Logistic Regression",
         "estimator": logreg(),
         "samplers": [("smote", SMOTE(sampling_strategy=0.5, random_state=seed))],
         "note": "oversample thiểu số lên 50% đa số"},
        {"group": "Non-E Mode", "kind": "data-level",
         "name": f"SMOTE + {BOOST_LABEL[boosting]}",
         "estimator": _make_boosting(boosting, seed),
         "samplers": [("smote", SMOTE(sampling_strategy=0.5, random_state=seed))],
         "note": "oversample + boosting"},
        {"group": "Non-E Mode", "kind": "data-level",
         "name": "RandomUnderSampler + Logistic Regression",
         "estimator": logreg(),
         "samplers": [("under", RandomUnderSampler(sampling_strategy=0.5, random_state=seed))],
         "note": "undersample đa số còn 2× thiểu số"},
        {"group": "Non-E Mode", "kind": "data-level",
         "name": "SMOTE + Tomek Links + Logistic Regression",
         "estimator": logreg(),
         "samplers": [("smote_tomek", SMOTETomek(random_state=seed))],
         "note": "oversample rồi dọn cặp mẫu nhiễu (combine)"},

        # --- b) Non-E Mode: cost-sensitive ------------------------------
        {"group": "Non-E Mode", "kind": "cost-sensitive",
         "name": "Logistic Regression (class_weight='balanced')",
         "estimator": logreg(), "samplers": None,
         "note": "đổi trọng số trong hàm mất mát"},
        {"group": "Non-E Mode", "kind": "cost-sensitive",
         "name": f"{BOOST_LABEL[boosting]} (scale_pos_weight động)",
         "estimator": PosWeightClassifier(backend=boosting, seed=seed), "samplers": None,
         "note": "trọng số = n_âm/n_dương tính trong fit"},

        # --- c) E-Mode: ensemble ----------------------------------------
        {"group": "E-Mode", "kind": "ensemble-bagging",
         "name": "BalancedRandomForestClassifier",
         "estimator": BalancedRandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
         "samplers": None, "note": "bagging + cân bằng mỗi cây"},
        {"group": "E-Mode", "kind": "ensemble-bagging",
         "name": "EasyEnsembleClassifier",
         "estimator": EasyEnsembleClassifier(n_estimators=10, random_state=seed, n_jobs=1),
         "samplers": None, "note": "nhiều balanced AdaBoost trên tập đa số khác nhau"},
        {"group": "E-Mode", "kind": "ensemble-boosting",
         "name": "RUSBoostClassifier",
         "estimator": RUSBoostRobust(seed=seed),
         "samplers": None,
         "note": "boosting + random undersampling mỗi vòng (imblearn; nếu lỗi do sklearn 1.6 bỏ "
                 "SAMME.R thì tự dùng bản nội bộ tương đương)"},
    ]

    # Cost-sensitive cho Logistic bằng tham số chuẩn của sklearn (nhánh riêng cho rõ nghĩa)
    methods[6]["estimator"] = LogisticRegression(max_iter=2000, random_state=seed,
                                                 class_weight="balanced")
    for method in methods:
        sampler_steps = method.get("samplers")
        method["factory"] = (lambda m=method: wrap(m["estimator"], m.get("samplers")))
        method["is_resampling"] = bool(sampler_steps)
    return methods


def evaluate_method(method: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    threshold: float = THRESHOLD) -> Dict[str, Any]:
    """Fit + chấm điểm một phương pháp trên test, kèm kiểm chứng chống rò rỉ dữ liệu."""
    X_test_before, y_test_before = X_test.copy(), y_test.copy()
    model = method["factory"]()

    before = _distribution(y_train)
    after = before
    if method["is_resampling"]:
        probe = ImbPipeline([*method["samplers"]])          # chỉ có sampler → fit_resample được
        _X_res, y_res = probe.fit_resample(X_train.copy(), y_train.copy())
        after = _distribution(y_res)
    expected_seen = after["n"]

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    recorder = model.steps[-1][1] if method["is_resampling"] else model
    inner_backend = getattr(getattr(recorder, "estimator", None), "backend_", None)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    checks = {
        "test_nguyên_vẹn": bool(np.array_equal(X_test, X_test_before)
                                and np.array_equal(y_test, y_test_before)),
        "classifier_nhận_đúng_dữ_liệu_train": bool(getattr(recorder, "n_seen_", -1) == expected_seen),
        "resample_chỉ_trên_train": bool(not method["is_resampling"]
                                        or getattr(recorder, "n_seen_", -1) != before["n"]),
    }
    return {
        "group": method["group"], "method": method["name"], "kind": method["kind"],
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "f1_minority": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "train_time_s": float(train_time),
        "n_train_before": before["n"], "n_train_after": after["n"],
        "ir_before": before["imbalance_ratio"], "ir_after": after["imbalance_ratio"],
        "pos_pct_after": after["positive_pct"],
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "leakage_ok": all(checks.values()), "leakage_detail": checks,
        "note": method.get("note", "") + (f" [backend: {inner_backend}]" if inner_backend else ""),
    }


def render_markdown(rows: Sequence[Dict[str, Any]]) -> str:
    """Bảng Markdown: dùng `pandas.DataFrame.to_markdown()` nếu có, ngược lại tự sinh."""
    columns = ["group", "method", "roc_auc", "pr_auc", "f1_minority", "balanced_accuracy",
               "precision", "recall", "train_time_s"]
    if HAS_PANDAS:  # pragma: no cover - phụ thuộc môi trường
        try:  # `to_markdown` cần gói `tabulate`; thiếu thì rơi về bảng tự sinh bên dưới
            return pd.DataFrame(list(rows))[columns].to_markdown(index=False, floatfmt=".4f")
        except Exception as exc:  # noqa: BLE001
            log(f"  (lưu ý) pandas.to_markdown không dùng được ({type(exc).__name__}: {exc}) "
                f"→ tự sinh bảng Markdown")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" if c in ("group", "method") else "---:" for c in columns) + "|"
    lines = [header, separator]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_group_summary(rows: Sequence[Dict[str, Any]]) -> str:
    """Trung bình theo nhóm + phương pháp tốt nhất từng chỉ số."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    header = ["Nhóm", "#phương pháp", "PR-AUC (tb)", "F1 thiểu số (tb)", "Balanced Acc (tb)",
              "Thời gian (tb, s)"]
    lines = ["| " + " | ".join(header) + " |", "|---|---:|---:|---:|---:|---:|"]
    for group, items in groups.items():
        lines.append("| " + " | ".join([
            group, str(len(items)),
            f"{np.mean([r['pr_auc'] for r in items]):.4f}",
            f"{np.mean([r['f1_minority'] for r in items]):.4f}",
            f"{np.mean([r['balanced_accuracy'] for r in items]):.4f}",
            f"{np.mean([r['train_time_s'] for r in items]):.3f}"]) + " |")
    best_lines = []
    for metric, higher_is_better in METRIC_COLUMNS:
        ordered = sorted(rows, key=lambda r: r[metric], reverse=higher_is_better)
        best = ordered[0]
        best_lines.append(f"- **{metric}**: `{best['method']}` ({best['group']}) = "
                          f"{best[metric]:.4f}")
    return "\n".join(lines) + "\n\n**Tốt nhất theo từng chỉ số**\n\n" + "\n".join(best_lines)


def stratified_folds(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                     seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Chia fold bằng `StratifiedKFold` — MỖI fold giữ tỉ lệ lớp như toàn bộ dữ liệu."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train_index, val_index) for train_index, val_index in splitter.split(X, y)]


def evaluate_cv(method: Dict[str, Any], X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                seed: int = SEED, threshold: float = THRESHOLD) -> Dict[str, Any]:
    """Đánh giá bằng `StratifiedKFold`: resampling CHỈ trên fold-train, fold-val nguyên vẹn.

    Kiểm chứng từng fold:
    - `fold_val_nguyên_vẹn`: fold validation không bị sửa (so với bản sao trước khi fit);
    - `classifier_chỉ_thấy_fold_train`: classifier nhận ĐÚNG dữ liệu của fold-train (đã resample nếu
      là phương pháp resampling) — số mẫu ghi bởi `RecordingClassifier`;
    - `tỉ_lệ_lớp_giữ_nguyên`: tỉ lệ dương của fold-val lệch ≤ 2 điểm % so với toàn bộ dữ liệu
      (bảo đảm chia stratified có hiệu lực, không fold nào méo tỉ lệ lớp).
    """
    global_pct = _distribution(y)["positive_pct"]
    records: List[Dict[str, Any]] = []
    for fold, (train_index, val_index) in enumerate(stratified_folds(X, y, n_splits, seed), 1):
        X_tr, y_tr = X[train_index], y[train_index]
        X_va, y_va = X[val_index], y[val_index]
        X_va_before, y_va_before = X_va.copy(), y_va.copy()

        before = _distribution(y_tr)
        after = before
        if method["is_resampling"]:
            probe = ImbPipeline([*method["samplers"]])       # chỉ có sampler → fit_resample
            _X_res, y_res = probe.fit_resample(X_tr.copy(), y_tr.copy())
            after = _distribution(y_res)

        model = method["factory"]()
        model.fit(X_tr, y_tr)                                # ⚠️ chỉ fold-train, không có fold-val
        proba = model.predict_proba(X_va)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        recorder = model.steps[-1][1] if method["is_resampling"] else model
        val_dist = _distribution(y_va)
        records.append({
            "fold": fold, "n_train": before["n"], "n_train_after": after["n"],
            "ir_before": before["imbalance_ratio"], "ir_after": after["imbalance_ratio"],
            "n_val": val_dist["n"], "pos_pct_val": val_dist["positive_pct"],
            "roc_auc": float(roc_auc_score(y_va, proba)),
            "pr_auc": float(average_precision_score(y_va, proba)),
            "f1_minority": float(f1_score(y_va, y_pred, pos_label=1, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_va, y_pred)),
            "checks": {
                "fold_val_nguyên_vẹn": bool(np.array_equal(X_va, X_va_before)
                                            and np.array_equal(y_va, y_va_before)),
                "classifier_chỉ_thấy_fold_train": bool(getattr(recorder, "n_seen_", -1) == after["n"]),
                "tỉ_lệ_lớp_giữ_nguyên": bool(abs(val_dist["positive_pct"] - global_pct) <= 2.0),
            },
        })

    metrics = ["roc_auc", "pr_auc", "f1_minority", "balanced_accuracy"]
    summary: Dict[str, Any] = {"group": method["group"], "method": method["name"],
                               "n_splits": n_splits, "folds": records}
    for metric in metrics:
        values = [r[metric] for r in records]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    summary["leakage_ok"] = all(all(r["checks"].values()) for r in records)
    summary["n_folds_pass"] = sum(1 for r in records if all(r["checks"].values()))
    summary["pos_pct_val_range"] = (min(r["pos_pct_val"] for r in records),
                                    max(r["pos_pct_val"] for r in records))
    return summary


def render_cv_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Bảng Markdown cho kết quả StratifiedKFold: trung bình ± độ lệch chuẩn qua các fold."""
    header = ["Nhóm", "Phương pháp", "#fold", "ROC-AUC", "PR-AUC", "F1 thiểu số",
              "Bal. Acc", "Rò rỉ"]
    lines = ["| " + " | ".join(header) + " |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        cells = [row["group"], row["method"], str(row["n_splits"])]
        for metric in ("roc_auc", "pr_auc", "f1_minority", "balanced_accuracy"):
            cells.append(f"{row[metric + '_mean']:.4f} ± {row[metric + '_std']:.4f}")
        cells.append(f"{row['n_folds_pass']}/{row['n_splits']} PASS")
        lines.append("| " + " | ".join(cells) + " |")
    lo, hi = (min(r["pos_pct_val_range"][0] for r in rows),
              max(r["pos_pct_val_range"][1] for r in rows)) if rows else (0.0, 0.0)
    lines += ["", f"*Tỉ lệ dương các fold validation: {lo:.2f}% – {hi:.2f}% "
                  f"(dataset gốc {CLASS_WEIGHTS[1] * 100:.2f}%).*"]
    return "\n".join(lines)


def write_artifacts(rows: Sequence[Dict[str, Any]], table: str, summary: str,
                    header: str, extra_sections: Optional[List[str]] = None) -> None:
    """Ghi `reports/benchmark_imbalanced.md` và `.csv` (không lỗi nếu thiếu thư mục)."""
    import pathlib

    md_path, csv_path = pathlib.Path(OUTPUT_MD), pathlib.Path(OUTPUT_CSV)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    notes = "\n".join(f"- {r['method']}: {r['note']}" for r in rows if r.get("note"))
    leakage = "\n".join(f"- {'PASS' if r['leakage_ok'] else 'FAIL'} — {r['method']}"
                        for r in rows)
    md_path.write_text("\n".join([
        "# Benchmark Non-E Mode vs E-Mode — phân loại mất cân bằng 95/5", "",
        header, "", "## Bảng so sánh tổng hợp", "", table, "", "## Trung bình theo nhóm",
        "", summary, "", "## Kiểm chứng chống rò rỉ dữ liệu", "", leakage, "",
        *(extra_sections or []),
        "## Ghi chú từng phương pháp", "", notes, "",
        "## Tái lập", "", "```powershell", "python benchmark_imbalanced.py", "```", "",
    ]), encoding="utf-8")
    columns = ["group", "method", "kind", "roc_auc", "pr_auc", "f1_minority", "balanced_accuracy",
               "precision", "recall", "train_time_s", "n_train_before", "n_train_after",
               "ir_before", "ir_after", "tp", "fp", "fn", "tn", "leakage_ok", "note"]
    if HAS_PANDAS:  # pragma: no cover - phụ thuộc môi trường
        pd.DataFrame(list(rows))[columns].to_csv(csv_path, index=False, encoding="utf-8")
        return
    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_cv_artifacts(cv_rows: Sequence[Dict[str, Any]], n_splits: int) -> None:
    """Ghi `reports/benchmark_imbalanced_cv.md` (bảng theo phương pháp + chi tiết từng fold) và `.csv`."""
    import pathlib

    md_path = pathlib.Path(OUTPUT_CV_MD)
    csv_path = pathlib.Path(OUTPUT_CV_CSV)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    detail_header = ["Phương pháp", "fold", "n_train", "→ sau resample", "IR trước→sau",
                     "n_val", "% dương (val)", "ROC-AUC", "PR-AUC", "F1", "Bal. Acc",
                     "fold-val nguyên vẹn", "classifier chỉ thấy fold-train", "tỉ lệ lớp giữ nguyên"]
    detail = ["| " + " | ".join(detail_header) + " |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|"]
    for row in cv_rows:
        for fold in row["folds"]:
            checks = fold["checks"]
            detail.append("| " + " | ".join([
                row["method"], str(fold["fold"]), str(fold["n_train"]), str(fold["n_train_after"]),
                f"{fold['ir_before']:.1f}→{fold['ir_after']:.1f}", str(fold["n_val"]),
                f"{fold['pos_pct_val']:.2f}%", f"{fold['roc_auc']:.4f}", f"{fold['pr_auc']:.4f}",
                f"{fold['f1_minority']:.4f}", f"{fold['balanced_accuracy']:.4f}",
                *("PASS" if value else "FAIL" for value in checks.values())]) + " |")
    md_path.write_text("\n".join([
        f"# Đánh giá StratifiedKFold {n_splits} fold — Non-E Mode vs E-Mode", "",
        f"- Chia fold: `StratifiedKFold(n_splits={n_splits}, shuffle=True, random_state={SEED})` "
        "trên **train split**; test vẫn khoá riêng làm holdout chốt.", "",
        "- Resampling chỉ chạy trên **fold-train** (trong `imblearn.pipeline.Pipeline`); "
        "fold validation không bao giờ được resample.", "",
        "## 1. Trung bình ± độ lệch chuẩn qua các fold", "", render_cv_table(cv_rows), "",
        "## 2. Chi tiết từng fold", "", "\n".join(detail), "", "## 3. Tái lập", "",
        "```powershell", f"python benchmark_imbalanced.py --cv {n_splits}", "```", "",
    ]), encoding="utf-8")
    flat = [{**{k: v for k, v in fold.items() if k != "checks"},
             **{f"check_{k}": v for k, v in fold["checks"].items()},
             "group": row["group"], "method": row["method"]} for row in cv_rows
            for fold in row["folds"]]
    columns = ["group", "method", "fold", "n_train", "n_train_after", "ir_before", "ir_after",
               "n_val", "pos_pct_val", "roc_auc", "pr_auc", "f1_minority", "balanced_accuracy",
               "check_fold_val_nguyên_vẹn", "check_classifier_chỉ_thấy_fold_train",
               "check_tỉ_lệ_lớp_giữ_nguyên"]
    if HAS_PANDAS:  # pragma: no cover - phụ thuộc môi trường
        pd.DataFrame(flat)[columns].to_csv(csv_path, index=False, encoding="utf-8")
        return
    import csv

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in flat:
            writer.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Chạy benchmark đầy đủ và in bảng so sánh ra terminal."""
    for stream in (sys.stdout, sys.stderr):  # tránh UnicodeEncodeError trên console cp1252
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - stream không hỗ trợ
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--quick", action="store_true",
                        help="Bỏ nhóm tốn thời gian (EasyEnsemble) để chạy nhanh.")
    parser.add_argument("--cv", type=int, default=0,
                        help="Số fold StratifiedKFold (>=2) chạy trên TRAIN SPLIT để kiểm tra độ ổn "
                             "định; test vẫn khoá làm holdout. 0 = tắt (mặc định).")
    args = parser.parse_args(argv)

    import sklearn
    import imblearn

    log("=" * 78)
    log("BENCHMARK: Non-E Mode (resampling/cost-sensitive) vs E-Mode (ensemble)")
    log("=" * 78)
    log(f"Thư viện: numpy {np.__version__} · scikit-learn {sklearn.__version__} · "
        f"imbalanced-learn {imblearn.__version__} · pandas "
        f"{(pd.__version__ if HAS_PANDAS else 'KHÔNG có — dùng bảng tự sinh')}")

    boosting = "xgboost" if _xgboost_usable() else "lightgbm"
    if boosting != "xgboost" and not _lightgbm_usable():
        boosting = "hist_gradient_boosting"
    log(f"Backend boosting dùng cho nhóm Baseline/cost-sensitive: {BOOST_LABEL[boosting]}")

    X, y = make_dataset(args.n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    log(f"\n[1] Dữ liệu: {_fmt_distribution(_distribution(y))}")
    log(f"    train (80%): {_fmt_distribution(_distribution(y_train))}")
    log(f"    test  (20%): {_fmt_distribution(_distribution(y_test))}  ← cố định cho MỌI phương pháp")
    log(f"    Lưu ý: đoán “toàn bộ là lớp đa số” đã đạt "
        f"{100 - _distribution(y_test)['positive_pct']:.2f}% accuracy ⇒ không dùng Accuracy.")

    methods = build_methods(SEED, boosting)
    if args.quick:
        methods = [m for m in methods if "EasyEnsemble" not in m["name"]]

    log(f"\n[2] Chạy {len(methods)} phương pháp (resampling chỉ trong imblearn.pipeline.Pipeline)")
    rows: List[Dict[str, Any]] = []
    for index, method in enumerate(methods, start=1):
        try:
            row = evaluate_method(method, X_train, y_train, X_test, y_test)
        except Exception as exc:  # noqa: BLE001 - benchmark phải chạy tiếp khi 1 phương pháp lỗi
            log(f"  [{index:2d}/{len(methods)}] LỖI {method['name']}: "
                f"{type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        log(f"  [{index:2d}/{len(methods)}] {row['group']:<10s} | {row['method']:<46s} "
            f"| train {row['n_train_before']}→{row['n_train_after']} (IR {row['ir_before']:.1f}"
            f"→{row['ir_after']:.1f}) | PR-AUC {row['pr_auc']:.4f} F1 {row['f1_minority']:.4f} "
            f"BAcc {row['balanced_accuracy']:.4f} | {row['train_time_s']:.2f}s "
            f"| rò rỉ: {'PASS' if row['leakage_ok'] else 'FAIL'}")

    table = render_markdown(rows)
    summary = render_group_summary(rows)
    header = (f"- Dataset: `make_classification(n_samples={args.n_samples}, n_classes=2, "
              f"weights={CLASS_WEIGHTS}, random_state={SEED})`; Stratified {1 - TEST_SIZE:.0%}/"
              f"{TEST_SIZE:.0%} (seed {SEED})\n"
              f"- Backend boosting: **{BOOST_LABEL[boosting]}**; ngưỡng quyết định {THRESHOLD}")
    log("\n[3] BẢNG SO SÁNH TỔNG HỢP (test cố định, ngưỡng 0.5)\n")
    log(table)
    log("\n[4] TRUNG BÌNH THEO NHÓM\n")
    log(summary)
    failed = [r["method"] for r in rows if not r["leakage_ok"]]
    log(f"\n[5] Kiểm chứng chống rò rỉ (holdout): "
        f"{'TẤT CẢ PASS' if not failed else 'FAIL ở: ' + ', '.join(failed)}")

    extra_sections: List[str] = []
    if args.cv >= 2:
        log(f"\n[6] STRATIFIEDKFOLD {args.cv} FOLD TRÊN TRAIN SPLIT "
            f"({len(y_train)} mẫu) — test vẫn khoá làm holdout")
        cv_rows: List[Dict[str, Any]] = []
        for index, method in enumerate(methods, start=1):
            try:
                cv_row = evaluate_cv(method, X_train, y_train, n_splits=args.cv)
            except Exception as exc:  # noqa: BLE001
                log(f"  [{index:2d}/{len(methods)}] LỖI CV {method['name']}: "
                    f"{type(exc).__name__}: {exc}")
                continue
            cv_rows.append(cv_row)
            log(f"  [{index:2d}/{len(methods)}] {cv_row['group']:<10s} | "
                f"{cv_row['method']:<46s} | PR-AUC {cv_row['pr_auc_mean']:.4f}±"
                f"{cv_row['pr_auc_std']:.4f} | F1 {cv_row['f1_minority_mean']:.4f} | "
                f"BAcc {cv_row['balanced_accuracy_mean']:.4f} | rò rỉ: "
                f"{cv_row['n_folds_pass']}/{cv_row['n_splits']} PASS")
        cv_table = render_cv_table(cv_rows)
        log("\n" + cv_table)
        write_cv_artifacts(cv_rows, args.cv)
        cv_failed = [r["method"] for r in cv_rows if not r["leakage_ok"]]
        log(f"\n[7] Kiểm chứng chống rò rỉ (StratifiedKFold): "
            f"{'TẤT CẢ PASS' if not cv_failed else 'FAIL ở: ' + ', '.join(cv_failed)}")
        log(f"Đã ghi: {OUTPUT_CV_MD} và {OUTPUT_CV_CSV}")
        failed += [f"{name} (CV)" for name in cv_failed]
        extra_sections = [f"## StratifiedKFold {args.cv} fold (trên train split)", "",
                          cv_table, "",
                          f"Chi tiết từng fold + 3 kiểm chứng mỗi fold: `{OUTPUT_CV_MD}`.", ""]

    write_artifacts(rows, table, summary, header, extra_sections)
    log(f"\nĐã ghi: {OUTPUT_MD} và {OUTPUT_CSV}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())




