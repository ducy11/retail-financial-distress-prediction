"""Supplementary experiments for the defense: LiteSVM and XGBoost outside the main pipeline.

The main pipeline deliberately uses four pure scikit-learn families, so this script runs LiteSVM on the
real corpus and on the 95/5 synthetic set, plus XGBoost with per-class reporting, confusion matrices and
train-to-validation gaps. It stays out of `run_all` and writes `results/defense_models.{json,md}`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting.config import RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio  # noqa: E402
from forecasting.data_loader import load_prepared  # noqa: E402
from forecasting.evaluation import evaluate_proba, metrics_at_threshold  # noqa: E402
from forecasting.features import build_feature_matrix, extract_labels  # noqa: E402
from forecasting.models import DEFAULT_MODEL_ORDER, make_model, predict_proba  # noqa: E402

#: Real operating threshold, read from the artifact rather than hard-coded.
OPERATING_THRESHOLD_FALLBACK = 0.7879126873178737
OUTPUT_JSON = RESULTS_DIR / "defense_models.json"
OUTPUT_MD = RESULTS_DIR / "defense_models.md"
#: Repeat count for the median inference latency and the number of rows measured.
LATENCY_REPEATS = 5
LATENCY_ROWS = 1000
#: Display names of the four official model families, matching `docs/BAO-CAO.md`.
PRETTY = {"logistic": "Logistic Regression", "random_forest": "Random Forest",
          "hist_gradient_boosting": "HistGradientBoosting", "mlp": "MLP (mạng nơ-ron)"}


def log(message: str = "") -> None:
    """Print a log line to the terminal."""
    print(message, flush=True)


def operating_threshold() -> float:
    """Operating threshold frozen in `summary.json`, with a fallback when the file is absent."""
    path = RESULTS_DIR / "summary.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return float(json.load(f).get("best_threshold", OPERATING_THRESHOLD_FALLBACK))
    return OPERATING_THRESHOLD_FALLBACK


def make_litesvm(kind: str, *, balanced: bool = False, random_state: int = RANDOM_SEED) -> Pipeline:
    """Pipeline of impute, scale and a calibrated LiteSVM.

    LiteSVM means a cheap linear SVM, either `LinearSVC` or `SGDClassifier` with the hinge loss, which
    lack `predict_proba` and are wrapped in `CalibratedClassifierCV` for Platt scaling. Scaling is needed
    because the SVM optimises a distance, and it lives in the pipeline so it learns from train only.
    """
    if kind == "linearsvc":
        # dual="auto" lets liblinear pick the solver mode from the feature-to-sample ratio.
        base: Any = LinearSVC(C=1.0, dual="auto", max_iter=5000,
                              class_weight="balanced" if balanced else None,
                              random_state=random_state)
    elif kind == "sgd_hinge":
        base = SGDClassifier(loss="hinge", alpha=1e-4, max_iter=2000, tol=1e-3,
                             class_weight="balanced" if balanced else None,
                             random_state=random_state)
    else:
        raise KeyError(f"LiteSVM chưa hỗ trợ: {kind!r}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("calibrate", CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)),
    ])


def classes_from(model: Pipeline) -> np.ndarray:
    """Class vector of a fitted pipeline, taken from the estimator inside the calibration wrapper."""
    return np.asarray(getattr(model, "classes_", [0, 1]))


def _per_class(y: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Per-class metrics plus the confusion matrix at the given threshold."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "per_class": {
            "0": {"precision": float(precision[0]), "recall": float(recall[0]),
                  "f1": float(f1[0]), "support": int(support[0])},
            "1": {"precision": float(precision[1]), "recall": float(recall[1]),
                  "f1": float(f1[1]), "support": int(support[1])},
        },
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _latency_ms(model: Any, X: np.ndarray, rows: int = LATENCY_ROWS) -> float:
    """Median inference latency in milliseconds for `rows` rows, tiling samples to reach that count."""
    if len(X) == 0:
        return float("nan")
    reps = int(np.ceil(rows / len(X)))
    big = np.tile(X, (reps, 1))[:rows]
    times: List[float] = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        _ = predict_proba(model, big)
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times))


def eval_model(model: Any, X: np.ndarray, y: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Every metric for a fitted model on an arbitrary split."""
    proba = predict_proba(model, X)
    full = evaluate_proba(y, proba, operating_threshold=threshold)
    return {
        "n": int(len(y)),
        "auroc": full["auroc"],
        "average_precision": full["average_precision"],
        "brier": full["brier"],
        "at_0.5": full["by_threshold"][[b["threshold"] for b in full["by_threshold"]].index(0.5)]
        if any(b["threshold"] == 0.5 for b in full["by_threshold"]) else metrics_at_threshold(y, proba, 0.5),
        "at_operating": metrics_at_threshold(y, proba, threshold),
        "per_class_at_0.5": _per_class(y, proba, 0.5),
        "per_class_at_operating": _per_class(y, proba, threshold),
    }


def _train_timed(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    """Fit the model and return the training time in seconds."""
    start = time.perf_counter()
    model.fit(X, y)
    return float(time.perf_counter() - start)


def _row(name: str, group: str, model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
         X_va: np.ndarray, y_va: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
         threshold: float) -> Dict[str, Any]:
    """Fit and evaluate one model on train, validation and test, returning a complete result row."""
    seconds = _train_timed(model, X_tr, y_tr)
    row = {"name": name, "group": group, "train_time_s": seconds,
           "latency_ms_per_1000": _latency_ms(model, X_te),
           "train": eval_model(model, X_tr, y_tr, threshold),
           "val": eval_model(model, X_va, y_va, threshold),
           "test": eval_model(model, X_te, y_te, threshold)}
    row["overfit_gap_auroc"] = float(row["train"]["auroc"] - row["val"]["auroc"])
    return row


def part_litesvm_real(threshold: float) -> Dict[str, Any]:
    """First part: LiteSVM next to the four project families on the real corpus.

    The hinge-loss estimators expose no `predict_proba`, so they are wrapped in
    `CalibratedClassifierCV` with Platt scaling, itself fitted on train only.
    """
    train_s, val_s, test_s = (load_prepared(s) for s in ("train", "validation", "test"))
    X_tr, y_tr = build_feature_matrix(train_s), extract_labels(train_s)
    X_va, y_va = build_feature_matrix(val_s), extract_labels(val_s)
    X_te, y_te = build_feature_matrix(test_s), extract_labels(test_s)
    variants = [("LinearSVC (lề cực đại)", "linearsvc", False),
                ("LinearSVC + class_weight=balanced", "linearsvc", True),
                ("SGD hinge (giảm gradient)", "sgd_hinge", False),
                ("SGD hinge + class_weight=balanced", "sgd_hinge", True)]
    rows = [_row(f"LiteSVM · {label}", "litesvm", make_litesvm(kind, balanced=balanced),
                 X_tr, y_tr, X_va, y_va, X_te, y_te, threshold)
            for label, kind, balanced in variants]
    for name in DEFAULT_MODEL_ORDER:
        rows.append(_row(PRETTY[name], "project", make_model(name),
                         X_tr, y_tr, X_va, y_va, X_te, y_te, threshold))
    log(f"  [LiteSVM/real] {len(rows)} models — test AUROC: " + ", ".join(
        f"{r['name'].split(' · ')[-1]}={r['test']['auroc']:.4f}" for r in rows))
    return {"protocol": "corpus thật (train 212 / val 32 / test 64); ngưỡng vận hành lấy từ summary.json",
            "threshold": threshold, "rows": rows}


def _synth_row(label: str, group: str, model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, Any]:
    """Fit and score with the exact columns of `reports/benchmark_imbalanced.md` on the 95/5 set."""
    seconds = _train_timed(model, X_tr, y_tr)
    proba = predict_proba(model, X_te)
    pred = (proba >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_te, pred, labels=[0, 1],
                                                               zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    return {"name": label, "group": group, "train_time_s": seconds,
            "roc_auc": float(roc_auc_score(y_te, proba)),
            "pr_auc": float(average_precision_score(y_te, proba)),
            "f1_minority": float(f1[1]),
            "balanced_accuracy": float((recall[1] + recall[0]) / 2.0),
            "precision": float(precision[1]), "recall": float(recall[1]),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def part_litesvm_synthetic() -> Dict[str, Any]:
    """Second part: LiteSVM on the 95/5 synthetic set, reusing `labs.benchmark.make_dataset`.

    The generator uses n=10000, weights [0.95, 0.05] and seed 42, so the numbers line up directly with
    the existing table in `reports/benchmark_imbalanced.md`.
    """
    from labs.benchmark import make_dataset

    X, y = make_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y,
                                              random_state=RANDOM_SEED)
    rows = [
        _synth_row("Logistic Regression (không can thiệp)", "baseline",
                   Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler()),
                             ("model", LogisticRegression(max_iter=2000,
                                                          random_state=RANDOM_SEED))]),
                   X_tr, y_tr, X_te, y_te),
    ]
    for label, kind, balanced in (("LiteSVM · LinearSVC", "linearsvc", False),
                                  ("LiteSVM · SGD hinge", "sgd_hinge", False),
                                  ("LiteSVM · LinearSVC + balanced", "linearsvc", True)):
        rows.append(_synth_row(label, "litesvm", make_litesvm(kind, balanced=balanced),
                               X_tr, y_tr, X_te, y_te))
    log("  [LiteSVM/synthetic 95-5] done")
    return {"protocol": "make_classification(n_samples=10000, weights=[0.95,0.05], seed 42), "
                        "stratified 80/20, ngưỡng 0.5",
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "positive_test": int((y_te == 1).sum()), "rows": rows}


def _xgboost(**kw: Any) -> Any:
    """Return a compatibility-checked `XGBClassifier`, or None when the environment cannot run it.

    XGBoost 2.x with scikit-learn 1.6 can fail at runtime, so a tiny probe fit runs first and the
    XGBoost section is skipped with an explicit reason instead of returning wrong numbers.
    """
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - library missing
        log(f"  (skipping XGBoost) import failed: {exc}")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            XGBClassifier(n_estimators=5, n_jobs=1).fit(np.zeros((20, 3)), np.array([0, 1] * 10))
    except Exception as exc:  # pragma: no cover - version conflict
        log(f"  (skipping XGBoost) probe fit failed: {type(exc).__name__}: {exc}")
        return None
    return XGBClassifier(**kw)


def part_xgboost_real(threshold: float) -> Dict[str, Any]:
    """Third part: XGBoost as a boosting reference on the real corpus, with per-class reporting.

    Three variants are run: default, `scale_pos_weight` from the train class ratio, and early stopping
    with validation as the eval set, so the round count is chosen on validation and stated in the
    artifact.
    """
    from forecasting.features import feature_names

    train_s, val_s, test_s = (load_prepared(s) for s in ("train", "validation", "test"))
    X_tr, y_tr = build_feature_matrix(train_s), extract_labels(train_s)
    X_va, y_va = build_feature_matrix(val_s), extract_labels(val_s)
    X_te, y_te = build_feature_matrix(test_s), extract_labels(test_s)
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    weight = n_neg / n_pos if n_pos else 1.0
    common = dict(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9,
                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1,
                  random_state=RANDOM_SEED, eval_metric="logloss")
    rows: List[Dict[str, Any]] = []
    for label, extra in (("XGBoost (mặc định, không can thiệp)", {}),
                         (f"XGBoost + scale_pos_weight = {weight:.3f}", {"scale_pos_weight": weight})):
        model = _xgboost(**common, **extra)
        if model is None:
            return {"skipped": "môi trường không dùng được XGBoost", "rows": []}
        rows.append(_row(label, "xgboost", model, X_tr, y_tr, X_va, y_va, X_te, y_te, threshold))

    es_params = dict(common, n_estimators=400, early_stopping_rounds=30, scale_pos_weight=weight)
    model = _xgboost(**es_params)
    if model is not None:
        start = time.perf_counter()
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        seconds = float(time.perf_counter() - start)
        row = {"name": "XGBoost + scale_pos_weight + early stopping (eval = validation)",
               "group": "xgboost", "train_time_s": seconds,
               "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
               "latency_ms_per_1000": _latency_ms(model, X_te),
               "train": eval_model(model, X_tr, y_tr, threshold),
               "val": eval_model(model, X_va, y_va, threshold),
               "test": eval_model(model, X_te, y_te, threshold)}
        row["overfit_gap_auroc"] = float(row["train"]["auroc"] - row["val"]["auroc"])
        try:
            gain = model.get_booster().get_score(importance_type="gain")
            names = feature_names()
            top = sorted(((names[int(str(k)[1:])], v) for k, v in gain.items() if str(k).startswith("f")),
                         key=lambda kv: -kv[1])[:10]
            row["importance_gain_top10"] = [{"feature": f, "gain": float(g)} for f, g in top]
        except Exception as exc:  # pragma: no cover - different xgboost version
            log(f"  (skipping importance) {exc}")
        rows.append(row)
    log("  [XGBoost/real] " + ", ".join(f"{r['name'][:22]}={r['test']['auroc']:.4f}" for r in rows))
    return {"protocol": "corpus thật; early stopping chọn số vòng trên validation (KHÔNG dùng test)",
            "threshold": threshold, "rows": rows}


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a number for the markdown tables, tolerating None and NaN."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def markdown(sections: Dict[str, Any], threshold: float) -> str:
    """Render the markdown tables for all three parts, for quick cross-checking during the defense."""
    lines = ["# Thực nghiệm phản biện: LiteSVM & XGBoost (ngoài pipeline chính)", "",
             f"- Ngưỡng vận hành của đồ án (đọc từ `summary.json`): **{threshold:.4f}**",
             "- Chống rò rỉ: impute/scale/hiệu chuẩn/trọng số lớp chỉ học từ train; riêng biến thể",
             "  early stopping chọn số vòng trên **validation** (ghi rõ, không dùng test).", "",
             "## 1. Corpus thật — LiteSVM đặt cạnh 4 họ mô hình của đồ án", "",
             "| Mô hình | Nhóm | Test AUROC | Test AP | F1@0,5 | F1@ngưỡng vận hành | Bal.Acc@0,5 | MCC@ngưỡng | TN/FP/FN/TP | Giây | ms/1000 dòng | Gap train−val |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in sections["litesvm_real"]["rows"]:
        c = r["test"]["per_class_at_operating"]["confusion"]
        a5 = r["test"]["at_0.5"]
        lines.append(
            f"| {r['name']} | {r['group']} | {_fmt(r['test']['auroc'])} | "
            f"{_fmt(r['test']['average_precision'])} | {_fmt(a5['f1'])} | "
            f"{_fmt(r['test']['at_operating']['f1'])} | "
            f"{_fmt((a5['recall'] + a5['specificity']) / 2)} | "
            f"{_fmt(r['test']['at_operating']['mcc'])} | "
            f"{c['tn']}/{c['fp']}/{c['fn']}/{c['tp']} | "
            f"{_fmt(r['train_time_s'], 3)} | {_fmt(r['latency_ms_per_1000'], 2)} | "
            f"{_fmt(r['overfit_gap_auroc'])} |")
    lines += ["", "## 2. Bộ giả lập 95/5 — LiteSVM trong bảng benchmark mất cân bằng", "",
              "| Phương pháp | Nhóm | ROC-AUC | PR-AUC | F1 thiểu số | Bal. Acc | Giây |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in sections["litesvm_synthetic"]["rows"]:
        lines.append(f"| {r['name']} | {r['group']} | {_fmt(r['roc_auc'])} | {_fmt(r['pr_auc'])} | "
                     f"{_fmt(r['f1_minority'])} | {_fmt(r['balanced_accuracy'])} | "
                     f"{_fmt(r['train_time_s'], 3)} |")
    xgb = sections.get("xgboost_real", {})
    if xgb.get("rows"):
        lines += ["", "## 3. XGBoost (tham chiếu boosting) — báo cáo TỪNG LỚP tại ngưỡng vận hành", "",
                  "| Biến thể | Test AUROC | Test AP | Class 0 (P/R/F1) | Class 1 (P/R/F1) | TN/FP/FN/TP | Gap train−val |",
                  "|---|---:|---:|---|---|---|---:|"]
        for r in xgb["rows"]:
            cls = r["test"]["per_class_at_operating"]["per_class"]
            c = r["test"]["per_class_at_operating"]["confusion"]
            lines.append(f"| {r['name']} | {_fmt(r['test']['auroc'])} | "
                         f"{_fmt(r['test']['average_precision'])} | "
                         f"{_fmt(cls['0']['precision'])}/{_fmt(cls['0']['recall'])}/{_fmt(cls['0']['f1'])} | "
                         f"{_fmt(cls['1']['precision'])}/{_fmt(cls['1']['recall'])}/{_fmt(cls['1']['f1'])} | "
                         f"{c['tn']}/{c['fp']}/{c['fn']}/{c['tp']} | {_fmt(r['overfit_gap_auroc'])} |")
    lines += ["", "## 4. Tái lập", "", "```powershell", "python -m scripts.experiment_defense", "```"]
    return "\n".join(lines) + "\n"


def run(write: bool = True) -> Dict[str, Any]:
    """Run the three experiment parts and optionally write the artifacts under `reports/results/`."""
    ensure_dirs()
    threshold = operating_threshold()
    log("=== Defense experiments: LiteSVM and XGBoost outside the main pipeline ===")
    sections = {"threshold": threshold,
                "litesvm_real": part_litesvm_real(threshold),
                "litesvm_synthetic": part_litesvm_synthetic(),
                "xgboost_real": part_xgboost_real(threshold)}
    if write:
        OUTPUT_JSON.write_text(json.dumps(sections, ensure_ascii=False, indent=2, default=float) + "\n",
                               encoding="utf-8")
        OUTPUT_MD.write_text(markdown(sections, threshold), encoding="utf-8")
        log(f"Wrote: reports/results/{OUTPUT_JSON.name} + reports/results/{OUTPUT_MD.name}")
    return sections


def main(argv=None) -> int:
    """Command-line entry point for `scripts.experiment_defense`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    run(write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
