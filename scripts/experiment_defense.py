"""Thực nghiệm BỔ SUNG phục vụ phản biện: **LiteSVM** và **XGBoost** (không thuộc pipeline chính).

Vì sao có script này (đọc trước khi trích số liệu):
- Hội đồng hỏi *"Có cần chạy thêm LiteSVM không?"* và *"Sao không so từng lớp trên XGBoost?"*.
  Pipeline chính của đồ án (`scripts/run_all.py`) **cố ý** chỉ dùng 4 họ thuần scikit-learn
  (logistic · random forest · hist-gradient-boosting · MLP). Script này chạy THÊM hai thứ đó trên
  **đúng bộ dữ liệu của đồ án** để câu trả lời có số liệu, chứ không phải suy đoán.
- Script KHÔNG nằm trong `run_all.py` và KHÔNG ghi vào `docs/BAO-CAO.md`, nên không làm lệch
  "22 bước / 234 test / 98.200 phép kiểm tra" đã công bố. Artifact riêng:
  `reports/results/defense_models.json` + `.md`.

Ba phần:
1. `litesvm_real`   — LiteSVM (LinearSVC lề cực đại + SGD hinge, có/không `class_weight`) trên
   **corpus thật** (train 212 / val 32 / test 64), đặt cạnh 4 họ mô hình của đồ án.
2. `litesvm_synth`  — cùng LiteSVM trên **bộ giả lập 95/5** dùng lại CHÍNH XÁC bộ sinh dữ liệu của
   `benchmark_imbalanced.py` (`make_dataset`, seed 42, stratified 80/20) ⇒ so trực tiếp được với
   bảng đã có `reports/benchmark_imbalanced.md`.
3. `xgboost_deep`   — XGBoost trên corpus thật: mặc định · `scale_pos_weight` động · early stopping
   trên validation; kèm **báo cáo theo từng lớp**, **ma trận nhầm lẫn** ở ngưỡng 0,5 và ở ngưỡng vận
   hành của đồ án, và **khoảng cách train↔val** (bằng chứng overfitting).

Lệnh: python -m scripts.experiment_defense            (chạy cả 3 phần, ghi artifact)
      python -m scripts.experiment_defense --no-write (chỉ in ra màn hình)

Chống rò rỉ: mọi bước học (impute/scale/calibration/trọng số lớp/early stopping) chỉ dùng train
(riêng early stopping dùng validation ⇒ được ghi rõ trong artifact là "chọn mô hình bằng val,
KHÔNG dùng test").
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

#: Ngưỡng vận hành thật của đồ án (đọc từ artifact, không hardcode).
OPERATING_THRESHOLD_FALLBACK = 0.7879126873178737
OUTPUT_JSON = RESULTS_DIR / "defense_models.json"
OUTPUT_MD = RESULTS_DIR / "defense_models.md"
#: Số lần lặp lại phép đo độ trễ suy luận (lấy trung vị) và số dòng dùng để đo.
LATENCY_REPEATS = 5
LATENCY_ROWS = 1000
#: Tên hiển thị của 4 họ mô hình chính thức (khớp `docs/BAO-CAO.md`).
PRETTY = {"logistic": "Logistic Regression", "random_forest": "Random Forest",
          "hist_gradient_boosting": "HistGradientBoosting", "mlp": "MLP (mạng nơ-ron)"}


def log(message: str = "") -> None:
    """In log ra terminal."""
    print(message, flush=True)


def operating_threshold() -> float:
    """Ngưỡng vận hành đã chốt trong `summary.json` (fallback nếu thiếu file)."""
    path = RESULTS_DIR / "summary.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return float(json.load(f).get("best_threshold", OPERATING_THRESHOLD_FALLBACK))
    return OPERATING_THRESHOLD_FALLBACK


def make_litesvm(kind: str, *, balanced: bool = False, random_state: int = RANDOM_SEED) -> Pipeline:
    """Pipeline LiteSVM: `impute(median) → scale → [Platt] LiteSVM`.

    "LiteSVM" = phiên bản SVM **tuyến tính, chi phí thấp** (không kernel, không lưu support vector
    dày đặc): `LinearSVC` (lề cực đại, loss hinge, liblinear) hoặc `SGDClassifier(loss="hinge")`
    (giảm gradient ngẫu nhiên). Cả hai đều KHÔNG có `predict_proba` ⇒ bọc
    `CalibratedClassifierCV(method="sigmoid", cv=3)` để đổi margin thành xác suất (Platt scaling),
    nhờ đó tính được F1/ngưỡng cùng thang với các mô hình khác.

    Vì sao phải scale: SVM nhạy với đơn vị đo (nó tối ưu KHOẢNG CÁCH tới siêu phẳng), khác mô hình
    cây. Scaler nằm trong Pipeline nên chỉ học từ train ⇒ không rò rỉ.
    """
    if kind == "linearsvc":
        # dual="auto" để liblinear/scipy tự chọn chế độ tối ưu theo n_features vs n_samples.
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
    """Vector lớp của pipeline đã fit (lấy từ estimator bên trong calibration)."""
    return np.asarray(getattr(model, "classes_", [0, 1]))


def _per_class(y: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Báo cáo theo TỪNG LỚP + ma trận nhầm lẫn tại `threshold`."""
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
    """Độ trễ suy luận trung vị (ms cho `rows` dòng) — nhân bản mẫu cho đủ số dòng."""
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
    """Metric đầy đủ của một mô hình đã fit trên một tập bất kỳ."""
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
    """Fit mô hình và trả về thời gian huấn luyện (giây)."""
    start = time.perf_counter()
    model.fit(X, y)
    return float(time.perf_counter() - start)


def _row(name: str, group: str, model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
         X_va: np.ndarray, y_va: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
         threshold: float) -> Dict[str, Any]:
    """Fit + đánh giá một mô hình trên train/val/test; trả một hàng kết quả đầy đủ."""
    seconds = _train_timed(model, X_tr, y_tr)
    row = {"name": name, "group": group, "train_time_s": seconds,
           "latency_ms_per_1000": _latency_ms(model, X_te),
           "train": eval_model(model, X_tr, y_tr, threshold),
           "val": eval_model(model, X_va, y_va, threshold),
           "test": eval_model(model, X_te, y_te, threshold)}
    row["overfit_gap_auroc"] = float(row["train"]["auroc"] - row["val"]["auroc"])
    return row


def part_litesvm_real(threshold: float) -> Dict[str, Any]:
    """Phần 1: LiteSVM (lề cực đại) + 4 họ mô hình của đồ án trên **corpus thật**.

    Lưu ý phương pháp: LinearSVC/SGD hinge không có `predict_proba` nên được bọc
    `CalibratedClassifierCV(sigmoid, cv=3)` (Platt scaling) — hiệu chuẩn cũng chỉ học từ train.
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
    log(f"  [LiteSVM/real] {len(rows)} mô hình — test AUROC: " + ", ".join(
        f"{r['name'].split(' · ')[-1]}={r['test']['auroc']:.4f}" for r in rows))
    return {"protocol": "corpus thật (train 212 / val 32 / test 64); ngưỡng vận hành lấy từ summary.json",
            "threshold": threshold, "rows": rows}


def _synth_row(label: str, group: str, model: Any, X_tr: np.ndarray, y_tr: np.ndarray,
               X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, Any]:
    """Fit + metric khớp đúng cột của `reports/benchmark_imbalanced.md` (bộ giả lập 95/5)."""
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
    """Phần 2: LiteSVM trên **bộ giả lập 95/5** dùng lại đúng bộ sinh dữ liệu của
    `benchmark_imbalanced.make_dataset` (n=10.000, weights [0.95, 0.05], seed 42) ⇒ so được
    trực tiếp với bảng đã có trong `reports/benchmark_imbalanced.md`."""
    from benchmark_imbalanced import make_dataset

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
    log("  [LiteSVM/synthetic 95-5] xong")
    return {"protocol": "make_classification(n_samples=10000, weights=[0.95,0.05], seed 42), "
                        "stratified 80/20, ngưỡng 0.5",
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "positive_test": int((y_te == 1).sum()), "rows": rows}


def _xgboost(**kw: Any) -> Any:
    """`XGBClassifier` đã kiểm tra tương thích; trả `None` nếu môi trường không dùng được.

    Vì sao phải thử trước: XGBoost 2.x + scikit-learn 1.6 có thể lỗi runtime (repo từng gặp); khi đó
    phần XGBoost bị BỎ QUA và ghi rõ lý do thay vì im lặng trả số sai.
    """
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - thiếu thư viện
        log(f"  (bỏ qua XGBoost) không import được: {exc}")
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            XGBClassifier(n_estimators=5, n_jobs=1).fit(np.zeros((20, 3)), np.array([0, 1] * 10))
    except Exception as exc:  # pragma: no cover - xung đột phiên bản
        log(f"  (bỏ qua XGBoost) fit thử thất bại: {type(exc).__name__}: {exc}")
        return None
    return XGBClassifier(**kw)


def part_xgboost_real(threshold: float) -> Dict[str, Any]:
    """Phần 3: XGBoost (tham chiếu boosting) trên **corpus thật**, kèm báo cáo TỪNG LỚP.

    Ba biến thể: mặc định · `scale_pos_weight = n_âm/n_dương` (tính từ train) · early stopping với
    `eval_set` = validation ⇒ **số vòng được chọn bằng validation** (ghi rõ trong artifact).
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
        except Exception as exc:  # pragma: no cover - phiên bản xgboost khác
            log(f"  (bỏ qua importance) {exc}")
        rows.append(row)
    log("  [XGBoost/real] " + ", ".join(f"{r['name'][:22]}={r['test']['auroc']:.4f}" for r in rows))
    return {"protocol": "corpus thật; early stopping chọn số vòng trên validation (KHÔNG dùng test)",
            "threshold": threshold, "rows": rows}


def _fmt(value: Any, digits: int = 4) -> str:
    """Định dạng số cho bảng markdown (an toàn với `None`/NaN)."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def markdown(sections: Dict[str, Any], threshold: float) -> str:
    """Sinh bảng markdown cho cả ba phần (để đối chiếu nhanh khi phản biện)."""
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
    """Chạy 3 phần thực nghiệm và (tuỳ chọn) ghi artifact vào `reports/results/`."""
    ensure_dirs()
    threshold = operating_threshold()
    log("=== Thực nghiệm phản biện: LiteSVM & XGBoost (ngoài pipeline chính) ===")
    sections = {"threshold": threshold,
                "litesvm_real": part_litesvm_real(threshold),
                "litesvm_synthetic": part_litesvm_synthetic(),
                "xgboost_real": part_xgboost_real(threshold)}
    if write:
        OUTPUT_JSON.write_text(json.dumps(sections, ensure_ascii=False, indent=2, default=float) + "\n",
                               encoding="utf-8")
        OUTPUT_MD.write_text(markdown(sections, threshold), encoding="utf-8")
        log(f"Đã ghi: reports/results/{OUTPUT_JSON.name} + reports/results/{OUTPUT_MD.name}")
    return sections


def main(argv=None) -> int:
    """CLI: `python -m scripts.experiment_defense [--no-write]`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    run(write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
