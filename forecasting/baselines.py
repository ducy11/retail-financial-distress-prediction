"""Baseline đối chứng — không được thiếu khi báo cáo metric.

Ba baseline theo thứ tự mạnh dần:
1. `dummy_most_frequent` — luôn dự đoán lớp đa số (thước đo "không học gì").
2. `ticker_prior` — xác suất = tỷ lệ distress trung bình của CHÍNH công ty đó trong train.
   Baseline mạnh nhất, cần thiết vì nhãn có thể gần như là thuộc tính của công ty: nếu mô
   hình không vượt được baseline này thì nó chỉ đang "nhớ mặt công ty".
3. `single_feature[debt_to_assets_latest]` — logistic trên 1 feature duy nhất.

Lệnh: python -m forecasting.baselines  → ghi reports/results/baselines.json
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import GROUP_KEY, RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from .data_loader import load_prepared
from .evaluation import evaluate_proba
from .features import build_feature_matrix, extract_labels, feature_names
from .models import MODEL_REGISTRY, make_model, predict_proba

#: Feature đơn lẻ dùng cho baseline "1 chỉ tiêu".
SINGLE_FEATURE = "debt_to_assets_latest"


def ticker_prior(samples: List[Dict[str, Any]], labels: np.ndarray) -> Dict[str, float]:
    """Tỷ lệ distress theo công ty, ước lượng trên train (không dùng val/test)."""
    buckets: Dict[str, List[int]] = {}
    for s, y in zip(samples, labels):
        buckets.setdefault(s[GROUP_KEY], []).append(int(y))
    return {t: float(np.mean(v)) for t, v in sorted(buckets.items())}


def predict_ticker_prior(samples: List[Dict[str, Any]], prior: Dict[str, float],
                         default: float = 0.5) -> np.ndarray:
    """Xác suất dự đoán = tỷ lệ distress của công ty đó trong train."""
    return np.asarray([prior.get(s[GROUP_KEY], default) for s in samples], dtype=float)


def fit_dummy(samples: List[Dict[str, Any]], labels: np.ndarray):
    """DummyClassifier(most_frequent) — cận dưới của mọi mô hình."""
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
    model.fit(build_feature_matrix(samples), labels)
    return model


def fit_single_feature(train_samples: List[Dict[str, Any]], labels: np.ndarray,
                       name: str = SINGLE_FEATURE) -> Tuple[Any, int]:
    """Logistic trên 1 feature (impute + scale); trả (model, chỉ số cột)."""
    names = feature_names()
    if name not in names:
        raise KeyError(f"Không có feature {name!r}; ví dụ hợp lệ: {names[:5]}")
    j = names.index(name)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)),
    ])
    model.fit(build_feature_matrix(train_samples)[:, [j]], labels)
    return model, j


def _metric_block(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Block metric gọn cho bảng so sánh (tại threshold 0.5 + AUROC/AP/Brier)."""
    full = evaluate_proba(y, proba)
    at = next((b for b in full["by_threshold"] if abs(b["threshold"] - threshold) < 1e-9), None)
    if at is None:
        from .evaluation import metrics_at_threshold

        at = metrics_at_threshold(y, proba, threshold)
    keys = ("accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1",
            "specificity", "mcc", "tn", "fp", "fn", "tp")
    return {
        "n": int(len(y)),
        "observed_distress": float(np.mean(y)),
        "auroc": full["auroc"],
        "average_precision": full["average_precision"],
        "brier": full["brier"],
        "at_0.5": {k: at[k] for k in keys},
    }


def run() -> Dict[str, Any]:
    """Đánh giá baseline + mô hình tham chiếu trên validation và test."""
    ensure_dirs()
    train_s = load_prepared("train")
    val_s = load_prepared("validation")
    test_s = load_prepared("test")
    y_tr, y_va, y_te = (extract_labels(s) for s in (train_s, val_s, test_s))

    prior = ticker_prior(train_s, y_tr)
    X_va, X_te = build_feature_matrix(val_s), build_feature_matrix(test_s)
    single, j = fit_single_feature(train_s, y_tr)

    predictors: List[Tuple[str, np.ndarray, np.ndarray]] = [
        ("dummy_most_frequent",
         fit_dummy(train_s, y_tr).predict_proba(X_va)[:, 1],
         fit_dummy(train_s, y_tr).predict_proba(X_te)[:, 1]),
        ("ticker_prior", predict_ticker_prior(val_s, prior), predict_ticker_prior(test_s, prior)),
        (f"single_feature[{SINGLE_FEATURE}]",
         predict_proba(single, X_va[:, [j]]), predict_proba(single, X_te[:, [j]])),
    ]
    for name in ("logistic", "random_forest", "hist_gradient_boosting", "lightgbm"):
        if name not in MODEL_REGISTRY:  # lightgbm/xgboost là phụ thuộc tuỳ chọn của môi trường
            continue
        model = make_model(name)
        model.fit(build_feature_matrix(train_s), y_tr)
        predictors.append((f"model[{name}]", predict_proba(model, X_va), predict_proba(model, X_te)))

    rows = [{"baseline": tag,
             "val": _metric_block(y_va, p_va),
             "test": _metric_block(y_te, p_te)}
            for tag, p_va, p_te in predictors]

    out: Dict[str, Any] = {"ticker_prior_train": prior, "threshold_for_table": 0.5, "rows": rows}
    (RESULTS_DIR / "baselines.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(f"  {'hệ thống':34s} {'AUROC':>6s} {'AP':>6s} {'F1':>6s} {'macroF1':>8s} {'Acc':>6s}")
    for r in rows:
        t = r["test"]
        print(f"  {r['baseline']:34s} {t['auroc']:6.3f} {t['average_precision']:6.3f} "
              f"{t['at_0.5']['f1']:6.3f} {t['at_0.5']['macro_f1']:8.3f} {t['at_0.5']['accuracy']:6.3f}")
    print("  (bảng in trên TEST, threshold 0.5)")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    print("=== Baseline đối chứng (fit trên train → đánh giá val/test) ===")
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
