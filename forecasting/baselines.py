"""Control baselines required when reporting metrics.

Runs four references from weakest to strongest: majority-class dummy, `ticker_prior` using the same
company's train distress rate, a single-feature logistic model, and the public Altman Z'' rule. A model
that cannot beat `ticker_prior` is only recognizing the company. Writes `results/baselines.json`.
"""
from __future__ import annotations

import json
import math
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
from .labels import ALTMAN_DISTRESS_BELOW, altman_z_double_prime
from .models import DEFAULT_MODEL_ORDER, make_model, predict_proba

#: Single feature used for the "1 indicator" baseline.
SINGLE_FEATURE = "debt_to_assets_latest"

#: Sigmoid slope turning Z'' into a raw probability (for AUROC/AP on the same 0..1 scale; no fitted parameters).
ALTMAN_SLOPE = 0.5


def ticker_prior(samples: List[Dict[str, Any]], labels: np.ndarray) -> Dict[str, float]:
    """Distress rate per company, estimated on train (never uses val/test)."""
    buckets: Dict[str, List[int]] = {}
    for s, y in zip(samples, labels):
        buckets.setdefault(s[GROUP_KEY], []).append(int(y))
    return {t: float(np.mean(v)) for t, v in sorted(buckets.items())}


def predict_ticker_prior(samples: List[Dict[str, Any]], prior: Dict[str, float],
                         default: float = 0.5) -> np.ndarray:
    """Predicted probability = that company's distress rate in train."""
    return np.asarray([prior.get(s[GROUP_KEY], default) for s in samples], dtype=float)


def altman_z_probability(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Altman Z''-score of the latest published quarter mapped to a risk probability by a public rule.

    Every distress-forecasting report must compare against the classic Altman threshold (1968, 2000).
    Nothing is learned here: Z'' is passed through sigmoid((Z'' - threshold) / slope), so this is a
    data-free baseline.

    Uses the same window as `forecasting.features`: `history[-1]` is the latest quarter with
    `available_on <= as_of`, so the target quarter's data never leaks. Missing components -> 0.5
    (neutral, no metric inflation).
    """
    out: List[float] = []
    for sample in samples:
        z = altman_z_double_prime(sample["request"]["history"][-1])
        out.append(0.5 if z is None else
                   1.0 / (1.0 + math.exp((z - ALTMAN_DISTRESS_BELOW) / ALTMAN_SLOPE)))
    return np.asarray(out, dtype=float)


def fit_dummy(samples: List[Dict[str, Any]], labels: np.ndarray):
    """DummyClassifier(most_frequent) - the lower bound of every model."""
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
    model.fit(build_feature_matrix(samples), labels)
    return model


def fit_single_feature(train_samples: List[Dict[str, Any]], labels: np.ndarray,
                       name: str = SINGLE_FEATURE) -> Tuple[Any, int]:
    """Logistic on 1 feature (impute + scale); returns (model, column index)."""
    names = feature_names()
    if name not in names:
        raise KeyError(f"Feature {name!r} does not exist; valid examples: {names[:5]}")
    j = names.index(name)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED)),
    ])
    model.fit(build_feature_matrix(train_samples)[:, [j]], labels)
    return model, j


def _metric_block(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compact metric block for the comparison table (at threshold 0.5 + AUROC/AP/Brier)."""
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
    """Evaluate baselines + reference models on validation and test."""
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
        # RULE baseline (learns no parameters): Altman Z'' of the latest published quarter.
        ("rule[altman_z_double_prime<1.1]",
         altman_z_probability(val_s), altman_z_probability(test_s)),
    ]
    for name in DEFAULT_MODEL_ORDER:
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

    print(f"  {'system':34s} {'AUROC':>6s} {'AP':>6s} {'F1':>6s} {'macroF1':>8s} {'Acc':>6s}")
    for r in rows:
        t = r["test"]
        print(f"  {r['baseline']:34s} {t['auroc']:6.3f} {t['average_precision']:6.3f} "
              f"{t['at_0.5']['f1']:6.3f} {t['at_0.5']['macro_f1']:8.3f} {t['at_0.5']['accuracy']:6.3f}")
    print("  (table above is on TEST, threshold 0.5)")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    print("=== Control baselines (fit on train -> score val/test) ===")
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
