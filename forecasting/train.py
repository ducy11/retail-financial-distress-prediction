"""Train candidate models and select one by cross-company average precision.

Fits each registered family on train, scores validation metrics plus out-of-fold average precision with
GroupKFold, and picks the winner by cross-company AP, then best-F1, AP, AUROC and overfit gap. Writes
`results/summary.json` and `models/best.joblib`. Run with `python -m forecasting.train`.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import (FIGURES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, ensure_dirs,
                     ensure_utf8_stdio)
from .data_loader import load_prepared
from .evaluation import (best_f1_point, cost_optimal_threshold, evaluate_proba,
                         metrics_at_threshold, threshold_from_validation)
from .features import build_feature_matrix, extract_labels, feature_names
from .models import DEFAULT_MODEL_ORDER, MODEL_REGISTRY, make_model, predict_proba
from .validation import bootstrap_ci

#: Default models = the project's 4 families (logistic, random forest, hist gradient boosting, MLP).
DEFAULT_MODELS = [m for m in DEFAULT_MODEL_ORDER if m in MODEL_REGISTRY]


def _safe_metric(x) -> float:
    """Coerce a metric (possibly None/NaN) to float; None/NaN -> -1 so ranking stays safe."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if v != v else v  # v != v means NaN


def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for logs (None/NaN -> '—')."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if number != number else f"{number:.{digits}f}"


def _rank(row: Dict[str, Any]):
    """Model ranking key: cross-company AP (GroupKFold) -> best-F1(val) -> AP(val) -> AUROC(val)
    -> smallest overfit gap.

    Cross-company AP leads because in-domain metrics are dominated by recognizing the company
    (ticker-prior baseline = 0.986), so ranking by in-domain score answers the wrong question.
    See `docs/ke-hoach-tiep-theo.md` (P0-#2) and report section 6.4.
    """
    return (_safe_metric(row.get("cross_company_ap")), _safe_metric(row["best_f1_val"]),
            _safe_metric(row["average_precision"]), _safe_metric(row["auroc"]),
            -_safe_metric(row["overfit_gap_f1"]))


def train_split(model_name: str, X_train, y_train, X_val, y_val, **params):
    """Fit one model; return (metric row, pipeline).

    The row holds TRAIN and VALIDATION metrics at the same 0.5 threshold (to compare overfitting),
    the best-F1 point on validation and the cost-optimal threshold on validation. `params` overrides
    hyperparameters.
    """
    model = make_model(model_name, **params)
    model.fit(X_train, y_train)
    proba_tr = predict_proba(model, X_train)
    proba_val = predict_proba(model, X_val)
    tr = metrics_at_threshold(y_train, proba_tr, 0.5)
    va = metrics_at_threshold(y_val, proba_val, 0.5)
    val_full = evaluate_proba(y_val, proba_val)
    best = best_f1_point(y_val, proba_val) or {"threshold": 0.5, "f1": float("nan"),
                                               "precision": float("nan"), "recall": float("nan")}
    train_auroc = (float(roc_auc_score(y_train, proba_tr))
                   if len(set(np.asarray(y_train).tolist())) > 1 else float("nan"))
    keys_tr = ("accuracy", "precision", "recall", "f1", "macro_f1")
    keys_va = ("accuracy", "precision", "recall", "f1", "macro_f1", "specificity", "mcc")
    return {
        "model": model_name,
        "params": dict(params) if params else None,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "prevalence_val": float(np.mean(y_val)),
        "auroc": val_full["auroc"],
        "average_precision": val_full["average_precision"],
        "brier_val": val_full["brier"],
        "train_auroc": train_auroc,
        "train_at_0.5": {k: tr[k] for k in keys_tr},
        "val_at_0.5": {k: va[k] for k in keys_va},
        "overfit_gap_f1": float(tr["f1"] - va["f1"]),
        "overfit_gap_auroc": float(train_auroc - val_full["auroc"]),
        "best_f1_val": best["f1"],
        "best_threshold": best["threshold"],
        "precision_at_best": best["precision"],
        "recall_at_best": best["recall"],
        "cost_optimal_val": cost_optimal_threshold(y_val, proba_val),
    }, model


def cross_company_metrics(model_name: str, samples: List[Dict[str, Any]],
                          n_splits: int = 4) -> Dict[str, Any]:
    """Out-of-fold AP and AUROC when splitting by company (GroupKFold), the main selection criterion.

    The in-domain test set contains exactly the companies seen in train, so every in-domain metric is
    dominated by recognizing the company (`ticker_prior` baseline = 0.986; see
    `forecasting/validation.py`). Cross-company AP measures the relevant question: how the model does
    on a company it never saw.

    Selection data is train plus validation (test is never touched). Each GroupKFold fold holds out a
    whole company, so its labels never appear in the fit portion.
    """
    from .validation import grouped_cv

    result = grouped_cv([model_name], samples, n_splits=n_splits)
    block = result.get(model_name) or {}
    return {
        "cross_company_ap": block.get("oof_average_precision"),
        "cross_company_auroc": block.get("oof_auroc"),
        "cross_company_n_oof": block.get("n_oof"),
        "cross_company_n_splits": result.get("n_splits"),
    }


def run(model_names: List[str] | None = None) -> Dict[str, Any]:
    """Train candidate models on train and pick the winner on validation."""
    ensure_dirs()
    model_names = list(model_names or DEFAULT_MODELS)
    train_samples = load_prepared("train")
    val_samples = load_prepared("validation")

    X_train = build_feature_matrix(train_samples)
    y_train = extract_labels(train_samples)
    X_val = build_feature_matrix(val_samples)
    y_val = extract_labels(val_samples)

    feats = feature_names()
    print(f"Train {X_train.shape}, Validation {X_val.shape}; n_features={len(feats)}")

    summaries: List[Dict[str, Any]] = []
    best_name, best_row, best_model = None, None, None
    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"  (skipped) model not in registry: {name}")
            continue
        print(f"  Fit {name} ...")
        try:
            row, model = train_split(name, X_train, y_train, X_val, y_val)
        except Exception as e:  # noqa: BLE001 - sklearn fit failure (bad data / too few rows)
            print(f"    Fit ERROR {name}: {e}. Skipping this model.")
            continue
        row["n_features"] = len(feats)
        # Primary selection signal: out-of-fold AP by company (train and validation; test untouched).
        row.update(cross_company_metrics(name, train_samples + val_samples, n_splits=4))
        summaries.append(row)
        print(f"    train: AUROC={row['train_auroc']:.3f} F1@0.5={row['train_at_0.5']['f1']:.3f}"
              f" | val: AUROC={row['auroc']:.3f} AP={row['average_precision']:.3f} "
              f"F1@0.5={row['val_at_0.5']['f1']:.3f} bestF1@{row['best_threshold']:.2f}="
              f"{row['best_f1_val']:.3f} | gapF1={row['overfit_gap_f1']:+.3f}"
              f" | cross-company AP={_fmt(row.get('cross_company_ap'))}"
              f" AUROC={_fmt(row.get('cross_company_auroc'))}")
        if best_row is None or _rank(row) > _rank(best_row):
            best_name, best_row, best_model = name, row, model

    summary = {
        "seed": RANDOM_SEED,
        "n_features": len(feats),
        "feature_names": feats,
        "models": summaries,
        "best_model": best_name,
        "best_threshold": threshold_from_validation({
            "best_f1": {
                "threshold": best_row["best_threshold"],
                "f1": best_row["best_f1_val"],
            }}),
        "best_threshold_cost_optimal": best_row["cost_optimal_val"]["threshold"],
        "selection_rule": ("max cross-company AP (GroupKFold, train+validation) -> best-F1(val) -> "
                           "AP(val) -> AUROC(val) -> smallest overfit gap"),
        "bootstrap_val_best": (bootstrap_ci(y_val, predict_proba(best_model, X_val))
                               if best_model is not None else None),
        "trained_at": None,  # no timestamp so output stays stable/diffable
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    if best_model is not None:
        joblib.dump({"model": best_model, "name": best_name, "features": feats,
                     "threshold": float(summary["best_threshold"])},
                    MODELS_DIR / "best.joblib")

    # Validation PR curve for each model (refit from the same fixed params)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plotted = False
        for row in summaries:
            try:
                pipe = make_model(row["model"])
                if row["model"] == best_name:
                    pipe = best_model
                else:
                    pipe.fit(X_train, y_train)
                proba_val = predict_proba(pipe, X_val)
            except Exception as e:  # noqa: BLE001
                print(f"  (skipped) plotting {row['model']}: {e}")
                continue
            prec, rec, _ = precision_recall_curve(y_val, proba_val)
            plt.plot(rec, prec, label=row["model"])
            plotted = True

        if plotted:
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall on validation")
            plt.legend()
            plt.grid(alpha=0.3)
            from .config import FIGURES_DIR
            plt.savefig(FIGURES_DIR / "validation_pr_curves.png", dpi=120)
        plt.close()
    except Exception as e:  # pragma: no cover
        print("Skipping PR figure:", e)

    return summary


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[],
                        help="Train only these models (default: all).")
    args = parser.parse_args(argv)
    models = args.model or DEFAULT_MODELS
    summary = run(models)
    print(f"\nDone. Best model: {summary['best_model']} - "
          f"using threshold {summary['best_threshold']:.2f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
