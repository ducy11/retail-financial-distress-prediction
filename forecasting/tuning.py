"""Hyperparameter tuning with GridSearchCV split by company using StratifiedGroupKFold.

Plain CV would mix all companies across folds and inflate the score, so every fold holds out a whole
company. Selection refits on average precision, which is threshold-free, then re-scores the best config
on validation. Writes `results/tuning.json` and `results/tuning.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score

from .config import GROUP_KEY, RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from .data_loader import load_prepared
from .evaluation import evaluate_proba
from .features import build_feature_matrix, extract_labels
from .models import DEFAULT_MODEL_ORDER, MODEL_REGISTRY, make_model, predict_proba

#: Full grid (20-40 configs/model depending on the machine).
GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {"model__C": [0.01, 0.1, 1.0, 10.0],
                 "model__class_weight": [None, "balanced"]},
    "random_forest": {"model__max_depth": [3, 6, None],
                      "model__min_samples_leaf": [1, 2, 4],
                      "model__n_estimators": [200, 500]},
    "hist_gradient_boosting": {"model__learning_rate": [0.03, 0.1],
                               "model__max_depth": [2, 3],
                               "model__max_iter": [200, 400]},
    "mlp": {"model__hidden_layer_sizes": [32, 64],
            "model__alpha": [1e-4, 1e-3, 1e-2]},
}

#: Reduced grid for weak machines (--quick).
QUICK_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {"model__C": [0.1, 1.0]},
    "random_forest": {"model__max_depth": [3, 6], "model__min_samples_leaf": [2]},
    "hist_gradient_boosting": {"model__learning_rate": [0.05], "model__max_depth": [3]},
    "mlp": {"model__alpha": [1e-3]},
}

#: Two CV scoring metrics: AUROC (ranking) and AP (what matters when the positive class is the target).
SCORING = {"auroc": "roc_auc", "average_precision": "average_precision"}


def tune_one(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
             quick: bool = False, n_splits: int = 4) -> Dict[str, Any]:
    """GridSearchCV for one model; returns a dict with the CV result table and the best config."""
    grid = QUICK_GRIDS[name] if quick else GRIDS[name]
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    search = GridSearchCV(make_model(name), grid, scoring=SCORING,
                          refit="average_precision", cv=cv, n_jobs=1, return_train_score=False)
    search.fit(X, y, groups=groups)

    rows: List[Dict[str, Any]] = []
    for params, mean_ap, mean_auroc in zip(search.cv_results_["params"],
                                          search.cv_results_["mean_test_average_precision"],
                                          search.cv_results_["mean_test_auroc"]):
        rows.append({
            "params": {k.replace("model__", ""): v for k, v in params.items()},
            "cv_average_precision": float(mean_ap),
            "cv_auroc": float(mean_auroc) if mean_auroc == mean_auroc else None,
        })
    rows.sort(key=lambda r: -r["cv_average_precision"])

    # Score the default config (from models.HYPERPARAMS) on the same splitter, showing whether tuning
    # actually improves anything rather than just reporting that GridSearch ran.
    default_scores = cross_val_score(make_model(name), X, y, groups=groups, cv=cv,
                                     scoring="average_precision", n_jobs=1)
    best = search.best_params_
    return {
        "model": name,
        "n_candidates": len(rows),
        "cv_folds": n_splits,
        "best_params": {k.replace("model__", ""): v for k, v in best.items()},
        "best_cv_average_precision": float(search.best_score_),
        "best_cv_auroc": float(search.cv_results_["mean_test_auroc"][search.best_index_]),
        "table": rows,
        "default_cv_reference": {"mean_average_precision": float(np.mean(default_scores)),
                                 "scores": [float(s) for s in default_scores]},
    }


def run(models: List[str] | None = None, quick: bool = False,
        n_splits: int = 4) -> Dict[str, Any]:
    """Tune models on train with company-split CV, then confirm on validation."""
    ensure_dirs()
    names = [m for m in (models or DEFAULT_MODEL_ORDER) if m in MODEL_REGISTRY]
    train_s, val_s = load_prepared("train"), load_prepared("validation")
    X, y = build_feature_matrix(train_s), extract_labels(train_s)
    groups = np.asarray([s[GROUP_KEY] for s in train_s])
    X_val, y_val = build_feature_matrix(val_s), extract_labels(val_s)

    results: List[Dict[str, Any]] = []
    for name in names:
        item = tune_one(name, X, y, groups, quick=quick, n_splits=n_splits)
        model = make_model(name, **item["best_params"])
        model.fit(X, y)
        val_metrics = evaluate_proba(y_val, predict_proba(model, X_val))
        item["validation"] = {
            "auroc": val_metrics["auroc"],
            "average_precision": val_metrics["average_precision"],
            "best_f1": val_metrics["best_f1"],
            "at_0.5": next(b for b in val_metrics["by_threshold"] if b["threshold"] == 0.5),
        }
        # Score the default config at the same threshold for a fair comparison
        base = make_model(name)
        base.fit(X, y)
        base_metrics = evaluate_proba(y_val, predict_proba(base, X_val))
        item["validation_default"] = {
            "auroc": base_metrics["auroc"],
            "average_precision": base_metrics["average_precision"],
            "best_f1": base_metrics["best_f1"],
        }
        results.append(item)
        print(f"  {name:24s} best={item['best_params']} "
              f"CV-AP={item['best_cv_average_precision']:.3f} "
              f"(default {item['default_cv_reference']['mean_average_precision']:.3f}) "
              f"val AP={item['validation']['average_precision']:.3f}")

    out = {"cv": "StratifiedGroupKFold by ticker",
           "refit_metric": "average_precision",
           "quick": quick, "results": results}
    (RESULTS_DIR / "tuning.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = ["# Hyperparameter tuning (CV split by company)", ""]
    lines.append("| Model | Best config | CV-AP | CV-AUROC | Val-AP | Val-AUROC | Val-F1* |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        v = r["validation"]
        lines.append(f"| {r['model']} | `{r['best_params']}` | {r['best_cv_average_precision']:.3f} | "
                     f"{r['best_cv_auroc']:.3f} | {v['average_precision']:.3f} | {v['auroc']:.3f} | "
                     f"{(v['best_f1'] or {}).get('f1', float('nan')):.3f} |")
    lines += ["", "*(F1* = best F1 on validation; AP = average precision.)*", ""]
    for r in results:
        lines.append(f"## {r['model']} - {r['n_candidates']} configs")
        lines.append("")
        lines.append("| # | Config | CV-AP | CV-AUROC |")
        lines.append("|---:|---|---:|---:|")
        for i, row in enumerate(r["table"][:12], start=1):
            lines.append(f"| {i} | `{row['params']}` | {row['cv_average_precision']:.3f} | "
                         f"{(row['cv_auroc'] if row['cv_auroc'] is not None else float('nan')):.3f} |")
        lines += ["", f"Default: CV-AP = {r['default_cv_reference']['mean_average_precision']:.3f} "
                      f"(delta {r['best_cv_average_precision'] - r['default_cv_reference']['mean_average_precision']:+.3f})",
                  ""]
    (RESULTS_DIR / "tuning.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Reduced grid for weak machines.")
    parser.add_argument("--models", default="", help="Comma-separated list of models.")
    args = parser.parse_args(argv)
    print("=== Hyperparameter tuning (split by company) ===")
    run(models=[m for m in args.models.split(",") if m], quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
