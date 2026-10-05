"""Score the frozen best model on the test split, once.

Loads `models/best.joblib`, applies the threshold chosen on validation, and writes
`results/test_evaluation.json` plus the confusion figures. Run with `python -m forecasting.evaluate`.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, auc, confusion_matrix,
                            precision_recall_curve, roc_curve)

from .config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from .data_loader import load_prepared
from .evaluation import evaluate_proba
from .features import build_feature_matrix, extract_labels
from .models import predict_proba
from .report import load_threshold


def load_cost_threshold() -> float | None:
    """Cost-optimal threshold (computed on validation in `forecasting.train`), if present."""
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary.get("best_threshold_cost_optimal")


def _confusion_figure(cm, threshold: float, path, cost_hint: str = "") -> None:
    """Draw and save a confusion matrix."""
    disp = ConfusionMatrixDisplay(cm, display_labels=["No distress", "Distress"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — test (threshold {threshold:.3f}){cost_hint}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _curve_figure(y_test: np.ndarray, proba: np.ndarray, threshold: float, path) -> Dict[str, float]:
    """ROC + PR on test (with AUROC/AP) to read the whole trade-off, not just one point."""
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fpr, tpr, label=f"AUROC={auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate (1 - specificity)")
    axes[0].set_ylabel("True Positive Rate (recall)")
    axes[0].set_title("ROC — test")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(rec, prec, label=f"AP={auc(rec, prec):.3f}")
    axes[1].axvline(threshold, color="k", ls="--", alpha=0.6, label=f"threshold={threshold:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall — test")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"auroc_from_curve": float(auc(fpr, tpr)), "ap_from_curve": float(auc(rec, prec))}


def run() -> Dict[str, Any]:
    ensure_dirs()
    artifact = joblib.load(MODELS_DIR / "best.joblib")
    model, name = artifact["model"], artifact["name"]
    threshold = load_threshold()

    test_samples = load_prepared("test")
    X_test = build_feature_matrix(test_samples)
    y_test = extract_labels(test_samples)
    proba = predict_proba(model, X_test)

    # Metrics at the operating threshold (matches the confusion figure), plus the 0.5 reference cut.
    metrics = evaluate_proba(y_test, proba, operating_threshold=threshold)
    metrics["threshold"] = threshold
    curve_stats = _curve_figure(y_test, proba, threshold, FIGURES_DIR / "test_roc_pr_curves.png")

    y_pred = (proba >= threshold).astype(int)
    misclassified = [
        {"sample_id": s["sample_id"], "ticker": s["ticker"], "actual": int(y),
         "probability": float(p), "predicted": int(pr)}
        for s, y, p, pr in zip(test_samples, y_test, proba, y_pred) if pr != int(y)
    ]

    summary = {
        "model": name,
        "threshold": threshold,
        "threshold_cost_optimal_from_val": load_cost_threshold(),
        "curve_stats": curve_stats,
        "test_metrics": metrics,
        "n_misclassified": len(misclassified),
        "misclassified": misclassified,
        "notes": "Metrics under 'operating' are at the operating threshold; 'confusion_at_0.5' keeps "
                 "the traditional 0.5 cut for comparison against other thresholds.",
    }
    (RESULTS_DIR / "test_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    _confusion_figure(confusion_matrix(y_test, y_pred, labels=[0, 1]), threshold,
                      FIGURES_DIR / "test_confusion.png")
    _confusion_figure(confusion_matrix(y_test, (proba >= 0.5).astype(int), labels=[0, 1]), 0.5,
                      FIGURES_DIR / "test_confusion_at_0.5.png")

    print(json.dumps({k: v for k, v in summary.items() if k != "misclassified"},
                     ensure_ascii=False, indent=2, default=float))
    return summary


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
