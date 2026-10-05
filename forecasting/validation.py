"""Generalization and uncertainty checks that guard against overstated performance.

Provides GroupKFold and leave-one-company-out evaluation, sample and company-cluster bootstrap intervals,
walk-forward evaluation over time with a publication-date purge, and agreement between model
probabilities. Writes `results/validation_checks.json` and `results/walk_forward.json`.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import (FIGURES_DIR, GROUP_KEY, N_BOOTSTRAP, RANDOM_SEED, RESULTS_DIR, ensure_dirs,
                     ensure_utf8_stdio)
from .data_loader import load_prepared
from .evaluation import metrics_at_threshold
from .features import build_feature_matrix, extract_labels
from .models import DEFAULT_MODEL_ORDER, MODEL_REGISTRY, make_model, predict_proba


def grouped_cv(model_names: Sequence[str], samples: List[Dict[str, Any]],
               n_splits: int = 4, min_test: int = 4) -> Dict[str, Any]:
    """GroupKFold by company; return out-of-fold metrics (all folds pooled) for each model."""
    groups = np.asarray([s[GROUP_KEY] for s in samples])
    X = build_feature_matrix(samples)
    y = extract_labels(samples)
    n_groups = len(set(groups.tolist()))
    n_splits = max(2, min(n_splits, n_groups))
    splitter = GroupKFold(n_splits=n_splits)

    out: Dict[str, Any] = {"n_splits": n_splits, "n_groups": n_groups, "held_out_groups": {}}
    for name in model_names:
        proba_oof = np.full(len(y), np.nan)
        fold_rows: List[Dict[str, Any]] = []
        for k, (tr, te) in enumerate(splitter.split(X, y, groups=groups)):
            if len(set(y[tr].tolist())) < 2 or len(te) < min_test:
                fold_rows.append({"fold": k, "skipped": True,
                                  "n_test": int(len(te)),
                                  "test_groups": sorted(set(groups[te].tolist()))})
                continue
            model = make_model(name)
            model.fit(X[tr], y[tr])
            proba_oof[te] = predict_proba(model, X[te])
            has_pos = bool(y[te].sum())
            fold_rows.append({
                "fold": k, "skipped": False,
                "test_groups": sorted(set(groups[te].tolist())),
                "n_test": int(len(te)),
                "observed_distress": float(y[te].mean()),
                "auroc": float(roc_auc_score(y[te], proba_oof[te])) if len(set(y[te].tolist())) > 1 else None,
                "average_precision": float(average_precision_score(y[te], proba_oof[te])) if has_pos else None,
            })
        mask = np.isfinite(proba_oof)
        pooled = metrics_at_threshold(y[mask], proba_oof[mask], 0.5) if mask.sum() else {}
        out[name] = {
            "folds": fold_rows,
            "n_oof": int(mask.sum()),
            "oof_auroc": (float(roc_auc_score(y[mask], proba_oof[mask]))
                          if len(set(y[mask].tolist())) > 1 else None),
            "oof_average_precision": (float(average_precision_score(y[mask], proba_oof[mask]))
                                      if mask.sum() and y[mask].sum() else None),
            "oof_at_0.5": pooled,
        }
    return out


def leave_one_company_out(model_names: Sequence[str], splits: Dict[str, List[Dict[str, Any]]],
                          min_test: int = 4) -> List[Dict[str, Any]]:
    """Time-ordered LOCO: train on 7 companies (train split), test on the held-out company.

    The held-out company's validation + test rows are both used so it has enough samples; a company
    with single-class labels has an undefined AUROC (flagged as `single_class`) - which is itself a
    finding worth reporting.
    """
    train_s = splits["train"]
    candidates = splits["validation"] + splits["test"]
    rows: List[Dict[str, Any]] = []
    for ticker in sorted({s[GROUP_KEY] for s in train_s}):
        tr = [s for s in train_s if s[GROUP_KEY] != ticker]
        te = [s for s in candidates if s[GROUP_KEY] == ticker]
        if len(te) < min_test or len(set(extract_labels(tr).tolist())) < 2:
            rows.append({"held_out": ticker, "skipped": True, "n_test": len(te)})
            continue
        X_tr, y_tr = build_feature_matrix(tr), extract_labels(tr)
        X_te, y_te = build_feature_matrix(te), extract_labels(te)
        for name in model_names:
            model = make_model(name)
            model.fit(X_tr, y_tr)
            proba = predict_proba(model, X_te)
            single_class = len(set(y_te.tolist())) < 2
            rows.append({
                "held_out": ticker, "model": name, "skipped": False, "n_test": len(te),
                "observed_distress": float(y_te.mean()),
                "single_class": bool(single_class),
                "auroc": None if single_class else float(roc_auc_score(y_te, proba)),
                "average_precision": (float(average_precision_score(y_te, proba))
                                      if y_te.sum() else None),
                "at_0.5": metrics_at_threshold(y_te, proba, 0.5),
            })
    return rows


def bootstrap_ci(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5,
                 n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """95% confidence intervals (percentile bootstrap) for AUROC / AP / F1 / macro-F1."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    acc: Dict[str, List[float]] = {"auroc": [], "average_precision": [], "f1": [], "macro_f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yb, pb = y_true[idx], y_prob[idx]
        if len(set(yb.tolist())) < 2:
            continue
        acc["auroc"].append(float(roc_auc_score(yb, pb)))
        acc["average_precision"].append(float(average_precision_score(yb, pb)))
        at = metrics_at_threshold(yb, pb, threshold)
        acc["f1"].append(at["f1"])
        acc["macro_f1"].append(at["macro_f1"])
    point = metrics_at_threshold(y_true, y_prob, threshold)
    out: Dict[str, Any] = {"n": int(len(y_true)), "n_boot": int(len(acc["auroc"])),
                           "threshold": float(threshold)}
    for key, values in acc.items():
        arr = np.asarray(values)
        out[key] = {
            "point": float(roc_auc_score(y_true, y_prob)) if key == "auroc"
            else float(average_precision_score(y_true, y_prob)) if key == "average_precision"
            else float(point[key]),
            "mean": float(arr.mean()) if len(arr) else None,
            "ci95_low": float(np.percentile(arr, 2.5)) if len(arr) else None,
            "ci95_high": float(np.percentile(arr, 97.5)) if len(arr) else None,
        }
    return out


def agreement(model_names: Sequence[str], train_s: List[Dict[str, Any]],
              eval_s: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Spearman rank correlation between model probabilities, showing when metrics coincide."""
    from scipy.stats import spearmanr

    X_tr, y_tr = build_feature_matrix(train_s), extract_labels(train_s)
    X_ev = build_feature_matrix(eval_s)
    proba = {name: predict_proba(_fit(name, X_tr, y_tr), X_ev) for name in model_names}
    names = list(proba)
    matrix: Dict[str, Dict[str, float]] = {a: {} for a in names}
    for a in names:
        for b in names:
            matrix[a][b] = 1.0 if a == b else float(spearmanr(proba[a], proba[b]).correlation)
    return {"n_eval": len(eval_s), "spearman": matrix}


def _fit(name: str, X, y):
    model = make_model(name)
    model.fit(X, y)
    return model


#: Models used for the checks = exactly the project's 4 families (from the registry).
MODELS_FOR_CHECKS = list(DEFAULT_MODEL_ORDER)


def cluster_bootstrap_ci(samples: List[Dict[str, Any]], y_true: Sequence[int],
                         y_prob: Sequence[float], threshold: float = 0.5,
                         n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """95% CI by company-cluster bootstrap, resampling tickers rather than individual quarters.

    Eight quarters of one company are not independent; the label is close to a company attribute
    (90.8% of consecutive-quarter pairs keep the label; see `eda_deep.md`). Bootstrapping individual
    samples therefore yields artificially narrow intervals. Cluster resampling instead answers whether
    the conclusion holds for a different random set of companies.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups: Dict[str, List[int]] = {}
    for i, sample in enumerate(samples):
        groups.setdefault(sample[GROUP_KEY], []).append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    aurocs: List[float] = []
    aps: List[float] = []
    f1s: List[float] = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[k]] for k in picked])
        y, p = y_true[idx], y_prob[idx]
        if len(set(y.tolist())) < 2:      # drawn cluster has only one class => AUROC/AP undefined
            continue
        aurocs.append(float(roc_auc_score(y, p)))
        aps.append(float(average_precision_score(y, p)))
        # Compute F1 inline (cheaper than re-running the full metric block 2,000 times; same result).
        predicted = (p >= threshold).astype(int)
        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        fn = int(((predicted == 0) & (y == 1)).sum())
        denominator = 2 * tp + fp + fn
        f1s.append((2 * tp / denominator) if denominator else 0.0)

    def block(point: float, values: List[float]) -> Dict[str, Any]:
        finite = [v for v in values if v == v]
        return {"point": point, "n_valid": len(finite),
                "mean": float(np.mean(finite)) if finite else None,
                "ci95_low": float(np.percentile(finite, 2.5)) if finite else None,
                "ci95_high": float(np.percentile(finite, 97.5)) if finite else None}

    return {"n_boot": n_boot, "n_clusters": len(keys), "threshold": threshold,
            "auroc": block(_safe_auc(y_true, y_prob), aurocs),
            "average_precision": block(_safe_ap(y_true, y_prob), aps),
            "f1": block(float(metrics_at_threshold(y_true, y_prob, threshold).get("f1", float("nan"))),
                        f1s)}


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC, returning NaN when there is only one class (must not raise)."""
    return float("nan") if len(set(y_true.tolist())) < 2 else float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision - returns NaN when there is only one class."""
    return (float("nan") if len(set(y_true.tolist())) < 2
            else float(average_precision_score(y_true, y_prob)))


def _as_date(text: str) -> date:
    """'YYYY-MM-DD' -> `datetime.date` (every timestamp in prepared is ISO)."""
    year, month, day = (int(part) for part in str(text).split("-")[:3])
    return date(year, month, day)


def walk_forward_folds(samples: List[Dict[str, Any]], n_folds: int = 4, purge_days: int = 90,
                       min_train: int = 60) -> List[Dict[str, Any]]:
    """Split by TIME (expanding window) + purge by label publication date.

    Time marker is `request.target_period_end`, the end of the quarter to forecast. For fold k, train is
    every sample whose target period is before the cut and whose label was published before
    `cut - purge_days`, which reproduces the information available at decision time, and test is the next
    time block.

    This complements GroupKFold and LOCO rather than replacing them: LOCO answers the new-company
    question on an old period, while walk-forward answers the old-company question on a new period.
    """
    order = sorted(range(len(samples)),
                   key=lambda i: _as_date(samples[i]["request"]["target_period_end"]))
    ends = [_as_date(samples[i]["request"]["target_period_end"]) for i in order]
    n = len(order)
    folds: List[Dict[str, Any]] = []
    for k in range(1, n_folds + 1):
        cut_pos = min((k * n) // (n_folds + 1), n - 1)
        stop_pos = n if k == n_folds else min(((k + 1) * n) // (n_folds + 1), n - 1)
        cut = ends[cut_pos]
        train_idx = [i for i in order
                     if _as_date(samples[i]["request"]["target_period_end"]) < cut
                     and _as_date(samples[i]["label_available_on"]) <= cut - timedelta(days=purge_days)]
        test_idx = [i for i in order
                    if _as_date(samples[i]["request"]["target_period_end"]) >= cut
                    and (stop_pos >= n
                         or _as_date(samples[i]["request"]["target_period_end"]) < ends[stop_pos])]
        bounds = [_as_date(samples[i]["request"]["target_period_end"]) for i in test_idx]
        folds.append({
            "fold": k, "cut": cut.isoformat(), "purge_days": purge_days,
            "train_idx": train_idx, "test_idx": test_idx,
            "test_period": f"{min(bounds)}→{max(bounds)}" if bounds else None,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "skipped": len(train_idx) < min_train or not test_idx,
        })
    return folds


def walk_forward_metrics(model_names: Sequence[str], samples: List[Dict[str, Any]],
                         n_folds: int = 4, purge_days: int = 90, min_train: int = 60,
                         min_test: int = 4) -> Dict[str, Any]:
    """Run walk-forward for each model; return the per-fold table plus per-model averages."""
    X, y = build_feature_matrix(samples), extract_labels(samples)
    folds = walk_forward_folds(samples, n_folds=n_folds, purge_days=purge_days, min_train=min_train)
    rows: List[Dict[str, Any]] = []
    for fold in folds:
        base = {"fold": fold["fold"], "test_period": fold["test_period"],
                "n_train": fold["n_train"], "n_test": fold["n_test"]}
        if fold["skipped"] or len(fold["test_idx"]) < min_test:
            rows.append({**base, "model": None, "skipped": True,
                         "reason": "train too small" if fold["skipped"] else "test too small"})
            continue
        train_idx, test_idx = fold["train_idx"], fold["test_idx"]
        if len(set(y[test_idx].tolist())) < 2:
            rows.append({**base, "model": None, "skipped": True, "reason": "single-class test labels"})
            continue
        for name in model_names:
            model = make_model(name)
            model.fit(X[train_idx], y[train_idx])
            proba = predict_proba(model, X[test_idx])
            rows.append({**base, "model": name, "skipped": False, "cut": fold["cut"],
                         "observed_distress": float(y[test_idx].mean()),
                         "auroc": float(roc_auc_score(y[test_idx], proba)),
                         "average_precision": float(average_precision_score(y[test_idx], proba))})

    summary: Dict[str, Any] = {}
    for name in model_names:
        valid = [r for r in rows if r.get("model") == name and not r.get("skipped")]
        summary[name] = {
            "n_folds_evaluated": len(valid),
            "mean_auroc": float(np.mean([r["auroc"] for r in valid])) if valid else None,
            "mean_average_precision": (float(np.mean([r["average_precision"] for r in valid]))
                                       if valid else None),
            "min_auroc": min((r["auroc"] for r in valid), default=None),
            "per_fold_auroc": [r["auroc"] for r in valid],
        }
    return {"protocol": ("expanding window on `target_period_end` + purge by label publication date; "
                         "uses ONLY train+validation+purged - TEST never participates"),
            "n_folds": n_folds, "purge_days": purge_days, "min_train": min_train,
            "rows": rows, "summary": summary}


def _walk_forward_figure(result: Dict[str, Any], path: Any) -> None:
    """Figure: AUROC/AP per time fold for each model (skipped if matplotlib is missing)."""
    valid = [r for r in result["rows"] if not r.get("skipped")]
    if not valid:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        models = sorted({r["model"] for r in valid})
        folds = sorted({r["fold"] for r in valid})
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        for metric, ax, label in (("auroc", axes[0], "AUROC"),
                                  ("average_precision", axes[1], "AP (PR-AUC)")):
            for name in models:
                xs = [r["fold"] for r in valid if r["model"] == name]
                ys = [r[metric] for r in valid if r["model"] == name]
                ax.plot(xs, ys, marker="o", label=name, linewidth=1.2)
            ax.set_xticks(folds)
            ax.set_xlabel("time fold (cut point increases)")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle("Walk-forward over time (expanding window + 90-day purge)", fontsize=11)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - just an illustrative figure
        print("  (skipped walk-forward figure)", e)


def run(model_names: Sequence[str] | None = None) -> Dict[str, Any]:
    """Run all checks: GroupKFold, LOCO, bootstrap (sample & cluster), walk-forward, agreement."""
    ensure_dirs()
    names = [m for m in (model_names or MODELS_FOR_CHECKS) if m in MODEL_REGISTRY]
    splits = {n: load_prepared(n) for n in ("train", "validation", "test", "purged")}
    pool = ([s for n in ("train", "validation", "test", "purged") for s in splits[n]])

    cv = grouped_cv(names, pool, n_splits=len({s[GROUP_KEY] for s in pool}))
    loco = leave_one_company_out(names, splits)

    X_tr, y_tr = build_feature_matrix(splits["train"]), extract_labels(splits["train"])
    X_te, y_te = build_feature_matrix(splits["test"]), extract_labels(splits["test"])
    in_domain: Dict[str, Any] = {}
    for name in names:
        proba = predict_proba(_fit(name, X_tr, y_tr), X_te)
        in_domain[name] = {
            "test_auroc": float(roc_auc_score(y_te, proba)),
            "test_average_precision": float(average_precision_score(y_te, proba)),
            "bootstrap_test": bootstrap_ci(y_te, proba),
            # Confidence interval by company cluster, the right uncertainty for the new-company question.
            "bootstrap_cluster_test": cluster_bootstrap_ci(splits["test"], y_te, proba),
        }

    out: Dict[str, Any] = {
        "models": names,
        "grouped_cv_all_samples": cv,
        "leave_one_company_out": loco,
        "in_domain_test": in_domain,
        "agreement": agreement(names, splits["train"], splits["test"]),
        "headline": {},
    }
    print(f"  {'model':24s} {'in-domain test AUROC':>21s} {'cross-company OOF AUROC':>24s}")
    for name in names:
        oof = cv[name]["oof_auroc"]
        out["headline"].setdefault("in_domain_test_auroc", {})[name] = in_domain[name]["test_auroc"]
        out["headline"].setdefault("cross_company_oof_auroc", {})[name] = oof
        print(f"  {name:24s} {in_domain[name]['test_auroc']:21.3f} "
              f"{(oof if oof is not None else float('nan')):24.3f}")
    loco_scores = [r["auroc"] for r in loco if not r.get("skipped") and r.get("auroc") is not None]
    out["headline"]["loco_mean_auroc"] = float(np.mean(loco_scores)) if loco_scores else None
    out["headline"]["loco_n_evaluable"] = len(loco_scores)
    skipped = sorted({r["held_out"] for r in loco if r.get("skipped")})
    out["headline"]["loco_skipped_companies"] = skipped
    print(f"  LOCO: mean AUROC={out['headline']['loco_mean_auroc']} over "
          f"{len(loco_scores)} comparisons; skipped (single-class labels): {skipped}")

    # Time-ordered walk-forward over train, validation and purged data (test not involved); this
    # answers the old-company, new-period case and complements LOCO above.
    wf_pool = [s for n in ("train", "validation", "purged") for s in splits[n]]
    walk_forward = walk_forward_metrics(names, wf_pool)
    out["walk_forward"] = walk_forward

    def _fmt3(value: Any) -> str:
        return "—" if value is None else f"{float(value):.3f}"

    out["headline"]["walk_forward_mean_auroc"] = {
        name: block["mean_auroc"] for name, block in walk_forward["summary"].items()}
    print("  Time-ordered walk-forward (train+validation+purged, 90-day label purge):")
    for name, block in walk_forward["summary"].items():
        print(f"    {name:24s} {block['n_folds_evaluated']} fold | mean AUROC "
              f"{_fmt3(block['mean_auroc'])} | mean AP {_fmt3(block['mean_average_precision'])}")

    (RESULTS_DIR / "walk_forward.json").write_text(
        json.dumps(walk_forward, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    _walk_forward_figure(walk_forward, FIGURES_DIR / "analysis" / "08_walk_forward.png")

    (RESULTS_DIR / "validation_checks.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    print("=== Generalization & uncertainty checks ===")
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
