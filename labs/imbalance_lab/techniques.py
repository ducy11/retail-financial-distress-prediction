"""Catalog of imbalance-handling techniques with a leak-free comparison runner.

Every sampler runs inside `imblearn.pipeline.Pipeline`, cost-sensitive and Focal Loss weights are
derived inside `fit`, and thresholds are tuned on out-of-fold probabilities of the train pool, so
validation and test never leak into fitting. Accuracy is reported as a diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C

# Catalog mapping groups to techniques, the single source for the runner, the report and the tests.
#: Required technique groups. `test_catalog_covers_every_required_group` checks this list directly, so no
#: entry may be dropped when editing the code.
REQUIRED_GROUPS: Dict[str, Tuple[str, ...]] = {
    "data-level/oversampling": ("ros", "smote", "borderline_smote", "adasyn"),
    "data-level/undersampling": ("rus", "tomek", "enn"),
    "hybrid": ("smote_tomek", "smote_enn"),
    "algorithm-level": ("cost_sensitive_scale_pos_weight", "cost_sensitive_class_weight",
                        "focal_loss"),
    "ensemble": ("balanced_rf", "easy_ensemble", "rusboost"),
}


#: Reference group that is not part of the required list but is needed for comparison: the unmodified
#: model with no imbalance handling.
REFERENCE_GROUPS: Dict[str, Tuple[str, ...]] = {"baseline": ("baseline",)}

#: Description of each technique, used in logs and the report.
TECHNIQUE_DOCS: Dict[str, str] = {
    "baseline": "Baseline: default boosting with no imbalance intervention, used as the reference",
    "ros": "RandomOverSampler: duplicate minority samples without creating new ones",
    "smote": "SMOTE: interpolate between a minority sample and its minority neighbors",
    "borderline_smote": "BorderlineSMOTE: interpolate only from minority samples on the boundary (DANGER region)",
    "adasyn": "ADASYN: generate more samples in proportion to each minority sample's difficulty",
    "rus": "RandomUnderSampler: randomly drop majority samples to reach the target ratio",
    "tomek": "Tomek Links: boundary cleaning that drops the majority sample in a mixed-class neighbor pair",
    "enn": "EditedNearestNeighbours: boundary cleaning that drops samples with differently labeled neighbors",
    "smote_tomek": "Hybrid SMOTE + Tomek Links",
    "smote_enn": "Hybrid SMOTE + EditedNearestNeighbours",
    "cost_sensitive_scale_pos_weight": "LightGBM with `scale_pos_weight = n_negative/n_positive` computed in fit",
    "cost_sensitive_class_weight": "`class_weight='balanced'`, deriving sample weights inside fit",
    "focal_loss": "Focal Loss as a LightGBM custom objective, with gamma down-weighting easy samples",
    "balanced_rf": "BalancedRandomForestClassifier, undersampling inside each tree during fit",
    "easy_ensemble": "EasyEnsembleClassifier, many AdaBoost models over balanced subsets",
    "rusboost": "RUSBoostClassifier, boosting with undersampling in each round",
}

#: Library requirements per technique; a missing library makes the runner record a `skipped` status.
REQUIRES: Dict[str, Tuple[str, ...]] = {
    "balanced_rf": ("imblearn",), "easy_ensemble": ("imblearn",), "rusboost": ("imblearn",),
    "focal_loss": ("lightgbm",),
}

#: Anchor table mapping each requirement to its technique and implementation site, for the report and tests.
IMPLEMENTATION: Dict[str, str] = {
    "baseline": "models.make_base_classifier, default boosting with no intervention",
    "ros": "samplers.RandomOverSampler or imblearn RandomOverSampler",
    "smote": "samplers.SMOTE or imblearn SMOTE",
    "borderline_smote": "samplers.BorderlineSMOTE or imblearn BorderlineSMOTE",
    "adasyn": "samplers.ADASYN or imblearn ADASYN",
    "rus": "samplers.RandomUnderSampler or imblearn RandomUnderSampler",
    "tomek": "samplers.TomekLinks or imblearn TomekLinks",
    "enn": "samplers.EditedNearestNeighbours or imblearn EditedNearestNeighbours",
    "smote_tomek": "samplers.make_hybrid_sampler('smote_tomek') or imblearn SMOTETomek",
    "smote_enn": "samplers.make_hybrid_sampler('smote_enn') or imblearn SMOTEENN",
    "cost_sensitive_scale_pos_weight": "models.ScalePosWeightClassifier",
    "cost_sensitive_class_weight": "models.BalancedWeightClassifier",
    "focal_loss": "losses.FocalLossClassifier using analytic grad/hess",
    "balanced_rf": "imblearn BalancedRandomForestClassifier",
    "easy_ensemble": "imblearn EasyEnsembleClassifier",
    "rusboost": "imblearn RUSBoostClassifier",
}

#: Run order: the baseline first as the reference, then the required groups.
TECHNIQUE_ORDER: Tuple[str, ...] = tuple(
    key for group in (*REFERENCE_GROUPS.values(), *REQUIRED_GROUPS.values()) for key in group)

#: Key of the unmodified model, used in the baseline comparison table.
BASELINE_TECHNIQUE = "baseline"


#: Threshold modes, the keys that carry a `threshold`. Every threshold function returns this set plus a
#: few summary values (`pr_auc`, `n_candidates`), so iteration filters by this list.
THRESHOLD_MODES: Tuple[str, ...] = ("best_f1", "best_cost", "min_precision", "fixed_0.5")


def technique_group(key: str) -> str:
    """Group of one technique, including the reference group `baseline`."""
    for groups in (REQUIRED_GROUPS, REFERENCE_GROUPS):
        for group, keys in groups.items():
            if key in keys:
                return group
    raise KeyError(f"Technique not in the catalog: {key!r}")


def all_techniques_available(prefer_imblearn: bool = True) -> Dict[str, List[str]]:
    """Map each technique to its missing libraries; an empty dict means everything can run."""
    from .losses import FocalLossClassifier
    from .samplers import resampling_backend

    have_imblearn = resampling_backend(prefer_imblearn) == "imblearn"
    have_lightgbm = FocalLossClassifier.available()
    missing: Dict[str, List[str]] = {}
    for key in TECHNIQUE_ORDER:
        gaps: List[str] = []
        for requirement in REQUIRES.get(key, ()):
            if requirement == "imblearn" and not have_imblearn:
                gaps.append("imbalanced-learn")
            elif requirement == "lightgbm" and not have_lightgbm:
                gaps.append("lightgbm")
        if gaps:
            missing[key] = gaps
    return missing


# Estimator construction for one technique.
def _ensemble_factory(key: str, random_state: int, quick: bool) -> Callable[[], Any]:
    """Factory for the ensemble group: BalancedRandomForest, EasyEnsemble and RUSBoost.

    All three undersample inside `fit` over the train set passed to them, so under CV they sample only
    from the fold train and never touch validation or test.
    """
    from imblearn.ensemble import (BalancedRandomForestClassifier, EasyEnsembleClassifier,
                                  RUSBoostClassifier)

    if key == "balanced_rf":
        return lambda: BalancedRandomForestClassifier(
            n_estimators=50 if quick else 100, max_depth=6, min_samples_leaf=2,
            sampling_strategy="auto", replacement=False, n_jobs=1, random_state=random_state)
    if key == "easy_ensemble":
        return lambda: EasyEnsembleClassifier(
            n_estimators=5 if quick else 10, sampling_strategy="auto", n_jobs=1,
            random_state=random_state)
    return lambda: RUSBoostClassifier(n_estimators=20 if quick else 50, learning_rate=0.1,
                                      random_state=random_state)


def build_technique(key: str, *, prefer_imblearn: bool = C.PREFER_IMBLEARN,
                    random_state: int = C.SEED, quick: bool = False) -> Dict[str, Any]:
    """Spec for one technique: `factory` for a fresh estimator per fold and `probe_factory` for resampling logs.

    `probe_factory` is returned only for resampling techniques. It is a sampler-only pipeline (`pipe[:-1]`)
    used to call `fit_resample` and record the fold-train label distribution before and after.
    """
    if key not in TECHNIQUE_ORDER:
        raise KeyError(f"Technique not in the catalog: {key!r}; have {list(TECHNIQUE_ORDER)}")

    from .models import BalancedWeightClassifier, ScalePosWeightClassifier, make_base_classifier
    from .samplers import (HYBRID_SAMPLERS, SINGLE_SAMPLERS, build_sampler_pipeline,
                           make_hybrid_sampler, make_single_sampler)

    join_kwargs = {"over_strategy": C.TECHNIQUE_OVER_STRATEGY,
                   "under_strategy": C.TECHNIQUE_UNDER_STRATEGY,
                   "k_neighbors": C.TECHNIQUE_K_NEIGHBORS, "random_state": random_state}
    samplers: Optional[List[Tuple[str, Any]]] = None
    if key in SINGLE_SAMPLERS:
        samplers = make_single_sampler(key, prefer_imblearn, **join_kwargs)
    elif key in HYBRID_SAMPLERS:
        samplers = make_hybrid_sampler(key, prefer_imblearn, **join_kwargs)

    if samplers is not None:
        def factory() -> Any:
            return build_sampler_pipeline(
                samplers, make_base_classifier(None, random_state=random_state),
                prefer_imblearn=prefer_imblearn)

        def probe_factory() -> Any:
            return factory()[:-1]          # sampler only, so fit_resample is available

        kind = "data-level"
    elif key == "baseline":
        # Reference model: no resampling, no class weights, no custom loss.
        factory = lambda: make_base_classifier(None, random_state=random_state)  # noqa: E731
        probe_factory = None
        kind = "baseline"
    elif key == "cost_sensitive_scale_pos_weight":
        factory = lambda: ScalePosWeightClassifier(random_state=random_state)  # noqa: E731
        probe_factory = None
        kind = "algorithm-level"
    elif key == "cost_sensitive_class_weight":
        factory = lambda: BalancedWeightClassifier(random_state=random_state)  # noqa: E731
        probe_factory = None
        kind = "algorithm-level"
    elif key == "focal_loss":
        from .losses import FocalLossClassifier

        factory = lambda: FocalLossClassifier(random_state=random_state)  # noqa: E731
        probe_factory = None
        kind = "algorithm-level"
    else:
        factory = _ensemble_factory(key, random_state, quick)
        probe_factory = None
        kind = "ensemble"

    return {"key": key, "group": technique_group(key), "kind": kind,
            "doc": TECHNIQUE_DOCS[key], "implementation": IMPLEMENTATION[key],
            "samplers": samplers, "estimator": factory(), "factory": factory,
            "probe_factory": probe_factory,
            "is_resampling": samplers is not None, "requires": list(REQUIRES.get(key, ()))}


# Run one technique: stratified CV on the train pool, thresholds from the OOF PR curve, then a test score.
def _write_json(path: Any, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float) + "\n",
                    encoding="utf-8")


def _write_csv(path: Any, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = header or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def evaluate_technique(spec: Dict[str, Any], X_pool: Any, y_pool: Any, X_test: Any, y_test: Any, *,
                       n_splits: int, seed: int, log: Callable[[str], None],
                       threshold_fn: Callable[..., Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate one technique without leakage: resampling inside the pipeline, thresholds from OOF.

    Returns `rows` (test metrics per threshold), `fold_rows`, `resample_rows`, `thresholds`,
    `checks` (PASS/FAIL anti-leak flags) and `oof_pr_auc`.
    """
    from .cv import cross_validate_strategy, refit_and_score
    from .metrics import metrics_at_threshold

    cv = cross_validate_strategy(spec, X_pool, y_pool, n_splits=n_splits, seed=seed,
                                 probe_factory=spec["probe_factory"], log=log)
    tuned = threshold_fn(y_pool, cv["oof_proba"], cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                         precision_target=C.PRECISION_TARGET)
    threshold_map = {mode: float(tuned[mode]["threshold"]) for mode in THRESHOLD_MODES
                     if mode in tuned}
    log("    Thresholds from the PR curve of OOF probabilities: "
        + ", ".join(f"{mode}={value:.4f}" for mode, value in threshold_map.items())
        + f" | OOF PR-AUC={tuned['pr_auc']:.4f} ({tuned['n_candidates']} candidates)")

    X_test_before, y_test_before = X_test.copy(), y_test.copy()
    scored = refit_and_score(spec, X_pool, y_pool, X_test, y_test, threshold_map)
    baseline_f1 = float(scored["at_threshold"]["fixed_0.5"]["f1"])
    baseline_recall = float(scored["at_threshold"]["fixed_0.5"]["recall"])

    rows: List[Dict[str, Any]] = []
    for mode, metrics in scored["at_threshold"].items():
        rows.append({"technique": spec["key"], "group": spec["group"], "kind": spec["kind"],
                     "threshold_mode": mode, "split": "test", "n_test": int(len(y_test)),
                     "delta_f1_vs_0.5": float(metrics["f1"] - baseline_f1),
                     "delta_recall_vs_0.5": float(metrics["recall"] - baseline_recall),
                     **metrics})
    checks = {
        "fold_validation_untouched": True,   # cv.assert_val_untouched ran inside each fold
        "test_untouched": bool(np.array_equal(X_test, X_test_before)
                               and np.array_equal(y_test, y_test_before)),
        "thresholds_from_oof": bool(set(threshold_map) == set(THRESHOLD_MODES)),
    }
    resample_rows = []
    for entry in cv["folds"]:
        before, after = entry["train_before"], entry["train_after"]
        resample_rows.append({
            "technique": spec["key"], "fold": entry["fold"],
            "train_n_before": before["n"], "train_pos_before": before["n_positive"],
            "train_ir_before": before["imbalance_ratio"], "train_n_after": after["n"],
            "train_pos_after": after["n_positive"], "train_ir_after": after["imbalance_ratio"],
            "val_n": entry["val"]["n"], "val_pos": entry["val"]["n_positive"],
            "val_pos_pct": entry["val"]["positive_pct"]})
    log(f"    -> test: F1@{threshold_map['best_f1']:.3f}="
        f"{scored['at_threshold']['best_f1']['f1']:.3f} (F1@0.5={baseline_f1:.3f}) | "
        f"PR-AUC={scored['at_threshold']['best_f1']['pr_auc']:.4f} | "
        f"leakage: {'PASS' if all(checks.values()) else 'FAIL'}")

    return {"technique": spec["key"], "group": spec["group"], "kind": spec["kind"],
            "doc": spec["doc"], "implementation": spec["implementation"],
            "is_resampling": spec["is_resampling"], "status": "ok", "reason": "",
            "rows": rows, "fold_rows": [{"technique": spec["key"], **row}
                                        for row in cv["fold_metrics"]],
            "resample_rows": resample_rows, "thresholds": threshold_map,
            "oof_pr_auc": float(tuned["pr_auc"]), "checks": checks,
            "oof_proba": cv["oof_proba"], "oof_y": y_pool,
            "oof_at_0.5": metrics_at_threshold(y_pool, cv["oof_proba"], 0.5)}



def _empty_result(spec: Dict[str, Any], status: str, reason: str) -> Dict[str, Any]:
    """Record for a skipped or failed technique, keeping the schema so the report needs no special case."""
    return {"technique": spec["key"], "group": spec["group"], "kind": spec["kind"],
            "doc": spec["doc"], "implementation": spec["implementation"],
            "is_resampling": spec["is_resampling"], "status": status, "reason": reason,
            "rows": [], "fold_rows": [], "resample_rows": [], "thresholds": {},
            "oof_pr_auc": None, "checks": {}, "oof_at_0.5": {}, "oof_proba": None, "oof_y": None}


def _has(package: str) -> bool:
    """True when `package` has version metadata, so it can be printed without importing it."""
    try:
        md.version(package)
    except Exception:
        return False
    return True


def run(n_samples: Optional[int] = None, n_splits: Optional[int] = None,
        techniques: Optional[Sequence[str]] = None, prefer_imblearn: bool = C.PREFER_IMBLEARN,
        write: bool = True, quick: bool = False) -> Dict[str, Any]:
    """Run the catalog: generate 98/2 data, cross-validate each technique, tune thresholds on OOF, then test."""
    from .data import (format_distribution, label_distribution, make_imbalanced_dataset,
                       stratified_holdout_split)
    from .models import classifier_backend
    from .samplers import resampling_backend
    from .thresholds import tune_thresholds_from_pr_curve

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        try:  # force UTF-8 so a redirected Windows console does not raise UnicodeEncodeError
            reconfigure = getattr(sys.stdout, "reconfigure", None)
            encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
            if reconfigure is not None and encoding not in ("utf-8", "utf8"):
                reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - stream does not support reconfigure
            pass
        print(message)
        log_lines.append(message)

    seed = C.SEED
    n = int(n_samples or C.CATALOG_N_SAMPLES)
    folds = int(n_splits or C.CATALOG_N_SPLITS)
    keys = list(techniques) if techniques else list(TECHNIQUE_ORDER)
    unknown = [key for key in keys if key not in TECHNIQUE_ORDER]
    if unknown:
        raise ValueError(f"Techniques not in the catalog: {unknown}; have {list(TECHNIQUE_ORDER)}")

    missing_libs = all_techniques_available(prefer_imblearn)
    log("=== Imbalance technique catalog: pipeline with no data leakage ===")
    log(f"Classifier backend  : {classifier_backend()} "
        f"(lightgbm {md.version('lightgbm') if _has('lightgbm') else '-'}, "
        f"scikit-learn {md.version('scikit-learn')})")
    log(f"Resampling backend  : {resampling_backend(prefer_imblearn)} "
        f"(imbalanced-learn {md.version('imbalanced-learn') if _has('imbalanced-learn') else '-'})")
    log(f"Catalog             : {len(keys)} techniques x {folds} folds | n_samples={n} | seed={seed}"
        + (" | --quick mode" if quick else ""))
    if missing_libs:
        log(f"Missing libraries   : {missing_libs}, so those techniques are skipped with a recorded reason")
    for group, group_keys in (*REFERENCE_GROUPS.items(), *REQUIRED_GROUPS.items()):
        log(f"  - {group:26s}: {', '.join(key for key in group_keys if key in keys)}")

    X, y = make_imbalanced_dataset(n_samples=n, random_state=seed)
    split = stratified_holdout_split(X, y, seed=seed)
    X_pool, y_pool = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    log("\n[1] Data with a 98/2 imbalance")
    log(f"    all         : {format_distribution(label_distribution(y))}")
    log(f"    train_pool  : {format_distribution(label_distribution(y_pool))}")
    log(f"    holdout test: {format_distribution(label_distribution(y_test))}"
        "  <- never resampled, scored once")

    results: List[Dict[str, Any]] = []
    for index, key in enumerate(keys, start=2):
        spec = build_technique(key, prefer_imblearn=prefer_imblearn, random_state=seed, quick=quick)
        log(f"\n[{index}] `{key}` ({spec['group']}): {spec['doc']}")
        gaps = missing_libs.get(key, [])
        if gaps:
            reason = f"missing {', '.join(gaps)}"
            log(f"    SKIPPED: {reason}")
            results.append(_empty_result(spec, "skipped", reason))
            continue
        try:
            results.append(evaluate_technique(spec, X_pool, y_pool, X_test, y_test,
                                              n_splits=folds, seed=seed, log=log,
                                              threshold_fn=tune_thresholds_from_pr_curve))
        except Exception as exc:  # noqa: BLE001 - one failing technique must not stop the catalog
            reason = f"{type(exc).__name__}: {exc}"
            log(f"    ERROR: {reason}")
            results.append(_empty_result(spec, "error", reason))

    failed = [item["technique"] for item in results
              if item["status"] == "ok" and not all(item["checks"].values())]
    log(f"\n[{len(keys) + 2}] Anti-leak checks: "
        f"{'ALL PASS' if not failed else 'FAIL at: ' + ', '.join(failed)}")

    result: Dict[str, Any] = {
        "n_samples": n, "n_splits": folds, "seed": seed, "quick": quick,
        "backend": classifier_backend(), "resampling_backend": resampling_backend(prefer_imblearn),
        "techniques": keys, "required_groups": {g: list(v) for g, v in REQUIRED_GROUPS.items()},
        "results": results,
        "rows": [row for item in results for row in item["rows"]],
        "resample_rows": [row for item in results for row in item["resample_rows"]],
        "fold_rows": [row for item in results for row in item["fold_rows"]],
        "missing_libraries": missing_libs,
        "label_distribution": {"all": label_distribution(y),
                               "train_pool": label_distribution(y_pool),
                               "test": label_distribution(y_test)},
        "leakage_all_pass": not failed,
        "log": "\n".join(log_lines),
    }
    if write:
        _write_artifacts(result)
    return result




# Reporting: Markdown, CSV, JSON and log artifacts.
def _fmt(value: Any, digits: int = 3) -> str:
    """Format a number for tables, mapping None and NaN to '-'."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return "-" if number != number else f"{number:.{digits}f}"


def _row_for(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Test metric row for one technique at one threshold mode."""
    return next((row for row in item["rows"] if row["threshold_mode"] == mode), {})


def _train_summary(item: Dict[str, Any]) -> str:
    """`IR before -> IR after` string, averaged over folds, for the label-distribution column."""
    rows = item["resample_rows"]
    if not rows:
        return "-"
    before = float(np.mean([row["train_ir_before"] for row in rows]))
    after = float(np.mean([row["train_ir_after"] for row in rows]))
    return f"{before:.1f} -> {after:.1f}"


def _val_pos_range(item: Dict[str, Any]) -> str:
    """Range of fold-validation positive percentages, showing that validation keeps the class ratio."""
    rows = item["resample_rows"]
    if not rows:
        return "-"
    values = [row["val_pos_pct"] for row in rows]
    return f"{min(values):.2f}-{max(values):.2f}%"


def compare_with_baseline(result: Dict[str, Any],
                          mode: str = "best_f1") -> Dict[str, Any]:
    """Comparison table of the baseline against every imbalance-handling technique.

    Each row lists the primary metrics at threshold `mode`, which defaults to `best_f1`, a threshold
    selected on the PR curve of out-of-fold probabilities: precision, recall, F1, F1-macro, F1-weighted,
    F-beta, PR-AUC, ROC-AUC, MCC and the confusion matrix (TN/FP/FN/TP). Deltas against the baseline are
    included for PR-AUC, F1, F1-macro, F-beta and recall.

    Accuracy is absent on purpose. It is a diagnostic (`metrics.accuracy_diagnostic`) because at a 98/2
    ratio the always-predict-the-majority-class rule already reaches roughly 98% accuracy.

    Returns:
        A dict with `mode`, `metrics` (column order), `baseline` (the reference row), `rows` (the baseline
        first, then catalog order), `n_better_than_baseline` and `best_by_metric`.
    """
    from .metrics import COMPARISON_COLUMNS

    by_key: Dict[str, Dict[str, Any]] = {}
    for item in result["results"]:
        if item["status"] != "ok":
            continue
        row = _row_for(item, mode)
        if not row:
            continue
        by_key[item["technique"]] = {
            "technique": item["technique"], "group": item["group"], "kind": item["kind"],
            "is_resampling": item["is_resampling"], "threshold": float(row["threshold"]),
            **{name: row.get(name) for name in COMPARISON_COLUMNS},
        }

    baseline = by_key.get(BASELINE_TECHNIQUE)
    for name, row in by_key.items():
        if baseline is None:
            continue
        for metric in ("pr_auc", "f1", "macro_f1", "weighted_f1", "fbeta", "recall", "roc_auc", "mcc"):
            current, reference = row.get(metric), baseline.get(metric)
            if current is None or reference is None or current != current or reference != reference:
                row[f"delta_{metric}"] = None
            else:
                row[f"delta_{metric}"] = float(current) - float(reference)

    order = [key for key in TECHNIQUE_ORDER if key in by_key]
    rows = [by_key[key] for key in order]
    deltas = [row["delta_pr_auc"] for row in rows
              if row["technique"] != BASELINE_TECHNIQUE and row.get("delta_pr_auc") is not None]
    best_by_metric: Dict[str, Any] = {}
    for metric in ("pr_auc", "f1", "macro_f1", "weighted_f1", "fbeta", "roc_auc", "mcc"):
        candidates = [row for row in rows if row.get(metric) is not None]
        if candidates:
            winner = max(candidates, key=lambda row: float(row[metric]))
            best_by_metric[metric] = {"technique": winner["technique"], "value": float(winner[metric])}
    return {"mode": mode, "metrics": list(COMPARISON_COLUMNS), "baseline": baseline, "rows": rows,
            "n_better_than_baseline": int(sum(1 for value in deltas if value > 0)),
            "n_techniques_compared": int(len(deltas)), "best_by_metric": best_by_metric}


def comparison_markdown(result: Dict[str, Any], mode: str = "best_f1") -> str:
    """Markdown table of the baseline against every handling technique, also used in the report."""
    table = compare_with_baseline(result, mode)
    lines = [f"| Technique | Group | PR-AUC | F1 | F1-macro | F1-weighted | F-beta({_fmt(C.FBETA_BETA, 1)}) "
             "| ROC-AUC | MCC | TN/FP/FN/TP | dPR-AUC | dF1 | dF1-macro |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in table["rows"]:
        confusion = (f"{row.get('tn')}/{row.get('fp')}/{row.get('fn')}/{row.get('tp')}"
                     if row.get("tn") is not None else "-")
        lines.append(
            f"| `{row['technique']}` | `{row['group']}` | {_fmt(row.get('pr_auc'), 4)} | "
            f"{_fmt(row.get('f1'))} | {_fmt(row.get('macro_f1'))} | {_fmt(row.get('weighted_f1'))} | "
            f"{_fmt(row.get('fbeta'))} | {_fmt(row.get('roc_auc'))} | {_fmt(row.get('mcc'))} | "
            f"{confusion} | {_fmt(row.get('delta_pr_auc'))} | {_fmt(row.get('delta_f1'))} | "
            f"{_fmt(row.get('delta_macro_f1'))} |")
    if table["baseline"] is None:
        lines.append("| _(baseline missing, so the table lists handling techniques only)_ |  |  |  |  |  |  |  |  |  |  |  |  |")
    else:
        better = table["n_better_than_baseline"]
        total = table["n_techniques_compared"]
        lines += ["", f"- Scoring threshold: `{mode}`, selected on the PR curve of out-of-fold "
                      f"probabilities. Baseline: **PR-AUC {_fmt(table['baseline'].get('pr_auc'), 4)}, "
                      f"F1 {_fmt(table['baseline'].get('f1'))}**.",
                  f"- Techniques with higher PR-AUC than the baseline: **{better}/{total}**"
                  + (f", the highest being `{table['best_by_metric']['pr_auc']['technique']}` "
                     f"({_fmt(table['best_by_metric']['pr_auc']['value'], 4)})."
                     if "pr_auc" in table["best_by_metric"] else "."),
                  "- The table omits accuracy because at a 98/2 ratio the majority-class rule reaches "
                  "roughly 98%, so accuracy is treated as a diagnostic only."]
    return "\n".join(lines) + "\n"


def _diagnostic_accuracy_lines(result: Dict[str, Any]) -> List[str]:
    """Diagnostic lines about accuracy, reported only to show why it is not the primary measure.

    Accuracy at the operating point is printed next to the majority-class baseline, which makes the
    issue with a 98/2 ratio immediately visible.
    """
    ok = [item for item in result["results"] if item["status"] == "ok"]
    rows = [(item["technique"], _row_for(item, "best_f1")) for item in ok]
    rows = [(name, row) for name, row in rows if row]
    if not rows:
        return []
    baseline = next((row for name, row in rows if name == BASELINE_TECHNIQUE), rows[0][1])
    majority = baseline.get("majority_baseline_accuracy_pct")
    values = [float(row["accuracy"]) * 100.0 for _name, row in rows
              if row.get("accuracy") is not None]
    lines = ["**Diagnostic metric, not used to draw conclusions:**",
             f"- The always-predict-the-majority-class rule reaches **{_fmt(majority, 2)}% accuracy**, "
             f"while the techniques fall between {_fmt(min(values), 2)}% and {_fmt(max(values), 2)}%.",
             "- Accuracy differences here do not show that one technique is better; read precision, "
             "recall, F1 with its macro, weighted and beta variants, PR-AUC, ROC-AUC and the confusion "
             "matrix instead."]
    return lines


def _markdown(result: Dict[str, Any]) -> str:
    """Full Markdown report: catalog, test metrics, PR thresholds, anti-leak checks and conclusions."""
    results = result["results"]
    ok = [item for item in results if item["status"] == "ok"]
    dist = result["label_distribution"]
    lines = ["# Imbalance technique catalog: leak-free comparison", "",
             f"- Classifier backend **{result['backend']}**, resampling "
             f"**{result['resampling_backend']}**, n_samples {result['n_samples']}, "
             f"{result['n_splits']} folds, seed {result['seed']}"
             + (" --quick mode" if result["quick"] else ""),
             "- Label distribution: all "
             f"n={dist['all']['n']} (positive {dist['all']['positive_pct']:.2f}%, IR="
             f"{dist['all']['imbalance_ratio']:.1f}); train_pool "
             f"n={dist['train_pool']['n']} (positive {dist['train_pool']['positive_pct']:.2f}%); test "
             f"n={dist['test']['n']} (positive {dist['test']['positive_pct']:.2f}%)",
             "- Threshold tuning: thresholds selected on the precision-recall curve of out-of-fold "
             "probabilities (train_pool only), namely `best_f1`, `best_cost` for FN/FP cost and "
             "`min_precision`, plus a 0.5 reference. Test is scored once after freezing.",
             "- Anti-leak: every sampler sits inside `imblearn.pipeline.Pipeline`; cost-sensitive and "
             "Focal Loss weights are computed inside `fit`; ensembles sample inside `fit`; "
             "`cv.assert_val_untouched` runs on every fold.", "",
             "## 1. Catalog, implementation and status", "",
             "| Group | Technique | Description | Implementation | Status |", "|---|---|---|---|---|"]
    for item in results:
        if item["status"] == "ok":
            status = "PASS" if all(item["checks"].values()) else "FAIL"
        else:
            status = f"{item['status'].upper()} ({item['reason']})"
        lines.append(f"| `{item['group']}` | `{item['technique']}` | {item['doc']} | "
                     f"{item['implementation']} | {status} |")

    lines += ["", "## 2. Metric trên holdout test — ngưỡng PR tốt nhất vs mốc 0.5 (yêu cầu #3)", "",
              "*(Accuracy KHÔNG có trong bảng: ở tỉ lệ 98/2 đoán 'lớp đa số' đã đạt ~98% ⇒ accuracy chỉ "
              "là chỉ số chẩn đoán.)*", "",
              "| Nhóm | Kỹ thuật | Ngưỡng | thr | Precision | Recall | F1 | F1-macro | F1-weighted | "
              f"F-beta({C.FBETA_BETA:g}) | ΔF1 vs 0.5 | ΔRecall | PR-AUC | ROC-AUC | Brier | MCC |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in ok:
        for mode in ("best_f1", "fixed_0.5", "best_cost", "min_precision"):
            row = _row_for(item, mode)
            if not row:
                continue
            lines.append(
                f"| `{item['group']}` | `{item['technique']}` | {mode} | {row['threshold']:.4f} | "
                f"{_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['f1'])} | "
                f"{_fmt(row.get('macro_f1'))} | {_fmt(row.get('weighted_f1'))} | "
                f"{_fmt(row.get('fbeta'))} | {_fmt(row.get('delta_f1_vs_0.5'))} | "
                f"{_fmt(row.get('delta_recall_vs_0.5'))} | {_fmt(row['pr_auc'], 4)} | "
                f"{_fmt(row['roc_auc'])} | {_fmt(row.get('brier'), 4)} | {_fmt(row['mcc'])} |")

    lines += ["", "## 3. Baseline (untreated) versus handling techniques", "",
              comparison_markdown(result).rstrip("\n")]
    diagnostic = _diagnostic_accuracy_lines(result)
    if diagnostic:
        lines += ["", *diagnostic]

    lines += ["", "## 4. Thresholds from out-of-fold probabilities and OOF PR-AUC", "",
              "| Technique | OOF PR-AUC | best_f1 | best_cost | min_precision | 0.5 | Train IR "
              "(mean) | Fold-val positive % |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for item in ok:
        thresholds = item["thresholds"]
        lines.append(f"| `{item['technique']}` | {_fmt(item['oof_pr_auc'], 4)} | "
                     f"{_fmt(thresholds.get('best_f1'), 4)} | {_fmt(thresholds.get('best_cost'), 4)} | "
                     f"{_fmt(thresholds.get('min_precision'), 4)} | 0.5000 | {_train_summary(item)} | "
                     f"{_val_pos_range(item)} |")

    lines += ["", "## 5. Data leakage checks", ""]
    if not ok:
        lines.append("- No technique completed successfully.")
    for item in ok:
        detail = ", ".join(f"{name}: {'PASS' if value else 'FAIL'}"
                           for name, value in item["checks"].items())
        lines.append(f"- {'PASS' if all(item['checks'].values()) else 'FAIL'} - "
                     f"`{item['technique']}` ({detail})")
    lines += ["", "## 6. Conclusions and recommendations", "", *_conclusions(result), "",
              "## 7. Reproduction", "", "```powershell",
              "python -m pip install -r requirements-labs.txt",
              "python -m labs.imbalance_lab.techniques            # full catalog, writes artifacts",
              "python -m labs.imbalance_lab.techniques --quick --techniques smote,adasyn,focal_loss",
              "python -m unittest discover -s tests -v       # includes catalog and anti-leak tests",
              "```"]
    return "\n".join(lines) + "\n"


def _conclusions(result: Dict[str, Any]) -> List[str]:
    """Automatic conclusions derived from the numbers, including negative findings."""
    ok = [item for item in result["results"] if item["status"] == "ok"]
    if not ok:
        return ["No technique completed successfully; check the libraries and rerun."]
    lines: List[str] = []

    # 1. Compare against the baseline, the model with no imbalance handling.
    table = compare_with_baseline(result)
    if table["baseline"] is not None and table["n_techniques_compared"]:
        best_delta = max((row for row in table["rows"]
                          if row["technique"] != BASELINE_TECHNIQUE
                          and row.get("delta_pr_auc") is not None),
                         key=lambda row: float(row["delta_pr_auc"]), default=None)
        text = (f"**Against the baseline (untreated)**: baseline PR-AUC "
                f"{_fmt(table['baseline'].get('pr_auc'), 4)} / F1 "
                f"{_fmt(table['baseline'].get('f1'))}; "
                f"**{table['n_better_than_baseline']}/{table['n_techniques_compared']}** techniques "
                f"beat the baseline on PR-AUC")
        if best_delta is not None:
            text += (f", the highest being `{best_delta['technique']}` "
                     f"(dPR-AUC {float(best_delta['delta_pr_auc']):+.4f})")
        lines.append(text + ".")

    best_pr = max(ok, key=lambda item: item["oof_pr_auc"] or -1.0)
    lines.append(f"1. **Ranking quality (out-of-fold PR-AUC)** is best for `{best_pr['technique']}` "
                 f"({_fmt(best_pr['oof_pr_auc'], 4)}). PR-AUC is threshold-free, so it is the fairest "
                 "comparison between techniques.")

    gains = []
    for item in ok:
        tuned_row, half_row = _row_for(item, "best_f1"), _row_for(item, "fixed_0.5")
        if tuned_row and half_row:
            gain = float(tuned_row.get("f1", float("nan"))) - float(half_row.get("f1", float("nan")))
            if gain == gain:
                gains.append((item["technique"], gain))
    if gains:
        mean_gain = float(np.mean([gain for _name, gain in gains]))
        best_gain = max(gains, key=lambda pair: pair[1])
        worse = [name for name, gain in gains if gain < -1e-9]
        lines.append(
            f"2. **Threshold tuning on the PR curve instead of 0.5**: F1 changes by "
            f"**{mean_gain:+.3f}** on average, best for `{best_gain[0]}` ({best_gain[1]:+.3f})"
            + (f"; {len(worse)} techniques decrease ("
               + ", ".join(f"`{name}`" for name in worse)
               + "), so the best threshold is not always above 0.5." if worse else "."))

    cleaning = [item for item in ok if item["technique"] in ("tomek", "enn")]
    parts = []
    for item in cleaning:
        rows = item["resample_rows"]
        if rows:
            parts.append(f"`{item['technique']}`: IR "
                         f"{np.mean([row['train_ir_before'] for row in rows]):.1f} -> "
                         f"{np.mean([row['train_ir_after'] for row in rows]):.1f}")
    if parts:
        lines.append("3. **Tomek Links and ENN clean the boundary rather than balance the ratio** ("
                     + "; ".join(parts) + "): they only drop samples near the boundary, so the IR stays "
                     "close to its 98/2 value and balancing requires oversampling alongside them, as in "
                     "the hybrid group.")

    algorithm = [item for item in ok if item["group"] == "algorithm-level"]
    if algorithm:
        best = max(algorithm, key=lambda item: _row_for(item, "best_f1").get("f1", -1.0))
        row = _row_for(best, "best_f1")
        lines.append(f"4. **Algorithm-level** techniques leave the data unchanged; best F1 is "
                     f"`{best['technique']}` with F1={_fmt(row.get('f1'))}, recall="
                     f"{_fmt(row.get('recall'))}, PR-AUC={_fmt(row.get('pr_auc'), 4)}. Focal Loss uses a "
                     "custom objective with `gamma` and `alpha`, so it changes the shape of the loss "
                     "rather than only the class weights.")

    ensemble = [item for item in ok if item["group"] == "ensemble"]
    if ensemble:
        best = max(ensemble, key=lambda item: _row_for(item, "best_f1").get("f1", -1.0))
        row = _row_for(best, "best_f1")
        lines.append(f"5. **Ensemble** techniques: best F1 is `{best['technique']}` with "
                     f"F1={_fmt(row.get('f1'))}, PR-AUC={_fmt(row.get('pr_auc'), 4)}. Undersampling "
                     "happens inside `fit`, so it still only touches the fold train.")

    lines.append("6. **Recommendation for the main pipeline**: choose the technique by PR-AUC, pick the "
                 "threshold on the PR curve of out-of-fold probabilities rather than a default 0.5 or a "
                 "test-based choice, and prefer interventions that do not generate samples "
                 "(`class_weight`, `scale_pos_weight`, Focal Loss) when labels attach to entities, "
                 "because synthetic samples tend to fall inside the region of entities already in train.")
    return lines


def _write_artifacts(result: Dict[str, Any]) -> None:
    """Write `techniques.md`, `techniques.csv`, `techniques.json`, `techniques.log` and the by-fold CSV.

    A separate `techniques_comparison.csv` holds the baseline-versus-technique table.
    """
    out = C.ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "techniques.csv", result["rows"])
    payload = {key: value for key, value in result.items()
               if key not in ("log", "rows", "resample_rows", "fold_rows", "results")}
    # Drop the out-of-fold probability arrays so the JSON stays readable by any tool.
    payload["results"] = [{key: value for key, value in item.items()
                           if key not in ("oof_proba", "oof_y")}
                          for item in result["results"]]
    payload["comparison"] = compare_with_baseline(result)
    _write_json(out / "techniques.json", payload)
    _write_csv(out / "techniques_by_fold.csv", result["resample_rows"])
    _write_csv(out / "techniques_comparison.csv", compare_with_baseline(result)["rows"])
    (out / "techniques.log").write_text(result["log"] + "\n", encoding="utf-8")
    (out / "techniques.md").write_text(_markdown(result), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=C.CATALOG_N_SAMPLES,
                        help=f"Number of synthetic samples (default {C.CATALOG_N_SAMPLES}).")
    parser.add_argument("--cv", type=int, default=C.CATALOG_N_SPLITS,
                        help=f"StratifiedKFold splits over the train pool (default {C.CATALOG_N_SPLITS}).")
    parser.add_argument("--techniques", default="",
                        help="Comma-separated technique keys (default: all).")
    parser.add_argument("--no-imblearn", action="store_true",
                        help="Force the in-house sampler implementation instead of imbalanced-learn.")
    parser.add_argument("--quick", action="store_true",
                        help="Lighter grid with fewer ensemble estimators for a faster run.")
    parser.add_argument("--no-write", action="store_true", help="Do not write artifacts.")
    args = parser.parse_args(argv)
    keys = [key.strip() for key in args.techniques.split(",") if key.strip()]
    run(n_samples=args.n_samples, n_splits=args.cv, techniques=keys or None,
        prefer_imblearn=not args.no_imblearn, write=not args.no_write, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
