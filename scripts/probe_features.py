"""Measure the effect of the proposed features on cross-company AUROC.

Compares the current feature set, the set plus every proposed feature, the proposed features alone and
each proposed group on the same company-grouped folds, so paired-by-fold deltas stay meaningful. Writes
no artifact and stays out of `scripts.run_all`.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]  # repo root, since this script lives in scripts/
sys.path.insert(0, str(ROOT))

import forecasting.features as features  # noqa: E402
from forecasting.data_loader import load_prepared, to_float  # noqa: E402
from forecasting.features import build_feature_matrix, extract_labels, feature_names  # noqa: E402
from forecasting.models import DEFAULT_MODEL_ORDER, make_model, predict_proba  # noqa: E402

SPLITS = ("train", "validation", "test", "purged")
#: Survey tool outside `run_all`; reuses the model list from the registry.
MODELS = tuple(DEFAULT_MODEL_ORDER)
LOOKBACK = 8
SUF = "_vnd"

#: Small grid for nested CV to keep runtime sane; the full grid lives in `forecasting/tuning.py`.
NESTED_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic": {"model__C": [0.03, 0.1, 0.3, 1.0]},
    "random_forest": {"model__max_depth": [3, 6], "model__min_samples_leaf": [2, 4]},
    "hist_gradient_boosting": {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]},
    "mlp": {"model__alpha": [1e-3]},
}


def _s(window, field):
    return [to_float(r.get(field + SUF)) for r in window]


def _last(values):
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def _ratio(num, den):
    if not (math.isfinite(num) and math.isfinite(den)) or den == 0:
        return float("nan")
    return num / den


def _yoy(cur, past):
    if not (math.isfinite(cur) and math.isfinite(past)) or past == 0:
        return float("nan")
    return (cur - past) / abs(past)


def _min_finite(values):
    vals = [v for v in values if math.isfinite(v)]
    return min(vals) if vals else float("nan")


def _streak(values, pred):
    n = 0
    for v in reversed(values):
        if math.isfinite(v) and pred(v):
            n += 1
        else:
            break
    return float(n)


def proposed_features(sample: Dict[str, Any]) -> Dict[str, float]:
    """Around 20 proposed features across five groups: coverage, accruals, days, path and scores."""
    window = sample["request"]["history"][-LOOKBACK:]
    f: Dict[str, float] = {}
    if not window:
        return f
    ta, eq = _last(_s(window, "total_assets")), _last(_s(window, "stockholders_equity"))
    ca, cl = _last(_s(window, "current_assets")), _last(_s(window, "current_liabilities"))
    inv, ar = _last(_s(window, "inventory")), _last(_s(window, "receivables"))
    rev, cogs = _last(_s(window, "revenue")), _last(_s(window, "cost_of_sales"))
    ni, ocf = _last(_s(window, "net_income")), _last(_s(window, "operating_cash_flow"))
    oi, re = _last(_s(window, "operating_income")), _last(_s(window, "retained_earnings"))
    sga = _last(_s(window, "selling_general_admin"))
    tl = ta - eq  # Liabilities derived from A = L + E, since the tag covers only 39 percent of quarters.

    # P1: coverage and capital structure.
    f["liab_derived_to_assets"] = _ratio(tl, ta)
    f["liab_derived_to_equity"] = _ratio(tl, eq)
    f["equity_to_assets"] = _ratio(eq, ta)

    # P2: earnings quality using Sloan-style accruals.
    f["accruals_to_assets"] = _ratio(ni - ocf, ta)
    ni4, ocf4 = _s(window, "net_income")[-4:], _s(window, "operating_cash_flow")[-4:]
    ok4 = len(ni4) == 4 and all(math.isfinite(x) for x in ni4 + ocf4)
    f["accruals4_to_assets"] = _ratio(sum(ni4) - sum(ocf4), ta) if ok4 else float("nan")

    # P3: turnover in days and growth that deviates from revenue growth.
    f["days_inventory"] = 91.25 * _ratio(inv, cogs)
    f["days_receivables"] = 91.25 * _ratio(ar, rev)
    for name, value in (("inventory", inv), ("selling_general_admin", sga)):
        key = f"{name}_growth_minus_revenue_growth"
        f[key] = (_yoy(value, _s(window, name)[-5]) - _yoy(rev, _s(window, "revenue")[-5])
                  if len(window) >= 5 else float("nan"))

    # P4: path extremes across the window, not just the latest quarter.
    f["current_ratio_min_8q"] = _min_finite(
        [_ratio(a, b) for a, b in zip(_s(window, "current_assets"),
                                      _s(window, "current_liabilities"))])
    f["ocf_to_sales_min_8q"] = _min_finite(
        [_ratio(a, b) for a, b in zip(_s(window, "operating_cash_flow"), _s(window, "revenue"))])
    f["net_margin_min_8q"] = _min_finite(
        [_ratio(a, b) for a, b in zip(_s(window, "net_income"), _s(window, "revenue"))])
    revs = [v for v in _s(window, "revenue") if math.isfinite(v)]
    f["revenue_drawdown_8q"] = (rev / max(revs) - 1.0
                                if revs and math.isfinite(rev) and max(revs) > 0 else float("nan"))
    f["neg_ni_streak"] = _streak(_s(window, "net_income"), lambda x: x < 0)
    f["neg_ocf_streak"] = _streak(_s(window, "operating_cash_flow"), lambda x: x < 0)

    # P5: academic scores, the book-value Altman Z and a reduced Ohlson O.
    wc = ca - cl
    z = (1.2 * _ratio(wc, ta) + 1.4 * _ratio(re, ta) + 3.3 * _ratio(oi, ta)
         + 0.6 * _ratio(eq, tl) + 1.0 * _ratio(rev, ta))
    f["altman_z_book"] = z if math.isfinite(z) else float("nan")
    ni_prev = _s(window, "net_income")[-2] if len(window) >= 2 else float("nan")
    size_term = -0.407 * math.log(ta) if math.isfinite(ta) and ta > 0 else 0.0
    o = (-1.32 + size_term + 6.03 * _ratio(tl, ta) - 1.43 * _ratio(wc, ta)
         + 0.0757 * _ratio(cl, ca)
         - 1.72 * (1.0 if (math.isfinite(tl) and math.isfinite(ta) and tl > ta) else 0.0)
         - 2.37 * _ratio(ni, ta) - 1.83 * _ratio(ocf, tl)
         + 0.285 * f["neg_ni_streak"]
         - 0.521 * _ratio(ni - ni_prev, abs(ni) + abs(ni_prev)))
    f["ohlson_o"] = o if math.isfinite(o) else float("nan")
    return f


PROPOSED_GROUPS = {
    "P1_coverage": ["liab_derived_to_assets", "liab_derived_to_equity", "equity_to_assets"],
    "P2_accruals": ["accruals_to_assets", "accruals4_to_assets"],
    "P3_days": ["days_inventory", "days_receivables", "inventory_growth_minus_revenue_growth",
                "selling_general_admin_growth_minus_revenue_growth"],
    "P4_path": ["current_ratio_min_8q", "ocf_to_sales_min_8q", "net_margin_min_8q",
                "revenue_drawdown_8q", "neg_ni_streak", "neg_ocf_streak"],
    "P5_scores": ["altman_z_book", "ohlson_o"],
}


def _proposed_matrix(samples: Sequence[Dict[str, Any]]):
    names = sorted(proposed_features(samples[0]).keys())
    X = np.full((len(samples), len(names)), np.nan)
    for i, s in enumerate(samples):
        feats = proposed_features(s)
        for j, n in enumerate(names):
            X[i, j] = feats.get(n, float("nan"))
    return X, names


def design(samples, prop_cols=None):
    """Design matrix holding the base features plus any selected proposed columns."""
    X_base = build_feature_matrix(samples)
    if prop_cols is None:
        return X_base
    X_prop, names = _proposed_matrix(samples)
    return np.hstack([X_base, X_prop[:, [names.index(c) for c in prop_cols]]])


def _fold_splits(X, y, groups, n_splits):
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))


def _pooled_and_fold(model_name, X, y, splits):
    """Return pooled out-of-fold AUROC and the per-fold AUROC values on the same fold set."""
    proba = np.full(len(y), np.nan)
    folds = []
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2:
            folds.append(float("nan"))
            continue
        model = make_model(model_name)
        model.fit(X[tr], y[tr])
        p = predict_proba(model, X[te])
        proba[te] = p
        folds.append(float(roc_auc_score(y[te], p)) if len(set(y[te].tolist())) > 1
                     else float("nan"))
    mask = np.isfinite(proba)
    return float(roc_auc_score(y[mask], proba[mask])), folds


def _legacy_matrix(samples: Sequence[Dict[str, Any]]):
    """Feature matrix of the version before P1 and P4 were integrated, kept as a before/after baseline.

    Reconstructed by pointing `debt_to_assets` and `debt_to_equity` back at the `liabilities` tag and
    dropping the six path features.
    """
    original = dict(features.RATIO_PARTS)
    features.RATIO_PARTS["debt_to_assets"] = ("liabilities", "total_assets")
    features.RATIO_PARTS["debt_to_equity"] = ("liabilities", "stockholders_equity")
    try:
        X = build_feature_matrix(samples)
        names = feature_names()
    finally:
        features.RATIO_PARTS.clear()
        features.RATIO_PARTS.update(original)
    keep = [i for i, name in enumerate(names) if name not in set(features.PATH_FEATURES)]
    return X[:, keep]


def _nested_oof(model_name: str, X, y, groups, outer_splits: List[Any],
                n_inner: int = 4) -> Dict[str, Any]:
    """Nested CV: tune hyperparameters on the inner loop and score on the outer fold.

    Selecting a configuration and then scoring it on the same fold inflates the result, so the small
    `NESTED_GRIDS` grid is tuned with StratifiedGroupKFold on the outer train part only, and the outer
    fold is used purely for scoring.
    """
    proba = np.full(len(y), np.nan)
    folds: List[float] = []
    chosen: List[Dict[str, Any]] = []
    for tr, te in outer_splits:
        if len(set(y[tr].tolist())) < 2:
            folds.append(float("nan"))
            continue
        search = GridSearchCV(
            make_model(model_name), NESTED_GRIDS[model_name], scoring="average_precision",
            cv=StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=42),
            n_jobs=1, return_train_score=False)
        search.fit(X[tr], y[tr], groups=groups[tr])
        p = predict_proba(search.best_estimator_, X[te])
        proba[te] = p
        chosen.append({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        folds.append(float(roc_auc_score(y[te], p)) if len(set(y[te].tolist())) > 1
                     else float("nan"))
    mask = np.isfinite(proba)
    return {"oof_auroc": float(roc_auc_score(y[mask], proba[mask])) if mask.any() else None,
            "folds": folds, "best_params": chosen}


def _loco(model_name, X, y, groups):
    scores = []
    for g in sorted(set(groups.tolist())):
        te = groups == g
        tr = ~te
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        model = make_model(model_name)
        model.fit(X[tr], y[tr])
        scores.append(float(roc_auc_score(y[te], predict_proba(model, X[te]))))
    return scores


def _in_domain(model_name, train_s, test_s, prop_cols):
    X_tr = design(train_s, prop_cols)
    y_tr = extract_labels(train_s)
    X_te = design(test_s, prop_cols)
    y_te = extract_labels(test_s)
    model = make_model(model_name)
    model.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, predict_proba(model, X_te)))


def main(argv=None) -> int:
    from forecasting.config import ensure_utf8_stdio

    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested", action="store_true",
                        help="Also run nested CV with inner-loop tuning for the main variants.")
    parser.add_argument("--models", default="", help="Models for nested CV, comma-separated.")
    args = parser.parse_args(argv)
    splits = {n: load_prepared(n) for n in SPLITS}
    pool = [s for n in SPLITS for s in splits[n]]
    y = extract_labels(pool)
    groups = np.asarray([s["ticker"] for s in pool])

    # Harness check: the column count must match the current pipeline's `feature_names()`.
    X_base = design(pool)
    print(f"Harness: {X_base.shape[1]} base features = {len(feature_names())} (expected match: "
          f"{X_base.shape[1] == len(feature_names())})")

    _, prop_names = _proposed_matrix(pool)
    X_prop = _proposed_matrix(pool)[0]
    print("\n=== Proposed feature coverage (share of samples with a value) ===")
    for j, n in enumerate(prop_names):
        print(f"  {n:46s} {float(np.mean(np.isfinite(X_prop[:, j]))):6.1%}")

    key_pipeline = f"A_pipeline({len(feature_names())})"
    configs: Dict[str, Any] = {key_pipeline: None,
                               "B_pipeline+ALL(proposed)": [c for c in prop_names],
                               "C_proposed_ONLY": "PROPOSED"}
    for gname, cols in PROPOSED_GROUPS.items():
        configs[f"{gname} (+{len(cols)})"] = cols

    n_splits = len(set(groups.tolist()))
    splits_ref = _fold_splits(X_base, y, groups, n_splits)
    print(f"\n=== GroupKFold by company: {n_splits} folds, {len(y)} samples ===")
    reference: Dict[str, tuple] = {}
    for m in MODELS:
        reference[m] = _pooled_and_fold(m, X_base, y, splits_ref)

    for cname, cols in configs.items():
        is_base = cname == key_pipeline
        if cols == "PROPOSED":
            X = X_prop
        elif cols is None:
            X = X_base
        else:
            X = design(pool, cols)
        print(f"\n{cname}  [n_feature={X.shape[1]}]")
        for m in MODELS:
            if is_base:
                pooled, folds = reference[m]
            else:
                pooled, folds = _pooled_and_fold(m, X, y, splits_ref)
            deltas = [f - b for f, b in zip(folds, reference[m][1])
                      if math.isfinite(f) and math.isfinite(b)]
            mean_d = float(np.mean(deltas)) if deltas else float("nan")
            better = sum(1 for d in deltas if d > 0)
            print(f"   {m:22s} OOF={pooled:.3f}  dFold_mean={mean_d:+.3f}  "
                  f"({better}/{len(deltas)} folds better)")

    print("\n=== LOCO (drop each company; only two-class companies are scorable) ===")
    for cname in (key_pipeline, "B_pipeline+ALL(proposed)", "C_proposed_ONLY"):
        X = X_prop if configs[cname] == "PROPOSED" else (
            X_base if configs[cname] is None else design(pool, configs[cname]))
        parts = []
        for m in MODELS:
            s = _loco(m, X, y, groups)
            parts.append(f"{m[:12]}={np.mean(s):.3f}(n={len(s)})")
        print(f"  {cname:26s} " + "  ".join(parts))

    print("\n=== In-domain train to test (logistic) ===")
    print(f"  pipeline            {_in_domain('logistic', splits['train'], splits['test'], None):.3f}")
    print(f"  pipeline+ALL(proposed) "
          f"{_in_domain('logistic', splits['train'], splits['test'], list(prop_names)):.3f}")

    if args.nested:
        models = [m for m in (args.models.split(",") if args.models else MODELS) if m]
        print(f"\n=== NESTED CV: outer GroupKFold ({n_splits} folds) x "
              f"inner StratifiedGroupKFold (4); configuration chosen in the inner loop ===")
        variants: Dict[str, Any] = {
            "LEGACY_41 (before integration)": _legacy_matrix,
            key_pipeline + " (after integration)": design,
            "LEGACY_41+P1+P4 (measured separately)": lambda s: design(
                s, PROPOSED_GROUPS["P1_coverage"] + PROPOSED_GROUPS["P4_path"]),
        }
        ref: Dict[str, Dict[str, Any]] = {}
        for label, builder in variants.items():
            X = builder(pool)
            print(f"\n{label}  [n_feature={X.shape[1]}]")
            for m in models:
                res = _nested_oof(m, X, y, groups, splits_ref)
                if label.startswith("LEGACY_41 (b"):
                    ref[m] = res
                folds = res["folds"]
                finite = [f for f in folds if math.isfinite(f)]
                base_folds = ref.get(m, {}).get("folds", [])
                deltas = [f - b for f, b in zip(folds, base_folds)
                          if math.isfinite(f) and math.isfinite(b)] if base_folds else []
                extra = (f"  dFold_vs_LEGACY={float(np.mean(deltas)):+.3f} "
                         f"({sum(1 for d in deltas if d > 0)}/{len(deltas)} folds better)"
                         if deltas else "")
                oof = res["oof_auroc"]
                print(f"   {m:22s} OOF={oof:.3f}  fold_mean={np.mean(finite):.3f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


