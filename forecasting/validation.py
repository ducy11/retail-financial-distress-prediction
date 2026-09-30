"""Kiểm chứng tổng quát hoá và độ bất định — phần "chống tự lừa mình" của đồ án.

Gồm 4 nhóm:
1. `grouped_cv` — GroupKFold theo công ty: mỗi fold giữ TOÀN BỘ một số công ty ra khỏi train,
   đo đúng câu hỏi "mô hình có dự báo được công ty chưa từng thấy không?".
2. `leave_one_company_out` — LOCO: bỏ từng công ty, train trên 7 công ty còn lại, test trên
   công ty bị bỏ (ghép cả validation + test của công ty đó).
3. `bootstrap_ci` — khoảng tin cậy 95% cho AUROC/AP/F1 bằng bootstrap (n nhỏ nên bắt buộc).
4. `agreement` — tương quan hạng giữa xác suất các mô hình (vì sao metric có thể trùng khít).

Lệnh: python -m forecasting.validation  → ghi reports/results/validation_checks.json
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .config import GROUP_KEY, N_BOOTSTRAP, RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from .data_loader import load_prepared
from .evaluation import metrics_at_threshold
from .features import build_feature_matrix, extract_labels
from .models import MODEL_REGISTRY, make_model, predict_proba


def grouped_cv(model_names: Sequence[str], samples: List[Dict[str, Any]],
               n_splits: int = 4, min_test: int = 4) -> Dict[str, Any]:
    """GroupKFold theo công ty; trả metric out-of-fold (gộp toàn bộ fold) cho từng mô hình."""
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
    """LOCO theo thời gian: train trên 7 công ty (train split), test trên công ty bị bỏ.

    Dùng cả validation + test của công ty bị bỏ để có đủ mẫu; công ty có nhãn đơn lớp thì
    AUROC không xác định (ghi rõ `single_class`) — đó cũng là một phát hiện cần báo cáo.
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
    """Khoảng tin cậy 95% (percentile bootstrap) cho AUROC / AP / F1 / macro-F1."""
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
    """Tương quan hạng (Spearman) giữa xác suất các mô hình — giải thích metric trùng khít."""
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


#: Các mô hình đưa vào kiểm chứng = đúng 3 họ mô hình của đồ án.
MODELS_FOR_CHECKS = ["logistic", "random_forest", "hist_gradient_boosting"]


def run(model_names: Sequence[str] | None = None) -> Dict[str, Any]:
    """Chạy toàn bộ kiểm chứng: GroupKFold, LOCO, bootstrap CI, agreement."""
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
        }

    out: Dict[str, Any] = {
        "models": names,
        "grouped_cv_all_samples": cv,
        "leave_one_company_out": loco,
        "in_domain_test": in_domain,
        "agreement": agreement(names, splits["train"], splits["test"]),
        "headline": {},
    }
    print(f"  {'mô hình':24s} {'in-domain test AUROC':>21s} {'cross-company OOF AUROC':>24s}")
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
    print(f"  LOCO: mean AUROC={out['headline']['loco_mean_auroc']} trên "
          f"{len(loco_scores)} phép so; bỏ qua (nhãn đơn lớp): {skipped}")

    (RESULTS_DIR / "validation_checks.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    print("=== Kiểm chứng tổng quát hoá & độ bất định ===")
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
