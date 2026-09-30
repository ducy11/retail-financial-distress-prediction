"""ĐO THỰC NGHIỆM tác động của các đặc trưng ĐỀ XUẤT lên cross-company AUROC.

Lệnh: python -m scripts.probe_features

Vì sao cần: ở đồ án này AUROC in-domain (~0.988) bị chi phối bởi baseline "nhớ mặt công ty"
(`ticker_prior` = 0.986), nên đặc trưng mới chỉ có giá trị nếu cải thiện **cross-company**
(GroupKFold theo công ty). Script so sánh trên CÙNG bộ fold → phép so ghép cặp theo fold:
- `A_baseline(41)`        : bộ feature hiện tại (`forecasting.features`),
- `B_baseline+ALL`        : thêm toàn bộ ~17 đặc trưng đề xuất,
- `C_proposed_ONLY`       : chỉ 17 đặc trưng đề xuất (kiểm tra feature gốc có dư thừa không),
- `P1..P5`                : từng nhóm đề xuất (coverage / accruals / days / path / scores).

Nguyên tắc chống rò rỉ: mọi đặc trưng mới chỉ dùng lịch sử của chính sample (`available_on ≤
as_of`); không dùng tag nào chưa có trong prepared. Không ghi artifact, không thuộc `run_all`.

Lưu ý khi đọc kết quả: n = 324 mẫu / 8 công ty, chỉ 5 fold có đủ 2 lớp để tính AUROC → chênh
lệch ±0.03 có thể là nhiễu; cần xác nhận bằng nested CV trước khi đổi `forecasting/features.py`.
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

ROOT = Path(__file__).resolve().parents[1]  # repo root (script nằm trong scripts/)
sys.path.insert(0, str(ROOT))

import forecasting.features as features  # noqa: E402
from forecasting.data_loader import load_prepared, to_float  # noqa: E402
from forecasting.features import build_feature_matrix, extract_labels, feature_names  # noqa: E402
from forecasting.models import DEFAULT_MODEL_ORDER, make_model, predict_proba  # noqa: E402

SPLITS = ("train", "validation", "test", "purged")
#: Công cụ khảo sát (không thuộc run_all): dùng đúng danh sách mô hình của registry.
MODELS = tuple(DEFAULT_MODEL_ORDER)
LOOKBACK = 8
SUF = "_vnd"

#: Lưới nhỏ cho nested CV (giữ thời gian chạy hợp lý; lưới đầy đủ ở `forecasting/tuning.py`).
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
    """~20 đặc trưng đề xuất, 5 nhóm: coverage / accruals / days / path / scores."""
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
    tl = ta - eq  # nợ phải trả SUY RA (tag `liabilities` chỉ phủ 39% số quý)

    # P1 — sửa độ phủ / cơ cấu vốn
    f["liab_derived_to_assets"] = _ratio(tl, ta)
    f["liab_derived_to_equity"] = _ratio(tl, eq)
    f["equity_to_assets"] = _ratio(eq, ta)

    # P2 — chất lượng lợi nhuận (accruals kiểu Sloan)
    f["accruals_to_assets"] = _ratio(ni - ocf, ta)
    ni4, ocf4 = _s(window, "net_income")[-4:], _s(window, "operating_cash_flow")[-4:]
    ok4 = len(ni4) == 4 and all(math.isfinite(x) for x in ni4 + ocf4)
    f["accruals4_to_assets"] = _ratio(sum(ni4) - sum(ocf4), ta) if ok4 else float("nan")

    # P3 — vòng quay theo ngày + lệch tăng trưởng so với doanh thu
    f["days_inventory"] = 91.25 * _ratio(inv, cogs)
    f["days_receivables"] = 91.25 * _ratio(ar, rev)
    for name, value in (("inventory", inv), ("selling_general_admin", sga)):
        key = f"{name}_growth_minus_revenue_growth"
        f[key] = (_yoy(value, _s(window, name)[-5]) - _yoy(rev, _s(window, "revenue")[-5])
                  if len(window) >= 5 else float("nan"))

    # P4 — đường đi/cực trị trong cửa sổ (không chỉ quý mới nhất)
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

    # P5 — điểm số học thuật: Altman Z (book value) và Ohlson O (bản rút gọn)
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
    """Ma trận thiết kế: 41 feature gốc (+ các cột đề xuất được chọn)."""
    X_base = build_feature_matrix(samples)
    if prop_cols is None:
        return X_base
    X_prop, names = _proposed_matrix(samples)
    return np.hstack([X_base, X_prop[:, [names.index(c) for c in prop_cols]]])


def _fold_splits(X, y, groups, n_splits):
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))


def _pooled_and_fold(model_name, X, y, splits):
    """(OOF AUROC gộp, [AUROC từng fold]) trên cùng bộ fold."""
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
    """Ma trận 41 feature của phiên bản TRƯỚC khi tích hợp P1+P4 (để so trước/sau).

    Tái lập bằng cách: (1) đưa `debt_to_assets` / `debt_to_equity` về dùng tag `liabilities`
    (không suy ra từ A = L + E), (2) bỏ 6 cột nhóm `path`.
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
    """Nested CV: chọn hyperparameter ở vòng TRONG (theo nhóm công ty), chấm điểm ở vòng NGOÀI.

    Vì sao cần: nếu chọn cấu hình rồi chấm ngay trên cùng fold thì điểm bị lạc quan hoá. Ở đây
    lưới nhỏ (`NESTED_GRIDS`) được tune bằng `StratifiedGroupKFold` chỉ trên phần train của fold
    ngoài; fold ngoài chỉ dùng để CHẤM ĐIỂM (không tham gia chọn cấu hình).
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
                        help="Thêm nested CV (tune ở vòng trong) cho các biến thể chính.")
    parser.add_argument("--models", default="", help="Mô hình cho nested CV, phân tách bằng dấu phẩy.")
    args = parser.parse_args(argv)
    splits = {n: load_prepared(n) for n in SPLITS}
    pool = [s for n in SPLITS for s in splits[n]]
    y = extract_labels(pool)
    groups = np.asarray([s["ticker"] for s in pool])

    # kiểm chứng harness: số cột phải khớp `feature_names()` của pipeline hiện tại
    X_base = design(pool)
    print(f"Harness: {X_base.shape[1]} feature gốc = {len(feature_names())} (khớp mong đợi: "
          f"{X_base.shape[1] == len(feature_names())})")

    _, prop_names = _proposed_matrix(pool)
    X_prop = _proposed_matrix(pool)[0]
    print("\n=== Độ phủ đặc trưng đề xuất (% mẫu có giá trị) ===")
    for j, n in enumerate(prop_names):
        print(f"  {n:46s} {float(np.mean(np.isfinite(X_prop[:, j]))):6.1%}")

    key_pipeline = f"A_pipeline({len(feature_names())})"
    configs: Dict[str, Any] = {key_pipeline: None,
                               "B_pipeline+ALL(đề xuất)": [c for c in prop_names],
                               "C_proposed_ONLY": "PROPOSED"}
    for gname, cols in PROPOSED_GROUPS.items():
        configs[f"{gname} (+{len(cols)})"] = cols

    n_splits = len(set(groups.tolist()))
    splits_ref = _fold_splits(X_base, y, groups, n_splits)
    print(f"\n=== GroupKFold theo công ty: {n_splits} fold, {len(y)} mẫu ===")
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
            print(f"   {m:22s} OOF={pooled:.3f}  Δfold_tb={mean_d:+.3f}  "
                  f"({better}/{len(deltas)} fold tốt hơn)")

    print("\n=== LOCO (bỏ từng công ty; chỉ công ty có 2 lớp mới tính được) ===")
    for cname in (key_pipeline, "B_pipeline+ALL(đề xuất)", "C_proposed_ONLY"):
        X = X_prop if configs[cname] == "PROPOSED" else (
            X_base if configs[cname] is None else design(pool, configs[cname]))
        parts = []
        for m in MODELS:
            s = _loco(m, X, y, groups)
            parts.append(f"{m[:12]}={np.mean(s):.3f}(n={len(s)})")
        print(f"  {cname:26s} " + "  ".join(parts))

    print("\n=== In-domain train→test (logistic) ===")
    print(f"  pipeline            {_in_domain('logistic', splits['train'], splits['test'], None):.3f}")
    print(f"  pipeline+ALL(đề xuất) "
          f"{_in_domain('logistic', splits['train'], splits['test'], list(prop_names)):.3f}")

    if args.nested:
        models = [m for m in (args.models.split(",") if args.models else MODELS) if m]
        print(f"\n=== NESTED CV: GroupKFold ngoài ({n_splits} fold) × "
              f"StratifiedGroupKFold trong (4) — chọn cấu hình ở vòng trong ===")
        variants: Dict[str, Any] = {
            "LEGACY_41 (trước tích hợp)": _legacy_matrix,
            key_pipeline + " (sau tích hợp)": design,
            "LEGACY_41+P1+P4 (đề xuất đo riêng)": lambda s: design(
                s, PROPOSED_GROUPS["P1_coverage"] + PROPOSED_GROUPS["P4_path"]),
        }
        ref: Dict[str, Dict[str, Any]] = {}
        for label, builder in variants.items():
            X = builder(pool)
            print(f"\n{label}  [n_feature={X.shape[1]}]")
            for m in models:
                res = _nested_oof(m, X, y, groups, splits_ref)
                if label.startswith("LEGACY_41 (t"):
                    ref[m] = res
                folds = res["folds"]
                finite = [f for f in folds if math.isfinite(f)]
                base_folds = ref.get(m, {}).get("folds", [])
                deltas = [f - b for f, b in zip(folds, base_folds)
                          if math.isfinite(f) and math.isfinite(b)] if base_folds else []
                extra = (f"  Δfold_vs_LEGACY={float(np.mean(deltas)):+.3f} "
                         f"({sum(1 for d in deltas if d > 0)}/{len(deltas)} fold tốt hơn)"
                         if deltas else "")
                oof = res["oof_auroc"]
                print(f"   {m:22s} OOF={oof:.3f}  fold_tb={np.mean(finite):.3f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


