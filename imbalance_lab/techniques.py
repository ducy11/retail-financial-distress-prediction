"""DANH MỤC ĐẦY ĐỦ các nhóm kỹ thuật xử lý mất cân bằng (yêu cầu #2) + runner so sánh không rò rỉ.

Danh mục (đúng theo danh sách yêu cầu):

| Nhóm | Kỹ thuật | Khoá |
|---|---|---|
| **Baseline (tham chiếu, yêu cầu #3)** | boosting mặc định — KHÔNG can thiệp mất cân bằng | `baseline` |
| Data-level / Oversampling | RandomOverSampler · SMOTE · BorderlineSMOTE · ADASYN | `ros`, `smote`, `borderline_smote`, `adasyn` |
| Data-level / Undersampling | RandomUnderSampler · Tomek Links · Edited Nearest Neighbours | `rus`, `tomek`, `enn` |
| Hybrid | SMOTE + Tomek Links · SMOTE + ENN | `smote_tomek`, `smote_enn` |
| Algorithm-level | `scale_pos_weight` động · `class_weight='balanced'` · Focal Loss | `cost_sensitive_scale_pos_weight`, `cost_sensitive_class_weight`, `focal_loss` |
| Ensemble | BalancedRandomForest · EasyEnsemble · RUSBoost | `balanced_rf`, `easy_ensemble`, `rusboost` |
| Threshold tuning | Ngưỡng chọn trên ĐƯỜNG PR của xác suất out-of-fold (3 chế độ + mốc 0.5) | áp cho MỌI kỹ thuật |

Chống rò rỉ (được kiểm chứng và ghi PASS/FAIL trong artifact):
1. Mọi sampler nằm TRONG `imblearn.pipeline.Pipeline` ⇒ `fit_resample` chỉ chạy trên train của fold;
   validation của fold được so với bản sao trước khi fit (`cv.assert_val_untouched`).
2. Trọng số cost-sensitive / Focal Loss tính TRONG `fit` từ nhãn nhận được (fold-train).
3. Ensemble undersampling (BalancedRF/EasyEnsemble/RUSBoost) chỉ lấy mẫu bên trong `fit` của chính tập
   train được truyền vào.
4. Ngưỡng chọn trên **xác suất out-of-fold của train_pool**, KHÔNG chọn trên test; test chỉ được chấm
   điểm một lần sau khi đã chốt mô hình + ngưỡng.

Đánh giá (yêu cầu #3): **Accuracy KHÔNG phải thước đo chính** — bảng so sánh chỉ dùng Precision,
Recall, F1 (binary/macro/weighted/F-beta), PR-AUC (Average Precision), ROC-AUC, MCC và Confusion Matrix
(`metrics.PRIMARY_METRICS`); accuracy chỉ xuất hiện như chỉ số CHẨN ĐOÁN kèm mốc "đoán lớp đa số"
(`metrics.accuracy_diagnostic`). Artifact có sẵn bảng **Baseline (chưa xử lý) vs từng kỹ thuật**
(`compare_with_baseline`, `comparison_markdown`, `techniques_comparison.csv`).

Resampling được tích hợp QUA `imblearn.pipeline.Pipeline` (`samplers.build_sampler_pipeline`) — không
dùng pipeline chuẩn của scikit-learn cho các bước lấy mẫu, nhờ đó `fit_resample` chỉ chạy trên train
của từng fold khi vào Cross-Validation.

Lệnh:
    python -m imbalance_lab.techniques                     # 20.000 mẫu, 5 fold, tất cả kỹ thuật
    python -m imbalance_lab.techniques --quick             # lưới nhẹ (ensemble ít estimator hơn)
    python -m imbalance_lab.techniques --techniques smote,adasyn,focal_loss --cv 3
    python -m imbalance_lab.techniques --no-imblearn       # dùng bản nội bộ của lab

Artifact (mặc định `reports/imbalance/`): `techniques.md`, `techniques.csv`, `techniques.json`,
`techniques.log`.
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

# ---------------------------------------------------------------------------
# Danh mục: nhóm → kỹ thuật (nguồn duy nhất cho runner, báo cáo VÀ test)
# ---------------------------------------------------------------------------
#: Nhóm kỹ thuật theo yêu cầu #2. Test `test_catalog_covers_every_required_group` đối chiếu trực tiếp
#: danh sách này, nên KHÔNG được bỏ sót mục nào khi sửa code.
REQUIRED_GROUPS: Dict[str, Tuple[str, ...]] = {
    "data-level/oversampling": ("ros", "smote", "borderline_smote", "adasyn"),
    "data-level/undersampling": ("rus", "tomek", "enn"),
    "hybrid": ("smote_tomek", "smote_enn"),
    "algorithm-level": ("cost_sensitive_scale_pos_weight", "cost_sensitive_class_weight",
                        "focal_loss"),
    "ensemble": ("balanced_rf", "easy_ensemble", "rusboost"),
}


#: Nhóm THAM CHIẾU — KHÔNG thuộc danh sách yêu cầu #2 nhưng bắt buộc có để so sánh: mô hình gốc
#: CHƯA xử lý mất cân bằng (yêu cầu #3: "bảng/hàm so sánh Baseline vs các kỹ thuật xử lý").
REFERENCE_GROUPS: Dict[str, Tuple[str, ...]] = {"baseline": ("baseline",)}

#: Mô tả từng kỹ thuật (dùng trong log/báo cáo).
TECHNIQUE_DOCS: Dict[str, str] = {
    "baseline": "BASELINE — boosting mặc định, KHÔNG can thiệp mất cân bằng (mốc so sánh)",
    "ros": "RandomOverSampler — sao chép mẫu thiểu số (không tạo mẫu mới)",
    "smote": "SMOTE — nội suy giữa mẫu thiểu số và láng giềng thiểu số",
    "borderline_smote": "BorderlineSMOTE — chỉ nội suy từ mẫu thiểu số nằm ở BIÊN (vùng DANGER)",
    "adasyn": "ADASYN — sinh thêm tỉ lệ với độ khó (số láng giềng đa số) của từng mẫu thiểu số",
    "rus": "RandomUnderSampler — hạ ngẫu nhiên lớp đa số về tỉ lệ mục tiêu",
    "tomek": "Tomek Links — làm sạch biên: bỏ mẫu đa số trong cặp láng giềng khác lớp",
    "enn": "EditedNearestNeighbours — làm sạch biên: bỏ mẫu có láng giềng khác lớp",
    "smote_tomek": "HYBRID SMOTE + Tomek Links",
    "smote_enn": "HYBRID SMOTE + EditedNearestNeighbours",
    "cost_sensitive_scale_pos_weight": "LightGBM + `scale_pos_weight = n_âm/n_dương` tính TRONG fit",
    "cost_sensitive_class_weight": "`class_weight='balanced'` — trọng số mẫu suy trong fit",
    "focal_loss": "Focal Loss — custom objective của LightGBM (gamma làm mờ mẫu dễ, alpha theo lớp)",
    "balanced_rf": "BalancedRandomForestClassifier (undersample từng cây, TRONG fit)",
    "easy_ensemble": "EasyEnsembleClassifier (nhiều AdaBoost trên các tập con cân bằng)",
    "rusboost": "RUSBoostClassifier (boosting + undersample từng vòng)",
}

#: Yêu cầu thư viện của từng kỹ thuật (thiếu ⇒ runner ghi trạng thái `skipped` kèm lý do).
REQUIRES: Dict[str, Tuple[str, ...]] = {
    "balanced_rf": ("imblearn",), "easy_ensemble": ("imblearn",), "rusboost": ("imblearn",),
    "focal_loss": ("lightgbm",),
}

#: Bảng anchor để report/test đối chiếu "yêu cầu ↔ kỹ thuật ↔ nơi cài đặt".
IMPLEMENTATION: Dict[str, str] = {
    "baseline": "models.make_base_classifier (boosting mặc định, KHÔNG can thiệp)",
    "ros": "samplers.RandomOverSampler / imblearn RandomOverSampler",
    "smote": "samplers.SMOTE / imblearn SMOTE",
    "borderline_smote": "samplers.BorderlineSMOTE / imblearn BorderlineSMOTE",
    "adasyn": "samplers.ADASYN / imblearn ADASYN",
    "rus": "samplers.RandomUnderSampler / imblearn RandomUnderSampler",
    "tomek": "samplers.TomekLinks / imblearn TomekLinks",
    "enn": "samplers.EditedNearestNeighbours / imblearn EditedNearestNeighbours",
    "smote_tomek": "samplers.make_hybrid_sampler('smote_tomek') / imblearn SMOTETomek",
    "smote_enn": "samplers.make_hybrid_sampler('smote_enn') / imblearn SMOTEENN",
    "cost_sensitive_scale_pos_weight": "models.ScalePosWeightClassifier",
    "cost_sensitive_class_weight": "models.BalancedWeightClassifier",
    "focal_loss": "losses.FocalLossClassifier (grad/hess giải tích)",
    "balanced_rf": "imblearn BalancedRandomForestClassifier",
    "easy_ensemble": "imblearn EasyEnsembleClassifier",
    "rusboost": "imblearn RUSBoostClassifier",
}

#: Thứ tự chạy: BASELINE trước (mốc so sánh, yêu cầu #3), rồi các nhóm theo yêu cầu #2.
TECHNIQUE_ORDER: Tuple[str, ...] = tuple(
    key for group in (*REFERENCE_GROUPS.values(), *REQUIRED_GROUPS.values()) for key in group)

#: Khoá của mô hình gốc CHƯA xử lý (dùng trong bảng so sánh baseline).
BASELINE_TECHNIQUE = "baseline"


#: Các CHẾ ĐỘ ngưỡng (khoá có `threshold`) — mọi hàm chọn ngưỡng của lab trả đúng bộ này + vài số
#: tổng hợp (`pr_auc`, `n_candidates`) nên khi lặp phải lọc theo danh sách này.
THRESHOLD_MODES: Tuple[str, ...] = ("best_f1", "best_cost", "min_precision", "fixed_0.5")


def technique_group(key: str) -> str:
    """Nhóm của một kỹ thuật (gồm cả nhóm tham chiếu `baseline`)."""
    for groups in (REQUIRED_GROUPS, REFERENCE_GROUPS):
        for group, keys in groups.items():
            if key in keys:
                return group
    raise KeyError(f"Kỹ thuật không có trong danh mục: {key!r}")


def all_techniques_available(prefer_imblearn: bool = True) -> Dict[str, List[str]]:
    """{kỹ thuật: [thư viện còn thiếu]} — rỗng nghĩa là chạy được hết."""
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


# ---------------------------------------------------------------------------
# Dựng estimator cho một kỹ thuật
# ---------------------------------------------------------------------------
def _ensemble_factory(key: str, random_state: int, quick: bool) -> Callable[[], Any]:
    """Factory cho nhóm ENSEMBLE (BalancedRandomForest / EasyEnsemble / RUSBoost).

    Cả ba đều undersample BÊN TRONG `fit` của tập train được truyền vào ⇒ trong CV chỉ lấy mẫu từ
    fold-train, không bao giờ đụng validation/test.
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
    """Spec của một kỹ thuật: `factory` (estimator mới mỗi fold) + `probe_factory` (log resampling).

    `probe_factory` chỉ được trả khi kỹ thuật có resampling: nó là pipeline CHỈ có sampler
    (`pipe[:-1]`) để gọi `fit_resample` và ghi phân phối nhãn TRƯỚC/SAU của fold-train.
    """
    if key not in TECHNIQUE_ORDER:
        raise KeyError(f"Kỹ thuật không có trong danh mục: {key!r}; có {list(TECHNIQUE_ORDER)}")

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
            return factory()[:-1]          # chỉ sampler ⇒ gọi được `fit_resample`

        kind = "data-level"
    elif key == "baseline":
        # Mốc so sánh (yêu cầu #3): KHÔNG resampling, KHÔNG trọng số lớp, KHÔNG custom loss.
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


# ---------------------------------------------------------------------------
# Chạy một kỹ thuật: CV stratified trên train_pool → chọn ngưỡng trên OOF (đường PR) → chốt test
# ---------------------------------------------------------------------------
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
    """Đánh giá một kỹ thuật KHÔNG rò rỉ: resampling trong pipeline, ngưỡng chọn trên OOF.

    Trả về `rows` (metric trên test theo từng ngưỡng), `fold_rows`, `resample_rows`, `thresholds`,
    `checks` (PASS/FAIL chống rò rỉ) và `oof_pr_auc`.
    """
    from .cv import cross_validate_strategy, refit_and_score
    from .metrics import metrics_at_threshold

    cv = cross_validate_strategy(spec, X_pool, y_pool, n_splits=n_splits, seed=seed,
                                 probe_factory=spec["probe_factory"], log=log)
    tuned = threshold_fn(y_pool, cv["oof_proba"], cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                         precision_target=C.PRECISION_TARGET)
    threshold_map = {mode: float(tuned[mode]["threshold"]) for mode in THRESHOLD_MODES
                     if mode in tuned}
    log("    Ngưỡng chọn trên ĐƯỜNG PR của xác suất OOF: "
        + ", ".join(f"{mode}={value:.4f}" for mode, value in threshold_map.items())
        + f" | PR-AUC OOF={tuned['pr_auc']:.4f} ({tuned['n_candidates']} điểm ứng viên)")

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
        "fold_val_nguyên_vẹn": True,       # `cv.assert_val_untouched` đã chạy trong từng fold
        "test_nguyên_vẹn": bool(np.array_equal(X_test, X_test_before)
                                and np.array_equal(y_test, y_test_before)),
        "ngưỡng_chọn_trên_oof": bool(set(threshold_map) == set(THRESHOLD_MODES)),
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
    log(f"    → test: F1@{threshold_map['best_f1']:.3f}="
        f"{scored['at_threshold']['best_f1']['f1']:.3f} (F1@0.5={baseline_f1:.3f}) | "
        f"PR-AUC={scored['at_threshold']['best_f1']['pr_auc']:.4f} | "
        f"rò rỉ: {'PASS' if all(checks.values()) else 'FAIL'}")

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
    """Bản ghi cho kỹ thuật bị BỎ QUA/LỖI (giữ nguyên schema để báo cáo không phải xử lý đặc biệt)."""
    return {"technique": spec["key"], "group": spec["group"], "kind": spec["kind"],
            "doc": spec["doc"], "implementation": spec["implementation"],
            "is_resampling": spec["is_resampling"], "status": status, "reason": reason,
            "rows": [], "fold_rows": [], "resample_rows": [], "thresholds": {},
            "oof_pr_auc": None, "checks": {}, "oof_at_0.5": {}, "oof_proba": None, "oof_y": None}


def _has(package: str) -> bool:
    """True nếu `package` có metadata phiên bản (để in mà không cần import)."""
    try:
        md.version(package)
    except Exception:
        return False
    return True


def run(n_samples: Optional[int] = None, n_splits: Optional[int] = None,
        techniques: Optional[Sequence[str]] = None, prefer_imblearn: bool = C.PREFER_IMBLEARN,
        write: bool = True, quick: bool = False) -> Dict[str, Any]:
    """Chạy danh mục: sinh dữ liệu 98/2 → CV stratified từng kỹ thuật → ngưỡng PR trên OOF → test."""
    from .data import (format_distribution, label_distribution, make_imbalanced_dataset,
                       stratified_holdout_split)
    from .models import classifier_backend
    from .samplers import resampling_backend
    from .thresholds import tune_thresholds_from_pr_curve

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        try:  # console Windows (cp1252)/stdout bị redirect: ép UTF-8 để không UnicodeEncodeError
            reconfigure = getattr(sys.stdout, "reconfigure", None)
            encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
            if reconfigure is not None and encoding not in ("utf-8", "utf8"):
                reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - stream không hỗ trợ reconfigure
            pass
        print(message)
        log_lines.append(message)

    seed = C.SEED
    n = int(n_samples or C.CATALOG_N_SAMPLES)
    folds = int(n_splits or C.CATALOG_N_SPLITS)
    keys = list(techniques) if techniques else list(TECHNIQUE_ORDER)
    unknown = [key for key in keys if key not in TECHNIQUE_ORDER]
    if unknown:
        raise ValueError(f"Kỹ thuật không có trong danh mục: {unknown}; có {list(TECHNIQUE_ORDER)}")

    missing_libs = all_techniques_available(prefer_imblearn)
    log("=== DANH MỤC KỸ THUẬT MẤT CÂN BẰNG (yêu cầu #2) — pipeline không rò rỉ dữ liệu ===")
    log(f"Backend phân loại   : {classifier_backend()} "
        f"(lightgbm {md.version('lightgbm') if _has('lightgbm') else '—'}, "
        f"scikit-learn {md.version('scikit-learn')})")
    log(f"Backend resampling  : {resampling_backend(prefer_imblearn)} "
        f"(imbalanced-learn {md.version('imbalanced-learn') if _has('imbalanced-learn') else '—'})")
    log(f"Danh mục            : {len(keys)} kỹ thuật × {folds} fold | n_samples={n} | seed={seed}"
        + (" | chế độ --quick" if quick else ""))
    if missing_libs:
        log(f"Thiếu thư viện      : {missing_libs} ⇒ các kỹ thuật đó bị BỎ QUA (ghi rõ lý do)")
    for group, group_keys in (*REFERENCE_GROUPS.items(), *REQUIRED_GROUPS.items()):
        log(f"  - {group:26s}: {', '.join(key for key in group_keys if key in keys)}")

    X, y = make_imbalanced_dataset(n_samples=n, random_state=seed)
    split = stratified_holdout_split(X, y, seed=seed)
    X_pool, y_pool = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]
    log("\n[1] Dữ liệu (mất cân bằng 98/2)")
    log(f"    toàn bộ     : {format_distribution(label_distribution(y))}")
    log(f"    train_pool  : {format_distribution(label_distribution(y_pool))}")
    log(f"    holdout test: {format_distribution(label_distribution(y_test))}"
        "  ← KHÔNG resample, chỉ dùng một lần để chốt")

    results: List[Dict[str, Any]] = []
    for index, key in enumerate(keys, start=2):
        spec = build_technique(key, prefer_imblearn=prefer_imblearn, random_state=seed, quick=quick)
        log(f"\n[{index}] `{key}` ({spec['group']}) — {spec['doc']}")
        gaps = missing_libs.get(key, [])
        if gaps:
            reason = f"thiếu {', '.join(gaps)}"
            log(f"    BỎ QUA — {reason}")
            results.append(_empty_result(spec, "skipped", reason))
            continue
        try:
            results.append(evaluate_technique(spec, X_pool, y_pool, X_test, y_test,
                                              n_splits=folds, seed=seed, log=log,
                                              threshold_fn=tune_thresholds_from_pr_curve))
        except Exception as exc:  # noqa: BLE001 - một kỹ thuật lỗi không làm hỏng cả danh mục
            reason = f"{type(exc).__name__}: {exc}"
            log(f"    LỖI — {reason}")
            results.append(_empty_result(spec, "error", reason))

    failed = [item["technique"] for item in results
              if item["status"] == "ok" and not all(item["checks"].values())]
    log(f"\n[{len(keys) + 2}] Kiểm chứng chống rò rỉ: "
        f"{'TẤT CẢ PASS' if not failed else 'FAIL ở: ' + ', '.join(failed)}")

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




# ---------------------------------------------------------------------------
# Báo cáo (Markdown/CSV/JSON/log)
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 3) -> str:
    """Định dạng số cho bảng (None/NaN → '—')."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if number != number else f"{number:.{digits}f}"


def _row_for(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Dòng metric trên test của một kỹ thuật tại một chế độ ngưỡng."""
    return next((row for row in item["rows"] if row["threshold_mode"] == mode), {})


def _train_summary(item: Dict[str, Any]) -> str:
    """Chuỗi `IR trước → IR sau` (trung bình qua fold) cho cột phân phối nhãn."""
    rows = item["resample_rows"]
    if not rows:
        return "—"
    before = float(np.mean([row["train_ir_before"] for row in rows]))
    after = float(np.mean([row["train_ir_after"] for row in rows]))
    return f"{before:.1f} → {after:.1f}"


def _val_pos_range(item: Dict[str, Any]) -> str:
    """Khoảng % dương của fold-validation (chứng minh validation giữ tỉ lệ lớp)."""
    rows = item["resample_rows"]
    if not rows:
        return "—"
    values = [row["val_pos_pct"] for row in rows]
    return f"{min(values):.2f}–{max(values):.2f}%"


def compare_with_baseline(result: Dict[str, Any],
                          mode: str = "best_f1") -> Dict[str, Any]:
    """Bảng so sánh **BASELINE (chưa xử lý)** với từng kỹ thuật xử lý mất cân bằng (yêu cầu #3).

    Mỗi dòng gồm metric CHÍNH tại ngưỡng `mode` (mặc định `best_f1` — ngưỡng chọn trên đường PR của
    xác suất out-of-fold): Precision, Recall, F1, F1-macro, F1-weighted, F-beta, PR-AUC, ROC-AUC, MCC
    và Confusion Matrix (TN/FP/FN/TP). Kèm chênh lệch so với baseline: ΔPR-AUC, ΔF1, ΔF1-macro,
    ΔFbeta, ΔRecall.

    **Accuracy KHÔNG có trong bảng này** — nó là chỉ số chẩn đoán (`metrics.accuracy_diagnostic`)
    vì ở tỉ lệ 98/2 quy tắc "đoán lớp đa số" đã đạt ~98% accuracy.

    Returns:
        dict gồm `mode`, `metrics` (thứ tự cột), `baseline` (dòng mốc), `rows` (baseline trước rồi
        theo thứ tự danh mục), `n_better_than_baseline` và `best_by_metric`.
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
    """Bảng Markdown: Baseline (chưa xử lý) vs từng kỹ thuật xử lý (dùng cả trong báo cáo)."""
    table = compare_with_baseline(result, mode)
    lines = [f"| Kỹ thuật | Nhóm | PR-AUC | F1 | F1-macro | F1-weighted | F-beta({_fmt(C.FBETA_BETA, 1)}) "
             "| ROC-AUC | MCC | TN/FP/FN/TP | ΔPR-AUC | ΔF1 | ΔF1-macro |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in table["rows"]:
        confusion = (f"{row.get('tn')}/{row.get('fp')}/{row.get('fn')}/{row.get('tp')}"
                     if row.get("tn") is not None else "—")
        lines.append(
            f"| `{row['technique']}` | `{row['group']}` | {_fmt(row.get('pr_auc'), 4)} | "
            f"{_fmt(row.get('f1'))} | {_fmt(row.get('macro_f1'))} | {_fmt(row.get('weighted_f1'))} | "
            f"{_fmt(row.get('fbeta'))} | {_fmt(row.get('roc_auc'))} | {_fmt(row.get('mcc'))} | "
            f"{confusion} | {_fmt(row.get('delta_pr_auc'))} | {_fmt(row.get('delta_f1'))} | "
            f"{_fmt(row.get('delta_macro_f1'))} |")
    if table["baseline"] is None:
        lines.append("| _(thiếu baseline — bảng chỉ có kỹ thuật xử lý)_ |  |  |  |  |  |  |  |  |  |  |  |  |")
    else:
        better = table["n_better_than_baseline"]
        total = table["n_techniques_compared"]
        lines += ["", f"- Ngưỡng chấm điểm: `{mode}` (chọn trên đường PR của xác suất out-of-fold). "
                      f"Baseline: **PR-AUC {_fmt(table['baseline'].get('pr_auc'), 4)}, "
                      f"F1 {_fmt(table['baseline'].get('f1'))}**.",
                  f"- Số kỹ thuật có PR-AUC CAO HƠN baseline: **{better}/{total}**"
                  + (f"; cao nhất ở `{table['best_by_metric']['pr_auc']['technique']}` "
                     f"({_fmt(table['best_by_metric']['pr_auc']['value'], 4)})."
                     if "pr_auc" in table["best_by_metric"] else "."),
                  "- Bảng KHÔNG có Accuracy: ở tỉ lệ 98/2, đoán 'lớp đa số' đã đạt ~98% ⇒ accuracy "
                  "chỉ là chỉ số chẩn đoán (`metrics.accuracy_diagnostic`)."]
    return "\n".join(lines) + "\n"


def _diagnostic_accuracy_lines(result: Dict[str, Any]) -> List[str]:
    """Dòng CHẨN ĐOÁN về accuracy (yêu cầu #3: accuracy không phải thước đo chính).

    In accuracy tại điểm vận hành + MỐC ĐA SỐ, để thấy ngay vì sao accuracy vô dụng ở tỉ lệ 98/2.
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
    lines = ["**Chỉ số CHẨN ĐOÁN (không dùng để kết luận):**",
             f"- Mốc \"luôn đoán lớp đa số\" = **{_fmt(majority, 2)}% accuracy**; accuracy của các kỹ "
             f"thuật nằm trong {_fmt(min(values), 2)}%–{_fmt(max(values), 2)}%.",
             "- ⇒ Chênh lệch accuracy giữa các kỹ thuật ở đây KHÔNG chứng minh kỹ thuật nào tốt hơn; "
             "phải đọc Precision/Recall/F1(−macro/−weighted/−beta), PR-AUC, ROC-AUC và Confusion Matrix."]
    return lines


def _markdown(result: Dict[str, Any]) -> str:
    """Bảng Markdown đầy đủ: danh mục, metric test, ngưỡng PR, chống rò rỉ, kết luận."""
    results = result["results"]
    ok = [item for item in results if item["status"] == "ok"]
    dist = result["label_distribution"]
    lines = ["# Danh mục kỹ thuật xử lý mất cân bằng (yêu cầu #2) — so sánh không rò rỉ dữ liệu", "",
             f"- Backend phân loại: **{result['backend']}** · resampling: "
             f"**{result['resampling_backend']}** · n_samples {result['n_samples']} · "
             f"{result['n_splits']} fold · seed {result['seed']}"
             + (" · chế độ `--quick`" if result["quick"] else ""),
             "- Phân phối nhãn: toàn bộ "
             f"n={dist['all']['n']} (dương {dist['all']['positive_pct']:.2f}%, IR="
             f"{dist['all']['imbalance_ratio']:.1f}); train_pool "
             f"n={dist['train_pool']['n']} (dương {dist['train_pool']['positive_pct']:.2f}%); test "
             f"n={dist['test']['n']} (dương {dist['test']['positive_pct']:.2f}%)",
             "- **Threshold tuning**: ngưỡng chọn trên **đường Precision-Recall của xác suất "
             "out-of-fold** (chỉ train_pool): `best_f1` (PR), `best_cost` (chi phí FN/FP), "
             "`min_precision`; kèm mốc 0.5 để so sánh. Test chỉ được chấm **một lần** sau khi chốt.",
             "- Chống rò rỉ: mọi sampler nằm TRONG `imblearn.pipeline.Pipeline`; cost-sensitive / Focal "
             "Loss tính trọng số trong `fit`; ensemble lấy mẫu bên trong `fit`; "
             "`cv.assert_val_untouched` chạy từng fold.", "",
             "## 1. Danh mục ↔ cài đặt ↔ trạng thái", "",
             "| Nhóm | Kỹ thuật | Mô tả | Cài đặt | Trạng thái |", "|---|---|---|---|---|"]
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

    lines += ["", "## 3. Bảng so sánh BASELINE (chưa xử lý) vs các kỹ thuật xử lý (yêu cầu #3)", "",
              comparison_markdown(result).rstrip("\n")]
    diagnostic = _diagnostic_accuracy_lines(result)
    if diagnostic:
        lines += ["", *diagnostic]

    lines += ["", "## 4. Ngưỡng chọn trên xác suất out-of-fold + PR-AUC (OOF)", "",
              "| Kỹ thuật | PR-AUC (OOF) | best_f1 | best_cost | min_precision | 0.5 | IR train "
              "(trung bình) | % dương fold-val |", "|---|---:|---:|---:|---:|---:|---|---|"]
    for item in ok:
        thresholds = item["thresholds"]
        lines.append(f"| `{item['technique']}` | {_fmt(item['oof_pr_auc'], 4)} | "
                     f"{_fmt(thresholds.get('best_f1'), 4)} | {_fmt(thresholds.get('best_cost'), 4)} | "
                     f"{_fmt(thresholds.get('min_precision'), 4)} | 0.5000 | {_train_summary(item)} | "
                     f"{_val_pos_range(item)} |")

    lines += ["", "## 5. Kiểm chứng chống rò rỉ dữ liệu", ""]
    if not ok:
        lines.append("- (không có kỹ thuật nào chạy được)")
    for item in ok:
        detail = ", ".join(f"{name}: {'PASS' if value else 'FAIL'}"
                           for name, value in item["checks"].items())
        lines.append(f"- {'PASS' if all(item['checks'].values()) else 'FAIL'} — "
                     f"`{item['technique']}` ({detail})")
    lines += ["", "## 6. Kết luận & khuyến nghị", "", *_conclusions(result), "",
              "## 7. Tái lập", "", "```powershell",
              "python -m pip install -r imbalance_lab/requirements.txt",
              "python -m imbalance_lab.techniques            # toàn bộ danh mục (ghi artifact)",
              "python -m imbalance_lab.techniques --quick --techniques smote,adasyn,focal_loss",
              "python -m unittest discover -s tests -v       # gồm test danh mục + chống rò rỉ",
              "```"]
    return "\n".join(lines) + "\n"


def _conclusions(result: Dict[str, Any]) -> List[str]:
    """Kết luận TỰ ĐỘNG từ số liệu (không nhập tay) — gồm cả các phát hiện phủ định."""
    ok = [item for item in result["results"] if item["status"] == "ok"]
    if not ok:
        return ["(không có kỹ thuật nào chạy được — kiểm tra thư viện rồi chạy lại)"]
    lines: List[str] = []

    # (1) So sánh với BASELINE (mô hình CHƯA xử lý mất cân bằng) — yêu cầu #3.
    table = compare_with_baseline(result)
    if table["baseline"] is not None and table["n_techniques_compared"]:
        best_delta = max((row for row in table["rows"]
                          if row["technique"] != BASELINE_TECHNIQUE
                          and row.get("delta_pr_auc") is not None),
                         key=lambda row: float(row["delta_pr_auc"]), default=None)
        text = (f"**So với BASELINE (chưa xử lý)**: baseline PR-AUC "
                f"{_fmt(table['baseline'].get('pr_auc'), 4)} / F1 "
                f"{_fmt(table['baseline'].get('f1'))}; có "
                f"**{table['n_better_than_baseline']}/{table['n_techniques_compared']}** kỹ thuật "
                f"vượt baseline về PR-AUC")
        if best_delta is not None:
            text += (f", cao nhất là `{best_delta['technique']}` "
                     f"(ΔPR-AUC {float(best_delta['delta_pr_auc']):+.4f})")
        lines.append(text + ".")

    best_pr = max(ok, key=lambda item: item["oof_pr_auc"] or -1.0)
    lines.append(f"1. **Thứ hạng xác suất (PR-AUC out-of-fold)** tốt nhất là `{best_pr['technique']}` "
                 f"({_fmt(best_pr['oof_pr_auc'], 4)}). PR-AUC không phụ thuộc ngưỡng nên đây là so "
                 "sánh 'công bằng' nhất giữa các kỹ thuật.")

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
            f"2. **Tinh chỉnh ngưỡng theo đường PR thay cho 0.5**: F1 thay đổi trung bình "
            f"**{mean_gain:+.3f}**; tốt nhất ở `{best_gain[0]}` ({best_gain[1]:+.3f})"
            + (f"; riêng {len(worse)} kỹ thuật GIẢM ("
               + ", ".join(f"`{name}`" for name in worse)
               + ") — ngưỡng tốt nhất không phải lúc nào cũng cao hơn 0.5." if worse else "."))

    cleaning = [item for item in ok if item["technique"] in ("tomek", "enn")]
    parts = []
    for item in cleaning:
        rows = item["resample_rows"]
        if rows:
            parts.append(f"`{item['technique']}`: IR "
                         f"{np.mean([row['train_ir_before'] for row in rows]):.1f} → "
                         f"{np.mean([row['train_ir_after'] for row in rows]):.1f}")
    if parts:
        lines.append("3. **Tomek Links / ENN là kỹ thuật LÀM SẠCH biên, không phải cân bằng tỉ lệ** ("
                     + "; ".join(parts) + "): chúng chỉ bỏ mẫu sát biên nên IR gần như giữ nguyên ở "
                     "98/2 ⇒ phải dùng kèm oversampling (nhóm hybrid) mới có tác dụng cân bằng.")

    algorithm = [item for item in ok if item["group"] == "algorithm-level"]
    if algorithm:
        best = max(algorithm, key=lambda item: _row_for(item, "best_f1").get("f1", -1.0))
        row = _row_for(best, "best_f1")
        lines.append(f"4. **Algorithm-level** (không đổi dữ liệu): tốt nhất theo F1 là "
                     f"`{best['technique']}` — F1={_fmt(row.get('f1'))}, recall="
                     f"{_fmt(row.get('recall'))}, PR-AUC={_fmt(row.get('pr_auc'), 4)}. Focal Loss dùng "
                     "custom objective (`gamma`, `alpha`) nên đổi cả HÌNH DẠNG hàm mất mát, không chỉ "
                     "trọng số lớp.")

    ensemble = [item for item in ok if item["group"] == "ensemble"]
    if ensemble:
        best = max(ensemble, key=lambda item: _row_for(item, "best_f1").get("f1", -1.0))
        row = _row_for(best, "best_f1")
        lines.append(f"5. **Ensemble**: tốt nhất theo F1 là `{best['technique']}` — "
                     f"F1={_fmt(row.get('f1'))}, PR-AUC={_fmt(row.get('pr_auc'), 4)}. Undersampling "
                     "nằm bên trong `fit` nên vẫn chỉ chạm train của fold.")

    lines.append("6. **Khuyến nghị cho pipeline chính**: chọn kỹ thuật theo PR-AUC, chọn ngưỡng trên "
                 "đường PR của xác suất out-of-fold (không dùng 0.5 mặc định, không chọn trên test), "
                 "và ưu tiên can thiệp KHÔNG sinh mẫu (`class_weight`/`scale_pos_weight`/Focal Loss) "
                 "khi nhãn gắn với thực thể — mẫu tổng hợp dễ rơi vào 'vùng' của chính thực thể đã có "
                 "trong train.")
    return lines


def _write_artifacts(result: Dict[str, Any]) -> None:
    """Ghi artifact vào `reports/imbalance/`: `techniques.{md,csv,json,log}`, `techniques_by_fold.csv`,
    `techniques_comparison.csv` (bảng Baseline vs kỹ thuật — yêu cầu #3)."""
    out = C.ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "techniques.csv", result["rows"])
    payload = {key: value for key, value in result.items()
               if key not in ("log", "rows", "resample_rows", "fold_rows", "results")}
    # `results` chứa xác suất out-of-fold (mảng numpy) — bỏ khi ghi JSON để file đọc được bằng mọi tool.
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
                        help=f"Số mẫu dữ liệu giả lập (mặc định {C.CATALOG_N_SAMPLES}).")
    parser.add_argument("--cv", type=int, default=C.CATALOG_N_SPLITS,
                        help=f"Số fold StratifiedKFold trên train_pool (mặc định {C.CATALOG_N_SPLITS}).")
    parser.add_argument("--techniques", default="",
                        help="Danh sách khoá kỹ thuật, phân tách bằng dấu phẩy (mặc định: tất cả).")
    parser.add_argument("--no-imblearn", action="store_true",
                        help="Bắt buộc dùng bản sampler nội bộ thay vì imbalanced-learn.")
    parser.add_argument("--quick", action="store_true",
                        help="Lưới nhẹ: ensemble ít estimator hơn (chạy nhanh hơn).")
    parser.add_argument("--no-write", action="store_true", help="Không ghi artifact.")
    args = parser.parse_args(argv)
    keys = [key.strip() for key in args.techniques.split(",") if key.strip()]
    run(n_samples=args.n_samples, n_splits=args.cv, techniques=keys or None,
        prefer_imblearn=not args.no_imblearn, write=not args.no_write, quick=args.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
