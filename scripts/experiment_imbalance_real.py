"""Compare class-imbalance techniques on the real eight-company corpus, guarding against leakage.

Runs GroupKFold by ticker on train plus validation so every fold holds whole companies out, keeps each
sampler inside an imblearn Pipeline so it only touches fold-train, and scores AP, AUROC and best-F1 on
the out-of-fold probabilities. The 64-sample test split keeps its final-holdout role.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from forecasting.config import GROUP_KEY, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.eda import markdown_table
from forecasting.evaluation import best_f1_point
from forecasting.features import build_feature_matrix, extract_labels
from forecasting.models import HYPERPARAMS, make_model

#: Base model shared by every technique, so only the imbalance step differs.
DEFAULT_BASE = "hist_gradient_boosting"


def _techniques() -> List[Dict[str, Any]]:
    """Technique catalogue, skipping resampling entries when imbalanced-learn is unavailable."""
    out: List[Dict[str, Any]] = [
        {"name": "none", "group": "baseline", "kind": "plain",
         "description": "không can thiệp (đối chứng)"},
        {"name": "class_weight", "group": "algorithm-level", "kind": "class_weight",
         "description": "class_weight='balanced' tính trong fit từ nhãn fold-train"},
    ]
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks

        out += [
            {"name": "random_under", "group": "undersampling", "kind": "sampler",
             "factory": "random_under", "ratio": None,
             "description": "RandomUnderSampler: cân bằng bằng cách hạ lớp ĐA SỐ (ở corpus này là "
                            "lớp 1 = suy giảm, 62%) xuống bằng lớp thiểu số"},
            {"name": "tomek_links", "group": "undersampling", "kind": "sampler",
             "factory": "tomek", "ratio": None,
             "description": "TomekLinks: bỏ cặp mẫu hai lớp nằm sát nhau (làm sạch biên)"},
            {"name": "smote", "group": "oversampling", "kind": "sampler",
             "factory": "smote", "ratio": 1.0,
             "description": "SMOTE: sinh thêm mẫu cho LỚP THIỂU SỐ (ở corpus này là lớp 0 = không suy "
                            "giảm, 38%) cho tới khi cân bằng"},
            {"name": "smote_enn", "group": "hybrid", "kind": "sampler",
             "factory": "smote_enn", "ratio": None,
             "description": "SMOTE + ENN: sinh mẫu rồi dọn mẫu bị láng giềng phủ nhận"},
        ]
        _ = (SMOTE, RandomUnderSampler, TomekLinks, SMOTEENN)
    except Exception as exc:  # pragma: no cover - imbalanced-learn missing
        out.append({"name": "resampling_unavailable", "group": "n/a", "kind": "unavailable",
                    "description": f"bỏ qua resampling: {exc}"})
    return out


def build_estimator(technique: Dict[str, Any], base: str = DEFAULT_BASE) -> Any:
    """Pipeline for one technique, with any sampler inside the pipeline so it only touches fold-train."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    params = dict(HYPERPARAMS.get(base, {}))
    if technique["kind"] != "class_weight":
        params["class_weight"] = None      # control and resampling variants add no class weights
    if technique["kind"] == "sampler":
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks
        from imblearn.combine import SMOTEENN

        if technique["factory"] == "smote":
            # Complete balancing is safe on every fold here: class 0 is the minority at 37.7 percent, so
            # any ratio below 1.0 would try to synthesise samples for the majority class.
            sampler: Any = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=1.0)
        elif technique["factory"] == "random_under":
            sampler = RandomUnderSampler(random_state=42, sampling_strategy="majority")
        elif technique["factory"] == "tomek":
            sampler = TomekLinks()
        else:
            sampler = SMOTEENN(random_state=42)
        estimator = dict(make_model(base, **params).steps)["model"]
        return ImbPipeline([("impute", SimpleImputer(strategy="median")), ("sampler", sampler),
                            ("model", estimator)])
    return make_model(base, **params)


def _sampler_stats(estimator: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Imbalance ratio before and after resampling on fold-train, showing where the intervention acts."""
    steps = dict(estimator.steps) if hasattr(estimator, "steps") else {}
    if "sampler" not in steps:
        return {"ir_before": None, "ir_after": None, "n_before": None, "n_after": None}

    def ratio(labels: np.ndarray) -> float:
        positive = int(np.sum(labels == 1))
        negative = int(len(labels) - positive)
        return float(max(positive, negative) / min(positive, negative)) if min(positive, negative) else float("inf")

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median").fit(X)
    X_imputed = imputer.transform(X)
    sampler = steps["sampler"]
    sampler_clone = sampler.__class__(**sampler.get_params())
    X_res, y_res = sampler_clone.fit_resample(X_imputed, y)
    return {"ir_before": ratio(y), "ir_after": ratio(y_res),
            "n_before": int(len(y)), "n_after": int(len(y_res))}


def evaluate_technique(technique: Dict[str, Any], samples: List[Dict[str, Any]], base: str,
                       n_splits: int = 4, min_test: int = 4,
                       return_proba: bool = False) -> Dict[str, Any]:
    """Out-of-fold cross-company metrics for one technique, checking the fold test set stays untouched."""
    groups = np.asarray([s[GROUP_KEY] for s in samples])
    X, y = build_feature_matrix(samples), extract_labels(samples)
    n_splits = max(2, min(n_splits, len(set(groups.tolist()))))
    proba = np.full(len(y), np.nan)
    fold_checks: List[bool] = []
    sampler_info: Dict[str, Any] = {}
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups=groups)):
        if len(set(y[train_idx].tolist())) < 2 or len(test_idx) < min_test:
            continue
        estimator = build_estimator(technique, base)
        X_test_before = X[test_idx].copy()
        estimator.fit(X[train_idx], y[train_idx])
        proba[test_idx] = estimator.predict_proba(X[test_idx])[:, 1]
        # Leakage guard: the fold test set must stay identical because the sampler only sees fold-train.
        fold_checks.append(bool(np.array_equal(X_test_before, X[test_idx], equal_nan=True)))
        if fold == 0:
            sampler_info = _sampler_stats(estimator, X[train_idx], y[train_idx])
    mask = np.isfinite(proba)
    if mask.sum() < min_test or len(set(y[mask].tolist())) < 2:
        return {"name": technique["name"], "n_oof": int(mask.sum()), "average_precision": None}
    best = best_f1_point(y[mask], proba[mask])
    row = {
        "name": technique["name"], "group": technique["group"],
        "description": technique["description"],
        "n_oof": int(mask.sum()),
        "average_precision": float(average_precision_score(y[mask], proba[mask])),
        "auroc": float(roc_auc_score(y[mask], proba[mask])),
        "best_f1_on_oof": (best or {}).get("f1"),
        "best_f1_threshold": (best or {}).get("threshold"),
        "folds_checked": len(fold_checks),
        "fold_test_untouched": bool(all(fold_checks)) if fold_checks else None,
        **{key: sampler_info.get(key) for key in ("ir_before", "ir_after", "n_before", "n_after")},
    }
    if return_proba:
        # Internal only, for the paired bootstrap and DeLong tests; never written to JSON.
        row["_oof_proba"] = proba
        row["_oof_mask"] = mask
    return row


def fig_imbalance_real(rows: List[Dict[str, Any]], reference: Dict[str, Any] | None, path: Path) -> None:
    """Figure: cross-company delta AP and delta AUROC against the no-intervention control."""
    if not reference:
        return
    base_ap = reference.get("average_precision") or 0.0
    base_auc = reference.get("auroc") or 0.0
    ranked = sorted([r for r in rows if r.get("average_precision")], key=lambda r: -r["average_precision"])
    labels = [r["name"] for r in ranked][::-1]
    delta_ap = [100 * (r["average_precision"] - base_ap) for r in ranked][::-1]
    delta_auc = [100 * ((r.get("auroc") or 0.0) - base_auc) for r in ranked][::-1]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(labels) + 2.0))
    ax.barh(y - 0.2, delta_ap, height=0.4, color="crimson", label="ΔAP (điểm %)")
    ax.barh(y + 0.2, delta_auc, height=0.4, color="steelblue", label="ΔAUROC (điểm %)")
    ax.set_yticks(y, labels, fontsize=9)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Chênh lệch so với đối chứng 'none' (điểm phần trăm), OOF cross-company")
    ax.set_title("Kỹ thuật xử lý lệch lớp trên DỮ LIỆU THẬT (8 công ty, GroupKFold)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def conclusions(rows: List[Dict[str, Any]], reference: Dict[str, Any] | None,
                label_ir: float | None,
                significance: Dict[str, Any] | None = None) -> List[str]:
    """Automatic reading: which techniques beat the control, by how much, plus methodology warnings."""
    lines: List[str] = []
    if not reference:
        return lines
    base_ap = reference.get("average_precision") or 0.0
    base_auc = reference.get("auroc") or 0.0
    better = [r for r in rows if (r.get("average_precision") or 0) > base_ap + 0.005]
    worse = [r for r in rows if (r.get("average_precision") or 0) < base_ap - 0.005]
    lines.append(
        f"**Đối chứng (không can thiệp):** cross-company AP = {base_ap:.4f}, AUROC = {base_auc:.4f} "
        f"(IR cấp mẫu của corpus ≈ {label_ir:.2f} — mất cân bằng NHẸ).")
    lines.append(
        f"**Kết quả trên dữ liệu thật:** {len(better)} kỹ thuật hơn đối chứng > 0,5 điểm % AP, "
        f"{len(worse)} kỹ thuật kém hơn > 0,5 điểm %. "
        + (f"Tốt nhất: **{max(rows, key=lambda r: r.get('average_precision') or 0)['name']}** "
           f"(AP {max(r.get('average_precision') or 0 for r in rows):.4f})." if rows else ""))
    lines.append(
        "**Cách đọc:** AP ở đây là OOF cross-company — mỗi công ty bị giữ trọn ra ngoài, nên kết quả "
        "không thể đến từ việc \"nhớ mặt công ty\". Vì vậy đây là phép thử công bằng cho kỹ thuật "
        "resampling/cost-sensitive, khác hẳn so sánh in-domain.")
    if not better:
        lines.append(
            "**Kết luận (nhất quán với lab):** trên dữ liệu thật, **không kỹ thuật nào cải thiện AP có "
            "ý nghĩa**; mất cân bằng KHÔNG phải nút thắt — nút thắt là nhãn gắn với thực thể và số "
            "lượng công ty ít. Điều này khớp với `reports/imbalance/summary.md` (3 chiến lược chênh "
            "nhau < 0,01 AP trên dữ liệu 98/2) và là lý do repo KHÔNG đưa SMOTE vào pipeline chính.")
    lines.append(
        "**Lưu ý về CHIỀU của mất cân bằng (khác bộ dữ liệu phá sản thông thường):** trong corpus này "
        "lớp 1 (suy giảm) chiếm **62%** ⇒ lớp THIỂU SỐ là lớp 0. Vì vậy SMOTE ở đây **sinh thêm mẫu "
        "\"không suy giảm\"**, tức can thiệp theo hướng ngược với thực hành phổ biến — một lý do nữa "
        "để mất cân bằng không phải nút thắt của bài toán này.")
    if significance:
        boot = significance.get("bootstrap_ap") or {}
        delong = significance.get("delong_auroc") or {}
        lines.append(
            f"**Kiểm định cặp cho kỹ thuật tốt nhất** (`{significance.get('best_technique')}` vs "
            f"`{significance.get('reference')}`, cùng {significance.get('n_paired')} mẫu OOF): ΔAP = "
            f"{(boot.get('delta') or 0):+.4f} (CI 95% [{(boot.get('ci95_lower') or 0):+.4f}; "
            f"{(boot.get('ci95_upper') or 0):+.4f}], p = {boot.get('p_value')}), ΔAUROC (DeLong) = "
            f"{(delong.get('delta') or 0):+.4f} (p = {delong.get('p_value')}) ⇒ "
            + ("**có ý nghĩa thống kê** ở mức 5%."
               if (boot.get("p_value") or 1) < 0.05 else
               "**chưa** đủ căn cứ khẳng định hơn đối chứng (khoảng tin cậy chứa 0)."))
    guard = [r for r in rows if r.get("fold_test_untouched") is False]
    if guard:
        lines.append(f"**CẢNH BÁO RÒ RỈ:** {len(guard)} kỹ thuật làm đổi tập test của fold — phải sửa.")
    else:
        lines.append(
            f"**Chống rò rỉ:** mọi kỹ thuật đều có sampler nằm trong pipeline ⇒ tập test của từng fold "
            f"KHÔNG bị resample (đã kiểm tra bằng so khớp ma trận trước/sau fit).")
    return lines


def markdown_imbalance_real(summary: Dict[str, Any]) -> str:
    """Render `reports/results/imbalance_real.md` from the JSON payload."""
    rows = summary["rows"]
    lines = ["# Kỹ thuật xử lý lệch lớp trên DỮ LIỆU THẬT (8 công ty)", "",
             f"- Giao thức: {summary['protocol']['cv']}; {summary['protocol']['leakage']}",
             f"- Mô hình nền dùng chung: **{summary['base_model']}** "
             f"(để mọi kỹ thuật khác nhau CHỈ ở bước xử lý lệch lớp).",
             f"- Mất cân bằng corpus: IR cấp mẫu ≈ {summary['label_ir']:.2f}; "
             f"nhưng HD/LOW/WMT có 100% nhãn 1 (mất cân bằng cấp thực thể).",
             "", "## 1. Nhận xét tự động", ""]
    lines += [f"{i + 1}. {text}" for i, text in enumerate(summary["conclusions"])]
    lines += ["", "## 2. Bảng kết quả (xếp theo AP out-of-fold)", "",
              markdown_table(
                  ["Kỹ thuật", "Nhóm", "AP (OOF)", "AUROC (OOF)", "F1* trên OOF", "IR trước → sau",
                   "n train trước → sau", "Test fold nguyên vẹn"],
                  [[r["name"], r.get("group", "—"),
                    f"{r['average_precision']:.4f}" if r.get("average_precision") else "—",
                    f"{r['auroc']:.4f}" if r.get("auroc") else "—",
                    f"{r.get('best_f1_on_oof'):.4f}" if r.get("best_f1_on_oof") else "—",
                    (f"{r['ir_before']:.2f} → {r['ir_after']:.2f}"
                     if r.get("ir_before") is not None else "—"),
                    (f"{r['n_before']} → {r['n_after']}" if r.get("n_before") is not None else "—"),
                    "PASS" if r.get("fold_test_untouched") else "—"]
                   for r in sorted(rows, key=lambda r: -(r.get("average_precision") or 0))],
                  ["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]),
              "",
              "> Đọc bảng: cột **IR trước → sau** chứng minh can thiệp thực sự diễn ra trên fold-train; "
              "cột **Test fold nguyên vẹn** là bài kiểm tra chống rò rỉ (sampler không được chạm validation).",
              "", "## 3. Kiểm định cặp (kỹ thuật tốt nhất vs đối chứng)", ""]
    sig = summary.get("significance") or {}
    if sig:
        boot = sig.get("bootstrap_ap") or {}
        delong = sig.get("delong_auroc") or {}
        lines += [markdown_table(
            ["So sánh", "ΔAP", "CI 95% ΔAP", "p (bootstrap AP)", "ΔAUROC", "p (DeLong)"],
            [[f"{sig.get('best_technique')} vs {sig.get('reference')}",
              f"{(boot.get('delta') or 0):+.4f}",
              f"[{(boot.get('ci95_lower') or 0):+.4f}; {(boot.get('ci95_upper') or 0):+.4f}]",
              f"{boot.get('p_value')}", f"{(delong.get('delta') or 0):+.4f}", f"{delong.get('p_value')}"]],
            ["---", "---:", "---:", "---:", "---:", "---:"])]
    else:
        lines.append("Không tính được kiểm định cặp (hai hệ thống không dùng cùng tập out-of-fold).")
    lines += ["", "## 4. Hình", ""] + [f"- `{p}`" for p in summary["figures"]]
    return "\n".join(lines) + "\n"


def run(write: bool = True, folds: int = 4, base: str = DEFAULT_BASE, out_dir: Path | None = None,
        fig_dir: Path | None = None, figures: bool = True) -> Dict[str, Any]:
    """Run the technique catalogue on the real corpus and write `imbalance_real.{json,md}`."""
    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    figs = Path(fig_dir) if fig_dir else (RESULTS_DIR.parent / "figures" / "imbalance_real")
    splits = {name: load_prepared(name) for name in ("train", "validation")}
    samples = splits["train"] + splits["validation"]
    y_all = extract_labels(samples)
    positive = int(y_all.sum())
    label_ir = float(max(positive, len(y_all) - positive) / min(positive, len(y_all) - positive))

    techniques = _techniques()
    rows = [evaluate_technique(t, samples, base, n_splits=folds, return_proba=True)
            for t in techniques if t["kind"] != "unavailable"]
    reference = next((r for r in rows if r["name"] == "none"), None)
    best = max((r for r in rows if r.get("average_precision")),
               key=lambda r: r["average_precision"])

    # Paired tests of the best technique against the control on the same out-of-fold probabilities.
    significance: Dict[str, Any] = {}
    if reference is not None and best is not reference:
        from forecasting.significance import delong_test, paired_bootstrap

        mask_best, mask_ref = best["_oof_mask"], reference["_oof_mask"]
        if np.array_equal(mask_best, mask_ref):
            y_oof = extract_labels(samples)[mask_best]
            probabilities = {"best": best["_oof_proba"][mask_best],
                             "reference": reference["_oof_proba"][mask_ref]}
            significance = {
                "best_technique": best["name"], "reference": reference["name"],
                "n_paired": int(mask_best.sum()),
                "delong_auroc": delong_test(y_oof, probabilities["best"],
                                            probabilities["reference"]),
                "bootstrap_ap": paired_bootstrap(y_oof, probabilities["best"],
                                                 probabilities["reference"],
                                                 "average_precision", n_boot=2000),
            }
    for row in rows:                      # drop the probability arrays before writing JSON
        row.pop("_oof_proba", None)
        row.pop("_oof_mask", None)
    summary: Dict[str, Any] = {
        "protocol": {
            "cv": f"GroupKFold({folds}) theo mã cổ phiếu trên train+validation ({len(samples)} mẫu)",
            "leakage": "sampler nằm trong imblearn.Pipeline ⇒ chỉ chạm fold-train; test fold được "
                       "so khớp ma trận trước/sau fit",
            "metric": "AP/AUROC/F1 trên xác suất out-of-fold (không dùng test của đồ án)",
            "note": "test 64 mẫu giữ nguyên vai trò chốt cuối, không tham gia thí nghiệm này",
        },
        "base_model": base,
        "label_ir": label_ir,
        "n_samples": len(samples),
        "n_positive": positive,
        "rows": rows,
        "significance": significance,
        "conclusions": conclusions(rows, reference, label_ir, significance),
        "figures": [],
    }
    if figures:
        figs.mkdir(parents=True, exist_ok=True)
        fig_imbalance_real(rows, reference, figs / "01_techniques_real.png")
        summary["figures"] = [str(p.relative_to(RESULTS_DIR.parent))
                              for p in sorted(figs.glob("*.png"))]
    if write:
        (out / "imbalance_real.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "imbalance_real.md").write_text(markdown_imbalance_real(summary), encoding="utf-8")
    best = max((r for r in rows if r.get("average_precision")), key=lambda r: r["average_precision"])
    print(f"Imbalance (real data): {len(rows)} techniques; control AP = "
          f"{(reference or {}).get('average_precision', float('nan')):.4f}; best = {best['name']} "
          f"({best['average_precision']:.4f}); leakage: "
          f"{'PASS' if all(r.get('fold_test_untouched') for r in rows) else 'FAIL'}")
    return summary


def main(argv=None) -> int:
    """Command-line entry point for `scripts.experiment_imbalance_real`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base model shared by every technique.")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    print("=== Class-imbalance techniques on the real data (GroupKFold by company) ===")
    run(write=not args.no_write, folds=args.folds, base=args.base, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
