"""Thí nghiệm tiền xử lý trên DỮ LIỆU THẬT: winsorize × scaler × họ mô hình.

Lệnh: python -m scripts.experiment_preprocessing [--quick] [--no-write] [--no-figures]

Vì sao cần: `reports/results/eda.md` đo 6/14 tỷ số có |skew| > 1 và `debt_to_equity` có **30,6%**
giá trị ngoài khoảng IQR, nhưng pipeline chính vẫn chỉ dùng `StandardScaler` (mean/std — chính là
đại lượng bị outlier chi phối). Thí nghiệm này trả lời bằng số:

1. **Winsorize (clip theo IQR hoặc P1–P99, ngưỡng học từ train)** có cải thiện AP/AUROC không?
2. **Scaler chịu đuôi nặng** (`RobustScaler`, `PowerTransformer`) tốt hơn `StandardScaler` bao nhiêu?
3. Với mô hình cây (bất biến với scale) thì can thiệp nào còn tác dụng?

Hai giao thức đánh giá (không dùng test để chọn cấu hình):
- **In-domain**: fit train → đo validation (AP/AUROC/F1 tốt nhất).
- **Cross-company**: `GroupKFold(4)` trên train+validation, gộp xác suất out-of-fold (AP/AUROC) —
  câu hỏi thật của đồ án vì nhãn gần như là thuộc tính công ty.

Ghi ra `reports/results/preprocessing_experiment.{json,md}` + 1 hình.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from forecasting.config import GROUP_KEY, RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.eda import markdown_table
from forecasting.evaluation import best_f1_point, evaluate_proba
from forecasting.features import build_feature_matrix, extract_labels
from forecasting.models import MODEL_REGISTRY, make_model

#: Cấu hình tham chiếu = pipeline chính hiện tại (logistic + StandardScaler, không winsorize).
REFERENCE = {"model": "logistic", "scaler": "standard", "winsorize": "none"}

#: Kích thước lưới cho mô hình cây / mô hình tuyến tính.
TREE_SCALERS: Tuple[str, ...] = ("none", "robust")
LINEAR_SCALERS: Tuple[str, ...] = ("none", "standard", "robust", "power")
WINSORIZERS: Tuple[str, ...] = ("none", "iqr", "p1p99")


def _variants(models: Sequence[str], quick: bool = False) -> List[Dict[str, str]]:
    """Sinh danh sách cấu hình (model × scaler × winsorize) — tuyến tính thử nhiều scaler hơn."""
    winsorizers = ("none", "iqr") if quick else WINSORIZERS
    out: List[Dict[str, str]] = []
    for name in models:
        scalers = ("standard", "robust") if name == "logistic" else TREE_SCALERS
        if quick and name != "logistic":
            scalers = ("none",)
        for scaler in scalers:
            for winsorize in winsorizers:
                out.append({"model": name, "scaler": scaler, "winsorize": winsorize})
    return out


def _cross_company_oof(model_name: str, scaler: str, winsorize: str,
                       samples: List[Dict[str, Any]], n_splits: int = 4,
                       min_test: int = 4) -> Dict[str, Any]:
    """AP/AUROC out-of-fold khi giữ TRỌN công ty ra khỏi fold-train (chống rò rỉ thực thể)."""
    groups = np.asarray([s[GROUP_KEY] for s in samples])
    X, y = build_feature_matrix(samples), extract_labels(samples)
    n_splits = max(2, min(n_splits, len(set(groups.tolist()))))
    proba = np.full(len(y), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
        if len(set(y[train_idx].tolist())) < 2 or len(test_idx) < min_test:
            continue
        model = make_model(model_name, scaler=scaler, winsorize=winsorize)
        model.fit(X[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    mask = np.isfinite(proba)
    if mask.sum() < min_test or len(set(y[mask].tolist())) < 2:
        return {"n_oof": int(mask.sum()), "oof_average_precision": None, "oof_auroc": None}
    return {"n_oof": int(mask.sum()),
            "oof_average_precision": float(average_precision_score(y[mask], proba[mask])),
            "oof_auroc": float(roc_auc_score(y[mask], proba[mask]))}


def evaluate_variant(variant: Dict[str, str], splits: Dict[str, List[Dict[str, Any]]]
                     ) -> Dict[str, Any]:
    """Fit một cấu hình trên train, đo validation + cross-company OOF (mọi thứ fit trên train)."""
    train, validation = splits["train"], splits["validation"]
    X_tr, y_tr = build_feature_matrix(train), extract_labels(train)
    X_va, y_va = build_feature_matrix(validation), extract_labels(validation)
    model = make_model(variant["model"], scaler=variant["scaler"], winsorize=variant["winsorize"])
    model.fit(X_tr, y_tr)
    proba_va = model.predict_proba(X_va)[:, 1]
    val = evaluate_proba(y_va, proba_va)
    best = val.get("best_f1") or {}
    row: Dict[str, Any] = {
        "model": variant["model"], "scaler": variant["scaler"], "winsorize": variant["winsorize"],
        "val_average_precision": val.get("average_precision"),
        "val_auroc": val.get("auroc"),
        "val_best_f1": best.get("f1"), "val_best_f1_threshold": best.get("threshold"),
        "val_brier": val.get("brier"),
        "n_clipped_train_pct": None, "n_clipped_val_pct": None,
    }
    winsorize_step = dict(model.steps).get("winsorize")
    if winsorize_step is not None:
        row["n_clipped_train_pct"] = 100.0 * winsorize_step.clip_share(X_tr)
        row["n_clipped_val_pct"] = 100.0 * winsorize_step.clip_share(X_va)
    row.update(_cross_company_oof(variant["model"], variant["scaler"], variant["winsorize"],
                                  train + validation))
    return row


def _reference_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    for row in rows:
        if all(row[key] == REFERENCE[key] for key in REFERENCE):
            return row
    return None


def fig_preprocessing(rows: Sequence[Dict[str, Any]], reference: Dict[str, Any] | None, path: Path,
                      top: int = 12) -> None:
    """Hình — ΔAP (validation và cross-company) so với cấu hình pipeline chính."""
    if not reference:
        return
    base_val = reference.get("val_average_precision") or 0.0
    base_oof = reference.get("oof_average_precision") or 0.0
    picked = sorted(rows, key=lambda r: -((r.get("oof_average_precision") or 0.0)))[:top][::-1]
    labels = [f"{r['model']} | {r['scaler']} | {r['winsorize']}" for r in picked]
    delta_val = [100 * ((r.get("val_average_precision") or 0.0) - base_val) for r in picked]
    delta_oof = [100 * ((r.get("oof_average_precision") or 0.0) - base_oof) for r in picked]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(labels) + 2.2))
    ax.barh(y - 0.2, delta_val, height=0.4, color="steelblue", label="Δ Val AP (điểm %)")
    ax.barh(y + 0.2, delta_oof, height=0.4, color="crimson", label="Δ Cross-company AP (điểm %)")
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Chênh lệch AP so với cấu hình pipeline chính (điểm phần trăm)")
    ax.set_title("Winsorize × scaler: cấu hình nào cải thiện AP thật sự?", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chosen_model() -> str | None:
    """Tên mô hình đã được `forecasting.train` chốt (đọc `reports/results/summary.json`)."""
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("best_model")
    except (OSError, json.JSONDecodeError):  # pragma: no cover - file hỏng
        return None


def conclusions(rows: Sequence[Dict[str, Any]], reference: Dict[str, Any] | None) -> List[str]:
    """Nhận xét tự động: can thiệp nào giúp, bao nhiêu, và cảnh báo về diễn giải."""
    lines: List[str] = []
    if not reference:
        return lines
    base_val = reference.get("val_average_precision") or 0.0
    base_oof = reference.get("oof_average_precision") or 0.0
    best_val = max(rows, key=lambda r: r.get("val_average_precision") or 0.0)
    best_oof = max(rows, key=lambda r: r.get("oof_average_precision") or 0.0)
    lines.append(
        f"**Cấu hình pipeline chính** ({REFERENCE['model']} + {REFERENCE['scaler']} + không winsorize): "
        f"Val AP = {base_val:.4f}, cross-company AP = {base_oof:.4f}.")
    lines.append(
        f"**Tốt nhất theo validation:** {best_val['model']} + {best_val['scaler']} + "
        f"winsorize={best_val['winsorize']} → Val AP = {(best_val.get('val_average_precision') or 0):.4f} "
        f"({100 * ((best_val.get('val_average_precision') or 0) - base_val):+.2f} điểm %), "
        f"cross-company AP = {(best_val.get('oof_average_precision') or 0):.4f}.")
    lines.append(
        f"**Tốt nhất theo cross-company:** {best_oof['model']} + {best_oof['scaler']} + "
        f"winsorize={best_oof['winsorize']} → cross-company AP = "
        f"{(best_oof.get('oof_average_precision') or 0):.4f} "
        f"({100 * ((best_oof.get('oof_average_precision') or 0) - base_oof):+.2f} điểm %).")
    pairs = {(r["model"], r["scaler"], r["winsorize"]): r for r in rows}
    deltas = []
    for (model, scaler, winsorize), row in pairs.items():
        if winsorize == "none":
            continue
        base = pairs.get((model, scaler, "none"))
        if base and row.get("oof_average_precision") and base.get("oof_average_precision"):
            deltas.append(100 * (row["oof_average_precision"] - base["oof_average_precision"]))
    if deltas:
        lines.append(
            f"**Riêng tác động của winsorize** (cùng model/scaler, so với không clip): trung bình "
            f"{np.mean(deltas):+.2f} điểm % cross-company AP, tốt nhất {max(deltas):+.2f}, xấu nhất "
            f"{min(deltas):+.2f} trên {len(deltas)} cặp so sánh ⇒ "
            + ("cải thiện rõ về trung bình, ĐẶC BIỆT cho mô hình tuyến tính; xem mục dưới để biết "
               "tác động trên đúng mô hình được chốt."
               if np.mean(deltas) > 0.5 else
               "thay đổi nhỏ ⇒ chưa đủ căn cứ đổi mặc định, nhưng vẫn nên giữ biến thể này trong "
               "ablation vì nó giảm phương sai của hệ số tuyến tính."))
    # Kết luận cho ĐÚNG mô hình được chốt: tránh việc báo cáo khuyến nghị "đưa winsorize vào
    # pipeline chính" trong khi pipeline chính (theo `forecasting/models.py`) vẫn không winsorize.
    chosen = _chosen_model()
    if chosen:
        variants = {r["winsorize"]: r for r in rows if r["model"] == chosen and r["scaler"] == "none"}
        base_row = variants.get("none")
        candidates = [v for k, v in variants.items() if k != "none"]
        if base_row and base_row.get("oof_average_precision") and candidates:
            best_variant = max(candidates, key=lambda r: r.get("oof_average_precision") or 0.0)
            delta = 100 * ((best_variant.get("oof_average_precision") or 0.0)
                           - base_row["oof_average_precision"])
            lines.append(
                f"**Trên ĐÚNG mô hình được chốt (`{chosen}`, đọc từ `summary.json`)**: "
                f"winsorize={best_variant['winsorize']} cho cross-company AP "
                f"{(best_variant.get('oof_average_precision') or 0):.4f} so với "
                f"{base_row['oof_average_precision']:.4f} khi không clip ⇒ {delta:+.2f} điểm % "
                + ("— đủ lớn để đổi mặc định." if delta > 0.5 else
                   "— nằm trong khoảng nhiễu của 244 mẫu out-of-fold, nên **pipeline chính giữ "
                   "không winsorize** (đơn giản, dễ diễn giải) và winsorize chỉ được dùng như một "
                   "biến thể ablation; muốn đổi mặc định cần thêm công ty/dữ liệu."))
    linear = [r for r in rows if r["model"] == "logistic" and r["winsorize"] == "none"]
    if len(linear) > 1:
        best_scaler = max(linear, key=lambda r: r.get("oof_average_precision") or 0.0)
        lines.append(
            f"**Scaler cho mô hình tuyến tính** (không winsorize): tốt nhất là "
            f"`{best_scaler['scaler']}` với cross-company AP = "
            f"{(best_scaler.get('oof_average_precision') or 0):.4f} (StandardScaler = {base_oof:.4f}) ⇒ "
            + ("đổi scaler mặc định sang cấu hình này có căn cứ."
               if (best_scaler.get("oof_average_precision") or 0) - base_oof > 0.005 else
               "khác biệt không đáng kể (dưới 0,5 điểm %), nên giữ StandardScaler cho gọn và ghi lại "
               "kết quả âm này như một kết luận trung thực."))
    return lines


def run(write: bool = True, quick: bool = False, out_dir: Path | None = None,
        fig_dir: Path | None = None, figures: bool = True) -> Dict[str, Any]:
    """Chạy thí nghiệm tiền xử lý, ghi `reports/results/preprocessing_experiment.{json,md}`."""
    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    figs = Path(fig_dir) if fig_dir else (RESULTS_DIR.parent / "figures" / "preprocessing")
    splits = {name: load_prepared(name) for name in ("train", "validation", "test")}
    models = [m for m in ("logistic", "random_forest", "hist_gradient_boosting")
              if m in MODEL_REGISTRY]
    variants = _variants(models, quick=quick)
    rows = [evaluate_variant(variant, splits) for variant in variants]
    reference = _reference_row(rows)

    summary: Dict[str, Any] = {
        "protocol": {
            "fit": "chỉ trên train (212 mẫu)",
            "in_domain": "validation (32 mẫu), ngưỡng chọn theo best-F1 trên chính validation",
            "cross_company": "GroupKFold(4) trên train+validation, gộp xác suất out-of-fold",
            "test": "KHÔNG dùng để chọn cấu hình (giữ vai trò chốt cuối)",
            "reference": REFERENCE,
            "n_variants": len(rows),
        },
        "models": models,
        "rows": rows,
        "conclusions": conclusions(rows, reference),
        "figures": [],
    }
    if figures:
        figs.mkdir(parents=True, exist_ok=True)
        fig_preprocessing(rows, reference, figs / "01_winsorize_scaler.png")
        summary["figures"] = [str(p.relative_to(RESULTS_DIR.parent))
                              for p in sorted(figs.glob("*.png"))]
    if write:
        (out / "preprocessing_experiment.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "preprocessing_experiment.md").write_text(markdown_preprocessing(summary),
                                                         encoding="utf-8")
    best = max(rows, key=lambda r: r.get("oof_average_precision") or 0.0)
    base = (reference or {}).get("oof_average_precision")
    print(f"Tiền xử lý: {len(rows)} cấu hình; tốt nhất cross-company = {best['model']}/"
          f"{best['scaler']}/{best['winsorize']} (AP {best.get('oof_average_precision'):.4f}); "
          f"tham chiếu AP = {base:.4f}" if base else "tham chiếu: không có")
    return summary


def markdown_preprocessing(summary: Dict[str, Any]) -> str:
    """Sinh `reports/results/preprocessing_experiment.md` (mọi số đọc từ JSON)."""
    rows = summary["rows"]
    protocol = summary["protocol"]
    ranked = sorted(rows, key=lambda r: -((r.get("oof_average_precision") or 0.0)))
    lines = ["# Thí nghiệm tiền xử lý trên dữ liệu thật (winsorize × scaler)", "",
             f"- Giao thức: fit **{protocol['fit']}**; đo in-domain trên {protocol['in_domain']}; "
             f"tổng quát hoá bằng {protocol['cross_company']}; {protocol['test']}.",
             f"- Số cấu hình đã chạy: **{protocol['n_variants']}** "
             f"({len({r['model'] for r in rows})} họ mô hình × scaler × winsorize).",
             f"- Tham chiếu: `{REFERENCE['model']}` + `{REFERENCE['scaler']}` + không winsorize "
             f"(= pipeline chính hiện tại).",
             "",
             "## 1. Nhận xét tự động", ""]
    lines += [f"{i + 1}. {text}" for i, text in enumerate(summary["conclusions"])]
    lines += ["", "## 2. Bảng đầy đủ (xếp theo cross-company AP)", "",
              markdown_table(
                  ["Mô hình", "Scaler", "Winsorize", "Val AP", "Val AUROC", "Val F1*",
                   "Cross-co. AP", "Cross-co. AUROC", "% clip trên val", "Brier"],
                  [[r["model"], r["scaler"], r["winsorize"],
                    f"{r.get('val_average_precision'):.4f}" if r.get("val_average_precision") else "—",
                    f"{r.get('val_auroc'):.4f}" if r.get("val_auroc") else "—",
                    f"{r.get('val_best_f1'):.4f}" if r.get("val_best_f1") else "—",
                    f"{r.get('oof_average_precision'):.4f}" if r.get("oof_average_precision") else "—",
                    f"{r.get('oof_auroc'):.4f}" if r.get("oof_auroc") else "—",
                    f"{r['n_clipped_val_pct']:.2f}" if r.get("n_clipped_val_pct") is not None else "—",
                    f"{r.get('val_brier'):.4f}" if r.get("val_brier") else "—"]
                   for r in ranked],
                  ["---", "---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "> Cột **% clip trên val** = tỉ lệ giá trị thực sự bị cắt khi áp ngưỡng học từ train "
              "(đo mức can thiệp, không phải mức cải thiện). Ngưỡng winsorize học **chỉ trên train** "
              "nên không rò rỉ; test vẫn không được dùng ở bước này.",
              "",
              "## 3. Hình", ""] + [f"- `{p}`" for p in summary["figures"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """CLI: `python -m scripts.experiment_preprocessing [--quick] [--no-write] [--no-figures]`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Lưới rút gọn (nhanh hơn).")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    print("=== Thí nghiệm tiền xử lý: winsorize × scaler (dữ liệu thật) ===")
    run(write=not args.no_write, quick=args.quick, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
