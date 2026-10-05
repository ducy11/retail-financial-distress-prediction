"""Explain the frozen model with KernelSHAP, complementing global permutation importance.

Computes per-sample Shapley values with the self-implemented KernelSHAP, records the efficiency
self-check and the rank agreement with permutation importance, and explains misclassified samples.
Writes `reports/results/shap.{json,md}` plus three figures under `reports/figures/shap/`.
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

from forecasting.config import FIGURES_DIR, RANDOM_SEED, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.eda import markdown_table
from forecasting.explain import (DEFAULT_N_BACKGROUND, DEFAULT_N_COALITIONS, kernel_shap_matrix,
                                 mean_abs_shap, permutation_importance_ranking, pipeline_predict_fn,
                                 rank_agreement)
from forecasting.features import build_feature_matrix, extract_labels, feature_names

#: Directory holding the SHAP figures.
SHAP_DIR = FIGURES_DIR / "shap"
#: Number of features shown in the global table and figure.
TOP_FEATURES = 15


def _short(name: str) -> str:
    return str(name).replace("_latest", "").replace("_window", "")


def fig_shap_summary(importance: np.ndarray, permutation: np.ndarray | None, names: List[str],
                     path: Path) -> None:
    """First figure: mean |phi| bars with permutation deltas overlaid to compare both methods."""
    order = np.argsort(-np.asarray(importance))[:TOP_FEATURES][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh([_short(names[i]) for i in order], [importance[i] for i in order],
            color="mediumpurple", alpha=0.9, label="SHAP: mean |φ|")
    if permutation is not None:
        scaled = np.asarray(permutation, dtype=float)
        share = scaled.max() if scaled.max() > 0 else 1.0
        for row, i in enumerate(order):
            ax.plot([importance[i], share * scaled[i] / share], [row, row], color="crimson",
                    lw=1.2, alpha=0.6)
            ax.plot(share * scaled[i] / share, row, "o", color="crimson", ms=4)
    ax.set_xlabel("mean |φ| (SHAP) — chấm đỏ: ΔAUROC khi hoán vị (permutation)")
    ax.set_title("Độ quan trọng feature: SHAP vs permutation importance", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_shap_beeswarm(phi: np.ndarray, X: np.ndarray, y: np.ndarray, names: List[str],
                      path: Path, top: int = 6) -> None:
    """Second figure: for each feature, the normalised value on x, the SHAP value on y, coloured by label."""
    order = np.argsort(-mean_abs_shap(phi))[:top]
    cols = 3
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows))
    for ax, j in zip(np.atleast_1d(axes).ravel(), order):
        values = X[:, j]
        finite = np.isfinite(values)
        if finite.sum() < 3 or float(np.nanstd(values)) == 0:
            ax.axis("off")
            continue
        lo, hi = np.nanpercentile(values[finite], [1, 99])
        scaled = np.clip((values - lo) / max(1e-9, hi - lo), 0, 1)
        for label, color in ((0, "steelblue"), (1, "crimson")):
            mask = finite & (np.asarray(y) == label)
            ax.scatter(scaled[mask], phi[mask, j], s=18, alpha=0.75, color=color,
                       label=f"nhãn {label}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(_short(names[j]), fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    for ax in np.atleast_1d(axes).ravel()[len(order):]:
        ax.axis("off")
    axes_flat = np.atleast_1d(axes).ravel()
    axes_flat[0].set_ylabel("φ (đóng góp vào log-odds)")
    axes_flat[0].legend(fontsize=7)
    fig.suptitle("SHAP beeswarm: giá trị feature (0–1 theo P1–P99) vs đóng góp φ", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_shap_local(phi: np.ndarray, base_value: float, predictions: np.ndarray,
                   sample_ids: List[str], indices: List[int], names: List[str], path: Path,
                   top: int = 8) -> None:
    """Third figure: local explanations for misclassified samples, showing the largest signed terms."""
    if not indices:
        return
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8))
    for ax, index in zip(np.atleast_1d(axes), indices):
        values = phi[index]
        order = np.argsort(-np.abs(values))[:top][::-1]
        colors = ["crimson" if values[j] > 0 else "steelblue" for j in order]
        ax.barh([_short(names[j]) for j in order], [values[j] for j in order], color=colors,
                alpha=0.9)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(f"{sample_ids[index]}\nP(nhãn 1)={predictions[index]:.2f}, nền={base_value:.2f}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("SHAP cục bộ cho các mẫu dự đoán sai (đỏ: đẩy về phía suy giảm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(write: bool = True, n_coalitions: int = DEFAULT_N_COALITIONS,
        n_background: int = DEFAULT_N_BACKGROUND, max_explain: int = 64,
        out_dir: Path | None = None, fig_dir: Path | None = None,
        figures: bool = True) -> Dict[str, Any]:
    """Compute SHAP values for the model in `reports/models/best.joblib` and write the artifacts."""
    import joblib

    from forecasting.config import MODELS_DIR

    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    figs = Path(fig_dir) if fig_dir else SHAP_DIR
    names = [str(n) for n in feature_names()]

    artifact = joblib.load(MODELS_DIR / "best.joblib")
    model, model_name = artifact["model"], artifact["name"]
    splits = {name: load_prepared(name) for name in ("train", "validation", "test")}
    X_train = build_feature_matrix(splits["train"])
    X_val, y_val = build_feature_matrix(splits["validation"]), extract_labels(splits["validation"])
    X_test, y_test = build_feature_matrix(splits["test"]), extract_labels(splits["test"])

    rng = np.random.default_rng(RANDOM_SEED)
    background_idx = np.sort(rng.choice(len(X_train), size=min(n_background, len(X_train)),
                                        replace=False))
    background = X_train[background_idx]

    # Explain the whole validation split plus part of test, keeping order to trace misclassified samples.
    n_test = max(0, min(len(X_test), max_explain - len(X_val)))
    explain_samples = splits["validation"] + splits["test"][:n_test]
    X_explain = np.vstack([X_val, X_test[:n_test]]) if n_test else X_val
    y_explain = np.concatenate([y_val, y_test[:n_test]]) if n_test else y_val
    sample_ids = [str(s["sample_id"]) for s in explain_samples]

    predict = pipeline_predict_fn(model)
    result = kernel_shap_matrix(predict, background, X_explain, n_coalitions=n_coalitions,
                                random_state=RANDOM_SEED)
    phi, predictions = result["phi"], result["predictions"]
    importance = mean_abs_shap(phi)
    permutation = permutation_importance_ranking(model, X_explain, y_explain, n_repeats=5,
                                                 random_state=RANDOM_SEED)
    agreement = rank_agreement(importance, permutation if permutation is not None
                               else np.zeros_like(importance))

    threshold = 0.5
    if (RESULTS_DIR / "summary.json").exists():
        threshold = float(json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
                          .get("best_threshold", 0.5))
    predicted = (predictions >= threshold).astype(int)
    wrong = [i for i in range(len(predicted)) if int(predicted[i]) != int(y_explain[i])]
    local = [{"sample_id": sample_ids[i], "actual": int(y_explain[i]),
              "probability": float(predictions[i]), "predicted": int(predicted[i]),
              "top_positive": [{"feature": names[j], "phi": float(phi[i, j])}
                               for j in np.argsort(-phi[i])[:4]],
              "top_negative": [{"feature": names[j], "phi": float(phi[i, j])}
                               for j in np.argsort(phi[i])[:4]]} for i in wrong[:3]]

    ranking = sorted(range(len(names)), key=lambda j: -importance[j])
    summary: Dict[str, Any] = {
        "model": model_name,
        "method": "KernelSHAP tự cài đặt (Lundberg & Lee 2017), π(S) = (M−1)/[C(M,|S|)·|S|·(M−|S|)]",
        "n_features": len(names),
        "n_explained": result["n_explained"],
        "n_coalitions": n_coalitions,
        "n_background": int(background.shape[0]),
        "evaluated_on": {"validation": int(len(X_val)), "test": int(n_test)},
        "base_value": result["base_value"],
        "threshold": threshold,
        "self_check": {
            "max_abs_efficiency_gap": result["max_abs_efficiency_gap"],
            "relative_efficiency_gap": result["relative_efficiency_gap"],
            "note": ("Σφ_j + E[f] phải bằng f(x); KernelSHAP ràng buộc đúng điều này nên sai số ≈ 0 "
                     "(kiểm chứng cài đặt, không phải kết quả mô hình)."),
        },
        "importance_top": [{"feature": names[j], "mean_abs_shap": float(importance[j])}
                           for j in ranking[:TOP_FEATURES]],
        "rank_agreement_with_permutation": agreement,
        "misclassified_explained": len(local),
        "local_explanations": local,
        "figures": [],
    }

    if figures:
        figs.mkdir(parents=True, exist_ok=True)
        fig_shap_summary(importance, permutation, names, figs / "01_shap_summary.png")
        fig_shap_beeswarm(phi, X_explain, y_explain, names, figs / "02_shap_beeswarm.png")
        fig_shap_local(phi, result["base_value"], predictions, sample_ids, wrong[:3], names,
                       figs / "03_shap_local_errors.png")
        summary["figures"] = [str(p.relative_to(FIGURES_DIR.parent))
                              for p in sorted(figs.glob("*.png"))]

    if write:
        (out / "shap.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "shap.md").write_text(markdown_shap(summary), encoding="utf-8")

    print(f"SHAP ({model_name}): {result['n_explained']} samples, {n_coalitions} coalitions per point; "
          f"efficiency gap = {result['max_abs_efficiency_gap']:.2e}; top-3 = "
          f"{', '.join(i['feature'] for i in summary['importance_top'][:3])}")
    return summary


def markdown_shap(summary: Dict[str, Any]) -> str:
    """Render `reports/results/shap.md` from the summary payload."""
    agreement = summary["rank_agreement_with_permutation"]
    lines = ["# Giải thích mô hình bằng SHAP (KernelSHAP tự cài đặt)", "",
             f"- Mô hình được giải thích: **{summary['model']}**; ngưỡng vận hành "
             f"{summary['threshold']:.3f}.",
             f"- Phương pháp: {summary['method']}",
             f"- Quy mô: **{summary['n_explained']} mẫu** (validation "
             f"{summary['evaluated_on']['validation']} + test {summary['evaluated_on']['test']}), "
             f"{summary['n_coalitions']} liên minh/điểm, nền {summary['n_background']} mẫu train; "
             f"giá trị nền E[f] = {summary['base_value']:.4f}.",
             "",
             "## 1. Tự kiểm chứng cài đặt",
             "",
             f"- **Efficiency**: sai số lớn nhất |Σφ + E[f] − f(x)| = "
             f"{summary['self_check']['max_abs_efficiency_gap']:.3e} "
             f"(tương đối {summary['self_check']['relative_efficiency_gap']:.3e}).",
             f"- {summary['self_check']['note']}",
             f"- Đối chiếu với permutation importance (phương pháp độc lập): Spearman = "
             f"{agreement.get('spearman')}, trùng top-{agreement.get('top_k')} = "
             f"{agreement.get('top_overlap')}.",
             "",
             "## 2. Độ quan trọng toàn cục (mean |φ|)",
             "",
             markdown_table(["#", "Feature", "mean |φ|"],
                            [[i + 1, r["feature"], f"{r['mean_abs_shap']:.4f}"]
                             for i, r in enumerate(summary["importance_top"])],
                            ["---:", "---", "---:"]),
             "",
             "> Đọc bảng: mean |φ| là mức đóng góp trung bình của feature vào log-odds quyết định. "
             "Đây là thước đo toàn cục, khác permutation importance (mức giảm metric khi hoán vị); "
             "hai phương pháp đồng thuận ⇒ kết luận về nhóm feature dẫn đầu không phụ thuộc một "
             "công cụ duy nhất. Khi feature đa cộng tuyến, SHAP chia \"công\" cho cả nhóm nên đọc "
             "kèm cụm tương quan ở `eda_deep.md`.",
             "",
             "## 3. Giải thích cục bộ các mẫu dự đoán sai",
             ""]
    if summary["local_explanations"]:
        lines += markdown_table(
            ["Mẫu", "Thực tế", "P(nhãn 1)", "Feature đẩy về suy giảm (φ > 0)",
             "Feature kéo về an toàn (φ < 0)"],
            [[r["sample_id"], r["actual"], f"{r['probability']:.3f}",
              ", ".join(f"{i['feature']} ({i['phi']:+.2f})" for i in r["top_positive"]),
              ", ".join(f"{i['feature']} ({i['phi']:+.2f})" for i in r["top_negative"])]
             for r in summary["local_explanations"]],
            ["---", "---:", "---:", "---", "---"])
        lines += ["",
                  "> Đọc bảng: mỗi mẫu sai được phân rã thành các đóng góp dương (đẩy về phía suy "
                  "giảm) và âm (kéo về phía an toàn) — đây là căn cứ giải trình từng hồ sơ, thứ mà "
                  "permutation importance (toàn cục) không cung cấp được."]
    else:
        lines.append("- Không có mẫu dự đoán sai trong tập được giải thích.")
    lines += ["", "## 4. Hình", ""] + [f"- `{p}`" for p in summary["figures"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """Command-line entry point for `scripts.explain_model`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-coalitions", type=int, default=DEFAULT_N_COALITIONS)
    parser.add_argument("--n-background", type=int, default=DEFAULT_N_BACKGROUND)
    parser.add_argument("--max-explain", type=int, default=64)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    print("=== Model explanation with SHAP (self-implemented KernelSHAP) ===")
    run(write=not args.no_write, n_coalitions=args.n_coalitions, n_background=args.n_background,
        max_explain=args.max_explain, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
