"""Hyperparameter random search with an experiment ledger.

Runs random search against cross-company average precision using StratifiedGroupKFold, the same protocol
as `forecasting.tuning`, records every trial in `reports/results/runs.csv`, and compares the result with
GridSearchCV and the defaults in `HYPERPARAMS`. Writes `reports/results/search.{json,md}`.
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

from forecasting.config import GROUP_KEY, RESULTS_DIR, RANDOM_SEED, ensure_dirs, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.eda import markdown_table
from forecasting.features import build_feature_matrix, extract_labels
from forecasting.models import DEFAULT_MODEL_ORDER, MODEL_REGISTRY
from forecasting.search import (SEARCH_SPACES, compare_with_grid, cross_company_ap, random_search,
                                write_ledger)

#: Models searched by default, taken from the single registry in `forecasting.models`.
DEFAULT_MODELS = list(DEFAULT_MODEL_ORDER)


def _grid_reference(model: str) -> Dict[str, Any] | None:
    """Read the GridSearchCV result for the same model from `tuning.json`, when present."""
    path = RESULTS_DIR / "tuning.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("results") or []:
        if row.get("model") == model:
            return {"best_cv_average_precision": row.get("best_cv_average_precision"),
                    "best_params": row.get("best_params"),
                    "default_cv_average_precision":
                        (row.get("default_cv_reference") or {}).get("mean_average_precision")}
    return None


def fig_search(rows: List[Dict[str, Any]], path: Path) -> None:
    """Plot the CV average precision of every trial, marking the best point and the grid reference."""
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        values = [r["cv_average_precision"] for r in rows
                  if r["model"] == model and r.get("cv_average_precision") is not None]
        if not values:
            ax.axis("off")
            continue
        ax.hist(values, bins=min(15, max(5, len(values) // 3)), color="mediumseagreen", alpha=0.85)
        best = max(values)
        ax.axvline(best, color="crimson", ls="--", lw=1.2, label=f"tốt nhất {best:.4f}")
        reference = next((r.get("grid_search_best_cv_ap") for r in rows
                          if r["model"] == model and r.get("grid_search_best_cv_ap")), None)
        if reference:
            ax.axvline(reference, color="steelblue", ls=":", lw=1.2, label=f"Grid {reference:.4f}")
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("CV-AP (cross-company)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Random search: phân bố CV-AP của các trial (so với GridSearchCV)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(write: bool = True, trials: int = 40, models: List[str] | None = None,
        folds: int = 4, winsorize: str = "none", out_dir: Path | None = None,
        fig_dir: Path | None = None, figures: bool = True) -> Dict[str, Any]:
    """Run random search per model and write the artifacts plus the `runs.csv` ledger."""
    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    figs = Path(fig_dir) if fig_dir else (RESULTS_DIR.parent / "figures" / "search")
    names = [m for m in (models or DEFAULT_MODELS) if m in MODEL_REGISTRY]
    splits = {n: load_prepared(n) for n in ("train", "validation")}
    samples = splits["train"] + splits["validation"]   # tune on train plus validation, never on test
    X, y = build_feature_matrix(samples), extract_labels(samples)
    groups = np.asarray([s[GROUP_KEY] for s in samples])

    trials_all: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for model in names:
        search = random_search(model, X, y, groups, n_trials=trials, seed=RANDOM_SEED,
                               n_splits=folds, winsorize=winsorize)
        reference = _grid_reference(model)
        comparison = compare_with_grid(search, (reference or {}).get("best_cv_average_precision"))
        default_metrics = cross_company_ap(model, {}, X, y, groups, n_splits=folds,
                                          winsorize=winsorize)
        rows.append({
            "model": model, "n_trials": search["n_trials"], "n_valid": search["n_valid"],
            "best_params": (search["best"] or {}).get("params"),
            "best_cv_average_precision": (search["best"] or {}).get("cv_average_precision"),
            "best_cv_ap_std": (search["best"] or {}).get("cv_ap_std"),
            "best_cv_auroc": (search["best"] or {}).get("cv_auroc"),
            "default_cv_average_precision": default_metrics.get("cv_average_precision"),
            "grid_search_best_cv_ap": (reference or {}).get("best_cv_average_precision"),
            "grid_search_best_params": (reference or {}).get("best_params"),
            "delta_vs_grid": comparison.get("delta_vs_grid"),
            "delta_vs_default": ((search["best"] or {}).get("cv_average_precision") or 0.0)
                                 - (default_metrics.get("cv_average_precision") or 0.0),
            **comparison,
        })
        trials_all += search["trials"]

    ledger_path = out / "runs.csv"
    if write:
        write_ledger(ledger_path, trials_all)
    summary: Dict[str, Any] = {
        "method": ("random search (log-uniform cho tham số scale) + log MỌI trial; môi trường không có "
                   "Optuna nên dùng cách này, không gian tham số mô tả ở `SEARCH_SPACES`"),
        "objective": "AP out-of-fold, StratifiedGroupKFold theo công ty trên train+validation",
        "n_trials_per_model": trials, "models": names, "seed": RANDOM_SEED, "winsorize": winsorize,
        "rows": rows,
        "ledger": {"path": str(ledger_path.relative_to(RESULTS_DIR.parent)), "n_rows": len(trials_all)},
        "figures": [],
    }
    if figures:
        figs.mkdir(parents=True, exist_ok=True)
        fig_search(trials_all, figs / "01_search_distribution.png")
        summary["figures"] = [str(p.relative_to(RESULTS_DIR.parent))
                              for p in sorted(figs.glob("*.png"))]
    if write:
        (out / "search.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "search.md").write_text(markdown_search(summary), encoding="utf-8")

    best_row = max(rows, key=lambda r: r.get("best_cv_average_precision") or 0) if rows else {}
    print(f"Search: {len(names)} models x {trials} trials; best {best_row.get('model')} "
          f"CV-AP={best_row.get('best_cv_average_precision')}; delta vs Grid="
          f"{best_row.get('delta_vs_grid')}; ledger: {ledger_path.name} ({len(trials_all)} rows)")
    return summary


def markdown_search(summary: Dict[str, Any]) -> str:
    """Render `reports/results/search.md` from the JSON payload."""
    rows = summary["rows"]
    lines = ["# Tìm kiếm siêu tham số: random search + sổ thực nghiệm", "",
             f"- Phương pháp: {summary['method']}",
             f"- Mục tiêu: {summary['objective']}; seed {summary['seed']}; "
             f"{summary['n_trials_per_model']} trial/mô hình; winsorize={summary['winsorize']}.",
             f"- Sổ thực nghiệm: `{summary['ledger']['path']}` — **{summary['ledger']['n_rows']} dòng** "
             f"(mỗi dòng = một trial, kèm params/seed/CV-AP/thời gian/trạng thái).",
             "",
             "## 1. Kết quả từng mô hình", "",
             markdown_table(
                 ["Mô hình", "#trial", "CV-AP tốt nhất", "± độ lệch", "CV-AP mặc định",
                  "CV-AP GridSearchCV", "Δ vs Grid", "Δ vs mặc định", "Thời gian (s)"],
                 [[r["model"], r["n_trials"],
                   f"{r['best_cv_average_precision']:.4f}" if r.get("best_cv_average_precision") else "—",
                   f"{r['best_cv_ap_std']:.4f}" if r.get("best_cv_ap_std") is not None else "—",
                   (f"{r['default_cv_average_precision']:.4f}"
                    if r.get("default_cv_average_precision") else "—"),
                   f"{r['grid_search_best_cv_ap']:.4f}" if r.get("grid_search_best_cv_ap") else "—",
                   f"{r['delta_vs_grid']:+.4f}" if r.get("delta_vs_grid") is not None else "—",
                   f"{r['delta_vs_default']:+.4f}" if r.get("delta_vs_default") is not None else "—",
                   f"{r['total_seconds']:.1f}" if r.get("total_seconds") else "—"]
                  for r in rows],
                 ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
             "",
             "**Cấu hình tốt nhất tìm được:**", ""]
    for r in rows:
        lines.append(f"- `{r['model']}`: `{json.dumps(r.get('best_params'), ensure_ascii=False)}` → "
                     f"CV-AP {r.get('best_cv_average_precision')}")
    lines += ["",
              "> Đọc bảng: **Δ vs Grid** cho biết tìm kiếm ngẫu nhiên có vượt lưới GridSearchCV hay không "
              "(cùng thước đo CV-AP và cùng splitter). Δ ≈ 0 nghĩa là lưới cũ đã đủ tốt và chi phí thêm "
              "trial là lãng phí — kết luận này cũng được ghi lại thay vì chỉ khoe con số đẹp.",
              "> Muốn dùng Bayesian search (Optuna TPE) khi môi trường có: thay `sample_params` bằng "
              "`trial.suggest_float(..., log=True)` với cùng các không gian ở `SEARCH_SPACES`; phần "
              "ghi sổ/so sánh giữ nguyên.",
              "", "## 2. Hình", ""] + [f"- `{p}`" for p in summary["figures"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """Command-line entry point for `scripts.search`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40, help="Number of trials per model.")
    parser.add_argument("--models", default="", help="Comma-separated list of models.")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--winsorize", default="none", choices=("none", "iqr", "p1p99"))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    models = [m for m in args.models.split(",") if m] or None
    print("=== Random search with experiment ledger (objective: cross-company CV-AP) ===")
    run(write=not args.no_write, trials=args.trials, models=models, folds=args.folds,
        winsorize=args.winsorize, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
