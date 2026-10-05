"""Orchestrate the single-vs-hybrid imbalance experiment and export its artifacts.

Runs `python -m labs.imbalance_experiment.main` end to end: builds the pipeline catalogue, cross-validates
every entry, scores the holdout once, prints the comparison tables, and writes summary.md, the CSV and JSON
files, the PR curve figure and run.log into the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from runtime_warnings import quiet_library_warnings

# Silence the harmless warnings this library pair emits so the experiment log stays readable.
# The exact filters are documented in `runtime_warnings.py`.
quiet_library_warnings()

from .config import DEFAULT_OUT_DIR, ExperimentConfig
from .data_loader import Dataset, load_dataset
from .evaluation import (METRIC_KEYS, METRIC_LABELS, TechniqueResult, _markdown_from_rows,
                         evaluate_all, format_mean_std, summary_rows)
from .insights import analyse as analyse_findings
from .insights import render_markdown
from .pipeline_builder import build_pipelines, pipeline_table

LOGGER = logging.getLogger("labs.imbalance_experiment")


def _setup_logging(log_path: Path | None) -> logging.Logger:
    """Attach a UTF-8 console handler plus an optional file handler and return the logger.

    The console handler is installed after stdout is reconfigured, because the default Windows codepage
    cannot encode the non-ASCII report text.
    """
    try:  # pragma: no cover - console dependent
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - the stream does not support reconfigure
        pass
    logger = logging.getLogger("labs.imbalance_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _cv_table(results: Sequence[TechniqueResult]) -> str:
    """Render the cross-validated mean and standard deviation per pipeline as Markdown."""
    rows: List[Dict[str, Any]] = []
    for result in results:
        row: Dict[str, Any] = {"group": result.group, "key": result.key,
                               "status": result.status or "—"}
        for metric in METRIC_KEYS:
            row[metric] = format_mean_std(result, metric) if result.ok else "—"
        row["fit_seconds"] = round(result.fit_seconds_mean, 3) if result.ok else None
        rows.append(row)
    columns = ["group", "key", "status", *METRIC_KEYS, "fit_seconds"]
    headers = ["Nhóm", "Kỹ thuật", "Trạng thái", *[METRIC_LABELS.get(m, m) for m in METRIC_KEYS],
               "giây/fold"]
    return _markdown_from_rows(rows, columns, headers)


def _holdout_table(results: Sequence[TechniqueResult]) -> str:
    """Render the holdout metrics and the compute cost per pipeline as Markdown."""
    rows = summary_rows(results, source="holdout")
    columns = ["group", "key", *METRIC_KEYS, "n_train_after", "resample_seconds_per_fold",
               "fit_seconds_per_fold"]
    headers = ["Nhóm", "Kỹ thuật", *[METRIC_LABELS.get(m, m) for m in METRIC_KEYS],
               "n train sau resample", "giây resample/fold", "giây fit/fold"]
    return _markdown_from_rows(rows, columns, headers)


def _group_table(results: Sequence[TechniqueResult]) -> str:
    """Render the per-group averages and their delta against the baseline as Markdown."""
    baseline = next((r for r in results if r.key == "baseline" and r.ok), None)
    base_pr = baseline.holdout.get("pr_auc", float("nan")) if baseline else float("nan")

    def mean_of(items: Sequence[TechniqueResult], metric: str) -> float:
        values = [r.holdout.get(metric, float("nan")) for r in items if r.ok]
        values = [value for value in values if value == value]
        return float(np.mean(values)) if values else float("nan")

    rows: List[Dict[str, Any]] = []
    groups = (("baseline", [r for r in results if r.group == "baseline" and r.ok]),
              ("single (đơn lẻ)", [r for r in results if r.ok and r.group.startswith("single")]),
              ("hybrid (kết hợp)", [r for r in results if r.ok and r.group == "hybrid"]))
    for name, items in groups:
        if not items:
            continue
        rows.append({"group": name, "n": len(items), "pr_auc": mean_of(items, "pr_auc"),
                     "delta_pr_auc": mean_of(items, "pr_auc") - base_pr, "f1": mean_of(items, "f1"),
                     "recall": mean_of(items, "recall"), "fpr": mean_of(items, "fpr"),
                     "seconds": float(np.mean([r.fit_seconds_mean for r in items]))})
    columns = ["group", "n", "pr_auc", "delta_pr_auc", "f1", "recall", "fpr", "seconds"]
    headers = ["Nhóm", "#kỹ thuật", "PR-AUC (TB)", "ΔPR-AUC vs baseline", "F1 (TB)", "Recall (TB)",
               "FPR (TB)", "giây/fold (TB)"]
    return _markdown_from_rows(rows, columns, headers)


def _ranking_table(results: Sequence[TechniqueResult]) -> str:
    """Render the pipelines ranked by PR-AUC, with F1, recall and FPR alongside."""
    baseline = next((r for r in results if r.key == "baseline" and r.ok), None)
    base_pr = baseline.holdout.get("pr_auc", float("nan")) if baseline else float("nan")
    ok = sorted((r for r in results if r.ok),
                key=lambda r: -float(r.holdout.get("pr_auc", float("-inf"))))
    rows: List[Dict[str, Any]] = []
    for rank, result in enumerate(ok, start=1):
        rows.append({"rank": rank, "key": result.key, "group": result.group,
                     "pr_auc": result.holdout.get("pr_auc", float("nan")),
                     "delta_pr_auc": result.holdout.get("pr_auc", float("nan")) - base_pr,
                     "f1": result.holdout.get("f1", float("nan")),
                     "macro_f1": result.holdout.get("macro_f1", float("nan")),
                     "recall": result.holdout.get("recall", float("nan")),
                     "fpr": result.holdout.get("fpr", float("nan")),
                     "balanced_accuracy": result.holdout.get("balanced_accuracy", float("nan"))})
    columns = ["rank", "key", "group", "pr_auc", "delta_pr_auc", "f1", "macro_f1", "recall", "fpr",
               "balanced_accuracy"]
    headers = ["#", "Kỹ thuật", "Nhóm", "PR-AUC", "ΔPR-AUC vs baseline", "F1 (thiểu số)", "F1-macro",
               "Recall", "FPR", "Balanced Acc"]
    return _markdown_from_rows(rows, columns, headers)


def _leak_table(results: Sequence[TechniqueResult]) -> str:
    """Render the per-pipeline leakage checks as a PASS/FAIL Markdown table."""
    lines = ["| Kỹ thuật | Nhóm | Kiểm chứng chống rò rỉ | Kết quả |", "|---|---|---|---|"]
    for result in results:
        if not result.ok:
            lines.append(f"| `{result.key}` | `{result.group}` | — | {result.status} "
                         f"({result.reason}) |")
            continue
        detail = ", ".join(f"{name}: {'PASS' if value else 'FAIL'}"
                           for name, value in result.checks.items())
        lines.append(f"| `{result.key}` | `{result.group}` | {detail} | "
                     f"{'PASS' if all(result.checks.values()) else 'FAIL'} |")
    return "\n".join(lines)


def plot_pr_curves(results: Sequence[TechniqueResult], path: Path,
                   *, highlight: Sequence[str] = ("baseline", "smote", "smote_class_weight")
                   ) -> bool:
    """Draw the two-panel comparison figure and report whether it could be produced.

    The left panel holds the precision-recall curves, the right panel places every pipeline at the 0.5
    operating point in recall-FPR space. Pipelines listed in `highlight` are drawn thicker.

    Returns:
        True when the PNG was written, False when matplotlib or the PR curve data is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is unavailable
        LOGGER.warning("Skipping the PR curve figure: %s", exc)
        return False

    # 1. Keep only the pipelines that produced a curve.
    plottable = [r for r in results if r.ok and r.pr_curve is not None]
    if not plottable:
        LOGGER.warning("No PR curve data to plot")
        return False

    # 2. Draw the curves, emphasising the highlighted techniques.
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for result in plottable:
        recall = np.asarray(result.pr_curve["recall"], dtype=float)
        precision = np.asarray(result.pr_curve["precision"], dtype=float)
        is_highlight = result.key in highlight
        linewidth = 2.4 if is_highlight else 1.0
        alpha = 1.0 if is_highlight else 0.5
        ap = result.holdout.get("pr_auc", float("nan"))
        axes[0].plot(recall, precision, linewidth=linewidth, alpha=alpha,
                     label=f"{result.key} (AP={ap:.3f})")
        axes[1].plot(result.holdout.get("fpr", float("nan")),
                     result.holdout.get("recall", float("nan")),
                     marker="o", markersize=5, alpha=alpha, label=result.key)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall trên holdout test (càng cao càng tốt)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=6, loc="upper right")
    axes[1].set_xlabel("FPR — báo động giả (càng thấp càng tốt)")
    axes[1].set_ylabel("Recall (càng cao càng tốt)")
    axes[1].set_title("Điểm vận hành tại ngưỡng 0.5 (Recall vs FPR)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=6, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    LOGGER.info("Wrote the PR curve figure: %s", path)
    return True


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]],
               header: Sequence[str] | None = None) -> None:
    """Write rows to CSV with the standard library, so pandas is not required."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(header) if header else list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _serialise(results: Sequence[TechniqueResult]) -> List[Dict[str, Any]]:
    """Flatten the results into a JSON-friendly payload, dropping the numpy PR curve arrays."""
    payload: List[Dict[str, Any]] = []
    for result in results:
        payload.append({
            "key": result.key, "name": result.name, "group": result.group, "note": result.note,
            "status": result.status, "reason": result.reason,
            "is_resampling": result.is_resampling,
            "n_train_after_mean": result.n_train_after_mean,
            "fit_seconds_mean": result.fit_seconds_mean,
            "resample_seconds_mean": result.resample_seconds_mean,
            "total_seconds": result.total_seconds,
            "cv_mean": result.mean, "cv_std": result.std, "holdout": result.holdout,
            "checks": result.checks,
            "folds": [{"fold": row.fold, "n_train": row.n_train, "n_val": row.n_val,
                       "n_train_after": row.n_train_after, "ir_before": row.ir_before,
                       "ir_after": row.ir_after, "fit_seconds": row.fit_seconds,
                       "resample_seconds": row.resample_seconds, **row.metrics}
                      for row in result.folds]})
    return payload


def _close_file_handlers(logger: logging.Logger) -> None:
    """Close and detach file handlers once the artifacts are written.

    On Windows an open run.log blocks deleting or moving the output directory, for example when
    `tempfile.TemporaryDirectory()` cleans up.
    """
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def run(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Run the experiment: load the dataset, evaluate every pipeline, then build the tables.

    Args:
        cfg: experiment configuration, see `ExperimentConfig`.

    Returns:
        Mapping with `dataset`, `results`, `findings`, `tables`, `artifacts`, `config` and
        `summary_markdown`.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logging(cfg.out_dir / "run.log" if cfg.write else None)

    # 1. Load the dataset and describe the run before any model is fitted.
    dataset = load_dataset(cfg)
    logger.info("EXPERIMENT: single techniques vs hybrid techniques on imbalanced data")
    logger.info("Data      : %s", dataset.describe())
    logger.info("Config    : %s", {key: value for key, value in cfg.to_dict().items()
                                  if key not in ("out_dir", "techniques")})
    logger.info("Leakage   : every sampler stays inside imblearn.pipeline.Pipeline; "
                "the holdout is scored once")

    # 2. Build the catalogue and evaluate every entry.
    specs = build_pipelines(cfg, keys=cfg.techniques)
    logger.info("\nCatalogue of %d pipelines:\n%s", len(specs), pipeline_table(specs))

    results = evaluate_all(specs, dataset, cfg, log=logger.info)
    findings = analyse_findings(results, cfg)

    # 3. Report the leakage verdict before the metric tables.
    failed = [r.key for r in results if r.ok and not all(r.checks.values())]
    logger.info("\nLeakage checks: %s",
                "ALL PASS" if not failed else f"FAIL for: {', '.join(failed)}")

    # 4. Render each table once, then reuse it for the console and the report.
    tables = {
        "pipeline_catalogue": pipeline_table(specs),
        "cv_mean_std": _cv_table(results),
        "holdout": _holdout_table(results),
        "by_group": _group_table(results),
        "ranking": _ranking_table(results),
        "leakage": _leak_table(results),
    }
    logger.info("\nRanking by PR-AUC (holdout test)\n%s", tables["ranking"])
    logger.info("\nAverages by group\n%s", tables["by_group"])
    logger.info("\nMetrics per pipeline: mean and standard deviation over %d folds\n%s", cfg.n_splits,
                tables["cv_mean_std"])

    # 5. Write the artifacts and release the log file handle.
    summary_md = _build_summary_markdown(dataset, results, findings, tables, cfg)
    artifacts: Dict[str, str] = {}
    if cfg.write:
        write_artifacts(cfg, results, findings, tables, summary_md, logger, artifacts)
    logger.info("\nDone. Best pipeline by PR-AUC: %s",
                findings["ranking"].get("pr_auc", {}).get("key", "—"))
    _close_file_handlers(logger)
    return {"dataset": dataset, "results": results, "findings": findings, "tables": tables,
            "artifacts": artifacts, "config": cfg.to_dict(), "summary_markdown": summary_md}


def _build_summary_markdown(dataset: Dataset, results: Sequence[TechniqueResult],
                            findings: Dict[str, Any], tables: Dict[str, str],
                            cfg: ExperimentConfig) -> str:
    """Assemble the full Markdown report from the rendered tables and the insight section."""
    lines = ["# Thực nghiệm: Phương pháp đơn lẻ vs Phương pháp kết hợp (imbalanced learning)", "",
             f"- Dữ liệu: **{dataset.describe()}**",
             f"- Cấu hình: `n_samples={cfg.n_samples}`, `imbalance_ratio=1:{cfg.imbalance_ratio}`, "
             f"`n_splits={cfg.n_splits}`, `base_model={cfg.base_model}`, "
             f"`threshold={cfg.threshold}`, `seed={cfg.seed}`",
             "- Chống rò rỉ: mọi resampling nằm TRONG `imblearn.pipeline.Pipeline` (chỉ chạy trên train "
             "của từng fold); holdout test chấm đúng MỘT lần sau khi refit trên toàn bộ train.",
             "- Metric chính: PR-AUC, ROC-AUC, F1 (thiểu số/macro), Balanced Accuracy, Recall, FPR — "
             "**Accuracy không dùng** để kết luận.", "",


             "## 1. Danh mục pipeline", "", tables["pipeline_catalogue"],
             "## 2. Trung bình theo nhóm (Baseline · Đơn lẻ · Kết hợp)", "", tables["by_group"],
             "## 3. Xếp hạng theo PR-AUC (holdout test)", "", tables["ranking"],
             "## 4. Metric chi tiết theo pipeline — CV mean ± std", "", tables["cv_mean_std"],
             "## 5. Metric trên holdout test + chi phí tính toán", "", tables["holdout"],
             "## 6. Kiểm chứng chống rò rỉ dữ liệu", "", tables["leakage"], "",
             render_markdown(results, cfg)]
    return "\n".join(lines) + "\n"


def write_artifacts(cfg: ExperimentConfig, results: Sequence[TechniqueResult],
                    findings: Dict[str, Any], tables: Dict[str, str], summary_md: str,
                    logger: logging.Logger, artifacts: Dict[str, str]) -> Dict[str, str]:
    """Write every artifact into `cfg.out_dir` and return the mapping of names to paths."""
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    (out / "summary.md").write_text(summary_md, encoding="utf-8")
    artifacts["summary_md"] = str(out / "summary.md")

    holdout_rows = summary_rows(results, source="holdout")
    _write_csv(out / "summary.csv", holdout_rows)
    artifacts["summary_csv"] = str(out / "summary.csv")

    cv_rows = summary_rows(results, source="cv")
    _write_csv(out / "cv_mean_std.csv", cv_rows)
    artifacts["cv_csv"] = str(out / "cv_mean_std.csv")

    (out / "results.json").write_text(
        json.dumps({"config": cfg.to_dict(), "findings": findings, "results": _serialise(results)},
                   ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    artifacts["results_json"] = str(out / "results.json")

    if plot_pr_curves(results, out / "pr_curves.png"):
        artifacts["pr_curves"] = str(out / "pr_curves.png")
    artifacts["run_log"] = str(out / "run.log")
    logger.info("Wrote artifacts: %s", ", ".join(sorted(Path(path).name
                                                        for path in artifacts.values())))
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for the experiment."""
    parser = argparse.ArgumentParser(
        description="Compare single techniques against hybrid techniques on imbalanced data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", default="synthetic",
                        help="'synthetic' or a path to a CSV file (for example data/creditcard.csv)")
    parser.add_argument("--target", default="Class", help="Label column name when reading a CSV file")
    parser.add_argument("--n-samples", type=int, default=ExperimentConfig.n_samples)
    parser.add_argument("--imbalance-ratio", type=int, default=ExperimentConfig.imbalance_ratio,
                        help="Majority to minority ratio (50 means 1:50)")
    parser.add_argument("--test-size", type=float, default=ExperimentConfig.test_size)
    parser.add_argument("--cv", type=int, default=ExperimentConfig.n_splits,
                        help="Number of StratifiedKFold folds")
    parser.add_argument("--threshold", type=float, default=ExperimentConfig.threshold)
    parser.add_argument("--model", default=ExperimentConfig.base_model,
                        choices=("lightgbm", "random_forest"), help="Base classifier")
    parser.add_argument("--techniques", default="",
                        help="Comma separated technique keys (default: all of them)")
    parser.add_argument("--over-strategy", type=float, default=ExperimentConfig.over_strategy)
    parser.add_argument("--under-strategy", type=float, default=ExperimentConfig.under_strategy)
    parser.add_argument("--quick", action="store_true", help="Fewer estimators, for a faster run")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--no-write", action="store_true", help="Skip writing artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: build the configuration from the arguments and run the experiment."""
    args = build_parser().parse_args(argv)
    keys = tuple(key.strip() for key in args.techniques.split(",") if key.strip())
    cfg = ExperimentConfig(
        source=args.data, target_column=args.target, n_samples=args.n_samples,
        imbalance_ratio=args.imbalance_ratio, test_size=args.test_size, n_splits=args.cv,
        threshold=args.threshold, base_model=args.model, over_strategy=args.over_strategy,
        under_strategy=args.under_strategy, quick=args.quick, techniques=keys,
        out_dir=Path(args.out_dir), write=not args.no_write)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
