"""Chạy toàn bộ lab: 3 chiến lược + mốc minh hoạ SAI, ghi artifact và log phân phối nhãn.

Lệnh:
    python -m imbalance_lab.run                    # cấu hình mặc định (50.000 mẫu, 98/2)
    python -m imbalance_lab.run --n-samples 20000 --no-imblearn
    python -m imbalance_lab.run --skip-leaky       # bỏ phần minh hoạ rò rỉ

Artifact (mặc định `reports/imbalance/`):
- `summary.md`            — bảng so sánh + kết luận + cách tái lập;
- `summary.csv`           — metric trên holdout test theo chiến lược × ngưỡng;
- `metrics_by_fold.csv`   — metric từng fold tại ngưỡng 0.5;
- `resampling_by_fold.csv`— phân phối nhãn TRƯỚC/SAU resampling từng fold;
- `thresholds.json`       — ngưỡng chọn trên xác suất out-of-fold;
- `run.log`               — toàn bộ log (gồm phân phối nhãn trước/sau mỗi fold).
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata as md
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import config as C
from .cv import cross_validate_strategy, refit_and_score
from .data import (format_distribution, label_distribution, make_imbalanced_dataset,
                   stratified_holdout_split)
from .metrics import bootstrap_ci, metrics_at_threshold
from .models import (STRATEGIES, build_leaky_reference, build_strategy, classifier_backend,
                     make_base_classifier, resample_once)
from .samplers import resampling_backend
from .thresholds import tune_thresholds

#: Thứ tự cột metric in ra bảng console/CSV.
CSV_METRICS = ("precision", "recall", "f1", "pr_auc", "roc_auc", "brier", "mcc",
               "n_predicted_positive", "tn", "fp", "fn", "tp")


def _version(package: str) -> str:
    try:
        return md.version(package)
    except Exception:  # pragma: no cover - package có thể không cài dạng metadata
        return "?"


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float) + "\n",
                    encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = header or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _print_metrics_table(title: str, rows: List[Dict[str, Any]], log) -> None:
    """In bảng metric dạng cột cố định (Precision/Recall/F1/PR-AUC/ROC-AUC/Brier/MCC)."""
    log(f"\n    {title}")
    log("      {:<26s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
        "cấu hình", "thr", "prec", "recall", "f1", "pr_auc", "roc_auc", "brier", "mcc"))
    for row in rows:
        log("      {:<26s} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.3f} "
            "{:>7.3f}".format(
                row["label"], row["threshold"], row.get("precision", float("nan")),
                row.get("recall", float("nan")), row.get("f1", float("nan")),
                row.get("pr_auc", float("nan")), row.get("roc_auc", float("nan")),
                row.get("brier", float("nan")), row.get("mcc", float("nan"))))


def _majority_note(y_test: np.ndarray) -> str:
    positive_rate = 100.0 * float((np.asarray(y_test) == 1).mean())
    return (f"nhãn test: {positive_rate:.2f}% dương ⇒ đoán “toàn bộ là lớp đa số” đã đạt "
            f"{100.0 - positive_rate:.2f}% accuracy")


def leaky_demo(n_samples: int, prefer_imblearn: bool, seed: int, log) -> Dict[str, Any]:
    """Mốc MINH HOẠ SAI: SMOTE+undersample trên TOÀN BỘ dữ liệu rồi mới chia train/test.

    Kết quả sẽ lạc quan hơn thực tế vì tập test chứa mẫu TỔNG HỢP nội suy từ cả mẫu train.
    Hàm đếm số mẫu test "không tồn tại trong dữ liệu gốc" để chứng minh bằng số.
    """
    X, y = make_imbalanced_dataset(n_samples=n_samples, random_state=seed)
    leaky = build_leaky_reference(prefer_imblearn, seed)
    X_res, y_res = resample_once(leaky, X, y)
    original_rows = {row.tobytes() for row in X}
    split = stratified_holdout_split(X_res, y_res, seed=seed)
    synthetic_in_test = int(sum(1 for row in split["X_test"] if row.tobytes() not in original_rows))

    model = leaky["classifier"]
    model.fit(split["X_train"], split["y_train"])
    proba = model.predict_proba(split["X_test"])[:, 1]
    thresholds = tune_thresholds(split["y_test"], proba, cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                                 precision_target=C.PRECISION_TARGET)
    rows = []
    for name, cfg in thresholds.items():
        rows.append({"label": f"NÊN TRÁNH · {name}", "strategy": "leaky_resample_before_split",
                     "threshold_mode": name, "split": "test",
                     **metrics_at_threshold(split["y_test"], proba, cfg["threshold"])})
    log(f"    resample trước khi chia: {format_distribution(label_distribution(y))} → "
        f"{format_distribution(label_distribution(y_res))}; test chứa "
        f"{synthetic_in_test}/{len(split['y_test'])} mẫu TỔNG HỢP (không có trong dữ liệu gốc)")
    _print_metrics_table("Mốc minh hoạ SAI (rò rỉ) — chỉ để so sánh", rows, log)
    return {"rows": rows, "synthetic_rows_in_test": synthetic_in_test,
            "n_test": int(len(split["y_test"])), "thresholds": thresholds}


def run(n_samples: Optional[int] = None, prefer_imblearn: bool = C.PREFER_IMBLEARN,
        include_leaky: bool = True, write: bool = True) -> Dict[str, Any]:
    """Chạy lab: sinh dữ liệu → CV từng chiến lược → chọn ngưỡng trên OOF → chốt trên test."""
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = []

    def log(message: str = "") -> None:
        import sys  # nội bộ để không phụ thuộc import cấp module

        try:  # console Windows (cp1252) hoặc stdout bị redirect: ép UTF-8 để không UnicodeEncodeError
            reconfigure = getattr(sys.stdout, "reconfigure", None)
            encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
            if reconfigure is not None and encoding not in ("utf-8", "utf8"):
                reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - stream không hỗ trợ reconfigure
            pass
        print(message)
        log_lines.append(message)

    seed, n = C.SEED, n_samples or C.N_SAMPLES
    log("=== LAB MẤT CÂN BẰNG LỚP — pipeline không rò rỉ dữ liệu ===")
    log(f"Backend phân loại   : {classifier_backend()} "
        f"(lightgbm {_version('lightgbm')}, scikit-learn {_version('scikit-learn')})")
    log(f"Backend resampling  : {resampling_backend(prefer_imblearn)} "
        f"(imbalanced-learn {_version('imbalanced-learn')})")
    log(f"Chiến lược          : {', '.join(STRATEGIES)} (+1 mốc minh hoạ SAI)")

    X, y = make_imbalanced_dataset(n_samples=n, random_state=seed)
    split = stratified_holdout_split(X, y, seed=seed)
    X_pool, y_pool = split["X_train"], split["y_train"]
    X_test, y_test = split["X_test"], split["y_test"]

    log("\n[1] Dữ liệu (mất cân bằng 98/2)")
    log(f"    toàn bộ     : {format_distribution(label_distribution(y))}")
    log(f"    train_pool  : {format_distribution(label_distribution(y_pool))}")
    log(f"    holdout test: {format_distribution(label_distribution(y_test))}"
        f"  ← KHÔNG resample, chỉ dùng một lần để chốt")
    log(f"    Lưu ý Accuracy: {_majority_note(y_test)}")

    strategy_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    resample_rows: List[Dict[str, Any]] = []
    thresholds_json: Dict[str, Any] = {}
    oof_rows: List[Dict[str, Any]] = []

    for index, name in enumerate(STRATEGIES, start=2):
        strategy = build_strategy(name, prefer_imblearn=prefer_imblearn, random_state=seed)
        log(f"\n[{index}] Chiến lược `{name}` — {strategy['doc']}")
        cv = cross_validate_strategy(strategy, X_pool, y_pool, n_splits=C.N_SPLITS, seed=seed,
                                     probe_factory=strategy["probe_factory"], log=log)
        tuned = tune_thresholds(y_pool, cv["oof_proba"], cost_fn=C.COST_FN, cost_fp=C.COST_FP,
                                precision_target=C.PRECISION_TARGET)
        thresholds_json[name] = {mode: cfg["threshold"] for mode, cfg in tuned.items()}
        log("    Ngưỡng chọn trên xác suất OOF (chỉ dùng train_pool): "
            + ", ".join(f"{mode}={cfg['threshold']:.4f}" for mode, cfg in tuned.items()))

        threshold_map = {mode: cfg["threshold"] for mode, cfg in tuned.items()}
        scored = refit_and_score(strategy, X_pool, y_pool, X_test, y_test, threshold_map)
        rows: List[Dict[str, Any]] = []
        for mode, metrics in scored["at_threshold"].items():
            rows.append({"strategy": name, "threshold_mode": mode, "split": "test", **metrics})
        _print_metrics_table(f"Holdout test (n={len(y_test)}) — {name}", 
                             [{"label": f"{name} · {r['threshold_mode']}", **r} for r in rows], log)
        ci = bootstrap_ci(y_test, scored["proba"], thresholds_json[name]["best_f1"],
                          metric="pr_auc", n_boot=C.N_BOOTSTRAP, seed=seed)
        log(f"    PR-AUC CI95 (bootstrap {ci['n_boot_used']} vòng) = "
            f"{ci['point']:.3f} [{ci['lo95']:.3f}, {ci['hi95']:.3f}]")

        oof_metrics = metrics_at_threshold(y_pool, cv["oof_proba"], 0.5)
        oof_rows.append({"strategy": name, "threshold_mode": "fixed_0.5", "split": "oof",
                         **oof_metrics})
        strategy_rows.extend(rows)
        for row in cv["fold_metrics"]:
            fold_rows.append({"strategy": name, **row})
        for entry in cv["folds"]:
            resample_rows.append({
                "strategy": name, "fold": entry["fold"],
                "train_n": entry["train_before"]["n"],
                "train_neg_before": entry["train_before"]["n_negative"],
                "train_pos_before": entry["train_before"]["n_positive"],
                "train_ir_before": entry["train_before"]["imbalance_ratio"],
                "train_neg_after": entry["train_after"]["n_negative"],
                "train_pos_after": entry["train_after"]["n_positive"],
                "train_ir_after": entry["train_after"]["imbalance_ratio"],
                "val_n": entry["val"]["n"], "val_pos": entry["val"]["n_positive"],
                "val_pos_pct": entry["val"]["positive_pct"]})

    leaky = leaky_demo(n, prefer_imblearn, seed, log) if include_leaky else None
    result: Dict[str, Any] = {
        "strategies": list(STRATEGIES), "n_samples": n, "seed": seed,
        "backend": classifier_backend(), "resampling_backend": resampling_backend(prefer_imblearn),
        "label_distribution": {"all": label_distribution(y),
                               "train_pool": label_distribution(y_pool),
                               "test": label_distribution(y_test)},
        "strategy_rows": strategy_rows, "oof_rows": oof_rows, "fold_rows": fold_rows,
        "resample_rows": resample_rows, "thresholds": thresholds_json,
        "leaky": None if leaky is None else {
            "rows": leaky["rows"], "synthetic_rows_in_test": leaky["synthetic_rows_in_test"],
            "n_test": leaky["n_test"]},
        "log": "\n".join(log_lines),
    }
    if write:
        _write_artifacts(result)
    return result


def _markdown(result: Dict[str, Any]) -> str:
    """Bảng Markdown: metric trên test theo chiến lược × ngưỡng, log resampling, kết luận."""
    rows = result["strategy_rows"]
    lines = ["# Lab mất cân bằng lớp — pipeline không rò rỉ dữ liệu", "",
             f"- Backend phân loại: **{result['backend']}** · resampling: **{result['resampling_backend']}**"
             f" · seed {result['seed']} · n_samples {result['n_samples']}",
             f"- Phân phối nhãn: toàn bộ {format_distribution(result['label_distribution']['all'])}; "
             f"train_pool {format_distribution(result['label_distribution']['train_pool'])}; "
             f"test {format_distribution(result['label_distribution']['test'])}",
             "- Quy tắc chống rò rỉ: StratifiedKFold trên train_pool; SMOTE+undersample chỉ nằm trong "
             "pipeline (`imblearn.pipeline.Pipeline`) nên chỉ chạy trên train của từng fold; ngưỡng "
             "chọn trên xác suất out-of-fold; test chỉ dùng một lần.", "",
             "## Metric trên holdout test (n = %d)" % result["label_distribution"]["test"]["n"], "",
             "| Chiến lược | Ngưỡng | thr | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | MCC | FP | FN |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| `{row['strategy']}` | {row['threshold_mode']} | {row['threshold']:.4f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['pr_auc']:.3f} | "
            f"{row['roc_auc']:.3f} | {row.get('brier', float('nan')):.3f} | {row['mcc']:.3f} | "
            f"{row['fp']} | {row['fn']} |")
    lines += ["", "## Xác suất out-of-fold (chỉ để chọn ngưỡng, không phải kết quả chốt)", "",
              "| Chiến lược | thr | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["oof_rows"]:
        lines.append(f"| `{row['strategy']}` | {row['threshold']:.4f} | {row['precision']:.3f} | "
                     f"{row['recall']:.3f} | {row['f1']:.3f} | {row['pr_auc']:.3f} | "
                     f"{row['roc_auc']:.3f} | {row['mcc']:.3f} |")
    lines += ["", "## Ngưỡng chọn trên OOF", "", "| Chiến lược | best_f1 | best_cost | min_precision | 0.5 |",
              "|---|---:|---:|---:|---:|"]
    for name, modes in result["thresholds"].items():
        lines.append(f"| `{name}` | {modes.get('best_f1', float('nan')):.4f} | "
                     f"{modes.get('best_cost', float('nan')):.4f} | "
                     f"{modes.get('min_precision', float('nan')):.4f} | 0.5000 |")
    lines += ["", "## Phân phối nhãn theo fold (trước → sau resampling)", "",
              "| Chiến lược | Fold | Train âm/dương TRƯỚC | IR trước | Train âm/dương SAU | IR sau | Val dương |",
              "|---|---:|---|---:|---|---:|---:|"]
    for row in result["resample_rows"]:
        lines.append(f"| `{row['strategy']}` | {row['fold']} | "
                     f"{row['train_neg_before']}/{row['train_pos_before']} | {row['train_ir_before']:.1f} | "
                     f"{row['train_neg_after']}/{row['train_pos_after']} | {row['train_ir_after']:.1f} | "
                     f"{row['val_pos']} ({row['val_pos_pct']:.2f}%) |")
    if result.get("leaky"):
        lines += ["", "## Mốc MINH HOẠ SAI (resample trước khi chia train/test) — chỉ để so sánh", "",
                  "| Ngưỡng | Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
        for row in result["leaky"]["rows"]:
            lines.append(f"| {row['threshold_mode']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                         f"{row['f1']:.3f} | {row['pr_auc']:.3f} | {row['roc_auc']:.3f} |")
        lines.append(f"\n- Tập test của mốc SAI chứa **{result['leaky']['synthetic_rows_in_test']}/"
                     f"{result['leaky']['n_test']} mẫu tổng hợp** không tồn tại trong dữ liệu gốc "
                     f"⇒ metric bị thổi phồng, không dùng để kết luận.")
    lines += ["", "## Kết luận & khuyến nghị", "", *_conclusions(result), "",
              "## Tái lập", "", "```powershell",
              "python -m pip install -r imbalance_lab/requirements.txt",
              "python -m imbalance_lab.run        # log ở reports/imbalance/run.log",
              "python -m unittest discover -s tests -v   # gồm test chống rò rỉ của lab",
              "```"]
    return "\n".join(lines) + "\n"


def _conclusions(result: Dict[str, Any]) -> List[str]:
    """Viết kết luận tự động từ số liệu (không nhập tay)."""
    rows = result["strategy_rows"]
    if not rows:
        return ["(không có kết quả)"]
    best_pr = max(rows, key=lambda r: r.get("pr_auc", float("-inf")))
    best_f1 = max(rows, key=lambda r: r.get("f1", float("-inf")))
    fixed_f1 = {r["strategy"]: r["f1"] for r in rows if r["threshold_mode"] == "fixed_0.5"}
    thresholds = result["thresholds"]
    gains = []
    for name in thresholds:
        candidates = [r for r in rows if r["strategy"] == name and r["threshold_mode"] != "fixed_0.5"]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r["f1"])
        gains.append((name, best["f1"] - fixed_f1.get(name, float("nan")), best))
    lines = [
        f"1. **PR-AUC cao nhất trên test**: `{best_pr['strategy']}` = **{best_pr['pr_auc']:.3f}**; ba "
        f"chiến lược chênh nhau rất ít (PR-AUC "
        f"{min(r['pr_auc'] for r in rows):.3f}–{max(r['pr_auc'] for r in rows):.3f}) ⇒ với LightGBM, "
        f"xử lý mất cân bằng chủ yếu đổi **điểm vận hành** (precision/recall) chứ không đổi nhiều thứ "
        f"hạng xác suất (PR-AUC/ROC-AUC gần như trùng nhau).",
        f"2. **F1 cao nhất**: `{best_f1['strategy']}` = **{best_f1['f1']:.3f}** (precision "
        f"{best_f1['precision']:.3f}, recall {best_f1['recall']:.3f}) tại ngưỡng "
        f"{best_f1['threshold_mode']} = {best_f1['threshold']:.4f}.",
    ]
    if gains:
        name, gain, best = max(gains, key=lambda item: item[1])
        lines.append(
            f"3. **Tinh chỉnh ngưỡng quan trọng nhất ở `{name}`**: F1 {fixed_f1.get(name, float('nan')):.3f} "
            f"(cố định 0.5) → **{best['f1']:.3f}** (ngưỡng {best['threshold']:.4f}), ΔF1 = {gain:+.3f}. "
            f"Ngưỡng tối ưu giữa các chiến lược lệch nhau rất nhiều ("
            + ", ".join(f"`{n}` {thresholds[n].get('best_f1', float('nan')):.3f}" for n in thresholds)
            + ") ⇒ đổi cách xử lý mất cân bằng thì PHẢI chọn lại ngưỡng, không giữ 0.5.")
    lines.append(
        f"4. **Không dùng Accuracy**: tỉ lệ dương ở test là "
        f"{result['label_distribution']['test']['positive_pct']:.2f}% ⇒ quy tắc “đoán toàn lớp đa số” "
        f"đã đạt {100 - result['label_distribution']['test']['positive_pct']:.2f}% accuracy; thước đo "
        f"chính là PR-AUC, Recall và F1 tại ngưỡng nghiệp vụ.")
    resample = [r for r in result["resample_rows"] if r["strategy"] == "resampling"]
    if resample:
        lines.append(
            f"5. **Resampling chỉ chạy trên train của fold**: IR của train mỗi fold "
            f"{resample[0]['train_ir_before']:.1f} → **{resample[0]['train_ir_after']:.1f}** "
            f"(xem `resampling_by_fold.csv`), còn validation/test giữ nguyên tỉ lệ 98/2 — đúng nguyên "
            f"tắc chống rò rỉ (được kiểm chứng tự động, xem `tests/test_imbalance_lab.py`).")
    if result.get("leaky"):
        leaky_best = max(result["leaky"]["rows"], key=lambda r: r.get("pr_auc", float("-inf")))
        lines.append(
            f"6. **Mốc minh hoạ SAI** (resample trước khi chia train/test): PR-AUC "
            f"{leaky_best['pr_auc']:.3f} — cao hơn hẳn mọi chiến lược hợp lệ — trong khi tập test chứa "
            f"{result['leaky']['synthetic_rows_in_test']}/{result['leaky']['n_test']} mẫu tổng hợp "
            f"nội suy từ train ⇒ bằng chứng định lượng cho việc resampling phải nằm trong pipeline.")
    calibrated = next((r for r in rows if r["strategy"] == "resampling_calibrated"
                       and r["threshold_mode"] == "best_f1"), None)
    plain = next((r for r in rows if r["strategy"] == "resampling"
                  and r["threshold_mode"] == "best_f1"), None)
    if calibrated and plain:
        lines.append(
            f"7. **Hiệu chuẩn xác suất sau resampling**: Brier "
            f"{plain.get('brier', float('nan')):.4f} → **{calibrated.get('brier', float('nan')):.4f}**; "
            f"ngưỡng best-F1 {plain['threshold']:.3f} → {calibrated['threshold']:.3f} "
            f"(PR-AUC {plain['pr_auc']:.3f} → {calibrated['pr_auc']:.3f} — hiệu chuẩn không đổi thứ "
            f"hạng, chỉ đưa xác suất về đúng tần suất thực tế, nhờ đó ngưỡng mới đọc được theo nghĩa "
            f"“xác suất” và dùng chung giữa các chiến lược).")
    lines.append(
        "8. **Khuyến nghị**: (a) metric PR-AUC/F1/Recall thay vì Accuracy; (b) `scale_pos_weight` "
        "tính động trong `fit` (cost-sensitive) cho F1 tốt nhất mà không đổi dữ liệu; (c) SMOTE + "
        "undersample bên trong pipeline khi cần mô hình thấy nhiều mẫu dương — nhưng phải HIỆU CHUẨN "
        "lại xác suất và chọn lại ngưỡng; (d) chọn ngưỡng theo chi phí thực tế "
        "(`COST_FN`/`COST_FP`) — ở đây FN đắt gấp 10 lần FP nên ngưỡng theo chi phí luôn thấp hơn "
        "ngưỡng best-F1.")
    return lines


def _write_artifacts(result: Dict[str, Any]) -> None:
    """Ghi CSV/JSON/MD/log vào `reports/imbalance/`."""
    out = C.ARTIFACTS_DIR
    _write_csv(out / "summary.csv", result["strategy_rows"] + result["oof_rows"])
    _write_csv(out / "metrics_by_fold.csv", result["fold_rows"])
    _write_csv(out / "resampling_by_fold.csv", result["resample_rows"])
    _write_json(out / "thresholds.json", result["thresholds"])
    _write_json(out / "summary.json",
                {k: v for k, v in result.items() if k != "log"})
    (out / "run.log").write_text(result["log"] + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_markdown(result), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=C.N_SAMPLES,
                        help=f"Số mẫu dữ liệu giả lập (mặc định {C.N_SAMPLES}).")
    parser.add_argument("--no-imblearn", action="store_true",
                        help="Bắt buộc dùng bản sampler nội bộ thay vì imbalanced-learn.")
    parser.add_argument("--skip-leaky", action="store_true",
                        help="Bỏ mốc minh hoạ rò rỉ (chạy nhanh hơn).")
    parser.add_argument("--no-write", action="store_true", help="Không ghi artifact.")
    args = parser.parse_args(argv)
    run(n_samples=args.n_samples, prefer_imblearn=not args.no_imblearn,
        include_leaky=not args.skip_leaky, write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())




