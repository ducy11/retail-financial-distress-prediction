"""Check that conclusions hold across four public label definitions rather than a single one.

Runs the same protocol on the original labels plus a stress-signal rule, the Altman Z'' rule and a
forward event rule, reporting class balance, agreement and in-domain and cross-company metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from forecasting.config import (GROUP_KEY, RESULTS_DIR, STRESS_MIN_SIGNALS, TARGET, ensure_dirs,
                                ensure_utf8_stdio)
from forecasting.data import list_indicator_files, split_policy
from forecasting.data_loader import load_prepared
from forecasting.eda import distribution_of, markdown_table
from forecasting.evaluation import evaluate_proba
from forecasting.features import build_feature_matrix, extract_labels
from forecasting.labels import LABEL_RULES, label_by_rule
from forecasting.models import make_model, predict_proba

MODEL_NAME = "random_forest"


def _samples_for_rule(file: Path, rule: str, min_signals: int) -> List[Dict[str, Any]]:
    """Samples for one label definition, with history limited to quarters before the target quarter."""
    from scripts.relabel import _compact_row, _read_json

    doc = _read_json(file)
    ticker, rows = doc["ticker"], doc["rows"]
    out: List[Dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        target = rows[idx + 1]
        label, active = label_by_rule(rule, target, rows, idx + 1, min_signals=min_signals)
        out.append({
            "sample_id": f"{ticker}-{target['fiscal_year']}Q{target['fiscal_quarter']}",
            "ticker": ticker,
            "request": {
                "history": [_compact_row(row) for row in rows[: idx + 1]],
                "as_of": rows[idx].get("available_on"),
                "target_period_start": target.get("period_start"),
                "target_period_end": target.get("period_end"),
                "signal_rule": rule,
                "signals": active,
            },
            TARGET: int(label),
            "label_available_on": target.get("available_on"),
        })
    return out


def _ticker_prior(train: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> np.ndarray:
    """Company-memorising baseline: each company's own positive rate in train."""
    prior: Dict[str, float] = {}
    for ticker in sorted({str(s["ticker"]) for s in train}):
        labels = [int(s[TARGET]) for s in train if str(s["ticker"]) == ticker]
        prior[ticker] = float(np.mean(labels))
    global_prior = float(np.mean([int(s[TARGET]) for s in train])) if train else 0.5
    return np.array([prior.get(str(s["ticker"]), global_prior) for s in test])


def evaluate_rule(rule: str, min_signals: int = STRESS_MIN_SIGNALS) -> Dict[str, Any]:
    """Rebuild the split for one label definition and measure in-domain, cross-company and agreement."""
    per_ticker = {file.stem.split("-")[0]: _samples_for_rule(file, rule, min_signals)
                  for file in list_indicator_files()}
    train, val, test, purged = split_policy(per_ticker)
    y_train, y_val, y_test = extract_labels(train), extract_labels(val), extract_labels(test)
    X_train, X_test = build_feature_matrix(train), build_feature_matrix(test)

    model = make_model(MODEL_NAME)
    model.fit(X_train, y_train)
    in_domain = evaluate_proba(y_test, predict_proba(model, X_test)) if len(set(y_test.tolist())) > 1 else {}
    prior_metrics = evaluate_proba(y_test, _ticker_prior(train, test)) if len(set(y_test.tolist())) > 1 \
        else {}

    pool = train + val
    groups = np.asarray([s[GROUP_KEY] for s in pool])
    X_pool, y_pool = build_feature_matrix(pool), extract_labels(pool)
    proba_oof = np.full(len(y_pool), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=4).split(X_pool, y_pool, groups=groups):
        if len(set(y_pool[train_idx].tolist())) < 2 or len(test_idx) < 4:
            continue
        fold_model = make_model(MODEL_NAME)
        fold_model.fit(X_pool[train_idx], y_pool[train_idx])
        proba_oof[test_idx] = predict_proba(fold_model, X_pool[test_idx])
    mask = np.isfinite(proba_oof)
    cross: Dict[str, Any] = {}
    if mask.sum() and len(set(y_pool[mask].tolist())) > 1:
        cross = {"n_oof": int(mask.sum()),
                 "oof_average_precision": float(average_precision_score(y_pool[mask], proba_oof[mask])),
                 "oof_auroc": float(roc_auc_score(y_pool[mask], proba_oof[mask]))}

    original = {s["sample_id"]: int(s[TARGET]) for name in ("train", "validation", "test", "purged")
                for s in load_prepared(name)}
    new_labels = {s["sample_id"]: int(s[TARGET]) for arr in (train, val, test, purged) for s in arr}
    common = [key for key in original if key in new_labels]
    agreement = float(np.mean([original[k] == new_labels[k] for k in common])) if common else None
    model_auroc, prior_auroc = in_domain.get("auroc"), prior_metrics.get("auroc")
    return {
        "rule": rule,
        "description": LABEL_RULES[rule]["description"],
        "counts": {"train": len(train), "validation": len(val), "test": len(test),
                   "purged": len(purged)},
        "positive_rate": {"train": float(np.mean(y_train)) if len(y_train) else None,
                          "validation": float(np.mean(y_val)) if len(y_val) else None,
                          "test": float(np.mean(y_test)) if len(y_test) else None},
        "imbalance_train": distribution_of(y_train.tolist()),
        "imbalance_test": distribution_of(y_test.tolist()),
        "agreement_with_original_labels": agreement,
        "n_comparable_samples": len(common),
        "in_domain": {
            "model_auroc": model_auroc, "model_average_precision": in_domain.get("average_precision"),
            "ticker_prior_auroc": prior_auroc,
            "ticker_prior_average_precision": prior_metrics.get("average_precision"),
            "delta_auroc_model_minus_prior": (model_auroc - prior_auroc)
            if (model_auroc is not None and prior_auroc is not None) else None,
        },
        "cross_company": cross,
    }


def conclusions(rows: List[Dict[str, Any]]) -> List[str]:
    """Automatic conclusions: which label definitions support the project's central claim."""
    lines: List[str] = []
    reproduced = [r for r in rows if r["rule"] != "original"]
    if not reproduced:
        return lines
    prior_wins = [r for r in reproduced
                  if (r["in_domain"].get("delta_auroc_model_minus_prior") or 0.0) <= 0.01]
    lines.append(
        f"**Số định nghĩa nhãn TÁI LẬP ĐƯỢC đã kiểm chứng:** {len(reproduced)} "
        f"({', '.join(r['rule'] for r in reproduced)}).")
    lines.append(
        f"**Luận điểm \"mô hình không vượt baseline ticker-prior\" xuất hiện ở "
        f"{len(prior_wins)}/{len(reproduced)} định nghĩa** ⇒ "
        + ("kết luận **ỔN ĐỊNH theo định nghĩa nhãn** — đây là bằng chứng mạnh nhất của đồ án."
           if len(prior_wins) >= 2 else
           "kết luận CHƯA ổn định (chỉ đúng ở một định nghĩa) ⇒ phải nêu rõ trong phần hạn chế."))
    for row in reproduced:
        cross = row.get("cross_company") or {}
        lines.append(
            f"**`{row['rule']}`:** tỉ lệ dương test = {(row['positive_rate'].get('test') or 0):.1%}, "
            f"IR train = {(row['imbalance_train'].get('imbalance_ratio') or 0):.2f}; khớp nhãn gốc = "
            f"{(row.get('agreement_with_original_labels') or 0):.1%}; in-domain AUROC (mô hình / "
            f"ticker-prior) = {(row['in_domain'].get('model_auroc') or 0):.3f} / "
            f"{(row['in_domain'].get('ticker_prior_auroc') or 0):.3f}; cross-company OOF AP = "
            f"{(cross.get('oof_average_precision') or 0):.3f}, AUROC = {(cross.get('oof_auroc') or 0):.3f}.")
    lines.append(
        "**Đọc kết quả:** cột khớp nhãn gốc cho biết định nghĩa mới có đo cùng khái niệm với nhãn gốc "
        "hay không; cột cross-company cho biết mô hình còn giữ được bao nhiêu khi công ty bị giữ trọn ra "
        "ngoài. Một định nghĩa chỉ đáng tin khi cả hai hợp lý (khớp > ~65% và cross-company ≈ 0,9).")
    return lines


def markdown_label_sensitivity(summary: Dict[str, Any]) -> str:
    """Render `reports/results/label_sensitivity.md` from the JSON payload."""
    rows = summary["rows"]
    lines = ["# Kiểm chứng độ nhạy của kết luận theo ĐỊNH NGHĨA NHÃN (RQ4)", "",
             f"- Mô hình cố định: **{summary['model']}** (cấu hình mặc định); mọi định nghĩa dùng cùng "
             f"chính sách split và cùng giao thức đo.",
             f"- Số định nghĩa: **{len(rows)}**, trong đó "
             f"{len([r for r in rows if r['rule'] != 'original'])} định nghĩa tái lập được từ dữ liệu "
             f"công bố.", "", "## 1. Nhận xét tự động", ""]
    lines += [f"{i + 1}. {text}" for i, text in enumerate(summary["conclusions"])]
    lines += ["", "## 2. Bảng so sánh các định nghĩa nhãn", "",
              markdown_table(
                  ["Định nghĩa", "Dương (train)", "IR train", "Dương (test)", "Khớp nhãn gốc",
                   "AUROC test (mô hình)", "AUROC test (ticker-prior)", "Cross-company AP",
                   "Cross-company AUROC"],
                  [[r["rule"], f"{(r['positive_rate'].get('train') or 0):.1%}",
                    f"{(r['imbalance_train'].get('imbalance_ratio') or 0):.2f}",
                    f"{(r['positive_rate'].get('test') or 0):.1%}",
                    f"{(r.get('agreement_with_original_labels') or 0):.1%}",
                    f"{(r['in_domain'].get('model_auroc') or 0):.3f}",
                    f"{(r['in_domain'].get('ticker_prior_auroc') or 0):.3f}",
                    f"{((r.get('cross_company') or {}).get('oof_average_precision') or 0):.3f}",
                    f"{((r.get('cross_company') or {}).get('oof_auroc') or 0):.3f}"]
                   for r in rows],
                  ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
              "", "**Mô tả từng định nghĩa:**", ""]
    for r in rows:
        lines.append(f"- `{r['rule']}`: {r['description']}")
    lines += ["",
              "> Đọc bảng: nhãn gốc không tái tạo được nên cột **Khớp nhãn gốc** đo xem định nghĩa công "
              "khai có đo cùng khái niệm hay không (mốc cao nhất trước đây là 74,7% với `stress_signals`). "
              "Cột **Cross-company** là phép thử khắt khe nhất: công ty bị giữ trọn ra ngoài train. Nếu "
              "kết luận \"mô hình ≈ baseline nhớ mặt công ty\" lặp lại ở nhiều định nghĩa thì kết luận "
              "không phụ thuộc vào cách gán nhãn.", ""]
    return "\n".join(lines) + "\n"


def run(write: bool = True, rules: List[str] | None = None,
        min_signals: int = STRESS_MIN_SIGNALS, out_dir: Path | None = None) -> Dict[str, Any]:
    """Run the check across label definitions and write `label_sensitivity.{json,md}`."""
    import json

    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    selected = [r for r in (rules or list(LABEL_RULES)) if r != "original"]
    rows: List[Dict[str, Any]] = []

    if rules is None or "original" in rules:
        # Original labels are measured on the existing split, never rebuilt, to avoid drift.
        from forecasting.significance import compare_systems
        from forecasting.validation import grouped_cv

        splits = {n: load_prepared(n) for n in ("train", "validation", "test", "purged")}
        y_test = extract_labels(splits["test"])
        model = make_model(MODEL_NAME)
        model.fit(build_feature_matrix(splits["train"]), extract_labels(splits["train"]))
        proba = predict_proba(model, build_feature_matrix(splits["test"]))
        prior_proba = _ticker_prior(splits["train"], splits["test"])
        in_domain, prior = evaluate_proba(y_test, proba), evaluate_proba(y_test, prior_proba)
        oof = grouped_cv([MODEL_NAME], splits["train"] + splits["validation"], n_splits=4)[MODEL_NAME]
        comparison = compare_systems(y_test, {"model": proba, "ticker_prior": prior_proba},
                                    n_boot=1000)["pairs"][0]["delong_auroc"]
        rows.append({
            "rule": "original",
            "description": ("nhãn gốc `is_distressed` trong data/prepared (KHÔNG tái tạo được — "
                            "dùng làm mốc tham chiếu)"),
            "counts": {n: len(splits[n]) for n in ("train", "validation", "test", "purged")},
            "positive_rate": {n: float(np.mean(extract_labels(splits[n]))) for n in splits},
            "imbalance_train": distribution_of(extract_labels(splits["train"]).tolist()),
            "imbalance_test": distribution_of(y_test.tolist()),
            "agreement_with_original_labels": 1.0,
            "n_comparable_samples": int(len(y_test)),
            "in_domain": {"model_auroc": in_domain.get("auroc"),
                          "model_average_precision": in_domain.get("average_precision"),
                          "ticker_prior_auroc": prior.get("auroc"),
                          "ticker_prior_average_precision": prior.get("average_precision"),
                          "delta_auroc_model_minus_prior": (in_domain.get("auroc") or 0.0)
                                                            - (prior.get("auroc") or 0.0)},
            "cross_company": {"n_oof": oof.get("n_oof"),
                              "oof_average_precision": oof.get("oof_average_precision"),
                              "oof_auroc": oof.get("oof_auroc")},
            "delong_model_vs_prior": comparison,
        })

    for rule in selected:
        rows.append(evaluate_rule(rule, min_signals=min_signals))

    summary: Dict[str, Any] = {
        "model": MODEL_NAME,
        "policy": ("mỗi định nghĩa nhãn dựng lại split bằng forecasting.data.split_policy "
                   "(train/validation/test/purged) rồi đo độc lập"),
        "min_signals": min_signals,
        "rows": rows,
        "conclusions": conclusions(rows),
    }
    if write:
        (out / "label_sensitivity.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "label_sensitivity.md").write_text(markdown_label_sensitivity(summary),
                                                 encoding="utf-8")
    reproduced = [r for r in rows if r["rule"] != "original"]
    stable = [r for r in reproduced
              if (r["in_domain"].get("delta_auroc_model_minus_prior") or 0.0) <= 0.01]
    print(f"Labels: {len(rows)} definitions; the claim 'the model does not beat ticker_prior' holds in "
          f"{len(stable)}/{len(reproduced)} reproducible definitions")
    return summary


def main(argv=None) -> int:
    """Command-line entry point for `scripts.label_sensitivity`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default="",
                        help="Label definitions to run (default: original plus every reproducible one).")
    parser.add_argument("--min-signals", type=int, default=STRESS_MIN_SIGNALS)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    rules = [r for r in args.rules.split(",") if r] or None
    print("=== Sensitivity of the conclusions to the label definition (RQ4) ===")
    run(write=not args.no_write, rules=rules, min_signals=args.min_signals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
