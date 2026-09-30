"""Kiểm chứng độ nhạy của KẾT LUẬN theo định nghĩa NHÃN — 4 định nghĩa, số liệu thật (RQ4).

Lệnh: python -m scripts.label_sensitivity [--no-write]

Vì sao: nhãn gốc `is_distressed` không tái tạo được (khớp tối đa 74,7% — `analysis.json::label_audit`).
Nếu kết luận chỉ đúng với một định nghĩa nhãn thì đồ án không đứng vững. Script chạy CÙNG một quy
trình trên **4 định nghĩa nhãn công khai** (`forecasting/labels.py::LABEL_RULES`):

1. `original` — nhãn gốc trong `data/prepared` (không tái tạo được, dùng làm mốc);
2. `stress_signals` — ≥1 trong 6 tín hiệu căng thẳng của quý target;
3. `altman_z` — Altman Z''-score < 1,1 (công thức 1968/2000, dùng đúng 16 chỉ tiêu của đồ án);
4. `forward_4q` — có ≥1 quý trong **4 quý TỚI** chạm ngưỡng tín hiệu (sự kiện sắp xảy ra).

Mỗi định nghĩa được đo: IR/cân bằng lớp, mức khớp nhãn gốc, và hai chỉ số quyết định — in-domain
(test) và cross-company OOF (GroupKFold) của mô hình so với baseline `ticker_prior`. Kết luận chỉ
được coi là ỔN ĐỊNH nếu nó xuất hiện ở ≥2 định nghĩa.

Ghi ra `reports/results/label_sensitivity.{json,md}`. Mô hình cố định (Random Forest, cấu hình mặc
định trong `HYPERPARAMS`) để mọi định nghĩa được so sánh công bằng; test không dùng để chọn cấu hình.
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
    """Sample cho một định nghĩa nhãn: lịch sử = quý TRƯỚC target (không chứa dữ liệu quý target)."""
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
    """Baseline "nhớ mặt công ty": tỉ lệ nhãn 1 của chính công ty trong train."""
    prior: Dict[str, float] = {}
    for ticker in sorted({str(s["ticker"]) for s in train}):
        labels = [int(s[TARGET]) for s in train if str(s["ticker"]) == ticker]
        prior[ticker] = float(np.mean(labels))
    global_prior = float(np.mean([int(s[TARGET]) for s in train])) if train else 0.5
    return np.array([prior.get(str(s["ticker"]), global_prior) for s in test])


def evaluate_rule(rule: str, min_signals: int = STRESS_MIN_SIGNALS) -> Dict[str, Any]:
    """Dựng split theo một định nghĩa nhãn rồi đo in-domain + cross-company + mức khớp nhãn gốc."""
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
    """Kết luận tự động: định nghĩa nhãn nào ủng hộ luận điểm trung tâm của đồ án."""
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
    """Sinh `reports/results/label_sensitivity.md` (mọi số đọc từ JSON)."""
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
    """Chạy kiểm chứng trên nhiều định nghĩa nhãn, ghi `label_sensitivity.{json,md}`."""
    import json

    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    selected = [r for r in (rules or list(LABEL_RULES)) if r != "original"]
    rows: List[Dict[str, Any]] = []

    if rules is None or "original" in rules:
        # Mốc: nhãn gốc, đo trên chính split có sẵn (không dựng lại để tránh sai lệch).
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
    print(f"Nhãn: {len(rows)} định nghĩa; luận điểm 'mô hình không vượt ticker-prior' đúng ở "
          f"{len(stable)}/{len(reproduced)} định nghĩa tái lập được")
    return summary


def main(argv=None) -> int:
    """CLI: `python -m scripts.label_sensitivity [--rules a,b] [--min-signals N] [--no-write]`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", default="",
                        help="Danh sách định nghĩa nhãn (mặc định: original + mọi định nghĩa tái lập được).")
    parser.add_argument("--min-signals", type=int, default=STRESS_MIN_SIGNALS)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    rules = [r for r in args.rules.split(",") if r] or None
    print("=== Kiểm chứng độ nhạy theo định nghĩa nhãn (RQ4) ===")
    run(write=not args.no_write, rules=rules, min_signals=args.min_signals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
