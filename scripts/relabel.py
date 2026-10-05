"""Build a reproducible rule-labelled split and compare conclusions across label definitions.

Rebuilds samples from `data/retail-expanded` with `forecasting.labels`, applies the original split
policy into `data/prepared-rule/`, and compares in-domain and cross-company conclusions between the
original labels and the rule labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from forecasting.config import (DATA_DIR, STRESS_MIN_SIGNALS, STRESS_RULE_NAME, SUFFIX,
                                ensure_utf8_stdio)
from forecasting.data import split_policy
from forecasting.data_loader import load_prepared
from forecasting.evaluation import evaluate_proba
from forecasting.features import build_feature_matrix, extract_labels
from forecasting.labels import SIGNAL_DOCS, label_row
from forecasting.models import DEFAULT_MODEL_ORDER, MODEL_REGISTRY, make_model, predict_proba
from forecasting.validation import grouped_cv

RULE_DIR = DATA_DIR / "prepared-rule"
RESULTS = DATA_DIR.parent / "reports" / "results"
MODELS = list(DEFAULT_MODEL_ORDER)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compress a retail-expanded row into the prepared history-row shape used by `forecasting.data`."""
    from forecasting.data import CANONICAL_VND_FIELDS

    out: Dict[str, Any] = {}
    for key in ("fiscal_year", "fiscal_quarter", "period_start", "period_end",
                "available_on", "currency"):
        if key in row:
            out[key] = row[key]
    for field in CANONICAL_VND_FIELDS:
        out[field + SUFFIX] = row.get(field + SUFFIX)
    if "source_url" in row:
        out["source_url"] = row["source_url"]
    return out


def build_samples(file: Path, min_signals: int) -> Tuple[str, List[Dict[str, Any]]]:
    """Rule-labelled samples whose history holds only quarters published before the target quarter."""
    doc = _read_json(file)
    ticker, rows = doc["ticker"], doc["rows"]
    samples: List[Dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        last_hist, target = rows[idx], rows[idx + 1]
        label, active = label_row(target, rows, idx + 1, min_signals=min_signals)
        samples.append({
            "sample_id": f"{ticker}-{target['fiscal_year']}Q{target['fiscal_quarter']}",
            "ticker": ticker,
            "request": {
                "history": [_compact_row(r) for r in rows[: idx + 1]],
                "as_of": last_hist["available_on"],
                "target_period_start": target["period_start"],
                "target_period_end": target["period_end"],
                "ratios": None,
            },
            "is_distressed": int(label),
            "label_available_on": target["available_on"],
            "label_signals": active,
            "target_source_url": target.get("source_url"),
        })
    return ticker, samples


def _compare_models(min_signals: int, quick: bool) -> Dict[str, Any]:
    """Compare models and the ticker-prior baseline on rule labels, in-domain and cross-company."""
    tr = load_prepared("train", RULE_DIR)
    va = load_prepared("validation", RULE_DIR)
    te = load_prepared("test", RULE_DIR)
    X_tr, X_va, X_te = (build_feature_matrix(s) for s in (tr, va, te))
    y_tr, y_va, y_te = (extract_labels(s) for s in (tr, va, te))

    buckets: Dict[str, List[int]] = {}
    for s, y in zip(tr, y_tr):
        buckets.setdefault(s["ticker"], []).append(int(y))
    prior = {t: float(np.mean(v)) for t, v in buckets.items()}

    def _block(m: Dict[str, Any]) -> Dict[str, Any]:
        at = next(b for b in m["by_threshold"] if b["threshold"] == 0.5)
        return {"auroc": m["auroc"], "ap": m["average_precision"], "f1": at["f1"],
                "macro_f1": at["macro_f1"]}

    names = [m for m in MODELS if m in MODEL_REGISTRY]
    rows: List[Dict[str, Any]] = []
    for name in names:
        model = make_model(name)
        model.fit(X_tr, y_tr)
        rows.append({"system": f"model[{name}]", "source": "train→test",
                     "val": _block(evaluate_proba(y_va, predict_proba(model, X_va))),
                     "test": _block(evaluate_proba(y_te, predict_proba(model, X_te)))})

    # Ticker-prior baseline: no features, only each company's positive rate in train.
    p_va = np.asarray([prior.get(s["ticker"], 0.5) for s in va])
    p_te = np.asarray([prior.get(s["ticker"], 0.5) for s in te])
    rows.append({"system": "baseline[ticker_prior]", "source": "baseline",
                 "val": _block(evaluate_proba(y_va, p_va)),
                 "test": _block(evaluate_proba(y_te, p_te))})

    pool = tr + va + te
    cv = grouped_cv(names, pool, n_splits=len({s["ticker"] for s in pool}))
    cross = [{"system": f"model[{name}] — cross-company", "source": "GroupKFold",
              "test": {"auroc": cv[name]["oof_auroc"], "ap": cv[name]["oof_average_precision"],
                       "f1": (cv[name].get("oof_at_0.5") or {}).get("f1"),
                       "macro_f1": (cv[name].get("oof_at_0.5") or {}).get("macro_f1")}}
             for name in names if name in cv]
    return {"in_domain": rows, "cross_company": cross,
            "label_counts": {"train": int(y_tr.sum()), "validation": int(y_va.sum()),
                             "test": int(y_te.sum())},
            "ticker_prior": prior, "min_signals": min_signals, "quick": quick}


def _markdown_relabel(manifest: Dict[str, Any], comparison: Dict[str, Any]) -> str:
    """Markdown tables for the label-definition sensitivity check."""
    lines = ["# Nhãn tái lập được + kiểm chứng độ nhạy của kết luận", "",
             "## 1. Công thức nhãn thay thế", "",
             f"`is_distressed_rule = 1` nếu quý target có ≥ **{manifest['min_signals']}** tín hiệu "
             "căng thẳng tài chính:", ""]
    lines += [f"- {name}: {doc}" for name, doc in manifest["signals"].items()]
    lines += ["", f"- Mức khớp với nhãn GỐC: **{manifest['agreement_with_original_labels']:.1%}** "
                  f"trên {manifest['n_comparable_samples']} mẫu → hai định nghĩa khác nhau rõ rệt, "
                  "vì thế phải kiểm chứng độ nhạy của kết luận.", "",
              f"- Số mẫu dương tính theo nhãn quy tắc: `{manifest['label_counts']}`", "",
              "## 2. In-domain (chia theo thời gian, cùng công ty)", "",
              "| Hệ thống | Val AUROC | Val AP | Val F1 | Test AUROC | Test AP | Test F1 | Test macro-F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in comparison["in_domain"]:
        v, t = r["val"], r["test"]
        lines.append(f"| {r['system']} | {v['auroc']:.3f} | {v['ap']:.3f} | {v['f1']:.3f} | "
                     f"{t['auroc']:.3f} | {t['ap']:.3f} | {t['f1']:.3f} | {t['macro_f1']:.3f} |")
    lines += ["", "## 3. Cross-company (GroupKFold — công ty chưa từng thấy)", "",
              "| Hệ thống | AUROC | AP | F1 | macro-F1 |", "|---|---:|---:|---:|---:|"]
    for r in comparison["cross_company"]:
        t = r["test"]
        auc = "—" if t["auroc"] is None else f"{t['auroc']:.3f}"
        f1 = "—" if t["f1"] is None else f"{t['f1']:.3f}"
        mf1 = "—" if t["macro_f1"] is None else f"{t['macro_f1']:.3f}"
        lines.append(f"| {r['system']} | {auc} | {t['ap']:.3f} | {f1} | {mf1} |")
    return "\n".join(lines) + "\n"


def run(min_signals: int = STRESS_MIN_SIGNALS, quick: bool = False) -> Dict[str, Any]:
    """Write `data/prepared-rule` and compare conclusions between the two label definitions."""
    from forecasting.data import list_indicator_files, sha256

    all_samples: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, str] = {}
    for file in list_indicator_files():
        ticker, samples = build_samples(file, min_signals)
        all_samples[ticker] = samples
        sources[file.name] = sha256(file)

    train, val, test, purged = split_policy(all_samples)
    for name, arr in (("train", train), ("validation", val), ("test", test), ("purged", purged)):
        _write_json(RULE_DIR / f"{name}.json", arr)

    original: Dict[str, int] = {}
    for name in ("train", "validation", "test", "purged"):
        for s in load_prepared(name):
            original[s["sample_id"]] = int(s["is_distressed"])
    rule_labels = {s["sample_id"]: int(s["is_distressed"])
                   for arr in (train, val, test, purged) for s in arr}
    common = [k for k in original if k in rule_labels]
    agreement = float(np.mean([original[k] == rule_labels[k] for k in common])) if common else None

    counts = {"train": len(train), "validation": len(val), "test": len(test), "purged": len(purged)}
    label_counts = {n: int(sum(s["is_distressed"] for s in a))
                    for n, a in (("train", train), ("validation", val), ("test", test),
                                 ("purged", purged))}
    manifest: Dict[str, Any] = {
        "rule": STRESS_RULE_NAME,
        "min_signals": min_signals,
        "signals": SIGNAL_DOCS,
        "label_definition": ("1 nếu quý target có >= min_signals tín hiệu căng thẳng; chỉ dùng "
                            "dữ liệu công bố ở label_available_on (SAU as_of) nên không rò rỉ "
                            "vào feature."),
        "policy": "giống prepared gốc: 8 quý cuối = test, 4 quý trước = validation, purge 2 quý",
        "counts": counts,
        "label_counts": label_counts,
        "agreement_with_original_labels": agreement,
        "n_comparable_samples": len(common),
        "source_sha256": sources,
        "generated_by": "scripts.relabel",
    }
    _write_json(RULE_DIR / "manifest.json", manifest)

    comparison = _compare_models(min_signals, quick)
    RESULTS.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS / "relabel.json", {"manifest": manifest, "comparison": comparison})
    (RESULTS / "relabel.md").write_text(_markdown_relabel(manifest, comparison), encoding="utf-8")

    logistic = next((r for r in comparison["in_domain"] if r["system"] == "model[logistic]"), None)
    print(f"Relabel: agreement with the original labels {agreement:.1%}; split in {RULE_DIR.name}/; "
          f"test AUROC (logistic) = {logistic['test']['auroc']:.3f}; "
          f"ticker-prior test AUROC = "
          f"{next(r['test']['auroc'] for r in comparison['in_domain'] if r['source'] == 'baseline'):.3f}")
    return {"manifest": manifest, "comparison": comparison}


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-signals", type=int, default=STRESS_MIN_SIGNALS,
                        help="Minimum number of active signals required to set the label to 1.")
    args = parser.parse_args(argv)
    print("=== Reproducible rule labels and label-definition comparison ===")
    run(min_signals=args.min_signals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
