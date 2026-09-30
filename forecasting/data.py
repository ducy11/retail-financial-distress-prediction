"""Tái tạo split dữ liệu prepared từ retail-expanded.

Lệnh: python -m forecasting.data [--force]

Chính sách (đồng bộ manifest hiện có), theo thứ tự thời gian mẫu từng công ty:
- 8 quý cuối → test.
- 4 quý liền trước → validation.
- 1 quý ngay trước validation và 1 quý ngay sau validation → purged
  (dải biên chống rò rỉ nhãn giữa train và validation/test).
- Phần còn lại → train.

Sample đặt tên theo quý target (`TICKER-<fy>Q<q>`); lịch sử là toàn bộ quý đã
công bố trước quý target (nén bỏ provenance `sources`, giữ `source_url`).

Nhãn `is_distressed` do pipeline prepare_sec gốc sinh ra (không suy ra lại được
từ 16 chỉ tiêu): khi tái tạo, giữ nguyên nhãn theo sample_id từ prepared có sẵn;
heuristic `net_income_vnd < 0` chỉ dùng cho mẫu hoàn toàn mới.

Mặc định *không* chạy (prepared đã sẵn trong repo); dùng --force để tái tạo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import (
    MANIFEST_FILE,
    PREPARED_DIR,
    RETAIL_DIR,
    SUFFIX,
    ensure_dirs,
    ensure_utf8_stdio,
)
from .data_loader import to_float

QUARTERS_TEST = 8
QUARTERS_VALIDATION = 4
VND_PER_USD = 25_000  # demo

#: Thứ tự khoá chỉ tiêu *_vnd trong history row của prepared (chuẩn hoá gốc).
CANONICAL_VND_FIELDS = [
    "revenue", "cost_of_sales", "inventory", "selling_general_admin",
    "operating_cash_flow", "total_assets", "cash_and_equivalents",
    "operating_income", "current_assets", "current_liabilities", "net_income",
    "liabilities", "stockholders_equity", "retained_earnings", "receivables",
    "short_term_investments",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def list_indicator_files() -> List[Path]:
    """File 16-indicators theo thứ tự công ty trong corpus.json (WMT, HD, ...)."""
    corpus_path = RETAIL_DIR / "corpus.json"
    if corpus_path.exists():
        order = list(_read_json(corpus_path).get("companies", {}).keys())
    else:
        order = sorted(p.name.split("-")[0] for p in RETAIL_DIR.glob("*-16-indicators-vnd.json"))
    files = []
    for ticker in order:
        p = RETAIL_DIR / f"{ticker}-16-indicators-vnd.json"
        if p.exists():
            files.append(p)
    return files


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Nén một row retail-expanded về dạng history row của prepared.

    Thứ tự khoá như file gốc: fiscal_year, fiscal_quarter, period_start,
    period_end, available_on, currency, 16 chỉ tiêu *_vnd (thứ tự chuẩn),
    source_url. Ticker nằm ở sample, không nằm trong row.
    """
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


def _load_existing_labels() -> Dict[str, Dict[str, Any]]:
    """Nhãn gốc theo sample_id từ các file prepared hiện có (nếu có)."""
    labels: Dict[str, Dict[str, Any]] = {}
    for name in ("train", "validation", "test", "purged"):
        path = PREPARED_DIR / f"{name}.json"
        if not path.exists():
            continue
        for s in _read_json(path):
            labels[s["sample_id"]] = {
                "is_distressed": s.get("is_distressed"),
                "label_available_on": s.get("label_available_on"),
                "target_source_url": s.get("target_source_url"),
            }
    return labels


def _build_samples(file: Path,
                   existing: Dict[str, Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Tạo sample: lịch sử đủ trước quý target; nhãn/URL giữ từ prepared cũ nếu có."""
    doc = _read_json(file)
    ticker = doc["ticker"]
    rows = doc["rows"]  # quý theo thứ tự thời gian
    samples: List[Dict[str, Any]] = []
    for idx in range(len(rows) - 1):
        last_hist, target = rows[idx], rows[idx + 1]
        sample_id = f"{ticker}-{target['fiscal_year']}Q{target['fiscal_quarter']}"
        prev = existing.get(sample_id)
        if prev is not None and prev.get("is_distressed") is not None:
            label = prev["is_distressed"]
            label_date = prev["label_available_on"] or target["available_on"]
            target_url = prev["target_source_url"] or target.get("source_url")
        else:
            # Heuristic chỉ cho dữ liệu hoàn toàn mới (không có nhãn gốc).
            label = 1 if to_float(target.get("net_income_vnd")) < 0 else 0
            label_date = target["available_on"]
            target_url = target.get("source_url")
        samples.append({
            "sample_id": sample_id,
            "ticker": ticker,
            "request": {
                "history": [_compact_row(r) for r in rows[: idx + 1]],
                "as_of": last_hist["available_on"],
                "target_period_start": target["period_start"],
                "target_period_end": target["period_end"],
                "ratios": None,
            },
            "is_distressed": int(label),
            "label_available_on": label_date,
            "target_source_url": target_url,
        })
    return ticker, samples



def split_policy(all_samples: Dict[str, List[Dict[str, Any]]]):
    """Chia theo công ty: [train][purged 1][validation 4][purged 1][test 8]."""
    train, validation, test, purged = [], [], [], []
    for ticker, samples in all_samples.items():
        n = len(samples)
        need = QUARTERS_TEST + QUARTERS_VALIDATION + 2
        if n <= need:
            train.extend(samples)  # quá ít dữ liệu: không tạo test/val/purged
            continue
        train.extend(samples[: n - need])
        purged.append(samples[n - need])                       # trước validation
        validation.extend(samples[n - need + 1: n - QUARTERS_TEST - 1])
        purged.append(samples[n - QUARTERS_TEST - 1])          # sau validation
        test.extend(samples[n - QUARTERS_TEST:])
    return train, validation, test, purged


def build_manifest(all_out: Dict[str, List[Dict[str, Any]]],
                   sources: Dict[str, str],
                   old: Dict[str, Any],
                   test_ranges_: Dict[str, Any],
                   split_sha: Dict[str, str]) -> Dict[str, Any]:
    """Manifest theo đúng thứ tự khoá của file gốc (để byte-identical)."""
    counts = {name: len(arr) for name, arr in all_out.items()}
    labels = {name: [s["is_distressed"] for s in arr] for name, arr in all_out.items()}
    m: Dict[str, Any] = {
        "policy": "8 quý cuối mỗi công ty làm test; 4 quý validation; purge ngày công bố toàn cục",
        "test_is_last_two_fiscal_years_per_company": True,
        "target": "is_distressed",
        "label_counts": {
            name: {"distressed": sum(labels[name]), "total": counts[name] or 0}
            for name in all_out
        },
    }
    for key in ("validation_fit_before", "final_fit_before"):
        if key in old:
            m[key] = old[key]
    m["source_sha256"] = sources
    m["currency"] = "VND"
    m["demo_vnd_per_usd"] = str(VND_PER_USD)
    m["counts"] = counts
    m["test_ranges"] = test_ranges_
    m["split_sha256"] = split_sha
    return m


def test_ranges(test: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Khoảng target period của test theo công ty: from = start đầu, to = end cuối."""
    by_ticker: Dict[str, List[Tuple[str, str]]] = {}
    for s in test:
        by_ticker.setdefault(s["ticker"], []).append(
            (s["request"]["target_period_start"], s["request"]["target_period_end"]))
    return {
        ticker: {"from": min(p[0] for p in periods),
                 "to": max(p[1] for p in periods),
                 "count": len(periods)}
        for ticker, periods in by_ticker.items()
    }


def _write_json(path: Path, obj: Any) -> None:
    # newline="\n": ghi LF (giống file gốc); thêm "\n" cuối file để sha256 khớp.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run(force: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    if not force:
        return {"skipped": True,
                "message": "Đã có prepared; dùng --force để tái tạo từ retail-expanded."}

    existing = _load_existing_labels()
    all_samples: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, str] = {}
    corpus_path = RETAIL_DIR / "corpus.json"
    if corpus_path.exists():
        sources["corpus.json"] = sha256(corpus_path)
    for file in list_indicator_files():
        ticker, samples = _build_samples(file, existing)
        all_samples[ticker] = samples
        sources[file.name] = sha256(file)

    train, validation, test, purged = split_policy(all_samples)
    splits = {"train": train, "validation": validation, "test": test}
    all_out = {**splits, "purged": purged}

    for name, arr in all_out.items():
        _write_json(PREPARED_DIR / f"{name}.json", arr)

    old = _read_json(MANIFEST_FILE) if MANIFEST_FILE.exists() else {}
    manifest = build_manifest(
        all_out, sources, old,
        test_ranges_=test_ranges(test),
        split_sha={name: sha256(PREPARED_DIR / f"{name}.json") for name in all_out},
    )
    _write_json(MANIFEST_FILE, manifest)
    return {"skipped": False, **{name: len(v) for name, v in all_out.items()}}


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Tái tạo split từ retail-expanded (mặc định bỏ qua nếu có sẵn).")
    args = parser.parse_args(argv)
    result = run(force=args.force)
    if result.get("skipped"):
        print(result["message"])
    else:
        print("Đã tái tạo:", {k: v for k, v in result.items() if k != "skipped"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

