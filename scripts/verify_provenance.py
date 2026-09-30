"""Kiểm chứng NGUỒN GỐC dữ liệu — chứng minh số liệu là THẬT, không bịa.

Lệnh: python -m scripts.verify_provenance [--quick]

Vì sao cần (và khác gì `scripts/audit_data.py`): `audit_data` kiểm tra tính nhất quán **bên trong**
repo (số liệu trong báo cáo ↔ artifact, split ↔ manifest, đẳng thức kế toán…). Script này đi **ra
ngoài** bộ dữ liệu prepared: mở lại snapshot SEC thô (`data/sec/raw/*-companyfacts.json`, ~129 MB —
đúng file đã tải từ API công khai của SEC) và xác nhận từng con số bằng ba phép kiểm độc lập:

1. **HASH**: SHA-256 của mọi file raw phải khớp `data/sec/downloads.json` (chống sửa tay dữ liệu).
2. **FACT TỒN TẠI THẬT**: với **mọi** fact ghi trong `sources` của
   `data/retail-expanded/*-16-indicators-vnd.json` (tag, start/end, val, accn, form), phải tìm thấy
   bản ghi y hệt trong companyfacts của SEC ⇒ con số trong repo chính là con số SEC đã công bố.
3. **QUY ĐỔI**: `*_vnd` phải bằng `val × 25.000` (đúng `fx_policy.vnd_per_usd` khai trong file),
   và chỉ tiêu không có fact (`method = "absent"`) phải là `null` (không được bịa số).

Kết quả ghi ra `reports/results/provenance.{json,md}`; exit code ≠ 0 nếu có bất kỳ mismatch nào.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from forecasting.config import BASE_FIELDS, DATA_DIR, RESULTS_DIR, ensure_utf8_stdio

SEC_DIR = DATA_DIR / "sec"
RAW_DIR = SEC_DIR / "raw"
DOWNLOADS = SEC_DIR / "downloads.json"
RETAIL_DIR = DATA_DIR / "retail-expanded"
VND_RATE_KEY = "vnd_per_usd"


def sha256_of(path: Path) -> str:
    """SHA-256 của một file (đọc theo khối để không nạp cả file lớn vào RAM)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_downloads(path: Path = DOWNLOADS) -> Dict[str, Dict[str, Any]]:
    """Đọc registry tải SEC (CIK, URL, sha256, thời điểm tải)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hashes(downloads: Dict[str, Dict[str, Any]], raw_dir: Path = RAW_DIR
                 ) -> List[Dict[str, str]]:
    """Đối chiếu SHA-256 từng file raw với registry; trả danh sách mismatch."""
    mismatches: List[Dict[str, str]] = []
    for ticker, meta in sorted(downloads.items()):
        raw = raw_dir / f"{ticker}-companyfacts.json"
        if not raw.exists():
            mismatches.append({"ticker": ticker, "reason": "thiếu file raw"})
            continue
        actual = sha256_of(raw)
        if actual.lower() != str(meta.get("sha256", "")).lower():
            mismatches.append({"ticker": ticker, "reason": "sha256 khác registry",
                               "expected": str(meta.get("sha256")), "actual": actual})
    return mismatches


def iter_facts(raw: Dict[str, Any], tag: str) -> Iterable[Dict[str, Any]]:
    """Mọi bản ghi của một tag trong companyfacts (mọi taxonomy, mọi unit)."""
    for _taxonomy, tags in (raw.get("facts") or {}).items():
        entry = tags.get(tag)
        if not entry:
            continue
        for series in (entry.get("units") or {}).values():
            for item in series or []:
                yield item


def fact_present(raw: Dict[str, Any], fact: Dict[str, Any]) -> bool:
    """True nếu có bản ghi SEC khớp (tag, end, val, accn, form[, start]) với fact đã ghi."""
    for item in iter_facts(raw, str(fact.get("tag"))):
        if item.get("end") != fact.get("end") or item.get("val") != fact.get("val"):
            continue
        if fact.get("accn") and item.get("accn") != fact.get("accn"):
            continue
        if fact.get("form") and item.get("form") != fact.get("form"):
            continue
        if fact.get("start") and item.get("start") != fact.get("start"):
            continue
        return True
    return False


def derived_vnd(facts: List[Dict[str, Any]], rate: int, method: str) -> int | None:
    """Giá trị VND mà file phải có, tính TỪ các fact SEC theo đúng `method` đã khai.

    Quy tắc (đối chiếu `data/retail-expanded/*-16-indicators-vnd.json`):
    - `instant` / `reported_quarter` / `reported_first_quarter`: một fact ⇒ `val × rate`;
    - `current_ytd_minus_previous_ytd`: hai fact luỹ kế ⇒ `(val_đầu − val_trước) × rate`;
    - method khác (chưa hỗ trợ) ⇒ `None` (báo riêng, KHÔNG coi là lỗi dữ liệu).
    """
    values = [int(f.get("val") or 0) for f in facts]
    if not values:
        return None
    if method == "current_ytd_minus_previous_ytd":
        return (values[0] - values[1]) * rate if len(values) >= 2 else None
    if method in ("instant", "reported_quarter", "reported_first_quarter"):
        return values[0] * rate if len(values) == 1 else sum(values) * rate
    return None


def verify_document(ticker: str, doc: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """Kiểm fact thật + quy đổi VND cho MỘT file retail-expanded."""
    rate = int(doc.get("fx_policy", {}).get(VND_RATE_KEY, 0) or 0)
    rows = doc.get("rows") or []
    checked = missing_fact = bad_vnd = absent_filled = absent_checked = unsupported = 0
    examples: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        sources = row.get("sources") or {}
        label = f"{ticker}-{row.get('fiscal_year')}Q{row.get('fiscal_quarter')}"
        for field in BASE_FIELDS:
            raw_value = row.get(field + "_vnd")
            source = sources.get(field) or {}
            facts = source.get("facts") or []
            method = str(source.get("method") or "")
            if not facts:  # `method = "absent"`: thiếu ở SEC ⇒ phải là null, KHÔNG được bịa
                absent_checked += 1
                if raw_value not in (None, ""):
                    absent_filled += 1
                    if len(examples) < 6:
                        examples.append({"row": index, "sample": label, "field": field,
                                         "problem": "không có fact nhưng vẫn có giá trị"})
                continue
            checked += 1
            for fact in facts:
                if not fact_present(raw, fact):
                    missing_fact += 1
                    if len(examples) < 6:
                        examples.append({"row": index, "sample": label, "field": field,
                                         "problem": "fact không tồn tại trong companyfacts SEC",
                                         "tag": fact.get("tag"), "end": fact.get("end"),
                                         "val": fact.get("val"), "accn": fact.get("accn")})
            expected = derived_vnd(facts, rate, method)
            if expected is None:
                unsupported += 1
                continue
            try:
                actual = int(raw_value)
            except (TypeError, ValueError):
                actual = None
            if actual != expected:
                bad_vnd += 1
                if len(examples) < 6:
                    examples.append({"row": index, "sample": label, "field": field,
                                     "problem": "quy đổi VND sai", "method": method,
                                     "expected_vnd": expected, "actual_vnd": raw_value})
    return {"ticker": ticker, "n_rows": len(rows), "fx_rate": rate,
            "n_cells": len(rows) * len(BASE_FIELDS), "n_absent_cells": absent_checked,
            "n_facts_checked": checked, "n_facts_missing_in_sec": missing_fact,
            "n_vnd_mismatch": bad_vnd, "n_absent_but_filled": absent_filled,
            "n_unsupported_method": unsupported, "examples": examples}



def run(quick: bool = False) -> Dict[str, Any]:
    """Chạy toàn bộ kiểm chứng; ghi `reports/results/provenance.{json,md}`."""
    ensure_utf8_stdio()
    downloads = load_downloads()
    hash_mismatches = check_hashes(downloads)
    docs = sorted(RETAIL_DIR.glob("*-16-indicators-vnd.json"))
    per_company: List[Dict[str, Any]] = []
    for path in docs:
        ticker = path.name.split("-")[0]
        doc = json.loads(path.read_text(encoding="utf-8"))
        raw_path = RAW_DIR / f"{ticker}-companyfacts.json"
        if not raw_path.exists():
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        per_company.append(verify_document(ticker, doc, raw))
        if quick and len(per_company) >= 2:
            break
    totals = {
        "n_raw_files_hashed": len(downloads),
        "n_hash_mismatch": len(hash_mismatches),
        "n_companies_checked": len(per_company),
        "n_cells": sum(c["n_cells"] for c in per_company),
        "n_absent_cells": sum(c["n_absent_cells"] for c in per_company),
        "n_facts_checked": sum(c["n_facts_checked"] for c in per_company),
        "n_facts_missing_in_sec": sum(c["n_facts_missing_in_sec"] for c in per_company),
        "n_vnd_mismatch": sum(c["n_vnd_mismatch"] for c in per_company),
        "n_absent_but_filled": sum(c["n_absent_but_filled"] for c in per_company),
        "n_unsupported_method": sum(c["n_unsupported_method"] for c in per_company),
    }
    totals["ok"] = (totals["n_hash_mismatch"] == 0 and totals["n_facts_missing_in_sec"] == 0
                    and totals["n_vnd_mismatch"] == 0 and totals["n_absent_but_filled"] == 0
                    and totals["n_unsupported_method"] == 0)
    result = {"downloads_registry": str(DOWNLOADS.relative_to(DATA_DIR.parent)),
              "raw_dir": str(RAW_DIR.relative_to(DATA_DIR.parent)),
              "quick": bool(quick), "totals": totals,
              "hash_mismatches": hash_mismatches, "per_company": per_company,
              "note": ("Nguồn SEC công khai: https://data.sec.gov/api/xbrl/companyfacts/. "
                       "Mọi fact ghi trong `sources` phải tồn tại trong file raw (đã hash) — "
                       "đây là bằng chứng dữ liệu THẬT, không sinh/sửa tay.")}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "provenance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (RESULTS_DIR / "provenance.md").write_text(markdown_provenance(result), encoding="utf-8")
    print(f"Kiểm chứng nguồn gốc: {totals['n_raw_files_hashed']} file SEC đã hash "
          f"({totals['n_hash_mismatch']} lệch); {totals['n_facts_checked']} fact đối chiếu raw "
          f"({totals['n_facts_missing_in_sec']} không tìm thấy); "
          f"{totals['n_vnd_mismatch']} lỗi quy đổi; {totals['n_absent_but_filled']} ô bịa số "
          f"⇒ {'ĐẠT' if totals['ok'] else 'CÓ VẤN ĐỀ'}")
    return result


def markdown_provenance(result: Dict[str, Any]) -> str:
    """Sinh `reports/results/provenance.md` (mọi số đọc từ dict kết quả)."""
    from forecasting.eda import markdown_table

    totals = result["totals"]
    lines = [
        "# Kiểm chứng nguồn gốc dữ liệu (SEC thật) — `scripts.verify_provenance`",
        "",
        f"- Registry: `{result['downloads_registry']}` · snapshot thô: `{result['raw_dir']}`",
        f"- Kết luận: **{'ĐẠT' if totals['ok'] else 'CÓ VẤN ĐỀ'}**"
        f"{' (chế độ --quick: chỉ 2 công ty đầu)' if result.get('quick') else ''}",
        "",
        markdown_table(["Phép kiểm", "Số lượng"],
                       [["File SEC đã băm SHA-256", totals["n_raw_files_hashed"]],
                        ["Hash lệch registry", totals["n_hash_mismatch"]],
                        ["Công ty đối chiếu fact", totals["n_companies_checked"]],
                        ["Ô dữ liệu (quý × chỉ tiêu)", totals["n_cells"]],
                        ["Ô không có fact ở SEC (phải null)", totals["n_absent_cells"]],
                        ["Ô có fact → đối chiếu companyfacts SEC", totals["n_facts_checked"]],
                        ["Fact không tồn tại trong SEC", totals["n_facts_missing_in_sec"]],
                        ["Quy đổi VND sai (theo từng `method`)", totals["n_vnd_mismatch"]],
                        ["Ô không có fact nhưng vẫn có số", totals["n_absent_but_filled"]],
                        ["Ô dùng method chưa hỗ trợ", totals["n_unsupported_method"]]]),
        "",
        "## Theo công ty",
        "",
        markdown_table(["Ticker", "Số quý", "Ô dữ liệu", "Ô không có fact", "Ô đối chiếu",
                        "Fact thiếu ở SEC", "Lỗi VND", "Ô bịa số"],
                       [[c["ticker"], c["n_rows"], c["n_cells"], c["n_absent_cells"],
                         c["n_facts_checked"], c["n_facts_missing_in_sec"], c["n_vnd_mismatch"],
                         c["n_absent_but_filled"]] for c in result["per_company"]]),
        "",
        "## Cách đọc",
        "",
        "1. **Hash khớp** nghĩa là file SEC thô trong repo không bị sửa tay kể từ lúc tải "
        f"({result.get('note', '')})",
        "2. **Fact tồn tại trong SEC** nghĩa là từng con số dùng để huấn luyện đều tra ngược được "
        "trong companyfacts công khai (kèm `accn`, `form`, ngày `filed`) — không có số liệu tự sinh.",
        "3. **Quy đổi VND** chỉ là phép nhân hằng số 25.000 khai trong `fx_policy` (minh hoạ đơn vị, "
        "không phải dữ liệu Việt Nam và không phải tỷ giá lịch sử)",
        "4. **Ô bịa số (phải = 0)**: với chỉ tiêu không có fact ở SEC, file phải để `null` — đây là "
        "phép kiểm bắt lỗi \"điền số cho đủ\".",
        "",
        "Lệnh tái lập: `python -m scripts.verify_provenance` (đầy đủ) hoặc `... --quick` (2 công ty).",
    ]
    if any(c["examples"] for c in result["per_company"]):
        lines += ["", "## Ví dụ bất thường (nếu có)", ""]
        for company in result["per_company"]:
            for example in company["examples"]:
                lines.append(f"- `{company['ticker']}` · {example}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ensure_utf8_stdio()
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="chỉ kiểm 2 công ty đầu (nhanh, không đọc hết 129 MB)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(quick=args.quick)
    return 0 if result["totals"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
