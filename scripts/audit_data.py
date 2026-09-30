"""Đối soát TOÀN BỘ bộ dữ liệu trong repo với SỐ LIỆU THẬT đã công bố trên SEC.

Lệnh: python -m scripts.audit_data [--tickers WMT,HD] [--skip-sec] [--no-write]

Vì sao cần: repo có nhiều "tập" do pipeline gốc sinh ra (`retail-expanded` 16 chỉ tiêu, bản cũ
10 chỉ tiêu, `prepared`, `prepared-rule`, `manifest`, artifact trong `reports/`) và nhãn
`is_distressed` KHÔNG tái tạo được từ chỉ tiêu đã công bố (`docs/dinh-nghia-nhan.md`). Script này
trả lời câu hỏi "các con số trong các tập đó có đúng số liệu thật không" bằng cách truy từng giá
trị VND về FACT XBRL trong snapshot SEC (`data/sec/raw/*-companyfacts.json`) rồi kiểm tra nhất
quán nội bộ giữa các tập và với con số đã in trong báo cáo. Script CHỈ ĐỌC — không sửa dữ liệu;
phát hiện được ghi vào `reports/results/data_audit.json`.

Nhóm kiểm tra
1. `sec_snapshots`   SHA-256 snapshot khớp sổ đăng ký; CIK/tên pháp nhân khớp companyfacts.
2. `retail_expanded` Mỗi quý × 16 chỉ tiêu: fact (tag, start, end, accn) tồn tại THẬT trong
                     companyfacts, `val` (USD) trùng khớp, giá trị VND đúng quy tắc của `method`
                     (×25.000; luỹ kế trừ luỹ kế), `available_on` = ngày `filed` mới nhất, kỳ quý
                     70–105 ngày, thứ tự quý liên tục.
3. `legacy_10`       File 10 chỉ tiêu (bản cũ) trùng giá trị với bản 16 chỉ tiêu.
4. `prepared`        4 tập khớp 100% `retail-expanded`, lịch sử không rò rỉ (`available_on` ≤
                     `as_of`), chính sách 8 test / 4 validation / 2 purge tái lập được từ file
                     nguồn, các tập rời nhau.
5. `manifest`        counts, label_counts, split_sha256, source_sha256, test_ranges, mốc fit.
6. `corpus`          quarter_count, phạm vi năm, tên pháp nhân, CIK/URL.
7. `prepared_rule`   Nhãn quy tắc tái lập được (`forecasting.labels`) + khớp `relabel.json`/docs.
8. `report_numbers`  Con số trong `eda_summary.json`, `analysis.json`, `test_evaluation.json`,
                     `test_predictions.csv`, `baselines.json` khớp tính lại, kể cả xác suất do
                     `reports/models/best.joblib` sinh ra.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from forecasting.config import (BASE_FIELDS, DATA_DIR, DOCS_DIR, MANIFEST_FILE,
                                MIN_HISTORY_QUARTERS, MODELS_DIR, PREPARED_DIR, RESULTS_DIR,
                                RETAIL_DIR, STRESS_MIN_SIGNALS, SUFFIX, TARGET,
                                ensure_utf8_stdio)
from forecasting.data import (CANONICAL_VND_FIELDS, QUARTERS_TEST, QUARTERS_VALIDATION,
                              VND_PER_USD, sha256)
from forecasting.features import RATIO_PARTS
from forecasting.labels import label_row, signal_flags

# ---------------------------------------------------------------------------
# Đường dẫn + hằng số
# ---------------------------------------------------------------------------
RAW_DIR = DATA_DIR / "sec" / "raw"
DOWNLOADS_FILE = DATA_DIR / "sec" / "downloads.json"
CORPUS_FILE = RETAIL_DIR / "corpus.json"
RULE_DIR = DATA_DIR / "prepared-rule"
SPLITS = ("train", "validation", "test", "purged")
GROUPS = ("sec_snapshots", "retail_expanded", "legacy_10", "prepared", "manifest", "corpus",
          "prepared_rule", "report_numbers")

#: Quy tắc suy giá trị VND từ provenance (khớp mô tả trong `scripts/prepare_sec.py`; đã đối
#: chiếu trên toàn bộ fact của 8 công ty).
QUARTER_METHOD = "reported_quarter"
FIRST_QUARTER_METHOD = "reported_first_quarter"
INSTANT_METHOD = "instant"
YTD_METHOD = "current_ytd_minus_previous_ytd"
ABSENT_METHOD = "absent"
DURATION_METHODS = (QUARTER_METHOD, FIRST_QUARTER_METHOD)
KNOWN_METHODS = DURATION_METHODS + (INSTANT_METHOD, YTD_METHOD, ABSENT_METHOD)

#: Một quý tài chính 13 tuần → khoảng ngày hợp lệ (khớp thông điệp lọc trong corpus.json).
MIN_QUARTER_DAYS, MAX_QUARTER_DAYS = 70, 105
_SAMPLE_ID_RE = re.compile(r"^(?P<ticker>[A-Z]+)-(?P<fy>\d{4})Q(?P<q>[1-4])$")

#: Cache companyfacts theo ticker (mỗi file ~5–8 MB).
_SEC_DOCS: Dict[str, Dict[str, Any]] = {}



# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """Bảo đảm context lỗi ghi được ra JSON (chỉ giữ kiểu đơn giản)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _day(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _quarter_days(row: Dict[str, Any]) -> int | None:
    start, end = _day(row.get("period_start")), _day(row.get("period_end"))
    return None if start is None or end is None else (end - start).days + 1


def _close(x: Any, y: Any, tol: float = 1e-9) -> bool:
    """So hai số (None/NaN được coi là khớp nhau)."""
    if x is None or y is None:
        return x is None and y is None
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return x == y
    if fx != fx and fy != fy:  # cả hai là NaN
        return True
    return abs(fx - fy) <= tol


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Nén row retail-expanded về dạng history row của prepared (giống forecasting.data)."""
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


def _vnd_fields(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Tên chỉ tiêu có trong file (16 với bản chuẩn, 10 với bản cũ)."""
    keys = {k for row in rows for k in row if k.endswith(SUFFIX)}
    return sorted(k[: -len(SUFFIX)] for k in keys)


def _parse_sample_id(sample_id: str) -> Tuple[str, int, int] | None:
    match = _SAMPLE_ID_RE.match(sample_id or "")
    return None if not match else (match.group("ticker"), int(match.group("fy")),
                                   int(match.group("q")))


def sec_doc(ticker: str) -> Dict[str, Any]:
    """companyfacts của ticker (đọc 1 lần, cache); {} nếu thiếu file."""
    if ticker not in _SEC_DOCS:
        path = RAW_DIR / f"{ticker}-companyfacts.json"
        _SEC_DOCS[ticker] = _read_json(path) if path.exists() else {}
    return _SEC_DOCS[ticker]


def sec_fact_index(ticker: str) -> Dict[Tuple[str, Any, Any, Any], List[Dict[str, Any]]]:
    """{(tag, start, end, accn): [fact, ...]} để tra fact thật trong companyfacts."""
    index: Dict[Tuple[str, Any, Any, Any], List[Dict[str, Any]]] = {}
    for _taxonomy, tags in (sec_doc(ticker).get("facts") or {}).items():
        for tag, body in tags.items():
            for _unit, entries in (body.get("units") or {}).items():
                for entry in entries:
                    key = (tag, entry.get("start"), entry.get("end"), entry.get("accn"))
                    index.setdefault(key, []).append(entry)
    return index


class Auditor:
    """Thu lỗi theo nhóm kiểm tra + đếm số phép kiểm tra đã chạy."""

    def __init__(self) -> None:
        self.issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.stats: Dict[str, Any] = {}
        self.checks = 0

    def check(self, ok: bool, group: str, message: str, **ctx: Any) -> bool:
        """Ghi một phép kiểm tra; trả về `ok` để dùng tiếp trong luồng xử lý."""
        self.checks += 1
        if not ok:
            self.issues[group].append({"message": message, **_jsonable(ctx)})
        return bool(ok)

    def eq(self, group: str, message: str, declared: Any, actual: Any, **ctx: Any) -> bool:
        """So bằng tuyệt đối (giá trị khai báo vs giá trị tính lại)."""
        return self.check(declared == actual, group, message,
                          declared=declared, actual=actual, **ctx)

    def close(self, group: str, message: str, declared: Any, actual: Any,
              tol: float = 1e-9, **ctx: Any) -> bool:
        """So gần đúng cho số thực (mặc định 1e-9)."""
        return self.check(_close(declared, actual, tol), group, message,
                          declared=declared, actual=actual, **ctx)

    def stat(self, key: str, value: Any) -> None:
        self.stats[key] = value

    def n_issues(self) -> int:
        return sum(len(v) for v in self.issues.values())

    def summary(self) -> Dict[str, Any]:
        return {"n_checks": self.checks, "n_issues": self.n_issues(),
                "issues_by_group": {g: len(self.issues.get(g, [])) for g in GROUPS},
                "stats": self.stats,
                "issues": {g: self.issues[g] for g in GROUPS if self.issues.get(g)}}

    def markdown(self) -> str:
        """Markdown ngắn: số phát hiện theo nhóm + số liệu đối chiếu + phát hiện chi tiết."""
        lines = ["# Đối soát dữ liệu với SỐ LIỆU THẬT (SEC) — `scripts.audit_data`", "",
                 f"- Số phép kiểm tra: **{self.checks}**",
                 f"- Số phát hiện: **{self.n_issues()}**", "",
                 "## Phát hiện theo nhóm", "", "| Nhóm kiểm tra | Số phát hiện |", "|---|---:|"]
        lines += [f"| `{g}` | {len(self.issues.get(g, []))} |" for g in GROUPS]
        lines += ["", "## Số liệu đối chiếu", "", "| Khoá | Giá trị |", "|---|---|"]
        lines += [f"| `{k}` | `{json.dumps(v, ensure_ascii=False, default=str)}` |"
                  for k, v in self.stats.items()]
        for group in GROUPS:
            items = self.issues.get(group, [])
            if not items:
                continue
            lines += ["", f"## Nhóm `{group}` — {len(items)} phát hiện", ""]
            for item in items[:40]:
                ctx = {k: v for k, v in item.items() if k != "message"}
                suffix = f" — `{json.dumps(ctx, ensure_ascii=False, default=str)}`" if ctx else ""
                lines.append(f"- {item['message']}{suffix}")
            if len(items) > 40:
                lines.append(f"- … còn {len(items) - 40} phát hiện nữa (xem `data_audit.json`)")
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Nhóm 1 — snapshot SEC thô (nguồn "số liệu thật")
# ---------------------------------------------------------------------------
def check_sec_snapshots(a: Auditor, tickers: Sequence[str] | None = None) -> None:
    """SHA-256 + CIK + tên pháp nhân của `data/sec/raw/*-companyfacts.json`."""
    registries: Dict[str, Dict[str, Any]] = {}
    if DOWNLOADS_FILE.exists():
        registries[DOWNLOADS_FILE.name] = _read_json(DOWNLOADS_FILE)
    else:
        a.check(False, "sec_snapshots", f"thiếu sổ đăng ký {DOWNLOADS_FILE.name}")
    corpus = _read_json(CORPUS_FILE) if CORPUS_FILE.exists() else {}
    if corpus.get("downloads"):
        registries["corpus.json → downloads"] = corpus["downloads"]
    a.stat("sec_registries", sorted(registries))

    for reg_name, registry in registries.items():
        for ticker, meta in registry.items():
            if tickers and ticker not in tickers:
                continue
            path = RAW_DIR / f"{ticker}-companyfacts.json"
            if not a.check(path.exists(), "sec_snapshots",
                           f"{reg_name}: thiếu snapshot {path.name}", ticker=ticker):
                continue
            a.eq("sec_snapshots", f"{reg_name}: SHA-256 lệch cho {ticker}",
                 meta.get("sha256"), sha256(path))
            doc = sec_doc(ticker)
            a.eq("sec_snapshots", f"{reg_name}: CIK khác companyfacts cho {ticker}",
                 meta.get("cik"), doc.get("cik"))
            a.eq("sec_snapshots", f"{reg_name}: tên pháp nhân khác companyfacts cho {ticker}",
                 meta.get("entity_name"), doc.get("entityName"))


# ---------------------------------------------------------------------------
# Nhóm 2/3 — retail-expanded: cấu trúc quý + provenance + giá trị VND
# ---------------------------------------------------------------------------
def check_quarter_row(a: Auditor, ticker: str, at: int, row: Dict[str, Any],
                      prev: Dict[str, Any] | None, group: str) -> None:
    """Kỳ 70–105 ngày, available_on sau khi quý kết thúc, thứ tự quý liên tục."""
    where = f"{ticker}-{row.get('fiscal_year')}Q{row.get('fiscal_quarter')}"
    a.eq(group, f"{where}: ticker trong row khác tên file", row.get("ticker"), ticker, at=at)
    a.eq(group, f"{where}: currency khác VND", row.get("currency"), "VND", at=at)
    days = _quarter_days(row)
    a.check(days is not None and MIN_QUARTER_DAYS <= days <= MAX_QUARTER_DAYS, group,
            f"{where}: kỳ {days} ngày — không phải một quý riêng", at=at)
    end, available = _day(row.get("period_end")), _day(row.get("available_on"))
    a.check(end is not None and available is not None and available >= end, group,
            f"{where}: available_on {row.get('available_on')} không sau khi quý kết thúc", at=at)
    quarter = row.get("fiscal_quarter")
    if not a.check(quarter in (1, 2, 3, 4), group,
                   f"{where}: fiscal_quarter ngoài 1..4", at=at):
        return
    prev_q = prev.get("fiscal_quarter") if prev else None
    if prev and prev_q in (1, 2, 3, 4):
        prev_end = _day(prev.get("period_end"))
        a.check(prev_end is not None and end is not None and end > prev_end, group,
                f"{where}: period_end không tăng so với quý trước", at=at)
        prev_fy = prev.get("fiscal_year")
        want = (prev_fy + 1, 1) if prev_q == 4 else (prev_fy, prev_q + 1)
        a.eq(group, f"{where}: nhãn quý không liên tục sau {prev_fy}Q{prev_q}",
             (row.get("fiscal_year"), quarter), want, at=at)


def _expected_vnd(a: Auditor, where: str, field: str, method: str, facts: List[Dict[str, Any]],
                  row: Dict[str, Any], group: str) -> int | None:
    """Giá trị VND suy từ fact THẬT theo `method`; ghi lỗi nếu kỳ/fact không khớp."""
    period_start, period_end = row.get("period_start"), row.get("period_end")
    if method == YTD_METHOD:
        if not a.check(len(facts) >= 2, group,
                       f"{where}: {field} cần ≥2 fact luỹ kế để trừ ra quý riêng", field=field):
            return None
        ordered = sorted(facts, key=lambda f: (f.get("end") or "", f.get("filed") or ""))
        current, previous = ordered[-1], ordered[-2]
        a.eq(group, f"{where}: {field} luỹ kế hiện tại không kết thúc tại period_end",
             current.get("end"), period_end, field=field)
        a.eq(group, f"{where}: {field} hai fact luỹ kế không cùng mốc đầu năm",
             current.get("start"), previous.get("start"), field=field)
        a.check((previous.get("end") or "9999-12-31") < (period_start or "0000-01-01"), group,
                f"{where}: {field} fact luỹ kế trước chồng lấn kỳ target",
                previous_end=previous.get("end"), period_start=period_start, field=field)
        cur_val, prev_val = _int_or_none(current.get("val")), _int_or_none(previous.get("val"))
        return None if cur_val is None or prev_val is None else (cur_val - prev_val) * VND_PER_USD

    fact = facts[0]
    if method in DURATION_METHODS:
        a.check(fact.get("start") == period_start and fact.get("end") == period_end, group,
                f"{where}: {field} fact không khớp kỳ quý", field=field,
                fact_start=fact.get("start"), fact_end=fact.get("end"))
    elif method == INSTANT_METHOD:
        a.check(fact.get("start") is None and fact.get("end") == period_end, group,
                f"{where}: {field} fact instant không khớp period_end", field=field,
                fact_start=fact.get("start"), fact_end=fact.get("end"))
    if len(facts) > 1:
        a.check(False, group, f"{where}: {field} có {len(facts)} fact nhưng method={method}",
                field=field)
    value = _int_or_none(fact.get("val"))
    return None if value is None else value * VND_PER_USD


def check_quarter_provenance(a: Auditor, ticker: str, row: Dict[str, Any],
                             index: Dict[Tuple[str, Any, Any, Any], List[Dict[str, Any]]],
                             fields: Sequence[str], group: str) -> None:
    """Mọi chỉ tiêu của quý phải truy được về fact thật + giá trị VND đúng."""
    where = f"{ticker}-{row.get('fiscal_year')}Q{row.get('fiscal_quarter')}"
    sources = row.get("sources") or {}
    a.check(set(sources) == set(fields), group,
            f"{where}: tập provenance khác tập chỉ tiêu của file",
            missing=sorted(set(fields) - set(sources)), extra=sorted(set(sources) - set(fields)))
    for field in fields:
        src = sources.get(field)
        stored = _int_or_none(row.get(field + SUFFIX))
        if not a.check(isinstance(src, dict), group,
                       f"{where}: thiếu provenance cho {field}", field=field):
            continue
        method, facts = src.get("method"), list(src.get("facts") or [])
        if not a.check(method in KNOWN_METHODS, group,
                       f"{where}: method lạ {method!r} cho {field}", field=field):
            continue
        if method == ABSENT_METHOD:
            a.check(not facts and stored is None, group,
                    f"{where}: {field} method=absent nhưng vẫn có fact/giá trị",
                    field=field, n_facts=len(facts), value=stored)
            continue
        if not a.check(bool(facts), group, f"{where}: {field} không có fact nào", field=field):
            continue

        # (1) fact phải tồn tại THẬT trong companyfacts với `val` trùng khớp
        for fact in facts:
            key = (fact.get("tag"), fact.get("start"), fact.get("end"), fact.get("accn"))
            hits = index.get(key)
            if not a.check(bool(hits), group,
                           f"{where}: fact {field} không có trong snapshot SEC",
                           field=field, fact=str(key)):
                continue
            a.check(any(h.get("val") == fact.get("val") for h in hits), group,
                    f"{where}: `val` provenance khác companyfacts cho {field}",
                    field=field, provenance_val=fact.get("val"),
                    sec_vals=sorted({h.get("val") for h in hits
                                     if isinstance(h.get("val"), int)})[:3])

        # (2) giá trị VND phải đúng quy tắc của method (×25.000; luỹ kế trừ luỹ kế)
        expected = _expected_vnd(a, where, field, method, facts, row, group)
        if expected is not None:
            a.eq(group, f"{where}: {field} VND khác số liệu thật trong companyfacts",
                 stored, expected, field=field)

        # (3) available_on = ngày filed mới nhất; source_url chứa accession của fact đó
        latest = max(facts, key=lambda f: (f.get("filed") or "", f.get("end") or ""))
        a.eq(group, f"{where}: available_on khác ngày filed của fact {field}",
             row.get("available_on"), latest.get("filed"), field=field)
        url = str(row.get("source_url") or "").replace("-", "")
        a.check(str(latest.get("accn") or "").replace("-", "") in url, group,
                f"{where}: source_url không chứa accession của fact {field}",
                field=field, accn=latest.get("accn"))


def check_retail_doc(a: Auditor, path: Path, group: str) -> Tuple[str | None, List[Dict[str, Any]]]:
    """Kiểm tra một file `*-indicators-vnd.json` (bản chuẩn 16 hoặc bản cũ 10 chỉ tiêu)."""
    doc = _read_json(path)
    ticker = doc.get("ticker")
    rows = list(doc.get("rows") or [])
    fields = _vnd_fields(rows)
    a.stat(f"fields[{path.name}]", fields)
    a.eq(group, f"{path.name}: currency khác VND", doc.get("currency"), "VND")
    a.eq(group, f"{path.name}: tỷ giá minh hoạ khác {VND_PER_USD}",
         str((doc.get("fx_policy") or {}).get("vnd_per_usd")), str(VND_PER_USD))
    a.eq(group, f"{path.name}: indicator_count khác số chỉ tiêu có trong row",
         doc.get("indicator_count"), len(fields))
    index = sec_fact_index(str(ticker))
    a.check(bool(index), group, f"{path.name}: không đọc được companyfacts của {ticker}")
    prev: Dict[str, Any] | None = None
    for i, row in enumerate(rows):
        check_quarter_row(a, str(ticker), i, row, prev, group)
        check_quarter_provenance(a, str(ticker), row, index, fields, group)
        prev = row
    return ticker, rows


def check_retail_expanded(a: Auditor, tickers: Sequence[str] | None = None
                          ) -> Dict[str, List[Dict[str, Any]]]:
    """Nhóm 2 (bản 16 chỉ tiêu) + nhóm 3 (bản cũ 10 chỉ tiêu phải trùng giá trị)."""
    rows_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    legacy: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(RETAIL_DIR.glob("*-16-indicators-vnd.json")):
        ticker, rows = check_retail_doc(a, path, "retail_expanded")
        if ticker and not (tickers and ticker not in tickers):
            rows_by_ticker[ticker] = rows
    for path in sorted(RETAIL_DIR.glob("*-10-indicators-vnd.json")):
        ticker, rows = check_retail_doc(a, path, "legacy_10")
        if ticker:
            legacy[ticker] = rows

    for ticker, rows in legacy.items():
        current = rows_by_ticker.get(ticker)
        if not current:
            a.stat(f"legacy_only[{ticker}]", len(rows))  # công ty không được giữ trong corpus
            continue
        by_key = {(r.get("fiscal_year"), r.get("fiscal_quarter")): r for r in current}
        shared = _vnd_fields(rows)
        for row in rows:
            key = (row.get("fiscal_year"), row.get("fiscal_quarter"))
            other = by_key.get(key)
            if not a.check(other is not None, "legacy_10",
                           f"{ticker}-{key[0]}Q{key[1]}: quý không có ở bản 16 chỉ tiêu"):
                continue
            for field in shared:
                a.eq("legacy_10",
                     f"{ticker}-{key[0]}Q{key[1]}: {field} khác giữa bản 10 và bản 16 chỉ tiêu",
                     row.get(field + SUFFIX), other.get(field + SUFFIX), field=field)

    a.stat("retail_quarters", {t: len(r) for t, r in sorted(rows_by_ticker.items())})
    a.stat("legacy_10_tickers", sorted(legacy))
    return rows_by_ticker


# ---------------------------------------------------------------------------
# Nhóm 4 — prepared: 4 tập khớp retail-expanded + chính sách split
# ---------------------------------------------------------------------------
def _expected_split_by_sample(rows_by_ticker: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
    """Chính sách split của `forecasting.data.split_policy`, tính lại từ file nguồn."""
    expected: Dict[str, str] = {}
    need = QUARTERS_TEST + QUARTERS_VALIDATION + 2
    for ticker, rows in rows_by_ticker.items():
        n = len(rows) - 1  # số cặp (lịch sử → quý target)
        for j in range(n):
            target = rows[j + 1]
            sid = f"{ticker}-{target['fiscal_year']}Q{target['fiscal_quarter']}"
            if n <= need or j < n - need:
                expected[sid] = "train"
            elif j in (n - need, n - QUARTERS_TEST - 1):
                expected[sid] = "purged"
            elif j < n - QUARTERS_TEST - 1:
                expected[sid] = "validation"
            else:
                expected[sid] = "test"
    return expected


def check_prepared_sample(a: Auditor, split: str, sample: Dict[str, Any],
                          rows_by_ticker: Dict[str, List[Dict[str, Any]]]) -> None:
    """Một sample: khớp 100% retail-expanded, lịch sử đúng quý, không rò rỉ tương lai."""
    sid = str(sample.get("sample_id"))
    ticker = sample.get("ticker")
    rows = rows_by_ticker.get(str(ticker))
    if not a.check(rows is not None, "prepared",
                   f"{sid}: ticker không có trong retail-expanded", split=split, sample_id=sid):
        return
    assert rows is not None
    parsed = _parse_sample_id(sid)
    if not a.check(parsed is not None, "prepared",
                   f"{sid}: sample_id sai định dạng TICKER-YYYYQn", split=split, sample_id=sid):
        return
    assert parsed is not None
    sid_ticker, fiscal_year, fiscal_quarter = parsed
    a.eq("prepared", f"{sid}: ticker trong sample_id khác field ticker", sid_ticker, ticker,
         split=split, sample_id=sid)
    idx = next((i for i, r in enumerate(rows) if r.get("fiscal_year") == fiscal_year
                and r.get("fiscal_quarter") == fiscal_quarter), None)
    if not a.check(idx is not None, "prepared",
                   f"{sid}: quý target không có trong retail-expanded", split=split,
                   sample_id=sid):
        return
    assert idx is not None
    if not a.check(idx >= 1, "prepared",
                   f"{sid}: quý target là quý đầu tiên — không có lịch sử", split=split,
                   sample_id=sid):
        return

    request = sample.get("request") or {}
    history = list(request.get("history") or [])
    a.eq("prepared",
         f"{sid}: lịch sử có {len(history)} quý, phải là {idx} quý trước quý target",
         len(history), idx, split=split, sample_id=sid)
    for i, (expected_row, got_row) in enumerate(zip([_compact_row(r) for r in rows[:idx]],
                                                    history)):
        if expected_row != got_row:
            diff = sorted(k for k in set(expected_row) | set(got_row)
                          if expected_row.get(k) != got_row.get(k))
            a.check(False, "prepared", f"{sid}: quý lịch sử #{i} khác retail-expanded",
                    split=split, sample_id=sid, fields_khac_nhau=diff[:6])

    target = rows[idx]
    a.eq("prepared", f"{sid}: as_of khác ngày công bố của quý lịch sử cuối",
         request.get("as_of"), rows[idx - 1].get("available_on"), split=split, sample_id=sid)
    a.eq("prepared", f"{sid}: kỳ target khác retail-expanded",
         [request.get("target_period_start"), request.get("target_period_end")],
         [target.get("period_start"), target.get("period_end")], split=split, sample_id=sid)
    a.eq("prepared", f"{sid}: label_available_on khác ngày công bố quý target",
         sample.get("label_available_on"), target.get("available_on"), split=split,
         sample_id=sid)
    a.eq("prepared", f"{sid}: target_source_url khác retail-expanded",
         sample.get("target_source_url"), target.get("source_url"), split=split, sample_id=sid)
    a.check(sample.get(TARGET) in (0, 1), "prepared", f"{sid}: nhãn không phải 0/1",
            split=split, sample_id=sid, value=sample.get(TARGET))


    as_of, target_start = _day(request.get("as_of")), _day(target.get("period_start"))
    label_day = _day(sample.get("label_available_on"))
    a.check(as_of is not None and label_day is not None and label_day >= as_of, "prepared",
            f"{sid}: nhãn công bố trước as_of", split=split, sample_id=sid,
            label_available_on=sample.get("label_available_on"), as_of=request.get("as_of"))
    for i, row in enumerate(history):
        available, end = _day(row.get("available_on")), _day(row.get("period_end"))
        a.check(available is not None and as_of is not None and available <= as_of, "prepared",
                f"{sid}: quý lịch sử #{i} công bố SAU as_of (rò rỉ)", split=split, sample_id=sid,
                available_on=row.get("available_on"), as_of=request.get("as_of"))
        a.check(end is not None and target_start is not None and end < target_start, "prepared",
                f"{sid}: quý lịch sử #{i} chồng lấn kỳ target", split=split, sample_id=sid,
                period_end=row.get("period_end"), target_start=target.get("period_start"))


def check_prepared(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]]
                   ) -> Dict[str, List[Dict[str, Any]]]:
    """Bốn tập phải rời nhau, phủ đủ cặp dự báo, đúng chính sách và khớp file nguồn."""
    splits: Dict[str, List[Dict[str, Any]]] = {}
    for name in SPLITS:
        path = PREPARED_DIR / f"{name}.json"
        if not a.check(path.exists(), "prepared", f"thiếu {path.name}"):
            splits[name] = []
            continue
        splits[name] = list(_read_json(path))

    seen: Dict[str, str] = {}
    for name in SPLITS:
        for sample in splits[name]:
            sid = str(sample.get("sample_id"))
            if sid in seen:
                a.check(False, "prepared", f"{sid} xuất hiện ở cả {seen[sid]} và {name}",
                        sample_id=sid)
            seen[sid] = name

    expected = _expected_split_by_sample(rows_by_ticker)
    for sid, want in expected.items():
        got = seen.get(sid)
        a.check(got is not None, "prepared", f"{sid}: không có trong tập nào", sample_id=sid)
        if got:
            a.eq("prepared", f"{sid}: ở tập {got}, chính sách yêu cầu {want}", got, want,
                 sample_id=sid)
    extra = sorted(set(seen) - set(expected))
    a.check(not extra, "prepared",
            f"có {len(extra)} sample không sinh được từ retail-expanded", samples=extra[:5])
    a.stat("prepared_counts", {n: len(splits[n]) for n in SPLITS})
    a.stat("expected_counts", {n: sum(1 for v in expected.values() if v == n) for n in SPLITS})
    a.stat("n_samples_total", len(seen))

    for name in SPLITS:
        for sample in splits[name]:
            check_prepared_sample(a, name, sample, rows_by_ticker)
    return splits


# ---------------------------------------------------------------------------
# Nhóm 5 — manifest: con số tổng hợp + hash + mốc fit
# ---------------------------------------------------------------------------
def _max_published(samples: List[Dict[str, Any]]) -> str | None:
    """Ngày công bố lớn nhất xuất hiện trong lịch sử của tập mẫu."""
    days = [h.get("available_on") for s in samples
            for h in ((s.get("request") or {}).get("history") or [])]
    days = [d for d in days if d]
    return max(days) if days else None


def _min_as_of(samples: List[Dict[str, Any]]) -> str | None:
    """`as_of` nhỏ nhất của tập mẫu."""
    days = [(s.get("request") or {}).get("as_of") for s in samples]
    days = [d for d in days if d]
    return min(days) if days else None


def check_manifest(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]],
                   splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Counts / label_counts / split_sha256 / source_sha256 / test_ranges / mốc fit."""
    if not a.check(MANIFEST_FILE.exists(), "manifest", "thiếu data/prepared/manifest.json"):
        return {}
    man = _read_json(MANIFEST_FILE)

    for name in SPLITS:
        arr = splits[name]
        a.eq("manifest", f"counts[{name}] khác số mẫu thật",
             (man.get("counts") or {}).get(name), len(arr), split=name)
        label_counts = (man.get("label_counts") or {}).get(name) or {}
        a.eq("manifest", f"label_counts[{name}].total khác số mẫu thật",
             label_counts.get("total"), len(arr), split=name)
        a.eq("manifest", f"label_counts[{name}].distressed khác nhãn thật",
             label_counts.get("distressed"), sum(int(s[TARGET]) for s in arr), split=name)
        path = PREPARED_DIR / f"{name}.json"
        if path.exists():
            a.eq("manifest", f"split_sha256[{name}] khác SHA-256 của file thật",
                 (man.get("split_sha256") or {}).get(name), sha256(path), split=name)

    for file_name, digest in (man.get("source_sha256") or {}).items():
        path = RETAIL_DIR / file_name
        if not a.check(path.exists(), "manifest",
                       f"source_sha256 trỏ tới file không tồn tại: {file_name}", file=file_name):
            continue
        a.eq("manifest", f"source_sha256[{file_name}] khác SHA-256 của file nguồn thật",
             digest, sha256(path), file=file_name)

    periods: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for sample in splits["test"]:
        request = sample.get("request") or {}
        periods[str(sample.get("ticker"))].append(
            (request.get("target_period_start"), request.get("target_period_end")))
    computed = {t: {"from": min(p[0] for p in v), "to": max(p[1] for p in v), "count": len(v)}
                for t, v in periods.items()}
    a.eq("manifest", "test_ranges khác khoảng thời gian thật của tập test",
         man.get("test_ranges"), computed)
    a.check(man.get("test_is_last_two_fiscal_years_per_company") is True, "manifest",
            "test_is_last_two_fiscal_years_per_company không phải true")

    for ticker, rows in rows_by_ticker.items():
        years = sorted({_parse_sample_id(str(s.get("sample_id")))[1]  # type: ignore[index]
                        for s in splits["test"] if s.get("ticker") == ticker})
        last_two = [rows[-1]["fiscal_year"] - 1, rows[-1]["fiscal_year"]]
        n_test = sum(1 for s in splits["test"] if s.get("ticker") == ticker)
        a.check(years == last_two and n_test == QUARTERS_TEST, "manifest",
                f"{ticker}: test không phải 8 quý = 2 năm tài chính cuối", ticker=ticker,
                test_years=years, n_test=n_test, expected=last_two)

    a.check(man.get("currency") == "VND"
            and str(man.get("demo_vnd_per_usd")) == str(VND_PER_USD), "manifest",
            "currency/tỷ giá minh hoạ trong manifest khác quy ước",
            currency=man.get("currency"), vnd_per_usd=man.get("demo_vnd_per_usd"))
    a.eq("manifest", f"target khác {TARGET}", man.get("target"), TARGET)
    a.check(bool(man.get("policy")), "manifest", "manifest thiếu mô tả policy")

    train_end, val_start = _max_published(splits["train"]), _min_as_of(splits["validation"])
    a.stat("validation_fit_before[manifest|max_train_pub|min_val_as_of]",
           [man.get("validation_fit_before"), train_end, val_start])
    a.check(bool(train_end and val_start and man.get("validation_fit_before")
                 and train_end <= man["validation_fit_before"] <= val_start), "manifest",
            "validation_fit_before không nằm giữa mốc công bố của train và as_of đầu validation",
            manifest=man.get("validation_fit_before"), train_end=train_end, val_start=val_start)

    fit_end = max([d for d in (_max_published(splits["train"]),
                               _max_published(splits["validation"])) if d], default=None)
    test_start = _min_as_of(splits["test"])
    a.stat("final_fit_before[manifest|max_train_val_pub|min_test_as_of]",
           [man.get("final_fit_before"), fit_end, test_start])
    a.check(bool(fit_end and test_start and man.get("final_fit_before")
                 and fit_end <= man["final_fit_before"] <= test_start), "manifest",
            "final_fit_before không nằm giữa mốc công bố của train+validation và as_of đầu test",
            manifest=man.get("final_fit_before"), fit_end=fit_end, test_start=test_start)
    return man


# ---------------------------------------------------------------------------
# Nhóm 6 — corpus.json: số quý, phạm vi năm, tên pháp nhân, CIK/URL
# ---------------------------------------------------------------------------
def check_corpus(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]]) -> None:
    """Mọi con số mô tả corpus phải khớp file dữ liệu thật + companyfacts."""
    if not a.check(CORPUS_FILE.exists(), "corpus", "thiếu data/retail-expanded/corpus.json"):
        return
    corpus = _read_json(CORPUS_FILE)
    companies = corpus.get("companies") or {}
    a.eq("corpus", "tập công ty trong corpus.json khác tập file 16 chỉ tiêu",
         sorted(companies), sorted(rows_by_ticker))

    for ticker, meta in companies.items():
        rows = rows_by_ticker.get(ticker)
        if not a.check(rows is not None, "corpus",
                       f"{ticker}: corpus.json có nhưng thiếu file 16 chỉ tiêu", ticker=ticker):
            continue
        assert rows is not None
        doc = sec_doc(ticker)
        a.eq("corpus", f"{ticker}: tên file khai báo khác quy ước",
             meta.get("file"), f"{ticker}-16-indicators-vnd.json", ticker=ticker)
        a.eq("corpus", f"{ticker}: quarter_count khác số quý thật",
             meta.get("quarter_count"), len(rows), ticker=ticker)
        a.eq("corpus", f"{ticker}: first_fiscal_year khác quý đầu thật",
             meta.get("first_fiscal_year"), rows[0].get("fiscal_year"), ticker=ticker)
        a.eq("corpus", f"{ticker}: last_fiscal_year khác quý cuối thật",
             meta.get("last_fiscal_year"), rows[-1].get("fiscal_year"), ticker=ticker)
        a.eq("corpus", f"{ticker}: tên công ty khác entityName thật của SEC",
             meta.get("name"), doc.get("entityName"), ticker=ticker)

    downloads = corpus.get("downloads") or {}
    a.check(set(downloads) >= set(companies), "corpus",
            "corpus.json → downloads thiếu công ty đang dùng",
            missing=sorted(set(companies) - set(downloads)))
    for ticker, meta in downloads.items():
        doc = sec_doc(ticker)
        if not a.check(bool(doc), "corpus",
                       f"{ticker}: không có snapshot companyfacts để đối chiếu", ticker=ticker):
            continue
        a.eq("corpus", f"{ticker}: CIK trong corpus khác companyfacts",
             meta.get("cik"), doc.get("cik"), ticker=ticker)
        a.eq("corpus", f"{ticker}: entity_name trong corpus khác companyfacts",
             meta.get("entity_name"), doc.get("entityName"), ticker=ticker)
        a.eq("corpus", f"{ticker}: URL companyfacts không khớp CIK", meta.get("url"),
             f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(meta.get('cik') or 0):010d}.json",
             ticker=ticker)

    excluded = corpus.get("excluded") or {}
    a.check(bool(excluded) and all(str(v).strip() for v in excluded.values()), "corpus",
            "có công ty bị loại nhưng thiếu lý do")
    a.stat("corpus_companies", sorted(companies))
    a.stat("corpus_excluded", sorted(excluded))


# ---------------------------------------------------------------------------
# Nhóm 7 — prepared-rule: nhãn quy tắc có tái lập được từ dữ liệu thật?
# ---------------------------------------------------------------------------
def check_prepared_rule(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]],
                        splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Nhãn `is_distressed_rule` phải tính lại được + khớp artifact và docs."""
    rule: Dict[str, List[Dict[str, Any]]] = {}
    for name in SPLITS:
        path = RULE_DIR / f"{name}.json"
        if not a.check(path.exists(), "prepared_rule", f"thiếu data/prepared-rule/{name}.json"):
            rule[name] = []
            continue
        rule[name] = list(_read_json(path))

    original_ids = {str(s.get("sample_id")) for arr in splits.values() for s in arr}
    rule_ids = {str(s.get("sample_id")) for arr in rule.values() for s in arr}
    a.check(rule_ids == original_ids, "prepared_rule",
            "tập sample của prepared-rule khác prepared",
            only_rule=sorted(rule_ids - original_ids)[:5],
            only_prepared=sorted(original_ids - rule_ids)[:5])

    original_labels = {str(s.get("sample_id")): int(s[TARGET])
                       for arr in splits.values() for s in arr}
    pairs: List[Tuple[int, int]] = []
    for name in SPLITS:
        for sample in rule[name]:
            sid = str(sample.get("sample_id"))
            parsed = _parse_sample_id(sid)
            rows = rows_by_ticker.get(str(sample.get("ticker")))
            if parsed is None or rows is None:
                a.check(False, "prepared_rule",
                        f"{sid}: không truy được quý trong retail-expanded", sample_id=sid)
                continue
            idx = next((i for i, r in enumerate(rows) if r.get("fiscal_year") == parsed[1]
                        and r.get("fiscal_quarter") == parsed[2]), None)
            if idx is None or idx < 1:
                a.check(False, "prepared_rule", f"{sid}: quý target không hợp lệ", sample_id=sid)
                continue
            expected, active = label_row(rows[idx], rows, idx, min_signals=STRESS_MIN_SIGNALS)
            a.eq("prepared_rule",
                 f"{sid}: nhãn quy tắc không tái lập được từ dữ liệu thật",
                 int(sample[TARGET]), expected, signals=active)
            if sid in original_labels:
                pairs.append((original_labels[sid], expected))

    counts = {n: len(rule[n]) for n in SPLITS}
    label_counts = {n: sum(int(s[TARGET]) for s in rule[n]) for n in SPLITS}
    agree = sum(1 for o, r in pairs if o == r) / len(pairs) if pairs else float("nan")
    a.stat("rule_counts", counts)
    a.stat("rule_label_counts", label_counts)
    a.stat("rule_agreement_recomputed", agree)

    manifest_path = RULE_DIR / "manifest.json"
    if a.check(manifest_path.exists(), "prepared_rule",
               "thiếu data/prepared-rule/manifest.json"):
        man = _read_json(manifest_path)
        a.check(man.get("counts") == counts and man.get("label_counts") == label_counts,
                "prepared_rule", "manifest của prepared-rule khác file thật",
                declared_counts=man.get("counts"), actual_counts=counts,
                declared_labels=man.get("label_counts"), actual_labels=label_counts)
        a.eq("prepared_rule", "min_signals trong manifest khác STRESS_MIN_SIGNALS",
             man.get("min_signals"), STRESS_MIN_SIGNALS)
        a.close("prepared_rule",
                "mức khớp nhãn trong manifest prepared-rule khác tính lại",
                man.get("agreement_with_original_labels"), agree)

    relabel_path = RESULTS_DIR / "relabel.json"
    if a.check(relabel_path.exists(), "prepared_rule", "thiếu reports/results/relabel.json"):
        relabel_man = (_read_json(relabel_path).get("manifest") or {})
        a.eq("prepared_rule",
             "reports/results/relabel.json báo số mẫu dương tính khác tính lại",
             relabel_man.get("label_counts"), label_counts)
        a.close("prepared_rule", "reports/results/relabel.json báo mức khớp nhãn khác tính lại",
                relabel_man.get("agreement_with_original_labels"), agree)

    doc_path = DOCS_DIR / "dinh-nghia-nhan.md"
    if doc_path.exists():
        text = doc_path.read_text(encoding="utf-8")
        match = re.search(r"nhãn quy tắc: `(\{[^`]*\})`", text)
        if match:
            try:
                declared = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                declared = None
            a.eq("prepared_rule",
                 "số mẫu dương tính ghi trong docs/dinh-nghia-nhan.md khác tính lại",
                 declared, label_counts)
        match = re.search(r"Mức khớp với nhãn gốc: \*\*([\d.,]+)%\*\*", text)
        if match:
            a.close("prepared_rule",
                    "mức khớp nhãn ghi trong docs/dinh-nghia-nhan.md khác tính lại",
                    float(match.group(1).replace(",", ".")), agree * 100, tol=0.05)


# ---------------------------------------------------------------------------
# Nhóm 8 — con số đã in trong reports/ phải khớp tính lại từ dữ liệu thật
# ---------------------------------------------------------------------------
def _samples_by_ticker(splits: Dict[str, List[Dict[str, Any]]]
                       ) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for name in SPLITS:
        for sample in splits[name]:
            out[str(sample.get("ticker"))].append(sample)
    return dict(out)


def _recompute_coverage(rows_by_ticker: Dict[str, List[Dict[str, Any]]]
                        ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(min, mean) % độ phủ theo chỉ tiêu — giống `scripts.eda.coverage_matrix`."""
    fields = sorted({k[: -len(SUFFIX)] for rows in rows_by_ticker.values() for r in rows
                     for k in r if k.endswith(SUFFIX)})
    mins: Dict[str, float] = {}
    means: Dict[str, float] = {}
    for field in fields:
        values = [100.0 * sum(1 for r in rows if r.get(field + SUFFIX) is not None) / len(rows)
                  for rows in rows_by_ticker.values()]
        mins[field], means[field] = min(values), sum(values) / len(values)
    return mins, means


def _median(values: List[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    return float(ordered[n // 2]) if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def check_eda_numbers(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]],
                      splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """`reports/results/eda_summary.json` phải khớp tính lại từ prepared + retail-expanded."""
    path = RESULTS_DIR / "eda_summary.json"
    if not a.check(path.exists(), "report_numbers", "thiếu reports/results/eda_summary.json"):
        return
    eda = _read_json(path)
    corpus_block, split_block = eda.get("corpus", {}), eda.get("splits", {})
    n_quarters = sum(len(rows) for rows in rows_by_ticker.values())
    n_samples = sum(len(arr) for arr in splits.values())

    a.eq("report_numbers", "eda_summary: n_companies khác số công ty thật",
         corpus_block.get("n_companies"), len(rows_by_ticker))
    a.eq("report_numbers", "eda_summary: n_quarters khác số quý thật",
         corpus_block.get("n_quarters"), n_quarters)
    a.eq("report_numbers", "eda_summary: n_samples khác số mẫu thật",
         corpus_block.get("n_samples"), n_samples)
    a.eq("report_numbers", "eda_summary: quarters_per_company khác dữ liệu thật",
         corpus_block.get("quarters_per_company"),
         {t: len(v) for t, v in sorted(rows_by_ticker.items())})

    for name in SPLITS:
        arr = splits[name]
        declared = split_block.get(name) or {}
        actual = {"distress": sum(int(s[TARGET]) for s in arr), "total": len(arr)}
        a.eq("report_numbers", f"eda_summary: cân bằng lớp của {name} khác nhãn thật",
             declared, actual, split=name)

    cov_min, cov_mean = _recompute_coverage(rows_by_ticker)
    declared_min = (eda.get("coverage_pct") or {}).get("min_by_field") or {}
    declared_mean = (eda.get("coverage_pct") or {}).get("mean_by_field") or {}
    a.eq("report_numbers", "eda_summary: danh sách chỉ tiêu độ phủ khác dữ liệu thật",
         sorted(declared_min), sorted(cov_min))
    for field, value in cov_min.items():
        a.close("report_numbers", f"eda_summary: độ phủ min của {field} khác tính lại",
                declared_min.get(field), value, field=field)
    for field, value in cov_mean.items():
        a.close("report_numbers", f"eda_summary: độ phủ trung bình của {field} khác tính lại",
                declared_mean.get(field), value, field=field)
    expected_low = sorted([f for f, v in cov_min.items() if v < 50], key=lambda f: cov_min[f])
    a.eq("report_numbers", "eda_summary: low_coverage_fields khác tính lại",
         eda.get("low_coverage_fields"), expected_low)

    # Mục 3.4–3.6 (EDA cơ bản): danh sách biến được mô tả + nhận xét tự động phải khớp định nghĩa
    a.eq("report_numbers", "eda_summary: ratio_stats khác danh sách 14 tỷ số `*_latest`",
         sorted(str(r.get("feature")) for r in (eda.get("ratio_stats") or [])),
         sorted(f"{name}_latest" for name in RATIO_PARTS))
    a.eq("report_numbers", "eda_summary: indicator_stats khác 16 chỉ tiêu BASE_FIELDS",
         sorted(str(r.get("feature")) for r in (eda.get("indicator_stats") or [])),
         sorted(BASE_FIELDS))
    a.eq("report_numbers", "eda_summary: ratio_vs_label phải có đủ 14 tỷ số",
         len(eda.get("ratio_vs_label") or []), len(RATIO_PARTS))
    for field, record in ((r.get("feature"), r) for r in (eda.get("ratio_stats") or [])):
        a.eq("report_numbers", f"eda_summary: {field} thiếu cột mô tả bắt buộc",
             sorted(k for k in ("missing_pct", "min", "q1", "median", "mean", "q3", "max", "skew",
                                "iqr_outlier_pct") if k in record),
             ["iqr_outlier_pct", "max", "mean", "median", "min", "missing_pct", "q1", "q3", "skew"])
    a.check(bool(eda.get("conclusions")), "report_numbers",
            "eda_summary: thiếu phần nhận xét tự động (mục 3.6 của báo cáo)")


    share: Dict[str, float] = {t: sum(int(s[TARGET]) for s in arr) / len(arr)
                               for t, arr in _samples_by_ticker(splits).items()}
    label_block = eda.get("label") or {}
    declared_share = label_block.get("label_share_by_ticker") or {}
    a.eq("report_numbers", "eda_summary: danh sách tỷ lệ nhãn theo công ty khác dữ liệu thật",
         sorted(declared_share), sorted(share))
    for ticker, value in share.items():
        a.close("report_numbers", f"eda_summary: tỷ lệ nhãn = 1 của {ticker} khác tính lại",
                declared_share.get(ticker), value, ticker=ticker)
    a.eq("report_numbers", "eda_summary: companies_all_one khác tính lại",
         label_block.get("companies_all_one"), sorted([t for t, v in share.items() if v == 1.0]))
    a.eq("report_numbers", "eda_summary: companies_all_zero khác tính lại",
         label_block.get("companies_all_zero"), sorted([t for t, v in share.items() if v == 0.0]))

    lengths = [len(s["request"]["history"]) for name in SPLITS for s in splits[name]]
    n_below = sum(1 for h in lengths if h < MIN_HISTORY_QUARTERS)
    actual_hist = {"n_samples": len(lengths), "min": min(lengths), "max": max(lengths),
                   "median": _median(lengths), "n_below_min": n_below,
                   "share_below_min": n_below / len(lengths)}
    declared_hist = eda.get("history_length") or {}
    for key, value in actual_hist.items():
        a.close("report_numbers", f"eda_summary: history_length[{key}] khác tính lại",
                declared_hist.get(key), value, key=key)


def check_label_audit(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]],
                      splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Kết luận "nhãn gốc không tái lập được" phải tính lại ra đúng như đã in."""
    path = RESULTS_DIR / "analysis.json"
    if not a.check(path.exists(), "report_numbers", "thiếu reports/results/analysis.json"):
        return
    declared = _read_json(path).get("label_audit") or {}
    samples = splits["train"] + splits["validation"] + splits["test"]
    index: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]], int]] = {}
    for ticker, rows in rows_by_ticker.items():
        for i, row in enumerate(rows):
            index[f"{ticker}-{row.get('fiscal_year')}Q{row.get('fiscal_quarter')}"] = (row, rows, i)

    def num(row: Dict[str, Any], field: str) -> float | None:
        value = _int_or_none(row.get(field + SUFFIX))
        return None if value is None else float(value)

    rules = {
        "net_income<0": lambda r: (num(r, "net_income") or 0) < 0,
        "operating_income<0": lambda r: (num(r, "operating_income") or 0) < 0,
        "operating_cash_flow<0": lambda r: (num(r, "operating_cash_flow") or 0) < 0,
        "ocf<0 hoặc ni<0": lambda r: ((num(r, "operating_cash_flow") or 0) < 0
                                      or (num(r, "net_income") or 0) < 0),
        "current_liabilities>current_assets": lambda r: (
            (num(r, "current_liabilities") or 0) > (num(r, "current_assets") or 0)),
        "stockholders_equity<0": lambda r: (num(r, "stockholders_equity") or 0) < 0,
        "retained_earnings<0": lambda r: (num(r, "retained_earnings") or 0) < 0,
    }
    target_agree: Dict[str, float] = {}
    history_agree: Dict[str, float] = {}
    for name, fn in rules.items():
        ok_target = ok_history = total = 0
        for sample in samples:
            entry = index.get(str(sample.get("sample_id")))
            if not entry:
                continue
            row, rows, i = entry
            total += 1
            label = int(sample[TARGET])
            ok_target += int(int(bool(fn(row))) == label)
            last = rows[i] if i > 0 else rows[0]
            ok_history += int(int(bool(fn(last))) == label)
        target_agree[name] = ok_target / total if total else float("nan")
        history_agree[name] = ok_history / total if total else float("nan")

    stress = {"stress_signals>=1": [], "stress_signals>=2": []}
    for sample in samples:
        entry = index.get(str(sample.get("sample_id")))
        if not entry:
            continue
        row, rows, i = entry
        active = sum(1 for on in signal_flags(row, rows, i).values() if on)
        label = int(sample[TARGET])
        stress["stress_signals>=1"].append(int(int(active >= 1) == label))
        stress["stress_signals>=2"].append(int(int(active >= 2) == label))
    for name, values in stress.items():
        target_agree[name] = sum(values) / len(values) if values else float("nan")

    a.eq("report_numbers", "analysis.json: label_audit.n_samples khác số mẫu thật",
         declared.get("n_samples"), len(samples))
    for name, value in target_agree.items():
        a.close("report_numbers", f"analysis.json: rule_agreement[{name}] khác tính lại",
                (declared.get("rule_agreement") or {}).get(name), value, rule=name)
    for name, value in history_agree.items():
        a.close("report_numbers",
                f"analysis.json: rule_agreement_last_history[{name}] khác tính lại",
                (declared.get("rule_agreement_last_history") or {}).get(name), value, rule=name)
    a.close("report_numbers", "analysis.json: max_rule_agreement khác tính lại",
            declared.get("max_rule_agreement"), max(target_agree.values()))

    # Phản chứng đã in trong docs/dinh-nghia-nhan.md: WMT-2015Q2 lãi nhưng nhãn = 1
    entry = index.get("WMT-2015Q2")
    if entry is None:
        a.stat("WMT-2015Q2", "không có trong prepared → không kiểm tra được phản chứng")
    else:
        net_income = _int_or_none(entry[0].get("net_income" + SUFFIX))
        a.check(net_income is not None and net_income > 0, "report_numbers",
                "WMT-2015Q2 phải có net_income dương (phản chứng ở docs/dinh-nghia-nhan.md)",
                net_income=net_income)

    # Con số nêu trong tài liệu phải khớp tính lại (không được nhập tay)
    label_doc = DOCS_DIR / "dinh-nghia-nhan.md"
    if label_doc.exists():
        doc_counts = {int(m) for m in re.findall(r"trên (\d+) mẫu",
                                                label_doc.read_text(encoding="utf-8"))}
        a.eq("report_numbers",
             "docs/dinh-nghia-nhan.md: số mẫu của label_audit khác tính lại", doc_counts,
             {len(samples)})
    bao_cao = DOCS_DIR / "BAO-CAO.md"
    if bao_cao.exists():
        simple = [v for name, v in target_agree.items()
                  if not str(name).startswith("stress_signals")]
        match = re.search(r"quy tắc kế toán đơn giản nhất chỉ (\d+)%–(\d+)%",
                          bao_cao.read_text(encoding="utf-8"))
        if match and simple:
            a.eq("report_numbers",
                 "docs/BAO-CAO.md: khoảng khớp của các quy tắc đơn giản khác tính lại",
                 [int(match.group(1)), int(match.group(2))],
                 [round(min(simple) * 100), round(max(simple) * 100)])


def check_provenance_artifact(a: Auditor) -> None:
    """`reports/results/provenance.json` phải tồn tại và báo ĐẠT — bằng chứng dữ liệu THẬT.

    Bộ kiểm chứng `scripts/verify_provenance.py` mở lại snapshot SEC thô để chứng minh từng con số
    đều tra ngược được trong companyfacts (xem mục 3.1 báo cáo).
    """
    path = RESULTS_DIR / "provenance.json"
    if not a.check(path.exists(), "provenance",
                   "thiếu reports/results/provenance.json — chạy `python -m scripts.verify_provenance`"):
        return
    totals = (_read_json(path).get("totals") or {})
    a.check(bool(totals.get("ok")), "provenance",
            "provenance.json: kết luận KHÔNG ĐẠT (còn hash lệch / fact thiếu / quy đổi sai / ô bịa số)",
            totals=totals)
    downloads = _read_json(DATA_DIR / "sec" / "downloads.json")
    a.eq("provenance", "provenance.json: số file SEC đã băm khác số mục trong downloads.json",
         totals.get("n_raw_files_hashed"), len(downloads))
    a.eq("provenance", "provenance.json: hash lệch registry phải = 0",
         totals.get("n_hash_mismatch"), 0)
    a.eq("provenance", "provenance.json: fact không tồn tại trong SEC phải = 0",
         totals.get("n_facts_missing_in_sec"), 0)
    a.eq("provenance", "provenance.json: ô bịa số (không có fact mà vẫn có giá trị) phải = 0",
         totals.get("n_absent_but_filled"), 0)
    a.eq("provenance", "provenance.json: tổng ô dữ liệu phải = 16 chỉ tiêu × 332 quý",
         totals.get("n_cells"), 16 * 332)


def check_model_report_alignment(a: Auditor) -> None:
    """Chặn tái phát 4 lỗi kiểm toán: trộn mô hình, giấu cấu hình đang chạy, thiếu mô hình, câu chữ test.

    1. Mọi phân tích trong `analysis.json` phải thuộc **mô hình đã chốt** (`summary.best_model`) —
       trước đây `scripts/analyze.py` hard-code `"logistic"` nên bảng ngưỡng + permutation importance
       + hình `04_threshold_curves.png` nói về mô hình khác với `best.joblib`.
    2. Báo cáo phải **công bố cấu hình THẬT đang chạy** (mặc định) chứ không để bảng tinh chỉnh gây
       hiểu là đã triển khai cấu hình CV tốt nhất.
    3. Bảng so sánh §6.1 phải có **đủ các mô hình** đã huấn luyện (gồm LightGBM khi môi trường có).
    4. Không được nói "chốt test đúng một lần" (test được chấm cho nhiều hệ thống; điều đúng là test
       KHÔNG tham gia chọn mô hình/ngưỡng).
    """
    summary = _read_json(RESULTS_DIR / "summary.json")
    best = summary.get("best_model")
    analysis = _read_json(RESULTS_DIR / "analysis.json")
    if analysis and best:
        for section in ("importance", "threshold", "calibration"):
            block = analysis.get(section) or {}
            a.eq("report_numbers",
                 f"analysis.json: {section}.model phải là mô hình đã chốt (không trộn mô hình)",
                 block.get("model"), best)
    report = DOCS_DIR / "BAO-CAO.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        flat = re.sub(r"\s+", " ", text)
        a.check("KHÔNG nằm trong mô hình chốt" in flat, "report_numbers",
                "docs/BAO-CAO.md §5.3: thiếu công bố 'cấu hình CV tốt nhất KHÔNG nằm trong mô hình chốt'")
        a.check("Cấu hình THẬT của mô hình đã triển khai" in flat, "report_numbers",
                "docs/BAO-CAO.md §5.3: thiếu công bố cấu hình thật đang chạy")
        a.check("Chốt trên test đúng một lần" not in flat, "report_numbers",
                "docs/BAO-CAO.md: còn câu 'chốt trên test đúng một lần' (không khớp artifact)")
        a.check("không thuộc ba tập" in flat, "report_numbers",
                "docs/BAO-CAO.md §1: thiếu câu giải thích 16 mẫu purge không thuộc 3 tập")
        models_in_registry = [m for m in ("logistic", "random_forest", "hist_gradient_boosting",
                                          "lightgbm") if m in _model_registry()]
        missing = [m for m in models_in_registry if f"model[{m}]" not in text]
        a.eq("report_numbers",
             "docs/BAO-CAO.md: thiếu mô hình trong bảng so sánh (mọi mô hình đã huấn luyện phải có)",
             missing, [])


def _model_registry() -> set:
    """Tên các mô hình mà môi trường hiện tại chạy được (đọc từ `forecasting.models`)."""
    try:
        from forecasting.models import MODEL_REGISTRY

        return set(MODEL_REGISTRY)
    except Exception:  # pragma: no cover - thiếu phụ thuộc tuỳ chọn
        return {"logistic", "random_forest", "hist_gradient_boosting"}


def check_docs_text_consistency(a: Auditor) -> None:
    """Câu chữ trong `docs/slide.md` và `docs/BAO-CAO.md` phải khớp artifact.

    Vì sao cần: hai câu dưới đây từng bị **hard-code** và lệch khỏi kết quả chạy lại (báo "tất cả
    là FN, precision = 1.000" trong khi có 1 FP; báo "Logistic được chọn" trong khi chốt là Random
    Forest). Kiểm tra này chặn tái phát: mọi con số phải suy ra từ `analysis.json` / `summary.json`.
    """
    analysis = _read_json(RESULTS_DIR / "analysis.json")
    summary = _read_json(RESULTS_DIR / "summary.json")
    errors = analysis.get("error_cases") or []
    n_fn = sum(1 for e in errors
               if int(e.get("actual", -1)) == 1 and int(e.get("predicted", -1)) == 0)
    n_fp = sum(1 for e in errors
               if int(e.get("actual", -1)) == 0 and int(e.get("predicted", -1)) == 1)
    best = summary.get("best_model")
    pretty = {"logistic": "Logistic Regression", "random_forest": "Random Forest",
              "hist_gradient_boosting": "HistGradientBoosting", "xgboost": "XGBoost"}.get(best, best)

    slide_path, report_path = DOCS_DIR / "slide.md", DOCS_DIR / "BAO-CAO.md"
    slide = slide_path.read_text(encoding="utf-8") if slide_path.exists() else ""
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    # Câu chữ trong báo cáo được ghép từ nhiều dòng nguồn ⇒ chuẩn hoá khoảng trắng trước khi so.
    flat_report = re.sub(r"\s+", " ", report)
    if slide:
        a.check(f"{n_fn} FN + {n_fp} FP" in slide, "report_numbers",
                "docs/slide.md: số FN/FP không khớp error_cases của analysis.json",
                declared_fn=n_fn, declared_fp=n_fp, n_errors=len(errors))
        a.check("tất cả là FN" not in slide and "precision = 1.000" not in slide, "report_numbers",
                "docs/slide.md: còn câu khẳng định cũ 'tất cả là FN / precision = 1.000'")
        a.check(best is None or pretty in slide, "report_numbers",
                "docs/slide.md: không nhắc đúng mô hình được chốt (summary.best_model)",
                best_model=best)
    if report:
        match = re.search(r"Tổng số mẫu sai trên test: \*\*(\d+) / (\d+)\*\* — gồm \*\*(\d+) FN\*\* "
                          r"\(bỏ sót suy giảm\) và \*\*(\d+) FP\*\*", flat_report)
        a.check(match is not None, "report_numbers",
                "docs/BAO-CAO.md §7.5: không tìm thấy câu 'Tổng số mẫu sai … FN … FP' "
                "(câu chữ phải suy ra từ error_cases)")
        if match:
            a.eq("report_numbers",
                 "docs/BAO-CAO.md §7.5: số mẫu sai / FN / FP khác error_cases",
                 [int(g) for g in match.groups()],
                 [len(errors), int(_read_json(RESULTS_DIR / "test_evaluation.json").get(
                     "test_metrics", {}).get("n") or 0), n_fn, n_fp])
        a.check("## 11. Tài liệu tham khảo" in report, "report_numbers",
                "docs/BAO-CAO.md: thiếu mục 11 (Tài liệu tham khảo)")
        a.check("không có FP** (precision = 1.000)" not in flat_report, "report_numbers",
                "docs/BAO-CAO.md: còn câu khẳng định cũ 'không có FP (precision = 1.000)'")
    defense = DOCS_DIR / "slide-bao-ve.md"
    checklist = DOCS_DIR / "checklist-doi-chieu-yeu-cau.md"
    a.check(defense.exists() and defense.stat().st_size > 2000, "report_numbers",
            "thiếu docs/slide-bao-ve.md (dàn slide bảo vệ để xuất .pptx)")
    a.check(checklist.exists() and checklist.stat().st_size > 2000, "report_numbers",
            "thiếu docs/checklist-doi-chieu-yeu-cau.md (đối chiếu tiêu chí)")


def check_test_artifacts(a: Auditor, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """`test_evaluation.json`, `test_predictions.csv` và `best.joblib` phải khớp nhau."""
    te_path = RESULTS_DIR / "test_evaluation.json"
    if not a.check(te_path.exists(), "report_numbers",
                   "thiếu reports/results/test_evaluation.json"):
        return
    te = _read_json(te_path)
    test = splits["test"]
    n = len(test)
    labels = [int(s[TARGET]) for s in test]
    metrics = te.get("test_metrics") or {}
    operating = metrics.get("operating") or {}
    cm = metrics.get("confusion_at_operating") or {}
    tn, fp = int(cm.get("tn") or 0), int(cm.get("fp") or 0)
    fn, tp = int(cm.get("fn") or 0), int(cm.get("tp") or 0)

    a.eq("report_numbers", "test_evaluation: n khác số mẫu test", metrics.get("n"), n)
    a.close("report_numbers", "test_evaluation: observed_distress khác tỷ lệ nhãn 1 thật",
            metrics.get("observed_distress"), sum(labels) / n)
    a.check(tn + fp + fn + tp == n, "report_numbers",
            "test_evaluation: confusion matrix không cộng đủ số mẫu test",
            confusion=[tn, fp, fn, tp])
    a.close("report_numbers", "test_evaluation: ngưỡng vận hành khác ngưỡng chung",
            operating.get("threshold"), te.get("threshold"))
    if tp + fn:
        a.close("report_numbers", "test_evaluation: recall không khớp confusion matrix",
                operating.get("recall"), tp / (tp + fn))
    if tp + fp:
        a.close("report_numbers", "test_evaluation: precision không khớp confusion matrix",
                operating.get("precision"), tp / (tp + fp))
    if tn + fp:
        a.close("report_numbers", "test_evaluation: specificity không khớp confusion matrix",
                operating.get("specificity"), tn / (tn + fp))
    a.close("report_numbers", "test_evaluation: accuracy không khớp confusion matrix",
            operating.get("accuracy"), (tp + tn) / n)
    a.close("report_numbers", "test_evaluation: AUROC khác AUROC của đường cong ROC",
            metrics.get("auroc"), (te.get("curve_stats") or {}).get("auroc_from_curve"))
    a.check(te.get("n_misclassified") == len(te.get("misclassified") or []) == fp + fn,
            "report_numbers", "test_evaluation: số mẫu sai khác confusion matrix",
            declared=te.get("n_misclassified"), n_listed=len(te.get("misclassified") or []),
            from_confusion=fp + fn)

    by_sample = {str(s.get("sample_id")): s for s in test}
    rows: List[Dict[str, str]] = []
    csv_path = RESULTS_DIR / "test_predictions.csv"
    if a.check(csv_path.exists(), "report_numbers",
               "thiếu reports/results/test_predictions.csv"):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        a.eq("report_numbers", "test_predictions.csv: số dòng khác số mẫu test", len(rows), n)
        a.check({r.get("sample_id") for r in rows} == set(by_sample), "report_numbers",
                "test_predictions.csv: tập sample_id khác tập test",
                missing=sorted(set(by_sample) - {r.get("sample_id") for r in rows})[:5])
        threshold = float(te.get("threshold") or 0.0)
        for row in rows:
            sample = by_sample.get(str(row.get("sample_id")))
            if sample is None:
                continue
            request = sample.get("request") or {}
            sid = row.get("sample_id")
            a.eq("report_numbers", f"{sid}: nhãn trong CSV khác nhãn trong prepared",
                 int(row["actual"]), int(sample[TARGET]))
            a.eq("report_numbers", f"{sid}: as_of trong CSV khác prepared",
                 row.get("as_of"), request.get("as_of"))
            a.eq("report_numbers", f"{sid}: kỳ target trong CSV khác prepared",
                 row.get("target_quarter"),
                 f"{request.get('target_period_start')} -> {request.get('target_period_end')}")
            a.eq("report_numbers", f"{sid}: predicted không khớp probability + ngưỡng",
                 int(row["predicted"]), int(float(row["probability"]) >= threshold))
        for item in te.get("misclassified") or []:
            sid = str(item.get("sample_id"))
            row = next((r for r in rows if r.get("sample_id") == sid), None)
            if row is None:
                a.check(False, "report_numbers",
                        f"{sid}: trong danh sách sai nhưng thiếu ở test_predictions.csv")
                continue
            a.check(_close(row.get("probability"), item.get("probability"), 1e-12)
                    and int(row["predicted"]) == int(item.get("predicted"))
                    and int(row["actual"]) == int(item.get("actual")), "report_numbers",
                    f"{sid}: dòng CSV khác danh sách misclassified của test_evaluation.json",
                    declared=item, csv=dict(row))


    # Xác suất trong CSV phải đúng bằng xác suất do mô hình đã lưu sinh ra
    model_path = MODELS_DIR / "best.joblib"
    if a.check(model_path.exists(), "report_numbers", "thiếu reports/models/best.joblib"):
        import joblib

        from forecasting.features import build_feature_matrix
        from forecasting.models import predict_proba

        artifact = joblib.load(model_path)
        a.eq("report_numbers", "best.joblib: mô hình khác mô hình trong test_evaluation.json",
             artifact.get("name"), te.get("model"))
        proba = [float(p) for p in predict_proba(artifact["model"], build_feature_matrix(test))]
        a.eq("report_numbers", "best.joblib: số xác suất khác số mẫu test", len(proba), n)
        for sample, value in zip(test, proba):
            sid = str(sample.get("sample_id"))
            row = next((r for r in rows if r.get("sample_id") == sid), None)
            if row is None:
                a.check(False, "report_numbers",
                        f"{sid}: thiếu ở test_predictions.csv (không so được với mô hình)")
                continue
            a.close("report_numbers",
                    f"{sid}: xác suất trong CSV khác xác suất mô hình best.joblib",
                    row.get("probability"), value, tol=1e-12)


def check_baselines(a: Auditor, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Baseline ticker-prior phải bằng đúng tỷ lệ nhãn 1 của từng công ty trên train."""
    path = RESULTS_DIR / "baselines.json"
    if not a.check(path.exists(), "report_numbers", "thiếu reports/results/baselines.json"):
        return
    baselines = _read_json(path)
    priors: Dict[str, List[int]] = defaultdict(list)
    for sample in splits["train"]:
        priors[str(sample.get("ticker"))].append(int(sample[TARGET]))
    computed = {t: sum(v) / len(v) for t, v in priors.items()}
    declared = baselines.get("ticker_prior_train") or {}
    a.eq("report_numbers", "baselines.json: ticker_prior_train khác công ty của train",
         sorted(declared), sorted(computed))
    for ticker, value in computed.items():
        a.close("report_numbers",
                f"baselines.json: ticker_prior_train[{ticker}] khác tỷ lệ nhãn 1 thật trên train",
                declared.get(ticker), value, ticker=ticker)

    te_path = RESULTS_DIR / "test_evaluation.json"
    if te_path.exists():
        te = _read_json(te_path)
        name = f"model[{te.get('model')}]"
        row = next((r for r in baselines.get("rows") or [] if r.get("baseline") == name), None)
        if a.check(row is not None, "report_numbers",
                   f"baselines.json: thiếu dòng {name} để đối chiếu với test_evaluation.json"):
            assert row is not None
            a.close("report_numbers",
                    f"baselines.json: AUROC test của {name} khác test_evaluation.json",
                    (row.get("test") or {}).get("auroc"),
                    (te.get("test_metrics") or {}).get("auroc"))


def check_balance_identity(a: Auditor, rows_by_ticker: Dict[str, List[Dict[str, Any]]]) -> None:
    """Khi quý có tag `liabilities`, đẳng thức A = L + E phải (gần) đúng.

    Vì sao cần: `forecasting/features.py` SUY RA nợ phải trả từ `total_assets - stockholders_equity`
    khi thiếu tag. Nếu đẳng thức sai ở dữ liệu thật thì cách suy ra này không hợp lệ.
    """
    checked = mismatched = 0
    worst = 0.0
    for ticker, rows in rows_by_ticker.items():
        for row in rows:
            liabilities = _int_or_none(row.get("liabilities" + SUFFIX))
            assets = _int_or_none(row.get("total_assets" + SUFFIX))
            equity = _int_or_none(row.get("stockholders_equity" + SUFFIX))
            if liabilities is None or assets is None or equity is None:
                continue
            checked += 1
            diff = liabilities - (assets - equity)
            relative = abs(diff) / abs(assets) if assets else float("inf")
            worst = max(worst, relative)
            if relative > 0.005:  # dung sai 0,5% tài sản (restatement/rounding nhỏ thì bỏ qua)
                mismatched += 1
                a.check(False, "retail_expanded",
                        f"{ticker}-{row.get('fiscal_year')}Q{row.get('fiscal_quarter')}: "
                        f"liabilities lệch khỏi A = L + E quá 0,5% tài sản",
                        diff_vnd=diff, relative=relative)
    a.stat("balance_identity[quý kiểm|quý lệch>0.5%|lệch tương đối lớn nhất]",
           [checked, mismatched, round(worst, 6)])


def check_advanced_artifacts(a: Auditor, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """Đối soát các artifact của mục 9: tiền xử lý, lệch lớp thật, tìm kiếm, SHAP, kiểm định, nhãn.

    Mục đích: các con số "nâng cấp để đạt mức Xuất sắc" cũng phải qua lưới an toàn như mọi số khác —
    sai số efficiency của KernelSHAP phải thật nhỏ, p-value trong [0, 1], sampler không được chạm tập
    test của fold, và mọi định nghĩa nhãn phải dựng được split đủ 4 tập.
    """
    from forecasting.features import feature_names

    n_features = len(feature_names())

    # 1) Thí nghiệm tiền xử lý
    path = RESULTS_DIR / "preprocessing_experiment.json"
    if a.check(path.exists(), "report_numbers", "thiếu reports/results/preprocessing_experiment.json"):
        prep = _read_json(path)
        rows = prep.get("rows") or []
        a.eq("report_numbers", "preprocessing_experiment: số cấu hình khác protocol.n_variants",
             len(rows), (prep.get("protocol") or {}).get("n_variants"))
        a.check(bool(rows), "report_numbers", "preprocessing_experiment: không có cấu hình nào")
        a.check(all({"model", "scaler", "winsorize"} <= set(r) for r in rows), "report_numbers",
                "preprocessing_experiment: thiếu khoá model/scaler/winsorize ở một cấu hình")
        shares = [r.get("n_clipped_val_pct") for r in rows if r.get("n_clipped_val_pct") is not None]
        a.check(all(0.0 <= s <= 100.0 for s in shares), "report_numbers",
                "preprocessing_experiment: % giá trị bị clip nằm ngoài [0, 100]")
        referenced = (prep.get("protocol") or {}).get("reference") or {}
        a.check(any(all(r.get(k) == v for k, v in referenced.items()) for r in rows), "report_numbers",
                "preprocessing_experiment: không tìm thấy cấu hình tham chiếu trong bảng")

    # 2) SHAP (KernelSHAP tự cài đặt)
    path = RESULTS_DIR / "shap.json"
    if a.check(path.exists(), "report_numbers", "thiếu reports/results/shap.json"):
        shap = _read_json(path)
        a.eq("report_numbers", "shap: n_features khác feature_names()", shap.get("n_features"), n_features)
        gap = (shap.get("self_check") or {}).get("max_abs_efficiency_gap")
        a.check(gap is not None and gap < 1e-6, "report_numbers",
                f"shap: sai số efficiency quá lớn ({gap}); KernelSHAP phải thoả Σφ + E[f] = f(x)")
        a.check(int(shap.get("n_explained") or 0) > 0, "report_numbers",
                "shap: không giải thích mẫu nào")
        top = shap.get("importance_top") or []
        names = set(feature_names())
        a.check(all(r.get("feature") in names for r in top), "report_numbers",
                "shap: có feature trong bảng quan trọng không thuộc feature_names()")
        summary_path = RESULTS_DIR / "summary.json"
        if summary_path.exists():
            a.eq("report_numbers", "shap: mô hình được giải thích khác mô hình đã chốt",
                 shap.get("model"), _read_json(summary_path).get("best_model"))

    # 3) Kiểm định ý nghĩa thống kê
    path = RESULTS_DIR / "significance.json"
    if a.check(path.exists(), "report_numbers", "thiếu reports/results/significance.json"):
        sig = _read_json(path)
        a.eq("report_numbers", "significance: số mẫu test khác số mẫu test thật",
             sig.get("n_samples"), len(splits["test"]))
        systems = sig.get("systems") or {}
        a.check("ticker_prior" in systems, "report_numbers",
                "significance: thiếu baseline ticker_prior trong bảng so sánh")
        for name, metrics in systems.items():
            a.check(0.0 <= float(metrics.get("average_precision", -1)) <= 1.0, "report_numbers",
                    f"significance: AP của {name} nằm ngoài [0, 1]")
        for pair in sig.get("pairs") or []:
            p_value = (pair.get("delong_auroc") or {}).get("p_value")
            if p_value is not None:
                a.check(0.0 <= float(p_value) <= 1.0, "report_numbers",
                        "significance: p-value của DeLong nằm ngoài [0, 1]")
            a.check(bool((pair.get("delong_auroc") or {}).get("auc_matches_sklearn")), "report_numbers",
                    "significance: AUROC cài trong module không khớp sklearn.roc_auc_score")
            boot = pair.get("bootstrap_ap") or {}
            a.check(boot.get("ci95_lower") is not None and boot.get("ci95_upper") is not None,
                    "report_numbers", "significance: thiếu khoảng tin cậy của ΔAP")

    # 4) Kỹ thuật xử lý lệch lớp trên dữ liệu thật
    path = RESULTS_DIR / "imbalance_real.json"
    if a.check(path.exists(), "report_numbers", "thiếu reports/results/imbalance_real.json"):
        imb = _read_json(path)
        rows = imb.get("rows") or []
        a.check(len(rows) >= 3, "report_numbers",
                f"imbalance_real: quá ít kỹ thuật được đo ({len(rows)})")
        leaked = [r["name"] for r in rows if r.get("fold_test_untouched") is False]
        a.check(not leaked, "report_numbers",
                f"imbalance_real: sampler chạm tập test của fold ở {leaked}")
        a.check(any(r["name"] == "none" for r in rows), "report_numbers",
                "imbalance_real: thiếu đối chứng 'none' để so sánh")
        a.eq("report_numbers", "imbalance_real: số mẫu OOF khác số mẫu train+validation",
             max((r.get("n_oof") or 0) for r in rows),
             len(splits["train"]) + len(splits["validation"]))

    # 5) Sổ thực nghiệm của tìm kiếm siêu tham số
    path = RESULTS_DIR / "search.json"
    ledger = RESULTS_DIR / "runs.csv"
    if a.check(path.exists(), "report_numbers", "thiếu reports/results/search.json"):
        search = _read_json(path)
        declared = (search.get("ledger") or {}).get("n_rows")
        a.check(ledger.exists(), "report_numbers", "thiếu sổ thực nghiệm reports/results/runs.csv")
        if ledger.exists():
            with open(ledger, "r", encoding="utf-8", newline="") as handle:
                n_rows = sum(1 for _ in csv.DictReader(handle))
            a.eq("report_numbers", "search: số dòng runs.csv khác ledger.n_rows", n_rows, declared)
        a.check(all(r.get("best_cv_average_precision") is not None
                    for r in search.get("rows") or []), "report_numbers",
                "search: có mô hình không tìm được cấu hình hợp lệ nào")

    # 6) Độ nhạy theo định nghĩa nhãn
    path = RESULTS_DIR / "label_sensitivity.json"
    if a.check(path.exists(), "report_numbers", "thiếu reports/results/label_sensitivity.json"):
        labels = _read_json(path)
        rows = labels.get("rows") or []
        expected = {"train": len(splits["train"]), "validation": len(splits["validation"]),
                    "test": len(splits["test"]), "purged": len(splits["purged"])}
        for row in rows:
            a.eq("report_numbers", f"label_sensitivity[{row['rule']}]: số mẫu mỗi tập khác split gốc",
                 row.get("counts"), expected)
            rate = (row.get("positive_rate") or {}).get("test")
            a.check(rate is None or 0.0 <= float(rate) <= 1.0, "report_numbers",
                    f"label_sensitivity[{row['rule']}]: tỉ lệ dương ngoài [0, 1]")
        a.check(len([r for r in rows if r["rule"] != "original"]) >= 2, "report_numbers",
                "label_sensitivity: cần ≥ 2 định nghĩa nhãn tái lập được để kết luận độ nhạy")
        a.check(bool(labels.get("conclusions")), "report_numbers",
                "label_sensitivity: thiếu phần kết luận tự động")


def check_feature_artifacts(a: Auditor) -> None:
    """Số cột feature phải khớp giữa code (`features.py`), artifact và báo cáo."""
    from forecasting.features import feature_names

    n_features = len(feature_names())
    a.stat("n_features_features_py", n_features)
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        a.eq("report_numbers", "summary.json: n_features khác số cột của features.py",
             _read_json(summary_path).get("n_features"), n_features)
    analysis_path = RESULTS_DIR / "analysis.json"
    if analysis_path.exists():
        a.eq("report_numbers", "analysis.json: importance.n_features khác số cột của features.py",
             (_read_json(analysis_path).get("importance") or {}).get("n_features"), n_features)
    doc_path = DOCS_DIR / "BAO-CAO.md"
    if doc_path.exists():
        match = re.search(r"`feature_names\(\)` — (\d+) cột", doc_path.read_text(encoding="utf-8"))
        if match:
            a.eq("report_numbers", "docs/BAO-CAO.md: số cột feature khác features.py",
                 int(match.group(1)), n_features)


def check_class_balance(a: Auditor, splits: Dict[str, List[Dict[str, Any]]]) -> None:
    """`reports/results/class_balance.json` phải khớp nhãn thật trong prepared (đếm lại + IR)."""
    path = RESULTS_DIR / "class_balance.json"
    if not a.check(path.exists(), "report_numbers", "thiếu reports/results/class_balance.json"):
        return
    declared = _read_json(path)
    for name in SPLITS:
        labels = [int(s[TARGET]) for s in splits[name]]
        positive = sum(labels)
        block = (declared.get("per_split") or {}).get(name) or {}
        a.eq("report_numbers", f"class_balance: counts[{name}] khác nhãn thật",
             block.get("counts"), {"1": positive, "0": len(labels) - positive}, split=name)
        minority = min(positive, len(labels) - positive)
        ratio = (max(positive, len(labels) - positive) / minority) if minority else float("inf")
        a.close("report_numbers", f"class_balance: imbalance_ratio[{name}] khác tính lại",
                block.get("imbalance_ratio"), ratio, split=name)

    pool = [s for name in SPLITS for s in splits[name]]
    positive = sum(int(s[TARGET]) for s in pool)
    total = len(pool)
    minority_pct = 100.0 * min(positive, total - positive) / total
    overall = declared.get("overall") or {}
    a.eq("report_numbers", "class_balance: counts toàn corpus khác nhãn thật",
         overall.get("counts"), {"1": positive, "0": total - positive})
    a.close("report_numbers", "class_balance: % lớp thiểu số toàn corpus khác tính lại",
            overall.get("minority_percent"), minority_pct)
    want_level = ("balanced" if minority_pct >= 40.0
                  else "slightly_imbalanced" if minority_pct >= 20.0 else "severely_imbalanced")
    a.eq("report_numbers", "class_balance: mức mất cân bằng toàn corpus khác ngưỡng công bố",
         overall.get("level"), want_level, minority_percent=minority_pct)
    a.stat("class_balance[IR toàn corpus | % thiểu số | mức]",
           [overall.get("imbalance_ratio"), round(minority_pct, 2), overall.get("level")])


# ---------------------------------------------------------------------------
# Chạy toàn bộ + ghi báo cáo
# ---------------------------------------------------------------------------
def run(tickers: Sequence[str] | None = None, skip_sec: bool = False,
        write: bool = True) -> Dict[str, Any]:
    """Chạy toàn bộ nhóm kiểm tra; trả báo cáo (và ghi artifact khi `write=True`)."""
    a = Auditor()
    if skip_sec:
        a.stat("sec_snapshots", "bỏ qua (--skip-sec)")
    else:
        check_sec_snapshots(a, tickers)
    rows_by_ticker = check_retail_expanded(a, tickers)
    check_balance_identity(a, rows_by_ticker)
    splits = check_prepared(a, rows_by_ticker)
    check_manifest(a, rows_by_ticker, splits)
    check_corpus(a, rows_by_ticker)
    check_prepared_rule(a, rows_by_ticker, splits)
    check_eda_numbers(a, rows_by_ticker, splits)
    check_label_audit(a, rows_by_ticker, splits)
    check_test_artifacts(a, splits)
    check_baselines(a, splits)
    check_class_balance(a, splits)
    check_feature_artifacts(a)
    check_advanced_artifacts(a, splits)
    check_docs_text_consistency(a)
    check_model_report_alignment(a)
    check_provenance_artifact(a)

    report = {"generated_by": "scripts.audit_data", **a.summary()}
    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "data_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=float) + "\n",
            encoding="utf-8")
        (RESULTS_DIR / "data_audit.md").write_text(a.markdown(), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default="",
                        help="Chỉ kiểm tra các ticker này, phân tách bằng dấu phẩy.")
    parser.add_argument("--skip-sec", action="store_true",
                        help="Bỏ qua nhóm snapshot SEC (nhanh hơn, không đọc ~60 MB JSON).")
    parser.add_argument("--no-write", action="store_true",
                        help="Không ghi reports/results/data_audit.*")
    args = parser.parse_args(argv)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    report = run(tickers=tickers, skip_sec=args.skip_sec, write=not args.no_write)

    print("=== Đối soát dữ liệu trong repo với số liệu thật SEC ===")
    print(f"  Số phép kiểm tra: {report['n_checks']} — số phát hiện: {report['n_issues']}")
    for group, count in report["issues_by_group"].items():
        print(f"  * {group}: {count}")
    for key in ("retail_quarters", "legacy_10_tickers", "prepared_counts", "expected_counts",
                "rule_label_counts", "rule_agreement_recomputed",
                "validation_fit_before[manifest|max_train_pub|min_val_as_of]",
                "final_fit_before[manifest|max_train_val_pub|min_test_as_of]"):
        if key in report["stats"]:
            print(f"  - {key}: {report['stats'][key]}")
    for group, items in report["issues"].items():
        print(f"\n[{group}] {len(items)} phát hiện (in tối đa 10):")
        for item in items[:10]:
            print("   *", item["message"])
    return 1 if report["n_issues"] else 0


if __name__ == "__main__":
    sys.exit(main())

