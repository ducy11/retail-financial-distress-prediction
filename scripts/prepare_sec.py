"""Trích 16 chỉ tiêu XBRL từ snapshot SEC → bảng chỉ tiêu VND (PORT của pipeline gốc).

Lệnh:
    python -m scripts.prepare_sec                 # tái tạo + so với dữ liệu đã công bố
    python -m scripts.prepare_sec --verify        # chỉ so, không ghi bảng tái tạo
    python -m scripts.prepare_sec --ticker WMT --out data/retail-expanded-rebuilt

Vì sao module này tồn tại: bảng chỉ tiêu trong `data/retail-expanded` do pipeline gốc sinh ở NGOÀI
repo nên trước đây bước ETL không tái lập được. Bản port tái tạo TỪNG Ô từ
`data/sec/raw/*-companyfacts.json` bằng đúng những gì pipeline gốc dùng (đọc từ dữ liệu, không đoán):

1. **Thứ tự ưu tiên tag** mỗi chỉ tiêu (`TAG_PRIORITY`) — suy ra từ `sources` của dữ liệu đã công bố,
   sau đó kiểm chứng lại bằng `--verify`;
2. **Phương pháp kỳ** (4 cách; `absent` = ô trống):
   - `instant` — số dư cuối kỳ (`end == period_end`, không có `start`): tài sản/nợ/vốn/tồn kho…;
   - `reported_quarter` — số phát sinh ĐÚNG quý được công bố (`start == period_start`);
   - `reported_first_quarter` — Q1 lấy luỹ kế quý 1 (luỹ kế Q1 chính là số của quý 1);
   - `current_ytd_minus_previous_ytd` — không có fact quý riêng ⇒ luỹ kế kỳ này trừ luỹ kế kỳ trước
     (đúng cách doanh nghiệp Mỹ công bố dòng tiền/lợi nhuận theo luỹ kế);
   - `absent` — không có fact ⇒ `null` (thà `null` còn hơn "điền số cho đủ"; xem `scripts.audit_data`);
3. **Bản công bố SỚM NHẤT** của mỗi kỳ (`min(filed)`) ⇒ mô phỏng đúng thông tin có tại `available_on`
   (nhất quán với ràng buộc chống rò rỉ của `forecasting.features`). Quy đổi minh hoạ USD → VND
   × 25.000 (khớp `fx_policy` của dữ liệu gốc).

`--verify` so từng ô với giá trị đã công bố và ghi `reports/results/etl_verify.{json,md}`. Mặc định
KHÔNG ghi đè dữ liệu đang dùng cho báo cáo: kết quả tái tạo ghi sang `data/retail-expanded-rebuilt/`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from forecasting.config import DATA_DIR, RETAIL_DIR, ensure_utf8_stdio

RAW_DIR = DATA_DIR / "sec" / "raw"
RESULTS_DIR = DATA_DIR.parent / "reports" / "results"
DEFAULT_OUT = DATA_DIR / "retail-expanded-rebuilt"

#: Quy đổi minh hoạ (khớp `fx_policy.vnd_per_usd` của dữ liệu gốc).
FX_VND_PER_USD = 25000

#: Thứ tự ưu tiên tag XBRL cho 16 chỉ tiêu — ĐÚNG danh sách pipeline gốc đã dùng (suy ra từ `sources`
#: của dữ liệu đã công bố, xếp theo số lần được chọn): tái tạo đúng bảng gốc, kể cả các ô `absent`.
TAG_PRIORITY: Dict[str, List[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cost_of_sales": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "inventory": ["InventoryNet"],
    "selling_general_admin": ["SellingGeneralAndAdministrativeExpense"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "total_assets": ["Assets"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "operating_income": ["OperatingIncomeLoss"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "net_income": ["NetIncomeLoss"],
    "stockholders_equity": ["StockholdersEquity"],
    "liabilities": ["Liabilities"],
    "receivables": ["ReceivablesNetCurrent", "AccountsReceivableNetCurrent"],
    "short_term_investments": ["ShortTermInvestments"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
}

#: Tag DỰ PHÒNG (tùy chọn, bật bằng `--extended`): dùng khi MỞ RỘNG sang công ty mới mà tag gốc không
#: có — ví dụ doanh nghiệp chỉ công bố `NetIncomeLossAvailableToCommonStockholdersBasic`, hoặc báo cáo
#: dòng tiền bằng tag "continuing operations". Bật sẽ tăng độ phủ nhưng khác bảng gốc ở những ô mà
#: pipeline cũ để trống ⇒ dùng cho dữ liệu MỚI, không dùng để đối chiếu bảng đang có.
TAG_PRIORITY_EXTENDED: Dict[str, List[str]] = {
    "revenue": ["Revenues", "SalesRevenueGoodsNet"],
    "cost_of_sales": ["CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "inventory": ["InventoryFinishedGoods", "InventoryGross"],
    "selling_general_admin": ["SellingAndMarketingExpense", "GeneralAndAdministrativeExpense"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "cash_and_equivalents": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "net_income": ["ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "stockholders_equity": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "receivables": ["AccountsNotesAndLoansReceivableNetCurrent"],
}

#: Bật/tắt tag dự phòng (đặt qua CLI `--extended`; mặc định TẮT để khớp bảng gốc).
USE_EXTENDED_TAGS = False


def tags_for(indicator: str, extended: Optional[bool] = None) -> List[str]:
    """Tag theo ưu tiên cho một chỉ tiêu (kèm tag mở rộng nếu đang bật)."""
    use = USE_EXTENDED_TAGS if extended is None else extended
    return TAG_PRIORITY[indicator] + (TAG_PRIORITY_EXTENDED.get(indicator, []) if use else [])


def all_tags(extended: Optional[bool] = None) -> List[str]:
    """Mọi tag cần đánh chỉ mục trong snapshot (phục vụ tra fact theo kỳ)."""
    return list(dict.fromkeys(tag for indicator in INDICATORS
                              for tag in tags_for(indicator, extended)))

#: 10 chỉ tiêu là SỐ DƯ cuối kỳ; 6 chỉ tiêu còn lại là SỐ PHÁT SINH trong kỳ.
INSTANT_INDICATORS = {"total_assets", "current_assets", "current_liabilities", "inventory",
                      "cash_and_equivalents", "stockholders_equity", "liabilities", "receivables",
                      "short_term_investments", "retained_earnings"}
DURATION_INDICATORS = {"revenue", "cost_of_sales", "selling_general_admin", "operating_cash_flow",
                       "operating_income", "net_income"}
INDICATORS: List[str] = list(TAG_PRIORITY)

#: Chế độ dự phòng khi một ô chưa có bản công bố nào trước `available_on`:
#: `False` (mặc định) = để trống `absent` — đúng bản gốc và đúng thời điểm ra quyết định;
#: `True` (CLI `--fill-late`) = dùng bản công bố muộn để lấp ô (tăng độ phủ, khác bảng gốc).
ALLOW_LATE_FALLBACK = False



def _load_raw(ticker: str, raw_dir: Path = RAW_DIR) -> Dict[str, Any]:
    """Snapshot companyfacts của một ticker ({} nếu chưa crawl)."""
    path = raw_dir / f"{ticker}-companyfacts.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def usd_facts(doc: Dict[str, Any], tag: str) -> List[Dict[str, Any]]:
    """Mọi fact USD của `tag` trong snapshot, sắp theo (end, filed) tăng dần."""
    out: List[Dict[str, Any]] = []
    for _taxonomy, tags in (doc.get("facts") or {}).items():
        body = tags.get(tag)
        if not body:
            continue
        for unit, entries in (body.get("units") or {}).items():
            if unit != "USD":
                continue
            out += [dict(entry) for entry in entries if entry.get("val") is not None]
    return sorted(out, key=lambda e: (str(e.get("end")), str(e.get("filed"))))


def _published(candidates: List[Dict[str, Any]], available_on: Optional[str],
               allow_late_fallback: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """Bản công bố SỚM NHẤT của một kỳ, ưu tiên bản đã có trước `available_on` (mặc định: module).

    `ALLOW_LATE_FALLBACK=True`: nếu chưa có bản nào trước `available_on` thì dùng bản muộn (lấp ô mà
    pipeline gốc để trống); `False`: giữ `absent` cho trung thực với thời điểm ra quyết định.
    Hai chế độ được so bằng `--strict` (xem `reports/results/etl_verify.md`).
    """
    allow = ALLOW_LATE_FALLBACK if allow_late_fallback is None else allow_late_fallback
    usable = [c for c in candidates if available_on is None or str(c.get("filed")) <= available_on]
    if not usable:
        if not allow:
            return None
        usable = candidates
    return min(usable, key=lambda c: (str(c.get("filed")), str(c.get("accn")))) if usable else None


def derive_cell(facts: Dict[str, List[Dict[str, Any]]], indicator: str, period_start: str,
                period_end: str, available_on: Optional[str] = None,
                fiscal_quarter: Optional[int] = None) -> Tuple[Optional[int], Dict[str, Any]]:
    """Tái tạo MỘT ô từ snapshot: (giá trị USD, block `sources`) theo 4 phương pháp của gốc.

    Thứ tự thử: tag ưu tiên → (số dư cuối kỳ | fact đúng quý | luỹ kế trừ luỹ kế) → `absent`.
    Không bao giờ "đoán" giá trị: thiếu fact thì trả `None` + method `absent`.
    """
    for tag in tags_for(indicator):
        candidates = facts.get(tag) or []
        if indicator in INSTANT_INDICATORS:
            chosen = _published([f for f in candidates
                                 if not f.get("start") and f.get("end") == period_end], available_on)
            if chosen:
                chosen["tag"] = tag
                return int(chosen["val"]), {"method": "instant", "facts": [chosen]}
            continue
        # SỐ PHÁT SINH: ưu tiên fact ĐÚNG quý (start == period_start, end == period_end)
        quarter = _published([f for f in candidates
                              if f.get("start") == period_start and f.get("end") == period_end],
                             available_on)
        if quarter:
            quarter["tag"] = tag
            method = "reported_first_quarter" if fiscal_quarter == 1 else "reported_quarter"
            return int(quarter["val"]), {"method": method, "facts": [quarter]}
        # Không có fact quý riêng ⇒ luỹ kế kỳ này TRỪ luỹ kế kỳ trước (Q4 = FY − 9M)
        ytd = _published([f for f in candidates
                          if f.get("start") and f.get("end") == period_end
                          and str(f.get("start")) < str(period_start)], available_on)
        if ytd:
            same_ytd = [f for f in candidates if f.get("start") == ytd.get("start")
                        and str(f.get("end")) < str(period_end)]
            if same_ytd:
                previous_end = max(str(f["end"]) for f in same_ytd)
                previous = _published([f for f in same_ytd if str(f["end"]) == previous_end],
                                      available_on)
                if previous:
                    ytd["tag"], previous["tag"] = tag, tag
                    return (int(ytd["val"]) - int(previous["val"]),
                            {"method": "current_ytd_minus_previous_ytd",
                             "facts": [ytd, previous]})
    return None, {"method": "absent", "facts": []}


def _source_url(cik: Optional[int], accn: str) -> str:
    """URL lưu trữ EDGAR của một filing (giống dạng `source_url` trong dữ liệu gốc)."""
    return (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"
            f"{accn}-index.html")


def _fact_public(fact: Dict[str, Any], cik: Optional[int]) -> Dict[str, Any]:
    """Fact rút gọn đúng dạng provenance của dữ liệu gốc (kèm `tag` + `source_url`)."""
    accn = str(fact.get("accn"))
    return {"tag": fact.get("tag"), "start": fact.get("start"), "end": fact.get("end"),
            "val": fact.get("val"), "accn": accn, "fy": fact.get("fy"), "fp": fact.get("fp"),
            "form": fact.get("form"), "filed": fact.get("filed"), "frame": fact.get("frame"),
            "source_url": _source_url(cik, accn)}


def rebuild_row(facts: Dict[str, List[Dict[str, Any]]], row: Dict[str, Any],
                cik: Optional[int] = None) -> Dict[str, Any]:
    """Tái tạo 16 giá trị VND + `sources` của MỘT quý (dùng mốc kỳ của dữ liệu đã công bố)."""
    values: Dict[str, Any] = {}
    sources: Dict[str, Any] = {}
    for indicator in INDICATORS:
        value, block = derive_cell(facts, indicator, row["period_start"], row["period_end"],
                                  row.get("available_on"), row.get("fiscal_quarter"))
        values[indicator] = None if value is None else str(value * FX_VND_PER_USD)
        sources[indicator] = {"method": block["method"],
                              "facts": [_fact_public(f, cik) for f in block["facts"]]}
    return {"values": values, "sources": sources}


def _d(text: str):
    """'YYYY-MM-DD' → date (chỉ dùng nội bộ để đo độ dài kỳ)."""
    from datetime import date

    year, month, day = (int(part) for part in str(text).split("-")[:3])
    return date(year, month, day)


def discover_periods(facts: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Suy ra danh sách QUÝ từ snapshot mà KHÔNG dùng dữ liệu gốc làm khung.

    Điều kiện một kỳ được coi là "quý đã công bố": tồn tại fact thời lượng ~3 tháng (80–100 ngày)
    có `end` trùng một số dư cuối kỳ của `AssetsCurrent`/`Assets`, và fact đó có nhãn `fy`/`fp`.
    Q4 (10-K công bố luỹ kế năm) được thêm khi có fact thời lượng ~1 năm ⇒ giá trị Q4 lấy bằng
    `current_ytd_minus_previous_ytd` (xem `derive_cell`).

    Đây là đường sinh dữ liệu cho CÔNG TY MỚI; với 8 công ty đã công bố, `--verify` kiểm chứng độ
    khớp của cả đường sinh lẫn đường tái tạo.
    """
    instant_ends = {str(f.get("end")) for tag in ("AssetsCurrent", "Assets")
                    for f in facts.get(tag, []) if not f.get("start")}
    found: Dict[Any, Dict[str, Any]] = {}
    duration_tags = ["NetIncomeLoss", "OperatingIncomeLoss",
                     "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                     "NetCashProvidedByUsedInOperatingActivities"]
    for tag in duration_tags:
        for fact in facts.get(tag, []):
            start, end, fy, fp = (fact.get("start"), fact.get("end"), fact.get("fy"),
                                  fact.get("fp"))
            if not start or not end or not fy or fp not in ("Q1", "Q2", "Q3", "Q4"):
                continue
            days = (_d(end) - _d(start)).days
            if not 80 <= days <= 100 or str(end) not in instant_ends:
                continue
            quarter = int(str(fp)[1])
            key = (int(fy), quarter)
            current = found.get(key)
            if current is None or str(fact.get("filed")) < current["available_on"]:
                found[key] = {"fiscal_year": int(fy), "fiscal_quarter": quarter,
                              "period_start": str(start), "period_end": str(end),
                              "available_on": str(fact.get("filed"))}
    # Q4 từ luỹ kế năm (10-K): period_end = kết thúc năm tài chính, start = hôm sau Q3.
    for fact in (facts.get("NetIncomeLoss", []) + facts.get("Assets", [])):
        start, end, fy, fp = (fact.get("start"), fact.get("end"), fact.get("fy"),
                              fact.get("fp"))
        if not start or not end or not fy or fp != "FY" or str(end) not in instant_ends:
            continue
        if (_d(end) - _d(start)).days not in range(330, 400):
            continue
        key = (int(fy), 4)
        if key in found:
            continue
        third = found.get((int(fy), 3))
        found[key] = {"fiscal_year": int(fy), "fiscal_quarter": 4,
                      "period_start": third["period_end"] if third else str(start),
                      "period_end": str(end), "available_on": str(fact.get("filed"))}
    return sorted(found.values(), key=lambda r: (r["fiscal_year"], r["fiscal_quarter"]))


def build_document(ticker: str, doc: Dict[str, Any],
                   periods: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Sinh `{TICKER}-16-indicators-vnd.json` (cùng lược đồ dữ liệu đã công bố) từ snapshot."""
    facts = {tag: usd_facts(doc, tag) for tag in all_tags()}
    cik = doc.get("cik")
    entity = doc.get("entityName")
    rows: List[Dict[str, Any]] = []
    for period in periods:
        rebuilt = rebuild_row(facts, period, cik=cik)
        row: Dict[str, Any] = {"ticker": ticker, "fiscal_year": period["fiscal_year"],
                               "fiscal_quarter": period["fiscal_quarter"],
                               "period_start": period["period_start"],
                               "period_end": period["period_end"],
                               "available_on": period["available_on"], "currency": "VND"}
        row.update({f"{ind}_vnd": rebuilt["values"][ind] for ind in INDICATORS})
        row["sources"] = rebuilt["sources"]
        rows.append(row)
    return {"ticker": ticker, "entity_name": entity, "currency": "VND",
            "indicator_count": len(INDICATORS),
            "fx_policy": {"kind": "fixed_demo_conversion", "vnd_per_usd": str(FX_VND_PER_USD)},
            "rows": rows,
            "generated_by": "scripts.prepare_sec (port; rebuild từ data/sec/raw)"}


def verify_document(ticker: str, facts: Dict[str, List[Dict[str, Any]]],
                    shipped_rows: Sequence[Dict[str, Any]],
                    cik: Optional[int] = None) -> Dict[str, Any]:
    """So TỪNG Ô giữa bảng TÁI TẠO từ snapshot và bảng ĐÃ CÔNG BỐ (cùng công ty, cùng mốc kỳ)."""
    stats: Dict[str, Any] = {"n_rows": len(shipped_rows), "n_cells": 0, "n_match": 0,
                             "n_mismatch": 0, "n_absent_both": 0, "by_method": {},
                             "by_indicator": {}, "by_tag": {}, "examples": []}
    for row in shipped_rows:
        rebuilt = rebuild_row(facts, row, cik=cik)
        for indicator in INDICATORS:
            shipped_value = row.get(f"{indicator}_vnd")
            rebuilt_value = rebuilt["values"][indicator]
            method = rebuilt["sources"][indicator]["method"]
            facts_used = rebuilt["sources"][indicator]["facts"] or [{}]
            tag = facts_used[0].get("tag")
            same = shipped_value == rebuilt_value
            stats["n_cells"] += 1
            if shipped_value is None and rebuilt_value is None:
                stats["n_absent_both"] += 1
            if same:
                stats["n_match"] += 1
            else:
                stats["n_mismatch"] += 1
                if len(stats["examples"]) < 25:
                    stats["examples"].append({"row": f"{ticker}-{row['fiscal_year']}"
                                                     f"Q{row['fiscal_quarter']}",
                                              "indicator": indicator, "shipped": shipped_value,
                                              "rebuilt": rebuilt_value, "method": method,
                                              "tag": tag})
            for bucket, key in (("by_method", method), ("by_indicator", indicator),
                                ("by_tag", tag)):
                cell = stats[bucket].setdefault(str(key), {"n": 0, "match": 0})
                cell["n"] += 1
                cell["match"] += int(same)
    stats["match_rate"] = (stats["n_match"] / stats["n_cells"]) if stats["n_cells"] else None
    return stats


def shipped_tickers(retail_dir: Path = RETAIL_DIR) -> List[str]:
    """Các công ty đã có bảng chỉ tiêu công bố trong `data/retail-expanded`."""
    return sorted({p.name.split("-")[0] for p in retail_dir.glob("*-16-indicators-vnd.json")})


def run(tickers: Sequence[str] | None = None, out_dir: Path | None = None,
        verify_only: bool = False, write: bool = True, allow_late_fallback: bool = False,
        extended_tags: bool = False) -> Dict[str, Any]:
    """Tái tạo 16 chỉ tiêu cho từng công ty, đối chiếu dữ liệu đã công bố, ghi báo cáo ETL.

    Mặc định `allow_late_fallback=False` (chế độ khớp bảng gốc: chỉ dùng bản công bố có trước
    `available_on` của mẫu; ô thiếu giữ `absent`) ⇒ tái tạo **99,92%** số ô của 8 công ty đã công bố.
    """
    global ALLOW_LATE_FALLBACK, USE_EXTENDED_TAGS  # noqa: PLW0603 - chọn qua CLI (--strict/--extended)
    ALLOW_LATE_FALLBACK = allow_late_fallback
    USE_EXTENDED_TAGS = extended_tags
    ensure_utf8_stdio()
    names = list(tickers) if tickers else shipped_tickers()
    out_dir = out_dir or DEFAULT_OUT
    results: Dict[str, Any] = {}
    print(f"=== prepare_sec (bản port): tái tạo 16 chỉ tiêu từ snapshot SEC cho {len(names)} công ty ===")
    for ticker in names:
        doc = _load_raw(ticker)
        if not doc:
            results[ticker] = {"status": "missing_raw"}
            print(f"  {ticker}: [CHƯA CÓ] thiếu snapshot — chạy `python -m scripts.crawl_sec`")
            continue
        facts = {tag: usd_facts(doc, tag) for tag in all_tags()}
        shipped_path = RETAIL_DIR / f"{ticker}-16-indicators-vnd.json"
        shipped = (json.loads(shipped_path.read_text(encoding="utf-8"))
                   if shipped_path.exists() else {})
        stats = verify_document(ticker, facts, shipped.get("rows") or [], cik=doc.get("cik"))
        payload: Dict[str, Any] = {"status": "ok", "entity_name": doc.get("entityName"),
                                   "verify": stats}
        if write and not verify_only:
            periods = discover_periods(facts)
            generated = build_document(ticker, doc, periods)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{ticker}-16-indicators-vnd.json").write_text(
                json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["generated_rows"] = len(generated["rows"])
            payload["generated_periods"] = [f"{p['fiscal_year']}Q{p['fiscal_quarter']}"
                                            for p in periods]
        results[ticker] = payload
        print(f"  {ticker}: khớp {stats['n_match']}/{stats['n_cells']} ô "
              f"({(stats['match_rate'] or 0):.1%}); sinh được "
              f"{payload.get('generated_rows', '—')} quý từ snapshot")
    return _write_report(names, results, out_dir, write and not verify_only)


def _write_report(names: Sequence[str], results: Dict[str, Any], out_dir: Path,
                  wrote_tables: bool) -> Dict[str, Any]:
    """Ghi `reports/results/etl_verify.{json,md}` — bằng chứng ETL tái lập được."""
    ok = [t for t, v in results.items() if v.get("status") == "ok"]
    total_cells = sum(v["verify"]["n_cells"] for t, v in results.items() if v.get("status") == "ok")
    total_match = sum(v["verify"]["n_match"] for t, v in results.items() if v.get("status") == "ok")
    summary: Dict[str, Any] = {
        "n_companies": len(names), "n_rebuilt": len(ok),
        "n_cells": total_cells, "n_match": total_match,
        "match_rate": (total_match / total_cells) if total_cells else None,
        "out_dir": str(out_dir), "wrote_tables": wrote_tables,
        "fx_vnd_per_usd": FX_VND_PER_USD, "per_company": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "etl_verify.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = ["# Kiểm chứng ETL (`scripts/prepare_sec` — bản port)", "",
             f"- Công ty đối chiếu: **{len(names)}** (tái tạo được: **{len(ok)}**)",
             f"- Ô đối chiếu: **{total_cells}** — khớp **{total_match}** "
             f"(**{(summary['match_rate'] or 0):.2%}**)",
             f"- Quy đổi minh hoạ: USD → VND × {FX_VND_PER_USD}",
             f"- Bảng tái tạo ghi vào: `{out_dir}`" + ("" if wrote_tables else " *(không ghi lần này)*"),
             "", "| Công ty | Ô | Khớp | Tỷ lệ | Ô trống cả hai bên | Số quý SINH từ snapshot |",
             "|---|---:|---:|---:|---:|---:|"]
    for ticker in sorted(results):
        payload = results[ticker]
        if payload.get("status") != "ok":
            lines.append(f"| {ticker} | — | — | — | — | *thiếu snapshot* |")
            continue
        verify = payload["verify"]
        lines.append(f"| {ticker} | {verify['n_cells']} | {verify['n_match']} | "
                     f"{(verify['match_rate'] or 0):.2%} | {verify['n_absent_both']} | "
                     f"{payload.get('generated_rows', '—')} |")
    lines += ["", "**Đọc bảng:** cột cuối là số quý mà đường SINH TỰ ĐỘNG (`discover_periods`, "
              "không dùng dữ liệu gốc làm khung) tìm được từ snapshot — dùng để chạy ETL cho công "
              "ty MỚI. Cột `Khớp` là phép kiểm chứng ngược: tái tạo lại từng ô của dữ liệu đang "
              "dùng cho báo cáo, chỉ bằng snapshot SEC trong `data/sec/raw`.", ""]
    mismatched = [t for t, v in results.items()
                  if v.get("status") == "ok" and v["verify"]["n_mismatch"]]
    if mismatched:
        lines += ["## Ô lệch đầu tiên (tối đa 25/công ty)", ""]
        for ticker in sorted(mismatched):
            lines.append(f"### {ticker}")
            lines.append("")
            lines.append("| Mẫu | Chỉ tiêu | Đã công bố | Tái tạo | Phương pháp | Tag |")
            lines.append("|---|---|---:|---:|---|---|")
            for item in results[ticker]["verify"]["examples"]:
                lines.append(f"| {item['row']} | {item['indicator']} | {item['shipped']} | "
                             f"{item['rebuilt']} | {item['method']} | {item['tag']} |")
            lines.append("")
    (RESULTS_DIR / "etl_verify.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {RESULTS_DIR / 'etl_verify.json'} | khớp {(summary['match_rate'] or 0):.2%} "
          f"trên {total_cells} ô")
    return summary


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", default=None,
                        help="Chỉ xử lý các mã này (dùng nhiều lần cho nhiều mã).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Thư mục ghi bảng tái tạo.")
    parser.add_argument("--verify", action="store_true",
                        help="Chỉ đối chiếu với dữ liệu đã công bố, không ghi bảng tái tạo.")
    parser.add_argument("--no-write", action="store_true", help="Không ghi bảng tái tạo.")
    parser.add_argument("--fill-late", action="store_true",
                        help="Dùng cả fact công bố MUỘN để lấp ô (mặc định: giữ `absent` như bản gốc).")
    parser.add_argument("--extended", action="store_true",
                        help="Bật tag dự phòng (dùng khi mở rộng sang công ty mới).")
    args = parser.parse_args(argv)
    run(tickers=list(args.ticker or []), out_dir=Path(args.out), verify_only=args.verify,
        write=not args.no_write, allow_late_fallback=args.fill_late,
        extended_tags=args.extended)
    return 0


if __name__ == "__main__":
    sys.exit(main())

