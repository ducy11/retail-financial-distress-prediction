"""Nhãn SỰ KIỆN phá sản/kiệt quệ lấy từ SEC (8-K item 1.03 + tín hiệu kiệt quệ) — không cần thư viện ngoài.

Lệnh: python -m scripts.fetch_events [--refresh] [--ticker WMT]

Vì sao cần: nhãn `is_distressed` trong `data/prepared` là TRẠNG THÁI kế toán của quý target (không tái
tạo được từ dữ liệu công bố — xem `docs/dinh-nghia-nhan.md`; mức khớp tối đa với quy tắc đơn giản là
74,7%) và KHÔNG gắn với sự kiện phá sản nào. Script này bổ sung loại nhãn thứ hai, có mốc thời gian
công khai và kiểm chứng được:

- 8-K có `items` chứa `1.03` (Bankruptcy or Receivership) ⇒ **sự kiện phá sản thật**;
- 8-K có `items` chứa `2.06` (impairment) / `4.02` (non-reliance) / `4.01` (thay kiểm toán)
  ⇒ **tín hiệu kiệt quệ**.

Kết quả: `data/events/{TICKER}.json` + `reports/results/events.{json,md}` (bảng đếm theo công ty và
KẾT LUẬN về tính hợp lệ của đề tài).

Lưu ý phạm vi: KHÔNG thuộc `scripts.run_all` (cần Internet; đây là công cụ khảo sát để trả lời câu
hỏi "8 công ty bán lẻ này có phá sản thật hay không?").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

from forecasting.config import DATA_DIR, ensure_utf8_stdio

EVENTS_DIR = DATA_DIR / "events"
RESULTS_DIR = DATA_DIR.parent / "reports" / "results"

#: SEC yêu cầu User-Agent thật; chờ 0,15 s mỗi request để tôn trọng giới hạn ~10 req/s.
USER_AGENT = "CS114-do-an research <student@example.edu>"
BANKRUPTCY_ITEM = "1.03"
DISTRESS_ITEMS = ("2.06", "4.02", "4.01")
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


def _get_json(url: str, pause: float = 0.15, timeout: int = 30) -> Dict[str, Any]:
    """GET JSON từ SEC (User-Agent + delay). Ném `RuntimeError` rõ ràng nếu không có mạng."""
    time.sleep(pause)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:  # pragma: no cover
        raise RuntimeError(f"Không truy cập được SEC ({url}): {exc}. Bước này cần Internet.") from exc


def ticker_to_cik() -> Dict[str, int]:
    """Bảng ticker → CIK công khai của SEC."""
    data = _get_json(TICKER_URL)
    return {row["ticker"]: int(row["cik_str"]) for row in data.values()}


def filings_of(cik: int) -> List[Dict[str, Any]]:
    """Toàn bộ filing gần đây của một CIK (gộp cả file phân trang trong `filings.files`)."""
    doc = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    keys = ("form", "filingDate", "accessionNumber", "items")
    recent = doc["filings"]["recent"]
    rows = [{k: recent[k][i] for k in keys} for i in range(len(recent["form"]))]
    for extra in doc["filings"].get("files", []):
        page = _get_json(f"https://data.sec.gov/submissions/{extra['name']}")
        rows += [{k: page[k][i] for k in keys} for i in range(len(page["form"]))]
    return rows


def corpus_tickers() -> List[str]:
    """8 công ty trong corpus (bảng 16 chỉ tiêu) — đọc từ dữ liệu, không hard-code."""
    retail = DATA_DIR / "retail-expanded"
    return sorted({path.name.split("-")[0] for path in retail.glob("*-16-indicators-vnd.json")})


def events_of(ticker: str, cik: int) -> List[Dict[str, Any]]:
    """Sự kiện phá sản/kiệt quệ của một công ty (lọc theo `items` của 8-K)."""
    out: List[Dict[str, Any]] = []
    for filing in filings_of(cik):
        items = {code.strip() for code in str(filing.get("items") or "").split(",") if code.strip()}
        kinds = (["bankruptcy"] if BANKRUPTCY_ITEM in items
                 else [f"distress_item_{code}" for code in sorted(items) if code in DISTRESS_ITEMS])
        for kind in kinds:
            accn = str(filing["accessionNumber"])
            out.append({"ticker": ticker, "kind": kind, "form": filing["form"],
                        "filing_date": filing["filingDate"], "items": sorted(items),
                        "accession": accn,
                        "source_url": (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                                       f"{accn.replace('-', '')}/{accn}-index.html")})
    return sorted(out, key=lambda e: e["filing_date"])

def run(tickers: Sequence[str] | None = None, refresh: bool = False) -> Dict[str, Any]:
    """Lấy sự kiện cho từng công ty, ghi cache + báo cáo; trả bảng tổng hợp."""
    ensure_utf8_stdio()
    names = list(tickers) if tickers else corpus_tickers()
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cik_map = ticker_to_cik()
    all_events: List[Dict[str, Any]] = []
    per_ticker: Dict[str, Any] = {}
    for ticker in names:
        cache = EVENTS_DIR / f"{ticker}.json"
        if cache.exists() and not refresh:
            events = json.loads(cache.read_text(encoding="utf-8"))
        else:
            if ticker not in cik_map:
                per_ticker[ticker] = {"status": "no_cik"}
                continue
            events = events_of(ticker, cik_map[ticker])
            cache.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
        per_ticker[ticker] = {"status": "ok", "n_events": len(events),
                              "n_bankruptcy": sum(e["kind"] == "bankruptcy" for e in events),
                              "first": events[0]["filing_date"] if events else None,
                              "last": events[-1]["filing_date"] if events else None}
        all_events += events
        print(f"  {ticker}: {len(events)} sự kiện "
              f"({per_ticker[ticker]['n_bankruptcy']} phá sản)")
    return _write_report(names, per_ticker, all_events)


def _write_report(names: Sequence[str], per_ticker: Dict[str, Any],
                  all_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ghi `reports/results/events.{json,md}` + kết luận về định vị đề tài."""
    n_bankruptcy = sum(e["kind"] == "bankruptcy" for e in all_events)
    n_done = sum(1 for v in per_ticker.values() if v.get("status") == "ok")
    conclusion = (
        f"**0 sự kiện phá sản (8-K item 1.03)** trong {n_done}/{len(names)} công ty đã tra "
        f"⇒ bộ dữ liệu hiện tại KHÔNG chứa sự kiện phá sản thật, nên đề tài được định vị là "
        f"**dự báo suy giảm tài chính (financial distress)**; muốn dự báo phá sản phải mở rộng "
        f"universe sang doanh nghiệp đã nộp 8-K item 1.03."
        if not n_bankruptcy else
        f"Có **{n_bankruptcy}** sự kiện phá sản thật ⇒ nhãn sự kiện dùng được cho bài toán phá sản.")
    summary: Dict[str, Any] = {"n_companies": len(names), "n_events": len(all_events),
                               "n_bankruptcy": n_bankruptcy, "per_ticker": per_ticker,
                               "conclusion": conclusion, "events": all_events}
    (RESULTS_DIR / "events.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Nhãn SỰ KIỆN từ SEC (8-K item 1.03 = phá sản; 2.06/4.02/4.01 = tín hiệu kiệt quệ)", "",
             f"- Công ty kiểm tra: **{len(names)}** — sự kiện: **{len(all_events)}** "
             f"(phá sản: **{n_bankruptcy}**)", "",
             "| Công ty | Sự kiện | Trong đó phá sản | Sự kiện đầu | Sự kiện cuối |",
             "|---|---:|---:|---|---|"]
    for ticker in sorted(per_ticker):
        info = per_ticker[ticker]
        if info.get("status") != "ok":
            lines.append(f"| {ticker} | — | — | — | *không tra được CIK* |")
            continue
        lines.append(f"| {ticker} | {info['n_events']} | {info['n_bankruptcy']} | "
                     f"{info['first'] or '—'} | {info['last'] or '—'} |")
    lines += ["", f"**Kết luận cho đề tài:** {conclusion}", "",
              "> Nguồn: `https://data.sec.gov/submissions/CIK##########.json` (trường `items` của 8-K).",
              ""]
    (RESULTS_DIR / "events.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → {RESULTS_DIR / 'events.md'} | phá sản: {n_bankruptcy}/{len(all_events)} sự kiện")
    return summary


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", action="append", default=None)
    parser.add_argument("--refresh", action="store_true", help="Bỏ cache, tải lại từ SEC.")
    args = parser.parse_args(argv)
    try:
        run(tickers=list(args.ticker or []), refresh=args.refresh)
    except RuntimeError as exc:
        print(f"[fetch_events] {exc}")
        return 2  # cần Internet: mã lỗi rõ ràng, không im lặng thất bại
    return 0


if __name__ == "__main__":
    sys.exit(main())
