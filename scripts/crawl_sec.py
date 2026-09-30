"""Tải lại snapshot SEC companyfacts cho các công ty trong downloads.json.

Lệnh: python -m scripts.crawl_sec [--refresh] [--ticker WMT]

Mặc định chỉ KIỂM TRA (so sha256 file local với downloads.json, báo thiếu/hụt).
--refresh tải lại toàn bộ (hoặc --ticker chọn lọc) từ data.sec.gov và cập nhật
sha256 + thời điểm tải trong data/sec/downloads.json.

Quy tắc SEC: request phải có User-Agent dạng "Tên <email>"; tối đa ~10 req/s.
Dữ liệu ghi vào data/sec/raw/<TICKER>-companyfacts.json (bị .gitignore).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from forecasting.config import DATA_DIR, ensure_utf8_stdio

DOWNLOADS = DATA_DIR / "sec" / "downloads.json"
RAW_DIR = DATA_DIR / "sec" / "raw"
USER_AGENT = "CS114-do-an research <student@example.edu>"  # TODO: điền email thật


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_downloads() -> Dict[str, Any]:
    with open(DOWNLOADS, encoding="utf-8") as f:
        return json.load(f)


def verify_one(ticker: str, meta: Dict[str, Any]) -> str:
    """Trạng thái một ticker: ok | mismatch | missing."""
    path = RAW_DIR / f"{ticker}-companyfacts.json"
    if not path.exists():
        return "missing"
    actual = sha256_bytes(path.read_bytes())
    return "ok" if actual == meta.get("sha256") else "mismatch"


def download_one(ticker: str, meta: Dict[str, Any]) -> str:
    """Tải companyfacts cho một ticker, ghi file + cập nhật meta. Trả trạng thái."""
    url = meta["url"]
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    path = RAW_DIR / f"{ticker}-companyfacts.json"
    path.write_bytes(data)
    meta["sha256"] = sha256_bytes(data)
    meta["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    return "downloaded"


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Tải lại từ SEC.")
    parser.add_argument("--ticker", action="append", default=[],
                        help="Chỉ xử lý các ticker này (mặc định: tất cả).")
    args = parser.parse_args(argv)

    downloads = load_downloads()
    targets = args.ticker or list(downloads)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    status_counts: Dict[str, int] = {}
    for ticker in targets:
        if ticker not in downloads:
            print(f"  {ticker}: không có trong downloads.json — bỏ qua")
            continue
        meta = downloads[ticker]
        if args.refresh:
            try:
                status = download_one(ticker, meta)
                time.sleep(0.15)  # lịch sự với SEC API
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  {ticker}: LỖI tải — {e}")
                status = "error"
        else:
            status = verify_one(ticker, meta)
        status_counts[status] = status_counts.get(status, 0) + 1
        print(f"  {ticker}: {status}")

    if args.refresh and status_counts.get("downloaded"):
        with open(DOWNLOADS, "w", encoding="utf-8", newline="\n") as f:
            json.dump(downloads, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Đã cập nhật {DOWNLOADS.name}")

    print("Tổng kết:", status_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
