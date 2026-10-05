"""Verify or refresh the SEC companyfacts snapshots listed in downloads.json.

By default the script compares each local file's SHA-256 against the registry and reports missing or
mismatched snapshots; `--refresh` re-downloads from data.sec.gov. SEC requires a real User-Agent and at
most ~10 requests per second. Run with `python -m scripts.crawl_sec [--refresh] [--ticker WMT]`.
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
USER_AGENT = "CS114-do-an research <student@example.edu>"  # TODO: replace with a real contact email


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_downloads() -> Dict[str, Any]:
    with open(DOWNLOADS, encoding="utf-8") as f:
        return json.load(f)


def verify_one(ticker: str, meta: Dict[str, Any]) -> str:
    """Status of one ticker: ok, mismatch or missing."""
    path = RAW_DIR / f"{ticker}-companyfacts.json"
    if not path.exists():
        return "missing"
    actual = sha256_bytes(path.read_bytes())
    return "ok" if actual == meta.get("sha256") else "mismatch"


def download_one(ticker: str, meta: Dict[str, Any]) -> str:
    """Download companyfacts for one ticker, write the file and update its metadata."""
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
    parser.add_argument("--refresh", action="store_true", help="Re-download from SEC.")
    parser.add_argument("--ticker", action="append", default=[],
                        help="Process only these tickers (default: all).")
    args = parser.parse_args(argv)

    downloads = load_downloads()
    targets = args.ticker or list(downloads)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    status_counts: Dict[str, int] = {}
    for ticker in targets:
        if ticker not in downloads:
            print(f"  {ticker}: not present in downloads.json — skipped")
            continue
        meta = downloads[ticker]
        if args.refresh:
            try:
                status = download_one(ticker, meta)
                time.sleep(0.15)  # stay polite towards the SEC API
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  {ticker}: download failed — {e}")
                status = "error"
        else:
            status = verify_one(ticker, meta)
        status_counts[status] = status_counts.get(status, 0) + 1
        print(f"  {ticker}: {status}")

    if args.refresh and status_counts.get("downloaded"):
        with open(DOWNLOADS, "w", encoding="utf-8", newline="\n") as f:
            json.dump(downloads, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Updated {DOWNLOADS.name}")

    print("Summary:", status_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
