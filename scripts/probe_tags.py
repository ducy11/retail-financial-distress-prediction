"""Đo ĐỘ KHẢ THI DỮ LIỆU của các đặc trưng đề xuất: tag XBRL nào có thật trong snapshot SEC?

Lệnh: python -m scripts.probe_tags

Vì sao cần: đề xuất thêm đặc trưng chỉ có nghĩa nếu chỉ tiêu đó tồn tại THẬT trong
`data/sec/raw/*-companyfacts.json` cho phần lớn số quý. Với từng tag ứng viên, script đếm:
- % quý có fact kết thúc đúng `period_end` (độ phủ),
- % quý có fact khớp CHÍNH XÁC kỳ quý (`period_start`..`period_end`) — phân biệt số quý riêng
  với số luỹ kế (luỹ kế vẫn dùng được nhưng phải trừ nhau, xem `current_ytd_minus_previous_ytd`).

In bảng tổng hợp min/mean theo công ty để chọn chỉ tiêu bổ sung. Không ghi artifact, không
thuộc `scripts.run_all` (đây là công cụ khảo sát, không phải bước pipeline).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from forecasting.config import DATA_DIR, RETAIL_DIR, ensure_utf8_stdio

RAW_DIR = DATA_DIR / "sec" / "raw"

#: (tag, dạng) — instant = số dư cuối kỳ; duration = số phát sinh trong kỳ.
CANDIDATES: List[Tuple[str, str]] = [
    # tài sản/nợ/vốn (chủ yếu đã có trong 16 chỉ tiêu — dùng làm đối chứng)
    ("Assets", "instant"), ("AssetsCurrent", "instant"), ("InventoryNet", "instant"),
    ("Liabilities", "instant"), ("LiabilitiesCurrent", "instant"),
    ("StockholdersEquity", "instant"), ("RetainedEarningsAccumulatedDeficit", "instant"),
    ("CashAndCashEquivalentsAtCarryingValue", "instant"),
    # ứng viên MỚI cho đặc trưng đề xuất
    ("AccountsPayableCurrent", "instant"),
    ("OperatingLeaseLiabilityCurrent", "instant"), ("OperatingLeaseLiabilityNoncurrent", "instant"),
    ("LongTermDebtNoncurrent", "instant"), ("LongTermDebtCurrent", "instant"),
    ("ShortTermBorrowings", "instant"), ("LongTermDebt", "instant"),
    ("Goodwill", "instant"), ("CommonStockSharesOutstanding", "instant"),
    # kết quả kinh doanh
    ("Revenues", "duration"), ("RevenueFromContractWithCustomerExcludingAssessedTax", "duration"),
    ("SalesRevenueNet", "duration"), ("CostOfRevenue", "duration"),
    ("CostOfGoodsAndServicesSold", "duration"), ("GrossProfit", "duration"),
    ("OperatingIncomeLoss", "duration"), ("NetIncomeLoss", "duration"),
    ("IncomeTaxExpenseBenefit", "duration"), ("ShareBasedCompensation", "duration"),
    ("PaymentsToAcquirePropertyPlantAndEquipment", "duration"),
    ("DepreciationDepletionAndAmortization", "duration"),
    ("DepreciationAmortizationAndAccretionNet", "duration"),
    ("InterestExpense", "duration"), ("InterestExpenseDebt", "duration"),
    ("OperatingLeaseCost", "duration"),
    ("WeightedAverageNumberOfDilutedSharesOutstanding", "duration"),
    # dòng tiền / vốn chủ
    ("NetCashProvidedByUsedInOperatingActivities", "duration"),
    ("NetCashProvidedByUsedInInvestingActivities", "duration"),
    ("NetCashProvidedByUsedInFinancingActivities", "duration"),
    ("PaymentsOfDividendsCommonStock", "duration"),
    ("PaymentsForRepurchaseOfCommonStock", "duration"),
]


def _index(ticker: str) -> Tuple[Dict[str, set], Dict[Tuple[str, str], set]]:
    """(ends_by_tag, exact_by_tag) từ companyfacts của một công ty."""
    path = RAW_DIR / f"{ticker}-companyfacts.json"
    if not path.exists():
        return {}, {}
    facts = json.loads(path.read_text(encoding="utf-8")).get("facts") or {}
    ends: Dict[str, set] = defaultdict(set)
    exact: Dict[Tuple[str, str], set] = defaultdict(set)
    for _taxonomy, tags in facts.items():
        for tag, body in tags.items():
            for _unit, entries in (body.get("units") or {}).items():
                for entry in entries:
                    ends[tag].add(entry.get("end"))
                    exact[(entry.get("start"), entry.get("end"))].add(tag)
    return ends, exact


def measure() -> Dict[str, Dict[str, Any]]:
    """{tag: {kind, cov: {ticker: %}, exact: {ticker: %}}} trên 8 công ty trong corpus."""
    out: Dict[str, Dict[str, Any]] = {tag: {"kind": kind, "cov": {}, "exact": {}}
                                      for tag, kind in CANDIDATES}
    for path in sorted(RETAIL_DIR.glob("*-16-indicators-vnd.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        ticker, rows = doc["ticker"], doc["rows"]
        ends, exact = _index(ticker)
        for tag, kind in CANDIDATES:
            hits = ends.get(tag, set())
            cov = sum(1 for r in rows if r["period_end"] in hits) / len(rows)
            if kind == "instant":
                ok = sum(1 for r in rows
                         if tag in exact.get((None, r["period_end"]), set()))
            else:
                ok = sum(1 for r in rows
                         if tag in exact.get((r["period_start"], r["period_end"]), set()))
            out[tag]["cov"][ticker] = 100.0 * cov
            out[tag]["exact"][ticker] = 100.0 * ok / len(rows)
    return out


def run() -> Dict[str, Dict[str, Any]]:
    """In bảng độ phủ tag (min/mean theo công ty) và trả số liệu thô."""
    ensure_utf8_stdio()
    data = measure()
    n_company = len(next(iter(data.values()))["cov"]) if data else 0
    print(f"=== Độ phủ tag XBRL trên {n_company} công ty / {len(CANDIDATES)} tag ứng viên ===")
    print(f"{'tag':52s} {'dạng':8s} {'min phủ':>7s} {'tb phủ':>7s} {'min khớp kỳ':>11s} {'#ct thiếu':>9s}")
    rows = []
    for tag, d in data.items():
        covs, exs = list(d["cov"].values()), list(d["exact"].values())
        missing = sum(1 for c in covs if c == 0.0)
        rows.append((min(covs), tag, d["kind"], sum(covs) / max(1, n_company),
                     min(exs), missing))
    for min_cov, tag, kind, mean_cov, min_exact, missing in sorted(rows, reverse=True):
        print(f"{tag:52s} {kind:8s} {min_cov:6.0f}% {mean_cov:6.0f}% {min_exact:10.0f}% {missing:9d}")
    return data


if __name__ == "__main__":
    run()
