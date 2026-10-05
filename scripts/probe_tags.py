"""Measure data feasibility for candidate features: which XBRL tags exist in the SEC snapshot?

For every candidate tag the script counts the share of quarters covered by `period_end` and the share
with an exact quarter match, then prints a min/mean table per company to guide indicator selection. It
writes no artifact and is a survey tool rather than a pipeline step.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from forecasting.config import DATA_DIR, RETAIL_DIR, ensure_utf8_stdio

RAW_DIR = DATA_DIR / "sec" / "raw"

#: (tag, kind) - instant for a period-end balance, duration for activity within the period.
CANDIDATES: List[Tuple[str, str]] = [
    # assets, liabilities and equity: mostly covered by the 16 indicators, kept as controls
    ("Assets", "instant"), ("AssetsCurrent", "instant"), ("InventoryNet", "instant"),
    ("Liabilities", "instant"), ("LiabilitiesCurrent", "instant"),
    ("StockholdersEquity", "instant"), ("RetainedEarningsAccumulatedDeficit", "instant"),
    ("CashAndCashEquivalentsAtCarryingValue", "instant"),
    # new candidates for the proposed features
    ("AccountsPayableCurrent", "instant"),
    ("OperatingLeaseLiabilityCurrent", "instant"), ("OperatingLeaseLiabilityNoncurrent", "instant"),
    ("LongTermDebtNoncurrent", "instant"), ("LongTermDebtCurrent", "instant"),
    ("ShortTermBorrowings", "instant"), ("LongTermDebt", "instant"),
    ("Goodwill", "instant"), ("CommonStockSharesOutstanding", "instant"),
    # income statement
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
    # cash flow and equity movements
    ("NetCashProvidedByUsedInOperatingActivities", "duration"),
    ("NetCashProvidedByUsedInInvestingActivities", "duration"),
    ("NetCashProvidedByUsedInFinancingActivities", "duration"),
    ("PaymentsOfDividendsCommonStock", "duration"),
    ("PaymentsForRepurchaseOfCommonStock", "duration"),
]


def _index(ticker: str) -> Tuple[Dict[str, set], Dict[Tuple[str, str], set]]:
    """Return (ends_by_tag, exact_by_tag) parsed from one company's companyfacts."""
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
    """Return {tag: {kind, cov: {ticker: percent}, exact: {ticker: percent}}} over the corpus."""
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
    """Print the tag coverage table (min and mean per company) and return the raw numbers."""
    ensure_utf8_stdio()
    data = measure()
    n_company = len(next(iter(data.values()))["cov"]) if data else 0
    print(f"=== XBRL tag coverage over {n_company} companies / {len(CANDIDATES)} candidate tags ===")
    print(f"{'tag':52s} {'kind':8s} {'min_cov':>7s} {'mean_cov':>8s} {'min_exact':>9s} {'missing':>7s}")
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
