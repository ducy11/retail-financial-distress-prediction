"""Verify the provenance checker in `scripts/verify_provenance.py`.

This suite guards the claim that the data is real rather than fabricated: a wrong comparison rule or a
wrong VND conversion would make the checker pass for no reason. The conversion rules are exercised
directly, and the fact matcher must agree exactly while rejecting a wrong value or accession.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_provenance import (RAW_DIR, RETAIL_DIR, check_hashes, derived_vnd,  # noqa: E402
                                       fact_present, load_downloads, verify_document)
from runtime_warnings import quiet_library_warnings  # noqa: E402

HAS_RAW = (RAW_DIR / "WMT-companyfacts.json").exists()
SKIP_REASON = "no raw SEC snapshot in data/sec/raw; run `python -m scripts.crawl_sec`"


def setUpModule() -> None:  # noqa: D103 - unittest hook
    quiet_library_warnings()


class TestDerivedVnd(unittest.TestCase):
    """Conversion rules: correct for a single fact and for the difference of two cumulative periods."""

    def test_single_fact_methods(self):
        facts = [{"val": 114_167_000_000}]
        for method in ("instant", "reported_quarter", "reported_first_quarter"):
            self.assertEqual(derived_vnd(facts, 25_000, method), 114_167_000_000 * 25_000)

    def test_ytd_difference_method(self):
        facts = [{"val": 1_097_289_000}, {"val": 548_648_000}]
        self.assertEqual(derived_vnd(facts, 25_000, "current_ytd_minus_previous_ytd"),
                         548_641_000 * 25_000)

    def test_unknown_method_returns_none(self):
        self.assertIsNone(derived_vnd([{"val": 1}], 25_000, "method_chua_biet"))
        self.assertIsNone(derived_vnd([], 25_000, "instant"))
        self.assertIsNone(derived_vnd([{"val": 1}], 25_000, "current_ytd_minus_previous_ytd"))


class TestFactMatching(unittest.TestCase):
    """The fact matcher must agree on tag, end, value and accession, and reject any single mismatch."""

    RAW = {"facts": {"us-gaap": {"SalesRevenueNet": {"units": {"USD": [
        {"start": "2014-02-01", "end": "2014-04-30", "val": 114_167_000_000,
         "accn": "0000104169-14-000033", "form": "10-Q", "filed": "2014-06-06"}]}}}}}
    FACT = {"tag": "SalesRevenueNet", "start": "2014-02-01", "end": "2014-04-30",
            "val": 114_167_000_000, "accn": "0000104169-14-000033", "form": "10-Q"}

    def test_matching_fact_is_found(self):
        self.assertTrue(fact_present(self.RAW, self.FACT))

    def test_wrong_value_or_accn_is_rejected(self):
        self.assertFalse(fact_present(self.RAW, {**self.FACT, "val": 1}))
        self.assertFalse(fact_present(self.RAW, {**self.FACT, "accn": "0000000000-00-000000"}))
        self.assertFalse(fact_present(self.RAW, {**self.FACT, "tag": "NotAPresentTag"}))


class TestProvenanceOnRealData(unittest.TestCase):
    """Run against the real SEC snapshot, limited to one company to keep the test fast."""

    @unittest.skipUnless(HAS_RAW, SKIP_REASON)
    def test_hash_and_facts_of_wmt(self):
        downloads = load_downloads()
        self.assertIn("WMT", downloads)
        self.assertEqual(check_hashes({"WMT": downloads["WMT"]}), [])
        doc = json.loads((RETAIL_DIR / "WMT-16-indicators-vnd.json").read_text(encoding="utf-8"))
        raw = json.loads((RAW_DIR / "WMT-companyfacts.json").read_text(encoding="utf-8"))
        stats = verify_document("WMT", doc, raw)
        self.assertEqual(stats["n_cells"], stats["n_rows"] * 16)
        self.assertEqual(stats["n_facts_missing_in_sec"], 0)
        self.assertEqual(stats["n_vnd_mismatch"], 0)
        self.assertEqual(stats["n_absent_but_filled"], 0)
        self.assertEqual(stats["n_unsupported_method"], 0)
        self.assertGreater(stats["n_facts_checked"], 0)


if __name__ == "__main__":
    unittest.main()
