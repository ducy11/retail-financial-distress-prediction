"""Kiểm thử bộ kiểm chứng nguồn gốc dữ liệu (`scripts/verify_provenance.py`).

Chạy: python -m unittest tests.test_verify_provenance -v

Vì sao cần: đây là bằng chứng "dữ liệu THẬT, không bịa". Nếu quy tắc đối chiếu hoặc quy tắc quy đổi
VND bị viết sai thì bộ kiểm chứng sẽ báo ĐẠT một cách vô nghĩa — nên phải kiểm cả hai chiều:
1. Quy tắc tính (một fact / hiệu hai fact luỹ kế / method lạ) đúng như dữ liệu khai.
2. Đối chiếu fact phải KHỚP chính xác và phải TỪ CHỐI when sai val/accn.
3. Trên dữ liệu thật (nếu có snapshot SEC): hash WMT khớp registry và WMT không có ô nào sai.
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
SKIP_REASON = "chưa có snapshot SEC thô (data/sec/raw) — chạy `python -m scripts.crawl_sec`"


def setUpModule() -> None:  # noqa: D103 - hook của unittest
    quiet_library_warnings()


class TestDerivedVnd(unittest.TestCase):
    """Quy tắc quy đổi: đúng cho cả fact đơn và hiệu hai kỳ luỹ kế."""

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
    """Đối chiếu fact phải khớp tag/end/val/accn; sai một trường là phải từ chối."""

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
        self.assertFalse(fact_present(self.RAW, {**self.FACT, "tag": "KhôngTồnTại"}))


class TestProvenanceOnRealData(unittest.TestCase):
    """Chạy trên snapshot SEC thật (chỉ 1 công ty để test nhanh)."""

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
