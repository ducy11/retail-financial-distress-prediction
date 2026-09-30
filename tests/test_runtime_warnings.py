"""Kiểm thử `runtime_warnings.py`: lọc ĐÚNG cảnh báo vô hại, không che cảnh báo khác.

Chạy: python -m unittest discover -s tests -v

Vì sao cần test này? Bộ phiên bản đang kiểm thử (scikit-learn 1.6 + scipy 1.18) in
`OptimizeWarning: Unknown solver options: iprint` ở MỌI lần fit `LogisticRegression`, còn
`unittest` lại gọi `simplefilter("default")` sau khi import module test — nếu cơ chế lọc hỏng thì
output test lại đầy cảnh báo (khó thấy FAIL thật). Test kiểm chứng 3 điều:
1. `quiet_library_warnings()` không chèn trùng filter (gọi nhiều lần vẫn idempotent);
2. cảnh báo `iprint` bị lọc ngay cả SAU khi ai đó `simplefilter("default")` (đúng tình huống unittest);
3. cảnh báo KHÁC (không nằm trong danh sách đã biết) vẫn được hiển thị — không tắt cảnh báo chung.
"""
from __future__ import annotations

import pathlib
import sys
import unittest
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime_warnings import (LIBRARY_WARNING_FILTERS, OptimizeWarning,  # noqa: E402
                              _filter_key, quiet_library_warnings)


def _known_filter_count() -> int:
    """Số bản của các filter đã biết đang nằm trong `warnings.filters`."""
    known = {_filter_key("ignore", flt.get("message", ""), flt["category"], flt.get("module", ""))
             for flt in LIBRARY_WARNING_FILTERS}
    return sum(1 for item in warnings.filters
               if _filter_key(item[0], item[1], item[2], item[3]) in known)


def setUpModule() -> None:  # noqa: D103 - `unittest` hook
    """Lọc lại cảnh báo thư viện vô hại sau khi `unittest` đặt `simplefilter("default")`."""
    quiet_library_warnings()


class TestQuietLibraryWarnings(unittest.TestCase):
    """Cơ chế lọc cảnh báo thư viện."""

    def test_filter_list_is_narrow_and_documented(self):
        """Danh sách filter chỉ gồm các cảnh báo ĐÃ BIẾT (iprint + deprecate matplotlib/pyparsing)."""
        self.assertEqual(len(LIBRARY_WARNING_FILTERS), 3)
        messages = [item.get("message", "") for item in LIBRARY_WARNING_FILTERS]
        modules = [item.get("module", "") for item in LIBRARY_WARNING_FILTERS]
        self.assertIn("Unknown solver options: iprint", messages)
        self.assertTrue(any(module.startswith("matplotlib") for module in modules))
        self.assertTrue(any(module.startswith("pyparsing") for module in modules))

    def test_idempotent_and_placed_at_front(self):
        """Gọi nhiều lần (như mỗi `setUpModule`) vẫn đúng 1 bản mỗi filter và nằm ở ĐẦU danh sách."""
        for _ in range(3):
            quiet_library_warnings()
        self.assertEqual(_known_filter_count(), len(LIBRARY_WARNING_FILTERS))
        head = {_filter_key(item[0], item[1], item[2], item[3]) for item in warnings.filters[:3]}
        self.assertEqual(len(head), len(LIBRARY_WARNING_FILTERS))  # 3 filter của ta đứng đầu

    def test_iprint_warning_is_hidden_even_after_simplefilter_default(self):
        """Sau khi `unittest` đặt `simplefilter("default")`, cảnh báo `iprint` vẫn bị lọc.

        Đây là hồi quy thật đã gặp: nếu chỉ "thêm filter khi chưa có" thì `default` do `unittest`
        chèn lên đầu sẽ thắng và `OptimizeWarning` lại in ra giữa output test.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")      # điều `unittest` làm trước khi chạy test
            quiet_library_warnings()              # điều `setUpModule` làm
            warnings.warn("Unknown solver options: iprint", OptimizeWarning, stacklevel=1)
        self.assertEqual([str(item.message) for item in caught], [])
        self.assertEqual(_known_filter_count(), len(LIBRARY_WARNING_FILTERS))

    def test_unknown_warning_still_visible(self):
        """Filter phải HẸP: cảnh báo khác vẫn nổi lên (không che lỗi thật của code)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            quiet_library_warnings()
            warnings.warn("cảnh báo thử nghiệm không thuộc danh sách đã biết", UserWarning,
                          stacklevel=1)
        self.assertEqual([str(item.message) for item in caught],
                         ["cảnh báo thử nghiệm không thuộc danh sách đã biết"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
