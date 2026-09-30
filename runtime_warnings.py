"""Lọc CẢNH BÁO VÔ HẠI của thư viện để log CLI/test sạch — không che cảnh báo của code dự án.

Vì sao cần một chỗ chung thay vì lọc rải rác?
1. Cảnh báo phát sinh từ CẶP PHIÊN BẢN thư viện, không phải từ logic của đề tài:
   - scikit-learn 1.6 truyền tuỳ chọn `iprint` xuống scipy 1.18 (scipy đã bỏ tuỳ chọn này) ⇒ MỖI lần
     fit `LogisticRegression` in ra `OptimizeWarning: Unknown solver options: iprint`;
   - matplotlib 3.9 gọi API đã deprecate của pyparsing ⇒ `PyparsingDeprecationWarning` khi vẽ hình.
   Cả hai KHÔNG làm đổi tham số/kết quả mô hình (đã kiểm chứng: metric giống hệt khi bật/tắt lọc).
2. `unittest` (khi chạy `python -m unittest ...`) gọi `warnings.simplefilter("default")` SAU khi đã
   import module test, nên filter đặt ở cấp module bị "che" và cảnh báo lại hiện ra. Vì vậy các
   module test gọi lại `quiet_library_warnings()` trong `setUpModule()` (chạy sau runner).

Chỉ lọc ĐÚNG thông điệp/đối tượng đã biết; mọi cảnh báo khác (kể cả của code dự án) vẫn hiển thị.
Gỡ module này khi nâng cấp scikit-learn/scipy/matplotlib sang bộ phiên bản tương thích.
"""
from __future__ import annotations

import warnings

try:  # `OptimizeWarning` là cảnh báo của scipy
    from scipy.optimize import OptimizeWarning
except Exception:  # pragma: no cover - môi trường không có scipy
    class OptimizeWarning(UserWarning):  # type: ignore[no-redef]
        """Fallback khi scipy không cung cấp `OptimizeWarning`."""


#: Các filter vô hại đã biết (kwarg đúng chuẩn `warnings.filterwarnings`).
LIBRARY_WARNING_FILTERS = (
    {"message": "Unknown solver options: iprint", "category": OptimizeWarning},
    {"category": DeprecationWarning, "module": r"matplotlib\..*"},
    {"category": DeprecationWarning, "module": r"pyparsing.*"},
)


def _pattern_text(value: object) -> str:
    """Chuỗi pattern của một filter (`warnings` lưu dạng regex đã compile hoặc `None`)."""
    if value is None:
        return ""
    return getattr(value, "pattern", str(value))


def _filter_key(action: str, message: object, category: type, module: object) -> tuple:
    """Khoá nhận dạng một filter (dùng để gỡ bản cũ trước khi chèn lại)."""
    return (action, _pattern_text(message), category, _pattern_text(module))


def quiet_library_warnings() -> None:
    """Đưa các filter vô hại lên ĐẦU `warnings.filters` (mỗi filter chỉ giữ đúng một bản).

    Phải chèn lại ở ĐẦU chứ không chỉ "thêm nếu chưa có": `unittest` gọi
    `warnings.simplefilter("default")` ngay trước khi chạy test, chèn `default` vào đầu danh sách
    và như vậy filter cũ (dù vẫn còn trong danh sách) bị "che" — cảnh báo `iprint` lại in ra. Vì
    vậy hàm này gỡ bản cũ rồi chèn lại lên đầu; gọi bao nhiêu lần cũng an toàn, danh sách không phình.
    """
    for flt in LIBRARY_WARNING_FILTERS:
        key = _filter_key("ignore", flt.get("message", ""), flt["category"], flt.get("module", ""))
        warnings.filters[:] = [item for item in warnings.filters
                               if _filter_key(item[0], item[1], item[2], item[3]) != key]
        warnings.filterwarnings("ignore", **flt)
