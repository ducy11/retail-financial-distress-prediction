"""Nạp dữ liệu prepared (train/validation/test/purged) và manifest.

Dữ liệu là JSON thuần — không cần pandas. Mỗi sample có cấu trúc:
{
  "sample_id": "WMT-2015Q2",
  "ticker": "WMT",
  "request": {
    "history": [ {fiscal_year, fiscal_quarter, period_start, period_end,
                  available_on, revenue_vnd, cost_of_sales_vnd, ...}, ... ],
    "as_of": "2014-06-06",
    "target_period_start": "2014-05-01",
    "target_period_end": "2014-07-31"
  },
  "is_distressed": 1,
  "label_available_on": "2014-09-05"
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .config import MANIFEST_FILE, PREPARED_FILES, SUFFIX, TARGET, manifest_file, prepared_files


def load_manifest(path: Path | None = None) -> Dict[str, Any]:
    """Đọc manifest (chính sách split, thống kê mẫu, sha256 nguồn).

    Mặc định đọc manifest của `prepared_dir()` (tôn trọng biến môi trường
    `FORECASTING_PREPARED_DIR`); truyền `path` để chỉ định file khác.
    """
    target = path or manifest_file()
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prepared(name: str, prepared_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Nạp một split prepared: "train" | "validation" | "test" | "purged".

    `prepared_dir` cho phép đọc split khác (vd `data/prepared-rule`); mặc định dùng
    `config.prepared_dir()` và tôn trọng biến môi trường `FORECASTING_PREPARED_DIR`.
    """
    files = prepared_files(prepared_dir) if prepared_dir else prepared_files()
    if name not in files:
        raise ValueError(f"Không có split chuẩn: {name!r}; có {sorted(files)}")
    path = files[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path.name} (trong {path.parent}). Chạy `python -m forecasting.data` "
            f"để tái tạo split, hoặc `python -m scripts.relabel` để tạo split theo quy tắc."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def history_array(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lịch sử quý gần nhất → hiện tại (mảng thô, giá trị chuỗi VND hoặc None)."""
    return sample["request"]["history"]


def to_float(value: Any) -> float:
    """Chuyển giá trị VND (chuỗi số nguyên) hoặc None thành float (nan hóa None)."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def field_names(history_row: Dict[str, Any]) -> List[str]:
    """Tên 16 chỉ tiêu có mặt trong một dòng lịch sử."""
    return [k for k in history_row if k.startswith("revenue") or k.endswith(SUFFIX)]


def label_of(sample: Dict[str, Any]) -> int:
    """Nhãn nhị phân suy giảm."""
    return int(sample[TARGET])


def to_analytic_columns(history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """Chuyển lịch sử về dict {field: [float, ...]} để tính feature numpy."""
    cols: Dict[str, List[float]] = {}
    fields = field_names(history[0])
    for field in fields:
        cols[field] = [to_float(row.get(field)) for row in history]
    return cols
