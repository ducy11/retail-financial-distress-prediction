"""Chuyển snapshot SEC thô (data/sec/raw) thành chuỗi chỉ tiêu (retail-expanded).

Lệnh: python -m scripts.prepare_sec [--dry-run]

CHƯA CÀI ĐẶT — pipeline gốc (trích 16 chỉ tiêu từ XBRL companyfacts, chọn tag
mỗi chỉ tiêu theo độ ưu tiên, quy đổi USD→VND ×25.000, đối soát quý/năm, ghi
provenance `sources`) chưa được port sang repo này. Dữ liệu retail-expanded
hiện có trong repo là kết quả của lần chạy gốc và vẫn dùng được:

- `python -m forecasting.data --force` tái tạo prepared từ retail-expanded
  (đã được kiểm thử byte-identical trong tests/test_pipeline.py).
- Chỉ cần chuẩn bị lại dữ liệu khi: crawl được snapshot mới, thêm công ty
  mới, hoặc đổi bộ chỉ tiêu.

Việc port pipeline này gồm các bước (tham khảo cấu trúc output trong
data/retail-expanded/HD-16-indicators-vnd.json):
  1. Đọc companyfacts JSON, nhóm facts theo tag × period (10-Q/10-K).
  2. Với mỗi chỉ tiêu (16), chọn fact đúng quý (reported_quarter) hoặc
     đúng thời điểm (instant), ưu tiên tag theo bảng độ ưu tiên.
  3. Quy đổi USD → VND (×25.000, làm tròn số nguyên), ghi provenance.
  4. Đối soát: tổng 4 quý = năm tài chính; số quý >= ngưỡng; ghi corpus.json.
"""
from __future__ import annotations

import argparse
import sys

from forecasting.config import ensure_utf8_stdio

VND_PER_USD = 25_000


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in kế hoạch, không ghi file.")
    _ = parser.parse_args(argv)
    print(__doc__)
    print("\n[prepare_sec] CHƯA CÀI ĐẶT — xem docstring module này để biết các bước port.")
    return 2  # mã lỗi rõ ràng, tránh nhầm là thành công


if __name__ == "__main__":
    sys.exit(main())
