"""EDA chuyên sâu: thống kê feature, nhãn, tương quan, dịch chuyển — sinh báo cáo tự động.

Lệnh: python -m scripts.eda_deep [--no-write] [--no-figures]

Khác `scripts/eda.py` (mô tả corpus: độ phủ chỉ tiêu, cân bằng lớp, boxplot, chuỗi thời gian),
script này trả lời bốn câu hỏi kỹ thuật phải xong trước khi tin vào metric:

1. Ma trận 47 feature có cột hằng / thiếu nhiều / đuôi nặng / nhiều outlier không?
2. Nhãn lệch tới mức nào theo split – công ty – quý, và có "dính" theo thời gian không?
3. Feature nào thật sự liên hệ với nhãn (AUC 1-feature, MI, BH-FDR) và nhóm nào trùng thông tin?
4. Test có khác train (KS/SMD/PSI) và còn rò rỉ nào (lịch sử chồng lấn, dòng trùng, thiếu-mang-nhãn)?

Kết quả: `reports/results/eda_deep.{json,md}` + 7 hình ở `reports/figures/eda_deep/`.
Toàn bộ logic nằm ở `forecasting/eda.py` (hàm nhận mảng/dict thuần ⇒ test độc lập được).
"""
from __future__ import annotations

import argparse
import sys

from forecasting.config import ensure_utf8_stdio
from forecasting.eda import run


def main(argv=None) -> int:
    """Chạy EDA chuyên sâu và (mặc định) ghi artifact vào `reports/`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true",
                        help="Chỉ in, không ghi reports/results/eda_deep.*")
    parser.add_argument("--no-figures", action="store_true",
                        help="Bỏ vẽ hình (nhanh hơn, dùng khi chỉ cần số liệu)")
    args = parser.parse_args(argv)
    print("=== EDA chuyên sâu: feature / nhãn / tương quan / dịch chuyển ===")
    run(write=not args.no_write, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
