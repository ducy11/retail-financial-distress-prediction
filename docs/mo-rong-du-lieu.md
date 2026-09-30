# Mở rộng dữ liệu bán lẻ Mỹ

Thiết kế/ lý do chọn corpus; tài liệu này được `data/README.md` tham chiếu.

## Corpus hiện tại

Từ SEC XBRL companyfacts (`data/sec/downloads.json` + `data/sec/raw/`) của
20 công ty bán lẻ Mỹ; **8 công ty được giữ**, 12 bị loại
(xem lý do từng công ty trong `data/retail-expanded/corpus.json`):

| Ticker | Tên | Số quý | Năm đầu → cuối |
|---|---:|---:|---:|
| WMT | Walmart | 48 | FY2015 → FY2026 |
| HD | The Home Depot | 44 | FY2015 → FY2026 |
| LOW | Lowe’s | 44 | FY2015 → FY2026 |
| ROST | Ross Stores | 44 | FY2015 → FY2026 |
| DG | Dollar General | 32 | FY2018 → FY2026 |
| ORLY | O’Reilly Automotive | 44 | FY2015 → FY2026 |
| DKS | Dick’s Sporting Goods | 44 | FY2015 → FY2026 |
| FIVE | Five Below | 32 | FY2018 → FY2026 |

- `*-16-indicators-vnd.json`: 16 chỉ tiêu/ quý, quy đổi minh họa **25.000 VND/USD**,
  mỗi chỉ tiêu có nguồn fact (`sources`): tag XBRL, form, ngày file, frame.
- `*_vnd` là chuỗi số nguyên (đồng Việt Nam), không phải BCTC Việt Nam.

## Tiêu chí chọn công ty

1. Ngành bán lẻ có quý tài chính 13 tuần (10-Q/10-K) đủ 16 chỉ tiêu chuẩn.
2. Tối thiểu 32 quý (8 năm) để đủ train/validation/test theo chính sách split.
3. Loại các công ty có chỉ tiêu thiếu quá ≥30% tổng quý hoặc thay đổi nghiệp vụ lớn
   (xem `corpus.json` → `excluded`).

## Chỉ tiêu (16)

`revenue, cost_of_sales, inventory, selling_general_admin, operating_cash_flow,
total_assets, cash_and_equivalents, operating_income, current_assets,
current_liabilities, net_income, stockholders_equity, liabilities, receivables,
short_term_investments, retained_earnings`.

Số liệu cuối kỳ (instant) lấy đúng ngày `end`, số phát sinh (duration) lấy đúng quý
(`reported_quarter`, tránh YTD gộp). Thiếu → `null`, xử lý như NaN ở tầng features.

## Tái tạo

```powershell
python -m scripts.crawl_sec --refresh   # tải lại companyfacts từ SEC (mạng cần)
python -m scripts.prepare_sec           # retail-expanded từ snaphot + ma trận đối soát
python -m forecasting.data --force      # split prepared từ các file 16-indicators
```

`data/sec/raw/` bị loại khỏi git (file lớn); `data/sec/downloads.json` ghi hash để
đối soát khi tải lại.
