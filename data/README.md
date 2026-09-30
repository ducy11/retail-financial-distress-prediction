# Dữ liệu huấn luyện và kiểm thử

Corpus hiện tại: **8 doanh nghiệp, 332 quý, 16 chỉ tiêu** — WMT, HD, LOW, ROST, DG, ORLY, DKS, FIVE. [Tệp dữ liệu và nguồn BCTC để review](../docs/mo-rong-du-lieu.md).

- `retail-expanded/corpus.json`: công ty được chọn và lý do loại 12 công ty còn lại.
- `retail-expanded/*-16-indicators-vnd.json`: dữ liệu tiền VND, provenance từng chỉ tiêu, đối soát quý/năm.
- `sec/downloads.json`: URL SEC, CIK, tên pháp nhân, thời điểm tải, SHA-256 của 20 snapshot.
- `sec/raw/`: snapshot thô lưu local, bị loại khỏi Git và Docker context để giữ repo nhỏ.

| File trong prepared | Mẫu | Mục đích |
|---|---:|---|
| [train.json](prepared/train.json) | 212 | Huấn luyện ứng viên chung |
| [validation.json](prepared/validation.json) | 32 | Chọn mô hình, 4 quý/công ty |
| [test.json](prepared/test.json) | 64 | Hai năm tài chính hoàn chỉnh cuối, 8 quý/công ty |
| [purged.json](prepared/purged.json) | 16 | Bảo vệ thứ tự công bố toàn cục, 2 quý/công ty bao 2 bên validation |
| [manifest.json](prepared/manifest.json) | — | Khoảng thời gian, chính sách, SHA-256 nguồn và split |

Tái tạo split: `python -m forecasting.data --force` (nhãn `is_distressed` giữ nguyên từ prepared cũ — do pipeline `prepare_sec` gốc sinh ra, không suy ra lại từ 16 chỉ tiêu); train: `python -m forecasting.train`. Hai lệnh chạy offline. Crawl lại: `python -m scripts.crawl_sec --refresh`, rồi `python -m scripts.prepare_sec`. Refresh có thể đổi corpus; cần review báo cáo lọc trước khi dùng.

**Chứng minh dữ liệu là thật:** `python -m scripts.verify_provenance` → `reports/results/provenance.{json,md}`. Script băm SHA-256 toàn bộ snapshot SEC, đối chiếu với `sec/downloads.json`, rồi **tra ngược từng fact** (tag, kỳ, `accn`, `form`) trong `sec/raw/*-companyfacts.json` và kiểm lại phép quy đổi VND — bắt cả trường hợp "điền số cho đủ" ở chỉ tiêu không có fact.

Tiền là chuỗi số nguyên VND, quy đổi minh họa 25.000 VND/USD. Không phải BCTC Việt Nam hoặc tỷ giá lịch sử. Mỗi sample chứa lịch sử và nhãn; lịch sử lặp giữa sample có chủ đích, không phải nhiều quan sát độc lập hơn. 332 quý tạo 324 cặp dự báo. Test là 8 quý cuối mỗi công ty: WMT FY2025–FY2026, các công ty khác cùng cửa sổ hai năm tài chính cuối (xem `test_ranges` trong manifest).

