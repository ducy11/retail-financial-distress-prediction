# Kiểm chứng nguồn gốc dữ liệu (SEC thật) — `scripts.verify_provenance`

- Registry: `data\sec\downloads.json` · snapshot thô: `data\sec\raw`
- Kết luận: **ĐẠT**

| Phép kiểm | Số lượng |
|---|---|
| File SEC đã băm SHA-256 | 20 |
| Hash lệch registry | 0 |
| Công ty đối chiếu fact | 8 |
| Ô dữ liệu (quý × chỉ tiêu) | 5312 |
| Ô không có fact ở SEC (phải null) | 703 |
| Ô có fact → đối chiếu companyfacts SEC | 4609 |
| Fact không tồn tại trong SEC | 0 |
| Quy đổi VND sai (theo từng `method`) | 0 |
| Ô không có fact nhưng vẫn có số | 0 |
| Ô dùng method chưa hỗ trợ | 0 |

## Theo công ty

| Ticker | Số quý | Ô dữ liệu | Ô không có fact | Ô đối chiếu | Fact thiếu ở SEC | Lỗi VND | Ô bịa số |
|---|---|---|---|---|---|---|---|
| DG | 32 | 512 | 92 | 420 | 0 | 0 | 0 |
| DKS | 44 | 704 | 167 | 537 | 0 | 0 | 0 |
| FIVE | 32 | 512 | 44 | 468 | 0 | 0 | 0 |
| HD | 44 | 704 | 44 | 660 | 0 | 0 | 0 |
| LOW | 44 | 704 | 60 | 644 | 0 | 0 | 0 |
| ORLY | 44 | 704 | 88 | 616 | 0 | 0 | 0 |
| ROST | 44 | 704 | 112 | 592 | 0 | 0 | 0 |
| WMT | 48 | 768 | 96 | 672 | 0 | 0 | 0 |

## Cách đọc

1. **Hash khớp** nghĩa là file SEC thô trong repo không bị sửa tay kể từ lúc tải (Nguồn SEC công khai: https://data.sec.gov/api/xbrl/companyfacts/. Mọi fact ghi trong `sources` phải tồn tại trong file raw (đã hash) — đây là bằng chứng dữ liệu THẬT, không sinh/sửa tay.)
2. **Fact tồn tại trong SEC** nghĩa là từng con số dùng để huấn luyện đều tra ngược được trong companyfacts công khai (kèm `accn`, `form`, ngày `filed`) — không có số liệu tự sinh.
3. **Quy đổi VND** chỉ là phép nhân hằng số 25.000 khai trong `fx_policy` (minh hoạ đơn vị, không phải dữ liệu Việt Nam và không phải tỷ giá lịch sử)
4. **Ô bịa số (phải = 0)**: với chỉ tiêu không có fact ở SEC, file phải để `null` — đây là phép kiểm bắt lỗi "điền số cho đủ".

Lệnh tái lập: `python -m scripts.verify_provenance` (đầy đủ) hoặc `... --quick` (2 công ty).