# Kiểm định ý nghĩa thống kê trên test (DeLong + paired bootstrap)

- Tập so sánh: **64 mẫu test** (38 dương); mọi hệ thống chấm trên **cùng** mẫu, test không được dùng để chọn cấu hình.
- Hệ thống: `model[random_forest]`, `ticker_prior`, `single_feature[debt_to_assets_latest]`, `dummy_most_frequent`.
- Kiểm định: DeLong (1988) cho ΔAUROC; paired bootstrap cho ΔAP/ΔAUROC (resample theo lớp, seed cố định).

## 1. Kết quả tổng hợp

| Hệ thống | AUROC | Average Precision |
|---|---:|---:|
| model[random_forest] | 0.9828 | 0.9908 |
| ticker_prior | 0.9858 | 0.9846 |
| single_feature[debt_to_assets_latest] | 0.8887 | 0.9379 |
| dummy_most_frequent | 0.5000 | 0.5938 |

## 2. Mô hình so với baseline `ticker_prior`

1. **ΔAUROC (DeLong):** model[random_forest] − ticker_prior = -0.0030, z = -0.3126, p = 0.7546 ⇒ **không** có ý nghĩa ở mức 5% (chưa thể khẳng định mô hình hơn baseline).
2. **ΔAP (paired bootstrap 2000 vòng):** +0.0061, CI 95% = [-0.0048; +0.0215], p = 0.3440 ⇒ kết luận tương tự DeLong.
3. **Cách đọc:** n = 64 mẫu nên khoảng tin cậy rộng; "không khác biệt" nghĩa là *dữ liệu chưa đủ để khẳng định*, không phải bằng chứng mô hình kém. Đây là lý do báo cáo dùng thêm so sánh cross-company (`validation_checks.json`) và lấy `ticker_prior` làm mốc trung thực.

## 3. Mọi cặp so sánh (ΔAUROC DeLong · ΔAP bootstrap)

| A | B | ΔAUROC (A−B) | p (DeLong) | ΔAP (A−B) | CI95 ΔAP | p (bootstrap AP) |
|---|---|---:|---:|---:|---:|---:|
| model[random_forest] | ticker_prior | -0.0030 | 0.7546 | +0.0061 | [-0.0048; +0.0215] | 0.3440 |
| model[random_forest] | single_feature[debt_to_assets_latest] | +0.0941 | 0.007555 | +0.0528 | [+0.0176; +0.0973] | 0.0010 |
| model[random_forest] | dummy_most_frequent | +0.4828 | 0 | +0.3970 | [+0.3794; +0.4062] | 0.0000 |
| ticker_prior | single_feature[debt_to_assets_latest] | +0.0972 | 0.01055 | +0.0467 | [+0.0074; +0.0941] | 0.0180 |
| ticker_prior | dummy_most_frequent | +0.4858 | 0 | +0.3909 | [+0.3703; +0.4042] | 0.0000 |
| single_feature[debt_to_assets_latest] | dummy_most_frequent | +0.3887 | 0 | +0.3442 | [+0.2962; +0.3828] | 0.0000 |

> Đọc bảng: p < 0,05 ⇒ khác biệt khó giải thích bằng ngẫu nhiên. Nếu `model[random_forest]` không khác `ticker_prior` về ý nghĩa thống kê thì kết luận trung thực là lợi thế của mô hình **chưa được chứng minh** trên bộ test này (n nhỏ) — phải đọc kèm kết quả cross-company.

