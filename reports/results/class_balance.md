# Mất cân bằng lớp của nhãn `is_distressed` (sinh tự động)

- Cột mục tiêu: **`is_distressed`** trong `data/prepared/*.json` (1 = suy giảm tài chính).
- Đếm thuần Python (repo không dùng pandas) — tương đương `value_counts()` và `value_counts(normalize=True) * 100`.
- Ngưỡng phân loại: thiểu số ≥ 40% → cân bằng; < 20% → nghiêm trọng.

## Theo tập

| Tập | n | Lớp 1 (distress) | Lớp 0 | % thiểu số | IR (đa số/thiểu số) | Acc nếu đoán lớp đa số | Mức |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 212 | 132 (62.3%) | 80 (37.7%) | 37.7% | 1.65 | 62.3% | mất cân bằng NHẸ |
| validation | 32 | 21 (65.6%) | 11 (34.4%) | 34.4% | 1.91 | 65.6% | mất cân bằng NHẸ |
| test | 64 | 38 (59.4%) | 26 (40.6%) | 40.6% | 1.46 | 59.4% | CÂN BẰNG |
| purged | 16 | 11 (68.8%) | 5 (31.2%) | 31.2% | 2.20 | 68.8% | mất cân bằng NHẸ |

## Toàn bộ corpus (train+validation+test+purged)

| Phạm vi | n | Lớp 1 (distress) | Lớp 0 | % thiểu số | IR (đa số/thiểu số) | Acc nếu đoán lớp đa số | Mức |
|---|---:|---:|---:|---:|---:|---:|---|
| toàn corpus | 324 | 202 (62.3%) | 122 (37.7%) | 37.7% | 1.66 | 62.3% | mất cân bằng NHẸ |

## Theo công ty (nhãn gần như là thuộc tính thực thể)

| Công ty | n | Lớp 1 (distress) | Lớp 0 | % thiểu số | IR (đa số/thiểu số) | Acc nếu đoán lớp đa số | Mức |
|---|---:|---:|---:|---:|---:|---:|---|
| DG | 31 | 17 (54.8%) | 14 (45.2%) | 45.2% | 1.21 | 54.8% | CÂN BẰNG |
| DKS | 43 | 5 (11.6%) | 38 (88.4%) | 11.6% | 7.60 | 88.4% | mất cân bằng NGHIÊM TRỌNG |
| FIVE | 31 | 6 (19.4%) | 25 (80.6%) | 19.4% | 4.17 | 80.6% | mất cân bằng NGHIÊM TRỌNG |
| HD | 43 | 43 (100.0%) | 0 (0.0%) | 0.0% | inf | 100.0% | chỉ một lớp |
| LOW | 43 | 43 (100.0%) | 0 (0.0%) | 0.0% | inf | 100.0% | chỉ một lớp |
| ORLY | 43 | 39 (90.7%) | 4 (9.3%) | 9.3% | 9.75 | 90.7% | mất cân bằng NGHIÊM TRỌNG |
| ROST | 43 | 2 (4.7%) | 41 (95.3%) | 4.7% | 20.50 | 95.3% | mất cân bằng NGHIÊM TRỌNG |
| WMT | 47 | 47 (100.0%) | 0 (0.0%) | 0.0% | inf | 100.0% | chỉ một lớp |

## Nhãn quy tắc tái lập được (`data/prepared-rule`)

| Phạm vi | n | Lớp 1 (distress) | Lớp 0 | % thiểu số | IR (đa số/thiểu số) | Acc nếu đoán lớp đa số | Mức |
|---|---:|---:|---:|---:|---:|---:|---|
| nhãn quy tắc | 324 | 137 (42.3%) | 187 (57.7%) | 42.3% | 1.36 | 57.7% | CÂN BẰNG |

## Đọc kết quả

- Toàn corpus: **202 mẫu lớp 1 / 122 mẫu lớp 0** (62.3% / 37.7%), IR = **1.66** → **mất cân bằng NHẸ**.
- Train: IR = 1.65 (mất cân bằng NHẸ); validation: IR = 1.91; test: IR = 1.46 (CÂN BẰNG).
- Ở cấp CÔNG TY mức mất cân bằng nặng hơn hẳn: DG 45% [CÂN BẰNG]; DKS 12% [mất cân bằng NGHIÊM TRỌNG]; FIVE 19% [mất cân bằng NGHIÊM TRỌNG]; HD 0% [chỉ một lớp]; LOW 0% [chỉ một lớp]; ORLY 9% [mất cân bằng NGHIÊM TRỌNG]; ROST 5% [mất cân bằng NGHIÊM TRỌNG]; WMT 0% [chỉ một lớp].
- Đường cơ sở “đoán lớp đa số” đã đạt 59.4% accuracy trên test ⇒ Accuracy là chỉ số gây hiểu nhầm; phải dùng PR-AUC/AP, F1, recall tại ngưỡng chi phí.
- Repo đã xử lý một phần: `class_weight` bật cho random_forest, hist_gradient_boosting; `forecasting/tuning` refit theo **average_precision**; `forecasting/evaluation` báo cáo F1/macro-F1, MCC và ngưỡng tối ưu theo chi phí `COST_FN`/`COST_FP`.
- **Không** dùng SMOTE ở bộ dữ liệu này: nhãn gần như là thuộc tính công ty và mọi đánh giá đều chia theo nhóm (`GroupKFold`/LOCO); sinh mẫu tổng hợp trong không gian 47 chiều từ 212 mẫu train của 8 thực thể sẽ tạo quan sát “của chính công ty đã có”, tức hợp thức hoá đúng loại rò rỉ cấp thực thể mà đồ án đang đo.
