# Kỹ thuật xử lý lệch lớp trên DỮ LIỆU THẬT (8 công ty)

- Giao thức: GroupKFold(4) theo mã cổ phiếu trên train+validation (244 mẫu); sampler nằm trong imblearn.Pipeline ⇒ chỉ chạm fold-train; test fold được so khớp ma trận trước/sau fit
- Mô hình nền dùng chung: **hist_gradient_boosting** (để mọi kỹ thuật khác nhau CHỈ ở bước xử lý lệch lớp).
- Mất cân bằng corpus: IR cấp mẫu ≈ 1.68; nhưng HD/LOW/WMT có 100% nhãn 1 (mất cân bằng cấp thực thể).

## 1. Nhận xét tự động

1. **Đối chứng (không can thiệp):** cross-company AP = 0.9588, AUROC = 0.9426 (IR cấp mẫu của corpus ≈ 1.68 — mất cân bằng NHẸ).
2. **Kết quả trên dữ liệu thật:** 1 kỹ thuật hơn đối chứng > 0,5 điểm % AP, 2 kỹ thuật kém hơn > 0,5 điểm %. Tốt nhất: **smote_enn** (AP 0.9683).
3. **Cách đọc:** AP ở đây là OOF cross-company — mỗi công ty bị giữ trọn ra ngoài, nên kết quả không thể đến từ việc "nhớ mặt công ty". Vì vậy đây là phép thử công bằng cho kỹ thuật resampling/cost-sensitive, khác hẳn so sánh in-domain.
4. **Lưu ý về CHIỀU của mất cân bằng (khác bộ dữ liệu phá sản thông thường):** trong corpus này lớp 1 (suy giảm) chiếm **62%** ⇒ lớp THIỂU SỐ là lớp 0. Vì vậy SMOTE ở đây **sinh thêm mẫu "không suy giảm"**, tức can thiệp theo hướng ngược với thực hành phổ biến — một lý do nữa để mất cân bằng không phải nút thắt của bài toán này.
5. **Kiểm định cặp cho kỹ thuật tốt nhất** (`smote_enn` vs `none`, cùng 244 mẫu OOF): ΔAP = +0.0095 (CI 95% [-0.0054; +0.0269], p = 0.21), ΔAUROC (DeLong) = +0.0087 (p = 0.4741232109103479) ⇒ **chưa** đủ căn cứ khẳng định hơn đối chứng (khoảng tin cậy chứa 0).
6. **Chống rò rỉ:** mọi kỹ thuật đều có sampler nằm trong pipeline ⇒ tập test của từng fold KHÔNG bị resample (đã kiểm tra bằng so khớp ma trận trước/sau fit).

## 2. Bảng kết quả (xếp theo AP out-of-fold)

| Kỹ thuật | Nhóm | AP (OOF) | AUROC (OOF) | F1* trên OOF | IR trước → sau | n train trước → sau | Test fold nguyên vẹn |
|---|---|---:|---:|---:|---:|---:|---|
| smote_enn | hybrid | 0.9683 | 0.9512 | 0.9158 | 1.32 → 1.12 | 186 → 159 | PASS |
| none | baseline | 0.9588 | 0.9426 | 0.9419 | — | — | PASS |
| class_weight | algorithm-level | 0.9569 | 0.9404 | 0.9419 | — | — | PASS |
| random_under | undersampling | 0.9561 | 0.9267 | 0.9265 | 1.32 → 1.00 | 186 → 160 | PASS |
| smote | oversampling | 0.9502 | 0.9500 | 0.9311 | 1.32 → 1.00 | 186 → 212 | PASS |
| tomek_links | undersampling | 0.9428 | 0.9160 | 0.9159 | 1.32 → 1.23 | 186 → 178 | PASS |

> Đọc bảng: cột **IR trước → sau** chứng minh can thiệp thực sự diễn ra trên fold-train; cột **Test fold nguyên vẹn** là bài kiểm tra chống rò rỉ (sampler không được chạm validation).

## 3. Kiểm định cặp (kỹ thuật tốt nhất vs đối chứng)

| So sánh | ΔAP | CI 95% ΔAP | p (bootstrap AP) | ΔAUROC | p (DeLong) |
|---|---:|---:|---:|---:|---:|
| smote_enn vs none | +0.0095 | [-0.0054; +0.0269] | 0.21 | +0.0087 | 0.4741232109103479 |

## 4. Hình

- `figures\imbalance_real\01_techniques_real.png`
