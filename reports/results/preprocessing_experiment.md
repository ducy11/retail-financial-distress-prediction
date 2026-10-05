# Thí nghiệm tiền xử lý trên dữ liệu thật (winsorize × scaler)

- Giao thức: fit **chỉ trên train (212 mẫu)**; đo in-domain trên validation (32 mẫu), ngưỡng chọn theo best-F1 trên chính validation; tổng quát hoá bằng GroupKFold(4) trên train+validation, gộp xác suất out-of-fold; KHÔNG dùng để chọn cấu hình (giữ vai trò chốt cuối).
- Số cấu hình đã chạy: **24** (4 họ mô hình × scaler × winsorize).
- Tham chiếu: `logistic` + `standard` + không winsorize (= pipeline chính hiện tại).

## 1. Nhận xét tự động

1. **Cấu hình pipeline chính** (logistic + standard + không winsorize): Val AP = 0.9869, cross-company AP = 0.9232.
2. **Tốt nhất theo validation:** mlp + standard + winsorize=p1p99 → Val AP = 0.9940 (+0.72 điểm %), cross-company AP = 0.7365.
3. **Tốt nhất theo cross-company:** random_forest + none + winsorize=iqr → cross-company AP = 0.9590 (+3.58 điểm %).
4. **Riêng tác động của winsorize** (cùng model/scaler, so với không clip): trung bình +0.74 điểm % cross-company AP, tốt nhất +5.37, xấu nhất -4.27 trên 16 cặp so sánh ⇒ cải thiện rõ về trung bình, ĐẶC BIỆT cho mô hình tuyến tính; xem mục dưới để biết tác động trên đúng mô hình được chốt.
5. **Trên ĐÚNG mô hình được chốt (`random_forest`, đọc từ `summary.json`)**: winsorize=iqr cho cross-company AP 0.9590 so với 0.9577 khi không clip ⇒ +0.13 điểm % — nằm trong khoảng nhiễu của 244 mẫu out-of-fold, nên **pipeline chính giữ không winsorize** (đơn giản, dễ diễn giải) và winsorize chỉ được dùng như một biến thể ablation; muốn đổi mặc định cần thêm công ty/dữ liệu.
6. **Scaler cho mô hình tuyến tính** (không winsorize): tốt nhất là `standard` với cross-company AP = 0.9232 (StandardScaler = 0.9232) ⇒ khác biệt không đáng kể (dưới 0,5 điểm %), nên giữ StandardScaler cho gọn và ghi lại kết quả âm này như một kết luận trung thực.

## 2. Bảng đầy đủ (xếp theo cross-company AP)

| Mô hình | Scaler | Winsorize | Val AP | Val AUROC | Val F1* | Cross-co. AP | Cross-co. AUROC | % clip trên val | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| random_forest | none | iqr | 0.9894 | 0.9740 | 0.9756 | 0.9590 | 0.9451 | 8.31 | 0.0424 |
| random_forest | robust | iqr | 0.9894 | 0.9740 | 0.9756 | 0.9590 | 0.9451 | 8.31 | 0.0424 |
| random_forest | none | p1p99 | 0.9869 | 0.9654 | 0.9756 | 0.9578 | 0.9433 | 4.78 | 0.0455 |
| random_forest | robust | p1p99 | 0.9869 | 0.9654 | 0.9756 | 0.9578 | 0.9433 | 4.78 | 0.0455 |
| random_forest | none | none | 0.9869 | 0.9654 | 0.9756 | 0.9577 | 0.9431 | — | 0.0459 |
| random_forest | robust | none | 0.9869 | 0.9654 | 0.9756 | 0.9577 | 0.9431 | — | 0.0459 |
| hist_gradient_boosting | none | iqr | 0.9908 | 0.9784 | 0.9756 | 0.9575 | 0.9416 | 8.31 | 0.0365 |
| hist_gradient_boosting | robust | iqr | 0.9908 | 0.9784 | 0.9756 | 0.9575 | 0.9416 | 8.31 | 0.0365 |
| hist_gradient_boosting | none | none | 0.9894 | 0.9740 | 0.9756 | 0.9569 | 0.9404 | — | 0.0368 |
| hist_gradient_boosting | none | p1p99 | 0.9894 | 0.9740 | 0.9756 | 0.9569 | 0.9404 | 4.78 | 0.0368 |
| hist_gradient_boosting | robust | none | 0.9894 | 0.9740 | 0.9756 | 0.9569 | 0.9404 | — | 0.0368 |
| hist_gradient_boosting | robust | p1p99 | 0.9894 | 0.9740 | 0.9756 | 0.9569 | 0.9404 | 4.78 | 0.0368 |
| logistic | standard | p1p99 | 0.9908 | 0.9784 | 0.9756 | 0.9417 | 0.9104 | 4.78 | 0.0433 |
| logistic | standard | iqr | 0.9881 | 0.9697 | 0.9756 | 0.9377 | 0.9081 | 8.31 | 0.0421 |
| logistic | standard | none | 0.9869 | 0.9654 | 0.9756 | 0.9232 | 0.8896 | — | 0.0505 |
| logistic | robust | iqr | 0.9857 | 0.9610 | 0.9756 | 0.9071 | 0.8672 | 8.31 | 0.0513 |
| logistic | robust | p1p99 | 0.9908 | 0.9784 | 0.9756 | 0.8936 | 0.8474 | 4.78 | 0.0455 |
| logistic | robust | none | 0.9881 | 0.9697 | 0.9756 | 0.8676 | 0.8102 | — | 0.0514 |
| mlp | standard | none | 0.9932 | 0.9870 | 0.9767 | 0.7792 | 0.7083 | — | 0.0790 |
| mlp | standard | iqr | 0.9857 | 0.9610 | 0.9756 | 0.7562 | 0.6578 | 8.31 | 0.0756 |
| mlp | standard | p1p99 | 0.9940 | 0.9870 | 0.9756 | 0.7365 | 0.6417 | 4.78 | 0.0761 |
| mlp | robust | iqr | 0.9790 | 0.9481 | 0.9302 | 0.7330 | 0.6177 | 8.31 | 0.1065 |
| mlp | robust | p1p99 | 0.9309 | 0.8788 | 0.9130 | 0.7065 | 0.6358 | 4.78 | 0.1353 |
| mlp | robust | none | 0.9495 | 0.9048 | 0.9130 | 0.6792 | 0.6456 | — | 0.1291 |

> Cột **% clip trên val** = tỉ lệ giá trị thực sự bị cắt khi áp ngưỡng học từ train (đo mức can thiệp, không phải mức cải thiện). Ngưỡng winsorize học **chỉ trên train** nên không rò rỉ; test vẫn không được dùng ở bước này.

## 3. Hình

- `figures\preprocessing\01_winsorize_scaler.png`
