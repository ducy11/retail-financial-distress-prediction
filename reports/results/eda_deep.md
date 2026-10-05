# EDA chuyên sâu — feature, nhãn, tương quan, dịch chuyển

*Sinh tự động bởi `python -m scripts.eda_deep` (module: `forecasting/eda.py`). Không có số nào nhập tay.*

## 0. Phạm vi

- Mẫu: train=212, validation=32, test=64, purged=16
- Feature: **47** cột (từ `forecasting.features.feature_names()`).
- Liên hệ feature–nhãn và tương quan được tính trên **train+validation — test không dùng để mô tả liên hệ**; test chỉ dùng cho chẩn đoán dịch chuyển (train → test).

## 1. Chất lượng ma trận feature

- Feature hằng số: **—**; gần hằng số (>95% một giá trị): **negative_ni_streak**.
- Thiếu > 20%: **8** feature; đuôi nặng: **15**; nhiều outlier IQR: **32**.

| Feature | Thiếu % | #giá trị | Median | P1–P99 | Skew | Kurtosis | Outlier IQR % | = 0 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| receivables_to_sales_yoy | 40.5738 | 145 | 0.007345 | -0.454116 … 0.762838 | 3.50968 | 27.0592 | 8.96552 | 0 |
| receivables_to_sales_latest | 32.377 | 165 | 0.050641 | 0.021041 … 0.102237 | 0.214321 | -1.47544 | 0 | 0 |
| operating_margin_yoy | 28.2787 | 175 | -0.014209 | -2.295724 … 2.292017 | -3.79438 | 39.3459 | 19.4286 | 0 |
| operating_income_yoy_growth | 28.2787 | 175 | 0.048531 | -2.019102 … 2.91259 | 0.586485 | 18.8863 | 15.4286 | 0 |
| ocf_to_sales_yoy | 27.0492 | 178 | 0.00014 | -2.626075 … 5.242927 | 5.7255 | 51.3623 | 15.1685 | 0 |
| operating_cash_flow_yoy_growth | 27.0492 | 178 | 0.075203 | -2.845131 … 5.395316 | 6.48374 | 60.4336 | 16.2921 | 0 |
| net_margin_yoy | 25 | 183 | 0.006118 | -2.477048 … 6.270881 | 11.9492 | 152.775 | 19.6721 | 0 |
| net_income_yoy_growth | 25 | 183 | 0.092659 | -1.863227 … 8.113403 | 11.3431 | 137.948 | 17.4863 | 0 |
| operating_margin_latest | 16.8033 | 203 | 0.096018 | -0.035219 … 0.225978 | -1.45748 | 9.34805 | 0.985222 | 0 |
| ocf_to_sales_latest | 13.9344 | 210 | 0.09681 | -0.159365 … 0.398105 | -1.20119 | 9.38285 | 4.28571 | 0 |
| ocf_to_sales_min_window | 13.9344 | 55 | 0.043286 | -0.574406 … 0.191266 | -2.2836 | 5.69173 | 7.61905 | 0 |
| net_margin_latest | 13.5246 | 211 | 0.078107 | -0.048781 … 0.164165 | -1.71297 | 9.43119 | 0.947867 | 0 |
| net_margin_min_window | 13.5246 | 61 | 0.052068 | -0.251778 … 0.139353 | -1.88155 | 3.86582 | 7.58294 | 0 |
| gross_margin_yoy | 13.1148 | 212 | -0.002376 | -0.417086 … 1.159562 | 13.3983 | 188.206 | 15.566 | 0 |
| sgna_pct_revenue_yoy | 13.1148 | 212 | 0.004312 | -0.320298 … 0.482996 | 1.79335 | 15.3247 | 12.7358 | 0 |
- debt_to_equity_latest: skew=-13.713345, kurtosis=203.976011, outlier IQR=27.459016%
- retained_to_assets_yoy: skew=-13.911288, kurtosis=198.769397, outlier IQR=7.075472%
- gross_margin_yoy: skew=13.398265, kurtosis=188.20579, outlier IQR=15.566038%
- net_margin_yoy: skew=11.949164, kurtosis=152.774887, outlier IQR=19.672131%
- net_income_yoy_growth: skew=11.343104, kurtosis=137.94847, outlier IQR=17.486339%

![Thiếu theo feature × split](../reports/figures/eda_deep/01_missing_pct_by_split.png)

![Missingness mang thông tin nhãn](../reports/figures/eda_deep/07_missingness_information.png)

## 2. Nhãn: mất cân bằng, entropy, chuỗi trạng thái

| Tập | n | Dương | Tỉ lệ dương % | IR | Entropy (bit) | Entropy chuẩn hoá | Gini | Baseline đa số % | Mức |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 212 | 132 | 62.2642 | 1.65 | 0.956155 | 0.956155 | 0.469918 | 62.2642 | slightly_imbalanced |
| validation | 32 | 21 | 65.625 | 1.90909 | 0.928362 | 0.928362 | 0.451172 | 65.625 | slightly_imbalanced |
| test | 64 | 38 | 59.375 | 1.46154 | 0.974489 | 0.974489 | 0.482422 | 59.375 | balanced |
| purged | 16 | 11 | 68.75 | 2.2 | 0.896038 | 0.896038 | 0.429688 | 68.75 | slightly_imbalanced |

| Công ty | n | Tỉ lệ dương % | IR | Mức | Lần đổi nhãn | Chuỗi 1 dài nhất | Chuỗi 0 dài nhất |
|---|---:|---:|---:|---|---:|---:|---:|
| HD | 43 | 100 | — | severely_imbalanced | 0 | 43 | 0 |
| LOW | 43 | 100 | — | severely_imbalanced | 0 | 43 | 0 |
| WMT | 47 | 100 | — | severely_imbalanced | 0 | 47 | 0 |
| ORLY | 43 | 90.6977 | 9.75 | severely_imbalanced | 5 | 37 | 2 |
| DG | 31 | 54.8387 | 1.21429 | balanced | 2 | 17 | 11 |
| FIVE | 31 | 19.3548 | 4.16667 | severely_imbalanced | 12 | 1 | 7 |
| DKS | 43 | 11.6279 | 7.6 | severely_imbalanced | 6 | 3 | 17 |
| ROST | 43 | 4.65116 | 20.5 | severely_imbalanced | 4 | 1 | 21 |

- Chuyển trạng thái giữa hai quý liên tiếp: `{'0->0': 103, '0->1': 15, '1->0': 14, '1->1': 184}`; P(1|1) = 0.929293, P(1|0) = 0.127119, giữ nguyên nhãn = **90.822785%**.
- Thực thể: **8** công ty, số thực thể hiệu dụng = **7.848086**; 41.049383% mẫu thuộc công ty chỉ có một lớp.
- Mức mất cân bằng nặng nhất: ROST (IR=20.5), ORLY (IR=9.75), DKS (IR=7.6).

![Tỉ lệ nhãn theo công ty](../reports/figures/eda_deep/02_label_by_ticker.png)

## 3. Quan hệ feature ↔ nhãn

- 47/47 feature tính được AUC 1-feature; 11 feature có |2·AUC−1| ≥ 0.30; 11 feature còn ý nghĩa sau BH-FDR.
- Lưu ý: 47 kiểm định ⇒ q-value BH; xếp hạng theo |2·AUC−1| để không phụ thuộc cỡ mẫu, không dùng p-value đơn lẻ.

| Feature | #cặp | AUC | Hướng | r (point-biserial) | q-value BH | MI | Lift decile trên |
|---|---:|---:|---|---:|---:|---:|---:|
| current_ratio_min_window | 244 | 0.0481 | giá trị thấp ⇒ nhãn 1 | -0.6965 | 0 | 0.4317 | 0.1914 |
| current_ratio_latest | 244 | 0.0738 | giá trị thấp ⇒ nhãn 1 | -0.6879 | 0 | 0.3671 | 0.2552 |
| working_capital_to_assets | 244 | 0.0776 | giá trị thấp ⇒ nhãn 1 | -0.6838 | 0 | 0.3355 | 0.1276 |
| debt_to_assets_latest | 244 | 0.8787 | giá trị cao ⇒ nhãn 1 | 0.623 | 0 | 0.3399 | 1.5948 |
| receivables_to_sales_latest | 165 | 0.8665 | giá trị cao ⇒ nhãn 1 | 0.5704 | 0 | 0.2169 | 1.5566 |
| quick_ratio_latest | 244 | 0.2185 | giá trị thấp ⇒ nhãn 1 | -0.5078 | 0 | 0.1722 | 0.2552 |
| debt_to_equity_latest | 244 | 0.7153 | giá trị cao ⇒ nhãn 1 | -0.0353 | 0.814958 | 0.3397 | 1.5948 |
| retained_to_assets_yoy | 212 | 0.2997 | giá trị thấp ⇒ nhãn 1 | -0.0818 | 0.442702 | 0.153 | 0.5506 |
| total_assets_yoy_growth | 212 | 0.3154 | giá trị thấp ⇒ nhãn 1 | -0.2544 | 0.000849 | 0.0461 | 0.6195 |
| cash_to_assets_latest | 244 | 0.3372 | giá trị thấp ⇒ nhãn 1 | -0.398 | 0 | 0.1565 | 0.3827 |
| revenue_cv | 236 | 0.3484 | giá trị thấp ⇒ nhãn 1 | -0.3364 | 1e-06 | 0.0775 | 0.2622 |
| debt_to_equity_yoy | 212 | 0.6454 | giá trị cao ⇒ nhãn 1 | -0.0252 | 0.889346 | 0.095 | 1.3766 |
| current_assets_yoy_growth | 212 | 0.356 | giá trị thấp ⇒ nhãn 1 | -0.2213 | 0.003969 | 0.0251 | 0.6883 |
| cost_of_sales_yoy_growth | 212 | 0.3582 | giá trị thấp ⇒ nhãn 1 | -0.2063 | 0.007024 | 0.0439 | 0.6883 |
| revenue_yoy_growth | 212 | 0.3605 | giá trị thấp ⇒ nhãn 1 | -0.1933 | 0.012381 | 0.0234 | 0.6195 |

![Liên hệ feature ↔ nhãn](../reports/figures/eda_deep/03_target_association.png)

![ECDF top feature theo nhãn](../reports/figures/eda_deep/06_top_feature_ecdf.png)

## 4. Quan hệ feature ↔ feature (đa cộng tuyến)

- **8** cặp có |r| ≥ 0.9; **15** feature nằm trong 7 cụm thông tin.
- Số chiều hiệu dụng (participation ratio) = **12.891743** trên 47 cột; cần **22** thành phần cho 95% phương sai.

**Cụm mạnh nhất (|r| ≥ 0.9):**

| # | Cụm feature | Số cột |
|---:|---|---:|
| 1 | current_ratio_latest, current_ratio_min_window, working_capital_to_assets | 3 |
| 2 | cash_and_equivalents_yoy_growth, cash_to_assets_yoy | 2 |
| 3 | cost_of_sales_yoy_growth, revenue_yoy_growth | 2 |
| 4 | net_income_yoy_growth, net_margin_yoy | 2 |
| 5 | net_margin_latest, operating_margin_latest | 2 |
| 6 | ocf_to_sales_yoy, operating_cash_flow_yoy_growth | 2 |
| 7 | operating_income_yoy_growth, operating_margin_yoy | 2 |

| Feature A | Feature B | Pearson | Spearman |
|---|---|---:|---:|
| cash_to_assets_yoy | cash_and_equivalents_yoy_growth | 0.9977 | 0.9663 |
| net_margin_yoy | net_income_yoy_growth | 0.9917 | 0.9399 |
| ocf_to_sales_yoy | operating_cash_flow_yoy_growth | 0.9899 | 0.9895 |
| operating_margin_latest | net_margin_latest | 0.9783 | 0.9727 |
| current_ratio_latest | working_capital_to_assets | 0.9665 | 0.982 |
| revenue_yoy_growth | cost_of_sales_yoy_growth | 0.9583 | 0.9522 |
| operating_margin_yoy | operating_income_yoy_growth | 0.9405 | 0.9295 |
| current_ratio_latest | current_ratio_min_window | 0.9302 | 0.927 |
| working_capital_to_assets | current_ratio_min_window | 0.8841 | 0.9064 |
| ocf_to_sales_min_window | net_margin_min_window | 0.861 | 0.8806 |
- Đọc cột Spearman để biết tương quan có bị outlier chi phối không.

![Heatmap tương quan theo cụm](../reports/figures/eda_deep/04_correlation_clustered.png)

## 5. Dịch chuyển phân phối train → test

- **5** feature vượt ngưỡng cảnh báo (KS ≥ 0.3 hoặc |SMD| ≥ 0.5); 41 feature có PSI > 0.2; 13 feature khác biệt có ý nghĩa sau BH-FDR.
- Cảnh báo đọc số: PSI chia 10 khoảng trên mẫu test chỉ 64 dòng nên mỗi khoảng ~6 quan sát ⇒ PSI nhạy nhiễu, dùng KS/SMD làm tiêu chí chính, PSI để theo dõi về sau.

| Feature | Mean train | Mean test | KS | KS p-value | q-value BH | SMD | PSI |
|---|---:|---:|---:|---:|---:|---:|---:|
| debt_to_assets_yoy | 0.067472 | -0.026369 | 0.6208 | 0 | 0 | -0.9284 | 5.61497 |
| debt_to_equity_yoy | 0.060465 | -0.160156 | 0.5608 | 0 | 0 | -0.1148 | 5.39814 |
| retained_to_assets_yoy | -0.646426 | 0.086477 | 0.3983 | 3.1e-07 | 5e-06 | 0.1756 | 1.82069 |
| receivables_to_sales_latest | 0.057035 | 0.078873 | 0.3268 | 0.00458353 | 0.021543 | 0.6559 | 4.0499 |
| revenue_per_asset_latest | 0.445017 | 0.386103 | 0.3249 | 4.114e-05 | 0.000483 | -0.4555 | 2.56747 |
| net_income_yoy_growth | 0.688024 | 0.550609 | 0.279 | 0.00250755 | 0.015634 | -0.0328 | 0.636642 |
| cost_of_sales_yoy_growth | 0.089944 | 0.054042 | 0.2788 | 0.0009715 | 0.009132 | -0.3077 | 0.556629 |
| current_assets_yoy_growth | 0.122252 | 0.052636 | 0.2656 | 0.0019652 | 0.015394 | -0.4539 | 0.566493 |
| sgna_pct_revenue_yoy | -0.001156 | 0.016934 | 0.2597 | 0.00266115 | 0.015634 | 0.1587 | 0.307948 |
| net_margin_yoy | 0.446498 | 0.423091 | 0.2592 | 0.0062354 | 0.026642 | -0.0068 | 0.346252 |
| receivables_to_sales_yoy | 0.013311 | 0.072009 | 0.2529 | 0.0618647 | 0.126419 | 0.2972 | 0.382145 |
| retained_to_assets_latest | 0.314169 | 0.268523 | 0.25 | 0.00338261 | 0.017665 | -0.1362 | 2.58863 |
| gross_margin_yoy | 0.079427 | 0.007445 | 0.2365 | 0.00822391 | 0.030773 | -0.1073 | 0.407503 |
| debt_to_assets_latest | 0.740463 | 0.835593 | 0.2311 | 0.0085117 | 0.030773 | 0.4576 | 1.50654 |
| ocf_to_sales_min_window | -0.002871 | 0.027356 | 0.2291 | 0.011621 | 0.039013 | 0.2405 | 3.58629 |

![Dịch chuyển train → test](../reports/figures/eda_deep/05_drift_ks_vs_smd.png)

## 6. Rò rỉ tiềm ẩn & tính độc lập của mẫu

- Hai mẫu liên tiếp cùng công ty chia sẻ lịch sử (Jaccard): mean **0.916954**, median 0.952381, min 0.5 trên 316 cặp.
- Kỳ lịch sử của test đã có trong train: **65.432099%**; kỳ target của test xuất hiện trong lịch sử train: **0** mẫu.
- Dòng feature trùng khít test↔train: **0**; trùng trong nội bộ train: 0.
- Thứ tự thời gian nhãn: 0 mẫu có nhãn công bố TRƯỚC khi kỳ target kết thúc (kỳ vọng 0); khoảng cách trung vị 34 ngày.
- Mật độ thiếu mỗi mẫu: trung bình 4.568182/47 feature, cao nhất 29; tỉ lệ mẫu thiếu > 50% feature = 0.103896.

**Feature thiếu mang thông tin nhãn (top 10 theo |Δ|):**

| Feature | #thiếu | Tỉ lệ nhãn 1 khi THIẾU % | khi CÓ % | Δ điểm % | q-value BH |
|---|---:|---:|---:|---:|---:|
| net_margin_latest | 38 | 13.1579 | 68.8889 | -55.731 | 0 |
| net_margin_min_window | 38 | 13.1579 | 68.8889 | -55.731 | 0 |
| operating_margin_latest | 46 | 21.7391 | 69.084 | -47.3448 | 0 |
| net_margin_yoy | 69 | 26.087 | 72.3849 | -46.298 | 0 |
| net_income_yoy_growth | 69 | 26.087 | 72.3849 | -46.298 | 0 |
| operating_margin_yoy | 77 | 29.8701 | 72.7273 | -42.8571 | 0 |
| operating_income_yoy_growth | 77 | 29.8701 | 72.7273 | -42.8571 | 0 |
| ocf_to_sales_latest | 34 | 35.2941 | 65.3285 | -30.0343 | 0.003664 |
| ocf_to_sales_min_window | 34 | 35.2941 | 65.3285 | -30.0343 | 0.003664 |
| revenue_cv | 8 | 37.5 | 62.6667 | -25.1667 | 0.280863 |

**Cặp feature thường thiếu cùng nhau (top 5):**

| Feature A | Feature B | #thiếu cùng | Jaccard |
|---|---|---:|---:|
| receivables_to_sales_latest | receivables_to_sales_yoy | 110 | 0.839695 |
| operating_margin_yoy | operating_income_yoy_growth | 77 | 1 |
| net_margin_yoy | net_income_yoy_growth | 69 | 1 |
| ocf_to_sales_yoy | operating_cash_flow_yoy_growth | 66 | 1 |
| net_margin_yoy | ocf_to_sales_yoy | 50 | 0.588235 |

## 7. Kết luận tự động

1. **Mức mất cân bằng (cấp mẫu):** entropy nhãn 0.955564 bit (tối đa 1.0), IR = 1.655738, lớp thiểu số 37.654321% ⇒ slightly_imbalanced; đoán lớp đa số đã đạt 62.345679% ⇒ không dùng Accuracy làm thước đo chính.
2. **Nhãn gần như là thuộc tính thực thể:** 90.822785% cặp quý liên tiếp giữ nguyên nhãn; P(nhãn=1 | quý trước nhãn=1) = 0.929293, P(nhãn=1 | quý trước nhãn=0) = 0.127119; 41.049383% mẫu thuộc công ty chỉ có MỘT lớp.
3. **Không được chia tập ngẫu nhiên theo mẫu:** công ty nhãn đơn lớp = HD, LOW, WMT ⇒ mọi đánh giá phải chia theo nhóm (`GroupKFold`/LOCO) và nêu IR theo công ty.
4. **Feature vô dụng/giả tín hiệu:** hằng số = —; gần hằng số (>95% một giá trị) = negative_ni_streak ⇒ cân nhắc loại.
5. **Feature bị impute chi phối** (thiếu > 20%): receivables_to_sales_yoy, receivables_to_sales_latest, operating_margin_yoy, operating_income_yoy_growth, ocf_to_sales_yoy, operating_cash_flow_yoy_growth, net_margin_yoy, net_income_yoy_growth ⇒ đọc kèm kiểm định missingness-mang-nhãn.
6. **Đuôi nặng/lệch mạnh** (|skew| > 5 hoặc kurtosis > 20): gross_margin_yoy, operating_margin_yoy, net_margin_yoy, debt_to_equity_latest, debt_to_equity_yoy, receivables_to_sales_yoy, cash_to_assets_yoy, ocf_to_sales_yoy, retained_to_assets_yoy, revenue_yoy_growth, cost_of_sales_yoy_growth, operating_cash_flow_yoy_growth ⇒ nên thêm winsorize/clip và dùng metric xếp hạng.
7. **Nhiều outlier theo IQR** (> 5% mẫu): gross_margin_latest, gross_margin_yoy, operating_margin_yoy, net_margin_yoy, sgna_pct_revenue_latest, sgna_pct_revenue_yoy, quick_ratio_latest, quick_ratio_yoy, debt_to_assets_yoy, debt_to_equity_latest, debt_to_equity_yoy, inventory_to_sales_yoy.
8. **Liên hệ feature ↔ nhãn:** 47/47 feature đủ mẫu để tính AUC 1-feature; 11 feature có |2·AUC−1| ≥ 0.30; 11 feature còn ý nghĩa sau BH-FDR.
9. **Đa cộng tuyến:** 8 cặp |r| ≥ 0.9, gom thành 7 cụm; số chiều hiệu dụng = 12.891743 (trên 47 cột).
10. **Dịch chuyển train → test:** 5 feature vượt ngưỡng (KS ≥ 0.3 hoặc |SMD| ≥ 0.5), 41 feature có PSI > 0.2.
11. **Mẫu không độc lập:** Jaccard lịch sử trung bình giữa hai mẫu liên tiếp cùng công ty = 0.916954 ⇒ số quan sát độc lập nhỏ hơn số mẫu; 0 dòng test trùng khít feature với train.
12. **Kiểm tra thứ tự nhãn:** 0 mẫu có `label_available_on` ≤ ngày kết thúc kỳ target (kỳ vọng 0); khoảng cách trung vị 34 ngày.
13. **Missingness:** 22 feature có tỉ lệ nhãn khác biệt có ý nghĩa giữa nhóm thiếu và nhóm có dữ liệu (BH-FDR < 5%) ⇒ dữ liệu thiếu theo cơ chế MNAR, median-impute một mình sẽ xoá tín hiệu; đề xuất thêm cờ `is_missing` khi tái huấn luyện.

## 8. Hình

- `figures\eda_deep\01_missing_pct_by_split.png`
- `figures\eda_deep\02_label_by_ticker.png`
- `figures\eda_deep\03_target_association.png`
- `figures\eda_deep\04_correlation_clustered.png`
- `figures\eda_deep\05_drift_ks_vs_smd.png`
- `figures\eda_deep\06_top_feature_ecdf.png`
- `figures\eda_deep\07_missingness_information.png`
