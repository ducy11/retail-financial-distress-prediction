# Phân tích kết quả chuyên sâu (sinh tự động)

Số liệu lấy từ `reports/results/*.json`; hình ở `reports/figures/analysis/`.

## 1. Overfitting: Train vs Validation

| Mô hình | Train AUROC | Val AUROC | Gap AUROC | Train F1 | Val F1 | Gap F1 | Val AP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.984 | 0.965 | 0.019 | 0.935 | 0.976 | -0.040 | 0.987 | 0.051 |
| random_forest | 1.000 | 0.965 | 0.034 | 0.985 | 0.976 | 0.009 | 0.987 | 0.046 |
| hist_gradient_boosting | 1.000 | 0.974 | 0.026 | 1.000 | 0.976 | 0.024 | 0.989 | 0.037 |
| mlp | 0.974 | 0.987 | -0.013 | 0.936 | 0.913 | 0.023 | 0.993 | 0.079 |

## 2. Feature quan trọng nhất (permutation, ΔAUROC khi hoán vị)

| # | Feature | Val ΔAUROC | Test ΔAUROC | Hệ số (n/a — mô hình cây) |
|---:|---|---:|---:|---:|
| 1 | gross_margin_latest | 0.008 | — | — |
| 2 | receivables_to_sales_latest | 0.001 | 0.003 | — |
| 3 | working_capital_to_assets | 0.001 | 0.007 | — |
| 4 | operating_margin_latest | 0.001 | — | — |
| 5 | net_margin_latest | 0.000 | 0.003 | — |
| 6 | retained_to_assets_latest | 0.000 | 0.003 | — |
| 7 | operating_margin_yoy | 0.000 | — | — |
| 8 | net_margin_yoy | 0.000 | — | — |
| 9 | cash_to_assets_latest | 0.000 | 0.001 | — |
| 10 | revenue_per_asset_latest | 0.000 | — | — |

## 3. Đa cộng tuyến (VIF)

Số feature VIF > 10: **33** / 47

| Feature | VIF |
|---|---:|
| cash_to_assets_yoy | 798.1 |
| cash_and_equivalents_yoy_growth | 765.9 |
| revenue_yoy_growth | 288.1 |
| current_ratio_latest | 244.2 |
| revenue_per_asset_yoy | 225.9 |
| working_capital_to_assets | 175.4 |
| current_assets_yoy_growth | 158.0 |
| total_assets_yoy_growth | 144.6 |
| net_income_yoy_growth | 139.1 |
| net_margin_yoy | 136.2 |

## 4. Ablation (bỏ nhóm feature / lọc mẫu non)

| Biến thể | #feature | #train | Val AUROC | Val AP | Test AUROC | Test AP | Test F1 | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tất cả feature | 47 | 212 | 0.965 | 0.987 | 0.983 | 0.991 | 0.949 | 0.934 |
| bỏ nhóm ratios_latest (14 cột) | 33 | 212 | 0.952 | 0.984 | 0.966 | 0.982 | 0.949 | 0.934 |
| bỏ nhóm ratios_yoy (14 cột) | 33 | 212 | 0.974 | 0.989 | 0.974 | 0.985 | 0.949 | 0.934 |
| bỏ nhóm growth (10 cột) | 37 | 212 | 0.970 | 0.988 | 0.985 | 0.993 | 0.949 | 0.934 |
| bỏ nhóm structure (1 cột) | 46 | 212 | 0.961 | 0.986 | 0.982 | 0.990 | 0.949 | 0.934 |
| bỏ nhóm stress (2 cột) | 45 | 212 | 0.965 | 0.987 | 0.986 | 0.993 | 0.949 | 0.934 |
| bỏ nhóm path (6 cột) | 41 | 212 | 0.961 | 0.986 | 0.977 | 0.987 | 0.949 | 0.934 |
| bỏ cột độ phủ <50% (0 cột) | 47 | 212 | 0.965 | 0.987 | 0.983 | 0.991 | 0.949 | 0.934 |
| chỉ mẫu có ≥5 quý lịch sử | 47 | 180 | 0.965 | 0.987 | 0.987 | 0.993 | 0.949 | 0.934 |

## 5. Phân tích lỗi (mẫu dự đoán sai trên test)

| Mẫu | Công ty | Thực tế | Dự đoán | P(distress) | n_history | current_ratio | WC/assets | net_margin | debt/assets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HD-2024Q2 | HD | 1 | 0 | 0.783 | 37 | 1.34 | 0.10 | 0.099 | 0.98 |
| DG-2025Q2 | DG | 0 | 1 | 0.798 | 29 | 1.23 | 0.05 | 0.038 | 0.75 |
| FIVE-2024Q3 | FIVE | 1 | 0 | 0.174 | 26 | 1.63 | 0.11 | 0.040 | 0.60 |

## 6. Ngưỡng quyết định (theo F1 và theo chi phí)

| Tập | Ngưỡng best-F1 | F1 | Precision | Recall | Ngưỡng tối ưu chi phí | Chi phí kỳ vọng |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.788 | 0.976 | 1.000 | 0.952 | 0.788 | 5.0 |
| test | 0.783 | 0.974 | 0.974 | 0.974 | 0.783 | 6.0 |

Chi phí giả định: FN = 5, FP = 1.

## 7. Hiệu chuẩn xác suất

| Tập | Brier | ECE |
|---|---:|---:|
| validation | 0.046 | 0.098 |
| test | 0.058 | 0.086 |

## 8. Learning curve (chia theo công ty)

| #mẫu train | Train AUROC | Val AUROC |
|---:|---:|---:|
| 56 | — | — |
| 83 | 1.000 | 0.731 |
| 109 | 1.000 | 0.818 |
| 135 | 1.000 | 0.855 |
| 162 | 1.000 | 0.895 |

Gap ở dữ liệu đầy đủ: **0.105** AUROC.

## 9. In-domain vs cross-company vs baseline

| Hệ thống | AUROC | AP | Nguồn |
|---|---:|---:|---|
| model[logistic] — in-domain | 0.983 | 0.989 | train→test |
| model[random_forest] — in-domain | 0.983 | 0.991 | train→test |
| model[hist_gradient_boosting] — in-domain | 0.977 | 0.987 | train→test |
| model[mlp] — in-domain | 0.971 | 0.982 | train→test |
| model[logistic] — cross-company | 0.912 | 0.942 | GroupKFold |
| model[random_forest] — cross-company | 0.933 | 0.957 | GroupKFold |
| model[hist_gradient_boosting] — cross-company | 0.926 | 0.953 | GroupKFold |
| model[mlp] — cross-company | 0.789 | 0.832 | GroupKFold |
| dummy_most_frequent | 0.500 | 0.594 | baseline |
| ticker_prior | 0.986 | 0.985 | baseline |
| single_feature[debt_to_assets_latest] | 0.889 | 0.938 | baseline |
| rule[altman_z_double_prime<1.1] | 0.758 | 0.865 | baseline |
| model[logistic] | 0.983 | 0.989 | baseline |
| model[random_forest] | 0.983 | 0.991 | baseline |
| model[hist_gradient_boosting] | 0.977 | 0.987 | baseline |
| model[mlp] | 0.971 | 0.982 | baseline |

LOCO trung bình: **0.628** AUROC trên 8 phép so; công ty không tính được (nhãn đơn lớp): —

## 10. Audit nhãn (nhãn gốc có tái tạo được?)

| Quy tắc thử nghiệm (trên quý target) | Khớp với nhãn gốc | Khớp khi áp trên dòng lịch sử cuối |
|---|---:|---:|
| net_income<0 | 39.6% | 39.6% |
| operating_income<0 | 39.3% | 39.3% |
| operating_cash_flow<0 | 40.3% | 40.3% |
| ocf<0 hoặc ni<0 | 40.9% | 40.9% |
| current_liabilities>current_assets | 65.3% | 65.3% |
| stockholders_equity<0 | 51.9% | 51.9% |
| retained_earnings<0 | 54.2% | 54.2% |
| stress_signals>=1 | 74.7% | — |
| stress_signals>=2 | 47.7% | — |

Khớp cao nhất: **74.7%** trên 308 mẫu ⇒ nhãn gốc không tái tạo được từ dữ liệu công bố.
