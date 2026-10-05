# EDA — dữ liệu bán lẻ Mỹ (SEC XBRL)

- Corpus: **8 công ty, 332 quý, 324 mẫu**
- Công ty có nhãn 1 toàn bộ: **HD, LOW, WMT**
- Công ty có nhãn 0 toàn bộ: **—**
- Mẫu có < 5 quý lịch sử: **32/324 (10%)**
- Chỉ tiêu độ phủ < 50% (theo công ty): **liabilities, receivables, short_term_investments, net_income, operating_income**

## Số quý mỗi công ty

| Công ty | Số quý |
|---|---:|
| DG | 32 |
| DKS | 44 |
| FIVE | 32 |
| HD | 44 |
| LOW | 44 |
| ORLY | 44 |
| ROST | 44 |
| WMT | 48 |

## Cân bằng lớp theo tập (đếm và %)

| Tập | Suy giảm (1) | Không (0) | Tổng | % lớp 1 | % lớp 0 |
|---|---:|---:|---:|---:|---:|
| train | 132 | 80 | 212 | 62.3% | 37.7% |
| validation | 21 | 11 | 32 | 65.6% | 34.4% |
| test | 38 | 26 | 64 | 59.4% | 40.6% |
| purged | 11 | 5 | 16 | 68.8% | 31.2% |

> Đọc bảng: lớp 1 chiếm đa số ở mọi tập ⇒ **không dùng Accuracy**; theo dõi F1/macro-F1, PR-AUC (AP) và MCC (chi tiết ở mục đánh giá của báo cáo).

## Tỷ lệ nhãn theo công ty (phát hiện quan trọng)

| Công ty | Tỷ lệ nhãn = 1 |
|---|---:|
| HD | 1.00 |
| LOW | 1.00 |
| WMT | 1.00 |
| ORLY | 0.91 |
| DG | 0.55 |
| FIVE | 0.19 |
| DKS | 0.12 |
| ROST | 0.05 |

## Độ phủ chỉ tiêu (min–mean theo công ty)

| Chỉ tiêu | Min % | Mean % |
|---|---:|---:|
| liabilities | 0.0 | 39.1 |
| receivables | 0.0 | 57.1 |
| short_term_investments | 0.0 | 29.5 |
| net_income | 9.1 | 88.6 |
| operating_income | 9.1 | 86.4 |
| operating_cash_flow | 59.1 | 89.7 |
| cash_and_equivalents | 81.2 | 97.7 |
| cost_of_sales | 100.0 | 100.0 |
| current_assets | 100.0 | 100.0 |
| current_liabilities | 100.0 | 100.0 |
| inventory | 100.0 | 100.0 |
| retained_earnings | 100.0 | 100.0 |
| revenue | 100.0 | 100.0 |
| selling_general_admin | 100.0 | 100.0 |
| stockholders_equity | 100.0 | 100.0 |
| total_assets | 100.0 | 100.0 |

## Thống kê mô tả 14 tỷ số (giá trị quý gần nhất; 324 mẫu; không đơn vị)

| Tỷ số | Thiếu % | Nhỏ nhất | Q1 | Trung vị | Trung bình | Q3 | Lớn nhất | Skew | Ngoại lai IQR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| debt_to_equity | 0.0 | -1699.564856 | 1.329627 | 2.143514 | -1.046676 | 3.452771 | 318.936709 | -14.907298 | 30.6 |
| gross_margin | 0.0 | -0.025679 | 0.28273 | 0.327324 | 0.334371 | 0.344779 | 0.533476 | 0.849176 | 14.2 |
| cash_to_assets | 0.0 | 0.003611 | 0.024864 | 0.042707 | 0.084305 | 0.112068 | 0.408248 | 1.6397 | 8.3 |
| retained_to_assets | 0.0 | -0.374136 | 0.102902 | 0.29772 | 0.295538 | 0.423514 | 1.097344 | 0.387015 | 6.5 |
| quick_ratio | 0.0 | 0.08205 | 0.18819 | 0.27628 | 0.421264 | 0.591044 | 1.744425 | 1.418304 | 4.6 |
| ocf_to_sales | 10.5 | -0.574406 | 0.055027 | 0.096288 | 0.104871 | 0.156372 | 0.458471 | -1.09136 | 3.8 |
| sgna_pct_revenue | 0.0 | 0.113943 | 0.186516 | 0.214906 | 0.223028 | 0.246399 | 0.461212 | 0.809222 | 3.1 |
| current_ratio | 0.0 | 0.686714 | 0.967935 | 1.191527 | 1.258503 | 1.551621 | 2.979182 | 0.87462 | 1.2 |
| operating_margin | 14.8 | -0.359365 | 0.051627 | 0.098422 | 0.105172 | 0.149578 | 0.238223 | -1.172796 | 0.7 |
| net_margin | 12.3 | -0.251778 | 0.038709 | 0.079135 | 0.075435 | 0.10428 | 0.171972 | -1.395264 | 0.7 |
| inventory_to_sales | 0.0 | 0.297705 | 0.477932 | 0.644766 | 0.717621 | 0.884965 | 1.829357 | 0.797479 | 0.6 |
| debt_to_assets | 0.0 | 0.34231 | 0.633366 | 0.723893 | 0.777362 | 0.938767 | 1.360091 | 0.696818 | 0.0 |
| receivables_to_sales | 36.1 | 0.015236 | 0.035373 | 0.054096 | 0.061722 | 0.087255 | 0.163595 | 0.54626 | 0.0 |
| revenue_per_asset | 0.0 | 0.103946 | 0.316466 | 0.417913 | 0.427054 | 0.538962 | 0.7109 | 0.099234 | 0.0 |

> Đọc bảng: cột **Ngoại lai IQR %** = % mẫu nằm ngoài [Q1 − 1,5·IQR, Q3 + 1,5·IQR]; cột **Skew** = độ lệch (|skew| > 1 là lệch rõ). Tỷ số vừa lệch vừa nhiều ngoại lai (thường là các tỷ số có mẫu số nhỏ như `debt_to_equity`) nên được **clip/winsorize** trước khi huấn luyện mô hình tuyến tính.

## Thống kê mô tả 16 chỉ tiêu gốc (nghìn tỷ VND; mọi công ty × mọi quý)

| Chỉ tiêu | Thiếu % | Nhỏ nhất | Trung vị | Trung bình | Lớn nhất | Ngoại lai IQR % |
|---|---:|---:|---:|---:|---:|---:|
| short_term_investments | 72.3 | 0.0 | 4.1 | 5.490086 | 46.3 | 3.3 |
| liabilities | 62.7 | 6.42805 | 968.9125 | 970.719337 | 2353.95 | 0.0 |
| receivables | 38.3 | 0.94925 | 9.376225 | 60.127987 | 302.875 | 0.0 |
| operating_income | 14.5 | -14.2 | 20.228862 | 56.080332 | 217.7 | 0.0 |
| net_income | 12.0 | -52.275 | 14.527475 | 35.890089 | 197.275 | 1.7 |
| operating_cash_flow | 10.2 | -93.95 | 18.968275 | 57.590947 | 417.8 | 6.7 |
| cash_and_equivalents | 1.8 | 0.6632 | 34.699188 | 75.780243 | 571.15 | 6.1 |
| revenue | 0.0 | 5.022475 | 163.331175 | 734.852054 | 4722.825 | 14.5 |
| cost_of_sales | 0.0 | 4.51095 | 113.756538 | 534.917416 | 3590.375 | 14.5 |
| inventory | 0.0 | 5.3844 | 121.412338 | 335.26724 | 1633.85 | 14.5 |
| selling_general_admin | 0.0 | 1.8133 | 36.518025 | 150.730636 | 958.325 | 14.5 |
| total_assets | 0.0 | 18.376 | 424.3719 | 1356.684549 | 7216.375 | 14.5 |
| current_assets | 0.0 | 13.228125 | 182.814362 | 469.966159 | 2323.0 | 14.5 |
| current_liabilities | 0.0 | 4.516575 | 151.682513 | 486.412871 | 2893.3 | 14.5 |
| stockholders_equity | 0.0 | -378.675 | 66.054513 | 327.756266 | 2490.425 | 19.0 |
| retained_earnings | 0.0 | -393.6 | 72.06635 | 524.230732 | 2619.35 | 3.6 |

> Đọc bảng: chênh lệch **quy mô** giữa các công ty rất lớn (WMT lớn hơn FIVE nhiều lần) nên không so sánh giá trị tuyệt đối giữa công ty; mô hình dùng **tỷ số** (bảng trên). Chỉ tiêu thiếu nhiều là hạn chế của dữ liệu gốc SEC, đã nêu ở phần độ phủ.

## Tương quan giữa các tỷ số (Pearson / Spearman)

- **6 cặp** có |r| ≥ 0.8 ⇒ gần như cùng một thông tin, gom thành **4 cụm**: cash_to_assets, current_ratio, quick_ratio; gross_margin, net_margin, operating_margin; debt_to_assets, receivables_to_sales; inventory_to_sales, sgna_pct_revenue.
- Số chiều hiệu dụng của 14 tỷ số: **5.083158** (cần 9 thành phần cho 95% phương sai).

| Cặp tỷ số tương quan mạnh nhất | Pearson | Spearman |
|---|---:|---:|
| operating_margin ↔ net_margin | 0.9783 | 0.9727 |
| gross_margin ↔ operating_margin | 0.8412 | 0.8975 |
| quick_ratio ↔ cash_to_assets | 0.8356 | 0.8665 |
| sgna_pct_revenue ↔ inventory_to_sales | 0.8329 | 0.7478 |
| current_ratio ↔ quick_ratio | 0.8214 | 0.747 |
| debt_to_assets ↔ receivables_to_sales | 0.8029 | 0.736 |
| gross_margin ↔ net_margin | 0.7637 | 0.7003 |
| operating_margin ↔ ocf_to_sales | 0.7301 | 0.7 |

**Cặp bị ngoại lai chi phối** (Pearson lệch Spearman > 0.30): debt_to_assets ↔ debt_to_equity (r -0.0752 vs hạng 0.4482), debt_to_equity ↔ receivables_to_sales (r -0.0382 vs hạng 0.385), gross_margin ↔ debt_to_equity (r -0.1226 vs hạng 0.2693), operating_margin ↔ debt_to_equity (r -0.1103 vs hạng 0.2055), debt_to_equity ↔ revenue_per_asset (r 0.0617 vs hạng -0.2492)

> Đọc bảng: cột **Spearman** là tương quan theo hạng — nếu hai cột chênh nhau nhiều (> 0,3) thì tương quan chủ yếu do **ngoại lai** tạo ra, cần xử lý ngoại lai trước khi kết luận. Với các cặp |r| cao, nên **giữ 1 đại diện mỗi cụm** hoặc dùng regularization (`C`, `l2_regularization`) để tránh hệ số bất ổn.

## Tương quan giữa tỷ số và NHÃN (train+validation)

| Tỷ số | AUC 1-tỷ số | Hướng | r (point-biserial) | q-value BH | Mutual information |
|---|---:|---|---:|---:|---:|
| current_ratio | 0.0738 | giá trị thấp ⇒ nhãn 1 | -0.6879 | 0.0 | 0.3671 |
| debt_to_assets | 0.8787 | giá trị cao ⇒ nhãn 1 | 0.623 | 0.0 | 0.3399 |
| receivables_to_sales | 0.8665 | giá trị cao ⇒ nhãn 1 | 0.5704 | 0.0 | 0.169 |
| quick_ratio | 0.2185 | giá trị thấp ⇒ nhãn 1 | -0.5078 | 0.0 | 0.1722 |
| debt_to_equity | 0.7153 | giá trị cao ⇒ nhãn 1 | -0.0353 | 0.680896 | 0.3397 |
| cash_to_assets | 0.3372 | giá trị thấp ⇒ nhãn 1 | -0.398 | 0.0 | 0.1565 |
| gross_margin | 0.6277 | giá trị cao ⇒ nhãn 1 | 0.2123 | 0.001978 | 0.2196 |
| operating_margin | 0.6057 | giá trị cao ⇒ nhãn 1 | 0.2124 | 0.0047 | 0.1056 |
| revenue_per_asset | 0.5907 | giá trị cao ⇒ nhãn 1 | 0.1732 | 0.011726 | 0.1014 |
| retained_to_assets | 0.4448 | giá trị thấp ⇒ nhãn 1 | -0.0042 | 0.97091 | 0.303 |
| net_margin | 0.4481 | giá trị thấp ⇒ nhãn 1 | 0.0025 | 0.97091 | 0.1229 |
| sgna_pct_revenue | 0.5431 | giá trị cao ⇒ nhãn 1 | 0.1263 | 0.075952 | 0.1666 |
| inventory_to_sales | 0.4578 | giá trị thấp ⇒ nhãn 1 | -0.0622 | 0.424541 | 0.0268 |
| ocf_to_sales | 0.4796 | giá trị thấp ⇒ nhãn 1 | 0.0715 | 0.423346 | 0.0803 |

> Đọc bảng: **AUC 1-tỷ số** là khả năng tự tách hai lớp của tỷ số đó; `|2·AUC−1|` càng lớn càng tách tốt, hướng cho biết giá trị cao hay thấp ứng với suy giảm. Cột **q-value BH** đã hiệu chỉnh cho 14 phép kiểm định (q < 0,05 mới coi là chắc chắn). 6 tỷ số có |2·AUC−1| ≥ 0,30; 3 tỷ số còn ý nghĩa sau hiệu chỉnh.

## Nhận xét (sinh tự động từ số liệu trên)

1. **Phân phối & ngoại lai:** 6/14 tỷ số lệch rõ (|skew| > 1), lệch nhất: debt_to_equity (skew -14.91, max 318.94), cash_to_assets (skew 1.64, max 0.41), quick_ratio (skew 1.42, max 1.74); 4/14 tỷ số có hơn 5% giá trị ngoại lai theo IQR. ⇒ Trước khi huấn luyện nên **clip/winsorize** các tỷ số này (mô hình tuyến tính bị outlier kéo mạnh; mô hình cây ít bị hơn nhưng vẫn nên xem lại giá trị cực trị).
2. **Giá trị thiếu (theo từng tỷ số):** 1 tỷ số thiếu hơn 20% — receivables_to_sales (36%). ⇒ Phần thiếu được `SimpleImputer(median)` xử lý **trong pipeline**; riêng các tỷ số dựa trên `receivables`/`short_term_investments` chỉ có nghĩa với một phần mẫu.
3. **Tỉ lệ lớp (bằng %):** lớp suy giảm chiếm 62.3% toàn corpus; theo tập: train 62.3%, validation 65.6%, test 59.4%, purged 68.8%. ⇒ Chỉ đoán lớp đa số đã đạt 62.3% ⇒ **KHÔNG dùng Accuracy** làm thước đo chính; dùng Precision/Recall/**F1 & macro-F1**/PR-AUC (AP)/MCC và nêu ngưỡng quyết định.
4. **Lệch lớp ở cấp công ty (quan trọng hơn cấp mẫu):** HD, LOW, WMT có **100% nhãn = 1** trong mọi quý; lệch ngược lại: DG chỉ 54.8% nhãn 1, FIVE chỉ 19.4% nhãn 1. ⇒ Nhãn gần như là thuộc tính của công ty, nên phải chia tập/CV **theo nhóm công ty** (GroupKFold/LOCO) và luôn so với baseline `ticker_prior`, nếu không AUROC in-domain sẽ bị thổi phồng.
5. **Tương quan giữa các tỷ số:** 6 cặp có |r| ≥ 0.8 (gần như trùng thông tin), gom thành 4 cụm: cash_to_assets, current_ratio, quick_ratio; gross_margin, net_margin, operating_margin; debt_to_assets, receivables_to_sales. ⇒ Nên **giữ 1 đại diện mỗi cụm** hoặc dùng mô hình có regularization để tránh hệ số bất ổn.
6. **Cảnh báo tương quan giả do ngoại lai:** 5 cặp có tương quan Pearson lệch xa Spearman hơn 0.30 — debt_to_assets↔debt_to_equity (r -0.0752 vs hạng 0.4482), debt_to_equity↔receivables_to_sales (r -0.0382 vs hạng 0.385), gross_margin↔debt_to_equity (r -0.1226 vs hạng 0.2693). ⇒ Các cặp này chỉ 'liên quan' ở vài quan sát dị biệt; phải xử lý ngoại lai trước khi kết luận về quan hệ giữa chúng.
7. **Tỷ số tách hai lớp tốt nhất (train+validation, xếp theo |2·AUC−1|):** current_ratio (2·AUC−1 = -0.85 ⇒ giá trị thấp ⇒ nhãn 1), debt_to_assets (2·AUC−1 = +0.76 ⇒ giá trị cao ⇒ nhãn 1), receivables_to_sales (2·AUC−1 = +0.73 ⇒ giá trị cao ⇒ nhãn 1), quick_ratio (2·AUC−1 = -0.56 ⇒ giá trị thấp ⇒ nhãn 1), debt_to_equity (2·AUC−1 = +0.43 ⇒ giá trị cao ⇒ nhãn 1). ⇒ Đây là các biến nên giữ lại khi lọc đặc trưng; lưu ý nhóm thanh khoản/cấu trúc vốn gần như không đổi theo quý nên phần lớn sức mạnh dự báo in-domain đến từ việc phân biệt **công ty**, không phải từ động lực suy giảm của từng quý.
8. **Ngoại lai ở dữ liệu gốc:** 10/16 chỉ tiêu tiền có hơn 5% quý nằm ngoài khoảng IQR — đây thường là khác biệt quy mô giữa các chuỗi bán lẻ (WMT lớn hơn FIVE nhiều lần), nên khi so sánh giữa công ty phải dùng **tỷ số** (đã có) thay vì giá trị tuyệt đối.

## Hình

- `figures\eda\01_coverage_by_company.png`
- `figures\eda\02_class_balance.png`
- `figures\eda\03_label_by_company_quarter.png`
- `figures\eda\04_ratio_boxplots_by_label.png`
- `figures\eda\05_timeseries_indicators.png`
- `figures\eda\06_history_length.png`
- `figures\eda\07_ratio_distributions.png`
- `figures\eda\08_ratio_correlation.png`
- `figures\eda\09_ratio_vs_label.png`
