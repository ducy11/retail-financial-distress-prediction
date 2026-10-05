# Nhãn tái lập được + kiểm chứng độ nhạy của kết luận

## 1. Công thức nhãn thay thế

`is_distressed_rule = 1` nếu quý target có ≥ **1** tín hiệu căng thẳng tài chính:

- net_income<0: Lỗ ròng trong quý target
- operating_cash_flow<0: Dòng tiền hoạt động âm
- operating_income<0: Lỗ hoạt động
- current_liabilities>current_assets: Vốn lưu động âm (thanh khoản ngắn hạn)
- stockholders_equity<0: Vốn chủ sở hữu âm
- revenue_yoy<-5%: Doanh thu giảm hơn 5% so với cùng kỳ

- Mức khớp với nhãn GỐC: **74.4%** trên 324 mẫu → hai định nghĩa khác nhau rõ rệt, vì thế phải kiểm chứng độ nhạy của kết luận.

- Số mẫu dương tính theo nhãn quy tắc: `{'train': 88, 'validation': 14, 'test': 27, 'purged': 8}`

## 2. In-domain (chia theo thời gian, cùng công ty)

| Hệ thống | Val AUROC | Val AP | Val F1 | Test AUROC | Test AP | Test F1 | Test macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| model[logistic] | 0.933 | 0.949 | 0.880 | 0.931 | 0.949 | 0.773 | 0.827 |
| model[random_forest] | 0.905 | 0.927 | 0.846 | 0.909 | 0.941 | 0.920 | 0.934 |
| model[hist_gradient_boosting] | 0.857 | 0.868 | 0.667 | 0.922 | 0.932 | 0.851 | 0.882 |
| model[mlp] | 0.893 | 0.884 | 0.812 | 0.887 | 0.886 | 0.706 | 0.686 |
| baseline[ticker_prior] | 0.944 | 0.945 | 0.727 | 0.960 | 0.958 | 0.744 | 0.807 |

## 3. Cross-company (GroupKFold — công ty chưa từng thấy)

| Hệ thống | AUROC | AP | F1 | macro-F1 |
|---|---:|---:|---:|---:|
| model[logistic] — cross-company | 0.369 | 0.399 | 0.247 | 0.365 |
| model[random_forest] — cross-company | 0.731 | 0.649 | 0.393 | 0.573 |
| model[hist_gradient_boosting] — cross-company | 0.638 | 0.608 | 0.391 | 0.575 |
| model[mlp] — cross-company | 0.403 | 0.395 | 0.270 | 0.407 |
