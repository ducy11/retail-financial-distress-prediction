# Thực nghiệm phản biện: LiteSVM & XGBoost (ngoài pipeline chính)

- Ngưỡng vận hành của đồ án (đọc từ `summary.json`): **0.7879**
- Chống rò rỉ: impute/scale/hiệu chuẩn/trọng số lớp chỉ học từ train; riêng biến thể
  early stopping chọn số vòng trên **validation** (ghi rõ, không dùng test).

## 1. Corpus thật — LiteSVM đặt cạnh 4 họ mô hình của đồ án

| Mô hình | Nhóm | Test AUROC | Test AP | F1@0,5 | F1@ngưỡng vận hành | Bal.Acc@0,5 | MCC@ngưỡng | TN/FP/FN/TP | Giây | ms/1000 dòng | Gap train−val |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| LiteSVM · LinearSVC (lề cực đại) | litesvm | 0.9211 | 0.9459 | 0.8750 | 0.6182 | 0.8259 | 0.4975 | 26/0/21/17 | 0.020 | 2.13 | 0.0322 |
| LiteSVM · LinearSVC + class_weight=balanced | litesvm | 0.9200 | 0.9454 | 0.8608 | 0.6429 | 0.8128 | 0.5174 | 26/0/20/18 | 0.020 | 2.13 | 0.0318 |
| LiteSVM · SGD hinge (giảm gradient) | litesvm | 0.9352 | 0.9573 | 0.8421 | 0.7333 | 0.8057 | 0.5987 | 26/0/16/22 | 0.014 | 2.26 | 0.0129 |
| LiteSVM · SGD hinge + class_weight=balanced | litesvm | 0.9646 | 0.9761 | 0.9136 | 0.6667 | 0.8715 | 0.5375 | 26/0/19/19 | 0.014 | 2.19 | 0.0267 |
| Logistic Regression | project | 0.9828 | 0.9887 | 0.9351 | 0.9143 | 0.9160 | 0.8272 | 26/0/6/32 | 0.005 | 0.95 | 0.0187 |
| Random Forest | project | 0.9828 | 0.9908 | 0.9487 | 0.9600 | 0.9291 | 0.9039 | 25/1/2/36 | 0.465 | 21.98 | 0.0342 |
| HistGradientBoosting | project | 0.9767 | 0.9868 | 0.9367 | 0.9067 | 0.9099 | 0.7750 | 23/3/4/34 | 0.170 | 10.61 | 0.0260 |
| MLP (mạng nơ-ron) | project | 0.9706 | 0.9819 | 0.9136 | 0.8947 | 0.8715 | 0.7409 | 22/4/4/34 | 0.074 | 1.19 | -0.0134 |

## 2. Bộ giả lập 95/5 — LiteSVM trong bảng benchmark mất cân bằng

| Phương pháp | Nhóm | ROC-AUC | PR-AUC | F1 thiểu số | Bal. Acc | Giây |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression (không can thiệp) | baseline | 0.7443 | 0.2823 | 0.1228 | 0.5328 | 0.020 |
| LiteSVM · LinearSVC | litesvm | 0.7366 | 0.2953 | 0.1228 | 0.5328 | 0.056 |
| LiteSVM · SGD hinge | litesvm | 0.7319 | 0.2385 | 0.0000 | 0.5000 | 0.086 |
| LiteSVM · LinearSVC + balanced | litesvm | 0.7744 | 0.1505 | 0.0185 | 0.5045 | 0.057 |

## 3. XGBoost (tham chiếu boosting) — báo cáo TỪNG LỚP tại ngưỡng vận hành

| Biến thể | Test AUROC | Test AP | Class 0 (P/R/F1) | Class 1 (P/R/F1) | TN/FP/FN/TP | Gap train−val |
|---|---:|---:|---|---|---|---:|
| XGBoost (mặc định, không can thiệp) | 0.9777 | 0.9872 | 0.8519/0.8846/0.8679 | 0.9189/0.8947/0.9067 | 23/3/4/34 | 0.0303 |
| XGBoost + scale_pos_weight = 0.606 | 0.9787 | 0.9876 | 0.8621/0.9615/0.9091 | 0.9714/0.8947/0.9315 | 25/1/4/34 | 0.0346 |
| XGBoost + scale_pos_weight + early stopping (eval = validation) | 0.9818 | 0.9897 | 0.8667/1.0000/0.9286 | 1.0000/0.8947/0.9444 | 26/0/4/34 | 0.0253 |

## 4. Tái lập

```powershell
python -m scripts.experiment_defense
```
