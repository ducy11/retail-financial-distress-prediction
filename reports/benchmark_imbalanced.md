# Benchmark Non-E Mode vs E-Mode — phân loại mất cân bằng 95/5

- Dataset: `make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)`; Stratified 80%/20% (seed 42)
- Backend boosting: **XGBoost**; ngưỡng quyết định 0.5

## Bảng so sánh tổng hợp

| group      | method                                        |   roc_auc |   pr_auc |   f1_minority |   balanced_accuracy |   precision |   recall |   train_time_s |
|:-----------|:----------------------------------------------|----------:|---------:|--------------:|--------------------:|------------:|---------:|---------------:|
| Baseline   | Logistic Regression (không can thiệp)         |    0.7443 |   0.2833 |        0.1228 |              0.5328 |      0.8750 |   0.0660 |         0.0294 |
| Baseline   | XGBoost (mặc định)                            |    0.9210 |   0.6740 |        0.5935 |              0.7162 |      0.9388 |   0.4340 |         0.7505 |
| Non-E Mode | SMOTE + Logistic Regression                   |    0.7717 |   0.1791 |        0.2201 |              0.6468 |      0.1474 |   0.4340 |         0.0301 |
| Non-E Mode | SMOTE + XGBoost                               |    0.8963 |   0.6235 |        0.5536 |              0.7777 |      0.5254 |   0.5849 |         0.7824 |
| Non-E Mode | RandomUnderSampler + Logistic Regression      |    0.7742 |   0.1701 |        0.2251 |              0.6413 |      0.1558 |   0.4057 |         0.0087 |
| Non-E Mode | SMOTE + Tomek Links + Logistic Regression     |    0.7736 |   0.1483 |        0.2069 |              0.7174 |      0.1204 |   0.7358 |         0.2676 |
| Non-E Mode | Logistic Regression (class_weight='balanced') |    0.7742 |   0.1480 |        0.2016 |              0.7075 |      0.1173 |   0.7170 |         0.0140 |
| Non-E Mode | XGBoost (scale_pos_weight động)               |    0.9164 |   0.6579 |        0.5755 |              0.7759 |      0.5755 |   0.5755 |         0.5998 |
| E-Mode     | BalancedRandomForestClassifier                |    0.9206 |   0.5441 |        0.4503 |              0.8212 |      0.3263 |   0.7264 |         1.0477 |
| E-Mode     | EasyEnsembleClassifier                        |    0.8183 |   0.2288 |        0.2297 |              0.7479 |      0.1349 |   0.7736 |         1.8547 |
| E-Mode     | RUSBoostClassifier                            |    0.8189 |   0.2465 |        0.3113 |              0.6364 |      0.3113 |   0.3113 |         1.3500 |

## Trung bình theo nhóm

| Nhóm | #phương pháp | PR-AUC (tb) | F1 thiểu số (tb) | Balanced Acc (tb) | Thời gian (tb, s) |
|---|---:|---:|---:|---:|---:|
| Baseline | 2 | 0.4786 | 0.3582 | 0.6245 | 0.390 |
| Non-E Mode | 6 | 0.3211 | 0.3305 | 0.7111 | 0.284 |
| E-Mode | 3 | 0.3398 | 0.3304 | 0.7352 | 1.417 |

**Tốt nhất theo từng chỉ số**

- **roc_auc**: `XGBoost (mặc định)` (Baseline) = 0.9210
- **pr_auc**: `XGBoost (mặc định)` (Baseline) = 0.6740
- **f1_minority**: `XGBoost (mặc định)` (Baseline) = 0.5935
- **balanced_accuracy**: `BalancedRandomForestClassifier` (E-Mode) = 0.8212
- **precision**: `XGBoost (mặc định)` (Baseline) = 0.9388
- **recall**: `EasyEnsembleClassifier` (E-Mode) = 0.7736
- **train_time_s**: `RandomUnderSampler + Logistic Regression` (Non-E Mode) = 0.0087

## Kiểm chứng chống rò rỉ dữ liệu

- PASS — Logistic Regression (không can thiệp)
- PASS — XGBoost (mặc định)
- PASS — SMOTE + Logistic Regression
- PASS — SMOTE + XGBoost
- PASS — RandomUnderSampler + Logistic Regression
- PASS — SMOTE + Tomek Links + Logistic Regression
- PASS — Logistic Regression (class_weight='balanced')
- PASS — XGBoost (scale_pos_weight động)
- PASS — BalancedRandomForestClassifier
- PASS — EasyEnsembleClassifier
- PASS — RUSBoostClassifier

## StratifiedKFold 5 fold (trên train split)

| Nhóm | Phương pháp | #fold | ROC-AUC | PR-AUC | F1 thiểu số | Bal. Acc | Rò rỉ |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | Logistic Regression (không can thiệp) | 5 | 0.7562 ± 0.0226 | 0.2785 ± 0.0163 | 0.0844 ± 0.0430 | 0.5221 ± 0.0116 | 5/5 PASS |
| Baseline | XGBoost (mặc định) | 5 | 0.9266 ± 0.0148 | 0.6589 ± 0.0246 | 0.5007 ± 0.0177 | 0.6723 ± 0.0082 | 5/5 PASS |
| Non-E Mode | SMOTE + Logistic Regression | 5 | 0.7741 ± 0.0131 | 0.1861 ± 0.0199 | 0.2457 ± 0.0117 | 0.6726 ± 0.0169 | 5/5 PASS |
| Non-E Mode | SMOTE + XGBoost | 5 | 0.9138 ± 0.0095 | 0.5964 ± 0.0357 | 0.5201 ± 0.0323 | 0.7572 ± 0.0245 | 5/5 PASS |
| Non-E Mode | RandomUnderSampler + Logistic Regression | 5 | 0.7721 ± 0.0160 | 0.1815 ± 0.0225 | 0.2394 ± 0.0215 | 0.6622 ± 0.0255 | 5/5 PASS |
| Non-E Mode | SMOTE + Tomek Links + Logistic Regression | 5 | 0.7766 ± 0.0130 | 0.1609 ± 0.0165 | 0.2035 ± 0.0094 | 0.7045 ± 0.0162 | 5/5 PASS |
| Non-E Mode | Logistic Regression (class_weight='balanced') | 5 | 0.7769 ± 0.0139 | 0.1594 ± 0.0129 | 0.2000 ± 0.0085 | 0.7018 ± 0.0160 | 5/5 PASS |
| Non-E Mode | XGBoost (scale_pos_weight động) | 5 | 0.9219 ± 0.0109 | 0.6301 ± 0.0223 | 0.5860 ± 0.0173 | 0.7637 ± 0.0138 | 5/5 PASS |
| E-Mode | BalancedRandomForestClassifier | 5 | 0.9170 ± 0.0069 | 0.5221 ± 0.0166 | 0.4354 ± 0.0133 | 0.8256 ± 0.0070 | 5/5 PASS |
| E-Mode | EasyEnsembleClassifier | 5 | 0.8115 ± 0.0184 | 0.2354 ± 0.0251 | 0.2268 ± 0.0051 | 0.7376 ± 0.0102 | 5/5 PASS |
| E-Mode | RUSBoostClassifier | 5 | 0.7561 ± 0.0335 | 0.1771 ± 0.0396 | 0.2080 ± 0.0332 | 0.6208 ± 0.0353 | 5/5 PASS |

*Tỉ lệ dương các fold validation: 5.25% – 5.31% (dataset gốc 5.00%).*

Chi tiết từng fold + 3 kiểm chứng mỗi fold: `reports/benchmark_imbalanced_cv.md`.

## Ghi chú từng phương pháp

- Logistic Regression (không can thiệp): mốc so sánh tuyến tính
- XGBoost (mặc định): mốc so sánh boosting, không chỉnh gì cho mất cân bằng
- SMOTE + Logistic Regression: oversample thiểu số lên 50% đa số
- SMOTE + XGBoost: oversample + boosting
- RandomUnderSampler + Logistic Regression: undersample đa số còn 2× thiểu số
- SMOTE + Tomek Links + Logistic Regression: oversample rồi dọn cặp mẫu nhiễu (combine)
- Logistic Regression (class_weight='balanced'): đổi trọng số trong hàm mất mát
- XGBoost (scale_pos_weight động): trọng số = n_âm/n_dương tính trong fit
- BalancedRandomForestClassifier: bagging + cân bằng mỗi cây
- EasyEnsembleClassifier: nhiều balanced AdaBoost trên tập đa số khác nhau
- RUSBoostClassifier: boosting + random undersampling mỗi vòng (imblearn; nếu lỗi do sklearn 1.6 bỏ SAMME.R thì tự dùng bản nội bộ tương đương) [backend: RUSBoostInternal (dự phòng — sklearn 1.6 bỏ SAMME.R)]

## Tái lập

```powershell
python benchmark_imbalanced.py
```
