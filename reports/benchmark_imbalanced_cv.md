# Đánh giá StratifiedKFold 5 fold — Non-E Mode vs E-Mode

- Chia fold: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` trên **train split**; test vẫn khoá riêng làm holdout chốt.

- Resampling chỉ chạy trên **fold-train** (trong `imblearn.pipeline.Pipeline`); fold validation không bao giờ được resample.

## 1. Trung bình ± độ lệch chuẩn qua các fold

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

## 2. Chi tiết từng fold

| Phương pháp | fold | n_train | → sau resample | IR trước→sau | n_val | % dương (val) | ROC-AUC | PR-AUC | F1 | Bal. Acc | fold-val nguyên vẹn | classifier chỉ thấy fold-train | tỉ lệ lớp giữ nguyên |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| Logistic Regression (không can thiệp) | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.7923 | 0.2672 | 0.0899 | 0.5235 | PASS | PASS | PASS |
| Logistic Regression (không can thiệp) | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7611 | 0.2877 | 0.1099 | 0.5291 | PASS | PASS | PASS |
| Logistic Regression (không can thiệp) | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7578 | 0.3008 | 0.0000 | 0.4993 | PASS | PASS | PASS |
| Logistic Regression (không can thiệp) | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7224 | 0.2542 | 0.1111 | 0.5294 | PASS | PASS | PASS |
| Logistic Regression (không can thiệp) | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7475 | 0.2829 | 0.1111 | 0.5294 | PASS | PASS | PASS |
| XGBoost (mặc định) | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.9260 | 0.6656 | 0.5128 | 0.6776 | PASS | PASS | PASS |
| XGBoost (mặc định) | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9410 | 0.6646 | 0.4957 | 0.6696 | PASS | PASS | PASS |
| XGBoost (mặc định) | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.8988 | 0.6147 | 0.5042 | 0.6752 | PASS | PASS | PASS |
| XGBoost (mặc định) | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9356 | 0.6905 | 0.5210 | 0.6814 | PASS | PASS | PASS |
| XGBoost (mặc định) | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9319 | 0.6590 | 0.4696 | 0.6578 | PASS | PASS | PASS |
| SMOTE + Logistic Regression | 1 | 6400 | 9090 | 17.8→2.0 | 1600 | 5.25% | 0.7949 | 0.1651 | 0.2507 | 0.6884 | PASS | PASS | PASS |
| SMOTE + Logistic Regression | 2 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.7729 | 0.1937 | 0.2462 | 0.6729 | PASS | PASS | PASS |
| SMOTE + Logistic Regression | 3 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.7643 | 0.1788 | 0.2609 | 0.6937 | PASS | PASS | PASS |
| SMOTE + Logistic Regression | 4 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.7574 | 0.1719 | 0.2249 | 0.6493 | PASS | PASS | PASS |
| SMOTE + Logistic Regression | 5 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.7810 | 0.2212 | 0.2458 | 0.6586 | PASS | PASS | PASS |
| SMOTE + XGBoost | 1 | 6400 | 9090 | 17.8→2.0 | 1600 | 5.25% | 0.9204 | 0.6354 | 0.5699 | 0.7993 | PASS | PASS | PASS |
| SMOTE + XGBoost | 2 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.9262 | 0.5943 | 0.5143 | 0.7499 | PASS | PASS | PASS |
| SMOTE + XGBoost | 3 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.8995 | 0.5704 | 0.4828 | 0.7315 | PASS | PASS | PASS |
| SMOTE + XGBoost | 4 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.9161 | 0.6361 | 0.5424 | 0.7678 | PASS | PASS | PASS |
| SMOTE + XGBoost | 5 | 6400 | 9091 | 17.9→2.0 | 1600 | 5.31% | 0.9069 | 0.5456 | 0.4914 | 0.7374 | PASS | PASS | PASS |
| RandomUnderSampler + Logistic Regression | 1 | 6400 | 1020 | 17.8→2.0 | 1600 | 5.25% | 0.7885 | 0.1414 | 0.2273 | 0.6629 | PASS | PASS | PASS |
| RandomUnderSampler + Logistic Regression | 2 | 6400 | 1017 | 17.9→2.0 | 1600 | 5.31% | 0.7820 | 0.1977 | 0.2529 | 0.6830 | PASS | PASS | PASS |
| RandomUnderSampler + Logistic Regression | 3 | 6400 | 1017 | 17.9→2.0 | 1600 | 5.31% | 0.7609 | 0.1854 | 0.2724 | 0.6948 | PASS | PASS | PASS |
| RandomUnderSampler + Logistic Regression | 4 | 6400 | 1017 | 17.9→2.0 | 1600 | 5.31% | 0.7462 | 0.1768 | 0.2102 | 0.6233 | PASS | PASS | PASS |
| RandomUnderSampler + Logistic Regression | 5 | 6400 | 1017 | 17.9→2.0 | 1600 | 5.31% | 0.7831 | 0.2064 | 0.2341 | 0.6468 | PASS | PASS | PASS |
| SMOTE + Tomek Links + Logistic Regression | 1 | 6400 | 12120 | 17.8→1.0 | 1600 | 5.25% | 0.7906 | 0.1463 | 0.2159 | 0.7246 | PASS | PASS | PASS |
| SMOTE + Tomek Links + Logistic Regression | 2 | 6400 | 12122 | 17.9→1.0 | 1600 | 5.31% | 0.7741 | 0.1719 | 0.1990 | 0.6989 | PASS | PASS | PASS |
| SMOTE + Tomek Links + Logistic Regression | 3 | 6400 | 12122 | 17.9→1.0 | 1600 | 5.31% | 0.7643 | 0.1569 | 0.2065 | 0.7090 | PASS | PASS | PASS |
| SMOTE + Tomek Links + Logistic Regression | 4 | 6400 | 12120 | 17.9→1.0 | 1600 | 5.31% | 0.7612 | 0.1425 | 0.1880 | 0.6767 | PASS | PASS | PASS |
| SMOTE + Tomek Links + Logistic Regression | 5 | 6400 | 12122 | 17.9→1.0 | 1600 | 5.31% | 0.7928 | 0.1867 | 0.2078 | 0.7133 | PASS | PASS | PASS |
| Logistic Regression (class_weight='balanced') | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.7912 | 0.1452 | 0.2085 | 0.7180 | PASS | PASS | PASS |
| Logistic Regression (class_weight='balanced') | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7717 | 0.1610 | 0.1950 | 0.6949 | PASS | PASS | PASS |
| Logistic Regression (class_weight='balanced') | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7666 | 0.1614 | 0.2034 | 0.7061 | PASS | PASS | PASS |
| Logistic Regression (class_weight='balanced') | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7599 | 0.1479 | 0.1858 | 0.6744 | PASS | PASS | PASS |
| Logistic Regression (class_weight='balanced') | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7952 | 0.1815 | 0.2074 | 0.7159 | PASS | PASS | PASS |
| XGBoost (scale_pos_weight động) | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.9223 | 0.6416 | 0.6061 | 0.7874 | PASS | PASS | PASS |
| XGBoost (scale_pos_weight động) | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9316 | 0.6123 | 0.5897 | 0.7623 | PASS | PASS | PASS |
| XGBoost (scale_pos_weight động) | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9008 | 0.6022 | 0.5641 | 0.7499 | PASS | PASS | PASS |
| XGBoost (scale_pos_weight động) | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9273 | 0.6655 | 0.5677 | 0.7502 | PASS | PASS | PASS |
| XGBoost (scale_pos_weight động) | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9274 | 0.6286 | 0.6026 | 0.7685 | PASS | PASS | PASS |
| BalancedRandomForestClassifier | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.9167 | 0.4958 | 0.4437 | 0.8298 | PASS | PASS | PASS |
| BalancedRandomForestClassifier | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9213 | 0.5147 | 0.4369 | 0.8289 | PASS | PASS | PASS |
| BalancedRandomForestClassifier | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9084 | 0.5291 | 0.4549 | 0.8280 | PASS | PASS | PASS |
| BalancedRandomForestClassifier | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9113 | 0.5248 | 0.4221 | 0.8116 | PASS | PASS | PASS |
| BalancedRandomForestClassifier | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.9276 | 0.5462 | 0.4194 | 0.8295 | PASS | PASS | PASS |
| EasyEnsembleClassifier | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.8199 | 0.2525 | 0.2301 | 0.7497 | PASS | PASS | PASS |
| EasyEnsembleClassifier | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.8194 | 0.2560 | 0.2188 | 0.7293 | PASS | PASS | PASS |
| EasyEnsembleClassifier | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.8218 | 0.2583 | 0.2340 | 0.7451 | PASS | PASS | PASS |
| EasyEnsembleClassifier | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7747 | 0.1996 | 0.2261 | 0.7223 | PASS | PASS | PASS |
| EasyEnsembleClassifier | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.8215 | 0.2105 | 0.2253 | 0.7414 | PASS | PASS | PASS |
| RUSBoostClassifier | 1 | 6400 | 6400 | 17.8→17.8 | 1600 | 5.25% | 0.7497 | 0.1320 | 0.1901 | 0.6037 | PASS | PASS | PASS |
| RUSBoostClassifier | 2 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7291 | 0.2128 | 0.2082 | 0.6132 | PASS | PASS | PASS |
| RUSBoostClassifier | 3 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7908 | 0.1878 | 0.2212 | 0.6895 | PASS | PASS | PASS |
| RUSBoostClassifier | 4 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7980 | 0.2233 | 0.2604 | 0.6089 | PASS | PASS | PASS |
| RUSBoostClassifier | 5 | 6400 | 6400 | 17.9→17.9 | 1600 | 5.31% | 0.7128 | 0.1294 | 0.1602 | 0.5887 | PASS | PASS | PASS |

## 3. Tái lập

```powershell
python -m labs.benchmark --cv 5
```
