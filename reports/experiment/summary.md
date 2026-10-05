# Thực nghiệm: Phương pháp đơn lẻ vs Phương pháp kết hợp (imbalanced learning)

- Dữ liệu: **synthetic(1:50): train n=16000 (dương 314 = 1.96%, IR=50.0) | test n=4000 (dương 79 = 1.98%, IR=49.6) | 20 đặc trưng**
- Cấu hình: `n_samples=20000`, `imbalance_ratio=1:50`, `n_splits=5`, `base_model=lightgbm`, `threshold=0.5`, `seed=42`
- Chống rò rỉ: mọi resampling nằm TRONG `imblearn.pipeline.Pipeline` (chỉ chạy trên train của từng fold); holdout test chấm đúng MỘT lần sau khi refit trên toàn bộ train.
- Metric chính: PR-AUC, ROC-AUC, F1 (thiểu số/macro), Balanced Accuracy, Recall, FPR — **Accuracy không dùng** để kết luận.

## 1. Danh mục pipeline

| Nhóm | Kỹ thuật | Mô tả | Resampling? |
|---|---|---|---|
| `baseline` | `baseline` — Baseline (không xử lý) | mốc so sánh: mô hình chuẩn trên dữ liệu gốc | không |
| `single-data` | `ros` — RandomOverSampler | sao chép mẫu thiểu số tới tỉ lệ mục tiêu | có (trong pipeline) |
| `single-data` | `smote` — SMOTE | nội suy mẫu thiểu số (dễ làm mờ biên) | có (trong pipeline) |
| `single-data` | `borderline_smote` — Borderline-SMOTE | chỉ nội suy từ mẫu thiểu số ở vùng DANGER | có (trong pipeline) |
| `single-data` | `adasyn` — ADASYN | sinh thêm theo độ khó của từng mẫu thiểu số | có (trong pipeline) |
| `single-data` | `rus` — RandomUnderSampler | hạ ngẫu nhiên lớp đa số (mất thông tin) | có (trong pipeline) |
| `single-data` | `tomek` — Tomek Links (cleaning) | làm sạch biên: bỏ cặp láng giềng khác lớp | có (trong pipeline) |
| `single-data` | `enn` — Edited Nearest Neighbours (cleaning) | làm sạch biên: bỏ mẫu bị láng giềng 'bỏ phiếu' khác lớp | có (trong pipeline) |
| `single-algorithm` | `class_weight` — Class Weights (balanced) | class_weight='balanced' — trọng số tính trong fit | không |
| `single-algorithm` | `focal_loss` — Focal Loss (custom objective) | Focal Loss: giảm trọng số mẫu dễ (gamma, alpha) | không |
| `single-ensemble` | `balanced_rf` — Balanced Random Forest | rừng cây, mỗi cây undersample lớp đa số | không |
| `single-ensemble` | `easy_ensemble` — EasyEnsemble | nhiều AdaBoost trên các tập con cân bằng | không |
| `single-ensemble` | `balanced_bagging` — Balanced Bagging | bagging với các tập con được cân bằng | không |
| `hybrid` | `smote_tomek` — HYBRID SMOTE + Tomek Links | SMOTE rồi DỌN cặp mẫu nhiễu (Tomek Links) | có (trong pipeline) |
| `hybrid` | `smote_enn` — HYBRID SMOTE + ENN | SMOTE rồi DỌN mẫu bị láng giềng phủ nhận (ENN) | có (trong pipeline) |
| `hybrid` | `smote_class_weight` — HYBRID SMOTE + Class Weights | SMOTE tỉ lệ vừa phải (0,5) + trọng số lớp | có (trong pipeline) |
| `hybrid` | `rusboost` — HYBRID RUSBoost (undersampling + boosting) | boosting có undersample từng vòng lặp | không |

## 2. Trung bình theo nhóm (Baseline · Đơn lẻ · Kết hợp)

| Nhóm             |   #kỹ thuật |   PR-AUC (TB) |   ΔPR-AUC vs baseline |   F1 (TB) |   Recall (TB) |   FPR (TB) |   giây/fold (TB) |
|:-----------------|------------:|--------------:|----------------------:|----------:|--------------:|-----------:|-----------------:|
| baseline         |           1 |      0.947382 |             0         |  0.871429 |      0.772152 | 0          |          1.02913 |
| single (đơn lẻ)  |          12 |      0.910406 |            -0.0369758 |  0.804232 |      0.859705 | 0.00913883 |          1.12381 |
| hybrid (kết hợp) |           4 |      0.922926 |            -0.0244564 |  0.830726 |      0.905063 | 0.00873502 |          1.28925 |
## 3. Xếp hạng theo PR-AUC (holdout test)

|   # | Kỹ thuật           | Nhóm             |   PR-AUC |   ΔPR-AUC vs baseline |   F1 (thiểu số) |   F1-macro |   Recall |         FPR |   Balanced Acc |
|----:|:-------------------|:-----------------|---------:|----------------------:|----------------:|-----------:|---------:|------------:|---------------:|
|   1 | smote_enn          | hybrid           | 0.95711  |           0.0097273   |        0.934211 |   0.966468 | 0.898734 | 0.000510074 |       0.949112 |
|   2 | adasyn             | single-data      | 0.95439  |           0.00700799  |        0.927152 |   0.962875 | 0.886076 | 0.000510074 |       0.942783 |
|   3 | smote              | single-data      | 0.953763 |           0.00638075  |        0.940397 |   0.969625 | 0.898734 | 0.000255037 |       0.94924  |
|   4 | smote_tomek        | hybrid           | 0.953763 |           0.00638075  |        0.940397 |   0.969625 | 0.898734 | 0.000255037 |       0.94924  |
|   5 | smote_class_weight | hybrid           | 0.953083 |           0.00570107  |        0.934211 |   0.966468 | 0.898734 | 0.000510074 |       0.949112 |
|   6 | class_weight       | single-algorithm | 0.951501 |           0.00411861  |        0.918919 |   0.958695 | 0.860759 | 0.000255037 |       0.930252 |
|   7 | tomek              | single-data      | 0.951156 |           0.00377356  |        0.879433 |   0.938635 | 0.78481  | 0           |       0.892405 |
|   8 | borderline_smote   | single-data      | 0.950631 |           0.00324906  |        0.926174 |   0.962387 | 0.873418 | 0.000255037 |       0.936581 |
|   9 | ros                | single-data      | 0.950226 |           0.00284332  |        0.90411  |   0.951164 | 0.835443 | 0.000255037 |       0.917594 |
|  10 | focal_loss         | single-algorithm | 0.948352 |           0.000969953 |        0.902778 |   0.950498 | 0.822785 | 0           |       0.911392 |
|  11 | enn                | single-data      | 0.947658 |           0.000275368 |        0.879433 |   0.938635 | 0.78481  | 0           |       0.892405 |
|  12 | baseline           | baseline         | 0.947382 |           0           |        0.871429 |   0.934569 | 0.772152 | 0           |       0.886076 |
|  13 | rus                | single-data      | 0.916965 |          -0.0304175   |        0.771739 |   0.883183 | 0.898734 | 0.00867126  |       0.945031 |
|  14 | balanced_rf        | single-ensemble  | 0.883186 |          -0.064196    |        0.581673 |   0.784062 | 0.924051 | 0.0252487   |       0.949401 |
|  15 | balanced_bagging   | single-ensemble  | 0.876129 |          -0.0712529   |        0.647619 |   0.81906  | 0.860759 | 0.0160673   |       0.922346 |
|  16 | rusboost           | hybrid           | 0.827748 |          -0.119635    |        0.514085 |   0.7481   | 0.924051 | 0.0336649   |       0.945193 |
|  17 | easy_ensemble      | single-ensemble  | 0.640921 |          -0.306462    |        0.371353 |   0.670131 | 0.886076 | 0.0581484   |       0.913964 |
## 4. Metric chi tiết theo pipeline — CV mean ± std

| Nhóm             | Kỹ thuật           | Trạng thái   | PR-AUC        | ROC-AUC       | F1 (thiểu số)   | F1-macro      | Balanced Acc   | Recall        | FPR           | Precision     | MCC           |   giây/fold |
|:-----------------|:-------------------|:-------------|:--------------|:--------------|:----------------|:--------------|:---------------|:--------------|:--------------|:--------------|:--------------|------------:|
| baseline         | baseline           | ok           | 0.961 ± 0.007 | 0.995 ± 0.006 | 0.866 ± 0.010   | 0.932 ± 0.005 | 0.885 ± 0.008  | 0.771 ± 0.015 | 0.000 ± 0.000 | 0.988 ± 0.010 | 0.870 ± 0.010 |       1.029 |
| single-data      | ros                | ok           | 0.958 ± 0.005 | 0.995 ± 0.005 | 0.898 ± 0.025   | 0.948 ± 0.013 | 0.919 ± 0.024  | 0.838 ± 0.048 | 0.001 ± 0.000 | 0.971 ± 0.018 | 0.900 ± 0.023 |       1.399 |
| single-data      | smote              | ok           | 0.965 ± 0.009 | 0.996 ± 0.005 | 0.906 ± 0.008   | 0.952 ± 0.004 | 0.939 ± 0.011  | 0.879 ± 0.023 | 0.001 ± 0.001 | 0.937 ± 0.027 | 0.905 ± 0.008 |       1.449 |
| single-data      | borderline_smote   | ok           | 0.960 ± 0.006 | 0.996 ± 0.005 | 0.903 ± 0.010   | 0.950 ± 0.005 | 0.934 ± 0.010  | 0.870 ± 0.021 | 0.001 ± 0.001 | 0.939 ± 0.025 | 0.902 ± 0.011 |       1.451 |
| single-data      | adasyn             | ok           | 0.961 ± 0.009 | 0.997 ± 0.004 | 0.902 ± 0.014   | 0.950 ± 0.007 | 0.943 ± 0.007  | 0.889 ± 0.014 | 0.002 ± 0.001 | 0.916 ± 0.032 | 0.900 ± 0.015 |       1.439 |
| single-data      | rus                | ok           | 0.918 ± 0.013 | 0.995 ± 0.004 | 0.707 ± 0.025   | 0.850 ± 0.013 | 0.961 ± 0.010  | 0.936 ± 0.020 | 0.014 ± 0.002 | 0.569 ± 0.029 | 0.723 ± 0.024 |       0.302 |
| single-data      | tomek              | ok           | 0.960 ± 0.007 | 0.995 ± 0.006 | 0.879 ± 0.016   | 0.939 ± 0.008 | 0.895 ± 0.013  | 0.790 ± 0.027 | 0.000 ± 0.000 | 0.992 ± 0.010 | 0.883 ± 0.015 |       1.252 |
| single-data      | enn                | ok           | 0.961 ± 0.006 | 0.995 ± 0.007 | 0.873 ± 0.010   | 0.936 ± 0.005 | 0.890 ± 0.010  | 0.780 ± 0.020 | 0.000 ± 0.000 | 0.992 ± 0.010 | 0.878 ± 0.009 |       1.231 |
| single-algorithm | class_weight       | ok           | 0.959 ± 0.002 | 0.996 ± 0.005 | 0.904 ± 0.010   | 0.951 ± 0.005 | 0.933 ± 0.010  | 0.866 ± 0.021 | 0.001 ± 0.001 | 0.946 ± 0.035 | 0.903 ± 0.011 |       1.118 |
| single-algorithm | focal_loss         | ok           | 0.959 ± 0.008 | 0.995 ± 0.005 | 0.882 ± 0.009   | 0.940 ± 0.004 | 0.898 ± 0.007  | 0.796 ± 0.015 | 0.000 ± 0.000 | 0.988 ± 0.015 | 0.885 ± 0.009 |       1.35  |
| single-ensemble  | balanced_rf        | ok           | 0.901 ± 0.020 | 0.994 ± 0.002 | 0.571 ± 0.035   | 0.778 ± 0.019 | 0.964 ± 0.006  | 0.955 ± 0.012 | 0.028 ± 0.004 | 0.408 ± 0.036 | 0.614 ± 0.028 |       0.415 |
| single-ensemble  | easy_ensemble      | ok           | 0.741 ± 0.030 | 0.977 ± 0.008 | 0.346 ± 0.012   | 0.655 ± 0.007 | 0.926 ± 0.013  | 0.921 ± 0.032 | 0.068 ± 0.006 | 0.213 ± 0.010 | 0.424 ± 0.006 |       1.462 |
| single-ensemble  | balanced_bagging   | ok           | 0.897 ± 0.019 | 0.994 ± 0.001 | 0.652 ± 0.038   | 0.821 ± 0.020 | 0.956 ± 0.007  | 0.930 ± 0.016 | 0.019 ± 0.004 | 0.504 ± 0.049 | 0.676 ± 0.030 |       0.617 |
| hybrid           | smote_tomek        | ok           | 0.965 ± 0.009 | 0.996 ± 0.005 | 0.906 ± 0.008   | 0.952 ± 0.004 | 0.939 ± 0.011  | 0.879 ± 0.023 | 0.001 ± 0.001 | 0.937 ± 0.027 | 0.905 ± 0.008 |       1.768 |
| hybrid           | smote_enn          | ok           | 0.965 ± 0.008 | 0.996 ± 0.006 | 0.916 ± 0.018   | 0.957 ± 0.009 | 0.950 ± 0.008  | 0.901 ± 0.015 | 0.001 ± 0.001 | 0.932 ± 0.035 | 0.915 ± 0.019 |       1.789 |
| hybrid           | smote_class_weight | ok           | 0.965 ± 0.009 | 0.996 ± 0.005 | 0.916 ± 0.014   | 0.957 ± 0.007 | 0.950 ± 0.013  | 0.901 ± 0.025 | 0.001 ± 0.000 | 0.932 ± 0.022 | 0.914 ± 0.014 |       1.396 |
| hybrid           | rusboost           | ok           | 0.808 ± 0.033 | 0.987 ± 0.004 | 0.523 ± 0.031   | 0.753 ± 0.017 | 0.938 ± 0.012  | 0.908 ± 0.024 | 0.031 ± 0.003 | 0.368 ± 0.029 | 0.566 ± 0.028 |       0.204 |
## 5. Metric trên holdout test + chi phí tính toán

| Nhóm             | Kỹ thuật           |   PR-AUC |   ROC-AUC |   F1 (thiểu số) |   F1-macro |   Balanced Acc |   Recall |      FPR |   Precision |      MCC |   n train sau resample |   giây resample/fold |   giây fit/fold |
|:-----------------|:-------------------|---------:|----------:|----------------:|-----------:|---------------:|---------:|---------:|------------:|---------:|-----------------------:|---------------------:|----------------:|
| baseline         | baseline           | 0.947382 |  0.990347 |        0.871429 |   0.934569 |       0.886076 | 0.772152 | 0        |    1        | 0.876712 |                  nan   |             nan      |           1.029 |
| single-data      | ros                | 0.950226 |  0.992762 |        0.90411  |   0.951164 |       0.917594 | 0.835443 | 0.000255 |    0.985075 | 0.905517 |                18822.8 |               0.0044 |           1.399 |
| single-data      | smote              | 0.953763 |  0.994118 |        0.940397 |   0.969625 |       0.94924  | 0.898734 | 0.000255 |    0.986111 | 0.940303 |                18822.8 |               0.0196 |           1.449 |
| single-data      | borderline_smote   | 0.950631 |  0.990509 |        0.926174 |   0.962387 |       0.936581 | 0.873418 | 0.000255 |    0.985714 | 0.926535 |                18822.8 |               0.0149 |           1.451 |
| single-data      | adasyn             | 0.95439  |  0.989941 |        0.927152 |   0.962875 |       0.942783 | 0.886076 | 0.00051  |    0.972222 | 0.926789 |                18805   |               0.0145 |           1.439 |
| single-data      | rus                | 0.916965 |  0.989217 |        0.771739 |   0.883183 |       0.945031 | 0.898734 | 0.008671 |    0.67619  | 0.774612 |                  753.6 |               0.0029 |           0.302 |
| single-data      | tomek              | 0.951156 |  0.993266 |        0.879433 |   0.938635 |       0.892405 | 0.78481  | 0        |    1        | 0.883981 |                12796.6 |               0.1888 |           1.252 |
| single-data      | enn                | 0.947658 |  0.990638 |        0.879433 |   0.938635 |       0.892405 | 0.78481  | 0        |    1        | 0.883981 |                12768.6 |               0.1654 |           1.231 |
| single-algorithm | class_weight       | 0.951501 |  0.992384 |        0.918919 |   0.958695 |       0.930252 | 0.860759 | 0.000255 |    0.985507 | 0.919579 |                  nan   |             nan      |           1.118 |
| single-algorithm | focal_loss         | 0.948352 |  0.991694 |        0.902778 |   0.950498 |       0.911392 | 0.822785 | 0        |    1        | 0.90546  |                  nan   |             nan      |           1.35  |
| single-ensemble  | balanced_rf        | 0.883186 |  0.991845 |        0.581673 |   0.784062 |       0.949401 | 0.924051 | 0.025249 |    0.424419 | 0.61649  |                  nan   |             nan      |           0.415 |
| single-ensemble  | easy_ensemble      | 0.640921 |  0.959139 |        0.371353 |   0.670131 |       0.913964 | 0.886076 | 0.058148 |    0.234899 | 0.43871  |                  nan   |             nan      |           1.462 |
| single-ensemble  | balanced_bagging   | 0.876129 |  0.983232 |        0.647619 |   0.81906  |       0.922346 | 0.860759 | 0.016067 |    0.519084 | 0.660352 |                  nan   |             nan      |           0.617 |
| hybrid           | smote_tomek        | 0.953763 |  0.994118 |        0.940397 |   0.969625 |       0.94924  | 0.898734 | 0.000255 |    0.986111 | 0.940303 |                18822.8 |               0.3856 |           1.768 |
| hybrid           | smote_enn          | 0.95711  |  0.994044 |        0.934211 |   0.966468 |       0.949112 | 0.898734 | 0.00051  |    0.972603 | 0.933694 |                18699.4 |               0.3935 |           1.789 |
| hybrid           | smote_class_weight | 0.953083 |  0.994609 |        0.934211 |   0.966468 |       0.949112 | 0.898734 | 0.00051  |    0.972603 | 0.933694 |                18822.8 |               0.0083 |           1.396 |
| hybrid           | rusboost           | 0.827748 |  0.984186 |        0.514085 |   0.7481   |       0.945193 | 0.924051 | 0.033665 |    0.356098 | 0.561833 |                  nan   |             nan      |           0.204 |
## 6. Kiểm chứng chống rò rỉ dữ liệu

| Kỹ thuật | Nhóm | Kiểm chứng chống rò rỉ | Kết quả |
|---|---|---|---|
| `baseline` | `baseline` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `ros` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `smote` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `borderline_smote` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `adasyn` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `rus` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `tomek` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `enn` | `single-data` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `class_weight` | `single-algorithm` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `focal_loss` | `single-algorithm` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `balanced_rf` | `single-ensemble` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `easy_ensemble` | `single-ensemble` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `balanced_bagging` | `single-ensemble` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `smote_tomek` | `hybrid` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `smote_enn` | `hybrid` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `smote_class_weight` | `hybrid` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=1): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |
| `rusboost` | `hybrid` | resampling dùng imblearn.pipeline: PASS, sampler chỉ nằm trong pipeline (n=0): PASS, fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS | PASS |

## Phân tích chuyên sâu (yêu cầu #5)

Bối cảnh: 20000 mẫu, mất cân bằng 1:50, 5 fold stratified, mô hình nền `lightgbm`, ngưỡng 0.5. Metric chính: PR-AUC, F1 (thiểu số), Recall, FPR.

**Mốc Baseline (không xử lý)**: PR-AUC 0.9474, F1 0.871, Recall 0.772, FPR 0.0000, 1.03s/fold. (Accuracy chẩn đoán 99.55% trong khi đoán lớp đa số đã được 98.03% ⇒ accuracy không dùng để kết luận.)

### 1. SMOTE đơn lẻ vs SMOTE + dọn biên (SMOTETomek / SMOTEENN)

- SMOTE đơn lẻ: PR-AUC **0.9538**, F1 0.940 (CV 0.9645 ± 0.009), Recall 0.899; train sau resample 18823 mẫu.
- SMOTE sinh mẫu bằng NỘI SUY giữa các láng giềng thiểu số nên dễ tạo mẫu nằm trong vùng đa số (boundary blur/nhiễu). SMOTETomek/SMOTEENN thêm bước DỌN: bỏ mẫu sát biên hoặc bị láng giềng 'phủ nhận'.
- SMOTETomek: PR-AUC **0.9538** (Δ +0.0000), F1 0.940 (Δ +0.0000), FPR 0.0003 (Δ +0.0000), train sau resample 18823 mẫu ⇒ KÉM HƠN so với SMOTE thuần.
- SMOTEENN: PR-AUC **0.9571** (Δ +0.0033), F1 0.934 (Δ -0.0062), FPR 0.0005 (Δ +0.0003), train sau resample 18699 mẫu ⇒ TỐT HƠN so với SMOTE thuần.

**Cách đọc**: dọn biên chỉ có lợi khi mẫu tổng hợp thực sự rơi vào vùng chồng lấn. Khi hai lớp tách rõ (`class_sep` lớn) hoặc mẫu thiểu số quá ít, phần lớn mẫu SMOTE vốn đã đúng vùng ⇒ dọn biên chỉ làm MẤT thông tin và có thể GIẢM Recall (bỏ cả mẫu sát biên 'khó nhưng thật'). Vì vậy phải đọc đồng thời ΔPR-AUC, ΔF1 và ΔFPR.

### 2. Chi phí tính toán: Resampling lớn vs Cost/Weight (algorithm-level)

| Kỹ thuật | n train sau resample | Hệ số kích thước | giây/fold (fit) | giây/fold (resample) | PR-AUC | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `smote` | 18823 | 1.47× | 1.449 | 0.0196 | 0.9538 | 0.899 | 0.0003 |
| `borderline_smote` | 18823 | 1.47× | 1.451 | 0.0149 | 0.9506 | 0.873 | 0.0003 |
| `adasyn` | 18805 | 1.47× | 1.439 | 0.0145 | 0.9544 | 0.886 | 0.0005 |
| `rus` | 754 | 0.06× | 0.302 | 0.0029 | 0.9170 | 0.899 | 0.0087 |
| `smote_tomek` | 18823 | 1.47× | 1.768 | 0.3856 | 0.9538 | 0.899 | 0.0003 |
| `smote_enn` | 18699 | 1.46× | 1.789 | 0.3935 | 0.9571 | 0.899 | 0.0005 |
| `class_weight` | — | 1.00× | 1.118 | — | 0.9515 | 0.861 | 0.0003 |
| `focal_loss` | — | 1.00× | 1.350 | — | 0.9484 | 0.823 | 0.0000 |
| `smote_class_weight` | 18823 | 1.47× | 1.396 | 0.0083 | 0.9531 | 0.899 | 0.0005 |

- Oversampling làm tập train PHÌNH RA (hệ số kích thước > 1×) ⇒ mỗi vòng lặp boosting phải xử lý nhiều mẫu hơn, cộng thêm chi phí sinh mẫu (cột giây/fold resample).
- `class_weight='balanced'` và Focal Loss KHÔNG đổi kích thước dữ liệu (1,0×, không tốn thời gian resample) nhưng vẫn đổi điểm vận hành (Recall/F1).
- Trên cấu hình này, fit với SMOTE chậm hơn `class_weight` khoảng **1.3×** (1.45s so với 1.12s mỗi fold); khoảng cách còn tăng khi tỉ lệ mất cân bằng cao hơn (1:100) hoặc số fold lớn hơn, vì mỗi fold phải resample lại từ đầu (đúng nguyên tắc chống rò rỉ).
- Chọn theo chi phí: dữ liệu lớn/nhiều fold ⇒ ưu tiên cost/weight (rẻ, không phình dữ liệu); resampling chỉ đáng dùng khi mô hình không hỗ trợ trọng số lớp hoặc cần 'thấy' nhiều mẫu dương hơn.

### 3. Khi nào kết hợp (hybrid) vượt trội so với đơn lẻ?

| Hybrid | Đơn lẻ tốt nhất | ΔPR-AUC | ΔF1 | ΔRecall | ΔFPR | Kết luận |
|---|---|---:|---:|---:|---:|---|
| `smote_tomek` | `smote` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | đơn lẻ tốt hơn |
| `smote_enn` | `smote` | +0.0033 | -0.0062 | +0.0000 | +0.0003 | HYBRID tốt hơn |
| `smote_class_weight` | `smote` | -0.0007 | -0.0062 | +0.0000 | +0.0003 | đơn lẻ tốt hơn |
| `rusboost` | `balanced_rf` | -0.0554 | -0.0676 | +0.0000 | +0.0084 | đơn lẻ tốt hơn |

- Hybrid thắng trên bộ này: `smote_enn`. Kết hợp có lợi nhất khi hai can thiệp BÙ TRỪ nhau: resampling đưa thêm mẫu thiểu số vào vùng khó, còn trọng số lớp/dọn biên chỉnh lại đúng chỗ mô hình còn yếu thay vì nhân bản nhiễu.
- Điều kiện hybrid thường thắng: (a) overlap giữa hai lớp lớn (mẫu SMOTE hay rơi vào vùng đa số) — dọn biên giúp; (b) mẫu thiểu số quá ít (1:100) — SMOTE + trọng số lớp ổn định hơn SMOTE thuần; (c) cần tỉ lệ dương cao trong train nhưng ngân sách tính toán hạn chế — RUSBoost/BalancedBagging rẻ hơn oversampling lớn.

### Khuyến nghị rút ra

- Tốt nhất theo PR_AUC: `smote_enn` (hybrid) = **0.9571** (baseline 0.9474, Δ +0.0097)
- Tốt nhất theo F1: `smote` (single-data) = **0.9404** (baseline 0.8714, Δ +0.0690)
- Tốt nhất theo RECALL: `balanced_rf` (single-ensemble) = **0.9241** (baseline 0.7722, Δ +0.1519)
- FPR thấp nhất: `baseline` (0.0000), `tomek` (0.0000), `enn` (0.0000)
- FPR cao nhất (cần chú ý báo động giả): `balanced_rf` (0.0252), `rusboost` (0.0337), `easy_ensemble` (0.0581)
- Quy trình khuyến nghị: (1) luôn lấy baseline làm mốc; (2) xếp hạng theo PR-AUC (không phụ thuộc ngưỡng) rồi kiểm tra F1/Recall và FPR; (3) chọn ngưỡng theo chi phí FN/FP trên xác suất out-of-fold; (4) chỉ giữ kỹ thuật khi mức cải thiện vượt độ lệch chuẩn giữa các fold.

### Giới hạn

- Dataset mặc định là GIẢ LẬP (`make_classification`): kết luận định tính chuyển được, con số thì không. Muốn chạy dữ liệu thật (credit card fraud) dùng `--data path/to/creditcard.csv --target Class`.
- Mỗi kỹ thuật dùng MỘT cấu hình hợp lý (tỉ lệ oversampling 0,5; Focal Loss `gamma=2`, `alpha=0,75`) chứ chưa grid-search ⇒ đây là so sánh 'cùng ngân sách', không phải 'tối ưu cho từng kỹ thuật'.
- Ngưỡng báo cáo cố định 0,5 để so sánh trực tiếp; muốn tối ưu ngưỡng hãy dùng `labs.imbalance_lab.thresholds.tune_thresholds_from_pr_curve` (chọn trên xác suất OOF).


