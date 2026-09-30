# Danh mục kỹ thuật xử lý mất cân bằng (yêu cầu #2) — so sánh không rò rỉ dữ liệu

- Backend phân loại: **lightgbm** · resampling: **imblearn** · n_samples 20000 · 5 fold · seed 42
- Phân phối nhãn: toàn bộ n=20000 (dương 2.00%, IR=49.0); train_pool n=16000 (dương 2.00%); test n=4000 (dương 2.00%)
- **Threshold tuning**: ngưỡng chọn trên **đường Precision-Recall của xác suất out-of-fold** (chỉ train_pool): `best_f1` (PR), `best_cost` (chi phí FN/FP), `min_precision`; kèm mốc 0.5 để so sánh. Test chỉ được chấm **một lần** sau khi chốt.
- Chống rò rỉ: mọi sampler nằm TRONG `imblearn.pipeline.Pipeline`; cost-sensitive / Focal Loss tính trọng số trong `fit`; ensemble lấy mẫu bên trong `fit`; `cv.assert_val_untouched` chạy từng fold.

## 1. Danh mục ↔ cài đặt ↔ trạng thái

| Nhóm | Kỹ thuật | Mô tả | Cài đặt | Trạng thái |
|---|---|---|---|---|
| `baseline` | `baseline` | BASELINE — boosting mặc định, KHÔNG can thiệp mất cân bằng (mốc so sánh) | models.make_base_classifier (boosting mặc định, KHÔNG can thiệp) | PASS |
| `data-level/oversampling` | `ros` | RandomOverSampler — sao chép mẫu thiểu số (không tạo mẫu mới) | samplers.RandomOverSampler / imblearn RandomOverSampler | PASS |
| `data-level/oversampling` | `smote` | SMOTE — nội suy giữa mẫu thiểu số và láng giềng thiểu số | samplers.SMOTE / imblearn SMOTE | PASS |
| `data-level/oversampling` | `borderline_smote` | BorderlineSMOTE — chỉ nội suy từ mẫu thiểu số nằm ở BIÊN (vùng DANGER) | samplers.BorderlineSMOTE / imblearn BorderlineSMOTE | PASS |
| `data-level/oversampling` | `adasyn` | ADASYN — sinh thêm tỉ lệ với độ khó (số láng giềng đa số) của từng mẫu thiểu số | samplers.ADASYN / imblearn ADASYN | PASS |
| `data-level/undersampling` | `rus` | RandomUnderSampler — hạ ngẫu nhiên lớp đa số về tỉ lệ mục tiêu | samplers.RandomUnderSampler / imblearn RandomUnderSampler | PASS |
| `data-level/undersampling` | `tomek` | Tomek Links — làm sạch biên: bỏ mẫu đa số trong cặp láng giềng khác lớp | samplers.TomekLinks / imblearn TomekLinks | PASS |
| `data-level/undersampling` | `enn` | EditedNearestNeighbours — làm sạch biên: bỏ mẫu có láng giềng khác lớp | samplers.EditedNearestNeighbours / imblearn EditedNearestNeighbours | PASS |
| `hybrid` | `smote_tomek` | HYBRID SMOTE + Tomek Links | samplers.make_hybrid_sampler('smote_tomek') / imblearn SMOTETomek | PASS |
| `hybrid` | `smote_enn` | HYBRID SMOTE + EditedNearestNeighbours | samplers.make_hybrid_sampler('smote_enn') / imblearn SMOTEENN | PASS |
| `algorithm-level` | `cost_sensitive_scale_pos_weight` | LightGBM + `scale_pos_weight = n_âm/n_dương` tính TRONG fit | models.ScalePosWeightClassifier | PASS |
| `algorithm-level` | `cost_sensitive_class_weight` | `class_weight='balanced'` — trọng số mẫu suy trong fit | models.BalancedWeightClassifier | PASS |
| `algorithm-level` | `focal_loss` | Focal Loss — custom objective của LightGBM (gamma làm mờ mẫu dễ, alpha theo lớp) | losses.FocalLossClassifier (grad/hess giải tích) | PASS |
| `ensemble` | `balanced_rf` | BalancedRandomForestClassifier (undersample từng cây, TRONG fit) | imblearn BalancedRandomForestClassifier | PASS |
| `ensemble` | `easy_ensemble` | EasyEnsembleClassifier (nhiều AdaBoost trên các tập con cân bằng) | imblearn EasyEnsembleClassifier | PASS |
| `ensemble` | `rusboost` | RUSBoostClassifier (boosting + undersample từng vòng) | imblearn RUSBoostClassifier | PASS |

## 2. Metric trên holdout test — ngưỡng PR tốt nhất vs mốc 0.5 (yêu cầu #3)

*(Accuracy KHÔNG có trong bảng: ở tỉ lệ 98/2 đoán 'lớp đa số' đã đạt ~98% ⇒ accuracy chỉ là chỉ số chẩn đoán.)*

| Nhóm | Kỹ thuật | Ngưỡng | thr | Precision | Recall | F1 | F1-macro | F1-weighted | F-beta(2) | ΔF1 vs 0.5 | ΔRecall | PR-AUC | ROC-AUC | Brier | MCC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | `baseline` | best_f1 | 0.0768 | 0.887 | 0.787 | 0.834 | 0.916 | 0.994 | 0.806 | 0.077 | 0.162 | 0.8891 | 0.991 | 0.0066 | 0.833 |
| `baseline` | `baseline` | fixed_0.5 | 0.5000 | 0.962 | 0.625 | 0.758 | 0.877 | 0.991 | 0.672 | 0.000 | 0.000 | 0.8891 | 0.991 | 0.0066 | 0.772 |
| `baseline` | `baseline` | best_cost | 0.0127 | 0.657 | 0.863 | 0.746 | 0.870 | 0.989 | 0.812 | -0.012 | 0.238 | 0.8891 | 0.991 | 0.0066 | 0.747 |
| `baseline` | `baseline` | min_precision | 0.0065 | 0.567 | 0.900 | 0.696 | 0.844 | 0.986 | 0.805 | -0.062 | 0.275 | 0.8891 | 0.991 | 0.0066 | 0.707 |
| `data-level/oversampling` | `ros` | best_f1 | 0.2408 | 0.877 | 0.800 | 0.837 | 0.917 | 0.994 | 0.814 | 0.011 | 0.062 | 0.8909 | 0.991 | 0.0053 | 0.834 |
| `data-level/oversampling` | `ros` | fixed_0.5 | 0.5000 | 0.937 | 0.738 | 0.825 | 0.911 | 0.993 | 0.770 | 0.000 | 0.000 | 0.8909 | 0.991 | 0.0053 | 0.828 |
| `data-level/oversampling` | `ros` | best_cost | 0.0396 | 0.673 | 0.850 | 0.751 | 0.873 | 0.989 | 0.808 | -0.074 | 0.112 | 0.8909 | 0.991 | 0.0053 | 0.751 |
| `data-level/oversampling` | `ros` | min_precision | 0.0162 | 0.530 | 0.887 | 0.664 | 0.827 | 0.984 | 0.782 | -0.162 | 0.150 | 0.8909 | 0.991 | 0.0053 | 0.678 |
| `data-level/oversampling` | `smote` | best_f1 | 0.7417 | 0.894 | 0.738 | 0.808 | 0.902 | 0.993 | 0.764 | 0.011 | -0.050 | 0.8703 | 0.989 | 0.0061 | 0.809 |
| `data-level/oversampling` | `smote` | fixed_0.5 | 0.5000 | 0.808 | 0.787 | 0.797 | 0.897 | 0.992 | 0.791 | 0.000 | 0.000 | 0.8703 | 0.989 | 0.0061 | 0.793 |
| `data-level/oversampling` | `smote` | best_cost | 0.0877 | 0.556 | 0.875 | 0.680 | 0.836 | 0.985 | 0.785 | -0.118 | 0.088 | 0.8703 | 0.989 | 0.0061 | 0.690 |
| `data-level/oversampling` | `smote` | min_precision | 0.0796 | 0.547 | 0.875 | 0.673 | 0.832 | 0.985 | 0.781 | -0.124 | 0.088 | 0.8703 | 0.989 | 0.0061 | 0.684 |
| `data-level/oversampling` | `borderline_smote` | best_f1 | 0.5949 | 0.886 | 0.775 | 0.827 | 0.912 | 0.993 | 0.795 | 0.016 | 0.000 | 0.8734 | 0.987 | 0.0057 | 0.825 |
| `data-level/oversampling` | `borderline_smote` | fixed_0.5 | 0.5000 | 0.849 | 0.775 | 0.810 | 0.903 | 0.993 | 0.789 | 0.000 | 0.000 | 0.8734 | 0.987 | 0.0057 | 0.808 |
| `data-level/oversampling` | `borderline_smote` | best_cost | 0.0462 | 0.585 | 0.900 | 0.709 | 0.851 | 0.987 | 0.813 | -0.101 | 0.125 | 0.8734 | 0.987 | 0.0057 | 0.719 |
| `data-level/oversampling` | `borderline_smote` | min_precision | 0.0462 | 0.585 | 0.900 | 0.709 | 0.851 | 0.987 | 0.813 | -0.101 | 0.125 | 0.8734 | 0.987 | 0.0057 | 0.719 |
| `data-level/oversampling` | `adasyn` | best_f1 | 0.7870 | 0.857 | 0.675 | 0.755 | 0.875 | 0.991 | 0.705 | -0.035 | -0.100 | 0.8700 | 0.991 | 0.0066 | 0.756 |
| `data-level/oversampling` | `adasyn` | fixed_0.5 | 0.5000 | 0.805 | 0.775 | 0.790 | 0.893 | 0.992 | 0.781 | 0.000 | 0.000 | 0.8700 | 0.991 | 0.0066 | 0.786 |
| `data-level/oversampling` | `adasyn` | best_cost | 0.1573 | 0.614 | 0.875 | 0.722 | 0.857 | 0.988 | 0.806 | -0.068 | 0.100 | 0.8700 | 0.991 | 0.0066 | 0.727 |
| `data-level/oversampling` | `adasyn` | min_precision | 0.1206 | 0.582 | 0.887 | 0.703 | 0.848 | 0.987 | 0.803 | -0.087 | 0.112 | 0.8700 | 0.991 | 0.0066 | 0.712 |
| `data-level/undersampling` | `rus` | best_f1 | 0.9447 | 0.725 | 0.625 | 0.671 | 0.832 | 0.987 | 0.643 | 0.190 | -0.250 | 0.7637 | 0.984 | 0.0286 | 0.667 |
| `data-level/undersampling` | `rus` | fixed_0.5 | 0.5000 | 0.332 | 0.875 | 0.481 | 0.731 | 0.970 | 0.659 | 0.000 | 0.000 | 0.7637 | 0.984 | 0.0286 | 0.525 |
| `data-level/undersampling` | `rus` | best_cost | 0.6092 | 0.398 | 0.875 | 0.547 | 0.766 | 0.976 | 0.706 | 0.066 | 0.000 | 0.7637 | 0.984 | 0.0286 | 0.579 |
| `data-level/undersampling` | `rus` | min_precision | 0.7470 | 0.493 | 0.850 | 0.624 | 0.807 | 0.982 | 0.742 | 0.143 | -0.025 | 0.7637 | 0.984 | 0.0286 | 0.638 |
| `data-level/undersampling` | `tomek` | best_f1 | 0.0799 | 0.900 | 0.787 | 0.840 | 0.918 | 0.994 | 0.808 | 0.082 | 0.162 | 0.8956 | 0.992 | 0.0065 | 0.839 |
| `data-level/undersampling` | `tomek` | fixed_0.5 | 0.5000 | 0.962 | 0.625 | 0.758 | 0.877 | 0.991 | 0.672 | 0.000 | 0.000 | 0.8956 | 0.992 | 0.0065 | 0.772 |
| `data-level/undersampling` | `tomek` | best_cost | 0.0207 | 0.756 | 0.850 | 0.800 | 0.898 | 0.992 | 0.829 | 0.042 | 0.225 | 0.8956 | 0.992 | 0.0065 | 0.797 |
| `data-level/undersampling` | `tomek` | min_precision | 0.0071 | 0.577 | 0.887 | 0.700 | 0.846 | 0.986 | 0.801 | -0.058 | 0.262 | 0.8956 | 0.992 | 0.0065 | 0.709 |
| `data-level/undersampling` | `enn` | best_f1 | 0.1043 | 0.886 | 0.775 | 0.827 | 0.912 | 0.993 | 0.795 | 0.054 | 0.138 | 0.8888 | 0.991 | 0.0064 | 0.825 |
| `data-level/undersampling` | `enn` | fixed_0.5 | 0.5000 | 0.981 | 0.637 | 0.773 | 0.884 | 0.992 | 0.685 | 0.000 | 0.000 | 0.8888 | 0.991 | 0.0064 | 0.788 |
| `data-level/undersampling` | `enn` | best_cost | 0.0072 | 0.534 | 0.887 | 0.667 | 0.829 | 0.984 | 0.784 | -0.106 | 0.250 | 0.8888 | 0.991 | 0.0064 | 0.681 |
| `data-level/undersampling` | `enn` | min_precision | 0.0073 | 0.534 | 0.887 | 0.667 | 0.829 | 0.984 | 0.784 | -0.106 | 0.250 | 0.8888 | 0.991 | 0.0064 | 0.681 |
| `hybrid` | `smote_tomek` | best_f1 | 0.7417 | 0.894 | 0.738 | 0.808 | 0.902 | 0.993 | 0.764 | 0.011 | -0.050 | 0.8703 | 0.989 | 0.0061 | 0.809 |
| `hybrid` | `smote_tomek` | fixed_0.5 | 0.5000 | 0.808 | 0.787 | 0.797 | 0.897 | 0.992 | 0.791 | 0.000 | 0.000 | 0.8703 | 0.989 | 0.0061 | 0.793 |
| `hybrid` | `smote_tomek` | best_cost | 0.0877 | 0.556 | 0.875 | 0.680 | 0.836 | 0.985 | 0.785 | -0.118 | 0.088 | 0.8703 | 0.989 | 0.0061 | 0.690 |
| `hybrid` | `smote_tomek` | min_precision | 0.0796 | 0.547 | 0.875 | 0.673 | 0.832 | 0.985 | 0.781 | -0.124 | 0.088 | 0.8703 | 0.989 | 0.0061 | 0.684 |
| `hybrid` | `smote_enn` | best_f1 | 0.8038 | 0.855 | 0.738 | 0.792 | 0.894 | 0.992 | 0.758 | 0.011 | -0.062 | 0.8636 | 0.990 | 0.0071 | 0.790 |
| `hybrid` | `smote_enn` | fixed_0.5 | 0.5000 | 0.762 | 0.800 | 0.780 | 0.888 | 0.991 | 0.792 | 0.000 | 0.000 | 0.8636 | 0.990 | 0.0071 | 0.776 |
| `hybrid` | `smote_enn` | best_cost | 0.0984 | 0.493 | 0.900 | 0.637 | 0.813 | 0.982 | 0.773 | -0.143 | 0.100 | 0.8636 | 0.990 | 0.0071 | 0.658 |
| `hybrid` | `smote_enn` | min_precision | 0.1405 | 0.556 | 0.875 | 0.680 | 0.836 | 0.985 | 0.785 | -0.101 | 0.075 | 0.8636 | 0.990 | 0.0071 | 0.690 |
| `algorithm-level` | `cost_sensitive_scale_pos_weight` | best_f1 | 0.3591 | 0.875 | 0.787 | 0.829 | 0.913 | 0.993 | 0.804 | 0.010 | 0.050 | 0.8919 | 0.988 | 0.0051 | 0.827 |
| `algorithm-level` | `cost_sensitive_scale_pos_weight` | fixed_0.5 | 0.5000 | 0.922 | 0.738 | 0.819 | 0.908 | 0.993 | 0.768 | 0.000 | 0.000 | 0.8919 | 0.988 | 0.0051 | 0.821 |
| `algorithm-level` | `cost_sensitive_scale_pos_weight` | best_cost | 0.0537 | 0.673 | 0.850 | 0.751 | 0.873 | 0.989 | 0.808 | -0.068 | 0.112 | 0.8919 | 0.988 | 0.0051 | 0.751 |
| `algorithm-level` | `cost_sensitive_scale_pos_weight` | min_precision | 0.0198 | 0.533 | 0.900 | 0.670 | 0.830 | 0.984 | 0.791 | -0.150 | 0.162 | 0.8919 | 0.988 | 0.0051 | 0.685 |
| `algorithm-level` | `cost_sensitive_class_weight` | best_f1 | 0.2912 | 0.844 | 0.812 | 0.828 | 0.912 | 0.993 | 0.819 | -0.005 | 0.062 | 0.8950 | 0.992 | 0.0050 | 0.825 |
| `algorithm-level` | `cost_sensitive_class_weight` | fixed_0.5 | 0.5000 | 0.938 | 0.750 | 0.833 | 0.915 | 0.994 | 0.781 | 0.000 | 0.000 | 0.8950 | 0.992 | 0.0050 | 0.836 |
| `algorithm-level` | `cost_sensitive_class_weight` | best_cost | 0.0522 | 0.639 | 0.863 | 0.734 | 0.864 | 0.988 | 0.806 | -0.099 | 0.113 | 0.8950 | 0.992 | 0.0050 | 0.736 |
| `algorithm-level` | `cost_sensitive_class_weight` | min_precision | 0.0275 | 0.549 | 0.912 | 0.685 | 0.838 | 0.985 | 0.806 | -0.148 | 0.162 | 0.8950 | 0.992 | 0.0050 | 0.701 |
| `algorithm-level` | `focal_loss` | best_f1 | 0.3038 | 0.823 | 0.812 | 0.818 | 0.907 | 0.993 | 0.815 | 0.041 | 0.162 | 0.8914 | 0.990 | 0.0073 | 0.814 |
| `algorithm-level` | `focal_loss` | fixed_0.5 | 0.5000 | 0.963 | 0.650 | 0.776 | 0.886 | 0.992 | 0.695 | 0.000 | 0.000 | 0.8914 | 0.990 | 0.0073 | 0.788 |
| `algorithm-level` | `focal_loss` | best_cost | 0.2148 | 0.723 | 0.850 | 0.782 | 0.888 | 0.991 | 0.821 | 0.005 | 0.200 | 0.8914 | 0.990 | 0.0073 | 0.779 |
| `algorithm-level` | `focal_loss` | min_precision | 0.1591 | 0.571 | 0.900 | 0.699 | 0.846 | 0.986 | 0.807 | -0.077 | 0.250 | 0.8914 | 0.990 | 0.0073 | 0.710 |
| `ensemble` | `balanced_rf` | best_f1 | 0.7454 | 0.860 | 0.537 | 0.662 | 0.828 | 0.988 | 0.581 | 0.317 | -0.325 | 0.6970 | 0.967 | 0.0907 | 0.675 |
| `ensemble` | `balanced_rf` | fixed_0.5 | 0.5000 | 0.216 | 0.863 | 0.345 | 0.655 | 0.953 | 0.539 | 0.000 | 0.000 | 0.6970 | 0.967 | 0.0907 | 0.412 |
| `ensemble` | `balanced_rf` | best_cost | 0.6250 | 0.466 | 0.675 | 0.551 | 0.770 | 0.980 | 0.619 | 0.206 | -0.188 | 0.6970 | 0.967 | 0.0907 | 0.550 |
| `ensemble` | `balanced_rf` | min_precision | 0.6423 | 0.525 | 0.650 | 0.581 | 0.786 | 0.982 | 0.621 | 0.236 | -0.213 | 0.6970 | 0.967 | 0.0907 | 0.575 |
| `ensemble` | `easy_ensemble` | best_f1 | 0.5755 | 0.564 | 0.388 | 0.459 | 0.725 | 0.980 | 0.413 | 0.266 | -0.450 | 0.4757 | 0.901 | 0.2041 | 0.458 |
| `ensemble` | `easy_ensemble` | fixed_0.5 | 0.5000 | 0.109 | 0.838 | 0.194 | 0.559 | 0.909 | 0.359 | 0.000 | 0.000 | 0.4757 | 0.901 | 0.2041 | 0.272 |
| `ensemble` | `easy_ensemble` | best_cost | 0.5538 | 0.386 | 0.487 | 0.431 | 0.709 | 0.976 | 0.463 | 0.237 | -0.350 | 0.4757 | 0.901 | 0.2041 | 0.421 |
| `ensemble` | `easy_ensemble` | min_precision | 0.5739 | 0.544 | 0.388 | 0.453 | 0.722 | 0.980 | 0.411 | 0.259 | -0.450 | 0.4757 | 0.901 | 0.2041 | 0.450 |
| `ensemble` | `rusboost` | best_f1 | 0.8808 | 0.157 | 0.138 | 0.147 | 0.565 | 0.967 | 0.141 | 0.019 | -0.212 | 0.0735 | 0.723 | 0.0773 | 0.131 |
| `ensemble` | `rusboost` | fixed_0.5 | 0.5000 | 0.078 | 0.350 | 0.127 | 0.538 | 0.933 | 0.206 | 0.000 | 0.000 | 0.0735 | 0.723 | 0.0773 | 0.130 |
| `ensemble` | `rusboost` | best_cost | 0.6493 | 0.142 | 0.300 | 0.193 | 0.583 | 0.958 | 0.245 | 0.065 | -0.050 | 0.0735 | 0.723 | 0.0773 | 0.183 |
| `ensemble` | `rusboost` | min_precision | 0.8808 | 0.000 | 0.000 | 0.000 | 0.495 | 0.970 | 0.000 | -0.127 | -0.350 | 0.0735 | 0.723 | 0.0773 | 0.000 |

## 3. Bảng so sánh BASELINE (chưa xử lý) vs các kỹ thuật xử lý (yêu cầu #3)

| Kỹ thuật | Nhóm | PR-AUC | F1 | F1-macro | F1-weighted | F-beta(2.0) | ROC-AUC | MCC | TN/FP/FN/TP | ΔPR-AUC | ΔF1 | ΔF1-macro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `baseline` | `baseline` | 0.8891 | 0.834 | 0.916 | 0.994 | 0.806 | 0.991 | 0.833 | 3912/8/17/63 | 0.000 | 0.000 | 0.000 |
| `ros` | `data-level/oversampling` | 0.8909 | 0.837 | 0.917 | 0.994 | 0.814 | 0.991 | 0.834 | 3911/9/16/64 | 0.002 | 0.002 | 0.001 |
| `smote` | `data-level/oversampling` | 0.8703 | 0.808 | 0.902 | 0.993 | 0.764 | 0.989 | 0.809 | 3913/7/21/59 | -0.019 | -0.026 | -0.013 |
| `borderline_smote` | `data-level/oversampling` | 0.8734 | 0.827 | 0.912 | 0.993 | 0.795 | 0.987 | 0.825 | 3912/8/18/62 | -0.016 | -0.008 | -0.004 |
| `adasyn` | `data-level/oversampling` | 0.8700 | 0.755 | 0.875 | 0.991 | 0.705 | 0.991 | 0.756 | 3911/9/26/54 | -0.019 | -0.079 | -0.040 |
| `rus` | `data-level/undersampling` | 0.7637 | 0.671 | 0.832 | 0.987 | 0.643 | 0.984 | 0.667 | 3901/19/30/50 | -0.125 | -0.163 | -0.083 |
| `tomek` | `data-level/undersampling` | 0.8956 | 0.840 | 0.918 | 0.994 | 0.808 | 0.992 | 0.839 | 3913/7/17/63 | 0.007 | 0.006 | 0.003 |
| `enn` | `data-level/undersampling` | 0.8888 | 0.827 | 0.912 | 0.993 | 0.795 | 0.991 | 0.825 | 3912/8/18/62 | -0.000 | -0.008 | -0.004 |
| `smote_tomek` | `hybrid` | 0.8703 | 0.808 | 0.902 | 0.993 | 0.764 | 0.989 | 0.809 | 3913/7/21/59 | -0.019 | -0.026 | -0.013 |
| `smote_enn` | `hybrid` | 0.8636 | 0.792 | 0.894 | 0.992 | 0.758 | 0.990 | 0.790 | 3910/10/21/59 | -0.025 | -0.042 | -0.022 |
| `cost_sensitive_scale_pos_weight` | `algorithm-level` | 0.8919 | 0.829 | 0.913 | 0.993 | 0.804 | 0.988 | 0.827 | 3911/9/17/63 | 0.003 | -0.005 | -0.003 |
| `cost_sensitive_class_weight` | `algorithm-level` | 0.8950 | 0.828 | 0.912 | 0.993 | 0.819 | 0.992 | 0.825 | 3908/12/15/65 | 0.006 | -0.006 | -0.003 |
| `focal_loss` | `algorithm-level` | 0.8914 | 0.818 | 0.907 | 0.993 | 0.815 | 0.990 | 0.814 | 3906/14/15/65 | 0.002 | -0.017 | -0.009 |
| `balanced_rf` | `ensemble` | 0.6970 | 0.662 | 0.828 | 0.988 | 0.581 | 0.967 | 0.675 | 3913/7/37/43 | -0.192 | -0.173 | -0.088 |
| `easy_ensemble` | `ensemble` | 0.4757 | 0.459 | 0.725 | 0.980 | 0.413 | 0.901 | 0.458 | 3896/24/49/31 | -0.413 | -0.375 | -0.191 |
| `rusboost` | `ensemble` | 0.0735 | 0.147 | 0.565 | 0.967 | 0.141 | 0.723 | 0.131 | 3861/59/69/11 | -0.816 | -0.688 | -0.350 |

- Ngưỡng chấm điểm: `best_f1` (chọn trên đường PR của xác suất out-of-fold). Baseline: **PR-AUC 0.8891, F1 0.834**.
- Số kỹ thuật có PR-AUC CAO HƠN baseline: **5/15**; cao nhất ở `tomek` (0.8956).
- Bảng KHÔNG có Accuracy: ở tỉ lệ 98/2, đoán 'lớp đa số' đã đạt ~98% ⇒ accuracy chỉ là chỉ số chẩn đoán (`metrics.accuracy_diagnostic`).

**Chỉ số CHẨN ĐOÁN (không dùng để kết luận):**
- Mốc "luôn đoán lớp đa số" = **98.00% accuracy**; accuracy của các kỹ thuật nằm trong 96.80%–99.40%.
- ⇒ Chênh lệch accuracy giữa các kỹ thuật ở đây KHÔNG chứng minh kỹ thuật nào tốt hơn; phải đọc Precision/Recall/F1(−macro/−weighted/−beta), PR-AUC, ROC-AUC và Confusion Matrix.

## 4. Ngưỡng chọn trên xác suất out-of-fold + PR-AUC (OOF)

| Kỹ thuật | PR-AUC (OOF) | best_f1 | best_cost | min_precision | 0.5 | IR train (trung bình) | % dương fold-val |
|---|---:|---:|---:|---:|---:|---|---|
| `baseline` | 0.8372 | 0.0768 | 0.0127 | 0.0065 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |
| `ros` | 0.8397 | 0.2408 | 0.0396 | 0.0162 | 0.5000 | 49.0 → 2.0 | 2.00–2.00% |
| `smote` | 0.8264 | 0.7417 | 0.0877 | 0.0796 | 0.5000 | 49.0 → 2.0 | 2.00–2.00% |
| `borderline_smote` | 0.8196 | 0.5949 | 0.0462 | 0.0462 | 0.5000 | 49.0 → 2.0 | 2.00–2.00% |
| `adasyn` | 0.8177 | 0.7870 | 0.1573 | 0.1206 | 0.5000 | 49.0 → 2.0 | 2.00–2.00% |
| `rus` | 0.7563 | 0.9447 | 0.6092 | 0.7470 | 0.5000 | 49.0 → 2.0 | 2.00–2.00% |
| `tomek` | 0.8379 | 0.0799 | 0.0207 | 0.0071 | 0.5000 | 49.0 → 48.9 | 2.00–2.00% |
| `enn` | 0.8312 | 0.1043 | 0.0072 | 0.0073 | 0.5000 | 49.0 → 48.6 | 2.00–2.00% |
| `smote_tomek` | 0.8264 | 0.7417 | 0.0877 | 0.0796 | 0.5000 | 49.0 → 2.0 | 2.00–2.00% |
| `smote_enn` | 0.8235 | 0.8038 | 0.0984 | 0.1405 | 0.5000 | 49.0 → 1.9 | 2.00–2.00% |
| `cost_sensitive_scale_pos_weight` | 0.8270 | 0.3591 | 0.0537 | 0.0198 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |
| `cost_sensitive_class_weight` | 0.8350 | 0.2912 | 0.0522 | 0.0275 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |
| `focal_loss` | 0.8386 | 0.3038 | 0.2148 | 0.1591 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |
| `balanced_rf` | 0.6984 | 0.7454 | 0.6250 | 0.6423 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |
| `easy_ensemble` | 0.4747 | 0.5755 | 0.5538 | 0.5739 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |
| `rusboost` | 0.1103 | 0.8808 | 0.6493 | 0.8808 | 0.5000 | 49.0 → 49.0 | 2.00–2.00% |

## 5. Kiểm chứng chống rò rỉ dữ liệu

- PASS — `baseline` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `ros` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `smote` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `borderline_smote` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `adasyn` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `rus` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `tomek` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `enn` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `smote_tomek` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `smote_enn` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `cost_sensitive_scale_pos_weight` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `cost_sensitive_class_weight` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `focal_loss` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `balanced_rf` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `easy_ensemble` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)
- PASS — `rusboost` (fold_val_nguyên_vẹn: PASS, test_nguyên_vẹn: PASS, ngưỡng_chọn_trên_oof: PASS)

## 6. Kết luận & khuyến nghị

**So với BASELINE (chưa xử lý)**: baseline PR-AUC 0.8891 / F1 0.834; có **5/15** kỹ thuật vượt baseline về PR-AUC, cao nhất là `tomek` (ΔPR-AUC +0.0065).
1. **Thứ hạng xác suất (PR-AUC out-of-fold)** tốt nhất là `ros` (0.8397). PR-AUC không phụ thuộc ngưỡng nên đây là so sánh 'công bằng' nhất giữa các kỹ thuật.
2. **Tinh chỉnh ngưỡng theo đường PR thay cho 0.5**: F1 thay đổi trung bình **+0.067**; tốt nhất ở `balanced_rf` (+0.317); riêng 2 kỹ thuật GIẢM (`adasyn`, `cost_sensitive_class_weight`) — ngưỡng tốt nhất không phải lúc nào cũng cao hơn 0.5.
3. **Tomek Links / ENN là kỹ thuật LÀM SẠCH biên, không phải cân bằng tỉ lệ** (`tomek`: IR 49.0 → 48.9; `enn`: IR 49.0 → 48.6): chúng chỉ bỏ mẫu sát biên nên IR gần như giữ nguyên ở 98/2 ⇒ phải dùng kèm oversampling (nhóm hybrid) mới có tác dụng cân bằng.
4. **Algorithm-level** (không đổi dữ liệu): tốt nhất theo F1 là `cost_sensitive_scale_pos_weight` — F1=0.829, recall=0.787, PR-AUC=0.8919. Focal Loss dùng custom objective (`gamma`, `alpha`) nên đổi cả HÌNH DẠNG hàm mất mát, không chỉ trọng số lớp.
5. **Ensemble**: tốt nhất theo F1 là `balanced_rf` — F1=0.662, PR-AUC=0.6970. Undersampling nằm bên trong `fit` nên vẫn chỉ chạm train của fold.
6. **Khuyến nghị cho pipeline chính**: chọn kỹ thuật theo PR-AUC, chọn ngưỡng trên đường PR của xác suất out-of-fold (không dùng 0.5 mặc định, không chọn trên test), và ưu tiên can thiệp KHÔNG sinh mẫu (`class_weight`/`scale_pos_weight`/Focal Loss) khi nhãn gắn với thực thể — mẫu tổng hợp dễ rơi vào 'vùng' của chính thực thể đã có trong train.

## 7. Tái lập

```powershell
python -m pip install -r imbalance_lab/requirements.txt
python -m imbalance_lab.techniques            # toàn bộ danh mục (ghi artifact)
python -m imbalance_lab.techniques --quick --techniques smote,adasyn,focal_loss
python -m unittest discover -s tests -v       # gồm test danh mục + chống rò rỉ
```
