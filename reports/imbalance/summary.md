# Lab mất cân bằng lớp — pipeline không rò rỉ dữ liệu

- Backend phân loại: **lightgbm** · resampling: **imblearn** · seed 42 · n_samples 50000
- Phân phối nhãn: toàn bộ n=50000 | âm=49000 | dương=1000 (2.00%) | IR=49.0; train_pool n=40000 | âm=39200 | dương=800 (2.00%) | IR=49.0; test n=10000 | âm=9800 | dương=200 (2.00%) | IR=49.0
- Quy tắc chống rò rỉ: StratifiedKFold trên train_pool; SMOTE+undersample chỉ nằm trong pipeline (`imblearn.pipeline.Pipeline`) nên chỉ chạy trên train của từng fold; ngưỡng chọn trên xác suất out-of-fold; test chỉ dùng một lần.

## Metric trên holdout test (n = 10000)

| Chiến lược | Ngưỡng | thr | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | MCC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | best_f1 | 0.0930 | 0.926 | 0.870 | 0.897 | 0.950 | 0.997 | 0.004 | 0.895 | 14 | 26 |
| `baseline` | best_cost | 0.0298 | 0.835 | 0.910 | 0.871 | 0.950 | 0.997 | 0.004 | 0.869 | 36 | 18 |
| `baseline` | min_precision | 0.0027 | 0.495 | 0.970 | 0.655 | 0.950 | 0.997 | 0.004 | 0.685 | 198 | 6 |
| `baseline` | fixed_0.5 | 0.5000 | 0.980 | 0.750 | 0.850 | 0.950 | 0.997 | 0.004 | 0.855 | 3 | 50 |
| `cost_sensitive` | best_f1 | 0.3869 | 0.956 | 0.875 | 0.914 | 0.944 | 0.996 | 0.003 | 0.913 | 8 | 25 |
| `cost_sensitive` | best_cost | 0.0765 | 0.768 | 0.910 | 0.833 | 0.944 | 0.996 | 0.003 | 0.832 | 55 | 18 |
| `cost_sensitive` | min_precision | 0.0091 | 0.496 | 0.955 | 0.653 | 0.944 | 0.996 | 0.003 | 0.681 | 194 | 9 |
| `cost_sensitive` | fixed_0.5 | 0.5000 | 0.966 | 0.865 | 0.913 | 0.944 | 0.996 | 0.003 | 0.913 | 6 | 27 |
| `resampling` | best_f1 | 0.8899 | 0.948 | 0.820 | 0.879 | 0.940 | 0.997 | 0.005 | 0.879 | 9 | 36 |
| `resampling` | best_cost | 0.5000 | 0.784 | 0.910 | 0.843 | 0.940 | 0.997 | 0.005 | 0.842 | 50 | 18 |
| `resampling` | min_precision | 0.0847 | 0.516 | 0.965 | 0.672 | 0.940 | 0.997 | 0.005 | 0.698 | 181 | 7 |
| `resampling` | fixed_0.5 | 0.5000 | 0.784 | 0.910 | 0.843 | 0.940 | 0.997 | 0.005 | 0.842 | 50 | 18 |
| `resampling_calibrated` | best_f1 | 0.4746 | 0.936 | 0.810 | 0.869 | 0.941 | 0.998 | 0.004 | 0.869 | 11 | 38 |
| `resampling_calibrated` | best_cost | 0.1552 | 0.764 | 0.905 | 0.828 | 0.941 | 0.998 | 0.004 | 0.828 | 56 | 19 |
| `resampling_calibrated` | min_precision | 0.0307 | 0.551 | 0.965 | 0.702 | 0.941 | 0.998 | 0.004 | 0.723 | 157 | 7 |
| `resampling_calibrated` | fixed_0.5 | 0.5000 | 0.946 | 0.795 | 0.864 | 0.941 | 0.998 | 0.004 | 0.865 | 9 | 41 |

## Xác suất out-of-fold (chỉ để chọn ngưỡng, không phải kết quả chốt)

| Chiến lược | thr | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.5000 | 0.995 | 0.802 | 0.889 | 0.961 | 0.997 | 0.892 |
| `cost_sensitive` | 0.5000 | 0.960 | 0.881 | 0.919 | 0.962 | 0.997 | 0.918 |
| `resampling` | 0.5000 | 0.794 | 0.932 | 0.858 | 0.949 | 0.997 | 0.858 |
| `resampling_calibrated` | 0.5000 | 0.952 | 0.848 | 0.897 | 0.948 | 0.997 | 0.896 |

## Ngưỡng chọn trên OOF

| Chiến lược | best_f1 | best_cost | min_precision | 0.5 |
|---|---:|---:|---:|---:|
| `baseline` | 0.0930 | 0.0298 | 0.0027 | 0.5000 |
| `cost_sensitive` | 0.3869 | 0.0765 | 0.0091 | 0.5000 |
| `resampling` | 0.8899 | 0.5000 | 0.0847 | 0.5000 |
| `resampling_calibrated` | 0.4746 | 0.1552 | 0.0307 | 0.5000 |

## Phân phối nhãn theo fold (trước → sau resampling)

| Chiến lược | Fold | Train âm/dương TRƯỚC | IR trước | Train âm/dương SAU | IR sau | Val dương |
|---|---:|---|---:|---|---:|---:|
| `baseline` | 1 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `baseline` | 2 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `baseline` | 3 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `baseline` | 4 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `baseline` | 5 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `cost_sensitive` | 1 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `cost_sensitive` | 2 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `cost_sensitive` | 3 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `cost_sensitive` | 4 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `cost_sensitive` | 5 | 31360/640 | 49.0 | 31360/640 | 49.0 | 160 (2.00%) |
| `resampling` | 1 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling` | 2 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling` | 3 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling` | 4 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling` | 5 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling_calibrated` | 1 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling_calibrated` | 2 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling_calibrated` | 3 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling_calibrated` | 4 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |
| `resampling_calibrated` | 5 | 31360/640 | 49.0 | 6272/3136 | 2.0 | 160 (2.00%) |

## Mốc MINH HOẠ SAI (resample trước khi chia train/test) — chỉ để so sánh

| Ngưỡng | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| best_f1 | 0.992 | 0.991 | 0.991 | 1.000 | 1.000 |
| best_cost | 0.971 | 1.000 | 0.985 | 1.000 | 1.000 |
| min_precision | 0.501 | 1.000 | 0.667 | 1.000 | 1.000 |
| fixed_0.5 | 0.986 | 0.993 | 0.989 | 1.000 | 1.000 |

- Tập test của mốc SAI chứa **786/2940 mẫu tổng hợp** không tồn tại trong dữ liệu gốc ⇒ metric bị thổi phồng, không dùng để kết luận.

## Kết luận & khuyến nghị

1. **PR-AUC cao nhất trên test**: `baseline` = **0.950**; ba chiến lược chênh nhau rất ít (PR-AUC 0.940–0.950) ⇒ với LightGBM, xử lý mất cân bằng chủ yếu đổi **điểm vận hành** (precision/recall) chứ không đổi nhiều thứ hạng xác suất (PR-AUC/ROC-AUC gần như trùng nhau).
2. **F1 cao nhất**: `cost_sensitive` = **0.914** (precision 0.956, recall 0.875) tại ngưỡng best_f1 = 0.3869.
3. **Tinh chỉnh ngưỡng quan trọng nhất ở `baseline`**: F1 0.850 (cố định 0.5) → **0.897** (ngưỡng 0.0930), ΔF1 = +0.047. Ngưỡng tối ưu giữa các chiến lược lệch nhau rất nhiều (`baseline` 0.093, `cost_sensitive` 0.387, `resampling` 0.890, `resampling_calibrated` 0.475) ⇒ đổi cách xử lý mất cân bằng thì PHẢI chọn lại ngưỡng, không giữ 0.5.
4. **Không dùng Accuracy**: tỉ lệ dương ở test là 2.00% ⇒ quy tắc “đoán toàn lớp đa số” đã đạt 98.00% accuracy; thước đo chính là PR-AUC, Recall và F1 tại ngưỡng nghiệp vụ.
5. **Resampling chỉ chạy trên train của fold**: IR của train mỗi fold 49.0 → **2.0** (xem `resampling_by_fold.csv`), còn validation/test giữ nguyên tỉ lệ 98/2 — đúng nguyên tắc chống rò rỉ (được kiểm chứng tự động, xem `tests/test_imbalance_lab.py`).
6. **Mốc minh hoạ SAI** (resample trước khi chia train/test): PR-AUC 1.000 — cao hơn hẳn mọi chiến lược hợp lệ — trong khi tập test chứa 786/2940 mẫu tổng hợp nội suy từ train ⇒ bằng chứng định lượng cho việc resampling phải nằm trong pipeline.
7. **Hiệu chuẩn xác suất sau resampling**: Brier 0.0051 → **0.0038**; ngưỡng best-F1 0.890 → 0.475 (PR-AUC 0.940 → 0.941 — hiệu chuẩn không đổi thứ hạng, chỉ đưa xác suất về đúng tần suất thực tế, nhờ đó ngưỡng mới đọc được theo nghĩa “xác suất” và dùng chung giữa các chiến lược).
8. **Khuyến nghị**: (a) metric PR-AUC/F1/Recall thay vì Accuracy; (b) `scale_pos_weight` tính động trong `fit` (cost-sensitive) cho F1 tốt nhất mà không đổi dữ liệu; (c) SMOTE + undersample bên trong pipeline khi cần mô hình thấy nhiều mẫu dương — nhưng phải HIỆU CHUẨN lại xác suất và chọn lại ngưỡng; (d) chọn ngưỡng theo chi phí thực tế (`COST_FN`/`COST_FP`) — ở đây FN đắt gấp 10 lần FP nên ngưỡng theo chi phí luôn thấp hơn ngưỡng best-F1.

## Tái lập

```powershell
python -m pip install -r requirements-labs.txt
python -m labs.imbalance_lab.run        # log ở reports/imbalance/run.log
python -m unittest discover -s tests -v   # gồm test chống rò rỉ của lab
```
