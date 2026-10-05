# `labs.imbalance_lab` — pipeline xử lý dữ liệu mất cân bằng, KHÔNG rò rỉ dữ liệu

Lab độc lập (không thuộc `scripts.run_all`) gồm **hai phần**:

1. `python -m labs.imbalance_lab.run` — so sánh 3 chiến lược + 1 biến thể hiệu chuẩn trên dữ liệu giả lập
   **98% / 2%**, kèm mốc MINH HOẠ SAI (resample trước khi chia) để đo mức lạc quan hoá.
2. `python -m labs.imbalance_lab.techniques` — **danh mục đầy đủ 5 nhóm kỹ thuật** theo yêu cầu #2
   (data-level oversampling/undersampling, hybrid, algorithm-level gồm Focal Loss, ensemble) và
   **threshold tuning trên đường Precision-Recall**. Chi tiết: mục 7 dưới đây.

Yêu cầu bắt buộc cho cả hai: mọi bước resampling chỉ chạy trên tập train của từng fold.

## 1. Cài đặt & chạy

```powershell
python -m pip install -r requirements-labs.txt
python -m labs.imbalance_lab.run                        # 50.000 mẫu, 5 fold, có mốc minh hoạ rò rỉ
python -m labs.imbalance_lab.run --n-samples 10000 --skip-leaky
python -m labs.imbalance_lab.run --no-imblearn          # dùng sampler nội bộ (khi offline)
python -m labs.imbalance_lab.techniques                 # danh mục 15 kỹ thuật × 5 fold (yêu cầu #2)
python -m labs.imbalance_lab.techniques --quick         # ensemble ít estimator hơn (chạy nhanh)
python -m unittest discover -s tests -v            # gồm test chống rò rỉ của cả hai phần
```

Thư viện: `scikit-learn`, `imbalanced-learn`, `lightgbm` (+ `numpy`, `scipy`). Nếu thiếu LightGBM,
lab tự hạ cấp sang XGBoost rồi `HistGradientBoosting` và ghi rõ backend đã dùng trong log; riêng
Focal Loss cần LightGBM (nếu thiếu, kỹ thuật đó bị BỎ QUA kèm lý do trong artifact).

## 2. Đầu ra (mặc định `reports/imbalance/`)

| File | Nội dung |
|---|---|
| `run.log` | Toàn bộ log, **gồm phân phối nhãn TRƯỚC/SAU resampling của từng fold** |
| `summary.md` | Bảng metric test theo chiến lược × ngưỡng, ngưỡng đã chọn, log fold, kết luận |
| `summary.csv` | Metric trên holdout test + xác suất out-of-fold |
| `metrics_by_fold.csv` | Metric từng fold tại ngưỡng 0.5 |
| `resampling_by_fold.csv` | Số mẫu âm/dương và IR trước/sau resampling mỗi fold |
| `thresholds.json` | Ngưỡng `best_f1` / `best_cost` / `min_precision` chọn trên OOF |

## 3. Ba chiến lược

| Chiến lược | Can thiệp | Ghi chú |
|---|---|---|
| `baseline` | không | LightGBM mặc định — mốc so sánh |
| `cost_sensitive` | hàm mất mát | `scale_pos_weight = n_âm/n_dương`, tính **trong `fit`** (chỉ từ nhãn fold train) |
| `resampling` | dữ liệu | `SMOTE(sampling_strategy=0.1)` → `RandomUnderSampler(sampling_strategy=0.5)` → LightGBM, tất cả trong `imblearn.pipeline.Pipeline` |
| `resampling_calibrated` | dữ liệu + xác suất | như trên, bọc thêm `CalibratedClassifierCV(method="isotonic", cv=5)` vì resampling làm lệch prior (ngưỡng tối ưu ~0,89 thay vì ~0,09) |

Ngoài ra có mốc **`leaky_resample_before_split`** — cố tình làm SAI (resample trước khi chia
train/test) để đo mức lạc quan hoá; lab còn đếm số mẫu test là mẫu tổng hợp.

## 4. Quy tắc chống rò rỉ (được code và test thực thi)

1. `StratifiedKFold` trên `train_pool`; holdout test tách trước bằng `train_test_split(stratify=y)` và
   **chỉ dùng một lần** để chốt kết quả.
2. Resampling nằm TRONG `imblearn.pipeline.Pipeline` ⇒ chỉ `fit_resample` trên train của fold.
   `cv.assert_val_untouched()` so `X_va/y_va` trước–sau để phát hiện thay đổi ngoài ý muốn.
3. Ngưỡng quyết định chọn trên xác suất **out-of-fold** của `train_pool`, không bao giờ trên test.
4. `scale_pos_weight` tính trong `fit` ⇒ tự động đúng theo từng fold (không dùng hằng số toàn cục).
5. Log phân phối "sau resampling" lấy từ một pipeline **thứ hai** chỉ có sampler, nên không ảnh
   hưởng mô hình đang huấn luyện.

Test tương ứng: `tests/test_imbalance_lab.py::TestNoLeakage` (classifier ghi lại số mẫu nó nhận,
validation phải nguyên vẹn từng byte) và `TestThresholds` (hàm ngưỡng khớp brute-force/sklearn).

## 5. Đánh giá (yêu cầu #3)

- **Accuracy KHÔNG phải thước đo chính** (`metrics.PRIMARY_METRICS` không chứa accuracy): đoán lớp đa số
  ở tỉ lệ 98/2 đã đạt ~98%. Accuracy chỉ được in như chỉ số CHẨN ĐOÁN kèm mốc so sánh
  (`metrics.accuracy_diagnostic` → `majority_baseline_accuracy_pct`, cờ `accuracy_better_than_majority`).
- Bộ metric chính: **Precision, Recall, F1 (binary / macro / weighted / F-beta)**, **PR-AUC (Average
  Precision)**, ROC-AUC, MCC — kèm **Confusion Matrix** (TN/FP/FN/TP) và balanced accuracy; Brier để
  đọc hiệu chuẩn. Có khoảng tin cậy bootstrap cho `pr_auc`, `f1`, `macro_f1`, `fbeta`, `recall`
  (`metrics.metric_scalar` + `bootstrap_ci`).
- **Tinh chỉnh ngưỡng** thay vì cố định 0.5: chọn trên lưới lượng tử *hoặc* trên **đường Precision-Recall**
  (`thresholds.tune_thresholds_from_pr_curve`), 3 chế độ `best_f1` / `best_cost`
  (tối thiểu `FN·COST_FN + FP·COST_FP`) / `min_precision`.
- **Bảng so sánh Baseline (chưa xử lý) vs kỹ thuật xử lý**: `techniques.compare_with_baseline`,
  `techniques.comparison_markdown` → `techniques_comparison.csv` + mục 3 của `techniques.md`
  (cột ΔPR-AUC, ΔF1, ΔF1-macro so với baseline).
- **Resampling chỉ chạy trong `imblearn.pipeline.Pipeline`** (`samplers.build_sampler_pipeline`), không
  dùng pipeline chuẩn của scikit-learn cho bước lấy mẫu ⇒ khi vào Cross-Validation `fit_resample` chỉ
  chạy trên train của từng fold (test `tests/test_evaluation_metrics.py::TestImblearnPipeline`).

## 6. Cấu trúc mã

```
labs/imbalance_lab/
  config.py     # tham số duy nhất: 98/2, 5 fold, SMOTE 0.1 → under 0.5, chi phí FN/FP, tham số danh mục
  data.py       # make_classification + chia tập stratified + thống kê phân phối nhãn
  samplers.py   # imblearn (đường chính) + bản nội bộ: SMOTE/RandomOverSampler/BorderlineSMOTE/ADASYN/
                # RandomUnderSampler/TomekLinks/ENN/SamplerChain/Pipeline
  models.py     # 3 chiến lược + ScalePosWeightClassifier + BalancedWeightClassifier + mốc minh hoạ SAI
  losses.py     # Focal Loss: grad/hess giải tích + FocalLossClassifier (custom objective LightGBM)
  metrics.py    # metric CHÍNH (precision, recall, F1 binary/macro/weighted/F-beta, PR-AUC, ROC-AUC, MCC)
                # + Confusion Matrix + bootstrap CI; accuracy chỉ để CHẨN ĐOÁN (kèm mốc lớp đa số)
  thresholds.py # quét ngưỡng vector hoá + chọn ngưỡng TRÊN ĐƯỜNG PR (precision_recall_curve)
  cv.py         # StratifiedKFold, log resampling theo fold, assert chống rò rỉ, refit + chốt test
  run.py        # điều phối 4 chiến lược, log, ghi artifact (reports/imbalance/summary.*)
  techniques.py # DANH MỤC 5 nhóm kỹ thuật (yêu cầu #2) + bảng BASELINE vs kỹ thuật (yêu cầu #3) +
                # runner + artifact (reports/imbalance/techniques.*)
```

Xem kết quả đo mới nhất trong `reports/imbalance/summary.md` và
`reports/imbalance/techniques.md`.

## 7. Danh mục kỹ thuật (yêu cầu #2) — `python -m labs.imbalance_lab.techniques`

| Nhóm | Kỹ thuật | Khoá |
|---|---|---|
| **Baseline (chưa xử lý — yêu cầu #3)** | boosting mặc định, không can thiệp | `baseline` |
| Data-level / Oversampling | RandomOverSampler · SMOTE · BorderlineSMOTE · ADASYN | `ros`, `smote`, `borderline_smote`, `adasyn` |
| Data-level / Undersampling | RandomUnderSampler · Tomek Links · ENN | `rus`, `tomek`, `enn` |
| Hybrid | SMOTE + Tomek Links · SMOTE + ENN | `smote_tomek`, `smote_enn` |
| Algorithm-level | `scale_pos_weight` động · `class_weight='balanced'` · **Focal Loss** | `cost_sensitive_scale_pos_weight`, `cost_sensitive_class_weight`, `focal_loss` |
| Ensemble | BalancedRandomForest · EasyEnsemble · RUSBoost | `balanced_rf`, `easy_ensemble`, `rusboost` |
| Threshold tuning | `best_f1` (PR) · `best_cost` · `min_precision`, so với mốc 0.5 | áp cho MỌI kỹ thuật |

Đầu ra (`reports/imbalance/`): `techniques.md` (bảng đầy đủ + **bảng so sánh Baseline vs kỹ thuật** +
kết luận tự động), `techniques.csv` (metric trên test theo kỹ thuật × 4 chế độ ngưỡng),
**`techniques_comparison.csv`** (Baseline vs kỹ thuật: metric chính + ΔPR-AUC/ΔF1/ΔF1-macro),
`techniques.json`, `techniques_by_fold.csv` (IR train trước→sau từng fold, % dương fold-val),
`techniques.log`.

Tùy chọn: `--n-samples`, `--cv N`, `--techniques a,b,c`, `--quick`, `--no-imblearn`, `--no-write`.

Ba phát hiện định lượng (sinh tự động trong mục "Kết luận" của `techniques.md`):

1. **Tomek Links / ENN là `clean-sampling`, không phải cân bằng**: chúng bỏ mẫu sát biên nên số mẫu
   thiểu số không đổi và IR gần như giữ nguyên ở 98/2 ⇒ chỉ hữu ích khi ghép với oversampling (nhóm
   hybrid). Test `test_cleaning_techniques_keep_minority_untouched` khoá hành vi này.
2. **Tinh chỉnh ngưỡng theo đường PR thay cho 0.5 đổi điểm vận hành, không đổi thứ hạng**: PR-AUC
   giữ nguyên giữa các chế độ ngưỡng (ngưỡng không nằm trong công thức PR-AUC); F1 thay đổi theo từng
   kỹ thuật, ví dụ `balanced_rf`/`easy_ensemble` được lợi nhiều vì xác suất hiệu chuẩn kém.
3. **Focal Loss** (custom objective của LightGBM, grad/hess giải tích có test sai phân) là can thiệp
   algorithm-level mạnh nhất trong danh mục ở ví dụ này; `scale_pos_weight` và `class_weight='balanced'`
   cho kết quả gần nhau (cùng họ trọng số lớp).

Mọi kỹ thuật đều được kiểm chứng chống rò rỉ trong artifact (mục "Kiểm chứng chống rò rỉ dữ liệu"):
fold-validation nguyên vẹn, test nguyên vẹn, ngưỡng chọn trên OOF — PASS/FAIL in rõ từng kỹ thuật.
