# Các nhóm kỹ thuật xử lý mất cân bằng (yêu cầu #2) — cài đặt, kiểm thử, kết quả

*Tài liệu viết tay (không sinh tự động). Mọi con số dưới đây trích từ artifact sinh bằng một lệnh:
`reports/imbalance/techniques.md` (+ `.csv`, `.json`, `.log`) — chạy lại được bằng
`python -m labs.imbalance_lab.techniques`.*

> **Phạm vi:** LightGBM/XGBoost trong lab này chỉ là **base learner cho các kỹ thuật mất cân bằng**
> (ví dụ Focal Loss cần custom objective của LightGBM), **KHÔNG thuộc bộ mô hình của đồ án**. Bộ mô hình
> chính thức gồm đúng **3 họ thuần scikit-learn**: Logistic Regression · Random Forest ·
> HistGradientBoosting (`forecasting/models.py`).

## 1. Bảng đối chiếu: yêu cầu ↔ cài đặt ↔ kiểm thử

| Mục trong yêu cầu #2 | Khoá | Cài đặt | Kiểm thử tự động |
|---|---|---|---|
| **(tham chiếu — yêu cầu #3) Baseline: chưa xử lý mất cân bằng** | `baseline` | `models.make_base_classifier` (boosting mặc định, không resampling/không trọng số lớp) | `test_baseline_is_present_and_runs_first`, `TestBaselineComparison` |
| Oversampling: Random Oversampling | `ros` | `labs/imbalance_lab/samplers.py::RandomOverSampler` (nội bộ) · `imblearn.over_sampling.RandomOverSampler` | `test_oversamplers_reach_target_ratio`, `test_samplers_never_mutate_inputs` |
| Oversampling: SMOTE | `smote` | `samplers.SMOTE` · `imblearn SMOTE` | như trên |
| Oversampling: BorderlineSMOTE | `borderline_smote` | `samplers.BorderlineSMOTE` (biến thể borderline-1: chỉ nội suy từ mẫu vùng DANGER) · `imblearn BorderlineSMOTE` | như trên |
| Oversampling: ADASYN | `adasyn` | `samplers.ADASYN` (số mẫu tỉ lệ với độ khó) · `imblearn ADASYN` | như trên |
| Undersampling: Random Undersampling | `rus` | `samplers.RandomUnderSampler` · `imblearn RandomUnderSampler` | `test_undersampler_reaches_target_ratio` |
| Undersampling: Tomek Links | `tomek` | `samplers.TomekLinks` · `imblearn TomekLinks` | `test_cleaning_techniques_keep_minority_untouched` |
| Undersampling: ENN | `enn` | `samplers.EditedNearestNeighbours` (khớp `kind_sel="all"`, `sampling_strategy="auto"` của imblearn) | như trên |
| Hybrid: SMOTE-ENN / SMOTE-Tomek | `smote_enn`, `smote_tomek` | `samplers.make_hybrid_sampler` · `imblearn SMOTEENN` / `SMOTETomek` | `test_hybrid_techniques_balance_then_clean` |
| Cost-sensitive: `scale_pos_weight` | `cost_sensitive_scale_pos_weight` | `models.ScalePosWeightClassifier` — trọng số tính **trong `fit`** | `tests/test_imbalance_lab.py::test_scale_pos_weight_uses_only_given_fold_labels` |
| Cost-sensitive: `class_weight='balanced'` | `cost_sensitive_class_weight` | `models.BalancedWeightClassifier` — `compute_class_weight('balanced')` trên nhãn nhận được | `test_balanced_class_weight_uses_fit_labels` |
| Focal Loss (custom loss) | `focal_loss` | `labs/imbalance_lab/losses.py` — grad/hess giải tích theo logit + `FocalLossClassifier` (LightGBM custom objective) | `test_gradient_matches_finite_difference`, `test_hessian_matches_finite_difference_of_gradient`, `test_gamma_zero_reduces_to_weighted_bce`, `test_hessian_is_strictly_positive` |
| Ensemble: BalancedRandomForest | `balanced_rf` | `imblearn.ensemble.BalancedRandomForestClassifier` | `TestCatalogCoverage`, `TestNoLeakageInCatalog` |
| Ensemble: EasyEnsemble (+ RUSBoost) | `easy_ensemble`, `rusboost` | `imblearn.ensemble.EasyEnsembleClassifier` / `RUSBoostClassifier` | như trên |
| Threshold tuning theo PR (thay 0.5) | áp cho **mọi** kỹ thuật | `labs/imbalance_lab/thresholds.py::tune_thresholds_from_pr_curve` (ứng viên = điểm của `precision_recall_curve`) | `TestPRThresholds` (5 test: PR-AUC khớp sklearn, ngưỡng nằm trên đường cong, F1 = max F1 của đường cong, cost tối thiểu, min-precision đạt mục tiêu) |

Danh mục được khoá bằng test `TestCatalogCoverage::test_required_groups_and_techniques_present`:
nếu ai sửa code và bỏ sót một kỹ thuật trong danh sách yêu cầu, test FAIL ngay.

## 2. Kết quả đo (n = 20.000 mẫu 98/2, 5 fold, seed 42)

Trích `reports/imbalance/techniques.md` (cột ΔF1 = F1 tại ngưỡng PR tốt nhất − F1 tại 0.5):

| Nhóm | Kỹ thuật | PR-AUC (OOF) | thr | F1 @PR | F1 @0.5 | ΔF1 | IR train (trước → sau) |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | `baseline` | 0.8372 | 0.0768 | 0.834 | 0.758 | +0.076 | 49.0 → 49.0 |
| Oversampling | `ros` | 0.8397 | 0.2408 | 0.837 | 0.825 | +0.011 | 49.0 → 2.0 |
| Oversampling | `smote` | 0.8264 | 0.7417 | 0.808 | 0.797 | +0.011 | 49.0 → 2.0 |
| Oversampling | `borderline_smote` | 0.8196 | 0.5949 | 0.827 | 0.810 | +0.016 | 49.0 → 2.0 |
| Oversampling | `adasyn` | 0.8177 | 0.7870 | 0.755 | 0.790 | −0.035 | 49.0 → 2.0 |
| Undersampling | `rus` (RandomUnderSampler) | 0.7563 | 0.9447 | 0.671 | 0.481 | +0.190 | 49.0 → 2.0 |
| Undersampling | `tomek` | 0.8379 | 0.0799 | 0.840 | 0.758 | +0.082 | 49.0 → **48.9** |
| Undersampling | `enn` | 0.8312 | 0.1043 | 0.827 | 0.773 | +0.054 | 49.0 → **48.6** |
| Hybrid | `smote_tomek` | 0.8264 | 0.7417 | 0.808 | 0.797 | +0.011 | 49.0 → 2.0 |
| Hybrid | `smote_enn` | 0.8235 | 0.8038 | 0.792 | 0.780 | +0.011 | 49.0 → 1.9 |
| Algorithm-level | `cost_sensitive_scale_pos_weight` | 0.8270 | 0.3591 | 0.829 | 0.819 | +0.010 | 49.0 → 49.0 |
| Algorithm-level | `cost_sensitive_class_weight` | 0.8350 | 0.2912 | 0.828 | 0.833 | −0.005 | 49.0 → 49.0 |
| Algorithm-level | `focal_loss` | **0.8386** | 0.3038 | 0.818 | 0.776 | +0.042 | 49.0 → 49.0 |
| Ensemble | `balanced_rf` | 0.6984 | 0.7454 | 0.662 | 0.345 | **+0.317** | 49.0 → 49.0 |
| Ensemble | `easy_ensemble` | 0.4747 | 0.5755 | 0.459 | 0.194 | +0.265 | 49.0 → 49.0 |
| Ensemble | `rusboost` | 0.1103 | 0.8808 | 0.147 | 0.127 | +0.020 | 49.0 → 49.0 |

Đọc bảng: **PR-AUC out-of-fold** là cột dùng để chọn kỹ thuật (không phụ thuộc ngưỡng); các cột ngưỡng
chỉ nói về điểm vận hành sau khi đã chọn. Mọi kỹ thuật đều PASS kiểm chứng chống rò rỉ (mục 5 của
`techniques.md`), và % dương của fold-validation giữ đúng 2.00% ở mọi fold.

## 3. Threshold tuning trên đường Precision-Recall (thay vì 0.5)

- `labs/imbalance_lab/thresholds.py::pr_curve_points` gọi đúng `sklearn.metrics.precision_recall_curve` và
  trả PR-AUC = `average_precision_score` (test `test_pr_auc_matches_sklearn`).
- `tune_thresholds_from_pr_curve` lấy **ứng viên ngưỡng chính là các điểm của đường PR** (không phải
  lưới lượng tử nội suy), rồi chọn 3 chế độ: `best_f1` (max F1 trên đường cong), `best_cost`
  (tối thiểu `FN·COST_FN + FP·COST_FP` với `COST_FN=10`, `COST_FP=1`), `min_precision` (ngưỡng nhỏ nhất
  đạt precision mục tiêu 0.5). Mốc `0.5` được tính song song để so sánh.
- **Chống rò rỉ**: ngưỡng chỉ chọn trên xác suất **out-of-fold của `train_pool`**; test chấm đúng một
  lần. Test `test_thresholds_are_recomputed_from_oof_only` suy lại ngưỡng từ OOF và so với giá trị đã
  dùng — phải khớp tới 12 chữ số, nên nếu code lấy ngưỡng từ test thì test đổ.
- Số liệu: tinh chỉnh ngưỡng đổi F1 trung bình **+0.067**; kỹ thuật được lợi nhiều nhất là
  `balanced_rf` (+0.317) vì xác suất của Balanced Random Forest hiệu chuẩn kém (F1@0.5 chỉ 0.345).
  Ngược lại `adasyn` (−0.035) và `cost_sensitive_class_weight` (−0.005) giảm nhẹ — **ngưỡng tốt nhất
  không phải lúc nào cũng cao hơn 0.5**, nên đây là điều phải đo chứ không được giả định.

## 4. Đánh giá mô hình (yêu cầu #3)

- **Accuracy KHÔNG dùng làm thước đo chính**: `labs/imbalance_lab/metrics.py::PRIMARY_METRICS` =
  Precision, Recall, F1 (binary) , F1-macro, F1-weighted, **F-beta** (`config.FBETA_BETA = 2,0`),
  PR-AUC (Average Precision), ROC-AUC, MCC; kèm Confusion Matrix (TN/FP/FN/TP) và balanced accuracy.
  Accuracy chỉ là chỉ số **chẩn đoán** (`DIAGNOSTIC_METRICS`, hàm `accuracy_diagnostic`) và luôn in kèm
  mốc "đoán lớp đa số".
- **Bằng chứng định lượng cho quy tắc đó** (từ `techniques.md`, mục 3.2): ở bộ 98/2, quy tắc đoán lớp
  đa số đã đạt **98,00% accuracy**, còn accuracy của 16 kỹ thuật nằm trong **96,80%–99,40%** —
  `rusboost` có accuracy 96,8% nhưng PR-AUC chỉ **0,0735** và MCC **0,131** (gần như ngẫu nhiên), còn
  `tomek` có 99,4% accuracy với PR-AUC 0,8956. Nếu xếp hạng theo accuracy thì `rusboost` "trông ổn"
  trong khi theo PR-AUC/MCC nó là kỹ thuật tệ nhất ⇒ đúng lý do phải bỏ accuracy khỏi tiêu chí chính.

**Bảng Baseline (chưa xử lý) vs các kỹ thuật** — `compare_with_baseline()` → mục 3 của
`techniques.md` và `reports/imbalance/techniques_comparison.csv` (Δ so với baseline):

| Kỹ thuật | PR-AUC | F1 | F1-macro | F1-weighted | F-beta(2) | ROC-AUC | MCC | TN/FP/FN/TP | ΔPR-AUC | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `baseline` | 0.8891 | 0.834 | 0.916 | 0.994 | 0.806 | 0.991 | 0.833 | 3912/8/17/63 | 0.000 | 0.000 |
| `ros` | 0.8909 | 0.837 | 0.917 | 0.994 | 0.814 | 0.991 | 0.834 | 3911/9/16/64 | +0.002 | +0.002 |
| `tomek` | **0.8956** | **0.840** | **0.918** | 0.994 | 0.808 | 0.992 | **0.839** | 3913/7/17/63 | **+0.007** | +0.006 |
| `cost_sensitive_class_weight` | 0.8950 | 0.828 | 0.912 | 0.993 | 0.819 | 0.992 | 0.825 | 3908/12/15/65 | +0.006 | −0.006 |
| `cost_sensitive_scale_pos_weight` | 0.8919 | 0.829 | 0.913 | 0.993 | 0.804 | 0.988 | 0.827 | 3911/9/17/63 | +0.003 | −0.005 |
| `focal_loss` | 0.8914 | 0.818 | 0.907 | 0.993 | 0.815 | 0.990 | 0.814 | 3906/14/15/65 | +0.002 | −0.017 |
| `borderline_smote` | 0.8734 | 0.827 | 0.912 | 0.993 | 0.795 | 0.987 | 0.825 | 3912/8/18/62 | −0.016 | −0.008 |
| `smote` / `smote_tomek` | 0.8703 | 0.808 | 0.902 | 0.993 | 0.764 | 0.989 | 0.809 | 3913/7/21/59 | −0.019 | −0.026 |
| `adasyn` | 0.8700 | 0.755 | 0.875 | 0.991 | 0.705 | 0.991 | 0.756 | 3911/9/26/54 | −0.019 | −0.079 |
| `smote_enn` | 0.8636 | 0.792 | 0.894 | 0.992 | 0.758 | 0.990 | 0.790 | 3910/10/21/59 | −0.025 | −0.042 |
| `rus` | 0.7637 | 0.671 | 0.832 | 0.987 | 0.643 | 0.984 | 0.667 | 3901/19/30/50 | −0.125 | −0.163 |
| `balanced_rf` | 0.6970 | 0.662 | 0.828 | 0.988 | 0.581 | 0.967 | 0.675 | 3913/7/37/43 | −0.192 | −0.173 |
| `easy_ensemble` | 0.4757 | 0.459 | 0.725 | 0.980 | 0.413 | 0.901 | 0.458 | 3896/24/49/31 | −0.413 | −0.375 |
| `rusboost` | 0.0735 | 0.147 | 0.565 | 0.967 | 0.141 | 0.723 | 0.131 | 3861/59/69/11 | −0.816 | −0.688 |

Đọc bảng: **chỉ 5/15 kỹ thuật vượt baseline về PR-AUC** (cao nhất `tomek` +0,0065), và cả 5 đều vượt
rất nhẹ — trong khi nhiều kỹ thuật làm **giảm** chất lượng (nhóm ensemble giảm mạnh nhất). Đây là bằng
chứng cho khuyến nghị "ưu tiên can thiệp KHÔNG sinh mẫu và luôn so với baseline".
F1-weighted gần như bão hoà (~0,99) ở mọi kỹ thuật vì lớp đa số chiếm 98% ⇒ thêm một lý do phải đọc
F1-macro / F-beta / PR-AUC / MCC song song.


## 5. Focal Loss — cài đặt và kiểm chứng

- Công thức (nhãn `y`, xác suất `p`, logit `z`):
  `FL = -[α·y·(1-p)^γ·ln p + (1-α)·(1-y)·p^γ·ln(1-p)]`.
- LightGBM cần **gradient/Hessian theo logit**; lab dùng công thức giải tích (`losses.focal_grad_hess`)
  và kiểm chứng bằng sai phân số trung tâm cho cả grad lẫn hessian.
- Hai chốt an toàn: `hessian` được kẹp sàn `1e-6` (LightGBM yêu cầu `h > 0` — có test trên cả điểm
  logit cực trị ±30) và xác suất được kẹp `[1e-6, 1-1e-6]` để tránh `log(0)`.
- `γ = 0` phải thoái hoá về **weighted BCE**: test `test_gamma_zero_reduces_to_weighted_bce` so với
  công thức `α(p−1)` / `(1−α)p` và Hessian `α·p(1−p)` — nếu công thức tổng quát sai thì test này đổ.
- `alpha=None` ⇒ `α = n_âm/n` tính **trong `fit`** từ nhãn nhận được (chỉ fold-train khi CV), test
  `test_classifier_probabilities_and_dynamic_alpha` khoá hành vi này cùng việc xác suất luôn ∈ [0,1] và
  cộng lại bằng 1 (LightGBM không áp hàm liên kết cho custom objective, nên lab tự sigmoid từ raw score).
- Kết quả ở ví dụ này: `focal_loss` có PR-AUC OOF cao nhất trong nhóm algorithm-level (0.8386) và lợi
  nhiều nhất từ việc hạ ngưỡng về 0.304 (F1 0.776 → 0.818) — đúng kỳ vọng "tập trung vào mẫu khó".

## 6. Phát hiện định lượng (kể cả phát hiện phủ định)

1. **Tomek Links / ENN không tự cân bằng được 98/2**: IR chỉ từ 49.0 → 48.9 (`tomek`) và 48.6 (`enn`)
   vì chúng là *clean-sampling* — chỉ bỏ mẫu sát biên. Muốn có tác dụng phải ghép với oversampling
   (`smote_tomek`, `smote_enn`). Đây là lý do hai kỹ thuật này được đánh giá ở dạng nguyên bản.
2. **Oversampling không cải thiện thứ hạng xác suất rõ rệt**: PR-AUC OOF của `ros` 0.8397, `tomek`
   0.8379, `focal_loss` 0.8386 gần như bằng nhau, trong khi `rus` (undersampling mạnh) tụt còn 0.7563
   do mất 98% dữ liệu đa số.
3. **Ensemble nhạy cảm với hiệu chuẩn và cấu hình**: `balanced_rf` 0.6984, `easy_ensemble` 0.4747,
   `rusboost` 0.1103 — đều thấp hơn hẳn boosting thường + cost-sensitive (0.827–0.8386) với tham số
   mặc định; các con số này đo trên n=20.000 và tham số cố định (mục 6 nêu giới hạn).
4. **Chọn ngưỡng là can thiệp "miễn phí" hơn resampling**: nó chỉ đổi điểm vận hành của cùng một mô
   hình (PR-AUC không đổi) — phù hợp với kết luận P0 của `docs/ke-hoach-tiep-theo.md`.
5. **Đối chiếu với pipeline chính**: repo chính **cố ý không** dùng SMOTE/oversample cho bộ dữ liệu
   bán lẻ, vì ở đó nhãn gần như là thuộc tính của CÔNG TY (`ticker_prior` đạt AUROC 0.986) nên mẫu tổng
   hợp dễ rơi vào "vùng" của chính thực thể đã có trong train — hợp thức hoá đúng loại rò rỉ cấp thực
   thể mà đồ án đang đo. Danh mục này vì vậy là **lab thí nghiệm trên dữ liệu giả lập**, không phải
   đường huấn luyện của mô hình chốt.

## 7. Giới hạn (đọc trước khi trích dẫn số liệu)

- Dữ liệu đo là **giả lập 98/2** (`make_classification`), không phải bộ bán lẻ của đồ án.
- `focal_loss` mới chạy **một cấu hình** `γ=2,0` / `α=0,75` (chưa grid-search `γ`, `α`); mọi kỹ thuật
  resampling cũng dùng một tỉ lệ mục tiêu 0,5. So sánh giữa các nhóm là "mỗi kỹ thuật một cấu hình hợp
  lý", chưa phải "tối ưu từng kỹ thuật".
- `focal_loss` và các ensemble đều phụ thuộc thư viện (`lightgbm`, `imbalanced-learn`); khi thiếu, kỹ
  thuật đó bị **BỎ QUA** và ghi rõ lý do trong `techniques.json` / `techniques.md` (không im lặng bỏ).
- Chưa có: nhiều lớp (multi-class), dữ liệu thời gian, và kiểm chứng trên bộ dữ liệu bán lẻ thật — các
  hướng này thuộc P1/P2 của `docs/ke-hoach-tiep-theo.md`.

## 8. Tái lập

```powershell
python -m pip install -r requirements-labs.txt
python -m labs.imbalance_lab.techniques            # toàn bộ danh mục + threshold tuning → reports/imbalance/
python -m labs.imbalance_lab.techniques --quick --techniques baseline,smote,adasyn,focal_loss
python -m unittest discover -s tests -v       # 30 test danh mục + 19 test ĐÁNH GIÁ (yêu cầu #3)
```

| Artifact | Nội dung |
|---|---|
| `reports/imbalance/techniques.md` | Bảng đầy đủ 16 kỹ thuật (gồm baseline), ngưỡng OOF, PASS/FAIL chống rò rỉ, **bảng Baseline vs kỹ thuật**, kết luận tự động |
| `reports/imbalance/techniques_comparison.csv` | Bảng so sánh Baseline vs kỹ thuật: metric chính + ΔPR-AUC/ΔF1/ΔF1-macro |
| `reports/imbalance/techniques.csv` | Metric trên test theo kỹ thuật × 4 chế độ ngưỡng |
| `reports/imbalance/techniques_by_fold.csv` | IR train trước→sau và % dương fold-val từng fold |
| `reports/imbalance/techniques.json` | Cấu hình chạy, trạng thái từng kỹ thuật, ngưỡng đã chọn, bảng so sánh |
| `reports/imbalance/techniques.log` | Log console đầy đủ (gồm log phân phối nhãn mỗi fold) |
