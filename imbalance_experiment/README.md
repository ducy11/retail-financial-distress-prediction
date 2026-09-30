# `imbalance_experiment` — Phương pháp ĐƠN LẺ vs PHƯƠNG PHÁP KẾT HỢP (imbalanced learning)

Thực nghiệm độc lập, modular, tái lập được: so sánh **17 pipeline** xử lý mất cân bằng trên dataset
**1:50** (tuỳ chọn 1:100 hoặc CSV như *Credit Card Fraud Detection*), có **Chống rò rỉ dữ liệu** ở mọi
bước resampling và **phân tích chuyên sâu tự động** từ số liệu đo được.

```powershell
# cấu hình đầy đủ (20.000 mẫu, 1:50, 5 fold) — vài phút tuỳ máy
python -m imbalance_experiment.main

# thử nhanh
python -m imbalance_experiment.main --quick --n-samples 5000

# mất cân bằng 1:100, mô hình nền Random Forest
python -m imbalance_experiment.main --imbalance-ratio 100 --model random_forest

# dữ liệu thật: Credit Card Fraud Detection (tải riêng, không kèm repo)
python -m imbalance_experiment.main --data data/creditcard.csv --target Class --n-samples 0

# chạy một vài kỹ thuật
python -m imbalance_experiment.main --techniques baseline,smote,smote_enn,class_weight
```

## 1. Cấu trúc mã (theo yêu cầu #4)

| Module | Trách nhiệm |
|---|---|
| `config.py` | `ExperimentConfig` — một nguồn duy nhất cho mọi tham số (seed, tỉ lệ mất cân bằng, mô hình nền, số fold, ngưỡng…) |
| `data_loader.py` | `make_synthetic_dataset` / `load_csv_dataset` / `load_dataset` (chia train/test **stratified**), `stratified_folds` — mỗi fold giữ nguyên tỉ lệ lớp |
| `pipeline_builder.py` | `PipelineSpec` + `build_pipelines` + `make_estimator` + `RobustRUSBoost` + `inspect_pipeline` (kiểm tra imblearn pipeline & sampler nằm trong pipeline) + `pipeline_table` |
| `evaluation.py` | `evaluate_technique` / `evaluate_all` (StratifiedKFold, đo thời gian, cờ chống rò rỉ, PR curve) + `summary_rows` + bảng Markdown (pandas nếu có) |
| `insights.py` | `analyse` + `render_markdown` — 3 câu hỏi bắt buộc của yêu cầu #5, mọi con số lấy từ kết quả đo |
| `main.py` | `run(cfg)`: điều phối, in bảng so sánh, ghi artifact + biểu đồ PR curve, CLI |

Tái sử dụng có kiểm thử từ `imbalance_lab` (sampler, `metrics_at_threshold`, Focal Loss, `imblearn`
pipeline) ⇒ không viết lại công thức, không lệch số liệu giữa hai nơi.

## 2. Danh mục 17 pipeline

| Nhóm | Kỹ thuật |
|---|---|
| Baseline | `baseline` (LightGBM mặc định, **không** can thiệp) |
| Single — data-level | `ros`, `smote`, `borderline_smote`, `adasyn`, `rus`, `tomek`, `enn` |
| Single — algorithm-level | `class_weight` (`class_weight='balanced'`), `focal_loss` (custom objective) |
| Single — ensemble | `balanced_rf`, `easy_ensemble`, `balanced_bagging` |
| Hybrid | `smote_tomek`, `smote_enn` (resampling + cleaning), `smote_class_weight` (resampling + cost), `rusboost` (undersampling + boosting) |

## 3. Chống Data Leakage (yêu cầu #1) — được kiểm chứng tự động

1. **Mọi sampler nằm TRONG `imblearn.pipeline.Pipeline`** (`samplers.build_sampler_pipeline`) ⇒
   `fit_resample` chỉ chạy trên dữ liệu mà pipeline được `fit` = **train của fold**.
   Không có lệnh resample nào ở ngoài pipeline; `inspect_pipeline` chặn cấu hình sai ngay khi dựng.
2. **Fold-validation được so với bản sao trước khi fit** (`imbalance_lab.cv.assert_val_untouched`);
   tập test cũng được so trước/sau và chỉ chấm **đúng một lần** sau khi refit trên toàn bộ train.
3. **Class weights / Focal Loss tính trong `fit`** từ nhãn nhận được (chỉ fold-train), không dùng hằng
   số tính trước trên toàn bộ dữ liệu.
4. **Ensemble lấy mẫu bên trong `fit`** (BalancedRF/EasyEnsemble/BalancedBagging/RUSBoost) nên không
   bao giờ chạm validation/test.
5. Cột “kiểm chứng chống rò rỉ” trong `summary.md` in **PASS/FAIL cho từng pipeline**.

## 4. Chỉ số đánh giá (yêu cầu #3)

**Không dùng Accuracy** làm tiêu chí chính: PR-AUC (Average Precision), ROC-AUC, F1 (lớp thiểu số),
F1-macro, **Balanced Accuracy**, Recall, **FPR** (+ Precision, MCC, confusion counts để chẩn đoán).
Accuracy chỉ in kèm **mốc “đoán lớp đa số”** (xem `imbalance_lab.metrics.accuracy_diagnostic`).

Trong bảng báo cáo, metric CV được trình bày dạng **mean ± std qua các fold** (`evaluation._cv_table`),
metric holdout kèm **thời gian fit** và **hệ số phình dữ liệu** (`n train sau resample`).

## 5. Kết quả đo — cấu hình mặc định (20.000 mẫu, 1:50, 5 fold, LightGBM, ngưỡng 0.5)

Trung bình theo nhóm (metric holdout, Δ so với baseline):

| Nhóm | #kỹ thuật | PR-AUC | ΔPR-AUC | F1 (thiểu số) | Recall | FPR | giây/fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline (không xử lý)** | 1 | 0.9474 | — | 0.8714 | 0.7722 | 0.0000 | 1.03 |
| Single (đơn lẻ) | 12 | 0.9104 | **−0.0370** | 0.8042 | 0.8597 | 0.0091 | 1.12 |
| Hybrid (kết hợp) | 4 | 0.9229 | −0.0245 | 0.8307 | 0.9051 | 0.0087 | 1.29 |

Xếp hạng theo PR-AUC (top 5 và bottom 4):

| # | Kỹ thuật | Nhóm | PR-AUC | ΔPR-AUC | F1 | Recall | FPR |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `smote_enn` | hybrid | 0.9571 | +0.0097 | 0.934 | 0.899 | 0.0005 |
| 2 | `adasyn` | single-data | 0.9544 | +0.0070 | 0.927 | 0.886 | 0.0005 |
| 3 | `smote` | single-data | 0.9538 | +0.0064 | **0.940** | 0.899 | 0.0003 |
| 4 | `smote_tomek` | hybrid | 0.9538 | +0.0064 | **0.940** | 0.899 | 0.0003 |
| 5 | `smote_class_weight` | hybrid | 0.9531 | +0.0057 | 0.934 | 0.899 | 0.0005 |
| … | | | | | | | |
| 14 | `balanced_rf` | single-ensemble | 0.8832 | −0.0642 | 0.582 | **0.924** | 0.0252 |
| 15 | `balanced_bagging` | single-ensemble | 0.8761 | −0.0713 | 0.648 | 0.861 | 0.0161 |
| 16 | `rusboost` | hybrid | 0.8277 | −0.1196 | 0.514 | **0.924** | 0.0337 |
| 17 | `easy_ensemble` | single-ensemble | 0.6409 | −0.3065 | 0.371 | 0.886 | **0.0581** |

**Bằng chứng “không dùng Accuracy”**: accuracy chẩn đoán của baseline = 99.55% trong khi quy tắc
“đoán lớp đa số” đã được **98.03%** — chỉ 1,5 điểm % cho toàn bộ quá trình học. Trong khi đó các kỹ
thuật ensemble (undersampling mạnh) có accuracy tương tự nhưng PR-AUC thấp hơn baseline tới **0,31**.

## 6. Phân tích chuyên sâu (yêu cầu #5)

### 6.1 SMOTE đơn lẻ vs SMOTETomek / SMOTEENN (dọn biên có giúp?)

- `smote`: PR-AUC **0.9538**, F1 0.940, Recall 0.899, train sau resample **18.823** mẫu (CV 0.9645 ± 0.009).
- `smote_tomek`: PR-AUC **0.9538 (Δ +0.0000)**, F1 0.940 (Δ 0.000), FPR 0.0003 → **không cải thiện**
  (bước dọn Tomek chỉ bỏ vài mẫu nên gần như không đổi), chi phí resample **tăng 18×** (0,37s so với 0,02s/fold).
- `smote_enn`: PR-AUC **0.9571 (Δ +0.0033)** — nhỉnh nhất bảng — nhưng **F1 giảm −0.0062** và FPR tăng
  nhẹ (+0.0003): ENN bỏ cả mẫu khó–nhưng–thật nên đổi thứ hạng tốt hơn mà điểm vận hành xấu hơn.
- **Kết luận**: trên dữ liệu hai lớp tách rõ (`class_sep=1.6`), mẫu SMOTE phần lớn đã rơi đúng vùng nên
  **dọn biên không đem lại lợi ích rõ rệt**, chỉ làm tăng chi phí. Dọn biên đáng dùng khi overlap lớn
  (mẫu tổng hợp hay rơi vào vùng đa số) — khi đó ΔPR-AUC/ΔF1 mới dương và đủ lớn để bù chi phí.
  Đây cũng là lý do phải đọc **cả ba** chỉ số ΔPR-AUC, ΔF1 và ΔFPR thay vì một chỉ số.

### 6.2 Resampling lớn vs Cost/Weight: thời gian và chi phí tính toán

| Kỹ thuật | n train sau resample | Hệ số kích thước | giây/fold (fit) | giây/fold (resample) |
|---|---:|---:|---:|---:|
| `smote` | 18.823 | 1.47× | 1.468 | 0.020 |
| `adasyn` | 18.805 | 1.47× | 1.475 | 0.016 |
| `smote_tomek` | 18.823 | 1.47× | 1.793 | **0.367** |
| `smote_enn` | 18.699 | 1.46× | 1.732 | **0.372** |
| `smote_class_weight` | 18.823 | 1.47× | 1.446 | 0.008 |
| `class_weight` | — | **1.00×** | **1.109** | — |
| `focal_loss` | — | **1.00×** | 1.404 | — |
| `rus` | 754 | **0.06×** | **0.302** | 0.003 |

- Oversampling **phình tập train 1,47×** ⇒ mỗi vòng boosting xử lý thêm ~47% mẫu, cộng chi phí sinh mẫu
  (0,02–0,37s/fold tuỳ kỹ thuật dọn biên).
- Cost/Weight (`class_weight`, Focal Loss) **không đổi kích thước dữ liệu (1,00×)** và **không tốn thời
  gian resample**, trong khi vẫn dịch được điểm vận hành: `class_weight` đạt PR-AUC 0.9515 (Δ +0.0041),
  Recall 0.861 so với baseline 0.772.
- Về thời gian: `smote` fit chậm hơn `class_weight` **≈1,3×** (1,47s vs 1,11s/fold) trên 16.000 mẫu,
  20 chiều; với 5 fold × 15 kỹ thuật resampling thì phần chênh này cộng dồn thành vài phút. Khoảng cách
  còn tăng theo `imbalance_ratio` (1:100 ⇒ nhiều mẫu tổng hợp hơn) và theo số fold.
- **Chọn theo chi phí**: dữ liệu lớn / nhiều vòng CV / inference cần nhanh ⇒ ưu tiên **cost-sensitive**;
  chỉ resampling khi mô hình không hỗ trợ trọng số lớp hoặc cần mô hình “thấy” nhiều mẫu dương hơn.

### 6.3 Khi nào HYBRID vượt trội so với đơn lẻ?

| Hybrid | Đơn lẻ tốt nhất | ΔPR-AUC | ΔF1 | ΔRecall | ΔFPR | Kết luận |
|---|---|---:|---:|---:|---:|---|
| `smote_enn` | `smote` | **+0.0033** | −0.0062 | 0.0000 | +0.0003 | **hybrid tốt hơn** (PR-AUC) |
| `smote_tomek` | `smote` | +0.0000 | 0.0000 | 0.0000 | 0.0000 | hoà |
| `smote_class_weight` | `smote` | −0.0007 | −0.0062 | 0.0000 | +0.0003 | đơn lẻ nhỉnh hơn |
| `rusboost` | `balanced_rf` | −0.0554 | −0.0676 | 0.0000 | +0.0084 | đơn lẻ tốt hơn |

- Trên bộ dữ liệu này **hybrid không thắng đậm**: chỉ `smote_enn` nhỉnh +0,0033 PR-AUC (≈ 1/3 độ lệch
  chuẩn giữa các fold, xem cột `PR-AUC` ở mục “Metric chi tiết” ⇒ chưa đủ để kết luận chắc).
- **Khi nào hybrid thực sự đáng dùng** (điều kiện, không phải tuyên bố suông):
  1. **Overlap lớn giữa hai lớp** — mẫu SMOTE hay rơi vào vùng đa số; khi đó dọn biên (ENN/Tomek) có
     việc để làm và ΔPR-AUC thường dương rõ, đổi lại chi phí resample tăng ~18×.
  2. **Mẫu thiểu số cực ít (1:100 trở lên)** — SMOTE + trọng số lớp ổn định hơn SMOTE thuần vì không
     phụ thuộc hoàn toàn vào mẫu nội suy, và ít nhạy với ngưỡng quyết định.
  3. **Ngân sách tính toán hạn chế nhưng cần tỉ lệ dương cao trong train** — nhóm undersampling+ensemble
     (`rusboost`, `balanced_bagging`) rẻ hơn oversampling lớn; đổi lại recall cao (0.924) nhưng
     **PR-AUC giảm mạnh** và **FPR tăng** (0.025–0.058 so với 0.000 của baseline).
- Ngược lại, khi dữ liệu **tách rõ và đã đủ mẫu** (như cấu hình mặc định), **baseline rất mạnh**: 12/16
  kỹ thuật đơn lẻ và 3/4 kỹ thuật hybrid **không vượt** baseline về PR-AUC ⇒ thêm kỹ thuật là “đánh
  cược”, phải đo bằng ΔPR-AUC so với độ lệch chuẩn giữa các fold.


## 7. Artifact (mặc định `reports/experiment/`)

| File | Nội dung |
|---|---|
| `summary.md` | Báo cáo đầy đủ: danh mục pipeline, trung bình theo nhóm, xếp hạng, métric CV mean ± std, holdout + chi phí, PASS/FAIL chống rò rỉ, **phân tích chuyên sâu** |
| `summary.csv` | Metric holdout từng pipeline (mở bằng pandas/Excel) |
| `cv_mean_std.csv` | Metric CV dạng mean ± std |
| `results.json` | Cấu hình + metric từng fold + findings (tái lập & phân tích tiếp) |
| `pr_curves.png` | (trái) đường Precision-Recall mọi pipeline, (phải) điểm Recall–FPR tại ngưỡng 0.5 |
| `run.log` | Log đầy đủ (phân phối nhãn, thời gian, cảnh báo fallback của RUSBoost) |

## 8. Tái lập & kiểm thử

```powershell
python -m pip install -r imbalance_lab/requirements.txt
python -m imbalance_experiment.main                    # tái tạo đúng bộ số liệu ở mục 6/7
python -m unittest tests.test_experiment -v            # test dữ liệu, danh mục pipeline, CV, insights
python -m unittest discover -s tests -v                # toàn bộ test của repo
```

Test đáng chú ý: `test_catalogue_covers_every_required_technique` (đối chiếu 17 kỹ thuật với đề bài),
`test_resampling_techniques_use_imblearn_pipeline_inside` (mọi kỹ thuật resampling dùng
`imblearn.pipeline` và sampler nằm TRONG pipeline), `test_stratified_folds_keep_class_ratio_and_cover_all_samples`,
`test_resampling_flags_are_pass_and_size_is_recorded` (cờ chống rò rỉ PASS + đo kích thước sau resample),
`test_markdown_has_three_required_analyses`.

## 9. Ghi chú kỹ thuật & giới hạn

- **RUSBoost**: `imblearn` dùng AdaBoost bên trong và AdaBoost từ chối fit khi một base estimator “worse
  than random” — gặp thật ở 1:50. `RobustRUSBoost` thử lần lượt cây sâu 3 → stump → **RandomUnderSampler +
  boosting** và **ghi lại đường đã dùng** (log + `implementation_`) nên không âm thầm đổi thuật toán.
- **Cân bằng giữa 17 pipeline × 5 fold**: mỗi fold resample lại từ đầu (đúng nguyên tắc chống rò rỉ) nên
  chi phí = `n_fold × (resample + fit)`; `--quick` giảm số estimator khi cần chạy nhanh.
- **Giới hạn**: dataset mặc định là giả lập (`make_classification`) — kết luận định tính chuyển được,
  con số thì không; mỗi kỹ thuật dùng MỘT cấu hình hợp lý (chưa grid-search) nên đây là so sánh “cùng
  ngân sách”; ngưỡng báo cáo cố định 0,5 để so sánh trực tiếp (muốn tối ưu ngưỡng dùng
  `imbalance_lab.thresholds.tune_thresholds_from_pr_curve`, chọn trên xác suất out-of-fold).
- **Bài học phương pháp luận**: (1) luôn báo cáo baseline cùng mọi kỹ thuật; (2) so bằng PR-AUC + F1 +
  FPR chứ không bằng accuracy; (3) mọi can thiệp resampling phải nằm trong pipeline để CV không rò rỉ;
  (4) cải thiện phải lớn hơn độ lệch chuẩn giữa các fold mới đáng tin.
- **Log sạch**: `runtime_warnings.py` lọc đúng các cảnh báo vô hại của cặp phiên bản thư viện
  (`OptimizeWarning: Unknown solver options: iprint` khi fit `LogisticRegression` do scikit-learn 1.6 →
  scipy 1.18; `PyparsingDeprecationWarning` khi matplotlib 3.9 vẽ hình). Lọc áp ngay khi import package và
  trong `setUpModule()` của test (vì `unittest` chèn `simplefilter("default")` lên đầu trước khi chạy test),
  nên `reports/experiment/run.log` và output `unittest` chỉ còn thông tin của thực nghiệm.

