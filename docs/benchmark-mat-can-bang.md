# Benchmark mất cân bằng: **Non-E Mode** vs **E-Mode** (`benchmark_imbalanced.py`)

Script độc lập ở gốc repo — chạy một lệnh, in bảng so sánh trực tiếp ra terminal:

```powershell
python -m pip install -r requirements-benchmark.txt
python benchmark_imbalanced.py     # ~75–120 giây (11 phương pháp)
```

Artifact tái tạo được: `reports/benchmark_imbalanced.md` (bảng + kiểm chứng + ghi chú),
`reports/benchmark_imbalanced.csv` (đủ mọi cột thô, gồm confusion matrix và IR trước/sau),
`reports/benchmark_imbalanced.log` (log terminal đầy đủ).

## 1. Dữ liệu và cách chia tập

| Mục | Giá trị |
|---|---|
| Sinh dữ liệu | `make_classification(n_samples=10000, n_classes=2, weights=[0.95, 0.05], random_state=42)` — 20 đặc trưng (8 informative, 4 redundant), `flip_y=0.01` |
| Phân phối | 9.470 âm / 530 dương (**5,30 %**, IR = 17,9) |
| Chia tập | **Stratified 80/20** (8.000 train / 2.000 test, seed 42); test **cố định** dùng chung cho mọi phương pháp |
| Đánh giá bổ sung | `--cv N`: **`StratifiedKFold(n_splits=N, shuffle=True, seed 42)` chạy TRÊN TRAIN SPLIT** (test vẫn khoá làm holdout chốt) — báo cáo trung bình ± độ lệch chuẩn qua fold |
| Nguyên tắc bất di bất dịch | **Cân bằng dữ liệu chỉ trên tập train** (fold-train khi CV). Validation/test **không bao giờ** được resample — xem §3 |
| Chỉ số vô nghĩa | Accuracy: đoán “toàn bộ là lớp đa số” đã được **94,70 %** ⇒ không dùng; thay bằng **Balanced Accuracy** + **F1 lớp thiểu số** + **PR-AUC** |

## 2. Ba nhóm phương pháp (11 cấu hình)

| Nhóm | Phương pháp | Can thiệp |
|---|---|---|
| **Baseline** | Logistic Regression (không can thiệp); XGBoost/LightGBM mặc định | không |
| **Non-E Mode** — data-level | SMOTE + LR; SMOTE + boosting; RandomUnderSampler + LR; SMOTE + Tomek Links + LR | dữ liệu, **chỉ trên train** |
| **Non-E Mode** — cost-sensitive | LR `class_weight='balanced'`; boosting `scale_pos_weight` | hàm mất mát |
| **E-Mode** — bagging | `BalancedRandomForestClassifier`; `EasyEnsembleClassifier` | cấu trúc ensemble |
| **E-Mode** — boosting | `RUSBoostClassifier` | cấu trúc ensemble |

## 3. Chống rò rỉ dữ liệu — 3 kiểm chứng tự động

1. **Mọi sampler nằm trong `imblearn.pipeline.Pipeline`** ⇒ `fit_resample()` chỉ chạy trên TRAIN;
   script không có bất kỳ lệnh resample nào ngoài pipeline.
2. **`RecordingClassifier`** bọc classifier cuối để ghi lại số mẫu/nhãn nó **thực sự** nhận khi
   `fit` → so với kỳ vọng (train gốc, hoặc train đã resample). Cost-sensitive tính trọng số **bên
   trong `fit`** từ nhãn của tập được truyền vào, không dùng hằng số toàn cục.
3. **Tập test nguyên vẹn**: sao chép `X_test`/`y_test` trước khi huấn luyện, so `np.array_equal`
   sau khi dự báo.

Test ÂM TÍNH trong `tests/test_benchmark_imbalanced.py` (`test_leaky_estimator_is_flagged`) đưa vào
một estimator **cố tình sửa `X_test`** trong `fit` — script phải đánh dấu **FAIL**. Nhờ đó bảo đảm
các dòng “PASS” ở bảng dưới là kết quả thật của kiểm chứng, không phải kiểm chứng rỗng.
Kết quả lần chạy hiện hành: **11/11 PASS** (`reports/benchmark_imbalanced.md`).

### 3.1. Khi bật `--cv N` (StratifiedKFold)

Mỗi fold được kiểm chứng **riêng** (3 phép kiểm tra × N fold, ghi trong
`reports/benchmark_imbalanced_cv.csv`):

| Kiểm chứng | Ý nghĩa |
|---|---|
| `fold_val_nguyên_vẹn` | Bản sao fold validation được so `np.array_equal` sau khi fit ⇒ sampler/estimator không sửa dữ liệu kiểm định |
| `classifier_chỉ_thấy_fold_train` | Số mẫu classifier nhận = số mẫu **fold-train sau resample** (và ≠ fold-train gốc với phương pháp resampling) |
| `tỉ_lệ_lớp_giữ_nguyên` | Tỉ lệ dương của từng fold validation lệch ≤ 2 điểm % so với toàn bộ ⇒ `StratifiedKFold` có hiệu lực |

Ngoài ra shell kiểm chứng fold có tính chất chia đúng: `train ∩ val = ∅` và các fold-val **phủ đúng
một lần** toàn bộ mẫu (`tests/test_benchmark_imbalanced.py::TestStratifiedCV`).

## 4. Kết quả (test 2.000 mẫu, ngưỡng 0,5, seed 42)

| Nhóm | Phương pháp | ROC-AUC | **PR-AUC** | **F1 thiểu số** | **Bal. Acc** | Precision | Recall | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Logistic Regression | 0,7443 | 0,2833 | 0,1228 | 0,5328 | 0,8750 | 0,0660 | 0,01 |
| Baseline | **XGBoost (mặc định)** | **0,9210** | **0,6740** | **0,5935** | 0,7162 | **0,9388** | 0,4340 | 0,53 |
| Non-E (data) | SMOTE + Logistic Regression | 0,7717 | 0,1791 | 0,2201 | 0,6468 | 0,1474 | 0,4340 | 0,02 |
| Non-E (data) | SMOTE + XGBoost | 0,8963 | 0,6235 | 0,5536 | 0,7777 | 0,5254 | 0,5849 | 0,71 |
| Non-E (data) | RandomUnderSampler + LR | 0,7742 | 0,1701 | 0,2251 | 0,6413 | 0,1558 | 0,4057 | **0,01** |
| Non-E (data) | SMOTE + Tomek Links + LR | 0,7736 | 0,1483 | 0,2069 | 0,7174 | 0,1204 | 0,7358 | 0,27 |
| Non-E (cost) | LR `class_weight='balanced'` | 0,7742 | 0,1480 | 0,2016 | 0,7075 | 0,1173 | 0,7170 | 0,01 |
| Non-E (cost) | XGBoost `scale_pos_weight` | 0,9164 | 0,6579 | 0,5755 | 0,7759 | 0,5755 | 0,5755 | 0,56 |
| E-Mode | **BalancedRandomForest** | 0,9206 | 0,5441 | 0,4503 | **0,8212** | 0,3263 | 0,7264 | 1,03 |
| E-Mode | EasyEnsemble | 0,8183 | 0,2288 | 0,2297 | 0,7479 | 0,1349 | **0,7736** | 1,73 |
| E-Mode | RUSBoost | 0,8189 | 0,2465 | 0,3113 | 0,6364 | 0,3113 | 0,3113 | 1,07 |

**Trung bình theo nhóm**

| Nhóm | # | PR-AUC | F1 thiểu số | Bal. Acc | Train (s) |
|---|---:|---:|---:|---:|---:|
| Baseline | 2 | **0,4786** | **0,3582** | 0,6245 | 0,27 |
| Non-E Mode | 6 | 0,3211 | 0,3305 | 0,7111 | **0,26** |
| E-Mode | 3 | 0,3398 | 0,3304 | **0,7352** | 1,28 |

## 4.1. Kiểm tra độ ổn định bằng `StratifiedKFold` 5 fold (trên train split)

`python benchmark_imbalanced.py --cv 5` — mỗi phương pháp chạy lại 5 fold, báo cáo **trung bình ±
độ lệch chuẩn**; test 2.000 mẫu vẫn khoá làm holdout. Tỉ lệ dương của các fold validation:
**5,25 % – 5,31 %** (dataset 5,00 %) ⇒ chia stratified có hiệu lực. Cả **55/55 fold PASS** 3 kiểm
chứng chống rò rỉ (`reports/benchmark_imbalanced_cv.md`, `.csv`).

| Nhóm | Phương pháp | ROC-AUC | PR-AUC | F1 thiểu số | Bal. Acc |
|---|---|---:|---:|---:|---:|
| Baseline | Logistic Regression | 0,7562 ± 0,0226 | 0,2785 ± 0,0163 | 0,0844 ± 0,0430 | 0,5221 ± 0,0116 |
| Baseline | **XGBoost (mặc định)** | **0,9266 ± 0,0148** | **0,6589 ± 0,0246** | 0,5007 ± 0,0177 | 0,6723 ± 0,0082 |
| Non-E (data) | SMOTE + Logistic | 0,7741 ± 0,0131 | 0,1861 ± 0,0199 | 0,2457 ± 0,0117 | 0,6726 ± 0,0169 |
| Non-E (data) | SMOTE + XGBoost | 0,9138 ± 0,0095 | 0,5964 ± 0,0357 | 0,5201 ± 0,0323 | 0,7572 ± 0,0245 |
| Non-E (data) | RandomUnderSampler + Logistic | 0,7721 ± 0,0160 | 0,1815 ± 0,0225 | 0,2394 ± 0,0215 | 0,6622 ± 0,0255 |
| Non-E (data) | SMOTE + Tomek + Logistic | 0,7766 ± 0,0130 | 0,1609 ± 0,0165 | 0,2035 ± 0,0094 | 0,7045 ± 0,0162 |
| Non-E (cost) | Logistic `class_weight='balanced'` | 0,7769 ± 0,0139 | 0,1594 ± 0,0129 | 0,2000 ± 0,0085 | 0,7018 ± 0,0160 |
| Non-E (cost) | XGBoost `scale_pos_weight` | 0,9219 ± 0,0109 | 0,6301 ± 0,0223 | **0,5860 ± 0,0173** | 0,7637 ± 0,0138 |
| E-Mode | BalancedRandomForest | 0,9170 ± 0,0069 | 0,5221 ± 0,0166 | 0,4354 ± 0,0133 | **0,8256 ± 0,0070** |
| E-Mode | EasyEnsemble | 0,8115 ± 0,0184 | 0,2354 ± 0,0251 | 0,2268 ± 0,0051 | 0,7376 ± 0,0102 |
| E-Mode | RUSBoost | 0,7561 ± 0,0335 | 0,1771 ± 0,0396 | 0,2080 ± 0,0332 | 0,6208 ± 0,0353 |

**Xếp hạng trùng khớp với holdout** (XGBoost mặc định cao nhất về PR-AUC; nhóm resampling thấp hơn cả
baseline; BalancedRandomForest cao nhất về Balanced Accuracy) ⇒ kết luận §5 **không** phải hiện tượng
của một lần chia tập; độ lệch chuẩn nhỏ (PR-AUC ± 0,007…0,040) cũng cho thấy các chênh lệch lớn
(0,66 vs 0,16) là thật.

## 5. Diễn giải

1. **Không can thiệp thắng về xếp hạng xác suất.** XGBoost mặc định đạt PR-AUC 0,6740 và ROC-AUC
   0,9210 — cao nhất trong 11 cấu hình. Trên dữ liệu mất cân bằng, **PR-AUC là chỉ số quyết định**;
   vì vậy “phải resampling mới tốt” là giả định sai ở đây.
2. **Resampling dữ liệu làm giảm PR-AUC** (SMOTE+LR 0,1791 vs LR 0,2833; SMOTE+Tomek 0,1483 —
   mức thấp nhất). Oversample/undersample làm lệch prior và mất dữ liệu đa số: recall tăng nhưng
   precision sụp (0,12–0,16), ròng lại là mất mát ở PR-AUC. XGBoost chịu SMOTE tốt hơn LR (0,6235)
   nhưng vẫn **thấp hơn** chính nó khi không can thiệp.
3. **Cost-sensitive là can thiệp “rẻ” nhất**: giữ PR-AUC gần baseline (0,6579, chỉ kém 0,016) mà
   tăng balanced accuracy 0,7162 → 0,7759 so với baseline, và gần như không tăng thời gian huấn luyện.
4. **E-Mode đổi precision/recall hiệu quả**: BalancedRandomForest đạt **Balanced Accuracy cao nhất
   0,8212**, EasyEnsemble đạt **recall cao nhất 0,7736**. Cái giá: precision thấp (0,13–0,33) và
   thời gian huấn luyện trung bình **1,28 s** (~5× Non-E Mode) — RUSBoost ≈ 1,07 s, BRF ≈ 1,03 s.
5. **Khuyến nghị triển khai:** (a) giữ mô hình mặc định/boosting làm bộ xếp hạng; (b) nếu cần recall
   cao, **dịch ngưỡng quyết định** trước khi nghĩ tới resampling — cùng một mô hình, đổi ngưỡng cho
   phổ precision/recall rộng mà **không** mất PR-AUC; (c) chỉ dùng E-Mode khi mục tiêu là balanced
   accuracy và chấp nhận chi phí + mất khả năng hiệu chuẩn xác suất (ensemble cân bằng làm lệch
   prior — xem `reports/imbalance/` để thấy hiệu ứng tương tự ở lab 98/2).

## 6. Tương thích phiên bản (đã gặp khi chạy — script tự xử lý)

| Vấn đề | Biểu hiện | Cách script xử lý |
|---|---|---|
| `imblearn.RUSBoostClassifier` dựa trên AdaBoost **SAMME.R** | scikit-learn 1.6 đã bỏ SAMME.R → `ValueError: BaseClassifier in AdaBoostClassifier ensemble is worse than random` | Thử imblearn trước (đường chuẩn); nếu lỗi → `RUSBoostInternal` (cài đúng RUSBoost: RUS theo trọng số mỗi vòng, bỏ qua weak learner có lỗi ≥ 0,5) và **ghi rõ backend** trong cột `note` của báo cáo |
| XGBoost 2.1 + scikit-learn 1.6 (`'super' object has no attribute '__sklearn_tags__'`) | lỗi khi xgboost nằm trong `sklearn.Pipeline` (ví dụ có bước imputer) | Script dùng xgboost **trực tiếp** (không bọc Pipeline sklearn) và **probe fit** trước; nếu probe lỗi thì tự hạ cấp `lightgbm` → `HistGradientBoosting` và ghi backend vào log |
| `pandas.DataFrame.to_markdown()` | `ImportError: Import tabulate failed` | Nếu thiếu `tabulate`, script **tự sinh bảng Markdown** tương đương (vẫn xuất `.csv` bằng DataFrame) |

## 7. Kiểm thử

`tests/test_benchmark_imbalanced.py` (11 test, chạy cùng `python -m unittest discover -s tests -v`):

| Test | Nội dung |
|---|---|
| `TestDataset` | tỉ lệ ~95/5; chia stratified giữ tỉ lệ; seed cho kết quả lặp lại |
| `TestNoLeakage` | SMOTE trong pipeline: classifier chỉ thấy train đã resample, `X_test`/`y_test` nguyên vẹn; baseline thấy đúng toàn bộ train; **test âm tính** estimator sửa test phải bị đánh dấu FAIL |
| `TestStratifiedCV` | 5 fold giữ tỉ lệ lớp (< 1,5 điểm %), `train ∩ val = ∅`, các fold-val phủ đúng một lần toàn bộ mẫu; `evaluate_cv` với SMOTE chỉ resample fold-train (fold-val nguyên vẹn, 3/3 fold PASS); phương pháp không resampling giữ nguyên số mẫu fold-train |
| `TestRusBoostFallback` | `RUSBoostInternal` cho xác suất trong [0,1], tổng = 1, mọi `alpha > 0`; wrapper `RUSBoostRobust` fit + predict được |
| `TestReporting` | bảng Markdown đủ cột/dòng; thống kê nhóm đủ 3 nhóm và có mục “Tốt nhất theo từng chỉ số” |

## 8. Hạn chế

- **Một split, một seed** (42): kết quả holdout là point estimate. Đã giảm rủi ro bằng chế độ `--cv 5`
  (StratifiedKFold trên train split ⇒ có độ lệch chuẩn qua fold, xếp hạng trùng holdout), nhưng vẫn
  chưa có **khoảng tin cậy bootstrap** cho holdout — muốn có thì dùng `imbalance_lab` (đã có
  StratifiedKFold + bootstrap CI + quét ngưỡng OOF) hoặc lặp nhiều seed rồi tổng hợp.
- Ngưỡng báo cáo cố định 0,5 cho mọi phương pháp (đúng yêu cầu so sánh thuần phân loại); tối ưu ngưỡng
  là chủ đề riêng — xem `imbalance_lab/thresholds.py`, `reports/imbalance/thresholds.json`.
- Thời gian huấn luyện đo trên máy dev (1 luồng, `n_jobs=1`) nên chỉ dùng để **so sánh tương đối**.
- Dữ liệu giả lập có cấu trúc mạnh (`n_informative=8`), nên “mô hình mặc định thắng” ở đây **không**
  suy ra được rằng resampling vô dụng với mọi dữ liệu mất cân bằng — chỉ nói rằng nó không cần thiết
  khi nhiễu thấp và ranh giới lớp phân tách rõ.
- Đây là **benchmark phương pháp**, không thay thế kết luận của pipeline chính (`reports/results/`),
  nơi vấn đề trung tâm là rò rỉ thông tin cấp thực thể chứ không phải mất cân bằng lớp.
