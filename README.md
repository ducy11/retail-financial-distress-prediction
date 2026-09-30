# CS114 — Đồ án: Dự báo suy giảm tài chính (financial distress) doanh nghiệp bán lẻ

Dự án dùng dữ liệu báo cáo tài chính (SEC XBRL — 10-K/10-Q) của các chuỗi bán lẻ lớn tại Mỹ để dự báo trước 1 quý (horizon ~90 ngày sau `as_of`) liệu công ty có rơi vào suy giảm tài chính (`is_distressed`) hay không.

## Cấu trúc repo

```
data/                      # Dữ liệu (đã có sẵn trong repo)
  retail-expanded/         #   16 chỉ tiêu, quy đổi minh họa VND, có provenance từng chỉ tiêu
  prepared/                #   Split train / validation / test / purged (manifest.json mô tả chính sách)
  sec/raw/                 #   Snapshot SEC thô (bị loại khỏi git, tải lại bằng scripts.crawl_sec)
  samples/                 #   Mẫu BCTC CSV định dạng đồ án
forecasting/               # Gói Python: data pipeline, features, mô hình, đánh giá, báo cáo
imbalance_lab/             # Lab mất cân bằng 98/2 (resampling, ngưỡng, hiệu chuẩn) — độc lập
benchmark_imbalanced.py    # Benchmark 1 lệnh: Non-E Mode (resampling/cost-sensitive) vs E-Mode
scripts/                   # Lệnh CLI chạy bằng `python -m scripts.*`
notebooks/                 # Khám phá, phân tích EDA
reports/                   # Đầu ra (bảng điểm, confusion matrix, figure) — tái tạo được
docs/                      # Tài liệu đề cương, mở rộng dữ liệu, báo cáo
```

## Môi trường

- Python ≥ 3.11 (dev: 3.13.12, Windows).
- Cài phụ thuộc: `python -m pip install -r requirements.txt`.
- Code **không phụ thuộc pandas** — data load thuần `json`, tính năng bằng `numpy`, **mô hình bằng
  `scikit-learn`**: đúng **3 họ mô hình** (Logistic Regression · Random Forest · HistGradientBoosting).
  Pipeline chính **không dùng xgboost/lightgbm**; chỉ module lab mất cân bằng (`imbalance_lab/`,
  `benchmark_imbalanced.py`, `imbalance_experiment/`) mới dùng boosting ngoài làm base learner cho các
  kỹ thuật resampling. Mọi thứ chạy offline.
- **Cảnh báo vô hại của thư viện**: bộ phiên bản đang kiểm thử (scikit-learn 1.6 + scipy 1.18, matplotlib 3.9)
  in `OptimizeWarning: Unknown solver options: iprint` ở **mỗi** lần fit `LogisticRegression` và
  `PyparsingDeprecationWarning` khi vẽ hình. Repo lọc **đúng** hai thông điệp này trong
  `runtime_warnings.py` (`quiet_library_warnings()`) để log CLI/test sạch — cảnh báo khác (kể cả của code
  dự án) vẫn hiển thị. Các module test gọi lại hàm này trong `setUpModule()` vì `unittest` chèn
  `simplefilter("default")` lên đầu danh sách filter sau khi đã import module test.

## Quick start

```powershell
# 0. (Tuỳ chọn) Tái tạo lại split từ retail-expanded — mặc định bỏ qua nếu prepared đã có.
#    Nhãn is_distressed được giữ nguyên từ prepared cũ (sinh bởi pipeline prepare_sec gốc).
python -m forecasting.data --force

# 1. Huấn luyện + đánh giá trên validation, chọn mô hình & threshold
python -m forecasting.train

# 2. Đánh giá trên test với mô hình/ngưỡng đã CỐ ĐỊNH (test KHÔNG dùng để chọn mô hình/ngưỡng)
python -m forecasting.evaluate

# 3. Xuất bảng dự báo chi tiết từng mẫu + biểu đồ phân phối xác suất
python -m forecasting.report

# 4. EDA nhanh + EDA chuyên sâu + kiểm thử pipeline
python -m scripts.eda            # 9 hình + thống kê mô tả + tương quan + nhận xét (mục 3.1–3.6)
python -m scripts.eda_deep       # kiểm tra 47 feature, nhãn, tương quan, drift + 7 hình (mục 3.7)
python -m unittest discover -s tests -v
```

## Kiểm chứng bổ sung để đạt mức Xuất sắc (mục 9 của báo cáo)

```powershell
# 5. Tiền xử lý đuôi nặng: winsorize × scaler × 3 họ mô hình trên dữ liệu THẬT
python -m scripts.experiment_preprocessing        # → reports/results/preprocessing_experiment.{json,md}
# 6. Giải thích mô hình bằng SHAP (KernelSHAP tự cài đặt, có tự kiểm chứng efficiency)
python -m scripts.explain_model --max-explain 64  # → reports/results/shap.{json,md} + 3 hình
# 7. Kiểm định "hơn nhau có thật không": DeLong + paired bootstrap trên cùng 64 mẫu test
python -m scripts.significance                    # → reports/results/significance.{json,md}
# 8. Kỹ thuật xử lý lệch lớp trên DỮ LIỆU THẬT (7 kỹ thuật, GroupKFold theo công ty)
python -m scripts.experiment_imbalance_real       # → reports/results/imbalance_real.{json,md}
# 9. Tìm kiếm siêu tham số bằng random search + SỔ THỰC NGHIỆM runs.csv
python -m scripts.search                          # mặc định 40 trial/mô hình × 3 mô hình → runs.csv (113 dòng)
python -m scripts.search --trials 25              # chạy nhanh hơn (sổ sẽ có ít dòng hơn)
# 10. Độ nhạy của kết luận theo 4 ĐỊNH NGHĨA NHÃN (original, stress_signals, Altman Z'', forward-4Q)
python -m scripts.label_sensitivity               # → reports/results/label_sensitivity.{json,md}
# 11. Demo (dùng khi bảo vệ): dự đoán MỘT quý + giải thích SHAP cục bộ
python -m scripts.predict --sample-id HD-2024Q2 --explain
# 12. Kiểm chứng dữ liệu là THẬT: băm SHA-256 snapshot SEC + tra ngược từng fact + quy đổi VND
python -m scripts.verify_provenance             # → reports/results/provenance.{json,md}
```

Bộ tài liệu bảo vệ đồ án (factsheet + dàn 11 slide + 8 câu hỏi phản biện kèm kịch bản trả lời) là tài liệu
viết tay: [docs/bo-tai-lieu-bao-ve.md](docs/bo-tai-lieu-bao-ve.md) — kèm
[docs/checklist-doi-chieu-yeu-cau.md](docs/checklist-doi-chieu-yeu-cau.md) (đối chiếu tiêu chí → bằng
chứng → lệnh) và [docs/slide-bao-ve.md](docs/slide-bao-ve.md) (deck bảo vệ, xuất được `.pptx` bằng
`python -m scripts.export_office`).

**File xuất để nộp / làm slide** (`python -m scripts.export_office` → `docs/`):

| File | Dùng để |
|---|---|
| `BAO-CAO.docx` | **Báo cáo hoàn chỉnh** dạng Word (11 mục + 3 phụ lục, bảng + hình) — bản chính để đọc/nộp |
| `BAO-CAO-slide.pptx` | Slide tự động **14 mục**, mọi số khớp artifact (dùng làm khung slide) |
| `BAO-CAO-slide-bao-ve.pptx` | **Deck bảo vệ 11 slide** (takeaway + bullet + hình) |
| `BAO-CAO-slide-bao-ve.docx` | Cùng nội dung deck bảo vệ ở dạng Word — **để dựng slide** theo ý mình |
| `bo-tai-lieu-bao-ve.docx` | **Bộ tài liệu bảo vệ đầy đủ**: factsheet số liệu + dàn 11 slide (có lời thoại) + 8 Q&A phản biện |

## Benchmark mất cân bằng: Non-E Mode vs E-Mode (`benchmark_imbalanced.py`)

*(Phần này — cùng 2 lab bên dưới — dùng XGBoost/LightGBM **chỉ làm base learner** cho các kỹ thuật mất
cân bằng; **KHÔNG thuộc bộ mô hình của đồ án** (3 họ thuần scikit-learn ở pipeline chính).)*

Script độc lập, tái lập toàn bộ từ một lệnh, so sánh **11 phương pháp** trên dữ liệu giả lập mất cân
bằng cao **95/5** (10.000 mẫu, chia Stratified 80/20, test cố định dùng chung cho mọi phương pháp):

```powershell
python -m pip install -r requirements-benchmark.txt
python benchmark_imbalanced.py            # in bảng so sánh trực tiếp ra terminal
python benchmark_imbalanced.py --cv 5     # thêm StratifiedKFold 5 fold trên train split
python benchmark_imbalanced.py --quick    # bỏ EasyEnsemble cho nhanh
```

**Chia tập (theo yêu cầu):** `train_test_split(test_size=0.2, stratify=y, random_state=42)` giữ nguyên
tỉ lệ lớp; tuỳ chọn `--cv N` dùng `StratifiedKFold(n_splits=N, shuffle=True, random_state=42)` **trên
train split** (test vẫn khoá làm holdout). Mọi kỹ thuật cân bằng dữ liệu (SMOTE / RandomUnderSampler /
SMOTE+Tomek) **chỉ** chạy trên train (và trên fold-train khi CV) — validation/test không bao giờ được
resample; script tự kiểm chứng và in PASS/FAIL.

| Nhóm | Phương pháp |
|---|---|
| Baseline | Logistic Regression (không can thiệp) · XGBoost/LightGBM (mặc định) |
| Non-E Mode (data-level) | SMOTE · RandomUnderSampler · SMOTE+Tomek Links — chạy **trong `imblearn.pipeline.Pipeline`**, chỉ trên train |
| Non-E Mode (cost-sensitive) | `class_weight='balanced'` · `scale_pos_weight` động (tính trong `fit`) |
| E-Mode (ensemble) | `BalancedRandomForestClassifier` · `EasyEnsembleClassifier` · `RUSBoostClassifier` |

Chỉ số trên test: ROC-AUC, PR-AUC (Average Precision), F1 lớp thiểu số, Balanced Accuracy,
Precision/Recall và **thời gian huấn luyện**. Kết quả ghi vào `reports/benchmark_imbalanced.md`
(+ `.csv`, `.log`); kiểm chứng chống rò rỉ dữ liệu chạy ngay trong script và in PASS/FAIL.
Chi tiết & diễn giải: [docs/benchmark-mat-can-bang.md](docs/benchmark-mat-can-bang.md).

## Danh mục kỹ thuật mất cân bằng (yêu cầu #2) — `imbalance_lab/techniques.py`

*(Lab độc lập: base learner mặc định là LightGBM — xem ghi chú phạm vi ở mục trên.)*

Lab mất cân bằng (`imbalance_lab/`) có thêm một **danh mục đầy đủ 5 nhóm kỹ thuật** + threshold tuning
trên đường PR, chạy bằng một lệnh:

```powershell
python -m imbalance_lab.techniques            # 16 kỹ thuật (gồm BASELINE) × 5 fold → reports/imbalance/
python -m imbalance_lab.techniques --quick --techniques baseline,smote,adasyn,focal_loss
```

| Nhóm | Kỹ thuật |
|---|---|
| **Baseline (chưa xử lý)** | boosting mặc định — mốc so sánh cho mọi kỹ thuật |
| Data-level / Oversampling | RandomOverSampler · SMOTE · BorderlineSMOTE · ADASYN |
| Data-level / Undersampling | RandomUnderSampler · Tomek Links · EditedNearestNeighbours |
| Hybrid | SMOTE + Tomek Links · SMOTE + ENN |
| Algorithm-level | `scale_pos_weight` động · `class_weight='balanced'` · **Focal Loss** (custom objective LightGBM, grad/hess giải tích) |
| Ensemble | BalancedRandomForestClassifier · EasyEnsembleClassifier · RUSBoostClassifier |
| Threshold tuning | `best_f1` / `best_cost` / `min_precision` chọn trên **đường Precision-Recall của xác suất out-of-fold**, so với mốc 0.5 |

- Mọi sampler nằm TRONG `imblearn.pipeline.Pipeline` (bản nội bộ tương thích khi thiếu `imbalanced-learn`);
  cost-sensitive / Focal Loss tính trọng số trong `fit`; ngưỡng không bao giờ chọn trên test.
- **Đánh giá (yêu cầu #3):** Accuracy **không** là thước đo chính — bảng so sánh chỉ dùng Precision,
  Recall, F1 (binary/macro/weighted/**F-beta**), **PR-AUC (Average Precision)**, ROC-AUC, MCC và
  **Confusion Matrix**; accuracy chỉ xuất hiện như chỉ số chẩn đoán kèm mốc "đoán lớp đa số"
  (`imbalance_lab/metrics.py::accuracy_diagnostic`).
- Artifact: `reports/imbalance/techniques.{md,csv,json,log}`, `techniques_by_fold.csv` và
  **`techniques_comparison.csv`** — bảng Baseline (chưa xử lý) vs từng kỹ thuật kèm ΔPR-AUC/ΔF1/
  ΔF1-macro (hàm `compare_with_baseline`, `comparison_markdown`); PASS/FAIL chống rò rỉ in cho từng
  kỹ thuật.
- Kiểm thử: `tests/test_imbalance_techniques.py` (30 test: danh mục đủ mục, tỉ lệ mục tiêu của từng
  sampler, grad/hess Focal Loss khớp sai phân số, ngưỡng PR hợp lệ, chống rò rỉ) và
  `tests/test_evaluation_metrics.py` (19 test: metric khớp sklearn, chính sách không dùng accuracy,
  `imblearn.pipeline.Pipeline`, bảng so sánh baseline).
- Chi tiết & kết quả đo: [docs/cac-ky-thuat-mat-can-bang.md](docs/cac-ky-thuat-mat-can-bang.md).

## Thực nghiệm: Phương pháp ĐƠN LẺ vs PHƯƠNG PHÁP KẾT HỢP (`imbalance_experiment/`)

*(Lab độc lập: base learner mặc định là LightGBM — xem ghi chú phạm vi ở mục benchmark phía trên.)*

Gói thực nghiệm độc lập so sánh **17 pipeline** trên dataset mất cân bằng **1:50** (tuỳ chọn 1:100 hoặc
CSV kiểu *Credit Card Fraud Detection*), mọi bước resampling nằm TRONG `imblearn.pipeline.Pipeline`:

```powershell
python -m imbalance_experiment.main                       # 20.000 mẫu, 1:50, 5 fold → reports/experiment/
python -m imbalance_experiment.main --quick --n-samples 5000
python -m imbalance_experiment.main --imbalance-ratio 100 --model random_forest
python -m imbalance_experiment.main --data data/creditcard.csv --target Class
```

| Nhóm | Pipeline |
|---|---|
| Baseline | `baseline` (không can thiệp mất cân bằng) |
| Single (data-level) | RandomOverSampler · SMOTE · Borderline-SMOTE · ADASYN · RandomUnderSampler · Tomek Links · ENN |
| Single (algorithm-level) | `class_weight='balanced'` · Focal Loss |
| Single (ensemble) | Balanced Random Forest · EasyEnsemble · Balanced Bagging |
| **Hybrid** | SMOTE+Tomek · SMOTE+ENN · **SMOTE + Class Weights** · **RUSBoost** |

- Module: `data_loader.py` (chia tập stratified) · `pipeline_builder.py` (danh mục pipeline) ·
  `evaluation.py` (StratifiedKFold, metric **mean ± std**) · `insights.py` (phân tích tự động) ·
  `main.py` (CLI + bảng + biểu đồ PR curve).
- Metric: PR-AUC, ROC-AUC, F1 (thiểu số/macro), Balanced Accuracy, Recall, **FPR** — không dùng Accuracy.
- Artifact: `reports/experiment/summary.{md,csv}`, `cv_mean_std.csv`, `results.json`, `pr_curves.png`,
  `run.log`; mỗi pipeline có dòng PASS/FAIL chống rò rỉ.
- Kiểm thử: `tests/test_experiment.py`; chi tiết & phân tích: [imbalance_experiment/README.md](imbalance_experiment/README.md).

## Chính sách dữ liệu (tóm tắt từ `data/prepared/manifest.json`)

- Mỗi công ty: 8 quý cuối → test, 4 quý trước đó → validation, còn lại → train.
- Purge: các sample có nhãn công bố sau một mốc toàn cục bị loại để tránh rò rỉ thứ tự công bố.
- **Giữ tỉ lệ lớp (stratified):** tỉ lệ dương của train/validation/test lệch ≤ 3,3 điểm % so với toàn bộ
  (62,3%); CV tinh chỉnh dùng `StratifiedGroupKFold` — vừa giữ tỉ lệ lớp vừa giữ TRỌN công ty ngoài
  fold-train. Kiểm thử: `tests/test_pipeline.py::TestSplits.test_class_ratio_preserved_across_splits`.
- **Cân bằng/tiền xử lý chỉ trên train:** pipeline chính KHÔNG resample (chỉ `class_weight='balanced*'`
  do sklearn tính trong `fit`); impute/scale nằm trong `Pipeline` nên chỉ học thống kê từ train;
  SMOTE / RandomUnderSampler chỉ tồn tại trong `imbalance_lab/` và `benchmark_imbalanced.py`, đặt TRONG
  `imblearn.pipeline.Pipeline` ⇒ chỉ chạy trên train (fold-train khi CV), validation/test không bao giờ
  bị resample. Kiểm thử: `tests/test_pipeline.py::TestNoLeakageInPreprocessing`.
- Đơn vị: chuỗi số nguyên VND, tỷ giá minh họa 25.000 VND/USD (demo, không phải BCTC Việt Nam).

Xem chi tiết: [docs/mo-rong-du-lieu.md](docs/mo-rong-du-lieu.md), [data/README.md](data/README.md).
