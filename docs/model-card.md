# Model card — Dự báo suy giảm tài chính doanh nghiệp bán lẻ (`is_distressed`)

*Tài liệu viết tay (không sinh tự động). Số liệu hiệu năng hiện hành nằm trong
`docs/BAO-CAO.md` và `reports/results/*.json`; mọi con số trong tài liệu này đều kiểm chứng được
bằng `python -m scripts.audit_data`.*

## 1. Thông tin chung

| Mục | Giá trị |
|---|---|
| Tên mô hình | `reports/models/best.joblib` — **Random Forest** (bagging cây), chốt theo quy tắc ở §5.2 báo cáo; **cấu hình mặc định** (xem §5) |
| Bài toán | Phân loại nhị phân: quý **kế tiếp** doanh nghiệp có rơi vào suy giảm tài chính không |
| Đơn vị dự báo | 1 mẫu = 1 (công ty, quý target); lịch sử ≤ 8 quý, chỉ dùng dữ liệu đã công bố trước `as_of` |
| Phiên bản dữ liệu | `data/prepared/*` (SHA-256 trong `data/prepared/manifest.json`) |
| Chủ sở hữu | Nhóm đồ án CS114 — dữ liệu công khai SEC XBRL |
| Trạng thái | **Nghiên cứu/giáo dục** — KHÔNG dùng để ra quyết định tín dụng |

## 2. Mục đích sử dụng

**Dùng được:** nghiên cứu phương pháp luận về rò rỉ dữ liệu cấp thực thể, thiết kế giao thức đánh giá
(baseline theo thực thể, cross-company CV), và minh hoạ quy trình ML tái lập được.

**KHÔNG dùng cho:** xếp hạng tín dụng, quyết định cho vay/đầu tư, đánh giá một công ty cụ thể —
vì (a) nhãn không có định nghĩa kiểm chứng được, (b) chỉ có 8 công ty, (c) metric in-domain bị chi
phối bởi "nhớ mặt công ty" (xem §7).

## 3. Dữ liệu huấn luyện

| Mục | Giá trị |
|---|---|
| Nguồn | SEC XBRL companyfacts (`data/sec/raw/`, SHA-256 trong `data/sec/downloads.json`) |
| Chuỗi chỉ tiêu | 16 chỉ tiêu/quý, quy đổi minh hoạ ×25.000 VND/USD (`data/retail-expanded/`) |
| Corpus | **8 công ty, 332 quý → 324 mẫu** (WMT, HD, LOW, ROST, DG, ORLY, DKS, FIVE) |
| Chia tập | test = 8 quý cuối/công ty (64), validation = 4 quý trước đó (32), purge = 2 quý (16), còn lại train (212) |
| Chống rò rỉ thời gian | mọi dòng lịch sử có `available_on <= as_of`; dải purge 2 quý |
| Nhãn | `is_distressed` **giữ nguyên từ pipeline gốc** — pipeline đó **đã được port** thành `scripts/prepare_sec.py` và đối chiếu ngược **99,92% ô** (`reports/results/etl_verify.md`); nhãn vẫn không tái tạo được từ dữ liệu công bố (mức khớp cao nhất với quy tắc đơn giản = 74,7%) ⇒ dùng kèm **nhãn quy tắc** (`stress_signals`/`altman_z`/`forward_4q`) và **nhãn sự kiện** (`scripts/fetch_events.py`) |
| Cân bằng lớp | toàn bộ 202/122 (62,3%/37,7%, IR 1,66); train IR 1,65; test 38/26; **theo công ty rất lệch**: HD/LOW/WMT 100% nhãn 1, ROST 4,7% (IR 20,5) — `reports/results/class_balance.md` |

## 4. Đặc trưng (features)

- **47 cột** = 14 tỷ số (giá trị mới nhất + YoY) + 10 tốc độ tăng trưởng + cấu trúc vốn + nhóm `path`
  (cực trị xấu nhất trong cửa sổ, mức giảm doanh thu so với đỉnh, chuỗi quý âm) + chỉ báo căng thẳng.
- Chỉ dùng lịch sử của chính mẫu; thiếu dữ liệu → NaN → `SimpleImputer(median)` **trong pipeline**.
- **Nợ phải trả suy ra** từ `total_assets − stockholders_equity` khi thiếu tag `liabilities`
  (tag này chỉ phủ 39% số quý) — đối chiếu 124 quý có tag trong `scripts/audit_data.py`.
- **Không** đưa `ticker` vào mô hình (sẽ hợp thức hoá rò rỉ thực thể).
- **Mọi phân tích trong báo cáo** (ngưỡng, permutation importance, hiệu chuẩn, ablation) được tính
  trên **chính mô hình này** — `reports/results/analysis.json` có khoá
  `{threshold,importance,calibration}.model` và audit bắt buộc chúng bằng `summary.best_model`
  (trước đây script phân tích hard-code `logistic`, khiến hình/bảng nói về mô hình khác).

## 5. Kiến trúc & quy trình

- Pipeline sklearn: `median-impute → (StandardScaler cho mô hình tuyến tính/MLP) → classifier`;
  **4 họ mô hình**: Logistic Regression, Random Forest, HistGradientBoosting, **MLP** (mạng nơ-ron phi
  tuyến, không dựa trên cây) — đều thuần scikit-learn; đồ án KHÔNG dùng xgboost/lightgbm.
- **Bốn baseline đối chứng**: dummy (lớp đa số), `ticker_prior` (nhớ mặt công ty), logistic 1 chỉ tiêu,
  và **quy tắc Altman Z'' < 1,1** (công thức công khai, không học tham số từ dữ liệu).
- **Chọn mô hình theo AP cross-company** (GroupKFold trên train+validation) → best-F1(val) → AP(val)
  → AUROC(val) → gap overfit nhỏ nhất. Quy tắc lưu trong `summary.json::selection_rule`.
- **Ngưỡng quyết định**: ngưỡng best-F1 trên validation + ngưỡng tối ưu theo chi phí kỳ vọng
  (`COST_FN=5`, `COST_FP=1` trong repo chính; `10:1` trong lab 98/2).
- **Đánh giá cuối trên test** (`python -m forecasting.evaluate`) với mô hình/ngưỡng đã cố định;
  **test không tham gia chọn mô hình/ngưỡng** (chọn trên validation + AP cross-company out-of-fold),
  và cũng không dùng để thử cấu hình khác.
- **Cấu hình đang chạy** = cấu hình **mặc định** trong `forecasting/models.py::HYPERPARAMS`
  (Random Forest: `n_estimators=300, max_depth=6, min_samples_leaf=2`,
  `class_weight='balanced_subsample'`, `random_state=42`); tinh chỉnh theo CV chỉ hơn +0,003 CV-AP
  nên **không** được triển khai (xem `reports/results/tuning.json`).

## 6. Hiệu năng (trạng thái tại lần chạy gần nhất — xem artifact để cập nhật)

| Góc nhìn | Chỉ số | Nguồn |
|---|---|---|
| In-domain (train→test) | AUROC test ≈ 0,97–0,98, AP ≈ 0,99 | `reports/results/test_evaluation.json` |
| Cross-company (GroupKFold) | AUROC ≈ 0,91–0,93 | `reports/results/validation_checks.json` |
| LOCO (bỏ trọn 1 công ty) | trung bình ≈ 0,61 trên **8/8** công ty tính được | cùng file |
| Baseline "ticker-prior" (không dùng feature) | AUROC test ≈ 0,986 | `reports/results/baselines.json` |
| Baseline 1 biến (`debt_to_assets_latest`) | AUROC test ≈ 0,89 | cùng file |

**Đọc đúng:** con số in-domain **không** chứng minh năng lực dự báo — baseline chỉ dùng danh tính công
ty đạt mức tương đương hoặc cao hơn. Chỉ số nên trích dẫn là **cross-company / LOCO** (kèm khoảng tin
cậy bootstrap).

## 7. Hạn chế đã biết (bắt buộc đọc trước khi dùng)

1. **Nhãn không kiểm chứng được**: quy tắc kế toán đơn giản khớp tối đa 74,7%; phản chứng
   `WMT-2015Q2` (lãi nhưng gán nhãn 1) — `docs/dinh-nghia-nhan.md`.
2. **Rò rỉ cấp thực thể**: nhãn gần như là thuộc tính công ty ⇒ metric in-domain bị thổi phồng.
3. **Chỉ 8 thực thể**: LOCO (giữ trọn 1 công ty ra ngoài) chỉ còn AUROC trung bình **0,610** trên cả
   8/8 công ty; learning curve chưa bão hoà (`analysis.json::learning_curve`, gap 0,105).
4. **Mẫu không độc lập**: cửa sổ 8 quý trượt nên các mẫu liền nhau chia sẻ phần lớn lịch sử.
5. **Winsorize chưa vào pipeline chính** — thí nghiệm `scripts/experiment_preprocessing.py` cho thấy với **mô hình được chốt** chỉ +0,13 điểm % cross-company AP (trong khoảng nhiễu), nên giữ pipeline đơn giản; chưa dùng mô hình chuỗi thời gian/survival.
6. **Thiếu tín hiệu ngoài báo cáo tài chính** (giá cổ phiếu, xếp hạng tín dụng, vĩ mô).
7. **Đơn vị tiền là minh hoạ** (×25.000 VND/USD) — không phải BCTC Việt Nam.

## 8. Sử dụng đúng cách (khuyến nghị triển khai)

1. Luôn báo cáo kèm **baseline ticker-prior** và **cross-company AUROC/AP**; không nêu in-domain AUROC
   đơn lẻ.
2. Chọn **ngưỡng theo chi phí nghiệp vụ** (FN thường đắt hơn FP), không dùng 0,5 mặc định.
3. Không dùng Accuracy cho bộ dữ liệu này (đoán lớp đa số đã đạt ~59–98% tuỳ tập/công ty).
4. Mọi quyết định dùng mô hình phải ghi `sample_id`, `P(distress)`, ngưỡng áp dụng, phiên bản model.

## 9. Bảo trì & giám sát

| Hoạt động | Tần suất | Cách làm |
|---|---|---|
| Giám sát dịch chuyển | mỗi lần cập nhật dữ liệu | PSI/K-S theo feature; tỉ lệ dương theo quý; AP cross-company (cảnh báo khi giảm > 0,05) |
| Huấn luyện lại | mỗi năm tài chính mới/có 10-K mới | `python -m scripts.run_all` (tái lập byte-identical cho split; test tự động) |
| Kiểm chứng số liệu | trước mỗi lần báo cáo | `python -m scripts.audit_data` (98.200 phép kiểm tra, 0 phát hiện) |
| Kiểm chứng NGUỒN GỐC | trước khi công bố/nộp | `python -m scripts.verify_provenance` — băm SHA-256 snapshot SEC, tra ngược từng fact trong companyfacts, kiểm quy đổi VND (0 lệch / 0 fact thiếu / 0 ô bịa số) |
| Kiểm thử | mỗi thay đổi code | `python -m unittest discover -s tests -v` (**234 test**, gồm chống rò rỉ + KernelSHAP + demo predict + `TestModelRegistry` chặn quay lại 3 mô hình + walk-forward/cluster bootstrap + ETL port) |
| Rollback | khi metric test/cross-company giảm | giữ artifact cũ; mọi artifact có SHA-256 trong manifest |

## 10. Tái lập & truy vết

```powershell
python -m scripts.run_all                 # toàn bộ artifact + docs (22 bước, gồm PORT ETL ở bước 2)
python -m scripts.audit_data              # đối chiếu với số liệu thật SEC: 0 phát hiện
python -m unittest discover -s tests -v   # 234 test (registry 4 mô hình, chống rò rỉ, ETL, walk-forward)
python -m imbalance_lab.run               # lab mất cân bằng (kèm hiệu chuẩn + mốc minh hoạ rò rỉ)
```

## 11. Chạy mô hình trên một mẫu/quý mới (demo)

```powershell
python -m scripts.predict --sample-id HD-2024Q2              # xác suất + quyết định theo ngưỡng vận hành
python -m scripts.predict --ticker FIVE --quarter 2024Q3 --explain   # + top-K đóng góp SHAP
python -m scripts.predict --input mau-moi.json --json        # tự cung cấp một mẫu theo định dạng prepared
```

Script in `P(distress)`, quyết định tại **ngưỡng vận hành** (không phải 0,5), **cảnh báo** khi mẫu
thuộc validation/test (đó là backtest, không phải dự báo tương lai), và (tuỳ chọn) đối chiếu nhãn
thật. Đây là công cụ phục vụ phần trình bày, không tạo artifact.

Liên quan: `docs/bo-tai-lieu-bao-ve.md` (factsheet + 11 slide + 8 Q&A), `docs/slide-bao-ve.md`,
`docs/checklist-doi-chieu-yeu-cau.md`, `docs/ke-hoach-tiep-theo.md` (lộ trình P0/P1/P2),
`docs/dinh-nghia-nhan.md` (truy vết nhãn), `docs/BAO-CAO.md` (kết quả đầy đủ + tài liệu tham khảo),
`data/prepared/manifest.json` (chính sách split + SHA-256).

