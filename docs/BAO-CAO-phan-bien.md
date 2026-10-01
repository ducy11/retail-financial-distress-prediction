# BÁO CÁO KỸ THUẬT TOÀN VĂN & GIẢI TRÌNH PHẢN BIỆN ĐỒ ÁN MÁY HỌC

**Tên đề tài:** Dự báo suy giảm tài chính (*financial distress*) doanh nghiệp bán lẻ niêm yết Hoa Kỳ
bằng học máy trên dữ liệu báo cáo tài chính công bố (SEC XBRL)

| Thông tin học vụ | Nội dung |
|---|---|
| Học phần | Máy học (*Machine Learning*) |
| Giảng viên hướng dẫn | «điền» |
| Nhóm tác giả | «điền họ tên – MSSV từng thành viên» |
| Hình thức nộp | Báo cáo kỹ thuật toàn văn + mã nguồn tái lập được + slide bảo vệ |
| Ngày hoàn thành | «điền» |

> **Nguyên tắc số liệu của báo cáo.** Mọi con số trong báo cáo này được đọc trực tiếp từ artifact do
> chính mã nguồn sinh ra (`reports/results/*.json|csv|md`, `reports/imbalance/*`,
> `reports/benchmark_imbalanced*.md`, `reports/experiment/*`). Không có số nào được nhập tay: chạy
> `python -m scripts.run_all` rồi `python -m scripts.make_report` là báo cáo tự khớp lại với kết quả.
> Vì vậy mọi bảng dưới đây có thể được Hội đồng đối chiếu lại ngay tại chỗ.

## Bảng ánh xạ 10 câu hỏi chất vấn ↔ chương giải trình

| # | Câu hỏi chất vấn | Giải trình ở | Bằng chứng trong repo |
|---|---|---|---|
| 1 | Bộ dữ liệu có bị mất cân bằng không? | 2.1 | `reports/results/class_balance.json`, `reports/results/eda_deep.md` |
| 2 | Trình bày công thức đánh giá? | 7.1–7.2 | `forecasting/evaluation.py`, `docs/cong-thuc-do-an.md` |
| 3 | Các cách xử lý mất cân bằng dữ liệu? | 9.1 | `docs/cac-ky-thuat-mat-can-bang.md`, `imbalance_lab/` |
| 4 | Ngoài baseline có đề xuất thêm đặc trưng mới gì? | 8.1 | `forecasting/features.py` (47 cột, 6 khối) |
| 5 | Tại sao dùng thực nghiệm loại bỏ dần và khác gì baseline? | 8.2–8.3 | `reports/results/analysis.json::ablation` |
| 6 | Non-E Mode và E-Mode là gì? | 9.2 | `reports/benchmark_imbalanced.md` |
| 7 | Kết quả Non-E, E-Mode và Class Weight ra sao? | 9.2 | `reports/benchmark_imbalanced.md`, `reports/results/imbalance_real.md` |
| 8 | Kết hợp nhiều phương pháp so với từng phương pháp riêng lẻ? | 9.3 | `reports/experiment/summary.md` |
| 9 | Có cần chạy thêm LiteSVM không? | 9.4 | `reports/results/defense_models.md` |
| 10 | So từng Class trên XGBoost? Phân tích Confusion Matrix, Train vs Val? | 6.1–6.4; phụ 9.5 | `reports/results/test_evaluation.json`, `analysis.json::overfit`, `defense_models.md` |

## Phân bổ thời gian trình bày (khớp barem chấm thi)

| Phần | Nội dung | Thời lượng | Chương |
|---|---|---|---|
| **I** | Trọng tâm cơ bản theo đề cương `DoAnMonHoc`: mục tiêu & dữ liệu → EDA → tiền xử lý → 4 mô hình → so sánh & chọn mô hình → từng lớp, ma trận nhầm lẫn, overfitting | 10 phút đầu | 1–6 |
| **II** | Giải trình chuyên sâu 10 câu hỏi: công thức & ngưỡng Bayes → 47 đặc trưng & ablation → cân bằng dữ liệu Non-E/E-Mode & LiteSVM → tổng kết, cảnh báo rò rỉ thực thể, cheat-sheet | 5 phút cuối + Q&A | 7–10 |

---

# PHẦN I — TRỌNG TÂM CƠ BẢN THEO CHUẨN ĐỒ ÁN

## CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI, MỤC TIÊU & BỘ DỮ LIỆU

### 1.1. Mục tiêu đồ án

Đồ án hướng tới một hệ thống học máy có thể **dự báo nguy cơ suy giảm tài chính của doanh nghiệp bán
lẻ** ở tầm quý, dựa trên chính dữ liệu báo cáo tài chính mà doanh nghiệp đã công bố. Cụ thể:

1. Xây dựng đường ống dữ liệu từ nguồn công khai (SEC XBRL) tới bộ mẫu học máy, trong đó mỗi mẫu là
   một **cửa sổ trượt 8 quý liên tiếp** (`LOOKBACK_QUARTERS = 8`, tương đương 2 năm tài chính) và nhãn
   là trạng thái tài chính của quý kế tiếp.
2. Thực hiện trọn vẹn chu trình khoa học dữ liệu: **EDA → tiền xử lý → huấn luyện đối chuẩn 4 họ mô
   hình → đánh giá, so sánh, chọn mô hình → phân tích lỗi và kiểm tra overfitting**.
3. Trả lời trung thực câu hỏi "mô hình này có thực sự dự báo được không?", bằng cách **đo lường rò rỉ
   thông tin ở cấp thực thể** — vấn đề mà phần lớn báo cáo về chủ đề này bỏ qua.

### 1.2. Mô tả bộ dữ liệu thực tế

| Thuộc tính | Giá trị | Nguồn kiểm chứng |
|---|---|---|
| Kiểu dữ liệu | Dạng bảng (*tabular*), chuỗi quý | `data/prepared/*.json` |
| Số doanh nghiệp | 8 chuỗi bán lẻ niêm yết (DG, DKS, FIVE, HD, LOW, ORLY, ROST, WMT) | `data/sec/downloads.json` |
| Số quý báo cáo | 332 quý | `reports/results/eda_summary.json` |
| Số mẫu dự báo | **N = 324** | `data/prepared/manifest.json` |
| Số đặc trưng | **47 cột** (6 khối, xem Chương 8) | `forecasting/features.py::feature_names()` |
| Bài toán | **Phân loại nhị phân** (`is_distressed` ∈ {0, 1}) | `forecasting/config.py::TARGET` |
| Kiểm chứng nguồn | 20 file SEC băm SHA-256 khớp registry (**0 lệch**); **4.609 ô** tra ngược được `accn/form/ngày nộp` (**0 ô bịa số**) | `reports/results/provenance.md` |

**Về nguồn gốc nhãn — nói rõ để Hội đồng đối chiếu:** nhãn `is_distressed` là **nhãn đi kèm bộ dữ
liệu**, *không* suy ra được từ dữ liệu công bố: quy tắc kế toán đơn giản nhất chỉ khớp tối đa
**74,7%** (`stress_signals ≥ 1`), các quy tắc khác khớp 39–65% (`reports/results/analysis.json::label_audit`).
Vì vậy nhóm **không** che giấu điểm yếu này mà xử lý theo hướng khoa học: bổ sung **ba định nghĩa nhãn
tái lập được** để kiểm chứng độ nhạy của mọi kết luận:

| Định nghĩa nhãn | Công thức | Kết luận có giữ nguyên? |
|---|---|---|
| `stress_signals` | `L = 1[Σ_{s=1..6} 1[s] ≥ 1]`, 6 tín hiệu: `NI<0`, `OCF<0`, `OI<0`, `CL>CA`, `SE<0`, `YoY(doanh thu) < −5%` | có |
| `altman_z` | `Z″ = 6,56·WC/TA + 3,26·RE/TA + 6,72·EBIT/TA + 1,05·BV_E/TL`; `L = 1[Z″ < 1,1]` (Altman 1968/2000, không học tham số) | có |
| `forward_4q` | `L = 1[∃ q ∈ {+1..+4}: Σ tín hiệu(q) ≥ 1]` (đo *sự kiện sắp xảy ra*, không phải trạng thái quý) | có |

Kết quả: **3/3 định nghĩa giữ nguyên kết luận** về thứ hạng mô hình và về hiện tượng rò rỉ thực thể
(`reports/results/label_sensitivity.md`) — đây là bằng chứng cho thấy kết luận không phụ thuộc vào một
định nghĩa nhãn duy nhất. Ngoài ra repo còn có **nhãn sự kiện 8-K** của SEC: **0 sự kiện phá sản
(item 1.03)** trên cả 8 doanh nghiệp, 11 hồ sơ tín hiệu kiệt quệ ⇒ đề tài được định vị đúng là
*suy giảm tài chính*, không phải *phá sản* (`reports/results/events.md`).

### 1.3. Phân chia tập dữ liệu chống rò rỉ thời gian

Dữ liệu là **chuỗi quý** nên chia ngẫu nhiên (`train_test_split` thuần) là sai về phương pháp: các quý
liền nhau của cùng một doanh nghiệp có tương quan chuỗi rất cao (**90,8%** cặp quý liền kề giữ nguyên
nhãn; xác suất một quý căng thẳng khi quý trước cũng căng thẳng là **0,929**). Chia ngẫu nhiên sẽ đưa
"tương lai" vào tập huấn luyện và làm điểm số bị thổi phồng. Nhóm áp dụng **chia theo thời gian kết
hợp dải thanh lọc (*purged time-split*)** cho từng doanh nghiệp:

| Tập | Số mẫu | Nhãn 1 | Nhãn 0 | Tỉ lệ dương | IR | Vai trò |
|---|---:|---:|---:|---:|---:|---|
| Train | 212 | 132 | 80 | 62,3% | 1,65 | huấn luyện mọi tham số (impute, scale, trọng số lớp) |
| **Purged** | **16** | 11 | 5 | 68,8% | 2,20 | **loại khỏi cả ba tập** — dải đệm cách ly theo ngày công bố |
| Validation | 32 | 21 | 11 | 65,6% | 1,91 | chọn ngưỡng vận hành & cấu hình |
| Test | 64 | 38 | 26 | **59,375%** | 1,46 | đánh giá cuối, chấm **đúng một lần** |

Ba điểm kỹ thuật của giao thức này:

1. **Test "khoá" hoàn toàn:** quy tắc chọn mô hình và ngưỡng được ghi trong mã
   (`max AP cross-company → best-F1(val) → AP(val) → AUROC(val) → gap nhỏ nhất`), test không tham gia
   bất kỳ quyết định nào.
2. **Purge theo ngày công bố:** mỗi mẫu gắn `available_on`; các mẫu nằm trong cửa sổ 90 ngày quanh mốc
   chia bị đẩy vào `purged` thay vì rơi vào train/validation, triệt tiêu rò rỉ nhãn do công bố muộn.
3. **Kiểm chứng tự động:** `tests/test_pipeline.py::test_no_label_leak_in_features` khẳng định mọi dòng
   lịch sử đều thoả `available_on ≤ as_of` và không chứa quý target; `scripts/audit_data.py` kiểm
   **98.200** phép kiểm tra, **0** phát hiện.

---

## CHƯƠNG 2: PHÂN TÍCH KHÁM PHÁ DỮ LIỆU (EDA)

### 2.1. Phân phối lớp và thực trạng mất cân bằng

**Trả lời trực tiếp câu hỏi "dữ liệu có mất cân bằng không?": có, nhưng theo hai cấp khác nhau — cấp
mẫu chỉ lệch nhẹ, cấp thực thể lệch cực nặng.**

| Chỉ số toàn corpus (324 mẫu) | Giá trị | Diễn giải |
|---|---:|---|
| Tỉ lệ nhãn dương | 62,35% | lớp **đa số là nhãn 1** (suy giảm) — ngược với bộ dữ liệu phá sản thông thường |
| Imbalance Ratio (IR = đa số/thiểu số) | **1,6557** | lệch **nhẹ** ở cấp mẫu |
| Shannon Entropy | **0,9556 bit** (max 1,0) | gần cân bằng |
| Gini Impurity | **0,4695** (max 0,5) | gần cân bằng |
| Accuracy của quy tắc "đoán lớp đa số" | 62,35% | mốc chứng minh Accuracy vô dụng để kết luận |
| Số thực thể hiệu dụng `1/Σp²` | 7,848 / 8 | phân bố mẫu giữa 8 công ty gần đều |
| Cặp quý liền kề giữ nguyên nhãn | 90,8% | bằng chứng dữ liệu có **tự tương quan mạnh** |

**Mất cân bằng cấp thực thể mới là vấn đề thật:**

| Doanh nghiệp | n | Tỉ lệ nhãn 1 | IR | Mức |
|---|---:|---:|---:|---|
| HD | 43 | **100,0%** | ∞ | một lớp |
| LOW | 43 | **100,0%** | ∞ | một lớp |
| WMT | 47 | **100,0%** | ∞ | một lớp |
| ORLY | 43 | 90,7% | 9,75 | lệch nghiêm trọng |
| DG | 31 | 54,8% | 1,21 | cân bằng |
| FIVE | 31 | 19,4% | 4,17 | lệch nghiêm trọng |
| DKS | 43 | 11,6% | 7,60 | lệch nghiêm trọng |
| ROST | 43 | **4,7%** | 20,50 | lệch nghiêm trọng |

41,0% số mẫu thuộc các công ty **chỉ có một lớp nhãn**. Đây chính là cơ chế khiến một mô hình có thể
đạt AUROC rất cao bằng cách "nhớ mặt công ty" — nội dung được định lượng ở Chương 10.2 và là phát hiện
khoa học trung tâm của đồ án.

**Mở rộng sang bộ dữ liệu mất cân bằng cực đoan:** vì corpus thật lệch nhẹ, nhóm chạy thêm hai bộ
giả lập để khảo sát *bản chất* kỹ thuật cân bằng: **95/5** (10.000 mẫu, `reports/benchmark_imbalanced.md`)
và **1:50** (20.000 mẫu, `reports/experiment/summary.md`) — kết quả ở Chương 9.

### 2.2. Phân tích tương quan thuộc tính ↔ nhãn

Ba thước đo bổ trợ nhau được dùng đồng thời (vì mỗi thước bắt một kiểu quan hệ): **hệ số điểm-biserial
Pearson** (quan hệ tuyến tính), **hiệu ứng hạng `|2·AUC − 1|`** (quan hệ đơn điệu, không phụ thuộc cỡ
mẫu) và **Mutual Information** (quan hệ phi tuyến). Vì có 47 kiểm định song song, mọi p-value được hiệu
chỉnh bằng **Benjamini–Hochberg (FDR)** — nếu không, ở mức α = 0,05 sẽ có ~2 "phát hiện" thuần ngẫu nhiên.

| Feature | #cặp | AUC 1-feature | \|2·AUC−1\| | Hướng | q-value BH | MI | Lift decile trên |
|---|---:|---:|---:|---|---:|---:|---:|
| `current_ratio_min_window` | 244 | 0,048 | **0,904** | giá trị thấp ⇒ nhãn 1 | 0,0000 | **0,432** | 0,191 |
| `current_ratio_latest` | 244 | 0,074 | 0,852 | giá trị thấp ⇒ nhãn 1 | 0,0000 | 0,367 | 0,255 |
| `working_capital_to_assets` | 244 | 0,078 | 0,844 | giá trị thấp ⇒ nhãn 1 | 0,0000 | 0,336 | 0,128 |
| `debt_to_assets_latest` | 244 | 0,879 | 0,758 | giá trị cao ⇒ nhãn 1 | 0,0000 | 0,340 | 1,595 |
| `receivables_to_sales_latest` | 165 | 0,867 | 0,734 | giá trị cao ⇒ nhãn 1 | 0,0000 | 0,217 | 1,557 |
| `quick_ratio_latest` | 244 | 0,218 | 0,564 | giá trị thấp ⇒ nhãn 1 | 0,0000 | 0,172 | 0,255 |
| `debt_to_equity_latest` | 244 | 0,715 | 0,430 | giá trị cao ⇒ nhãn 1 | 0,8150 | 0,340 | 1,595 |
| `retained_to_assets_yoy` | 212 | 0,300 | 0,400 | giá trị thấp ⇒ nhãn 1 | 0,4427 | 0,153 | 0,551 |
| `total_assets_yoy_growth` | 212 | 0,315 | 0,370 | giá trị thấp ⇒ nhãn 1 | 0,0008 | 0,046 | 0,620 |
| `cash_to_assets_latest` | 244 | 0,337 | 0,326 | giá trị thấp ⇒ nhãn 1 | 0,0000 | 0,157 | 0,383 |

- **47/47** feature đều tính được AUC đơn biến; **11** feature có `|2·AUC−1| ≥ 0,30`; **11** feature còn ý
  nghĩa sau hiệu chỉnh BH.
- Nhận xét nghiệp vụ: tín hiệu mạnh nhất là **thanh khoản ngắn hạn ở mức xấu nhất trong 8 quý**
  (`current_ratio_min_window`, MI = 0,432) và **đòn bẩy** (`debt_to_assets_latest`). Điều này hợp lý về
  kinh tế: suy giảm tài chính của chuỗi bán lẻ biểu hiện trước hết ở khả năng thanh toán ngắn hạn suy
  yếu kéo dài, không phải ở một quý lợi nhuận xấu đơn lẻ.
- **Cảnh báo kỹ thuật rất dễ bị hỏi:** `debt_to_equity_latest` có AUC 0,715 nhưng điểm-biserial gần 0 và
  **không** ý nghĩa sau BH (q = 0,815) vì đuôi cực nặng (skew −14,9) chi phối Pearson, trong khi AUC
  không bị outlier chi phối. Đây là lý do nhóm **không** dùng p-value đơn lẻ để chọn đặc trưng.

### 2.3. Đa cộng tuyến và số chiều hiệu dụng

| Chỉ số | Giá trị | Ý nghĩa |
|---|---:|---|
| Cặp feature có `\|r\| ≥ 0,90` | 8 cặp | thông tin dư thừa trực tiếp |
| Số cụm tương quan (ngưỡng 0,9) | 7 cụm / 15 feature | các biến đo cùng một chiều kinh tế |
| **VIF > 10** | **33/47 cột** | đa cộng tuyến nghiêm trọng; cực đại **798,10** (`cash_to_assets_yoy`) |
| **Participation Ratio** `(Σλ)²/Σλ²` | **12,892** | không gian 47 chiều chỉ tương đương ~13 chiều thông tin |
| Thành phần cần cho 95% phương sai | 22 | phần còn lại gần như là nhiễu dư |

Hệ quả phương pháp luận (dùng xuyên suốt báo cáo): (i) **không** dùng RFE/biến chọn đơn biến vì nó sẽ
loại nhầm các biến tương quan cao nhưng hữu ích — thay bằng **ablation theo nhóm** (Chương 8); (ii) hệ
số của mô hình tuyến tính không đọc được như "trọng số nhân quả"; (iii) giá trị SHAP phải đọc kèm cụm
tương quan, vì SHAP chia đều "công" cho các biến tương quan với nhau.

### 2.4. Dịch chuyển phân phối train → test (chẩn đoán, KHÔNG dùng để chọn mô hình)

| Feature | Mean train | Mean test | KS | SMD | PSI |
|---|---:|---:|---:|---:|---:|
| `debt_to_assets_yoy` | 0,067 | −0,026 | **0,621** | −0,928 | 5,615 |
| `debt_to_equity_yoy` | 0,060 | −0,160 | 0,561 | −0,115 | 5,398 |
| `retained_to_assets_yoy` | −0,646 | 0,086 | 0,398 | 0,176 | 1,821 |
| `receivables_to_sales_latest` | 0,057 | 0,079 | 0,327 | 0,656 | 4,050 |
| `revenue_per_asset_latest` | 0,445 | 0,386 | 0,325 | −0,456 | 2,567 |
| `net_income_yoy_growth` | 0,688 | 0,551 | 0,279 | −0,033 | 0,637 |

**5 feature** vượt ngưỡng cảnh báo (KS ≥ 0,30 hoặc `|SMD| ≥ 0,50`). Đây là lý do nhóm **bắt buộc** báo
cáo kèm giao thức đánh giá theo thời gian (walk-forward) và cảnh báo drift khi triển khai: kết luận
trên test có thể phản ánh *phân phối mới*, không chỉ năng lực mô hình.

---

## CHƯƠNG 3: QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU

### 3.1. Xử lý giá trị khuyết thiếu

Bộ dữ liệu có **703 ô thiếu** (quý × chỉ tiêu) được để `null` đúng như SEC công bố — nhóm **không** điền
số cho đủ (kiểm chứng độc lập bằng `scripts/verify_provenance.py`: 0 phát hiện).

| Nhóm chỉ tiêu | Độ phủ | Ghi chú |
|---|---|---|
| Nhóm đầy đủ | 100% | `total_assets`, `revenue`, `current_assets`, `current_liabilities`, `inventory`, `cost_of_sales`, `selling_general_admin`, `stockholders_equity`, `retained_earnings` |
| `liabilities` | 39% quý (min 16% theo công ty) | ⇒ nợ phải trả được **suy ra** từ `TL = TA − SE`; đối chiếu 124 quý có tag: 122 khớp tuyệt đối, 2 lệch do restatement |
| `receivables`, `short_term_investments` | thấp và không đồng đều | các tỷ số liên quan bị impute nhiều |
| Mẫu có < 5 quý lịch sử | **32/324 mẫu (9,9%)** | feature YoY của các mẫu này là `NaN` |

Cách xử lý: `SimpleImputer(strategy='median')` đặt **bên trong** `Pipeline`, nên **trung vị chỉ được tính
trên train của từng lần fit** rồi mới áp sang validation/test. Impute trước khi chia tập là rò rỉ thống
kê kinh điển; ở đây bị chặn bằng cả thiết kế pipeline lẫn kiểm thử tự động.

**Một phát hiện EDA phải công bố — giá trị thiếu mang thông tin nhãn (MNAR):**

| Feature | #thiếu | Tỉ lệ nhãn 1 khi **thiếu** | khi **có** dữ liệu | Δ |
|---|---:|---:|---:|---:|
| `net_margin_latest` | 38 | 13,2% | 68,9% | −55,7 điểm % |
| `net_margin_min_window` | 38 | 13,2% | 68,9% | −55,7 điểm % |
| `operating_margin_latest` | 46 | 21,7% | 69,1% | −47,3 điểm % |

Nghĩa là "thiếu dữ liệu" ở đây **không ngẫu nhiên** mà tương quan mạnh với trạng thái lành mạnh (doanh
nghiệp khoẻ ít phải công bố chi tiết). Nhóm vẫn dùng median-impute (an toàn, không sinh thông tin) nhưng
**báo cáo hiện tượng này** thay vì im lặng, đồng thời kiểm chứng kết luận không phụ thuộc cách xử lý bằng
biến thể ablation `chỉ mẫu có ≥ 5 quý lịch sử` và `bỏ cột có độ phủ < 50%`.

### 3.2. Chuẩn hoá và co giãn thang đo

| Mô hình | Scaler | Công thức | Lý do |
|---|---|---|---|
| Logistic Regression | **StandardScaler** | `z = (x − μ_train)/σ_train` | mô hình tuyến tính tối ưu theo độ dốc, nhạy đơn vị đo |
| MLP | **StandardScaler** | như trên | lan truyền ngược theo gradient nên rất nhạy đơn vị đo |
| Random Forest, HistGradientBoosting | **không scale** (`none`) | — | mô hình cây chỉ quan tâm **thứ tự** giá trị tại điểm cắt; scale không đổi cây |

Các tham số `μ, σ` nằm trong `Pipeline` ⇒ chỉ học từ train. Nhóm còn **đối sánh thực nghiệm** với
`RobustScaler` (median/IQR), `PowerTransformer` (Yeo–Johnson) và `QuantileTransformer` (xếp hạng →
phân phối chuẩn) trong `scripts/experiment_preprocessing.py`, rồi mới ra khuyến nghị theo mô hình được
chốt (`reports/results/preprocessing_experiment.md`) — không giả định "scaler nào cũng như nhau".

### 3.3. Xử lý giá trị ngoại lai (Outlier)

**6/14 tỷ số có `|skew| > 1`**, riêng `debt_to_equity` có skew **−14,9** và **30,6%** giá trị nằm ngoài
khoảng IQR. `StandardScaler` lấy mean/std nên chính các cực trị này quyết định tỉ lệ scale ⇒ mô hình
tuyến tính bị kéo lệch. Nhóm cài `Winsorizer` clip theo ngưỡng **học từ train**:

- `method='iqr'`: `clip(x, Q1 − 1,5·IQR, Q3 + 1,5·IQR)`
- `method='p1p99'`: `clip(x, P1, P99)`
- `clip_share()` báo tỉ lệ giá trị thực sự bị clip, để biết mức can thiệp thật.

Winsorization được khảo sát như **một biến thể** và công bố rõ là **không dùng ở pipeline chính**
(`winsorize = none`): mô hình được chốt là mô hình **cây** (không nhạy scale/outlier), nên thêm một phép
biến đổi chỉ tăng độ phức tạp mà không cải thiện kiểm chứng — quyết định này có số liệu đối chứng trong
`reports/results/preprocessing_experiment.md`.

### 3.4. Đóng gói pipeline chuẩn mực

```
JSON SEC → 16 chỉ tiêu/quý → mẫu (lịch sử 8 quý + quý target + nhãn)
        → 47 feature → Pipeline( SimpleImputer(median) → [StandardScaler] → Model )
```

Toàn bộ nằm trong `sklearn.pipeline.Pipeline` (và `GridSearchCV`/cross-validation theo nhóm công ty tái
dùng đúng pipeline đó), nên **không phép biến đổi nào được học ngoài fold train**. Đây là "7 lớp kiểm
soát rò rỉ" ở `docs/BAO-CAO.md` §4.6, được khẳng định lại bằng kiểm thử tự động.

---

## CHƯƠNG 4: THIẾT LẬP VÀ HUẤN LUYỆN 4 HỌ MÔ HÌNH MACHINE LEARNING

Bốn họ mô hình được chọn để **khác nhau về cơ chế học**, không phải để "thử cho nhiều": tuyến tính (L2)
→ học kết hợp dạng đóng bao (*bagging*) → học kết hợp dạng tăng cường độ dốc (*boosting*) → mạng nơ-ron
phi tuyến **không** dựa trên cây. Toàn bộ siêu tham số nằm ở **một nguồn duy nhất**
`forecasting/models.py::HYPERPARAMS`; mô hình được chốt giữ **cấu hình mặc định** đã công bố (không dùng
cấu hình tinh chỉnh — lý do ở Chương 5.3).

| # | Mô hình | Cấu hình chính xác trong mã | Cơ chế học |
|---|---|---|---|
| 1 | **Logistic Regression** | `max_iter=2000`, `C=0.1` ⇒ điều chuẩn **L2/Ridge**, solver `lbfgs`, `random_state=42`; **không** đặt `class_weight` ở cấu hình triển khai | tuyến tính trong không gian log-odds của 47 feature đã chuẩn hoá; hệ số đọc được trực tiếp |
| 2 | **Random Forest** | `n_estimators=300`, `max_depth=6`, `min_samples_leaf=2`, **`class_weight='balanced_subsample'`**, `n_jobs=1`, `random_state=42` | **bagging**: 300 cây quyết định trên các mẫu bootstrap; mỗi điểm rẽ nhánh chỉ xét một tập con feature ngẫu nhiên ⇒ khử tương quan giữa các cây; `balanced_subsample` tính lại trọng số lớp **theo từng mẫu bootstrap của mỗi cây** |
| 3 | **HistGradientBoosting** | `max_iter=300`, `learning_rate=0.05`, `max_depth=3`, `l2_regularization=1.0`, `class_weight='balanced'`, `random_state=42` | **boosting**: học tuần tự, mỗi cây sửa sai số của các cây trước; đặc trưng được rời rạc hoá thành histogram để tăng tốc (cùng họ ý tưởng với LightGBM nhưng thuần scikit-learn) |
| 4 | **MLP Classifier** | `hidden_layer_sizes=32` (một lớp ẩn 32 nơ-ron), `alpha=1e-3` ⇒ **L2** trên trọng số, `learning_rate_init=1e-3`, solver **Adam**, `max_iter=3000`, **`early_stopping=True`**, `n_iter_no_change=30` | mạng nơ-ron truyền thẳng; biểu diễn **phi tuyến không dựa trên cây** (mặt quyết định trơn) nên bổ sung một giả thuyết học khác hẳn; bắt buộc scale |

**Kiểm soát overfitting được thiết kế ngay trong cấu hình** (Chương 6 sẽ đo lại): Random Forest bị chặn
độ sâu `max_depth=6` và buộc lá có ≥ 2 mẫu (không để cây mọc tự do trên 212 mẫu); HistGradientBoosting
dùng cây rất nông `max_depth=3`, học chậm `learning_rate=0.05`, điều chuẩn `l2_regularization=1.0`; MLP
đặt `alpha=1e-3` và **early stopping** với độ kiên nhẫn 30 vòng.

> **Ghi chú minh bạch (rất dễ bị hỏi):** `early_stopping` của `HistGradientBoostingClassifier` mặc định
> là `'auto'`; theo quy tắc của scikit-learn nó chỉ tự bật khi tập huấn luyện **> 10.000 mẫu**, mà ở đây
> chỉ có **212** mẫu ⇒ **không** kích hoạt. Việc chống overfitting cho mô hình này đến từ ba ràng buộc
> cấu trúc vừa nêu, và điều đó được kiểm chứng bằng số ở Chương 6.4 (gap train−val = 0,0260).

---

## CHƯƠNG 5: SO SÁNH HIỆU NĂNG VÀ LỰA CHỌN MÔ HÌNH TỐI ƯU

### 5.1. Bảng tổng hợp so sánh hiệu năng trên tập Test độc lập (N = 64)

| Mô hình | AUROC | AP (PR-AUC) | Brier | F1@0,5 | Bal.Acc@0,5 | F1@0,7879 | MCC@0,7879 | Thời gian huấn luyện | Độ trễ suy luận (1.000 dòng) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0,9828** | 0,9887 | **0,053** | 0,9351 | 0,9160 | 0,9143 | 0,8272 | **0,005 s** | **0,95 ms** |
| **Random Forest** | **0,9828** | **0,9908** | 0,058 | 0,9487 | **0,9291** | **0,9600** | **0,9039** | 0,465 s | 21,98 ms |
| HistGradientBoosting | 0,9767 | 0,9868 | 0,061 | 0,9367 | 0,9099 | 0,9067 | 0,7750 | 0,170 s | 10,61 ms |
| MLP (mạng nơ-ron) | 0,9706 | 0,9819 | 0,094 | 0,9136 | 0,8715 | 0,8947 | 0,7409 | 0,074 s | 1,19 ms |

*Cột Brier in 3 chữ số; hai cột cuối đo trên cùng máy, cùng quy trình (`reports/results/defense_models.md`).*

Bảng xếp hạng theo **giao thức nghiêm ngặt hơn** (*cross-company*, giữ trọn công ty ra khỏi train —
chống rò rỉ thực thể) thay đổi thứ tự một chút và là **căn cứ chọn mô hình thật của đồ án**:

| Mô hình | AP cross-company (OOF) | AUROC cross-company (OOF) | AUROC in-domain (test) | **Suy giảm khi đổi giao thức** |
|---|---:|---:|---:|---:|
| **Random Forest** | **0,9577** | **0,9431** | 0,9828 | −0,0397 |
| HistGradientBoosting | 0,9569 | 0,9404 | 0,9767 | −0,0363 |
| Logistic Regression | 0,9232 | 0,8896 | 0,9828 | −0,0932 |
| MLP (mạng nơ-ron) | 0,7792 | 0,7083 | 0,9706 | **−0,2623** |

### 5.2. Kết luận mô hình tốt nhất

**Random Forest là mô hình được chọn**, và điều này được quyết định **trước khi** nhìn test, theo quy tắc
đã công bố trong `reports/results/summary.json`:

> `max AP cross-company (GroupKFold trên train+validation) → best-F1(val) → AP(val) → AUROC(val) → gap overfit nhỏ nhất`

Random Forest thắng ở cả bốn chỉ số cốt lõi: **AP cross-company 0,9577** (cao nhất), **test AP 0,9908**,
**test F1 0,9600**, **MCC 0,9039**; đồng thời có Balanced Accuracy cao nhất (0,9291) và mức suy giảm khi
đổi sang cross-company nhỏ nhất trong nhóm mô hình cây (−0,0397).

### 5.3. Nhận xét vì sao Random Forest vượt trội — và ba điều phải nói cho trung thực

1. **Dữ liệu tài chính có nhiều ranh giới dạng ngưỡng bậc thang** (ví dụ "nợ/vốn chủ vượt mức nào thì
   rủi ro tăng vọt"). Cây quyết định biểu diễn trực tiếp dạng `x_j ≤ θ` nên mô hình hoá các ngưỡng này tự
   nhiên hơn; mạng nơ-ron với chỉ 212 mẫu train không đủ dữ liệu để "mượt hoá" các ngưỡng đó (MLP tụt
   mạnh nhất khi sang cross-company: −0,2623 AUROC).
2. **33 biến đa cộng tuyến (VIF > 10) không phá được Random Forest**, vì mỗi điểm rẽ nhánh chỉ xét một
   tập con feature ngẫu nhiên ⇒ rừng "trải rủi ro" qua nhiều biến tương quan thay vì dựa vào một biến.
   Với mô hình tuyến tính, đa cộng tuyến làm hệ số bất ổn định.
3. **`balanced_subsample` xử lý lệch lớp mà không cần sinh mẫu nhân tạo** — quan trọng vì mọi kỹ thuật
   resampling đều có nguy cơ tạo mẫu lai giữa các quý của cùng một công ty, tức hợp thức hoá rò rỉ cấp
   thực thể (Chương 9.3).

**Ba điều nói cho trung thực (không tô hồng kết quả):**

- **Lợi thế so với "mô hình một-biến" là rõ, nhưng so với bộ dự báo ngây thơ theo công ty thì KHÔNG có ý
  nghĩa thống kê.** Kiểm định DeLong cho ΔAUROC(RF − ticker-prior) = −0,0030 với **p = 0,7546**; bootstrap
  theo cặp cho ΔAP = +0,0061, CI 95% = [−0,0048; +0,0215], p = 0,344. Nói cách khác: trên 64 mẫu test,
  *chưa đủ bằng chứng* để khẳng định RF hơn baseline "nhớ mặt công ty".
- **RF không thắng mọi chỉ số:** trên Brier, Logistic Regression nhỉnh hơn (0,053 so với 0,058) ⇒ nếu mục
  tiêu là *hiệu chuẩn xác suất* thì mô hình tuyến tính có lợi thế.
- **Chi phí tính toán lớn hơn nhiều lần:** RF mất 0,465 s để huấn luyện và ~22 ms cho 1.000 dòng suy luận,
  tức **~93× thời gian huấn luyện** và **~23× độ trễ** so với Logistic Regression. Với bối cảnh chỉ chấm
  vài chục doanh nghiệp mỗi quý thì mức này chấp nhận được, nhưng khi mở rộng lên hàng nghìn doanh nghiệp
  đây là thông số phải cân nhắc (và là lý do đồ án đề xuất quy tắc chọn ngưỡng/chi phí thay vì "đổi mô
  hình to hơn").

---

## CHƯƠNG 6: CHI TIẾT TỪNG CLASS, MA TRẬN NHẦM LẪN & KIỂM TRA OVERFITTING

### 6.1. Ma trận nhầm lẫn trên mô hình tốt nhất (Random Forest)

Tập test độc lập: **N = 64** (38 dương thực tế). Hai điểm vận hành được báo cáo song song:

| Ngưỡng | TN | FP | FN | TP | Accuracy | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,5000 (quy ước) | 23 | 3 | 1 | 37 | 93,75% | 0,9250 | 0,9737 | 0,8846 |
| **0,7879 (vận hành, chọn trên val)** | **25** | **1** | **2** | **36** | **95,31%** | 0,9730 | 0,9474 | 0,9615 |

Accuracy tổng thể tại ngưỡng vận hành: `(TN + TP)/N = (25 + 36)/64 = 95,31%`.

**So với quy tắc "đoán lớp đa số" (62,35%)**, mức 95,31% thật sự có ý nghĩa — đây là lý do mọi bảng trong
báo cáo đều in kèm mốc đa số, vì với dữ liệu mất cân bằng, Accuracy đứng một mình là vô nghĩa.

### 6.2. Phân tích chi tiết theo từng lớp (Per-Class)

| Lớp | Số mẫu | Precision | Recall | F1 | Diễn giải nghiệp vụ |
|---|---:|---:|---:|---:|---|
| **Class 0 — lành mạnh** (ngưỡng vận hành) | 26 | 0,9259 | 0,9615 | 0,9434 | bắt đúng 25/26 doanh nghiệp khoẻ; chỉ 1 báo động giả |
| **Class 1 — suy giảm** (ngưỡng vận hành) | 38 | 0,9730 | 0,9474 | **0,9600** | bắt đúng 36/38 ca suy giảm; bỏ sót 2 ca |
| Class 0 (tại ngưỡng 0,5) | 26 | 0,9583 | 0,8846 | 0,9200 | — |
| Class 1 (tại ngưỡng 0,5) | 38 | 0,9250 | 0,9737 | 0,9487 | — |

Cơ chế xử lý lệch lớp của Random Forest ở đây là **`class_weight='balanced_subsample'`**: trọng số lớp
được tính lại **trên từng mẫu bootstrap của mỗi cây** (chứ không phải một trọng số duy nhất cho cả rừng),
nên mỗi cây con đều nhìn thấy hai lớp với mức quan trọng tương đương. Kết quả thực nghiệm cho thấy cơ chế
này **không** làm hỏng Precision của lớp đa số (vẫn 0,926) trong khi giữ Precision lớp thiểu số ở 0,973.

### 6.3. Phân tích lỗi chi tiết (Error Analysis)

Cả tập test chỉ có **3 mẫu sai** trên 64; repo lưu vết tới từng mẫu kèm giá trị feature
(`reports/results/error_cases.csv`) và có cả phân rã SHAP cục bộ (`reports/results/shap.json`):

| Mẫu | Thực tế | P(dự báo) | Loại lỗi | Bằng chứng feature (rút từ artifact) |
|---|---:|---:|---|---|
| `HD-2024Q2` | 1 | 0,7835 | **FN** (sát ngưỡng 0,788) | `current_ratio = 1,339`; `WC/TA = 0,104`; `net_margin = 9,9%`; `OCF/doanh thu = 15,1%`; lịch sử 37 quý ⇒ **các chỉ số quý đó vẫn dương** |
| `FIVE-2024Q3` | 1 | 0,1745 | **FN** (rõ rệt) | `current_ratio = 1,633`; `WC/TA = 0,108`; `net_margin = 4,0%`; `debt_to_assets = 0,599` ⇒ quý "trông khoẻ" |
| `DG-2025Q2` | 0 | 0,7975 | **FP** | `debt_to_assets = 0,751`; `net_margin = 3,8%`; `OCF/doanh thu = 8,1%` ⇒ **trông giống doanh nghiệp căng thẳng** |

**Đọc lỗi cho đúng bản chất:** hai ca FN đều thuộc doanh nghiệp mà **nhãn ở cấp công ty gần như bất
biến** (HD có 100% quý mang nhãn 1), nhưng tại chính quý đó các chỉ tiêu ngắn hạn lại hồi phục. Nghĩa là
mô hình sai vì **nhãn là trạng thái dài hạn của thực thể**, trong khi feature chỉ mô tả 8 quý gần nhất —
không phải vì "ranh giới quyết định mờ". Ca FP của DG cho thấy mặt trái: khi chỉ số đòn bẩy/biên lợi
nhuận xuống mức xấu, mô hình gán rủi ro cao dù nhãn là 0. Kết luận: **3/64 lỗi này không thể sửa bằng
tinh chỉnh siêu tham số**, phải sửa ở tầng **định nghĩa nhãn** (đúng như khuyến nghị P0 của đồ án).

### 6.4. Đánh giá hiện tượng Overfitting (Train vs Validation)

| Mô hình | Train AUROC | Val AUROC | **Gap AUROC** | Train F1 | Val F1 | Val AP | Brier (val) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0,9841 | 0,9654 | **0,0187** | 0,9354 | 0,9756 | 0,9869 | 0,0505 |
| **Random Forest** | 0,9995 | 0,9654 | **0,0342** | 0,9847 | 0,9756 | 0,9869 | 0,0459 |
| HistGradientBoosting | 1,0000 | 0,9740 | **0,0260** | 1,0000 | 0,9756 | 0,9894 | **0,0368** |
| MLP (mạng nơ-ron) | 0,9736 | 0,9870 | **−0,0134** | 0,9363 | 0,9130 | 0,9932 | 0,0790 |

Cách đọc đúng — rất quan trọng khi bị hỏi:

1. **Random Forest có gap 0,0342** (train 0,9995 ≈ 1,0; val 0,9654). Gap dương là hiện tượng "học vẹt"
   có thật, nhưng **được kiểm soát**: `max_depth=6` và `min_samples_leaf=2` chặn cây mọc tự do trên
   **212 mẫu**; nếu để mặc định `max_depth=None`, gap sẽ lớn hơn hẳn.
2. **MLP có gap âm (−0,0134)** — val tốt "hơn" train. Đây **không** phải điều kỳ diệu mà là hệ quả của
   `early_stopping=True` (dừng ở điểm tối ưu trên tập tách ra) cộng với **độ bất định của 32 mẫu
   validation**; nó nhắc rằng gap trên val chỉ đáng tin khi n ≥ vài trăm.
3. **Không mô hình nào underfitting:** cả bốn đều có AUROC train ≥ 0,97 và val ≥ 0,965 ⇒ năng lực biểu
   diễn đủ dùng; đường học (*learning curve*) còn chỉ ra dữ liệu **chưa bão hoà** (gap train−val tại cỡ
   dữ liệu lớn nhất = **0,1049**), tức thêm dữ liệu vẫn cải thiện được — đây chính là lý do khuyến nghị
   P1 "mở rộng lên 50–100 doanh nghiệp" của đồ án.
4. **Brier trên validation** tốt nhất thuộc HistGradientBoosting (0,0368), kế đến RF (0,0459) — tức
   *hiệu chuẩn* của hai mô hình cây tốt hơn MLP (0,0790); điều này hỗ trợ việc dùng trực tiếp xác suất
   để ra quyết định theo chi phí (Chương 7.3).

---

# PHẦN II — GIẢI TRÌNH CHUYÊN SÂU 10 CÂU HỎI PHẢN BIỆN CỦA HỘI ĐỒNG

## CHƯƠNG 7: CƠ SỞ TOÁN HỌC CỦA BỘ CHỈ SỐ & TỐI ƯU HOÁ NGƯỠNG THEO CHI PHÍ

### 7.1. Định nghĩa toán học các chỉ số chạy trong mã nguồn (số tại ngưỡng vận hành t = 0,788)

Ma trận nhầm lẫn trên test (N = 64): **TN = 25, FP = 1, FN = 2, TP = 36**.

| Chỉ số | Công thức | Thay số | Kết quả |
|---|---|---|---|
| Accuracy *(chỉ dùng chẩn đoán)* | `(TP + TN)/N` | `(36+25)/64` | 0,9531 |
| Recall (TPR, độ nhạy) | `TP/(TP + FN)` | `36/(36+2)` | **0,9474** |
| Precision (độ chuẩn xác) | `TP/(TP + FP)` | `36/(36+1)` | **0,9730** |
| Specificity (TNR) | `TN/(TN + FP)` | `25/(25+1)` | 0,9615 |
| FPR (báo động giả) | `FP/(TN + FP)` | `1/26` | 0,0385 |
| NPV | `TN/(TN + FN)` | `25/27` | 0,9259 |
| F1 | `2·P·R/(P + R)` | `2·0,9730·0,9474/(0,9730+0,9474)` | **0,9600** |
| F-macro | `(F1₀ + F1₁)/2` | `(0,9434 + 0,9600)/2` | 0,9517 |
| F-weighted | `Σ (n_c/N)·F1_c` | `(26·0,9434 + 38·0,9600)/64` | 0,9533 |
| F-beta (β = 2, dùng ở lab) | `(1+β²)·P·R/(β²·P + R)` | — | `reports/imbalance/techniques.md` |
| **Balanced Accuracy** | `(TPR + TNR)/2` | `(0,9474 + 0,9615)/2` | **0,9545** |
| **MCC** | `(TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | `(36·25 − 1·2)/√(37·38·26·27)` | **0,9039** |
| AUROC | `P(S₊ > S₋) + ½·P(S₊ = S₋)` = `(1/n₊n₋)·ΣΣ[𝟙(sᵢ>sⱼ) + ½·𝟙(sᵢ=sⱼ)]` | — | **0,9828** |
| AP (PR-AUC) | `Σ_k (R_k − R_{k−1})·P_k` | — | **0,9908** |
| Brier | `(1/N)·Σ(pᵢ − yᵢ)²` | — | **0,0580** |
| Brier Skill Score | `1 − BS/BS_null` với `BS_null = p̄(1−p̄) = 0,2412` | `1 − 0,0580/0,2412` | **0,7594** |

### 7.2. Luận giải bản chất kỹ thuật (phần dễ bị hỏi nhất)

**(a) Vì sao dùng Balanced Accuracy mà không dùng G-mean?** — *Trả lời thẳng: mã nguồn **không** dùng
G-mean; đã kiểm tra toàn repo, không có dòng nào tính `√(TPR·TNR)`.*

- Về toán: `Balanced Accuracy = (TPR + TNR)/2` là **trung bình cộng**, còn `G-mean = √(TPR·TNR)` là **trung
  bình nhân**. Trung bình nhân sụp rất nhanh về 0 khi *một trong hai* vế tiến về 0 (ví dụ TNR = 0,05 thì
  G-mean = √(0,95·0,05) = 0,218 trong khi BA = 0,50) — hữu ích như một "cảnh báo đỏ" cho mất cân bằng
  cực đoan.
- Về bối cảnh đồ án: mức lệch lớp ở corpus thật chỉ **IR = 1,46–1,66** nên hai chỉ số gần như song hành;
  BA lại **có sẵn trong scikit-learn** (`balanced_accuracy_score`) nên dùng thống nhất cho cả pipeline
  chính lẫn lab, không phải tự cài công thức rồi phát sinh chênh số giữa hai nơi.
- Bù cho trường hợp cực đoan: ở lab mất cân bằng **98/2** và **1:50**, nhóm dùng thêm **F-beta với β = 2**
  (coi trọng Recall gấp đôi Precision) và **PR-AUC** thay vì dựa vào một chỉ số duy nhất — cách này chặt
  hơn cả G-mean.

**(b) Vì sao ưu tiên AP (PR-AUC) hơn ROC-AUC?**

- ROC-AUC tích phân theo `FPR = FP/(TN + FP)`. Khi lớp âm chiếm áp đảo, mẫu số `TN + FP` rất lớn nên cải
  thiện hàng trăm ca dương tính thật chỉ làm FPR nhích vài phần nghìn ⇒ ROC-AUC **lạc quan giả tạo**. AP
  tích phân trên **Recall** với trọng số Precision, mỗi bước đều liên quan trực tiếp tới lớp quan tâm.
- Ở **corpus thật**, lớp dương (suy giảm) lại là lớp **đa số** (62,35%) nên hiệu ứng trên *nhẹ hơn* — nhóm
  nói rõ điều này thay vì phóng đại. AP vẫn được chọn làm **tiêu chí chọn mô hình** vì: (i) AP là thước đo
  phù hợp khi mục tiêu là xếp hạng rủi ro; (ii) trên hai bộ giả lập của lab (95/5 và 1:50), chính AP mới
  phân biệt được các phương pháp — ROC-AUC ở đó gần như không đổi (95/5: nhóm Baseline có PR-AUC 0,4786
  nhưng ROC-AUC tới 0,9210).
- Bằng chứng ngay trên corpus thật: baseline `single_feature[debt_to_assets_latest]` đạt ROC-AUC 0,8887
  nhưng AP chỉ 0,9379, trong khi RF đạt 0,9828 / 0,9908 — AP **khoét sâu** vào vùng xác suất cao, đúng
  vùng dùng để ra quyết định.

### 7.3. Tối ưu ngưỡng quyết định theo hàm chi phí thực tế

**Hàm chi phí kỳ vọng.** Với một mẫu `x`, đặt `η = P(y=1|x)`; bỏ sót một doanh nghiệp suy giảm tốn kém
hơn nhiều lần một báo động nhầm (trong mã: `C_FN = 5`, `C_FP = 1`):

```
E[Cost | x] = C_FN · η · 𝟙[p̂ < t]  +  C_FP · (1 − η) · 𝟙[p̂ ≥ t]
```

**Dẫn xuất ngưỡng Bayes.** Quyết định dự đoán lớp 1 khi và chỉ khi chi phí kỳ vọng của nó không lớn hơn:

```
C_FN · η  ≥  C_FP · (1 − η)
⟺  η · (C_FN + C_FP)  ≥  C_FP
⟺  p* = C_FP / (C_FN + C_FP) = 1 / (1 + C_FN/C_FP)
```

Với `C_FN/C_FP = 5` ⇒ **p\* = 1/6 ≈ 0,1667** (dạng odds: `η/(1−η) ≥ C_FP/C_FN = 0,2`). Ngưỡng này **không
phụ thuộc tỉ lệ lớp**, chỉ phụ thuộc tỉ lệ chi phí.

**Ngưỡng vận hành của đồ án** `t = 0,7879` được chọn bằng cực đại F1 trên **validation** (không dùng test).
Áp lên test và đối chiếu với ngưỡng lý thuyết:

| Ngưỡng áp lên test (N = 64) | FN | FP | Chi phí `5·FN + 1·FP` | Ghi chú |
|---|---:|---:|---:|---|
| **0,1667 = p\* (Bayes lý thuyết)** | 0 | 15 | **15,0** | tệ nhất — giải thích bên dưới |
| 0,5000 (quy ước) | 1 | 3 | 8,0 | |
| **0,7879 (vận hành, chọn trên val)** | 2 | 1 | **11,0** | ngưỡng triển khai |
| 0,7835 (argmin **trên chính test**) | 1 | 1 | 6,0 | chỉ là cận dưới lạc quan, **in-sample** |

**Ba lý do ngưỡng Bayes lý thuyết KHÔNG cho chi phí thấp nhất ở đây — câu trả lời "ăn điểm":**

1. `p*` là ngưỡng trên **xác suất hậu nghiệm thật η**, không phải trên điểm số của mô hình. Xác suất của
   Random Forest chưa hiệu chuẩn hoàn hảo: ECE ≈ **0,109**, riêng dải [0,7–0,8) mô hình dự báo trung bình
   0,783 nhưng tần suất thực tế chỉ 0,571 ⇒ "cắm p̂ vào luật Bayes" không còn tối ưu.
2. Hàm chi phí thực nghiệm là **hàm bậc thang** (chỉ đổi giá trị tại các điểm xác suất có trong dữ liệu)
   nên `argmin` là một **khoảng**; giá trị 6,0 đạt được bằng cách tối ưu **trên chính tập test** — đó là
   con số *in-sample*, không phải hiệu năng triển khai.
3. Với n = 64, chênh lệch chi phí giữa các ngưỡng do **1–2 mẫu** quyết định: mẫu `HD-2024Q2` có
   `p = 0,78348` nằm sát ngay ngưỡng 0,7879 nên đổi ngưỡng một chút là đổi 5 đơn vị chi phí.

⇒ Quy trình đúng mà đồ án áp dụng: **chọn ngưỡng trên validation** (được 0,7879, chi phí val = 5,0) rồi
**chỉ đo** trên test (11,0). Muốn dùng đúng ngưỡng Bayes `p* = 1/6` thì phải **hiệu chuẩn xác suất trước**
(Platt/isotonic, fit trên train/validation) — chính là khuyến nghị P2 của đồ án.

---

## CHƯƠNG 8: HỆ THỐNG 47 ĐẶC TRƯNG MỚI & THỰC NGHIỆM BÓC TÁCH THEO NHÓM (GROUP ABLATION)

### 8.1. Đề xuất đặc trưng mới ngoài baseline

Baseline "1 chỉ tiêu" trong đồ án dùng **một** tỷ số duy nhất (`debt_to_assets_latest`, AUROC test 0,8887).
Hệ thống đề xuất mở rộng thành **47 cột trong 6 khối logic**, mỗi khối trả lời một câu hỏi nghiệp vụ khác
nhau — thay vì ném thêm biến vào mô hình:

| Khối | #cột | Đại diện | Câu hỏi nghiệp vụ mà khối trả lời |
|---|---:|---|---|
| `ratios_latest` | 14 | `current_ratio_latest`, `debt_to_assets_latest`, `net_margin_latest` | *Hiện trạng* quý mới nhất đã công bố ra sao? |
| `ratios_yoy` | 14 | `current_ratio_yoy`, `debt_to_assets_yoy` | Hiện trạng đó **tốt lên hay xấu đi** so với cùng kỳ năm trước? |
| `growth` | 10 | `revenue_yoy_growth`, `operating_cash_flow_yoy_growth`, `total_assets_yoy_growth` | Quy mô kinh doanh đang **co lại** không? |
| `structure` | 1 | `working_capital_to_assets = (CA − CL)/TA` | Cơ cấu vốn lưu động trên tổng tài sản |
| `stress` | 2 | `negative_ocf_streak`, `negative_ni_streak`, `distress_quarters_in_window`, `revenue_cv` | Căng thẳng **kéo dài bao nhiêu quý**? Doanh thu biến động thế nào? |
| `path` | 6 | `current_ratio_min_window`, `net_margin_min_window`, `ocf_to_sales_min_window`, `revenue_drawdown_window` | **Quỹ đạo 8 quý**: mức xấu nhất từng chạm, mức giảm so với đỉnh doanh thu |

**Điểm mới thực sự (nói được ngay khi bị hỏi "có gì mới ngoài baseline?"):** ba nhóm
`ratios_yoy` + `stress` + `path` biến dữ liệu từ **một báo cáo tĩnh** thành **một quỹ đạo theo thời gian**.
Cụ thể, khối `path` lấy **cực trị xấu nhất** và **độ bền** của 8 quý (`min`, drawdown, chuỗi âm liên tiếp)
— đúng trực giác tài chính rằng suy giảm là một *quá trình*, không phải một điểm. Việc nhóm `path` có mặt
là **kết quả đo**, không phải suy đoán: `scripts/probe_features.py` cho thấy nhóm này cộng thêm
**+0,03…+0,12 AUROC cross-company** so với chỉ dùng giá trị quý mới nhất, và biến quan trọng nhất toàn hệ
thống theo SHAP cũng như theo liên hệ đơn biến chính là `current_ratio_min_window` (Chương 2.2).

### 8.2. Phương pháp luận Group Ablation — vì sao **không** dùng RFE ở bài toán này

RFE (*Recursive Feature Elimination*) là kỹ thuật chuẩn, nhưng với **bộ dữ liệu này** nó có ba nhược điểm
có thể chứng minh bằng số liệu:

1. **Đa cộng tuyến làm RFE chọn "người anh em sinh đôi" một cách tuỳ ý.** Có **33/47** cột có VIF > 10 và
   **8 cặp** feature có `|r| ≥ 0,90`; chỉ cần loại cột này thì cột tương quan với nó lập tức "đội giá trị
   lên". RFE vì thế thường **loại nhầm biến hữu ích** và giữ lại biến trùng thông tin — kết quả thay đổi
   theo từng lần chạy mà ý nghĩa nghiệp vụ không đổi.
2. **Cỡ mẫu nhỏ (212 mẫu train, 47 cột).** Loại dần từng biến đòi hỏi hàng chục lần refit trên 212 mẫu;
   độ bất định của thứ hạng importance lớn hơn khoảng cách giữa các biến ⇒ "xếp hạng" mất ý nghĩa.
3. **Không trả lời được câu hỏi của Hội đồng.** RFE trả lời *"bỏ cột nào thì điểm giảm?"*, còn câu hỏi ở
   đây là *"bỏ một KHỐI THÔNG TIN nghiệp vụ (ví dụ toàn bộ quỹ đạo 8 quý) thì mô hình mất gì?"* — cần
   ablation theo nhóm.

**Cách làm (đã cài trong `scripts/analyze.py`, artifact `analysis.json::ablation`):** giữ nguyên mô hình,
chia tập, seed và mọi tiền xử lý; **chỉ thay đổi tập cột đầu vào** theo từng biến thể (bỏ trọn một khối),
rồi đo lại AUROC trên validation và test. Mỗi biến thể là một phép so sánh có đối chứng, không phải một
phân tích tự do.

**Khác biệt cốt lõi so với baseline:** baseline "ném toàn bộ biến thô" và **không** kiểm chứng biến nào
đóng góp gì; Group Ablation **đo đóng góp biên của từng khối thông tin** theo cùng một giao thức đánh giá,
nhờ đó biết được đâu là nút thắt thật của bài toán.

### 8.3. Kết quả thực nghiệm bóc tách trên Random Forest

| Biến thể | #feature | Val AUROC | Test AUROC | Test F1 | Δ Test AUROC so với đủ 47 |
|---|---:|---:|---:|---:|---:|
| **Đủ 47 feature** | 47 | 0,9654 | **0,9828** | 0,9487 | — |
| Bỏ `ratios_latest` (14) | 33 | 0,9524 | 0,9656 | 0,9487 | −0,0172 |
| Bỏ `ratios_yoy` (14) | 33 | 0,9740 | 0,9737 | 0,9487 | −0,0091 |
| Bỏ `growth` (10) | 37 | 0,9697 | 0,9848 | 0,9487 | +0,0020 |
| Bỏ `structure` (1) | 46 | 0,9610 | 0,9818 | 0,9487 | −0,0010 |
| Bỏ `stress` (2) | 45 | 0,9654 | 0,9858 | 0,9487 | +0,0030 |
| Bỏ `path` (6) | 41 | 0,9610 | 0,9767 | 0,9487 | −0,0061 |
| Bỏ cột có độ phủ < 50% | 47 | 0,9654 | 0,9828 | 0,9487 | 0,0000 |
| Chỉ giữ mẫu có ≥ 5 quý lịch sử | 47 | 0,9654 | 0,9868 | 0,9487 | +0,0040 |

**Ba kết luận định lượng (đây là nội dung trả lời câu hỏi "tại sao loại bỏ từ từ"):**

1. **Loại bỏ bất kỳ khối nào cũng không làm Test AUROC lệch quá 0,003 so với đường cơ sở** (trừ khối
   `ratios_latest` −0,0172 — hợp lý vì đó là 14 chỉ số hiện trạng cốt lõi). Riêng `ratios_latest` là khối
   **duy nhất** có đóng góp âm rõ rệt khi bị bỏ đi ⇒ mọi khối còn lại đều **thay thế được cho nhau**, dấu
   hiệu đặc trưng đã **bão hoà**.
2. **Không có khối nào đẩy điểm lên đáng kể** ⇒ nút thắt của bài toán **không** nằm ở feature engineering.
   Đây là bằng chứng trực tiếp cho phát hiện trung tâm ở Chương 10.2: nút thắt nằm ở **cấu trúc thực thể**
   (nhãn gần như là thuộc tính của công ty).
3. **Hai biến thể có điểm nhích lên (bỏ `growth` +0,0020; bỏ `stress` +0,0030; chỉ giữ mẫu ≥ 5 quý
   +0,0040) là NHIỄU, không phải cải thiện.** Chênh lệch này nhỏ hơn độ bất định của 64 mẫu test (CI 95%
   của AUROC là [0,947; 1,000], bề rộng 0,053) và nhỏ hơn sai số giữa các seed. Nhóm **không** bỏ khối nào
   để "khoe điểm cao hơn" — nếu bỏ đi rồi báo cáo 0,9868 sẽ là *data snooping* trên chính tập test. Đây là
   lựa chọn có chủ đích và nên được nêu trước khi bị hỏi.

---

## CHƯƠNG 9: THỰC NGHIỆM CÂN BẰNG DỮ LIỆU & ĐỐI CHỨNG LITESVM

### 9.1. Bản đồ 4 cấp độ kỹ thuật xử lý mất cân bằng (đã cài đặt và kiểm thử)

| Cấp độ | Kỹ thuật | Cài đặt trong đồ án | Kiểm chứng |
|---|---|---|---|
| **1. Cấp dữ liệu** (data-level) | Random Oversampling, **SMOTE**, Borderline-SMOTE, ADASYN, Random Undersampling, **Tomek Links**, **ENN** | `imbalance_lab/samplers.py` + `imblearn.*`, chạy **trong** `imblearn.pipeline.Pipeline` | `test_oversamplers_reach_target_ratio`, `test_undersampler_reaches_target_ratio`, `test_cleaning_techniques_keep_minority_untouched`, `test_samplers_never_mutate_inputs` |
| **2. Cấp thuật toán** (algorithm-level) | `class_weight='balanced'`, **`scale_pos_weight`**, **Focal Loss** (custom objective) | `models.BalancedWeightClassifier`, `ScalePosWeightClassifier`, `imbalance_lab/losses.py` | `test_balanced_class_weight_uses_fit_labels`, `test_scale_pos_weight_uses_only_given_fold_labels`, `test_gradient_matches_finite_difference`, `test_gamma_zero_reduces_to_weighted_bce` |
| **3. Cấp tập hợp** (ensemble-level) | **Balanced Random Forest**, EasyEnsemble, Balanced Bagging, RUSBoost | `imblearn.ensemble.*`, `RobustRUSBoost` (dự phòng khi AdaBoost từ chối fit ở 1:50) | `TestCatalogCoverage`, `TestNoLeakageInCatalog` |
| **4. Cấp hậu xử lý** (post-processing) | **Threshold moving** theo chi phí / theo đường PR | `imbalance_lab/thresholds.py::tune_thresholds_from_pr_curve` (3 chế độ: `best_f1`, `best_cost`, `min_precision`) | `TestPRThresholds` (5 test: PR-AUC khớp sklearn, ngưỡng nằm trên đường cong, F1 = max F1 của đường cong, chi phí tối thiểu, min-precision đạt mục tiêu) |

**Chống rò rỉ ở mọi cấp:** mọi sampler nằm trong `imblearn.pipeline.Pipeline` ⇒ `fit_resample` chỉ chạy
trên train của fold; trọng số lớp/Focal Loss được tính **trong `fit`** từ nhãn nhận được; ensemble lấy
mẫu **bên trong** `fit`. Kiểm chứng tự động: **11/11 phương pháp PASS** ở lần chạy holdout và **5/5 fold
PASS** ở StratifiedKFold (`reports/benchmark_imbalanced.md`, `_cv.md`).

### 9.2. Non-E Mode vs E-Mode vs Baseline — kết quả đo trên hai bộ dữ liệu

**Định nghĩa dùng thống nhất trong báo cáo:**

- **Non-E Mode (Non-Ensemble):** can thiệp **tĩnh, một lần, ngoài mô hình** — tái lấy mẫu ở bước tiền xử
  lý (SMOTE, RUS, SMOTE+Tomek) hoặc đổi trọng số hàm mất mát, rồi đưa vào **một** mô hình đơn.
- **E-Mode (Ensemble):** can thiệp **động, bên trong từng mô hình con** — mỗi cây/ mỗi vòng boosting tự
  lấy mẫu cân bằng (Balanced Random Forest, EasyEnsemble, RUSBoost).

**(a) Trên bộ giả lập 95/5 (n = 10.000, test 2.000) — `reports/benchmark_imbalanced.md`:**

| Nhóm | #phương pháp | PR-AUC (tb) | F1 thiểu số (tb) | Balanced Acc (tb) | Thời gian (tb, s) |
|---|---:|---:|---:|---:|---:|
| **Baseline** (không can thiệp) | 2 | **0,4786** | **0,3582** | 0,6245 | 0,390 |
| **Non-E Mode** (data/cost) | 6 | 0,3211 | 0,3305 | 0,7111 | **0,284** |
| **E-Mode** (ensemble cân bằng) | 3 | 0,3398 | 0,3304 | **0,7352** | 1,417 |

Kết quả chi tiết theo từng chỉ số (để trả lời "kết quả Non-E, E-Mode, Class Weight ra sao?"):

| Phương pháp (đại diện) | Nhóm | ROC-AUC | PR-AUC | F1 thiểu số | Bal. Acc | Thời gian (s) |
|---|---|---:|---:|---:|---:|---:|
| XGBoost (mặc định, không can thiệp) | Baseline | **0,9210** | **0,6740** | **0,5935** | 0,7162 | 0,751 |
| Logistic Regression (không can thiệp) | Baseline | 0,7443 | 0,2833 | 0,1228 | 0,5328 | 0,029 |
| **SMOTE + XGBoost** | Non-E | 0,8963 | 0,6235 | 0,5536 | 0,7777 | 0,782 |
| **XGBoost + `scale_pos_weight` động** | Non-E (cost) | 0,9164 | 0,6579 | 0,5755 | 0,7759 | 0,600 |
| **Logistic + `class_weight='balanced'`** | Non-E (cost) | 0,7742 | 0,1480 | 0,2016 | 0,7075 | **0,014** |
| SMOTE + Logistic | Non-E | 0,7717 | 0,1791 | 0,2201 | 0,6468 | 0,030 |
| RandomUnderSampler + Logistic | Non-E | 0,7742 | 0,1701 | 0,2251 | 0,6413 | **0,009** |
| **BalancedRandomForest** | E-Mode | 0,9206 | 0,5441 | 0,4503 | **0,8212** | 1,048 |
| EasyEnsemble | E-Mode | 0,8183 | 0,2288 | 0,2297 | 0,7479 | 1,855 |
| RUSBoost | E-Mode | 0,8189 | 0,2465 | 0,3113 | 0,6364 | 1,350 |

**Đọc bảng cho đúng:** E-Mode nâng **Balanced Accuracy** cao nhất (0,8212 với BalancedRandomForest) vì nó
buộc mô hình con phải thấy cả hai lớp; nhưng **PR-AUC vẫn thua baseline XGBoost** (0,5441 so với 0,6740) và
thời gian **gấp ~5 lần**. Nói cách khác: can thiệp cân bằng *dịch chuyển điểm vận hành* (tăng Recall lớp
thiểu số, giảm Precision) chứ **không tạo thêm năng lực xếp hạng**. Ở StratifiedKFold 5 fold,
BalancedRandomForest là phương pháp **ổn định nhất** (ROC-AUC 0,9170 ± 0,0069; PR-AUC 0,5221 ± 0,0166),
còn RUSBoost kém ổn định nhất (0,7561 ± 0,0335).

**(b) Trên corpus THẬT của đồ án (GroupKFold 4 theo công ty) — `reports/results/imbalance_real.md`:**

| Kỹ thuật | Nhóm | AP (OOF) | AUROC (OOF) | F1* trên OOF | IR trước → sau |
|---|---|---:|---:|---:|---:|
| **smote_enn** | hybrid | **0,9683** | 0,9512 | 0,9158 | 1,32 → 1,12 |
| none (đối chứng) | baseline | 0,9588 | 0,9426 | 0,9419 | — |
| class_weight | algorithm | 0,9569 | 0,9404 | 0,9419 | — |
| random_under | undersampling | 0,9561 | 0,9267 | 0,9265 | 1,32 → 1,00 |
| smote | oversampling | 0,9502 | 0,9500 | 0,9311 | 1,32 → 1,00 |
| tomek_links | undersampling | 0,9428 | 0,9160 | 0,9159 | 1,32 → 1,23 |

Kiểm định cặp cho kỹ thuật tốt nhất: **ΔAP = +0,0095**, CI 95% = [−0,0054; +0,0269], **p = 0,21**;
ΔAUROC (DeLong) = +0,0087, p = 0,474 ⇒ **chưa đủ căn cứ** khẳng định hơn đối chứng. Đây là kết quả quan
trọng: trên dữ liệu thật, **không** kỹ thuật cân bằng nào tạo ra cải thiện có ý nghĩa thống kê.

### 9.3. Phương pháp đơn lẻ vs phương pháp kết hợp (hybrid) — và vì sao pipeline chính KHÔNG resampling

**(a) Hạn chế định lượng của từng phương pháp đơn lẻ** (số lấy từ 95/5, 1:50 và corpus thật):

| Phương pháp đơn lẻ | Hạn chế | Bằng chứng số |
|---|---|---|
| **SMOTE thuần** | nội suy mẫu mới ở vùng chồng lấn giữa hai lớp ⇒ mẫu tổng hợp rơi vào vùng đa số, làm **giảm Precision** và méo thứ hạng xác suất | 95/5: SMOTE+LR có Precision **0,1474** và PR-AUC 0,1791 (so với LR trơn 0,2833); corpus thật: AP 0,9502 < đối chứng 0,9588 |
| **Undersampling thuần** | vứt bỏ thông tin lớp đa số ⇒ mô hình mất khả năng phân biệt ở vùng biên | 95/5: RUS+LR PR-AUC **0,1701** (thấp nhất nhóm Non-E); corpus thật: `random_under` AP 0,9561 < 0,9588 |
| **Class weight thuần** | đẩy Recall lên nhưng **bùng nổ False Positive**; xác suất đầu ra **không còn là xác suất** (bị đổi thang) | 95/5: LR + `class_weight='balanced'` có Recall 0,7170 (từ 0,0660) nhưng Precision chỉ **0,1173** ⇒ PR-AUC 0,1480 |
| **Ensemble cân bằng thuần** | hiệu chuẩn kém, chi phí thời gian cao | 95/5: EasyEnsemble 1,855 s cho PR-AUC 0,2288; BalancedRF F1@0,5 chỉ 0,345 ⇒ cần threshold-moving (+0,317 F1 khi hạ ngưỡng) |

**(b) Sức mạnh của phương pháp kết hợp (hybrid) = hai can thiệp BÙ TRỪ nhau:**
resampling đưa thêm mẫu thiểu số vào vùng khó, còn dọn biên (Tomek/ENN) hoặc trọng số lớp chỉnh lại đúng
chỗ mô hình còn yếu — thay vì nhân bản cả nhiễu.

| Nhóm | #kỹ thuật | PR-AUC (tb) | Δ so với baseline | F1 (tb) | Recall (tb) | FPR (tb) | Giây/fold |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** (không xử lý) | 1 | **0,9474** | 0 | 0,8714 | 0,7722 | **0,0000** | 1,03 |
| **Đơn lẻ** (12 kỹ thuật) | 12 | 0,9104 | **−0,0370** | 0,8042 | 0,8597 | 0,0091 | 1,12 |
| **Kết hợp** (4 kỹ thuật) | 4 | 0,9229 | **−0,0245** | 0,8307 | **0,9051** | 0,0087 | 1,29 |

So từng cặp "hybrid vs đơn lẻ tốt nhất" (bộ 1:50, `reports/experiment/summary.md`):

| Hybrid | Đơn lẻ tốt nhất | ΔPR-AUC | ΔF1 | ΔRecall | Kết luận |
|---|---|---:|---:|---:|---|
| `smote_enn` (SMOTE + dọn ENN) | `smote` | **+0,0033** | −0,0062 | +0,0000 | **HYBRID tốt hơn** ở PR-AUC |
| `smote_tomek` (SMOTE + Tomek) | `smote` | 0,0000 | 0,0000 | +0,0000 | hoà |
| `smote_class_weight` | `smote` | −0,0007 | −0,0062 | +0,0000 | đơn lẻ nhỉnh |
| `rusboost` (RUS + boosting) | `balanced_rf` | −0,0554 | −0,0676 | +0,0000 | đơn lẻ tốt hơn rõ |

Chỉ số tốt nhất theo từng mục tiêu (1:50): PR-AUC → **`smote_enn` 0,9571** (+0,0097 so với baseline);
F1 → **`smote` 0,9404** (+0,069); Recall → **`balanced_rf` 0,9241** (+0,152); FPR thấp nhất → baseline,
`tomek`, `enn` (0,0000); **FPR cao nhất (phải cảnh báo) → `easy_ensemble` 0,0581**.

**Quy tắc kết luận (đúng và dùng được ngay khi bị hỏi):** 12/16 kỹ thuật đơn lẻ và 3/4 kỹ thuật kết hợp
**không vượt** baseline về PR-AUC; cải thiện chỉ đáng tin khi **lớn hơn độ lệch chuẩn giữa các fold**. Vì
vậy thứ tự ưu tiên đúng là: **(1)** luôn có baseline; **(2)** xếp hạng theo PR-AUC (không phụ thuộc ngưỡng);
**(3)** kiểm tra F1/Recall **và** FPR; **(4)** tối ưu ngưỡng theo chi phí trên xác suất out-of-fold;
**(5)** chỉ giữ kỹ thuật khi mức cải thiện vượt độ bất định.

**(c) Vì sao pipeline CHÍNH của đồ án cố ý không dùng resampling:**

1. **Mất cân bằng ở corpus thật rất nhẹ (IR = 1,66).** Không có "bệnh" để chữa: mô hình đã học được cả hai
   lớp, và sửa tỉ lệ lớp chỉ đổi điểm vận hành — việc mà **threshold moving** làm tốt hơn và rẻ hơn nhiều
   (Chương 7.3).
2. **Resampling ở dữ liệu này sẽ HỢP THỨC HOÁ rò rỉ cấp thực thể.** Mẫu được nhóm theo *quý của cùng một
   công ty*: SMOTE nội suy giữa hai quý của **cùng công ty**, nên điểm tổng hợp sinh ra nằm ngay trong
   "vùng" của thực thể đó. Khi thực thể ấy đã có trong train, mẫu tổng hợp không mang thông tin mới mà chỉ
   làm mô hình **tự tin hơn** vào việc nhận diện công ty. Đây đúng là loại rò rỉ mà đồ án đang đo lường.
3. **Trọng số lớp là can thiệp rẻ và minh bạch hơn:** Random Forest dùng `balanced_subsample`,
   HistGradientBoosting dùng `class_weight='balanced'`; cả hai tính trọng số **trong `fit`** từ nhãn train
   (kiểm chứng bằng `test_balanced_class_weight_uses_fit_labels`), không đụng tới dữ liệu.
4. **Nhưng nhóm KHÔNG phủ nhận resampling:** toàn bộ 17 pipeline vẫn được cài đặt, kiểm thử và chạy công
   khai trong `imbalance_lab/` + `imbalance_experiment/`, kèm cả kết quả **trên dữ liệu thật**
   (`imbalance_real.md`). Kết luận "không dùng" là **kết quả đo**, không phải định kiến.

### 9.4. Đối chứng **LiteSVM** — trả lời câu hỏi "Có cần chạy thêm LiteSVM không?"

**Nhóm đã chạy thật** (artifact `reports/results/defense_models.md`, tái lập bằng
`python -m scripts.experiment_defense`). Định nghĩa dùng trong báo cáo: **LiteSVM = SVM tuyến tính chi phí
thấp** — hai biến thể `LinearSVC` (lề cực đại, loss hinge, liblinear) và `SGDClassifier(loss='hinge')`
(giảm gradient ngẫu nhiên), mỗi biến thể có/không `class_weight='balanced'`. Vì hai mô hình này **không**
có `predict_proba`, chúng được bọc `CalibratedClassifierCV(method='sigmoid', cv=3)` (Platt scaling) để đưa
margin về xác suất — công bằng với các mô hình khác, và **hiệu chuẩn cũng chỉ học từ train**.

| Mô hình (corpus thật, test N = 64) | Test AUROC | Test AP | F1@0,5 | F1@ngưỡng 0,788 | Giây | ms/1.000 dòng |
|---|---:|---:|---:|---:|---:|---:|
| LiteSVM · LinearSVC (lề cực đại) | 0,9211 | 0,9459 | 0,8750 | 0,6182 | **0,020** | 2,13 |
| LiteSVM · LinearSVC + `class_weight=balanced` | 0,9200 | 0,9454 | 0,8608 | 0,6429 | 0,020 | 2,13 |
| LiteSVM · SGD hinge | 0,9352 | 0,9573 | 0,8421 | 0,7333 | **0,014** | 2,26 |
| **LiteSVM · SGD hinge + `balanced`** (tốt nhất của họ) | **0,9646** | **0,9761** | 0,9136 | 0,6667 | 0,014 | 2,19 |
| *(tham chiếu)* Logistic Regression | 0,9828 | 0,9887 | 0,9351 | 0,9143 | 0,005 | 0,95 |
| *(tham chiếu)* **Random Forest** | **0,9828** | **0,9908** | **0,9487** | **0,9600** | 0,465 | 21,98 |
| *(tham chiếu)* HistGradientBoosting | 0,9767 | 0,9868 | 0,9367 | 0,9067 | 0,170 | 10,61 |
| *(tham chiếu)* MLP | 0,9706 | 0,9819 | 0,9136 | 0,8947 | 0,074 | 1,19 |

**Kết luận dứt khoát:** LiteSVM (bản tốt nhất, SGD hinge + `balanced`) đạt AUROC **0,9646** — **thấp hơn
cả bốn mô hình chính** (logistic 0,9828; RF 0,9828; HGB 0,9767; MLP 0,9706). Chênh lệch so với MLP chỉ
0,006 nhưng chênh so với RF là **0,018 AUROC + 0,0146 AP**. Do đó:

- **Có đáng chạy để khảo sát đa dạng thuật toán? CÓ.** Nó là **phép thử biên tuyến tính**: LiteSVM chỉ đạt
  0,92–0,96 AUROC trong khi mô hình cây đạt 0,98 ⇒ tồn tại cấu trúc **phi tuyến/ngưỡng** mà siêu phẳng
  tuyến tính không nắm hết. Đây là bằng chứng độc lập cho lựa chọn mô hình cây.
- **Có cần đưa LiteSVM vào pipeline chính? KHÔNG.** Nó không vượt mô hình nào, lại kém ổn định hơn khi bỏ
  đi tính phi tuyến, và **phải hiệu chuẩn lại ngưỡng** (xem cảnh báo dưới).
- **Điểm rất đáng nói khi bị hỏi (bẫy kỹ thuật):** bản LiteSVM tốt nhất có `F1@0,5 = 0,9136` nhưng
  `F1@0,788 = 0,6667` — tức **ngưỡng 0,788 của Random Forest KHÔNG chuyển được sang LiteSVM**. Nguyên nhân
  là phép hiệu chuẩn Platt cho ra xác suất khác thang; cắm nguyên ngưỡng của mô hình này sang mô hình khác
  làm Recall sụp (TN/FP/FN/TP = 26/0/19/19). Bài học: **ngưỡng là tài sản của từng mô hình**, phải chọn lại
  trên validation cho từng họ mô hình.

Kiểm chứng thêm trên **bộ giả lập 95/5** (cùng bộ sinh dữ liệu với `benchmark_imbalanced.py`, nên so được
trực tiếp với bảng ở Chương 9.2):

| Phương pháp | Nhóm | ROC-AUC | PR-AUC | F1 thiểu số | Bal. Acc |
|---|---|---:|---:|---:|---:|
| Logistic Regression (không can thiệp) | Baseline | 0,7443 | 0,2823 | 0,1228 | 0,5328 |
| LiteSVM · LinearSVC | LiteSVM | 0,7366 | **0,2953** | 0,1228 | 0,5328 |
| LiteSVM · SGD hinge | LiteSVM | 0,7319 | 0,2385 | 0,0000 | 0,5000 |
| LiteSVM · LinearSVC + `balanced` | LiteSVM | 0,7744 | 0,1505 | 0,0185 | 0,5045 |

LiteSVM có PR-AUC nhỉnh hơn logistic (0,2953 so với 0,2823) nhưng ROC-AUC thấp hơn ⇒ **không tạo ra bước
nhảy về năng lực**; kết luận không đổi ở cả hai bộ dữ liệu.

### 9.5. Phụ lục trả lời "So Class trên XGBoost thì sao?" (báo cáo 3 lớp trong 1 bảng)

Vì Hội đồng từng hỏi riêng về XGBoost, nhóm chạy thêm XGBoost trên **đúng corpus** với ba biến thể
(`reports/results/defense_models.md` §3):

| Biến thể XGBoost | Test AUROC | Test AP | Class 0 (P/R/F1) | Class 1 (P/R/F1) | TN/FP/FN/TP |
|---|---:|---:|---|---|---|
| Mặc định (không can thiệp) | 0,9777 | 0,9872 | 0,8519 / 0,8846 / 0,8679 | 0,9189 / 0,8947 / 0,9067 | 23/3/4/34 |
| `scale_pos_weight = 0,606` | 0,9787 | 0,9876 | 0,8621 / 0,9615 / 0,9091 | 0,9714 / 0,8947 / 0,9315 | 25/1/4/34 |
| `scale_pos_weight` + **early stopping** (chọn vòng trên validation) | **0,9818** | **0,9897** | 0,8667 / **1,0000** / 0,9286 | **1,0000** / 0,8947 / 0,9444 | **26/0/4/34** |

Hai kết luận rút ra — đều ủng hộ kết luận chung của đồ án:

1. **Thêm một thư viện boosting mạnh (XGBoost) cũng KHÔNG vượt Random Forest** (0,9818 so với 0,9828), dù
   có thêm `scale_pos_weight` **và** early stopping. Đây là bằng chứng định lượng mạnh cho kết luận
   *"**thuật toán không phải nút thắt** — nút thắt nằm ở dữ liệu/nhãn"* (trùng khớp với kết luận từ
   `reports/experiment/summary.md`: 12/16 kỹ thuật và mọi mô hình đều bám quanh cùng một mức điểm).
2. **`scale_pos_weight` là "con dao hai lưỡi" có số liệu:** ở biến thể `spw` lớp thiểu số (class 0) tăng
   Recall 0,8846 → 0,9615, nhưng số FN của lớp dương vẫn là **4** (so với RF chỉ 2) ⇒ trọng số lớp không
   sửa được lỗi nằm ở tầng nhãn (Chương 6.3).

---

## CHƯƠNG 10: TỔNG KẾT ĐỒ ÁN, ĐÓNG GÓP THỰC TIỄN & DEFENSE CHEAT-SHEET

### 10.1. Đúc kết kiến trúc tối ưu (Best Pipeline)

```
Dữ liệu SEC XBRL (8 công ty, 332 quý)
   → mẫu hoá: cửa sổ 8 quý + quý target + nhãn           (purged time-split: 212/16/32/64)
   → 47 đặc trưng trong 6 khối (ratios_latest · ratios_yoy · growth · structure · stress · path)
   → Pipeline: SimpleImputer(median) → [StandardScaler cho logistic/MLP] → Model
   → Model: Random Forest (n_estimators=300, max_depth=6, min_samples_leaf=2,
                           class_weight='balanced_subsample')  ← cấu hình MẶC ĐỊNH, không tinh chỉnh
   → Ngưỡng vận hành t = 0,7879 (chọn bằng max-F1 trên validation; C_FN/C_FP = 5/1)
   → Quyết định: P(distress) ≥ 0,7879 ⇒ đưa vào danh sách theo dõi/đánh giá lại tín dụng
```

Kết quả trên test độc lập: **AUROC 0,9828 · AP 0,9908 · F1 0,9600 · Balanced Acc 0,9545 · MCC 0,9039 ·
Brier 0,0580 · Accuracy 95,31%** — vượt xa quy tắc kế toán cổ điển (Altman Z'' cho AUROC 0,7581) nhưng
**không** vượt được baseline "nhớ mặt công ty" ở mức có ý nghĩa thống kê (xem 10.2).

**Khuyến nghị vận hành kèm theo (đã có trong mã):** (i) mọi quyết định ghi log `sample_id`, xác suất,
ngưỡng, phiên bản mô hình, SHA-256 artifact; (ii) giám sát dịch chuyển bằng KS/PSI — 5 feature đã vượt
ngưỡng cảnh báo (nặng nhất `debt_to_assets_yoy` KS = 0,621); (iii) huấn luyện lại theo quý và **chọn lại
ngưỡng trên validation mới**, không bê ngưỡng cũ sang (bài học từ Chương 9.4).

### 10.2. Phát hiện khoa học quan trọng nhất: **rò rỉ cấp thực thể** (điểm mạnh khi bị chất vấn)

Đây là phần nhóm chủ động công bố **trước khi** Hội đồng hỏi:

| Bằng chứng | Số liệu | Ý nghĩa |
|---|---|---|
| Baseline "nhớ mặt công ty" (`ticker_prior`) trên test | **AUROC 0,9858** (AP 0,9846) | cao **hơn** cả mô hình RF (0,9828) |
| Kiểm định DeLong RF − `ticker_prior` | Δ = −0,0030; **p = 0,7546** | **không** khác biệt có ý nghĩa thống kê |
| Riêng tỷ lệ nhãn theo công ty | HD/LOW/WMT = **100%** nhãn 1; ROST = **4,7%** | nhãn gần như là **thuộc tính của công ty** |
| Tỉ lệ mẫu thuộc công ty chỉ có một lớp | **41,0%** | hơn 1/3 dữ liệu không đóng góp khả năng phân biệt |
| Quy tắc kế toán truyền thống (Altman Z″ < 1,1) | AUROC 0,7581; F1 0,656 | nhãn **không** sinh ra từ quy tắc kế toán ⇒ nó phản ánh **thực thể**, không phản ánh quý |
| Đo lại bằng giao thức khắt khe: cross-company | AUROC **0,9334** (AP 0,9577) | tụt 0,05 khi giữ trọn công ty ra khỏi train |
| **Leave-One-Company-Out (8 công ty)** | AUROC trung bình **0,6280** | đây mới là năng lực dự báo **thực sự** cho công ty chưa từng thấy |
| Walk-forward theo thời gian (3 fold) | RF 0,9229 (min 0,8109); HGB 0,9490 (min 0,9195); logistic 0,7532 (min 0,5069); MLP 0,7143 (min 0,5000) | có fold gần như **ngẫu nhiên** ⇒ kết luận "0,98" không bền theo thời gian |
| Nhãn sự kiện 8-K của SEC | **0** sự kiện phá sản (item 1.03); 11 hồ sơ tín hiệu kiệt quệ | dữ liệu **không** chứa phá sản thật ⇒ đề tài phải gọi đúng tên là *suy giảm tài chính* |
| Nhãn gốc có tái tạo được từ dữ liệu công bố? | không — khớp tối đa **74,7%** với quy tắc đơn giản nhất | nhóm công bố điểm yếu này thay vì che; kiểm chứng độ nhạy bằng **3 định nghĩa nhãn khác** (3/3 giữ kết luận) |

**Thông điệp nhóm muốn Hội đồng ghi nhận:** một mô hình đạt AUROC 0,983 trên bộ dữ liệu này **không**
chứng minh năng lực dự báo suy giảm — vì một quy tắc chỉ cần *biết tên công ty* đã đạt 0,986. Giá trị học
thuật của đồ án nằm ở chỗ **đo lường và công bố sự thật đó**, kèm bộ giao thức đánh giá đúng (baseline thực
thể + cross-company + LOCO + walk-forward + khoảng tin cậy), thay vì trình bày một con số 0,98 gây ngộ nhận.

### 10.3. Đóng góp thực tiễn, hạn chế và hướng phát triển

**Đóng góp:** (1) đường ống dữ liệu tái lập được từ SEC, có kiểm chứng SHA-256 và tra ngược từng ô
(4.609 ô, 0 lỗi; ETL port tái tạo **99,92%** số ô); (2) bộ 47 đặc trưng có căn cứ kinh tế, được **đo** đóng
góp bằng ablation; (3) khung đánh giá đầy đủ (≥ 4 baseline + 4 họ mô hình + 4 giao thức + kiểm định ý
nghĩa + khoảng tin cậy bootstrap theo cụm); (4) báo cáo trung thực về rò rỉ cấp thực thể và nhãn.

**Hạn chế đã biết (nói trước để không bị "đánh úp"):** 8 công ty / 324 mẫu, các mẫu liên tiếp chồng lấn
(Jaccard lịch sử trung bình = **0,917** ⇒ số quan sát độc lập nhỏ hơn 324 nhiều); nhãn gốc không tái tạo
được; nhãn sự kiện 8-K không có ca phá sản nào; kết luận không suy rộng ra toàn ngành bán lẻ.

**Hướng phát triển theo ưu tiên:** **P0** chốt định nghĩa nhãn chính thức với giảng viên (hạ tầng đã xong:
3 định nghĩa nhãn quy tắc + nhãn sự kiện 8-K + ETL port); **P1** mở rộng lên 50–100 doanh nghiệp (đường
sinh quý đã có; learning curve **chưa bão hoà**: gap train−val 0,1049), lọc đặc trưng theo cụm/VIF (33/47
cột VIF > 10; chỉ 12,89 chiều hiệu dụng), và thử mô hình panel/survival cho chuỗi quý (90,8% cặp quý liền
nhau giữ nguyên nhãn); **P2** hiệu chuẩn xác suất để dùng được ngưỡng Bayes `p* = 1/6`, monitoring PSI/KS,
và kiểm thử trên dữ liệu ngoài phân phối (*out-of-distribution*).

### 10.4. Phụ lục: Defense Cheat-Sheet — 10 câu hỏi & đáp án nhanh (mỗi câu 3 ý, kèm số)

| # | Câu hỏi của Hội đồng | Trả lời nhanh (3 ý) |
|---|---|---|
| **1** | Bộ dữ liệu có mất cân bằng không? | ① **Cấp mẫu: nhẹ** — IR = **1,6557** (62,35% dương), entropy 0,9556 bit, Gini 0,4695. ② **Cấp thực thể: cực nặng** — HD/LOW/WMT **100%** nhãn 1, ROST **4,7%**; 41% mẫu thuộc công ty một lớp. ③ Vì nhẹ ở cấp mẫu nên **không** dùng resampling; vấn đề thật là **rò rỉ cấp thực thể** (câu 9–10). |
| **2** | Trình bày công thức đánh giá? | ① Dùng **F1 = 2PR/(P+R)**, **Balanced Accuracy = (TPR+TNR)/2**, **MCC**, **AP = Σ(R_k−R_{k−1})P_k**, **Brier**. ② Tại t = 0,788: P = 0,9730, R = 0,9474, F1 = **0,9600**, BA = **0,9545**, MCC = **0,9039**, Brier = **0,0580**, BSS = **0,7594**. ③ **Accuracy không dùng để kết luận** — mốc đoán lớp đa số đã 62,35%; mã nguồn **không** dùng G-mean (đã kiểm tra toàn repo). |
| **3** | Các cách xử lý mất cân bằng? | ① **4 cấp:** dữ liệu (SMOTE/ADASYN/RUS/Tomek/ENN) · thuật toán (`class_weight`, `scale_pos_weight`, Focal Loss) · tập hợp (BalancedRF/EasyEnsemble/RUSBoost) · hậu xử lý (**threshold moving**). ② **17 pipeline đã cài + kiểm thử**, mọi resampling nằm **trong** pipeline ⇒ 11/11 PASS chống rò rỉ. ③ Pipeline chính dùng **`balanced_subsample`/`class_weight`** vì corpus chỉ lệch 1,66 và resampling sẽ hợp thức hoá rò rỉ thực thể. |
| **4** | Ngoài baseline có đề xuất đặc trưng mới gì? | ① Từ **1** chỉ tiêu lên **47 cột / 6 khối**: 14 `ratios_latest` + 14 `ratios_yoy` + 10 `growth` + `structure` + `stress` + 6 `path`. ② Điểm mới: biến dữ liệu **tĩnh** thành **quỹ đạo 8 quý** (`current_ratio_min_window`, `revenue_drawdown_window`, `negative_ocf_streak`) — đo được **+0,03…+0,12 AUROC cross-company**. ③ Biến mạnh nhất toàn hệ thống: `current_ratio_min_window` (`\|2AUC−1\| = 0,904`, MI = 0,432). |
| **5** | Tại sao dùng loại bỏ dần và khác gì baseline? | ① Vì **33/47 cột VIF > 10** (max 798,1) và chỉ **12,89 chiều hiệu dụng**, RFE đơn biến sẽ loại nhầm biến tương quan hữu ích ⇒ dùng **Group Ablation** bỏ trọn từng khối. ② Baseline "ném toàn bộ biến thô" **không kiểm chứng**; ablation đo **đóng góp biên** của từng khối thông tin nghiệp vụ. ③ Bỏ khối nào test AUROC cũng lệch **≤ 0,003** (trừ `ratios_latest` −0,0172) ⇒ đặc trưng **bão hoà**, nút thắt không ở feature. |
| **6** | Non-E Mode và E-Mode là gì? | ① **Non-E Mode:** can thiệp **tĩnh, ngoài mô hình** (SMOTE/RUS/class weight) rồi dùng **một** mô hình đơn. ② **E-Mode:** lấy mẫu cân bằng **động, bên trong từng mô hình con** (BalancedRF, EasyEnsemble, RUSBoost). ③ Khác biệt bản chất: Non-E đổi **dữ liệu/điểm vận hành**, E-Mode đổi **quá trình học của từng estimator**. |
| **7** | Kết quả Non-E, E-Mode, Class Weight? | ① Trên 95/5: Baseline PR-AUC **0,4786** > Non-E **0,3211** ≈ E-Mode **0,3398**; E-Mode dẫn **Balanced Acc 0,7352** nhưng thời gian **×5**. ② `class_weight` thuần: Recall 0,066 → **0,717** nhưng Precision rơi còn **0,117** (PR-AUC 0,148). ③ Trên **dữ liệu thật**: tốt nhất `smote_enn` AP 0,9683 so với đối chứng 0,9588 — ΔAP +0,0095, **p = 0,21** ⇒ **chưa** có ý nghĩa. |
| **8** | Kết hợp vs từng phương pháp riêng lẻ? | ① Trên 1:50: baseline PR-AUC **0,9474** > hybrid 0,9229 > đơn lẻ 0,9104; **12/16** kỹ thuật đơn lẻ và **3/4** hybrid **không** vượt baseline. ② Hybrid chỉ thắng khi hai can thiệp **bù trừ**: `smote_enn` (+0,0033 PR-AUC), cũng là kỹ thuật tốt nhất trên **dữ liệu thật** (0,9683). ③ Luôn đọc kèm **FPR**: `easy_ensemble` Recall cao nhưng FPR **0,0581**. |
| **9** | Có cần chạy thêm LiteSVM không? | ① **Đã chạy thật**: LiteSVM tốt nhất (SGD hinge + balanced) đạt AUROC **0,9646** / AP 0,9761 — **thấp hơn cả 4 mô hình chính** (RF 0,9828). ② Giá trị của nó là **phép thử biên tuyến tính**: siêu phẳng tuyến tính không nắm hết cấu trúc ngưỡng của dữ liệu tài chính ⇒ ủng hộ mô hình cây. ③ Cảnh báo kỹ thuật: ngưỡng **không chuyển được** giữa các họ mô hình (F1 của LiteSVM tụt 0,9136 → 0,6667 khi áp ngưỡng 0,788 của RF). |
| **10** | So Class trên XGBoost? Confusion Matrix, Train vs Val? | ① XGBoost (kèm `scale_pos_weight` + early stopping) đạt AUROC **0,9818** — **vẫn không vượt RF 0,9828** ⇒ *thuật toán không phải nút thắt*. ② RF @0,788: TN/FP/FN/TP = **25/1/2/36**; class 0 P/R/F1 = 0,9259/0,9615/0,9434; class 1 = 0,9730/0,9474/**0,9600**. ③ Overfit: RF gap train−val **0,0342** kiểm soát bởi `max_depth=6` + `min_samples_leaf=2`; HGB 0,0260; MLP −0,0134; learning curve **chưa bão hoà** (gap 0,1049). |

### 10.5. Phụ lục: 6 con số "đinh" phải nhớ

| Con số | Giá trị | Xuất hiện ở |
|---|---|---|
| Bộ dữ liệu | **324 mẫu · 47 cột · 8 công ty** (train 212 / purged 16 / val 32 / test 64) | Chương 1.3 |
| Mô hình chốt | **Random Forest** → AUROC **0,9828** · AP **0,9908** · F1 **0,9600** · ngưỡng **0,7879** | Chương 5, 6 |
| Ngưỡng Bayes lý thuyết | **p\* = 1/6 ≈ 0,1667** (`C_FN/C_FP = 5/1`) so với ngưỡng vận hành 0,7879 | Chương 7.3 |
| Rò rỉ thực thể | `ticker_prior` **0,9858** (p DeLong **0,7546**) → cross-company **0,9334** → **LOCO 0,6280** | Chương 10.2 |
| Bão hoà đặc trưng | bỏ khối nào cũng lệch **≤ 0,003** AUROC; VIF 33/47 > 10; 12,89 chiều hiệu dụng | Chương 8.3 |
| Cân bằng dữ liệu | chất lượng xếp hạng (PR-AUC) do Baseline dẫn; E-Mode chỉ dịch **điểm vận hành**, chi phí **×5** | Chương 9.2–9.3 |

---

## PHỤ LỤC A — Công thức dạng LaTeX (dán trực tiếp vào Equation Editor của Word)

```
Precision = TP / (TP + FP)                              Recall = TP / (TP + FN)

F1 = 2PR / (P + R)                                      F_beta = (1+b^2)PR / (b^2 P + R)

BalancedAcc = (TPR + TNR)/2 = (1/2)*[ TP/(TP+FN) + TN/(TN+FP) ]

MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

AUROC = P(S+ > S-) + (1/2)P(S+ = S-) = (1/(n+ n-)) * SUM_i SUM_j [ 1(s_i>s_j) + (1/2)1(s_i=s_j) ]

AP = SUM_k (R_k - R_(k-1)) * P_k                        Brier = (1/N) SUM_i (p_i - y_i)^2
BrierSkill = 1 - BS_BS_null,   BS_null = p_bar(1 - p_bar)

E[Cost|x] = C_FN * eta * 1[p_hat < t] + C_FP * (1-eta) * 1[p_hat >= t]
     ==>  p* = C_FP / (C_FN + C_FP) = 1 / (1 + C_FN/C_FP)

pi(S) = (M - 1) / [ C(M,|S|) * |S| * (M - |S|) ]
phi_j = SUM_{S subset M\j} [ |S|! (M-|S|-1)! / M! ] * [ v(S u j) - v(S) ]
Efficiency: f(x) = E[f(x)] + SUM_j phi_j

VIF_j = 1 / (1 - R_j^2)
PSI   = SUM_b (q_new,b - q_ref,b) * ln( q_new,b / q_ref,b )
SMD   = (mu_new - mu_ref) / sqrt( (sigma_new^2 + sigma_ref^2) / 2 )

z_DeLong = (AUC_A - AUC_B) / sqrt( Var(AUC_A - AUC_B) )
Var(AUC_A - AUC_B) = (1/n+) Var(V10_A - V10_B) + (1/n-) Var(V01_A - V01_B)

q_(i) = min_{j >= i} (m/j) * p_(j)                      [Benjamini-Hochberg]

Z'' = 6.56*(WC/TA) + 3.26*(RE/TA) + 6.72*(EBIT/TA) + 1.05*(BV_E/TL),  WC = CA - CL

total_liabilities = liabilities neu co, nguoc lai TL = TA - SE
SMOTE:  x_new = x_i + delta * (x_z(i) - x_i),  delta ~ U(0,1),  z(i) in k-NN
FocalLoss: FL = -[ alpha*y*(1-p)^gamma*ln p + (1-alpha)*(1-y)*p^gamma*ln(1-p) ]
```

## PHỤ LỤC B — Bản đồ số liệu ↔ artifact (để Hội đồng đối chiếu ngay tại chỗ)

| Số liệu dùng ở | Artifact gốc | Lệnh sinh lại |
|---|---|---|
| Ch.1–2 (mẫu, split, mất cân bằng, EDA) | `class_balance.json`, `eda_summary.json`, `eda_deep.json` | `python -m scripts.eda_deep`, `python -m scripts.class_balance` |
| Ch.3 (thiếu dữ liệu, outlier, scaler) | `preprocessing_experiment.json`, `data_audit.json` | `python -m scripts.experiment_preprocessing`, `python -m scripts.audit_data` |
| Ch.4–6 (4 mô hình, so sánh, per-class, overfit) | `baselines.json`, `summary.json`, `test_evaluation.json`, `analysis.json`, `defense_models.md` | `python -m forecasting.train`, `python -m forecasting.baselines`, `python -m scripts.analyze`, `python -m scripts.experiment_defense` |
| Ch.7 (metric, ngưỡng, chi phí) | `test_evaluation.json`, `analysis.json::threshold` | `python -m forecasting.evaluate` |
| Ch.8 (47 feature, ablation) | `analysis.json::ablation`, `eda_deep.json` | `python -m scripts.analyze` |
| Ch.9 (cân bằng, LiteSVM, XGBoost) | `benchmark_imbalanced*.md`, `experiment/summary.md`, `imbalance_real.md`, `techniques.md`, `defense_models.md` | `python benchmark_imbalanced.py`, `python -m imbalance_experiment.main`, `python -m scripts.experiment_imbalance_real`, `python -m scripts.experiment_defense` |
| Ch.10 (rò rỉ thực thể, kiểm định, nhãn) | `validation_checks.json`, `significance.md`, `walk_forward.json`, `label_sensitivity.json`, `events.md`, `provenance.md` | `python -m forecasting.validation`, `python -m scripts.significance`, `python -m scripts.label_sensitivity`, `python -m scripts.fetch_events`, `python -m scripts.verify_provenance` |
| Toàn bộ | — | `python -m scripts.run_all` (22 bước) → `python -m scripts.make_report` |

> **Ghi chú định dạng:** công thức trong thân báo cáo viết ở dạng Unicode-math (hiển thị đúng trong cả Word
> và Markdown); Phụ lục A là bản LaTeX tương ứng để chèn vào Equation Editor khi cần trình bày chuẩn.
