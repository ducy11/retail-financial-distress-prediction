# Bộ tài liệu bảo vệ đồ án — Dự báo suy giảm tài chính doanh nghiệp bán lẻ

*Tài liệu VIẾT TAY (không sinh tự động), dùng để làm slide và ôn phản biện. Mọi số liệu đọc trực tiếp
từ artifact trong repo: `reports/results/*.json`, `docs/BAO-CAO.md`, `reports/figures/**`.
Kiểm chứng: `python -m scripts.audit_data` (98.192 phép kiểm tra, 0 phát hiện) và
`python -m unittest discover -s tests` (220 test PASS).*

**Cách dùng**

- **Phần 1 (Factsheet):** số liệu nền để trả lời mọi câu hỏi "số này ở đâu ra".
- **Phần 2 (Dàn slide):** 11 slide, mỗi slide có tiêu đề · takeaway · bullet hiển thị · lời thoại.
- **Phần 3 (Mock defense):** 8 câu hỏi phản biện kèm kịch bản 4 bước (thừa nhận → kỹ thuật → số liệu → kết luận).
- **Đi kèm:** `docs/checklist-doi-chieu-yeu-cau.md` (đối chiếu 19 tiêu chí → bằng chứng → lệnh),
  `docs/slide-bao-ve.md` (deck 11 slide, xuất được `.pptx`), `python -m scripts.predict` (demo dự đoán
  1 quý + SHAP), `docs/BAO-CAO.md` §11 (tài liệu tham khảo 24 mục gắn vị trí dùng thật).

---

# PHẦN 1 — ML FACTSHEET (BẢNG DỮ LIỆU KỸ THUẬT CỐT LÕI)

## 1.1. Bài toán & Bộ dữ liệu

| Tiêu chí | Thông tin |
|---|---|
| **Tên bài toán** | **Phân loại nhị phân có giám sát** (quý kế tiếp doanh nghiệp có suy giảm tài chính?) + **xếp hạng rủi ro** (AP, ngưỡng theo chi phí) |
| **Định dạng dữ liệu** | **Bảng (tabular)** — nguồn gốc là **chuỗi thời gian quý** (SEC XBRL company-facts), vector hoá thành 47 đặc trưng/dòng |
| **Quy mô** | **324 mẫu** (từ **332 quý** của **8 công ty**), **47 features**, **2 classes** (`is_distressed` 0/1) |
| **Đơn vị dữ liệu** | 1 mẫu = (cửa sổ lịch sử ≤ `as_of`, tối đa 8 quý) + (quý target) + nhãn; lịch sử **không** chứa quý target (`available_on ≤ as_of`) |
| **Nhãn (cấp MẪU)** | Toàn corpus **62,3% / 37,7%** (202 dương / 122 âm) → **hơi lệch**; **IR = 1,66**; entropy **0,956 bit**; Gini 0,470 ⇒ "mất cân bằng nhẹ" |
| **Nhãn (cấp THỰC THỂ)** | **HD/LOW/WMT = 100% nhãn 1**; ORLY 90,7%; DG 54,8%; FIVE 19,4%; DKS 11,6%; **ROST 4,7%** ⇒ **IR tới 20,5**; **41,0% mẫu** thuộc công ty chỉ có một lớp ⇒ **mất cân bằng NGHIÊM TRỌNG** |
| **Chia tập** | Theo **THỜI GIAN trong từng công ty**: **train 212 / validation 32 / test 64** + **purge 16**. Tỉ lệ dương: train 62,3% · val 65,6% · test 59,4% · purged 68,8% (**lệch tối đa 3,6 điểm %** so với corpus) |
| **Có Stratified Split?** | **Không dùng stratified-random** (gây rò rỉ thời gian + thực thể). Dùng **chia theo thời gian** (test tự động `test_class_ratio_preserved_across_splits`) và **`StratifiedGroupKFold`** cho mọi bước CV/tuning (giữ tỉ lệ lớp **và** giữ trọn công ty ngoài fold-train) |
| **Tính xác thực (dữ liệu THẬT, không bịa)** | `python -m scripts.verify_provenance` mở lại **snapshot SEC thô 129 MB**: **20 file** companyfacts băm SHA-256 **khớp registry** (`data/sec/downloads.json`, ghi cả CIK + URL + thời điểm tải); **4.609/4.609 ô** (quý × chỉ tiêu) **tra ngược được** trong companyfacts kèm `accn`/`form`/ngày nộp; **0 lỗi quy đổi VND**; **0 ô "điền số cho đủ"** — 703 ô không có fact ở SEC đều để `null`. Kết quả: `reports/results/provenance.{json,md}` |

**Bằng chứng dữ liệu thật (câu trả lời cho "số liệu ở đâu ra?")**

1. **Raw có thật**: `data/sec/raw/*-companyfacts.json` (**41 file, 128,74 MB**) tải từ API công khai
   `https://data.sec.gov/api/xbrl/companyfacts/` — mỗi file có `sha256` + `downloaded_at` trong
   `downloads.json` (ví dụ WMT: CIK 104169, `f4a7691f…84d5d`).
2. **Hash khớp độc lập**: băm lại bằng cả Python và PowerShell đều ra đúng hash trong registry.
3. **Từng con số tra ngược được**: mọi fact ghi trong `sources` của
   `data/retail-expanded/*-16-indicators-vnd.json` (tag, kỳ, `accn`, `form`) tồn tại y hệt trong
   companyfacts ⇒ không có số liệu nào được sinh/sửa tay.
4. **Không "điền số cho đủ"**: 703 ô thiếu ở SEC để `null` (chính là nguồn gốc của các chỗ khuyết
   trong mục 1.2) — đây là phép kiểm bắt lỗi bịa số.
5. **Đơn vị**: VND chỉ là **phép nhân hằng số 25.000** khai trong `fx_policy` (minh hoạ đơn vị, không
   phải BCTC Việt Nam, không phải tỷ giá lịch sử) — đã kiểm đúng theo từng `method`, kể cả
   `current_ytd_minus_previous_ytd` (hiệu 2 kỳ luỹ kế).

## 1.2. EDA & Tiền xử lý

**3 phát hiện EDA nổi bật nhất**

| # | Phát hiện | Số liệu chứng minh |
|---|---|---|
| 1 | **Nhãn gần như là thuộc tính THỰC THỂ, không phải sự kiện của quý** | 90,8% cặp quý liền nhau giữ nguyên nhãn; P(nhãn1\|nhãn1) = **0,929** vs P(nhãn1\|nhãn0) = **0,127**; HD/LOW/WMT 100% nhãn 1; `ticker_prior` (chỉ dùng danh tính công ty) đạt **AUROC 0,986 > mô hình 0,983** |
| 2 | **Khuyết thiếu có cấu trúc và mang thông tin nhãn (MNAR)** | **5/16 chỉ tiêu có độ phủ TỐI THIỂU < 50%** (`liabilities`: min 0% – mean 39,1%; `receivables`: min 0% – mean 57,1%; `short_term_investments`: mean 29,5%); **22/47 feature** có tỉ lệ nhãn khác biệt giữa nhóm thiếu/có dữ liệu sau BH-FDR — nặng nhất `net_margin_latest`: **13,2% vs 68,9% (Δ = −55,7 điểm %)** |
| 3 | **Đuôi nặng + đa cộng tuyến mạnh** | 6/14 tỷ số \|skew\| > 1 (`debt_to_equity` skew **−14,9**, **30,6%** giá trị ngoài IQR); 15/47 cột đuôi nặng, 32/47 cột > 5% ngoại lai; **5 cặp** Pearson lệch Spearman > 0,3 (tương quan do ngoại lai: `debt_to_assets ↔ debt_to_equity` r = −0,075 nhưng hạng = 0,448); **VIF > 10 ở 33/47 cột**; 8 cặp \|r\| ≥ 0,9 → **7 cụm**; **số chiều hiệu dụng 12,89/47** |

*Phát hiện bổ trợ dùng để phản biện:* Jaccard lịch sử giữa 2 mẫu liền nhau **0,917** (324 mẫu ≪ 324 quan sát độc lập); **5 feature** dịch chuyển train→test (KS ≥ 0,30 hoặc \|SMD\| ≥ 0,50; nặng nhất `debt_to_assets_yoy` KS **0,621**); độ phủ chỉ tiêu thấp nhất theo công ty = **0%**.

**Pipeline tiền xử lý (toàn bộ nằm TRONG `sklearn.Pipeline` — fit chỉ trên train)**

```
X (47 features, có NaN)
  -> [1] Winsorizer(method="iqr")   # clip ngoài [Q1-1.5IQR, Q3+1.5IQR], ngưỡng HỌC TỪ TRAIN
  -> [2] SimpleImputer(median)      # median học từ train của từng fold
  -> [3] Scaler (theo mô hình): StandardScaler cho Logistic; none cho mô hình cây
        (đã thí nghiệm thêm: RobustScaler / PowerTransformer / QuantileTransformer)
  -> [4] Model
```

- **Mã hoá categorical:** không có biến phân loại; **cố ý KHÔNG one-hot `ticker`** (sẽ hợp thức hoá đúng loại rò rỉ thực thể đang đo).
- **Mất cân bằng:** `class_weight='balanced_subsample'` (RF) / `'balanced'` (HGB, LightGBM) — trọng số **tính trong `fit`** từ nhãn fold-train. **Không** resample ở pipeline chính; SMOTE/RUS/Focal Loss đo riêng (lab 98/2 và thí nghiệm trên dữ liệu thật) — **không cải thiện có ý nghĩa** (`smote_enn` 0,9683 vs đối chứng 0,9588, p = 0,21).
- **Chống rò rỉ (7 lớp):** split theo thời gian + purge; impute/scale trong Pipeline; CV theo nhóm công ty; test chấm **1 lần**; manifest có SHA-256 nguồn & split; 220 test tự động (gồm `test_no_label_leak_in_features`, `test_main_pipeline_has_no_balancer`).

## 1.3. Kiến trúc mô hình & Siêu tham số

| Mô hình | Họ | Siêu tham số chạy chính | Sau tinh chỉnh (CV-AP cross-company) |
|---|---|---|---|
| **Logistic Regression** | Tuyến tính | `max_iter=2000`, `C=0.1`, kèm `StandardScaler` | `C=0.01`, `class_weight=None` → **CV-AP 0,960** (mặc định 0,935, **+0,026**) |
| **Random Forest** ✅ *chốt* | Bagging (cây) | `n_estimators=300`, `max_depth=6`, `min_samples_leaf=2`, `class_weight='balanced_subsample'` | `max_depth=3`, `min_samples_leaf=2`, `n_estimators=500` → **CV-AP 0,989** (+0,003) |
| **HistGradientBoosting** | Boosting (sklearn) | `max_iter=300`, `learning_rate=0.05`, `max_depth=3`, `l2_regularization=1.0`, `class_weight='balanced'` | `lr=0.03`, `depth=2`, `max_iter=200` → **CV-AP 0,939** (+0,015) |
| **LightGBM** | Boosting hiện đại | `n_estimators=400`, `lr=0.05`, `num_leaves=15`, `min_child_samples=10`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `class_weight='balanced'` | `lr=0.05`, `num_leaves=15`, `min_child_samples=10` → CV-AP 0,917 (= mặc định) |
| *XGBoost* | Boosting | — | **Bị môi trường chặn** (xgboost 2.1.3 ↔ sklearn 1.6: `'super' object has no attribute '__sklearn_tags__'`); cách sửa: `pip install "scikit-learn>=1.7"` hoặc `xgboost<2.1` |

**Cách tối ưu (bài bản, có lưu vết):**

- `GridSearchCV` (8–18 cấu hình/mô hình) + **`StratifiedGroupKFold(4)`** + **`refit="average_precision"`**; luôn in kèm điểm của cấu hình mặc định trên cùng splitter ⇒ trả lời "tinh chỉnh có thật sự cải thiện không".
- Bổ sung **random search** (25 trial/mô hình, mẫu log-uniform cho tham số scale) + **sổ thực nghiệm** `reports/results/runs.csv` (**98 dòng**: run_id, model, params, seed, CV-AP, ±std, thời gian, trạng thái).
- Δ random search vs GridSearchCV: **LightGBM +0,049**, HGB **+0,026**, logistic +0,007, **Random Forest −0,004** ⇒ mô hình được chốt không hưởng lợi thêm từ tìm kiếm rộng hơn (kết quả âm được báo cáo).
- **Tiêu chí dừng:** không còn cải thiện AP cross-company có ý nghĩa; **không bao giờ dùng test** để chọn cấu hình.

## 1.4. Bảng tổng hợp kết quả (TEST — n = 64, ngưỡng 0,5)

> Phạm vi số liệu: cột *Cross-company OOF AUROC* = `GroupKFold(8)` giữ trọn công ty ra ngoài, gộp
> xác suất out-of-fold trên **toàn bộ 324 mẫu** (`reports/results/validation_checks.json`, mục
> `headline`). Các metric test của 4 mô hình lấy từ bước `forecasting.validation` (cùng 64 mẫu test);
> riêng mô hình được chốt còn có bộ metric **chính thức chạy 1 lần** ở `forecasting.evaluate`
> (đúng ngưỡng vận hành) — đó là hàng thứ 8 trong bảng.

| Model | Accuracy | Precision | Recall | F1 | Macro-F1 | MCC | AUROC | AP (PR-AUC) | Cross-company OOF AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (lớp đa số) | 0,594 | 0,594 | **1,000** | 0,745 | 0,373 | 0,000 | 0,500 | 0,594 | — |
| **`ticker_prior`** (baseline "nhớ mặt công ty") | 0,906 | **1,000** | 0,842 | 0,914 | 0,905 | 0,827 | **0,986** | 0,985 | — |
| `single_feature[debt_to_assets_latest]` | 0,797 | 0,714 | 0,947 | 0,847 | 0,772 | 0,355 | 0,889 | 0,938 | — |
| Logistic Regression | 0,922 | 0,923 | 0,947 | 0,935 | 0,919 | 0,838 | 0,983 | 0,989 | 0,912 |
| **Random Forest** ✅ | 0,938 | 0,925 | 0,974 | 0,949 | 0,934 | 0,871 | 0,983 | **0,991** | **0,933** |
| HistGradientBoosting | 0,922 | 0,902 | 0,974 | 0,937 | 0,917 | 0,839 | 0,977 | 0,987 | 0,926 |
| LightGBM | — | — | — | 0,949 | 0,934 | — | 0,979 | 0,988 | **0,933** |
| **Random Forest @ ngưỡng vận hành 0,788** | **0,953** | **0,973** | 0,947 | **0,960** | **0,952** | **0,904** | 0,983 | 0,991 | 0,933 |

*LightGBM: bước `validation` có chấm cả 4 mô hình trên test (AUROC 0,979 · AP 0,988 · F1 0,949 ·
macro-F1 0,934) nhưng chỉ in F1/macro-F1 — bộ Precision/Recall/Confusion đầy đủ chỉ được tính cho
mô hình **được chốt** (thiết kế "test chấm 1 lần"), nên các ô đó để "—" chứ không suy diễn.*

**Mô hình chiến thắng & chênh lệch**

- Chốt **Random Forest** theo **quy tắc công bố trước**: `max AP cross-company (GroupKFold) → best-F1(val) → AP(val) → AUROC(val) → gap overfit nhỏ nhất`.
- AP cross-company (OOF trên train+validation, 244 mẫu): RF **0,9577** > HGB 0,9569 > LightGBM 0,9471 > logistic 0,9232 ⇒ RF chỉ hơn HGB **0,0008** (tie-break bằng gap overfit F1: RF +0,009 so với HGB +0,024).
- Hơn logistic: ΔAP test +0,002; **Δcross-company AUROC +0,021**. Hơn HGB: ΔAP +0,004; Δcross-company +0,007.
- **Đối chiếu trung thực:** ba họ mô hình chênh nhau **< 0,01 AUROC** trên test ⇒ *thuật toán không phải nút thắt*; `ticker_prior` (không dùng feature) vẫn **cao hơn mô hình 0,003 AUROC**.


**Lý giải kỹ thuật vì sao RF nhỉnh hơn (có định lượng)**

1. **Dữ liệu nhỏ + nhiều cột đa cộng tuyến** (VIF > 10 ở 33/47): RF (bagging + `max_features` ngẫu nhiên) giảm phương sai và **bền với cột trùng thông tin**; boosting dễ dồn importance vào vài cột mạnh.
2. **Đuôi nặng/outlier** (skew tới −14,9): mô hình cây **bất biến đơn vị**, ít bị ngoại lai kéo; logistic phụ thuộc `StandardScaler` (hệ số `debt_to_equity_latest` gần 0 dù AUC 0,715).
3. **`class_weight='balanced_subsample'`** giúp RF nhạy với mẫu dương ở từng bootstrap: precision 0,925 vs HGB 0,902 (cùng recall 0,974).
4. **Thí nghiệm tiền xử lý (mục 9.1):** `random_forest + winsorize IQR` đạt AP cross-company **0,9590** — cao nhất trong 24 cấu hình (tham chiếu `logistic + StandardScaler` = 0,9232). Nhưng so **cùng mô hình được chốt**: winsorize chỉ **+0,13 điểm %** (0,9590 vs 0,9577) ⇒ nằm trong khoảng nhiễu của 244 mẫu out-of-fold, nên **pipeline chính giữ không winsorize** cho đơn giản; lợi ích thật nằm ở mô hình **tuyến tính** (+1,85 điểm % với `p1p99`).
5. **Kiểm định thống kê nói thật:** RF vs `ticker_prior`: ΔAUROC = −0,0030, **p = 0,7546** (DeLong); ΔAP = +0,0061, CI95 [−0,0048; +0,0215], p = 0,344.

## 1.5. Phân tích chuyên sâu (Confusion Matrix · Error Analysis · Overfit)

**Confusion Matrix (Random Forest, test n = 64)**

| Ngưỡng | TN | FP | FN | TP | Precision | Recall | F1 | Chi phí kỳ vọng (FN = 5, FP = 1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,500 | 23 | 3 | 1 | 37 | 0,925 | 0,974 | 0,949 | 8,0 |
| **0,783** (tối ưu chi phí, chọn trên VAL) | 25 | 1 | 2 | 36 | **0,973** | 0,947 | **0,960** | **6,0** ✅ |
| 0,788 (vận hành) | 25 | 1 | 2 | 36 | 0,973 | 0,947 | 0,960 | 11,0 |

→ **Nhầm lẫn chính = FALSE NEGATIVE** (bỏ sót doanh nghiệp suy giảm). Ở ngưỡng 0,5 mô hình chỉ có
1 FN nhưng 3 FP ⇒ hạ/ nâng ngưỡng là **đổi chỗ FP ↔ FN**, không phải "cải thiện mô hình".

**Phân tích lỗi thực tế (3/64 mẫu sai, đã truy vết tới công ty & chỉ tiêu)**

| Mẫu | Công ty | Thực tế | P(distress) | Loại lỗi | Đặc điểm chung |
|---|---|---:|---:|---|---|
| `FIVE-2024Q3` | FIVE | 1 | 0,174 | **FN (nặng)** | Công ty có nhãn rất thưa (19,4%) ⇒ mô hình thiên về "an toàn"; SHAP: `current_ratio_min_window` (−0,12), `current_ratio_latest` (−0,11), `debt_to_assets_latest` (−0,10) đều **kéo về an toàn** |
| `HD-2024Q2` | HD | 1 | 0,783 | FN (sát ngưỡng) | HD có 100% nhãn 1 nhưng quý này P = 0,783 < 0,788; SHAP: `debt_to_assets_latest` (+0,09), `debt_to_equity_latest` (+0,08) đẩy về suy giảm, bị `current_ratio_min_window` (−0,07) kéo lại |
| `DG-2025Q2` | DG | 0 | 0,798 | FP | Nhãn DG lưỡng cực (54,8%) ⇒ quý "khỏe" nằm sát biên quyết định |

**Nguyên nhân gốc (có bằng chứng):** đặc trưng hai lớp **chồng lấn** (ECDF top feature: `current_ratio_latest`
chỉ AUC 0,074 theo hướng nghịch); **nhãn không tái tạo được** từ dữ liệu công bố (quy tắc kế toán khớp
tối đa **74,7%**; phản chứng `WMT-2015Q2` lãi dương nhưng nhãn = 1) ⇒ có **nhiễu nhãn**, trần hiệu
năng bị giới hạn bởi **chất lượng nhãn**, không phải thuật toán.

**Overfitting / Underfitting (Train vs Validation vs Test)**

| Mô hình | Train AUROC | Val AUROC | Gap | Train F1 | Val F1 | Gap F1 | Val AP | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 0,984 | 0,965 | **0,019** | 0,935 | 0,976 | −0,040 | 0,987 | 0,051 |
| **Random Forest** | **1,000** | 0,965 | 0,034 | 0,985 | 0,976 | **+0,009** | 0,987 | 0,046 |
| HistGradientBoosting | 1,000 | **0,974** | 0,026 | 1,000 | 0,976 | +0,024 | 0,989 | **0,037** |
| LightGBM | 1,000 | 0,965 | 0,035 | 1,000 | 0,976 | +0,024 | 0,987 | 0,035 |

- **Có overfit nhẹ** (Train AUROC 1,000 vs Val 0,965–0,974; gap 0,019–0,035) — điển hình dữ liệu nhỏ (212 mẫu train).
- **Kiểm soát đã áp dụng:** giới hạn độ phức tạp (`max_depth` 3–6, `min_samples_leaf=2`,
  `l2_regularization=1.0`, `num_leaves=15`, `reg_lambda=1.0`); `class_weight` thay resample;
  **`StratifiedGroupKFold` + LOCO** để chẩn đoán tổng quát hoá; **ngưỡng chọn trên validation/OOF**;
  **bootstrap CI 2.000 vòng**; scaler chỉ dùng cho mô hình tuyến tính.
- **Học chưa bão hoà:** learning curve (chia theo công ty): train 0,998→0,989 nhưng **val 0,795 → 0,902**
  khi train 83→162 mẫu ⇒ cần **thêm dữ liệu/thực thể**, không phải thêm thuật toán.

**Kiểm chứng độ nhạy theo định nghĩa nhãn (RQ4)** — 4 định nghĩa, mô hình cố định RF:

| Định nghĩa | Dương (test) | IR train | Khớp nhãn gốc | AUROC test (mô hình / ticker-prior) | Cross-company AP |
|---|---:|---:|---:|---:|---:|
| `original` (mốc) | 59,4% | 1,65 | 100% | 0,983 / 0,986 | 0,958 |
| `stress_signals` | 42,2% | 1,41 | 74,4% | 0,909 / **0,960** | 0,610 |
| `altman_z` (Z'' < 1,1) | 31,2% | 3,42 | 63,9% | 0,999 / 0,991 | 0,460 |
| `forward_4q` (4 quý tới) | 48,4% | 1,99 | 66,7% | 0,836 / **0,911** | 0,582 |

⇒ Luận điểm "mô hình không vượt `ticker_prior`" đúng ở **3/3 định nghĩa tái lập được** ⇒ kết luận
**ổn định theo định nghĩa nhãn**; nhưng khả năng tổng quát hoá cross-company **sụp mạnh** dưới nhãn thay
thế (AP 0,46–0,61 so với 0,958) ⇒ phải nêu trong hạn chế.

---

# PHẦN 2 — DÀN BÀI SLIDE BÁO CÁO (11 SLIDES / 10 PHÚT)

> Quy ước: mỗi bullet **tối đa 10 từ**; phần *Speaker notes* là lời thoại 3–4 câu.

### Slide 1 — Trang tiêu đề

- **Takeaway:** Một bài toán "điểm cao" nhưng kết luận trung thực về giới hạn dữ liệu.
- **Hiển thị:** Tên đề tài · GVHD · Thành viên · dòng lệnh tái lập duy nhất.
- **Bullets:**
  - Dự báo suy giảm tài chính doanh nghiệp bán lẻ Mỹ
  - Dữ liệu SEC XBRL · 8 công ty · 324 mẫu · 47 features
  - Tái lập 100%: seed 42, split byte-identical
  - Audit 98.192 phép kiểm tra · 220 test tự động
- **Speaker notes:** "Đồ án không chỉ dừng ở AUROC 0,983 mà còn chỉ ra vì sao con số đó **chưa** chứng minh năng lực dự báo. Toàn bộ số liệu trên slide đều sinh từ artifact trong repo và tái lập bằng một lệnh, nên nhóm sẵn sàng chạy lại ngay trong buổi bảo vệ nếu thầy/cô yêu cầu."

### Slide 2 — Đặt vấn đề & Mục tiêu

- **Takeaway:** Dự báo rủi ro quý kế tiếp, KHÔNG nhận diện danh tính công ty.
- **Hiển thị:** sơ đồ `lịch sử ≤ as_of → quý target → nhãn 0/1`.
- **Bullets:**
  - Câu hỏi: quý tới doanh nghiệp có suy giảm?
  - 4 RQ: mô hình · đặc trưng · công ty mới · nhãn
  - Ngoài phạm vi: giá cổ phiếu, tín dụng, vĩ mô
  - Sản phẩm: pipeline + báo cáo + bộ kiểm chứng
- **Speaker notes:** "Mục tiêu không phải đoán tên công ty mà dự báo sự kiện. Nhóm tự đặt 4 câu hỏi nghiên cứu, trong đó RQ3 và RQ4 là phần phản biện chính: mô hình có tổng quát cho công ty chưa từng thấy, và kết luận có phụ thuộc định nghĩa nhãn."

### Slide 3 — Khám phá Dữ liệu (Dataset & Phân phối nhãn)

- **Takeaway:** Mất cân bằng **nhẹ ở cấp mẫu** nhưng **nghiêm trọng ở cấp công ty**.
- **Hiển thị:** bảng split + barh tỉ lệ nhãn theo công ty.
- **Bullets:**
  - 332 quý → 324 mẫu; 47 features; 2 lớp
  - Train/Val/Test/Purge = 212/32/64/16
  - Dương 62,3% (IR 1,66); entropy 0,956 bit
  - HD/LOW/WMT 100% nhãn 1; ROST 4,7%
  - 41% mẫu thuộc công ty đơn lớp
- **Speaker notes:** "Tỉ lệ 62/38 thoạt nhìn 'dễ chịu' nên dễ tưởng không cần xử lý mất cân bằng. Nhưng khi tách theo công ty, ba doanh nghiệp có 100% nhãn 1 còn ROST chỉ 4,7% — IR lên tới 20,5. Vì vậy nhóm coi accuracy là chẩn đoán, không phải thước đo chính."

### Slide 4 — EDA trọng tâm

- **Takeaway:** Ba phát hiện định hình toàn bộ thiết kế thực nghiệm.
- **Hiển thị:** coverage heatmap · histogram `debt_to_equity` · heatmap tương quan có cụm.
- **Bullets:**
  - Khuyết có cấu trúc: 5/16 chỉ tiêu phủ < 50%
  - MNAR: 22/47 feature; Δ nhãn tới −55,7 điểm %
  - Đuôi nặng: skew −14,9; 30,6% ngoại lai
  - Đa cộng tuyến: VIF > 10 ở 33/47; 7 cụm
  - Số chiều hiệu dụng chỉ 12,89/47
- **Speaker notes:** "Ba con số cần nhớ: 22/47 feature có tỉ lệ nhãn khác nhau giữa nhóm thiếu và có dữ liệu, tức giá trị thiếu **mang thông tin**; `debt_to_equity` lệch −14,9 nên Pearson vô dụng; và 33/47 cột VIF > 10 nghĩa là hệ số logistic không thể diễn giải nhân quả."

### Slide 5 — Quy trình Tiền xử lý

- **Takeaway:** Mọi phép biến đổi nằm TRONG Pipeline, fit chỉ trên train.
- **Hiển thị:** sơ đồ ngang 5 khối Input → Winsorize → Impute → Scale → Model.
- **Bullets:**
  - Winsorize IQR (ngưỡng học từ train)
  - Imputer median (trong pipeline, mỗi fold)
  - Scale: StandardScaler (tuyến tính); none (cây)
  - class_weight="balanced*" tính trong fit
  - Không resample; không ticker one-hot
- **Speaker notes:** "Đây là slide chống rò rỉ. Winsorize, impute và scale đều nằm trong Pipeline nên mỗi fold học thống kê của riêng nó; test chỉ được transform. Nhóm cũng cố ý không one-hot mã cổ phiếu vì làm vậy sẽ hợp thức hoá đúng loại rò rỉ mà đồ án đang đo."

### Slide 6 — Thuật toán & Tinh chỉnh siêu tham số

- **Takeaway:** 4 họ mô hình; chọn theo AP cross-company, không theo F1 in-domain.
- **Hiển thị:** bảng siêu tham số + sơ đồ quy tắc chọn mô hình.
- **Bullets:**
  - Logistic · Random Forest · HGB · LightGBM
  - GridSearchCV 8–18 cấu hình + StratifiedGroupKFold
  - refit = average_precision (không phụ thuộc ngưỡng)
  - Random search 25 trial + sổ `runs.csv` 98 dòng
  - XGBoost bị chặn phiên bản; đã ghi cách sửa
- **Speaker notes:** "Điểm khác biệt là tiêu chí chọn: nhóm xếp hạng theo AP cross-company, nghĩa là mô hình phải chịu được công ty chưa từng thấy. GridSearchCV chạy trên StratifiedGroupKFold và refit theo AP; nhóm còn chạy random search và ghi mọi trial vào sổ để so sánh công bằng với lưới cũ — kết quả âm cũng được lưu."

### Slide 7 — Bảng so sánh hiệu năng (Test set)

- **Takeaway:** RF thắng sát nút; khoảng cách giữa các mô hình < 0,01 AUROC.
- **Hiển thị:** bảng 6 dòng + hàng "RF @ ngưỡng vận hành".
- **Bullets:**
  - RF: Acc 0,938 · P 0,925 · R 0,974 · F1 0,949
  - RF@0,788: Acc 0,953 · P 0,973 · F1 0,960
  - MCC 0,904 · Brier 0,058 · AP 0,991
  - `ticker_prior`: AUROC 0,986 > mô hình 0,983
  - Cross-company 0,933 (in-domain 0,983)
- **Speaker notes:** "Bảng này có một dòng gây khó chịu cho nhóm: baseline chỉ dùng **tỉ lệ nhãn trung bình của công ty** đạt AUROC 0,986, cao hơn mô hình học máy. Nhóm giữ nguyên dòng đó vì che nó đi thì phần kết quả còn lại trở nên vô nghĩa."

### Slide 8 — Ma trận nhầm lẫn

- **Takeaway:** Lỗi chính là **bỏ sót suy giảm (FN)**; ngưỡng chỉ đổi chỗ FN/FP.
- **Hiển thị:** 2 confusion matrix (0,5 và 0,788) + đường Precision-Recall.
- **Bullets:**
  - @0,5: TN 23 · FP 3 · FN 1 · TP 37
  - @0,788: TN 25 · FP 1 · FN 2 · TP 36
  - Chi phí kỳ vọng: 8,0 → 6,0 tại 0,783
  - FN đắt gấp 5 lần FP (COST_FN=5)
- **Speaker notes:** "Nhóm biến ngưỡng thành **tham số nghiệp vụ**: nếu bỏ sót đắt gấp 5 lần báo động giả thì ngưỡng tối ưu là 0,783 và chi phí kỳ vọng giảm từ 8 xuống 6. Không có ngưỡng nào đúng cho mọi tổ chức — nó phải để trong file cấu hình và ghi log khi ra quyết định."

### Slide 9 — Phân tích lỗi & Overfitting

- **Takeaway:** Sai số có cấu trúc, liên quan chất lượng NHÃN hơn là thuật toán.
- **Hiển thị:** bảng 3 mẫu sai + SHAP cục bộ + bảng gap train/val.
- **Bullets:**
  - 3/64 sai: 2 FN (HD, FIVE), 1 FP (DG)
  - FN nặng: FIVE-2024Q3 chỉ P = 0,174
  - SHAP: `current_ratio` kéo về "an toàn"
  - Train AUROC 1,000 vs Val 0,965 (gap 0,034)
  - Learning curve chưa bão hoà (0,795 → 0,902)
- **Speaker notes:** "Nhóm không dừng ở '3 mẫu sai' mà truy vết bằng SHAP: FIVE-2024Q3 bị bỏ sót vì mọi chỉ số thanh khoản đều kéo về an toàn trong khi nhãn thì ngược lại. Kết hợp với việc quy tắc kế toán chỉ khớp 74,7% nhãn gốc, nhóm kết luận trần hiệu năng bị giới hạn bởi **định nghĩa nhãn**."

### Slide 10 — Kết luận & Ứng dụng thực tế

- **Takeaway:** Dùng làm **công cụ sàng lọc + ưu tiên thẩm định**, không phải kết luận tự động.
- **Hiển thị:** 3 khối: Kết quả · Giới hạn · Cách dùng.
- **Bullets:**
  - RF + ngưỡng theo chi phí: F1 0,960
  - Nhưng ΔAUROC vs ticker-prior p = 0,7546
  - Luận điểm ổn định ở 3/3 định nghĩa nhãn
  - Dùng: xếp hạng rủi ro, review thủ công
  - Không dùng: tự động từ chối/hạ hạng tín dụng
- **Speaker notes:** "Ứng dụng đúng nhất là **xếp hạng để ưu tiên thẩm định**, vì mô hình chỉ chắc chắn hơn ngẫu nhiên ở chỗ đó. Kiểm định DeLong cho thấy lợi thế so với baseline chưa có ý nghĩa ở n = 64, nên nhóm khuyến nghị không dùng để ra quyết định tự động."

### Slide 11 — Hạn chế & Hướng phát triển

- **Takeaway:** Ba nút thắt thật: nhãn, số thực thể, dữ liệu ngoài BCTC.
- **Hiển thị:** bảng P0/P1/P2 ngắn.
- **Bullets:**
  - P0: chốt định nghĩa nhãn chính thức
  - P1: thêm công ty; lọc feature theo cụm/VIF
  - P1: mô hình panel/survival cho chuỗi quý
  - P2: monitoring PSI/KS; ngưỡng cấu hình hoá
  - Đã có sẵn: SHAP, drift, kiểm định ý nghĩa
- **Speaker notes:** "Nhóm chủ động liệt kê hạn chế thay vì chờ bị hỏi: nhãn không tái lập được, chỉ 8 thực thể, thiếu dữ liệu phi tài chính. Phần kỹ thuật cho hai hạn chế đầu đã xong — gồm 4 định nghĩa nhãn, kiểm định thống kê, SHAP và monitoring drift — nên bước tiếp theo là mở rộng dữ liệu."

---

# PHẦN 3 — 8 CÂU HỎI PHẢN BIỆN & KỊCH BẢN TRẢ LỜI XUẤT SẮC

> Khuôn 4 bước cho MỌI câu: **① Thừa nhận** → **② Luận điểm kỹ thuật** → **③ Số liệu** → **④ Kết luận**.

### Q1. Mất cân bằng — vì sao (không) dùng Accuracy? Đo gì đáng tin hơn?

- **① Thừa nhận:** "Dữ liệu lệch lớp cả ở cấp mẫu và cấp công ty, nên Accuracy là chỉ số gây hiểu nhầm."
- **② Luận điểm kỹ thuật:** Chỉ đoán lớp đa số đã đạt 62,3%; với 8 thực thể, accuracy còn bị chi phối bởi
  việc "nhận ra công ty". Vì vậy thước đo phải (a) **không phụ thuộc ngưỡng** và (b) **đánh giá được
  khả năng tổng quát hoá theo nhóm**.
- **③ Số liệu:**
  - `dummy_most_frequent`: Accuracy **0,594** nhưng F1 **0,745** (đoán bừa toàn lớp 1) ⇒ F1 cũng bị lừa.
  - IR cấp mẫu 1,66 nhưng **IR cấp ROST lên 20,5**; entropy 0,956 bit; 41,0% mẫu thuộc công ty đơn lớp.
  - `ticker_prior` (không dùng feature) **AUROC 0,986 > mô hình 0,983**.
  - Khi giữ trọn công ty ngoài train, AUROC tụt còn **0,933**; LOCO **0,610**.
- **④ Kết luận:** Báo cáo dùng **PR-AUC (AP) làm chỉ số chính**, kèm F1/macro-F1, MCC, Brier và **CI bootstrap
  2.000 vòng**; mọi so sánh đều in cạnh baseline `ticker_prior` cùng **kiểm định DeLong (p = 0,7546)** để
  không tự lừa mình.

### Q2. Chuẩn hoá/xử lý trước hay sau khi chia tập — có rò rỉ không?

- **① Thừa nhận:** "Nếu impute/scale trước khi chia tập thì đó là rò rỉ thống kê kinh điển, và nhóm đã
  thiết kế để điều đó không thể xảy ra."
- **② Luận điểm kỹ thuật:** Toàn bộ phép biến đổi nằm trong `sklearn.Pipeline` ⇒ mỗi lần `fit` chỉ học thống kê
  từ dữ liệu train của **đúng fold đó**, `transform` mới áp cho validation/test. Split là **theo thời gian
  trong từng công ty** + **purge 16 mẫu**; feature chỉ tính từ lịch sử có `available_on ≤ as_of`.
- **③ Số liệu:** 7 lớp chống rò rỉ (mục 4.6 của `BAO-CAO.md`); test tự động
  `test_no_label_leak_in_features`, `test_main_pipeline_has_no_balancer`,
  `test_balancing_weights_come_from_fit_labels_only`, `test_cv_folds_are_stratified_and_grouped`;
  thí nghiệm lệch lớp **so khớp ma trận test trước/sau `fit` → PASS cho 7/7 kỹ thuật**;
  audit **98.192 phép kiểm tra, 0 phát hiện**.
- **④ Kết luận:** Không có rò rỉ thống kê. Rò rỉ **thực thể** (mẫu cùng công ty ở cả train và test) là hiện
  tượng **được đo và báo cáo** (in-domain 0,983 vs cross-company 0,933), không phải điều bị che.

### Q3. Vì sao chọn Random Forest mà không phải boosting/logistic? Bản chất có ưu thế gì?

- **① Thừa nhận:** "Ba họ mô hình gần như **hoà nhau** (< 0,01 AUROC) — chọn RF không phải vì nó vượt trội
  về thuật toán."
- **② Luận điểm kỹ thuật:** Trên 212 mẫu train với **33/47 cột VIF > 10** và đuôi nặng (`debt_to_equity`
  skew −14,9), RF (bagging + chọn ngẫu nhiên tập feature tại mỗi split) **giảm phương sai** và bền với
  cột trùng thông tin/ngoại lai, không cần scaler; logistic phụ thuộc scaler và bị ngoại lai chi phối;
  boosting dễ dồn importance vào vài cột mạnh.
- **③ Số liệu:**
  - Cross-company OOF AUROC: RF **0,933** = LightGBM 0,933 > HGB 0,926 > logistic 0,912.
  - AP cross-company: RF **0,958** > HGB 0,957 > LightGBM 0,947.
  - Test AP 0,991 (RF) vs 0,989 (logistic); precision 0,925 vs HGB 0,902 ở cùng recall 0,974.
  - `random_forest + winsorize IQR` = **0,9590 AP cross-company**, cao nhất trong 24 cấu hình
    (tham chiếu `logistic + StandardScaler` = 0,9232).
  - Ablation (mục 7.4): bỏ nhóm `ratios_latest` làm AUROC test giảm mạnh nhất ⇒ tín hiệu tập trung ở nhóm cấu trúc vốn/thanh khoản hiện tại.
- **④ Kết luận:** RF là lựa chọn **theo quy tắc công bố trước** (max AP cross-company) và hợp lý về bias–variance;
  đi kèm đó, kết luận định lượng quan trọng hơn là **"thuật toán không phải nút thắt"**.

### Q4. Train vs Validation có khoảng cách không? Chống overfitting bằng gì?

- **① Thừa nhận:** "Có — train AUROC chạm 1,000 trong khi validation chỉ 0,965–0,974, gap 0,019–0,035."
- **② Luận điểm kỹ thuật:** Với 212 mẫu và 47 cột, cây sâu sẽ "nhớ" dữ liệu. Nhóm kiểm soát bằng
  **giới hạn độ phức tạp** (depth 3–6, `min_samples_leaf=2`, `l2=1.0`, `num_leaves=15`, `reg_lambda=1.0`),
  **`class_weight` thay cho resample**, **CV theo nhóm công ty**, **ngưỡng chọn trên OOF/val**
  và **bootstrap 2.000 vòng** để báo cáo độ bất định.
- **③ Số liệu:**
  - gap F1 của RF chỉ **+0,009** (logistic −0,040; HGB +0,024; LightGBM +0,024).
  - val AP 0,987–0,989; Brier val 0,035–0,051; test AUROC 0,983 với **CI95 = [0,947; 1,000]**.
  - Learning curve: train 0,998→0,989 nhưng **val 0,795→0,902** khi train tăng 83→162 mẫu
    ⇒ **underfitting do thiếu dữ liệu**, không phải cần regularize thêm.
- **④ Kết luận:** Overfitting ở mức chấp nhận được và có kiểm soát; nút thắt thật là **số lượng thực thể**,
  không phải độ phức tạp mô hình.

### Q5. Lớp nào bị nhầm sang lớp nào nhiều nhất? Hậu quả FN/FP khi triển khai?

- **① Thừa nhận:** "Nhầm lẫn chủ yếu là **âm tính giả (FN)** — mô hình nói 'an toàn' trong khi doanh nghiệp
  thực sự suy giảm."
- **② Luận điểm kỹ thuật:** Nhãn gắn với thực thể (HD/LOW/WMT luôn = 1; ROST 4,7%) và hai lớp chồng lấn mạnh,
  nên mô hình có xu hướng "đóng dấu" công ty; **ngưỡng** là biến quyết định trực tiếp tỉ lệ FN/FP.
- **③ Số liệu:**
  - @0,5: TN 23 / FP 3 / **FN 1** / TP 37 (precision 0,925 · recall 0,974).
  - @0,783 (tối ưu chi phí với FN = 5·FP): TN 25 / **FP 1** / **FN 2** / TP 36, **chi phí kỳ vọng 6,0 so với 8,0**.
  - 2 FN là `HD-2024Q2` (P = 0,783) và `FIVE-2024Q3` (P = 0,174); FP là `DG-2025Q2` (P = 0,798).
- **④ Kết luận:** Hệ thống chỉ nên là **bộ lọc xếp hạng rủi ro + ưu tiên thẩm định thủ công**; ngưỡng để trong
  cấu hình và **ghi log mọi quyết định** (`sample_id`, P, ngưỡng, phiên bản model, SHA-256 artifact).

### Q6. Siêu tham số chọn thế nào? Tiêu chí dừng là gì?

- **① Thừa nhận:** "Nhóm dùng cả lưới (`GridSearchCV`) lẫn random search và ghi lại toàn bộ trial."
- **② Luận điểm kỹ thuật:** Lưới chạy trên **`StratifiedGroupKFold(4)`** với **`refit="average_precision"`**
  (AP không phụ thuộc ngưỡng), luôn so với **cấu hình mặc định trên cùng splitter**; random search dùng mẫu
  **log-uniform** cho tham số scale; mọi trial ghi vào **`reports/results/runs.csv`** ⇒ không có lựa chọn nào
  không truy vết được.
- **③ Số liệu:**
  - Lưới so với mặc định: logistic **+0,026**, HGB +0,015, RF +0,003.
  - Random search so với lưới: LightGBM **+0,049**, HGB **+0,026**, logistic +0,007, **RF −0,004**.
  - `runs.csv`: 98 dòng (run_id, model, params, seed, CV-AP ± std, thời gian, trạng thái).
- **④ Kết luận:** Tiêu chí dừng = **không còn cải thiện AP có ý nghĩa trên CV** so với mặc định/lưới; mọi lựa
  chọn mô hình **không bao giờ dùng test**. Việc tinh chỉnh rộng không giúp RF được **báo cáo thay vì che**.

### Q7. Cho xem 1–2 mẫu sai tiêu biểu. Tại sao mô hình "nghĩ" khác?

- **① Thừa nhận:** "Nhóm chọn 2 mẫu sai có phân rã SHAP đầy đủ: `FIVE-2024Q3` và `HD-2024Q2`."
- **② Luận điểm kỹ thuật:** KernelSHAP **tự cài đặt** (môi trường không có gói `shap`), 64 mẫu × 200 liên minh,
  có **kiểm chứng giải tích** cho hàm tuyến tính ⇒ giải thích đáng tin, không phải hình minh hoạ.
  - `FIVE-2024Q3` (thực tế = 1, P = 0,174): đóng góp **âm** chi phối — `current_ratio_min_window` **−0,12**,
    `current_ratio_latest` −0,11, `debt_to_assets_latest` −0,10, `working_capital_to_assets` −0,07
    ⇒ mô hình thấy thanh khoản/đòn bẩy "khoẻ" nên kết luận an toàn.
  - `HD-2024Q2` (P = 0,783, sát ngưỡng 0,788): đóng góp **dương** `debt_to_assets_latest` +0,09,
    `debt_to_equity_latest` +0,08 nhưng bị `current_ratio_min_window` −0,07 kéo lại ⇒ lệch chỉ 0,005.
- **③ Số liệu:** sai số **efficiency** |Σφ + E[f] − f(x)| = **2,2e-16**; top feature toàn cục:
  `current_ratio_latest` (mean|φ| 0,0765), `current_ratio_min_window` (0,0764), `debt_to_assets_latest` (0,0583);
  đồng thuận với permutation importance ở mức Spearman **0,362**; quy tắc kế toán chỉ khớp **74,7%** nhãn gốc.
- **④ Kết luận:** Đây là **lỗi của nhãn/dữ liệu hơn là lỗi thuật toán**, và chính vì vậy nhóm chạy kiểm chứng
  độ nhạy trên nhiều định nghĩa nhãn — kết luận giữ nguyên ở **3/3 định nghĩa tái lập được**.

### Q8. Nếu đưa vào production, thách thức về độ trễ và drift là gì?

- **① Thừa nhận:** "Về độ trễ thì đơn giản; **drift và độ tin cậy của nhãn** mới là vấn đề lớn."
- **② Luận điểm kỹ thuật:**
  - (a) Suy luận rất nhẹ: RF 300–500 cây trên 47 đặc trưng, chạy CPU đơn; nút thắt độ trễ nằm ở
    **pipeline lấy dữ liệu XBRL** (crawl → chuẩn hoá quý), không ở model.
  - (b) Nhãn có **độ trễ công bố** (`label_available_on` trung vị **34 ngày** sau kỳ target) ⇒ monitoring
    phải tách "feature drift" khỏi "label lag".
  - (c) Ngưỡng là **biến nghiệp vụ** (FN đắt gấp 5 lần FP) ⇒ cấu hình hoá, không hardcode.
- **③ Số liệu:** drift train→test đã đo: **5 feature** vượt ngưỡng (nặng nhất `debt_to_assets_yoy` KS **0,621**,
  SMD −0,928); **41 feature PSI > 0,2** (cần ≥ 64 dòng nên PSI nhiễu — dùng KS/SMD làm tiêu chí chính);
  **22/47 feature MNAR** ⇒ khi dữ liệu thiếu lan rộng, phân phối score sẽ dịch; `ticker_prior` = 0,986
  cho thấy rủi ro lớn nhất là **model chỉ học danh tính công ty** (gặp công ty mới: 0,933; LOCO 0,610).
- **④ Kết luận:** Kế hoạch vận hành: (1) **monitoring PSI/KS theo feature + AP cross-company định kỳ**
  (cảnh báo khi PSI > 0,2 hoặc AP cross-company giảm > 0,05); (2) **rollback** được vì mọi artifact có
  SHA-256 trong manifest; (3) **log quyết định** từng hồ sơ; (4) tái huấn luyện khi có công ty/nhãn mới và
  **chốt lại định nghĩa nhãn** trước khi dùng cho bất kỳ quyết định tín dụng nào.

---

## PHỤ LỤC — 6 con số "đinh" nên nhớ để phản biện

| Con số | Ý nghĩa |
|---|---|
| **324 / 47 / 212-32-64** | Mẫu · số feature · train-val-test |
| **62,3% · IR 1,66 · IR công ty tới 20,5** | Mất cân bằng nhẹ ở cấp mẫu, **nặng ở cấp thực thể** |
| **0,983 AUROC · 0,991 AP · F1 0,960 · MCC 0,904** | Mô hình được chốt trên test |
| **0,986 (ticker_prior) · 0,933 (cross-company) · 0,610 (LOCO)** | Ba con số phản biện — mô hình chưa chắc hơn baseline |
| **p = 0,7546 (DeLong) · CI95 AUROC [0,947; 1,000]** | Độ bất định trên n = 64 |
| **3/3 định nghĩa nhãn · ~1e-16 (SHAP) · 0 phát hiện / 98.192 phép kiểm tra · 220 test** | Độ vững của kết luận & mức độ tái lập |

---

## PHỤ LỤC — Bản đồ artifact ↔ số liệu (để truy vết khi bị hỏi)

| Số liệu trong tài liệu | File nguồn |
|---|---|
| Bảng so sánh test, confusion matrix, per-class | `reports/results/baselines.json`, `reports/results/test_evaluation.json` |
| Cross-company AUROC/AP, LOCO, learning curve | `reports/results/validation_checks.json`, `reports/results/analysis.md` |
| EDA (skew, VIF, missing, drift, ECDF) | `reports/results/eda_deep.md`, `reports/results/eda.md`, `reports/figures/**` |
| Winsorize/scaler + ΔAP cross-company | `reports/results/preprocessing_experiment.md` |
| SMOTE/RUS/Focal Loss, so khớp ma trận test | `reports/results/imbalance_real.md`, `reports/results/class_balance.md` |
| SHAP toàn cục/cục bộ, kiểm chứng efficiency | `reports/results/shap.md` |
| DeLong + bootstrap, p-value | `reports/results/significance.md` |
| 4 định nghĩa nhãn, độ nhạy | `reports/results/label_sensitivity.md`, `forecasting/labels.py` |
| Grid/random search, sổ trial | `reports/results/tuning.md`, `reports/results/search.md`, `reports/results/runs.csv` |
| Tổng hợp & kết luận toàn đồ án | `reports/results/summary.md`, `docs/BAO-CAO.md` |
| Kiểm tra dữ liệu & toàn vẹn | `reports/results/data_audit.md`, `scripts/audit_data.py` |

> Lệnh tái lập toàn bộ: `python -m scripts.run_all` · kiểm định: `python -m unittest discover -s tests`
> · audit: `python -m scripts.audit_data`.
