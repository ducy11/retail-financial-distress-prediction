---
marp: true
theme: default
paginate: true
size: "16:9"
---

<!--
=========================================================================
  FILE MARKDOWN TỰ CHỨA ĐỂ SINH SLIDE BẰNG CÔNG CỤ AI
  Đồ án CS114 — Dự báo suy giảm tài chính (financial distress) doanh nghiệp bán lẻ

  CÁCH DÙNG:
  1) Vào Gamma / Canva Magic Design / Tome / ChatGPT / Copilot / SlidesAI…
     dán nội dung file này TỪ dòng "## Slide 1" ĐẾN hết slide "Cảm ơn thầy/cô đã lắng nghe"
     (BỎ Phụ lục A/B/C ở cuối — đó là bảng tra cứu, không lên slide), kèm câu lệnh gợi ý ở
     mục "PROMPT GỢI Ý" (Phụ lục C) ở cuối file.
  2) Nếu dùng Marp / Slidev / Obsidian: `---` = hết slide. Mở trực tiếp file này là ra slide.
  3) Mỗi slide có 3 khối:
       • Dòng "**Thông điệp chính:**"  → chữ LỚN nhất trên slide (takeaway).
       • Gạch đầu dòng                  → nội dung hiển thị.
       • Dòng in nghiêng trong ngoặc *(...)* → LỜI THOẠI, KHÔNG đưa lên slide.
     Dòng "Ảnh: ..." là đường dẫn ảnh trong repo để bạn tải ảnh lên và chèn vào slide.
  4) Mọi số liệu đều sinh từ artifact trong repo (reports/results/*.json) nên
     KHÔNG chế thêm số khi để AI trình bày lại.
  5) File này thay thế docs/BAO-CAO-slide-bao-ve.pptx (bản .pptx do export_office
     sinh ra hiện bị PowerPoint từ chối mở dù zip hợp lệ). Bản .docx
     docs/BAO-CAO-slide-bao-ve.docx vẫn mở được bằng Word nếu cần.
=========================================================================
-->

<!-- _class: lead -->

# Dự báo suy giảm tài chính doanh nghiệp bán lẻ từ chỉ số tài chính

## Đồ án môn học CS114 — nhóm 1 người

- Dữ liệu **SEC XBRL thật**: 8 chuỗi bán lẻ Mỹ · **324 mẫu** · **47 đặc trưng**
- **4 họ mô hình** (Logistic Regression · Random Forest · HistGradientBoosting · MLP)
- Kiểm chứng 3 góc: in-domain · **công ty chưa từng thấy** · **giai đoạn mới**
- **234 test tự động · 98.200 phép kiểm tra đối soát, 0 phát hiện**
- Kết luận trung thực: mô hình **chưa** chắc hơn baseline “nhớ mặt công ty”

*(Thời lượng ~12 phút; bộ tài liệu bảo vệ đầy đủ ở docs/bo-tai-lieu-bao-ve.md)*

---

## Slide 1. Vấn đề & 4 câu hỏi nghiên cứu

**Thông điệp chính:** Dự báo **quý kế tiếp** doanh nghiệp có rơi vào suy giảm tài chính — không phải nhận diện công ty.

- **Bài toán:** phân loại nhị phân + xếp hạng rủi ro cho 1 quý tới (~90 ngày sau `as_of`)
- **Dữ liệu đầu vào:** 16 chỉ tiêu báo cáo tài chính quý (SEC XBRL), cửa sổ lịch sử ≤ 8 quý
- **RQ1** — 4 họ mô hình (tuyến tính / bagging / boosting / mạng nơ-ron) khác nhau thế nào?
- **RQ2** — đặc trưng nào quyết định kết quả?
- **RQ3** — mô hình có tổng quát hoá sang **công ty chưa từng thấy** (và **giai đoạn mới**) không?
- **RQ4** — kết luận có giữ nguyên khi đổi **định nghĩa nhãn** không?
- **Ngoài phạm vi:** giá cổ phiếu, dữ liệu vĩ mô, xếp hạng tín dụng

*(Điểm nhấn: RQ3 và RQ4 là phần nhóm tự phản biện chính kết quả của mình.)*

---

## Slide 2. Dữ liệu: nguồn thật, có kiểm chứng ngược

**Thông điệp chính:** Mọi con số **tra ngược được** về filing gốc của SEC — không có ô nào được “điền cho đủ”.

- **Nguồn:** SEC XBRL `companyfacts` (10-K/10-Q); 8 chuỗi bán lẻ: WMT, HD, LOW, ROST, DG, ORLY, DKS, FIVE
- **Quy mô:** 332 quý → **324 mẫu** dự báo; **47 đặc trưng** dựng từ lịch sử của chính mẫu
- **Chia tập theo thời gian:** train/validation/test = **212/32/64** + **16 mẫu purge** (dải đệm chống rò rỉ)
- **Chống rò rỉ thời gian:** mọi dòng lịch sử có `available_on ≤ as_of` (chỉ dùng thông tin đã công bố)
- **Kiểm chứng nguồn gốc:** 20 file SEC đã băm SHA-256 (**0 lệch**) · **4.609 ô** tra ngược trong companyfacts (**0 thiếu**) · **0 lỗi quy đổi** · **703 ô không có dữ liệu → để `null`**
- **Tỉ lệ lớp:** dương 62,3% (IR 1,66) — lệch **nhẹ** ở cấp mẫu nhưng **rất nặng** ở cấp công ty

Ảnh: `../reports/figures/eda/02_class_balance.png` → *Phân bố lớp theo tập*

*(Nhóm không tự sinh dữ liệu. Nếu thầy/cô yêu cầu, chạy `python -m scripts.verify_provenance` để băm lại toàn bộ snapshot SEC ngay tại buổi bảo vệ.)*

---

## Slide 3. EDA: ba phát hiện định hình thiết kế

**Thông điệp chính:** Dữ liệu tài chính quý **thiếu có cấu trúc, đuôi nặng và đa cộng tuyến** — cả ba đều được xử lý tường minh.

- **Thiếu có cấu trúc:** **5/16 chỉ tiêu có độ phủ TỐI THIỂU < 50%** (`liabilities` mean 39,1% · min 0%) ⇒ nợ phải trả được **suy ra** từ A = L + E (nhờ đó `debt_to_assets`/`debt_to_equity` phủ 100%)
- **Thiếu dữ liệu MANG thông tin (MNAR):** 22/47 đặc trưng có q < 5%; chênh tỉ lệ nhãn tới **−55,7 điểm %** ⇒ **không** xoá dòng thiếu
- **Đuôi nặng:** skew tới **−14,9**; **30,6%** giá trị `debt_to_equity` ngoài khoảng IQR ⇒ ưu tiên mô hình cây / impute median
- **Đa cộng tuyến:** **VIF > 10 ở 33/47 cột**; số chiều hiệu dụng chỉ **12,89** ⇒ hệ số mô hình tuyến tính **không** đọc như quan hệ nhân quả
- **Drift train→test:** 5 đặc trưng vượt ngưỡng (nặng nhất KS 0,621)
- Nhãn rất “dính” theo thời gian: **90,8%** cặp quý liền nhau giữ nguyên nhãn

Ảnh: `../reports/figures/eda/08_ratio_correlation.png` → *Tương quan giữa các tỷ số & cụm đa cộng tuyến*

*(Ba con số cần nhớ: thiếu dữ liệu mang thông tin; lệch −14,9 nên Pearson vô dụng; 33/47 cột VIF > 10.)*

---

## Slide 4. Phát hiện then chốt: nhãn gần như là thuộc tính của CÔNG TY

**Thông điệp chính:** Vì nhãn “dính” theo công ty, **điểm in-domain bị thổi phồng** — phải đo bằng công ty chưa từng thấy.

- Tỉ lệ dương 62,3% ở cấp **mẫu** (IR 1,66) — nghe “dễ chịu” nên rất dễ bỏ qua vấn đề
- Nhưng ở cấp **công ty**: **HD, LOW, WMT = 100% nhãn 1**; **ROST chỉ 4,7%** ⇒ IR tới **20,5**
- Baseline **chỉ dùng mã công ty** (`ticker_prior`, không dùng đặc trưng nào) đạt AUROC **0,986**
- Mô hình học máy: AUROC in-domain 0,983 (**không vượt** baseline)
- ⇒ Đây là **rò rỉ cấp thực thể** — loại lỗi thực nghiệm dễ bị bỏ qua nhất

Ảnh: `../reports/figures/eda/03_label_by_company_quarter.png` → *Tỉ lệ nhãn theo công ty và quý*

*(Con số 0,986 này là con số quan trọng nhất của đồ án: nó nói rằng phần lớn “năng lực” in-domain chỉ là nhận diện công ty. Nhóm giữ nguyên nó trong báo cáo.)*

---

## Slide 5. Tiền xử lý & 7 lớp chống rò rỉ

**Thông điệp chính:** Mọi phép biến đổi nằm **TRONG** `Pipeline` — mỗi fold chỉ học thống kê của chính nó.

- Pipeline chính: `impute(median) → (scaler cho mô hình tuyến tính/MLP) → model`
- **Winsorize KHÔNG** dùng ở pipeline chính: thí nghiệm cho thấy với mô hình được chốt chỉ **+0,13 điểm % AP** (trong khoảng nhiễu)
- **Không resample** trong pipeline chính (chỉ `class_weight`, tính trong `fit`); SMOTE/RUS đo riêng ở lab
- **Cố ý KHÔNG one-hot mã cổ phiếu** — làm vậy sẽ hợp thức hoá đúng loại rò rỉ đang đo
- 7 lớp kiểm soát: chia theo thời gian + purge 16 mẫu · impute/scale trong Pipeline · CV theo nhóm công ty · test không tham gia chọn mô hình/ngưỡng · manifest SHA-256 · test tự động bắt rò rỉ · provenance từng ô dữ liệu
- Split **byte-identical** khi chạy lại (có test chặn hồi quy)

Ảnh: `../reports/figures/preprocessing/01_winsorize_scaler.png` → *Winsorize × scaler (thí nghiệm mục 9.1)*

*(Nếu thầy/cô hỏi “sao không chuẩn hoá trước khi chia tập?” — đó là slide này: mọi thứ nằm trong Pipeline và chỉ fit trên train.)*

---

## Slide 6. Bốn họ mô hình + bốn baseline đối chứng

**Thông điệp chính:** Bốn **cơ chế học khác nhau** đặt cạnh **quy tắc tài chính cổ điển (Altman Z'' đạt 0,758 AUROC)**.

- **4 họ mô hình** (thuần scikit-learn, không xgboost/lightgbm):
  Logistic Regression (tuyến tính) · Random Forest (bagging) · HistGradientBoosting (boosting) · **MLP (mạng nơ-ron, phi tuyến không dựa trên cây)**
- **4 baseline:** dummy (lớp đa số) · **ticker-prior** (nhớ mặt công ty) · logistic 1 chỉ tiêu · **quy tắc Altman Z'' < 1,1** (công thức công khai 1968/2000, **không học tham số**)
- **Bảng so sánh trên cùng 64 mẫu test** (AUROC / AP):

| Hệ thống | AUROC | AP |
|---|---:|---:|
| Quy tắc **Altman Z'' < 1,1** (không học) | 0,758 | 0,865 |
| Logistic 1 chỉ tiêu (`debt_to_assets`) | 0,889 | 0,938 |
| Dummy (lớp đa số) | 0,500 | 0,594 |
| Baseline **ticker-prior** | **0,986** | 0,985 |
| Logistic Regression | 0,983 | 0,989 |
| **Random Forest** (được chốt) | 0,983 | 0,991 |
| HistGradientBoosting | 0,977 | 0,987 |
| MLP | 0,971 | 0,982 |

- Ngưỡng vận hành **0,788**; chọn mô hình theo **AP cross-company** (không theo F1 in-domain)

Ảnh: `../reports/figures/validation_pr_curves.png` → *Đường Precision–Recall trên validation*

*(Điểm mạnh của đồ án là dòng ticker-prior: mô hình không vượt nổi một quy tắc chỉ dùng danh tính công ty — nhóm để nguyên thay vì che.)*

---

## Slide 7. Tinh chỉnh siêu tham số & sổ thực nghiệm

**Thông điệp chính:** Tinh chỉnh rộng **không cải thiện** — và điều đó được **báo cáo**, không bị che.

- `GridSearchCV` (6–18 cấu hình/mô hình: 8 logistic · 18 RF · 8 HGB · 6 MLP) + **StratifiedGroupKFold(4)** + `refit=average_precision`
- Luôn in kèm điểm của **cấu hình mặc định** trên cùng splitter ⇒ trả lời “tinh chỉnh có cải thiện thật không”
- **Random search** 40 trial/mô hình + **sổ thực nghiệm `runs.csv` 153 dòng** (40 logistic · 33 RF · 40 HGB · 40 MLP; mỗi dòng có run_id, params, seed, CV-AP ± std, thời gian, trạng thái)
- Kết quả trung thực (cả hai chiều đều có số): GridSearch chỉ hơn cấu hình mặc định **+0,003 CV-AP (RF)** · +0,026 (logistic) · +0,015 (HGB) · +0,046 (MLP) ⇒ mô hình chốt **giữ nguyên cấu hình mặc định**
- Random search so với **lưới GridSearch** (cùng splitter, cùng thước đo): logistic **+0,007** · **RF −0,004** · HGB **+0,026** · MLP **+0,068** ⇒ lưới đã đủ tốt cho RF, thêm trial là lãng phí
- Tiêu chí dừng: không còn cải thiện AP cross-company có ý nghĩa; **không bao giờ dùng test** để chọn cấu hình

Ảnh: `../reports/figures/search/01_search_distribution.png` → *Phân bố CV-AP của các trial random search*

*(Kết quả âm cũng là kết quả: nhóm ghi rõ random search không giúp Random Forest, thay vì chỉ khoe lần tìm kiếm thành công.)*

---

## Slide 8. Kết quả trên test (n = 64) tại ngưỡng vận hành

**Thông điệp chính:** Tại ngưỡng **0,788**: Precision **0,973** · Recall **0,947** · F1 **0,960** — nhưng baseline ticker-prior vẫn ngang bằng.

- Mô hình: **Random Forest** — test **AUROC 0,983 · AP 0,991 · Brier 0,058**
- **Confusion matrix @0,788:** TN 25 · **FP 1** · **FN 2** · TP 36
- Ngưỡng 0,788 được chọn **trên validation** (best-F1 0,976); test chỉ dùng để báo cáo
- **Quy tắc Altman Z''** (không học gì): AUROC 0,758 — thấp hơn mô hình nhưng **cùng hướng** (Z'' càng thấp ⇒ rủi ro càng cao)
- **Kiểm định:** ΔAUROC (RF − ticker-prior) = −0,0030, **p = 0,7546** (DeLong) ⇒ *chưa* đủ căn cứ khẳng định mô hình hơn baseline; ΔAP = +0,0061, CI95 [−0,0048; +0,0215]
- Ngược lại, mô hình **vượt rõ** baseline 1 chỉ tiêu: ΔAUROC **+0,094**, **p = 0,0076**

Ảnh: `../reports/figures/test_confusion.png` → *Confusion matrix trên test tại ngưỡng vận hành 0,788*

*(Câu chốt: mô hình hơn “1 chỉ tiêu”, nhưng chưa hơn “nhận diện công ty”. Đó là kết luận trung thực nhất mà dữ liệu cho phép.)*

---

## Slide 9. Ngưỡng quyết định theo CHI PHÍ (không dùng 0,5)

**Thông điệp chính:** Ngưỡng là **tham số nghiệp vụ** — bỏ sót đắt gấp 5 lần báo động giả.

- Giả định chi phí: **FN = 5 · FP = 1** (có phân tích độ nhạy trong báo cáo mục 7.6)
- **Validation:** ngưỡng best-F1 = **0,788** (F1 0,976 · Precision 1,000 · Recall 0,952) và đây cũng là ngưỡng **tối ưu chi phí**
- **Test** (chỉ để phân tích): ngưỡng 0,783 → F1 0,974 · chi phí kỳ vọng **6,0**
- Đổi ngưỡng kéo Recall từ ~0,86 lên 0,95+ **mà không cần đổi mô hình** — can thiệp rẻ nhất về điểm vận hành
- Ngưỡng **cấu hình hoá** (không hardcode) + ghi log mọi quyết định: `sample_id`, `P(distress)`, ngưỡng, phiên bản mô hình
- Hiệu chuẩn: Brier validation 0,046 · test 0,058 ⇒ nếu muốn đọc xác suất như “xác suất thật” thì **phải calibrate** thêm

Ảnh: `../reports/figures/analysis/04_threshold_curves.png` → *Đường ngưỡng – F1 – chi phí kỳ vọng*

*(Không có ngưỡng “đúng” cho mọi tổ chức: nơi ưu tiên không bỏ sót thì hạ ngưỡng, nơi ưu tiên không làm phiền khách thì nâng ngưỡng — nên nó phải nằm trong file cấu hình.)*

---

## Slide 10. Tổng quát hoá sang CÔNG TY chưa từng thấy

**Thông điệp chính:** Bỏ công ty ra khỏi train thì AUROC rơi từ 0,983 → **0,933** (GroupKFold) và LOCO chỉ còn **0,628**.

- **Cross-company (GroupKFold, gộp xác suất out-of-fold):** RF **0,933 AUROC / 0,957 AP** · HGB 0,926/0,953 · logistic 0,912/0,942 · MLP 0,789/0,832
- **LOCO (bỏ trọn 1 công ty):** trung bình **0,628 AUROC** trên **8/8** công ty tính được
- ⇒ Hai mức “rơi” này chính là phần điểm in-domain đến từ việc **nhận ra công ty**
- Đây cũng là lý do **quy tắc chọn mô hình dùng AP cross-company**, không dùng F1 in-domain
- Vẫn hơn hẳn **quy tắc Altman** (0,758) và **1 chỉ tiêu** (0,889) trên cùng mẫu test

Ảnh: `../reports/figures/analysis/07_in_domain_vs_cross_company.png` → *In-domain vs cross-company vs baseline*

*(Nếu chỉ trình bày AUROC 0,983 thì đồ án chưa chứng minh gì. Con số 0,933 và 0,628 là phần “tự phản biện” quan trọng nhất.)*

---

## Slide 11. Kiểm chứng theo THỜI GIAN (walk-forward)

**Thông điệp chính:** Câu hỏi *“công ty cũ, giai đoạn mới”* còn khó hơn: AUROC trung bình **0,949 (HGB)** và có fold xuống **0,507**.

- Giao thức: **expanding window** cắt theo `target_period_end` + **purge theo ngày công bố nhãn (90 ngày)**; chỉ dùng train+validation+purged ⇒ **test không tham gia**
- 4 fold; fold nào train < 60 mẫu hoặc test đơn lớp thì **ghi rõ là bỏ qua** (không im lặng che)

| Mô hình | #fold | AUROC trung bình | AP trung bình | AUROC thấp nhất | AUROC từng fold |
|---|---:|---:|---:|---:|---|
| HistGradientBoosting | 3 | **0,949** | 0,977 | 0,920 | 0,920 · 0,948 · 0,979 |
| Random Forest | 3 | 0,923 | 0,952 | 0,811 | 0,811 · 0,988 · 0,970 |
| Logistic Regression | 3 | 0,753 | 0,833 | **0,507** | 0,778 · 0,507 · 0,975 |
| MLP | 3 | 0,714 | 0,783 | 0,500 | 0,698 · 0,500 · 0,944 |

- **Đọc bảng:** ba phép đo — in-domain · cross-company · walk-forward — trả lời ba câu hỏi khác nhau; không con số nào được trình bày đơn lẻ
- Bootstrap CI cũng được tính lại **theo cụm công ty** (mẫu lại ticker, không mẫu lại từng quý) vì 8 quý của cùng công ty không độc lập

Ảnh: `../reports/figures/analysis/08_walk_forward.png` → *AUROC/AP theo từng fold thời gian*

*(Đây là phần nhóm bổ sung mới: nếu chỉ kiểm chứng theo công ty thì chưa trả lời được câu “sang giai đoạn mới thì sao?”.)*

---

## Slide 12. Giải thích mô hình: vì sao dự đoán như vậy?

**Thông điệp chính:** Hai phương pháp độc lập (KernelSHAP tự cài + permutation importance) **đồng thuận**: thanh khoản & đòn bẩy dẫn đầu.

- **KernelSHAP tự cài** (Lundberg & Lee 2017) bằng numpy — môi trường không có gói `shap`; **tự kiểm chứng** bằng sai số efficiency **2,2e-16**
- Top đóng góp: `current_ratio_latest` (0,0765) · `current_ratio_min_window` (0,0764) · `debt_to_assets_latest` (0,0583) · `working_capital_to_assets` (0,0561)
- Đối chiếu permutation importance: Spearman **0,362**, trùng top-15 **0,43** ⇒ cùng nhóm đặc trưng dẫn đầu, không phụ thuộc một công cụ
- **Ablation:** bỏ nhóm `ratios_latest` làm test AUROC giảm mạnh nhất (0,983 → 0,966)
- **Cảnh báo đọc kết quả:** 33/47 cột VIF > 10 ⇒ SHAP “chia công” cho cả cụm tương quan; **không** suy ra quan hệ nhân quả

Ảnh: `../reports/figures/shap/01_shap_summary.png` → *SHAP toàn cục (mean |φ|)* · Ảnh phụ: `../reports/figures/shap/02_shap_beeswarm.png`

*(Nhóm không thuyết trình bằng hình minh hoạ suông: sai số efficiency 2e-16 là bằng chứng cài đặt đúng, và có test giải tích cho hàm tuyến tính.)*

---

## Slide 13. Phân tích lỗi: chỉ 3/64 mẫu sai và có cấu trúc

**Thông điệp chính:** Cả 3 mẫu sai đều **truy vết được** bằng SHAP — và trần hiệu năng bị giới hạn bởi **chất lượng nhãn**.

- Sai 3/64 mẫu = **2 FN + 1 FP** (đúng như confusion matrix @0,788)

| Mẫu | Thực tế | Dự đoán | P(distress) | Ghi chú |
|---|---:|---:|---:|---|
| HD-2024Q2 | 1 | 0 (FN) | 0,783 | sát ngưỡng 0,788 |
| FIVE-2024Q3 | 1 | 0 (FN) | **0,174** | mọi chỉ số thanh khoản đều kéo về “an toàn” |
| DG-2025Q2 | 0 | 1 (FP) | 0,798 | vừa vượt ngưỡng |

- SHAP cục bộ (FIVE-2024Q3): đẩy về “an toàn” gồm `current_ratio_min_window` (−0,12), `current_ratio_latest` (−0,11), `debt_to_assets_latest` (−0,10) — trong khi nhãn thực tế = 1
- **Overfit nhẹ:** train AUROC 1,000 vs validation 0,965 (gap 0,034)
- **Learning curve chưa bão hoà:** val AUROC 0,731 → 0,818 → 0,855 → 0,895 khi train 83 → 162 mẫu; gap 0,105 ⇒ **thiếu thực thể**, không phải thiếu thuật toán

Ảnh: `../reports/figures/analysis/01_overfit_train_vs_val.png` → *Overfitting: train vs validation* · Ảnh phụ: `../reports/figures/analysis/06_learning_curve.png` · `../reports/figures/shap/03_shap_local_errors.png`

*(Điểm quan trọng: quy tắc kế toán đơn giản chỉ khớp 74,7% nhãn gốc — nghĩa là có mẫu mà “đáp án” không khớp bất kỳ định nghĩa công khai nào. Nhóm kết luận trần hiệu năng bị giới hạn bởi nhãn.)*

---

## Slide 14. Truy vết nhãn: 4 định nghĩa, kết luận ỔN ĐỊNH

**Thông điệp chính:** Nhãn gốc **không tái tạo được**, nhưng kết luận quan trọng nhất giữ nguyên ở **3/3 định nghĩa công khai**.

- Nhãn `is_distressed` giữ nguyên từ pipeline gốc; quy tắc kế toán đơn giản khớp tối đa **74,7%** (trên 308 mẫu)
- Phản chứng cụ thể: **WMT-2015Q2** có lãi và dòng tiền dương nhưng bị gán nhãn 1
- Báo cáo định nghĩa thêm **3 nhãn công khai** và chạy lại **cùng một quy trình** trên từng nhãn:

| Định nghĩa nhãn | Dương (test) | Khớp nhãn gốc | AUROC test (mô hình / ticker-prior) | Cross-company AUROC |
|---|---:|---:|---:|---:|
| `original` (nhãn gốc) | 59,4% | 100% | 0,983 / 0,986 | 0,943 |
| `stress_signals` (≥1 trong 6 tín hiệu căng thẳng) | 42,2% | 74,4% | 0,909 / **0,960** | 0,730 |
| `altman_z` (Z'' < 1,1) | 31,2% | 63,9% | 0,999 / 0,991 | 0,772 |
| `forward_4q` (distress trong 4 quý tới) | 48,4% | 66,7% | 0,836 / **0,911** | 0,410 |

- **Luận điểm “mô hình không vượt baseline nhớ mặt công ty” xuất hiện ở 3/3 định nghĩa tái lập được** ⇒ kết luận **không phụ thuộc cách gán nhãn** — đây là bằng chứng mạnh nhất của đồ án
- Cũng phải nói rõ: dưới nhãn thay thế, AP cross-company tụt còn 0,46–0,61 ⇒ khả năng tổng quát hoá **thấp hơn** vẻ ngoài in-domain

Ảnh: `../reports/figures/eda/09_ratio_vs_label.png` → *Liên hệ tỷ số ↔ nhãn (EDA)*

*(Nếu bị hỏi “sao dám chắc kết quả không phải do cách gán nhãn?” — trả lời bằng cột “khớp nhãn gốc” và việc lặp lại kết luận trên 3 định nghĩa.)*

---

## Slide 15. Nhãn SỰ KIỆN từ SEC: dữ liệu có phá sản thật không?

**Thông điệp chính:** Tra **toàn bộ filing 8-K của 8/8 công ty**: **0 sự kiện phá sản (item 1.03)** ⇒ đề tài được định vị là **suy giảm tài chính (financial distress)**.

- Nguồn: `https://data.sec.gov/submissions/CIK##########.json` — đọc trường `items` của 8-K (script `scripts/fetch_events.py`, chỉ đọc, không suy diễn)
- Kết quả: **11 filing tín hiệu kiệt quệ** / **0 filing phá sản**
  - LOW **7** (2005–2018: 4.02 non-reliance, 2.06 impairment, 2.05 restructuring) · DG **2** · DKS **2** · FIVE · HD · ORLY · ROST · WMT = 0
- Ý nghĩa: nhãn gốc là **trạng thái kế toán quý**, **không** phải sự kiện phá sản ⇒ muốn làm “dự báo phá sản” phải **mở rộng universe** sang doanh nghiệp đã nộp 8-K item 1.03
- Đây là điểm yếu gốc của đề tài, và nhóm **chủ động đo nó** thay vì để người chấm phát hiện
- Song song: **ETL đã được port** để chứng minh dữ liệu tái tạo được (slide sau)

Ảnh: (không bắt buộc) — có thể chụp màn hình `reports/results/events.md` nếu muốn minh hoạ

*(Câu chốt với hội đồng: “Bọn em không làm dự báo phá sản; bọn em làm dự báo suy giảm tài chính, và đây là bằng chứng 8 công ty này chưa từng phá sản.”)*

---

## Slide 16. Tái lập & kiểm chứng: 1 lệnh, 234 test, 0 phát hiện

**Thông điệp chính:** Người chấm chạy lại được **toàn bộ** kết quả bằng **một lệnh**, và mọi con số trong báo cáo được **đối soát tự động**.

- `python -m scripts.run_all` → **22 bước** (data → **ETL port** → provenance → EDA → train → baselines → validation → tuning → evaluate → … → báo cáo Word + slide)
- **234 test tự động** (chống rò rỉ, registry 4 mô hình, walk-forward, cluster bootstrap, ETL port, split tái lập byte-identical)
- **98.200 phép kiểm tra đối soát / 0 phát hiện** (`python -m scripts.audit_data`) giữa báo cáo ↔ artifact
- **Kiểm chứng dữ liệu thật:** 20 file SEC băm SHA-256 · **4.609 ô** tra ngược companyfacts · 0 lệch / 0 thiếu / 0 ô “điền cho đủ”
- **ETL đã port** (`scripts/prepare_sec.py`): tái tạo **5.308/5.312 ô = 99,92%** từ snapshot SEC, và tự tìm được **496 quý** ⇒ chạy được cho công ty mới
- Seed 42 · artifact **không** chứa timestamp (diff được giữa 2 lần chạy) · CI GitHub Actions chạy test + audit mỗi lần push
- Demo trực tiếp trong buổi bảo vệ: `python -m scripts.predict --sample-id HD-2024Q2 --explain`

Ảnh: `../reports/figures/analysis/05_calibration.png` → *Hiệu chuẩn xác suất (Brier 0,046 val / 0,058 test)*

*(Nếu thầy/cô muốn kiểm tra tại chỗ: chạy `python -m scripts.audit_data` — 98.200 phép kiểm tra in ra 0 phát hiện.)*

---

## Slide 17. Hạn chế (nhóm tự liệt kê trước khi bị hỏi)

**Thông điệp chính:** Bảy giới hạn đã biết — mỗi giới hạn đều có **số liệu đo kèm**, không phải câu nói suông.

1. **Nhãn gốc không tái tạo được** — quy tắc công khai khớp tối đa 74,7% ⇒ đã bù bằng 3 nhãn quy tắc + nhãn sự kiện
2. **Rò rỉ cấp thực thể** — in-domain 0,983 ≈ ticker-prior 0,986; cross-company 0,933; walk-forward 0,949 (HGB)
3. **Chỉ 8 thực thể** — LOCO 0,628; learning curve chưa bão hoà (gap 0,105)
4. **Mẫu không độc lập** — cửa sổ 8 quý trượt; **90,8%** cặp quý liền nhau giữ nguyên nhãn
5. **Không có tín hiệu ngoài báo cáo tài chính** (giá cổ phiếu, xếp hạng tín dụng, vĩ mô)
6. **Đơn vị tiền là minh hoạ** (×25.000 VND/USD) — không phải BCTC Việt Nam, không dùng tỷ giá lịch sử
7. **Không có sự kiện phá sản thật** trong 8 công ty (0/8 có 8-K item 1.03) ⇒ không thể gọi là “dự báo phá sản”

Ảnh: `../reports/figures/eda_deep/05_drift_ks_vs_smd.png` → *Drift train → test (KS/SMD)*

*(Cách trả lời tốt nhất cho câu “hạn chế lớn nhất là gì?” là đưa luôn số đo: LOCO 0,628 và nhãn khớp 74,7%.)*

---

## Slide 18. Kết luận & hướng phát triển

**Thông điệp chính:** Đóng góp của đồ án **không** phải “AUROC 0,983” mà là **phát hiện & định lượng rò rỉ cấp thực thể** + bộ khung kiểm chứng ba góc tái lập được.

- **Kết luận 1:** 4 họ mô hình chênh nhau **< 0,01 AUROC** ở chế độ in-domain ⇒ **thuật toán không phải nút thắt**; nút thắt là **nhãn** và **giao thức đánh giá**
- **Kết luận 2:** baseline không dùng đặc trưng nào (ticker-prior) đạt 0,986 ⇒ phần lớn “năng lực” in-domain là **nhận diện công ty**
- **Kết luận 3:** nhưng mô hình **có** học tín hiệu thật — vượt dummy rất xa và vượt baseline 1 chỉ tiêu với **p = 0,0076**
- **Ứng dụng đúng:** **xếp hạng ưu tiên thẩm định** (kèm lý do SHAP cho từng hồ sơ) — **không** tự động từ chối/hạ hạng tín dụng
- **P0:** chốt định nghĩa nhãn với giảng viên (hạ tầng đã xong: 3 nhãn quy tắc + nhãn sự kiện)
- **P1:** mở rộng lên 50–100 công ty (đường sinh quý đã có trong `prepare_sec`); lọc đặc trưng theo cụm/VIF; mô hình panel/survival
- **P2:** monitoring PSI/KS + AP cross-company định kỳ; ngưỡng cấu hình hoá; log mọi quyết định

Ảnh: `../reports/figures/test_roc_pr_curves.png` → *ROC & PR curve trên test*

*(Xin cảm ơn và sẵn sàng nhận câu hỏi — nhóm có thể chạy lại bất kỳ bước nào ngay trong buổi bảo vệ.)*

---

<!-- _class: lead -->

# Cảm ơn thầy/cô đã lắng nghe

## Câu hỏi & thảo luận

- Báo cáo đầy đủ: `reports/BAO-CAO.md` · tài liệu bảo vệ: `docs/bo-tai-lieu-bao-ve.md`
- Tái lập: `python -m scripts.run_all` (22 bước) · `python -m scripts.audit_data` (98.200 phép kiểm tra)
- Kiểm chứng dữ liệu thật: `python -m scripts.verify_provenance`
- Demo: `python -m scripts.predict --sample-id HD-2024Q2 --explain`

*(Ba câu dễ bị hỏi nhất và câu trả lời ngắn nằm ở docs/bo-tai-lieu-bao-ve.md — phần “10 câu hỏi phản biện”.)*

---

<!--
=========================================================================
  PHẦN PHỤ LỤC — KHÔNG đưa lên slide trình bày.
  Mục đích: (1) để AI/công cụ sinh slide KHÔNG bịa số; (2) để bạn tra cứu nhanh
  khi bị hỏi; (3) danh mục ảnh cần tải lên; (4) prompt gợi ý.
=========================================================================
-->

## PHỤ LỤC A — Bảng số liệu gốc (chỉ để tra cứu, KHÔNG lên slide)

### A1. Số liệu nền về dữ liệu

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Công ty | 8 (WMT, HD, LOW, ROST, DG, ORLY, DKS, FIVE) | `data/prepared/corpus.json` |
| Số quý trong snapshot | 332 quý | `data/prepared` |
| Số mẫu dự báo | **324** | `data/prepared` |
| Chia tập (theo thời gian) | train 212 · validation 32 · test 64 · **purge 16** | `data/prepared/split.json` |
| Số đặc trưng | **47** | `feature_names.json` |
| Tỉ lệ dương | train 62,3% · val 65,6% · test 59,4% | `eda.json` |
| Imbalance ratio (train) | 1,65 | `eda.json` |
| Ô `null` giữ nguyên | 703 | `verify_provenance.json` |
| Ô tra ngược companyfacts | **4.609** (0 lệch · 0 thiếu) | `verify_provenance.json` |
| File SEC băm SHA-256 | 20 (0 lệch) | `verify_provenance.json` |
| Ô ETL tái tạo lại được | **5.308/5.312 = 99,92%** (DKS lệch 4 ô) | `etl_verify.json` |
| Quý ETL tự tìm được | 496 | `etl_verify.json` |

### A2. Kết quả trên test (n = 64) — mô hình chốt Random Forest

| Chỉ số | Giá trị |
|---|---:|
| AUROC | 0,983 |
| Average Precision (AP) | 0,991 |
| Brier | 0,058 |
| Ngưỡng vận hành | 0,7879 (≈ 0,788) |
| Precision @ngưỡng | 0,973 |
| Recall @ngưỡng | 0,947 |
| F1 @ngưỡng | 0,960 |
| Macro-F1 @ngưỡng | 0,952 |
| Confusion @ngưỡng | TN 25 · FP 1 · FN 2 · TP 36 |
| Ngưỡng best-F1 (validation) | 0,788 (F1 0,976 · P 1,000 · R 0,952) |
| Chi phí kỳ vọng @0,783 (test) | 6,0 (giả định FN = 5 · FP = 1) |

### A3. Kiểm định ý nghĩa (cùng 64 mẫu test)

| So sánh | ΔAUROC | p (DeLong) | ΔAP | CI95 ΔAP | p (bootstrap) |
|---|---:|---:|---:|---:|---:|
| RF − ticker_prior | −0,0030 | 0,7546 | +0,0061 | [−0,0048; +0,0215] | 0,3440 |
| RF − single_feature | +0,0941 | 0,0076 | +0,0528 | [+0,0176; +0,0973] | 0,0010 |
| RF − dummy | +0,4828 | 0,0000 | +0,3970 | [+0,3794; +0,4062] | 0,0000 |
| ticker_prior − single_feature | +0,0972 | 0,0106 | +0,0467 | [+0,0074; +0,0941] | 0,0180 |

### A4. Ba giao thức kiểm chứng (không trộn số giữa các giao thức)

| Hệ thống | In-domain (train→test) | Cross-company (GroupKFold OOF) | Walk-forward (trung bình fold) |
|---|---:|---:|---:|
| HistGradientBoosting | 0,977 / AP 0,987 | 0,926 / 0,953 | **0,949** / 0,977 |
| Random Forest | 0,983 / 0,991 | 0,933 / 0,957 | 0,923 / 0,952 |
| Logistic Regression | 0,983 / 0,989 | 0,912 / 0,942 | 0,753 / 0,833 |
| MLP | 0,971 / 0,982 | 0,789 / 0,832 | 0,714 / 0,783 |
| Baseline ticker-prior | 0,986 / 0,985 | — | — |
| Baseline 1 chỉ tiêu | 0,889 / 0,938 | — | — |
| Quy tắc Altman Z'' | 0,758 / 0,865 | — | — |
| Dummy | 0,500 / 0,594 | — | — |

- **LOCO:** trung bình **0,628 AUROC** trên **8/8** công ty
- **Walk-forward:** 4 fold (1 fold bỏ qua vì train < 60 mẫu); AUROC từng fold: HGB 0,920 · 0,948 · 0,979; RF 0,811 · 0,988 · 0,970; logistic 0,778 · 0,507 · 0,975; MLP 0,698 · 0,500 · 0,944

### A5. Nhãn & độ nhạy theo định nghĩa nhãn (RQ4)

| Định nghĩa | Dương (train) | IR train | Dương (test) | Khớp nhãn gốc | AUROC test (mô hình / ticker-prior) | Cross-company AP | Cross-company AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original` | 62,3% | 1,65 | 59,4% | 100,0% | 0,983 / 0,986 | 0,958 | 0,943 |
| `stress_signals` | 41,5% | 1,41 | 42,2% | 74,4% | 0,909 / 0,960 | 0,610 | 0,730 |
| `altman_z` | 22,6% | 3,42 | 31,2% | 63,9% | 0,999 / 0,991 | 0,460 | 0,772 |
| `forward_4q` | 66,5% | 1,99 | 48,4% | 66,7% | 0,836 / 0,911 | 0,582 | 0,410 |

- Audit nhãn gốc (trên 308 mẫu, quy tắc áp trên quý target): khớp cao nhất **74,7%** (`stress_signals ≥ 1`); các quy tắc khác: `current_liabilities > current_assets` 65,3% · `retained_earnings < 0` 54,2% · `stockholders_equity < 0` 51,9% · `ocf < 0 hoặc ni < 0` 40,9% · `operating_cash_flow < 0` 40,3% · `net_income < 0` 39,6%
- ⇒ **Nhãn gốc KHÔNG tái tạo được** từ dữ liệu công bố.

### A6. SHAP, VIF, ablation, overfit

| Hạng mục | Giá trị |
|---|---|
| Phương pháp giải thích | KernelSHAP tự cài (numpy) — 64 mẫu × 200 liên minh, nền 40 mẫu train |
| Sai số efficiency (tự kiểm chứng) | 2,220e-16 (tương đối 6,337e-15) |
| Đồng thuận với permutation importance | Spearman 0,362 · trùng top-15 = 0,43 |
| Top-4 SHAP (mean \|φ\|) | `current_ratio_latest` 0,0765 · `current_ratio_min_window` 0,0764 · `debt_to_assets_latest` 0,0583 · `working_capital_to_assets` 0,0561 |
| Top permutation (val ΔAUROC) | `gross_margin_latest` 0,008 · `receivables_to_sales_latest` 0,001 · `working_capital_to_assets` 0,001 |
| VIF > 10 | **33/47** cột (cao nhất `cash_to_assets_yoy` = 798,1) |
| Số chiều hiệu dụng | 12,89 |
| Ablation mạnh nhất | bỏ `ratios_latest` ⇒ test AUROC 0,983 → 0,966 · AP 0,991 → 0,982 |
| Overfit (train − val AUROC) | RF 1,000 − 0,965 = **0,034** · HGB 0,026 · logistic 0,019 · MLP −0,013 |
| Learning curve | 83 mẫu → val 0,731 · 109 → 0,818 · 135 → 0,855 · 162 → 0,895; gap cuối **0,105** |

### A7. Tinh chỉnh siêu tham số

| Mô hình | Cấu hình tốt nhất (CV) | #cấu hình | CV-AP | CV-AP mặc định | Chênh |
|---|---|---:|---:|---:|---:|
| Logistic Regression | `{'C': 0.01, 'class_weight': None}` | 8 | 0,960 | 0,935 | +0,026 |
| Random Forest | `{'max_depth': 3, 'min_samples_leaf': 2, 'n_estimators': 500}` | 18 | 0,989 | 0,986 | +0,003 |
| HistGradientBoosting | `{'learning_rate': 0.03, 'max_depth': 2, 'max_iter': 200}` | 8 | 0,939 | 0,924 | +0,015 |
| MLP | `{'alpha': 0.0001, 'hidden_layer_sizes': 64}` | 6 | 0,845 | 0,799 | +0,046 |

- **Cấu hình THẬT của mô hình chốt** (Random Forest) = **mặc định**: `{'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample', 'n_jobs': 1}`
- Random search (40 trial/mô hình, `runs.csv` 153 dòng) so với **lưới** GridSearch: logistic +0,007 · RF **−0,004** · HGB +0,026 · MLP +0,068 (CV-AP tốt nhất: RF 0,9855 · logistic 0,9669 · HGB 0,9648 · MLP 0,9129)
- Kỹ thuật xử lý lệch lớp trên dữ liệu thật (6 kỹ thuật, GroupKFold): tốt nhất `smote_enn` AP 0,9683 vs đối chứng 0,9588 — **không có ý nghĩa (p = 0,21)** ⇒ **không** đưa vào pipeline chính

### A8. Nhãn sự kiện SEC (8-K) — 8/8 công ty

| Công ty | Filing phá sản (item 1.03) | Filing tín hiệu kiệt quệ | Chi tiết |
|---|---:|---:|---|
| LOW | 0 | **7** | 4.02 non-reliance (2005–2015) · 2.06 impairment (2011, 2018) · 2.05 restructuring |
| DG | 0 | 2 | 2.06 impairment (2005, 2006) |
| DKS | 0 | 2 | 2.06 (2007) · 2.05 (2009) |
| FIVE · HD · ORLY · ROST · WMT | 0 | 0 | — |
| **Tổng** | **0** | **11** | ⇒ đề tài định vị là **financial distress**, không phải dự báo phá sản |

### A9. Tái lập & kiểm chứng

| Hạng mục | Giá trị |
|---|---|
| Một lệnh chạy toàn bộ | `python -m scripts.run_all` — **22/22 bước ok** (`reports/results/run_all.log`) |
| Test tự động | **234 PASS** (`python -m unittest discover -s tests`) |
| Đối soát báo cáo ↔ artifact | **98.200 phép kiểm tra · 0 phát hiện** (`scripts/audit_data.py`) |
| Seed | 42 (mọi mô hình/search/CV) |
| Tái lập split | **byte-identical** (có test chặn hồi quy) |
| Artifact có timestamp? | **Không** ⇒ `git diff` sau khi chạy lại phải rỗng |
| CI | `.github/workflows/tests.yml` (test + audit mỗi lần push) |
| Môi trường | Python 3 · scikit-learn · numpy · pandas · matplotlib · python-docx · python-pptx (không có `shap`, `optuna`, `xgboost`, `lightgbm`) |

### A10. “Sáu con số đinh” (slide dự phòng nếu bị yêu cầu tóm tắt)

1. **324** mẫu · **47** đặc trưng · **8** công ty · chia **212/32/64** (+16 purge)
2. In-domain AUROC **0,983** ≈ baseline ticker-prior **0,986** ⇒ p = **0,7546**
3. Cross-company **0,933** · LOCO **0,628** · walk-forward tốt nhất **0,949**
4. Nhãn gốc khớp quy tắc công khai tối đa **74,7%** ⇒ nhãn không tái tạo được
5. **0** filing phá sản / **11** filing tín hiệu kiệt quệ trong 8 công ty
6. **1** lệnh tái lập · **234** test · **98.200** phép kiểm tra · **0** phát hiện

---

## PHỤ LỤC B — Danh mục ảnh cần tải lên (theo thứ tự slide)

> Đường dẫn trong bảng tính từ thư mục `docs/` (khớp với dòng “Ảnh:” trong từng slide).

| Slide | File | Nội dung cần thể hiện |
|---|---|---|
| 2 | `../reports/figures/eda/02_class_balance.png` | Tỉ lệ lớp theo tập train/val/test |
| 3 | `../reports/figures/eda/08_ratio_correlation.png` | Ma trận tương quan + cụm đa cộng tuyến |
| 4 | `../reports/figures/eda/03_label_by_company_quarter.png` | **Tỉ lệ nhãn theo công ty × quý** (HD/LOW/WMT = 100%) |
| 5 | `../reports/figures/preprocessing/01_winsorize_scaler.png` | So sánh winsorize × scaler |
| 6 | `../reports/figures/validation_pr_curves.png` | Đường PR validation của 4 mô hình + baseline |
| 7 | `../reports/figures/search/01_search_distribution.png` | Phân bố CV-AP của random search |
| 8 | `../reports/figures/test_confusion.png` | **Confusion matrix test @0,788** |
| 9 | `../reports/figures/analysis/04_threshold_curves.png` | F1 & chi phí kỳ vọng theo ngưỡng |
| 10 | `../reports/figures/analysis/07_in_domain_vs_cross_company.png` | In-domain (0,983) vs cross-company (0,933) vs LOCO |
| 11 | `../reports/figures/analysis/08_walk_forward.png` | **AUROC/AP theo từng fold thời gian** |
| 12 | `../reports/figures/shap/01_shap_summary.png` + `shap/02_shap_beeswarm.png` | SHAP toàn cục & beeswarm |
| 13 | `../reports/figures/shap/03_shap_local_errors.png` | SHAP cục bộ 3 mẫu sai |
| 13 | `../reports/figures/analysis/01_overfit_train_vs_val.png` + `analysis/06_learning_curve.png` | Overfit & learning curve |
| 14 | `../reports/figures/eda/09_ratio_vs_label.png` | Liên hệ tỷ số ↔ nhãn |
| 16 | `../reports/figures/analysis/05_calibration.png` | Hiệu chuẩn (Brier) |
| 17 | `../reports/figures/eda_deep/05_drift_ks_vs_smd.png` | Drift train→test (KS/SMD) |
| 18 | `../reports/figures/test_roc_pr_curves.png` | ROC & PR curve trên test |
| (dự phòng) | `../reports/figures/eda_deep/02_label_by_ticker.png` | Tỉ lệ nhãn theo công ty (IR tới 20,5) |
| (dự phòng) | `../reports/figures/analysis/02_feature_importance.png` | Permutation importance |
| (dự phòng) | `../reports/figures/imbalance_real/01_techniques_real.png` | 6 kỹ thuật chống lệch lớp trên dữ liệu thật |
| (dự phòng) | `../reports/figures/eda/01_coverage_by_company.png` | Độ phủ chỉ tiêu theo công ty |

Nếu công cụ AI không nhận đường dẫn tương đối: nhớ **kéo ảnh trực tiếp** vào từng slide theo đúng thứ tự trên (số trong tên file = thứ tự trong thư mục).

---

## PHỤ LỤC C — Prompt gợi ý cho công cụ AI + checklist

### C1. Prompt dán kèm file này (Gamma / Canva / Tome / ChatGPT / Copilot)

> Đây là nội dung một đồ án môn học về **dự báo suy giảm tài chính doanh nghiệp bán lẻ**.
> Hãy tạo **20 slide** (1 trang bìa + Slide 1–18 + 1 trang cảm ơn) theo đúng nội dung tôi cung cấp:
> 1. Giữ **nguyên mọi con số** (không làm tròn khác, không thêm số mới, không bịa kết quả).
> 2. Mỗi slide có: **tiêu đề ngắn**, 1 dòng **thông điệp chính** (chữ lớn nhất), 4–7 gạch đầu dòng ngắn (mỗi dòng ≤ 12 từ), và chỗ đặt ảnh theo dòng “Ảnh:”.
> 3. **Không đưa** các dòng in nghiêng trong ngoặc và **không đưa** Phụ lục A/B/C lên slide (đó là ghi chú nói và bảng tra cứu).
> 4. Phong cách: học thuật, nghiêm túc, tối giản, nền sáng, 1 màu nhấn (xanh navy), font không chân (Inter/Roboto), bảng dùng cho slide 6, 11, 14.
> 5. Giữ **tiếng Việt**, dùng dấu phẩy thập phân (0,983) như bản gốc.
> 6. Slide 8, 10, 11 phải thể hiện rõ sự **trung thực khoa học**: baseline ticker-prior 0,986 ≈ mô hình, cross-company 0,933, LOCO 0,628 — không làm mờ các số này.

### C2. Checklist trước khi trình bày

- [ ] Slide 4 có đúng 4 con số: **62,3% / 20,5 / 0,986 / 0,983**
- [ ] Slide 6 có **đủ 4 mô hình + 4 baseline**, trong đó có **quy tắc Altman Z''**
- [ ] Slide 8 ghi rõ test **n = 64** và ngưỡng **0,788** (không trình bày @0,5)
- [ ] Slide 10 có **cả ba cột**: in-domain 0,983 · cross-company 0,933 · LOCO 0,628
- [ ] Slide 11 có bảng walk-forward và **nêu fold bị bỏ qua**
- [ ] Slide 14 nêu kết luận ổn định **3/3 định nghĩa nhãn**
- [ ] Slide 15 nêu **0 filing phá sản / 11 filing kiệt quệ** ⇒ định vị đúng đề tài
- [ ] Slide 16 ghi **234 test · 98.200 phép kiểm tra · 0 phát hiện · 22 bước · 1 lệnh**
- [ ] Slide 17 liệt kê **đủ 7 hạn chế**, có số đo kèm
- [ ] Toàn bộ slide **không** xuất hiện con số nào không có trong Phụ lục A

### C3. Nếu muốn tự dựng slide từ file này (không qua AI)

```powershell
# 1) Cách nhanh nhất: Marp CLI (ra PDF/PPTX)
npx @marp-team/marp-cli docs/slide-cho-ai.md --pptx --allow-local-files -o docs/slide-cho-ai.pptx
npx @marp-team/marp-cli docs/slide-cho-ai.md --pdf  --allow-local-files -o docs/slide-cho-ai.pdf

# 2) Hoặc mở file này bằng VS Code + extension "Marp for VS Code" → Export slide deck
# 3) Hoặc mở trong Obsidian (chế độ Slide) / Slidev
```

*(Nếu muốn xoá phần phụ lục khi đưa cho AI: cắt từ dòng `## PHỤ LỤC A` trở xuống; phần này chỉ để tra cứu và kiểm tra chéo.)*






