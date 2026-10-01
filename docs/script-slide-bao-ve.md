# KỊCH BẢN THUYẾT TRÌNH — khớp 1:1 với `docs/Predicting_Retail_Distress_SEC_XBRL.pdf` (12 slide)

*Tài liệu VIẾT TAY, để cầm tay đọc khi bảo vệ. Mỗi slide có: **(a)** thời lượng mục tiêu, **(b)** lời thoại
viết như nói, **(c)** số liệu phải nhấn, **(d)** việc cần sửa trên Canva. Mọi con số đều truy được về
artifact trong repo (xem Phụ lục B của `docs/BAO-CAO-phan-bien.md`).*

**Phân bổ thời gian:** Slide 1–10 ≈ **10 phút** (trọn barem Phần I) · Slide 11–12 + 2 slide bổ sung ≈
**4–5 phút** (Phần II) · phần còn lại là Q&A. Nếu bị nhắc "còn 3 phút", **bỏ Slide 5 và Slide 6**
(xem *Kế hoạch nén* ở cuối).

---

## SLIDE 1 — BÌA (0:20)

> **Lời thoại:** "Kính chào Thầy và Hội đồng. Nhóm 18 trình bày đề tài *Dự báo suy giảm tài chính doanh
> nghiệp bán lẻ Mỹ từ dữ liệu SEC XBRL*. Nhóm gồm em Nguyễn Đức Ý, MSSV 25730094, dưới sự hướng dẫn của
> Thầy Cáp Đình La Thăng. Em xin trình bày 10 phút phần cơ bản theo đề cương, rồi 5 phút giải trình chuyên
> sâu 10 câu hỏi của Thầy."

- **Nhấn:** đúng tên đề tài, và định vị đúng bài toán là **"suy giảm tài chính"** — *không phải* "phá sản":
  trong dữ liệu có **0 sự kiện phá sản** (8-K item 1.03), chỉ có 11 hồ sơ tín hiệu kiệt quệ
  (`reports/results/events.md`).
- **Chuyển:** "Em bắt đầu từ bài toán và đặc tả dữ liệu vào–ra."

## SLIDE 2 — INPUT & OUTPUT SPECIFICATIONS (0:45)

> **Lời thoại:** "Đầu vào là 16 chỉ tiêu tài chính thô từ 10-K/10-Q của 8 chuỗi bán lẻ. Mỗi chỉ tiêu đều
> gắn mốc công bố, và nhóm **chỉ dùng dữ liệu có `available_on ≤ as_of`** — tức mô hình chỉ thấy thông tin
> đã công bố tại thời điểm dự báo. Từ đó nhóm dựng **47 đặc trưng** trong cửa sổ trượt 8 quý. Nhóm **cố ý
> loại bỏ** giá cổ phiếu, chỉ số vĩ mô, xếp hạng tín nhiệm, và **cấm one-hot mã cổ phiếu** — vì đó chính là
> đường tắt để 'nhớ mặt công ty'. Đầu ra là nhãn nhị phân cho quý kế tiếp, xác suất dùng để ra quyết định với
> **ngưỡng vận hành 0,7879**, và giải thích cục bộ bằng **KernelSHAP**."

- **Nhấn:** `available_on ≤ as_of` · cấm one-hot ticker · **KernelSHAP** (đúng: nhóm tự cài, không phải
  TreeSHAP) · ngưỡng 0,7879 ứng với giả định **C_FN = 5 × C_FP**.
- **Canva:** ✅ bản 12 trang đã sửa đúng ("KernelSHAP", "operational threshold").

## SLIDE 3 — CORPUS ARCHITECTURE & PURGED TIME-SPLIT (1:00)

> **Lời thoại:** "Corpus gồm **332 quý** báo cáo của **8 công ty**, tạo ra **324 mẫu** dự báo; mỗi mẫu là
> một cửa sổ 8 quý. Về kiểm chứng nguồn: nhóm băm SHA-256 **20 file SEC gốc** — **0 lệch** so với registry;
> và tra ngược từng ô về filing gốc được **4.609 ô**, **0 ô bịa số**. Về chia tập: vì dữ liệu là chuỗi thời
> gian — **90,8% cặp quý liền kề giữ nguyên nhãn** — nhóm **không** chia ngẫu nhiên mà chia theo thời gian
> kèm **dải đệm purge 90 ngày**: train **212**, purge **16**, validation **32**, test **64**. Test bị
> **khoá**: chọn mô hình và chọn ngưỡng đều làm trên validation/cross-company, test chỉ chấm **một lần**."

- **Nhấn:** 332 / 324 / 8 công ty · 4.609 ô · 0 lệch SHA-256 · **212 / 16 / 32 / 64** ·
  tỉ lệ dương 62,3% / 65,6% / **59,38%**.
- **Câu chốt:** "Vì vậy nhóm gọi tập test là *locked test*."
- **Canva:** nhãn `IFiling latency buffer` nhiều khả năng là **icon bị trích thành chữ "I"** — kiểm tra
  bằng mắt; nếu hiện ra chữ thì sửa thành **"Filing latency buffer"**.

## SLIDE 4 — DUAL-LEVEL IMBALANCE PARADOX (1:10)

> **Lời thoại:** "Thầy hỏi dữ liệu có mất cân bằng không — câu trả lời là **có, nhưng ở hai cấp khác nhau**.
> Ở **cấp mẫu** lệch chỉ nhẹ: tỉ lệ dương 62,35%, imbalance ratio **1,66**, entropy Shannon **0,9556 bit** —
> nhìn qua tưởng gần cân bằng. Nhưng ở **cấp thực thể** thì cực nặng: **41% số mẫu thuộc các công ty chỉ có
> một lớp nhãn** — HD, LOW, WMT có **100%** quý mang nhãn suy giảm, còn ROST chỉ **4,7%**. Nghĩa là entropy
> cao ở cấp mẫu **đánh lừa** mô hình ngây thơ: nó chỉ cần nhận ra công ty là xong. Đây là tiền đề cho phát
> hiện quan trọng nhất ở slide 11."

- **Nhấn:** IR **1,66** · entropy **0,9556 bit** · **41,0%** mẫu một-lớp · HD/LOW/WMT **100%** · ROST **4,7%**.
- **Canva (⚠️ BẮT BUỘC SỬA):** xoá **text quảng cáo template** còn sót — *"Reduces human error"*,
  *"Improves decision-making"*, *"AI systems process complex data with high precision…"*,
  *"Automates repetitive tasks"*, *"Routine processes such as document verification, data entry, and
  compliance checks…"*. Đây là **text mẫu của Canva, không liên quan đồ án**; kể cả khi đang bị khối màu
  che thì vẫn nên xoá cho sạch file.

## SLIDE 5 — FEATURE SIGNALS & MULTICOLLINEARITY (0:50)

> **Lời thoại:** "Đặc trưng mạnh nhất toàn hệ thống là **`current_ratio_min_window`** — mức thanh khoản
> ngắn hạn **xấu nhất** trong 8 quý: AUC đơn biến **0,048**, tức hướng nghịch — thanh khoản càng thấp thì
> rủi ro càng cao; hiệu ứng hạng |2·AUC−1| = **0,904**. Điều này hợp lý về tài chính: suy giảm là một
> **quá trình**, không phải một quý. Về đa cộng tuyến: **33/47 cột có VIF > 10**, cao nhất **798,1**; số
> chiều hiệu dụng (participation ratio) chỉ **12,89** — 47 cột nhưng thực chất chưa tới **13 chiều thông
> tin**. Vì vậy nhóm **không** dùng loại bỏ đặc trưng đơn biến (RFE), mà dùng **ablation theo nhóm** —
> em sẽ nói ở slide 10."

- **Nhấn:** AUC 0,048 (nghịch đảo) · VIF 33/47 (max 798,1) · **12,89 chiều hiệu dụng**.
- **Nén:** slide này **có thể bỏ** nếu thiếu thời gian (thông tin đã có ở slide 10).

## SLIDE 6 — DATA PREPROCESSING & MNAR MECHANISM (0:50)

> **Lời thoại:** "Tiền xử lý có ba điểm. Một: impute **median đặt bên trong Pipeline**, nên trung vị chỉ
> học từ train của từng lần fit — không rò rỉ thống kê sang validation/test. Hai: winsorize theo IQR được
> **khảo sát có đối chứng** nhưng **không dùng ở pipeline chính**, vì mô hình chốt là mô hình cây — không
> nhạy thang đo và outlier. Ba — điều nhóm muốn công bố **trước khi** bị hỏi: **giá trị thiếu ở đây mang
> thông tin nhãn**. Có **703 ô thiếu**; khi `net_margin_latest` thiếu thì chỉ **13,2%** mẫu là nhãn suy
> giảm, còn khi có dữ liệu thì **68,9%** — lệch **55,7 điểm phần trăm**. Nghĩa là 'thiếu dữ liệu' **không
> ngẫu nhiên** (MNAR). Nhóm vẫn dùng median-impute nhưng **báo cáo hiện tượng này** thay vì im lặng, và có
> biến thể ablation để chứng minh kết luận không phụ thuộc cách xử lý."

- **Nhấn:** 703 ô thiếu · median-impute trong Pipeline · MNAR **13,2% vs 68,9%** (Δ −55,7 điểm %).
- **Nén:** slide này cũng **có thể bỏ** nếu thiếu thời gian.

## SLIDE 7 — FOUR FAMILY MODEL BENCHMARKING (1:20)

> **Lời thoại:** "Nhóm chạy **4 họ mô hình** khác nhau về cơ chế học: tuyến tính (Logistic L2), bagging
> (Random Forest), boosting (HistGradientBoosting) và mạng nơ-ron (MLP). Trên test: **RF và Logistic cùng
> đạt AUROC 0,9828**; AP **0,9908** và 0,9887; F1 **0,9600** và 0,9351; HGB 0,9767; MLP 0,9706. Nhưng
> tiêu chí chọn mô hình của nhóm **không phải điểm test**, mà là **AP cross-company** — giữ **trọn** công
> ty ra khỏi train. Theo đó RF thắng với **AP 0,9577**, và quan trọng hơn: **MLP sụt 0,2623 AUROC** khi
> chuyển sang cross-company — bằng chứng mạng nơ-ron không đủ dữ liệu để tổng quát hoá sang công ty mới.
> Nhóm cũng so với **4 baseline**, trong đó **quy tắc kế toán Altman Z″ < 1,1 chỉ đạt AUROC 0,7581**."

- **Nhấn:** 4 họ · RF **0,9828 / 0,9908 / 0,9600** · cross-company AP **0,9577** · MLP **−0,2623** ·
  Altman Z″ **0,7581**.
- **Nếu bị hỏi "sao chọn RF khi Logistic bằng AUROC?"** → "Vì AP cross-company 0,9577 so với 0,9232, F1
  0,9600 so với 0,9351, MCC 0,9039 so với 0,8272, và RF chịu được 33 biến đa cộng tuyến; hơn nữa RF giữ
  **cấu hình mặc định** nên ít rủi ro overfit theo cấu hình."

## SLIDE 8 — CONFUSION MATRIX & ERROR AUDIT (1:10)

> **Lời thoại:** "Ở ngưỡng vận hành 0,7879 trên 64 mẫu test: **TN 25, FP 1, FN 2, TP 36** — accuracy
> **95,31%**, trong khi quy tắc 'đoán lớp đa số' chỉ được 62,35%. Theo từng lớp: lớp **suy giảm** có
> Precision **0,9730**, Recall **0,9474**, F1 **0,9600**; lớp **lành mạnh** F1 **0,9434**. Chỉ **3 mẫu sai**
> trên 64, và nhóm mổ xẻ từng ca: **2 trong 3 lỗi nằm trong 0,01 của biên quyết định** — `HD-2024Q2` lệch
> **0,0044** và `DG-2025Q2` lệch **0,0096**; ca còn lại (`FIVE-2024Q3`) lệch xa. Đáng chú ý: cả hai ca bỏ
> sót đều thuộc doanh nghiệp có nhãn **gần như bất biến ở cấp công ty**, nhưng quý đó chỉ số ngắn hạn lại
> hồi phục. Kết luận: **lỗi nằm ở tầng định nghĩa nhãn, không sửa được bằng tinh chỉnh siêu tham số**."

- **Nhấn:** **25/1/2/36** · 95,31% · F1 lớp suy giảm **0,9600** · 2/3 lỗi sát biên (0,0044 – 0,0096) ·
  overfit gap train−val **0,0342** (train 0,9995 / val 0,9654) đã kiểm soát bằng `max_depth=6` +
  `min_samples_leaf=2`; HGB gap 0,0260; MLP −0,0134.
- **Câu chốt nên nói để "ghi điểm" độc lập:** "Chaotic margin thấp nhưng ta vẫn nói rõ: **hai lỗi đó là
  lỗi của nhãn, không phải của mô hình**."

## SLIDE 9 — COST & CALIBRATION OPTIMIZATION (1:10)

> **Lời thoại:** "Vì sao ngưỡng là 0,7879 mà không phải 0,5? Nhóm đặt bài toán thành **tối thiểu chi phí
> kỳ vọng** `C_FN·FN + C_FP·FP`, với giả định **bỏ sót đắt gấp 5 lần báo động giả**. Về lý thuyết, ngưỡng
> Bayes là **p\* = C_FP/(C_FP + C_FN) = 1/6 ≈ 0,1667** — và ngưỡng này **không phụ thuộc tỉ lệ lớp**. Nhưng
> khi áp lên test: ngưỡng lý thuyết cho **0 bỏ sót và 15 báo động giả ⇒ chi phí 15**; còn ngưỡng thực
> nghiệm **0,7879 đánh đổi lấy 2 bỏ sót + 1 báo động giả ⇒ chi phí 11**. Lý do lệch khỏi ngưỡng Bayes là
> xác suất của Random Forest **chưa hiệu chuẩn hoàn hảo** — sai số hiệu chuẩn **ECE ≈ 0,109**. Vì vậy quy
> tắc nhóm áp dụng: **chọn ngưỡng trên validation, test chỉ để đo**; muốn dùng đúng p\* = 1/6 thì phải
> hiệu chuẩn lại xác suất."

- **Nhấn:** `C_FN = 5, C_FP = 1` · p\* = **1/6 = 0,1667** · chi phí **15 → 11** · ECE **0,109** ·
  "chọn trên val, đo trên test".
- **Canva (⚠️ BẮT BUỘC SỬA):** xoá tiếp **các khối text template** ở slide này (chúng đang chen vào
  **Panel A**). Nội dung Panel A/B hiện **đã đúng kỹ thuật** — không cần sửa số.

## SLIDE 10 — 47-FEATURE TAXONOMY & ABLATION (0:50)

> **Lời thoại:** "Thầy hỏi ngoài baseline nhóm có đề xuất đặc trưng mới gì, và tại sao lại loại bỏ dần.
> Baseline của đồ án là **một** tỷ số (`debt_to_assets_latest`, AUROC 0,8887). Nhóm mở rộng thành **47 cột
> trong 6 khối**: 14 tỷ số hiện trạng, 14 biến YoY, 10 chỉ tiêu tăng trưởng, cấu trúc vốn lưu động, chỉ báo
> căng thẳng (chuỗi quý âm liên tiếp), và **quỹ đạo 8 quý** — cực trị xấu nhất, mức giảm so với đỉnh doanh
> thu. Điểm mới nằm ở chỗ biến dữ liệu **tĩnh thành quỹ đạo**. Về loại bỏ dần: nhóm **không dùng RFE đơn
> biến** vì có **33/47 cột VIF > 10** — nó sẽ loại nhầm biến tương quan nhưng hữu ích. Nhóm làm **ablation
> theo khối**: bỏ trọn từng khối rồi đo lại. Kết quả: bỏ `ratios_latest` giảm **0,0172 AUROC** — lớn nhất
> trong 6 khối, nhưng **vẫn nằm trong độ bất định của 64 mẫu test**; còn 5 khối kia chỉ **≤ 0,003**. Kết
> luận: **đặc trưng đã bão hoà**, nút thắt **không** nằm ở feature engineering."

- **Nhấn:** 6 khối / 47 cột · bỏ `ratios_latest` **−0,0172**, các khối khác **≤ 0,003** ·
  "nằm trong độ bất định của 64 mẫu" (đừng nói "có ý nghĩa thống kê"!).
- **Canva:** sửa chữ **"6 folds" → "6 ablation blocks"** (nhóm đặc trưng, không phải fold CV).

## SLIDE 11 — ENTITY SHORTCUT LEAKAGE (1:20) — slide quan trọng nhất phần giải trình

> **Lời thoại:** "Đây là phát hiện nhóm muốn Hội đồng ghi nhận. Nhóm tự xây một baseline **ngây thơ: chỉ
> biết tên công ty** — lấy tỉ lệ nhãn trung bình của chính công ty đó trong train, không dùng bất kỳ chỉ
> tiêu tài chính nào. Baseline đó đạt **AUROC 0,9858** trên test — **cao hơn cả Random Forest 0,9828**.
> Kiểm định DeLong cho chênh lệch này cho **p = 0,7546**, tức **không có khác biệt ý nghĩa thống kê**.
> Nghĩa là: con số 0,98 trong bảng kết quả thông thường **không** chứng minh năng lực dự báo suy giảm —
> vì chỉ cần nhớ tên công ty là đủ. Khi đo lại nghiêm ngặt: giữ trọn công ty ra khỏi train (cross-company)
> AUROC còn **0,9334**, và khi bỏ hẳn từng công ty một (**Leave-One-Company-Out**) AUROC chỉ còn **0,6280**
> — có công ty mô hình gần như đoán ngẫu nhiên. Vì vậy nhóm kết luận: **rò rỉ ở tầng thực thể là vấn đề
> trung tâm của bộ dữ liệu này**, và đóng góp thật của đồ án là **đo lường + đề xuất giao thức đánh giá
> đúng** (baseline thực thể, cross-company, LOCO, walk-forward, khoảng tin cậy) — thay vì trình bày một
> con số 0,98 gây ngộ nhận."

- **Nhấn:** ticker-prior **0,9858** · DeLong **p = 0,7546** · cross-company **0,9334** · LOCO **0,6280**.
- **Nếu bị hỏi "vậy mô hình có vô dụng không?"** → "Không: mô hình vẫn **hơn xa** quy tắc kế toán Altman
  (0,7581) và mô hình một-biến (0,8887); điều nhóm nói là **lợi thế so với việc chỉ nhớ tên công ty** chưa
  được chứng minh trên 64 mẫu — và muốn chứng minh thì phải mở rộng lên 50–100 công ty."

## SLIDE 12 — THANK YOU (0:20)

> **Lời thoại:** "Em xin hết phần trình bày. Ba điều nhóm muốn để lại: một là đường ống dữ liệu **tái lập
> được** từ SEC với kiểm chứng SHA-256 và tra ngược từng ô; hai là bộ **47 đặc trưng** có căn cứ kinh tế,
> được đo đóng góp bằng ablation; ba là **đo lường trung thực rò rỉ cấp thực thể**. Em sẵn sàng nhận câu hỏi
> của Thầy và Hội đồng."

---

## HAI SLIDE KHUYẾN NGHỊ BỔ SUNG (nếu bạn kịp thêm — 2 slide này trả lời trực tiếp 2 câu hỏi của Thầy)

### [BỔ SUNG 1] — XỬ LÝ MẤT CÂN BẰNG: 4 CẤP ĐỘ & NON-E vs E-MODE (0:50)

**Text dán vào slide (đã kiểm chứng, lấy từ artifact):**

| Mục | Nội dung |
|---|---|
| 4 cấp độ | **Dữ liệu** (SMOTE, ADASYN, RUS, Tomek, ENN) · **Thuật toán** (`class_weight`, `scale_pos_weight`, Focal Loss) · **Tập hợp** (BalancedRF, EasyEnsemble, RUSBoost) · **Hậu xử lý** (threshold moving) — **17 pipeline** đã cài, **11/11 PASS** kiểm chứng chống rò rỉ |
| Bộ 95/5 (đo thực) | Baseline PR-AUC **0,4786** > Non-E **0,3211** ≈ E-Mode **0,3398**; E-Mode dẫn Balanced Acc **0,7352** nhưng **thời gian ×5** |
| Đơn lẻ vs kết hợp (1:50) | Baseline **0,9474** > hybrid **0,9229** > đơn lẻ **0,9104**; **12/16** kỹ thuật đơn lẻ và **3/4** hybrid **không** vượt baseline |
| Dữ liệu thật | tốt nhất `smote_enn` AP **0,9683** vs đối chứng **0,9588**; ΔAP +0,0095, **p = 0,21** ⇒ **chưa** có ý nghĩa |

> **Lời thoại:** "Vì corpus chỉ lệch **1,66**, nhóm không resampling ở pipeline chính — và đó là **kết luận
> có đo lường**, không phải bỏ qua: trên cả ba bộ dữ liệu (95/5, 1:50, corpus thật), can thiệp cân bằng
> chỉ **dịch điểm vận hành** chứ không tạo thêm năng lực xếp hạng. Ngoài ra resampling trên dữ liệu này
> còn **nguy hiểm**: SMOTE nội suy giữa hai quý của **cùng một công ty**, tức hợp thức hoá rò rỉ thực thể.
> Cách xử lý cân bằng của đồ án là **trọng số lớp** (`balanced_subsample`) + **tối ưu ngưỡng theo chi phí**."

### [BỔ SUNG 2] — TÌM KIẾM SIÊU THAM SỐ & GIAO THỨC CHỐNG RÒ RỈ (0:50)

**Text dán vào slide:**

- **GridSearchCV** với `StratifiedGroupKFold(4)`, `refit = average_precision` — lưới **6–18 cấu hình/họ**
  (logistic 8 · RF 18 · HGB 8 · MLP 6).
- **Random Search 40 trial/mô hình**, tham số thang đo lấy **log-uniform**, ghi sổ **153 dòng** `runs.csv`.
- **Tuyệt đối không dùng test** để tìm hoặc chọn siêu tham số (test chỉ chấm một lần).
- **Cấu hình chốt: RF giữ mặc định** (`max_depth=6`, `n_estimators=300`, `min_samples_leaf=2`,
  `class_weight='balanced_subsample'`) vì lưới chỉ cải thiện **+0,003 CV-AP** — trong khoảng nhiễu.

> **Lời thoại:** "Nhóm chạy cả lưới lẫn random search và **ghi lại toàn bộ trial**, nhưng **không triển
> khai** cấu hình tốt nhất theo CV: chênh lệch chỉ **+0,003 CV-AP**, nhỏ hơn độ lệch giữa các fold. Giữ
> mặc định là lựa chọn **có chủ đích** để tránh overfit vào lưới, và nhóm công bố rõ điều này trong
> `summary.json`."

---

## Q&A — 10 CÂU THẦY ĐÃ HỎI, TRẢ LỜI "NÓI" NGẮN (mỗi câu ≤ 25 giây)

| # | Câu hỏi | Trả lời miệng (đọc là nói được ngay) |
|---|---|---|
| 1 | **Dữ liệu có mất cân bằng không?** | "Có, ở **hai cấp**. Cấp mẫu lệch nhẹ: IR **1,66**, entropy **0,9556 bit**. Cấp thực thể nặng: **41%** mẫu thuộc công ty một lớp — HD/LOW/WMT **100%** nhãn suy giảm, ROST **4,7%**. Chính vì vậy nhóm không resampling mà đo rò rỉ thực thể." |
| 2 | **Trình bày công thức đánh giá?** | "Nhóm dùng **F1**, **Balanced Accuracy = (TPR+TNR)/2**, **MCC**, **AP = Σ(R_k−R_{k−1})P_k** và **Brier**. Tại ngưỡng 0,788: P **0,9730**, R **0,9474**, F1 **0,9600**, BA **0,9545**, MCC **0,9039**, Brier **0,0580**. Accuracy chỉ để chẩn đoán — mốc đoán lớp đa số đã **62,35%**." |
| 3 | **Các cách xử lý mất cân bằng?** | "Bốn cấp: **dữ liệu** (SMOTE/ADASYN/RUS/Tomek/ENN), **thuật toán** (`class_weight`, `scale_pos_weight`, Focal Loss), **tập hợp** (BalancedRF, EasyEnsemble, RUSBoost), **hậu xử lý** (threshold moving). Nhóm cài **17 pipeline**, tất cả resampling nằm **trong** pipeline — **11/11 PASS** kiểm chứng chống rò rỉ." |
| 4 | **Ngoài baseline có đề xuất đặc trưng mới gì?** | "Từ **1** tỷ số lên **47 cột / 6 khối**. Điểm mới là **quỹ đạo 8 quý**: thanh khoản xấu nhất trong cửa sổ, mức giảm so với đỉnh doanh thu, chuỗi quý dòng tiền âm. Biến mạnh nhất là `current_ratio_min_window` — AUC đơn biến 0,048, |2AUC−1| **0,904**." |
| 5 | **Tại sao loại bỏ từ từ, khác gì baseline?** | "Nhóm dùng **ablation theo khối** thay vì RFE, vì **33/47 cột VIF > 10** nên RFE đơn biến sẽ loại nhầm biến tương quan hữu ích. Kết quả: chỉ khối `ratios_latest` đóng góp rõ (**−0,0172 AUROC**), 5 khối còn lại **≤ 0,003** ⇒ **đặc trưng đã bão hoà**." |
| 6 | **Non-E Mode và E-Mode là gì?** | "**Non-E**: can thiệp **tĩnh, ngoài mô hình** — resample hoặc đổi trọng số rồi dùng **một** mô hình. **E-Mode**: cân bằng **động, trong từng mô hình con** — mỗi cây/mỗi vòng boosting tự lấy mẫu. Non-E đổi dữ liệu/điểm vận hành, E-Mode đổi quá trình học." |
| 7 | **Kết quả Non-E, E-Mode, Class Weight?** | "Trên bộ 95/5: Baseline PR-AUC **0,4786** > Non-E **0,3211** ≈ E-Mode **0,3398**; E-Mode dẫn Balanced Acc **0,7352** nhưng **chậm gấp 5 lần**. `class_weight` thuần: Recall 0,066 → **0,717** nhưng Precision rơi còn **0,117**. Trên dữ liệu thật, kỹ thuật tốt nhất `smote_enn` chỉ hơn đối chứng **+0,0095 AP với p = 0,21**." |
| 8 | **Kết hợp so với từng phương pháp riêng lẻ?** | "Trên 1:50, Baseline **0,9474** > hybrid **0,9229** > đơn lẻ **0,9104**; **12/16** đơn lẻ và **3/4** hybrid không vượt baseline. Hybrid chỉ thắng khi hai can thiệp **bù trừ** nhau: `smote_enn` **+0,0033 PR-AUC** — và nó cũng là kỹ thuật tốt nhất trên dữ liệu thật (0,9683)." |
| 9 | **Có cần chạy thêm LiteSVM không?** | "Nhóm **đã chạy**: LiteSVM tốt nhất (SGD hinge + `balanced`) đạt AUROC **0,9646**, thấp hơn **cả 4** mô hình chính (RF 0,9828). Giá trị của nó là **phép thử biên tuyến tính** — cho thấy dữ liệu có cấu trúc ngưỡng phi tuyến. Có một bẫy: ngưỡng **không chuyển được** giữa các họ mô hình — F1 của LiteSVM tụt 0,9136 → **0,6667** khi áp ngưỡng 0,788 của RF." |
| 10 | **So Class trên XGBoost? Confusion Matrix, Train vs Val?** | "XGBoost (có `scale_pos_weight` + early stopping) đạt AUROC **0,9818** — **vẫn không vượt RF 0,9828** ⇒ **thuật toán không phải nút thắt**. Confusion của RF @0,788 là **TN 25 / FP 1 / FN 2 / TP 36**; lớp suy giảm P/R/F1 = **0,9730 / 0,9474 / 0,9600**. Overfit: gap train−val **0,0342**, kiểm soát bằng `max_depth=6` + `min_samples_leaf=2`." |

**Ba câu trả lời dự phòng (khả năng cao bị hỏi thêm):**

- *"Tại sao AUROC 0,98 mà còn nói mô hình chưa chứng minh được gì?"* → "Vì baseline **chỉ biết tên công
  ty** đã đạt **0,9858**, DeLong **p = 0,7546**; LOCO chỉ **0,6280**. Nên 0,98 là **trần của dữ liệu**,
  không phải năng lực mô hình."
- *"Sao không hiệu chuẩn để dùng ngưỡng p\* = 1/6?"* → "Đã nằm trong hướng P2: hiệu chuẩn Platt/isotonic
  fit trên train/validation rồi mới dùng ngưỡng Bayes; ở báo cáo nhóm **nói rõ** ngưỡng 0,7879 **không**
  tối ưu Bayes vì ECE 0,109."
- *"Sao chỉ 8 công ty mà dám kết luận?"* → "Nhóm **không** suy rộng: mọi kết luận đều kèm khoảng tin cậy
  (AUROC CI 95% **[0,947; 1,000]**) và nhóm đã ghi rõ **learning curve chưa bão hoà** (gap **0,1049**) ⇒
  việc cần làm là **mở rộng lên 50–100 công ty**, không phải khoe thêm điểm."

---

## KẾ HOẠCH NÉN NẾU HẾT THỜI GIAN (bị nhắc "còn 3 phút")

1. **Bỏ Slide 5 và Slide 6** (multicollinearity, MNAR) — thông tin vẫn xuất hiện một phần ở slide 10.
2. **Bỏ Panel B của Slide 9** — chỉ giữ Panel A (p\* = 1/6 và chi phí 15 → 11).
3. **Giữ nguyên tuyệt đối Slide 11** (rò rỉ thực thể) — đây là slide ghi điểm.
4. Câu chốt 20 giây: "Tóm lại: **dữ liệu tái lập được**, **4 họ mô hình được đối chuẩn**, và **phát hiện
   trung tâm là rò rỉ cấp thực thể** — LOCO chỉ 0,6280."

## CHECKLIST TRƯỚC KHI IN / TRÌNH BÀY

- [ ] **Xoá text template Canva** ở slide 4 và slide 9 (khối *"Reduces human error / Improves
      decision-making / Automates repetitive tasks…"*).
- [ ] Sửa chữ **"6 folds" → "6 ablation blocks"** (slide 10).
- [ ] Kiểm tra mắt nhãn **`IFiling latency buffer`** (slide 3) → sửa thành **"Filing latency buffer"**.
- [ ] Kiểm tra mắt các **tiêu đề lặp** (P6 3×, P12 2×): nếu là hiệu ứng **shadow/outline** của Canva thì
      **bỏ qua**; nếu là chữ chồng chữ thật thì xoá lớp dư.
- [ ] Nếu thêm 2 slide bổ sung: đặt **sau slide 10** (trước slide 11) để mạch "đặc trưng → cân bằng → rò rỉ".
- [ ] Chuẩn bị **file dự phòng**: `docs/BAO-CAO-phan-bien.docx` (39 bảng số thật) và `.pptx` nếu có, vì
      **PDF Canva là ảnh — không copy/không search/không kiểm chứng máy được**.
- [ ] Thuộc **3 con số "đinh"**: **0,9828 · 1/6 = 0,1667 · 0,6280** (mô hình · ngưỡng Bayes · LOCO).

> Nếu Hội đồng hỏi số liệu bất kỳ mà bạn chưa chắc: mở ngay `docs/BAO-CAO-phan-bien.md` (Phụ lục B có bảng
> "số liệu ↔ artifact ↔ lệnh sinh lại") — mọi con số đều truy được về file trong repo.
