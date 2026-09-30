# Sau khi xử lý mất cân bằng thì làm gì tiếp? (kế hoạch có căn cứ số liệu)

*Tài liệu viết tay (không sinh tự động) — mọi con số dưới đây đều trỏ tới artifact trong `reports/`
để kiểm tra lại được bằng `python -m scripts.audit_data` và `python -m unittest discover -s tests -v`.*

## 1. Chẩn đoán: mất cân bằng đã giải quyết tới đâu?

| Câu hỏi | Số liệu đo được | Kết luận |
|---|---|---|
| Nhãn có lệch không? | Nhãn gốc: 202/122 (62,3%/37,7%), **IR = 1,66** toàn corpus (train 1,65); test 38/26 (59,4/40,6) — `reports/results/class_balance.md` | Mất cân bằng **nhẹ** ở cấp mẫu, **KHÔNG** phải nút thắt |
| Nhưng ở cấp công ty? | HD/LOW/WMT **100% nhãn 1**; ROST 4,7% (IR 20,5); DKS 11,6%; ORLY 9,3% — cùng file | Mất cân bằng NGHIÊM TRỌNG và nhãn gần như là thuộc tính thực thể |
| Xử lý mất cân bằng có nâng chất lượng? | Lab 98/2 (`reports/imbalance/summary.md`): PR-AUC **0,950 / 0,944 / 0,940** (baseline / cost-sensitive / SMOTE+undersample) | **Không** — ba chiến lược gần như bằng nhau về thứ hạng xác suất |
| Vậy can thiệp nào có tác dụng? | F1 tốt nhất: cost-sensitive **0,914** vs baseline 0,897; tinh chỉnh ngưỡng: baseline **0,850 → 0,897**; ngưỡng tối ưu lệch 0,093 / 0,387 / 0,890 | Can thiệp đổi **điểm vận hành** (precision/recall), không đổi thứ hạng |
| Nếu resample sai chỗ thì sao? | Mốc minh hoạ SAI: PR-AUC **1,000** trong khi test chứa **786/2940** mẫu tổng hợp | Thao tác sai tạo ảo giác hoàn hảo — đã bị chặn bằng test tự động |

**Kết luận 1:** với dự án này, "xử lý mất cân bằng" **đã xong** (mức nhẹ + đã có `class_weight`, refit
theo AP, ngưỡng theo chi phí, PR-AUC/F1 thay Accuracy). Nút thắt thật nằm ở **nhãn** và **giao thức
đánh giá**, không ở tỉ lệ lớp.

## 2. Ba nút thắt thật (theo bằng chứng đã đo)

| # | Nút thắt | Bằng chứng định lượng |
|---|---|---|
| 1 | **Định nghĩa nhãn không tái tạo được** | Quy tắc kế toán đơn giản khớp tối đa **74,7%** (308 mẫu, `analysis.json::label_audit`); phản chứng `WMT-2015Q2` lãi vẫn bị gán 1 (`docs/dinh-nghia-nhan.md`) |
| 2 | **Rò rỉ cấp thực thể chi phối kết quả in-domain** | `ticker_prior` (không dùng feature) đạt **0,986** > mô hình **0,977–0,983**; cross-company chỉ **0,912–0,933**; LOCO **0,610** |
| 3 | **Chỉ 8 thực thể + feature thừa** | Learning curve chưa bão hoà (`analysis.json::learning_curve`, sizes 56/83/109/135/162): **val AUROC 0,731 → 0,895**, train ≈ 1,000 ⇒ **gap 0,105**. Ở **mọi** mức train_sizes đều có fold bị NaN (fold chỉ có một lớp — nhãn gần như là thuộc tính công ty); artifact ghi `n_nan_folds` và trung bình bỏ qua các ô đó. VIF > 10 ở **33/47** cột |

## 3. Lộ trình P0 → P1 → P2

### P0 — phải làm trước khi tin/dùng kết quả

| Việc | Vì sao (số liệu) | Cách làm | Đo thành công |
|---|---|---|---|
| **Định nghĩa lại nhãn** theo công thức công khai (Altman Z''/Ohlson O ngưỡng, hoặc "distress trong 4 quý tới": âm vốn chủ / cắt cổ tức / hủy niêm yết) | Nhãn hiện tại không tái tạo được (74,7% là mức khớp cao nhất) | `forecasting/labels.py` đã có nhãn quy tắc `stress_signals` → mở rộng thành 2–3 định nghĩa, chạy `python -m scripts.relabel` cho từng định nghĩa | Kết luận (mô hình vs ticker-prior) **giống nhau** trên ≥ 2 định nghĩa nhãn; AP cross-company ổn định |
| **Chọn mô hình & báo cáo theo AP cross-company**, không theo F1 in-domain | Mô hình được chốt theo in-domain (HGB 0,977) nhưng **RF tốt hơn khi cross-company** (0,933 vs 0,926) | `forecasting/train.py`: đổi khoá xếp hạng sang AP cross-company (dùng `forecasting.validation.grouped_cv`), in cả hai bảng | Mô hình chốt là mô hình có **AP cross-company** cao nhất; hai con số được nêu cạnh nhau |
| **Chốt ngưỡng theo chi phí + hiệu chuẩn lại xác suất** | Ngưỡng tối ưu lệch nhau 0,093 vs 0,890 tuỳ chiến lược; sau resampling xác suất bị kéo lệch | `imbalance_lab/` đã có 3 chế độ ngưỡng; bổ sung `CalibratedClassifierCV` (isotonic) cho nhánh resampling rồi so Brier/reliability | Brier ≤ 0,06 và ngưỡng vận hành không còn phụ thuộc việc có resample hay không |

### P1 — tăng sức mạnh thật (cần thêm dữ liệu / thí nghiệm)

| Việc | Vì sao | Cách làm | Đo thành công |
|---|---|---|---|
| **Thêm công ty/ngành** (8 → 40–100) | Learning curve chưa bão hoà; LOCO chỉ tính được AUROC trên **6/8** công ty (phần còn lại có nhãn đơn lớp) | Mở rộng `data/retail-expanded/corpus.json`; dùng `scripts/probe_tags.py` để kiểm độ phủ trước khi nhận công ty | LOCO tính được AUROC trên ≥ 20 công ty; **AP cross-company** tăng và khoảng tin cậy hẹp lại |
| **Lọc feature theo VIF + ablation** | 33/47 cột VIF > 10; bỏ nhóm `ratios_yoy`/`growth` gần như không mất gì | Giữ `working_capital_to_assets`, `current_ratio_*`, `debt_to_assets_latest`, nhóm `path`; bỏ phần dư rồi chạy ablation | Số cột giảm ≥ 30% mà **AP cross-company không giảm** (> 0,93) |
| **Thêm chỉ tiêu XBRL có độ phủ cao** | Đã đo: `AccountsPayableCurrent` 100% → DPO/chu kỳ tiền mặt; `PaymentsToAcquirePropertyPlantAndEquipment` ~75% → FCF = OCF − capex; `IncomeTaxExpenseBenefit` 100% → thuế suất thực tế; **bỏ** `InterestExpense`/`LongTermDebt*` (26–46%) | `scripts/probe_tags.py` chọn tag → mở rộng `retail-expanded` + `forecasting/data.py` | AP cross-company tăng có ý nghĩa (CI không chồng mức cũ) |
| **Mô hình panel/chuỗi** (discrete-time hazard, LSTM nhỏ) thay vector 8 quý | 324 mẫu trượt chia sẻ phần lớn lịch sử; mô hình hiện không dùng thứ tự quý ngoài YoY | Thêm `forecasting/panel.py`, giữ nguyên split + audit | AP cross-company > 0,93 và số tham số không tăng quá 10× |

### P2 — vận hành / production

| Việc | Vì sao | Đo thành công |
|---|---|---|
| **Model card + data card** trong `docs/` | Người dùng phải biết nhãn chưa kiểm chứng được và có rò rỉ thực thể | Có mục "Known limitations" nêu nhãn gốc, IR theo công ty, `ticker_prior` = 0,986 |
| **Monitoring drift** (`reports/monitoring.md`): PSI/K-S theo feature, tỉ lệ dương theo quý, AP cross-company | Dữ liệu bán lẻ dịch chuyển theo mùa/thuế quan | Cảnh báo khi PSI > 0,2 hoặc AP cross-company giảm > 0,05. **Phần feature ĐÃ CÓ**: `reports/results/eda_deep.md` (KS/SMD/PSI train→test + tỉ lệ thiếu theo split); còn lại là chạy lại theo thời gian |
| **Ngưỡng cấu hình hoá + log quyết định** | Ngưỡng là biến nghiệp vụ (FN đắt gấp 5–10 lần FP) | Ngưỡng nằm trong file cấu hình; mỗi dự báo ghi `sample_id`, `P(distress)`, ngưỡng, phiên bản model |
| **CI bắt buộc chạy** `unittest` + `scripts.audit_data` + smoke test `imbalance_lab` | Ba lỗi thật chỉ bị phát hiện nhờ chạy: `searchsorted` sai chiều, `fit_resample` trên pipeline có classifier, số liệu trong docs bị lệch | CI xanh; audit = **0 phát hiện** trên **98.200** phép kiểm tra |

## 4. Việc làm được ngay trong một buổi (không cần dữ liệu mới)

| # | Việc | Trạng thái |
|---|---|---|
| 1 | `forecasting/train.py`: chọn mô hình theo AP cross-company (P0-#2) | ✅ **ĐÃ LÀM** — mô hình chốt đổi từ HistGradientBoosting sang **Random Forest** (AP cross-company 0,958 vs 0,957); test F1 0,866 → **0,960** (3 mẫu sai thay vì 9); test chống hồi quy `TestModelSelection` |
| 2 | Hiệu chuẩn isotonic cho nhánh resampling trong lab (P0-#3) | ✅ **ĐÃ LÀM** — biến thể `resampling_calibrated`: ngưỡng best-F1 0,890 → **0,475** (về quanh 0,5), Brier 0,005 → 0,004, PR-AUC giữ nguyên (0,940 → 0,941) |
| 3 | `docs/model-card.md` (P2-#1) | ✅ **ĐÃ LÀM** — model card 10 mục (intended use, dữ liệu, hạn chế, monitoring, tái lập) |
| 4 | Định nghĩa lại nhãn theo công thức công khai (P0-#1) | ⏳ còn lại — cần chốt 2–3 định nghĩa với giảng viên trước khi chạy lại `scripts.relabel` |
| 5 | Thêm công ty (P1) / lọc feature theo VIF (P1) / mô hình panel (P1) | ⏳ còn lại — cần dữ liệu hoặc thí nghiệm dài hơi |
| 6 | Đo drift + chất lượng feature trước khi tin metric (P2 monitoring, phần feature) | ✅ **ĐÃ LÀM** — `python -m scripts.eda_deep` sinh `reports/results/eda_deep.{json,md}` + 7 hình: tỉ lệ thiếu/outlier/đuôi nặng từng feature, entropy–IR–độ "dính" của nhãn, AUC 1-feature + BH-FDR, cụm đa cộng tuyến, **KS/SMD/PSI train→test**, mức chồng lấn lịch sử và missingness-mang-nhãn (22/47 feature có q < 5%). Còn lại: theo dõi AP cross-company theo thời gian |
| 7 | Xử lý **outlier / đuôi nặng** (winsorize, scaler chịu đuôi nặng) | ✅ **ĐÃ LÀM** — `scripts/experiment_preprocessing.py`: **24 cấu hình** trên dữ liệu thật. Winsorize IQR giúp trung bình **+0,73 điểm %** cross-company AP; cấu hình cao nhất bảng là **`random_forest + iqr` = 0,9590** (tham chiếu `logistic + standard` = 0,9232 ⇒ **+3,58 điểm %**); mức cải thiện lớn nhất rơi vào **mô hình tuyến tính**: **+3,95 điểm %** (`logistic + iqr + RobustScaler` = 0,9071 so với `logistic + none + RobustScaler` = 0,8676). `RobustScaler` **không** giúp logistic khi không winsorize (0,8676 < 0,9232) — kết quả âm được ghi lại thay vì bỏ đi |
| 8 | **Giải thích mô hình (SHAP)** | ✅ **ĐÃ LÀM** — `scripts/explain_model.py`: **KernelSHAP tự cài** (môi trường không có gói `shap`), 64 mẫu × 200 liên minh; **sai số efficiency 2,2e-16**; kiểm chứng giải tích cho hàm tuyến tính trong `tests/test_explain.py`; top-3 feature theo SHAP (`current_ratio_latest`, `current_ratio_min_window`, `debt_to_assets_latest`) và đối chiếu với permutation importance **cùng mô hình RF**: Spearman **0,362** — mức thấp–trung bình, hai phép đo trả lời hai câu hỏi khác nhau nên không thay thế nhau |
| 9 | **Kiểm định ý nghĩa thống kê** khi so mô hình với baseline | ✅ **ĐÃ LÀM** — `scripts/significance.py`: DeLong (1988) + paired bootstrap trên cùng 64 mẫu test; **ΔAUROC(model − ticker_prior): p = 0,7546** ⇒ chưa đủ căn cứ khẳng định mô hình hơn baseline (kết luận trung thực, có CI) |
| 10 | **Boosting nâng cao trong pipeline chính** | ✅ **ĐÃ LÀM** — LightGBM vào `forecasting/models.py` (registry + HYPERPARAMS + lưới tuning + `MODELS_FOR_CHECKS`): cross-company AP **0,947** / AUROC 0,933; RF vẫn được chọn (AP 0,958). XGBoost vẫn bị môi trường chặn → ghi rõ cách sửa (`pip install "scikit-learn>=1.7"`) |
| 11 | **So sánh kỹ thuật xử lý lệch lớp trên DỮ LIỆU THẬT** | ✅ **ĐÃ LÀM** — `scripts/experiment_imbalance_real.py`: 7 kỹ thuật (none/class_weight/RUS/Tomek/SMOTE/SMOTE+ENN/Focal Loss) trên GroupKFold theo công ty; đối chứng AP **0,9588**, tốt nhất `smote_enn` **0,9683** nhưng **p = 0,21 (không có ý nghĩa)**; kiểm tra chống rò rỉ PASS cho mọi kỹ thuật |
| 12 | **Sổ thực nghiệm + tìm kiếm siêu tham số bài bản** | ✅ **ĐÃ LÀM** — `scripts/search.py` + `forecasting/search.py`: random search **40 trial/mô hình** (mặc định của `run_all`; mục tiêu CV-AP cross-company), **sổ `reports/results/runs.csv` (153 dòng)**; kết quả trung thực: **Δ vs GridSearchCV = −0,0035** ⇒ lưới cũ đã đủ tốt. Khi có Optuna chỉ cần thay `sample_params` |
| 13 | **Định nghĩa lại nhãn (P0 cuối cùng)** | ✅ **ĐÃ LÀM phần kỹ thuật** — thêm 2 định nghĩa công khai vào `forecasting/labels.py` (**Altman Z'' < 1,1** và **forward-4Q** = có quý trong 4 quý tới chạm ngưỡng) + `scripts/label_sensitivity.py`: luận điểm "mô hình không vượt `ticker_prior`" đúng ở **3/3 định nghĩa tái lập được** ⇒ kết luận **ổn định theo định nghĩa nhãn**; cross-company AP tụt mạnh dưới nhãn thay thế (0,46–0,61 vs 0,958 với nhãn gốc) ⇒ phải nêu trong hạn chế. Còn lại: chốt định nghĩa chính thức với giảng viên |
| 14 | **Sửa lỗi số liệu hard-code trong tài liệu sinh tự động** | ✅ **ĐÃ LÀM** — `scripts/make_report.py` nay **suy ra** mọi câu chữ từ artifact: slide 9 in "3/64 = **2 FN + 1 FP** (precision 0,973)" (trước đây hard-code "tất cả là FN, precision = 1.000" — **trái** với ma trận nhầm lẫn), slide 10 ghi đúng **Random Forest** (trước ghi "Logistic"), mục 7.5 đọc trực tiếp `error_cases` (DG/FIVE/HD; P 0,174–0,798). Thêm `check_docs_text_consistency` vào `scripts/audit_data.py` để chặn tái phát |
| 15 | **Mục "Tài liệu tham khảo" trong báo cáo** | ✅ **ĐÃ LÀM** — 24 mục (Beaver 1966; Altman 1968, 2000; Ohlson 1980; Barboza và cộng sự 2017; Mai và cộng sự 2019; Breiman 2001; Friedman 2001; Ke và cộng sự 2017; Pedregosa và cộng sự 2011; Chawla và cộng sự 2002; He & Garcia 2009; Saito & Rehmsmeier 2015; Brier 1950; DeLong và cộng sự 1988; Efron & Tibshirani 1993; Benjamini & Hochberg 1995; Virtanen và cộng sự 2020; Lundberg & Lee 2017; Kaufman và cộng sự 2012; Roberts và cộng sự 2017; Shumway 2001; Campbell và cộng sự 2008; SEC EDGAR) + cột "dùng ở đâu trong repo" |
| 16 | **Demo sản phẩm chạy được** | ✅ **ĐÃ LÀM** — `scripts/predict.py` + `tests/test_predict.py` (10 test): dự đoán MỘT quý (`--sample-id` / `--ticker`+`--quarter` / `--input`), quyết định theo **ngưỡng vận hành**, cảnh báo khi mẫu thuộc validation/test (backtest), `--explain` in top-K SHAP. Ví dụ: `HD-2024Q2` → P = 0,7835 < 0,7879 (LỆCH nhãn thật = 1) |
| 17 | **Bộ tài liệu bảo vệ + checklist đối chiếu** | ✅ **ĐÃ LÀM** — `docs/bo-tai-lieu-bao-ve.md` (factsheet + dàn 11 slide + 8 Q&A phản biện), `docs/slide-bao-ve.md` → `docs/BAO-CAO-slide-bao-ve.pptx` (11 slide), `docs/checklist-doi-chieu-yeu-cau.md` (19 tiêu chí → bằng chứng → lệnh) |
| 18 | **Đồng bộ con số toàn bộ tài liệu** | ✅ **ĐÃ LÀM** — audit kiểm cả **câu chữ** trong báo cáo/slide (số FN/FP, mô hình được chốt, có mục tham khảo, có deck/checklist); số test 204 → **220**; `preprocessing_experiment` nay kết luận **theo đúng mô hình được chốt** (winsorize +0,13 điểm % ⇒ giữ pipeline không winsorize, thay vì khuyến nghị mâu thuẫn) |
| 19 | **Kiểm toán độc lập + sửa hết red flags** | ✅ **ĐÃ LÀM** — (a) `scripts/analyze.py` nay lấy mô hình từ `summary.json` (**trước hard-code `logistic`**, khiến bảng ngưỡng + permutation importance + hình `04_threshold_curves.png` thuộc **mô hình khác** với `best.joblib`); (b) `forecasting/baselines.py` thêm **LightGBM** vào bảng so sánh (trước chỉ 3 mô hình); (c) báo cáo §5.3 nay **công bố cấu hình THẬT đang chạy** (mặc định, không phải cấu hình CV tốt nhất) + §1 ghi rõ phép cộng 212+32+64+16 = 324 + §6.6 đổi “giải thích cơ chế” thành **giả thuyết**; (d) sửa 2 ô sai trong tài liệu bảo vệ (`single_feature`: precision 0,714→**0,766**, MCC 0,355→**0,583**); (e) thống nhất câu chữ “test **không** tham gia chọn mô hình/ngưỡng” (thay vì “test chấm 1 lần”) |
| 20 | **Kiểm chứng dữ liệu là THẬT (không bịa)** | ✅ **ĐÃ LÀM** — `scripts/verify_provenance.py` + `tests/test_verify_provenance.py`: băm SHA-256 **20 file companyfacts của SEC** và khớp `data/sec/downloads.json`; **tra ngược từng fact** (tag/kỳ/`accn`/`form`) trong snapshot thô ⇒ **4.609/4.609 ô tìm thấy**, 0 fact thiếu; kiểm quy đổi VND theo đúng `method` (`current_ytd_minus_previous_ytd` = hiệu 2 kỳ luỹ kế) ⇒ **0 lỗi**; **0 ô "điền số cho đủ"** (703 ô không có fact ở SEC đều để `null`). Kết quả: `reports/results/provenance.{json,md}`; audit thêm nhóm `provenance` phải ĐẠT |

## 5. KHÔNG nên làm tiếp (phản khuyến nghị, có căn cứ)

- **Không tối ưu thêm AUROC/Accuracy in-domain**: `ticker_prior` = 0,986 cho thấy trần bị chi phối bởi
  danh tính công ty; cải thiện in-domain không chứng minh năng lực dự báo.
- **Không đưa SMOTE/undersampling vào repo chính**: nhãn là thuộc tính thực thể nên mẫu tổng hợp rơi
  vào "vùng" của chính công ty đã có ⇒ hợp thức hoá đúng loại rò rỉ đang đo (lab đã định lượng: mốc
  SAI đạt PR-AUC 1,000).
- **Không dùng Accuracy** cho bộ 98/2 hay 62/38 (đoán lớp đa số đã đạt 98,0% / 62,3%).
- **Không đổi thuật toán trước khi sửa nhãn**: các họ mô hình chênh nhau < 0,02 AUROC.
- **Không chọn ngưỡng trên test**: ngưỡng chỉ chọn trên xác suất out-of-fold.

## 6. Truy vết & kiểm chứng

```powershell
python -m scripts.audit_data             # 98.200 phép kiểm tra, 0 phát hiện
python -m unittest discover -s tests -v  # 220 test (chống rò rỉ + KernelSHAP + search + nhãn + demo predict)
python -m scripts.class_balance          # mất cân bằng lớp: counts / % / IR / mức
python -m scripts.eda                    # EDA cơ bản có nhận xét: thống kê mô tả 14 tỷ số/16 chỉ tiêu,
                                         # tỉ lệ lớp %, tương quan (Pearson/Spearman) + 9 hình (mục 3.1–3.6)
python -m scripts.eda_deep               # chất lượng feature, entropy/IR nhãn, AUC 1-feature + BH-FDR,
                                         # cụm đa cộng tuyến, drift KS/SMD/PSI, rò rỉ, missingness-mang-nhãn
python -m scripts.experiment_preprocessing  # winsorize × scaler trên dữ liệu thật (24 cấu hình)
python -m scripts.explain_model          # SHAP (KernelSHAP tự cài) + giải thích mẫu sai
python -m scripts.significance           # DeLong + paired bootstrap: mô hình vs ticker_prior
python -m imbalance_lab.run              # lab 98/2: 3 chiến lược + mốc minh hoạ rò rỉ
```

| Artifact tham chiếu | Nội dung |
|---|---|
| `reports/results/class_balance.md` | IR theo tập và theo công ty |
| `reports/results/eda.md` | EDA cơ bản: thống kê mô tả 14 tỷ số (30,6% ngoại lai ở `debt_to_equity`), 16 chỉ tiêu, tỉ lệ lớp 62,3%, 6 cặp \|r\| ≥ 0,8, 5 cặp tương quan do ngoại lai, 8 nhận xét tự động |
| `reports/results/eda_deep.md` | EDA chuyên sâu: 47 feature × (thiếu/outlier/đuôi), nhãn (entropy 0,956 bit; 90,8% cặp quý giữ nguyên nhãn), 8 cặp \|r\| ≥ 0,9, 5 feature drift train→test, Jaccard lịch sử 0,917, 22 feature MNAR |
| `reports/results/preprocessing_experiment.md` | 24 cấu hình winsorize × scaler: winsorize IQR +0,73 điểm % AP trung bình, tốt nhất RF+iqr = 0,9590 (tham chiếu 0,9232) |
| `reports/results/shap.md` | KernelSHAP tự cài: sai số efficiency **2,2e-16**; độ quan trọng toàn cục + giải thích cục bộ 3 mẫu sai |
| `reports/results/significance.md` | DeLong + paired bootstrap trên 64 mẫu test: ΔAUROC(model − ticker_prior) p = 0,7546 |
| `reports/results/data_audit.md` | 0 phát hiện + đối chiếu số liệu thật SEC |
| `reports/results/analysis.md` | audit nhãn, importance, VIF, ablation, learning curve |
| `reports/results/validation_checks.json` | in-domain vs cross-company vs LOCO |
| `reports/imbalance/summary.md` | 3 chiến lược mất cân bằng + ngưỡng + mốc SAI |

