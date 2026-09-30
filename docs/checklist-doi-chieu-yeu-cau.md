# Checklist đối chiếu yêu cầu & tự đánh giá (dùng trước khi nộp / bảo vệ)

*Tài liệu VIẾT TAY. Mục đích: để người chấm đối chiếu nhanh từng tiêu chí → bằng chứng → lệnh
kiểm chứng; và để nhóm tự rà trước khi nộp. Mọi đường dẫn đều tồn tại trong repo.*

## 1. Bảng đối chiếu theo tiêu chí

| # | Tiêu chí | Đã làm gì | Bằng chứng trong repo | Lệnh kiểm chứng |
|---|---|---|---|---|
| 1 | Xác định bài toán, mục tiêu, phạm vi | Phân loại nhị phân 1 quý tới + xếp hạng rủi ro; 4 câu hỏi nghiên cứu; nêu rõ ngoài phạm vi | `docs/BAO-CAO.md` §1–2; `docs/bo-tai-lieu-bao-ve.md` §1.1 | `python -m scripts.make_report` |
| 2 | Nguồn dữ liệu công khai + provenance | SEC XBRL company-facts; mỗi chỉ tiêu có nguồn; SHA-256 nguồn & từng split | `data/README.md`, `data/prepared/manifest.json`, `scripts/crawl_sec.py`, `scripts/prepare_sec.py` | `python -m scripts.audit_data` |
| 2b | **Dữ liệu THẬT, không bịa** | **20 file SEC** băm SHA-256 khớp registry; **4.609/4.609 ô** tra ngược được trong companyfacts (kèm `accn`/`form`); **0 lỗi quy đổi VND**; **0 ô điền số cho đủ** (703 ô thiếu ở SEC để `null`) | `scripts/verify_provenance.py`, `tests/test_verify_provenance.py`, `reports/results/provenance.{json,md}` | `python -m scripts.verify_provenance` |
| 3 | Tiền xử lý & chống rò rỉ | Impute median + scaler **trong** `Pipeline`; winsorize nghiên cứu riêng; 7 lớp kiểm soát rò rỉ | `forecasting/preprocessing.py`, `forecasting/models.py`, `docs/BAO-CAO.md` §4.6, §9.1 | `python -m unittest tests.test_preprocessing tests.test_pipeline` |
| 4 | Xây dựng đặc trưng có căn cứ | 47 feature: 14 tỷ số (latest/YoY) + 10 growth + cấu trúc vốn + nhóm `path`; đo bằng ablation | `forecasting/features.py`, `docs/BAO-CAO.md` §4.5, §7.4 | `python -m scripts.analyze` |
| 5 | EDA (khám phá dữ liệu) | 9 hình EDA cơ bản (mô tả, phân phối, tương quan) + 7 hình chuyên sâu (thiếu/outlier/đuôi, drift, cụm) + nhận xét tự động | `reports/figures/**`, `reports/results/eda.md`, `reports/results/eda_deep.md` | `python -m scripts.eda`, `python -m scripts.eda_deep` |
| 6 | **≥ 3 mô hình** | 4 họ: Logistic · Random Forest · HistGradientBoosting · LightGBM (+ XGBoost bị chặn phiên bản, ghi rõ cách sửa) | `forecasting/models.py`, `docs/BAO-CAO.md` §5, §6.1 | `python -m forecasting.train` |
| 7 | Tinh chỉnh siêu tham số | `GridSearchCV` + `StratifiedGroupKFold` + `refit=average_precision`; thêm random search 25 trial/mô hình + **sổ thực nghiệm** `runs.csv` | `forecasting/tuning.py`, `forecasting/search.py`, `reports/results/runs.csv` | `python -m forecasting.tuning`, `python -m scripts.search` |
| 8 | Đánh giá đầy đủ & đúng | Precision/Recall/F1/macro-F1/MCC/Brier/AUROC/AP; confusion matrix ở **ngưỡng vận hành**; bootstrap CI 2.000 vòng | `forecasting/evaluation.py`, `reports/results/test_evaluation.json`, `docs/BAO-CAO.md` §6.2–6.5 | `python -m forecasting.evaluate` |
| 9 | So sánh với baseline & kiểm định | 3 baseline (dummy, `ticker_prior`, 1-feature) + DeLong + paired bootstrap | `forecasting/baselines.py`, `forecasting/significance.py`, `docs/BAO-CAO.md` §9.3 | `python -m scripts.significance` |
| 10 | Xử lý mất cân bằng | `class_weight='balanced*'` trong `fit`; lab 98/2 + thí nghiệm trên dữ liệu thật; **không** resample ở pipeline chính | `imbalance_lab/`, `scripts/experiment_imbalance_real.py`, `docs/BAO-CAO.md` §9.4 | `python -m imbalance_lab.run`, `python -m scripts.experiment_imbalance_real` |
| 11 | Giải thích mô hình | KernelSHAP **tự cài đặt** + tự kiểm chứng efficiency; permutation importance; VIF; ablation | `forecasting/explain.py`, `reports/results/shap.md`, `docs/BAO-CAO.md` §7.2, §9.2 | `python -m scripts.explain_model` |
| 12 | Phân tích lỗi | 3 mẫu sai cụ thể + SHAP cục bộ + ngưỡng theo chi phí (FN đắt gấp 5 lần FP) | `reports/results/error_cases.csv`, `docs/BAO-CAO.md` §7.5–7.6 | `python -m scripts.analyze` |
| 13 | **Sản phẩm chạy được (demo)** | CLI dự đoán cho một quý/mẫu mới, in P(distress) theo ngưỡng vận hành, cảnh báo backtest, `--explain` SHAP | `scripts/predict.py`, `tests/test_predict.py` | `python -m scripts.predict --sample-id HD-2024Q2 --explain` |
| 14 | Báo cáo khoa học | 11 mục + 3 phụ lục + **tài liệu tham khảo (24 mục, gắn vị trí dùng thật)** | `docs/BAO-CAO.md` (≈1.100 dòng) | `python -m scripts.make_report` |
| 15 | Slide thuyết trình | Slide tự động 14 mục (khớp artifact) + dàn 11 slide bảo vệ có speaker notes + xuất `.docx`/`.pptx` | `docs/slide.md`, `docs/slide-bao-ve.md`, `docs/BAO-CAO-slide.pptx`, `docs/BAO-CAO-slide-bao-ve.pptx` | `python -m scripts.export_office` |
| 16 | Tái lập (reproducibility) | Một lệnh chạy **21 bước** (kể cả `provenance` + demo `predict`); seed 42; không ghi timestamp; log từng bước; hướng dẫn tái lập | `scripts/run_all.py`, `docs/huong-dan-tai-lap.md`, `reports/results/run_all.log` | `python -m scripts.run_all` |
| 17 | Kiểm thử tự động | 220 test (chống rò rỉ, KernelSHAP, tái lập byte-identical, mutation test cho auditor) | `tests/*.py` | `python -m unittest discover -s tests -v` |
| 18 | Kiểm tra toàn vẹn số liệu | 98.192 phép kiểm tra, 0 phát hiện; đối chiếu với SEC thật; audit cả câu chữ trong báo cáo | `scripts/audit_data.py`, `reports/results/data_audit.md` | `python -m scripts.audit_data` |
| 19 | Trung thực khoa học | Nêu baseline **không thua** mô hình (`ticker_prior` 0,986 vs 0,983); p = 0,7546; mục hạn chế + hướng phát triển | `docs/BAO-CAO.md` §1, §6.4, §9.3, §10.2 | đọc trực tiếp (không script) |

## 2. Ba điểm nhấn nên nói ngay khi mở đầu phần trình bày

1. **Có baseline đối chứng và nó rất mạnh.** Baseline chỉ dùng tỷ lệ nhãn trung bình của công ty
   (`ticker_prior`) đạt AUROC 0,986 — *không thua* mô hình học máy (0,983). Vì vậy nhóm không hô
   "AUROC 0,98" mà định lượng phần nào đến từ **nhận diện công ty** (in-domain 0,983 vs
   cross-company 0,933 vs LOCO 0,610) — xem `docs/BAO-CAO.md` §6.4.
2. **Mọi con số đều truy vết được.** 98.192 phép kiểm tra tự động (0 phát hiện) so số liệu trong
   báo cáo/slide với artifact, 220 test tự động, và toàn bộ báo cáo **sinh từ** `reports/**` — kể cả
   mục 7.5 (phân tích lỗi) nay đọc trực tiếp từ `error_cases` nên không thể lệch.
3. **Có sản phẩm chạy được.** `python -m scripts.predict --sample-id HD-2024Q2 --explain` in
   P(distress), quyết định theo **ngưỡng vận hành 0,788**, cảnh báo backtest và top-6 đóng góp SHAP
   (sai số efficiency ~1e-16) — demo được ngay trong buổi bảo vệ.

## 3. Nếu bị hỏi X, mở ngay file nào?

| Câu hỏi | Mở file / chạy lệnh |
|---|---|
| "Nhãn ở đâu ra, có tin được không?" | `docs/dinh-nghia-nhan.md`; `reports/results/analysis.json::label_audit`; `scripts/relabel.py` |
| "Sao không dùng SMOTE?" | `docs/cac-ky-thuat-mat-can-bang.md`; `reports/results/imbalance_real.md` (§9.4) |
| "Có rò rỉ dữ liệu không?" | `docs/BAO-CAO.md` §4.6 (7 lớp); `tests/test_pipeline.py`; `reports/results/imbalance_real.md` (so khớp ma trận test) |
| "Vì sao chọn Random Forest?" | `docs/BAO-CAO.md` §5.2, §6.6; `reports/results/tuning.md`; `python -m forecasting.tuning` |
| "Mô hình sai ở đâu?" | `reports/results/error_cases.csv`; `docs/BAO-CAO.md` §7.5; `python -m scripts.predict --sample-id FIVE-2024Q3 --explain` |
| "Khác biệt có ý nghĩa không?" | `reports/results/significance.md` (DeLong p = 0,7546; ΔAP p = 0,344) |
| "Winsorize có giúp không, sao không dùng?" | `reports/results/preprocessing_experiment.md` (khuyến nghị tường minh theo mô hình được chốt) |
| "Định nghĩa nhãn khác thì sao?" | `reports/results/label_sensitivity.md` (3/3 định nghĩa giữ nguyên kết luận) |
| "Chạy lại có ra đúng số này không?" | `python -m scripts.run_all` (21 bước) + `docs/huong-dan-tai-lap.md` |
| "Số liệu có phải bịa không?" | `python -m scripts.verify_provenance` — băm SHA-256 20 file SEC, tra ngược 4.609 fact trong companyfacts, bắt cả 'điền số cho đủ' (0 phát hiện) |

## 4. Việc còn lại (nếu có thêm thời gian — đã ghi trong `docs/ke-hoach-tiep-theo.md`)

| Ưu tiên | Việc | Vì sao (bằng chứng) |
|---|---|---|
| P0 | **Chốt định nghĩa nhãn chính thức với giảng viên** (nhãn gốc không tái tạo được: quy tắc khớp tối đa 74,7%) | `docs/BAO-CAO.md` §8; chỉ còn bước ra quyết định, phần kỹ thuật đã xong (4 định nghĩa + độ nhạy) |
| P1 | Mở rộng 50–100 công ty | Learning curve chưa bão hoà (val AUROC 0,795 → 0,902 khi train 83 → 162 mẫu) |
| P1 | Lọc đặc trưng theo cụm/VIF (33/47 cột VIF > 10; chỉ 12,89 chiều hiệu dụng) | `reports/results/eda_deep.md` §cụm đa cộng tuyến |
| P1 | Mô hình panel/survival cho chuỗi quý | 90,8% cặp quý liền nhau giữ nguyên nhãn — dữ liệu có cấu trúc thời gian |
| P2 | Monitoring PSI/KS + ngưỡng cấu hình hoá khi triển khai | 5 feature drift train→test (nặng nhất KS 0,621) |

> Trạng thái tổng: **19/19 hạng mục ở mục 1 đã có bằng chứng trong repo**; các mục P0–P2 ở mục 4 là
> *giới hạn đã biết*, được nêu minh bạch trong báo cáo (§10.2) chứ không phải việc còn thiếu sót.

