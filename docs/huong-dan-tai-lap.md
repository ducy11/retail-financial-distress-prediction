# Hướng dẫn tái lập (từ dữ liệu trong repo đến báo cáo)

## 0. Môi trường

```powershell
python --version            # ≥ 3.11 (đã kiểm thử trên 3.13, Windows)
python -m pip install -r requirements.txt
```

Mã nguồn **không phụ thuộc pandas** (dữ liệu JSON thuần + numpy + scikit-learn).
Xuất Word cần `python-docx`; xuất Slide dùng bộ ghi OOXML nội bộ (không cần `python-pptx`).
Log CLI/test đã lọc cảnh báo **vô hại** của thư viện (`runtime_warnings.py`): `OptimizeWarning: Unknown solver options: iprint` (scikit-learn 1.6 + scipy 1.18, in ở mỗi lần fit `LogisticRegression`) và `PyparsingDeprecationWarning` (matplotlib 3.9 dùng API pyparsing đã deprecate). Cảnh báo khác — kể cả của code dự án — vẫn hiển thị.

## 1. Chạy tất cả trong một lệnh

```powershell
python -m scripts.run_all
```

Thứ tự các bước (19): `data → eda → eda_deep → train → baselines → validation → tuning → evaluate → report → analyze → prep_exp → imbalance_real → search → explain → significance → label_sensitivity → relabel → make_report → export_office`. Log chi tiết ở `reports/results/run_all.log` (kèm lý do nếu một bước bị bỏ qua); `train` và `evaluate` là hai bước lõi, các bước còn lại lỗi thì ghi rõ rồi đi tiếp.

Chạy chọn lọc:

```powershell
python -m scripts.run_all --only train,evaluate          # chỉ chạy một số bước
python -m scripts.run_all --skip tuning,analyze          # bỏ bước nặng
python -m forecasting.tuning --quick                     # lưới tìm kiếm rút gọn
python -m scripts.analyze --quick                        # giảm số lần hoán vị
```

## 2. Chạy từng bước (khi cần kiểm tra riêng)

| Lệnh | Sinh ra | Ý nghĩa |
|---|---|---|
| `python -m forecasting.data` | `data/prepared/*` | Tái tạo split gốc (mặc định bỏ qua nếu đã có; `--force` để ghi lại, byte-identical) |
| `python -m scripts.eda` | `reports/figures/eda/*.png`, `reports/results/eda.md` | 9 hình EDA + thống kê mô tả 14 tỷ số/16 chỉ tiêu, tỉ lệ lớp %, tương quan, nhận xét tự động (mục 3.1–3.6) |
| `python -m scripts.eda_deep` | `reports/results/eda_deep.{json,md}`, `reports/figures/eda_deep/*.png` | EDA chuyên sâu: chất lượng 47 feature, entropy/IR nhãn, liên hệ feature–nhãn, cụm đa cộng tuyến, drift KS/SMD/PSI, rò rỉ & missingness-mang-nhãn |
| `python -m forecasting.train` | `reports/results/summary.json`, `reports/models/best.joblib` | Fit các họ mô hình (logistic/RF/HGB/LightGBM), chọn mô hình trên validation, tính ngưỡng |
| `python -m forecasting.baselines` | `reports/results/baselines.json` | Dummy, ticker-prior, single-feature (đối chứng bắt buộc) |
| `python -m forecasting.validation` | `reports/results/validation_checks.json` | GroupKFold, LOCO, bootstrap CI, tương quan hạng |
| `python -m forecasting.tuning` | `reports/results/tuning.{json,md}` | GridSearchCV chia theo công ty + so với cấu hình mặc định |
| `python -m forecasting.evaluate` | `reports/results/test_evaluation.json` | Chốt trên test đúng một lần, confusion matrix tại ngưỡng vận hành |
| `python -m forecasting.report` | `reports/results/test_predictions.csv` | Xác suất từng mẫu test + histogram |
| `python -m scripts.analyze` | `reports/results/analysis.{json,md}`, 7 hình | Overfit, importance, VIF, ablation, audit nhãn, lỗi, ngưỡng, calibration |
| `python -m scripts.relabel` | `data/prepared-rule/*`, `reports/results/relabel.*` | Nhãn quy tắc tái lập được + kiểm chứng độ nhạy |
| `python -m scripts.audit_data` | `reports/results/data_audit.{json,md}` | Đối soát từng con số với số liệu thật trong snapshot SEC (provenance, VND, split, nhãn quy tắc, artifact báo cáo) |
| `python -m scripts.class_balance` | `reports/results/class_balance.{json,md}` | Mất cân bằng lớp: đếm/ tỉ lệ/ Imbalance Ratio theo tập, theo công ty, theo nhãn quy tắc |
| `python -m imbalance_lab.run` | `reports/imbalance/summary.{md,csv,json}`, `run.log` | Lab 98/2: 3 chiến lược + hiệu chuẩn + mốc minh hoạ SAI (rò rỉ) |
| `python -m imbalance_lab.techniques` | `reports/imbalance/techniques.{md,csv,json}`, `techniques_by_fold.csv`, `techniques.log` | Danh mục 15 kỹ thuật (yêu cầu #2) + threshold tuning trên đường PR, PASS/FAIL chống rò rỉ |
| `python -m imbalance_experiment.main` | `reports/experiment/summary.{md,csv}`, `cv_mean_std.csv`, `results.json`, `pr_curves.png` | Thực nghiệm ĐƠN LẺ vs KẾT HỢP: 17 pipeline, mất cân bằng 1:50, metric mean ± std, chống rò rỉ |
| `python -m scripts.make_report` | `docs/*.md` | Báo cáo + slide + tài liệu |
| `python -m scripts.export_office` | `docs/BAO-CAO.docx`, `docs/BAO-CAO-slide.pptx` | Bản Word và Slide để nộp |
| `python -m unittest discover -s tests -v` | — | Kiểm thử: tái lập byte-identical, chống rò rỉ, tính nhất quán metric, mutation test cho auditor |
| `python -m scripts.predict --sample-id HD-2024Q2 --explain` | — (in ra màn hình) | Demo MỘT mẫu: P(distress), quyết định theo ngưỡng vận hành, cảnh báo backtest, top-K đóng góp SHAP (dùng cho phần trình bày) |
| `python -m scripts.make_report` + `export_office` | `docs/BAO-CAO.docx`, `docs/*.pptx` | Báo cáo/Word/2 bộ slide (tự động 14 slide + dàn bảo vệ 11 slide) |
| `python -m scripts.audit_data` | `reports/results/data_audit.md` | Toàn vẹn số liệu & câu chữ trong báo cáo/slide so với artifact |

## 3. Chạy trên bộ nhãn khác (tùy chọn)

```powershell
$env:FORECASTING_PREPARED_DIR = "e:\CS114\Do-an\do-an\data\prepared-rule"
python -m forecasting.train      # chạy trên split theo nhãn quy tắc
Remove-Item Env:FORECASTING_PREPARED_DIR
```

## 4. Kiểm tra nhanh sau khi chạy

```powershell
Get-ChildItem reports\results, reports\figures, docs | Select-Object Name, Length
python -m unittest discover -s tests -v
```

Nếu một bước bị bỏ qua, log ghi rõ `[CHƯA CÓ]` hoặc `[LỖI]` kèm traceback — không có bước nào im lặng thất bại.