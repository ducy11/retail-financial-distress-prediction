"""Lab xử lý dữ liệu MẤT CÂN BẰNG (imbalanced data) — pipeline không rò rỉ dữ liệu.

Gói này là một lab độc lập với pipeline chính của đồ án bán lẻ (`forecasting/`), vì nó cần
`imbalanced-learn` (repo chính cố ý không phụ thuộc thư viện này).

Lệnh chạy:
- `python -m imbalance_lab.run`         — 3 chiến lược + 1 biến thể hiệu chuẩn + mốc MINH HOẠ SAI.
- `python -m imbalance_lab.techniques`  — DANH MỤC ĐẦY ĐỦ 5 nhóm kỹ thuật (yêu cầu #2) + threshold
  tuning trên đường Precision-Recall.

Thành phần:
- `config`     — mọi tham số (mất cân bằng 98/2, số fold, tham số SMOTE/undersampling, chi phí…).
- `data`       — sinh dữ liệu bảng bằng `make_classification` + chia tập stratified.
- `samplers`   — `imblearn.pipeline.Pipeline` (đường chính) và bản cài đặt nội bộ: SMOTE,
                 RandomOverSampler, BorderlineSMOTE, ADASYN, RandomUnderSampler, TomekLinks, ENN.
- `models`     — baseline, cost-sensitive (`scale_pos_weight`/`class_weight='balanced'`), resampling.
- `losses`     — Focal Loss: grad/hess theo logit + bộ phân loại LightGBM (custom objective).
- `thresholds` — tìm ngưỡng quyết định tối ưu (F1, chi phí kỳ vọng, precision mục tiêu) trên LƯỚI và
                 trên ĐƯỜNG PRECISION-RECALL.
- `metrics`    — metric CHÍNH (precision, recall, F1 binary/macro/weighted/F-beta, PR-AUC, ROC-AUC, MCC),
                 Confusion Matrix, bootstrap CI; **accuracy chỉ là chỉ số chẩn đoán** (kèm mốc lớp đa số).
- `cv`         — `StratifiedKFold` + log phân phối nhãn TRƯỚC/SAU resampling từng fold.
- `run`        — chạy 4 chiến lược, ghi artifact vào `reports/imbalance/`.
- `techniques` — danh mục kỹ thuật (yêu cầu #2) + runner + artifact `reports/imbalance/techniques.*`.
"""
from __future__ import annotations

__all__ = ["config", "data", "samplers", "models", "losses", "thresholds", "metrics", "cv", "run",
           "techniques"]
