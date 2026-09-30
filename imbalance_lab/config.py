"""Cấu hình duy nhất cho mọi tham số của lab (không hardcode rải rác trong code)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Thư mục gốc repo + nơi ghi artifact (CSV/JSON/MD/log).
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "reports" / "imbalance"

# ---------------------------------------------------------------------------
# Sinh dữ liệu (mất cân bằng 98/2 như yêu cầu)
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 50_000
N_FEATURES = 20
N_INFORMATIVE = 8
N_REDUNDANT = 6
CLASS_SEP = 1.0
FLIP_Y = 0.0  # giữ đúng tỉ lệ 98/2 (không đảo nhãn)
#: (tỉ lệ lớp âm, tỉ lệ lớp dương) — 98% lớp 0, 2% lớp 1.
CLASS_WEIGHTS: Tuple[float, float] = (0.98, 0.02)

# ---------------------------------------------------------------------------
# Chia tập & cross-validation (bảo toàn tỉ lệ lớp)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20      # holdout test chỉ được dùng ĐÚNG MỘT LẦN, không resample
N_SPLITS = 5          # StratifiedKFold trên tập train_pool

# ---------------------------------------------------------------------------
# Resampling (chỉ áp dụng trên train của từng fold)
# ---------------------------------------------------------------------------
SMOTE_SAMPLING_STRATEGY = 0.10    # SMOTE: nâng thiểu số lên 10% số mẫu đa số
UNDER_SAMPLING_STRATEGY = 0.50    # RandomUnderSampler: hạ đa số để tỉ lệ thiểu/đa = 0.5
SMOTE_K_NEIGHBORS = 5
#: True → ưu tiên `imblearn.pipeline.Pipeline` (đường chính); False → dùng bản nội bộ.
PREFER_IMBLEARN = True

# ---------------------------------------------------------------------------
# Danh mục kỹ thuật (yêu cầu #2) — xem `imbalance_lab/techniques.py`
# ---------------------------------------------------------------------------
#: Tỉ lệ (thiểu/đa) mục tiêu sau OVERSAMPLING trong danh mục kỹ thuật
#: (RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN).
TECHNIQUE_OVER_STRATEGY = 0.50
#: Tỉ lệ (thiểu/đa) mục tiêu sau UNDERSAMPLING trong danh mục kỹ thuật (RandomUnderSampler).
TECHNIQUE_UNDER_STRATEGY = 0.50
#: Số láng giềng cho BorderlineSMOTE/ADASYN ở bản cài đặt nội bộ.
TECHNIQUE_K_NEIGHBORS = 5
#: Số mẫu mặc định của danh mục (nhẹ hơn N_SAMPLES vì chạy ~14 kỹ thuật × K fold).
CATALOG_N_SAMPLES = 20_000
#: Số fold mặc định của danh mục.
CATALOG_N_SPLITS = 5
#: Focal Loss (custom objective của LightGBM): gamma (làm mờ mẫu dễ) và alpha (trọng số lớp dương).
#: `None` ⇒ alpha tính ĐỘNG trong `fit` = n_âm/n (chỉ từ nhãn nhận được — không rò rỉ).
FOCAL_GAMMA = 2.0
FOCAL_ALPHA: Optional[float] = 0.75
#: Số vòng lặp tối đa / learning rate cho mô hình Focal Loss (LightGBM custom objective).
FOCAL_N_ESTIMATORS = 400
FOCAL_LEARNING_RATE = 0.05


#: True → thêm biến thể `resampling_calibrated` (hiệu chuẩn xác suất sau resampling).
CALIBRATE_RESAMPLING = True
#: `isotonic` (không tham số, hợp dữ liệu nhiều) hoặc `sigmoid` (Platt, ít tham số hơn).
CALIBRATION_METHOD = "isotonic"
#: Số fold nội bộ của `CalibratedClassifierCV` (cross-fitting, chỉ trên train của fold).
CALIBRATION_FOLDS = 5

# ---------------------------------------------------------------------------
# Ngưỡng quyết định & chi phí
# ---------------------------------------------------------------------------
#: Chi phí tương đối của bỏ sót (FN) so với báo động giả (FP) khi tìm ngưỡng theo chi phí.
COST_FN = 10.0
COST_FP = 1.0
#: Ngưỡng precision mục tiêu cho chế độ "threshold for min precision".
PRECISION_TARGET = 0.50
#: Hệ số beta của F-beta khi báo cáo (beta = 2 ⇒ coi trọng RECALL gấp đôi precision,
#: hợp bài toán bỏ sót đắt hơn báo động giả). F1 là trường hợp riêng beta = 1.
FBETA_BETA = 2.0
#: Số vòng bootstrap cho khoảng tin cậy 95% của PR-AUC/F1 trên holdout test.
N_BOOTSTRAP = 500

# ---------------------------------------------------------------------------
# Bộ phân loại
# ---------------------------------------------------------------------------
#: Tham số LightGBM (đổi sang xgboost/HGB nếu thiếu lightgbm — xem `models.make_base_classifier`).
LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 1,        # 1 luồng → kết quả ổn định/tái lập được
    "verbose": -1,
}
