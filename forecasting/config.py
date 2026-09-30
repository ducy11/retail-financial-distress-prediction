"""Cấu hình chung: đường dẫn, chỉ tiêu (features), hyperparameter.

Mọi module nên import cấu hình từ đây thay vì hardcode đường dẫn.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn cây thư mục (tính tương đối từ nơi đặt file này)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RETAIL_DIR = DATA_DIR / "retail-expanded"
PREPARED_DIR = DATA_DIR / "prepared"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPORTS_DIR / "models"
DOCS_DIR = ROOT / "docs"
NOTEBOOKS_DIR = ROOT / "notebooks"

PREPARED_FILES = {
    "train": PREPARED_DIR / "train.json",
    "validation": PREPARED_DIR / "validation.json",
    "test": PREPARED_DIR / "test.json",
    "purged": PREPARED_DIR / "purged.json",
}
MANIFEST_FILE = PREPARED_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# Chỉ tiêu và tính năng
# ---------------------------------------------------------------------------
#: Tên đầy đủ các chỉ tiêu (bậc 1) trích từ SEC XBRL.
BASE_FIELDS = [
    "revenue",
    "cost_of_sales",
    "inventory",
    "selling_general_admin",
    "operating_cash_flow",
    "total_assets",
    "cash_and_equivalents",
    "operating_income",
    "current_assets",
    "current_liabilities",
    "net_income",
    "stockholders_equity",
    "liabilities",
    "receivables",
    "short_term_investments",
    "retained_earnings",
]

#: Hậu tố giá trị trong JSON prepared (chuỗi số nguyên VND).
SUFFIX = "_vnd"

#: Nhãn mục tiêu (cột target).
TARGET = "is_distressed"

#: Cửa sổ lịch sử tính năng — dùng 8 quý gần nhất (2 năm tài chính).
LOOKBACK_QUARTERS = 8

#: Các tỷ số tài chính (bậc 2) được tính trong features — key: công thức tiếng Anh để log/diễn giải.
#: Các tỷ số tài chính (bậc 2) được tính trong features — key: công thức tiếng Anh để log/diễn giải.
#: `total_liabilities` = tag `liabilities` nếu quý đó có, ngược lại suy ra từ `total_assets -
#: stockholders_equity` (tag `liabilities` chỉ phủ 39% số quý; xem `forecasting/features.py`).
RATIOS = {
    "gross_margin": "(revenue - cost_of_sales) / revenue",
    "operating_margin": "operating_income / revenue",
    "net_margin": "net_income / revenue",
    "sgna_pct_revenue": "selling_general_admin / revenue",
    "current_ratio": "current_assets / current_liabilities",
    "quick_ratio": "(current_assets - inventory) / current_liabilities",
    "debt_to_assets": "total_liabilities / total_assets",
    "debt_to_equity": "total_liabilities / stockholders_equity",
    "inventory_to_sales": "inventory / revenue",
    "receivables_to_sales": "receivables / revenue",
    "cash_to_assets": "cash_and_equivalents / total_assets",
    "ocf_to_sales": "operating_cash_flow / revenue",
    "retained_to_assets": "retained_earnings / total_assets",
    "revenue_per_asset": "revenue / total_assets",
}

#: Tên cột chuẩn tạo ra bởi features — dùng làm cột X sau khi mã hoá dạng cột.
#: LƯU Ý: cột feature THẬT là `forecasting.features.feature_names()` (36 tỷ số/growth + cấu trúc
#: vốn + cực trị theo cửa sổ + chỉ báo căng thẳng). Hằng số này chỉ liệt kê 14 tỷ số gốc để tham
#: chiếu — đừng hardcode số cột ở bất kỳ đâu, hãy dùng `len(feature_names())`.
FEATURE_COLS = RATIOS.keys()

# ---------------------------------------------------------------------------
# Huấn luyện / đánh giá
# ---------------------------------------------------------------------------
#: Bỏ ngẫu nhiên rollback độc lập với môi trường (SPLIT_TRAIN/EVAL là deterministic vì dựa trên thời gian).
RANDOM_SEED = 42

#: Cần ≥ 2 nhãn ở hai lớp khi train — threshold tối thiểu của tỷ lệ distress.
EVAL_THRESHOLDS = [0.30, 0.35, 0.40, 0.50, 0.60, 0.656]

#: Khoá nhóm cho cross-validation theo thực thể (không để lộ công ty giữa train và test).
GROUP_KEY = "ticker"

#: Số quý lịch sử tối thiểu để feature YoY có nghĩa (YoY cần 5 quý: hiện tại + 4 quý trước).
MIN_HISTORY_QUARTERS = 5

#: Số vòng bootstrap cho khoảng tin cậy metric (mục "độ bất định" của báo cáo).
N_BOOTSTRAP = 2000

#: Chi phí tương đối cho phân tích ngưỡng theo lợi ích kỳ vọng: bỏ sót (FN) đắt hơn báo động giả.
COST_FN = 5.0
COST_FP = 1.0

#: Cấu hình quy tắc nhãn thay thế tái lập được (xem forecasting/labels.py).
STRESS_MIN_SIGNALS = 1
STRESS_RULE_NAME = "stress_signals"

#: Biến môi trường cho phép chạy toàn bộ pipeline trên split khác (vd: data/prepared-rule).
ENV_PREPARED_DIR = "FORECASTING_PREPARED_DIR"


def ensure_dirs() -> None:
    """Tạo thư mục output nếu chưa có."""
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_utf8_stdio() -> None:
    """Ép stdout/stderr dùng UTF-8 (chống UnicodeEncodeError khi pipe/redirect ở console cp1252).

    Python 3.7+: sys.stdout.reconfigure sẵn; trên nền tảng không hỗ trợ thì bỏ qua.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - thiết bị/stream không reconfigure được
            pass


def prepared_dir() -> Path:
    """Thư mục split đang dùng.

    Mặc định `PREPARED_DIR` (nhãn gốc). Đặt biến môi trường `FORECASTING_PREPARED_DIR`
    để chạy pipeline trên split khác — ví dụ `data/prepared-rule` do `scripts.relabel` sinh ra.
    """
    import os

    override = os.environ.get(ENV_PREPARED_DIR)
    return Path(override).resolve() if override else PREPARED_DIR


def prepared_files(base: Path | None = None) -> dict[str, Path]:
    """Đường dẫn 4 split theo `base` (mặc định `prepared_dir()`)."""
    root = base or prepared_dir()
    return {name: root / f"{name}.json" for name in ("train", "validation", "test", "purged")}


def manifest_file(base: Path | None = None) -> Path:
    """Đường dẫn manifest theo `base` (mặc định `prepared_dir()`)."""
    root = base or prepared_dir()
    return root / "manifest.json"

