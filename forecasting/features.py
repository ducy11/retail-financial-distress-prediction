"""Tính năng (features) cho từng sample — thuần numpy, không dùng pandas.

Nguyên tắc chống rò rỉ (leak):
- Chỉ dùng lịch sử trong prepared (mọi quý đã có `available_on <= as_of`).
- Tỷ số / tăng trưởng đều tính trong cửa sổ lịch sử; không nhìn target period hay nhãn.
- Giá trị thiếu → NaN; model downstream xử lý (impute mean hoặc tree tự loại).

Feature (47 cột): 14 tỷ số (giá trị mới nhất + YoY) + tăng trưởng YoY của 10 chỉ tiêu chính +
cấu trúc vốn (`working_capital_to_assets`) + cực trị/độ bền theo cửa sổ (nhóm `path`) +
chỉ báo căng thẳng & biến động doanh thu (`distress_quarters_in_window`, `revenue_cv`).

Hai lựa chọn thiết kế được ĐO bằng `scripts/probe_features.py` (cross-company GroupKFold) rồi mới
đưa vào:
1. Nợ phải trả SUY RA: tag `liabilities` chỉ phủ 39% số quý (min 16%) nên `debt_to_assets` /
   `debt_to_equity` trước đây là NaN ở phần lớn mẫu. Nhóm `debt_*` nay dùng
   `total_liabilities` = `liabilities` nếu có, ngược lại `total_assets - stockholders_equity`
   (đẳng thức kế toán; đối chiếu 124 quý có tag: 122 khớp tuyệt đối, 2 quý lệch do restatement —
   xem `scripts/audit_data.py`).
2. Nhóm `path` — cực trị xấu nhất trong cửa sổ, mức giảm so với đỉnh doanh thu và số quý ÂM liên
   tiếp — vì `*_latest` chỉ nhìn 1 quý, còn suy giảm tài chính là một quá trình.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .config import LOOKBACK_QUARTERS, RATIOS, SUFFIX
from .data_loader import history_array, to_float


def _num(row: Dict[str, Any], field: str) -> float:
    return to_float(row.get(field + SUFFIX))


def _window(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tối đa LOOKBACK_QUARTERS quý gần nhất của lịch sử sample."""
    return history_array(sample)[-LOOKBACK_QUARTERS:]


def _last(values: List[float]) -> float:
    for v in reversed(values):
        if math.isfinite(v):
            return v
    return float("nan")


def _yoy(current: float, past: float) -> float:
    if not (math.isfinite(current) and math.isfinite(past)) or past == 0:
        return float("nan")
    return (current - past) / abs(past)


def _series(window: List[Dict[str, Any]], field: str) -> List[float]:
    return [_num(row, field) for row in window]


def _safe_ratio(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)) or den == 0:
        return float("nan")
    return num / den


def _min_finite(values: List[float]) -> float:
    """Giá trị nhỏ nhất trong các phần tử hữu hạn (cực trị XẤU NHẤT của chuỗi)."""
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def _trailing_streak(values: List[float], predicate) -> float:
    """Số quý LIÊN TIẾP gần nhất thoả `predicate` (vd. net income < 0)."""
    streak = 0
    for value in reversed(values):
        if math.isfinite(value) and predicate(value):
            streak += 1
        else:
            break
    return float(streak)


def _total_liabilities(window: List[Dict[str, Any]]) -> float:
    """Nợ phải trả của quý gần nhất: tag `liabilities`, thiếu thì suy ra từ A = L + E.

    Vì sao cần suy ra: tag `liabilities` chỉ có ở 39% số quý (min 16% theo công ty) trong khi
    `total_assets` và `stockholders_equity` phủ 100%. Không suy ra thì `debt_to_assets_latest` /
    `debt_to_equity_latest` (2 trong 10 feature quan trọng nhất) là NaN ở phần lớn mẫu và mô hình
    mất hẳn thông tin đòn bẩy. Đẳng thức A = L + E đã được đối chiếu trên 124 quý có tag.
    """
    liabilities = _last(_series(window, "liabilities"))
    if math.isfinite(liabilities):
        return liabilities
    return _last(_series(window, "total_assets")) - _last(_series(window, "stockholders_equity"))


def _resolve(expr: str, window: List[Dict[str, Any]]) -> float:
    """Giá trị mới nhất của một biểu thức: `a`, `a-b`, hoặc `total_liabilities` (nợ suy ra)."""
    if expr.strip() == "total_liabilities":
        return _total_liabilities(window)
    if "-" in expr:
        a, b = expr.split("-", 1)
        return _last(_series(window, a.strip())) - _last(_series(window, b.strip()))
    return _last(_series(window, expr.strip()))


def _ratio_latest(window: List[Dict[str, Any]], numerator: str, denominator: str) -> float:
    """Tỷ số từ quý gần nhất; hỗ trợ tử/mẫu là chỉ tiêu, phép trừ 'a-b' hoặc `total_liabilities`."""
    return _safe_ratio(_resolve(numerator, window), _resolve(denominator, window))


#: Map tên tỷ số → (tử, mẫu). Tên khớp config.RATIOS để diễn giải nhất quán.
RATIO_PARTS = {
    "gross_margin": ("revenue - cost_of_sales", "revenue"),
    "operating_margin": ("operating_income", "revenue"),
    "net_margin": ("net_income", "revenue"),
    "sgna_pct_revenue": ("selling_general_admin", "revenue"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "quick_ratio": ("current_assets - inventory", "current_liabilities"),
    "debt_to_assets": ("total_liabilities", "total_assets"),
    "debt_to_equity": ("total_liabilities", "stockholders_equity"),
    "inventory_to_sales": ("inventory", "revenue"),
    "receivables_to_sales": ("receivables", "revenue"),
    "cash_to_assets": ("cash_and_equivalents", "total_assets"),
    "ocf_to_sales": ("operating_cash_flow", "revenue"),
    "retained_to_assets": ("retained_earnings", "total_assets"),
    "revenue_per_asset": ("revenue", "total_assets"),
}

GROWTH_FIELDS = [
    "revenue", "cost_of_sales", "inventory", "operating_cash_flow",
    "total_assets", "cash_and_equivalents", "operating_income",
    "current_assets", "current_liabilities", "net_income",
]

#: Nhóm `path` — cực trị xấu nhất trong cửa sổ, mức giảm so với đỉnh doanh thu, chuỗi quý âm.
PATH_FEATURES = [
    "current_ratio_min_window",
    "ocf_to_sales_min_window",
    "net_margin_min_window",
    "revenue_drawdown_window",
    "negative_ni_streak",
    "negative_ocf_streak",
]


def _feature_names() -> List[str]:
    """Thứ tự cột feature ổn định (đồng bộ với _build_features)."""
    names: List[str] = []
    for name in RATIO_PARTS:
        names += [f"{name}_latest", f"{name}_yoy"]
    names += [f"{field}_yoy_growth" for field in GROWTH_FIELDS]
    names += [
        "working_capital_to_assets",
        "distress_quarters_in_window",
        "revenue_cv",
    ]
    names += PATH_FEATURES
    return names


def _build_features(sample: Dict[str, Any]) -> Dict[str, float]:
    """Tính feature của một sample, trả dict {tên: float}."""
    window = _window(sample)
    feats: Dict[str, float] = {}
    if not window:
        return feats

    # 1) 14 tỷ số: mới nhất + YoY (so 4 quý trước)
    for name, (num, den) in RATIO_PARTS.items():
        latest = _ratio_latest(window, num, den)
        feats[f"{name}_latest"] = latest
        if len(window) >= 5:
            feats[f"{name}_yoy"] = _yoy(latest, _ratio_latest(window[:-4], num, den))
        else:
            feats[f"{name}_yoy"] = float("nan")

    # 2) Tăng trưởng YoY của 10 chỉ tiêu chính
    for field in GROWTH_FIELDS:
        series = _series(window, field)
        past = _last(series[:-4]) if len(series) >= 5 else float("nan")
        feats[f"{field}_yoy_growth"] = _yoy(_last(series), past)

    # 3) Cấu trúc vốn / thanh khoản
    #    Lưu ý: `cash_to_assets_latest` đã có từ vòng RATIO_PARTS phía trên — không tính lại
    #    để tránh cột trùng tên (double counting khi tính feature importance).
    total_assets = _last(_series(window, "total_assets"))
    ca = _last(_series(window, "current_assets"))
    cl = _last(_series(window, "current_liabilities"))
    feats["working_capital_to_assets"] = _safe_ratio(ca - cl, total_assets)

    # 4) Chỉ báo căng thẳng + độ biến động doanh thu
    ocf = _series(window, "operating_cash_flow")
    feats["distress_quarters_in_window"] = float(
        sum(1 for v in ocf if math.isfinite(v) and v < 0))
    rev = [v for v in _series(window, "revenue") if math.isfinite(v)]
    if len(rev) > 1:
        mean = float(np.mean(rev))
        feats["revenue_cv"] = float(np.std(rev) / mean) if mean != 0 else float("nan")
    else:
        feats["revenue_cv"] = float("nan")

    # 5) Nhóm `path` — đường đi/độ bền trong cửa sổ (đo bằng scripts/probe_features.py:
    #    +0.03…+0.12 AUROC cross-company so với chỉ dùng giá trị quý mới nhất).
    feats["current_ratio_min_window"] = _min_finite(
        [_safe_ratio(a, b) for a, b in zip(_series(window, "current_assets"),
                                           _series(window, "current_liabilities"))])
    feats["ocf_to_sales_min_window"] = _min_finite(
        [_safe_ratio(a, b) for a, b in zip(ocf, _series(window, "revenue"))])
    feats["net_margin_min_window"] = _min_finite(
        [_safe_ratio(a, b) for a, b in zip(_series(window, "net_income"),
                                           _series(window, "revenue"))])
    latest_rev = _last(_series(window, "revenue"))
    peak_rev = max(rev) if rev else float("nan")
    feats["revenue_drawdown_window"] = (
        _safe_ratio(latest_rev, peak_rev) - 1.0
        if math.isfinite(latest_rev) and math.isfinite(peak_rev) and peak_rev > 0
        else float("nan"))
    feats["negative_ni_streak"] = _trailing_streak(_series(window, "net_income"), lambda v: v < 0)
    feats["negative_ocf_streak"] = _trailing_streak(ocf, lambda v: v < 0)
    return feats


def feature_names() -> List[str]:
    """Tên cột feature theo thứ tự cố định."""
    return _feature_names()


def build_feature_matrix(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Ma trận X (n_samples x n_features); NaN cho giá trị thiếu."""
    names = _feature_names()
    rows = np.zeros((len(samples), len(names)), dtype=float)
    for i, sample in enumerate(samples):
        feats = _build_features(sample)
        for j, name in enumerate(names):
            rows[i, j] = feats.get(name, float("nan"))
    return rows


def extract_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Vector nhãn y (int 0/1)."""
    return np.asarray([int(s["is_distressed"]) for s in samples], dtype=int)


# ---------------------------------------------------------------------------
# Nhóm feature (phục vụ ablation), lọc theo độ dài lịch sử, bảng feature thô
# ---------------------------------------------------------------------------
#: Nhóm feature theo bản chất kinh tế — dùng để ablation "yếu tố nào ảnh hưởng nhiều nhất".
FEATURE_GROUPS: Dict[str, List[str]] = {
    "ratios_latest": [f"{name}_latest" for name in RATIO_PARTS],
    "ratios_yoy": [f"{name}_yoy" for name in RATIO_PARTS],
    "growth": [f"{field}_yoy_growth" for field in GROWTH_FIELDS],
    "structure": ["working_capital_to_assets"],
    "stress": ["distress_quarters_in_window", "revenue_cv"],
    "path": list(PATH_FEATURES),
}


def feature_groups() -> Dict[str, List[str]]:
    """Map nhóm → tên cột feature thật (đã đối chiếu với `feature_names()`)."""
    valid = set(_feature_names())
    return {group: [n for n in names if n in valid] for group, names in FEATURE_GROUPS.items()}


def history_length(sample: Dict[str, Any]) -> int:
    """Số quý lịch sử có sẵn của sample (trước as_of)."""
    return len(history_array(sample))


def filter_by_history(samples: List[Dict[str, Any]],
                      min_quarters: int) -> List[Dict[str, Any]]:
    """Giữ các sample có ≥ `min_quarters` quý lịch sử (loại mẫu quá non, YoY gần như toàn NaN)."""
    return [s for s in samples if history_length(s) >= min_quarters]


def feature_rows(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bảng feature thô (list dict) — dùng cho phân tích lỗi/kiểm toán từng mẫu."""
    names = _feature_names()
    out: List[Dict[str, Any]] = []
    for s in samples:
        feats = _build_features(s)
        row: Dict[str, Any] = {
            "sample_id": s["sample_id"],
            "ticker": s["ticker"],
            "is_distressed": int(s["is_distressed"]),
            "n_history": history_length(s),
        }
        row.update({n: feats.get(n, float("nan")) for n in names})
        out.append(row)
    return out


