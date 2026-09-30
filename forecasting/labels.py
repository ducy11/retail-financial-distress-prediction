"""Nhãn thay thế TÁI LẬP ĐƯỢC cho kiểm chứng độ nhạy của kết luận.

Vì sao cần: nhãn gốc `is_distressed` trong `data/prepared` không tái tạo được từ dữ liệu công bố
(xem `docs/dinh-nghia-nhan.md`): pipeline sinh nhãn gốc chưa được port, và không quy tắc kế toán
đơn giản nào khớp >65%. Module này định nghĩa một nhãn THAY THẾ có công thức công khai.

Nguyên tắc chống rò rỉ: nhãn được tính trên **quý target** (quý mà mô hình phải dự báo). Dữ liệu
của quý target chỉ được công bố ở `label_available_on` (sau `as_of`), nên không nằm trong feature.

`is_distressed_rule = 1` nếu quý target có ≥ `min_signals` tín hiệu căng thẳng tài chính:
1. `net_income < 0` (lỗ ròng)
2. `operating_cash_flow < 0` (dòng tiền hoạt động âm)
3. `operating_income < 0` (lỗ hoạt động)
4. `current_liabilities > current_assets` (vốn lưu động âm)
5. `stockholders_equity < 0` (vốn chủ sở hữu âm)
6. `revenue` giảm > 5% so với cùng kỳ năm trước

Tín hiệu thiếu dữ liệu được coi là KHÔNG xảy ra (bảo thủ, không suy diễn).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import STRESS_MIN_SIGNALS, SUFFIX

#: Mô tả từng tín hiệu (in ra manifest/báo cáo để truy vết).
SIGNAL_DOCS: Dict[str, str] = {
    "net_income<0": "Lỗ ròng trong quý target",
    "operating_cash_flow<0": "Dòng tiền hoạt động âm",
    "operating_income<0": "Lỗ hoạt động",
    "current_liabilities>current_assets": "Vốn lưu động âm (thanh khoản ngắn hạn)",
    "stockholders_equity<0": "Vốn chủ sở hữu âm",
    "revenue_yoy<-5%": "Doanh thu giảm hơn 5% so với cùng kỳ",
}


def _num(row: Dict[str, Any], field: str) -> float | None:
    value = row.get(field + SUFFIX)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def signal_flags(row: Dict[str, Any], rows: List[Dict[str, Any]], idx: int,
                 yoy_threshold: float = -0.05) -> Dict[str, bool]:
    """Cờ 6 tín hiệu căng thẳng của quý target (rows[idx]); thiếu dữ liệu → False."""
    ni, ocf, oi = _num(row, "net_income"), _num(row, "operating_cash_flow"), _num(row, "operating_income")
    ca, cl = _num(row, "current_assets"), _num(row, "current_liabilities")
    eq = _num(row, "stockholders_equity")
    rev = _num(row, "revenue")
    past_rev = _num(rows[idx - 4], "revenue") if idx >= 4 else None
    yoy = ((rev - past_rev) / abs(past_rev)) if (rev is not None and past_rev) else None
    return {
        "net_income<0": ni is not None and ni < 0,
        "operating_cash_flow<0": ocf is not None and ocf < 0,
        "operating_income<0": oi is not None and oi < 0,
        "current_liabilities>current_assets": ca is not None and cl is not None and cl > ca,
        "stockholders_equity<0": eq is not None and eq < 0,
        "revenue_yoy<-5%": yoy is not None and yoy < yoy_threshold,
    }


def label_row(row: Dict[str, Any], rows: List[Dict[str, Any]], idx: int,
              min_signals: int = STRESS_MIN_SIGNALS) -> Tuple[int, List[str]]:
    """(nhãn 0/1, danh sách tín hiệu đã bật) cho quý target tại `rows[idx]`."""
    flags = signal_flags(row, rows, idx)
    active = [name for name, on in flags.items() if on]
    return int(len(active) >= min_signals), active


# ---------------------------------------------------------------------------
# Định nghĩa nhãn THAY THẾ theo công thức công khai (mục RQ4 của báo cáo)
# ---------------------------------------------------------------------------
#: Ngưỡng Altman Z'' cho doanh nghiệp phi sản xuất (Altman 1968/2000): < 1,1 = vùng nguy hiểm.
ALTMAN_DISTRESS_BELOW = 1.1
#: Số quý "tương lai" cho định nghĩa distress-trong-tương-lai (1 năm).
FORWARD_HORIZON_QUARTERS = 4


def _total_liabilities(row: Dict[str, Any]) -> float | None:
    """Nợ phải trả: dùng tag `liabilities` nếu có, ngược lại suy ra từ A = L + E (như features)."""
    liabilities = _num(row, "liabilities")
    if liabilities is not None:
        return liabilities
    assets, equity = _num(row, "total_assets"), _num(row, "stockholders_equity")
    if assets is None or equity is None:
        return None
    return assets - equity


def altman_z_double_prime(row: Dict[str, Any]) -> float | None:
    """Altman Z''-score (doanh nghiệp phi sản xuất, dùng đúng 16 chỉ tiêu của đồ án).

    Z'' = 6,56·(WC/TA) + 3,26·(RE/TA) + 6,72·(EBIT/TA) + 1,05·(BV_E/TL)
    với WC = current_assets − current_liabilities, EBIT = operating_income, TL suy ra từ A = L + E.
    Trả `None` nếu thiếu thành phần bắt buộc (không suy diễn).
    """
    ta = _num(row, "total_assets")
    ca, cl = _num(row, "current_assets"), _num(row, "current_liabilities")
    re_, ebit, equity = _num(row, "retained_earnings"), _num(row, "operating_income"), \
        _num(row, "stockholders_equity")
    tl = _total_liabilities(row)
    if None in (ta, ca, cl, re_, ebit, equity, tl) or ta == 0 or tl == 0:
        return None
    return (6.56 * (ca - cl) / ta + 3.26 * re_ / ta + 6.72 * ebit / ta + 1.05 * equity / tl)


def label_altman(row: Dict[str, Any], rows: List[Dict[str, Any]], idx: int) -> Tuple[int, List[str]]:
    """Nhãn theo Altman Z'': 1 nếu Z'' < `ALTMAN_DISTRESS_BELOW` (thiếu dữ liệu ⇒ 0, bảo thủ)."""
    z = altman_z_double_prime(row)
    if z is None:
        return 0, ["altman_z_missing"]
    return int(z < ALTMAN_DISTRESS_BELOW), [f"altman_z={z:.2f}"]


def label_forward_stress(rows: List[Dict[str, Any]], idx: int,
                         horizon: int = FORWARD_HORIZON_QUARTERS,
                         min_signals: int = STRESS_MIN_SIGNALS) -> Tuple[int, List[str]]:
    """Nhãn "suy giảm trong `horizon` quý TỚI": 1 nếu CÓ ÍT NHẤT một quý sau target có ≥ min tín hiệu.

    Vì sao cần định nghĩa này: nhãn gốc là "trạng thái của quý target" nên rất dễ "dính" theo thời gian
    (90,8% cặp quý liền nhau giữ nguyên nhãn — xem `eda_deep.md`). Nhãn forward đo **sự kiện sắp xảy ra**,
    sát câu hỏi nghiệp vụ hơn ("doanh nghiệp có rủi ro trong 4 quý tới?").
    Trả kèm số tên tín hiệu + số quý tương lai thực sự quan sát được (để biết mẫu có bị "hụt đuôi" không).
    """
    observed = 0
    active: List[str] = []
    for step in range(1, horizon + 1):
        if idx + step >= len(rows):
            break
        observed += 1
        flags = signal_flags(rows[idx + step], rows, idx + step)
        fired = [name for name, on in flags.items() if on]
        if len(fired) >= min_signals:
            active.append(f"q+{step}:{','.join(fired)}")
    label = int(bool(active))
    return label, active + [f"observed_quarters={observed}"]


#: Danh mục định nghĩa nhãn dùng cho kiểm chứng độ nhạy (name → mô tả + hàm gọi).
LABEL_RULES: Dict[str, Dict[str, Any]] = {
    "stress_signals": {
        "description": "≥1 trong 6 tín hiệu căng thẳng của QUÝ TARGET (quy tắc kế toán đơn giản)",
        "function": "stress_signals",
    },
    "altman_z": {
        "description": f"Altman Z''-score < {ALTMAN_DISTRESS_BELOW} (công thức công khai 1968/2000)",
        "function": "altman_z",
    },
    "forward_4q": {
        "description": (f"có ≥1 quý trong {FORWARD_HORIZON_QUARTERS} quý TỚI chạm ngưỡng tín hiệu căng "
                        f"thẳng (sự kiện sắp xảy ra, không phải trạng thái quý target)"),
        "function": "forward_4q",
    },
}


def label_by_rule(rule: str, row: Dict[str, Any], rows: List[Dict[str, Any]],
                  idx: int, min_signals: int = STRESS_MIN_SIGNALS) -> Tuple[int, List[str]]:
    """Áp một định nghĩa nhãn theo tên (`LABEL_RULES`) — dùng thống nhất cho mọi script."""
    if rule not in LABEL_RULES:
        raise KeyError(f"Định nghĩa nhãn không có: {rule!r}; có {sorted(LABEL_RULES)}")
    if rule == "altman_z":
        return label_altman(row, rows, idx)
    if rule == "forward_4q":
        return label_forward_stress(rows, idx, min_signals=min_signals)
    return label_row(row, rows, idx, min_signals=min_signals)
