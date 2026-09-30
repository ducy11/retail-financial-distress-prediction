"""EDA chuyên sâu — thuần numpy/scipy/scikit-learn, KHÔNG cần pandas.

Bổ sung cho `scripts/eda.py` (đã có: độ phủ chỉ tiêu, cân bằng lớp, boxplot, chuỗi thời gian)
bốn nhóm kiểm tra còn thiếu trước khi tin vào metric:

1. Chất lượng MA TRẬN FEATURE (47 cột): tỉ lệ thiếu theo split, cột hằng/gần hằng, độ lệch
   (`skew`), đuôi nặng (`kurtosis`), outlier theo IQR và |z| > 3, dải giá trị vô lý.
2. NHÃN ở cấp thực thể: entropy/Gini, Imbalance Ratio theo split/công ty/năm/quý, độ dài "spell"
   liên tiếp, xác suất chuyển trạng thái, công ty nhãn đơn lớp, số thực thể hiệu dụng.
3. QUAN HỆ: feature ↔ nhãn (point-biserial, AUC 1 feature + hướng, mutual information, lift theo
   decile, BH-FDR) và feature ↔ feature (Pearson/Spearman, cụm đa cộng tuyến, số chiều hiệu dụng).
4. RÒ RỈ & DỊCH CHUYỂN: KS 2 mẫu + standardized mean difference + PSI train→test, mức chồng lấn
   lịch sử giữa các mẫu (temporal overlap), tỉ lệ dòng trùng feature giữa các split, và
   "missingness có mang thông tin nhãn" (chi-square).

Nguyên tắc: mọi hàm nhận mảng/dict thuần (test độc lập được, không đọc file), CLI nằm ở
`scripts/eda_deep.py` — giống cách tách `scripts/eda.py` (CLI) khỏi `forecasting/features.py` (logic).
EDA **không** fit mô hình và **không** dùng test để chọn cấu hình.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from .config import (BASE_FIELDS, FIGURES_DIR, RANDOM_SEED, RESULTS_DIR, SUFFIX, ensure_dirs,
                     ensure_utf8_stdio)
from .data_loader import load_prepared, to_float
from .features import RATIO_PARTS, build_feature_matrix, extract_labels, feature_names

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend phải đặt trước khi import pyplot)

#: Bốn split chuẩn trong prepared.
SPLITS = ("train", "validation", "test", "purged")
#: Split dùng để tính thống kê feature (purged là dải đệm, không phải tập học).
FEATURE_SPLITS = ("train", "validation", "test")
#: Cặp so sánh dịch chuyển phân phối (chuỗi thời gian ⇒ train cũ hơn test).
REFERENCE_SPLIT = "train"
COMPARISON_SPLIT = "test"

# ---------------------------------------------------------------------------
# Ngưỡng cảnh báo (dùng cho phần "kết luận tự động" trong báo cáo)
# ---------------------------------------------------------------------------
#: Cỡ mẫu tối thiểu sau khi bỏ NaN để tin một hệ số tương quan / AUC 1 feature.
MIN_PAIR_N = 30
#: Tỉ lệ thiếu trên 20% ⇒ feature phải impute nhiều, cảnh báo.
MISSING_WARN_PCT = 20.0
#: |skew| > 5 và excess kurtosis > 20 ⇒ đuôi nặng, tỷ số có thể "nổ" khi mẫu số nhỏ.
SKEW_WARN = 5.0
KURTOSIS_WARN = 20.0
#: Trên 5% số mẫu nằm ngoài [Q1 - 1.5 IQR, Q3 + 1.5 IQR] ⇒ nên xem lại vì sao.
OUTLIER_WARN_PCT = 5.0
#: KS statistic ≥ 0.30 hoặc |SMD| ≥ 0.50 ⇒ phân phối train/test dịch chuyển đáng kể.
DRIFT_KS_WARN = 0.30
DRIFT_SMD_WARN = 0.50
#: |r| ≥ 0.90 được coi là cùng một cụm thông tin (đa cộng tuyến).
CORR_CLUSTER_THRESHOLD = 0.90
#: Ngưỡng phân loại mất cân bằng — giữ GIỐNG `scripts/class_balance.py` để không mâu thuẫn.
BALANCED_MIN_MINORITY_PCT = 40.0
SEVERE_MAX_MINORITY_PCT = 20.0


# ---------------------------------------------------------------------------
# 0. Thống kê cơ bản dùng chung (entropy, Gini, Imbalance Ratio, BH-FDR)
# ---------------------------------------------------------------------------
def shannon_entropy(counts: Sequence[int]) -> float:
    """Entropy Shannon (bit): 0 bit = một lớp, 1 bit = cân bằng nhị phân."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts if c > 0))


def normalized_entropy(counts: Sequence[int]) -> float:
    """Entropy / log2(số lớp) — so sánh được giữa các bài toán khác số lớp."""
    classes = sum(1 for c in counts if c > 0)
    if classes <= 1:
        return 0.0
    return float(shannon_entropy(counts) / math.log2(classes))


def gini_impurity(counts: Sequence[int]) -> float:
    """Gini impurity: 0 = thuần nhất, 0.5 = cân bằng nhị phân."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(1.0 - sum((c / total) ** 2 for c in counts))


def imbalance_ratio(counts: Sequence[int]) -> float:
    """IR = mẫu đa số / mẫu thiểu số (``inf`` nếu chỉ có một lớp)."""
    if len(counts) < 2:
        return float("inf")
    positive, negative = int(counts[0]), int(counts[1])
    if min(positive, negative) == 0:
        return float("inf")
    return max(positive, negative) / min(positive, negative)


def effective_number(counts: Sequence[int]) -> float:
    """Số thực thể hiệu dụng 1/Σp² — 8 công ty cân bằng = 8.0; một công ty chiếm 90% ⇒ gần 1."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(1.0 / sum((c / total) ** 2 for c in counts if c > 0))


def imbalance_level(minority_pct: float) -> str:
    """Phân loại mức mất cân bằng theo tỉ lệ % lớp thiểu số (khớp `scripts/class_balance.py`)."""
    if minority_pct >= BALANCED_MIN_MINORITY_PCT:
        return "balanced"
    if minority_pct >= SEVERE_MAX_MINORITY_PCT:
        return "slightly_imbalanced"
    return "severely_imbalanced"


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """q-value Benjamini–Hochberg (giữ nguyên vị trí; `None` cho giá trị thiếu).

    Vì sao cần: 47 feature × nhiều kiểm định ⇒ với α = 0.05 sẽ có ~2 "phát hiện" thuần ngẫu nhiên.
    """
    indexed = [(i, float(p)) for i, p in enumerate(p_values)
               if p is not None and math.isfinite(float(p))]
    out: List[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return out
    indexed.sort(key=lambda kv: kv[1])
    m = len(indexed)
    running = 1.0
    for rank in range(m, 0, -1):
        index, p = indexed[rank - 1]
        running = min(running, p * m / rank)
        out[index] = float(running)
    return out


def _finite(values: np.ndarray) -> np.ndarray:
    """Bỏ NaN/inf khỏi một dãy."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _round(value: Any, digits: int = 6) -> Any:
    """Làm tròn để JSON/markdown ổn định giữa các lần chạy; NaN/inf → None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _median_impute(X: np.ndarray) -> np.ndarray:
    """Median-impute cho MỤC ĐÍCH EDA (MI, ma trận tương quan, PCA).

    Cố ý KHÔNG dùng cho mô hình: giá trị thay thế phải được học trong Pipeline trên từng
    fold-train, nếu không sẽ là rò rỉ thống kê (xem `forecasting/models.py`).
    """
    arr = np.array(X, dtype=float, copy=True)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            col[np.isnan(col)] = float(np.median(finite))
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 0b. Tiện ích chọn dữ liệu cho EDA CƠ BẢN (mục 3.1–3.6 của báo cáo)
# ---------------------------------------------------------------------------
#: Hệ số quy đổi chỉ tiêu tiền VND → "nghìn tỷ VND" cho bảng mô tả dễ đọc.
VND_PER_TRILLION = 1e12


def latest_ratio_names() -> List[str]:
    """Tên 14 tỷ số ở dạng "giá trị quý gần nhất" (`*_latest`) — đơn vị so sánh được giữa công ty."""
    return [f"{name}_latest" for name in RATIO_PARTS]


def latest_ratio_matrix(samples: Sequence[Dict[str, Any]]
                        ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Ma trận 14 tỷ số `*_latest` + vector nhãn (chọn cột từ `feature_names()` theo tên)."""
    names = latest_ratio_names()
    full = [str(name) for name in feature_names()]
    index = [full.index(name) for name in names]
    return build_feature_matrix(list(samples))[:, index], names, extract_labels(list(samples))


def indicator_value_matrix(rows_by_ticker: Dict[str, List[Dict[str, Any]]],
                           fields: Sequence[str] | None = None,
                           scale: float = VND_PER_TRILLION) -> Tuple[np.ndarray, List[str]]:
    """Ma trận giá trị 16 chỉ tiêu (mọi công ty × mọi quý) đã quy đổi đơn vị (mặc định nghìn tỷ VND).

    Dùng cho bảng thống kê mô tả DỮ LIỆU GỐC (khác ma trận feature 47 cột của mô hình).
    """
    columns = [str(f) for f in (fields or BASE_FIELDS)]
    rows = [row for rows in rows_by_ticker.values() for row in rows]
    matrix = np.full((len(rows), len(columns)), np.nan)
    for i, row in enumerate(rows):
        for j, field in enumerate(columns):
            value = to_float(row.get(field + SUFFIX))
            matrix[i, j] = value / scale if math.isfinite(value) else np.nan
    return matrix, columns


# ---------------------------------------------------------------------------
# 1. Phân tích sâu nhãn/lớp (imbalance, entropy, spell, chuyển trạng thái)
# ---------------------------------------------------------------------------
def _target_end(sample: Dict[str, Any]) -> str:
    """Ngày kết thúc kỳ target — dùng để sắp thứ tự thời gian trong mỗi công ty."""
    return str(sample["request"]["target_period_end"])


def distribution_of(labels: Sequence[int]) -> Dict[str, Any]:
    """Thống kê một dãy nhãn 0/1: đếm, %, IR, entropy, Gini, baseline lớp đa số."""
    values = [int(v) for v in labels]
    n = len(values)
    if n == 0:
        return {"n": 0}
    positive = sum(1 for v in values if v == 1)
    negative = n - positive
    minority = 0 if positive <= negative else 1
    minority_pct = 100.0 * min(positive, negative) / n
    return {
        "n": n,
        "positive": positive,
        "negative": negative,
        "positive_pct": _round(100.0 * positive / n),
        "minority_class": minority,
        "minority_pct": _round(minority_pct),
        "imbalance_ratio": _round(imbalance_ratio([positive, negative])),
        "level": imbalance_level(minority_pct),
        "entropy_bits": _round(shannon_entropy([positive, negative])),
        "entropy_normalized": _round(normalized_entropy([positive, negative])),
        "gini": _round(gini_impurity([positive, negative])),
        "majority_baseline_accuracy_pct": _round(100.0 * max(positive, negative) / n),
    }


def _switches(labels: Sequence[int]) -> int:
    """Số lần nhãn ĐỔI trạng thái giữa hai quý liên tiếp (0 ⇒ nhãn là hằng số của công ty)."""
    return int(sum(1 for a, b in zip(labels, labels[1:]) if a != b))


def _max_run(labels: Sequence[int], value: int) -> int:
    """Chuỗi (spell) dài nhất của một giá trị nhãn — đo mức "dính" theo thời gian."""
    best = current = 0
    for v in labels:
        current = current + 1 if int(v) == value else 0
        best = max(best, current)
    return int(best)


def _transition_counts(sequences: Sequence[Sequence[int]]) -> Dict[str, int]:
    """Đếm chuyển trạng thái nhãn giữa hai quý liên tiếp, gộp trên mọi công ty."""
    counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[f"{int(a)}->{int(b)}"] += 1
    return counts


def _calendar_quarter(date_text: str) -> str:
    """``"2014-07-31"`` → ``"Q3"`` (quý dương lịch của kỳ target — kiểm tra mùa vụ)."""
    text = str(date_text)
    month = int(text[5:7]) if len(text) >= 7 else 0
    return f"Q{(month - 1) // 3 + 1}" if month else "?"


def label_deep_dive(samples_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Phân tích nhãn theo 4 trục: split, công ty, thời gian, chuỗi trạng thái.

    Trả về dict thuần JSON: ``per_split``, ``overall``, ``per_ticker``, ``per_year``,
    ``per_quarter``, ``transitions``, ``entities``, ``flags``.
    """
    pooled = [s for name in SPLITS for s in samples_by_split.get(name, [])]
    per_split = {name: distribution_of([int(s["is_distressed"]) for s in samples_by_split.get(name, [])])
                 for name in SPLITS}
    overall = distribution_of([int(s["is_distressed"]) for s in pooled])

    sequences: List[List[int]] = []
    per_ticker: Dict[str, Dict[str, Any]] = {}
    for ticker in sorted({str(s["ticker"]) for s in pooled}):
        ordered = sorted((s for s in pooled if str(s["ticker"]) == ticker), key=_target_end)
        labels = [int(s["is_distressed"]) for s in ordered]
        sequences.append(labels)
        record = distribution_of(labels)
        record.update({
            "n_switches": _switches(labels),
            "max_run_one": _max_run(labels, 1),
            "max_run_zero": _max_run(labels, 0),
            "first_target": _target_end(ordered[0])[:7],
            "last_target": _target_end(ordered[-1])[:7],
        })
        per_ticker[ticker] = record

    per_year: Dict[str, Dict[str, Any]] = {}
    for year in sorted({_target_end(s)[:4] for s in pooled}):
        per_year[year] = distribution_of([int(s["is_distressed"]) for s in pooled
                                          if _target_end(s)[:4] == year])

    per_quarter: Dict[str, Dict[str, Any]] = {}
    for quarter in ("Q1", "Q2", "Q3", "Q4"):
        rows = [int(s["is_distressed"]) for s in pooled if _calendar_quarter(_target_end(s)) == quarter]
        if rows:
            per_quarter[quarter] = distribution_of(rows)

    transitions = _transition_counts(sequences)
    total_pairs = sum(transitions.values())
    all_one = sorted(t for t, r in per_ticker.items() if r.get("positive") == r.get("n"))
    all_zero = sorted(t for t, r in per_ticker.items() if r.get("negative") == r.get("n"))
    single_class_samples = sum(r["n"] for r in per_ticker.values()
                               if r.get("positive") == r.get("n") or r.get("negative") == r.get("n"))
    worst_ir = sorted(per_ticker.items(), key=lambda kv: -(kv[1].get("imbalance_ratio") or 0.0))

    return {
        "per_split": per_split,
        "overall": overall,
        "per_ticker": per_ticker,
        "per_year": per_year,
        "per_quarter": per_quarter,
        "transitions": {
            "counts": transitions,
            "p_one_given_one": _round(transitions["1->1"] / max(1, transitions["1->1"] + transitions["1->0"])),
            "p_one_given_zero": _round(transitions["0->1"] / max(1, transitions["0->0"] + transitions["0->1"])),
            "persistence_pct": _round(100.0 * (transitions["0->0"] + transitions["1->1"]) / max(1, total_pairs)),
        },
        "entities": {
            "n_tickers": len(per_ticker),
            "effective_number_of_tickers": _round(effective_number([r["n"] for r in per_ticker.values()])),
            "tickers_all_one": all_one,
            "tickers_all_zero": all_zero,
            "share_samples_in_single_class_tickers_pct": _round(
                100.0 * single_class_samples / max(1, len(pooled))),
            "worst_imbalance": [{"ticker": t, "imbalance_ratio": r.get("imbalance_ratio"),
                                 "positive_pct": r.get("positive_pct"), "level": r.get("level")}
                                for t, r in worst_ir[:3]],
            "tickers_needing_entity_level_cv": all_one + all_zero,
        },
        "flags": {
            "overall_entropy_bits": overall.get("entropy_bits"),
            "overall_imbalance_ratio": overall.get("imbalance_ratio"),
            "overall_level": overall.get("level"),
        },
    }


# ---------------------------------------------------------------------------
# 2. Chất lượng ma trận feature (missing, hằng số, đuôi nặng, outlier)
# ---------------------------------------------------------------------------
def feature_statistics(X: np.ndarray, names: Sequence[str]) -> List[Dict[str, Any]]:
    """Thống kê từng cột feature: missing %, hằng/gần hằng, phân vị, skew, đuôi, outlier.

    Vì sao: 47 cột được tạo bởi công thức tỷ số ⇒ dễ có cột gần như hằng số (vô dụng),
    cột thiếu > 50% (impute chi phối) và cột đuôi cực nặng (`debt_to_equity` khi vốn chủ nhỏ).
    """
    records: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        col = np.asarray(X[:, j], dtype=float)
        finite = _finite(col)
        n = int(col.size)
        record: Dict[str, Any] = {
            "feature": str(name),
            "n": n,
            "n_missing": int(n - finite.size),
            "missing_pct": _round(100.0 * (n - finite.size) / max(1, n)),
            "n_unique": int(np.unique(finite).size),
        }
        if finite.size == 0:
            record.update({"is_constant": None, "is_near_constant": None})
            records.append(record)
            continue
        values, counts = np.unique(finite, return_counts=True)
        q1, q3 = (float(v) for v in np.percentile(finite, [25, 75]))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = int(np.sum((finite < lo) | (finite > hi)))
        std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        z_hi = int(np.sum(np.abs(finite - float(np.mean(finite))) / std > 3.0)) if std > 0 else 0
        record.update({
            "is_constant": bool(values.size == 1),
            # "gần hằng số" = gần như vô dụng nhưng KHÁC hằng số tuyệt đối ⇒ hai cờ loại trừ nhau
            "is_near_constant": bool(values.size > 1 and counts.max() / finite.size >= 0.95),
            "top_value_share": _round(float(counts.max() / finite.size), 4),
            "mean": _round(float(np.mean(finite))),
            "std": _round(std),
            "min": _round(float(finite.min())),
            "p1": _round(float(np.percentile(finite, 1))),
            "q1": _round(q1),
            "median": _round(float(np.median(finite))),
            "q3": _round(q3),
            "p99": _round(float(np.percentile(finite, 99))),
            "max": _round(float(finite.max())),
            "skew": _round(float(sps.skew(finite, bias=False))) if finite.size >= 3 and std > 0 else None,
            "kurtosis_excess": (_round(float(sps.kurtosis(finite, bias=False, fisher=True)))
                                if finite.size >= 4 and std > 0 else None),
            "zero_pct": _round(100.0 * float(np.sum(finite == 0)) / finite.size),
            "negative_pct": _round(100.0 * float(np.sum(finite < 0)) / finite.size),
            "iqr_outlier_n": outliers,
            "iqr_outlier_pct": _round(100.0 * outliers / finite.size),
            "z3_outlier_pct": _round(100.0 * z_hi / finite.size),
        })
        records.append(record)
    return records


def feature_quality_flags(stats: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Danh sách feature cần xem lại theo từng loại vấn đề (dùng cho kết luận tự động)."""
    def pick(predicate) -> List[str]:
        return [str(r["feature"]) for r in stats if predicate(r)]

    return {
        "constant": pick(lambda r: r.get("is_constant")),
        "near_constant": pick(lambda r: r.get("is_near_constant")),
        "missing_gt_20pct": sorted(pick(lambda r: (r.get("missing_pct") or 0) > MISSING_WARN_PCT),
                                   key=lambda f: -next(r["missing_pct"] for r in stats
                                                       if r["feature"] == f)),
        "heavy_tail": pick(lambda r: (abs(r.get("skew") or 0) > SKEW_WARN)
                           or (r.get("kurtosis_excess") or 0) > KURTOSIS_WARN),
        "many_outliers": pick(lambda r: (r.get("iqr_outlier_pct") or 0) > OUTLIER_WARN_PCT),
    }


# ---------------------------------------------------------------------------
# 3a. Quan hệ feature ↔ nhãn (point-biserial, AUC 1 feature, MI, lift, BH-FDR)
# ---------------------------------------------------------------------------
def target_association(X: np.ndarray, y: Sequence[int], names: Sequence[str],
                       min_pairs: int = MIN_PAIR_N) -> Dict[str, Any]:
    """Mức liên hệ của TỪNG feature với nhãn, kèm kiểm định và hiệu chỉnh đa so sánh.

    Trả về 4 chỉ số bổ trợ nhau (không thay thế permutation importance ở `scripts/analyze.py`):
    - ``auc`` / ``effect_rank_biserial`` = 2·AUC − 1: đo mức TÁCH hai lớp, có hướng;
    - ``point_biserial_r`` + ``p_value`` + ``q_value_bh``: ý nghĩa thống kê sau hiệu chỉnh;
    - ``mutual_information``: bắt được liên hệ PHI TUYẾN mà Pearson bỏ qua;
    - ``lift_top_decile``: nhãn dương ở 10% giá trị cao nhất so với tỉ lệ nền (đọc theo nghiệp vụ).
    """
    y_arr = np.asarray([int(v) for v in y], dtype=int)
    mi_available = False
    mi_values = np.zeros(len(names))
    if np.unique(y_arr).size == 2 and len(y_arr) > 0:
        mi_values = mutual_info_classif(_median_impute(X), y_arr, discrete_features=False,
                                        random_state=RANDOM_SEED)
        mi_available = True

    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        col = np.asarray(X[:, j], dtype=float)
        mask = np.isfinite(col)
        n_pairs = int(mask.sum())
        row: Dict[str, Any] = {
            "feature": str(name),
            "n_pairs": n_pairs,
            "missing_pct": _round(100.0 * (col.size - n_pairs) / max(1, col.size)),
            "auc": None, "effect_rank_biserial": None, "direction": None,
            "point_biserial_r": None, "p_value": None, "q_value_bh": None,
            "lift_top_decile": None,
            "mutual_information": _round(float(mi_values[j]), 4) if mi_available else None,
        }
        if n_pairs >= min_pairs:
            x = col[mask]
            labels = y_arr[mask]
            if np.unique(labels).size == 2 and float(np.std(x, ddof=1)) > 0:
                auc = float(roc_auc_score(labels, x))
                r_value, p_value = sps.pearsonr(x, labels)
                base = float(labels.mean())
                threshold = float(np.quantile(x, 0.9))
                top = labels[x >= threshold]
                row.update({
                    "auc": _round(auc, 4),
                    "effect_rank_biserial": _round(2.0 * auc - 1.0, 4),
                    "direction": "giá trị cao ⇒ nhãn 1" if auc > 0.5 else "giá trị thấp ⇒ nhãn 1",
                    "point_biserial_r": _round(float(r_value), 4),
                    "p_value": _round(float(p_value), 8),
                    "lift_top_decile": _round(float(top.mean()) / base, 4) if base > 0 and top.size else None,
                })
        rows.append(row)

    q_values = benjamini_hochberg([r["p_value"] for r in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value_bh"] = _round(q_value, 6) if q_value is not None else None

    by_effect = sorted(rows, key=lambda r: -abs(r["effect_rank_biserial"] or 0.0))
    by_mi = sorted(rows, key=lambda r: -(r["mutual_information"] or 0.0))
    return {
        "per_feature": rows,
        "ranked_by_effect": by_effect,
        "ranked_by_mutual_information": by_mi,
        "n_significant_after_bh_5pct": int(sum(1 for r in rows if (r["q_value_bh"] or 1.0) < 0.05)),
        "n_effect_ge_0_30": int(sum(1 for r in rows if abs(r["effect_rank_biserial"] or 0.0) >= 0.30)),
        "n_usable_features": int(sum(1 for r in rows if r["auc"] is not None)),
        "multiple_testing_note": ("47 kiểm định ⇒ q-value BH; xếp hạng theo |2·AUC−1| để không phụ "
                                 "thuộc cỡ mẫu, không dùng p-value đơn lẻ."),
    }


# ---------------------------------------------------------------------------
# 3b. Quan hệ feature ↔ feature (tương quan, cụm đa cộng tuyến, số chiều hiệu dụng)
# ---------------------------------------------------------------------------
def pairwise_correlation(X: np.ndarray, method: str = "pearson",
                         min_pairs: int = MIN_PAIR_N) -> np.ndarray:
    """Ma trận tương quan theo cặp GIÁ TRỊ HỮU HẠN (pairwise complete case).

    Vì sao không impute rồi mới tính: 47 cột có lượng missing rất khác nhau; impute trước sẽ
    tạo tương quan giả giữa các cột cùng bị thiếu (artefact của median).
    """
    arr = np.asarray(X, dtype=float)
    k = arr.shape[1]
    if method == "spearman":
        ranked = np.full_like(arr, np.nan)
        for j in range(k):
            column = arr[:, j]
            mask = np.isfinite(column)
            if mask.any():
                ranked[mask, j] = sps.rankdata(column[mask])
        arr = ranked
    out = np.full((k, k), np.nan)
    for i in range(k):
        out[i, i] = 1.0
        for j in range(i + 1, k):
            mask = np.isfinite(arr[:, i]) & np.isfinite(arr[:, j])
            if int(mask.sum()) >= min_pairs:
                first, second = arr[mask, i], arr[mask, j]
                if float(np.std(first)) > 0 and float(np.std(second)) > 0:
                    value = float(np.corrcoef(first, second)[0, 1])
                    out[i, j] = out[j, i] = value
    return out


def correlation_clusters(corr: np.ndarray, names: Sequence[str],
                         threshold: float = CORR_CLUSTER_THRESHOLD) -> List[List[str]]:
    """Gom feature thành cụm "cùng một thông tin" bằng union-find trên |r| ≥ ngưỡng."""
    n = len(names)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            value = corr[i, j]
            if math.isfinite(value) and abs(value) >= threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
    groups: Dict[int, List[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(str(names[i]))
    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    return sorted(clusters, key=lambda c: (-len(c), c[0]))


def effective_dimensionality(X: np.ndarray) -> Dict[str, Any]:
    """Số chiều hiệu dụng của ma trận feature (median-impute + chuẩn hoá, chỉ để EDA).

    - ``participation_ratio`` = (Σλ)²/Σλ² — nhỏ hơn nhiều so với số cột ⇒ feature thừa.
    - ``n_components_for_95pct_variance`` — số thành phần cần để giữ 95% phương sai.
    """
    arr = _median_impute(X)
    std = arr.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    standardized = (arr - arr.mean(axis=0)) / std
    singular = np.linalg.svd(standardized, compute_uv=False)
    variance = singular ** 2
    total = float(variance.sum())
    if total <= 0:
        return {"n_columns": int(arr.shape[1]), "participation_ratio": None,
                "n_components_for_95pct_variance": None}
    cumulative = np.cumsum(variance) / total
    return {
        "n_columns": int(arr.shape[1]),
        "participation_ratio": _round(float(total ** 2 / float((variance ** 2).sum()))),
        "n_components_for_95pct_variance": int(np.searchsorted(cumulative, 0.95) + 1),
        "top_eigenvalue_share_pct": _round(100.0 * float(variance[0]) / total),
    }


def correlation_analysis(X: np.ndarray, names: Sequence[str],
                         threshold: float = CORR_CLUSTER_THRESHOLD) -> Dict[str, Any]:
    """Tương quan feature ↔ feature: cặp mạnh nhất, cụm đa cộng tuyến, số chiều hiệu dụng.

    So sánh Pearson (tuyến tính) với Spearman (hạng) để biết quan hệ có bị outlier chi phối không:
    chênh lệch lớn giữa hai hệ số ở cùng một cặp là dấu hiệu đuôi nặng.
    """
    pearson = pairwise_correlation(X, "pearson")
    spearman = pairwise_correlation(X, "spearman")
    n = len(names)
    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            value = pearson[i, j]
            if not math.isfinite(value):
                continue
            pairs.append({"a": str(names[i]), "b": str(names[j]),
                          "pearson": _round(value, 4),
                          "spearman": _round(spearman[i, j], 4),
                          "abs_r": _round(abs(value), 4)})
    strongest = sorted(pairs, key=lambda p: -p["abs_r"])
    strong = [p for p in strongest if p["abs_r"] >= threshold]
    clusters = correlation_clusters(pearson, names, threshold)
    return {
        "threshold": threshold,
        "n_pairs_computed": len(pairs),
        "n_pairs_abs_ge_threshold": len(strong),
        "strongest_pairs": strongest[:20],
        "clusters_abs_ge_threshold": clusters,
        "n_features_in_clusters": int(sum(len(c) for c in clusters)),
        "effective_dimensionality": effective_dimensionality(X),
        "pearson_matrix": [[_round(v, 4) for v in row] for row in pearson],
        "spearman_matrix": [[_round(v, 4) for v in row] for row in spearman],
        "feature_order": [str(name) for name in names],
        "note": ("Tương quan cặp dùng chung giá trị hữu hạn (không impute) nên cỡ mẫu mỗi cặp "
                 "khác nhau — đọc kèm `missing_pct` trong mục feature."),
    }


# ---------------------------------------------------------------------------
# 4a. Dịch chuyển phân phối train → test (KS, SMD, PSI)
# ---------------------------------------------------------------------------
def population_stability_index(reference: Sequence[float], current: Sequence[float],
                               bins: int = 10) -> Optional[float]:
    """PSI theo decile của mẫu tham chiếu — chỉ số vận hành quen thuộc (PSI > 0.2 ⇒ dịch chuyển)."""
    ref, cur = _finite(np.asarray(reference, dtype=float)), _finite(np.asarray(current, dtype=float))
    if ref.size < bins or cur.size < bins:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return None
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size
    epsilon = 1e-6
    ref_p = np.clip(ref_share, epsilon, None)
    cur_p = np.clip(cur_share, epsilon, None)
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())


def drift_analysis(X_ref: np.ndarray, X_new: np.ndarray, names: Sequence[str],
                   ref_name: str = REFERENCE_SPLIT, new_name: str = COMPARISON_SPLIT) -> Dict[str, Any]:
    """So phân phối từng feature giữa hai split theo thời gian (train → test).

    Ba thước đo vì mỗi cái bắt một kiểu dịch chuyển: KS (toàn phân phối, không giả định),
    SMD (dịch trung bình theo độ lệch chuẩn gộp), PSI (chuẩn vận hành, dễ đặt ngưỡng cảnh báo).
    Dịch chuyển mạnh ⇒ kết luận trên test có thể phản ánh phân bố mới, không phải năng lực mô hình.
    """
    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        ref = _finite(X_ref[:, j])
        new = _finite(X_new[:, j])
        row: Dict[str, Any] = {"feature": str(name), "n_ref": int(ref.size), "n_new": int(new.size),
                              "mean_ref": _round(float(ref.mean())) if ref.size else None,
                              "mean_new": _round(float(new.mean())) if new.size else None,
                              "ks_statistic": None, "ks_p_value": None, "smd": None, "psi": None}
        if ref.size >= 3 and new.size >= 3:
            ks = sps.ks_2samp(ref, new)
            denominator = math.sqrt(float(ref.var(ddof=1) + new.var(ddof=1)) / 2.0)
            row.update({
                "ks_statistic": _round(float(ks.statistic), 4),
                "ks_p_value": _round(float(ks.pvalue), 8),
                "smd": _round((float(new.mean()) - float(ref.mean())) / denominator, 4) if denominator > 0 else None,
                "psi": _round(population_stability_index(ref, new)),
            })
        rows.append(row)

    q_values = benjamini_hochberg([r["ks_p_value"] for r in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value_bh"] = _round(q_value, 6) if q_value is not None else None
    ranked = sorted(rows, key=lambda r: -(r["ks_statistic"] or 0.0))
    drifted = [r["feature"] for r in rows
               if (r["ks_statistic"] or 0.0) >= DRIFT_KS_WARN or abs(r["smd"] or 0.0) >= DRIFT_SMD_WARN]
    return {
        "reference": ref_name,
        "comparison": new_name,
        "per_feature": rows,
        "ranked_by_ks": ranked,
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "n_psi_above_0_2": int(sum(1 for r in rows if (r["psi"] or 0.0) > 0.2)),
        "n_significant_after_bh_5pct": int(sum(1 for r in rows if (r["q_value_bh"] or 1.0) < 0.05)),
    }


# ---------------------------------------------------------------------------
# 4b. Missingness: mức thiếu theo split + "thiếu có mang thông tin nhãn?"
# ---------------------------------------------------------------------------
def missingness_analysis(X_by_split: Dict[str, np.ndarray], y_by_split: Dict[str, Sequence[int]],
                         names: Sequence[str]) -> Dict[str, Any]:
    """Phân tích giá trị thiếu: theo split, theo nhãn (MNAR), đồng-thiếu, mật độ thiếu mỗi mẫu.

    Câu hỏi kỹ thuật: nếu tỉ lệ nhãn 1 khác nhau CÓ Ý NGHĨA giữa nhóm thiếu và nhóm có dữ liệu
    thì giá trị thiếu không phải "nhiễu vô hại" mà mang thông tin ⇒ median-impute có thể xoá
    tín hiệu (hoặc tạo tín hiệu giả) và cần thêm cờ `is_missing` cho feature đó.
    """
    missing_by_split: Dict[str, Dict[str, Any]] = {}
    for split, matrix in X_by_split.items():
        n = max(1, matrix.shape[0])
        missing_by_split[split] = {
            str(name): _round(100.0 * float(np.sum(~np.isfinite(matrix[:, j]))) / n)
            for j, name in enumerate(names)
        }

    parts = [X_by_split[s] for s in FEATURE_SPLITS if s in X_by_split]
    pooled = np.vstack(parts) if parts else np.zeros((0, len(names)))
    labels = np.concatenate([np.asarray(y_by_split[s], dtype=int)
                             for s in FEATURE_SPLITS if s in X_by_split]) if parts \
        else np.zeros(0, dtype=int)

    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        column = pooled[:, j]
        present = np.isfinite(column)
        absent = ~present
        row: Dict[str, Any] = {"feature": str(name), "n_missing": int(absent.sum()),
                               "n_present": int(present.sum()), "target_rate_when_missing_pct": None,
                               "target_rate_when_present_pct": None, "delta_pct_points": None,
                               "chi2_p_value": None, "q_value_bh": None}
        if absent.sum() >= 5 and present.sum() >= 5:
            rate_absent = 100.0 * float(labels[absent].mean())
            rate_present = 100.0 * float(labels[present].mean())
            table = np.array([
                [int(labels[absent].sum()), int(absent.sum() - labels[absent].sum())],
                [int(labels[present].sum()), int(present.sum() - labels[present].sum())],
            ])
            try:
                _, p_value, _, _ = sps.chi2_contingency(table)
            except ValueError:  # bảng suy biến (một hàng toàn 0)
                p_value = 1.0
            row.update({"target_rate_when_missing_pct": _round(rate_absent),
                        "target_rate_when_present_pct": _round(rate_present),
                        "delta_pct_points": _round(rate_absent - rate_present),
                        "chi2_p_value": _round(float(p_value), 8)})
        rows.append(row)
    for row, q_value in zip(rows, benjamini_hochberg([r["chi2_p_value"] for r in rows])):
        row["q_value_bh"] = _round(q_value, 6) if q_value is not None else None
    ranked = sorted(rows, key=lambda r: -abs(r["delta_pct_points"] or 0.0))

    absent_matrix = ~np.isfinite(pooled) if pooled.size else np.zeros((0, len(names)), dtype=bool)
    co_missing: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = int(np.sum(absent_matrix[:, i] & absent_matrix[:, j]))
            union = int(np.sum(absent_matrix[:, i] | absent_matrix[:, j]))
            if both:
                co_missing.append({"a": str(names[i]), "b": str(names[j]), "n_both_missing": both,
                                   "jaccard": _round(both / union)})
    co_missing.sort(key=lambda r: (-r["n_both_missing"], r["a"]))

    patterns: Dict[bytes, List[str]] = {}
    for j, name in enumerate(names):
        patterns.setdefault(absent_matrix[:, j].tobytes(), []).append(str(name))
    identical_groups = sorted([g for g in patterns.values() if len(g) > 1],
                             key=lambda g: (-len(g), g[0]))

    per_sample = absent_matrix.sum(axis=1) if absent_matrix.size else np.zeros(0, dtype=int)
    n_features = max(1, len(names))
    return {
        "missing_pct_by_split": missing_by_split,
        "label_informativeness": {
            "per_feature": rows,
            "ranked_by_abs_delta": ranked,
            "n_significant_after_bh_5pct": int(sum(1 for r in rows if (r["q_value_bh"] or 1.0) < 0.05)),
            "test": "chi-square 2×2 (thiếu/có dữ liệu × nhãn) trên train+validation+test",
        },
        "co_missing_top": co_missing[:15],
        "identical_missing_pattern_groups": identical_groups,
        "missing_density_per_sample": {
            "mean_features_missing": _round(float(per_sample.mean())) if per_sample.size else None,
            "max_features_missing": int(per_sample.max()) if per_sample.size else None,
            "share_rows_with_gt_50pct_missing": _round(
                float(np.mean(per_sample > 0.5 * n_features))) if per_sample.size else None,
            "n_features": len(names),
        },
    }


# ---------------------------------------------------------------------------
# 4c. Rò rỉ tiềm ẩn: chồng lấn lịch sử, dòng trùng feature, thứ tự thời gian nhãn
# ---------------------------------------------------------------------------
def _history_periods(sample: Dict[str, Any]) -> List[str]:
    """Danh sách kỳ kết thúc (`period_end`) của các quý trong lịch sử mẫu."""
    return [str(row.get("period_end")) for row in sample["request"]["history"]]


def history_overlap_analysis(samples_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Đo mức CHỒNG LẤN lịch sử giữa các mẫu — nguyên nhân khiến 324 mẫu ≠ 324 quan sát độc lập.

    Hai phép đo:
    1. Trong cùng công ty: hệ số Jaccard giữa lịch sử của hai mẫu liên tiếp (cùng cửa sổ 8 quý
       ⇒ phần lớn dòng lịch sử dùng chung, hiệu dụng mẫu nhỏ hơn nhiều).
    2. Giữa train và test: số kỳ lịch sử của test đã xuất hiện trong lịch sử train, và số kỳ
       target của tập này xuất hiện trong lịch sử tập kia (rò rỉ mục tiêu chéo).
    """
    pooled = [s for name in SPLITS for s in samples_by_split.get(name, [])]
    jaccards: List[float] = []
    for ticker in sorted({str(s["ticker"]) for s in pooled}):
        ordered = sorted((s for s in pooled if str(s["ticker"]) == ticker), key=_target_end)
        for first, second in zip(ordered, ordered[1:]):
            set_a, set_b = set(_history_periods(first)), set(_history_periods(second))
            union = len(set_a | set_b)
            if union:
                jaccards.append(len(set_a & set_b) / union)

    def period_keys(split: str) -> set:
        return {(str(s["ticker"]), period) for s in samples_by_split.get(split, [])
                for period in _history_periods(s)}

    def target_keys(split: str) -> set:
        return {(str(s["ticker"]), _target_end(s)) for s in samples_by_split.get(split, [])}

    train_periods, test_periods = period_keys(REFERENCE_SPLIT), period_keys(COMPARISON_SPLIT)
    shared = train_periods & test_periods
    return {
        "consecutive_samples_same_ticker": {
            "n_pairs": len(jaccards),
            "mean_history_jaccard": _round(float(np.mean(jaccards))) if jaccards else None,
            "median_history_jaccard": _round(float(np.median(jaccards))) if jaccards else None,
            "min_history_jaccard": _round(float(np.min(jaccards))) if jaccards else None,
        },
        f"history_periods_{REFERENCE_SPLIT}_vs_{COMPARISON_SPLIT}": {
            "n_shared_periods": len(shared),
            "share_of_test_history_periods_in_train_pct": _round(
                100.0 * len(shared) / max(1, len(test_periods))),
        },
        "cross_split_target_leak": {
            "n_test_targets_seen_in_train_history":
                len(target_keys(COMPARISON_SPLIT) & period_keys(REFERENCE_SPLIT)),
            "n_train_targets_seen_in_test_history":
                len(target_keys(REFERENCE_SPLIT) & period_keys(COMPARISON_SPLIT)),
            "n_test_samples": len(samples_by_split.get(COMPARISON_SPLIT, [])),
        },
        "note": ("Jaccard cao ⇒ các mẫu liên tiếp chia sẻ phần lớn lịch sử: số quan sát ĐỘC LẬP "
                 "nhỏ hơn số mẫu, và khoảng tin cậy của metric cần đọc theo nhóm công ty."),
    }


def _row_keys(X: np.ndarray, decimals: int = 6) -> List[Tuple[Any, ...]]:
    """Khoá so trùng một dòng feature (NaN thay bằng `None` để so sánh được)."""
    keys: List[Tuple[Any, ...]] = []
    for row in np.asarray(X, dtype=float):
        keys.append(tuple(None if not math.isfinite(v) else round(float(v), decimals) for v in row))
    return keys


def duplicate_feature_rows(X_ref: np.ndarray, X_new: np.ndarray, ref_name: str = REFERENCE_SPLIT,
                           new_name: str = COMPARISON_SPLIT, decimals: int = 6) -> Dict[str, Any]:
    """Đếm dòng feature trùng khít giữa hai split (dấu hiệu bản ghi lặp/rò rỉ cơ học)."""
    ref_keys = _row_keys(X_ref, decimals)
    new_keys = _row_keys(X_new, decimals)
    ref_set = set(ref_keys)
    return {
        "ref": ref_name, "new": new_name,
        "n_ref_rows": len(ref_keys), "n_new_rows": len(new_keys),
        "n_new_rows_duplicated_in_ref": int(sum(1 for key in new_keys if key in ref_set)),
        "n_duplicate_rows_within_ref": int(len(ref_keys) - len(ref_set)),
        "n_duplicate_rows_within_new": int(len(new_keys) - len(set(new_keys))),
    }


def label_availability_analysis(samples_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Kiểm tra thứ tự thời gian của nhãn: nhãn phải có SAU khi kỳ target kết thúc."""
    bad = 0
    gaps: List[int] = []
    total = 0
    import datetime as _dt

    for name in SPLITS:
        for sample in samples_by_split.get(name, []):
            available, target_end = sample.get("label_available_on"), _target_end(sample)
            if not available:
                continue
            total += 1
            try:
                first = _dt.date.fromisoformat(str(available)[:10])
                second = _dt.date.fromisoformat(str(target_end)[:10])
            except ValueError:
                continue
            gaps.append((first - second).days)
            if first <= second:
                bad += 1
    return {
        "n_samples_checked": total,
        "n_label_available_before_target_end": bad,
        "min_gap_days": int(min(gaps)) if gaps else None,
        "median_gap_days": int(np.median(gaps)) if gaps else None,
    }


# ---------------------------------------------------------------------------
# 5. Hình minh hoạ (matplotlib Agg; CLI ở `scripts/eda_deep.py`)
# ---------------------------------------------------------------------------
def fig_missing_pct_by_split(missing_by_split: Dict[str, Dict[str, Any]], path: Any) -> None:
    """Hình 1 — heatmap % thiếu theo [feature × split], sắp theo mức thiếu của train."""
    splits = [s for s in FEATURE_SPLITS if s in missing_by_split]
    features = sorted(missing_by_split[splits[0]], key=lambda f: -(missing_by_split[splits[0]][f] or 0))
    matrix = np.array([[missing_by_split[s].get(f) or 0.0 for s in splits] for f in features])
    fig, ax = plt.subplots(figsize=(4.2, 11))
    ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0, vmax=100)
    ax.set_xticks(range(len(splits)), splits, rotation=30, ha="right")
    ax.set_yticks(range(len(features)), features, fontsize=7)
    for i in range(len(features)):
        for j in range(len(splits)):
            if matrix[i, j] > 0:
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] > 55 else "black")
    ax.set_title("Giá trị thiếu theo feature × split (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_label_by_ticker(label: Dict[str, Any], path: Any) -> None:
    """Hình 2 — tỉ lệ nhãn 1 theo công ty (kèm IR và số lần nhãn đổi trạng thái)."""
    per_ticker = label.get("per_ticker", {})
    tickers = sorted(per_ticker, key=lambda t: -(per_ticker[t].get("positive_pct") or 0))
    values = [(per_ticker[t].get("positive_pct") or 0.0) for t in tickers]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(tickers[::-1], values[::-1], color="crimson", alpha=0.8)
    for bar, ticker in zip(bars, tickers[::-1]):
        record = per_ticker[ticker]
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"IR={record.get('imbalance_ratio')} | đổi nhãn {record.get('n_switches')} lần",
                va="center", fontsize=7)
    ax.axvline(50, color="k", ls="--", lw=1, label="cân bằng 50%")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Tỉ lệ mẫu có nhãn = 1 (%)")
    ax.set_title("Mất cân bằng ở CẤP CÔNG TY (nhãn là thuộc tính thực thể)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_target_association(association: Dict[str, Any], path: Any, top: int = 20) -> None:
    """Hình 3 — top feature tách hai lớp mạnh nhất (|2·AUC−1|), kèm mutual information."""
    ranked = association.get("ranked_by_effect", [])[:top]
    names = [str(r["feature"]) for r in ranked][::-1]
    effects = [(r["effect_rank_biserial"] or 0.0) for r in ranked][::-1]
    mis = [(r["mutual_information"] or 0.0) for r in ranked][::-1]
    colors = ["crimson" if e > 0 else "steelblue" for e in effects]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    bars = ax.barh(names, effects, color=colors, alpha=0.85)
    for bar, mi in zip(bars, mis):
        ax.text(bar.get_width() + (0.01 if bar.get_width() >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2, f"MI={mi:.3f}",
                va="center", ha="left" if bar.get_width() >= 0 else "right", fontsize=7)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("2·AUC − 1 (đỏ: giá trị cao ⇒ nhãn 1; xanh: giá trị thấp ⇒ nhãn 1)")
    ax.set_title(f"Liên hệ feature ↔ nhãn (train+validation, top {top})", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_correlation(matrix: Sequence[Sequence[Optional[float]]], names: Sequence[str],
                    clusters: Sequence[Sequence[str]], path: Any) -> None:
    """Hình 4 — heatmap tương quan Pearson, sắp theo cụm đa cộng tuyến |r| ≥ ngưỡng."""
    array = np.array([[np.nan if v is None else float(v) for v in row] for row in matrix])
    order: List[str] = []
    for cluster in clusters:
        order.extend(cluster)
    order.extend([str(n) for n in names if str(n) not in order])
    index = [list(map(str, names)).index(n) for n in order]
    ordered = array[np.ix_(index, index)]
    fig, ax = plt.subplots(figsize=(12, 10.5))
    image = ax.imshow(ordered, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), order, fontsize=6)
    for cluster in clusters:
        positions = [order.index(n) for n in cluster]
        if positions:
            ax.add_patch(plt.Rectangle((min(positions) - 0.5, min(positions) - 0.5),
                                       len(positions), len(positions), fill=False,
                                       edgecolor="black", lw=1))
    ax.set_title("Tương quan Pearson giữa các feature (ô vuông = cụm |r| ≥ 0.90)", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_drift_ks_vs_smd(drift: Dict[str, Any], path: Any) -> None:
    """Hình 5 — dịch chuyển train → test: KS (toàn phân phối) vs SMD (dịch trung bình)."""
    rows = [r for r in drift.get("per_feature", []) if r.get("ks_statistic") is not None]
    ks = [r["ks_statistic"] for r in rows]
    smd = [(r["smd"] or 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.scatter(smd, ks, s=70, alpha=0.75, color="slateblue", edgecolor="white")
    for row, x_value, y_value in zip(rows, smd, ks):
        if y_value >= DRIFT_KS_WARN or abs(x_value) >= DRIFT_SMD_WARN:
            ax.annotate(str(row["feature"]), (x_value, y_value), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    for boundary in (DRIFT_SMD_WARN, -DRIFT_SMD_WARN):
        ax.axvline(boundary, color="crimson", ls="--", lw=1)
    ax.axhline(DRIFT_KS_WARN, color="crimson", ls="--", lw=1)
    ax.set_xlabel("SMD (test − train, theo SD gộp)")
    ax.set_ylabel("KS statistic")
    ax.set_title("Dịch chuyển phân phối train → test (đỏ = ngưỡng cảnh báo)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_top_feature_ecdf(X: np.ndarray, y: Sequence[int], names: Sequence[str],
                         ranked: Sequence[Dict[str, Any]], path: Any, top: int = 4) -> None:
    """Hình 6 — ECDF của top feature theo nhãn: thấy mức chồng lấn, không chỉ trung bình."""
    labels = np.asarray([int(v) for v in y], dtype=int)
    picked = [r for r in ranked if r.get("auc") is not None][:top]
    fig, axes = plt.subplots(1, max(1, len(picked)), figsize=(3.4 * max(1, len(picked)), 3.6))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, picked):
        j = list(map(str, names)).index(str(row["feature"]))
        column = np.asarray(X[:, j], dtype=float)
        for value, color, label in ((0, "steelblue", "không distress"), (1, "crimson", "distress")):
            subset = np.sort(_finite(column[labels == value]))
            if subset.size:
                ax.plot(subset, np.arange(1, subset.size + 1) / subset.size, color=color, lw=1.4,
                        label=label)
        ax.set_title(f"{row['feature']}\nAUC={row['auc']:.3f}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("giá trị", fontsize=8)
    axes[0].set_ylabel("ECDF", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Phân bố (ECDF) của các feature tách lớp mạnh nhất — theo nhãn", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_missingness_information(missingness: Dict[str, Any], path: Any, top: int = 12) -> None:
    """Hình 7 — chênh tỉ lệ nhãn 1 giữa nhóm THIẾU và nhóm CÓ dữ liệu (cảnh báo MNAR)."""
    ranked = [r for r in missingness.get("label_informativeness", {}).get("ranked_by_abs_delta", [])
              if r.get("delta_pct_points") is not None][:top]
    names = [str(r["feature"]) for r in ranked][::-1]
    deltas = [float(r["delta_pct_points"]) for r in ranked][::-1]
    colors = ["crimson" if d > 0 else "steelblue" for d in deltas]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.barh(names, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Δ tỉ lệ nhãn 1 (thiếu − có dữ liệu), điểm %")
    ax.set_title("Giá trị thiếu có mang thông tin nhãn? (chi-square + BH-FDR)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Kết luận tự động + báo cáo markdown
# ---------------------------------------------------------------------------
def auto_conclusions(summary: Dict[str, Any]) -> List[str]:
    """Sinh danh sách kết luận/cảnh báo TRỰC TIẾP từ số liệu (không viết tay)."""
    label = summary["label"]
    overall = label["overall"]
    flags = summary["feature_quality"]["flags"]
    transitions = label["transitions"]
    entities = label["entities"]
    drift = summary["drift"]
    leakage = summary["leakage"]
    missing = summary["missingness"]
    lines: List[str] = []

    lines.append(
        f"**Mức mất cân bằng (cấp mẫu):** entropy nhãn {overall['entropy_bits']} bit "
        f"(tối đa 1.0), IR = {overall['imbalance_ratio']}, lớp thiểu số {overall['minority_pct']}% "
        f"⇒ {overall['level']}; đoán lớp đa số đã đạt {overall['majority_baseline_accuracy_pct']}% "
        f"⇒ không dùng Accuracy làm thước đo chính.")
    lines.append(
        f"**Nhãn gần như là thuộc tính thực thể:** {transitions['persistence_pct']}% cặp quý liên tiếp "
        f"giữ nguyên nhãn; P(nhãn=1 | quý trước nhãn=1) = {transitions['p_one_given_one']}, "
        f"P(nhãn=1 | quý trước nhãn=0) = {transitions['p_one_given_zero']}; "
        f"{entities['share_samples_in_single_class_tickers_pct']}% mẫu thuộc công ty chỉ có MỘT lớp.")
    if entities["tickers_all_one"] or entities["tickers_all_zero"]:
        lines.append(
            f"**Không được chia tập ngẫu nhiên theo mẫu:** công ty nhãn đơn lớp = "
            f"{', '.join(entities['tickers_all_one'] + entities['tickers_all_zero'])} ⇒ mọi đánh giá "
            f"phải chia theo nhóm (`GroupKFold`/LOCO) và nêu IR theo công ty.")
    if flags["constant"] or flags["near_constant"]:
        lines.append(
            f"**Feature vô dụng/giả tín hiệu:** hằng số = "
            f"{', '.join(flags['constant']) or '—'}; gần hằng số (>95% một giá trị) = "
            f"{', '.join(flags['near_constant']) or '—'} ⇒ cân nhắc loại.")
    if flags["missing_gt_20pct"]:
        lines.append(
            f"**Feature bị impute chi phối** (thiếu > {MISSING_WARN_PCT:.0f}%): "
            f"{', '.join(flags['missing_gt_20pct'][:12])} ⇒ đọc kèm kiểm định missingness-mang-nhãn.")
    if flags["heavy_tail"]:
        lines.append(
            f"**Đuôi nặng/lệch mạnh** (|skew| > {SKEW_WARN:.0f} hoặc kurtosis > {KURTOSIS_WARN:.0f}): "
            f"{', '.join(flags['heavy_tail'][:12])} ⇒ nên thêm winsorize/clip và dùng metric xếp hạng.")
    if flags["many_outliers"]:
        lines.append(
            f"**Nhiều outlier theo IQR** (> {OUTLIER_WARN_PCT:.0f}% mẫu): "
            f"{', '.join(flags['many_outliers'][:12])}.")
    lines.append(
        f"**Liên hệ feature ↔ nhãn:** {summary['feature_vs_label']['n_usable_features']}/"
        f"{summary['feature_quality']['n_features']} feature đủ mẫu để tính AUC 1-feature; "
        f"{summary['feature_vs_label']['n_effect_ge_0_30']} feature có |2·AUC−1| ≥ 0.30; "
        f"{summary['feature_vs_label']['n_significant_after_bh_5pct']} feature còn ý nghĩa sau BH-FDR.")
    lines.append(
        f"**Đa cộng tuyến:** {summary['feature_vs_feature']['n_pairs_abs_ge_threshold']} cặp |r| ≥ "
        f"{summary['feature_vs_feature']['threshold']}, gom thành "
        f"{len(summary['feature_vs_feature']['clusters_abs_ge_threshold'])} cụm; "
        f"số chiều hiệu dụng = "
        f"{summary['feature_vs_feature']['effective_dimensionality']['participation_ratio']} "
        f"(trên {summary['feature_quality']['n_features']} cột).")
    lines.append(
        f"**Dịch chuyển train → test:** {drift['n_drifted']} feature vượt ngưỡng (KS ≥ {DRIFT_KS_WARN} "
        f"hoặc |SMD| ≥ {DRIFT_SMD_WARN}), {drift['n_psi_above_0_2']} feature có PSI > 0.2.")
    lines.append(
        f"**Mẫu không độc lập:** Jaccard lịch sử trung bình giữa hai mẫu liên tiếp cùng công ty = "
        f"{leakage['history_overlap']['consecutive_samples_same_ticker']['mean_history_jaccard']} "
        f"⇒ số quan sát độc lập nhỏ hơn số mẫu; "
        f"{leakage['duplicates']['n_new_rows_duplicated_in_ref']} dòng test trùng khít feature với train.")
    lines.append(
        f"**Kiểm tra thứ tự nhãn:** {leakage['label_availability']['n_label_available_before_target_end']}"
        f" mẫu có `label_available_on` ≤ ngày kết thúc kỳ target (kỳ vọng 0); khoảng cách trung vị "
        f"{leakage['label_availability']['median_gap_days']} ngày.")
    informative = missing["label_informativeness"]["n_significant_after_bh_5pct"]
    lines.append(
        f"**Missingness:** {informative} feature có tỉ lệ nhãn khác biệt có ý nghĩa giữa nhóm thiếu và "
        f"nhóm có dữ liệu (BH-FDR < 5%) ⇒ dữ liệu thiếu theo cơ chế MNAR, median-impute một mình sẽ "
        f"xoá tín hiệu; đề xuất thêm cờ `is_missing` khi tái huấn luyện.")
    return lines


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           aligns: Sequence[str] | None = None) -> str:
    """Bảng markdown đơn giản (thay cho `DataFrame.to_markdown` khi không có pandas)."""
    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:g}"
        return str(value)

    separator = list(aligns) if aligns else ["---"] * len(headers)
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "|" + "|".join(separator) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _top(stats: Sequence[Dict[str, Any]], key: str, count: int, reverse: bool = True
         ) -> List[Dict[str, Any]]:
    """Lấy `count` bản ghi có `key` lớn nhất (bỏ bản ghi thiếu giá trị)."""
    available = [r for r in stats if r.get(key) is not None]
    return sorted(available, key=lambda r: r[key], reverse=reverse)[:count]


def outlier_driven_pairs(correlation: Dict[str, Any],
                         gap: float = 0.30) -> List[Dict[str, Any]]:
    """Cặp biến có tương quan Pearson lệch xa Spearman ⇒ tương quan chủ yếu do NGOẠI LAI.

    Vì sao cần: Pearson nhạy với giá trị cực trị, Spearman thì không. Nếu hai hệ số chênh nhau
    nhiều thì kết luận "hai biến liên quan" là hệ quả của vài quan sát dị biệt, không phải quy luật.
    Xét **mọi cặp** (không chỉ các cặp mạnh nhất) để không bỏ sót tương quan giả.
    """
    pearson = correlation.get("pearson_matrix")
    spearman = correlation.get("spearman_matrix")
    order = [str(n) for n in (correlation.get("feature_order") or [])]
    if not pearson or not spearman or not order:
        return []
    pairs: List[Dict[str, Any]] = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            first, second = pearson[i][j], spearman[i][j]
            if first is None or second is None:
                continue
            if abs(float(first) - float(second)) > gap:
                pairs.append({"a": order[i], "b": order[j], "pearson": _round(first, 4),
                              "spearman": _round(second, 4),
                              "gap": _round(abs(float(first) - float(second)), 4)})
    return sorted(pairs, key=lambda p: -p["gap"])


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
                   aligns: Sequence[str] | None = None) -> str:
    """Bảng markdown (API công khai — dùng chung cho `scripts/eda.py` và báo cáo tự động)."""
    return _table(headers, rows, aligns)


def markdown_eda_deep(summary: Dict[str, Any]) -> str:
    """Sinh `reports/results/eda_deep.md` từ summary — mọi số đều lấy từ JSON, không nhập tay."""
    scope, label = summary["scope"], summary["label"]
    stats, flags = summary["feature_quality"]["stats"], summary["feature_quality"]["flags"]
    association = summary["feature_vs_label"]
    corr = summary["feature_vs_feature"]
    drift, missing = summary["drift"], summary["missingness"]
    leakage = summary["leakage"]

    lines = ["# EDA chuyên sâu — feature, nhãn, tương quan, dịch chuyển", "",
             "*Sinh tự động bởi `python -m scripts.eda_deep` (module: `forecasting/eda.py`). "
             "Không có số nào nhập tay.*", "",
             "## 0. Phạm vi", "",
             f"- Mẫu: " + ", ".join(f"{k}={v}" for k, v in scope["n_samples"].items()),
             f"- Feature: **{scope['n_features']}** cột (từ `forecasting.features.feature_names()`).",
             f"- Liên hệ feature–nhãn và tương quan được tính trên **{scope['association_eval_set']}**; "
             f"test chỉ dùng cho chẩn đoán dịch chuyển (train → test).",
             "",
             "## 1. Chất lượng ma trận feature", "",
             f"- Feature hằng số: **{', '.join(flags['constant']) or '—'}**; gần hằng số (>95% một giá "
             f"trị): **{', '.join(flags['near_constant']) or '—'}**.",
             f"- Thiếu > {MISSING_WARN_PCT:.0f}%: **{len(flags['missing_gt_20pct'])}** feature; "
             f"đuôi nặng: **{len(flags['heavy_tail'])}**; nhiều outlier IQR: "
             f"**{len(flags['many_outliers'])}**.",
             ""]
    lines.append(_table(
        ["Feature", "Thiếu %", "#giá trị", "Median", "P1–P99", "Skew", "Kurtosis", "Outlier IQR %", "= 0 %"],
        [[r["feature"], r.get("missing_pct"), r.get("n_unique"), r.get("median"),
          f"{r.get('p1')} … {r.get('p99')}", r.get("skew"), r.get("kurtosis_excess"),
          r.get("iqr_outlier_pct"), r.get("zero_pct")]
         for r in _top(stats, "missing_pct", 15)],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    lines += [f"- {r['feature']}: skew={r.get('skew')}, kurtosis={r.get('kurtosis_excess')}, "
              f"outlier IQR={r.get('iqr_outlier_pct')}%"
              for r in _top(stats, "kurtosis_excess", 5)]
    lines += ["", "![Thiếu theo feature × split]"
              "(../reports/figures/eda_deep/01_missing_pct_by_split.png)", "",
              "![Missingness mang thông tin nhãn]"
              "(../reports/figures/eda_deep/07_missingness_information.png)", ""]

    lines += ["## 2. Nhãn: mất cân bằng, entropy, chuỗi trạng thái", "",
              _table(["Tập", "n", "Dương", "Tỉ lệ dương %", "IR", "Entropy (bit)",
                      "Entropy chuẩn hoá", "Gini", "Baseline đa số %", "Mức"],
                     [[name, r.get("n"), r.get("positive"), r.get("positive_pct"),
                       r.get("imbalance_ratio"), r.get("entropy_bits"), r.get("entropy_normalized"),
                       r.get("gini"), r.get("majority_baseline_accuracy_pct"), r.get("level")]
                      for name, r in label["per_split"].items()],
                     ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
              "",
              _table(["Công ty", "n", "Tỉ lệ dương %", "IR", "Mức", "Lần đổi nhãn",
                      "Chuỗi 1 dài nhất", "Chuỗi 0 dài nhất"],
                     [[t, r.get("n"), r.get("positive_pct"), r.get("imbalance_ratio"), r.get("level"),
                       r.get("n_switches"), r.get("max_run_one"), r.get("max_run_zero")]
                      for t, r in sorted(label["per_ticker"].items(),
                                         key=lambda kv: -(kv[1].get("positive_pct") or 0))],
                     ["---", "---:", "---:", "---:", "---", "---:", "---:", "---:"]),
              "",
              f"- Chuyển trạng thái giữa hai quý liên tiếp: `{label['transitions']['counts']}`; "
              f"P(1|1) = {label['transitions']['p_one_given_one']}, "
              f"P(1|0) = {label['transitions']['p_one_given_zero']}, "
              f"giữ nguyên nhãn = **{label['transitions']['persistence_pct']}%**.",
              f"- Thực thể: **{label['entities']['n_tickers']}** công ty, số thực thể hiệu dụng "
              f"= **{label['entities']['effective_number_of_tickers']}**; "
              f"{label['entities']['share_samples_in_single_class_tickers_pct']}% mẫu thuộc công ty "
              f"chỉ có một lớp.",
              f"- Mức mất cân bằng nặng nhất: " + ", ".join(
                  f"{r['ticker']} (IR={r['imbalance_ratio']})"
                  for r in label["entities"]["worst_imbalance"]) + ".",
              "",
              "![Tỉ lệ nhãn theo công ty]"
              "(../reports/figures/eda_deep/02_label_by_ticker.png)", ""]

    lines += ["## 3. Quan hệ feature ↔ nhãn", "",
              f"- {association['n_usable_features']}/{scope['n_features']} feature tính được AUC "
              f"1-feature; {association['n_effect_ge_0_30']} feature có |2·AUC−1| ≥ 0.30; "
              f"{association['n_significant_after_bh_5pct']} feature còn ý nghĩa sau BH-FDR.",
              f"- Lưu ý: {association['multiple_testing_note']}", "",
              _table(["Feature", "#cặp", "AUC", "Hướng", "r (point-biserial)", "q-value BH",
                      "MI", "Lift decile trên"],
                     [[r["feature"], r.get("n_pairs"), r.get("auc"), r.get("direction"),
                       r.get("point_biserial_r"), r.get("q_value_bh"), r.get("mutual_information"),
                       r.get("lift_top_decile")]
                      for r in association["ranked_by_effect"][:15]],
                     ["---", "---:", "---:", "---", "---:", "---:", "---:", "---:"]),
              "",
              "![Liên hệ feature ↔ nhãn]"
              "(../reports/figures/eda_deep/03_target_association.png)",
              "",
              "![ECDF top feature theo nhãn]"
              "(../reports/figures/eda_deep/06_top_feature_ecdf.png)", ""]

    dims = corr["effective_dimensionality"]
    lines += ["## 4. Quan hệ feature ↔ feature (đa cộng tuyến)", "",
              f"- **{corr['n_pairs_abs_ge_threshold']}** cặp có |r| ≥ {corr['threshold']}; "
              f"**{corr['n_features_in_clusters']}** feature nằm trong "
              f"{len(corr['clusters_abs_ge_threshold'])} cụm thông tin.",
              f"- Số chiều hiệu dụng (participation ratio) = **{dims.get('participation_ratio')}** "
              f"trên {dims.get('n_columns')} cột; cần "
              f"**{dims.get('n_components_for_95pct_variance')}** thành phần cho 95% phương sai.",
              "", f"**Cụm mạnh nhất (|r| ≥ {corr['threshold']}):**", "",
              _table(["#", "Cụm feature", "Số cột"],
                     [[index + 1, ", ".join(cluster), len(cluster)]
                      for index, cluster in enumerate(corr["clusters_abs_ge_threshold"][:10])],
                     ["---:", "---", "---:"]),
              "",
              _table(["Feature A", "Feature B", "Pearson", "Spearman"],
                     [[p["a"], p["b"], p["pearson"], p["spearman"]]
                      for p in corr["strongest_pairs"][:10]],
                     ["---", "---", "---:", "---:"]),
              "- Đọc cột Spearman để biết tương quan có bị outlier chi phối không.",
              "",
              "![Heatmap tương quan theo cụm]"
              "(../reports/figures/eda_deep/04_correlation_clustered.png)", ""]

    lines += ["## 5. Dịch chuyển phân phối train → test", "",
              f"- **{drift['n_drifted']}** feature vượt ngưỡng cảnh báo "
              f"(KS ≥ {DRIFT_KS_WARN} hoặc |SMD| ≥ {DRIFT_SMD_WARN}); "
              f"{drift['n_psi_above_0_2']} feature có PSI > 0.2; "
              f"{drift['n_significant_after_bh_5pct']} feature khác biệt có ý nghĩa sau BH-FDR.",
              f"- Cảnh báo đọc số: PSI chia 10 khoảng trên mẫu test chỉ {scope['n_samples'].get(COMPARISON_SPLIT)} "
              f"dòng nên mỗi khoảng ~{max(1, scope['n_samples'].get(COMPARISON_SPLIT, 1) // 10)} quan sát "
              f"⇒ PSI nhạy nhiễu, dùng KS/SMD làm tiêu chí chính, PSI để theo dõi về sau.",
              "",
              _table(["Feature", "Mean train", "Mean test", "KS", "KS p-value", "q-value BH",
                      "SMD", "PSI"],
                     [[r["feature"], r.get("mean_ref"), r.get("mean_new"), r.get("ks_statistic"),
                       r.get("ks_p_value"), r.get("q_value_bh"), r.get("smd"), r.get("psi")]
                      for r in drift["ranked_by_ks"][:15]],
                     ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "![Dịch chuyển train → test]"
              "(../reports/figures/eda_deep/05_drift_ks_vs_smd.png)", ""]

    overlap = leakage["history_overlap"]
    consecutive = overlap["consecutive_samples_same_ticker"]
    duplicates = leakage["duplicates"]
    availability = leakage["label_availability"]
    density = missing["missing_density_per_sample"]
    split_history_key = f"history_periods_{REFERENCE_SPLIT}_vs_{COMPARISON_SPLIT}"
    lines += ["## 6. Rò rỉ tiềm ẩn & tính độc lập của mẫu", "",
              f"- Hai mẫu liên tiếp cùng công ty chia sẻ lịch sử (Jaccard): mean "
              f"**{consecutive['mean_history_jaccard']}**, median {consecutive['median_history_jaccard']}, "
              f"min {consecutive['min_history_jaccard']} trên {consecutive['n_pairs']} cặp.",
              f"- Kỳ lịch sử của test đã có trong train: "
              f"**{overlap[split_history_key]['share_of_test_history_periods_in_train_pct']}%**; "
              f"kỳ target của test xuất hiện trong lịch sử train: "
              f"**{overlap['cross_split_target_leak']['n_test_targets_seen_in_train_history']}** mẫu.",
              f"- Dòng feature trùng khít test↔train: **{duplicates['n_new_rows_duplicated_in_ref']}**; "
              f"trùng trong nội bộ train: {duplicates['n_duplicate_rows_within_ref']}.",
              f"- Thứ tự thời gian nhãn: {availability['n_label_available_before_target_end']} mẫu có "
              f"nhãn công bố TRƯỚC khi kỳ target kết thúc (kỳ vọng 0); khoảng cách trung vị "
              f"{availability['median_gap_days']} ngày.",
              f"- Mật độ thiếu mỗi mẫu: trung bình {density['mean_features_missing']}/"
              f"{density['n_features']} feature, cao nhất {density['max_features_missing']}; "
              f"tỉ lệ mẫu thiếu > 50% feature = {density['share_rows_with_gt_50pct_missing']}.",
              "",
              "**Feature thiếu mang thông tin nhãn (top 10 theo |Δ|):**", "",
              _table(["Feature", "#thiếu", "Tỉ lệ nhãn 1 khi THIẾU %", "khi CÓ %", "Δ điểm %",
                      "q-value BH"],
                     [[r["feature"], r.get("n_missing"), r.get("target_rate_when_missing_pct"),
                       r.get("target_rate_when_present_pct"), r.get("delta_pct_points"),
                       r.get("q_value_bh")]
                      for r in missing["label_informativeness"]["ranked_by_abs_delta"][:10]],
                     ["---", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "**Cặp feature thường thiếu cùng nhau (top 5):**", "",
              _table(["Feature A", "Feature B", "#thiếu cùng", "Jaccard"],
                     [[r["a"], r["b"], r["n_both_missing"], r["jaccard"]]
                      for r in missing["co_missing_top"][:5]],
                     ["---", "---", "---:", "---:"]), ""]

    lines += ["## 7. Kết luận tự động", ""]
    lines += [f"{index + 1}. {text}" for index, text in enumerate(summary["auto_conclusions"])]
    lines += ["", "## 8. Hình", ""]
    lines += [f"- `{path}`" for path in summary["figures"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 7. Điều phối: chạy toàn bộ EDA chuyên sâu + ghi artifact
# ---------------------------------------------------------------------------
def run(write: bool = True, out_dir: Any = None, fig_dir: Any = None,
        figures: bool = True) -> Dict[str, Any]:
    """Chạy EDA chuyên sâu, ghi `eda_deep.{json,md}` (+ 7 hình) và trả summary.

    `out_dir` / `fig_dir` cho phép test trỏ vào thư mục tạm thay vì `reports/`.
    Liên hệ feature–nhãn và tương quan dùng **train+validation** (không chạm test); test chỉ
    xuất hiện trong phép so dịch chuyển phân phối.
    """
    ensure_dirs()
    ensure_utf8_stdio()  # hàm này cũng được gọi trực tiếp từ test/notebook ⇒ đừng phụ thuộc CLI
    out = Path(out_dir) if out_dir else RESULTS_DIR
    figs = Path(fig_dir) if fig_dir else (FIGURES_DIR / "eda_deep")
    out.mkdir(parents=True, exist_ok=True)

    samples_by_split = {name: load_prepared(name) for name in SPLITS}
    names = [str(n) for n in feature_names()]
    X_by_split = {name: build_feature_matrix(samples_by_split[name]) for name in FEATURE_SPLITS}
    y_by_split = {name: extract_labels(samples_by_split[name]) for name in FEATURE_SPLITS}
    X_analysis = np.vstack([X_by_split["train"], X_by_split["validation"]])
    y_analysis = np.concatenate([y_by_split["train"], y_by_split["validation"]])

    label = label_deep_dive(samples_by_split)
    stats = feature_statistics(X_analysis, names)
    association = target_association(X_analysis, y_analysis, names)
    correlation = correlation_analysis(X_analysis, names)
    drift = drift_analysis(X_by_split[REFERENCE_SPLIT], X_by_split[COMPARISON_SPLIT], names)
    missingness = missingness_analysis(X_by_split, y_by_split, names)
    overlap = history_overlap_analysis(samples_by_split)

    figure_paths: List[str] = []
    if figures:
        figs.mkdir(parents=True, exist_ok=True)
        drawings = [
            ("01_missing_pct_by_split.png",
             lambda path: fig_missing_pct_by_split(missingness["missing_pct_by_split"], path)),
            ("02_label_by_ticker.png", lambda path: fig_label_by_ticker(label, path)),
            ("03_target_association.png", lambda path: fig_target_association(association, path)),
            ("04_correlation_clustered.png",
             lambda path: fig_correlation(correlation["pearson_matrix"], names,
                                          correlation["clusters_abs_ge_threshold"], path)),
            ("05_drift_ks_vs_smd.png", lambda path: fig_drift_ks_vs_smd(drift, path)),
            ("06_top_feature_ecdf.png",
             lambda path: fig_top_feature_ecdf(X_analysis, y_analysis, names,
                                               association["ranked_by_effect"], path)),
            ("07_missingness_information.png",
             lambda path: fig_missingness_information(missingness, path)),
        ]
        for filename, draw in drawings:
            path = figs / filename
            draw(path)
            try:
                figure_paths.append(str(path.relative_to(FIGURES_DIR.parent)))
            except ValueError:  # fig_dir ngoài cây repo (test dùng thư mục tạm)
                figure_paths.append(str(path))

    summary: Dict[str, Any] = {
        "generated_by": "forecasting.eda (CLI: python -m scripts.eda_deep)",
        "scope": {
            "n_samples": {name: len(samples_by_split[name]) for name in SPLITS},
            "n_features": len(names),
            "feature_names": names,
            "association_eval_set": "train+validation — test không dùng để mô tả liên hệ",
            "reference_split": REFERENCE_SPLIT,
            "comparison_split": COMPARISON_SPLIT,
        },
        "label": label,
        "feature_quality": {"stats": stats, "flags": feature_quality_flags(stats),
                            "n_features": len(names)},
        "feature_vs_label": association,
        "feature_vs_feature": correlation,
        "drift": drift,
        "missingness": missingness,
        "leakage": {"history_overlap": overlap,
                    "duplicates": duplicate_feature_rows(X_by_split[REFERENCE_SPLIT],
                                                         X_by_split[COMPARISON_SPLIT]),
                    "label_availability": label_availability_analysis(samples_by_split)},
        "figures": figure_paths,
    }
    summary["auto_conclusions"] = auto_conclusions(summary)

    if write:
        (out / "eda_deep.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "eda_deep.md").write_text(markdown_eda_deep(summary), encoding="utf-8")

    print(f"EDA chuyên sâu: {summary['feature_quality']['n_features']} feature trên "
          f"{len(X_analysis)} mẫu (train+validation); {drift['n_drifted']} feature dịch chuyển "
          f"train→test; {correlation['n_pairs_abs_ge_threshold']} cặp |r| ≥ "
          f"{correlation['threshold']}; nhãn: {label['overall']['level']} "
          f"(entropy {label['overall']['entropy_bits']} bit)")
    return summary
