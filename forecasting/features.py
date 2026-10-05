"""Compute the 47 per-sample features with numpy only, from the prepared history window.

Features cover latest and year-over-year ratios, YoY growth, capital structure, window extremes and
stress flags, with missing values left as NaN for the downstream imputer. Only history with
`available_on <= as_of` is read, so the target period and label never leak.
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
    """At most LOOKBACK_QUARTERS most-recent quarters of the sample history."""
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
    """Smallest finite value, i.e. the worst extreme of the series."""
    finite = [v for v in values if math.isfinite(v)]
    return min(finite) if finite else float("nan")


def _trailing_streak(values: List[float], predicate) -> float:
    """Count of the most recent consecutive quarters satisfying `predicate` (for example net income < 0)."""
    streak = 0
    for value in reversed(values):
        if math.isfinite(value) and predicate(value):
            streak += 1
        else:
            break
    return float(streak)


def _total_liabilities(window: List[Dict[str, Any]]) -> float:
    """Total liabilities of the latest quarter: the `liabilities` tag, else derived from A = L + E.

    Derivation is required because the `liabilities` tag exists in only 39% of quarters (min 16% per
    company), while `total_assets` and `stockholders_equity` cover 100%. Without it,
    `debt_to_assets_latest` and `debt_to_equity_latest` (2 of the 10 most important features) would be
    NaN for most samples and the model would lose leverage information. The identity A = L + E was
    cross-checked on 124 tagged quarters.
    """
    liabilities = _last(_series(window, "liabilities"))
    if math.isfinite(liabilities):
        return liabilities
    return _last(_series(window, "total_assets")) - _last(_series(window, "stockholders_equity"))


def _resolve(expr: str, window: List[Dict[str, Any]]) -> float:
    """Latest value of one expression: `a`, `a-b`, or `total_liabilities` (derived debt)."""
    if expr.strip() == "total_liabilities":
        return _total_liabilities(window)
    if "-" in expr:
        a, b = expr.split("-", 1)
        return _last(_series(window, a.strip())) - _last(_series(window, b.strip()))
    return _last(_series(window, expr.strip()))


def _ratio_latest(window: List[Dict[str, Any]], numerator: str, denominator: str) -> float:
    """Ratio from the latest quarter; numerator/denominator may be an indicator, a subtraction 'a-b', or `total_liabilities`."""
    return _safe_ratio(_resolve(numerator, window), _resolve(denominator, window))


#: Map ratio name -> (numerator, denominator). Names match config.RATIOS for consistent explanations.
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

#: The `path` group - worst window extreme, drawdown against the revenue peak, negative-quarter streaks.
PATH_FEATURES = [
    "current_ratio_min_window",
    "ocf_to_sales_min_window",
    "net_margin_min_window",
    "revenue_drawdown_window",
    "negative_ni_streak",
    "negative_ocf_streak",
]


def _feature_names() -> List[str]:
    """Stable feature column order (kept in sync with _build_features)."""
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
    """Compute features for one sample, returning a {name: float} dict."""
    window = _window(sample)
    feats: Dict[str, float] = {}
    if not window:
        return feats

    # 1. Fourteen ratios: latest value and YoY against four quarters earlier.
    for name, (num, den) in RATIO_PARTS.items():
        latest = _ratio_latest(window, num, den)
        feats[f"{name}_latest"] = latest
        if len(window) >= 5:
            feats[f"{name}_yoy"] = _yoy(latest, _ratio_latest(window[:-4], num, den))
        else:
            feats[f"{name}_yoy"] = float("nan")

    # 2. YoY growth of the ten core indicators.
    for field in GROWTH_FIELDS:
        series = _series(window, field)
        past = _last(series[:-4]) if len(series) >= 5 else float("nan")
        feats[f"{field}_yoy_growth"] = _yoy(_last(series), past)

    # 3. Capital structure and liquidity. `cash_to_assets_latest` already comes from the ratio loop
    #    above; recomputing it would add a duplicate column and double-count it in feature importance.
    total_assets = _last(_series(window, "total_assets"))
    ca = _last(_series(window, "current_assets"))
    cl = _last(_series(window, "current_liabilities"))
    feats["working_capital_to_assets"] = _safe_ratio(ca - cl, total_assets)

    # 4. Stress signal and revenue volatility.
    ocf = _series(window, "operating_cash_flow")
    feats["distress_quarters_in_window"] = float(
        sum(1 for v in ocf if math.isfinite(v) and v < 0))
    rev = [v for v in _series(window, "revenue") if math.isfinite(v)]
    if len(rev) > 1:
        mean = float(np.mean(rev))
        feats["revenue_cv"] = float(np.std(rev) / mean) if mean != 0 else float("nan")
    else:
        feats["revenue_cv"] = float("nan")

    # 5. The `path` group captures trajectory and persistence inside the window. `scripts/probe_features.py`
    #    measured +0.03 to +0.12 cross-company AUROC over using the latest quarter alone.
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
    """Feature column names in fixed order."""
    return _feature_names()


def build_feature_matrix(samples: List[Dict[str, Any]]) -> np.ndarray:
    """X matrix (n_samples x n_features); NaN for missing values."""
    names = _feature_names()
    rows = np.zeros((len(samples), len(names)), dtype=float)
    for i, sample in enumerate(samples):
        feats = _build_features(sample)
        for j, name in enumerate(names):
            rows[i, j] = feats.get(name, float("nan"))
    return rows


def extract_labels(samples: List[Dict[str, Any]]) -> np.ndarray:
    """Label vector y (int 0/1)."""
    return np.asarray([int(s["is_distressed"]) for s in samples], dtype=int)


#: Feature groups by economic nature, used to ablate which factor matters most.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "ratios_latest": [f"{name}_latest" for name in RATIO_PARTS],
    "ratios_yoy": [f"{name}_yoy" for name in RATIO_PARTS],
    "growth": [f"{field}_yoy_growth" for field in GROWTH_FIELDS],
    "structure": ["working_capital_to_assets"],
    "stress": ["distress_quarters_in_window", "revenue_cv"],
    "path": list(PATH_FEATURES),
}


def feature_groups() -> Dict[str, List[str]]:
    """Map group -> real feature column names (checked against `feature_names()`)."""
    valid = set(_feature_names())
    return {group: [n for n in names if n in valid] for group, names in FEATURE_GROUPS.items()}


def history_length(sample: Dict[str, Any]) -> int:
    """Number of history quarters available for the sample (before as_of)."""
    return len(history_array(sample))


def filter_by_history(samples: List[Dict[str, Any]],
                      min_quarters: int) -> List[Dict[str, Any]]:
    """Keep samples with >= `min_quarters` history quarters (drop too-young rows whose YoY is mostly NaN)."""
    return [s for s in samples if history_length(s) >= min_quarters]


def feature_rows(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Raw feature table (list of dicts) - used for per-sample error analysis/audit."""
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


