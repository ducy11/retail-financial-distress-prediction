"""Deep exploratory analysis of the feature matrix, labels, relationships and drift.

Every function takes plain arrays or dicts so it is independently testable, and the CLI lives in
`scripts/eda_deep.py`. The analysis fits no model and never uses test data to pick a configuration, but
does compare train and test distributions for drift.
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
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before importing pyplot)

#: The four standard splits in prepared.
SPLITS = ("train", "validation", "test", "purged")
#: Splits used for feature statistics (purged is a buffer band, not a learning set).
FEATURE_SPLITS = ("train", "validation", "test")
#: Distribution-drift comparison pair (time series => train is older than test).
REFERENCE_SPLIT = "train"
COMPARISON_SPLIT = "test"

# Warning thresholds used by the automatic conclusions in the report.
#: Minimum sample size after dropping NaN to trust a correlation / single-feature AUC.
MIN_PAIR_N = 30
#: Missing rate above 20% => heavy imputation needed, warn.
MISSING_WARN_PCT = 20.0
#: |skew| > 5 and excess kurtosis > 20 => heavy tails, a ratio can "blow up" on small denominators.
SKEW_WARN = 5.0
KURTOSIS_WARN = 20.0
#: More than 5% of samples outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR] => worth revisiting why.
OUTLIER_WARN_PCT = 5.0
#: KS statistic >= 0.30 or |SMD| >= 0.50 => train/test distributions drift noticeably.
DRIFT_KS_WARN = 0.30
DRIFT_SMD_WARN = 0.50
#: |r| >= 0.90 is treated as the same information cluster (collinearity).
CORR_CLUSTER_THRESHOLD = 0.90
#: Imbalance classification thresholds, kept identical to `scripts/class_balance.py` for consistency.
BALANCED_MIN_MINORITY_PCT = 40.0
SEVERE_MAX_MINORITY_PCT = 20.0


# Shared statistics: entropy, Gini impurity, imbalance ratio and BH-FDR correction.
def shannon_entropy(counts: Sequence[int]) -> float:
    """Shannon entropy (bits): 0 bits = one class, 1 bit = balanced binary."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts if c > 0))


def normalized_entropy(counts: Sequence[int]) -> float:
    """Entropy / log2(number of classes) - comparable across problems with different class counts."""
    classes = sum(1 for c in counts if c > 0)
    if classes <= 1:
        return 0.0
    return float(shannon_entropy(counts) / math.log2(classes))


def gini_impurity(counts: Sequence[int]) -> float:
    """Gini impurity: 0 = pure, 0.5 = balanced binary."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(1.0 - sum((c / total) ** 2 for c in counts))


def imbalance_ratio(counts: Sequence[int]) -> float:
    """IR = majority samples / minority samples (``inf`` when there is only one class)."""
    if len(counts) < 2:
        return float("inf")
    positive, negative = int(counts[0]), int(counts[1])
    if min(positive, negative) == 0:
        return float("inf")
    return max(positive, negative) / min(positive, negative)


def effective_number(counts: Sequence[int]) -> float:
    """Effective number of entities 1/sum(p^2) - 8 balanced companies = 8.0; one company at 90% -> near 1."""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(1.0 / sum((c / total) ** 2 for c in counts if c > 0))


def imbalance_level(minority_pct: float) -> str:
    """Classify imbalance by minority-class percentage (matches `scripts/class_balance.py`)."""
    if minority_pct >= BALANCED_MIN_MINORITY_PCT:
        return "balanced"
    if minority_pct >= SEVERE_MAX_MINORITY_PCT:
        return "slightly_imbalanced"
    return "severely_imbalanced"


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg q-values (keeps positions; `None` for missing values).

    With 47 features under many tests, a raw alpha of 0.05 would yield roughly two purely random
    findings, so q-values control the false discovery rate.
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
    """Drop NaN/inf from a sequence."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _round(value: Any, digits: int = 6) -> Any:
    """Round so JSON/markdown stay stable across runs; NaN/inf -> None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _median_impute(X: np.ndarray) -> np.ndarray:
    """Median-impute for EDA PURPOSES (MI, correlation matrix, PCA).

    Deliberately NOT used for the model: replacement values must be learned inside the Pipeline per
    fold-train, otherwise it is statistical leakage (see `forecasting/models.py`).
    """
    arr = np.array(X, dtype=float, copy=True)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = col[np.isfinite(col)]
        if finite.size:
            col[np.isnan(col)] = float(np.median(finite))
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


# Data-selection helpers for the basic EDA report sections.
#: Conversion factor from VND amounts to "trillion VND" for readable descriptive tables.
VND_PER_TRILLION = 1e12


def latest_ratio_names() -> List[str]:
    """Names of the 14 ratios in "latest quarter value" form (`*_latest`) - comparable across companies."""
    return [f"{name}_latest" for name in RATIO_PARTS]


def latest_ratio_matrix(samples: Sequence[Dict[str, Any]]
                        ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Matrix of the 14 `*_latest` ratios + label vector (columns selected from `feature_names()` by name)."""
    names = latest_ratio_names()
    full = [str(name) for name in feature_names()]
    index = [full.index(name) for name in names]
    return build_feature_matrix(list(samples))[:, index], names, extract_labels(list(samples))


def indicator_value_matrix(rows_by_ticker: Dict[str, List[Dict[str, Any]]],
                           fields: Sequence[str] | None = None,
                           scale: float = VND_PER_TRILLION) -> Tuple[np.ndarray, List[str]]:
    """Matrix of the 16 indicator values (every company x every quarter) in converted units (default trillion VND).

    Used for descriptive statistics of the RAW DATA (different from the model's 47-column feature matrix).
    """
    columns = [str(f) for f in (fields or BASE_FIELDS)]
    rows = [row for rows in rows_by_ticker.values() for row in rows]
    matrix = np.full((len(rows), len(columns)), np.nan)
    for i, row in enumerate(rows):
        for j, field in enumerate(columns):
            value = to_float(row.get(field + SUFFIX))
            matrix[i, j] = value / scale if math.isfinite(value) else np.nan
    return matrix, columns


# Label and class analysis: imbalance, entropy, run length and state transitions.
def _target_end(sample: Dict[str, Any]) -> str:
    """Target-period end date - used to sort by time within each company."""
    return str(sample["request"]["target_period_end"])


def distribution_of(labels: Sequence[int]) -> Dict[str, Any]:
    """Statistics for a 0/1 label sequence: counts, %, IR, entropy, Gini, majority-class baseline."""
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
    """Number of label FLIPS between consecutive quarters (0 => the label is constant per company)."""
    return int(sum(1 for a, b in zip(labels, labels[1:]) if a != b))


def _max_run(labels: Sequence[int], value: int) -> int:
    """Longest run (spell) of one label value - measures how "sticky" the label is over time."""
    best = current = 0
    for v in labels:
        current = current + 1 if int(v) == value else 0
        best = max(best, current)
    return int(best)


def _transition_counts(sequences: Sequence[Sequence[int]]) -> Dict[str, int]:
    """Count label state transitions between consecutive quarters, pooled over all companies."""
    counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[f"{int(a)}->{int(b)}"] += 1
    return counts


def _calendar_quarter(date_text: str) -> str:
    """``"2014-07-31"`` -> ``"Q3"`` (calendar quarter of the target period - seasonality check)."""
    text = str(date_text)
    month = int(text[5:7]) if len(text) >= 7 else 0
    return f"Q{(month - 1) // 3 + 1}" if month else "?"


def label_deep_dive(samples_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Analyze labels along 4 axes: split, company, time, state runs.

    Returns a pure JSON dict: ``per_split``, ``overall``, ``per_ticker``, ``per_year``,
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


# Feature matrix quality: missingness, constant columns, heavy tails and outliers.
def feature_statistics(X: np.ndarray, names: Sequence[str]) -> List[Dict[str, Any]]:
    """Per-column feature stats: missing %, constant/near-constant, quantiles, skew, tails, outliers.

    The 47 columns come from ratio formulas, so an almost-constant column (useless), a column missing
    more than 50% (imputation dominates) and a very heavy-tailed column (`debt_to_equity` when equity is
    small) can all appear.
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
            # A near-constant column is almost useless yet differs from a constant one, so the two flags
            # are mutually exclusive.
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
    """List of features to revisit by problem type (used for automatic conclusions)."""
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


# Feature to label association: point-biserial, single-feature AUC, mutual information, lift, BH-FDR.
def target_association(X: np.ndarray, y: Sequence[int], names: Sequence[str],
                       min_pairs: int = MIN_PAIR_N) -> Dict[str, Any]:
    """Strength of association of EACH feature with the label, with tests and multiplicity correction.

    Returns 4 complementary measures (does not replace permutation importance in `scripts/analyze.py`):
    - ``auc`` / ``effect_rank_biserial`` = 2*AUC - 1: measures class SEPARATION, with direction;
    - ``point_biserial_r`` + ``p_value`` + ``q_value_bh``: statistical significance after correction;
    - ``mutual_information``: catches NON-LINEAR association that Pearson misses;
    - ``lift_top_decile``: positive-label rate in the top 10% of values vs the base rate (business read).
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
                    "direction": "high value => label 1" if auc > 0.5 else "low value => label 1",
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
        "multiple_testing_note": ("47 tests => BH q-values; rank by |2*AUC-1| to avoid depending "
                                 "on sample size, never read a lone p-value."),
    }


# Feature to feature association: correlation, collinear clusters and effective dimensionality.
def pairwise_correlation(X: np.ndarray, method: str = "pearson",
                         min_pairs: int = MIN_PAIR_N) -> np.ndarray:
    """Pairwise correlation over finite values only (pairwise complete case).

    Imputing before correlating would create spurious correlation between columns that are missing
    together, a median artefact, because the 47 columns have very different missing rates.
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
    """Group features into "same information" clusters using union-find over |r| >= threshold."""
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
    """Effective dimensionality of the feature matrix (median-impute + standardize, EDA only).

    - ``participation_ratio`` = (sum lambda)^2 / sum(lambda^2) - much smaller than the column count
      means redundant features.
    - ``n_components_for_95pct_variance`` - components needed to keep 95% of the variance.
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
    """Feature <-> feature correlation: strongest pairs, collinear clusters, effective dimensionality.

    Compare Pearson (linear) with Spearman (rank) to know whether outliers dominate the relation:
    a large gap between the two on the same pair flags heavy tails.
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
        "note": ("Pairwise correlation uses shared finite values (no imputation), so each pair has a "
                 "different sample size - read alongside `missing_pct` in the feature section."),
    }


# Distribution drift from train to test: KS, SMD and PSI.
def population_stability_index(reference: Sequence[float], current: Sequence[float],
                               bins: int = 10) -> Optional[float]:
    """PSI on reference deciles - a familiar monitoring metric (PSI > 0.2 => drift)."""
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
    """Compare each feature's distribution between the two time splits (train -> test).

    Three measures because each catches a different kind of drift: KS (whole distribution, no
    assumptions), SMD (mean shift in pooled standard deviations), PSI (deployment standard, easy to
    threshold). Strong drift means test conclusions may reflect a new distribution, not model skill.
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


# Missingness: rate per split and whether missingness carries a label signal.
def missingness_analysis(X_by_split: Dict[str, np.ndarray], y_by_split: Dict[str, Sequence[int]],
                         names: Sequence[str]) -> Dict[str, Any]:
    """Missing-value analysis: per split, by label (MNAR), co-missing, per-sample missing density.

    Technical question: if the positive-label rate differs MEANINGFULLY between the missing group and
    the present group, then a missing value is not "harmless noise" but information => median impute
    can erase signal (or create fake signal), and the feature needs an extra `is_missing` flag.
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
            except ValueError:  # degenerate table (one row all zeros)
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
            "test": "chi-square 2x2 (missing/present x label) on train+validation+test",
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


# Potential leakage: history overlap, duplicate feature rows and label time order.
def _history_periods(sample: Dict[str, Any]) -> List[str]:
    """List of period-end (`period_end`) values for the quarters in the sample history."""
    return [str(row.get("period_end")) for row in sample["request"]["history"]]


def history_overlap_analysis(samples_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Measure HISTORY OVERLAP between samples - why 324 samples != 324 independent observations.

    Two measurements:
    1. Within a company: Jaccard between the histories of two consecutive samples (same 8-quarter
       window => most history rows are shared, so the effective sample is far smaller).
    2. Between train and test: how many test history periods already appear in train history, and how
       many target periods of one set appear in the other set's history (cross-target leakage).
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
        "note": ("High Jaccard => consecutive samples share most of their history: the number of "
                 "INDEPENDENT observations is smaller than the sample count, and metric confidence "
                 "intervals must be read by company cluster."),
    }


def _row_keys(X: np.ndarray, decimals: int = 6) -> List[Tuple[Any, ...]]:
    """Deduplication key for a feature row (NaN replaced by `None` so rows are comparable)."""
    keys: List[Tuple[Any, ...]] = []
    for row in np.asarray(X, dtype=float):
        keys.append(tuple(None if not math.isfinite(v) else round(float(v), decimals) for v in row))
    return keys


def duplicate_feature_rows(X_ref: np.ndarray, X_new: np.ndarray, ref_name: str = REFERENCE_SPLIT,
                           new_name: str = COMPARISON_SPLIT, decimals: int = 6) -> Dict[str, Any]:
    """Count exactly-duplicated feature rows between two splits (a sign of duplicated records/mechanical leakage)."""
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
    """Check the label time order: the label must exist AFTER the target period ends."""
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


# Illustrative figures (matplotlib Agg backend).
def fig_missing_pct_by_split(missing_by_split: Dict[str, Dict[str, Any]], path: Any) -> None:
    """Figure 1 - heatmap of missing % by [feature x split], sorted by the train missing rate."""
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
    ax.set_title("Missing values by feature x split (%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_label_by_ticker(label: Dict[str, Any], path: Any) -> None:
    """Figure 2 - positive-label rate by company (with IR and number of label flips)."""
    per_ticker = label.get("per_ticker", {})
    tickers = sorted(per_ticker, key=lambda t: -(per_ticker[t].get("positive_pct") or 0))
    values = [(per_ticker[t].get("positive_pct") or 0.0) for t in tickers]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(tickers[::-1], values[::-1], color="crimson", alpha=0.8)
    for bar, ticker in zip(bars, tickers[::-1]):
        record = per_ticker[ticker]
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"IR={record.get('imbalance_ratio')} | label flips {record.get('n_switches')} times",
                va="center", fontsize=7)
    ax.axvline(50, color="k", ls="--", lw=1, label="balanced 50%")
    ax.set_xlim(0, 118)
    ax.set_xlabel("Share of samples with label = 1 (%)")
    ax.set_title("Imbalance at the COMPANY level (the label is an entity attribute)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_target_association(association: Dict[str, Any], path: Any, top: int = 20) -> None:
    """Figure 3 - top features separating the two classes best (|2*AUC-1|), with mutual information."""
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
    ax.set_xlabel("2*AUC - 1 (red: high value => label 1; blue: low value => label 1)")
    ax.set_title(f"Feature <-> label association (train+validation, top {top})", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_correlation(matrix: Sequence[Sequence[Optional[float]]], names: Sequence[str],
                    clusters: Sequence[Sequence[str]], path: Any) -> None:
    """Figure 4 - Pearson correlation heatmap, ordered by collinear clusters with |r| >= threshold."""
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
    ax.set_title("Pearson correlation between features (boxes = clusters |r| >= 0.90)", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_drift_ks_vs_smd(drift: Dict[str, Any], path: Any) -> None:
    """Figure 5 - train -> test drift: KS (whole distribution) vs SMD (mean shift)."""
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
    ax.set_xlabel("SMD (test - train, in pooled SD)")
    ax.set_ylabel("KS statistic")
    ax.set_title("Train -> test distribution drift (red = warning threshold)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_top_feature_ecdf(X: np.ndarray, y: Sequence[int], names: Sequence[str],
                         ranked: Sequence[Dict[str, Any]], path: Any, top: int = 4) -> None:
    """Figure 6 - ECDF of the top features by label: see the overlap, not just the mean."""
    labels = np.asarray([int(v) for v in y], dtype=int)
    picked = [r for r in ranked if r.get("auc") is not None][:top]
    fig, axes = plt.subplots(1, max(1, len(picked)), figsize=(3.4 * max(1, len(picked)), 3.6))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, picked):
        j = list(map(str, names)).index(str(row["feature"]))
        column = np.asarray(X[:, j], dtype=float)
        for value, color, label in ((0, "steelblue", "no distress"), (1, "crimson", "distress")):
            subset = np.sort(_finite(column[labels == value]))
            if subset.size:
                ax.plot(subset, np.arange(1, subset.size + 1) / subset.size, color=color, lw=1.4,
                        label=label)
        ax.set_title(f"{row['feature']}\nAUC={row['auc']:.3f}", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("value", fontsize=8)
    axes[0].set_ylabel("ECDF", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle("Distribution (ECDF) of the strongest class-separating features - by label", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_missingness_information(missingness: Dict[str, Any], path: Any, top: int = 12) -> None:
    """Figure 7 - gap in positive-label rate between the MISSING group and PRESENT group (MNAR warning)."""
    ranked = [r for r in missingness.get("label_informativeness", {}).get("ranked_by_abs_delta", [])
              if r.get("delta_pct_points") is not None][:top]
    names = [str(r["feature"]) for r in ranked][::-1]
    deltas = [float(r["delta_pct_points"]) for r in ranked][::-1]
    colors = ["crimson" if d > 0 else "steelblue" for d in deltas]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.barh(names, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Delta positive-label rate (missing - present), percentage points")
    ax.set_title("Does a missing value carry a label signal? (chi-square + BH-FDR)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Automatic conclusions and the markdown report.
def auto_conclusions(summary: Dict[str, Any]) -> List[str]:
    """Generate the list of conclusions/warnings DIRECTLY from the numbers (nothing hand-written)."""
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
        f"**Imbalance level (sample level):** label entropy {overall['entropy_bits']} bits "
        f"(max 1.0), IR = {overall['imbalance_ratio']}, minority class {overall['minority_pct']}% "
        f"=> {overall['level']}; majority-class guessing already reaches {overall['majority_baseline_accuracy_pct']}% "
        f"=> do not use Accuracy as the primary metric.")
    lines.append(
        f"**The label is almost an entity attribute:** {transitions['persistence_pct']}% of consecutive-quarter "
        f"pairs keep the same label; P(label=1 | previous label=1) = {transitions['p_one_given_one']}, "
        f"P(label=1 | previous label=0) = {transitions['p_one_given_zero']}; "
        f"{entities['share_samples_in_single_class_tickers_pct']}% of samples belong to a company with only ONE class.")
    if entities["tickers_all_one"] or entities["tickers_all_zero"]:
        lines.append(
            f"**Never split randomly by sample:** single-class companies = "
            f"{', '.join(entities['tickers_all_one'] + entities['tickers_all_zero'])} => every evaluation "
            f"must split by group (`GroupKFold`/LOCO) and report IR by company.")
    if flags["constant"] or flags["near_constant"]:
        lines.append(
            f"**Useless features / fake signal:** constant = "
            f"{', '.join(flags['constant']) or '—'}; near-constant (>95% one value) = "
            f"{', '.join(flags['near_constant']) or '—'} => consider dropping.")
    if flags["missing_gt_20pct"]:
        lines.append(
            f"**Features dominated by imputation** (missing > {MISSING_WARN_PCT:.0f}%): "
            f"{', '.join(flags['missing_gt_20pct'][:12])} => read alongside the missingness-carries-label test.")
    if flags["heavy_tail"]:
        lines.append(
            f"**Heavy tails / strong skew** (|skew| > {SKEW_WARN:.0f} or kurtosis > {KURTOSIS_WARN:.0f}): "
            f"{', '.join(flags['heavy_tail'][:12])} => add winsorize/clip and use ranking metrics.")
    if flags["many_outliers"]:
        lines.append(
            f"**Many IQR outliers** (> {OUTLIER_WARN_PCT:.0f}% of samples): "
            f"{', '.join(flags['many_outliers'][:12])}.")
    lines.append(
        f"**Feature <-> label association:** {summary['feature_vs_label']['n_usable_features']}/"
        f"{summary['feature_quality']['n_features']} features have enough samples for a single-feature AUC; "
        f"{summary['feature_vs_label']['n_effect_ge_0_30']} features have |2*AUC-1| >= 0.30; "
        f"{summary['feature_vs_label']['n_significant_after_bh_5pct']} features remain significant after BH-FDR.")
    lines.append(
        f"**Collinearity:** {summary['feature_vs_feature']['n_pairs_abs_ge_threshold']} pairs with |r| >= "
        f"{summary['feature_vs_feature']['threshold']}, grouped into "
        f"{len(summary['feature_vs_feature']['clusters_abs_ge_threshold'])} clusters; "
        f"effective dimensionality = "
        f"{summary['feature_vs_feature']['effective_dimensionality']['participation_ratio']} "
        f"(out of {summary['feature_quality']['n_features']} columns).")
    lines.append(
        f"**Train -> test drift:** {drift['n_drifted']} features exceed the threshold (KS >= {DRIFT_KS_WARN} "
        f"or |SMD| >= {DRIFT_SMD_WARN}), {drift['n_psi_above_0_2']} features have PSI > 0.2.")
    lines.append(
        f"**Samples are not independent:** mean history Jaccard between two consecutive same-company samples = "
        f"{leakage['history_overlap']['consecutive_samples_same_ticker']['mean_history_jaccard']} "
        f"=> the number of independent observations is smaller than the sample count; "
        f"{leakage['duplicates']['n_new_rows_duplicated_in_ref']} test rows are exact duplicates feature với train.")
    lines.append(
        f"**Label time-order check:** {leakage['label_availability']['n_label_available_before_target_end']}"
        f" samples have `label_available_on` <= the target period end date (expected 0); median gap "
        f"{leakage['label_availability']['median_gap_days']} days.")
    informative = missing["label_informativeness"]["n_significant_after_bh_5pct"]
    lines.append(
        f"**Missingness:** {informative} features have a significantly different label rate between the missing "
        f"group and the present group (BH-FDR < 5%) => data is missing under an MNAR mechanism, so median "
        f"imputation alone erases signal; suggest adding an `is_missing` flag when retraining.")
    return lines


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
           aligns: Sequence[str] | None = None) -> str:
    """Simple markdown table (replaces `DataFrame.to_markdown` when pandas is unavailable)."""
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
    """Take the `count` records with the largest `key` (dropping records missing the value)."""
    available = [r for r in stats if r.get(key) is not None]
    return sorted(available, key=lambda r: r[key], reverse=reverse)[:count]


def outlier_driven_pairs(correlation: Dict[str, Any],
                         gap: float = 0.30) -> List[Dict[str, Any]]:
    """Variable pairs whose Pearson correlation diverges from Spearman, indicating outlier-driven correlation.

    Pearson is sensitive to extreme values while Spearman is not, so a large gap between the two
    coefficients on a pair means the apparent relation comes from a few unusual observations rather
    than a rule. Every pair is checked, not just the strongest, so no spurious correlation is missed.
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
    """Markdown table (public API - shared by `scripts/eda.py` and the automatic report)."""
    return _table(headers, rows, aligns)


def markdown_eda_deep(summary: Dict[str, Any]) -> str:
    """Build `reports/results/eda_deep.md` from the summary - every number comes from JSON, none typed by hand."""
    scope, label = summary["scope"], summary["label"]
    stats, flags = summary["feature_quality"]["stats"], summary["feature_quality"]["flags"]
    association = summary["feature_vs_label"]
    corr = summary["feature_vs_feature"]
    drift, missing = summary["drift"], summary["missingness"]
    leakage = summary["leakage"]

    lines = ["# Deep EDA - features, labels, correlation, drift", "",
             "*Generated automatically by `python -m scripts.eda_deep` (module: `forecasting/eda.py`). "
             "No number is typed by hand.*", "",
             "## 0. Scope", "",
             f"- Samples: " + ", ".join(f"{k}={v}" for k, v in scope["n_samples"].items()),
             f"- Features: **{scope['n_features']}** columns (from `forecasting.features.feature_names()`).",
             f"- Feature-label association and correlation are computed on **{scope['association_eval_set']}**; "
             f"test is used only for the drift diagnostic (train -> test).",
             "",
             "## 1. Feature matrix quality", "",
             f"- Constant features: **{', '.join(flags['constant']) or '—'}**; near-constant (>95% one "
             f"value): **{', '.join(flags['near_constant']) or '—'}**.",
             f"- Missing > {MISSING_WARN_PCT:.0f}%: **{len(flags['missing_gt_20pct'])}** features; "
             f"heavy tail: **{len(flags['heavy_tail'])}**; many IQR outliers: "
             f"**{len(flags['many_outliers'])}**.",
             ""]
    lines.append(_table(
        ["Feature", "Missing %", "#values", "Median", "P1-P99", "Skew", "Kurtosis", "IQR outlier %", "= 0 %"],
        [[r["feature"], r.get("missing_pct"), r.get("n_unique"), r.get("median"),
          f"{r.get('p1')} … {r.get('p99')}", r.get("skew"), r.get("kurtosis_excess"),
          r.get("iqr_outlier_pct"), r.get("zero_pct")]
         for r in _top(stats, "missing_pct", 15)],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    lines += [f"- {r['feature']}: skew={r.get('skew')}, kurtosis={r.get('kurtosis_excess')}, "
              f"outlier IQR={r.get('iqr_outlier_pct')}%"
              for r in _top(stats, "kurtosis_excess", 5)]
    lines += ["", "![Missing by feature x split]"
              "(../reports/figures/eda_deep/01_missing_pct_by_split.png)", "",
              "![Missingness carries a label signal]"
              "(../reports/figures/eda_deep/07_missingness_information.png)", ""]

    lines += ["## 2. Labels: imbalance, entropy, state runs", "",
              _table(["Split", "n", "Positive", "Positive rate %", "IR", "Entropy (bits)",
                      "Entropy normalized", "Gini", "Majority baseline %", "Level"],
                     [[name, r.get("n"), r.get("positive"), r.get("positive_pct"),
                       r.get("imbalance_ratio"), r.get("entropy_bits"), r.get("entropy_normalized"),
                       r.get("gini"), r.get("majority_baseline_accuracy_pct"), r.get("level")]
                      for name, r in label["per_split"].items()],
                     ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
              "",
              _table(["Company", "n", "Positive rate %", "IR", "Level", "Label flips",
                      "Longest run of 1", "Longest run of 0"],
                     [[t, r.get("n"), r.get("positive_pct"), r.get("imbalance_ratio"), r.get("level"),
                       r.get("n_switches"), r.get("max_run_one"), r.get("max_run_zero")]
                      for t, r in sorted(label["per_ticker"].items(),
                                         key=lambda kv: -(kv[1].get("positive_pct") or 0))],
                     ["---", "---:", "---:", "---:", "---", "---:", "---:", "---:"]),
              "",
              f"- Transitions between consecutive quarters: `{label['transitions']['counts']}`; "
              f"P(1|1) = {label['transitions']['p_one_given_one']}, "
              f"P(1|0) = {label['transitions']['p_one_given_zero']}, "
              f"label stays the same = **{label['transitions']['persistence_pct']}%**.",
              f"- Entities: **{label['entities']['n_tickers']}** companies, effective number of entities "
              f"= **{label['entities']['effective_number_of_tickers']}**; "
              f"{label['entities']['share_samples_in_single_class_tickers_pct']}% of samples belong to a " 
              f"company with only one class.",
              f"- Worst imbalance: " + ", ".join(
                  f"{r['ticker']} (IR={r['imbalance_ratio']})"
                  for r in label["entities"]["worst_imbalance"]) + ".",
              "",
              "![Label rate by company]"
              "(../reports/figures/eda_deep/02_label_by_ticker.png)", ""]

    lines += ["## 3. Feature <-> label association", "",
              f"- {association['n_usable_features']}/{scope['n_features']} features allow a "
              f"single-feature AUC; {association['n_effect_ge_0_30']} features have |2*AUC-1| >= 0.30; "
              f"{association['n_significant_after_bh_5pct']} features remain significant after BH-FDR.",
              f"- Reading: {association['multiple_testing_note']}", "",
              _table(["Feature", "#pairs", "AUC", "Direction", "r (point-biserial)", "BH q-value",
                      "MI", "Top-decile lift"],
                     [[r["feature"], r.get("n_pairs"), r.get("auc"), r.get("direction"),
                       r.get("point_biserial_r"), r.get("q_value_bh"), r.get("mutual_information"),
                       r.get("lift_top_decile")]
                      for r in association["ranked_by_effect"][:15]],
                     ["---", "---:", "---:", "---", "---:", "---:", "---:", "---:"]),
              "",
              "![Feature <-> label association]"
              "(../reports/figures/eda_deep/03_target_association.png)",
              "",
              "![ECDF of top features by label]"
              "(../reports/figures/eda_deep/06_top_feature_ecdf.png)", ""]

    dims = corr["effective_dimensionality"]
    lines += ["## 4. Feature <-> feature relationship (collinearity)", "",
              f"- **{corr['n_pairs_abs_ge_threshold']}** pairs with |r| >= {corr['threshold']}; "
              f"**{corr['n_features_in_clusters']}** features sit in "
              f"{len(corr['clusters_abs_ge_threshold'])} information clusters.",
              f"- Effective dimensionality (participation ratio) = **{dims.get('participation_ratio')}** "
              f"out of {dims.get('n_columns')} columns; "
              f"**{dims.get('n_components_for_95pct_variance')}** components cover 95% of the variance.",
              "", f"**Strongest clusters (|r| >= {corr['threshold']}):**", "",
              _table(["#", "Feature cluster", "Columns"],
                     [[index + 1, ", ".join(cluster), len(cluster)]
                      for index, cluster in enumerate(corr["clusters_abs_ge_threshold"][:10])],
                     ["---:", "---", "---:"]),
              "",
              _table(["Feature A", "Feature B", "Pearson", "Spearman"],
                     [[p["a"], p["b"], p["pearson"], p["spearman"]]
                      for p in corr["strongest_pairs"][:10]],
                     ["---", "---", "---:", "---:"]),
              "- Read the Spearman column to see whether a correlation is driven by outliers.",
              "",
              "![Correlation heatmap by cluster]"
              "(../reports/figures/eda_deep/04_correlation_clustered.png)", ""]

    lines += ["## 5. Train -> test distribution drift", "",
              f"- **{drift['n_drifted']}** features exceed the warning threshold "
              f"(KS >= {DRIFT_KS_WARN} or |SMD| >= {DRIFT_SMD_WARN}); "
              f"{drift['n_psi_above_0_2']} features have PSI > 0.2; "
              f"{drift['n_significant_after_bh_5pct']} features differ significantly after BH-FDR.",
              f"- Reading caveat: PSI splits the test sample of only {scope['n_samples'].get(COMPARISON_SPLIT)} "
              f"rows into 10 bins, so ~{max(1, scope['n_samples'].get(COMPARISON_SPLIT, 1) // 10)} observations "
              f"per bin => PSI is noise-sensitive; use KS/SMD as the main criterion and PSI for monitoring.",
              "",
              _table(["Feature", "Mean train", "Mean test", "KS", "KS p-value", "BH q-value",
                      "SMD", "PSI"],
                     [[r["feature"], r.get("mean_ref"), r.get("mean_new"), r.get("ks_statistic"),
                       r.get("ks_p_value"), r.get("q_value_bh"), r.get("smd"), r.get("psi")]
                      for r in drift["ranked_by_ks"][:15]],
                     ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "![Train -> test drift]"
              "(../reports/figures/eda_deep/05_drift_ks_vs_smd.png)", ""]

    overlap = leakage["history_overlap"]
    consecutive = overlap["consecutive_samples_same_ticker"]
    duplicates = leakage["duplicates"]
    availability = leakage["label_availability"]
    density = missing["missing_density_per_sample"]
    split_history_key = f"history_periods_{REFERENCE_SPLIT}_vs_{COMPARISON_SPLIT}"
    lines += ["## 6. Potential leakage & sample independence", "",
              f"- Two consecutive same-company samples share history (Jaccard): mean "
              f"**{consecutive['mean_history_jaccard']}**, median {consecutive['median_history_jaccard']}, "
              f"min {consecutive['min_history_jaccard']} over {consecutive['n_pairs']} pairs.",
              f"- Test history periods already in train: "
              f"**{overlap[split_history_key]['share_of_test_history_periods_in_train_pct']}%**; "
              f"test target periods appearing in train history: "
              f"**{overlap['cross_split_target_leak']['n_test_targets_seen_in_train_history']}** samples.",
              f"- Exact duplicate feature rows test<->train: **{duplicates['n_new_rows_duplicated_in_ref']}**; "
              f"duplicates within train: {duplicates['n_duplicate_rows_within_ref']}.",
              f"- Label time order: {availability['n_label_available_before_target_end']} samples have a "
              f"label published BEFORE the target period ends (expected 0); median gap "
              f"{availability['median_gap_days']} days.",
              f"- Missing density per sample: mean {density['mean_features_missing']}/"
              f"{density['n_features']} features, max {density['max_features_missing']}; "
              f"share of samples missing > 50% of features = {density['share_rows_with_gt_50pct_missing']}.",
              "",
              "**Missing features that carry a label signal (top 10 by |delta|):**", "",
              _table(["Feature", "#missing", "Label-1 rate when MISSING %", "when PRESENT %", "Delta pp",
                      "BH q-value"],
                     [[r["feature"], r.get("n_missing"), r.get("target_rate_when_missing_pct"),
                       r.get("target_rate_when_present_pct"), r.get("delta_pct_points"),
                       r.get("q_value_bh")]
                      for r in missing["label_informativeness"]["ranked_by_abs_delta"][:10]],
                     ["---", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "**Feature pairs that often go missing together (top 5):**", "",
              _table(["Feature A", "Feature B", "#both missing", "Jaccard"],
                     [[r["a"], r["b"], r["n_both_missing"], r["jaccard"]]
                      for r in missing["co_missing_top"][:5]],
                     ["---", "---", "---:", "---:"]), ""]

    lines += ["## 7. Automatic conclusions", ""]
    lines += [f"{index + 1}. {text}" for index, text in enumerate(summary["auto_conclusions"])]
    lines += ["", "## 8. Figures", ""]
    lines += [f"- `{path}`" for path in summary["figures"]]
    return "\n".join(lines) + "\n"


# Orchestration: run the whole deep EDA and write artifacts.
def run(write: bool = True, out_dir: Any = None, fig_dir: Any = None,
        figures: bool = True) -> Dict[str, Any]:
    """Run the deep EDA, write `eda_deep.{json,md}` (+ 7 figures) and return the summary.

    `out_dir` and `fig_dir` let tests point at a temp folder instead of `reports/`.
    Feature-label association and correlation use train and validation only (test untouched); test
    appears only in the distribution-drift comparison.
    """
    ensure_dirs()
    ensure_utf8_stdio()  # also called directly from tests/notebooks, so don't depend on the CLI
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
            "association_eval_set": "train+validation - test is not used to describe association",
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

    print(f"Deep EDA: {summary['feature_quality']['n_features']} features over "
          f"{len(X_analysis)} samples (train+validation); {drift['n_drifted']} features drift "
          f"train->test; {correlation['n_pairs_abs_ge_threshold']} pairs with |r| >= "
          f"{correlation['threshold']}; labels: {label['overall']['level']} "
          f"(entropy {label['overall']['entropy_bits']} bits)")
    return summary
