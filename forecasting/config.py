"""Shared config: paths, indicators (features), hyperparameters.

Every module should import config from here instead of hardcoding paths.
"""
from __future__ import annotations

from pathlib import Path

# Directory tree paths, resolved relative to this file.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RETAIL_DIR = DATA_DIR / "retail-expanded"
PREPARED_DIR = DATA_DIR / "prepared"
REPORTS_DIR = ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = REPORTS_DIR / "models"
DOCS_DIR = ROOT / "docs"

PREPARED_FILES = {
    "train": PREPARED_DIR / "train.json",
    "validation": PREPARED_DIR / "validation.json",
    "test": PREPARED_DIR / "test.json",
    "purged": PREPARED_DIR / "purged.json",
}
MANIFEST_FILE = PREPARED_DIR / "manifest.json"

#: Full names of the level-1 indicators pulled from SEC XBRL.
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

#: Value suffix used in the prepared JSON (integer strings in VND).
SUFFIX = "_vnd"

#: Target label (the target column).
TARGET = "is_distressed"

#: Feature history window - use the most recent 8 quarters (2 fiscal years).
LOOKBACK_QUARTERS = 8

#: Level-2 financial ratios computed in features - key: English formula for logging/explaining.
#: `total_liabilities` = the `liabilities` tag when that quarter has it, otherwise derived from
#: `total_assets - stockholders_equity` (the `liabilities` tag covers only 39% of quarters;
#: see `forecasting/features.py`).
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

#: Reference column names produced by features, used as X columns after column encoding.
#: The real feature columns come from `forecasting.features.feature_names()` (ratio, growth,
#: capital structure, window extremes, stress flags). This constant lists only the 14 base ratios
#: for reference; never hardcode a column count, use `len(feature_names())`.
FEATURE_COLS = RATIOS.keys()

#: Fixed seed so runs are reproducible (splits themselves are time-based, hence deterministic).
RANDOM_SEED = 42

#: Training needs both classes present; operating thresholds swept for the distress rate.
EVAL_THRESHOLDS = [0.30, 0.35, 0.40, 0.50, 0.60, 0.656]

#: Group key for entity-wise cross-validation (never leak a company across train/test).
GROUP_KEY = "ticker"

#: Minimum history quarters for YoY features to make sense (YoY needs 5: current + 4 ago).
MIN_HISTORY_QUARTERS = 5

#: Bootstrap rounds for metric confidence intervals (the "uncertainty" section of the report).
N_BOOTSTRAP = 2000

#: Relative costs for expected-utility threshold analysis: a miss (FN) costs more than a false alarm.
COST_FN = 5.0
COST_FP = 1.0

#: Reproducible replacement-label rule config (see forecasting/labels.py).
STRESS_MIN_SIGNALS = 1
STRESS_RULE_NAME = "stress_signals"

#: Env var to run the whole pipeline on a different split (e.g. data/prepared-rule).
ENV_PREPARED_DIR = "FORECASTING_PREPARED_DIR"


def ensure_dirs() -> None:
    """Create output folders if missing."""
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def ensure_utf8_stdio() -> None:
    """Force stdout/stderr to UTF-8 (avoids UnicodeEncodeError when piping on cp1252 consoles).

    Python 3.7+: sys.stdout.reconfigure exists; skip silently on platforms without it.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover - device/stream cannot be reconfigured
            pass


def prepared_dir() -> Path:
    """Split folder currently in use.

    Defaults to `PREPARED_DIR` (original labels). Set env var `FORECASTING_PREPARED_DIR`
    to run the pipeline on another split - e.g. `data/prepared-rule` built by `scripts.relabel`.
    """
    import os

    override = os.environ.get(ENV_PREPARED_DIR)
    return Path(override).resolve() if override else PREPARED_DIR


def prepared_files(base: Path | None = None) -> dict[str, Path]:
    """Paths of the 4 splits under `base` (defaults to `prepared_dir()`)."""
    root = base or prepared_dir()
    return {name: root / f"{name}.json" for name in ("train", "validation", "test", "purged")}


def manifest_file(base: Path | None = None) -> Path:
    """Manifest path under `base` (defaults to `prepared_dir()`)."""
    root = base or prepared_dir()
    return root / "manifest.json"

