"""Single source of configuration for the lab; no parameter is hardcoded in the modules."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

#: Repository root and the directory for artifacts (CSV, JSON, Markdown, log).
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "reports" / "imbalance"

# Dataset generation for an imbalanced setting, using a 98/2 split.
SEED = 42
N_SAMPLES = 50_000
N_FEATURES = 20
N_INFORMATIVE = 8
N_REDUNDANT = 6
CLASS_SEP = 1.0
FLIP_Y = 0.0  # keep the exact 98/2 ratio by leaving labels unflipped
#: (negative-class rate, positive-class rate): 98% class 0, 2% class 1.
CLASS_WEIGHTS: Tuple[float, float] = (0.98, 0.02)

# Split and cross-validation settings that preserve class proportions.
TEST_SIZE = 0.20      # the holdout test is used exactly once and is never resampled
N_SPLITS = 5          # StratifiedKFold over the train pool

# Resampling parameters, applied only to the train part of each fold.
SMOTE_SAMPLING_STRATEGY = 0.10    # SMOTE raises the minority to 10% of the majority count
UNDER_SAMPLING_STRATEGY = 0.50    # RandomUnderSampler lowers the majority until minority/majority is 0.5
SMOTE_K_NEIGHBORS = 5
#: True selects the imblearn Pipeline as the main path; False selects the in-house implementation.
PREFER_IMBLEARN = True

# Technique catalog from the rubric; see labs/imbalance_lab/techniques.py.
#: Target minority/majority ratio after oversampling, used by RandomOverSampler, SMOTE, BorderlineSMOTE
#: and ADASYN.
TECHNIQUE_OVER_STRATEGY = 0.50
#: Target minority/majority ratio after undersampling, used by RandomUnderSampler.
TECHNIQUE_UNDER_STRATEGY = 0.50
#: Neighbor count for BorderlineSMOTE and ADASYN in the in-house implementation.
TECHNIQUE_K_NEIGHBORS = 5
#: Default sample count for the catalog, lower than N_SAMPLES because it runs roughly 14 techniques
#: across K folds.
CATALOG_N_SAMPLES = 20_000
#: Default fold count for the catalog.
CATALOG_N_SPLITS = 5
#: Focal Loss as a LightGBM custom objective: gamma down-weights easy samples, alpha weights the
#: positive class. alpha=None is computed inside fit as n_negative/n from the received labels only,
#: which avoids leakage.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA: Optional[float] = 0.75
#: Maximum boosting rounds and learning rate for the Focal Loss model.
FOCAL_N_ESTIMATORS = 400
FOCAL_LEARNING_RATE = 0.05

#: True adds a resampling_calibrated variant that calibrates probabilities after resampling.
CALIBRATE_RESAMPLING = True
#: Calibration method: isotonic (non-parametric, suits more data) or sigmoid (Platt, fewer parameters).
CALIBRATION_METHOD = "isotonic"
#: Inner fold count for CalibratedClassifierCV, running cross-fitting on the train part of the fold.
CALIBRATION_FOLDS = 5

# Decision thresholds and cost sensitivity.
#: Relative cost of a miss (FN) against a false alarm (FP) when searching a cost-based threshold.
COST_FN = 10.0
COST_FP = 1.0
#: Target precision for the "threshold for min precision" mode.
PRECISION_TARGET = 0.50
#: Beta for the reported F-beta score. A beta of 2 weights recall twice as much as precision, which
#: fits a task where a miss costs more than a false alarm. F1 is the special case beta = 1.
FBETA_BETA = 2.0
#: Bootstrap rounds for the 95% confidence interval of PR-AUC and F1 on the holdout test.
N_BOOTSTRAP = 500

# Base classifier parameters.
#: LightGBM parameters, switched to xgboost or a histogram gradient boosting model when lightgbm is
#: missing; see models.make_base_classifier.
LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 1,        # single thread keeps results stable and reproducible
    "verbose": -1,
}
