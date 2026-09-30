"""Tìm kiếm siêu tham số bằng RANDOM SEARCH + **sổ thực nghiệm** (thay Optuna khi môi trường thiếu).

Vì sao: mục 3 của phiếu chấm yêu cầu "tối ưu siêu tham số bài bản (Optuna/GridSearchCV) **có lưu vết
thực nghiệm**". Môi trường đồ án **không có `optuna`** (không cài thêm được), nên ở đây:

1. `random_search` — lấy mẫu ngẫu nhiên có kiểm soát (log-uniform cho tham số scale như `learning_rate`,
   `C`; rời rạc cho `max_depth`, `num_leaves`…), đánh giá bằng **StratifiedGroupKFold** (giữ trọn công ty
   ngoài fold-train), mục tiêu **AP** (không phụ thuộc ngưỡng) — cùng giao thức với `forecasting.tuning`.
2. `write_ledger` — ghi **mọi trial** ra `reports/results/runs.csv` (run_id, model, params, seed, cv_ap,
   độ lệch chuẩn, thời gian, trạng thái) ⇒ tra cứu lại bất kỳ thí nghiệm nào, không phụ thuộc log văn bản.
3. `compare_with_grid` — so kết quả random search với `tuning.json` (GridSearchCV) trên cùng thước đo
   để biết tìm kiếm rộng hơn có đáng chi phí không.

Nếu môi trường có Optuna: chỉ cần thay `sample_params` bằng `trial.suggest_*` — không gian ở đây cố ý
mô tả dạng dữ liệu (`SEARCH_SPACES`) để chuyển đổi máy móc được.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import RANDOM_SEED
from .models import HYPERPARAMS, MODEL_REGISTRY, make_model

#: Không gian tìm kiếm mặc định: tên tham số → ("loguniform"|"int"|"float"|"choice", tham số...).
SEARCH_SPACES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "logistic": {
        "C": ("loguniform", 1e-3, 10.0),
        "class_weight": ("choice", (None, "balanced")),
        "max_iter": ("choice", (1000, 2000)),
    },
    "random_forest": {
        "max_depth": ("choice", (3, 4, 6, 8, None)),
        "min_samples_leaf": ("int", 1, 5),
        "n_estimators": ("choice", (200, 300, 500, 800)),
        "max_features": ("choice", ("sqrt", "log2", 0.5)),
    },
    "hist_gradient_boosting": {
        "learning_rate": ("loguniform", 0.01, 0.2),
        "max_depth": ("int", 2, 5),
        "max_iter": ("choice", (200, 300, 400, 600)),
        "l2_regularization": ("loguniform", 1e-3, 10.0),
    },
}


def sample_params(space: Mapping[str, Tuple[Any, ...]], rng: np.random.Generator) -> Dict[str, Any]:
    """Lấy mẫu một cấu hình từ không gian tìm kiếm (log-uniform cho tham số scale)."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "loguniform":
            params[name] = float(math.exp(rng.uniform(math.log(float(spec[1])),
                                                      math.log(float(spec[2])))))
        elif kind == "float":
            params[name] = float(rng.uniform(float(spec[1]), float(spec[2])))
        elif kind == "int":
            params[name] = int(rng.integers(int(spec[1]), int(spec[2]) + 1))
        elif kind == "choice":
            options = list(spec[1])
            params[name] = options[int(rng.integers(0, len(options)))]
        else:  # pragma: no cover - cấu hình sai
            raise ValueError(f"kiểu không gian không hợp lệ: {kind!r}")
    return params


def _safe_key(params: Mapping[str, Any]) -> str:
    """Khoá so trùng cấu hình (để không đánh giá lại cùng một điểm)."""
    normalized = {k: (None if v is None else round(float(v), 6) if isinstance(v, float) else v)
                  for k, v in sorted(params.items())}
    return json.dumps(normalized, sort_keys=True, default=str)


def cross_company_ap(model_name: str, params: Mapping[str, Any], X: np.ndarray, y: np.ndarray,
                     groups: np.ndarray, n_splits: int = 4, winsorize: str = "none") -> Dict[str, Any]:
    """AP out-of-fold khi giữ TRỌN công ty ra khỏi fold-train (cùng giao thức với `tuning`)."""
    n_splits = max(2, min(n_splits, len(set(groups.tolist()))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores: List[float] = []
    aurocs: List[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        if len(set(y[train_idx].tolist())) < 2 or len(set(y[test_idx].tolist())) < 2:
            continue
        model = make_model(model_name, winsorize=winsorize, **params)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        scores.append(float(average_precision_score(y[test_idx], proba)))
        aurocs.append(float(roc_auc_score(y[test_idx], proba)))
    if not scores:
        return {"cv_average_precision": None, "cv_ap_std": None, "cv_auroc": None, "n_folds": 0}
    return {"cv_average_precision": float(np.mean(scores)),
            "cv_ap_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "cv_auroc": float(np.mean(aurocs)), "n_folds": len(scores)}


def random_search(model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  n_trials: int = 40, seed: int = RANDOM_SEED, n_splits: int = 4,
                  space: Optional[Mapping[str, Tuple[Any, ...]]] = None,
                  winsorize: str = "none") -> Dict[str, Any]:
    """Random search cho một mô hình; trả mọi trial (kèm thời gian) + trial tốt nhất."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Model chưa đăng ký: {model_name!r}")
    space = space or SEARCH_SPACES.get(model_name) or {}
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(n_trials):
        params = sample_params(space, rng)
        key = _safe_key(params)
        if key in seen:                      # bỏ trùng: mỗi trial phải là một điểm mới
            continue
        seen.add(key)
        started = time.perf_counter()
        try:
            metrics = cross_company_ap(model_name, params, X, y, groups, n_splits=n_splits,
                                       winsorize=winsorize)
            status = "ok"
        except Exception as exc:  # pragma: no cover - cấu hình không fit được
            metrics, status = {"cv_average_precision": None}, f"failed: {type(exc).__name__}"
        trials.append({"run_id": f"{model_name}-{seed}-{index:03d}", "model": model_name,
                       "params": params, "seed": int(seed), "status": status,
                       "seconds": float(time.perf_counter() - started), **metrics})
    valid = [t for t in trials if t.get("cv_average_precision") is not None]
    best = max(valid, key=lambda t: t["cv_average_precision"]) if valid else None
    return {"model": model_name, "n_trials": len(trials), "n_valid": len(valid), "best": best,
            "trials": trials, "default_config_in_hyperparams":
                HYPERPARAMS.get(model_name, {})}


def write_ledger(path: Path, rows: Sequence[Dict[str, Any]]) -> int:
    """Ghi sổ thực nghiệm ra CSV (một dòng = một trial) — tra cứu lại được sau này."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "model", "params", "seed", "status", "cv_average_precision", "cv_ap_std",
              "cv_auroc", "n_folds", "seconds"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (json.dumps(row.get(name), ensure_ascii=False, default=str)
                                    if name == "params" else row.get(name)) for name in fields})
    return len(rows)


def compare_with_grid(search: Dict[str, Any], grid_best_cv_ap: Optional[float]) -> Dict[str, Any]:
    """So random search với `GridSearchCV` (số liệu lấy từ `reports/results/tuning.json`)."""
    best = search.get("best") or {}
    best_cv = best.get("cv_average_precision")
    delta = (best_cv - grid_best_cv_ap) if (best_cv is not None and grid_best_cv_ap is not None) else None
    return {"random_search_best_cv_ap": best_cv, "grid_search_best_cv_ap": grid_best_cv_ap,
            "delta_vs_grid": delta, "n_trials": search.get("n_trials"),
            "total_seconds": float(sum(t.get("seconds") or 0.0 for t in search.get("trials") or []))}
