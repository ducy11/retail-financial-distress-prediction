"""FOCAL LOSS cho bài toán mất cân bằng — custom objective của LightGBM.

Vì sao cần (algorithm-level): `scale_pos_weight` / `class_weight` chỉ dịch mức độ quan trọng giữa hai
lớp; Focal Loss còn **giảm trọng số mẫu DỄ** bằng hệ số `(1 - p_t)^gamma`, nên mô hình tập trung vào
mẫu khó (thường là mẫu thiểu số nằm sát biên). Với dữ liệu 98/2, đây là can thiệp vào HÀM MẤT MÁT
(không đổi dữ liệu, không sinh mẫu) nên không thể gây rò rỉ.

Công thức (nhãn `y ∈ {0,1}`, xác suất dự đoán `p`, logit `z = logit(p)`):

    FL(p) = -[ alpha·y·(1-p)^gamma·ln p + (1-alpha)·(1-y)·p^gamma·ln(1-p) ]

Đạo hàm bậc 1/2 theo **logit** (LightGBM yêu cầu grad/hess theo raw score) — suy ra giải tích và được
kiểm chứng bằng sai phân số trong `tests/test_imbalance_techniques.py`:

    y = 1:  g = alpha·(1-p)^gamma·(gamma·p·ln p + p - 1)
            h = alpha·p·(1-p)·[ -gamma·(1-p)^(gamma-1)·(gamma·p·ln p + p - 1)
                               + (1-p)^gamma·(gamma·ln p + gamma + 1) ]
    y = 0:  g = (1-alpha)·p^gamma·(p - gamma·(1-p)·ln(1-p))
            h = (1-alpha)·p·(1-p)·[ gamma·p^(gamma-1)·(p - gamma·(1-p)·ln(1-p))
                                   + p^gamma·(1 + gamma·ln(1-p) + gamma) ]

`gamma = 0` ⇒ thoái hoá về weighted BCE (kiểm chứng trong test). `alpha` là trọng số lớp DƯƠNG;
`alpha=None` ⇒ tính ĐỘNG trong `fit` bằng tỉ lệ lớp nhận được (`n_âm/n`) — chỉ dùng nhãn của tập được
truyền vào, nên khi nằm trong cross-validation nó chỉ thấy fold-train (chống rò rỉ).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from . import config as C

#: Sàn cho xác suất (tránh log(0)) và sàn cho Hessian (LightGBM yêu cầu h > 0).
EPS = 1e-6
HESS_FLOOR = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid ổn định số (không tràn với |z| lớn)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def focal_grad_hess(y_true: np.ndarray, raw_score: np.ndarray, gamma: float,
                    alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """(gradient, hessian) của Focal Loss theo logit — công thức giải tích (xem docstring module)."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(_sigmoid(np.asarray(raw_score, dtype=float).ravel()), EPS, 1.0 - EPS)
    log_p, log_q = np.log(p), np.log1p(-p)
    one_minus_p = 1.0 - p

    grad_pos = alpha * one_minus_p ** gamma * (gamma * p * log_p + p - 1.0)
    hess_pos = alpha * p * one_minus_p * (
        -gamma * one_minus_p ** (gamma - 1.0) * (gamma * p * log_p + p - 1.0)
        + one_minus_p ** gamma * (gamma * log_p + gamma + 1.0))

    grad_neg = (1.0 - alpha) * p ** gamma * (p - gamma * one_minus_p * log_q)
    hess_neg = (1.0 - alpha) * p * one_minus_p * (
        gamma * p ** (gamma - 1.0) * (p - gamma * one_minus_p * log_q)
        + p ** gamma * (1.0 + gamma * log_q + gamma))

    grad = np.where(y > 0.5, grad_pos, grad_neg)
    hess = np.where(y > 0.5, hess_pos, hess_neg)
    return grad, np.maximum(hess, HESS_FLOOR)


def focal_loss_value(y_true: np.ndarray, proba: np.ndarray, gamma: float = C.FOCAL_GAMMA,
                     alpha: float = 0.75) -> float:
    """Giá trị Focal Loss trung bình (để báo cáo so sánh với BCE) — không dùng để huấn luyện."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(proba, dtype=float).ravel(), EPS, 1.0 - EPS)
    loss = -(y * alpha * (1.0 - p) ** gamma * np.log(p)
             + (1.0 - y) * (1.0 - alpha) * p ** gamma * np.log1p(-p))
    return float(np.mean(loss))


def make_focal_objective(gamma: float, alpha: float) -> Callable[[np.ndarray, np.ndarray],
                                                                Tuple[np.ndarray, np.ndarray]]:
    """Trả `objective` cho LightGBM: `(y_true, raw_score) → (grad, hess)`."""

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return focal_grad_hess(y_true, y_pred, gamma, alpha)

    objective.__name__ = f"focal_loss_gamma{gamma:g}"
    return objective


class FocalLossClassifier(BaseEstimator, ClassifierMixin):
    """Bộ phân loại LightGBM huấn luyện bằng FOCAL LOSS (custom objective).

    `alpha=None` ⇒ alpha = n_âm/n tính TRONG `fit` từ nhãn được truyền vào (chỉ fold-train khi CV).
    Xác suất dự đoán suy từ raw score (`predict(..., raw_score=True)` rồi sigmoid) vì LightGBM KHÔNG
    áp hàm liên kết cho custom objective.
    """

    def __init__(self, gamma: float = C.FOCAL_GAMMA, alpha: Optional[float] = C.FOCAL_ALPHA,
                 n_estimators: int = C.FOCAL_N_ESTIMATORS,
                 learning_rate: float = C.FOCAL_LEARNING_RATE,
                 num_leaves: int = 31, min_child_samples: int = 20, subsample: float = 0.9,
                 colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 n_jobs: int = 1, random_state: int = C.SEED) -> None:
        self.gamma = gamma
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.random_state = random_state

    @staticmethod
    def available() -> bool:
        """Focal Loss của lab cần LightGBM (backend chính của lab)."""
        try:
            import lightgbm  # noqa: F401
        except Exception:  # pragma: no cover - môi trường thiếu lightgbm
            return False
        return True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FocalLossClassifier":
        if not self.available():  # pragma: no cover - môi trường thiếu lightgbm
            raise RuntimeError("Focal Loss cần lightgbm; môi trường hiện tại không có. "
                               "Cài `python -m pip install lightgbm`.")
        from lightgbm import LGBMClassifier

        y = np.asarray(y, dtype=int).ravel()
        n_positive = int((y == 1).sum())
        n_negative = int(len(y) - n_positive)
        self.alpha_ = (float(self.alpha) if self.alpha is not None
                       else (n_negative / len(y) if len(y) else 0.5))
        self.model_ = LGBMClassifier(
            objective=make_focal_objective(self.gamma, self.alpha_),
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            subsample=self.subsample, subsample_freq=1, colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda, n_jobs=self.n_jobs, verbose=-1,
            random_state=self.random_state)
        self.model_.fit(np.asarray(X, dtype=float), y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model_.predict(np.asarray(X, dtype=float), raw_score=True),
                         dtype=float).ravel()
        positive = _sigmoid(raw)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Tham số công khai (đủ để clone/pickle lại mô hình trong CV)."""
        return {"gamma": self.gamma, "alpha": self.alpha, "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate, "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples, "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree, "reg_lambda": self.reg_lambda,
                "n_jobs": self.n_jobs, "random_state": self.random_state}
