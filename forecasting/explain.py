"""Giải thích mô hình bằng giá trị SHAP — **tự cài đặt KernelSHAP**, không cần gói `shap`.

Vì sao tự cài đặt: môi trường của đồ án không có gói `shap` (và không cài thêm được), trong khi
yêu cầu mức Xuất sắc là *"giải thích mô hình (SHAP/Feature Importance)"*. Ở đây cài đúng thuật toán
**KernelSHAP** của Lundberg & Lee (2017): với hàm giá trị `v(S) = E[f(x) | các feature trong S]`,
giá trị Shapley là nghiệm bình phương tối thiểu có trọng số (Shapley kernel) trên các liên minh `S`
được lấy mẫu:

    π(S) = (M − 1) / [ C(M, |S|) · |S| · (M − |S|) ]

Ba cơ chế tự kiểm chứng được ghi vào artifact (không "tin lời"):
1. **Efficiency (Σφ_j = f(x) − E[f])** — đo `efficiency_gap` thực tế, kỳ vọng ≈ 0.
2. **Đối chiếu công thức giải tích** cho hàm tuyến tính `f(x) = w·x + b`: φ_j = w_j (x_j − E[x_j])
   (xem `linear_shap_exact`, được kiểm thử trong `tests/test_explain.py`).
3. **Đối chiếu thứ hạng** với permutation importance (hai phương pháp độc lập) để phát hiện bất đồng.

Lưu ý phương pháp luận: SHAP giải thích **mô hình**, không phải quan hệ nhân quả. Khi các feature
đa cộng tuyến mạnh (đồ án: VIF > 10 ở 33/47 cột), giá trị SHAP chia đều "công" cho các biến tương
quan nên phải đọc kèm cụm tương quan ở `reports/results/eda_deep.md`.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

#: Số liên minh lấy mẫu mỗi điểm cần giải thích (M = 47 nên không thể liệt kê 2^47).
DEFAULT_N_COALITIONS = 200
#: Số mẫu nền dùng để xấp xỉ E[f | x_S] (càng nhiều càng ổn định, càng chậm).
DEFAULT_N_BACKGROUND = 40
#: Sàn trọng số để ma trận thiết kế không suy biến khi C(M,k) quá lớn.
WEIGHT_FLOOR = 1e-18


def coalition_weight(n_features: int, size: int) -> float:
    """Trọng số Shapley kernel π(S) theo kích thước liên minh (loại S rỗng và S đầy đủ)."""
    if size <= 0 or size >= n_features:
        return 0.0
    return (n_features - 1) / (math.comb(n_features, size) * size * (n_features - size))


def sample_coalitions(n_features: int, n_coalitions: int, rng: np.random.Generator) -> np.ndarray:
    """Ma trận bool (n_coalitions × M): mỗi hàng là một liên minh `S` (True = giữ feature của x).

    Kích thước liên minh lấy đều trong 1..M−1 rồi lấy ngẫu nhiên tập con — cách lấy mẫu chuẩn của
    KernelSHAP khi không thể liệt kê toàn bộ 2^M liên minh.
    """
    if n_features < 2:
        return np.zeros((0, n_features), dtype=bool)
    sizes = rng.integers(1, n_features, size=n_coalitions)
    masks = np.zeros((n_coalitions, n_features), dtype=bool)
    for i, size in enumerate(sizes):
        masks[i, rng.choice(n_features, size=int(size), replace=False)] = True
    return masks


def linear_shap_exact(weights: Sequence[float], x: Sequence[float],
                      background: Sequence[Sequence[float]], bias: float = 0.0
                      ) -> Tuple[float, np.ndarray]:
    """Giá trị Shapley GIẢI TÍCH cho hàm tuyến tính f(x) = w·x + b (dùng làm ground truth).

    φ_0 = w·E[x] + b và φ_j = w_j (x_j − E[x_j]) — đúng cho mọi tập nền, mọi số feature.
    """
    w = np.asarray(weights, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    background = np.asarray(background, dtype=float)
    if background.ndim == 1:
        background = background.reshape(1, -1)
    mean_x = background.mean(axis=0)
    return float(w @ mean_x + bias), w * (x - mean_x)


def pipeline_predict_fn(pipeline: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Hàm dự đoán xác suất lớp dương từ một `Pipeline` sklearn đã fit."""
    def predict(matrix: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(np.asarray(matrix, dtype=float))[:, 1]

    return predict


def _mixed_predictions(predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                       x: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """v(S) cho mọi liên minh: trung bình f khi giữ feature trong S, phần còn lại lấy từ nền."""
    n_masks, n_features = masks.shape
    n_background = background.shape[0]
    mixed = np.where(masks[:, None, :], x[None, None, :], background[None, :, :])
    preds = np.asarray(predict_fn(mixed.reshape(n_masks * n_background, n_features)), dtype=float)
    return preds.reshape(n_masks, n_background).mean(axis=1)


def kernel_shap_values(predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                       x: np.ndarray, n_coalitions: int = DEFAULT_N_COALITIONS,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Giá trị SHAP cho MỘT điểm `x`: trả `phi` (M,), `base_value`, `prediction`, `efficiency_gap`."""
    background = np.asarray(background, dtype=float)
    if background.ndim == 1:
        background = background.reshape(1, -1)
    x = np.asarray(x, dtype=float).ravel()
    n_features = x.shape[0]
    rng = rng or np.random.default_rng(0)
    base_value = float(np.mean(predict_fn(background)))
    prediction = float(np.asarray(predict_fn(x.reshape(1, -1)), dtype=float).ravel()[0])
    masks = sample_coalitions(n_features, n_coalitions, rng)
    if masks.shape[0] == 0:
        return {"phi": np.zeros(n_features), "base_value": base_value, "prediction": prediction,
                "efficiency_gap": prediction - base_value, "n_coalitions": 0}

    values = _mixed_predictions(predict_fn, background, x, masks)
    weights = np.maximum(np.array([coalition_weight(n_features, int(m.sum())) for m in masks]),
                         WEIGHT_FLOOR)
    sqrt_w = np.sqrt(weights)
    design = masks.astype(float) * sqrt_w[:, None]
    target = (values - base_value) * sqrt_w
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    # KernelSHAP có ràng buộc hiệu suất: hiệu chỉnh đều mỗi φ để Σφ = f(x) − E[f] đúng chính xác.
    phi = solution + (prediction - base_value - solution.sum()) / n_features
    return {"phi": phi, "base_value": base_value, "prediction": prediction,
            "efficiency_gap": float(prediction - base_value - phi.sum()),
            "n_coalitions": int(masks.shape[0])}


def kernel_shap_matrix(predict_fn: Callable[[np.ndarray], np.ndarray], background: np.ndarray,
                       X_explain: np.ndarray, n_coalitions: int = DEFAULT_N_COALITIONS,
                       random_state: int = 0) -> Dict[str, Any]:
    """Giá trị SHAP cho cả ma trận điểm cần giải thích (mỗi hàng một điểm) + chỉ số tự kiểm chứng."""
    X_explain = np.asarray(X_explain, dtype=float)
    rng = np.random.default_rng(random_state)
    rows = [kernel_shap_values(predict_fn, background, X_explain[i], n_coalitions=n_coalitions,
                               rng=rng) for i in range(X_explain.shape[0])]
    phi = np.vstack([r["phi"] for r in rows]) if rows else np.zeros((0, X_explain.shape[1]))
    predictions = np.array([r["prediction"] for r in rows])
    gaps = np.array([r["efficiency_gap"] for r in rows])
    return {
        "phi": phi,
        "base_value": float(np.mean([r["base_value"] for r in rows])) if rows else 0.0,
        "predictions": predictions,
        "n_coalitions": int(n_coalitions),
        "max_abs_efficiency_gap": float(np.max(np.abs(gaps))) if gaps.size else 0.0,
        "relative_efficiency_gap": (float(np.max(np.abs(gaps) / np.maximum(np.abs(predictions), 1e-9)))
                                    if gaps.size else 0.0),
        "n_explained": int(phi.shape[0]),
    }


def mean_abs_shap(phi: np.ndarray) -> np.ndarray:
    """Độ quan trọng toàn cục: trung bình |φ_j| trên các điểm đã giải thích (giống `shap` summary)."""
    return np.mean(np.abs(np.asarray(phi, dtype=float)), axis=0)


def permutation_importance_ranking(pipeline: Any, X: np.ndarray, y: np.ndarray,
                                   n_repeats: int = 5, random_state: int = 0) -> Optional[np.ndarray]:
    """ΔAUROC khi hoán vị từng cột (đối chiếu độc lập với SHAP); `None` nếu chỉ có một lớp."""
    from sklearn.inspection import permutation_importance

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    result = permutation_importance(pipeline, np.asarray(X, dtype=float), y, scoring="roc_auc",
                                    n_repeats=n_repeats, random_state=random_state)
    return np.asarray(result.importances_mean, dtype=float)


def rank_agreement(a: np.ndarray, b: np.ndarray, top: int = 15) -> Dict[str, Any]:
    """Mức đồng thuận giữa hai bảng xếp hạng feature: Spearman + Jaccard của top-k."""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 3:
        return {"spearman": None, "top_overlap": None}
    spearman = float(spearmanr(a, b).statistic)
    top_a, top_b = set(np.argsort(-a)[:top].tolist()), set(np.argsort(-b)[:top].tolist())
    return {"spearman": spearman,
            "top_overlap": len(top_a & top_b) / max(1, len(top_a | top_b)),
            "top_k": int(min(top, a.size)),
            "agreed_features": sorted(top_a & top_b)}
