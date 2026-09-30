"""Resampling KHÔNG rò rỉ: `imblearn.pipeline.Pipeline` là đường chính.

Vì sao phải dùng `imblearn.pipeline.Pipeline` (không phải `sklearn.pipeline.Pipeline`):
- Chỉ `imblearn` mới chạy được bước `fit_resample` **bên trong** pipeline, nên khi pipeline được
  fit trên tập train của một fold thì resampling chỉ xảy ra trên chính tập đó.
- Nếu resample TRƯỚC khi chia fold (hoặc sau khi chia nhưng ngoài pipeline) thì mẫu tổng hợp của
  lớp thiểu số trong train có thể sinh ra từ thông tin của fold validation ⇒ rò rỉ dữ liệu.

`resampling_backend()` trả về `"imblearn"` nếu thư viện có mặt. Nếu môi trường offline không cài
được `imbalanced-learn`, lab dùng bản cài đặt nội bộ tương thích API (`SMOTE`, `RandomUnderSampler`,
`Pipeline` có `fit_resample`) — bản này được log rõ ràng và đã có unit test, nhưng **đường chuẩn của
đồ án vẫn là imblearn** (`pip install -r imbalance_lab/requirements.txt`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

MINORITY_LABEL = 1


def resampling_backend(prefer_imblearn: bool = True) -> str:
    """`"imblearn"` nếu dùng được `imbalanced-learn`, ngược lại `"builtin"`."""
    if not prefer_imblearn:
        return "builtin"
    try:  # pragma: no cover - phụ thuộc môi trường
        import imblearn  # noqa: F401
    except Exception:
        return "builtin"
    return "imblearn"


# ---------------------------------------------------------------------------
# Bản nội bộ (chỉ dùng khi thiếu imbalanced-learn) — tương thích API fit_resample
# ---------------------------------------------------------------------------
class SMOTE(BaseEstimator):
    """SMOTE tối giản: nội suy giữa mẫu thiểu số và một trong `k_neighbors` láng giềng thiểu số.

    `sampling_strategy` = tỉ lệ (thiểu/đa) MONG MUỐN sau khi oversample (giống imblearn).
    """

    def __init__(self, sampling_strategy: float = 0.1, k_neighbors: int = 5,
                 random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if len(minority) < 2 or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()
        k = max(1, min(self.k_neighbors, len(minority) - 1))
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X[minority])
        _, indices = neighbours.kneighbors(X[minority])
        base = rng.randint(0, len(minority), size=n_new)
        partner = np.array([indices[i, rng.randint(1, k + 1)] for i in base])
        gap = rng.random_sample((n_new, 1))
        synthetic = X[minority][base] + gap * (X[minority][partner] - X[minority][base])
        return (np.vstack([X, synthetic]),
                np.concatenate([y, np.full(n_new, MINORITY_LABEL, dtype=int)]))


class RandomUnderSampler(BaseEstimator):
    """Giữ toàn bộ lớp thiểu số và lấy ngẫu nhiên lớp đa số để đạt tỉ lệ `sampling_strategy`."""

    def __init__(self, sampling_strategy: float = 0.5, random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if not len(minority) or not len(majority):
            return X.copy(), y.copy()
        target = int(round(len(minority) / self.sampling_strategy))
        if target >= len(majority):
            return X.copy(), y.copy()
        keep = np.concatenate([minority, rng.choice(majority, size=target, replace=False)])
        keep.sort()
        return X[keep], y[keep]


class RandomOverSampler(BaseEstimator):
    """Oversampling NGẪU NHIÊN: sao chép mẫu thiểu số tới tỉ lệ `sampling_strategy` (thiểu/đa)."""

    def __init__(self, sampling_strategy: float = 0.5, random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if not len(minority) or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()
        picks = rng.randint(0, len(minority), size=n_new)
        return (np.vstack([X, X[minority][picks]]),
                np.concatenate([y, np.full(n_new, MINORITY_LABEL, dtype=int)]))


class BorderlineSMOTE(BaseEstimator):
    """Borderline-SMOTE (biến thể borderline-1): chỉ nội suy từ mẫu thiểu số nằm ở BIÊN.

    Mẫu thiểu số gọi là "DANGER" nếu trong `m_neighbors` láng giềng gần nhất (toàn bộ dữ liệu) số
    mẫu đa số NHIỀU HƠN số mẫu thiểu số — tức nó nằm sát vùng đa số. Nội suy chỉ diễn ra giữa mẫu
    DANGER và láng giềng thiểu số của nó, nên mẫu tổng hợp bám biên quyết định.
    """

    def __init__(self, sampling_strategy: float = 0.5, k_neighbors: int = 5,
                 m_neighbors: int = 10, random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state

    def _danger_indices(self, X: np.ndarray, y: np.ndarray,
                        minority: np.ndarray) -> List[int]:
        """Chỉ số (trong `minority`) của các mẫu thiểu số nằm vùng DANGER."""
        m = max(1, min(self.m_neighbors, len(X) - 1))
        neighbours = NearestNeighbors(n_neighbors=m + 1).fit(X).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        return [i for i, row in enumerate(neighbours)
                if int((y[row] != MINORITY_LABEL).sum()) > int((y[row] == MINORITY_LABEL).sum())]

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if len(minority) < 2 or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()

        danger = self._danger_indices(X, y, minority)
        base_pool = danger if danger else list(range(len(minority)))
        k = max(1, min(self.k_neighbors, len(minority) - 1))
        minority_neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X[minority]).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        base = np.asarray([base_pool[rng.randint(0, len(base_pool))] for _ in range(n_new)])
        partner = np.asarray([minority_neighbours[i][rng.randint(0, k)] for i in base])
        gap = rng.random_sample((n_new, 1))
        synthetic = X[minority][base] + gap * (X[minority][partner] - X[minority][base])
        return (np.vstack([X, synthetic]),
                np.concatenate([y, np.full(n_new, MINORITY_LABEL, dtype=int)]))


class ADASYN(BaseEstimator):
    """ADASYN: số mẫu tổng hợp cho mỗi mẫu thiểu số TỈ LỆ với số láng giềng đa số của nó.

    Mẫu thiểu số càng "khó" (càng nhiều láng giềng đa số) càng được sinh thêm ⇒ tập trung vào vùng
    quyết định thay vì rải đều như SMOTE.
    """

    def __init__(self, sampling_strategy: float = 0.5, n_neighbors: int = 5,
                 random_state: Any = None) -> None:
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = check_random_state(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        minority = np.flatnonzero(y == MINORITY_LABEL)
        majority = np.flatnonzero(y != MINORITY_LABEL)
        if len(minority) < 2 or not len(majority):
            return X.copy(), y.copy()
        target = int(round(self.sampling_strategy * len(majority)))
        n_new = max(0, target - len(minority))
        if n_new == 0:
            return X.copy(), y.copy()

        k = max(1, min(self.n_neighbors, len(X) - 1))
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        difficulty = np.array([np.mean(y[row] != MINORITY_LABEL) for row in neighbours], dtype=float)
        if difficulty.sum() <= 0:            # không có láng giềng đa số ⇒ thoái hoá về SMOTE
            difficulty = np.ones(len(minority), dtype=float)
        quotas = difficulty / difficulty.sum() * n_new
        quotas_int = np.floor(quotas).astype(int)
        quotas_int[int(np.argmax(quotas - quotas_int))] += n_new - int(quotas_int.sum())

        k_min = max(1, min(self.n_neighbors, len(minority) - 1))
        minority_neighbours = NearestNeighbors(n_neighbors=k_min + 1).fit(X[minority]).kneighbors(
            X[minority], return_distance=False)[:, 1:]
        synthetic: List[np.ndarray] = []
        for i, count in enumerate(quotas_int):
            for _ in range(int(count)):
                partner = minority_neighbours[i][rng.randint(0, k_min)]
                gap = rng.random_sample()
                synthetic.append(X[minority][i] + gap * (X[minority][partner] - X[minority][i]))
        if not synthetic:
            return X.copy(), y.copy()
        return (np.vstack([X, np.asarray(synthetic)]),
                np.concatenate([y, np.full(len(synthetic), MINORITY_LABEL, dtype=int)]))


class TomekLinks(BaseEstimator):
    """Tomek Links — LÀM SẠCH biên: bỏ mẫu ĐA SỐ trong cặp (đa số, thiểu số) là láng giềng của nhau.

    Đây là "clean-sampling" nên KHÔNG nhận tỉ lệ mục tiêu (giống imblearn): nó gọt biên chứ không tự
    cân bằng tập 98/2 — muốn cân bằng phải dùng kèm oversampling (xem hybrid `SMOTE+TomekLinks`).
    """

    def __init__(self, n_neighbors: int = 1) -> None:
        self.n_neighbors = n_neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        if len(np.unique(y)) < 2 or len(X) < 3:
            return X.copy(), y.copy()
        nearest = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X, return_distance=False)[:, 1]
        keep = np.ones(len(y), dtype=bool)
        for i, j in enumerate(nearest):
            if y[i] == y[j]:
                continue
            if y[i] != MINORITY_LABEL and nearest[j] == i:      # cặp Tomek ⇒ bỏ mẫu đa số
                keep[i] = False
        return X[keep], y[keep]


class EditedNearestNeighbours(BaseEstimator):
    """ENN — LÀM SẠCH biên: bỏ mẫu mà `n_neighbors` láng giềng gần nhất bầu cho lớp KHÁC.

    Tương thích `imblearn`: láng giềng tìm trên TOÀN BỘ dữ liệu (loại chính nó), mặc định
    `sampling_strategy="auto"` chỉ dọn lớp ĐA SỐ, và `kind_sel="all"` bỏ mẫu nếu có BẤT KỲ láng giềng
    nào khác lớp (imblearn 0.14: `np.all(nhood_label == target_class)`); `kind_sel="mode"` bỏ mẫu nếu
    lớp của nó khác lớp chiếm đa số trong các láng giềng. Cũng là "clean-sampling" (không có tỉ lệ).
    """

    def __init__(self, n_neighbors: int = 3, kind_sel: str = "all",
                 sampling_strategy: str = "auto") -> None:
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        labels = np.unique(y)
        if len(labels) < 2 or len(X) < 3:
            return X.copy(), y.copy()
        k = max(1, min(self.n_neighbors, len(X) - 1))
        all_neighbours = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
            X, return_distance=False)[:, 1:]
        majority = int(labels[np.argmax(np.bincount(y, minlength=2))])
        targets = labels if self.sampling_strategy == "all" else np.array([majority])
        keep = np.ones(len(y), dtype=bool)
        for label in targets:
            for i in np.flatnonzero(y == label):
                votes = y[all_neighbours[i]]
                if self.kind_sel == "all":
                    keep[i] = bool(np.all(votes == label))
                else:                                    # kind_sel == "mode"
                    keep[i] = bool(int(np.argmax(np.bincount(votes, minlength=2))) == label)
        return X[keep], y[keep]


class SamplerChain(BaseEstimator):
    """Ghép nhiều sampler cho bản nội bộ (dùng cho hybrid SMOTE + TomekLinks / + ENN)."""

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        self.steps = list(steps)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_out, y_out = np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel()
        for _name, sampler in self.steps:
            X_out, y_out = sampler.fit_resample(X_out, y_out)
        return X_out, y_out


class Pipeline(BaseEstimator):
    """Bản `imblearn.pipeline.Pipeline` tối giản: sampler chạy TRƯỚC, classifier fit SAU.

    Hỗ trợ: `fit`, `fit_resample` (chỉ chạy các bước sampler — dùng để LOG phân phối sau resample),
    `predict`, `predict_proba`, và `pipe[:-1]`.
    """

    def __init__(self, steps: Sequence[Tuple[str, Any]]) -> None:
        self.steps = list(steps)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Pipeline(self.steps[index])
        return self.steps[index]

    @property
    def classifier(self):
        return self.steps[-1][1]

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Chạy tuần tự các bước sampler trên dữ liệu TRAIN của fold."""
        return self._resample(X, y)

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_out, y_out = np.asarray(X, dtype=float), np.asarray(y, dtype=int).ravel()
        for _name, step in self.steps[:-1]:
            if hasattr(step, "fit_resample"):
                X_out, y_out = step.fit_resample(X_out, y_out)
            else:  # transformer thường (vd. scaler) — vẫn giữ đúng thứ tự
                X_out = step.fit_transform(X_out, y_out)
        return X_out, y_out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Pipeline":
        X_res, y_res = self._resample(X, y)
        self.classifier.fit(X_res, y_res)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(X)


def build_sampler_pipeline(steps: Sequence[Tuple[str, Any]], classifier: Any,
                           prefer_imblearn: bool = True) -> Any:
    """Ghép các bước sampler + classifier thành pipeline (imblearn nếu có, ngược lại bản nội bộ).

    Args:
        steps: danh sách `(tên, sampler)` — sampler phải có `fit_resample`.
        classifier: estimator cuối cùng (có `fit`/`predict_proba`).
        prefer_imblearn: ưu tiên `imblearn.pipeline.Pipeline` (mặc định, đường chuẩn).
    """
    backend = resampling_backend(prefer_imblearn)
    if backend == "imblearn":  # pragma: no cover - phụ thuộc môi trường
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(list(steps) + [("classifier", classifier)])
    return Pipeline(list(steps) + [("classifier", classifier)])


def make_samplers(prefer_imblearn: bool = True, *, smote_strategy: float, k_neighbors: int,
                  under_strategy: float, random_state: Any) -> List[Tuple[str, Any]]:
    """Tạo cặp sampler hỗn hợp: SMOTE (oversample) → RandomUnderSampler (undersample)."""
    if resampling_backend(prefer_imblearn) == "imblearn":  # pragma: no cover - môi trường
        from imblearn.over_sampling import SMOTE as ImbSMOTE
        from imblearn.under_sampling import RandomUnderSampler as ImbRUS
        return [("smote", ImbSMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors,
                                   random_state=random_state)),
                ("under", ImbRUS(sampling_strategy=under_strategy, random_state=random_state))]
    return [("smote", SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors,
                            random_state=random_state)),
            ("under", RandomUnderSampler(sampling_strategy=under_strategy,
                                         random_state=random_state))]


# ---------------------------------------------------------------------------
# Danh mục kỹ thuật (yêu cầu #2) — tên kỹ thuật → bước sampler
# ---------------------------------------------------------------------------
#: Kỹ thuật ĐƠN của danh mục: `key` → nhãn hiển thị.
SINGLE_SAMPLERS: Dict[str, str] = {
    "ros": "RandomOverSampler",
    "smote": "SMOTE",
    "borderline_smote": "BorderlineSMOTE",
    "adasyn": "ADASYN",
    "rus": "RandomUnderSampler",
    "tomek": "TomekLinks (làm sạch biên)",
    "enn": "EditedNearestNeighbours (làm sạch biên)",
}

#: Kỹ thuật HYBRID của danh mục: `key` → nhãn hiển thị.
HYBRID_SAMPLERS: Dict[str, str] = {
    "smote_tomek": "SMOTE + TomekLinks",
    "smote_enn": "SMOTE + EditedNearestNeighbours",
}


def make_single_sampler(kind: str, prefer_imblearn: bool = True, *,
                        over_strategy: float, under_strategy: float, k_neighbors: int,
                        random_state: Any) -> List[Tuple[str, Any]]:
    """Bước sampler cho MỘT kỹ thuật đơn (imblearn nếu có, ngược lại bản nội bộ)."""
    if kind not in SINGLE_SAMPLERS:
        raise KeyError(f"Kỹ thuật không hợp lệ: {kind!r}; có {sorted(SINGLE_SAMPLERS)}")
    if resampling_backend(prefer_imblearn) == "imblearn":  # pragma: no cover - môi trường
        return _imblearn_single(kind, over_strategy, under_strategy, k_neighbors, random_state)
    return _builtin_single(kind, over_strategy, under_strategy, k_neighbors, random_state)


def make_hybrid_sampler(kind: str, prefer_imblearn: bool = True, *,
                        over_strategy: float, under_strategy: float, k_neighbors: int,
                        random_state: Any) -> List[Tuple[str, Any]]:
    """Bước sampler cho kỹ thuật HYBRID: SMOTE (oversample) rồi làm sạch bằng TomekLinks / ENN."""
    if kind not in HYBRID_SAMPLERS:
        raise KeyError(f"Hybrid không hợp lệ: {kind!r}; có {sorted(HYBRID_SAMPLERS)}")
    if resampling_backend(prefer_imblearn) == "imblearn":  # pragma: no cover - môi trường
        from imblearn.combine import SMOTEENN as ImbSMOTEENN
        from imblearn.combine import SMOTETomek as ImbSMOTETomek
        from imblearn.over_sampling import SMOTE as ImbSMOTE

        smote = ImbSMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                         random_state=random_state)
        if kind == "smote_tomek":
            return [(kind, ImbSMOTETomek(smote=smote, random_state=random_state))]
        return [(kind, ImbSMOTEENN(smote=smote, random_state=random_state))]
    smote = SMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                  random_state=random_state)
    cleaner = TomekLinks() if kind == "smote_tomek" else EditedNearestNeighbours()
    return [(kind, SamplerChain([("smote", smote), ("clean", cleaner)]))]


def _imblearn_single(kind: str, over_strategy: float, under_strategy: float, k_neighbors: int,
                     random_state: Any) -> List[Tuple[str, Any]]:  # pragma: no cover - môi trường
    """Nhánh imblearn của `make_single_sampler` (đường chuẩn)."""
    from imblearn.over_sampling import ADASYN as ImbADASYN
    from imblearn.over_sampling import BorderlineSMOTE as ImbBorderline
    from imblearn.over_sampling import RandomOverSampler as ImbROS
    from imblearn.over_sampling import SMOTE as ImbSMOTE
    from imblearn.under_sampling import EditedNearestNeighbours as ImbENN
    from imblearn.under_sampling import RandomUnderSampler as ImbRUS
    from imblearn.under_sampling import TomekLinks as ImbTomek

    if kind == "ros":
        return [(kind, ImbROS(sampling_strategy=over_strategy, random_state=random_state))]
    if kind == "smote":
        return [(kind, ImbSMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                                random_state=random_state))]
    if kind == "borderline_smote":
        return [(kind, ImbBorderline(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                                     random_state=random_state))]
    if kind == "adasyn":
        return [(kind, ImbADASYN(sampling_strategy=over_strategy, n_neighbors=k_neighbors,
                                 random_state=random_state))]
    if kind == "rus":
        return [(kind, ImbRUS(sampling_strategy=under_strategy, random_state=random_state))]
    if kind == "tomek":
        return [(kind, ImbTomek())]
    return [(kind, ImbENN())]


def _builtin_single(kind: str, over_strategy: float, under_strategy: float, k_neighbors: int,
                    random_state: Any) -> List[Tuple[str, Any]]:
    """Nhánh nội bộ của `make_single_sampler` (dùng khi thiếu imbalanced-learn)."""
    if kind == "ros":
        return [(kind, RandomOverSampler(sampling_strategy=over_strategy,
                                         random_state=random_state))]
    if kind == "smote":
        return [(kind, SMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                             random_state=random_state))]
    if kind == "borderline_smote":
        return [(kind, BorderlineSMOTE(sampling_strategy=over_strategy, k_neighbors=k_neighbors,
                                       random_state=random_state))]
    if kind == "adasyn":
        return [(kind, ADASYN(sampling_strategy=over_strategy, n_neighbors=k_neighbors,
                              random_state=random_state))]
    if kind == "rus":
        return [(kind, RandomUnderSampler(sampling_strategy=under_strategy,
                                          random_state=random_state))]
    if kind == "tomek":
        return [(kind, TomekLinks())]
    return [(kind, EditedNearestNeighbours())]

