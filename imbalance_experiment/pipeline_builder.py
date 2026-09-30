"""Khởi tạo danh sách PIPELINE cho thực nghiệm: Baseline · Single · Hybrid (yêu cầu #4).

Mọi kỹ thuật resampling được ghép bằng **`imblearn.pipeline.Pipeline`** (không dùng pipeline chuẩn của
scikit-learn) để `fit_resample` chỉ chạy trên dữ liệu train của từng fold — đây là điều kiện bắt buộc
để chống rò rỉ dữ liệu khi vào Cross-Validation.

Danh mục (17 pipeline):

| Nhóm | Pipeline |
|---|---|
| Baseline | `baseline` (boosting mặc định, không can thiệp) |
| Single — Data-level | `ros`, `smote`, `borderline_smote`, `adasyn`, `rus`, `tomek`, `enn` |
| Single — Algorithm-level | `class_weight` (`class_weight='balanced'`), `focal_loss` (custom loss) |
| Single — Ensemble | `balanced_rf`, `easy_ensemble`, `balanced_bagging` |
| Hybrid | `smote_tomek`, `smote_enn` (resampling + cleaning), `smote_class_weight` (resampling + cost), `rusboost` (undersampling + boosting) |

Việc dựng sampler/metric/classifier được TÁI SỬ DỤNG từ `imbalance_lab` (đã có test riêng cho tỉ lệ
mục tiêu, grad/hess Focal Loss, và cơ chế chống rò rỉ) ⇒ thực nghiệm không viết lại công thức.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.base import BaseEstimator, ClassifierMixin

from .config import ExperimentConfig

LOGGER = logging.getLogger("imbalance_experiment")

#: Kỹ thuật bắt buộc theo đề bài: khoá → nhóm.
REQUIRED_TECHNIQUES: Dict[str, str] = {
    "baseline": "baseline",
    "ros": "single-data", "smote": "single-data", "borderline_smote": "single-data",
    "adasyn": "single-data", "rus": "single-data", "tomek": "single-data", "enn": "single-data",
    "class_weight": "single-algorithm", "focal_loss": "single-algorithm",
    "balanced_rf": "single-ensemble", "easy_ensemble": "single-ensemble",
    "balanced_bagging": "single-ensemble",
    "smote_tomek": "hybrid", "smote_enn": "hybrid", "smote_class_weight": "hybrid",
    "rusboost": "hybrid",
}

#: Thứ tự chạy (baseline trước để làm mốc so sánh).
TECHNIQUE_ORDER: Tuple[str, ...] = tuple(REQUIRED_TECHNIQUES)

#: Tên hiển thị của từng pipeline.
NAMES: Dict[str, str] = {
    "baseline": "Baseline (không xử lý)",
    "ros": "RandomOverSampler",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "rus": "RandomUnderSampler",
    "tomek": "Tomek Links (cleaning)",
    "enn": "Edited Nearest Neighbours (cleaning)",
    "class_weight": "Class Weights (balanced)",
    "focal_loss": "Focal Loss (custom objective)",
    "balanced_rf": "Balanced Random Forest",
    "easy_ensemble": "EasyEnsemble",
    "balanced_bagging": "Balanced Bagging",
    "smote_tomek": "HYBRID SMOTE + Tomek Links",
    "smote_enn": "HYBRID SMOTE + ENN",
    "smote_class_weight": "HYBRID SMOTE + Class Weights",
    "rusboost": "HYBRID RUSBoost (undersampling + boosting)",
}

#: Mô tả ngắn (in trong bảng và log).
NOTES: Dict[str, str] = {
    "baseline": "mốc so sánh: mô hình chuẩn trên dữ liệu gốc",
    "ros": "sao chép mẫu thiểu số tới tỉ lệ mục tiêu",
    "smote": "nội suy mẫu thiểu số (dễ làm mờ biên)",
    "borderline_smote": "chỉ nội suy từ mẫu thiểu số ở vùng DANGER",
    "adasyn": "sinh thêm theo độ khó của từng mẫu thiểu số",
    "rus": "hạ ngẫu nhiên lớp đa số (mất thông tin)",
    "tomek": "làm sạch biên: bỏ cặp láng giềng khác lớp",
    "enn": "làm sạch biên: bỏ mẫu bị láng giềng 'bỏ phiếu' khác lớp",
    "class_weight": "class_weight='balanced' — trọng số tính trong fit",
    "focal_loss": "Focal Loss: giảm trọng số mẫu dễ (gamma, alpha)",
    "balanced_rf": "rừng cây, mỗi cây undersample lớp đa số",
    "easy_ensemble": "nhiều AdaBoost trên các tập con cân bằng",
    "balanced_bagging": "bagging với các tập con được cân bằng",
    "smote_tomek": "SMOTE rồi DỌN cặp mẫu nhiễu (Tomek Links)",
    "smote_enn": "SMOTE rồi DỌN mẫu bị láng giềng phủ nhận (ENN)",
    "smote_class_weight": "SMOTE tỉ lệ vừa phải (0,5) + trọng số lớp",
    "rusboost": "boosting có undersample từng vòng lặp",
}

#: Yêu cầu thư viện/mô hình nền của từng pipeline (thiếu ⇒ bỏ qua kèm lý do).
REQUIRES: Dict[str, Tuple[str, ...]] = {
    "ros": ("imblearn",), "smote": ("imblearn",), "borderline_smote": ("imblearn",),
    "adasyn": ("imblearn",), "rus": ("imblearn",), "tomek": ("imblearn",), "enn": ("imblearn",),
    "balanced_rf": ("imblearn",), "easy_ensemble": ("imblearn",), "balanced_bagging": ("imblearn",),
    "smote_tomek": ("imblearn",), "smote_enn": ("imblearn",), "smote_class_weight": ("imblearn",),
    "rusboost": ("imblearn",),
    "focal_loss": ("lightgbm",),
}


@dataclass(frozen=True)
class PipelineSpec:
    """Một pipeline của thực nghiệm.

    Attributes:
        key: khoá kỹ thuật (dùng trong CLI/artifact).
        name: tên hiển thị.
        group: `baseline` | `single-data` | `single-algorithm` | `single-ensemble` | `hybrid`.
        note: mô tả ngắn.
        factory: hàm KHÔNG tham số trả về estimator mới (pipeline) cho mỗi fold.
        is_resampling: có bước lấy mẫu lại trong pipeline hay không.
        sampler_probe: pipeline CHỈ có sampler (`pipe[:-1]`) để đo/t log resampling mỗi fold.
        requires: thư viện cần có.
    """

    key: str
    name: str
    group: str
    note: str
    factory: Callable[[], Any]
    is_resampling: bool
    sampler_probe: Optional[Callable[[], Any]]
    requires: Tuple[str, ...] = ()



# ---------------------------------------------------------------------------
# Bộ phân loại nền & các estimator đặc biệt
# ---------------------------------------------------------------------------
def base_classifier(cfg: ExperimentConfig, *, seed: Optional[int] = None) -> Any:
    """Bộ phân loại NỀN (dùng chung cho baseline và mọi kỹ thuật để so sánh công bằng).

    - `lightgbm` (mặc định): tái sử dụng `imbalance_lab.models.make_base_classifier` (tự hạ cấp sang
      XGBoost/HistGradientBoosting nếu thiếu LightGBM).
    - `random_forest`: `RandomForestClassifier` của scikit-learn (không chỉnh trọng số lớp).
    """
    random_state = int(cfg.seed if seed is None else seed)
    if cfg.base_model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=60 if cfg.quick else 300, max_depth=6,
            min_samples_leaf=4 if cfg.quick else 2, n_jobs=1, random_state=random_state)

    from imbalance_lab.models import make_base_classifier

    return make_base_classifier(None, random_state=random_state)


def cost_sensitive_classifier(cfg: ExperimentConfig, *, seed: Optional[int] = None) -> Any:
    """Cost-sensitive: `class_weight='balanced'`.

    - `random_forest`: dùng thẳng `class_weight='balanced'` của scikit-learn.
    - `lightgbm`: dùng `imbalance_lab.models.BalancedWeightClassifier` (suy `class_weight` balanced bằng
      `compute_class_weight` TRONG `fit` rồi truyền `sample_weight`) — cùng công thức, không dùng hằng
      số toàn cục nên không rò rỉ.
    """
    random_state = int(cfg.seed if seed is None else seed)
    if cfg.base_model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=60 if cfg.quick else 300, max_depth=6,
            min_samples_leaf=4 if cfg.quick else 2, class_weight="balanced", n_jobs=1,
            random_state=random_state)

    from imbalance_lab.models import BalancedWeightClassifier

    return BalancedWeightClassifier(random_state=random_state)


class RobustRUSBoost(BaseEstimator, ClassifierMixin):
    """RUSBoost BỀN VỮNG: undersampling + boosting, có phương án dự phòng khi AdaBoost từ chối fit.

    Vì sao cần: `imblearn.ensemble.RUSBoostClassifier` dùng AdaBoost bên trong, mà AdaBoost có kiểm tra
    "base estimator phải TỐT HƠN NGẪU NHIÊN" trên tập con đã undersample. Với mất cân bằng rất cao
    (1:50, 1:100) kiểm tra này có thể thất bại và ném
    `ValueError: BaseClassifier in AdaBoostClassifier ensemble is worse than random`.

    Thứ tự thử (tất cả đều đúng tinh thần "undersampling + boosting"):
    1. `RUSBoostClassifier` với cây sâu 3 (cấu hình mặc định của imblearn);
    2. `RUSBoostClassifier` với STUMP (`max_depth=1`);
    3. `imblearn.pipeline.Pipeline([RandomUnderSampler, boosting])` — undersample một lần rồi boost.

    Bước nào thành công được ghi vào `self.implementation_` và log ra để artifact nói rõ đã chạy bản nào
    (không âm thầm đổi thuật toán). Mọi bước lấy mẫu đều nằm trong `fit` của tập train được truyền vào
    ⇒ vẫn an toàn khi vào Cross-Validation.
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RobustRUSBoost":
        """Fit theo thứ tự ưu tiên; ghi lại `implementation_` (đường đã chạy được)."""
        from imblearn.ensemble import RUSBoostClassifier
        from sklearn.tree import DecisionTreeClassifier

        n_estimators = 30 if self.cfg.quick else 100
        candidates = (
            ("RUSBoost(DecisionTree max_depth=3)",
             lambda: RUSBoostClassifier(
                 estimator=DecisionTreeClassifier(max_depth=3, random_state=int(self.cfg.seed)),
                 n_estimators=n_estimators, learning_rate=0.1, random_state=int(self.cfg.seed))),
            ("RUSBoost(stump max_depth=1)",
             lambda: RUSBoostClassifier(
                 estimator=DecisionTreeClassifier(max_depth=1, random_state=int(self.cfg.seed)),
                 n_estimators=n_estimators, learning_rate=0.1, random_state=int(self.cfg.seed))),
        )
        for label, build in candidates:
            try:
                estimator = build()
                estimator.fit(X, y)
                self.estimator_, self.implementation_ = estimator, label
                LOGGER.info("    rusboost: dùng %s", label)
                return self
            except Exception as exc:  # noqa: BLE001 - AdaBoost có thể từ chối vì lý do dữ liệu
                LOGGER.warning("    rusboost: %s không fit được (%s) — thử phương án tiếp theo",
                               label, type(exc).__name__)

        from imbalance_lab.samplers import build_sampler_pipeline
        from imbalance_lab.samplers import make_single_sampler

        steps = make_single_sampler("rus", prefer_imblearn=True,
                                    over_strategy=float(self.cfg.over_strategy),
                                    under_strategy=float(self.cfg.under_strategy),
                                    k_neighbors=int(self.cfg.k_neighbors),
                                    random_state=int(self.cfg.seed))
        pipeline = build_sampler_pipeline(steps, base_classifier(self.cfg), prefer_imblearn=True)
        pipeline.fit(X, y)
        self.estimator_ = pipeline
        self.implementation_ = "RandomUnderSampler + boosting (imblearn pipeline)"
        LOGGER.info("    rusboost: dùng %s", self.implementation_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Xác suất lớp dương từ estimator đã fit."""
        return self.estimator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nhãn dự đoán (ngưỡng 0.5)."""
        return self.estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Tham số công khai (giữ `cfg` để clone/pickle lại được)."""
        return {"cfg": self.cfg}


def _ensemble(cfg: ExperimentConfig, key: str) -> Any:  # pragma: no cover - phụ thuộc imblearn
    """Ensemble chuyên biệt cho dữ liệu mất cân bằng (imbalanced-learn).

    Cả bốn estimator đều lấy mẫu lại BÊN TRONG `fit` của chính tập train được truyền vào ⇒ an toàn khi
    vào Cross-Validation (không cần wrapper imblearn Pipeline cho bước lấy mẫu của chúng).
    """
    from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier,
                                   EasyEnsembleClassifier, RUSBoostClassifier)

    random_state = int(cfg.seed)
    if key == "balanced_rf":
        return BalancedRandomForestClassifier(
            n_estimators=50 if cfg.quick else 100, max_depth=6, sampling_strategy="auto",
            replacement=False, n_jobs=1, random_state=random_state)
    if key == "easy_ensemble":
        return EasyEnsembleClassifier(n_estimators=5 if cfg.quick else 10, sampling_strategy="auto",
                                      n_jobs=1, random_state=random_state)
    if key == "balanced_bagging":
        from sklearn.ensemble import RandomForestClassifier

        return BalancedBaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=10 if cfg.quick else 30, max_depth=6,
                                             n_jobs=1, random_state=random_state),
            n_estimators=5 if cfg.quick else 10, sampling_strategy="auto", replacement=False,
            random_state=random_state)
    # RUSBoost = boosting + undersample mỗi vòng; dùng wrapper bền vững vì AdaBoost có thể từ chối
    # fit trên dữ liệu mất cân bằng cực đoan (xem docstring `RobustRUSBoost`).
    return RobustRUSBoost(cfg)


def sampler_steps(cfg: ExperimentConfig, key: str) -> List[Tuple[str, Any]]:
    """Bước sampler (imblearn) của một kỹ thuật resampling."""
    from imbalance_lab.samplers import SINGLE_SAMPLERS, make_hybrid_sampler, make_single_sampler

    kwargs = {"over_strategy": float(cfg.over_strategy), "under_strategy": float(cfg.under_strategy),
              "k_neighbors": int(cfg.k_neighbors), "random_state": int(cfg.seed)}
    if key in ("smote_tomek", "smote_enn"):
        return make_hybrid_sampler(key, prefer_imblearn=True, **kwargs)
    if key == "smote_class_weight":
        return make_single_sampler("smote", prefer_imblearn=True, **kwargs)
    if key in SINGLE_SAMPLERS:
        return make_single_sampler(key, prefer_imblearn=True, **kwargs)
    raise KeyError(f"Kỹ thuật {key!r} không có bước sampler")



# ---------------------------------------------------------------------------
# Dựng pipeline & danh mục
# ---------------------------------------------------------------------------
def make_estimator(cfg: ExperimentConfig, key: str) -> Any:
    """Dựng estimator (pipeline) MỚI cho một kỹ thuật.

    - Baseline / algorithm-level / ensemble: trả thẳng estimator.
    - Resampling (đơn) và hybrid: `imblearn.pipeline.Pipeline([sampler..., classifier])` để
      `fit_resample` chỉ chạy trên dữ liệu mà pipeline được `fit` (train của fold).
    """
    if key == "baseline":
        return base_classifier(cfg)
    if key == "class_weight":
        return cost_sensitive_classifier(cfg)
    if key == "focal_loss":
        from imbalance_lab.losses import FocalLossClassifier

        return FocalLossClassifier(random_state=int(cfg.seed))
    if key in ("balanced_rf", "easy_ensemble", "balanced_bagging", "rusboost"):
        return _ensemble(cfg, key)

    from imbalance_lab.samplers import build_sampler_pipeline

    steps = sampler_steps(cfg, key)
    classifier = (cost_sensitive_classifier(cfg) if key == "smote_class_weight"
                  else base_classifier(cfg))
    pipeline = build_sampler_pipeline(steps, classifier, prefer_imblearn=True)
    checks = inspect_pipeline(pipeline, expect_samplers=True)
    if not checks["sampler_nằm_trong_pipeline"]:  # pragma: no cover - phòng vệ
        raise AssertionError(f"{key}: sampler không nằm trong pipeline ⇒ nguy cơ rò rỉ dữ liệu")
    return pipeline


def inspect_pipeline(pipeline: Any, *, expect_samplers: bool) -> Dict[str, Any]:
    """Kiểm tra cấu trúc pipeline: có dùng imblearn pipeline? có mấy bước sampler?

    Returns:
        dict gồm `imblearn_pipeline` (bool), `n_sampler_steps` (int) và
        `sampler_nằm_trong_pipeline` (bool đã thoả kỳ vọng `expect_samplers`).
    """
    from imbalance_lab.samplers import resampling_backend

    steps = list(getattr(pipeline, "steps", []))
    module = type(pipeline).__module__
    imblearn_ok = (module.startswith("imblearn.pipeline") if resampling_backend(True) == "imblearn"
                   else module.startswith("imbalance_lab"))
    n_samplers = sum(1 for _name, step in steps if hasattr(step, "fit_resample"))
    return {"imblearn_pipeline": bool(imblearn_ok), "n_sampler_steps": int(n_samplers),
            "sampler_nằm_trong_pipeline": bool(n_samplers > 0) if expect_samplers
            else bool(n_samplers == 0)}


def missing_requirements(cfg: ExperimentConfig, key: str) -> List[str]:
    """Thư viện/mô hình nền còn thiếu cho một kỹ thuật (rỗng = chạy được)."""
    from imbalance_lab.losses import FocalLossClassifier
    from imbalance_lab.samplers import resampling_backend

    gaps: List[str] = []
    for requirement in REQUIRES.get(key, ()):
        if requirement == "imblearn" and resampling_backend(True) != "imblearn":
            gaps.append("imbalanced-learn")
        elif requirement == "lightgbm" and not FocalLossClassifier.available():
            gaps.append("lightgbm")
    if key == "focal_loss" and cfg.base_model != "lightgbm":
        gaps.append(f"Focal Loss cần base_model=lightgbm (đang là {cfg.base_model})")
    return gaps


def build_pipelines(cfg: ExperimentConfig, *, keys: Tuple[str, ...] = ()) -> List[PipelineSpec]:
    """Danh mục pipeline (đủ 17 kỹ thuật theo đề bài), lọc theo `keys` nếu có.

    Args:
        cfg: cấu hình thực nghiệm.
        keys: chỉ dựng các kỹ thuật này (rỗng = tất cả, theo `TECHNIQUE_ORDER`).

    Raises:
        KeyError: nếu `keys` chứa khoá không có trong danh mục.
    """
    selected = list(keys) if keys else list(TECHNIQUE_ORDER)
    unknown = [key for key in selected if key not in REQUIRED_TECHNIQUES]
    if unknown:
        raise KeyError(f"Kỹ thuật không có trong danh mục: {unknown}; có {list(TECHNIQUE_ORDER)}")

    specs: List[PipelineSpec] = []
    for key in selected:
        is_resampling = key in ("ros", "smote", "borderline_smote", "adasyn", "rus", "tomek", "enn",
                                "smote_tomek", "smote_enn", "smote_class_weight")
        if is_resampling:
            steps = sampler_steps(cfg, key)

            def factory(steps=steps, key=key) -> Any:
                """Pipeline resampling + classifier (mỗi lần gọi là một instance MỚI)."""
                from imbalance_lab.samplers import build_sampler_pipeline

                classifier = (cost_sensitive_classifier(cfg) if key == "smote_class_weight"
                              else base_classifier(cfg))
                return build_sampler_pipeline(steps, classifier, prefer_imblearn=True)

            def probe(steps=steps) -> Any:
                """Pipeline CHỈ có sampler ⇒ gọi `fit_resample` để đo/ log phân phối sau resampling."""
                from imbalance_lab.samplers import build_sampler_pipeline

                return build_sampler_pipeline(steps, base_classifier(cfg),
                                              prefer_imblearn=True)[:-1]
        else:
            def factory(key=key) -> Any:  # type: ignore[misc]
                return make_estimator(cfg, key)

            probe = None  # type: ignore[assignment]
        specs.append(PipelineSpec(key=key, name=NAMES[key], group=REQUIRED_TECHNIQUES[key],
                                  note=NOTES[key], factory=factory, is_resampling=is_resampling,
                                  sampler_probe=probe, requires=REQUIRES.get(key, ())))
    return specs


def pipeline_table(specs: List[PipelineSpec]) -> str:
    """Bảng Markdown mô tả danh mục pipeline (nhóm · kỹ thuật · cách làm)."""
    lines = ["| Nhóm | Kỹ thuật | Mô tả | Resampling? |", "|---|---|---|---|"]
    for spec in specs:
        lines.append(f"| `{spec.group}` | `{spec.key}` — {spec.name} | {spec.note} | "
                     f"{'có (trong pipeline)' if spec.is_resampling else 'không'} |")
    return "\n".join(lines) + "\n"

