"""Tạo/tải dataset MẤT CÂN BẰNG và chia tập stratified (yêu cầu #2 của đề bài).

Hai nguồn dữ liệu:
1. `"synthetic"` (mặc định): `sklearn.datasets.make_classification` với tỉ lệ 1:50 (hoặc 1:100 qua
   `--imbalance-ratio`). Ưu điểm: chạy offline, tái lập tuyệt đối, biết chắc nhãn ⇒ đo được ảnh hưởng
   của từng kỹ thuật mà không lẫn nhiễu của dữ liệu thật.
2. Đường dẫn CSV: dùng cho `Credit Card Fraud Detection` (cột nhãn `Class`) hoặc bất kỳ CSV nhị phân
   nào. File CSV **không** đi kèm repo (dung lượng lớn) nên phải trỏ `--data path/to/creditcard.csv`.

Chia tập: `train_test_split(test_size=cfg.test_size, stratify=y, random_state=seed)` — giữ nguyên tỉ lệ
lớp; test tách TRƯỚC và chỉ dùng một lần để chốt kết quả.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """Dataset đã chia tập: train (dùng cho CV) + test (holdout, chốt một lần).

    Attributes:
        name: tên dataset để ghi log/artifact.
        X_train, y_train: tập train (CV chạy trên đây).
        X_test, y_test: tập holdout.
        meta: thông tin phụ (nguồn, tỉ lệ lớp, số đặc trưng...).
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]

    def distribution(self, y: np.ndarray) -> Dict[str, float]:
        """Tỉ lệ lớp dương và imbalance ratio (đa số/thiểu số) của một mảng nhãn."""
        y = np.asarray(y, dtype=int)
        n_positive = int((y == 1).sum())
        n_negative = int(len(y) - n_positive)
        return {"n": int(len(y)), "n_positive": n_positive, "n_negative": n_negative,
                "positive_pct": 100.0 * n_positive / len(y) if len(y) else float("nan"),
                "imbalance_ratio": (n_negative / n_positive) if n_positive else float("inf")}

    def describe(self) -> str:
        """Chuỗi một dòng để log: kích thước + tỉ lệ lớp của train/test."""
        train = self.distribution(self.y_train)
        test = self.distribution(self.y_test)
        return (f"{self.name}: train n={train['n']} (dương {train['n_positive']} = "
                f"{train['positive_pct']:.2f}%, IR={train['imbalance_ratio']:.1f}) | "
                f"test n={test['n']} (dương {test['n_positive']} = {test['positive_pct']:.2f}%, "
                f"IR={test['imbalance_ratio']:.1f}) | {self.X_train.shape[1]} đặc trưng")


def make_synthetic_dataset(cfg: ExperimentConfig) -> Dataset:
    """Sinh dataset mất cân bằng bằng `make_classification` rồi chia stratified."""
    X, y = make_classification(
        n_samples=int(cfg.n_samples),
        n_features=int(cfg.n_features),
        n_informative=int(cfg.n_informative),
        n_redundant=int(cfg.n_redundant),
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=float(cfg.class_sep),
        weights=list(cfg.class_weights),
        flip_y=float(cfg.flip_y),
        shuffle=True,
        random_state=int(cfg.seed),
    )
    X = X.astype(np.float64)
    y = y.astype(int)
    return _split_dataset(
        X, y,
        name=f"synthetic(1:{cfg.imbalance_ratio})",
        meta={"source": "synthetic", "imbalance_ratio_target": int(cfg.imbalance_ratio),
              "n_features": int(cfg.n_features), "n_informative": int(cfg.n_informative),
              "n_redundant": int(cfg.n_redundant), "class_sep": float(cfg.class_sep),
              "flip_y": float(cfg.flip_y), "seed": int(cfg.seed)},
        cfg=cfg)



def load_csv_dataset(cfg: ExperimentConfig) -> Dataset:
    """Đọc dataset nhị phân từ CSV (ví dụ `creditcard.csv`) rồi chia stratified.

    Hỗ trợ pandas nếu có (nhanh hơn với file lớn), ngược lại dùng `csv` thuần. Nhãn phải là 0/1 và
    cột nhãn do `cfg.target_column` chỉ định.
    """
    path = cfg.source
    meta_extra: Dict[str, Any]
    try:  # pragma: no cover - fast path phụ thuộc môi trường
        import pandas as pd

        frame = pd.read_csv(path)
        if cfg.target_column not in frame.columns:
            raise KeyError(f"CSV không có cột nhãn {cfg.target_column!r}; "
                           f"có {list(frame.columns)[:10]}")
        y = frame[cfg.target_column].to_numpy().astype(int)
        X = frame.drop(columns=[cfg.target_column]).to_numpy().astype(np.float64)
        meta_extra = {"reader": "pandas"}
    except ImportError:  # pragma: no cover - môi trường không có pandas
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            if cfg.target_column not in header:
                raise KeyError(f"CSV không có cột nhãn {cfg.target_column!r}")
            target_index = header.index(cfg.target_column)
            feature_indexes = [i for i in range(len(header)) if i != target_index]
            rows: List[List[float]] = []
            labels: List[int] = []
            for row in reader:
                if not row:
                    continue
                labels.append(int(float(row[target_index])))
                rows.append([float(row[i]) for i in feature_indexes])
        X = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        meta_extra = {"reader": "csv"}

    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("Nhãn phải là 0/1 (bài toán phân loại nhị phân)")
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return _split_dataset(X, y, name=f"csv:{name}",
                          meta={"source": str(path), **meta_extra}, cfg=cfg)


def _split_dataset(X: np.ndarray, y: np.ndarray, *, name: str, meta: Dict[str, Any],
                   cfg: ExperimentConfig) -> Dataset:
    """Chia stratified train/test (giữ nguyên tỉ lệ lớp) và kiểm tra chênh lệch tỉ lệ."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(cfg.test_size), stratify=y, random_state=int(cfg.seed), shuffle=True)
    dataset = Dataset(name=name, X_train=X_train, y_train=y_train.astype(int), X_test=X_test,
                      y_test=y_test.astype(int), meta=meta)
    train_rate = dataset.distribution(dataset.y_train)["positive_pct"]
    test_rate = dataset.distribution(dataset.y_test)["positive_pct"]
    if abs(train_rate - test_rate) > 1.0:
        LOGGER.warning("Tỉ lệ lớp train/test lệch %.2f điểm %% — kiểm tra lại stratify",
                       abs(train_rate - test_rate))
    LOGGER.info("Đã chia tập — %s", dataset.describe())
    return dataset


def load_dataset(cfg: ExperimentConfig) -> Dataset:
    """Nạp dataset theo `cfg.source` (`"synthetic"` hoặc đường dẫn CSV)."""
    if str(cfg.source).lower() in ("synthetic", "make", ""):
        return make_synthetic_dataset(cfg)
    return load_csv_dataset(cfg)


def stratified_folds(y: np.ndarray, cfg: ExperimentConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Danh sách (train_index, val_index) của `StratifiedKFold` — mỗi fold GIỮ NGUYÊN tỉ lệ lớp."""
    splitter = StratifiedKFold(n_splits=int(cfg.n_splits), shuffle=True, random_state=int(cfg.seed))
    folds = [(train_index, val_index)
             for train_index, val_index in splitter.split(np.zeros(len(y)), y)]
    if len(folds) != int(cfg.n_splits):  # pragma: no cover - phòng vệ
        raise AssertionError(f"StratifiedKFold trả {len(folds)} fold, mong đợi {cfg.n_splits}")
    return folds
