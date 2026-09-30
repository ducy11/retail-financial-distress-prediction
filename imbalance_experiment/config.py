"""Cấu hình thực nghiệm (một nguồn duy nhất) — dataclass bất biến, có type hint đầy đủ.

Mọi tham số của thực nghiệm nằm ở đây để `data_loader`, `pipeline_builder`, `evaluation` và `main`
dùng CHUNG một cấu hình ⇒ log/artifact luôn ghi lại đúng cấu hình đã chạy (tái lập được).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

#: Thư mục gốc repo (file này nằm trong `imbalance_experiment/`).
ROOT = Path(__file__).resolve().parents[1]
#: Nơi ghi artifact của thực nghiệm.
DEFAULT_OUT_DIR = ROOT / "reports" / "experiment"

#: Nhóm kỹ thuật theo yêu cầu đề bài (dùng cho báo cáo VÀ test đối chiếu).
GROUPS: Tuple[str, ...] = ("baseline", "single-data", "single-algorithm", "single-ensemble", "hybrid")


@dataclass(frozen=True)
class ExperimentConfig:
    """Toàn bộ tham số của một lần chạy thực nghiệm.

    Attributes:
        source: `"synthetic"` (mặc định) hoặc đường dẫn tới file CSV (ví dụ credit card fraud).
        target_column: tên cột nhãn khi đọc CSV (nhãn phải là 0/1).
        n_samples: số mẫu của dataset giả lập.
        imbalance_ratio: tỉ lệ đa số : thiểu số (50 ⇒ 1:50, tức lớp dương ≈ 1,96%).
        n_features/n_informative/n_redundant/class_sep/flip_y: tham số `make_classification`.
        test_size: tỉ lệ holdout test (chỉ dùng MỘT lần để chốt).
        seed: seed chung (chia tập, sampler, mô hình) để tái lập.
        base_model: `"lightgbm"` (mặc định) hoặc `"random_forest"` làm bộ phân loại nền.
        over_strategy/under_strategy/k_neighbors: tỉ lệ mục tiêu và số láng giềng của sampler.
        n_splits: số fold StratifiedKFold.
        threshold: ngưỡng quyết định khi báo cáo metric (0.5 = mặc định, không tinh chỉnh).
        quick: chế độ nhẹ (ít estimator hơn) cho thử nghiệm nhanh.
        techniques: danh sách khoá kỹ thuật cần chạy (rỗng = tất cả).
        out_dir: thư mục ghi artifact.
        write: có ghi artifact hay không.
    """

    # --- dữ liệu ---
    source: str = "synthetic"
    target_column: str = "Class"
    n_samples: int = 20_000
    imbalance_ratio: int = 50
    n_features: int = 20
    n_informative: int = 8
    n_redundant: int = 4
    class_sep: float = 1.6
    flip_y: float = 0.0
    test_size: float = 0.20
    seed: int = 42

    # --- mô hình nền & sampler ---
    base_model: str = "lightgbm"
    over_strategy: float = 0.50
    under_strategy: float = 0.50
    k_neighbors: int = 5

    # --- đánh giá ---
    n_splits: int = 5
    threshold: float = 0.5
    quick: bool = False
    techniques: Tuple[str, ...] = ()
    out_dir: Path = field(default=DEFAULT_OUT_DIR)
    write: bool = True

    @property
    def positive_rate(self) -> float:
        """Tỉ lệ lớp thiểu số (dương) suy từ `imbalance_ratio`."""
        return 1.0 / (float(self.imbalance_ratio) + 1.0)

    @property
    def class_weights(self) -> Tuple[float, float]:
        """(tỉ lệ lớp âm, tỉ lệ lớp dương) để truyền vào `make_classification(weights=...)`."""
        return (1.0 - self.positive_rate, self.positive_rate)

    def with_(self, **changes: Any) -> "ExperimentConfig":
        """Bản sao có ghi đè một vài trường (dataclass là bất biến)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """Cấu hình dạng dict để ghi JSON/artifact (Path → str)."""
        payload = dataclasses.asdict(self)
        payload["out_dir"] = str(self.out_dir)
        payload["class_weights"] = list(self.class_weights)
        payload["positive_rate"] = self.positive_rate
        return payload
