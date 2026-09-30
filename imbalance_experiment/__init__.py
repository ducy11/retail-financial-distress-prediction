"""Thực nghiệm so sánh PHƯƠNG PHÁP ĐƠN LẺ vs PHƯƠNG PHÁP KẾT HỢP cho dữ liệu mất cân bằng.

Gói này là một **thực nghiệm độc lập** (không thuộc `scripts.run_all`), tái sử dụng các "viên gạch"
đã được kiểm thử của `imbalance_lab` (sampler, metric, Focal Loss, pipeline imblearn) để tránh viết
lại công thức — nhưng có cấu trúc module riêng theo yêu cầu:

| Module | Trách nhiệm |
|---|---|
| `data_loader.py`    | tạo/tải dataset mất cân bằng 1:50 (hoặc 1:100), chia train/test stratified, sinh fold stratified |
| `pipeline_builder.py` | danh mục pipeline: Baseline · Single (data/algorithm/ensemble) · Hybrid, tất cả qua `imblearn.pipeline.Pipeline` |
| `evaluation.py`     | chạy StratifiedKFold, thu metric từng fold, tính **mean ± std**, dựng dữ liệu PR curve |
| `insights.py`       | phân tích chuyên sâu bằng số liệu: SMOTE vs SMOTETomek/SMOTEENN, resampling vs cost-sensitive, khi nào hybrid thắng |
| `main.py`           | điều phối, in bảng Markdown (pandas nếu có), xuất artifact + biểu đồ PR curve |

Chạy nhanh:

```powershell
python -m imbalance_experiment.main --quick --n-samples 5000          # thử nghiệm mẫu (~1 phút)
python -m imbalance_experiment.main                                    # cấu hình đầy đủ 20.000 mẫu, 5 fold
python -m imbalance_experiment.main --imbalance-ratio 100 --cv 5       # mất cân bằng 1:100
```

**Chống rò rỉ dữ liệu**: mọi bước resampling nằm TRONG `imblearn.pipeline.Pipeline` nên
`fit_resample` chỉ chạy trên train của từng fold; validation của fold và tập test được so với bản sao
trước khi fit và ghi PASS/FAIL vào artifact.
"""
from __future__ import annotations

__all__ = ["config", "data_loader", "pipeline_builder", "evaluation", "insights", "main"]
