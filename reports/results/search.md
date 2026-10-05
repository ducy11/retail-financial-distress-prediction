# Tìm kiếm siêu tham số: random search + sổ thực nghiệm

- Phương pháp: random search (log-uniform cho tham số scale) + log MỌI trial; môi trường không có Optuna nên dùng cách này, không gian tham số mô tả ở `SEARCH_SPACES`
- Mục tiêu: AP out-of-fold, StratifiedGroupKFold theo công ty trên train+validation; seed 42; 40 trial/mô hình; winsorize=none.
- Sổ thực nghiệm: `results\runs.csv` — **153 dòng** (mỗi dòng = một trial, kèm params/seed/CV-AP/thời gian/trạng thái).

## 1. Kết quả từng mô hình

| Mô hình | #trial | CV-AP tốt nhất | ± độ lệch | CV-AP mặc định | CV-AP GridSearchCV | Δ vs Grid | Δ vs mặc định | Thời gian (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logistic | 40 | 0.9669 | 0.0095 | 0.9583 | 0.9603 | +0.0066 | +0.0086 | 1.5 |
| random_forest | 33 | 0.9855 | 0.0086 | 0.9810 | 0.9890 | -0.0035 | +0.0045 | 104.9 |
| hist_gradient_boosting | 40 | 0.9648 | 0.0314 | 0.9095 | 0.9386 | +0.0262 | +0.0553 | 36.2 |
| mlp | 40 | 0.9129 | 0.0894 | 0.7216 | 0.8452 | +0.0677 | +0.1913 | 7.7 |

**Cấu hình tốt nhất tìm được:**

- `logistic`: `{"C": 0.035474273997484596, "class_weight": "balanced", "max_iter": 1000}` → CV-AP 0.9668948290859732
- `random_forest`: `{"max_depth": 6, "min_samples_leaf": 2, "n_estimators": 800, "max_features": "log2"}` → CV-AP 0.9855306446217781
- `hist_gradient_boosting`: `{"learning_rate": 0.1289148645723342, "max_depth": 2, "max_iter": 200, "l2_regularization": 0.0017108461891527731}` → CV-AP 0.9648090717396978
- `mlp`: `{"hidden_layer_sizes": 64, "alpha": 0.07634702129562373, "learning_rate_init": 0.006112828938332493, "max_iter": 1000}` → CV-AP 0.912867733950903

> Đọc bảng: **Δ vs Grid** cho biết tìm kiếm ngẫu nhiên có vượt lưới GridSearchCV hay không (cùng thước đo CV-AP và cùng splitter). Δ ≈ 0 nghĩa là lưới cũ đã đủ tốt và chi phí thêm trial là lãng phí — kết luận này cũng được ghi lại thay vì chỉ khoe con số đẹp.
> Muốn dùng Bayesian search (Optuna TPE) khi môi trường có: thay `sample_params` bằng `trial.suggest_float(..., log=True)` với cùng các không gian ở `SEARCH_SPACES`; phần ghi sổ/so sánh giữ nguyên.

## 2. Hình

- `figures\search\01_search_distribution.png`
