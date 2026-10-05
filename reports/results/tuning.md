# Tinh chỉnh hyperparameter (CV chia theo công ty)

| Mô hình | Cấu hình tốt nhất | CV-AP | CV-AUROC | Val-AP | Val-AUROC | Val-F1* |
|---|---|---:|---:|---:|---:|---:|
| logistic | `{'C': 0.01, 'class_weight': None}` | 0.960 | 0.917 | 0.989 | 0.974 | 0.976 |
| random_forest | `{'max_depth': 3, 'min_samples_leaf': 2, 'n_estimators': 500}` | 0.989 | 0.976 | 0.986 | 0.961 | 0.976 |
| hist_gradient_boosting | `{'learning_rate': 0.03, 'max_depth': 2, 'max_iter': 200}` | 0.939 | 0.934 | 0.988 | 0.970 | 0.976 |
| mlp | `{'alpha': 0.0001, 'hidden_layer_sizes': 64}` | 0.845 | 0.750 | 1.000 | 1.000 | 1.000 |

*(F1* = F1 tốt nhất trên validation; AP = average precision.)*

## logistic — 8 cấu hình

| # | Cấu hình | CV-AP | CV-AUROC |
|---:|---|---:|---:|
| 1 | `{'C': 0.01, 'class_weight': None}` | 0.960 | 0.917 |
| 2 | `{'C': 0.01, 'class_weight': 'balanced'}` | 0.960 | 0.916 |
| 3 | `{'C': 0.1, 'class_weight': 'balanced'}` | 0.940 | 0.918 |
| 4 | `{'C': 0.1, 'class_weight': None}` | 0.935 | 0.913 |
| 5 | `{'C': 1.0, 'class_weight': 'balanced'}` | 0.888 | 0.884 |
| 6 | `{'C': 1.0, 'class_weight': None}` | 0.886 | 0.882 |
| 7 | `{'C': 10.0, 'class_weight': 'balanced'}` | 0.851 | 0.836 |
| 8 | `{'C': 10.0, 'class_weight': None}` | 0.851 | 0.838 |

Mặc định: CV-AP = 0.935 (chênh +0.026)

## random_forest — 18 cấu hình

| # | Cấu hình | CV-AP | CV-AUROC |
|---:|---|---:|---:|
| 1 | `{'max_depth': 3, 'min_samples_leaf': 2, 'n_estimators': 500}` | 0.989 | 0.976 |
| 2 | `{'max_depth': 6, 'min_samples_leaf': 2, 'n_estimators': 500}` | 0.988 | 0.976 |
| 3 | `{'max_depth': 3, 'min_samples_leaf': 1, 'n_estimators': 500}` | 0.988 | 0.975 |
| 4 | `{'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 500}` | 0.987 | 0.974 |
| 5 | `{'max_depth': 6, 'min_samples_leaf': 4, 'n_estimators': 500}` | 0.987 | 0.973 |
| 6 | `{'max_depth': 3, 'min_samples_leaf': 4, 'n_estimators': 500}` | 0.987 | 0.971 |
| 7 | `{'max_depth': None, 'min_samples_leaf': 4, 'n_estimators': 500}` | 0.987 | 0.972 |
| 8 | `{'max_depth': 3, 'min_samples_leaf': 2, 'n_estimators': 200}` | 0.984 | 0.968 |
| 9 | `{'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 200}` | 0.984 | 0.967 |
| 10 | `{'max_depth': 6, 'min_samples_leaf': 2, 'n_estimators': 200}` | 0.984 | 0.968 |
| 11 | `{'max_depth': 3, 'min_samples_leaf': 4, 'n_estimators': 200}` | 0.984 | 0.966 |
| 12 | `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 500}` | 0.984 | 0.971 |

Mặc định: CV-AP = 0.986 (chênh +0.003)

## hist_gradient_boosting — 8 cấu hình

| # | Cấu hình | CV-AP | CV-AUROC |
|---:|---|---:|---:|
| 1 | `{'learning_rate': 0.03, 'max_depth': 2, 'max_iter': 200}` | 0.939 | 0.934 |
| 2 | `{'learning_rate': 0.03, 'max_depth': 2, 'max_iter': 400}` | 0.932 | 0.924 |
| 3 | `{'learning_rate': 0.1, 'max_depth': 2, 'max_iter': 400}` | 0.932 | 0.922 |
| 4 | `{'learning_rate': 0.1, 'max_depth': 3, 'max_iter': 400}` | 0.929 | 0.908 |
| 5 | `{'learning_rate': 0.1, 'max_depth': 2, 'max_iter': 200}` | 0.929 | 0.920 |
| 6 | `{'learning_rate': 0.03, 'max_depth': 3, 'max_iter': 200}` | 0.927 | 0.924 |
| 7 | `{'learning_rate': 0.03, 'max_depth': 3, 'max_iter': 400}` | 0.926 | 0.920 |
| 8 | `{'learning_rate': 0.1, 'max_depth': 3, 'max_iter': 200}` | 0.918 | 0.905 |

Mặc định: CV-AP = 0.924 (chênh +0.015)

## mlp — 6 cấu hình

| # | Cấu hình | CV-AP | CV-AUROC |
|---:|---|---:|---:|
| 1 | `{'alpha': 0.0001, 'hidden_layer_sizes': 64}` | 0.845 | 0.750 |
| 2 | `{'alpha': 0.001, 'hidden_layer_sizes': 64}` | 0.845 | 0.750 |
| 3 | `{'alpha': 0.01, 'hidden_layer_sizes': 64}` | 0.845 | 0.750 |
| 4 | `{'alpha': 0.0001, 'hidden_layer_sizes': 32}` | 0.799 | 0.741 |
| 5 | `{'alpha': 0.001, 'hidden_layer_sizes': 32}` | 0.799 | 0.741 |
| 6 | `{'alpha': 0.01, 'hidden_layer_sizes': 32}` | 0.799 | 0.741 |

Mặc định: CV-AP = 0.799 (chênh +0.046)
