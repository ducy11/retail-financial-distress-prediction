# Công thức dùng trong đồ án — bảng tra cứu (kèm nơi cài đặt + số thực)

*Tài liệu VIẾT TAY. Mục đích: liệt kê **mọi công thức đang thực sự chạy trong mã**, kèm vị trí
(`file:dòng`), hằng số, và số thực lấy từ artifact. Công thức chuẩn nhưng **không** dùng trong đồ án
được ghi rõ ở **mục 10** để tránh hiểu nhầm khi phản biện.*

## 0. Quy ước & nguồn số

| Ký hiệu | Nghĩa |
|---|---|
| `N`, `n₊`, `n₋` | số mẫu; số mẫu nhãn 1; số mẫu nhãn 0 |
| `y`, `p` | nhãn thật ∈ {0,1}; xác suất mô hình cho lớp 1 |
| `TP/FP/FN/TN` | ô của ma trận nhầm lẫn (nhãn 0/1, `labels=[0,1]`) |
| `K` (M) | số feature (47); trong SHAP dùng `M` |
| `C_FN`, `C_FP` | chi phí bỏ sót / báo động giả = **5 / 1** (repo chính), 10 / 1 (lab 98/2) |
| `IR` | imbalance ratio = đa số / thiểu số |

- Nguồn số: `reports/results/*.json|csv|md`, `reports/imbalance/*`, `reports/experiment/*`,
  `reports/benchmark_imbalanced*.md`. Không nhập tay số nào trong tài liệu này.
- Tập test của đồ án: **N = 64** (38 dương = 59,375%); train 212; validation 32; purged 16.

## 1. Dữ liệu, mẫu và nhãn

| # | Công thức / quy tắc | Ý nghĩa | Code | Số thực |
|---|---|---|---|---|
| 1.1 | mẫu = (lịch sử ≤ `LOOKBACK_QUARTERS`=8 quý với `available_on ≤ as_of`, quý target, nhãn) | chống rò rỉ thời gian | `forecasting/data.py`, `features._window` | 324 mẫu, 332 quý, 8 công ty |
| 1.2 | `TL = liabilities` nếu có, ngược lại `TL = TA − SE` (A = L + E) | tag `liabilities` chỉ phủ 39% quý | `features._total_liabilities` | đối chiếu 124 quý: 122 khớp tuyệt đối |
| 1.3 | `v_VND = v_USD × 25.000` | quy đổi minh hoạ (không phải tỷ giá lịch sử) | `data.VND_PER_USD` | — |
| 1.4 | tỷ số `r = a/b` nếu `b ≠ 0` và hữu hạn, ngược lại `NaN` | không suy diễn | `features._safe_ratio` | — |
| 1.5 | `YoY(x) = (x_t − x_{t−4}) / |x_{t−4}|` | so cùng kỳ (NaN nếu `x_{t−4}=0`) | `features._yoy` | cần ≥ 5 quý |
| 1.6 | nhãn `stress_signals`: `L = 1[Σ_{s=1..6} 1[s] ≥ m]`, `m = 1` | nhãn thay thế tái lập được | `labels.label_row` | khớp nhãn gốc tối đa **74,7%** |
| 1.7 | 6 tín hiệu: `NI<0`; `OCF<0`; `OI<0`; `CL>CA`; `SE<0`; `YoY(revenue) < −5%` | thiếu dữ liệu ⇒ tín hiệu = 0 (bảo thủ) | `labels.signal_flags` | — |
| 1.8 | Altman Z'': `Z″ = 6,56·WC/TA + 3,26·RE/TA + 6,72·EBIT/TA + 1,05·BV_E/TL` | công thức công khai 1968/2000, **không học tham số** | `labels.altman_z_double_prime` | nhãn 1 nếu `Z″ < 1,1` |
| 1.9 | `L = 1[∃ q ∈ {+1..+4}: Σ signals(q) ≥ 1]` | nhãn "sự kiện 4 quý tới" | `labels.label_forward_stress` | 3 định nghĩa nhãn, kết luận giữ ở 3/3 |
| 1.10 | ticker-prior: `p̂(x) = mean{ y_i : ticker_i = ticker(x) }` (tính trên train) | baseline "nhớ mặt công ty" | `baselines.ticker_prior` | test AUROC **0,9858** vs mô hình 0,9828 |
| 1.11 | Altman→xác suất: `p̂ = 1/(1+exp((Z″ − 1,1)/0,5))` | đưa quy tắc về cùng thang 0..1 | `baselines.altman_z_probability` | test AUROC 0,7581, F1 0,656 |

## 2. Đặc trưng — 47 cột

### 2.1. 14 tỷ số (`config.RATIOS`, `features.RATIO_PARTS`) — mỗi tỷ số có 2 cột: `_latest` và `_yoy`

| Tỷ số | Công thức |
|---|---|
| `gross_margin` | (revenue − cost_of_sales) / revenue |
| `operating_margin` | operating_income / revenue |
| `net_margin` | net_income / revenue |
| `sgna_pct_revenue` | selling_general_admin / revenue |
| `current_ratio` | current_assets / current_liabilities |
| `quick_ratio` | (current_assets − inventory) / current_liabilities |
| `debt_to_assets` | total_liabilities / total_assets |
| `debt_to_equity` | total_liabilities / stockholders_equity |
| `inventory_to_sales` | inventory / revenue |
| `receivables_to_sales` | receivables / revenue |
| `cash_to_assets` | cash_and_equivalents / total_assets |
| `ocf_to_sales` | operating_cash_flow / revenue |
| `retained_to_assets` | retained_earnings / total_assets |
| `revenue_per_asset` | revenue / total_assets |

### 2.2. Tăng trưởng, cấu trúc vốn, chỉ báo căng thẳng, nhóm `path`

| # | Công thức | Ghi chú | Code |
|---|---|---|---|
| 2.2.1 | `Δr = (r_latest − r_{t−4}) / |r_{t−4}|` cho cả 14 tỷ số | NaN nếu < 5 quý | `features._build_features` (1) |
| 2.2.2 | `g_x = (x_t − x_{t−4}) / |x_{t−4}|` cho 10 chỉ tiêu (`GROWTH_FIELDS`) | 10 cột `*_yoy_growth` | (2) |
| 2.2.3 | `working_capital_to_assets = (CA − CL) / TA` | nhóm `structure` | (3) |
| 2.2.4 | `distress_quarters_in_window = #{q ∈ window : OCF_q < 0}` | nhóm `stress` | (4) |
| 2.2.5 | `revenue_cv = σ(rev) / μ(rev)` | hệ số biến thiên (độ bất định doanh thu) | (4) |
| 2.2.6 | `current_ratio_min_window = min_q (CA_q / CL_q)` | cực trị **xấu nhất** trong cửa sổ | (5) |
| 2.2.7 | `ocf_to_sales_min_window = min_q (OCF_q / REV_q)` | " | (5) |
| 2.2.8 | `net_margin_min_window = min_q (NI_q / REV_q)` | " | (5) |
| 2.2.9 | `revenue_drawdown_window = REV_last / max_q(REV_q) − 1` | mức giảm so với đỉnh | (5) |
| 2.2.10 | `negative_ni_streak` = số quý `NI<0` **liên tiếp** ở cuối chuỗi | `_trailing_streak` | (5) |
| 2.2.11 | `negative_ocf_streak` = số quý `OCF<0` liên tiếp | " | (5) |

**Quy tắc số học an toàn (áp cho mọi feature):** `_last` bỏ qua NaN và lấy phần tử hữu hạn cuối;
`_min_finite` lấy min trên các phần tử hữu hạn; mọi phép chia đi qua `_safe_ratio` (mẫu = 0 ⇒ NaN).

**Nhóm feature dùng cho ablation (`features.FEATURE_GROUPS`) và kết quả thực (Random Forest, test):**

| Biến thể | #feature | Val AUROC | Test AUROC | Test F1 |
|---|---:|---:|---:|---:|
| tất cả feature | 47 | 0,9654 | **0,9828** | 0,9487 |
| bỏ `ratios_latest` (14) | 33 | 0,9524 | 0,9656 | 0,9487 |
| bỏ `ratios_yoy` (14) | 33 | 0,9740 | 0,9737 | 0,9487 |
| bỏ `growth` (10) | 37 | 0,9697 | 0,9848 | 0,9487 |
| bỏ `structure` (1) | 46 | 0,9610 | 0,9818 | 0,9487 |
| bỏ `stress` (2) | 45 | 0,9654 | 0,9858 | 0,9487 |
| bỏ `path` (6) | 41 | 0,9610 | 0,9767 | 0,9487 |

⇒ bỏ nhóm nào cũng **không** làm test tốt hơn một cách có ý nghĩa (chênh ≤ 0,003) ⇒ nút thắt của bài
toán **không** nằm ở feature mà ở **rò rỉ cấp thực thể** (mục 8).

## 3. Tiền xử lý (`forecasting/preprocessing.py`, `models.make_model`)

| # | Công thức | Ghi chú | Số thực |
|---|---|---|---|
| 3.0 | thứ tự pipeline: `[winsorize] → SimpleImputer(median) → [scaler] → estimator` | mọi bước **học trong `fit`** ⇒ chỉ train | pipeline chính: `winsorize = none` |
| 3.1 | impute: `x̂ = median(X_train[:, j])` cho ô thiếu | không rò rỉ thống kê | 703 ô thiếu ở SEC để `null` |
| 3.2 | winsorize IQR: `clip(x, Q1 − 1,5·IQR, Q3 + 1,5·IQR)` | `iqr_factor = 1,5` | 30,6% giá trị `debt_to_equity` ngoài IQR |
| 3.3 | winsorize phân vị: `clip(x, P1, P99)` | chế độ `p1p99` | `clip_share()` báo tỷ lệ bị clip |
| 3.4 | `z = (x − μ_train) / σ_train` (Standard) | logistic & MLP | `DEFAULT_SCALER` |
| 3.5 | `z = (x − median) / (Q3 − Q1)` (Robust) | `quantile_range=(25,75)` | thí nghiệm `preprocessing_experiment` |
| 3.6 | Yeo–Johnson (PowerTransformer, MLE cho λ) | `power` | " |
| 3.7 | Quantile → chuẩn: `x ↦ Φ⁻¹(rank(x)/(n+1))` | `n_quantiles = 200` | " |

## 4. Bộ chỉ số đánh giá (`forecasting/evaluation.py`, `labs/imbalance_lab/metrics.py`)

| # | Chỉ số | Công thức | Giá trị (test, ngưỡng vận hành 0,788) |
|---|---|---|---|
| 4.1 | ma trận nhầm lẫn | `confusion_matrix(y, ŷ, labels=[0,1])` | TN 25 / FP 1 / FN 2 / TP 36 |
| 4.2 | Accuracy *(chỉ chẩn đoán)* | `(TP+TN)/N` | 0,9531 (mốc đoán lớp đa số = **0,5938**) |
| 4.3 | Precision | `TP/(TP+FP)` | 0,9730 |
| 4.4 | Recall (TPR) | `TP/(TP+FN)` | 0,9474 |
| 4.5 | Specificity (TNR) | `TN/(TN+FP)` | 0,9615 |
| 4.6 | FPR | `FP/(TN+FP)` | 0,0385 |
| 4.7 | NPV | `TN/(TN+FN)` | 0,9259 |
| 4.8 | F1 | `2PR/(P+R)` | 0,9600 |
| 4.9 | F-macro | `(F1₀ + F1₁)/2` | 0,9517 |
| 4.10 | F-weighted | `Σ (n_c/N)·F1_c` | 0,9533 |
| 4.11 | F-beta (β=2, lab) | `(1+β²)PR/(β²P+R)` | lab: dùng trong `techniques.md` |
| 4.12 | MCC | `(TP·TN − FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | 0,9039 |
| 4.13 | Balanced accuracy | `(TPR + TNR)/2` | lab: 0,8212 (BalancedRF, E-Mode) |
| 4.14 | AUROC | `P(S₊>S₋) + ½P(hòa)` = `(1/(n₊n₋))ΣΣ[1(s_i>s_j)+½1(=)]` | 0,9828 |
| 4.15 | AP (PR-AUC) | `Σ_k (R_k − R_{k−1})·P_k` | 0,9908 |
| 4.16 | Brier | `(1/N)Σ(p_i − y_i)²` | 0,0580 |
| 4.17 | BSS *(audit)* | `1 − BS/BS_null`, `BS_null = p̄(1−p̄)` | 0,7594 (null 0,2412) |
| 4.18 | Chi phí kỳ vọng | `C_FN·FN + C_FP·FP` | 11,0 tại 0,788; 6,0 tại 0,7835 |

**Lưu ý đã đo (audit):** phân rã Murphy `BS = REL − RES + UNC` cho 0,0588 ≠ BS 0,0580 vì đẳng thức
chỉ đúng cho BS **đã chia bin**; ECE 0,1090; hiệu chuẩn-tổng-thể p̄ = 0,6281 vs ȳ = 0,5938.

## 5. Ngưỡng quyết định & quy tắc theo chi phí

| # | Công thức | Ý nghĩa | Code | Số thực |
|---|---|---|---|---|
| 5.1 | `E[Cost \| x] = C_FN·η·1[p̂<t] + C_FP·(1−η)·1[p̂≥t]` | chi phí kỳ vọng của một quyết định | `evaluation.metrics_at_threshold` | — |
| 5.2 | **`p* = C_FP/(C_FN+C_FP) = 1/(1+C_FN/C_FP)`** | ngưỡng Bayes tối ưu (dạng odds: `η/(1−η) ≥ C_FP/C_FN`) | — (dẫn xuất) | 5:1 ⇒ **p\* = 1/6 ≈ 0,1667** |
| 5.3 | `t_bestF1 = argmax_t F1(t)` trên đường PR | ngưỡng vận hành của đồ án | `evaluation.best_f1_point` | 0,7835 (test) / **0,7879 (chọn trên val)** |
| 5.4 | `t_cost = argmin_t (C_FN·FN(t) + C_FP·FP(t))` | ngưỡng theo chi phí | `evaluation.cost_optimal_threshold` | 0,7835 → cost 6,0 (test); 0,7879 → cost 5,0 (val) |
| 5.5 | `t_prec = argmax_t Recall(t)` với `Precision(t) ≥ 0,5` | chế độ `min_precision` (lab) | `labs/imbalance_lab/thresholds.py` | `techniques.md` |
| 5.6 | ứng viên ngưỡng = điểm của `precision_recall_curve` ∪ {0; 0,5; 1} (∪ lưới lượng tử 1001) | chỉ xét điểm thực của đường cong | `thresholds.candidate_thresholds` | — |

**Đã đo (audit):** `p* = 1/6` cho chi phí **15** trên test, trong khi argmin thực nghiệm (0,7835) cho 6
và ngưỡng vận hành (0,788) cho 11. Nguyên nhân: `p*` chỉ tối ưu khi `p̂` **đã hiệu chuẩn** (ECE ở đây
0,109), argmin trên chính tập đánh giá là đại lượng **in-sample**, và hàm chi phí là hàm bậc thang nên
argmin là một **plateau**. Vì vậy repo chọn ngưỡng trên **validation** rồi mới đo trên test.

## 6. Xử lý mất cân bằng (`labs/imbalance_lab/`, `labs/imbalance_experiment/`, `labs/benchmark.py`)

### 6.1. Đo mức mất cân bằng (`forecasting/eda.py`)

| # | Công thức | Code | Số thực |
|---|---|---|---|
| 6.1.1 | `IR = max(n₊,n₋) / min(n₊,n₋)` | `eda.imbalance_ratio` | corpus 1,68 (train 1,65; test 1,46); DKS 7,6 |
| 6.1.2 | `H = −Σ_c p_c·log₂ p_c` (bit) | `eda.shannon_entropy` | 1 bit = cân bằng |
| 6.1.3 | `H_norm = H / log₂ k` | `eda.normalized_entropy` | — |
| 6.1.4 | `Gini = 1 − Σ_c p_c²` | `eda.gini_impurity` | 0,5 = cân bằng nhị phân |
| 6.1.5 | `N_eff = 1 / Σ_c p_c²` | `eda.effective_number` | 8 công ty cân bằng = 8,0 |
| 6.1.6 | phân loại: `≥40%` balanced · `≥20%` slightly · `<20%` severely | `eda.imbalance_level` | HD/LOW/WMT = single_class (100% nhãn 1) |

### 6.2. Cấp thuật toán (cost-sensitive)

| # | Công thức | Code | Ghi chú |
|---|---|---|---|
| 6.2.1 | `class_weight='balanced'`: `w_c = N / (k · n_c)` | `models.HYPERPARAMS` (RF/HGB) | dùng cho RF `balanced_subsample` & HGB `balanced` |
| 6.2.2 | `balanced_subsample`: `w_c` tính lại **theo từng bootstrap sample** của mỗi cây | sklearn | RandomForest |
| 6.2.3 | `scale_pos_weight = n₋ / n₊` | `labs.benchmark.PosWeightClassifier` | tính **trong `fit`** ⇒ chỉ thấy train |
| 6.2.4 | Focal Loss: `FL = −[α·y·(1−p)^γ·ln p + (1−α)(1−y)·p^γ·ln(1−p)]` | `labs/imbalance_lab/losses.py` | `γ = 2,0`; `α = 0,75` (hoặc `n₋/N` động) |
| 6.2.5 | grad/hess giải tích theo logit (y=1, y=0) | `losses.focal_grad_hess` | kiểm chứng bằng sai phân số; `h ≥ 1e-6`; `γ=0` ⇒ weighted BCE |

### 6.3. Cấp dữ liệu (resampling — chỉ chạy trên train của từng fold, trong `imblearn.pipeline`)

| # | Công thức | Code |
|---|---|---|
| 6.3.1 | SMOTE: `x_new = x_i + δ·(x_{z_i} − x_i)`, `δ ~ U(0,1)`, `z_i` ∈ k-NN **thiểu số** (`k=5`) | `samplers.SMOTE` |
| 6.3.2 | kích thước đích: `n_new = round(s · n_maj) − n_min` (`s = 0,5` danh mục, 0,1 lab) | " |
| 6.3.3 | Borderline-SMOTE: vùng `DANGER = {xᵢ thiểu số : #đa số > #thiểu số trong m_neighbors}`; chỉ nội suy từ DANGER | `samplers.BorderlineSMOTE` |
| 6.3.4 | ADASYN: `r_i = Δ_i/k`; `r̂_i = r_i/Σr_j`; `g_i = r̂_i·G` (G = số mẫu cần sinh) | `samplers.ADASYN` |
| 6.3.5 | RandomOverSampler: sao chép ngẫu nhiên mẫu thiểu số tới `round(s·n_maj)` | `samplers.RandomOverSampler` |
| 6.3.6 | RandomUnderSampler: giữ trọn thiểu số + lấy ngẫu nhiên `n_maj' = round(n_min/s)` mẫu đa số | `samplers.RandomUnderSampler` |
| 6.3.7 | Tomek Links: cặp `(i,j)` **khác lớp** và là láng giềng gần nhất của nhau ⇒ bỏ mẫu **đa số** | `imblearn.TomekLinks` |
| 6.3.8 | ENN: bỏ mẫu nếu **đa số** trong `k` láng giềng có nhãn khác (`kind_sel="all"`, `sampling_strategy="auto"`) | `imblearn.EditedNearestNeighbours` |
| 6.3.9 | hybrid: `SMOTE → Tomek` / `SMOTE → ENN` / `SMOTE + class_weight` / `RUS + boosting` | `samplers.make_hybrid_sampler`, `imblearn` |

### 6.4. Cấp hậu xử lý

| # | Kỹ thuật | Công thức / cơ chế | Số thực |
|---|---|---|---|
| 6.4.1 | hiệu chuẩn lại sau resampling | `CalibratedClassifierCV` `isotonic` (hoặc Platt) + cross-fitting 5 fold | Brier 0,0051 → 0,0038; ngưỡng best-F1 **0,890 → 0,475** |
| 6.4.2 | ngưỡng chọn trên **xác suất out-of-fold của train** | không bao giờ chọn trên val/test | `thresholds.tune_thresholds_from_pr_curve` |
| 6.4.3 | ngưỡng theo chi phí (lab) | `C_FN = 10`, `C_FP = 1` | ngưỡng cost luôn thấp hơn ngưỡng best-F1 |

### 6.5. Kết quả thực của 3 nhóm (bộ giả lập 95/5, `reports/benchmark_imbalanced.md`)

| Nhóm | #phương pháp | PR-AUC (tb) | F1 thiểu số (tb) | Balanced Acc (tb) | Thời gian (s) |
|---|---:|---:|---:|---:|---:|
| Baseline | 2 | 0,4786 | 0,3582 | 0,6245 | 0,390 |
| Non-E Mode (data/cost) | 6 | 0,3211 | 0,3305 | 0,7111 | 0,284 |
| E-Mode (ensemble cân bằng) | 3 | 0,3398 | 0,3304 | **0,7352** | 1,417 |

⇒ E-Mode nâng Balanced Accuracy cao nhất (0,7352) nhưng **thời gian gấp ~5×** và PR-AUC vẫn thấp hơn
baseline XGBoost (0,6740) — kết luận "can thiệp cân bằng không phải nút thắt" được lặp lại ở **cả ba**
bộ dữ liệu (95/5, 1:50, và corpus thật 1,68).

## 7. Giải thích mô hình (`forecasting/explain.py`)

| # | Công thức | Ghi chú | Số thực |
|---|---|---|---|
| 7.1 | **Shapley:** `φ_j = Σ_{S ⊆ M\{j}} [|S|!·(M−|S|−1)!/M!]·[v(S∪{j}) − v(S)]` | định nghĩa chuẩn | — |
| 7.2 | **KernelSHAP weight:** `π(S) = (M−1) / [C(M,\|S\|)·\|S\|·(M−\|S\|)]` | 0 < \|S\| < M; ∅/M không xác định | `coalition_weight` — khớp công thức |
| 7.3 | `v(S) = E[f(x) \| x_S] ≈ (1/n_bg)·Σ_b f(x_S, x̄_{bg})` | Monte-Carlo trên 40 mẫu nền | `_mixed_predictions` |
| 7.4 | `φ = argmin_φ Σ_S π(S)·[v(S) − (φ₀ + Σ_{j∈S} φ_j)]²` (wLS) | `design = mask·√π`, `target = (v − E[f])·√π` | `kernel_shap_values` |
| 7.5 | **hiệu suất:** `f(x) = E[f(x)] + Σ_j φ_j` | được **cưỡng bức** đúng bằng phép chiếu đều | gap = **2,2e-16** (artifact) |
| 7.6 | `φ_j = w_j·(x_j − E[x_j])`, `φ₀ = w·E[x] + b` (f tuyến tính) | ground truth giải tích để tự kiểm | `linear_shap_exact`, `tests/test_explain.py` |
| 7.7 | `mean\|φ_j\|` trên các điểm giải thích | importance toàn cục | top-1 `current_ratio_latest` = 0,0765 |
| 7.8 | permutation importance = `AUROC(X) − AUROC(X với cột j hoán vị)` | đối chiếu độc lập, 5 lần lặp | Spearman 0,3618; Jaccard top-15 = 0,4286 |
| 7.9 | `VIF_j = 1 / (1 − R²_j)` (`R²_j` từ hồi quy `x_j` theo các cột còn lại) | đa cộng tuyến | **33/47** cột > 10; max 798,1 |
| 7.10 | learning curve: `gap = AUROC_train − AUROC_val` | theo cỡ tập train | train 162 → gap **0,1049** |

**Cảnh báo đã đo (audit):** `efficiency_gap ≈ 2,2e-16` là **tautology** (do phép chiếu ở 7.5), không đo
chất lượng ước lượng; với f phi tuyến (M=10, liệt kê đủ 2^M) sai số thật RMSE(φ̂) = 0,023 ở K=200 —
cùng bậc với các φ xếp hạng 9–15 (0,017–0,022) ⇒ thứ hạng top-15 không nên đọc như kết luận chắc chắn.

## 8. Thống kê, kiểm định, EDA, drift

| # | Công thức | Code | Số thực |
|---|---|---|---|
| 8.1 | **DeLong:** `V10_i = (1/n₋)Σ_j[1(s_i>s_j) + ½·1(s_i=s_j)]`, `ÂUC = mean(V10)` | `significance._placement_values` | khớp sklearn 6/6 cặp |
| 8.2 | `S10 = cov(V10)/n₊`, `S01 = cov(V01)/n₋` | `delong_test` | — |
| 8.3 | `Var(ΔÂUC) = (1/n₊)Var(V10^A−V10^B) + (1/n₋)Var(V01^A−V01^B)` | giữ **hiệp phương sai** giữa 2 mô hình | — |
| 8.4 | `z = ΔÂUC/√Var`, `p = 2(1−Φ(\|z\|))` | " | RF − ticker_prior: Δ = −0,0030, z = −0,3126, **p = 0,7546** |
| 8.5 | paired bootstrap **phân tầng theo lớp** cho ΔAP: CI 95% percentile (2,5/97,5), `p = 2·min(P(Δ≤0), P(Δ≥0))` | `paired_bootstrap` | ΔAP = +0,0061, CI [−0,0048; +0,0215], p = 0,344 |
| 8.6 | bootstrap CI của metric (2000 vòng) | `validation.bootstrap_ci` | AUROC [0,947; 1,000] |
| 8.7 | **cluster bootstrap** theo 8 công ty | `validation.cluster_bootstrap_ci` | AUROC [0,925; 1,000] (rộng hơn) |
| 8.8 | walk-forward: cửa sổ mở rộng theo `target_period_end` + **purge 90 ngày**, `min_train = 60` | `validation.walk_forward_folds` | HGB mean AUROC 0,9490 |
| 8.9 | điểm-biserial `r` + `p`; hiệu ứng hạng `\|2·AUC−1\|`; MI; lift decile | `eda.target_association` | top `current_ratio_min_window`: `\|2AUC−1\|` = 0,904, MI 0,432 |
| 8.10 | `lift@decile = mean(y \| x ≥ Q90) / mean(y)` | " | — |
| 8.11 | **BH-FDR:** `q_(i) = min_{j≥i} (p_(j)·m/j)` | `eda.benjamini_hochberg` | 47 feature × kiểm định ⇒ tránh ~2 phát hiện ngẫu nhiên |
| 8.12 | Pearson & Spearman theo cặp hoàn chỉnh; cụm `\|r\| ≥ 0,90` | `eda.correlation_analysis` | — |
| 8.13 | participation ratio `PR = (Σλ)²/Σλ²`; `n_components(95% phương sai)` | `eda.effective_dimensionality` | chỉ **12,89 chiều hiệu dụng** trên 47 cột |
| 8.14 | KS 2 mẫu; `SMD = (μ_new − μ_ref)/√((σ²_ref+σ²_new)/2)` | `eda.drift_analysis` | KS nặng nhất 0,621; 5 feature drift |
| 8.15 | `PSI = Σ_b (q_new,b − q_ref,b)·ln(q_new,b / q_ref,b)` | `eda.population_stability_index` | PSI > 0,2 ⇒ dịch chuyển |
| 8.16 | match rate ETL = `n_match / n_cells` | `scripts/prepare_sec.py` | **5.308/5.312 = 99,92%** |
| 8.17 | audit dữ liệu: số phép kiểm tra / số phát hiện | `scripts/audit_data.py` | **98.200 / 0** |

## 9. Tìm kiếm siêu tham số & giao thức đánh giá

| # | Công thức / quy tắc | Code | Số thực |
|---|---|---|---|
| 9.1 | random search **40 trial/mô hình**, tham số thang đo lấy **log-uniform** | `forecasting/search.sample_params` | 153 dòng `runs.csv` |
| 9.2 | hàm mục tiêu = **AP out-of-fold** trên `StratifiedGroupKFold` theo `ticker` | `search.cross_company_ap` | RF CV-AP 0,9855 |
| 9.3 | so với lưới: `Δ vs Grid`, `Δ vs mặc định` | `search.compare_with_grid` | RF: −0,0035 vs grid, +0,0045 vs mặc định |
| 9.4 | `GridSearchCV(refit=average_precision)` + `StratifiedGroupKFold` | `forecasting/tuning.py` | — |
| 9.5 | quy tắc chọn mô hình: `max AP cross-company → best-F1(val) → AP(val) → AUROC(val) → gap nhỏ nhất` | `summary.json::selection_rule` | chọn **Random Forest**, giữ cấu hình **mặc định** |
| 9.6 | 4 giao thức đánh giá | `validation_checks.json` | in-domain **0,9828** · cross-company OOF **0,9334** · LOCO **0,6280** · walk-forward 0,9489 |

## 10. Công thức chuẩn nhưng **KHÔNG** dùng trong đồ án (nói rõ khi phản biện)

| Công thức | Trạng thái trong đồ án | Lý do / thay thế bằng |
|---|---|---|
| **G-mean** `= √(TPR·TNR)` | **không có trong mã** (đã kiểm tra toàn repo) | dùng **Balanced accuracy = (TPR+TNR)/2**; nếu bị hỏi: trình bày công thức chuẩn rồi nêu lý do chọn trung bình cộng (nhạy hơn khi lớp thiểu số rất nhỏ) |
| **Focal Loss** | chỉ ở `labs/imbalance_lab/` (backend LightGBM) | **không** thuộc pipeline chính (4 họ thuần sklearn) |
| SMOTE / ADASYN / Borderline-SMOTE / ENN / Tomek | lab + thí nghiệm phụ | pipeline chính **cố ý không** resampling: nhãn gần như thuộc tính công ty ⇒ mẫu tổng hợp dễ rơi vào "vùng" của chính thực thể ⇒ hợp thức hoá rò rỉ cấp thực thể |
| XGBoost / LightGBM | đã **gỡ** khỏi pipeline chính | bảo đảm tái lập offline; boosting do `HistGradientBoosting` đảm nhiệm |
| trapezoid PR-AUC, DOR, Cohen κ, lift curve | không dùng | AP (average precision) + MCC + `lift_top_decile` |
| liệt kê 2^M liên minh cho SHAP | không dùng (M = 47) | KernelSHAP lấy mẫu 200 liên minh/điểm |

## 11. Bảng ánh xạ nhanh: nhóm công thức → file

| Nhóm | File chính |
|---|---|
| Mẫu, split, purge | `forecasting/data.py`, `data/prepared/manifest.json` |
| Nhãn (stress/Altman/forward) | `forecasting/labels.py` |
| Feature 47 cột | `forecasting/features.py`, `forecasting/config.py::RATIOS` |
| Tiền xử lý | `forecasting/preprocessing.py`, `forecasting/models.py` |
| Metric & ngưỡng | `forecasting/evaluation.py`, `labs/imbalance_lab/metrics.py`, `labs/imbalance_lab/thresholds.py` |
| Mất cân bằng | `labs/imbalance_lab/`, `labs/imbalance_experiment/`, `labs/benchmark.py`, `scripts/experiment_imbalance_real.py` |
| Giải thích | `forecasting/explain.py`, `scripts/explain_model.py` |
| Thống kê/kiểm định | `forecasting/significance.py`, `forecasting/validation.py` |
| EDA & drift | `forecasting/eda.py`, `scripts/eda.py`, `scripts/eda_deep.py` |
| Tìm kiếm siêu tham số | `forecasting/search.py`, `forecasting/tuning.py` |
| Kiểm chứng dữ liệu | `scripts/audit_data.py`, `scripts/verify_provenance.py`, `scripts/prepare_sec.py` |
