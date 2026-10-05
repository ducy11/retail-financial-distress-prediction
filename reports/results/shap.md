# Giải thích mô hình bằng SHAP (KernelSHAP tự cài đặt)

- Mô hình được giải thích: **random_forest**; ngưỡng vận hành 0.788.
- Phương pháp: KernelSHAP tự cài đặt (Lundberg & Lee 2017), π(S) = (M−1)/[C(M,|S|)·|S|·(M−|S|)]
- Quy mô: **64 mẫu** (validation 32 + test 32), 200 liên minh/điểm, nền 40 mẫu train; giá trị nền E[f] = 0.6391.

## 1. Tự kiểm chứng cài đặt

- **Efficiency**: sai số lớn nhất |Σφ + E[f] − f(x)| = 2.220e-16 (tương đối 6.337e-15).
- Σφ_j + E[f] phải bằng f(x); KernelSHAP ràng buộc đúng điều này nên sai số ≈ 0 (kiểm chứng cài đặt, không phải kết quả mô hình).
- Đối chiếu với permutation importance (phương pháp độc lập): Spearman = 0.3618137820239713, trùng top-15 = 0.42857142857142855.

## 2. Độ quan trọng toàn cục (mean |φ|)

| # | Feature | mean |φ| |
|---:|---|---:|
| 1 | current_ratio_latest | 0.0765 |
| 2 | current_ratio_min_window | 0.0764 |
| 3 | debt_to_assets_latest | 0.0583 |
| 4 | working_capital_to_assets | 0.0561 |
| 5 | debt_to_equity_latest | 0.0356 |
| 6 | quick_ratio_latest | 0.0230 |
| 7 | receivables_to_sales_latest | 0.0218 |
| 8 | gross_margin_latest | 0.0212 |
| 9 | retained_to_assets_latest | 0.0208 |
| 10 | operating_margin_latest | 0.0187 |
| 11 | ocf_to_sales_min_window | 0.0180 |
| 12 | inventory_yoy_growth | 0.0179 |
| 13 | debt_to_assets_yoy | 0.0176 |
| 14 | inventory_to_sales_latest | 0.0175 |
| 15 | total_assets_yoy_growth | 0.0172 |

> Đọc bảng: mean |φ| là mức đóng góp trung bình của feature vào log-odds quyết định. Đây là thước đo toàn cục, khác permutation importance (mức giảm metric khi hoán vị); hai phương pháp đồng thuận ⇒ kết luận về nhóm feature dẫn đầu không phụ thuộc một công cụ duy nhất. Khi feature đa cộng tuyến, SHAP chia "công" cho cả nhóm nên đọc kèm cụm tương quan ở `eda_deep.md`.

## 3. Giải thích cục bộ các mẫu dự đoán sai

|
 
M
ẫ
u
 
|
 
T
h
ự
c
 
t
ế
 
|
 
P
(
n
h
ã
n
 
1
)
 
|
 
F
e
a
t
u
r
e
 
đ
ẩ
y
 
v
ề
 
s
u
y
 
g
i
ả
m
 
(
φ
 
>
 
0
)
 
|
 
F
e
a
t
u
r
e
 
k
é
o
 
v
ề
 
a
n
 
t
o
à
n
 
(
φ
 
<
 
0
)
 
|


|
-
-
-
|
-
-
-
:
|
-
-
-
:
|
-
-
-
|
-
-
-
|


|
 
F
I
V
E
-
2
0
2
3
Q
3
 
|
 
1
 
|
 
0
.
1
3
2
 
|
 
o
p
e
r
a
t
i
n
g
_
i
n
c
o
m
e
_
y
o
y
_
g
r
o
w
t
h
 
(
+
0
.
0
7
)
,
 
n
e
g
a
t
i
v
e
_
n
i
_
s
t
r
e
a
k
 
(
+
0
.
0
6
)
,
 
q
u
i
c
k
_
r
a
t
i
o
_
y
o
y
 
(
+
0
.
0
5
)
,
 
g
r
o
s
s
_
m
a
r
g
i
n
_
l
a
t
e
s
t
 
(
+
0
.
0
4
)
 
|
 
c
u
r
r
e
n
t
_
r
a
t
i
o
_
m
i
n
_
w
i
n
d
o
w
 
(
-
0
.
1
2
)
,
 
c
u
r
r
e
n
t
_
r
a
t
i
o
_
l
a
t
e
s
t
 
(
-
0
.
1
1
)
,
 
d
e
b
t
_
t
o
_
a
s
s
e
t
s
_
l
a
t
e
s
t
 
(
-
0
.
1
0
)
,
 
w
o
r
k
i
n
g
_
c
a
p
i
t
a
l
_
t
o
_
a
s
s
e
t
s
 
(
-
0
.
0
7
)
 
|


|
 
H
D
-
2
0
2
4
Q
2
 
|
 
1
 
|
 
0
.
7
8
3
 
|
 
d
e
b
t
_
t
o
_
a
s
s
e
t
s
_
l
a
t
e
s
t
 
(
+
0
.
0
9
)
,
 
d
e
b
t
_
t
o
_
e
q
u
i
t
y
_
l
a
t
e
s
t
 
(
+
0
.
0
8
)
,
 
r
e
c
e
i
v
a
b
l
e
s
_
t
o
_
s
a
l
e
s
_
l
a
t
e
s
t
 
(
+
0
.
0
4
)
,
 
o
p
e
r
a
t
i
n
g
_
m
a
r
g
i
n
_
l
a
t
e
s
t
 
(
+
0
.
0
3
)
 
|
 
c
u
r
r
e
n
t
_
r
a
t
i
o
_
m
i
n
_
w
i
n
d
o
w
 
(
-
0
.
0
7
)
,
 
c
u
r
r
e
n
t
_
r
a
t
i
o
_
l
a
t
e
s
t
 
(
-
0
.
0
5
)
,
 
s
g
n
a
_
p
c
t
_
r
e
v
e
n
u
e
_
l
a
t
e
s
t
 
(
-
0
.
0
3
)
,
 
d
e
b
t
_
t
o
_
a
s
s
e
t
s
_
y
o
y
 
(
-
0
.
0
2
)
 
|

> Đọc bảng: mỗi mẫu sai được phân rã thành các đóng góp dương (đẩy về phía suy giảm) và âm (kéo về phía an toàn) — đây là căn cứ giải trình từng hồ sơ, thứ mà permutation importance (toàn cục) không cung cấp được.

## 4. Hình

- `figures\shap\01_shap_summary.png`
- `figures\shap\02_shap_beeswarm.png`
- `figures\shap\03_shap_local_errors.png`
