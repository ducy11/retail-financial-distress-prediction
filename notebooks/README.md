# Notebooks

Nơi để notebook khám phá (EDA), thử nghiệm tính năng, vẽ biểu đồ cho báo cáo.

- Chạy thử nhanh từ terminal trước: `python -m scripts.eda`.
- Khởi tạo notebook: `jupyter notebook` hoặc dùng VS Code (extension Jupyter).
- Quy ước: notebook chỉ *đọc* dữ liệu trong `data/`, mọi logic dùng lại được
  đưa vào gói `forecasting/` (notebook import: `from forecasting.features import ...`).
- Đặt tên: `01-eda.ipynb`, `02-feature-analysis.ipynb`, ...; output lớn nên
  Clear trước khi commit.

## Notebook hiện có

| Notebook | Nội dung |
|---|---|
| `01-tong-quan-du-lieu.ipynb` | Tổng quan corpus (8 công ty/332 quý/324 mẫu), phân bố nhãn **theo công ty**, phân bố lớp theo tập, bảng so sánh mô hình/baseline, metric chính thức tại ngưỡng vận hành, 3 mẫu sai và **demo `scripts.predict`** (P(distress) + top-4 SHAP). Notebook **chỉ đọc** `reports/results/*.json` và gọi lại hàm của repo nên số liệu không thể lệch với báo cáo — cần chạy `python -m scripts.run_all` trước. |

