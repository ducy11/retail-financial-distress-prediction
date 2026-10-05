# Kiểm chứng ETL (`scripts/prepare_sec` — bản port)

- Công ty đối chiếu: **8** (tái tạo được: **8**)
- Ô đối chiếu: **5312** — khớp **5308** (**99.92%**)
- Quy đổi minh hoạ: USD → VND × 25000
- Bảng tái tạo ghi vào: `E:\CS114\Do-an\do-an\data\retail-expanded-rebuilt`

| Công ty | Ô | Khớp | Tỷ lệ | Ô trống cả hai bên | Số quý SINH từ snapshot |
|---|---:|---:|---:|---:|---:|
| DG | 512 | 512 | 100.00% | 92 | 56 |
| DKS | 704 | 700 | 99.43% | 167 | 51 |
| FIVE | 512 | 512 | 100.00% | 44 | 56 |
| HD | 704 | 704 | 100.00% | 44 | 69 |
| LOW | 704 | 704 | 100.00% | 60 | 64 |
| ORLY | 704 | 704 | 100.00% | 88 | 65 |
| ROST | 704 | 704 | 100.00% | 112 | 66 |
| WMT | 768 | 768 | 100.00% | 96 | 69 |

**Đọc bảng:** cột cuối là số quý mà đường SINH TỰ ĐỘNG (`discover_periods`, không dùng dữ liệu gốc làm khung) tìm được từ snapshot — dùng để chạy ETL cho công ty MỚI. Cột `Khớp` là phép kiểm chứng ngược: tái tạo lại từng ô của dữ liệu đang dùng cho báo cáo, chỉ bằng snapshot SEC trong `data/sec/raw`.

## Ô lệch đầu tiên (tối đa 25/công ty)

### DKS

| Mẫu | Chỉ tiêu | Đã công bố | Tái tạo | Phương pháp | Tag |
|---|---|---:|---:|---|---|
| DKS-2015Q4 | operating_income | 5194825000000 | 5194850000000 | reported_quarter | OperatingIncomeLoss |
| DKS-2016Q4 | operating_income | 3455375000000 | 3455350000000 | reported_quarter | OperatingIncomeLoss |
| DKS-2017Q4 | operating_income | 4457850000000 | 4457875000000 | reported_quarter | OperatingIncomeLoss |
| DKS-2019Q4 | operating_income | 2472800000000 | 2472775000000 | reported_quarter | OperatingIncomeLoss |
