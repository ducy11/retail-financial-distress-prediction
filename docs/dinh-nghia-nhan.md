# Truy vết nhãn `is_distressed` (bắt buộc đọc trước khi dùng kết quả)

*Sinh tự động bởi `python -m scripts.make_report` từ `reports/results/analysis.json` và `reports/results/relabel.json`.*

## 1. Nhãn gốc đến từ đâu?

Nhãn trong `data/prepared/*.json` **được giữ nguyên** từ pipeline sinh dữ liệu ban đầu (`scripts.prepare_sec` — hiện **chưa được port**, hàm `main` trả mã lỗi 2 và ghi rõ “CHƯA CÀI ĐẶT”). `forecasting/data.py::_build_samples` chỉ dùng heuristic `net_income < 0` cho mẫu **hoàn toàn mới**; với dữ liệu hiện có, nhãn được đọc lại theo `sample_id`.

## 2. Nhãn gốc có tái tạo được không? — KHÔNG

Kiểm chứng tự động (`scripts/analyze.py::label_audit`) áp từng quy tắc kế toán đơn giản trên 308 mẫu (train + validation + test) và so với nhãn gốc:

| Quy tắc (trên quý target) | Khớp nhãn gốc | Khớp khi áp lên dòng lịch sử cuối |
|---|---:|---:|
| `net_income<0` | 39.6% | 39.6% |
| `operating_income<0` | 39.3% | 39.3% |
| `operating_cash_flow<0` | 40.3% | 40.3% |
| `ocf<0 hoặc ni<0` | 40.9% | 40.9% |
| `current_liabilities>current_assets` | 65.3% | 65.3% |
| `stockholders_equity<0` | 51.9% | 51.9% |
| `retained_earnings<0` | 54.2% | 54.2% |
| `stress_signals>=1` | 74.7% | — |
| `stress_signals>=2` | 47.7% | — |

→ Mức khớp cao nhất: **74.7%** ⇒ nhãn gốc **không** tương ứng với bất kỳ quy tắc đơn giản nào có thể viết lại từ dữ liệu công bố.

**Phản chứng cụ thể:** `WMT-2015Q2` (quý 2014-05-01 → 2014-07-31) có net income dương (89.825 tỷ VND ≈ 3,59 tỷ USD) và operating cash flow dương, nhưng nhãn = 1.

## 3. Nhãn gần như là thuộc tính của CÔNG TY

| Công ty | Tỷ lệ nhãn = 1 |
|---|---:|
| HD | 1.00 |
| LOW | 1.00 |
| WMT | 1.00 |
| ORLY | 0.91 |
| DG | 0.55 |
| FIVE | 0.19 |
| DKS | 0.12 |
| ROST | 0.05 |

- Công ty có 100% nhãn = 1: **HD, LOW, WMT**.
- Công ty có 0% nhãn = 1: **—** (nếu danh sách rỗng nghĩa là mọi công ty đều có ít nhất một quý nhãn 1).
- Hệ quả: ở chế độ in-domain, mô hình chỉ cần nhận ra công ty là đạt AUROC rất cao — đây là lý do đồ án bổ sung baseline ticker-prior và đánh giá cross-company (GroupKFold/LOCO).

## 4. Nhãn thay thế tái lập được (dùng để kiểm chứng độ nhạy)

`forecasting/labels.py` định nghĩa nhãn công khai: `is_distressed_rule = 1` nếu quý target có ≥ **1** tín hiệu trong 6 tín hiệu căng thẳng tài chính:

- net_income<0: Lỗ ròng trong quý target
- operating_cash_flow<0: Dòng tiền hoạt động âm
- operating_income<0: Lỗ hoạt động
- current_liabilities>current_assets: Vốn lưu động âm (thanh khoản ngắn hạn)
- stockholders_equity<0: Vốn chủ sở hữu âm
- revenue_yoy<-5%: Doanh thu giảm hơn 5% so với cùng kỳ

- Nhãn được tính trên **quý target** — dữ liệu chỉ công bố ở `label_available_on` (sau `as_of`) nên **không** rò rỉ vào feature.
- Số mẫu dương tính theo nhãn quy tắc: `{'train': 88, 'validation': 14, 'test': 27, 'purged': 8}`.
- Mức khớp với nhãn gốc: **74.4%** — đủ khác để phép thử độ nhạy có ý nghĩa.

Chạy lại nhánh dữ liệu này:

```powershell
python -m scripts.relabel              # sinh data/prepared-rule + so sánh hai định nghĩa nhãn
```

## 5. Khuyến nghị khi sử dụng kết quả

1. **Không** diễn giải AUROC in-domain là “khả năng dự báo suy giảm tài chính”.
2. Luôn kèm baseline ticker-prior và kết quả cross-company khi báo cáo.
3. Nếu dùng cho nghiên cứu tiếp: nên **tái sinh nhãn** bằng định nghĩa công khai (mục 4) hoặc quy tắc học thuật (Altman Z-score / O-score) rồi chạy lại toàn bộ pipeline.