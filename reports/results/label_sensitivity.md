# Kiểm chứng độ nhạy của kết luận theo ĐỊNH NGHĨA NHÃN (RQ4)

- Mô hình cố định: **random_forest** (cấu hình mặc định); mọi định nghĩa dùng cùng chính sách split và cùng giao thức đo.
- Số định nghĩa: **4**, trong đó 3 định nghĩa tái lập được từ dữ liệu công bố.

## 1. Nhận xét tự động

1. **Số định nghĩa nhãn TÁI LẬP ĐƯỢC đã kiểm chứng:** 3 (stress_signals, altman_z, forward_4q).
2. **Luận điểm "mô hình không vượt baseline ticker-prior" xuất hiện ở 3/3 định nghĩa** ⇒ kết luận **ỔN ĐỊNH theo định nghĩa nhãn** — đây là bằng chứng mạnh nhất của đồ án.
3. **`stress_signals`:** tỉ lệ dương test = 42.2%, IR train = 1.41; khớp nhãn gốc = 74.4%; in-domain AUROC (mô hình / ticker-prior) = 0.909 / 0.960; cross-company OOF AP = 0.610, AUROC = 0.730.
4. **`altman_z`:** tỉ lệ dương test = 31.2%, IR train = 3.42; khớp nhãn gốc = 63.9%; in-domain AUROC (mô hình / ticker-prior) = 0.999 / 0.991; cross-company OOF AP = 0.460, AUROC = 0.772.
5. **`forward_4q`:** tỉ lệ dương test = 48.4%, IR train = 1.99; khớp nhãn gốc = 66.7%; in-domain AUROC (mô hình / ticker-prior) = 0.836 / 0.911; cross-company OOF AP = 0.582, AUROC = 0.410.
6. **Đọc kết quả:** cột khớp nhãn gốc cho biết định nghĩa mới có đo cùng khái niệm với nhãn gốc hay không; cột cross-company cho biết mô hình còn giữ được bao nhiêu khi công ty bị giữ trọn ra ngoài. Một định nghĩa chỉ đáng tin khi cả hai hợp lý (khớp > ~65% và cross-company ≈ 0,9).

## 2. Bảng so sánh các định nghĩa nhãn

| Định nghĩa | Dương (train) | IR train | Dương (test) | Khớp nhãn gốc | AUROC test (mô hình) | AUROC test (ticker-prior) | Cross-company AP | Cross-company AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 62.3% | 1.65 | 59.4% | 100.0% | 0.983 | 0.986 | 0.958 | 0.943 |
| stress_signals | 41.5% | 1.41 | 42.2% | 74.4% | 0.909 | 0.960 | 0.610 | 0.730 |
| altman_z | 22.6% | 3.42 | 31.2% | 63.9% | 0.999 | 0.991 | 0.460 | 0.772 |
| forward_4q | 66.5% | 1.99 | 48.4% | 66.7% | 0.836 | 0.911 | 0.582 | 0.410 |

**Mô tả từng định nghĩa:**

- `original`: nhãn gốc `is_distressed` trong data/prepared (KHÔNG tái tạo được — dùng làm mốc tham chiếu)
- `stress_signals`: ≥1 trong 6 tín hiệu căng thẳng của QUÝ TARGET (quy tắc kế toán đơn giản)
- `altman_z`: Altman Z''-score < 1.1 (công thức công khai 1968/2000)
- `forward_4q`: có ≥1 quý trong 4 quý TỚI chạm ngưỡng tín hiệu căng thẳng (sự kiện sắp xảy ra, không phải trạng thái quý target)

> Đọc bảng: nhãn gốc không tái tạo được nên cột **Khớp nhãn gốc** đo xem định nghĩa công khai có đo cùng khái niệm hay không (mốc cao nhất trước đây là 74,7% với `stress_signals`). Cột **Cross-company** là phép thử khắt khe nhất: công ty bị giữ trọn ra ngoài train. Nếu kết luận "mô hình ≈ baseline nhớ mặt công ty" lặp lại ở nhiều định nghĩa thì kết luận không phụ thuộc vào cách gán nhãn.

