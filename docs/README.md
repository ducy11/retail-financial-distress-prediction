# Báo cáo / tài liệu của đồ án

Các file trong thư mục này mô tả quyết định lớn và kết quả để review (không phải mã chạy).

## Tài liệu VIẾT TAY (do người viết, không bị `scripts.make_report` ghi đè)

| File | Nội dung |
|---|---|
| `bo-tai-lieu-bao-ve.md` | **Bộ tài liệu bảo vệ:** Phần 1 — ML Factsheet (bài toán, dữ liệu, nhãn, split, EDA, tiền xử lý, siêu tham số, bảng benchmark ≥3 mô hình, confusion matrix, error analysis, overfit); Phần 2 — dàn 11 slide kèm takeaway/bullet/speaker notes; Phần 3 — 8 câu hỏi phản biện theo khuôn 4 bước (thừa nhận → kỹ thuật → số liệu → kết luận) + 2 phụ lục (6 con số "đinh" và bản đồ artifact ↔ số liệu) |
| `checklist-doi-chieu-yeu-cau.md` | **Đối chiếu 21 tiêu chí** → bằng chứng → lệnh kiểm chứng; 3 điểm nhấn khi trình bày; bảng "bị hỏi X thì mở file nào"; việc còn lại P0–P2 (trong đó **ETL port + nhãn quy tắc + nhãn sự kiện + walk-forward** đã xong) |
| `slide-bao-ve.md` | **Deck bảo vệ 11 slide** (cùng định dạng `slide.md` nên `export_office` xuất được `.pptx`) kèm lời thoại |
| `model-card.md` | Model card: mục đích, phạm vi, giới hạn, cách dùng/cách không dùng |
| `ke-hoach-tiep-theo.md` | Lộ trình P0/P1/P2 sau khi đóng các khoảng trống phản biện |
| `cac-ky-thuat-mat-can-bang.md` | Lab mất cân bằng 98/2: 17 pipeline, metric, kết quả |
| `benchmark-mat-can-bang.md` | Benchmark 95/5 (Non-E Mode vs E-Mode), 11 phương pháp |
| `mo-rong-du-lieu.md` | Chính sách mở rộng dữ liệu & provenance từng chỉ tiêu |

## Tài liệu SINH TỰ ĐỘNG (chạy `python -m scripts.make_report` để cập nhật)

`BAO-CAO.md` (báo cáo đầy đủ) · `slide.md` (dàn slide tự động) · `dinh-nghia-nhan.md` ·
`huong-dan-tai-lap.md`. Xuất Word/PowerPoint: `python -m scripts.export_office` →
- `BAO-CAO.docx` — **báo cáo hoàn chỉnh** (bảng + hình), để đọc/nộp;
- `BAO-CAO-slide.pptx` — slide tự động 14 mục (khớp artifact);
- `BAO-CAO-slide-bao-ve.pptx` + `BAO-CAO-slide-bao-ve.docx` — **deck bảo vệ 11 slide** (bản `.docx`
  giữ nguyên takeaway/bullet/lời thoại để **dựng slide** theo ý mình);
- `bo-tai-lieu-bao-ve.docx` — **bộ tài liệu bảo vệ đầy đủ** (factsheet + dàn 11 slide + 8 Q&A phản biện).

