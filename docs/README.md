# Báo cáo / tài liệu của đồ án

Các file trong thư mục này mô tả quyết định lớn và kết quả để review (không phải mã chạy).

## Tài liệu VIẾT TAY (do người viết, không bị `scripts.make_report` ghi đè)

| File | Nội dung |
|---|---|
| `bo-tai-lieu-bao-ve.md` | **Bộ tài liệu bảo vệ:** Phần 1 — ML Factsheet (bài toán, dữ liệu, nhãn, split, EDA, tiền xử lý, siêu tham số, bảng benchmark ≥3 mô hình, confusion matrix, error analysis, overfit); Phần 2 — dàn 11 slide kèm takeaway/bullet/speaker notes; Phần 3 — 10 câu hỏi phản biện theo khuôn 4 bước (thừa nhận → kỹ thuật → số liệu → kết luận) + 2 phụ lục (6 con số "đinh" và bản đồ artifact ↔ số liệu) |
| `checklist-doi-chieu-yeu-cau.md` | **Đối chiếu 21 tiêu chí** → bằng chứng → lệnh kiểm chứng; 3 điểm nhấn khi trình bày; bảng "bị hỏi X thì mở file nào"; việc còn lại P0–P2 (trong đó **ETL port + nhãn quy tắc + nhãn sự kiện + walk-forward** đã xong) |
| `slide-bao-ve.md` | **Deck bảo vệ 11 slide** (cùng định dạng `slide.md` nên `export_office` xuất được `.pptx`) kèm lời thoại |
| `slide-cho-ai.md` | **File Markdown TỰ CHỨA để sinh slide bằng công cụ AI** (Gamma/Canva/Tomei/ChatGPT) hoặc Marp/Slidev: 20 slide (bìa + 18 nội dung + cảm ơn) theo cấu trúc *thông điệp chính → gạch đầu dòng → lời thoại → Ảnh*, kèm Phụ lục A (toàn bộ số liệu gốc để AI không bịa số), Phụ lục B (danh mục ảnh cần chèn) và Phụ lục C (prompt gợi ý + checklist + lệnh Marp CLI). Dùng khi bản `.pptx` không mở được |
| `cong-thuc-do-an.md` | **Bảng tra cứu CÔNG THỨC của đồ án** (11 mục): dữ liệu & nhãn (6 tín hiệu, Altman Z″), 47 feature (14 tỷ số + YoY + growth + nhóm `path`), tiền xử lý (impute/winsorize/scaler), bộ chỉ số (F1/MCC/AUROC/AP/Brier), ngưỡng theo chi phí (`p* = C_FP/(C_FN+C_FP)`), mất cân bằng (class weight/SMOTE/ADASYN/Tomek/ENN/Focal Loss), SHAP (`π(S)`, hiệu suất), thống kê (DeLong/boostrap/BH-FDR/KS/SMD/PSI), tìm kiếm siêu tham số — mỗi công thức kèm `file:dòng` và **số thực từ artifact**; mục 10 liệt kê công thức chuẩn **không** dùng (G-mean, focal loss trong pipeline chính, SMOTE…) để tránh hiểu nhầm khi phản biện |
| `BAO-CAO-phan-bien.md` | **Báo cáo kỹ thuật toàn văn & giải trình phản biện 10 chương** (dùng để NỘP/đọc khi bảo vệ): Phần I 10 phút đầu theo barem đề cương (mục tiêu & dữ liệu → EDA → tiền xử lý → 4 họ mô hình → so sánh & chọn mô hình → per-class, confusion matrix, overfitting); Phần II giải trình sâu (công thức metric & ngưỡng Bayes `p* = 1/6` · 47 đặc trưng & Group Ablation · Non-E/E-Mode & LiteSVM/XGBoost đối chứng · tổng kết, cảnh báo rò rỉ thực thể LOCO 0,628 + cheat-sheet 10 câu + Phụ lục LaTeX). Xuất Word: `python -m scripts.export_office` → `BAO-CAO-phan-bien.docx` |
| `script-slide-bao-ve.md` | **Kịch bản thuyết trình** khớp 1:1 với deck Canva `Predicting_Retail_Distress_SEC_XBRL.pdf` (12 slide): mỗi slide có thời lượng mục tiêu → lời thoại viết như nói → số liệu phải nhấn → câu chuyển → việc cần sửa trên Canva; kèm **2 slide khuyến nghị bổ sung** (xử lý mất cân bằng · tìm kiếm siêu tham số), **Q&A 10 câu trả lời nói ngắn**, 3 câu trả lời dự phòng, **kế hoạch nén khi hết thời gian** và checklist trước khi in. Mọi số liệu đều truy được về artifact. Xuất Word: `python -m scripts.export_office` → `script-slide-bao-ve.docx` |
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
- `bo-tai-lieu-bao-ve.docx` — **bộ tài liệu bảo vệ đầy đủ** (factsheet + dàn 11 slide + 10 Q&A phản biện).

