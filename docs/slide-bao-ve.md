# Dàn slide bảo vệ (11 slide) — bản trình bày 10 phút

*Tài liệu VIẾT TAY. Cùng định dạng với `slide.md` nên `python -m scripts.export_office` xuất được
`docs/BAO-CAO-slide-bao-ve.pptx`. Phần chữ in nghiêng là lời thoại (không lên slide).*

## 1. Dự báo suy giảm tài chính doanh nghiệp bán lẻ

- Dữ liệu SEC XBRL · 8 công ty · 324 mẫu · 47 feature
- Train/Validation/Test = 212/32/64, chia theo thời gian
- Tái lập 1 lệnh · 222 test · 98.200 phép kiểm tra, 0 phát hiện
- **Dữ liệu SEC thật:** 20 file hash khớp, 4.609 fact tra ngược, 0 bịa
- Kết luận trung thực: mô hình **không** chắc hơn baseline

*Đồ án không chỉ dừng ở AUROC 0,983 mà chỉ ra vì sao con số đó chưa chứng minh năng lực dự báo. Mọi số trên slide tái lập được bằng một lệnh.*

## 2. Bài toán và câu hỏi nghiên cứu

- Quý kế tiếp: doanh nghiệp có suy giảm tài chính?
- RQ1 mô hình · RQ2 đặc trưng · RQ3 công ty mới · RQ4 định nghĩa nhãn
- Ngoài phạm vi: giá cổ phiếu, tín dụng, vĩ mô

*Mục tiêu là dự báo sự kiện, không nhận diện công ty. RQ3 và RQ4 là phần nhóm tự phản biện.*

## 3. Phát hiện then chốt: nhãn là thuộc tính của CÔNG TY

- Nhãn dương 62,3% ở cấp mẫu (IR toàn corpus = 1,66; train = 1,65)
- Nhưng HD/LOW/WMT = 100% nhãn 1; ROST chỉ 4,7%
- IR ở cấp công ty lên tới **20,5**; 41% mẫu thuộc công ty đơn lớp
- Baseline chỉ dùng mã công ty đạt AUROC **0,986**

![Phát hiện then chốt: nhãn là thuộc tính của CÔNG TY](../reports/figures/eda/03_label_by_company_quarter.png)

*Tỉ lệ 62/38 nghe dễ chịu nên dễ tưởng không cần xử lý mất cân bằng. Nhưng tách theo công ty thì ba doanh nghiệp có 100% nhãn 1, còn ROST chỉ 4,7% — IR lên 20,5. Vì vậy accuracy chỉ là chỉ số chẩn đoán.*

## 4. EDA: ba phát hiện định hình thiết kế

- Khuyết có cấu trúc: 5/16 chỉ tiêu phủ dưới 50%
- MNAR: 22/47 feature; mạnh nhất Δ = −55,7 điểm % nhãn
- Đuôi nặng: skew −14,9; 30,6% ngoại lai ở `debt_to_equity`
- Đa cộng tuyến: VIF > 10 ở 33/47 cột; chỉ 12,89 chiều hiệu dụng

![Tương quan và cụm đa cộng tuyến](../reports/figures/eda/08_ratio_correlation.png)

*Ba con số cần nhớ: giá trị thiếu mang thông tin; lệch −14,9 nên Pearson vô dụng; 33/47 cột VIF > 10 nên hệ số logistic không diễn giải nhân quả được.*

## 5. Tiền xử lý và chống rò rỉ

- Pipeline chính: impute median → (scaler cho tuyến tính) → model
- Mọi phép biến đổi nằm TRONG Pipeline, fit chỉ trên train
- Winsorize IQR chỉ nằm ở thí nghiệm 9.1 (lợi ích trong khoảng nhiễu)
- Không resample; cố ý KHÔNG one-hot mã cổ phiếu
- 7 lớp kiểm soát rò rỉ + dải purge 16 mẫu

*Đây là slide chống rò rỉ. Impute và scale nằm trong Pipeline nên mỗi fold học thống kê của riêng nó. Nhóm cố ý không one-hot mã cổ phiếu vì làm vậy sẽ hợp thức hoá đúng loại rò rỉ đang đo. Winsorize không được đưa vào pipeline chính vì lợi ích đo được nằm trong khoảng nhiễu.*

## 6. Ba họ mô hình và cách chọn

- Logistic Regression · Random Forest · HistGradientBoosting (3 họ, thuần scikit-learn)
- GridSearchCV + StratifiedGroupKFold, refit theo AP
- **Mô hình triển khai giữ cấu hình MẶC ĐỊNH** (tinh chỉnh chỉ +0,003 CV-AP ⇒ nhiễu)
- Chọn theo **AP cross-company**, không theo F1 in-domain
- Random search + sổ thực nghiệm `runs.csv` (113 dòng)
- Chỉ 3 họ mô hình (không xgboost/lightgbm) ⇒ mọi báo cáo/slide lấy từ 1 registry

![Đường Precision-Recall trên validation](../reports/figures/validation_pr_curves.png)

*Tiêu chí chọn là AP cross-company — mô hình phải chịu được công ty chưa từng thấy. Kết quả âm cũng được ghi lại: random search không giúp Random Forest, còn HGB thì +0,026.*

## 7. Kết quả trên test (n = 64) — kèm baseline đối chứng

- Random Forest: AUROC 0,983 · AP 0,991 · Brier 0,058
- Tại ngưỡng vận hành 0,788: P 0,973 · R 0,947 · F1 0,960
- Baseline `ticker_prior`: AUROC **0,986** (không thua mô hình)
- Cross-company AUROC 0,933 · LOCO 0,610
- DeLong ΔAUROC p = 0,7546 ⇒ chưa có ý nghĩa

![Confusion matrix trên test](../reports/figures/test_confusion.png)

*Bảng này có một dòng gây khó chịu cho nhóm: baseline chỉ dùng tỉ lệ nhãn trung bình của công ty đạt AUROC 0,986, cao hơn mô hình. Nhóm giữ nguyên dòng đó vì che nó đi thì phần kết quả còn lại trở nên vô nghĩa.*

## 8. Ma trận nhầm lẫn và ngưỡng theo chi phí

- @0,5: TN 23 · FP 3 · FN 1 · TP 37
- @0,788: TN 25 · FP 1 · FN 2 · TP 36 (P 0,973)
- @0,783 (tính trên test để phân tích): TN 25 · FP 1 · FN 1 · TP 37
- Chi phí kỳ vọng giảm 8,0 → 6,0; **ngưỡng vận hành = 0,788** (chọn trên validation)
- FN đắt gấp 5 lần FP (tham số nghiệp vụ)
- Ngưỡng là biến cấu hình, không hardcode

![Ngưỡng, chi phí và hiệu chuẩn](../reports/figures/analysis/04_threshold_curves.png)

*Nhóm biến ngưỡng thành tham số nghiệp vụ: nếu bỏ sót đắt gấp 5 lần báo động giả thì ngưỡng tối ưu là 0,783. Không có ngưỡng nào đúng cho mọi tổ chức — nó phải để trong file cấu hình và ghi log khi ra quyết định.*

## 9. Phân tích lỗi: 3/64 mẫu, có cấu trúc

- 2 FN (HD-2024Q2 P 0,783; FIVE-2024Q3 P 0,174) + 1 FP (DG-2025Q2)
- Lỗi trải ở 3 công ty: HD nhãn hằng 100%; DG 0,55; FIVE 0,19
- SHAP cục bộ: `current_ratio_min_window` kéo về "an toàn"
- Train AUROC 1,000 vs validation 0,965 ⇒ overfit nhẹ
- Learning curve chưa bão hoà ⇒ thiếu thực thể

![Overfitting: train so với validation](../reports/figures/analysis/01_overfit_train_vs_val.png)

*Nhóm không dừng ở "3 mẫu sai" mà truy vết bằng SHAP: FIVE-2024Q3 bị bỏ sót vì mọi chỉ số thanh khoản đều kéo về an toàn, trong khi nhãn thì ngược lại. Kèm việc quy tắc kế toán chỉ khớp 74,7% nhãn gốc, nhóm kết luận trần hiệu năng bị giới hạn bởi chất lượng nhãn.*

## 10. Kiểm chứng độ vững và demo sản phẩm

- 4 định nghĩa nhãn; kết luận giữ nguyên ở 3/3 tái lập được
- KernelSHAP tự cài: sai số efficiency ~1e-16
- Demo: `python -m scripts.predict --sample-id HD-2024Q2 --explain`
- Kiểm chứng dữ liệu thật: `python -m scripts.verify_provenance` (0 lệch)
- 222 test tự động · audit 98.200 phép kiểm tra, 0 phát hiện

![Trong tập so với cross-company](../reports/figures/analysis/07_in_domain_vs_cross_company.png)

*Môi trường không có gói `shap` nên nhóm tự cài KernelSHAP bằng numpy và tự kiểm chứng bằng sai số efficiency — không thuyết trình bằng hình minh hoạ suông. Phần demo chạy trực tiếp trong buổi bảo vệ nếu thầy cô yêu cầu.*

## 11. Kết luận, ứng dụng và hạn chế

- Dùng được làm bộ lọc xếp hạng + ưu tiên thẩm định
- Không dùng để tự động từ chối/hạ hạng tín dụng
- P0: chốt định nghĩa nhãn với giảng viên
- P1: thêm 50–100 công ty; lọc feature theo cụm/VIF
- P2: monitoring PSI/KS; ngưỡng cấu hình hoá

*Ứng dụng đúng nhất là xếp hạng để ưu tiên thẩm định, vì mô hình chỉ chắc chắn hơn ngẫu nhiên ở chỗ đó. Nhóm chủ động liệt kê hạn chế: nhãn không tái lập được, chỉ 8 thực thể, thiếu dữ liệu phi tài chính.*
