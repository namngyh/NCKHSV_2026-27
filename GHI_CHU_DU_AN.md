# Ghi chú dự án: dự báo rủi ro và ký quỹ động VN30F1M bằng EGARCH-X

Cập nhật: 20/10/2026. Tệp này tổng hợp ngữ cảnh, tiến trình và các điểm cần lưu ý. Quy ước vận hành chi tiết cho Claude nằm ở `CLAUDE.md`; số liệu chính thức nằm ở `Bài báo/tong_hop_ket_qua.tex` và `code thuật toán/results/`.

## 1. Ngữ cảnh

- **Đề tài:** dự báo rủi ro và xây dựng cơ chế ký quỹ động cho hợp đồng tương lai VN30F1M bằng EGARCH-X.
- **Nhóm:** Nguyễn Hoài Nam, Nguyễn Thuận Hiếu, Đặng Chí Hiếu (Khoa Toán – Thống kê, ĐH Tôn Đức Thắng). GVHD: TS. Châu, muốn ứng dụng thực tế rõ hơn.
- **Đề cương hiện hành:** `../../de_cuong_EGARCH-X_ky_quy_dong.md` (và `.tex`, `.pdf`), bản tháng 10/2026.
- **Dữ liệu:** 525.635 nến 1 phút `VN30F1M` (06/11/2017 – 17/07/2026) → 2.170 lợi suất ngày; chia theo thời gian 1.735 huấn luyện / 435 kiểm định (từ 17/10/2024).
- **Hai giai đoạn:** (1) mô hình rủi ro EGARCH-X (MLE, VB mean-field, VB full-rank, NUTS); (2) chuyển VaR/ES nhiều phiên thành tỷ lệ ký quỹ động, so với các quy tắc đối chứng.
- **Câu hỏi nghiên cứu:** CH1 biến ngoại sinh có cải thiện dự báo không; CH2 phân phối sai số và phương pháp suy diễn ảnh hưởng thế nào; CH3 ký quỹ động có tốt hơn đối chứng không.
- **Tiến độ dự kiến:** 10/2026 – 04/2027, báo cáo cấp Trường 05/2027. Kinh phí đề nghị 3.000.000 đồng.

## 2. Lịch sử quyết định

| Ngày | Sự kiện |
|---|---|
| 13–15/09/2026 | Bước 1, 2, 3 hoàn tất. EGARCH-X Student-t thắng; MCMC hội tụ (MS-GARCH-X trước đó không hội tụ). |
| 21–22/09/2026 | Kiểm toán độc lập kết quả, đổi tên dự án, dọn thư mục. |
| 25–26/09/2026 | Chốt "mô hình chính là EGARCH-X". Gỡ MS-GARCH-X (lưu ở nhánh/thẻ `luu-tru/ms-garch-x`). Viết lại thư viện thành `garch_vn30`. |
| 29–30/09/2026 | Chú thích tiếng Việt cho mọi tệp/hàm; `chay.bat` tự cài môi trường. |
| 16/10/2026 | Công bố phần mở rộng: skewed-t, GED, RV khoảng nghỉ trưa/qua đêm, nối F1M/F2M, loại cửa sổ đáo hạn. |

Đừng đưa MS-GARCH-X trở lại nếu chưa được yêu cầu.

## 3. Kết quả đã có

### 3.1. Ba bước nghiên cứu gốc (6 mô hình, 435 phiên kiểm định)
- **Bước 1:** EGARCH-X Student-t + MLE thắng cả bốn tiêu chí: AIC 5606,87 · BIC 5650,54 · QLIKE 0,9234 · NLS 1,5203.
- **Bước 2 (mô phỏng, E1–E3, B=100, T=2000):** E1/E3 θ̂ âm 100/100 lần; E2 (θ₀=0) không bịa ra đòn bẩy; coverage MLE (Wald) 0,89–0,98.
- **Bước 3 (NUTS):** R̂ 1,0125, ESS 1.340, 0 phân kỳ, 8,1 giờ. θ = −0,1399, KTC 95% [−0,2125; −0,0710] loại trừ 0. VB-MF đánh giá thấp bất định của ω và β (R_sd ≈ 0,37 và 0,29).
- **Phân rã ΔAIC:** đuôi dày −236,2 (GARCH) / −232,0 (EGARCH); biến ngoại sinh −49,2 (EGARCH) so với −12,8 (GARCH); dạng EGARCH −42,0.

### 3.2. Phần mở rộng (MLE, ngày giao dịch thông thường, 1.568 quan sát huấn luyện, 391 ngoài mẫu)

| Phân phối | AIC | BIC | QLIKE | NLS |
|---|---:|---:|---:|---:|
| Student-t | 5035,724 | 5078,585 | 1,159331 | **1,520979** |
| Skewed-t | **5024,098** | **5072,316** | 1,155516 | 1,522792 |
| GED | 5045,817 | 5088,677 | **1,154070** | 1,537779 |

Chưa kiểm định ý nghĩa thống kê của các chênh lệch ngoài mẫu. Không so AIC/BIC/QLIKE giữa các bản có tập quan sát khác nhau (xem 4.1).

### 3.3. Thực hiện theo đề cương, chỉ MLE (20/10/2026)

**Quyết định của người dùng 20/10/2026:** không làm VB nữa, từ nay chỉ ước lượng MLE; làm bám sát đề cương. Mã mới nằm ở gói `code thuật toán/de_cuong/` (`du_lieu`, `mo_hinh`, `danh_gia`, `ky_quy`) và `scripts/dc_chay.py` (bước 1–5), `dc_mo_phong.py`, `dc_ky_quy.py`; kết quả ở `results/de_cuong/`. Gói `garch_vn30` cũ không bị sửa.

Thiết kế: bỏ 61 phiên đầu (thiếu lịch sử 60 phiên cho biến ngoại sinh) → 2.110 phiên mô hình; UL = 700, XT = 420, KT = 990 phiên (942 lợi suất, vì 102 phiên chuyển hợp đồng là quan sát khuyết, không bị loại khỏi chuỗi). Phiên chuyển hợp đồng J_t và ngày đáo hạn được xử lý đúng mục 2.3/3.4 (khuyết + biến giả D_đh, D_sau, D_nghỉ), không bỏ ngày đáo hạn. Ước lượng lại mỗi 21 phiên, cửa sổ mở rộng, thống kê chuẩn hóa chỉ dùng dữ liệu đến thời điểm ước lượng lại.

| Bước | Kết quả chính |
|---|---|
| Dữ liệu (mục 2.8) | 2.171 phiên, 105 phiên chuyển, 105 đáo hạn, 13 phiên sau nghỉ dài, 0 phiên khóa biên, 8 phiên |r| > 7% cần đối chiếu tay. |qua đêm| trung vị 0,64% ở phiên chuyển so với 0,21% ở phiên thường. c_rho = 1,46 |
| Ước lượng (UL, skewed-t, 20 điểm khởi tạo đều hội tụ cùng nghiệm) | ν = 4,52; λ = −0,15; θ = −0,20; β = 0,75; γ_G = 0,30; δ_đh = 0,91. Chẩn đoán Ljung–Box, Berkowitz, hồi quy z²−1 lên D và τ đều không bác bỏ. LR γ = 0 bị bác bỏ ở cả ba phân phối (p < 0,001) |
| Chọn biến (XT, Student-t) | Chọn tiến chỉ giữ **G** (DM p = 0,017); thêm biến thứ hai không có ý nghĩa (p = 0,13). **d_X = 1, khác cấu hình chính (A, G) có d_X = 2: phải sửa đề cương (mục 6).** SPA p = 0,09 (chưa bác bỏ ở 5%). LR theo biến sau Holm: G và F có ý nghĩa, A, B, O không. γ_G không đổi dấu qua các lần ước lượng lại |
| Chọn phân phối (XT) | skewed-t có LS cao nhất; DM không phân biệt được skewed-t với Student-t (p = 0,33); chuẩn bị bác bỏ (p = 0,02) |
| Ngoài mẫu KT | 1% VaR: skewed-t 7 vi phạm (0,74%), Kupiec p = 0,41, toàn bộ 741 cửa sổ 250 phiên ở vùng xanh. Phân phối chuẩn thua rõ (13 vi phạm, phân vị động p ≈ 0, vùng vàng 201 cửa sổ). **CH1 chỉ được ủng hộ yếu ngoài mẫu:** LS tốt hơn EGARCH (DM p = 0,07), nhưng FZ0 không tốt hơn (p = 0,37) và QLIKE của EGARCH thấp hơn (1,480 so với 1,510) |
| Mô phỏng khôi phục (E1–E4, B = 100) | 100/100 lần thành công; độ phủ Wald 0,87–0,99; E2 bác bỏ sai θ = 0 ở mức 0,05; E4 bác bỏ sai γ = 0 ở mức 0,06. Đạt điều kiện chuyển sang Giai đoạn 2 |
| Ký quỹ (7) | Xem 3.4 |

Khác biệt cần ghi vào báo cáo: gradient tính số (không dùng autodiff); 20 điểm khởi tạo cho ước lượng chính, 8 điểm kèm khởi tạo ấm cho các lần ước lượng lại cuốn chiếu; biên độ giá, bước làm tròn g = 0,5% và giới hạn giảm D = 1%/phiên chưa xác minh theo thông lệ.

### 3.4. Ký quỹ động (mục 7), mô hình skewed-t với biến G

Thiết kế: H và α là tham số thiết kế nên được báo cáo trên lưới H ∈ {1, 2, 3} × α ∈ {5%, 2,5%, 1%}; với mỗi cặp, (w, b, φ, η) được chọn trên XT bằng chi phí kỳ vọng với c_th/c_v = 1/α rồi cố định trên KT. Trình bày chính: H = 2, α = 1% (lựa chọn của người thực hiện, chưa được người dùng chốt).

- **H = 2, α = 1%:** KT có 894 cửa sổ hợp lệ, 3 vi phạm (0,34%), ký quỹ trung bình 6,1%, tối đa 15,5%. H = 1, α = 1%: 0,96% vi phạm, ký quỹ trung bình 4,1%. Mức 13% không có vi phạm nào nhưng mức này chưa xác minh.
- **Quy tắc năm bước:** trên XT, mọi cặp chọn w = 0, b = 0, η = 0; chỉ sàn φ (0,5–1,0) có tác dụng. Dữ liệu hiện tại không ủng hộ ES, hệ số đệm hay làm trơn bất đối xứng.
- **CH3 (H = 2, 1%), so tại cùng ký quỹ trung bình:** thiếu hụt mỗi phiên thấp hơn ký quỹ cố định (DM p = 0,026), nhưng **không phân biệt được** với VaR thô EGARCH-X (p = 0,16), mô phỏng lịch sử có lọc (p = 0,18), EWMA (p = 0,07), lịch sử 250 phiên (p = 0,09), và **không tốt hơn EGARCH không có X (p = 0,77)**. Giả thuyết 3 chưa bác bỏ được so với các đối chứng động.
- **Giới hạn:** chỉ 3–9 vi phạm mỗi cặp nên lực kiểm định thấp; các cửa sổ H ≥ 2 chồng lấn. Trong 47 phiên căng thẳng của KT, H = 1 và α = 1% vi phạm 10,6% (cũng bằng ký quỹ cố định). Đèn giao thông chỉ có nghĩa với H = 1 và 99%: các cặp α = 5% ở vùng đỏ là điều dự kiến.

## 4. Điểm cần ghi chú và cảnh báo

### 4.1. Phạm vi xử lý lỗi đáo hạn/chuyển hợp đồng (đánh giá 20/10/2026)

**Đã sửa, chỉ trong nhánh MLE mở rộng** (`scripts/uoc_luong_mle_*.py`, kết quả ở `results/phan_phoi_moi/`):
- Nối lợi suất và lợi suất qua đêm của phiên đầu sau đáo hạn bằng giá đóng cửa F2M hôm trước (105 lần nối, đủ giá F2M). Ví dụ 22/05/2020: lợi suất qua đêm −9,08% → +0,38%; RV 84,40 → 2,09.
- Bỏ ngày đáo hạn khỏi hàm hợp lý; phiên đầu sau đáo hạn chỉ khởi tạo lại đường phương sai (h0), không tính điểm.
- Phương sai khởi tạo và chuẩn hóa biến ngoại sinh chỉ dùng các ngày huấn luyện được giữ lại.

**Chưa sửa:**
- Thư viện `garch_vn30` không có xử lý đáo hạn nào (không có từ khóa dao_han/expiry/roll/F2M). Sáu mô hình, Bước 2 (mô phỏng) và Bước 3 (NUTS) vẫn chạy trên chuỗi chưa nối hợp đồng. Số liệu 3.1 vì thế chưa chịu lỗi đáo hạn được loại.
- VB không có cho chuỗi đã xử lý đáo hạn. Chỉ có MLE.
- Đây là hàm hợp lý có điều kiện theo đoạn, không phải hàm hợp lý đầy đủ của EGARCH có quan sát khuyết (đề cương mục 3.4 mô tả cách khác: giữ đệ quy liên tục, thay cú sốc bằng kỳ vọng 0).
- Mô hình không dự báo rủi ro ngày đáo hạn và phiên đầu sau đó. Điều này mâu thuẫn với Giai đoạn 2, vì ký quỹ phải được ấn định cho mọi phiên.
- Hệ số động học đổi mạnh khi loại cửa sổ: alpha 0,0551 → 0,0058 (GED: −0,0048); beta khoảng 0,68 → 0,58; gamma_rv 0,289 → 0,407. Báo cáo ghi nhận sự nhạy cảm này nhưng chưa tìm nguyên nhân. Giả thuyết chưa kiểm tra: cắt và khởi tạo lại đường phương sai mỗi ~21 phiên làm yếu cấu trúc động.
- Chuyển hợp đồng sớm (trước đáo hạn) chỉ thử được trên 2024–2026 (F2M 5 phút), chưa có kiểm định ý nghĩa.
- Tệp nến F2M 5 phút gốc không nằm trong repo.
- Chưa xác minh ai là tác giả phần mở rộng ngoài việc nó nằm ở commit 16/10/2026; kiểm tra bằng `git log --format='%an'` nếu cần ghi công.

### 4.2. Lệch giữa đề cương và code
- Chia mẫu: đề cương dùng ba khối (UL ≈ 750, XT ≈ 420, KT ≈ 1.000); code dùng hai khối 1.735/435. Với 435 phiên, lực kiểm định Kupiec chỉ ≈ 0,37.
- Biến ngoại sinh: đề cương dùng A_t (khối lượng bất thường đã khử chu kỳ đáo hạn) và G_t; code dùng log(1+volume) và log(RV+1e-6).
- Đại lượng đại diện phương sai: đề cương dùng Garman–Klass + qua đêm có hiệu chỉnh thang đo; code mở rộng dùng RV từ nến 1 phút.
- Quy trình chọn biến (đề cương mục 6: chọn tiến, SPA, Holm) chưa có code.

### 4.3. Thông số thể chế cần xác minh trước khi dùng
Chưa xác minh từ nguồn chính thức: mức 13% của VSD, biên độ giá ±7%, hệ số nhân 100.000 đồng/điểm, bước làm tròn tỷ lệ ký quỹ, lịch đáo hạn qua các thời kỳ. Dữ liệu không có chuỗi ký quỹ lịch sử.

## 5. Việc còn mở (theo thứ tự đề xuất)

Đã xong ngày 20/10/2026 (gói `de_cuong/`): ký quỹ động, walk-forward, Diebold–Mariano, chọn biến, mô phỏng khôi phục, MLE theo đề cương. Còn lại:

1. **Chốt với người dùng/GVHD:** cặp thiết kế (H, α) trình bày chính; chấp nhận d_X = 1 (chỉ G) thay cho (A, G) và sửa đề cương mục 2.6, 6; xác minh mức 13% VSD, biên ±7%, bước làm tròn g, giới hạn giảm D.
2. **Báo cáo:** sinh bảng và hình từ `results/de_cuong/` (chưa có generator, chưa có hình); cập nhật `tai_lieu_ma_nguon.tex` cho gói `de_cuong` và thêm vào `chay.bat`/thư mục nộp GVHD.
3. **Kiểm tra độ bền ký quỹ còn thiếu:** đổi w, η; so ký quỹ MLE với hậu nghiệm (bỏ vì chỉ dùng MLE); bootstrap tham số khi nghiệm sát biên; kiểm thử đơn vị cho `de_cuong` (chưa có).
4. **Vấn đề kết quả cần suy nghĩ:** CH1 yếu ngoài mẫu và quy tắc năm bước thu về VaR + sàn. Cân nhắc có nên dùng ES, mở rộng lưới w/b/η, hay nêu đây là kết quả âm có kiểm soát.
5. **Cập nhật hai bản thảo** `Bài báo chính.tex`, `document.tex` (còn mô tả MS-GARCH-X). Chỉ làm khi được yêu cầu.
7. Cập nhật thư mục nộp GVHD `D:\Hoai Nam Ng\Project\GARCH-VN30F1M_Nop_GVHD\` khi code thay đổi.

## 6. Lưu ý vận hành

- Python có torch/pyro: `D:\App\miniconda\envs\quant311\python.exe`. XeLaTeX (MiKTeX) biên dịch 2 lần, dọn `.aux/.log/.out/.toc`.
- Viết tiếng Việt học thuật; trong khung chat không dùng ký hiệu LaTeX; dùng "trường hợp" thay "kịch bản".
- Mọi con số phải truy được về `results/`; tách rõ "bằng chứng cho thấy gì" và "chưa thể kết luận gì".
- Sửa code thì cập nhật docstring và `code thuật toán/tai_lieu/tai_lieu_ma_nguon.tex`.
- `tong_hop_ket_qua.tex` có chỉnh sửa tay đã đưa vào generator; đối chiếu diff trước khi sửa generator.
- Có tiến trình Python của dự án khác trên máy (qp-tracking, distributional_bands); không dừng tiến trình khi chưa liệt kê dòng lệnh.
