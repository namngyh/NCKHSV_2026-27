# Bản thảo cần sửa gì sau khi Bước 1 và Bước 2 hoàn tất

Lập ngày 13/09/2026, sau khi Bước 1 (17/17 mục) và Bước 2 (1.800/1.800 lần lặp) chạy xong.

**Kết quả trung tâm: EGARCH-X Student-t + MLE thắng cả bốn tiêu chí** — AIC 5606,87 · BIC 5650,54 · QLIKE 0,9234 · NLS 1,5203.

Số liệu đầy đủ nằm ở `ket_qua_moi.tex`, **sinh tự động** từ `results/*.pt` bằng `code thuật toán/scripts/bao_cao/sinh_ket_qua_moi.py`. Chạy lại script là số liệu tự khớp; đừng chép tay.

---

## Vấn đề lớn nhất: mô hình được đề xuất đã thua

`Bài báo chính.tex` xây dựng toàn bộ lập luận quanh MS-GARCH-X Student-t như **mô hình đề xuất**. Kết quả thực nghiệm bác bỏ nó:

| Mô hình | $d$ | AIC | QLIKE |
|---|---|---|---|
| **EGARCH-X Student-t** | **8** | **5606,87** | **0,9234** |
| GARCH-X Student-t | 7 | 5648,87 | 0,9520 |
| MS-GARCH-X Student-t *(mô hình đề xuất)* | 15 | 5650,58 | 0,9688 |

Mô hình đề xuất dùng gần gấp đôi số tham số nhưng **thua cả mô hình một trạng thái đơn giản nhất có biến ngoại sinh**. Chênh lệch AIC khi thêm cấu trúc chuyển trạng thái vào GARCH-X Student-t là **+1,71** — tức làm mô hình *tệ đi*.

`Bài báo chính.tex` hiện nhắc tới EGARCH **đúng một lần** trong 171 KB. Họ mô hình thắng cuộc gần như vắng mặt khỏi bài.

Đây là **quyết định của tác giả**, không phải việc sửa số liệu, nên tôi không tự đổi. Ba hướng khả dĩ:

1. **Đổi khung bài thành "so sánh có hệ thống"** thay vì "đề xuất một mô hình". Thiết kế lưới 8 mô hình × 3 phương pháp vốn đã phù hợp với khung này, và kết quả âm về chuyển trạng thái trở thành một phát hiện chứ không phải thất bại.
2. **Giữ MS-GARCH-X làm mô hình trung tâm** nhưng báo cáo trung thực rằng nó thua, và lý giải vì sao.
3. **Đổi mô hình đề xuất thành EGARCH-X Student-t**, viết lại phần lý thuyết quanh nó.

Tôi nghiêng về (1): nó đúng với những gì đã thực sự làm, và biến ba kết quả âm (chuyển trạng thái không đáng, MCMC không hội tụ, VB thua MLE) thành nội dung có giá trị.

---

## Những chỗ sai sự thật, phải sửa bất kể chọn hướng nào

### `Bài báo chính.tex`

| Dòng | Đang viết | Thực tế |
|---|---|---|
| **1931** | "VB mean-field … cho điểm log tốt nhất toàn bảng, $1{,}5354$" | **Sai.** NLS 1,5354 nay xếp **thứ 4**. Ba dòng EGARCH-X Student-t đều tốt hơn: 1,5203 (MLE), 1,5209 (VB-FR), 1,5215 (VB-MF) |
| **1856, 1931** | "trên tám kịch bản, độ bao phủ … $0{,}236$ đến $0{,}440$, trung bình khoảng $0{,}33$" | Số liệu này lấy từ lần chạy **có lỗi không sắp nhãn trạng thái**. Số đúng sau khi sửa, $B=100$: VB-MF 0,321 / 0,453 / 0,339; VB-FR 0,427 / 0,505 / 0,423 |
| **1412–1436** | "tám kịch bản", $B=30$ cho MLE và $B=15$ cho VB | Bước xác nhận dùng **ba kịch bản, $B=100$, cả ba phương pháp**. Bộ tám kịch bản $B$ thấp giữ làm phụ lục bổ trợ |
| **1856, 1931** | RQ3 chỉ thảo luận VB mean-field | VB full-rank nay có số liệu đầy đủ và **tốt hơn VB-MF ở cả sáu cấu hình**. Bỏ qua nó là bỏ mất một nửa câu trả lời cho RQ3 |

### `bao_cao_ket_qua.tex`

**Toàn bộ file phải coi là chưa xác minh.** Nó được sinh từ `report_data.json`, mà file JSON đó:
- được duy trì **thủ công**, không có script tái tạo;
- tổ chức quanh bộ mã mô hình cũ (b1–b4, m1–m4) nên **không biểu diễn được** danh mục tám mô hình;
- nhắc EGARCH **0 lần**;
- chứa kết quả mô phỏng từ trước khi sửa lỗi sắp nhãn.

Cách sửa đúng không phải chép tay số mới vào JSON mà là mở rộng `sinh_ket_qua_moi.py` để sinh nốt các bảng còn lại, rồi bỏ hẳn `report_data.json`.

### `document.tex`

File này khớp nhất trong ba (nhắc EGARCH 23 lần, đã mô tả thiết kế ba bước và ba kịch bản $B=100$). Chủ yếu cần **bổ sung kết quả thực tế** — hiện mới có phần thiết kế. Dùng `\input{ket_qua_moi}`.

---

## Nội dung mới có thể dùng ngay

`ket_qua_moi.tex` chứa bốn bảng, tất cả sinh tự động:

1. **`tab:buoc1-so-sanh`** — 24 ô: 8 mô hình × 3 phương pháp, kèm $d$, log-lik, AIC, BIC, QLIKE, NLS.
2. **`tab:buoc1-phan-ra`** — đóng góp riêng của từng thành phần cấu trúc, đo bằng $\Delta$AIC. Đây là bảng đắt giá nhất cho phần thảo luận:

| Thành phần | $\Delta$AIC |
|---|---|
| Đuôi dày (GARCH) | **−236,2** |
| Đuôi dày (EGARCH) | **−232,0** |
| Biến ngoại sinh (EGARCH) | −49,2 |
| Dạng hàm EGARCH (biến thể X-$t$) | −42,0 |
| Biến ngoại sinh (GARCH) | −12,8 |
| **Chuyển trạng thái** | **+1,7** |

3. **`tab:pra-coverage`** — độ phủ của mô phỏng MS-GARCH-X, đã sắp nhãn.
4. **`tab:prb-theta`** — hệ số đòn bẩy $\theta$ ở ba kịch bản EGARCH.

---

## Ba phát hiện mới nên đưa vào phần thảo luận

**Đuôi dày quan trọng hơn mọi thứ khác, gấp khoảng năm lần.** $\Delta$AIC của Student-t là −236 và −232, trong khi thành phần tốt thứ hai chỉ −49. Đây là câu trả lời dứt khoát cho RQ1.

**Biến ngoại sinh hữu ích gấp gần bốn lần trong EGARCH** (−49,2 so với −12,8). Cơ chế giải thích được: GARCH-X phải chặn phương sai ở $10^{-8}$ vì $\bmgamma^\top\Xvec$ có thể âm, còn EGARCH mô hình hoá $\ln h_t$ nên $\bmgamma$ tự do hoàn toàn. Dữ liệu xác nhận ràng buộc dương đó thực sự làm mất thông tin.

**Ràng buộc dương của GARCH-X phá hoại suy diễn Bayes, không chỉ gây bất tiện lý thuyết.** QLIKE của GARCH-X Student-t: MLE 0,9520 nhưng VB-MF **1,5357** và VB-FR **1,4223** — tệ hơn 50%. Trên EGARCH-X Student-t thì VB gần như ngang MLE (0,9281 và 0,9258 so với 0,9234). Giải thích: VB lấy mẫu tham số từ hậu nghiệm, một phần mẫu cho $\omega + \bmgamma^\top\Xvec < 0$ nên bị chặn, làm hỏng dự báo phương sai; MLE chỉ có một điểm nên tránh được. Đây là lập luận mạnh và mới cho việc chọn EGARCH.

---

## Ý nghĩa cho Bước 3

Mô hình thắng cuộc là **mô hình một trạng thái**, đúng như thiết kế ba bước đã tiên liệu. MCMC ở Bước 3 sẽ **không có đối xứng hoán vị nhãn** — nguyên nhân đã xác định của thất bại 64 giờ trước đây — và chỉ phải làm việc với 8 tham số thay vì 15.

Chướng ngại còn lại là sống núi cong $\omega/(1-\beta)$, thứ vừa làm VB chệch vừa ép NUTS hạ bước tích phân xuống $10^{-4}$. Nên **đổi tham số hoá sang (mức, độ dai dẳng) trước khi chạy Bước 3**.

---

## Việc tôi đã làm và việc tôi cố ý không làm

**Đã làm**: viết `sinh_ket_qua_moi.py`, sinh `ket_qua_moi.tex`, kiểm tra cân bằng môi trường LaTeX và số cột từng bảng (không có trình biên dịch trên máy này nên **chưa biên dịch thử được** — cần chạy XeLaTeX một lần trước khi tin).

**Cố ý không làm**: sửa trực tiếp `Bài báo chính.tex`, `document.tex` hay `bao_cao_ket_qua.tex`. Hai lý do. Thứ nhất, việc mô hình đề xuất bị bác bỏ là quyết định về hướng của bài, thuộc về tác giả. Thứ hai, không có trình biên dịch để kiểm chứng nên sửa mù vào file 171 KB đang biên dịch được là rủi ro không đáng.
