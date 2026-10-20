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
| **1856, 1931** | "trên tám kịch bản, độ bao phủ … $0{,}236$ đến $0{,}440$, trung bình khoảng $0{,}33$" | Số liệu này lấy từ lần chạy **có lỗi không sắp nhãn trạng thái**. Số đúng sau khi sửa, $B=100$: VB-MF 0,323 / 0,495 / 0,339; VB-FR 0,469 / 0,567 / 0,455 (đã đính chính 21/09 sau khi sửa lỗi giá trị thật của $p_{ss}$) |
| **1412–1436** | "tám kịch bản", $B=30$ cho MLE và $B=15$ cho VB | Bước xác nhận dùng **ba kịch bản, $B=100$, cả ba phương pháp**. Bộ tám kịch bản $B$ thấp giữ làm phụ lục bổ trợ |
| **1856, 1931** | RQ3 chỉ thảo luận VB mean-field | VB full-rank nay có số liệu đầy đủ và **tốt hơn VB-MF ở cả sáu cấu hình**. Bỏ qua nó là bỏ mất một nửa câu trả lời cho RQ3 |

### `bao_cao_ket_qua.tex`

> **Cập nhật 21/09.** Commit 70ccac5 đã đồng bộ file này với kết quả thật (thêm mục "ba bước", `\input{ket_qua_moi}`, chuyển 7 mục cũ thành Phụ lục A). Khối `sim` của `report_data.json` khi đó được **sửa tay** — nên khi chạy lại `01_sinh_hinh.py` nó bị ghi đè về cấu trúc cũ và `02_sinh_tex.py` hỏng. Nay đã vá `01_sinh_hinh.py` để tự sinh đúng cấu trúc (S1–S3 + E1–E3, ba phương pháp, gộp hai tầng), tức `report_data.json` **tái tạo được**, không còn phải sửa tay. Số liệu $p_{ss}$ trong đó cũng đã đúng sau khi sửa lỗi giá trị thật.

Đoạn dưới là nhận xét ban đầu (13/09), giữ lại làm lịch sử:

Nó được sinh từ `report_data.json`, mà file JSON đó lúc ấy:
- được duy trì **thủ công**, không có script tái tạo;
- tổ chức quanh bộ mã mô hình cũ (b1–b4, m1–m4) nên **không biểu diễn được** danh mục tám mô hình;
- nhắc EGARCH **0 lần**;
- chứa kết quả mô phỏng từ trước khi sửa lỗi sắp nhãn.

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

## Bước 3 đã chạy và HỘI TỤ (15/09) — bổ sung cho bản ghi chú này

Dự đoán ở bản đầu ("Bước 3 sẽ không vướng đối xứng hoán vị nhãn") đã được xác nhận, **mà không cần đổi tham số hoá** như tôi từng khuyên. Khởi tạo bốn chuỗi phân tán quanh nghiệm MLE (2 sai số chuẩn) là đủ.

| Chẩn đoán | EGARCH-X-t | MS-GARCH-X (lần thất bại) |
|---|---|---|
| $\hat R$ lớn nhất | **1,0125** | 2,84 |
| ESS nhỏ nhất | **1.340** / 1.600 | ≈ 5 |
| Phân kỳ | 0 | — |
| Thời gian | 8,1 giờ (4 chuỗi tuần tự) | 64 giờ, không hội tụ |

Cùng thư viện, cùng thuật toán, cùng dữ liệu — bỏ cấu trúc chuyển trạng thái thì hội tụ. Đây là bằng chứng dứt khoát cho luận điểm trung tâm của thiết kế ba bước, và là **kết quả dương về MCMC** mà bài trước đây không có.

**Hai kết quả dùng thẳng trong bài:**

1. **Hiệu ứng đòn bẩy được xác lập thống kê**: $\theta = -0{,}1399$, khoảng tin cậy 95% **[−0,2125; −0,0710] loại trừ 0**. Kết hợp với kịch bản $E_2$ của PR-B (mô hình không bịa ra đòn bẩy khi $\theta_0 = 0$), kết luận được kiểm chứng cả hai chiều.

2. **RQ3 trả lời được lần đầu tiên** — tỷ số $R_{sd} = \mathrm{sd}_{VB}/\mathrm{sd}_{MCMC}$ mà bài định nghĩa nhưng chưa từng tính được:

| | $\mu$ | $\omega$ | $\alpha$ | $\theta$ | $\beta$ | $\nu$ |
|---|---|---|---|---|---|---|
| VB-MF | 0,95 | **0,37** | 0,88 | 0,92 | **0,29** | 0,63 |
| VB-FR | 0,94 | **0,75** | 1,01 | 1,00 | **0,75** | 0,97 |

Mean-field đánh giá thấp bất định của $\omega$ và $\beta$ khoảng **ba lần** — đúng hai tham số trên sống núi cong $\omega/(1-\beta)$ mà mô phỏng đã chỉ ra. Kết quả mô phỏng và kết quả dữ liệu thật nói cùng một điều bằng hai con đường độc lập. Bảng đầy đủ: `tab:buoc3-bon-phuong-phap` và `tab:buoc3-rsd` trong `ket_qua_moi.tex`.

**Hệ quả cho dòng 1856 và 1931 của `Bài báo chính.tex`**: câu "Chuỗi MCMC không hội tụ nên không dùng làm chuẩn tham chiếu được" nay **sai** với mô hình thắng cuộc. RQ3 phải viết lại: có chuẩn tham chiếu, và kết luận về VB mean-field giờ là định lượng ($R_{sd} = 0{,}29$–$0{,}37$ ở $\omega, \beta$) chứ không còn phải suy từ mô phỏng.

Sống núi cong vẫn hiện diện — chính nó làm $R_{sd}$ của VB thấp ở $\omega$, $\beta$ — nhưng NUTS đi qua được; VB thì không, vì họ Gauss không mô tả được đường cong. Đổi tham số hoá vẫn là hướng tốt để **VB** làm việc tốt hơn, không còn là điều kiện cho MCMC.

---

## Việc tôi đã làm và việc tôi cố ý không làm

**Đã làm**: viết `sinh_ket_qua_moi.py`, sinh `ket_qua_moi.tex`, kiểm tra cân bằng môi trường LaTeX và số cột từng bảng (không có trình biên dịch trên máy này nên **chưa biên dịch thử được** — cần chạy XeLaTeX một lần trước khi tin).

**Cố ý không làm**: sửa trực tiếp `Bài báo chính.tex`, `document.tex` hay `bao_cao_ket_qua.tex`. Hai lý do. Thứ nhất, việc mô hình đề xuất bị bác bỏ là quyết định về hướng của bài, thuộc về tác giả. Thứ hai, không có trình biên dịch để kiểm chứng nên sửa mù vào file 171 KB đang biên dịch được là rủi ro không đáng.
