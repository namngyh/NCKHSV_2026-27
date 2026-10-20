# Mở rộng EGARCH-X: phân phối nhiễu, khoảng nghỉ và đáo hạn

Tài liệu này nối các kết quả mở rộng với quy trình sáu mô hình của nghiên cứu gốc. **Kết quả được chọn để phân tích biến động của ngày giao dịch thông thường** là EGARCH-X MLE trên chuỗi 2017–2026, sau khi tính cả khoảng nghỉ trưa và qua đêm, nối giá khi đổi hợp đồng, rồi loại cửa sổ đáo hạn khỏi hàm hợp lý. Bản giữ nguyên ngày đáo hạn và bản chuyển hợp đồng sớm từ 2024 được lưu làm đối chiếu. Chưa có VB cho chuỗi đã xử lý khoảng nghỉ và đáo hạn.

## 1. Dữ liệu và mô hình

- Dữ liệu gốc: [`ohlc_export.csv`](ohlc_export.csv), 525.635 nến 1 phút của ký hiệu liên tục `VN30F1M`, từ 06/11/2017 đến 17/07/2026. Sau ngày đầu không có giá đóng cửa trước đó: 2.170 lợi suất ngày, chia theo thời gian thành 1.735 huấn luyện và 435 kiểm định (từ 17/10/2024).
- Lợi suất mục tiêu là `r_t = 100 log(C_t / C*_{t-1})` (%). Vào ngày đầu sau đáo hạn, `C*_{t-1}` là **giá đóng cửa F2M của ngày trước**; ở các ngày khác là giá đóng cửa F1M của ngày trước. Cách nối này tránh đưa chênh lệch giá giữa hai hợp đồng vào lợi suất qua đêm. Kỹ thuật điều chỉnh bước nối trên chuỗi hợp đồng tương lai được giải thích trong [nghiên cứu đăng trên CME](https://www.cmegroup.com/education/files/improving-time-series-momentum-strategies.pdf).
- Nhiễu `z_t = (r_t - mu)/sqrt(h_t)` được chuẩn hóa về trung bình 0, phương sai 1. Ba lựa chọn: Student-t (`nu`), Hansen skewed-t (`eta`, `lambda_skew`) và GED (`ged_shape`). `theta` trong EGARCH là hệ số phản ứng bất đối xứng của phương sai; `lambda_skew` là độ lệch của **phân phối nhiễu**, hai tham số có ý nghĩa khác nhau.
- Hai biến ngoại sinh tại ngày `t`: `log(1 + volume_{t-1})` và `log(RV_{t-1} + 1e-6)`. Chỉ dùng thông tin của phiên trước; chuẩn hóa bằng dữ liệu huấn luyện. Trong bản loại cửa sổ, thống kê chuẩn hóa chỉ dùng những ngày huấn luyện được giữ lại.
- Phương trình phương sai: `log h_t = omega + alpha(|z_{t-1}| - E|z|) + theta z_{t-1} + beta log h_{t-1} + gamma_volume X_volume,t + gamma_rv X_RV,t`. `E|z|` được tính theo đúng phân phối đã chọn.

## 2. Xử lý nghỉ trưa, qua đêm và đổi hợp đồng

Các nến được tách thành hai phiên tại khoảng trống trong ngày dài hơn 30 phút. Mỗi ngày trong tệp có đúng một khoảng như vậy. RV của một phiên là tổng bình phương lợi suất từ **giá mở đến giá đóng nến đầu**, rồi giữa các giá đóng nến liên tiếp. Đặt `L_t = 100 log(O_chiều,t/C_sáng,t)` và `ON_t = 100 log(O_sáng,t/C*_{t-1})`. Khi đó:

`RV_t = RV_sáng,t + RV_chiều,t + L_t² + ON_t²` (%²).

`ON_t²` là **proxy** cho biến động qua khoảng thị trường đóng cửa; không có giao dịch thực tế trong đêm để tính RV tần suất cao. Phép chia này làm cho khoảng thời gian của RV phù hợp với lợi suất đóng cửa đến đóng cửa.

Ngày đáo hạn được xác định theo quy tắc [thứ Năm thứ ba của tháng của HNX](https://cbtd.hnx.vn/vi-vn/huong-dan/chi-tiet-thu-tuc-36-65.html); nếu nghỉ giao dịch, dùng phiên quan sát ngay trước đó trong tháng. Giá F2M theo ngày lấy từ [VNDIRECT DChart](https://dchart-api.vndirect.com.vn/dchart/history), đã đóng băng tại [`vndirect_f1m_f2m_daily.csv`](results/phan_phoi_moi/dao_han_f2m/vndirect_f1m_f2m_daily.csv). Giá F1M từ API khớp giá đóng cửa nến phút ở 2.170/2.171 ngày; một ngày thiếu là 29/04/2021, không thuộc ranh giới đáo hạn. Có **105 lần nối hợp đồng**, đều có giá F2M của ngày trước.

Ví dụ 21–22/05/2020: F1M cũ đóng 864, F2M đóng 786, phiên sau F1M mới mở 789. Lợi suất qua đêm ghép trực tiếp `100 log(789/864) = -9,0806%`; lợi suất qua đêm trên hợp đồng mới `100 log(789/786) = +0,3810%`. Giá 864 và 786 được ghi trong [bản tin VNCS](https://vncsi.com.vn/data/data/anhpnh/files/ban-tin-thi-truong-ngay-21_05_2020.pdf). Giá giao dịch của hợp đồng cũ **ngay trong ngày đáo hạn** vẫn là quan sát có thật và được giữ trong bản nối đầy đủ.

## 3. Vì sao loại cửa sổ đáo hạn

Với câu hỏi về ngày giao dịch thông thường, bản được chọn áp dụng quy tắc nhất quán cho cả 2017–2026:

1. Nối lợi suất phiên đầu sau đáo hạn bằng giá F2M của ngày trước như Mục 2.
2. Bỏ ngày đáo hạn khỏi hàm hợp lý.
3. Dùng phiên đầu sau đáo hạn làm điểm khởi tạo một đường phương sai mới; không đưa điểm này vào hàm hợp lý hoặc chấm điểm.
4. Từ phiên tiếp theo nữa, ước lượng/chấm điểm bình thường. Phương sai khởi tạo và thống kê chuẩn hóa biến ngoại sinh chỉ tính từ ngày huấn luyện được giữ lại.

Đây là **ước lượng có điều kiện theo các đoạn quan sát**, không phải hàm hợp lý đầy đủ của một EGARCH có quan sát bị thiếu. Vì trạng thái EGARCH phụ thuộc cú sốc hôm trước, chỉ xóa các hàng ngày đáo hạn rồi chạy đệ quy liên tục sẽ không đạt cùng mục tiêu; nghiên cứu về [quan sát thiếu trong mô hình GARCH](https://www.sciencedirect.com/science/article/pii/S0304407620302670) cho thấy xử lý thiếu quan sát cần được nói rõ. Bản này **không dự báo rủi ro trong ngày đáo hạn hoặc phiên đầu sau đó**. Bản giữ nguyên các ngày đó trong [`dao_han_f2m/REPORT_VI.md`](results/phan_phoi_moi/dao_han_f2m/REPORT_VI.md) là đối chiếu.

Trong 105 kỳ đáo hạn, 83 kỳ nằm ở huấn luyện và 22 kỳ ở kiểm định. Hàm hợp lý còn **1.568 quan sát huấn luyện**; **391 ngày ngoài mẫu** được chấm. Mỗi phân phối được tối ưu bằng L-BFGS-B từ **30 điểm khởi tạo**. Bảng số chi tiết và toàn bộ tham số nằm trong [`egarch_x_expiry_window_sensitivity.json`](results/phan_phoi_moi/dao_han_f2m/egarch_x_expiry_window_sensitivity.json).

| Phân phối | AIC ↓ | BIC ↓ | QLIKE ngoài mẫu ↓ | NLS ngoài mẫu ↓ | Điểm gần nghiệm tốt nhất |
|---|---:|---:|---:|---:|---:|
| Student-t | 5035,724 | 5078,585 | 1,159331 | **1,520979** | 30/30 |
| Skewed-t | **5024,098** | **5072,316** | 1,155516 | 1,522792 | 30/30 |
| GED | 5045,817 | 5088,677 | **1,154070** | 1,537779 | 29/30 |

AIC/BIC huấn luyện ưu tiên skewed-t; QLIKE kiểm định ưu tiên GED; NLS kiểm định ưu tiên Student-t. Chênh lệch QLIKE giữa GED và skewed-t khoảng 0,00145. Chưa kiểm định ý nghĩa thống kê của các chênh lệch ngoài mẫu. **Không so trực tiếp AIC, BIC hoặc QLIKE của bảng này với bản giữ ngày đáo hạn**, vì tập quan sát và cách chấm khác nhau.

## 4. Các kết quả bổ sung và giới hạn so sánh

| Kết quả | Dữ liệu và mục đích | Báo cáo |
|---|---|---|
| MLE skewed-t/GED, VB mean-field/full-rank | Dữ liệu gốc của sáu mô hình; chưa cộng riêng trưa/đêm, chưa sửa chuyển hợp đồng | [`REPORT_VI.md`](results/phan_phoi_moi/REPORT_VI.md), [`REPORT_VB_VI.md`](results/phan_phoi_moi/REPORT_VB_VI.md) |
| MLE sau khi tính đủ hai khoảng nghỉ | Giữ lợi suất gốc, thay RV và biến RV trễ; chưa nối hợp đồng | [`khoang_nghi/REPORT_MLE_VI.md`](results/phan_phoi_moi/khoang_nghi/REPORT_MLE_VI.md) |
| MLE nối F2M theo ngày | Cùng 2.170 lợi suất, sửa 105 lần đổi hợp đồng; giữ ngày đáo hạn | [`dao_han_f2m/REPORT_VI.md`](results/phan_phoi_moi/dao_han_f2m/REPORT_VI.md) |
| MLE chuyển sang F2M từ ngày đáo hạn | Dùng F2M 5 phút, chỉ có phần giao nhau 2024–2026; so với chuyển sau đáo hạn trên cùng giai đoạn | [`dao_han_f2m_5m/REPORT_VI.md`](results/phan_phoi_moi/dao_han_f2m_5m/REPORT_VI.md) |

Riêng F2M **5 phút** hiện có từ 02/01/2024; F2M **theo ngày** có từ 2017. Do đó phép thử chuyển sớm không bao phủ trường hợp 05/2020. Phương pháp chuyển sớm còn giả định thực hiện giao dịch tại giá đóng cửa phiên trước đáo hạn, chưa tính chênh lệch mua–bán và trượt giá. Các kết quả MLE/VB trên dữ liệu gốc không phải kết quả VB của bản loại cửa sổ.

## 5. Chạy lại và kiểm toán

Chạy từ thư mục `code thuật toán` sau khi cài `python -m pip install -r requirements.txt`. Tệp nguồn F1M và các kết quả gốc nằm trong repo. Nguồn giá F2M ngày đã đóng băng; lệnh mặc định **không tải lại từ mạng**.

```powershell
python scripts/uoc_luong_egarch_x_phan_phoi_moi.py --self-test
python scripts/uoc_luong_vb_egarch_x_phan_phoi_moi.py --self-test
python scripts/uoc_luong_egarch_x_phan_phoi_moi.py --starts 10
python scripts/uoc_luong_mle_khoang_nghi.py --starts 30
python scripts/uoc_luong_mle_roll_f2m.py --starts 30
python scripts/uoc_luong_mle_loai_ngay_dao_han.py --starts 30
```

Nếu muốn chạy lại hai bản VB trên dữ liệu **gốc**, dùng `python scripts/uoc_luong_vb_egarch_x_phan_phoi_moi.py --steps 3000 --draws 1000`; script có thể dùng kết quả đã lưu nếu đủ bước. Thêm `--fetch` vào `uoc_luong_mle_roll_f2m.py` chỉ khi cần cập nhật giá ngày từ VNDIRECT; nguồn có thể thay đổi sau thời điểm chốt dữ liệu, xem [`source_metadata.json`](results/phan_phoi_moi/dao_han_f2m/source_metadata.json).

Phép thử F2M 5 phút cần người dùng cung cấp tệp OHLCV ngoài repo:

```powershell
python scripts/uoc_luong_mle_roll_som_5m.py --import-source --source "<duong-dan-den-VN30F2M-5m.csv>" --starts 30
```

Tệp phải có `datetime_vn,open,high,low,close,volume`, bao phủ ít nhất 02/01/2024–17/07/2026. Mã SHA-256 của tệp đã dùng nằm trong [`egarch_x_5m_early_vs_late_mle.json`](results/phan_phoi_moi/dao_han_f2m_5m/egarch_x_5m_early_vs_late_mle.json). Tệp nến F2M 5 phút gốc không có trong repo công khai; các bảng ngày đã xử lý, tham số và điểm số đều được lưu.

Các script mới ghi vào `results/phan_phoi_moi/`, không ghi đè kết quả sáu mô hình gốc. `python -m pytest tests -q` kiểm tra thư viện hiện hành; `python scripts/kiem_toan_ket_qua.py` kiểm toán bảng số của nghiên cứu gốc. Hai lệnh này **không tự kiểm toán** toàn bộ bảng số của phần mở rộng; có thể đối chiếu từng ngày trong các tệp CSV và tham số/điểm số trong JSON đã lưu.
