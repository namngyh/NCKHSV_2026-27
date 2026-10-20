# EGARCH-X MLE sau khi tách qua đêm và nghỉ trưa

## Thiết kế

- Đầu vào: 525.635 nến phút của chuỗi VN30F1M, từ 2.171 ngày giao dịch trong giai đoạn 06/11/2017–17/07/2026. Lợi suất mục tiêu vẫn là `100 log(C_t/C_{t-1})` theo ngày; giữ nguyên 2.170 quan sát, 1.735 huấn luyện và 435 kiểm định, bắt đầu kiểm định ngày 17/10/2024.
- Tách hai phiên theo đúng một khoảng trống trong ngày dài hơn 30 phút; các khoảng quan sát dài 88–95 phút, trung vị 90 phút.
- Trong từng phiên, RV cộng bình phương lợi suất từ **mở cửa nến đầu → đóng cửa nến đầu**, rồi từ đóng cửa nến này → đóng cửa nến tiếp theo. Gọi `ON_t = 100 log(O_sáng,t/C_{t-1})` và `L_t = 100 log(O_chiều,t/C_sáng,t)`. Khi đó `RV_full,t = RV_sáng,t + RV_chiều,t + ON_t² + L_t²` (%²). Ngày đầu chưa có giá đóng cửa hôm trước nên `ON` thiếu; RV ngày đầu vẫn được tính từ phần quan sát được để dùng làm biến trễ ngày thứ hai.
- Biến ngoại sinh giữ nguyên cấu trúc: `log(1+VOL_{t-1})` và `log(RV_full,t-1 + 1e-6)`; chuẩn hóa theo tập huấn luyện. Phương trình EGARCH và lợi suất mục tiêu không đổi. MLE dùng **30 điểm khởi tạo khác nhau cho mỗi phân phối**: một lưới tham số đuôi/hình dạng/độ lệch và các điểm ngẫu nhiên trải rộng trên các tham số EGARCH. Mỗi điểm được tối ưu bằng L-BFGS-B; giữ nghiệm có log-hợp lý cao nhất.
- Đường lợi suất ngày và ngày chia mẫu được kiểm tra trùng với dữ liệu gốc. Log-hợp lý Student-t được đối chiếu độc lập với mã EGARCH gốc; QLIKE, NLS, MAE và RMSE ngoài mẫu cũng khớp.

## Kết quả mới

| Phân phối | Log-likelihood | AIC ↓ | BIC ↓ | QLIKE ↓ | NLS ↓ | Vi phạm VaR 1% / 5% |
|---|---:|---:|---:|---:|---:|---:|
| Student-t | -2801.391317 | 5618.782633 | 5662.452735 | 1.138424 | **1.525222** | 6 / 24 |
| Skewed-t | **-2793.750980** | **5605.501961** | **5654.630825** | 1.132862 | 1.527179 | 6 / 20 |
| GED | -2810.497502 | 5636.995005 | 5680.665106 | **1.128532** | 1.539141 | 6 / 21 |

AIC/BIC trên tập huấn luyện ưu tiên **skewed-t**. QLIKE trên tập kiểm định ưu tiên **GED**, còn NLS ưu tiên **Student-t**. Chênh lệch NLS giữa Student-t và skewed-t chỉ khoảng 0,002; bảng này chưa kiểm định ý nghĩa thống kê của khác biệt dự báo.

### Toàn bộ tham số MLE

| Tham số | Student-t | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0.084259 | 0.038084 | 0.080968 |
| `omega` | 0.159377 | 0.153968 | 0.135517 |
| `alpha` | 0.073204 | 0.078518 | 0.069574 |
| `theta` | -0.143328 | -0.128176 | -0.115395 |
| `beta` | 0.691797 | 0.706783 | 0.716551 |
| `gamma_volume` | -0.061695 | -0.059021 | -0.054344 |
| `gamma_rv` | 0.289368 | 0.270164 | 0.263702 |
| `nu` | 4.207035 | — | — |
| `eta` | — | 4.404485 | — |
| `lambda_skew` | — | -0.127379 | — |
| `ged_shape` | — | — | 1.121373 |

### Chẩn đoán 30 điểm khởi tạo

Một lần chạy được tính là về cùng nghiệm tốt nhất khi log-hợp lý thấp hơn nghiệm tốt nhất dưới `1e-4`.

| Phân phối | Số lần về cùng nghiệm / 30 | Độ trải log-hợp lý trong nhóm này | Độ trải lớn nhất của một tham số |
|---|---:|---:|---:|
| Student-t | 29 / 30 | 8,7e-8 | 1,35e-4 |
| Skewed-t | **30 / 30** | 1,3e-7 | 1,23e-4 |
| GED | 26 / 30 | 1,2e-6 | 1,15e-4 |

Một điểm Student-t và bốn điểm GED dừng ở nghiệm có log-hợp lý kém xa. Dù bộ tối ưu báo `success=True`, các điểm này **không phải nghiệm MLE được chọn**. Nghiệm tốt nhất và tham số tương ứng gần như không đổi so với lần chạy 10 điểm trước.

## So với cách tính RV cũ

| Phân phối | AIC cũ → mới | BIC cũ → mới | NLS cũ → mới |
|---|---:|---:|---:|
| Student-t | 5606.868 → 5618.783 | 5650.538 → 5662.453 | 1.520313 → 1.525222 |
| Skewed-t | 5592.364 → 5605.502 | 5641.493 → 5654.631 | 1.523476 → 1.527179 |
| GED | 5624.852 → 5636.995 | 5668.522 → 5680.665 | 1.535838 → 1.539140 |

Lợi suất và ngày chia mẫu giống nhau nên AIC/BIC/NLS cũ và mới có thể đối chiếu như hai đặc tả biến ngoại sinh khác nhau. **Không so trực tiếp QLIKE cũ với QLIKE mới**: đối tượng RV dùng để chấm điểm đã thay đổi. RV trung bình tăng từ 1,3275 lên 1,7682 %²; bình phương lợi suất qua đêm chiếm khoảng 24,68% tổng RV mới, phần nghỉ trưa khoảng 0,75%.

## Giới hạn quan trọng

Tệp nguồn chỉ có `SYMBOL=VN30F1M`, không ghi mã hợp đồng cụ thể hoặc phương pháp ghép chuỗi. Vì vậy **chưa thể xử lý chính xác lợi suất khi chuyển hợp đồng**. Đây không chỉ là rủi ro giả định: ngày 21/05/2020, nến 14:30 đóng tại 809,5 nhưng nến 14:45 của VN30F2005 đóng tại 864; sáng 22/05/2020 chuỗi `VN30F1M` mở tại 789. Dữ liệu hiện tính `100 log(789/864) = -9,08%` vào lợi suất qua đêm và bình phương của nó vào RV. Trong khi đó, [bản tin VNCS ngày 21/05/2020](https://vncsi.com.vn/data/data/anhpnh/files/ban-tin-thi-truong-ngay-21_05_2020.pdf) ghi VN30F2005 đóng 864 còn hợp đồng tháng kế tiếp VN30F2006 đóng 786; `100 log(789/786) ≈ +0,38%` là mốc tham khảo của hợp đồng tháng kế tiếp, không phải một giá trị thay thế đã được áp vào dữ liệu. [HNX quy định ngày giao dịch cuối cùng là thứ Năm thứ ba của tháng đáo hạn](https://cbtd.hnx.vn/vi-vn/huong-dan/chi-tiet-thu-tuc-36-65.html).

Hai ngày 21–22/05/2020 chiếm khoảng 3,33% tổng RV của toàn mẫu, dù chỉ là 2 trong 2.171 ngày. Giá 864 được ghi nhận trong bản tin thị trường nên không thể tự ý sửa thành giá nến trước hay giá thanh toán. Để đánh giá tác động đáo hạn một cách sạch sẽ, cần chuỗi giá có **mã hợp đồng cụ thể và quy tắc roll**, sau đó xây dựng lại chuỗi giá/lợi suất liên tục hoặc phân tích riêng những ngày roll. Kết luận về phân phối tốt nhất hiện vẫn **mang tính tạm thời** đối với rủi ro này. Bình phương lợi suất qua đêm cũng chỉ là một proxy đơn giản cho biến động ngoài giờ giao dịch.

Mã có thể chạy lại tại `scripts/uoc_luong_mle_khoang_nghi.py`; số đầy đủ và chẩn đoán từng điểm khởi tạo nằm trong `egarch_x_full_day_mle.json`. Tệp `daily_rv_components.csv` ghi các thành phần biến động từng ngày để kiểm toán.
