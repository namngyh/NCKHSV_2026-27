# EGARCH-X MLE với nến VN30F2M 5 phút: chuyển hợp đồng sớm

## Phạm vi và quy tắc

Tệp VN30F2M 5 phút được dùng trong lần chạy này bắt đầu 02/01/2024. Giữ phần giao nhau với dữ liệu F1M của nghiên cứu đến 17/07/2026: **631 phiên**, 630 lợi suất ngày, 503 quan sát huấn luyện, 127 quan sát kiểm định (từ 12/01/2026). Có **31 ngày đáo hạn**, bảy ngày trong tập kiểm định. Giá đóng cửa ngày lấy từ nến VN30F2M khớp **631/631** giá ngày VNDIRECT đã lưu. Mã SHA-256 của tệp nguồn ghi trong JSON kết quả; bản sao nến 5 phút là dữ liệu nguồn bên ngoài, không đưa lên GitHub.

Gộp nến F1M 1 phút thành nến 5 phút theo thời gian Việt Nam. Loại bản ghi F1M lúc 08:59 vì chuỗi F2M bắt đầu lúc 09:00; giá đóng cửa ngày F1M sau gộp vẫn khớp 631/631 phiên với nguồn ngày. Tách phiên sáng/chiều bằng khoảng trống trưa; `RV = RV_sáng + RV_chiều + L² + ON²`, trong đó RV phiên gồm lợi suất từ mở đến đóng nến đầu tiên và lợi suất giữa các giá đóng nến tiếp theo. Biến ngoại sinh: `log(1+volume)` trễ một ngày và `log(RV+1e-6)` trễ một ngày, chuẩn hóa trên tập huấn luyện của từng biến thể.

So sánh hai chiến lược xử lý cùng một tập ngày:

- **Chuyển sau đáo hạn (late):** dùng F1M trong ngày đáo hạn, rồi dùng giá F2M đóng ngày đáo hạn làm mốc cho phiên kế tiếp. Đây là cách nối trong [bản kết quả toàn kỳ](../dao_han_f2m/REPORT_VI.md), nhưng tính RV từ nến 5 phút và chỉ trên giai đoạn chung.
- **Chuyển trước đáo hạn (early):** giả định chuyển vị thế tại giá đóng cửa phiên ngay trước ngày đáo hạn; dùng F2M cho toàn bộ ngày đáo hạn. Lợi suất ngày đáo hạn và khoảng qua đêm bắt đầu từ **giá F2M đóng phiên trước**, nên không trộn giá hai hợp đồng. Từ phiên sau đáo hạn, chuỗi F1M đại diện cho cùng hợp đồng vừa là F2M.

Trước và trong tất cả 31 ngày đáo hạn, F2M có ít nhất **49 nến/phiên**, bắt đầu 09:00 và có nến cuối 14:45. Tuy nhiên, trung vị khối lượng F2M ở phiên trước đáo hạn chỉ bằng **7,67%** F1M, vì vậy giá thực hiện và chi phí chuyển vị thế cần được đánh giá riêng.

## MLE 30 điểm khởi tạo mỗi mô hình

| Cách chuyển | Phân phối | AIC ↓ | BIC ↓ | QLIKE ngoài mẫu chung ↓ | NLS ngoài mẫu chung ↓ | Điểm gần nghiệm tốt nhất |
|---|---|---:|---:|---:|---:|---:|
| Sau đáo hạn | Student-t | **1420,354** | **1454,119** | 1,346718 | 1,576056 | 30/30 |
| Sau đáo hạn | Skewed-t | 1422,338 | 1460,323 | 1,347023 | **1,575177** | 30/30 |
| Sau đáo hạn | GED | 1424,429 | 1458,194 | **1,289102** | 1,593360 | 27/30 |
| Trước đáo hạn | Student-t | **1429,309** | **1463,074** | 1,377380 | 1,572756 | 30/30 |
| Trước đáo hạn | Skewed-t | 1431,034 | 1469,020 | 1,378519 | **1,570855** | 30/30 |
| Trước đáo hạn | GED | 1431,994 | 1465,759 | **1,295843** | 1,591799 | 22/30 |

**AIC/BIC chỉ so giữa các phân phối trong cùng một cách chuyển.** Lợi suất ngày đáo hạn khác nhau giữa hai cách chuyển nên không so AIC/BIC giữa hai khối. QLIKE và NLS ở bảng trên được chấm trên đúng **120 ngày ngoài mẫu không phải ngày đáo hạn**, với lợi suất và RV mục tiêu giống hệt nhau giữa hai cách chuyển. Đường dự báo phương sai vẫn có thể khác vì quá trình huấn luyện và thông tin từ ngày đáo hạn khác nhau.

Trên 120 ngày chung, cách chuyển trước đáo hạn có QLIKE cao hơn một chút ở cả ba phân phối, nhưng NLS thấp hơn một chút. Ví dụ với GED, QLIKE 1,295843 so với 1,289102; NLS 1,591799 so với 1,593360. **Không có bằng chứng trong bảng này rằng chuyển sớm cải thiện đồng thời cả hai thước đo dự báo.** Chưa kiểm định ý nghĩa thống kê của chênh lệch; tập kiểm định chỉ có 127 ngày.

Trong giai đoạn 2024–2026 và RV nến 5 phút, **Student-t có AIC/BIC thấp nhất** ở cả hai cách chuyển, trong khi GED có QLIKE thấp nhất. Điều này khác kết quả toàn kỳ 2017–2026 với RV nến 1 phút. Vì khoảng dữ liệu và tần suất RV đều đổi, chưa thể quy khác biệt cho riêng quy tắc chuyển hợp đồng.

### Bộ tham số MLE: chuyển trước đáo hạn

| Tham số | Student-t | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,121493 | 0,104641 | 0,104058 |
| `omega` | 0,130370 | 0,126825 | 0,040161 |
| `alpha` | 0,026347 | 0,034960 | 0,136654 |
| `theta` | −0,123315 | −0,116191 | −0,096059 |
| `beta` | 0,673174 | 0,683398 | 0,756436 |
| `gamma_volume` | 0,027576 | 0,023263 | 0,037378 |
| `gamma_rv` | 0,212525 | 0,210581 | 0,110546 |
| `nu` | 2,926716 | — | — |
| `eta` | — | 2,940318 | — |
| `lambda_skew` | — | −0,030959 | — |
| `ged_shape` | — | — | 0,935353 |

### Bộ tham số MLE: chuyển sau đáo hạn, cùng mẫu và tần suất

| Tham số | Student-t | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,124603 | 0,120372 | 0,104058 |
| `omega` | 0,194548 | 0,191292 | 0,044824 |
| `alpha` | −0,107421 | −0,102077 | 0,095475 |
| `theta` | −0,116412 | −0,114403 | −0,096952 |
| `beta` | 0,492289 | 0,500523 | 0,710011 |
| `gamma_volume` | 0,118755 | 0,115426 | 0,071691 |
| `gamma_rv` | 0,258613 | 0,256741 | 0,101601 |
| `nu` | 2,915319 | — | — |
| `eta` | — | 2,918010 | — |
| `lambda_skew` | — | −0,007699 | — |
| `ged_shape` | — | — | 0,928231 |

## Tái lập và giới hạn

Từ thư mục `code thuật toán`:

```powershell
python scripts/uoc_luong_mle_roll_som_5m.py --import-source --source "<duong-dan-den-VN30F2M-5m.csv>" --starts 30
```

Tệp nguồn không được đưa vào repo công khai; sau khi đã nhập một lần, có thể chạy lại với `--starts 30` mà không cần `--import-source`. Tệp cần có các cột `datetime_vn,open,high,low,close,volume`, bao phủ ít nhất 02/01/2024–17/07/2026; kiểm tra SHA-256 trong JSON để tái lập đúng số liệu. JSON kết quả ghi toàn bộ điểm khởi tạo, tham số và điểm số; `early_daily_data.csv` và `late_daily_data.csv` ghi lợi suất, RV, hợp đồng được chọn và giá làm mốc từng ngày.

Chiến lược chuyển sớm đã tránh dùng giao dịch hợp đồng sắp hết hạn trong chính ngày đáo hạn trên phần dữ liệu **từ 2024**. Các tệp chỉ ghi ký hiệu liên tục F1M/F2M, không ghi mã hợp đồng thực tế trên từng dòng; quan hệ F2M ngày đáo hạn thành F1M phiên sau được suy ra từ quy tắc ký hiệu và lịch đáo hạn. Tệp F2M 5 phút hiện có không cho phép tính lại giai đoạn 2017–2023, gồm trường hợp bất thường tháng 05/2020. Việc chuyển vị thế ở giá đóng cửa phiên trước đáo hạn là một giả định; chi phí giao dịch, trượt giá và chênh lệch mua–bán chưa được tính. RV 5 phút cũng là một proxy khác với RV 1 phút của nghiên cứu ban đầu.
