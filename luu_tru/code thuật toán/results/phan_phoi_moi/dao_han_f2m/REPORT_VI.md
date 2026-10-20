# EGARCH-X MLE sau khi xử lý bước chuyển hợp đồng VN30F1M

## Dữ liệu và cách nối chuỗi

- Giữ 2.170 lợi suất ngày từ 07/11/2017 đến 17/07/2026: 1.735 quan sát huấn luyện, 435 quan sát kiểm định. Dữ liệu nến phút VN30F1M dùng để tính biến động thực hiện (`RV`) trong hai phiên, khoảng nghỉ trưa và qua đêm như [bản chạy trước](../khoang_nghi/REPORT_MLE_VI.md).
- Lấy giá đóng cửa ngày VN30F1M và VN30F2M từ [VNDIRECT DChart API](https://dchart-api.vndirect.com.vn/dchart/history). Giá đóng cửa F1M từ API khớp chính xác với CSV nến phút ở 2.170/2.171 ngày; một ngày F1M thiếu từ API là 29/04/2021, không phải ngày đáo hạn. Bản tải, thời điểm tải và tham số nguồn đã lưu trong `vndirect_f1m_f2m_daily.csv` và `source_metadata.json`.
- Theo [lịch đáo hạn HNX](https://cbtd.hnx.vn/vi-vn/huong-dan/chi-tiet-thu-tuc-36-65.html), xác định thứ Năm thứ ba mỗi tháng; khi ngày đó nghỉ giao dịch, dùng phiên quan sát ngay trước đó trong tháng. Có 105 ranh giới đáo hạn và đủ giá F2M tại tất cả các ranh giới.
- Ở phiên đầu tiên sau đáo hạn, thay mẫu số giá đóng cửa F1M hôm trước bằng giá đóng cửa **F2M hôm trước** khi tính `r_t = 100 log(C_t/C_{t-1}^*)` và lợi suất qua đêm `ON_t = 100 log(O_t/C_{t-1}^*)`. Sau đó `RV_t = RV_sáng,t + RV_chiều,t + L_t² + ON_t²`. Hai biến ngoại sinh vẫn là `log(1 + volume_{t-1})` và `log(RV_{t-1} + 1e-6)`, chuẩn hóa bằng thống kê tập huấn luyện. Các ngày khác dùng giá F1M như cũ.
- Đây là phép **nối đúng hợp đồng qua ngày chuyển tháng**. Ngày đáo hạn vẫn chứa giá giao dịch của hợp đồng sắp hết hạn; chưa thể tạo chuỗi chuyển sang F2M từ trước ngày đáo hạn vì chỉ có giá F2M theo ngày, chưa có nến phút F2M cho toàn kỳ.

Tại trường hợp rõ nhất, 21/05/2020 F1M đóng 864, F2M đóng 786; phiên sau mở 789 và đóng 768,9. Lợi suất qua đêm thô là **−9,0806%**, sau khi nối là **+0,3810%**. `RV` ngày 22/05/2020 giảm từ **84,3995** xuống **2,0865 %²**; lợi suất đóng cửa đến đóng cửa đổi từ **−11,6612%** thành **−2,1996%**. Giao dịch đóng 864 ngày 21/05/2020 được giữ nguyên, phù hợp với [bản tin VNCS](https://vncsi.com.vn/data/data/anhpnh/files/ban-tin-thi-truong-ngay-21_05_2020.pdf). Trên 105 phiên sau đáo hạn, trung vị trị tuyệt đối lợi suất qua đêm giảm từ 0,6438% xuống 0,3080%; cực đại giảm từ 9,0806% xuống 2,9317%.

## Bản giữ ngày đáo hạn để đối chiếu: 30 điểm khởi tạo mỗi phân phối

| Phân phối | AIC ↓ | BIC ↓ | QLIKE ngoài mẫu ↓ | NLS ngoài mẫu ↓ | Điểm về gần nghiệm tốt nhất |
|---|---:|---:|---:|---:|---:|
| Student-t | 5551,700 | 5595,370 | 1,113486 | **1,503768** | 28/30 |
| Skewed-t | **5540,191** | **5589,320** | 1,108390 | 1,504970 | 29/30 |
| GED | 5561,546 | 5605,216 | **1,107817** | 1,518386 | 25/30 |

Skewed-t dẫn đầu AIC/BIC trong huấn luyện. GED có QLIKE ngoài mẫu thấp nhất, nhưng chỉ thấp hơn skewed-t khoảng **0,000573**; Student-t có NLS ngoài mẫu thấp nhất, hơn skewed-t khoảng **0,001203**. Chưa có kiểm định ý nghĩa thống kê cho các chênh lệch dự báo nhỏ này. QLIKE dùng `RV` đã sửa làm đại lượng đối chiếu, vì vậy không so trực tiếp trị số QLIKE của bảng này với bản chưa nối hợp đồng. AIC/NLS cũng dùng chuỗi lợi suất đã đổi ở ngày chuyển hợp đồng.

### Toàn bộ tham số MLE

| Tham số | Student-t | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,089441 | 0,047801 | 0,085251 |
| `omega` | 0,149364 | 0,147602 | 0,126538 |
| `alpha` | 0,048249 | 0,055075 | 0,038085 |
| `theta` | −0,150696 | −0,136618 | −0,136950 |
| `beta` | 0,682894 | 0,692457 | 0,707589 |
| `gamma_volume` | −0,058009 | −0,055973 | −0,054075 |
| `gamma_rv` | 0,303572 | 0,289009 | 0,273479 |
| `nu` | 4,309120 | — | — |
| `eta` | — | 4,479145 | — |
| `lambda_skew` | — | −0,117926 | — |
| `ged_shape` | — | — | 1,126255 |

`lambda_skew < 0` biểu thị độ lệch trái theo tham số hóa Hansen. Phân phối nhiễu đều được chuẩn hóa về trung bình 0 và phương sai 1. Hệ số `gamma_volume` và `gamma_rv` tác động lên log-phương sai thông qua hai biến trễ đã chuẩn hóa; không đọc chúng như tác động nhân quả.

## Bản đề xuất cho toàn kỳ: loại cửa sổ đáo hạn

Chạy lại ước lượng có điều kiện theo các đoạn với **30 điểm khởi tạo/phân phối**. Bỏ ngày đáo hạn khỏi hàm hợp lý; phiên kế tiếp chỉ dùng để khởi tạo một đường phương sai mới, không tính vào hàm hợp lý hoặc điểm ngoài mẫu. Từ phiên tiếp theo nữa, tính bình thường với lợi suất và RV đã nối. Phương sai khởi tạo và thống kê chuẩn hóa hai biến ngoại sinh cũng chỉ tính từ các ngày huấn luyện được giữ lại. Như vậy ngày đáo hạn không truyền cú sốc hay `RV` của nó sang phần được chấm điểm. Còn 1.568 quan sát huấn luyện trong hàm hợp lý và 391 ngày ngoài mẫu được chấm; trong đó có 83 kỳ đáo hạn ở huấn luyện và 22 kỳ ở kiểm định. Bản này phù hợp khi câu hỏi nghiên cứu là biến động của **các ngày giao dịch thông thường**, không phải dự báo riêng chính ngày đáo hạn.

| Phân phối | AIC ↓ | BIC ↓ | QLIKE ngoài mẫu ↓ | NLS ngoài mẫu ↓ | Điểm về gần nghiệm tốt nhất |
|---|---:|---:|---:|---:|---:|
| Student-t | 5035,724 | 5078,585 | 1,159331 | **1,520979** | 30/30 |
| Skewed-t | **5024,098** | **5072,316** | 1,155516 | 1,522792 | 30/30 |
| GED | 5045,817 | 5088,677 | **1,154070** | 1,537779 | 29/30 |

Thứ hạng theo từng thước đo giữ nguyên. Tham số độ lệch skewed-t đổi ít (`lambda_skew`: −0,1179 → −0,1222; `eta`: 4,479 → 4,625), nhưng `alpha` đổi từ 0,0551 xuống 0,0058 và `gamma_rv` từ 0,2890 lên 0,4074. Các hệ số động học nhạy với việc cắt và khởi tạo lại đường phương sai hằng tháng. **Không so trực tiếp AIC/BIC hoặc QLIKE giữa hai bảng** vì số quan sát và cách chấm đã khác. Đây là hàm hợp lý có điều kiện sau khi chủ động loại các cửa sổ sự kiện; nó không phải hàm hợp lý đầy đủ của một EGARCH có quan sát bị thiếu.

### Toàn bộ tham số của bản loại cửa sổ đáo hạn

| Tham số | Student-t | Skewed-t | GED |
|---|---:|---:|---:|
| `mu` | 0,082754 | 0,042613 | 0,080286 |
| `omega` | 0,171793 | 0,172795 | 0,163461 |
| `alpha` | 0,002096 | 0,005826 | −0,004799 |
| `theta` | −0,185045 | −0,173522 | −0,174564 |
| `beta` | 0,581459 | 0,590699 | 0,580134 |
| `gamma_volume` | −0,072365 | −0,069187 | −0,073867 |
| `gamma_rv` | 0,422462 | 0,407419 | 0,408177 |
| `nu` | 4,465561 | — | — |
| `eta` | — | 4,624653 | — |
| `lambda_skew` | — | −0,122171 | — |
| `ged_shape` | — | — | 1,124029 |

## Tái lập và giới hạn

Từ thư mục `code thuật toán`:

```powershell
python scripts/uoc_luong_mle_roll_f2m.py --starts 30
python scripts/uoc_luong_mle_loai_ngay_dao_han.py --starts 30
```

Lệnh đầu dùng bản sao giá ngày VNDIRECT đã lưu; thêm `--fetch` nếu muốn cập nhật nguồn, nhưng dữ liệu nguồn có thể được nhà cung cấp sửa về sau. Hai tệp `egarch_x_roll_bridge_mle.json` và `egarch_x_expiry_window_sensitivity.json` chứa tham số đầy đủ, mọi điểm khởi tạo, điểm số và thông tin kiểm toán. `corrected_daily_data.csv` cho phép kiểm tra từng ngày trước/sau nối hợp đồng.

Kết luận có điều kiện theo quy tắc chuyển hợp đồng tại đáo hạn. Chuỗi F2M 5 phút hiện có chỉ bắt đầu từ 2024 nên không thể chuyển sớm trên toàn kỳ 2017–2026 mà vẫn giữ cách tính RV trong phiên thống nhất. Bản loại cửa sổ không đo lường hoặc dự báo rủi ro trong các ngày bị loại. Thứ hạng phân phối tương đối bền, nhưng chưa kiểm định ý nghĩa thống kê của chênh lệch dự báo.
