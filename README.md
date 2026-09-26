# GARCH-VN30F1M-MLE-VB-MCMC

So sánh họ mô hình GARCH trên lợi suất ngày hợp đồng tương lai VN30F1M bằng ba
phương pháp suy diễn dùng chung một hàm log-hợp lý: ước lượng hợp lý cực đại
(MLE), Bayes biến phân (mean-field, full-rank) và MCMC (NUTS).

Thiết kế ba bước:

1. **So sánh mô hình** — sáu mô hình một trạng thái (GARCH, EGARCH; phân phối chuẩn,
   Student-t, Student-t kèm biến ngoại sinh) × ba phương pháp, trên dữ liệu thật.
2. **Mô phỏng khôi phục tham số** — EGARCH-X Student-t, ba trường hợp E1–E3, 100 lần lặp mỗi trường hợp.
3. **MCMC** cho mô hình thắng cuộc (EGARCH-X Student-t).

## Thư mục

| Thư mục | Nội dung |
|---|---|
| [`code thuật toán/`](code%20thuật%20toán/) | Thư viện `garch_vn30`, script chạy, kiểm thử, kết quả |
| [`code thuật toán/tai_lieu/tai_lieu_ma_nguon.pdf`](code%20thuật%20toán/tai_lieu/tai_lieu_ma_nguon.pdf) | Tài liệu mã nguồn: cách chạy, thư viện, công thức toán, mô tả từng tệp và hàm |
| [`Bài báo/tong_hop_ket_qua.pdf`](Bài%20báo/tong_hop_ket_qua.pdf) | Toàn bộ bảng và hình kết quả (sinh tự động từ `results/`) |
| `Bài báo/` | Bản thảo bài báo |
| `Tài liệu tham khảo/` | Tài liệu tham khảo |

Cách cài đặt và chạy: xem [`code thuật toán/README.md`](code%20thuật%20toán/README.md).

Phiên bản trước, gồm mô hình chuyển trạng thái MS-GARCH-X và kết quả của nó, được
lưu ở nhánh `luu-tru/ms-garch-x` và thẻ `luu-tru-ms-garch-x`.

Tác giả: Nguyễn Hoài Nam (namngyh@gmail.com) và cộng sự.
