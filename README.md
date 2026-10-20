# Dự báo rủi ro và ký quỹ động VN30F1M bằng EGARCH-X

Đường ống thực hiện đúng đề cương nghiên cứu (bản tháng 10/2026, `tai_lieu/de_cuong/`): EGARCH-X với ba phân phối sai số, bốn phương pháp suy diễn (MLE, VB mean-field, VB full-rank, NUTS), đánh giá ngoài mẫu theo ba khối thời gian, và cơ chế ký quỹ động.

## Chạy

Mở **`chay.bat`** và chọn trong bảng:

| Mục | Việc |
|---|---|
| 2 | **Chạy chủ trì** (máy mạnh, cùng Colab, đồng bộ qua GitHub): dùng khi chạy chính thức |
| 3 | Chạy một máy (không Colab, không đồng bộ) |
| 4 | Chạy làm đơn vị (máy phụ khác, nếu có) |
| 5 | Xem tiến độ (lấy kết quả mới nhất từ GitHub trước) |
| 6 | Chạy thử nhanh toàn bộ để kiểm tra đường ống (kết quả không có giá trị nghiên cứu) |
| 7–11 | Kiểm thử; chạy vài bước; xóa và chạy lại một bước; sinh lại báo cáo và kiểm toán; đóng gói nộp GVHD |

Dừng lúc nào cũng được (Ctrl+C, tắt máy); chạy lại sẽ tiếp tục từ điểm kiểm tra.

**Colab:** mở `colab/chay_tren_colab.ipynb` trên Colab (File → Open notebook → GitHub), lưu token GitHub vào mục Secrets với tên `GITHUB_TOKEN`, chọn môi trường CPU, rồi Run all. Chỉ dùng một phiên Colab (`colab_1`), nhận khoảng 1/10 phần việc; máy mạnh (`may_manh`) nhận 9/10 và làm mọi phần tổng hợp. Hai nơi không làm trùng việc: mỗi nhóm đơn vị có đúng một máy sở hữu (xem `THIET_KE_CODE.md` mục 6c). Nếu Colab vắng mặt quá 6 giờ, máy mạnh tự làm thay phần của Colab.

`chay.bat` tự cài Python 3.11 và thư viện nếu máy chưa có. Muốn dùng một môi trường Python có sẵn, ghi đường dẫn `python.exe` vào `cau_hinh\python.txt`.

Dòng lệnh tương đương: `python quy_trinh/chay.py --buoc tat_ca [--ho_so nhanh]`, `--tien_do`, `--xoa b05`.

**Trước khi chạy hồ sơ đầy đủ:** điền nguồn xác minh cho Bảng 3 trong `cau_hinh/the_che.toml` (bước b00 dừng nếu còn trống). Nếu có chuỗi ký quỹ lịch sử của VSDC, đặt tại `du_lieu/ngoai/ky_quy_lich_su.csv` (cột `ngay`, `ty_le`).

## Thư mục

| Thư mục | Nội dung |
|---|---|
| `cau_hinh/` | `de_cuong.toml`: mọi tham số (một nguồn duy nhất, hai hồ sơ `day_du`, `nhanh`); `the_che.toml`: Bảng 3 |
| `du_lieu/goc/` | `ohlc_export.csv`: nến phút VN30F1M 06/11/2017 – 17/07/2026 |
| `ma_nguon/de_cuong/` | Thư viện: khung điểm kiểm tra, dữ liệu, mô hình JAX, suy diễn, đánh giá, mô phỏng, ký quỹ |
| `quy_trinh/` | 15 bước b00 – b14 gộp trong 6 tệp, điểm vào `chay.py` |
| `ket_qua/` | Đầu ra của từng bước (`bNN_ten/`), hồ sơ (`_ho_so/`), khóa cấu hình; `ket_qua_nhanh/` cho hồ sơ nhanh |
| `bao_cao/` | Báo cáo và hình sinh tự động (b13) |
| `kiem_thu/` | pytest: Bảng 11, phân phối, khóa khối kiểm tra, điểm kiểm tra |
| `tai_lieu/` | Đề cương (md/tex/pdf), tài liệu tham khảo |
| `luu_tru/` | Chỉ đọc: thiết kế sáu mô hình cũ (`code thuật toán/`), bản thảo cũ (`Bài báo/`) |

Thiết kế chi tiết: [`THIET_KE_CODE.md`](THIET_KE_CODE.md). Ngữ cảnh, tiến độ, việc đang làm: [`GHI_CHU_DU_AN.md`](GHI_CHU_DU_AN.md).

## Các bước

| Bước | Nội dung | Mục đề cương |
|---|---|---|
| b00 – b02 | Thông số thể chế; dữ liệu và kiểm tra chất lượng; kiểm tra cài đặt | 2, C.1 – C.5 |
| b03 | Mô phỏng khôi phục tham số E1 – E4, SBC | 4.4 |
| b04 | MLE trên khối UL, sai số chuẩn, chẩn đoán, LR | 3.6, 4.2 |
| b05 – b06 | Chọn biến ngoại sinh, chọn phân phối trên khối XT | 5, 6 |
| b07 | Kiểm tra điều kiện chuyển giai đoạn, khóa cấu hình (từ đây mới thấy khối KT) | 8.1, C.7 |
| b08 – b09 | Dự báo cuốn chiếu bốn phương pháp; đánh giá ngoài mẫu trên KT | 4.3, 5 |
| b10 – b12 | Dự báo rủi ro H phiên; quy tắc ký quỹ và đối chứng; độ bền | 7 |
| b13 – b14 | Báo cáo; đóng gói | 8.3 |

Tác giả: Nguyễn Hoài Nam, Nguyễn Thuận Hiếu, Đặng Chí Hiếu (Khoa Toán – Thống kê, ĐH Tôn Đức Thắng). Phiên bản trước khi cấu trúc lại: thẻ git `truoc-cau-truc-lai`; mô hình MS-GARCH-X: thẻ `luu-tru-ms-garch-x`.
