# NCKH-2026-2027 — MS-GARCH-X Student-t cho VN30F1M

Đề tài nghiên cứu khoa học sinh viên: xây dựng khung **Markov-Switching
GARCH-X (MS-GARCH-X) với sai số Student-t** để mô hình hóa và dự báo biến
động lợi suất hợp đồng tương lai **VN30F1M**, so sánh ba phương pháp suy diễn
trên cùng một cấu trúc mô hình — **ước lượng hợp lý cực đại (MLE)**,
**Variational Bayes (VB)** và **Markov Chain Monte Carlo (MCMC)**.

Trạng thái biến động thấp/cao được điều khiển bởi một chuỗi Markov ẩn hai
trạng thái; các biến ngoại sinh có thể bao gồm thanh khoản, biến động thực tế
(RV), chênh lệch giá so với chỉ số cơ sở, ngày đáo hạn và thời điểm chuyển
hợp đồng. Khung đánh giá gồm độ chính xác ước lượng qua mô phỏng, chất lượng
dự báo biến động và mật độ ngoài mẫu, kiểm định VaR, độ hiệu chỉnh xác suất
trạng thái và chi phí tính toán.

## Cấu trúc repo

```
Bài báo/                 Bản thảo LaTeX (Bài báo chính.tex, .pdf) và logo
Tài liệu tham khảo/       Tài liệu, bài báo gốc dùng làm tham khảo
code thuật toán/          Cài đặt Python đầy đủ (package msgarchx, script, test)
```

Không gian làm việc chính cho phần lập trình là **[`code thuật toán/`](code%20thuật%20toán/)**
— xem **[`code thuật toán/README.md`](code%20thuật%20toán/README.md)** để biết chi tiết
cấu trúc module, cách cài đặt, chạy thử, chạy ước lượng đầy đủ
(`run_full_study.py`), xuất báo cáo HTML (`build_html_report.py`), hai file
`.bat` chạy nhanh trên Windows, cùng các lưu ý về hiệu năng và giới hạn dữ
liệu hiện có.

## Bắt đầu nhanh (Windows)

Trong thư mục `code thuật toán/`, double-click lần lượt:

1. **`cai_dat_va_chay_full.bat`** — tự cài thư viện rồi chạy toàn bộ quy
   trình ước lượng (kiểm định sơ bộ → MLE → VB → MCMC → mô phỏng). An toàn
   chạy lại nhiều lần, tự bỏ qua các bước đã hoàn tất.
2. **`xem_bao_cao_html.bat`** — tạo và mở báo cáo HTML tổng hợp từ kết quả
   hiện có trong `results/` (dùng được ngay cả khi bước 1 đang chạy dở).

Chi tiết cấu hình, thời gian chạy ước tính và các quyết định thiết kế nằm
trong README của thư mục `code thuật toán/`.

## Dữ liệu

Dữ liệu giá VN30F1M theo phút (`ohlc_export.csv`, nhà cung cấp Data Pro,
2017-11-06 đến 2026-07-17, ~526.000 dòng) đã có sẵn trong `code thuật toán/`.
Do là chuỗi liên tục đã gộp sẵn (không có mã hợp đồng/ngày đáo hạn riêng),
hai biến ngoại sinh `EXP_t`/`ROLL_t` và `BASIS_t` chưa dùng được — xem mục
"Dữ liệu thật đã có" trong README của `code thuật toán/` để biết chi tiết.
