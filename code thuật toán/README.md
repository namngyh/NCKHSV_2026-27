# Code thuật toán — GARCH-VN30F1M-MLE-VB-MCMC

So sánh sáu mô hình GARCH/EGARCH một trạng thái (chuẩn, Student-t, Student-t + biến
ngoại sinh) trên lợi suất ngày VN30F1M, bằng ba phương pháp suy diễn dùng chung một
hàm log-hợp lý: ước lượng hợp lý cực đại (MLE), Bayes biến phân (mean-field,
full-rank) và MCMC (NUTS).

Tài liệu đầy đủ (cách chạy, thư viện, công thức toán, mô tả từng tệp và từng hàm):
**[`tai_lieu/tai_lieu_ma_nguon.pdf`](tai_lieu/tai_lieu_ma_nguon.pdf)**.

## Kết quả mở rộng EGARCH-X

Xem [hướng dẫn mở rộng EGARCH-X](EGARCH_X_MO_RONG.md) để hiểu ba phân phối Student-t, skewed-t và GED, cách tính khoảng nghỉ trưa/qua đêm, nối hợp đồng F1M/F2M, lựa chọn loại cửa sổ đáo hạn, toàn bộ kết quả MLE/VB và lệnh tái lập. Bản được dùng cho ngày giao dịch thông thường trên toàn kỳ là MLE có điều kiện theo đoạn, với 30 điểm khởi tạo cho mỗi phân phối; VB hiện chỉ áp dụng cho dữ liệu gốc.

## Cài đặt và kiểm tra

```
pip install -r requirements.txt
python -m pytest tests -q               # 43 kiểm thử
python scripts/kiem_toan_ket_qua.py     # tính lại mọi con số trong results/ và đối chiếu
```

Trên Windows: mở `chay.bat` để chọn thao tác từ bảng chọn.

## Ba bước nghiên cứu

| Lệnh | Nội dung | Đầu ra |
|---|---|---|
| `python scripts/chay_buoc1_so_sanh_mo_hinh.py` | 6 mô hình × 3 phương pháp trên dữ liệu thật | `results/00_data.pt`, `results/pipeline/*.pt` |
| `python scripts/chay_buoc2_mo_phong.py --jobs 8` | Mô phỏng khôi phục tham số EGARCH-X, trường hợp E1–E3, B = 100 | `results/mo_phong_egarch/` |
| `python scripts/chay_buoc3_mcmc.py` | NUTS cho mô hình thắng cuộc | `results/pipeline/mcmc_<mô hình>.pt` |

Mỗi script có cờ `--nhanh` (cấu hình nhỏ, ghi vào thư mục `*_nhanh` riêng) để thử
đường chạy trong vài phút. Chạy lại một bước sẽ bỏ qua phần đã có kết quả.

Sinh báo cáo và hình: `python scripts/bao_cao/ve_vector_tham_so.py`,
`python scripts/bao_cao/sinh_hinh_tong_hop.py`,
`python scripts/bao_cao/sinh_tong_hop_ket_qua.py` → `Bài báo/tong_hop_ket_qua.tex`.

## Cấu trúc

```
garch_vn30/            thư viện
  du_lieu/             đọc dữ liệu phút, lợi suất, RV, biến ngoại sinh, chia tập
  kiem_dinh/           kiểm định ban đầu (ADF, PP, KPSS, Ljung-Box, ARCH-LM, JB)
  mo_hinh/             phân phối sai số, GARCH(-X), EGARCH(-X), danh mục 6 mô hình
  suy_dien/            MLE, mô hình Bayes (Pyro), VB, MCMC
  danh_gia/            QLIKE, NLS, VaR (Kupiec, Christoffersen), Diebold-Mariano
  mo_phong/            sinh dữ liệu, trường hợp E1-E3, Bias/RMSE/Coverage
scripts/               điều phối ba bước, kiểm toán, sinh báo cáo
tests/                 kiểm thử pytest
results/               kết quả (tệp torch, chỉ chứa dict/tensor)
tai_lieu/              tài liệu mã nguồn (TeX + PDF)
```

Phiên bản trước (có mô hình chuyển trạng thái MS-GARCH-X và kết quả của nó) được
lưu ở nhánh `luu-tru/ms-garch-x` và thẻ `luu-tru-ms-garch-x`.
