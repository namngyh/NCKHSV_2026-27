# Thiết kế mã nguồn: đường ống chạy bằng `.bat`, có điểm kiểm tra, bám đề cương

Bản nháp thiết kế ngày 20/10/2026, **chưa viết code theo thiết kế này**. Nguồn chuẩn: `de_cuong_EGARCH-X_ky_quy_dong.md` (bản tháng 10/2026). Mục nào trong tài liệu này ghi "[CẦN QUYẾT]" thì chưa chốt.

## 1. Nguyên tắc

1. **Đề cương là yêu cầu bắt buộc.** Mỗi mục của đề cương (2 – 8, Phụ lục A – C) ứng với một bước hoặc một hàm trong mã. Mục 6 bên dưới là ma trận đối chiếu; bước nào lệch đề cương phải ghi lý do vào báo cáo, không lệch ngầm.
2. **Một cấu hình duy nhất** (`cau_hinh/de_cuong.toml`): khối thời gian, lưới siêu tham số, số điểm khởi tạo, hạt giống, ngưỡng chẩn đoán. Không có hằng số rải rác trong mã (Phụ lục C.8).
3. **Điểm kiểm tra ở hai tầng:** mỗi bước có hồ sơ hoàn thành; mỗi đơn vị công việc bên trong bước (một lần ước lượng lại, một lần lặp mô phỏng, một chuỗi NUTS) ghi một tệp riêng. Tắt máy bất kỳ lúc nào chỉ mất tối đa một đơn vị.
4. **Khóa khối kiểm tra (Phụ lục C.7).** Trước bước "chốt cấu hình", hàm truy cập dữ liệu **không trả về** quan sát thuộc KT (cắt cứng, không dựa vào kỷ luật của người viết). Bước sau chỉ chạy khi có tệp khóa chứa mã băm cấu hình đã chọn.
5. **Cổng chuyển giai đoạn (mục 8.1).** Bước ký quỹ chỉ chạy khi cả ba điều kiện: mô phỏng khôi phục đạt độ phủ ≈ 95% và bác bỏ sai ≈ 5% ở E2, E4; trung bình z² theo nhóm τ không khác 1; VaR một phiên không bị bác bỏ trên XT. Cổng ghi kết quả thật, không tự bỏ qua.
6. **Tái lập:** hạt giống cố định theo (bước, đơn vị); lưu `pip freeze`, mã băm mã nguồn và dữ liệu vào hồ sơ mỗi bước.

## 2. Cấu trúc thư mục đề xuất

```
NCKHSV_2026-27/
├─ chay.bat                      Bảng chọn duy nhất (cài đặt, chạy, xem tiến độ, báo cáo)
├─ README.md   CLAUDE.md   GHI_CHU_DU_AN.md   THIET_KE_CODE.md
├─ cau_hinh/
│   ├─ de_cuong.toml             Mọi tham số của đề cương
│   └─ the_che.toml              Bảng 3: thông số thể chế, nguồn, ngày xác minh (người dùng điền)
├─ du_lieu/
│   ├─ goc/ohlc_export.csv       Dữ liệu nến phút (chỉ đọc)
│   └─ ngoai/                    Chuỗi ký quỹ lịch sử nếu thu thập được (mục 7.5, đối chứng 5)
├─ ma_nguon/de_cuong/            Thư viện (không đọc/ghi tệp kết quả)
│   ├─ du_lieu/        nen_phut, ngay, lich (đáo hạn, nghỉ), dac_trung, khoi (có khóa KT)
│   ├─ mo_hinh/        phan_phoi (chuẩn, t*, SkT, GED), egarch_x, uoc_luong_mle
│   ├─ suy_dien/       mle, bayes_mo_hinh, vb (mean-field, full-rank), nuts, chan_doan_bayes
│   ├─ danh_gia/       diem_so, var_es, kiem_dinh, so_sanh (DM, MCS, SPA), chan_doan_mau
│   ├─ mo_phong/       khoi_phuc (E1–E4), sbc, du_bao_H_phien
│   ├─ ky_quy/         quy_tac_5_buoc, doi_chung, danh_gia_ky_quy
│   └─ quy_trinh/      diem_kiem_tra, buoc (lớp cơ sở), cong, ho_so
├─ quy_trinh/                    Các bước, mỗi bước một tệp mỏng gọi thư viện
│   b00_the_che   b01_du_lieu   b02_kiem_tra_cai_dat   b03_mo_phong_khoi_phuc
│   b04_mle_ban_dau   b05_chon_bien   b06_chon_phan_phoi   b07_chot_cau_hinh
│   b08_suy_dien_bayes   b09_danh_gia_KT   b10_du_bao_H_phien   b11_ky_quy
│   b12_do_ben   b13_bao_cao   b14_dong_goi
├─ ket_qua/
│   ├─ b01_du_lieu/ … b14_…/     Đầu ra cuối của từng bước (json, csv, pt)
│   ├─ _diem_kiem_tra/<bước>/    Tệp từng đơn vị công việc
│   ├─ _ho_so/<bước>.json        Mã băm vào/ra, thời gian, hạt giống, trạng thái
│   └─ _nhat_ky/<bước>.log
├─ bao_cao/                      Bảng, hình, tex sinh tự động từ ket_qua/
├─ tai_lieu/                     Đề cương (md/tex/pdf), tài liệu mã nguồn, tài liệu tham khảo
├─ kiem_thu/                     pytest: phân phối, rò rỉ thông tin, khóa KT, điểm kiểm tra
└─ luu_tru/                      ĐÓNG BĂNG, chỉ đọc
    ├─ garch_vn30_sau_mo_hinh/   Thư viện, scripts, results của thiết kế sáu mô hình
    ├─ mo_rong_2026_10_16/       Các script phần mở rộng (MLE roll F2M, loại ngày đáo hạn)
    └─ ban_thao_cu/              Bài báo chính.tex, document.tex, tong_hop_ket_qua…
```

Di chuyển bằng `git mv` để giữ lịch sử; trước khi di chuyển commit trạng thái hiện tại và gắn thẻ `truoc-cau-truc-lai`. Gói `de_cuong/` và các script `dc_*.py` hiện có sẽ được tách vào cấu trúc trên (đã chạy đúng, có thể dùng làm đối chiếu).

## 3. Cơ chế điểm kiểm tra

```
Buoc.chay(ctx):
    h_vao  = băm(cấu hình của bước + tệp đầu vào + mã nguồn của bước)
    nếu ho_so[bước].h_vao == h_vao và đầu ra còn đủ  →  "ĐÃ XONG, bỏ qua"
    cho từng đơn vị u trong danh_sach_don_vi(ctx):
        nếu có _diem_kiem_tra/<bước>/<u>.pkl hợp lệ  →  bỏ qua
        kết quả = thuc_hien_don_vi(u)                 # hạt giống = f(bước, u)
        ghi tệp tạm rồi đổi tên (nguyên tử) + băm
    tong_hop(tất cả đơn vị) → đầu ra cuối, ghi hồ sơ
```

| Bước | Đơn vị công việc (điểm kiểm tra) | Ghi chú |
|---|---|---|
| b03 mô phỏng khôi phục | một lần lặp (kịch bản, phương pháp, b) | 4 × 100 × số phương pháp |
| b04 MLE ban đầu | một phân phối × một cấu hình | ≥ 20 điểm khởi tạo |
| b05 chọn biến | một (tập biến, mốc ước lượng lại) | ước lượng lại mỗi 21 phiên trên XT |
| b08 Bayes | một chuỗi NUTS; một lần VB | NUTS chỉ tiếp tục ở mức chuỗi (giới hạn của Pyro) |
| b09, b10 | một mốc ước lượng lại; một khối phiên gốc mô phỏng | |
| b11 | một cặp (H, α) | lưới hiệu chỉnh trên XT |

Cổng `cong_*` và khóa `KHOA_CAU_HINH.json` nằm ở `quy_trinh/diem_kiem_tra`. Bước nào đổi cấu hình hoặc mã thì mã băm đổi và chỉ bước đó cùng các bước phụ thuộc chạy lại.

## 4. Bảng chọn `chay.bat`

```
1. Kiem tra moi truong / cai thu vien       (giữ cơ chế tự cài Python 3.11 hiện có)
2. Chay toan bo (tiep tuc tu diem kiem tra)
3. Chay mot buoc (nhap so)  ... và tuy chon chay lai
4. Xem tien do (bang: buoc, trang thai, don vi xong/tong, thoi gian, cong)
5. Chay thu nhanh (cau hinh nho, ghi vao ket_qua_nhanh/)
6. Kiem thu + kiem toan (tinh lai cac con so trong ket_qua/)
7. Sinh bao cao (bang, hinh, tex → bao_cao/)
8. Chay lai mot buoc (xoa ho so + diem kiem tra cua buoc do, co xac nhan)
9. Dong goi nop GVHD
```

Mọi lựa chọn đều ghi nhật ký và có thể dừng bằng Ctrl+C; chạy lại ở mục 2 sẽ tiếp tục.

## 5. Phân loại ước lượng theo đề cương

Đề cương yêu cầu **bốn** phương pháp trên cùng một hàm hợp lý (MLE, VB mean-field, VB full-rank, NUTS) và gradient bằng vi phân tự động (mục 4.2, 4.3). Mã tính số hiện có (numba + L-BFGS-B sai phân hữu hạn) chỉ là giải pháp tạm. Thiết kế đề xuất: **một hàm log-hợp lý duy nhất bằng PyTorch** (autodiff), dùng chung cho MLE (L-BFGS), VB và NUTS (Pyro), kiểm tra khớp số với bản numba hiện có (so sánh đến sai số làm tròn) trước khi thay thế.

**Đã chốt (20/10/2026): đủ bốn phương pháp ở mọi mốc ước lượng lại** (khoảng 68 mốc trên XT + KT), Bayes trên cấu hình được chọn (mục 4.3.5).

**Ngân sách tính toán (rủi ro lớn nhất của thiết kế).** NUTS trước đây mất 8,1 giờ cho một lần ước lượng (bốn chuỗi chạy tuần tự, Pyro, đệ quy viết bằng Python). Nhân 68 mốc ≈ 550 giờ tuần tự; chạy song song (12 lõi, 3 mốc × 4 chuỗi) vẫn ≈ 180 giờ (≈ 7,5 ngày). Vì vậy bước 0 của việc triển khai là **đo tốc độ** trên một mốc:

| Động cơ | Tình trạng | Ghi chú |
|---|---|---|
| PyTorch + Pyro | đã cài, kết quả cũ đã kiểm toán | chậm; đề cương không cấm |
| JAX + NumPyro | chưa cài (đề cương mục 8.5 nêu đây là công cụ chính) | biên dịch XLA thường nhanh hơn nhiều lần; cần cài thêm gói, chưa kiểm tra bánh xe Windows |

Sẽ chọn động cơ sau khi đo (khớp số log-hợp lý đến sai số làm tròn với bản numba hiện có là điều kiện bắt buộc). Điểm kiểm tra của NUTS ở mức chuỗi (một chuỗi ≈ 1/4 mốc); các chuỗi và các mốc độc lập nên chạy song song. Nếu sau khi đo vẫn vượt khả năng máy, đề cương mục 8.4 cho phép tăng bước ước lượng lại NUTS lên 42 phiên và thuê tài nguyên (đã dự trù 2,5 triệu đồng); lúc đó phải ghi vào báo cáo như một thay đổi.

## 5b. Các quyết định đã chốt

| Vấn đề | Quyết định |
|---|---|
| Bayes | Đủ bốn phương pháp (MLE, VB-MF, VB-FR, NUTS) ở mọi mốc ước lượng lại |
| Cỡ ba khối | UL = 750, XT = 420, KT = 940 phiên (sau khi bỏ 61 phiên đầu thiếu lịch sử biến ngoại sinh) |
| Di chuyển thư mục | Commit, gắn thẻ `truoc-cau-truc-lai`, rồi `git mv`; phần cũ sang `luu_tru/` chỉ đọc |
| Bảng 3 (thể chế) | Người dùng cung cấp nguồn xác minh; mã chỉ đọc `cau_hinh/the_che.toml` và dừng nếu còn trống |

## 6. Ma trận đối chiếu đề cương → mã

Trạng thái: ✔ đã có và đúng, ◐ có nhưng lệch/thiếu, ✘ chưa có.

| Mục đề cương | Nội dung | Bước | Trạng thái hiện tại |
|---|---|---|---|
| 2.1, 2.2, Bảng 3, C.1 | Thông số thể chế, xác minh từng thời kỳ | b00 | ✘ chưa xác minh; **người dùng phải cung cấp nguồn** |
| 2.3 | Lợi suất, phiên chuyển J_t theo lịch | b01 | ✔ |
| 2.4 | ρ^P, ρ^GK, ρ^RS, ρ*, ĉ_ρ, Nhận xét 1 | b01 | ◐ có GK; thiếu P và RS |
| 2.5, 2.6 | Biến lịch; A, G, F, B, O | b01 | ✔ |
| 2.7 | Chuẩn hóa không rò rỉ | b01, kiem_thu | ◐ đúng nhưng chưa có kiểm thử chống rò rỉ |
| 2.8, C.2, C.3 | Kiểm tra chất lượng, danh sách phiên đặc biệt | b01 | ◐ có số liệu; chưa đối chiếu lịch giao dịch chính thức |
| 3.1 – 3.3 | Đặc tả, ba phân phối, chính quy | b02 | ✔ (có kiểm tra tích phân số; còn phải kiểm tra ν→∞, λ=0 theo C.5 bằng kiểm thử) |
| 3.3.5 | Khả nghịch: lọc từ hai giá trị khởi tạo | b02 | ✘ |
| 3.4, 3.5 | Quan sát khuyết, hàm hợp lý | b02 | ✔ |
| 3.6 | LR kiểm định γ = 0 | b04 | ✔ |
| 4.1 | Tham số hóa không ràng buộc | b02 | ◐ dùng biên thay vì sigmoid/softplus |
| 4.2 | MLE ≥ 20 điểm khởi tạo, khởi tạo từ mô hình lồng | b04 | ◐ 20 điểm ở ước lượng chính; chỉ 8 điểm ở các lần ước lượng lại; chưa khởi tạo từ chuẩn/γ = 0 |
| 4.2 | Sai số chuẩn Hessian và bánh kẹp; bootstrap khi β sát biên | b04 | ◐ chưa có bootstrap |
| 4.2 (i)–(v) | Ljung–Box, hồi quy z²−1, Berkowitz, điểm gãy, ổn định | b04, b09 | ◐ điểm gãy dùng Inclan–Tiao (cần đối chiếu với tài liệu [31]); Berkowitz mới trong mẫu |
| 4.3.1 – 4.3.5 | Tiên nghiệm, VB-MF, VB-FR, NUTS, so sánh bốn phương pháp | b08 | ✘ **chưa có** (xung đột với yêu cầu "chỉ MLE") |
| 4.3.3 | Pareto k̂, hiệu chuẩn dựa trên mô phỏng (SBC) | b03, b08 | ✘ |
| 4.4, Bảng 6 | Mô phỏng khôi phục E1–E4, B = 100, T = 2000 | b03 | ◐ chỉ MLE; thiếu độ phủ khoảng Bayes và SBC |
| 5.1, Bảng 7 | Ba khối UL ≈ 750, XT ≈ 420, KT ≈ 1.000; ước lượng lại mỗi 21 phiên | b05, b09 | ◐ đang 700 / 420 / 990; **đã chốt 750 / 420 / 940** |
| 5.2 | QLIKE trên ρ* | b09 | ✔ |
| 5.3 | LS, mật độ dự báo hậu nghiệm, PIT | b09 | ◐ LS có; mật độ hậu nghiệm và PIT ngoài mẫu chưa có |
| 5.4 | VaR/ES, Kupiec, Christoffersen, phân vị động, FZ0, kiểm định ES | b09 | ✔ |
| 5.5 | DM (HAC), Giacomini–White, tập mô hình tin cậy | b09 | ◐ có DM; thiếu Giacomini–White và MCS |
| 5.6 | Nhóm phiên (đáo hạn, nghỉ dài, căng thẳng) | b09 | ✔ |
| 6 | Chọn tiến, DM một phía, SPA, Holm, VIF, ổn định, minh bạch | b05 | ◐ thiếu VIF; chọn bằng LS với Student-t |
| 7.1, 7.2 | Phân vị tổn thất; mô phỏng H phiên, S = 10⁴ | b10 | ✔ (MLE); ◐ hậu nghiệm chưa có |
| 7.3 | Quy tắc năm bước | b11 | ✔ |
| 7.4 | Lưới siêu tham số, chi phí kỳ vọng, đường biên | b11 | ◐ lưới đủ cho w, b, φ, η; có chi phí, **chưa vẽ đường biên** |
| 7.5 | Sáu đối chứng | b11 | ◐ thiếu chuỗi ký quỹ lịch sử của tổ chức bù trừ (chưa thu thập) |
| 7.6, Nhận xét 3, 4 | Chỉ tiêu, dãy thưa, loại cửa sổ chuyển hợp đồng | b11 | ✔ |
| 7.6 | Kiểm tra độ bền: đổi H, α, w, η; MLE so với hậu nghiệm | b12 | ◐ đổi H, α có; w, η chưa; hậu nghiệm chưa có |
| 8.1 | Điều kiện chuyển Giai đoạn 1 → 2 | cổng | ◐ thực hiện thủ công, chưa thành cổng tự động |
| 8.5, C.8 | Tái lập, tệp cấu hình, hạt giống, khóa phiên bản | mọi bước | ✘ thiết kế này |
| Phụ lục A | Công thức VaR/ES, Bảng 11 | kiem_thu | ◐ cần kiểm thử khớp Bảng 11 |
| Phụ lục B | GED (phương án thay thế) | b06 (tùy chọn) | ◐ có trong mã cũ, chưa tích hợp |
| Phụ lục C | Danh mục kiểm tra | cổng | ✘ chuyển thành cổng tự động |

## 7. Lộ trình triển khai (nếu thiết kế được chấp nhận)

0. **Đo tốc độ NUTS** (Pyro so với NumPyro) trên một mốc, chốt động cơ và ngân sách giờ chạy.
1. Commit, gắn thẻ, chuyển tệp sang cấu trúc mới; `luu_tru/` chỉ đọc. Kiểm tra `chay.bat` cũ và các kiểm thử cũ vẫn chạy trong `luu_tru/`.
2. Khung `quy_trinh` + `diem_kiem_tra` + khóa KT + bảng chọn `.bat`, có kiểm thử (tắt giữa chừng rồi chạy lại phải cho kết quả trùng bit).
3. Chuyển phần đã đúng (dữ liệu, mô hình, đánh giá, ký quỹ) vào khung; kết quả mới phải trùng `results/de_cuong/` hiện có.
4. Bổ sung phần còn thiếu trong ma trận (ưu tiên: Bảng 3 → autodiff → Bayes → MCS và Giacomini–White → đường biên và độ bền).
5. Sinh báo cáo, cập nhật tài liệu mã nguồn và thư mục nộp GVHD.
