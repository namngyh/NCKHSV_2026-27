# Thiết kế mã nguồn: đường ống chạy bằng `.bat`, có điểm kiểm tra, bám đề cương

Thiết kế ngày 20/10/2026; **mã đã được triển khai** theo thiết kế này (xem mục 0 của GHI_CHU_DU_AN.md cho tiến độ). Ma trận ở mục 6 phản ánh trạng thái TRƯỚC khi triển khai; trạng thái hiện tại ở mục 6b. Nguồn chuẩn: `de_cuong_EGARCH-X_ky_quy_dong.md` (bản tháng 10/2026). Mục nào trong tài liệu này ghi "[CẦN QUYẾT]" thì chưa chốt.

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
├─ ma_nguon/de_cuong/            Thư viện (không tự đọc/ghi tệp kết quả, trừ khung)
│   ├─ khung.py        cấu hình theo hồ sơ, Buoc, điểm kiểm tra, hồ sơ băm, khóa KT, cổng
│   ├─ du_lieu.py      bảng ngày, lịch đáo hạn/nghỉ, rho^P/GK/RS, biến ngoại sinh không rò rỉ
│   ├─ mo_hinh_jax.py  hàm hợp lý (16) bằng JAX, bốn phân phối, đổi biến (18), tiên nghiệm + Jacobian (20)
│   ├─ phan_phoi.py    NumPy/SciPy: cdf, phân vị, ES (Phụ lục A), lấy mẫu
│   ├─ uoc_luong.py    MLE >= 20 điểm, SE Hessian/bánh kẹp, khả nghịch, bootstrap, VB-MF/FR, NUTS, dự báo
│   ├─ danh_gia.py     điểm số, kiểm định VaR/ES, DM, Giacomini-White, MCS, SPA, VIF, chẩn đoán 4.2
│   ├─ mo_phong.py     E1–E4 (Bảng 6), SBC
│   ├─ ky_quy.py       mô phỏng H phiên (MLE/hậu nghiệm/FHS), quy tắc 5 bước, đối chứng, chỉ tiêu, đường biên
│   └─ mo_hinh.py      bản numba độc lập, chỉ dùng để đối chiếu số ở b02
├─ quy_trinh/                    Các bước, gộp theo giai đoạn (mỗi bước vẫn có hồ sơ và điểm kiểm tra riêng)
│   chay.py (điểm vào)   _chung.py
│   b00_b02_chuan_bi.py      b00 the_che, b01 du_lieu, b02 kiem_tra_cai_dat
│   b03_mo_phong.py          b03 mo_phong_khoi_phuc (+ SBC)
│   b04_b07_giai_doan_1.py   b04 mle_ban_dau, b05 chon_bien, b06 chon_phan_phoi, b07 chot_cau_hinh
│   b08_b09_ngoai_mau.py     b08 cuon_chieu_bon_phuong_phap, b09 danh_gia_KT
│   b10_b12_ky_quy.py        b10 du_bao_H_phien, b11 ky_quy, b12 do_ben
│   b13_b14_bao_cao.py       b13 bao_cao, b14 dong_goi
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
| 2.5, 2.6 | Biến lịch; A, G, F, B, O (+ RV bổ sung 25/10/2026) | b01 | ✔ |
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

## 6b. Trạng thái sau khi triển khai (20/10/2026)

Đã có mã cho mọi mục của ma trận trên. Các mục đã chuyển từ ✘/◐ sang có mã:
- 2.4: tính cả ρ^P, ρ^RS (báo cáo tương quan với r²); ρ* theo (6).
- 3.3.5: kiểm tra khả nghịch.
- 4.1: đổi biến sigmoid, softplus, tanh.
- 4.2:
  - gradient vi phân tự động (JAX);
  - ≥ 20 điểm khởi tạo ở MỌI lần ước lượng, kể cả cuốn chiếu;
  - điểm khởi tạo từ mô hình lồng chuẩn/γ = 0;
  - bootstrap khi β > 0,98.
- 4.3:
  - tiên nghiệm Bảng 5 với Jacobian (20);
  - VB-MF, VB-FR (nhiều lần khởi tạo, Pareto k̂);
  - NUTS 4 chuỗi, ma trận khối lượng đầy đủ, R̂/ESS/phân kỳ;
  - so sánh bốn phương pháp ở mọi mốc.
- 4.4: E1–E4 cho ba phân phối × bốn phương pháp, cộng SBC.
- 5.3: mật độ dự báo hậu nghiệm và PIT/Berkowitz ngoài mẫu.
- 5.5: Giacomini–White và MCS.
- 6: VIF.
- 7.4: đường biên.
- 7.6: độ bền theo w, η và so sánh MLE với hậu nghiệm.
- 8.1: cổng tự động.
- C.8: tệp cấu hình duy nhất và hạt giống theo đơn vị.
- Phụ lục B: GED chạy song song để báo cáo.

Còn lại:
1. **Bảng 3 và chuỗi ký quỹ lịch sử:** người dùng cung cấp. Hồ sơ `day_du` dừng ở b00 khi thiếu nguồn.
2. **Điểm gãy phương sai:** đang dùng Inclan–Tiao, cần đối chiếu với tài liệu [31] của đề cương.
3. **Chi phí tính toán của hồ sơ đầy đủ:** rất lớn, chủ yếu là NUTS ở b03 (4.800 lần lặp + 900 SBC) và b08 (68 mốc × 4 chuỗi). Cần ước tính trên máy mạnh trước khi chạy hết.
4. **Hồ sơ `nhanh`:** chỉ kiểm tra đường ống; con số của nó không có giá trị nghiên cứu.

## 6c. Thiết kế cho Google Colab, mức minh bạch cho bài Scopus (ĐÃ TRIỂN KHAI 20/10/2026)

Trạng thái: các mục A–C dưới đây đã có mã. Đã kiểm chứng:
- Chuỗi chính b00 → b15 chạy trọn ở hồ sơ nhanh.
- Kiểm thử phân mảnh rời nhau, khóa KT, điểm kiểm tra, NUTS/VB chạy tiếp qua tiến trình khác.
- b02 khớp gói `arch`: EGARCH chênh 0,0004; MCS trùng tập; SPA cùng kết luận.

Phát hiện nhờ b02k: bản đầu của kiểm định phân vị động với HAC (ước lượng vững kiểu White) bác bỏ 37–99% dưới H0, do biến hồi quy là chỉ thị hiếm. Đã thay bằng phương sai dưới H0, gồm α(1−α) cộng tự hiệp phương sai của chỉ thị tới bậc H − 1. Cỡ thực tế sau khi sửa (1.000 lần lặp, N = 940):
- α = 5%: khoảng 6–7%.
- α = 2,5%: khoảng 7–9%.
- α = 1%: khoảng 12%. Đây là đặc tính cỡ mẫu nhỏ của kiểm định phân vị động; b02k đo và đưa vào báo cáo.

Vai trò và chia việc: `[phan_tan]` trong cấu hình; điểm vào `quy_trinh/chay.py --vai_tro`; đồng bộ `ma_nguon/de_cuong/dong_bo.py`; Colab `colab/chay_tren_colab.ipynb`. b03 đã tách thành b03a (MLE, chặn cổng) và b03b (Bayes + SBC, chạy nền). Bước mới: b02k, b04b, b08b, b15.

Yêu cầu của người dùng: chạy mọi thứ trên Colab (gói miễn phí); không giới hạn thời gian huấn luyện; đúng cấu hình
đầy đủ của đề cương (không cắt giảm theo mục 8.4); kiểm tra thật kỹ, rõ ràng, minh bạch.

### A. Lỗ hổng điểm kiểm tra phải vá trước khi chạy trên Colab

| Chỗ | Hiện trạng | Hệ quả trên Colab | Sửa |
|---|---|---|---|
| b03 Bayes trong mô phỏng | 1 đơn vị = 1 lần lặp chứa trọn 4 chuỗi NUTS nối tiếp (khoảng 17 giờ với T = 2000) | Vượt thời lượng phiên, **không bao giờ xong** | Tách thành đơn vị (lần lặp, chuỗi); MLE khởi tạo là đơn vị riêng; tổng hợp lần lặp khi đủ 4 chuỗi |
| Chuỗi NUTS (b03, b08) | Một khối khoảng 2,4–4 giờ | Mất phiên là mất trắng | **Tầng điểm kiểm tra thứ ba**: lưu trạng thái sau khởi động (bước nhảy, ma trận khối lượng) rồi lấy mẫu theo đoạn 100 vòng, lưu trạng thái và mẫu sau mỗi đoạn (`MCMC.post_warmup_state`/`last_state` của NumPyro); khởi động cũng chia đoạn |
| VB | 3 lần khởi tạo × 20.000 bước trong một đơn vị | Có thể mất khoảng 30 phút | Mỗi lần khởi tạo là một đơn vị; lưu trạng thái tối ưu mỗi 2.000 bước |
| Phần tổng hợp | So sánh bốn phương pháp (b08), MCS (b09), SPA (b05), MLE trong cổng b07 chạy trong `tong_hop`, không có điểm kiểm tra | Mất phiên là tính lại | Chuyển thành đơn vị có điểm kiểm tra |
| Nhiều phiên Colab cùng chạy | Không có cơ chế giữ chỗ | Hai phiên có thể làm trùng một đơn vị | Tệp giữ chỗ có hạn (ghi giờ và nhịp tim); giữ chỗ quá hạn (phiên chết) được phiên khác lấy lại |
| Hết thời lượng phiên | Bị cắt ngang | Mất đoạn đang dở | `--gio_toi_da`: hết giờ thì hoàn tất đoạn đang dở, ghi điểm kiểm tra, thoát |

### B. Cách chạy trên Colab

- **Mã:** clone từ GitHub (`namngyh/NCKHSV_2026-27`); bản đang dùng được ghi lại bằng mã commit.
- **Kết quả, điểm kiểm tra, nhật ký:** nằm trên Google Drive (`MyDrive/NCKHSV_ket_qua/`). Mất phiên không mất dữ liệu.
- **Notebook `colab/chay_tren_colab.ipynb`:**
  1. Gắn Drive.
  2. Clone hoặc cập nhật repo và ghi mã commit.
  3. Cài `numpyro` (dùng `jax` CPU có sẵn).
  4. In phiên bản thư viện, số lõi, thông tin máy.
  5. Chạy `quy_trinh/chay.py --buoc tat_ca --ket_qua <Drive> --gio_toi_da 11 --phan_manh i/k`.
  6. Xem bảng tiến độ.
  7. Đóng gói kết quả.
- **Chia việc:** mở nhiều notebook cùng lúc, mỗi notebook một phân mảnh `i/k`. Phân mảnh và giữ chỗ bảo đảm không làm trùng.
- **Môi trường:** chọn CPU, không dùng GPU/TPU (đệ quy tuần tự và số thực 64 bit không hợp với GPU/TPU phổ thông). Đặt `spawn` cho tiến trình con.
- **Khi phiên bị ngắt:** mở lại notebook, chạy lại các ô; đường ống tự tiếp tục.

### C. Kiểm chứng và minh bạch (mức nộp tạp chí)

1. **Đối chiếu độc lập các thủ tục thống kê:**
   - MCS và SPA so với gói `arch`.
   - EGARCH phân phối chuẩn và Student-t không biến ngoại sinh so với `arch.arch_model`.
   - Kupiec, Christoffersen, DM so với giá trị tính tay hoặc ví dụ đã công bố.
   - FZ0: kiểm tra tính nhất quán bằng mô phỏng (kỳ vọng nhỏ nhất tại VaR, ES thật).
2. **Kiểm tra cỡ và lực của các kiểm định** (Kupiec, Christoffersen, phân vị động, DM, Berkowitz) bằng mô phỏng dưới H0 với cỡ mẫu bằng khối KT, để biết các kiểm định có đúng mức 5% không.
3. **Nguồn gốc của mọi con số:** mỗi tệp kết quả ghi mã commit, băm mã nguồn, băm cấu hình, băm dữ liệu gốc, `pip freeze`, thông tin máy, hạt giống, thời điểm.
4. **Kiểm toán:** một bước tính lại mọi con số trong báo cáo từ tệp điểm kiểm tra; khác nhau là báo lỗi.
5. **Nhật ký sai lệch so với đề cương:** tự động thu mọi lần chấp nhận khi cổng không đạt, mọi tham số khác giá trị đề cương, mọi đơn vị thất bại; xuất thành phụ lục.
6. **Khóa trước khi xem KT:** giữ nguyên cơ chế khóa cấu hình. Tệp khóa ghi lại toàn bộ lựa chọn và băm của chúng; lựa chọn nào làm sau khóa đều bị đánh dấu.
7. **Rà soát mã độc lập** trước khi chạy đầy đủ.
8. Bổ sung bốn chỗ còn thiếu so với đề cương: kiểm tra dự báo tiên nghiệm và độ nhạy tiên nghiệm (4.3.1), kiểm định phân vị động với HAC (Nhận xét 3), khóa phiên bản (8.5), băm dữ liệu.

### D. Thời gian dự kiến (đúng cấu hình đề cương)

Khoảng 27.000 giờ-lõi theo số đo hiện tại, sai số có thể 2–3 lần.
- Colab miễn phí khoảng 2 vCPU mỗi phiên: một phiên mất khoảng 13.500 giờ chạy; 5 phiên song song mất khoảng 110 ngày chạy liên tục.
- Gói miễn phí có hạn mức sử dụng và không bảo đảm tài nguyên. Thời gian thực tế phụ thuộc vào việc Colab cấp phiên.
- Tăng tốc mà không đổi đề cương: đo lại NUTS khởi tạo từ MLE ở độ sâu cây 10, tối ưu đệ quy JAX. Tham số hóa lại theo mục 8.4 là thay đổi đề cương, chỉ làm khi người dùng đồng ý.

## 7. Lộ trình triển khai (nếu thiết kế được chấp nhận)

0. **Đo tốc độ NUTS** (Pyro so với NumPyro) trên một mốc, chốt động cơ và ngân sách giờ chạy.
1. Commit, gắn thẻ, chuyển tệp sang cấu trúc mới; `luu_tru/` chỉ đọc. Kiểm tra `chay.bat` cũ và các kiểm thử cũ vẫn chạy trong `luu_tru/`.
2. Khung `quy_trinh` + `diem_kiem_tra` + khóa KT + bảng chọn `.bat`, có kiểm thử (tắt giữa chừng rồi chạy lại phải cho kết quả trùng bit).
3. Chuyển phần đã đúng (dữ liệu, mô hình, đánh giá, ký quỹ) vào khung; kết quả mới phải trùng `results/de_cuong/` hiện có.
4. Bổ sung phần còn thiếu trong ma trận (ưu tiên: Bảng 3 → autodiff → Bayes → MCS và Giacomini–White → đường biên và độ bền).
5. Sinh báo cáo, cập nhật tài liệu mã nguồn và thư mục nộp GVHD.
