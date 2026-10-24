# Ghi chú dự án: dự báo rủi ro và ký quỹ động VN30F1M bằng EGARCH-X

## 000000. GED ĐƯỢC CHỌN, THÊM MÔ PHỎNG KHÔI PHỤC GED (23/10/2026) — mới nhất

Lần chạy có VN30F2M: b03a xong 12.000 đơn vị (09:56), b05 chọn G (DM p = 0,045; thêm B: p = 0,106, loại), b06 chọn
**GED**, cổng b07 KHÔNG ĐẠT (12:06) vì điều kiện (i): "phan phoi ged chua co mo phong khoi phuc tham so (b03a)";
(ii) đạt (F p = 0,84), (iii) đạt (Kupiec p 0,81 / 0,65 / 0,52). Ước lượng UL+XT của ged|G: nu = 1,11, theta = -0,13,
gamma_G = 0,17. Người dùng chọn phương án 1: chạy mô phỏng khôi phục cho GED.
- `quy_trinh/b03a_ged.py` (nạp trong chay.py `MO_DUN`): thêm 4 x B_mle = 4.000 đơn vị `mp|ged|mle|E*|b`, đặt
  `mo_rong_don_vi` cho b03a và tính điều kiện (i) cho cả GED. Không sửa b03_mo_phong.py nên 12.000 đơn vị cũ được dùng lại.
  Thử một đơn vị GED E1 với cấu hình đầy đủ: 55-65 s, khôi phục tốt (nu 1,51 so với 1,5; beta 0,955 so với 0,95).
- Kiểm thử mới `test_b03a_them_ged`; 23/23 đạt.
- Khi cổng không đạt, chay.py trả mã 3 và chay.bat dừng hẳn: phải mở lại chay.bat mục 2 trên máy mạnh.
- PHÁT HIỆN 14:31: lần chạy b03a-b07 vừa xong dùng dữ liệu CŨ (16/08/2019 vẫn khuyết): máy không khởi động lại khi chỉ
  dữ liệu đổi (commit 19:55 ngày 22/10), vì `dong_bo.bam_ma` chỉ băm mã và cấu hình. Khi mở lại (14:31 ngày 23/10), máy kéo
  dữ liệu mới nên chạy lại từ b00, b03a làm đủ 16.000 đơn vị (khoảng 18-20 giờ). Kết quả chọn G/GED và cổng ở trên chỉ
  tham khảo. Đã sửa `bam_ma` băm thêm `du_lieu/goc`, `du_lieu/ngoai` (trừ .md) để lần sau sửa dữ liệu thì máy tự khởi động lại.
- 21:27 ngày 23/10 (người dùng chọn): GED cũng được thêm vào b03b (`b03b_so_lan_lap.py` dùng `cac_phan_phoi`): thêm
  29.800 đơn vị VB-MF, VB-FR, NUTS, SBC; tổng 119.200 (trước 89.400); ước tính b03b khoảng 20 ngày với 15 tiến trình.
  Đã chạy thử GED ở hồ sơ nhanh cho cả bốn loại đơn vị. Kiểm thử 23/23.
- Sự cố máy mạnh đêm 23/10: đồng bộ cuối 22:22, chạy lại 00:10 ngày 24/10, tiếp tục đúng chỗ dở (điểm kiểm tra từng đơn vị).
  Lần khởi động 14:39 và 21:51 chỉ chạy khoảng 8-9 tiến trình (RAM trống thấp lúc khởi động).
- Tài liệu tham khảo truy cập mở đã tải về `Desktop/Google Drive/NCKHSV_EGARCHX_Tai_lieu` (danh mục ở DANH_MUC_TAI_LIEU.md);
  qua đó xác nhận murphy2016 (BoE Staff WP 597), pfmi2012 (CPSS-IOSCO, tháng 4/2012) và thông báo VSDC 18/10/2024 (17%).
- Vẫn hoãn: sửa `danh_gia.chan_doan` (hồi quy theo biến lịch bỏ cột phiên chuyển). Sửa danh_gia.py làm b02 đổi h_ra nên
  b03a (16.000 đơn vị) chạy lại từ đầu; chỉ sửa khi chấp nhận chạy lại.

## 00000. XỬ LÝ ĐÁO HẠN BẰNG VN30F2M (22/10/2026, chiều)

Người dùng bổ sung `VN30F2M_1d` (18/08/2017 - 21/10/2026, phủ 2.170/2.171 phiên F1M) và chấp nhận làm lại toàn bộ.
- `[du_lieu].xu_ly_dao_han = "f2m"`: tại phiên chuyển r_t = 100 ln(F1M_t / F2M_{t-1}), qua đêm tương tự với giá mở.
  Kiểm tra: độ lệch chuẩn lợi suất phiên chuyển 1,95% (cách cũ, chứa cú nhảy cơ sở) -> 1,27% (cùng hợp đồng),
  ngày thường 1,48%. Chỉ còn 1 lợi suất khuyết (phiên đầu mẫu); F2M ngày 15/08/2019 vắng trong tệp gốc đã bổ sung 877,0 (xem du_lieu/goc/VN30F2M_bo_sung.md).
- Cột `J` = chỉ thị phiên chuyển theo lịch (biến giả D_sau, nhóm phiên); cột mới `khuyet` = lợi suất khuyết (hàm hợp
  lý, đệ quy). `dac_trung()["J"]` nay trả `khuyet`, thêm `J_chuyen`.
- Ký quỹ: `tong_that_thuc` tính R_{t,H} từ chuỗi lợi suất cùng hợp đồng, nên cửa sổ qua phiên chuyển không còn bị loại.
- Mô phỏng: `[mo_phong].khuyet_phien_chuyen = false` (dữ liệu giả không khuyết), b03a chạy lại (người dùng chọn).
- `bam_du_lieu` băm cả tệp F2M, nên mọi bước chạy lại. Kết quả cũ gắn thẻ git `ket-qua-truoc-f2m`.
- Chưa làm: chạy cách cũ ("loai") bằng MLE làm độ bền; quy đổi ký quỹ ra tiền; độ bền theo tập biến và cửa sổ (BM02 mục 6).

## 0000. BÁM ĐỀ CƯƠNG ĐÃ ĐĂNG KÝ BM02 (22/10/2026) — mới nhất, đọc trước

Người dùng chốt: "muốn bám sát đề cương đã đăng ký nhất" (`BM02_Decuong.docx`). Thay đổi so với đề cương chi tiết:
- **Chia mẫu theo BM02 mục 4**: 80% phiên đầu (đến 16/10/2024) là trong mẫu = UL (750) + XT (925); KT = 20% cuối, 435 phiên
  (17/10/2024 - 17/07/2026). Cấu hình: `[khoi_bm02].XT = 925` ghi đè `[khoi].XT = 420` (giữ giá trị cũ ở `[khoi]` chỉ để
  mã băm của b01, b02, b03a không đổi, nên 8.000+ đơn vị b03a đã chạy không bị mất; xem `BUOC_KHONG_PHU_THUOC_XT` trong khung.py).
  b02k, b04 trở đi chạy lại với khối mới. Bảng 7 của báo cáo tính lại từ cấu hình hiện hành (b13 `_bang_khoi`).
- **GED tham gia chọn phân phối** (b06): `[bam_sat_bm02].ged_tham_gia_chon = true`. Mô phỏng khôi phục (b03a) KHÔNG có GED
  (BM02 không có mô phỏng); nếu GED được chọn thì cổng b07 điều kiện (i) ghi "chưa có mô phỏng" và không đạt, khi đó cần
  thêm b03c (mô phỏng MLE cho GED) hoặc ghi nhận.
- **GARCH(1,1) làm mô hình nền** (BM02 mục 3, 5, câu hỏi 1): `ma_nguon/de_cuong/garch_nen.py`, chạy MLE cuốn chiếu ở b08
  (phân phối đã chọn), so sánh ở b09 (`CH1_EGARCH-X_vs_GARCH`). Kiểm thử: `kiem_thu/test_garch_nen.py` (khớp gói arch).
- **b03b số lần lặp riêng** (22/10/2026, người dùng chọn): `[mo_phong_bayes]` B_vb = 1000 (VB-MF, VB-FR), B_nuts = 300.
  Mục này không được băm và b03b có cờ `mo_rong_don_vi`, nên tăng B sau này chỉ chạy thêm phần thiếu. Mã ở
  `quy_trinh/b03b_so_lan_lap.py` (không sửa b03_mo_phong.py vì b03a băm cả tệp). `[mo_phong].B` không còn dùng.
- Chưa làm (cần xác minh): độ bền theo tập biến ngoại sinh và cửa sổ ước lượng (BM02 mục 5, 6) trong b12; quy đổi ký quỹ ra
  tiền theo hệ số nhân K = 100.000 đồng/điểm.

## 000. TRẠNG THÁI MỚI NHẤT (20/10/2026 khoảng 21:30) — ĐÃ PUSH GitHub, commit 39c13b2

- Đã triển khai và push toàn bộ đường ống chạy song song **1 máy mạnh (chủ trì, 9/10 việc) + 1 phiên Colab (`colab_1`, 1/10)**. Hướng dẫn chạy: README.md và mục "Cách chạy" trong cuộc trao đổi ngày 20/10 (A–F).
- Người dùng yêu cầu bỏ phép thử giả lập nhiều máy và push luôn. Vì vậy **đồng bộ git giữa hai máy CHƯA được kiểm chứng trọn vòng**. Phép thử dở cho thấy:
  - Máy làm đơn vị chờ đúng khi máy chủ trì chưa xong.
  - Lỗi `dong_bo.day` không push (tên đường dẫn ngắn 8.3) đã được sửa nhưng chưa chạy lại.
  - Lần chạy chính thức đầu tiên cần theo dõi: sau khoảng 30 phút, kiểm tra trên GitHub có commit "ket qua may_manh ..." và "ket qua colab_1 ..."; `chay.bat` mục 5 phải thấy đơn vị của cả hai máy.
- Kiểm thử 15/15 đạt. Chạy thử nhanh một máy trọn b00–b15 đạt (trước các sửa đổi cuối về nhận thay, bộ nhớ và đồng bộ).
- Việc người dùng phải làm trước khi chạy chính thức:
  1. Nguồn xác minh Bảng 3 (`cau_hinh/the_che.toml`).
  2. Token GitHub (quyền Contents read/write) lưu trong Colab Secrets với tên `GITHUB_TOKEN`.
  3. Cân nhắc chuyển repo sang riêng tư (kết quả sẽ công khai khi push).
- Thẻ `truoc-cau-truc-lai` chỉ có ở máy này (chưa push thẻ).

## 00. (cũ) ĐANG LÀM (20/10/2026 chiều): triển khai chạy song song máy mạnh + Colab, rồi push GitHub

Người dùng chốt:
- Mục tiêu bài Scopus: kiểm tra thật kỹ, minh bạch, không ngại thời gian; cấu hình ĐẦY ĐỦ theo đề cương (phương án 1).
- 1 máy mạnh (Windows, dưới 16 lõi, chủ trì) + Colab (gói miễn phí, chỉ làm đơn vị).
- Đồng bộ qua GitHub (repo CÔNG KHAI `namngyh/NCKHSV_2026-27`), push kết quả tuần tự; giữ dữ liệu trong repo.
- Đồng ý tách b03 thành b03a (MLE, chặn cổng) và b03b (Bayes + SBC, chạy nền).
- Người dùng đã yêu cầu push ("OK bắt đầu thiết kế và push lên git đi").

Đã làm trong lượt này:
- `khung.py` viết lại: vai trò mot_minh/chu_tri/don_vi; chủ sở hữu nhóm theo băm (`chu_so_huu`, NUTS chỉ cho `may_nuts`); `HetGio`/`con_thoi_gian` (biến NCKHSV_HAN); tham số `han` (hộp thời gian); `ctx.thu_muc_doan` (điểm kiểm tra tầng 3); băm dữ liệu gốc trong h_vao; `nguon_goc.json`; nhật ký theo máy; tiến trình con dùng spawn; NCKHSV_KET_QUA đổi thư mục kết quả.
- `de_cuong.toml`: thêm [do_nhay_tien_nghiem], [phan_tan], [de_cuong_goc], `nuts_vong_moi_doan`, `vb_buoc_moi_doan`.
- `mo_hinh_jax.py`: phương án tiên nghiệm `pa` (TIEN_NGHIEM_GOC = Bảng 5). `uoc_luong.py`: `vb` và `nuts_chuoi` chia đoạn, chạy tiếp được (đã thử chạy tiếp qua 2 tiến trình: đạt).
- `danh_gia.phan_vi_dong_hac` + `ky_quy.kiem_dinh_vuot` có `dq_hac_p`.
- `mo_phong.py` viết lại (du_lieu_lan, don_vi_mle, don_vi_bayes, gop_bayes, hang_sbc). `quy_trinh/b03_mo_phong.py` thành b03a + b03b.

CẬP NHẬT (20/10 tối): ĐÃ XONG các mục 1–6 dưới đây.
- Mục 1–4: đã thêm b02 đối chiếu arch (đạt: EGARCH chênh 0,0004; MCS trùng; SPA cùng kết luận), b02k, b04b, b08b, b13 có phụ lục sai lệch, b15 kiểm toán.
- Mục 5: `dong_bo.py` và `chay.py` vòng lặp.
- Mục 6: chay.bat mới, notebook `colab/chay_tren_colab.ipynb`, requirements ghim, .gitignore theo dõi ket_qua/.
- 15/15 kiểm thử đạt.

PHÁT HIỆN QUAN TRỌNG: b02k cho thấy kiểm định phân vị động với HAC (`danh_gia.phan_vi_dong_hac`) bác bỏ 39–95% dưới H0. Nghi do mô phỏng dùng VaR gần như hằng số (gần trùng hằng số chặn). Đã sửa mô phỏng (biến hồi quy độc lập) và đang chạy lại kiểm tra cỡ 2.000 lần. Nếu vẫn sai cỡ thì phải sửa chính kiểm định trước khi chạy thật.

ĐANG CHẠY: thử nhanh toàn bộ (log `ket_qua_nhanh_chay.log`).

CÒN LẠI:
- Xem kết quả 2 việc đang chạy.
- Giả lập 2 máy với repo git trần cục bộ.
- Cập nhật THIET_KE_CODE 6c thành "đã triển khai".
- Commit, push.

CÒN PHẢI LÀM (danh sách gốc, mục 1–6 đã xong):
1. `b04_b07`: b07 phụ thuộc b03a (thay b03); đặt `chi_chu_tri=True` cho b00, b01, b02, b04, b07, b11–b14; thêm b04b (kiểm tra dự báo tiên nghiệm, UL); đọc `mo_phong_khoi_phuc.json` từ b03a.
2. `b08_b09`:
   - VB mỗi lần khởi tạo là 1 đơn vị (`vb_mf|kkk|l#`); NUTS dùng `thu_muc_doan`.
   - `_so_sanh_bon` và MCS thành đơn vị; thêm b08b (độ nhạy tiên nghiệm bằng NUTS tại mốc đầu KT, các phương án trong cấu hình).
3. `b00_b02`: thêm đối chiếu `arch` (MCS, SPA, EGARCH chuẩn) vào b02; thêm b02k (cỡ và lực kiểm định bằng mô phỏng).
4. Thêm b15 kiểm toán; nhật ký sai lệch (so với [de_cuong_goc]) trong b13; b13 không phụ thuộc b03b, làm lại khi b03b xong.
5. `chay.py`:
   - Thêm `--vai_tro --may --gio_toi_da --git --phut_dong_bo`.
   - Vòng lặp chủ trì/đơn vị: chuỗi chính rồi b03b làm việc lấp chỗ trống; push định kỳ.
   - Phát hiện mã đổi sau pull thì dừng.
   - Thêm `dong_bo.py` (git pull/push có thử lại; đăng ký máy `ket_qua/_may/<may>.json`) và ghi `pip freeze` vào `ket_qua/_moi_truong/<may>.txt`.
6. `chay.bat`: mục "Chạy chủ trì", "Chạy làm đơn vị". Notebook `colab/chay_tren_colab.ipynb` (token trong Colab Secrets). `requirements.txt` ghim `==`. `.gitignore` theo dõi `ket_qua/` (gồm `_diem_kiem_tra`, `_nhat_ky`), vẫn bỏ `ket_qua_nhanh/`.
7. Kiểm thử: phân mảnh rời nhau và phủ đủ; chạy tiếp NUTS/VB; DQ-HAC; arch. Chạy nhanh toàn bộ; giả lập 2 máy với repo git trần cục bộ.
8. Cập nhật README, THIET_KE_CODE (6c đã triển khai), GHI_CHU; commit; push.

## 0. (cũ) ĐANG LÀM DỞ (tạm dừng 20/10/2026 khoảng 11:50)

**Mục tiêu đang làm:** triển khai `THIET_KE_CODE.md`: đường ống chạy bằng `chay.bat`, có điểm kiểm tra, bám đề cương; trên máy này chỉ chạy thử hồ sơ `nhanh`, bản đầy đủ chạy trên máy khác mạnh hơn.

**Quyết định đã chốt:**
- Đủ bốn phương pháp (MLE, VB-MF, VB-FR, NUTS) ở mọi mốc ước lượng lại.
- Khối UL/XT/KT = 750/420/940.
- Đã commit `b2294ef`, gắn thẻ `truoc-cau-truc-lai`, rồi `git mv`. Các thay đổi sau thẻ CHƯA commit; hoàn tác bằng `git reset --hard truoc-cau-truc-lai && git clean -fd`.
- Đã cài `jax` 0.10.2 và `numpyro` 0.22.0 vào môi trường `quant311`.

**Đã xong:**
- **Cấu trúc thư mục mới:**
  - `cau_hinh/`: `de_cuong.toml`, `the_che.toml`, `python.txt` (python.txt không theo dõi bằng git).
  - `du_lieu/goc/ohlc_export.csv`.
  - `ma_nguon/de_cuong/`: khung, du_lieu, mo_hinh_jax, phan_phoi, uoc_luong, danh_gia, ky_quy, mo_phong; `mo_hinh.py` (numba) chỉ giữ để đối chiếu ở b02.
  - `quy_trinh/`: `chay.py`, `_chung.py` và các tệp bước b00_b02, b03, b04_b07, b08_b09, b10_b12, b13_b14.
  - `kiem_thu/test_duong_ong.py`, `chay.bat` ở gốc, `requirements.txt` mới.
  - `luu_tru/`: `code thuật toán/`, `Bài báo/`, chỉ đọc.
  - `tai_lieu/de_cuong/`, `tai_lieu/tham_khao/`.
- **Đã kiểm chứng:**
  - Bảng 11 khớp đến 3 chữ số; ES của SkT theo công thức đóng khớp cầu phương.
  - JAX khớp numba: log-hợp lý chênh 1,4e-12, MLE chênh 5e-8, sai số chuẩn trùng mọi chữ số.
  - Khả nghịch: chênh lệch tắt dưới 1e-6 sau 37 phiên.
  - b00–b02 chạy được ở hồ sơ nhanh.
  - b03 nhanh chạy được 170/171 đơn vị. Đơn vị lỗi (SBC student VB-MF, β làm tròn thành 1 làm log tiên nghiệm = −∞) đã sửa trong `mo_hinh_jax.log_tien_nghiem` (tính theo ξ), và b03 nay ghi nhận đơn vị thất bại.
- **Đo tốc độ NUTS:** 7,3 ms mỗi gradient, mọi vòng chạm trần 1.023 bước leapfrog khi khởi tạo ngẫu nhiên, khoảng 9,5 giờ một mốc nếu 4 chuỗi chạy tuần tự. Khởi tạo từ MLE thì khoảng 29 bước ở độ sâu 5.

**Đang chạy lúc dừng:** `quy_trinh/chay.py --ho_so nhanh --buoc tat_ca --chap_nhan_cong`, nhật ký ở `ket_qua_nhanh_chay.log` (thư mục gốc). Đây là lần chạy thử đầu tiên của b04–b14, CHƯA biết kết quả. Có thể đang lỗi hoặc đã xong.

**Việc tiếp theo, theo thứ tự:**
1. Đọc `ket_qua_nhanh_chay.log` và `python quy_trinh/chay.py --tien_do --ho_so nhanh`; sửa lỗi của b04–b14 cho tới khi hồ sơ nhanh chạy hết b14.
2. `python -m pytest kiem_thu -q` (chưa chạy lần nào).
3. Chạy thử `chay.bat` (mục 4 và 5) trên máy này.
4. Cập nhật `THIET_KE_CODE.md`: các bước nay gộp theo giai đoạn trong 6 tệp, không phải 15 tệp. Cập nhật `README.md` và `CLAUDE.md` theo cấu trúc mới, và ma trận đối chiếu (nhiều mục ✘ nay đã có mã).
5. Dọn tệp thừa: `ma_nguon/de_cuong/jax_mo_hinh.py` (đã thay bằng `mo_hinh_jax.py`), `quy_trinh/do_toc_do_nuts_tam.py`, các tệp `ket_qua_nhanh_*.log`. Bản chụp `jax_mo_hinh.py` trong `luu_tru/` vô hại.
6. Commit sau khi người dùng đồng ý.
7. Ước tính giờ chạy cho máy mạnh và báo người dùng. b03 đầy đủ có 3 phân phối × 4 phương pháp × 4 kịch bản × 100 = 4.800 đơn vị cộng 900 đơn vị SBC, trong đó có NUTS, nên rất nặng. Đề cương mục 8.4 cho phép giảm nếu cần, nhưng phải ghi là thay đổi và người dùng quyết.

**Còn phải người dùng cung cấp:** nguồn xác minh Bảng 3 (`cau_hinh/the_che.toml`). Hồ sơ `day_du` sẽ dừng ở b00 nếu còn trống. Chuỗi ký quỹ lịch sử VSDC nếu có, đặt ở `du_lieu/ngoai/ky_quy_lich_su.csv` (cột ngay, ty_le).

Cập nhật: 20/10/2026. Tệp này tổng hợp ngữ cảnh, tiến trình và các điểm cần lưu ý. Quy ước vận hành chi tiết cho Claude nằm ở `CLAUDE.md`; số liệu chính thức nằm ở `Bài báo/tong_hop_ket_qua.tex` và `code thuật toán/results/`.

## 1. Ngữ cảnh

- **Đề tài:** dự báo rủi ro và xây dựng cơ chế ký quỹ động cho hợp đồng tương lai VN30F1M bằng EGARCH-X.
- **Nhóm:** Nguyễn Hoài Nam, Nguyễn Thuận Hiếu, Đặng Chí Hiếu (Khoa Toán – Thống kê, ĐH Tôn Đức Thắng). GVHD: TS. Châu, muốn ứng dụng thực tế rõ hơn.
- **Đề cương hiện hành:** `../../de_cuong_EGARCH-X_ky_quy_dong.md` (và `.tex`, `.pdf`), bản tháng 10/2026.
- **Dữ liệu:** 525.635 nến 1 phút `VN30F1M` (06/11/2017 – 17/07/2026) → 2.170 lợi suất ngày; chia theo thời gian 1.735 huấn luyện / 435 kiểm định (từ 17/10/2024).
- **Hai giai đoạn:** (1) mô hình rủi ro EGARCH-X (MLE, VB mean-field, VB full-rank, NUTS); (2) chuyển VaR/ES nhiều phiên thành tỷ lệ ký quỹ động, so với các quy tắc đối chứng.
- **Câu hỏi nghiên cứu:** CH1 biến ngoại sinh có cải thiện dự báo không; CH2 phân phối sai số và phương pháp suy diễn ảnh hưởng thế nào; CH3 ký quỹ động có tốt hơn đối chứng không.
- **Tiến độ dự kiến:** 10/2026 – 04/2027, báo cáo cấp Trường 05/2027. Kinh phí đề nghị 3.000.000 đồng.

## 2. Lịch sử quyết định

| Ngày | Sự kiện |
|---|---|
| 13–15/09/2026 | Bước 1, 2, 3 hoàn tất. EGARCH-X Student-t thắng; MCMC hội tụ (MS-GARCH-X trước đó không hội tụ). |
| 21–22/09/2026 | Kiểm toán độc lập kết quả, đổi tên dự án, dọn thư mục. |
| 25–26/09/2026 | Chốt "mô hình chính là EGARCH-X". Gỡ MS-GARCH-X (lưu ở nhánh/thẻ `luu-tru/ms-garch-x`). Viết lại thư viện thành `garch_vn30`. |
| 29–30/09/2026 | Chú thích tiếng Việt cho mọi tệp/hàm; `chay.bat` tự cài môi trường. |
| 16/10/2026 | Công bố phần mở rộng: skewed-t, GED, RV khoảng nghỉ trưa/qua đêm, nối F1M/F2M, loại cửa sổ đáo hạn. |

Đừng đưa MS-GARCH-X trở lại nếu chưa được yêu cầu.

## 3. Kết quả đã có

### 3.1. Ba bước nghiên cứu gốc (6 mô hình, 435 phiên kiểm định)
- **Bước 1:** EGARCH-X Student-t + MLE thắng cả bốn tiêu chí: AIC 5606,87 · BIC 5650,54 · QLIKE 0,9234 · NLS 1,5203.
- **Bước 2 (mô phỏng, E1–E3, B=100, T=2000):** E1/E3 θ̂ âm 100/100 lần; E2 (θ₀=0) không bịa ra đòn bẩy; coverage MLE (Wald) 0,89–0,98.
- **Bước 3 (NUTS):** R̂ 1,0125, ESS 1.340, 0 phân kỳ, 8,1 giờ. θ = −0,1399, KTC 95% [−0,2125; −0,0710] loại trừ 0. VB-MF đánh giá thấp bất định của ω và β (R_sd ≈ 0,37 và 0,29).
- **Phân rã ΔAIC:** đuôi dày −236,2 (GARCH) / −232,0 (EGARCH); biến ngoại sinh −49,2 (EGARCH) so với −12,8 (GARCH); dạng EGARCH −42,0.

### 3.2. Phần mở rộng (MLE, ngày giao dịch thông thường, 1.568 quan sát huấn luyện, 391 ngoài mẫu)

| Phân phối | AIC | BIC | QLIKE | NLS |
|---|---:|---:|---:|---:|
| Student-t | 5035,724 | 5078,585 | 1,159331 | **1,520979** |
| Skewed-t | **5024,098** | **5072,316** | 1,155516 | 1,522792 |
| GED | 5045,817 | 5088,677 | **1,154070** | 1,537779 |

Chưa kiểm định ý nghĩa thống kê của các chênh lệch ngoài mẫu. Không so AIC/BIC/QLIKE giữa các bản có tập quan sát khác nhau (xem 4.1).

### 3.3. Thực hiện theo đề cương, chỉ MLE (20/10/2026)

**Quyết định của người dùng 20/10/2026:** không làm VB nữa, từ nay chỉ ước lượng MLE; làm bám sát đề cương. Mã mới nằm ở gói `code thuật toán/de_cuong/` (`du_lieu`, `mo_hinh`, `danh_gia`, `ky_quy`) và `scripts/dc_chay.py` (bước 1–5), `dc_mo_phong.py`, `dc_ky_quy.py`; kết quả ở `results/de_cuong/`. Gói `garch_vn30` cũ không bị sửa.

Thiết kế: bỏ 61 phiên đầu (thiếu lịch sử 60 phiên cho biến ngoại sinh) → 2.110 phiên mô hình; UL = 700, XT = 420, KT = 990 phiên (942 lợi suất, vì 102 phiên chuyển hợp đồng là quan sát khuyết, không bị loại khỏi chuỗi). Phiên chuyển hợp đồng J_t và ngày đáo hạn được xử lý đúng mục 2.3/3.4 (khuyết + biến giả D_đh, D_sau, D_nghỉ), không bỏ ngày đáo hạn. Ước lượng lại mỗi 21 phiên, cửa sổ mở rộng, thống kê chuẩn hóa chỉ dùng dữ liệu đến thời điểm ước lượng lại.

| Bước | Kết quả chính |
|---|---|
| Dữ liệu (mục 2.8) | 2.171 phiên, 105 phiên chuyển, 105 đáo hạn, 13 phiên sau nghỉ dài, 0 phiên khóa biên, 8 phiên |r| > 7% cần đối chiếu tay. |qua đêm| trung vị 0,64% ở phiên chuyển so với 0,21% ở phiên thường. c_rho = 1,46 |
| Ước lượng (UL, skewed-t, 20 điểm khởi tạo đều hội tụ cùng nghiệm) | ν = 4,52; λ = −0,15; θ = −0,20; β = 0,75; γ_G = 0,30; δ_đh = 0,91. Chẩn đoán Ljung–Box, Berkowitz, hồi quy z²−1 lên D và τ đều không bác bỏ. LR γ = 0 bị bác bỏ ở cả ba phân phối (p < 0,001) |
| Chọn biến (XT, Student-t) | Chọn tiến chỉ giữ **G** (DM p = 0,017); thêm biến thứ hai không có ý nghĩa (p = 0,13). **d_X = 1, khác cấu hình chính (A, G) có d_X = 2: phải sửa đề cương (mục 6).** SPA p = 0,09 (chưa bác bỏ ở 5%). LR theo biến sau Holm: G và F có ý nghĩa, A, B, O không. γ_G không đổi dấu qua các lần ước lượng lại |
| Chọn phân phối (XT) | skewed-t có LS cao nhất; DM không phân biệt được skewed-t với Student-t (p = 0,33); chuẩn bị bác bỏ (p = 0,02) |
| Ngoài mẫu KT | 1% VaR: skewed-t 7 vi phạm (0,74%), Kupiec p = 0,41, toàn bộ 741 cửa sổ 250 phiên ở vùng xanh. Phân phối chuẩn thua rõ (13 vi phạm, phân vị động p ≈ 0, vùng vàng 201 cửa sổ). **CH1 chỉ được ủng hộ yếu ngoài mẫu:** LS tốt hơn EGARCH (DM p = 0,07), nhưng FZ0 không tốt hơn (p = 0,37) và QLIKE của EGARCH thấp hơn (1,480 so với 1,510) |
| Mô phỏng khôi phục (E1–E4, B = 100) | 100/100 lần thành công; độ phủ Wald 0,87–0,99; E2 bác bỏ sai θ = 0 ở mức 0,05; E4 bác bỏ sai γ = 0 ở mức 0,06. Đạt điều kiện chuyển sang Giai đoạn 2 |
| Ký quỹ (7) | Xem 3.4 |

Khác biệt cần ghi vào báo cáo: gradient tính số (không dùng autodiff); 20 điểm khởi tạo cho ước lượng chính, 8 điểm kèm khởi tạo ấm cho các lần ước lượng lại cuốn chiếu; biên độ giá, bước làm tròn g = 0,5% và giới hạn giảm D = 1%/phiên chưa xác minh theo thông lệ.

### 3.4. Ký quỹ động (mục 7), mô hình skewed-t với biến G

Thiết kế: H và α là tham số thiết kế nên được báo cáo trên lưới H ∈ {1, 2, 3} × α ∈ {5%, 2,5%, 1%}; với mỗi cặp, (w, b, φ, η) được chọn trên XT bằng chi phí kỳ vọng với c_th/c_v = 1/α rồi cố định trên KT. Trình bày chính: H = 2, α = 1% (lựa chọn của người thực hiện, chưa được người dùng chốt).

- **H = 2, α = 1%:** KT có 894 cửa sổ hợp lệ, 3 vi phạm (0,34%), ký quỹ trung bình 6,1%, tối đa 15,5%. H = 1, α = 1%: 0,96% vi phạm, ký quỹ trung bình 4,1%. Mức 13% không có vi phạm nào nhưng mức này chưa xác minh.
- **Quy tắc năm bước:** trên XT, mọi cặp chọn w = 0, b = 0, η = 0; chỉ sàn φ (0,5–1,0) có tác dụng. Dữ liệu hiện tại không ủng hộ ES, hệ số đệm hay làm trơn bất đối xứng.
- **CH3 (H = 2, 1%), so tại cùng ký quỹ trung bình:** thiếu hụt mỗi phiên thấp hơn ký quỹ cố định (DM p = 0,026), nhưng **không phân biệt được** với VaR thô EGARCH-X (p = 0,16), mô phỏng lịch sử có lọc (p = 0,18), EWMA (p = 0,07), lịch sử 250 phiên (p = 0,09), và **không tốt hơn EGARCH không có X (p = 0,77)**. Giả thuyết 3 chưa bác bỏ được so với các đối chứng động.
- **Giới hạn:** chỉ 3–9 vi phạm mỗi cặp nên lực kiểm định thấp; các cửa sổ H ≥ 2 chồng lấn. Trong 47 phiên căng thẳng của KT, H = 1 và α = 1% vi phạm 10,6% (cũng bằng ký quỹ cố định). Đèn giao thông chỉ có nghĩa với H = 1 và 99%: các cặp α = 5% ở vùng đỏ là điều dự kiến.

## 4. Điểm cần ghi chú và cảnh báo

### 4.1. Phạm vi xử lý lỗi đáo hạn/chuyển hợp đồng (đánh giá 20/10/2026)

**Đã sửa, chỉ trong nhánh MLE mở rộng** (`scripts/uoc_luong_mle_*.py`, kết quả ở `results/phan_phoi_moi/`):
- Nối lợi suất và lợi suất qua đêm của phiên đầu sau đáo hạn bằng giá đóng cửa F2M hôm trước (105 lần nối, đủ giá F2M). Ví dụ 22/05/2020: lợi suất qua đêm −9,08% → +0,38%; RV 84,40 → 2,09.
- Bỏ ngày đáo hạn khỏi hàm hợp lý; phiên đầu sau đáo hạn chỉ khởi tạo lại đường phương sai (h0), không tính điểm.
- Phương sai khởi tạo và chuẩn hóa biến ngoại sinh chỉ dùng các ngày huấn luyện được giữ lại.

**Chưa sửa:**
- Thư viện `garch_vn30` không có xử lý đáo hạn nào (không có từ khóa dao_han/expiry/roll/F2M). Sáu mô hình, Bước 2 (mô phỏng) và Bước 3 (NUTS) vẫn chạy trên chuỗi chưa nối hợp đồng. Số liệu 3.1 vì thế chưa chịu lỗi đáo hạn được loại.
- VB không có cho chuỗi đã xử lý đáo hạn. Chỉ có MLE.
- Đây là hàm hợp lý có điều kiện theo đoạn, không phải hàm hợp lý đầy đủ của EGARCH có quan sát khuyết (đề cương mục 3.4 mô tả cách khác: giữ đệ quy liên tục, thay cú sốc bằng kỳ vọng 0).
- Mô hình không dự báo rủi ro ngày đáo hạn và phiên đầu sau đó. Điều này mâu thuẫn với Giai đoạn 2, vì ký quỹ phải được ấn định cho mọi phiên.
- Hệ số động học đổi mạnh khi loại cửa sổ: alpha 0,0551 → 0,0058 (GED: −0,0048); beta khoảng 0,68 → 0,58; gamma_rv 0,289 → 0,407. Báo cáo ghi nhận sự nhạy cảm này nhưng chưa tìm nguyên nhân. Giả thuyết chưa kiểm tra: cắt và khởi tạo lại đường phương sai mỗi ~21 phiên làm yếu cấu trúc động.
- Chuyển hợp đồng sớm (trước đáo hạn) chỉ thử được trên 2024–2026 (F2M 5 phút), chưa có kiểm định ý nghĩa.
- Tệp nến F2M 5 phút gốc không nằm trong repo.
- Chưa xác minh ai là tác giả phần mở rộng ngoài việc nó nằm ở commit 16/10/2026; kiểm tra bằng `git log --format='%an'` nếu cần ghi công.

### 4.2. Lệch giữa đề cương và code
- Chia mẫu: đề cương dùng ba khối (UL ≈ 750, XT ≈ 420, KT ≈ 1.000); code dùng hai khối 1.735/435. Với 435 phiên, lực kiểm định Kupiec chỉ ≈ 0,37.
- Biến ngoại sinh: đề cương dùng A_t (khối lượng bất thường đã khử chu kỳ đáo hạn) và G_t; code dùng log(1+volume) và log(RV+1e-6).
- Đại lượng đại diện phương sai: đề cương dùng Garman–Klass + qua đêm có hiệu chỉnh thang đo; code mở rộng dùng RV từ nến 1 phút.
- Quy trình chọn biến (đề cương mục 6: chọn tiến, SPA, Holm) chưa có code.

### 4.3. Thông số thể chế cần xác minh trước khi dùng
Chưa xác minh từ nguồn chính thức: mức 13% của VSD, biên độ giá ±7%, hệ số nhân 100.000 đồng/điểm, bước làm tròn tỷ lệ ký quỹ, lịch đáo hạn qua các thời kỳ. Dữ liệu không có chuỗi ký quỹ lịch sử.

## 5. Việc còn mở (theo thứ tự đề xuất)

Đã xong ngày 20/10/2026 (gói `de_cuong/`): ký quỹ động, walk-forward, Diebold–Mariano, chọn biến, mô phỏng khôi phục, MLE theo đề cương. Còn lại:

1. **Chốt với người dùng/GVHD:** cặp thiết kế (H, α) trình bày chính; chấp nhận d_X = 1 (chỉ G) thay cho (A, G) và sửa đề cương mục 2.6, 6; xác minh mức 13% VSD, biên ±7%, bước làm tròn g, giới hạn giảm D.
2. **Báo cáo:** sinh bảng và hình từ `results/de_cuong/` (chưa có generator, chưa có hình); cập nhật `tai_lieu_ma_nguon.tex` cho gói `de_cuong` và thêm vào `chay.bat`/thư mục nộp GVHD.
3. **Kiểm tra độ bền ký quỹ còn thiếu:** đổi w, η; so ký quỹ MLE với hậu nghiệm (bỏ vì chỉ dùng MLE); bootstrap tham số khi nghiệm sát biên; kiểm thử đơn vị cho `de_cuong` (chưa có).
4. **Vấn đề kết quả cần suy nghĩ:** CH1 yếu ngoài mẫu và quy tắc năm bước thu về VaR + sàn. Cân nhắc có nên dùng ES, mở rộng lưới w/b/η, hay nêu đây là kết quả âm có kiểm soát.
5. **Cập nhật hai bản thảo** `Bài báo chính.tex`, `document.tex` (còn mô tả MS-GARCH-X). Chỉ làm khi được yêu cầu.
7. Cập nhật thư mục nộp GVHD `D:\Hoai Nam Ng\Project\GARCH-VN30F1M_Nop_GVHD\` khi code thay đổi.

## 6. Lưu ý vận hành

- Python có torch/pyro: `D:\App\miniconda\envs\quant311\python.exe`. XeLaTeX (MiKTeX) biên dịch 2 lần, dọn `.aux/.log/.out/.toc`.
- Viết tiếng Việt học thuật; trong khung chat không dùng ký hiệu LaTeX; dùng "trường hợp" thay "kịch bản".
- Mọi con số phải truy được về `results/`; tách rõ "bằng chứng cho thấy gì" và "chưa thể kết luận gì".
- Sửa code thì cập nhật docstring và `code thuật toán/tai_lieu/tai_lieu_ma_nguon.tex`.
- `tong_hop_ket_qua.tex` có chỉnh sửa tay đã đưa vào generator; đối chiếu diff trước khi sửa generator.
- Có tiến trình Python của dự án khác trên máy (qp-tracking, distributional_bands); không dừng tiến trình khi chưa liệt kê dòng lệnh.
