# CLAUDE.md — Dự án GARCH-VN30F1M-MLE-VB-MCMC

Ghi chú vận hành cho Claude khi làm việc trong repo này.

## Trạng thái hiện hành (20/10/2026) — cấu trúc mới, bám đề cương

- Người dùng yêu cầu **bám đề cương bắt buộc** (`tai_lieu/de_cuong/`), đủ bốn phương pháp (MLE, VB-MF, VB-FR,
  NUTS) ở mọi mốc ước lượng lại; từ 22/10/2026 bám đề cương đã đăng ký BM02: khối UL/XT/KT = 750/925/435 (80/20),
  GED tham gia chọn phân phối, thêm GARCH(1,1) nền (xem mục 0000 của `GHI_CHU_DU_AN.md`); chạy bằng `chay.bat`; máy này chỉ chạy hồ sơ
  `nhanh`, bản đầy đủ chạy trên máy mạnh hơn.
- Cấu trúc: `cau_hinh/` (một tệp tham số), `ma_nguon/de_cuong/` (thư viện, JAX/NumPyro), `quy_trinh/` (b00–b14,
  điểm vào `chay.py`), `ket_qua/` (theo bước, có `_ho_so/`, `_diem_kiem_tra/`), `kiem_thu/`, `luu_tru/` (chỉ
  đọc: toàn bộ mã và bản thảo cũ, giữ nguyên để vẫn chạy được). Thiết kế: `THIET_KE_CODE.md`.
- Việc đang làm dở và tiến độ: mục 0 của `GHI_CHU_DU_AN.md`. Đọc trước khi làm tiếp.
- Python của dự án: `D:\App\miniconda\envs\quant311\python.exe` (đã ghi vào `cau_hinh\python.txt`); thêm `-X utf8`.
- Sửa `khung.py` không làm các bước chạy lại; sửa mô-đun thư viện nào thì bước khai báo nó trong `thu_vien` chạy lại.
- Các mục bên dưới mô tả thiết kế sáu mô hình cũ, nay nằm trong `luu_tru/code thuật toán/` (đường dẫn tương đối
  bên dưới tính từ thư mục đó).

## Trạng thái cũ (26/09/2026) — thiết kế sáu mô hình, nay ở luu_tru/

**MS-GARCH-X đã bị gỡ khỏi dự án** (người dùng chốt 25/09/2026: "mô hình chính bây
giờ là EGARCH-X"). Toàn bộ code, kết quả và báo cáo liên quan tới mô hình chuyển
trạng thái (MS-GARCH-X, MS-GARCH chuẩn, PR-A/S1–S8, MCMC hai trạng thái, `msgarchx/`)
được lưu ở **nhánh `luu-tru/ms-garch-x` và thẻ `luu-tru-ms-garch-x`** (đã đẩy lên
GitHub). Khôi phục một tệp: `git checkout luu-tru-ms-garch-x -- <đường dẫn>`.
Đừng đưa MS-GARCH-X trở lại nếu người dùng không yêu cầu.

Thư viện cũ `msgarchx` đã được **viết lại thành `garch_vn30`** (tên tệp tiếng Việt).
Trước khi thay thế đã đối chiếu song song cũ/mới trên cùng dữ liệu và seed: dữ liệu,
log-hợp lý, chỉ tiêu ngoài mẫu, nghiệm MLE, sai số chuẩn, mẫu VB, mẫu MCMC, mô phỏng
đều trùng (lệch lớn nhất 4,4e-16). Kết quả cũ được chuyển sang dạng dict thuần
(không pickle lớp Python) và gộp lại cho kết quả trùng tuyệt đối.

Remote GitHub đã được người dùng đổi tên thành `namngyh/NCKHSV_2026-27`; remote cục
bộ vẫn trỏ URL cũ (GitHub tự chuyển hướng). Đổi remote cần người dùng cho phép.

## Thiết kế ba bước — cả ba đã XONG

| Bước | Nội dung | Kết quả chính |
|---|---|---|
| 1 | 6 mô hình một trạng thái × 3 phương pháp (MLE, VB-MF, VB-FR) trên VN30F1M | EGARCH-X Student-t + MLE thấp nhất cả AIC 5606,87 · BIC 5650,54 · QLIKE 0,9234 · NLS 1,5203 |
| 2 | Mô phỏng khôi phục tham số EGARCH-X Student-t, E1/E2/E3, B=100, T=2000 | E1/E3: 100/100 lần θ̂ âm; E2 (θ₀=0): không bịa ra đòn bẩy; coverage MLE (Wald) 0,89–0,98, 100/100 Hessian hợp lệ |
| 3 | NUTS trên EGARCH-X Student-t | R̂ 1,0125, ESS 1.340/1.600, 0 phân kỳ, 8,1 giờ; KTC 95% của θ loại trừ 0; R_sd VB-MF của ω, β ≈ 0,37 và 0,29 |

Danh mục (`garch_vn30/mo_hinh/danh_muc.py`, nguồn sự thật duy nhất): garch_n (d=4),
garch_t (5), garch_x_t (7), egarch_n (5), egarch_t (6), egarch_x_t (8).

## Cấu trúc code (`code thuật toán/`)

- `garch_vn30/` — thư viện: `du_lieu/`, `kiem_dinh/`, `mo_hinh/`, `suy_dien/`,
  `danh_gia/`, `mo_phong/`. Thư viện không đọc/ghi tệp kết quả.
- `scripts/chay_buoc1_so_sanh_mo_hinh.py`, `chay_buoc2_mo_phong.py`,
  `chay_buoc3_mcmc.py` (đều có `--nhanh`, ghi vào thư mục `*_nhanh` riêng),
  `kiem_toan_ket_qua.py` (tính lại 380 con số, phải ra 0 HONG), `chung.py`.
- `scripts/bao_cao/` — `sinh_tong_hop_ket_qua.py` → `Bài báo/tong_hop_ket_qua.tex`,
  `sinh_hinh_tong_hop.py`, `ve_vector_tham_so.py` (đều đọc thẳng `results/`).
- `tests/` — 43 kiểm thử. `chay.bat` — bảng chọn trên Windows.
- `tai_lieu/tai_lieu_ma_nguon.tex/.pdf` — tài liệu mã nguồn cho GVHD (cách chạy,
  thư viện, công thức, từng tệp/hàm). Sửa code thì cập nhật tài liệu này.

Định dạng `results/`: `00_data.pt`, `01_preliminary_tests.pt`,
`pipeline/<mô hình>_<mle|vb_mean_field|vb_full_rank>.pt` (MLE: `params` là dict
tensor, `runs` là list dict), `pipeline/mcmc_egarch_x_t.pt`,
`mo_phong_egarch/parts/<E>_<pp>_<bbb>.pt` (`uoc_luong`, `can_duoi`, `can_tren`),
`mo_phong_egarch/<E>.pt` (`bias`, `rmse`, `coverage`, `n_co_khoang`, ...).
Ba mô hình GARCH MLE được tính với 6 điểm khởi tạo (lần chạy cũ), EGARCH với 10.

## Môi trường

- Python có torch/pyro: `D:\App\miniconda\envs\quant311\python.exe` (không có trong
  PATH). arviz ở môi trường này là 0.23 — `mcmc.inference_data()` tự chọn cú pháp
  `from_dict` theo phiên bản (truyền sai dạng không báo lỗi mà làm R̂ thành NaN).
- XeLaTeX có sẵn (MiKTeX). Biên dịch 2 lần; dọn `.aux/.log/.out/.toc` sau khi xong.
- In tiếng Việt ra console: `PYTHONIOENCODING=utf-8`.
- Chỉ dừng tiến trình theo PID sau khi liệt kê dòng lệnh; máy có tiến trình Python
  của dự án khác (qp-tracking, distributional_bands) — không được giết nhầm.
- Hook GateGuard yêu cầu nêu "facts" trước lần ghi đầu tiên mỗi tệp và trước lệnh xóa.

## Sở thích & phản hồi của người dùng

- Viết tiếng Việt, văn phong học thuật. Trong khung chat KHÔNG dùng ký hiệu LaTeX
  (hiển thị lỗi) — dùng chữ thường.
- Dùng "trường hợp", không dùng "kịch bản". Sau dấu hai chấm viết hoa.
- Phân biệt rõ: bằng chứng cho thấy gì / chưa thể kết luận gì. Không bịa số liệu;
  mọi con số phải truy được về `results/`.
- Code đơn giản, dễ đọc để người dùng tự kiểm tra. GVHD yêu cầu mỗi tệp/hàm có
  giải thích lý thuyết, thư viện, tác dụng: docstring tiếng Việt CÓ DẤU theo mẫu
  "LÝ THUYẾT / THƯ VIỆN / Tham số / Trả về" (chuỗi in ra console vẫn không dấu).
  Sửa code thì sửa docstring và `tai_lieu/tai_lieu_ma_nguon.tex` cùng lúc.
  Thay đổi tối ưu hóa phải giữ bản gốc làm đối chứng và có test so khớp.
- Thư mục nộp GVHD: `D:\Hoai Nam Ng\Project\GARCH-VN30F1M_Nop_GVHD\` (bản sao code,
  results, .bat, tài liệu PDF, `ĐỌC TRƯỚC.md`; ngoài git). Cập nhật lại bằng cách
  chép lại khi code thay đổi.
- `tong_hop_ket_qua.tex` có chỉnh sửa tay của người dùng đã được đưa vào generator;
  khi sửa generator, đối chiếu diff để không làm mất chỉnh sửa đó.
- Không động vào `Bài báo chính.tex` / `document.tex` cho tới khi người dùng yêu cầu
  (hai bản thảo này vẫn còn mô tả MS-GARCH-X).
- Khi phát hiện bất nhất về phạm vi nghiên cứu, hỏi trước khi tự quyết.
- Người hướng dẫn: TS. Châu — muốn ứng dụng thực tế rõ hơn (hướng đã thống nhất:
  ký quỹ động dựa trên VaR, đối chiếu tỷ lệ của VSD (13% từ 18/7/2018, 17% từ 15/12/2022), Basel traffic-light).

## Việc còn mở

- **B1. Ứng dụng ký quỹ động** (đã có trong đường ống mới, b10-b12): VaR_t → tỷ lệ ký quỹ, so với tỷ lệ VSD (17% từ 15/12/2022),
  chấm Basel traffic-light cửa sổ 250 phiên.
- **B3. Walk-forward** cho EGARCH-X (tái ước lượng theo quý) để tăng lực kiểm định VaR
  (435 phiên kiểm định chỉ cho lực ~0,37).
- Kiểm định Diebold–Mariano giữa các mô hình (hàm đã có, chưa dùng trong bảng).
- Cập nhật hai bản thảo (`Bài báo chính.tex`, `document.tex`) theo khung EGARCH-X.
