# CLAUDE.md — Bối cảnh dự án MS-GARCH-X Student-t (VN30F1M)

Ghi chú vận hành cho Claude khi làm việc trong repo này. Đọc file này trước khi
đưa ra giả định về trạng thái dữ liệu/kết quả — nhiều thứ trong repo không như
vẻ ngoài của nó (xem phần "Bẫy đã gặp" bên dưới).

## Tổng quan đề tài

Mô hình MS-GARCH-X với sai số Student-t cho lợi suất VN30F1M, so sánh 3 phương
pháp suy diễn trên **cùng một hàm hợp lý**: MLE, Variational Bayes (mean-field
& full-rank), MCMC (NUTS). Hai phần nghiên cứu:
- Phần 1: thành phần cấu trúc nào (đuôi dày, biến ngoại sinh, chuyển trạng
  thái) cải thiện mô hình hóa biến động — so sánh B1→B2→B3→B4→M1 bằng MLE.
- Phần 2: phương pháp suy diễn nào đáng chi phí — cố định mô hình chính
  (MS-GARCH-X Student-t), đổi phương pháp (M1=MLE, M2=VB-MF, M3=VB-FR,
  M4=MCMC).

Tác giả: Nguyễn Hoài Nam (namngyh@gmail.com) và cộng sự. Người phản biện/hướng
dẫn: TS. Châu.

## Bản đồ file quan trọng

| File | Vai trò | Lưu ý |
|---|---|---|
| `Bài báo/document.tex` | Khung lý thuyết chính | **KHÔNG được git theo dõi** (xem cảnh báo bên dưới) |
| `Bài báo/bao_cao_ket_qua.tex` | Báo cáo kết quả thực nghiệm | **File SINH RA** — không sửa tay, sửa script sinh ra nó rồi chạy lại |
| `code thuật toán/scripts/bao_cao/report_data.json` | Nguồn số liệu cho báo cáo | Sinh thủ công/bán tự động từ `results/*.pt`; KHÔNG có script tự động tạo lại toàn bộ file này — nếu số liệu trông như `NaN`/`0`, kiểm tra xem file `.pt` gốc có tồn tại không trước khi kết luận "chưa chạy" |
| `code thuật toán/scripts/bao_cao/02_sinh_tex.py` | Sinh `bao_cao_ket_qua.tex` từ `report_data.json` + gọi `appendix_gen.py` | Chạy: `python "code thuật toán/scripts/bao_cao/02_sinh_tex.py"` |
| `code thuật toán/scripts/bao_cao/appendix_gen.py` | Sinh phần phụ lục (đọc trực tiếp từ `results/*.pt`) | |
| `code thuật toán/msgarchx/simulation/scenarios.py` | Định nghĩa 8 kịch bản mô phỏng S1–S8 | Giá trị tham số là MINH HỌA, không hiệu chỉnh theo VN30F1M thật (tự ghi trong docstring) |
| `code thuật toán/msgarchx/simulation/run_simulation.py` | Runner mô phỏng, hỗ trợ methods `mle`, `vb_mean_field`, `vb_full_rank`, `mcmc` | `vb_full_rank` có code nhưng **chưa từng được gọi** trong lần chạy 8 kịch bản đã có — không có trong `results/07_simulation_S*.pt` |
| `code thuật toán/results/*.pt` | Nguồn sự thật duy nhất cho mọi số liệu | Đọc bằng `torch.load(path, weights_only=False)`; cần `sys.path.insert(0, ".../code thuật toán")` trước khi load vì pickle tham chiếu class trong `msgarchx` |

## ĐỢT 21/09/2026 — KIỂM TOÁN, BỔ SUNG COVERAGE MLE + BRIER, TỔNG HỢP 8 KỊCH BẢN

Người dùng yêu cầu "chắc chắn" kết quả trước khi làm phần ứng dụng. Đã làm:

**1. Kiểm toán độc lập — `scripts/tien_ich/kiem_toan_ket_qua.py`** →
`results/kiem_toan_ket_qua.txt`. Mọi con số trong bảng cuối được tính LẠI từ
nguồn thô (checkpoint từng lần lặp, tham số đã lưu, dữ liệu gốc) bằng đường
viết lại (sắp nhãn, Bias/RMSE/Coverage, QLIKE/NLS/AIC/BIC, R̂), rồi đối chiếu
với file tổng hợp. Kết quả: **545 đạt, 0 hỏng thực sự** (2 "hỏng" là khối
S1/mcmc 3 lần lặp thăm dò không có trong file gộp — đã hạ thành cảnh báo).
Bước 1: 24/24 ô tái lập được loglik/AIC/BIC/QLIKE/NLS/h_hat/NLS-hỗn-hợp từ
tham số hoặc mẫu hậu nghiệm đã lưu (lệch 0). Bước 3: summary tái lập đến
1e-16; R̂ tách đôi viết lại ≤ 1,0053 (Pyro báo 1,0125 — cùng kết luận).
Cờ: `--spot N` tính lại Brier từ đầu (sinh lại dữ liệu + bộ lọc), `--grad N`
đo gradient tại nghiệm MLE. Chạy lại sau MỌI lần gộp.

**2. `chay_bo_sung.bat` đã xong 1.200/1.200 (08:50→09:53) và gộp.** Sau đó
chạy thêm S4–S8 (`--kich-ban S4 S5 S6 S7 S8`, 213 tác vụ, 10:05→10:33).
Bản gốc trước gộp: `results/07_simulation_truoc_bosung_20260921_0953/` và
`_1033/`. Khối đã sửa có khoá `brier_da_sap_nhan=True`, `n_se_ok`, `n_se_hong`,
`coverage_nguon`.

**3. Bốn phát hiện từ bổ sung + kiểm toán — PHẢI nêu khi báo cáo:**

| Phát hiện | Số liệu | Hệ quả |
|---|---|---|
| Brier đúng (sau sắp nhãn) thấp hơn hẳn bản sai | S1: MLE 0,397 / MF 0,495 / FR 0,469 (bản sai 0,530/0,561/0,679); S2: 0,661/0,498/0,489; S3: 0,399/0,476/0,443 | MLE thắng Brier ở S1, S3; VB thắng ở S2 (nhận dạng yếu). VB-FR không còn "tệ hơn ngẫu nhiên" |
| **Hessian MLE của MS-GARCH-X chỉ xác định dương ở ~50 % lần lặp** | S1 50/100, S2 49/100, S3 42/100; S4 15/29, S5 3/29, S6 8/30, S7 8/21, S8 20/29 | Coverage Wald của MLE là **coverage có điều kiện** trên tập con đó; bảng phải ghi n_se_ok. Bảng tổng hợp bỏ trống CP khi n_se_ok < 10 |
| Coverage MLE (PR-A, trên tập con) 0,74–0,80 trung bình; $p_{11}$ và $\beta_1$, $\nu$ thấp nhất (0,45–0,62) | VB-MF 0,32–0,50, VB-FR 0,46–0,57 | MLE vẫn tốt nhất ở coverage nhưng xa 0,95 — sống núi $\omega/(1-\beta)$ + khoảng Wald vượt biên ($p_{ss}>1$ ở 37–86 lần) |
| Coverage MLE PR-B (EGARCH-X) **0,89–0,98, đủ 100/100 Hessian** | E1 0,92–0,97; E2 0,93–0,97; E3 0,89–0,98 | Đúng dự đoán 0,93–0,95 trong A2; MLE trên EGARCH đạt chuẩn |

Thêm một điều **chưa kết luận**: gradient của $-\ell$ theo tham số raw tại
nghiệm MLE đã lưu của MS-GARCH-X lớn ở ~70 % lần lặp (trung vị 1e1–1e3, max
1e7), nhưng trị riêng lớn nhất của Hessian tới 1e8 nên gradient lớn có thể chỉ
là điều kiện xấu (bước Newton ~1e-6, thay đổi $\ell$ ~1e-4) chứ không phải
dừng sớm. Đang thử LBFGS tiếp 25 bước từ nghiệm đã lưu (script trong
scratchpad); nếu $\ell$ tăng đáng kể thì MLE mô phỏng (n_starts=3, Adam 400 +
LBFGS 200) chưa hội tụ hẳn và phải ghi rõ. Không ảnh hưởng PR-B.

**4. `Bài báo/tong_hop_ket_qua.tex`** — sinh bởi
`scripts/bao_cao/sinh_tong_hop_ket_qua.py`: CHỈ BẢNG, 48 bảng: dữ liệu; Bước 1
(trong mẫu, ngoài mẫu, VaR Kupiec/Christoffersen); PR-A **8 kịch bản** ($\theta_0$
+ 3 bảng/kịch bản, Brier trong caption, CP MLE kèm n_se_ok); PR-B 3 kịch bản;
Bước 3 (hậu nghiệm + $R_{sd}$); bảng đếm thắng RMSE. S4–S8 gộp lại từ
checkpoint với CẢ HAI bản vá (sắp nhãn + $p_{ss}$ từ P0), $B$ = 29/15 như bản
gốc — bản cũ trong `07_simulation_thamdo/` có cả hai lỗi, KHÔNG dùng.
Đã kiểm tra tĩnh: cân bằng môi trường, đúng số cột, không ký tự chưa escape.
**Máy này không có XeLaTeX** — PDF phải biên dịch nơi khác (`xelatex` 2 lần).

**5. Sửa nhỏ:** `Bài báo chính.tex` eq:brier-state đổi $\xi_{t\mid T}$ →
$\xi_{t\mid t}$ (code luôn dùng xác suất lọc); `bo_sung_coverage_brier.py` nay
áp `_apply_misspecification` cho S8 (trước bỏ sót — S1–S3 không ảnh hưởng),
nhận `--kich-ban`, đọc số lần lặp từ checkpoint có sẵn; `ket_qua_moi.tex` có
cột MLE (Wald) trong tab:pra-coverage; ghi chú Brier trong `tom_tat_mo_phong.py`
cập nhật (Phần 1 của file txt đó vẫn chưa in Brier/coverage MLE — dùng
`tong_hop_ket_qua.tex` làm nguồn chính).

**6. Dọn thư mục (người dùng duyệt "ok làm đi").** Đã xoá: `results_dryrun/`,
`results/pipeline_nhanh/`, `results/mo_phong_egarch_nhanh/` (chạy thử), hai
file `.bak` trong `Bài báo/`, `results/07_simulation_thamdo/` (8 file cũ có
cả hai lỗi gộp — tái tạo được từ `parts/sim/`). Đã gom vào
`results/lich_su_sua_loi/`: `07_simulation_truoc_sua_p11/`,
`07_simulation_truoc_bosung_*` (2), và **hai file lần chạy hỏng**
`04_mcmc_lan1_duongcheo.pt`, `06_oos_lan1_M4_hong.pt` (bằng chứng MCMC lần 1
$\hat R\approx4$ mà phụ lục trích — KHÔNG xoá; 4 script đọc chúng đã trỏ lại
đường mới). `logs/` bỏ theo dõi git (`git rm --cached`, thêm vào .gitignore),
giữ trên đĩa. Giữ nguyên: `report.html`, `regime_signal_vn30f1m.csv` (đầu ra
có script sinh), ba thư mục `hinh*` (mỗi thư mục do một `.tex` khác tham chiếu).
`bo_sung_coverage_brier.py --combine` nay sao lưu vào `lich_su_sua_loi/`.

**Quyết định chốt trong đợt:** bảng 8 kịch bản dùng mọi lần lặp có sẵn ($B$
ghi trong caption, S1–S3 = 100, S4–S8 = 29/15) thay vì cắt S1–S3 xuống $B$
thấp — tránh hai bộ số khác nhau cho cùng S1 trong một tài liệu. Không đụng
skew-$t$ (người dùng chốt chỉ làm 1–5).

---

## TIẾN ĐỘ VÀ VIỆC CẦN LÀM (cập nhật 15/09/2026)

### Toàn bộ thiết kế ba bước ĐÃ HOÀN TẤT — không còn tính toán nào đang chạy

**Bước 1 xong 13/09 19:09**, đủ 24/24 ô (8 mô hình × 3 phương pháp).
**EGARCH-X Student-t + MLE thắng cả bốn tiêu chí**: AIC 5606,87 · BIC 5650,54 ·
QLIKE 0,9234 · NLS 1,5203. Bảng đầy đủ và phân rã đóng góp từng thành phần
nằm trong `Bài báo/ket_qua_moi.tex` (sinh tự động).

Tín hiệu từ lần chạy thử đã được bản đầy đủ **xác nhận nguyên vẹn**: các dòng
EGARCH ở bản `--nhanh` chỉ là cận dưới, và bản đầy đủ cho đúng cùng con số
AIC 5606,87 — tức MLE đã tìm được cực đại ngay cả với cấu hình bị cắt xén.

Ba phát hiện chính của Bước 1 (chi tiết trong `Bài báo/CAN_SUA_SAU_BUOC_1.md`):
đuôi dày quan trọng nhất ($\Delta$AIC ≈ −235); biến ngoại sinh hữu ích gấp bốn
lần trong EGARCH vì không vướng ràng buộc dương; **chuyển trạng thái làm mô hình
tệ đi** ($\Delta$AIC = +1,7) — tức mô hình được đề xuất trong tên đề tài đã thua.

**Bước 3 xong 15/09**, hội tụ — xem mục riêng ngay dưới.

### Trạng thái tổng thể của ba bước

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Danh mục 8 mô hình × 3 phương pháp, dữ liệu thật | **XONG** 24/24 ô — EGARCH-X Student-t thắng cả bốn tiêu chí |
| 2 | Mô phỏng khôi phục tham số (PR-A + PR-B) | **XONG** 1.800/1.800 lần lặp |
| 3 | MCMC trên mô hình thắng cuộc | **XONG, HỘI TỤ** — $\hat R$ 1,0125, ESS 1.340, 0 phân kỳ |

---

## BƯỚC 3 — MCMC TRÊN MÔ HÌNH THẮNG CUỘC: HỘI TỤ (cập nhật 15/09/2026)

**Cả ba bước của thiết kế đã hoàn tất.** Bước 3 chạy 14/09 19:37 → 15/09 ~03:45,
**8,1 giờ**, trên EGARCH-X Student-t bằng NUTS, 4 chuỗi × (1.000 warmup + 400
mẫu), độ sâu cây tối đa 7, khởi tạo phân tán quanh nghiệm MLE.

| Chẩn đoán | Lần này (EGARCH-X-t) | Lần MS-GARCH-X thất bại |
|---|---|---|
| $\hat R$ lớn nhất | **1,0125** | 2,84 |
| ESS nhỏ nhất | **1.340** / 1.600 mẫu (84%) | ≈ 5 |
| Số phân kỳ | **0** | — |
| Thời gian | 8,1 giờ, 4 chuỗi **tuần tự** | 64 giờ, 4 chuỗi song song |

Đây là bằng chứng dứt khoát cho luận điểm của thiết kế ba bước: **MCMC thất bại
trước đây là do đối xứng hoán vị nhãn của cấu trúc chuyển trạng thái**, không
phải do MCMC. Cùng thư viện, cùng thuật toán, cùng dữ liệu — bỏ cấu trúc chuyển
trạng thái thì hội tụ.

### Kết quả trên VN30F1M, bốn phương pháp cùng một likelihood

| Tham số | MLE | VB-MF | VB-FR | MCMC | KTC 95% (MCMC) |
|---|---|---|---|---|---|
| $\theta$ | −0,1452 | −0,1407 | −0,1241 | **−0,1399** | **[−0,2125; −0,0710]** |
| $\beta$ | 0,6832 | 0,6264 | 0,6390 | 0,6393 | [0,4912; 0,7562] |
| $\nu$ | 4,2910 | 4,3897 | 4,3647 | 4,3644 | [3,5544; 5,4315] |
| $\omega$ | 0,1599 | 0,1941 | 0,1897 | 0,1870 | [0,1093; 0,2948] |
| $\alpha$ | 0,0721 | 0,1081 | 0,1066 | 0,1121 | [0,0268; 0,2110] |

Hai phát hiện dùng thẳng trong bài:
- **Hiệu ứng đòn bẩy của VN30F1M được xác lập thống kê**: khoảng tin cậy 95% của
  $\theta$ **loại trừ 0**. Kết hợp với PR-B (E2: mô hình không bịa ra đòn bẩy khi
  $\theta_0=0$), kết luận này được kiểm chứng cả hai chiều.
- **$\nu \approx 4{,}36$** với KTC [3,55; 5,43] — đuôi rất dày, khớp $\hat\nu\approx4{,}08$
  của MS-GARCH-X trước đây. E3 ($\nu_0=4$) đã xác nhận bộ ước lượng đáng tin ở đúng vùng này.

### RQ3 trả lời được lần đầu tiên: $R_{sd} = \mathrm{sd}_{VB}/\mathrm{sd}_{MCMC}$

| | $\mu$ | $\omega$ | $\alpha$ | $\theta$ | $\beta$ | $\nu$ |
|---|---|---|---|---|---|---|
| VB-MF | 0,95 | **0,37** | 0,88 | 0,92 | **0,29** | 0,63 |
| VB-FR | 0,94 | **0,75** | 1,01 | 1,00 | **0,75** | 0,97 |

Mean-field **đánh giá thấp bất định của $\omega$ và $\beta$ khoảng ba lần** —
đúng hai tham số nằm trên sống núi cong $\omega/(1-\beta)$ mà mô phỏng đã chỉ ra
(coverage của $\omega$, $\beta$ tệ nhất ở cả PR-A lẫn PR-B). Full-rank thu hẹp
nhưng vẫn hẹp hơn 25% ở đúng hai tham số đó. Ở $\mu$, $\theta$, $\alpha$ thì cả
hai họ VB gần khớp MCMC. **Kết quả mô phỏng và kết quả dữ liệu thật nói cùng một
điều, bằng hai con đường độc lập.**

### Hai sự cố vận hành ngày 13–14/09, đã ghi để không lặp lại

1. **`--chi-bang` từng chảy tiếp xuống Bước 3.** Lệnh "chỉ in bảng" không kèm
   `--bo-qua-mcmc` đã âm thầm khởi động MCMC 8 tiếng, chạy nền song song với
   một lần chạy khác, **hai tiến trình ghi chồng cùng file tiến độ** → mọi số
   đọc từ file đó hôm 13/09 không đáng tin và dẫn tới một loạt chẩn đoán sai.
   Đã vá: `--chi-bang` nay dừng sau khi in bảng.
2. **Không truyền `initial_params`** → Pyro dùng `init_to_uniform`, một chuỗi
   rơi vào $\theta=+1{,}03$ (ngược dấu đòn bẩy), bước tích phân sụp xuống
   $10^{-9}$. Đã thêm `initial_params_single_from_mle()` trong `mcmc.py`:
   chuỗi 0 đúng tại nghiệm MLE, ba chuỗi còn lại phân tán **2 sai số chuẩn**
   (từ `standard_errors_single`), có trần 1,0 trên thang đã biến đổi vì phương
   pháp delta phóng đại khi tham số gần biên. Hằng số nhiễu loạn cố định là
   sai: 0,3 trên $\theta$ có sai số chuẩn 0,035 là 15 độ lệch chuẩn.
3. **`taskkill /F /IM python.exe` giết nhầm backend của dự án khác.** Từ nay chỉ
   giết theo PID sau khi liệt kê dòng lệnh. `TaskStop` của harness **không giết
   tiến trình con** — phải kiểm tra lại bằng `Get-CimInstance Win32_Process`.

### Công cụ

- `chay_buoc3_mcmc.bat`: chạy Bước 3 và hiện thanh tqdm trực tiếp trên cmd,
  kèm ba chỉ số cần theo dõi (`s/it`, `step size`, `acc. prob`).
- `sinh_ket_qua_moi.py` nay sinh **6 bảng**: thêm bảng bốn phương pháp và bảng
  $R_{sd}$ cho `Bài báo/ket_qua_moi.tex`.


## VIỆC CẦN LÀM, xếp theo mức độ chặn

### Nhóm A — chặn: thiếu thì bài không hoàn chỉnh

**A1. Đồng bộ bản thảo với kết quả thực tế — RỦI RO LỚN NHẤT HIỆN NAY.**

Đây không phải việc tính toán mà là việc *bài đang mô tả một nghiên cứu khác
với nghiên cứu đã chạy*:

| File | Đang mô tả | Thực tế |
|---|---|---|
| `Bài báo chính.tex` | 8 kịch bản, B=30/15 | 3 kịch bản, B=100 |
| `README.md` §11.2 | "4 kịch bản hiệu chuẩn theo VN30F1M" | 8 kịch bản tham số minh hoạ |
| `bao_cao_ket_qua.tex` | sinh từ `report_data.json` | JSON **không tự đồng bộ** với `results/*.pt` |
| nhiều chỗ | MS-GARCH-X là "mô hình chính" | có thể EGARCH-X-t mới thắng |

`report_data.json` là điểm nguy hiểm nhất: nó được duy trì **thủ công**, không
có script tái tạo. Từ lúc viết đến nay đã có 1.800 lần lặp mới, một lỗi sắp nhãn
được sửa, và ba mô hình EGARCH được thêm. **Mọi con số trong
`bao_cao_ket_qua.tex` hiện phải coi là CHƯA XÁC MINH.**

Việc cần làm: viết script trích xuất `report_data.json` **tự động** từ
`results/*.pt`, theo đúng tinh thần "một nguồn sự thật" mà `tom_tat_mo_phong.py`
đã làm. Ước tính ~1 ngày, loại bỏ vĩnh viễn cả một họ lỗi.

**A2. ~~Tính coverage cho MLE~~ ĐÃ XONG 21/09 (xem đợt 21/09 ở đầu file — chỉ ~50 % Hessian PD ở PR-A).** Ghi chú cũ: Bảng so sánh phương pháp hiện **thiên
vị VB** ở cột khoảng tin cậy chỉ vì MLE không được tính; RQ3 và RQ4 không trả
lời đầy đủ được nếu thiếu. Hàm cho mô hình một trạng thái
(`standard_errors_single` trong `garch_mle.py`) **đã viết và kiểm chứng sơ bộ**
(tỷ số SE/sd Monte Carlo 0,97–1,41 trên một lần lặp E1); PR-A thì
`natural_scale_standard_errors` có sẵn, chỉ chưa gọi. Ước tính gián tiếp từ tỷ
số |chệch|/sd (MLE ≤ 0,37 ở mọi tham số EGARCH) cho thấy MLE nhiều khả năng đạt
**0,93–0,95** và thắng luôn cột này. Cảnh báo: với PR-A, Hessian không hợp lệ khi
nghiệm sát biên — đúng tình huống $p_{ss}$.

**A3. ~~Sửa Brier của PR-A~~ ĐÃ XONG 21/09.** Ghi chú cũ: Hiện sai vì tính *trước* khi sắp nhãn
trạng thái (VB-FR 0,63–0,68, tệ hơn đoán ngẫu nhiên 0,5). Checkpoint chỉ lưu con
số vô hướng chứ không lưu $\xi_t$ nên không sửa được từ dữ liệu đã có; phải chạy
lại **bộ lọc tiến** một lượt với tham số đã lưu — **không cần ước lượng lại**.

**A4. ~~Bước 3 — MCMC~~ ĐÃ XONG 15/09, hội tụ không cần đổi tham số hoá.** Ghi chú cũ giữ lại để biết lý do từng hoãn: Nên **đổi tham số
hoá trước**: Bước 2 đã chỉ ra sống núi cong $\omega/(1-\beta)$ vừa làm VB chệch
vừa ép NUTS hạ bước tích phân xuống $10^{-4}$ — chạy MCMC trước khi sửa là lặp
lại thất bại 64 giờ cũ. Bài vẫn hoàn chỉnh nếu Bước 3 báo cáo kết quả âm.

### Nhóm B — nâng chất lượng đáng kể

**B1. Phần ứng dụng ký quỹ động — QUAN TRỌNG NHẤT VỀ MẶT PHẢN BIỆN, ~2 ngày.**
Hướng đã thống nhất với TS. Châu và là câu trả lời cho "ứng dụng thực tế thế
nào", nhưng **chưa có dòng code nào**. Cần: chuyển $\widehat{\VaR}_t$ thành tỷ lệ
ký quỹ, đối chiếu mức cố định 13% của VSD, chấm điểm bằng Basel traffic-light
trên cửa sổ trượt 250 phiên. Tận dụng được ngay kiểm định Kupiec đã có.

**B2. Tách "chệch do xấp xỉ" khỏi "chệch do tiên nghiệm" — ~3 giờ.** Phát hiện
mạnh nhất của Bước 2 là VB bị tiên nghiệm kéo ($p_{ss}$ về 0,810, sát trung
bình tiên nghiệm 0,800). Chạy lại VB trên một kịch bản mô phỏng với tiên nghiệm
Beta(20; 2): nếu chệch biến mất thì **chứng minh** được cơ chế thay vì chỉ suy
luận. Đây là loại bằng chứng biến một nhận xét thành một đóng góp.

**B3. Kiểm định trượt gốc (walk-forward) cho mô hình thắng cuộc — ~4,5 giờ.**
Cửa sổ khởi động $W=1200$, tái ước lượng theo quý (16 lần): 970 quan sát kiểm
định thay vì 435, **lực kiểm định VaR từ 0,37 lên 0,74**, gần 4 cửa sổ Basel.
Chỉ khả thi với mô hình một trạng thái (~17 phút/lần ước lượng); MS-GARCH-X sẽ
mất ~113 giờ. Phải khai báo rõ đây là thiết kế **có tái ước lượng**, không so
trực tiếp QLIKE với phần chính (tham số cố định).

### Nhóm C — nhỏ, gom một đợt (~1 ngày)

Từ chính danh sách README §16: **CRPS** đã cài nhưng chưa bao giờ được gọi;
**biểu đồ PIT** mới vẽ cho M1, cần cho từng phương pháp; **ESS phần đuôi** cho
MCMC; **kiểm định Diebold–Mariano** giữa các mô hình (có code, chưa rõ đã dùng).

### Nếu phải chọn ba việc: A1 → B1 → A2

A1 vì rủi ro lớn nhất hiện nay không phải thiếu kết quả mà là **báo cáo sai kết
quả đã có**. B1 vì đó là phần phản biện sẽ hỏi và hiện đang trống. A2 vì rẻ và
đang làm lệch kết luận trung tâm về so sánh phương pháp.

### Đã bỏ khỏi phạm vi (người dùng chốt 13/09)

- **Batch hoá / GPU** — đã đo khả thi (10–89× trên CPU) nhưng hai chỗ dùng chính
  là PR-A và PR-B đều đã chạy xong; Bước 1 chỉ 17 mục tuần tự nên không giúp gì.
- **MLE có phạt** (penalized MLE để trị đuôi chạy trốn của $\omega_2$) — vẫn là
  khuyến nghị tốt, nhưng ra ngoài phạm vi bài.

---

## Nhận xét về cách chia dữ liệu (phân tích 13/09)

Chia 80/20 theo thời gian: huấn luyện 1.735 quan sát (07/11/2017 → 16/10/2024),
kiểm định 435 (17/10/2024 → 17/07/2026). Tỷ lệ thực là **79,95/20,05** vì mặt nạ
đặt trên 2.171 phiên gốc rồi mất một quan sát do lấy sai phân.

**Hợp lý**: chia theo thời gian không xáo trộn; chuẩn hoá biến ngoại sinh dùng
đúng thống kê tập huấn luyện và đúng thứ tự (mặt nạ trước, lọc sau) nên không rò
rỉ; $T=1735$ đủ để $p_{11}$, $p_{22}$, $\nu$, $\alpha_1$ bão hoà (bằng chứng từ
S6/S1/S7: RMSE của $p_{22}$ là 0,099 ở $T=500$ nhưng 0,025 ở $T=2000$ và 0,023 ở
$T=5000$).

**Bất hợp lý**:
- **Lực kiểm định VaR chỉ 0,37** ở mức sai gấp đôi (435 phiên, vùng chấp nhận
  [1, 9] vi phạm). Đây là điểm yếu nghiêm trọng nhất vì VaR là trọng tâm ứng dụng.
- 435 phiên chỉ đủ **1,7 cửa sổ Basel 250 phiên**.
- Giả định tham số ổn định bị chính dữ liệu phản bác: chia tập huấn luyện làm
  bốn, tỷ lệ thời gian ở trạng thái biến động cao là **23,6% → 36,2% → 23,5% →
  4,1%**, rồi tập kiểm định quay lại 25,0%.
- Trạng thái hiếm vẫn đói dữ liệu: $\omega_2$, $\alpha_2$, $\beta_2$ **chưa bão
  hoà** ở $T=2000$ (RMSE của $\omega_2$: 1,152 ở $T=2000$ → 0,535 ở $T=5000$).

**Đặc điểm tập kiểm định cần khai báo**: phương sai **thấp hơn** (sd 1,367 so với
1,515 sau khi bỏ quan sát ngoại lai −11,66%) nhưng **đuôi dày hơn** (độ nhọn
10,24 so với 8,74) và **nhiều thời gian ở trạng thái biến động cao hơn** (25,0%
so với 21,8%). Đây đúng là cấu hình khiến mô hình trông đẹp ở QLIKE/NLS nhưng
hỏng ở độ phủ đuôi — trong khi kiểm định VaR lại thiếu lực để phát hiện.


## ~~Cảnh báo: `document.tex` không có trong git~~ — ĐÃ XỬ LÝ (09/2026)

File này nay **đã được track** cùng `Bài báo chính.tex`, `bao_cao_ket_qua.tex`
và toàn bộ `hinh*/`. Cảnh báo cũ không còn hiệu lực; giữ lại tiêu đề để ai đọc
tài liệu cũ biết vấn đề đã được giải quyết chứ không phải bị bỏ quên.

## TRẠNG THÁI BƯỚC 2 — MÔ PHỎNG KHÔI PHỤC THAM SỐ — ĐÃ XONG (cập nhật 13/09/2026)

Đây là mục mới nhất; nếu mâu thuẫn với mô tả cũ bên dưới thì **mục này đúng**.

**PR-A (MS-GARCH-X, S1/S2/S3, B=100, n=2000, 3 phương pháp): ĐÃ XONG.**
**900/900 lần lặp** (897 trong lượt chính, 3 lần lặp `b=28` chạy bù sau theo quy
tắc thử lại ở dưới), chạy 10/09 09:18 → 11/09 19:19 = **34,0 giờ** trên 12 tiến
trình. Kết quả đã gộp vào `results/07_simulation_S1..S3.pt`. Bản sao kết quả
**bước thăm dò cũ** (8 kịch bản, B=30/15, *có khối MCMC 3 lần lặp*) nằm ở
`results/07_simulation_thamdo/` — **đừng xóa**, vì `combine()` chỉ ghi ba khóa
`mle/vb_mean_field/vb_full_rank` nên nó đã ghi đè và làm mất khối `mcmc` trong
`07_simulation_S1.pt`.

**PR-B (EGARCH-X, E1/E2/E3, B=100, n=2000): ĐÃ XONG.** 900/900 lần lặp, chạy
11/09 19:19 → 13/09 01:33 = **30,2 giờ** trên 12 tiến trình. Không lỗi nào,
không file `.tmp` sót, và **không lần lặp nào phải dùng quy tắc thử lại điểm
khởi tạo** — khác hẳn PR-A. Kết quả ở `results/mo_phong_egarch/E1..E3.pt`.

### Kết quả Bước 2 — xếp hạng phương pháp

Trên **59 ô RMSE** của cả hai nghiên cứu: **MLE thắng 45, VB-MF thắng 14,
VB-FR thắng 0**. MLE cũng rẻ nhất (20–26 phút/lần lặp so với 23–40).

| | VB-MF | VB-FR | Ghi chú |
|---|---|---|---|
| PR-A S1 / S2 / S3 | 0,323 / 0,495 / 0,339 | 0,469 / 0,567 / 0,455 | **không dùng được** |
| PR-B E1 / E2 / E3 | 0,715 / 0,652 / 0,703 | 0,815 / 0,725 / 0,814 | dùng được có chọn lọc |

Coverage (lý tưởng 0,95) tăng rõ khi bỏ cấu trúc chuyển trạng thái. VB-FR hơn
VB-MF ở **cả sáu** cấu hình, không ngoại lệ. **MLE chưa có cột này** vì quy trình
không gọi hàm sai số chuẩn Hessian — đây là khoảng trống còn lại, xem dưới.

### Quy luật: VB chỉ thắng khi HAI điều kiện cùng đúng

**(a) tham số nhận dạng yếu, VÀ (b) tiên nghiệm đặt đúng chỗ.** Bằng chứng đối
chiếu nằm trong cùng bộ dữ liệu:

| Tham số | Nhận dạng | Tiên nghiệm | Kết quả |
|---|---|---|---|
| $\omega_2$ (PR-A) | yếu | LogNormal phủ 0,40 hợp lý | VB thắng **3,5–12×** |
| $p_{ss}$ (PR-A) | yếu | Beta(8;2) TB 0,80 vs thật 0,98 — **sai chỗ** | VB thua: VB-MF coverage 0,00–0,05, VB-FR 0,23–0,69 |
| $\theta$ (E2) | mạnh | Normal đúng tâm 0 | hoà RMSE, VB thắng coverage 0,96–0,98 |
| $\nu$ ở $\nu_0=8$ | yếu | Gamma hợp lý | VB-MF thắng 5–9% |
| $\nu$ ở $\nu_0=4$ | yếu hơn | Gamma | **MLE thắng** (0,435 vs 0,464) |

Hai hàng cuối quan trọng cho bài: lợi thế của VB về $\nu$ **biến mất ở vùng đuôi
dày**, tức đúng vùng của VN30F1M ($\hat\nu\approx4{,}08$). Không được ngoại suy
kết quả $\nu_0=8$ sang dữ liệu thật.

Kết luận: **giá trị của VB ở đây là giá trị của TIÊN NGHIỆM, không phải của
thuật toán biến phân.** Cùng một cơ chế vừa cứu $\omega_2$ vừa giết $p_{ss}$.

### Cơ chế MLE hỏng ở $\omega_2$ (chỗ duy nhất VB thắng áp đảo)

MLE có **đuôi chạy trốn**: 13–15% số lần lặp cho $\omega_2>2{,}0$ (thật 0,40),
cực đại 5,05 (S1) và 6,47 (S3); VB không bao giờ vượt 0,74. Truy được: 13 lần
lặp chạy trốn đó đều là những lần $\alpha_2+\beta_2$ sụp xuống 0,443 thay vì
0,940 — $\omega_2$ và độ dai dẳng đánh đổi nhau, MLE thỉnh thoảng rơi vào nhánh
xấu. Nhưng VB cũng sai kiểu khác: trung vị 0,206 so với thật 0,40 (co quá mức),
MLE 0,791 (nở quá mức). VB thắng RMSE vì tránh thảm hoạ đuôi, không phải vì
đúng hơn ở tâm.

**Khuyến nghị chưa có trong bài: MLE có phạt** (penalized MLE) — giữ độ sắc của
MLE, thêm đúng số hạng phạt log-chuẩn lên $\omega$ mà VB đang dùng. Rẻ hơn VB,
loại bỏ rủi ro chạy trốn. Trả lời thẳng câu "vận hành thế nào trong thực tế".

### PR-B xác nhận bộ ước lượng EGARCH theo CẢ HAI chiều

| | $\theta_0$ | Kết quả |
|---|---|---|
| E1 | −0,10 | **100/100** lần lặp cho $\hat\theta$ âm |
| E2 | 0 | $\hat\theta$ TB **−0,0006**, t = −0,36, **47% âm** — không bịa ra đòn bẩy |
| E3 | −0,10, $\nu_0=4$ | **100/100** âm, chệch −0,0031 |

Hai phân bố E1 và E2 **tách rời hoàn toàn** (E1 cực đại −0,050; E2 cực tiểu
−0,037): không một lần lặp nào trong 200 nằm vùng chồng lấn. Coverage của
$\theta$ dưới VB-FR: 0,95 / 0,98 / 0,96 — đạt chuẩn ở cả ba kịch bản.

$\nu$ ở E3 khôi phục gần như hoàn hảo: MLE cho TB **4,121** so với thật 4,0,
trung vị 4,04, chệch chỉ **+3%** (so với +7% ở $\nu_0=8$). Lo ngại "mô hình đánh
giá thấp độ dày đuôi" được hoá giải ở đúng vùng quan trọng nhất.

### Sống núi cong $\omega/(1-\beta)$ — chướng ngại chung của VB và MCMC

Mọi phương pháp khôi phục **chính xác** mức log-phương sai vô điều kiện
(0,789–0,817 so với 0,800 thật) nhưng VB **trượt dọc sống núi**: ở E2, VB-FR cho
$\omega$ sai gần 3 lần và $\beta=0{,}860$ thay vì 0,950 — nửa đời cú sốc 4,6
phiên thay vì 13,5. Full-rank không cứu được vì họ Gauss nắm tương quan **tuyến
tính**, còn sống núi này **cong** (hyperbol).

Đây chính là thứ README ghi nhận làm NUTS phải hạ bước tích phân xuống $10^{-4}$
"do tương quan mạnh giữa $\omega_s,\alpha_s,\beta_s$" — **một chướng ngại, hai
nạn nhân**. Cách chữa không phải đổi phương pháp mà là **đổi tham số hoá** sang
(mức, độ dai dẳng); khi đó sống núi thành trục toạ độ, mean-field cũng đủ và
MCMC sẽ mixing nhanh hơn nhiều. Đáng cân nhắc **trước** khi chạy Bước 3.

### Hai khoảng trống còn lại

1. **Coverage của MLE chưa tính** ở cả hai nghiên cứu. Hàm
   `natural_scale_standard_errors()` đã có cho MS-GARCH-X (chỉ chưa gọi);
   `garch_mle.py` **chưa có** hàm tương ứng cho mô hình một trạng thái. Ước tính
   gián tiếp từ tỷ số |chệch|/sd (MLE ≤ 0,37 ở mọi tham số EGARCH) cho thấy MLE
   sẽ đạt **0,93–0,95**, tức nhiều khả năng thắng luôn cột khoảng tin cậy. Chi
   phí: PR-B ~12 phút + ~30 dòng code; PR-A ~2 giờ nhưng vướng nghiệm sát biên.
2. **Brier của PR-A vẫn sai** (tính trước khi sắp nhãn, checkpoint không lưu
   $\xi_t$). Chạy lại bộ lọc tiến với tham số đã lưu là đủ, ~25 phút.


### BẪY LỚN NHẤT của đợt này: nhãn trạng thái không được sắp trước khi gộp

`run_simulation.py` tính Bias/RMSE/Coverage bằng cách so `omega1` ước lượng với
`omega1` thật, nhưng **không** sắp nhãn — trong khi mô hình chuyển trạng thái có
đối xứng hoán vị nhãn nên bộ ước lượng trả nhãn theo thứ tự tùy ý. Đo được:
**VB đảo nhãn 47–54% số lần lặp** (đúng bằng tung đồng xu), MLE 23–29%. Hậu quả
là hai trạng thái bị trộn vào nhau: VB-FR ở S1 cho cả $\omega_1$ lẫn $\omega_2$
xấp xỉ 0,198 trong khi giá trị thật là 0,05 và 0,40.

**Đã sửa**: thêm `sap_nhan_trang_thai()` trong
`msgarchx/simulation/run_simulation.py` (tiêu chí: phương sai vô điều kiện
$\omega_s/(1-\alpha_s-\beta_s)$, trạng thái 1 = biến động thấp), gọi ở **cả hai**
đường gộp, kèm 5 test. Sau khi sửa, VB-FR ở S1 cho $\omega_1=0{,}224$ và
$\omega_2=0{,}542$ — tách được hai trạng thái.

Ba dấu hiệu nhận ra lỗi này nếu nó tái xuất ở chỗ khác: (1) ước lượng hai trạng
thái gần trùng khít dù DGP đặt chúng xa nhau; (2) chỉ những tham số **giống nhau
ở cả hai trạng thái** mới có coverage lành lặn (ở đây là $\mu$ và
$\boldsymbol{\gamma}$, vì cả ba kịch bản đặt $\mu_1=\mu_2=0$ và
$\boldsymbol{\gamma}_1=\boldsymbol{\gamma}_2$); (3) Brier **tệ hơn đoán ngẫu
nhiên** (0,5).

### ~~Brier hiện vẫn SAI~~ — ĐÃ SỬA 21/09 bằng bo_sung_coverage_brier.py (ghi chú cũ giữ lại)

Brier được tính trong từng lần lặp từ $\xi_t$ **trước** khi sắp nhãn, mà
checkpoint chỉ lưu con số vô hướng chứ không lưu $\xi_t$. Giá trị hiện tại
0,51–0,68 (riêng VB-FR 0,63–0,68, tệ hơn ngẫu nhiên). Muốn đúng phải chạy lại
**bộ lọc tiến** một lượt với tham số đã lưu — **không cần ước lượng lại**,
khoảng 25 phút. Chưa làm.

### Coverage thấp là do CHỆCH, và chệch phần lớn do TIÊN NGHIỆM

Sắp nhãn xong coverage gần như không đổi (S1: 0,317 → 0,323 với VB-MF). Lý do:
nó đã thấp từ trước vì độ chệch quá lớn. Tiên nghiệm $p_{ss}\sim$ Beta(8; 2)
có trung bình **0,800**, trong khi giá trị thật của S1 là $p_{11}=0{,}98$. VB-MF
cho **0,810** — sát trung bình tiên nghiệm; VB-FR cho 0,906; MLE (không có tiên
nghiệm) cho **0,983** — gần như trùng giá trị thật. Coverage của $p_{11},p_{22}$:
VB-MF **0,00–0,05** ở mọi kịch bản, VB-FR **0,23–0,69**.

> **ĐÍNH CHÍNH 21/09.** Các số ở đoạn trên từng bị ghi sai (VB-MF 0,790, MLE
> 0,963, và "coverage bằng 0,00 ở mọi kịch bản và mọi phương pháp Bayes"). Nguyên
> nhân: `combine()` trong `chay_mo_phong_song_song.py` lấy giá trị thật của
> $p_{ss}$ từ `params.P` — vốn chỉ là placeholder `eye(2)` — thay vì từ `P0`, nên
> mọi Bias/RMSE/Coverage của $p_{11}, p_{22}$ trong PR-A được tính so với **1,0**
> thay vì 0,98/0,95 (S1, S3) và 0,95/0,90 (S2). Đã sửa (đọc từ `P0`, giống bản
> vá `run_scenario` của commit 70ccac5), gộp lại cả ba kịch bản, sinh lại mọi báo
> cáo phái sinh. Bản gộp sai giữ ở `results/07_simulation_truoc_sua_p11/` để đối
> chiếu. RMSE thắng/thua **không đổi** (MLE 19/27); coverage trung bình tăng
> 0,00–0,06 điểm; tuyên bố "$p_{ss}$ coverage 0,00 ở mọi phương pháp" **rút lại**.

Điều này khớp với phân tích độ
nhạy đã có (đổi sang Beta(20; 2) thì $p_{11}$ trên dữ liệu thật nhảy từ 0,733
lên 0,986). **Khi viết bài, không được nói "VB-MF đánh giá thấp bất định" như
thể đó là tính chất thuần túy của mean-field.**

### ĐỪNG nâng `sim_vb_n_steps` từ 800 lên 3000

Đã kiểm chứng bằng `scripts/kiem_tra_hoi_tu_vb.py` (một lần lặp S1, 3000 bước,
đọc thẳng tham số thang đo của guide): thang đo **đi lang quang cả hai chiều**,
tỷ lệ scale(3000)/scale(800) trải từ 0,55× đến 1,52×. Tức **chưa hội tụ ở cả 800
lẫn 3000** — nâng số bước không mua được gì mà đắt gấp 3,75×. Nguyên nhân nằm ở
`Trace_ELBO(num_particles=1)` cộng Adam `lr=0,01` cố định; muốn sửa đúng hướng
thì tăng `num_particles` hoặc cho tốc độ học giảm dần. Cột ELBO của script đó
**không dùng được** (ghi một mẫu đơn lẻ chứ không phải trung bình trượt).

### b=28: thủ phạm là ĐIỂM KHỞI TẠO, không phải dữ liệu — đã phân định

`b=28` của MLE hỏng ở **cả ba** kịch bản S1/S2/S3 dù chúng có DGP khác nhau. Đã
phân định dứt điểm bằng thực nghiệm: **giữ nguyên dữ liệu seed 28**, chỉ đổi
`base_seed` từ 28 sang 500 thì ước lượng **chạy trơn** (loglik = −3131,84).
Xác nhận cơ chế trong code: `init_raw(config, seed)` chỉ nhận `config` và `seed`,
**không đụng đến dữ liệu**, nên bộ ba điểm khởi tạo sinh từ seed 28/29/30 giống
hệt nhau ở mọi kịch bản — đó là lý do cùng một `b` hỏng ở cả ba.

**KHÔNG báo cáo thành "tỷ lệ không hội tụ của MLE".** Con số đó ngụ ý "khoảng 1%
bộ dữ liệu khó ước lượng", trong khi sự thật là **0% bộ dữ liệu khó**: chỉ một
lần rút ngẫu nhiên điểm khởi tạo bị xấu, độc lập hoàn toàn với dữ liệu. Nói như
vậy là mô tả sai bản chất của bộ ước lượng.

**Đã cài quy tắc thử lại, khai báo trước** (`OFFSET_THU_LAI = 1000` trong
`scripts/chay_mo_phong_song_song.py`): nếu cả ba điểm khởi tạo phân kỳ thì rút
một bộ điểm khởi tạo khác bằng `seed + 1000`, **giữ nguyên dữ liệu**. Đây không
phải chọn lọc kết quả — quy tắc cố định trước khi chạy, không nhìn vào dữ liệu
lẫn giá trị ước lượng thu được, và chỉ nói "gặp điểm xuất phát hỏng thì đổi điểm
xuất phát". Checkpoint nay lưu thêm khóa `seed_khoi_tao` để kiểm toán được lần
lặp nào đã phải thử lại. **Khi viết bài phải nêu rõ quy tắc này.**

Lưu ý vận hành vẫn đúng: lần lặp thất bại **không sinh checkpoint**, nên trước
khi có quy tắc thử lại thì mỗi lần chạy lại đều thử lại rồi hỏng lại (~20 giây).

### Công cụ mới của đợt này

| File | Vai trò |
|---|---|
| `chay_mo_phong.bat` | Menu: [1] chạy PR-A rồi **tự động sang PR-B**, [4] tiến độ, [5] gộp cả hai, [6] chạy thử, [7] đổi số tiến trình |
| `scripts/tien_do_mo_phong.py` | Bảng tiến độ; chỉ đọc đĩa nên chạy lúc nào cũng được, bao nhiêu cửa sổ cũng được |
| `scripts/tom_tat_mo_phong.py` | Sinh `results/tom_tat_mo_phong.txt`; đọc giá trị tham số thật thẳng từ `scenarios.py` |
| `scripts/kiem_tra_hoi_tu_vb.py` | Chẩn đoán hội tụ VB qua tham số thang đo của guide |

Thay đổi đáng kể khác: `chay_mo_phong_song_song.py` có thêm cờ `--B`; `_spawn`
được tổng quát hoá (thêm `script`, `args_chung`) để **PR-B dùng chung một bản
điều phối** thay vì viết bản thứ hai; `chay_mo_phong_egarch.py` viết lại thành
checkpoint **theo từng lần lặp** và chạy song song (trước đó tuần tự một luồng,
ước tính ~145 giờ); cả hai driver nay ghi checkpoint **nguyên tử** (`.tmp` rồi
`os.replace`) để mất điện giữa chừng không sinh ra file `.pt` cụt đôi.

### Số đo thời gian trên máy hiện tại (12 tiến trình song song)

| | Phút / lần lặp |
|---|---|
| MS-GARCH-X, MLE | 23–26 |
| MS-GARCH-X, VB-MF | 30–39 |
| MS-GARCH-X, VB-FR | 31–40 |
| MS-GARCH-X, MCMC | **2.466** (khoảng 41 giờ — lý do MCMC đứng ngoài mô phỏng) |
| EGARCH-X, MLE | 20,4 |
| EGARCH-X, VB-FR | 24,3 |
| EGARCH-X, VB-MF | 26,7 |

## Bẫy đã gặp (đọc kỹ trước khi kết luận "chưa chạy" / "không tồn tại")

1. **File `.pt` "biến mất" không có nghĩa là chưa từng chạy.** Từng có trường
   hợp `04_mcmc.pt` và `07_simulation_S1..S8.pt` bị xóa cục bộ (chưa commit
   xóa) trong khi dữ liệu thật vẫn nằm nguyên trong `git show HEAD:<path>`.
   Luôn kiểm tra `git log --all --oneline -- <path>` và `git ls-tree -r HEAD
   --name-only | grep ...` trước khi báo cáo "không có kết quả".
2. **`report_data.json` không tự động đồng bộ với `results/*.pt`.** Không có
   script "chạy lại toàn bộ, tự cập nhật JSON" — ai đó phải tự viết
   script trích xuất mỗi khi có kết quả mới. JSON có `NaN`/`0` không chứng
   minh được là "chưa chạy" — phải đối chiếu trực tiếp với file `.pt`.
3. **Bash tool: thư mục Tiếng Việt gây lỗi `cd` âm thầm.** `cd "Bài báo" &&
   ...` đôi khi báo lỗi nhưng thư mục làm việc vẫn đổi (hoặc ngược lại, báo
   thành công nhưng không đổi) — luôn `pwd` để xác nhận trước khi dùng đường
   dẫn tương đối. Ưu tiên đường dẫn tuyệt đối.
4. **Tên script không được trùng module chuẩn của Python.** Đặt tên
   `inspect.py` sẽ làm `import torch` lỗi vòng lặp (torch tự `import
   inspect`). Dùng scratchpad với tên file không trùng tên thư viện.
5. **In tiếng Việt ra console cần `PYTHONIOENCODING=utf-8`**, nếu không sẽ
   lỗi `UnicodeEncodeError` với `cp1252`.
6. **`python` chạy được; `python3` thì không** (bị Windows App Execution
   Alias chặn). *Máy hiện tại (09/2026) khác hẳn máy mô tả trong các mục cũ
   bên dưới*: **AMD Ryzen 7 5700X3D, 8 nhân / 16 luồng, NVIDIA RTX 4060,
   Python 3.13.14, torch 2.13.0+cu130 (`torch.cuda.is_available()` là `True`),
   pyro 1.9.1.** Mọi chỗ trong file này nói "12 lõi logic" hay "máy này không
   có GPU" là nói về một máy CŨ — đừng dùng để lập kế hoạch.
7. **`appendix_gen.py` từng có bug cột `Y` chưa định nghĩa** trong
   `bao_cao_ket_qua.tex` (chỉ có `C`/`R`/`L`, không có `Y` như bên
   `document.tex`) — khiến file chưa từng biên dịch thành công trước đây. Đã
   sửa (`Y`→`L`), nhưng nếu thêm bảng mới trong `appendix_gen.py`, nhớ chỉ
   dùng cột đã định nghĩa trong preamble của `bao_cao_ket_qua.tex`.
8. **Không tự bịa số liệu.** Mọi con số phải truy được nguồn `.pt`/JSON cụ
   thể. Nếu không tìm thấy nguồn, nói rõ `[KHÔNG XÁC MINH ĐƯỢC]` thay vì suy
   diễn.

## Trạng thái kết quả đã biết (snapshot — kiểm tra lại nếu nghi ngờ đã đổi)

- **B3 (GARCH-X Student-t, 1 chế độ) thắng AIC/BIC/QLIKE** — cả trong mẫu lẫn
  ngoài mẫu, dù M1 (MS-GARCH-X) có log-likelihood cao hơn.
- **VB mean-field (M2) cho NLS tốt nhất**, chi phí chỉ ~78 phút so với 424,7
  phút của MLE (M1).
- **MCMC không hội tụ sau 3 lần thử** (đã áp đủ 3 biện pháp chuẩn: mass
  matrix đặc, warmup gấp 4, khởi tạo từ MLE chính quy) — $\hat R=2{,}84$,
  ESS$\approx 5$. Chẩn đoán: hậu nghiệm đa đỉnh thật, không phải lỗi cấu
  hình (so sánh phân tán trong/giữa chuỗi giữa lần 1 và lần 3). Khuyến nghị:
  **không chạy lại MCMC trên chính MS-GARCH-X** — ROI thấp, đã đủ căn cứ báo
  cáo kết quả âm cho RQ5. Lưu ý: điều này **không** mâu thuẫn với Bước 3 của
  thiết kế mới; Bước 3 chạy MCMC trên *mô hình thắng cuộc*, có thể là một mô
  hình một trạng thái — bài toán hoàn toàn khác vì không có đối xứng hoán vị
  nhãn.
- **Mô phỏng 8 kịch bản (S1–S8)** đã hoàn thành cho MLE (21–30 lần lặp/kịch
  bản) và VB mean-field (15 lần lặp/kịch bản); MCMC chỉ hoàn thành ở S1 (3
  lần lặp, do chi phí ~40–45 giờ/lần lặp); VB full-rank **chưa từng chạy**
  trong mô phỏng. Coverage 95% của VB-MF trung bình ~0,33 trên cả 8 kịch bản
  — bằng chứng có hệ thống rằng VB-MF đánh giá thấp bất định hậu nghiệm.
- Tám kịch bản dùng **tham số minh họa** (không hiệu chỉnh VN30F1M thật),
  thiết kế để phủ: mức phân biệt trạng thái, độ dày đuôi (2 hướng), cỡ mẫu (2
  hướng), độ hiếm trạng thái, sai đặc tả.

**⚠️ Lỗi đã tìm thấy và sửa (10/2026) — ảnh hưởng số liệu $p_{11}$/$p_{22}$ đã
báo cáo của S1–S8.** `run_scenario()` (`msgarchx/simulation/run_simulation.py`)
và `combine()` (`scripts/pipeline/chay_mo_phong_song_song.py`) từng lấy giá trị "thật"
để tính Bias/RMSE bằng `_named_params(scenario.params)` — nhưng
`scenario.params.P` chỉ là **placeholder `eye(2)`** (tự ghi trong docstring
`_base_params`: "khong dung truc tiep, P0 rieng duoc truyen vao DGP"). Nghĩa
là Bias/RMSE của $p_{11}$/$p_{22}$ trong **toàn bộ `07_simulation_S1..S8.pt`
đã lưu, và mọi số liệu trích từ đó trong `bao_cao_ket_qua.tex`, đều tính sai**
— so với $p_{11}=p_{22}=1{,}0$ thay vì $P_0$ thật. Đo cụ thể: S1/MLE bias báo
cáo $-0{,}0222$, giá trị đúng chỉ $-0{,}0022$ (sai 10 lần); S2/MLE dấu bias
còn **đổi chiều** ($-0{,}0275\to+0{,}0225$). VB-MF/MCMC vẫn tệ ở cả hai cách
tính (kết luận định tính không đổi).

- **Đã sửa tại nguồn**: cả hai hàm trên giờ ghi đè `true_named["p11"]`/
  `["p22"]` bằng `scenario.P0` sau khi gọi `_named_params`. Có test hồi quy
  `tests/test_simulation.py::test_run_scenario_true_p11_p22_come_from_P0_not_placeholder`.
- **Bias/Mean của S1–S8 SỬA ĐƯỢC bằng đại số** từ chính `07_simulation_S*.pt`
  đã lưu, không cần chạy lại: $\mathrm{Var}=\mathrm{RMSE}_{sai}^2-\mathrm{Bias}_{sai}^2$
  không phụ thuộc $\theta_0$ (không đổi), nên
  $\mathrm{Bias}_{đúng}=(1{,}0+\mathrm{Bias}_{sai})-P_{0,đúng}$, rồi
  $\mathrm{RMSE}_{đúng}=\sqrt{\mathrm{Var}+\mathrm{Bias}_{đúng}^2}$.
- **Coverage KHÔNG sửa được bằng đại số** (cần chặn CI từng lần lặp, dữ liệu
  đó nằm trong `results/parts/sim/` — đã bị xóa ở commit `adceec7`) — muốn
  đúng phải chạy lại hoặc ghi chú giới hạn khi báo cáo.
- **CHƯA cập nhật `bao_cao_ket_qua.tex`/`appendix_gen.py`/`8_kich_ban_mo_phong.tex`
  với số liệu đã sửa** — đang chờ người dùng quyết định cách xử lý (sửa lại
  Bias/RMSE bằng đại số + ghi chú giới hạn Coverage, hay chạy lại toàn bộ).
  **Đã xử lý (21/09/2026)**: `bao_cao_ket_qua.tex`/`appendix_gen.py` sinh lại
  đúng từ dữ liệu PR-A/PR-B thật. `8_kich_ban_mo_phong.tex` đổi tên thành
  `Bài báo/mo_phong_khoi_phuc_tham_so.tex` và **bỏ hẳn phần phụ lục S1-S8**
  (người dùng chốt: chỉ giữ PR-A ba kịch bản, không cần bộ tám kịch bản minh
  họa cũ song song nữa — bộ tám kịch bản gộp đúng cách đã có sẵn trong
  `tong_hop_ket_qua.tex`). File PR-A trong tài liệu đổi tên vẫn đang hiển thị
  số liệu $p_{11}/p_{22}$ CŨ (trước khi vá lỗi `combine()` cùng ngày) — xem
  cảnh báo ngay trong tài liệu, cần chạy lại `sinh_pr_a_pr_b_tex.py` khi có
  Python.

## Quyết định thiết kế đã chốt (09/2026)

Đây là trạng thái hiện hành, **thay thế** mọi mô tả cũ mâu thuẫn trong
`document.tex` hay `bao_cao_ket_qua.tex`.

**Cấu trúc nghiên cứu ba bước — thay cho thiết kế "hai phần" cũ**

**MS-GARCH-X Student-t KHÔNG còn là "mô hình chính".** Vai trò đó giờ là
*kết quả* của Bước 1–2, không phải giả định đầu vào. Mọi mô tả cũ gọi nó là
mô hình chính đều đã lỗi thời.

| Bước | Nội dung | Tiêu chí |
|---|---|---|
| 1 | So sánh **toàn bộ** danh mục mô hình × **3 phương pháp** (MLE, VB-MF, VB-FR) | AIC/BIC trong mẫu; QLIKE/NLS/kiểm định VaR ngoài mẫu |
| 2 | **Mô phỏng khôi phục tham số** xác nhận bộ ước lượng đáng tin: 3 kịch bản × $B=100$ × $T=2000$, cùng 3 phương pháp | Bias, RMSE, coverage, Brier |
| 3 | **MCMC chạy một lần duy nhất** cho mô hình thắng cuộc từ Bước 1–2 | Chẩn đoán $\hat R$, ESS |

Vì sao thiết kế này tốt hơn thiết kế cũ: MCMC lần trước thất bại vì **đối xứng
hoán vị nhãn của cấu trúc chuyển trạng thái**. Nếu mô hình thắng cuộc là mô
hình **một trạng thái** (kết quả hiện có đang nghiêng về GARCH-X Student-t —
thắng cả AIC, BIC lẫn QLIKE), thì **không tồn tại đối xứng hoán vị nhãn nào
cả**, bài toán MCMC dễ hơn hẳn và nhiều khả năng hội tụ bình thường. Kết quả
MCMC cũ (không hội tụ trên MS-GARCH-X) vẫn giữ nguyên trong bài như bằng chứng
cho kết luận âm; hai thứ bổ sung cho nhau chứ không mâu thuẫn.

**Khoảng trống cài đặt nói trên ĐÃ ĐƯỢC XỬ LÝ (09/2026)**

Trước đây `bayes_model.py` chỉ hỗ trợ MS-GARCH-X hai trạng thái, nên VB/MCMC
không chạy được cho mô hình một trạng thái. Đã bổ sung
`msgarchx/inference/bayes_single.py` (mô hình Pyro cho một trạng thái, cả
GARCH lẫn EGARCH) và thêm tham số `model_fn`/`model_args` vào `fit_vb` và
`fit_mcmc` — mặc định giữ nguyên hành vi cũ.

## Pipeline: `scripts/pipeline/pipeline_all_models.py`

```
python scripts/pipeline/pipeline_all_models.py --kiem-ke      # xem trước: chạy gì, bỏ qua gì (không tính toán)
python scripts/pipeline/pipeline_all_models.py                # chạy đầy đủ
python scripts/pipeline/pipeline_all_models.py --nhanh        # cấu hình nhỏ, thử đường chạy
python scripts/pipeline/pipeline_all_models.py --bo-qua-mcmc  # dừng trước Bước 3
python scripts/pipeline/pipeline_all_models.py --chi-bang     # chỉ in bảng từ checkpoint đã có
python scripts/pipeline/pipeline_all_models.py --tieu-chi nls # đổi tiêu chí chọn mô hình
```

Hai nguyên tắc đã cài vào pipeline:

1. **Không chạy lại thứ đã có.** Kết quả hai mô hình chuyển trạng thái được
   **nạp lại** từ `results/02_mle.pt`, `03_vb.pt`, `06_oos.pt` thay vì ước
   lượng lại (tiết kiệm ~9 giờ). Bảng kết quả đánh dấu `*` ở những hàng nạp
   lại. Pipeline chỉ tính phần **mới**: 6 mô hình một trạng thái × 3 phương
   pháp.
2. **Checkpoint riêng cho từng (mô hình, phương pháp)** trong
   `results/pipeline/`. Chạy lại sẽ bỏ qua phần đã xong — cần thiết vì toàn
   bộ mất nhiều giờ và có thể bị ngắt.

`--nhanh` ghi vào **thư mục riêng** `results/pipeline_nhanh/`. Nếu dùng chung
thư mục thì các checkpoint chất lượng thấp sẽ bị lần chạy thật coi là "đã
xong" và bỏ qua — một cái bẫy im lặng.

**Files mới của đợt này**: `models/egarch.py`, `inference/model_spec.py`
(danh mục mô hình — nguồn sự thật duy nhất), `inference/bayes_single.py`,
`evaluation/oos_single.py`, `simulation/run_egarch.py`,
`dgp.simulate_egarch`, `scripts/pipeline/pipeline_all_models.py`,
`scripts/pipeline/chay_mo_phong_egarch.py`, cùng `tests/test_egarch.py` và
`tests/test_pipeline_parts.py`. **69 test pass.**

## Tiến trình chạy sắp tới (thứ tự khuyến nghị trên PC mới)

| # | Lệnh | Nội dung | Đã sẵn sàng? |
|---|---|---|---|
| 0 | `pipeline_all_models.py --kiem-ke` | Xem trước sẽ chạy gì | ✅ |
| 1 | `pipeline_all_models.py --nhanh` | Thử đường chạy (~10 phút) | ✅ |
| 2 | `pipeline_all_models.py` | **15 mục mới** (3 EGARCH MLE + 12 VB) rồi MCMC cho mô hình thắng | ✅ |
| 3 | `chay_mo_phong_egarch.py --nhanh` | Thử PR-B | ✅ |
| 4 | `chay_mo_phong_egarch.py` | **PR-B đầy đủ**: 3 kịch bản × 3 phương pháp × B=100 | ✅ |
| 5 | PR-A với B=100, 3 kịch bản | **XONG (900/900) trên máy PC, đã pull về** (11/2026) — `scripts/pipeline/chay_mo_phong_song_song.py` | ✅ |
| 6 | Batch hóa bộ lọc tiến | Đo được 10–89× ngay trên CPU | ⚠️ **chưa làm** |

Mục 6 là việc còn lại duy nhất trong bảng này.

## PR-A đã chạy xong — kết quả thật, đã pull về (11/2026)

Máy PC chạy xong PR-A, hai đợt pull liên tiếp:
1. Commit `87c84fe` ("Hoan tat PR-A (897/900) va sua loi khong sap nhan trang
   thai truoc khi gop") — 897/900, thiếu `b=28` ở cả ba kịch bản (MLE phân kỳ
   NaN cả ba điểm khởi tạo).
2. Commit `3c44311` ("PR-A du 900/900: them quy tac thu lai diem khoi tao khi
   ca ba deu phan ky") — thêm quy tắc: nếu cả ba điểm khởi tạo phân kỳ, thử
   lại với bộ điểm khởi tạo khác (`seed + 1000`), **giữ nguyên dữ liệu**. Quy
   tắc cố định trước, không nhìn vào kết quả — không phải chọn lọc kết quả.
   Đã xác nhận thủ phạm là điểm khởi tạo (không phải dữ liệu):
   `init_raw()` chỉ nhận `(config, seed)`, không dùng dữ liệu, nên seed
   28/29/30 hỏng giống hệt ở cả ba kịch bản dù DGP khác nhau.

**Kết quả cuối: PR-A đã 900/900, đầy đủ.** Cả hai lần đều xử lý theo cùng quy
trình: `git stash` code cục bộ đang dở, `pull` sạch, `stash pop` rồi giải
quyết xung đột (nếu có) — xem chi tiết bên dưới. Tóm tắt đầy đủ:
`results/tom_tat_mo_phong.txt`.

**Phát hiện quan trọng — cần biết trước khi diễn giải bất kỳ con số PR-A nào:**

1. **Lỗi thứ hai, độc lập với lỗi $p_{11}$/$p_{22}$ đã ghi ở trên, được tìm ra
   và sửa NGAY TRÊN máy PC**: đối xứng hoán vị nhãn (label-switching). Mỗi
   lần lặp mô phỏng, bộ ước lượng có thể trả về "trạng thái 1"/"trạng thái 2"
   theo thứ tự tùy ý (đổi nhãn cho ra đúng một hàm hợp lý) — gộp trực tiếp
   $B=100$ lần lặp mà không sắp nhãn trước sẽ trộn $\omega_1$ với $\omega_2$
   thành vô nghĩa. Đã sửa bằng `sap_nhan_trang_thai()` trong
   `msgarchx/simulation/run_simulation.py` (tiêu chí: phương sai vô điều kiện
   $\omega_s/(1-\alpha_s-\beta_s)$, quy ước trạng thái 1 = biến động thấp),
   áp dụng trước khi gộp ở cả đường chạy song song lẫn `run_scenario()` tuần
   tự. **VB đảo nhãn gần 50% số lần lặp** — tức gần như đúng bằng tung đồng
   xu, một phát hiện khoa học độc lập đáng đưa vào bài (VB không phân biệt
   được nhãn trạng thái tốt hơn ngẫu nhiên).
2. **Coverage của VB-MF/VB-FR RẤT THẤP trên phần lớn tham số cấu trúc** —
   nhiều tham số ($\omega,\alpha,\beta,p_{11},p_{22}$) có coverage $\approx
   0{,}00$ (lý tưởng $0{,}95$), tệ hơn nhiều so với mức $\approx 0{,}33$ quan
   sát được ở bộ 8 kịch bản cũ ($B=15$). Đây là bằng chứng mạnh hơn nhiều,
   cần đối chiếu cẩn thận trước khi đưa vào kết luận bài — $B=100$ đáng tin
   hơn hẳn $B=15$ về mặt sai số Monte Carlo.
3. **MLE thiếu 1 lần lặp mỗi kịch bản** (99/100 ở cả S1, S2, S3) — do phân kỳ
   NaN, không phải lỗi tính — ghi rõ khi báo cáo $n$.
4. Số liệu $p_{11}$/$p_{22}$ trong PR-A **đã đúng ngay từ đầu** vì
   `run_scenario()` sửa lỗi P0-placeholder trước khi máy PC chạy — không cần
   sửa lại đại số như trường hợp S1-S8 cũ.

**Xử lý xung đột hợp nhất (đáng biết nếu gặp lại tình huống tương tự)**:
`chay_mo_phong_egarch.py` và `chay_mo_phong_song_song.py` bị xung đột "sửa ở
xa / xóa (di chuyển) ở máy này" vì cùng lúc PC viết lại cả hai file (thêm hạ
tầng song song theo từng lần lặp, dùng chung `_spawn`) trong khi máy này di
chuyển chúng vào `scripts/pipeline/`. Đã lấy nội dung mới từ xa, đặt lại vào
`scripts/pipeline/`, áp lại hai sửa đổi đường dẫn đã làm trước đó
(`parents[1]`→`parents[2]`, sửa `from scripts.chay_mo_phong_song_song
import` → `from scripts.pipeline.chay_mo_phong_song_song import`). File
`run_simulation.py` tự động hợp nhất sạch (không xung đột) vì lỗi $p_{11}$/
$p_{22}$ và lỗi sắp nhãn nằm ở hai vị trí khác nhau trong cùng file — cả hai
sửa đổi cùng tồn tại đúng. `tests/test_simulation.py` xung đột nội dung do cả
hai bên cùng thêm test mới ở cuối file — gộp thủ công, giữ cả hai bộ test
(78 test tổng, đều pass). Ba script mới từ máy PC
(`kiem_tra_hoi_tu_vb.py`, `tien_do_mo_phong.py`, `tom_tat_mo_phong.py`) đã
được xếp vào `scripts/tien_ich/` theo đúng quy ước mới.

**Đã kiểm chứng (2 phép, đều đối chiếu với kết quả cũ đã báo cáo)**:
1. Tái cấu trúc `garch_mle.py` (thêm `family`) cho log-likelihood **trùng từng
   bit** với bản trong git trên cả ba mô hình GARCH.
2. `evaluate_single_regime` tái lập **chênh lệch 0,00e+00** mọi chỉ tiêu ngoài
   mẫu đã lưu của b1/b2/b3 (QLIKE, NLS, MAE, RMSE, số vi phạm VaR).

Nhờ (2), pipeline **nạp lại** cả ba mô hình GARCH một trạng thái thay vì chạy
lại (tiết kiệm thêm ~42 phút). Chỉ ba mô hình EGARCH và phần VB là tính mới.
Dùng `--chay-lai-tat-ca` nếu muốn ép ước lượng lại toàn bộ.

**Ràng buộc $\beta\in(0,1)$ cho EGARCH — quyết định có chủ đích.** Lý thuyết
chỉ đòi $|\beta|<1$, nhưng tiên nghiệm Bayes là Beta trên $(0,1)$; nếu để MLE
tìm trên $(-1,1)$ thì ba phương pháp tìm trên **hai không gian tham số khác
nhau**, phá đúng nguyên tắc cốt lõi của bài. Nên MLE dùng `sigmoid` chứ không
phải `tanh`. Đã ghi thành Nhận xét riêng trong `document.tex`.

*Lưu ý về họ GARCH (tồn tại từ trước, chưa sửa)*: MLE dùng `softplus` cho
$\beta$ (miền $(0,\infty)$) trong khi tiên nghiệm là Beta trên $(0,1)$ — cùng
loại bất nhất. **Không sửa** vì sẽ làm lệch toàn bộ kết quả đã báo cáo và đã
kiểm chứng tái lập; nghiệm thực tế nằm trong $(0,1)$ nên không ảnh hưởng kết
luận. Nếu viết lại từ đầu thì nên thống nhất.

**Khôi phục tham số có HAI nghiên cứu riêng, phục vụ hai mục đích khác nhau:**

| | Mô hình sinh dữ liệu | Kịch bản | Mục đích |
|---|---|---|---|
| PR-A | MS-GARCH-X Student-t (15 tham số) | S1/S2/S3 | Kiểm định bộ ước lượng ở nơi khó nhất; giải thích vì sao các phương pháp bất đồng về $p_{11}$ |
| PR-B | EGARCH-X Student-t, **một trạng thái** | E1/E2/E3 | Kiểm định bộ ước lượng MỚI, nhất là hệ số đòn bẩy $\theta$ |

**KHÔNG chạy khôi phục tham số cho mọi mô hình.** Với GARCH một trạng thái,
MLE là bài toán sách giáo khoa; làm mô phỏng cho nó tốn 900 lần ước lượng để
xác nhận điều gần như chắc chắn đúng. Nếu làm cho cả 8 mô hình thì thành
7.200 lần ước lượng, phần lớn dư thừa.

**Kịch bản E2 ($\theta_0=0$) là kịch bản không được bỏ.** Nó kiểm tra chiều
NGƯỢC LẠI của E1: mô hình có *bịa ra* hiệu ứng đòn bẩy khi dữ liệu vốn không
có không? Nếu $\hat\theta$ lệch khỏi 0 một cách hệ thống trên dữ liệu sinh với
$\theta_0=0$ thì mọi kết luận "VN30F1M có đòn bẩy" đều mất giá trị. Code:
`msgarchx/simulation/run_egarch.py`, DGP: `dgp.simulate_egarch`.

**Ngân sách mô phỏng: $B=100$, $T=2000$, 3 kịch bản.**

Vì sao 3 chứ không phải 8: $B$ và số kịch bản mua hai thứ **khác nhau, không
thay thế nhau**. $B$ mua *độ chính xác trong một điều kiện* — sai số Monte
Carlo của coverage giảm từ ±12,1 điểm % ở $B=15$ xuống ±4,7 điểm % ở $B=100$.
Số kịch bản mua *độ phủ qua nhiều điều kiện*. Khi $B=15$ phải dùng nhiều kịch
bản để bù nhiễu; khi $B=100$ mỗi kịch bản đã cho một câu trả lời đủ chắc, nên
3 kịch bản chọn đúng còn nhiều thông tin hơn 8 kịch bản nhiễu.

| Kịch bản | Trả lời câu hỏi gì |
|---|---|
| Mốc nền — hai trạng thái tách biệt rõ | Bộ ước lượng có khôi phục đúng trong ca "dễ" không |
| Nhận dạng yếu — hai trạng thái gần nhau | Vì sao các phương pháp bất đồng về $p_{11}$; **và** là nơi VB-MF vs VB-FR khác nhau rõ nhất (mean-field bỏ tương quan hậu nghiệm, mà tương quan mạnh nhất nằm đúng ở vùng nhận dạng yếu) |
| Đuôi dày — $\nu$ nhỏ, gần biên tồn tại | $\nu$ có khôi phục được không khi giá trị thật gần biên ($\hat\nu\approx 4{,}08$ trên VN30F1M, rất gần) |

Bộ 8 kịch bản đã chạy ($B$ thấp) **giữ nguyên làm phụ lục bổ trợ** — đã tồn
tại, không tốn thêm gì.

*Ràng buộc cần biết*: chốt $T=2000$ đồng nhất thì mất cặp kịch bản cỡ mẫu
($T=500$/$T=5000$), tức mất phát hiện phản trực giác "mẫu lớn nhất lại cho
coverage tệ nhất (0,236)". Muốn giữ phát hiện đó thì cần một kịch bản thứ tư
với $T$ khác.

*Mốc chi phí đã đo*: một lần lặp MLE ở $T=2000$, `n_starts=3`, một luồng chạy
**quá 10 phút chưa xong** (phép đo bị dừng giữa chừng) — dùng làm cận dưới khi
lập ngân sách, chưa phải số cuối. Máy có **12 lõi logic**.

## Kế hoạch mở rộng EGARCH (chưa cài đặt)

Lưới hai thang song song, mỗi cặp kề nhau chỉ khác **đúng một yếu tố**:

| 1 chế độ | Chuẩn | Student-t | Student-t + X |
|---|---|---|---|
| GARCH | G-N (4) ✓ | G-t (5) ✓ | G-X-t (7) ✓ |
| **EGARCH** | **E-N (5)** | **E-t (6)** | **E-X-t (8)** |

Hai chế độ: MS-G-N (10) ✓, MS-G-X-t (15) ✓, **MS-E-X-t (17)** — tùy chọn, đắt
(~9 giờ). EGARCH luôn hơn GARCH **+1 tham số mỗi trạng thái** ($\theta_s$, hệ
số đòn bẩy). Ba mô hình EGARCH một chế độ: ~45 phút chạy, ~1 ngày cài đặt.

Động cơ khoa học (không chỉ là "thêm một mô hình"): EGARCH mô hình hóa
$\ln h_t$ nên $h_t>0$ **theo cấu trúc** — giải đúng hạn chế bài đã tự ghi
trong Nhận xét về tính dương (GARCH-X phải clamp ở $10^{-8}$ vì
$\bmgamma^\top\Xvec$ có thể âm). Thêm nữa, skewness của $r_t$ là $-0{,}829$,
đúng dấu hiệu hiệu ứng đòn bẩy.

**Không chuyển neo Phần 2 sang MS-EGARCH-X** kể cả khi EGARCH thắng Phần 1:
kết luận "MCMC thất bại do đa đỉnh từ đối xứng hoán vị nhãn" là tính chất của
*cấu trúc chuyển trạng thái*, không phải của dạng hàm phương sai.

Chi tiết cài đặt dễ mắc lỗi thầm lặng:
- $\E|z|$: chuẩn $=\sqrt{2/\pi}$; Student-t chuẩn hóa
  $=\dfrac{2\sqrt{\nu-2}\,\Gamma(\frac{\nu+1}{2})}{(\nu-1)\sqrt{\pi}\,\Gamma(\frac{\nu}{2})}$.
  Test bắt buộc: $\nu\to\infty$ phải hội tụ về $\sqrt{2/\pi}$.
- $\beta$ đổi transform `softplus` → `tanh` ($|\beta|<1$, không cần
  $\beta\ge0$); $\omega,\alpha,\theta$ tự do hoàn toàn.
- Bỏ `positive_variance_link`, thay bằng guard chặn $\ln h_t\in[-30,30]$
  chống tràn `exp()`.
- $z_{t-1}$ phụ thuộc $h_{t-1}$ nên **không** tính sẵn ngoài vòng lặp được như
  `eps_sq` của GARCH → vòng lặp chậm hơn ~10–20%.
- `garch_mle.py` đã có sẵn tham số `dist`; chỉ cần thêm `family` song song.
- Nếu làm MS-EGARCH: **phải chốt và công bố** thu gọn Gray trên thang phương
  sai rồi lấy log, hay thu gọn trực tiếp $\ln h$.

## Tối ưu tính toán — TRẠNG THÁI: CHƯA CÀI ĐẶT, nhưng ĐÃ ĐO

**Thư viện `msgarchx/` hiện KHÔNG có một dòng nào về GPU hay batch.**
`grep -rn "cuda\|torch.device" msgarchx/` trả về rỗng. Đừng nói với người dùng
là "đã tối ưu GPU".

Đã có sẵn `scripts/tien_ich/do_gpu_vs_cpu.py` (từ commit 9e42233) — script đo độc lập,
chứa một bản `filter_batched` tham chiếu. Nó **chỉ để đo**, không được thư
viện dùng.

**Số đo trên MÁY CŨ (12 lõi, không có GPU, float64, T=1735,
forward+backward) — chưa đo lại trên máy hiện tại có RTX 4060:**

| B (số bộ tham số) | Tổng (giây) | **Giây / mỗi bộ** | Tăng tốc so với B=1 |
|---|---|---|---|
| 1 | 0,953 | 0,9527 | — |
| 16 | 1,433 | 0,0896 | **10,6×** |
| 64 | 1,606 | 0,0251 | **38×** |
| 256 | 2,749 | 0,0107 | **89×** |

**Kết luận từ số đo**: batch hóa là phần **chắc thắng và rất lớn**, ngay cả khi
chỉ chạy CPU — vì ở $B=1$ gần như toàn bộ thời gian là chi phí vòng lặp Python
(1735 vòng), và chi phí đó được chia đều cho $B$ bộ tham số. GPU là phần
**chưa đo được** (máy này không có GPU) và vẫn chưa chắc thắng vì tensor quá
nhỏ.

**Thứ tự đúng khi làm**: (1) batch hóa `ms_garch_forward_filter` — tensor
$(2,)\to(B,2)$, giữ bản vô hướng gốc làm oracle → (2) đo lại CPU → (3) thử GPU
→ (4) đo → (5) chỉ giữ GPU nếu số đo ủng hộ.

**Nơi batch hóa có tác dụng ngay**: $B=100$ lần lặp mô phỏng (PR-A và PR-B),
và nhiều điểm khởi tạo MLE ở giai đoạn Adam. Giai đoạn L-BFGS khó batch vì mỗi
điểm khởi tạo cần số bước tìm kiếm đường thẳng khác nhau.

## Nghiên cứu "độ thiên vị hiệu chỉnh" (calibration bias) — 10/2026

**Trạng thái: MS-GARCH-X đã code + smoke-test xong, sẵn sàng chạy thật.
EGARCH-X CHỦ ĐỘNG hoãn** — người dùng yêu cầu chờ máy PC (GPU) chạy xong Bước
1 vào ngày mai để có $\hat\theta$ EGARCH-X-t **thật** trước khi thêm vào
nghiên cứu này ("để đảm bảo tính trung thực") — KHÔNG được tự chế số liệu
EGARCH-X để chạy thử lưới này.

**Giả thuyết kiểm định**: nếu chọn $\theta_0$ bằng đúng nghiệm MLE ước lượng
từ dữ liệu thật, thì khi mô phỏng lại và ước lượng lại, bản thân MLE sẽ có vẻ
"thắng" — không phải vì MLE tốt hơn, mà vì $\theta_0$ tình cờ nằm đúng ở điểm
MLE dễ tìm lại nhất. Lặp lại tương tự với $\theta_0=$ nghiệm VB mean-field,
và $\theta_0=$ nghiệm VB full-rank. Đây chính là mối lo circularity đã nêu
khi người dùng từng đề xuất "chỉ cần 1 kịch bản hiệu chỉnh theo dữ liệu
thật": *"nếu lấy $\hat\theta$ của MLE làm $\theta_0$, ta đã ngầm giả định MLE
đúng trước khi bắt đầu thí nghiệm"*. Nghiên cứu này biến mối lo đó thành một
giả thuyết **kiểm định được bằng số**.

**Thiết kế đã chốt** (qua `AskUserQuestion`, 10/2026): lưới $3\times 3$ — 3
điểm hiệu chỉnh (θ0 = nghiệm MLE / VB-MF / VB-FR thật trên MS-GARCH-X
Student-t) × 3 phương pháp ước lượng lại (MLE/VB-MF/VB-FR, **không** MCMC).
$T=2000$, $B=100$ — đồng bộ PR-A/PR-B. Đọc theo hàng: nếu đường chéo (phương
pháp trùng nguồn hiệu chỉnh) luôn tốt hơn ngoài đường chéo, giả thuyết được
xác nhận.

**Code đã có**:
- `msgarchx/simulation/calibration_bias.py` — `build_calibration_scenarios()`
  dựng 3 `SimulationScenario` từ $\theta_0$ thật (KHÔNG tự đọc file — nhận
  `dict[str, MSGARCHXParams]` làm đầu vào, giữ thư viện không gắn đường dẫn
  dự án cụ thể, đúng quy ước của `run_egarch.py`).
- `scripts/pipeline/chay_mo_phong_hieu_chinh.py` — script nạp $\theta_0$ thật từ
  `results/02_mle.pt`/`03_vb.pt`, chạy song song nhiều tiến trình (mô phỏng
  theo đúng kiến trúc đã kiểm chứng của `chay_mo_phong_song_song.py`: mỗi
  lần lặp một checkpoint riêng trong `results/parts/hieu_chinh/`, ngắt giữa
  chừng chạy lại tự bỏ qua phần đã xong), gộp kết quả vào
  `results/08_hieu_chinh_CAL_<nguồn>.pt`, in bảng chẩn đoán $3\times3$.
  - `--nhanh`: smoke test 1 lần lặp/ô, đã chạy thử thành công (không dùng để
    báo cáo).
  - Chạy thật: `python scripts/pipeline/chay_mo_phong_hieu_chinh.py --jobs N` rồi
    `--combine`. Chi phí ước tính: 900 lần ước lượng (3×3×100) — cùng cấp độ
    với S1-S4 đã hoàn thành trước đây bằng đúng hạ tầng song song này.
- `tests/test_calibration_bias.py` (3 test) — 73 test tổng đều pass.
- Rename `_params_from_summary_mean` → `params_from_summary_mean` (công khai)
  trong `run_simulation.py` để tái dùng khi chuyển VB summary → `MSGARCHXParams`.

**Việc còn thiếu**: (1) chạy thật lưới MS-GARCH-X ($B=100$) trên máy có nhiều
lõi hơn — chưa chạy trên máy hiện tại; (2) khi Bước 1 EGARCH-X xong trên máy
GPU, lặp lại đúng pattern này cho EGARCH-X-t (dùng `bayes_single.py` thay vì
`bayes_model.py`, `run_egarch.py` làm tham chiếu thay vì `run_simulation.py`).

## Đã dọn cấu trúc thư mục `scripts/` (10/2026)

`code thuật toán/scripts/` trước có ~19 script rời ở cấp trên cùng, lẫn lộn
demo / driver chính / tiện ích / sinh báo cáo. Đã chia thành 4 thư mục con —
**từ nay script mới phải đặt đúng thư mục, đừng thả lại ở cấp trên cùng**:

| Thư mục | Nội dung | Ví dụ |
|---|---|---|
| `scripts/demo/` | Minh họa quy mô nhỏ, KHÔNG phải nguồn kết quả chính thức | `01_run_preliminary_tests.py`…`04_run_simulation_demo.py` |
| `scripts/pipeline/` | Driver chạy nghiên cứu thật, có checkpoint | `run_full_study.py`, `pipeline_all_models.py`, `run_one_model.py`, `run_models_parallel.py`, `chay_mo_phong_egarch.py`, `chay_mo_phong_hieu_chinh.py`, `chay_mo_phong_song_song.py`, `chay_do_nhay_prior.py` |
| `scripts/bao_cao/` | Sinh báo cáo/hình/bảng cho bài báo (đã có sẵn, thêm 2 file) | `01_sinh_hinh.py`, `02_sinh_tex.py`, `appendix_gen.py`, `report_data.json`, `ve_hinh_bai_bao.py`, `cap_nhat_bai_bao.py` |
| `scripts/tien_ich/` | Theo dõi tiến độ, tổng hợp, đo hiệu năng, xuất dữ liệu ngoài | `tien_do.py`, `tom_tat_ket_qua.py`, `build_html_report.py`, `do_gpu_vs_cpu.py`, `05_export_regime_signal_for_qp.py` |

**Đã xử lý khi di chuyển** (để tránh đúng cái bẫy đã cảnh báo trước đây):
- Mọi `Path(__file__).resolve().parents[1]` → `parents[2]` (thêm một cấp thư
  mục) ở tất cả file vừa chuyển vào `demo/`, `pipeline/`, `tien_ich/`, và hai
  file mới chuyển vào `bao_cao/` (`ve_hinh_bai_bao.py`, `cap_nhat_bai_bao.py`
  — ba file gốc trong `bao_cao/` đã dùng đúng `parents[2]`/`parents[3]` từ
  trước, không đổi).
- `from scripts.run_full_study import ...` → `from scripts.pipeline.run_full_study import ...`
  trong `chay_do_nhay_prior.py`, `chay_mo_phong_song_song.py`, `run_models_parallel.py`.
- 3 file `.bat` (`cai_dat_va_chay_full.bat`, `chon_cach_chay.bat`,
  `xem_bao_cao_html.bat`) và `README.md` đã cập nhật đường dẫn gọi lệnh.
- Đã kiểm chứng: `pytest` 73/73 pass, `pipeline_all_models.py --kiem-ke`,
  `run_one_model.py --list`, `run_models_parallel.py --list`,
  `chay_mo_phong_hieu_chinh.py --bang`, và import trực tiếp
  `scripts.pipeline.chay_do_nhay_prior` đều chạy đúng từ vị trí mới.
- Chưa động tới `Bài báo/` — yêu cầu gốc chỉ nhắm vào `scripts/`.

## Sở thích & phản hồi của người dùng

- Viết bằng tiếng Việt, văn phong học thuật.
- **Luôn phân biệt rõ: tác giả tuyên bố gì / bằng chứng trực tiếp cho thấy gì
  / chưa thể kết luận gì.** Không được nói mạnh hơn dữ liệu thực tế cho phép
  (đã từng xảy ra: mô tả kịch bản mô phỏng "hiệu chỉnh theo VN30F1M" trong khi
  code thực chạy dùng tham số minh họa — phải sửa lại mô tả, không sửa dữ
  liệu để khớp mô tả).
- TS. Châu (người hướng dẫn/phản biện) nhận xét bài **quá nặng lý thuyết,
  cần ứng dụng thực tế mạnh hơn** — trọng tâm câu hỏi hay gặp: vận hành ra
  sao, ứng dụng thế nào trong thực tế, độ tin cậy và kiểm định ra sao.
- Hướng ứng dụng đã thống nhất: công cụ hỗ trợ **ký quỹ động (dynamic
  margin) dựa trên $\widehat{\VaR}_t$** của mô hình, đối chiếu với tỷ lệ ký
  quỹ cố định hiện hành của VSD (13% từ 19/12/2025, từng là 10%/17% các giai
  đoạn trước) — không đề xuất thay quy định, mà làm công cụ đối chiếu/hỗ trợ
  quyết định margin nội bộ của CTCK. Kiểm định độ tin cậy dùng khung Basel
  traffic-light (đếm vi phạm VaR trong cửa sổ trượt 250 phiên, 3 vùng
  xanh/vàng/đỏ) — tái dùng trực tiếp kiểm định Kupiec đã có, không cần tính
  toán mới. Tận dụng phát hiện "MCMC không hội tụ" làm bài học quản trị rủi
  ro mô hình: không đưa phương pháp chưa kiểm chứng vào production.
- Người dùng muốn **giữ lại MCMC** trong khung lý thuyết (đã có lúc bị lược
  bỏ trong một đợt viết lại, đã khôi phục theo yêu cầu) — MCMC thất bại là
  một phát hiện khoa học có giá trị, không phải nội dung cần giấu.
- Khi phát hiện bất nhất giữa các file (VD: 4 vs 8 kịch bản, MCMC có/không
  trong thiết kế), **hỏi rõ trước khi tự quyết định hướng nào** — đây là
  quyết định phạm vi nghiên cứu, không phải lỗi kỹ thuật đơn thuần.
- **Code phải đơn giản và dễ đọc để người dùng tự kiểm tra lại.** Đây là yêu
  cầu tường minh, không phải sở thích phụ. Cụ thể:
  - Giữ **bản vô hướng gốc** làm bản tham chiếu và làm "oracle" kiểm tra tính
    đúng đắn; viết bản batch thành **hàm riêng bên cạnh**, không viết đè lên.
  - Mọi hàm tối ưu mới phải có test khẳng định **kết quả batch trùng kết quả
    vô hướng** (dùng `torch.allclose` hoặc chặt hơn).
  - `device` là **tham số tường minh**, mặc định `"cpu"` — không dùng trạng
    thái toàn cục ẩn, không tự động chuyển sang GPU.
  - Ưu tiên vài dòng rõ ràng hơn một dòng thông minh; chú thích tiếng Việt
    theo đúng phong cách code hiện có (xem `models/garch.py` làm mẫu).
