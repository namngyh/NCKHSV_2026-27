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

## Cảnh báo quan trọng: `document.tex` không có trong git

`git ls-files` xác nhận file này **chưa từng được add/commit**. Lịch sử
restructuring (bỏ MCMC, viết lại theo "hai phần", đổi ký hiệu...) nằm ở các
commit của một file cũ tên `khung_ly_thuyet_mo_hinh.tex` — file đó hiện đang
bị đánh dấu xóa (`D`) trong git status nhưng **chưa commit việc xóa**.
Nói cách khác: toàn bộ nội dung `document.tex` hiện tại chỉ tồn tại trên đĩa,
không có bản sao lưu nào trong git. Nếu người dùng đồng ý, nên `git add` và
commit file này (và xử lý gọn file cũ) sớm nhất có thể — đừng tự ý commit nếu
chưa được yêu cầu, nhưng nên chủ động nhắc lại nếu thấy file này bị sửa nhiều
mà vẫn chưa có git history.

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
6. **`python` chạy được (3.11.9, có torch 2.13 CPU); `python3` thì không**
   (bị Windows App Execution Alias chặn).
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

## Pipeline: `scripts/pipeline_all_models.py`

```
python scripts/pipeline_all_models.py --kiem-ke      # xem trước: chạy gì, bỏ qua gì (không tính toán)
python scripts/pipeline_all_models.py                # chạy đầy đủ
python scripts/pipeline_all_models.py --nhanh        # cấu hình nhỏ, thử đường chạy
python scripts/pipeline_all_models.py --bo-qua-mcmc  # dừng trước Bước 3
python scripts/pipeline_all_models.py --chi-bang     # chỉ in bảng từ checkpoint đã có
python scripts/pipeline_all_models.py --tieu-chi nls # đổi tiêu chí chọn mô hình
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
`dgp.simulate_egarch`, `scripts/pipeline_all_models.py`,
`scripts/chay_mo_phong_egarch.py`, cùng `tests/test_egarch.py` và
`tests/test_pipeline_parts.py`. **69 test pass.**

## Tiến trình chạy sắp tới (thứ tự khuyến nghị trên PC mới)

| # | Lệnh | Nội dung | Đã sẵn sàng? |
|---|---|---|---|
| 0 | `pipeline_all_models.py --kiem-ke` | Xem trước sẽ chạy gì | ✅ |
| 1 | `pipeline_all_models.py --nhanh` | Thử đường chạy (~10 phút) | ✅ |
| 2 | `pipeline_all_models.py` | **15 mục mới** (3 EGARCH MLE + 12 VB) rồi MCMC cho mô hình thắng | ✅ |
| 3 | `chay_mo_phong_egarch.py --nhanh` | Thử PR-B | ✅ |
| 4 | `chay_mo_phong_egarch.py` | **PR-B đầy đủ**: 3 kịch bản × 3 phương pháp × B=100 | ✅ |
| 5 | PR-A với B=100, 3 kịch bản | Chưa có driver riêng — `run_full_study.py` hiện chạy cả 8 kịch bản | ⚠️ **còn thiếu** |
| 6 | Batch hóa bộ lọc tiến | Đo được 10–89× ngay trên CPU | ⚠️ **chưa làm** |

Mục 5 và 6 là hai việc còn lại. Mục 6 nên làm **trước** mục 5 nếu có thời
gian, vì nó rút ngắn chính mục 5.

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

Đã có sẵn `scripts/do_gpu_vs_cpu.py` (từ commit 9e42233) — script đo độc lập,
chứa một bản `filter_batched` tham chiếu. Nó **chỉ để đo**, không được thư
viện dùng.

**Số đo thật trên máy này (12 lõi, không có GPU, float64, T=1735,
forward+backward):**

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
