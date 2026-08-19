# Code thuật toán — MS-GARCH-X Student-t cho VN30F1M

Cài đặt Python cho khung nghiên cứu mô tả trong `Bài báo/Bài báo chính.tex`:
ước lượng và dự báo biến động theo trạng thái thị trường bằng MS-GARCH-X với
sai số Student-t, so sánh ba phương pháp suy diễn — **MLE, Variational Bayes,
MCMC** — trên cùng một hàm hợp lý và cùng một cơ chế thu gọn (Gray).

## Cấu trúc

```
msgarchx/
  distributions.py     Student-t chuẩn hóa (eq:standardized-t-density)
  transforms.py         Biến đổi tham số không ràng buộc <-> miền hợp lệ
  models/
    garch.py            GARCH(1,1) / GARCH-X một trạng thái (mô hình đối chứng B1-B3)
    ms_garch.py          MS-GARCH-X: bộ lọc tiến (Phụ lục A), làm trơn Kim, xử lý đổi nhãn
  inference/
    param_spec.py        Vector tham số raw (MLE) + cấu hình prior (VB/MCMC)
    mle.py                MLE đa điểm khởi tạo (Adam + L-BFGS)
    bayes_model.py        Mô hình Pyro DÙNG CHUNG cho VB và MCMC (cùng forward filter)
    variational.py        Variational Bayes: mean-field (AutoNormal) & full-rank (AutoMultivariateNormal)
    mcmc.py                MCMC bằng NUTS, chẩn đoán R-hat/ESS qua arviz
    forecast.py            Mô phỏng dự báo nhiều bước, dự báo hậu nghiệm
    summary.py             Tóm tắt hậu nghiệm, tỷ số độ lệch chuẩn VB/MCMC
  evaluation/
    volatility_loss.py    MAE, RMSE, QLIKE
    density_scores.py      Điểm log, CRPS, PIT
    var_backtests.py       Kiểm định Kupiec, Christoffersen
    dm_test.py              Kiểm định Diebold-Mariano
  stats_tests/
    preliminary_tests.py  ADF, Phillips-Perron, KPSS, Ljung-Box, ARCH-LM, Jarque-Bera
  simulation/
    dgp.py                 Cơ chế sinh dữ liệu MS-GARCH-X Student-t
    scenarios.py            8 kịch bản mô phỏng S1-S8 (giá trị ví dụ, cần thay bằng giá trị chính thức)
    metrics.py               Bias, RMSE, Coverage, Brier score
    run_simulation.py        Điều phối B lần lặp x {MLE, VB, MCMC}
  data/
    loader.py               Nạp ohlc_export.csv (dữ liệu phút VN30F1M thật), gộp ngày, RV_t
    cleaning.py               Loại trùng, kiểm tra OHLC, lưới thời gian
    continuous_contract.py    Xây chuỗi hợp đồng liên tục (cần dữ liệu nhiều hợp đồng + ngày đáo hạn)
    realized_volatility.py    RV_t từ dữ liệu nội ngày
    exogenous.py               Bảng biến ngoại sinh X_{t-1}, chuẩn hóa chỉ theo tập huấn luyện

scripts/
  01_run_preliminary_tests.py        Demo: kiểm định sơ bộ trên VN30F1M thật
  02_fit_mle_demo.py                  Demo: MLE cho B1, B2, M1 (cấu hình rút gọn)
  03_fit_vb_mcmc_demo.py              Demo: VB (mean-field, full-rank) + MCMC (cấu hình rút gọn)
  04_run_simulation_demo.py           Demo: nghiên cứu mô phỏng (kịch bản S1)
  05_export_regime_signal_for_qp.py   Xuất xác suất chế độ biến động ra CSV cho dự án QP Analysis riêng
  run_full_study.py                   Điều phối TOÀN BỘ pipeline chính thức (data->MLE->VB->MCMC->R1->OOS->simulation)
  run_one_model.py                    Chạy TỪNG MỘT trong 20 mô hình MLE của kế hoạch so sánh, tự chọn lúc nào chạy
  build_html_report.py                Gộp mọi checkpoint trong results/ thành một báo cáo HTML

chon_cach_chay.bat            Menu Windows: chọn chạy toàn bộ / từng bước / từng mô hình / tạo báo cáo
cai_dat_va_chay_full.bat       Cài thư viện rồi chạy `run_full_study.py --preset full`
xem_bao_cao_html.bat           Tạo report.html từ results/ hiện có rồi mở trình duyệt

tests/       pytest cho toàn bộ module
```

## Cài đặt

```
pip install -r requirements.txt
```

Backend suy diễn dùng **PyTorch + Pyro** (không dùng JAX/NumPyro) vì có wheel
Windows ổn định sẵn trong môi trường này.

## Chạy thử

```
python -m pytest tests/ -q                       # ~42 test nhanh (vài giây)
python -m pytest tests/test_inference_smoke.py    # smoke test MLE/VB/MCMC (~2-3 phút)

python scripts/01_run_preliminary_tests.py        # kiểm định sơ bộ trên VN30F1M thật
python scripts/02_fit_mle_demo.py                 # MLE cho B1, B2, M1 (cấu hình demo rút gọn)
python scripts/03_fit_vb_mcmc_demo.py             # VB (mean-field, full-rank) + MCMC (demo)
python scripts/04_run_simulation_demo.py          # demo nghiên cứu mô phỏng (kịch bản S1)
```

## Chạy ước lượng CHÍNH THỨC cho bản thảo: `run_full_study.py`

`scripts/run_full_study.py` điều phối TOÀN BỘ quy trình (kiểm định sơ bộ ->
B1-B4/M1 MLE -> M2/M3 VB -> M4 MCMC -> R1 -> đánh giá ngoài mẫu -> mô phỏng
S1-S8), mỗi bước được **checkpoint** vào thư mục `results/` (mặc định) — nếu
dừng/tắt máy giữa chừng, chạy lại lệnh cũ sẽ bỏ qua các bước đã xong.

```
python scripts/run_full_study.py --preset dry-run   # kiem tra pipeline chay dung trong vai phut
python scripts/run_full_study.py --preset full       # chay that, xem canh bao thoi gian ben duoi
python scripts/run_full_study.py --preset full --only simulation   # chi chay lai 1 buoc cu the
```

Sau khi (hoặc trong khi) `results/` có dữ liệu, tạo báo cáo HTML tổng hợp:

```
python scripts/build_html_report.py --results-dir results --out report.html
```

Báo cáo đọc bất kỳ file `.pt` nào đã có trong `results/`, các bước chưa chạy
xong sẽ hiển thị "chưa có dữ liệu" thay vì lỗi — có thể tạo báo cáo TẠM THỜI
bất cứ lúc nào trong quá trình chạy dài. Báo cáo gồm: thống kê mô tả, kiểm
định sơ bộ, bảng tham số MLE/VB/MCMC (đã xử lý đổi nhãn trạng thái theo
eq:label-identification, kèm sai số chuẩn cho M1 tính bằng delta method từ
Hessian), đánh giá ngoài mẫu (QLIKE/MAE/RMSE/log-score/VaR), kiểm định
Diebold-Mariano giữa các mô hình, biểu đồ (giá/lợi suất/RV, xác suất trạng
thái, histogram PIT), kết quả mô phỏng S1-S8, và bảng chi phí tính toán.

### File .bat cho Windows (double-click, không cần gõ lệnh)

- **`cai_dat_va_chay_full.bat`**: tự cài `requirements.txt` rồi chạy
  `run_full_study.py --preset full`. An toàn chạy lại nhiều lần (bỏ qua các
  bước đã checkpoint xong). Ghi log vào `logs\run_full_<timestamp>.log`.
- **`xem_bao_cao_html.bat`**: tạo `report.html` từ `results\` hiện có rồi mở
  bằng trình duyệt mặc định. Chạy được bất cứ lúc nào, kể cả khi file trên
  đang chạy ở cửa sổ khác (xem tiến độ tạm thời).
- **`chon_cach_chay.bat`**: menu 4 lựa chọn, gộp mọi cách chạy bên dưới vào
  một cửa sổ duy nhất (xem "Bốn cách chạy pipeline").

## Bốn cách chạy pipeline

Cùng một pipeline (`run_full_study.py`) có thể chạy theo 4 cách khác nhau,
tùy bạn muốn "bấm một lần rồi để đó" hay "tự kiểm soát từng bước". Cả 4 cách
đều ghi vào **cùng thư mục `results/`** và **cùng checkpoint theo tên file**,
nên có thể trộn lẫn tùy ý (vd. chạy MLE bằng cách 2, rồi chạy MCMC bằng cách
1) mà không xung đột.

### Cách 1 — Chạy hết một lượt (`cai_dat_va_chay_full.bat` hoặc `--preset full`)

```
python scripts/run_full_study.py --preset full
```

Chạy tuần tự `data -> prelim -> mle -> vb -> mcmc -> r1 -> oos -> simulation`.
Đơn giản nhất nhưng sẽ **kẹt ở bước `mcmc` rất lâu** (xem cảnh báo thời gian
bên dưới) trước khi bao giờ tới `oos`/`simulation`. Phù hợp khi bạn chấp nhận
để máy chạy nhiều ngày liên tục không can thiệp.

### Cách 2 — Chạy từng bước của pipeline chính, tự chọn thứ tự (`--only`)

Dùng khi muốn **MLE/VB/R1 chạy trước, MCMC chạy sau** (khuyến nghị, vì MCMC
đắt hơn MLE ~100 lần — xem mục 13.2 trong `../README.md`):

```
python scripts/run_full_study.py --preset full --only data prelim mle vb r1
python scripts/run_full_study.py --preset full --only mcmc
python scripts/run_full_study.py --preset full --only oos simulation
```

Lưu ý thứ tự 3 lệnh trên **không tùy tiện**:
- Lệnh 1 và 2 có thể đảo cho nhau (mle/vb/r1 không phụ thuộc mcmc).
- Lệnh 3 (`oos`) **phải chạy SAU CÙNG**, sau khi `mcmc` đã xong: `run_oos_step`
  nhận `mcmc_res` làm tham số tùy chọn — nếu chạy `oos` trước khi có
  `04_mcmc.pt`, nó vẫn chạy được nhưng bảng so sánh sẽ **thiếu hẳn mô hình
  M4** (không báo lỗi, chỉ âm thầm bỏ qua). Nếu lỡ chạy `oos` sớm rồi mới
  chạy `mcmc`, phải xóa `results/06_oos.pt` và chạy lại `--only oos` để có
  đủ M4.
- Các bước hợp lệ cho `--only`: `data prelim mle vb mcmc r1 oos simulation`
  (đúng tên biến trong `STAGE_ORDER`, có thể liệt kê nhiều bước cách nhau
  bởi dấu cách).

### Cách 3 — Chạy TỪNG MỘT trong 20 mô hình MLE riêng lẻ (`run_one_model.py`)

Dùng khi muốn tự chọn đúng một mô hình để chạy, xem tiến độ trực tiếp trên
terminal, và giữ lại lịch sử mọi lần chạy (không phải là `run_full_study.py`,
mà là kế hoạch sàng lọc 20 đặc tả MLE riêng — xem bảng bên dưới):

```
python scripts/run_one_model.py --list              # xem 20 mo hinh, cai nao chay duoc
python scripts/run_one_model.py 5                    # chay mo hinh #05
python scripts/run_one_model.py 12 --n-starts 6 --n-adam 500 --n-lbfgs 250
python scripts/run_one_model.py 3 --force            # ep chay lai du da co checkpoint
```

| # | Mô hình | Trạng thái |
|---|---|---|
| 01 | GARCH(1,1) chuẩn, không X | 1 |
| 02 | GARCH(1,1) Student-t, không X | 1 |
| 03 | MS-GARCH(1,1) chuẩn (nu cố định lớn), không X | 2 |
| 04 | MS-GARCH(1,1) Student-t, không X | 2 |
| 05 | GARCH-X chuẩn, X = log_volume_lag1 | 1 |
| 06 | GARCH-X chuẩn, X = log_rv_lag1 | 1 |
| 07 | GARCH-X chuẩn, X = log_volume_lag1 + log_rv_lag1 | 1 |
| 08–11 | *(đã bỏ — cần biến ngoại sinh thứ 3 trở lên, dữ liệu thô chưa có)* | — |
| 12 | MS-GARCH-X chuẩn, X = log_volume_lag1 | 2 |
| 13 | MS-GARCH-X chuẩn, X = log_rv_lag1 | 2 |
| 14 | MS-GARCH-X chuẩn, X = log_volume_lag1 + log_rv_lag1 | 2 |
| 15–18 | *(đã bỏ — cùng lý do 08–11)* | — |
| 19 | HAR-RV | *(chưa có model trong `msgarchx/models/`)* |
| 20 | Realized GARCH | *(chưa có model trong `msgarchx/models/`)* |

Kết quả:
- Checkpoint riêng từng mô hình: `results/screening/<id>.pt` (nạp lại tự
  động, bỏ qua tính toán nếu đã có — trừ khi thêm `--force`).
- **Log gộp, chỉ ghi thêm (append), không bao giờ xóa**:
  `results/screening_log.txt` — mỗi lần chạy (kể cả khi nạp từ cache) đều
  thêm một khối mới kèm dấu thời gian, log-hợp-lý, AIC/BIC, tham số, chẩn
  đoán hội tụ từng điểm khởi tạo.

Mục 21 (mô hình thắng cuộc → Variational Bayes) và 22 (→ MCMC) trong kế
hoạch KHÔNG nằm trong `run_one_model.py`: chúng chạy trên mô hình đã chọn
sau khi so sánh kết quả 01–14, không phải một đặc tả cố định từ đầu.

### Cách 4 — Menu gộp cả 3 cách trên (`chon_cach_chay.bat`)

Double-click `chon_cach_chay.bat`, chọn số 1–4 trong menu hiện ra. Lựa chọn
2 và 3 chính là hai lệnh `--only` và `run_one_model.py` ở trên, chỉ bọc lại
cho tiện gõ (không cần nhớ cú pháp).

## Dữ liệu thật đã có: `ohlc_export.csv`

File này chứa dữ liệu **giá VN30F1M theo phút** (SYMBOL, TRADING_DATE,
OPEN/HIGH/LOW/CLOSE_PX, VOL, TRADING_TIME, ...) từ 2017-11-06 đến 2026-07-17
(~526.000 dòng, ~2170 phiên). Đây là một ký hiệu liên tục DUY NHẤT do nhà
cung cấp dữ liệu tự chọn hợp đồng gần nhất — **không có** cột mã hợp đồng hay
ngày đáo hạn riêng, và **không có** dữ liệu chỉ số VN30 cơ sở.

Do đó, với dữ liệu hiện có:
- Đã dùng được: `log(1+VOL_{t-1})`, `log(RV_{t-1}+eps)` (RV tính thật từ lợi
  suất nội ngày theo phút, không phải đại diện `r_t^2`).
- **Chưa dùng được**: `BASIS_{t-1}` (cần dữ liệu chỉ số VN30 cơ sở),
  `EXP_t`/`ROLL_t` (cần lịch đáo hạn hợp đồng và/hoặc dữ liệu nhiều hợp đồng
  đồng thời để `data/continuous_contract.py` tự phát hiện ngày chuyển hợp
  đồng). Khi có thêm hai nguồn dữ liệu này, ghép vào `build_exogenous_matrix`
  trong `data/exogenous.py` là đủ, không cần đổi mô hình.

Kết quả kiểm định sơ bộ trên dữ liệu thật (script 01) đã xác nhận đúng các
đặc điểm mà bài báo giả định: đuôi dày (độ nhọn ≈ 9.0, Jarque-Bera bác bỏ
chuẩn mạnh), có hiệu ứng ARCH (ARCH-LM p < 0.001), lợi suất dừng (ADF/PP bác
bỏ nghiệm đơn vị, KPSS không bác bỏ dừng).

## Lưu ý hiệu năng (quan trọng)

Bộ lọc tiến MS-GARCH-X (`models/ms_garch.py::ms_garch_forward_filter`) là
**một vòng lặp Python tuần tự theo T** (không vector hóa), vì phương sai tại
`t` phụ thuộc đệ quy vào các moment thu gọn tại `t-1`. Đây là đặc điểm cố hữu
của MS-GARCH thu gọn kiểu Gray, không phải lỗi cài đặt.

Đã thử `torch.jit.trace` / `jit_compile=True` (Pyro) để tăng tốc nhưng đo
thực nghiệm cho thấy **không cải thiện** (thậm chí chậm hơn) trên CPU với
vòng lặp tuần tự này, nên toàn bộ pipeline chạy ở chế độ eager thuần túy.

**Số liệu dưới đây đo TRỰC TIẾP trên máy chạy pipeline này**, dùng đúng dữ
liệu VN30F1M thật đã cache (`results/00_data.pt`, T huấn luyện = 1735 quan
sát) — không phải số liệu suy diễn hay đo trên máy khác. Đo lại bằng cách
gọi thẳng `ms_garch_forward_filter`/`garch_forward` và các hàm `fit_*` trên
đúng dữ liệu này nếu nghi ngờ máy bạn cho tốc độ khác:

| Thao tác (T=1735) | 1 trạng thái (GARCH/GARCH-X) | 2 trạng thái (MS-GARCH-X) |
|---|---|---|
| 1 lần forward (không gradient) | ~160 ms | ~999 ms |
| 1 lần forward+backward (có gradient, dùng trong MLE/NUTS) | ~647 ms | ~3.57 s |
| 1 bước tối ưu quy đổi (Adam+L-BFGS trộn lẫn, đo thực nghiệm) | ~741 ms | ~1.97 s |
| 1 bước SVI (Variational Bayes, mean-field) | — | ~1.39 s |
| 1 bước SVI (Variational Bayes, full-rank) | — | ~3.35 s |

Hệ quả thực tế cho cấu hình `full` mặc định
(`mle_n_starts=6, n_adam=500, n_lbfgs=250`; `vb_n_steps=3000`):

- **MLE một mô hình 1 trạng thái** (B1/B2/B3, hoặc từng mô hình 01/02/05-07
  của `run_one_model.py`): $6\times750\times0.741\text{s}\approx$ **~56
  phút/mô hình**.
- **MLE một mô hình 2 trạng thái** (B4/M1, hoặc 03/04/12-14 của
  `run_one_model.py`): $6\times750\times1.97\text{s}\approx$ **~2 giờ 28
  phút/mô hình**.
- **VB mean-field (M2)**: $3000\times1.39\text{s}\approx$ **~1 giờ 9 phút**.
  **VB full-rank (M3)**: $3000\times3.35\text{s}\approx$ **~2 giờ 47 phút**.
- Sai số chuẩn Hessian cho M1 (delta method, `natural_scale_standard_errors`):
  chi phí cố định **~30 giây**, không phụ thuộc `n_starts`.

**MCMC (NUTS) — không có một con số đơn lẻ đáng tin cậy**: mỗi mẫu NUTS tốn
một số bước leapfrog *thích nghi* (không cố định), mỗi bước leapfrog = một
lần forward+backward (~3.57s ở trên). Với `max_tree_depth=7` (tối đa
$2^7-1=127$ bước leapfrog/mẫu) và cấu hình 4 chuỗi × (250 khởi động + 400
mẫu) = 2600 vòng lặp:
  - **Giới hạn trên lý thuyết** (MỌI mẫu đều chạm trần 127 bước — hiếm khi
    xảy ra thực tế): $2600\times127\times3.57\text{s}\approx$ **~328 giờ
    (~13.7 ngày)**.
  - Đo thực nghiệm ở một cấu hình CŨ (`max_tree_depth=5`, tối đa 31
    bước/mẫu — KHÔNG PHẢI cấu hình `full` hiện tại) từng cho ~53.5
    giây/vòng lặp trung bình, tức khoảng 35-40 giờ cho *cấu hình đó*; con số
    này **không áp đặt trực tiếp được** cho `max_tree_depth=7` (cho phép
    nhiều bước leapfrog hơn mỗi mẫu).
  - **Khuyến nghị**: khi chạy MCMC thật (`--only mcmc`, `verbose=True` mặc
    định), theo dõi tốc độ `s/it` mà thanh tiến trình tqdm báo trực tiếp
    trong vài phút đầu — đó là ước tính đáng tin cậy NHẤT cho đúng máy và
    đúng lúc bạn đang chạy, đáng tin hơn bất kỳ con số tính sẵn nào ở đây.
- MLE một điểm khởi tạo trên toàn mẫu thật, vài trăm bước Adam + L-BFGS:
  **vài phút**. Nhân với số điểm khởi tạo (`n_starts`) để ra tổng thời gian
  MLE một mô hình (xem bảng trên).
- Các script demo (`02`, `03`, `04`) và preset `dry-run` của `run_full_study.py`
  đều dùng cấu hình RÚT GỌN để chạy trong vài phút. Preset `full` dùng ngân
  sách đầy đủ (đã chốt cùng người dùng) và **nên chạy trên máy mạnh / qua
  đêm nhiều ngày**, không phải trong một phiên tương tác — xem "Bốn cách
  chạy pipeline" ở trên để tách MLE/VB (vài giờ) ra chạy trước, MCMC (có
  thể hơn một ngày) chạy riêng sau.

Hướng tối ưu khả thi cho tương lai (chưa triển khai, xem `sec:han-che`/hướng
phát triển của bài báo): viết lại bộ lọc tiến bằng C++/Cython, dùng GPU nếu
có, tham số hóa lại (non-centered) để cải thiện hình học cho NUTS, hoặc giảm
tần suất tái ước lượng ngoài mẫu.

## Các quyết định thiết kế cần lưu ý

- **Cùng một hàm hợp lý cho cả ba phương pháp suy diễn**: `mle.py` gọi trực
  tiếp `ms_garch_forward_filter`; `bayes_model.py` (dùng chung bởi
  `variational.py` và `mcmc.py`) cũng gọi đúng hàm đó qua `pyro.factor`. Đây
  là yêu cầu cốt lõi của bài báo (Mục 4, checklist Phụ lục B mục 7).
- **`float64` mặc định toàn cục** (`msgarchx/__init__.py`) vì log-likelihood
  MS-GARCH nhạy với sai số làm tròn (logsumexp, tỷ lệ phương sai giữa hai
  trạng thái).
- **Ràng buộc dương của phương sai** (`transforms.positive_variance_link`):
  mặc định `mode="clamp"` (giữ nguyên dạng cộng của eq:main-variance, chỉ cắt
  đáy); có tùy chọn `mode="softplus"` cho ràng buộc dương tuyệt đối, đúng như
  bài báo thảo luận sau eq:garchx.
- **Prior cho `alpha_s`, `beta_s`** (Beta(2,2) trên (0,1)) là lựa chọn của
  cài đặt này vì bài báo không đặc tả prior riêng cho hai tham số này — xem
  docstring `PriorConfig` trong `param_spec.py`.
- **8 kịch bản mô phỏng S1-S8** dùng giá trị tham số VÍ DỤ hợp lý về mặt định
  tính (xem `simulation/scenarios.py`); bài báo đánh dấu `\CanBoSung` cho giá
  trị chính thức — cần thay trước khi báo cáo kết quả mô phỏng trong bản thảo.

## TODO còn lại theo bài báo

- Bổ sung `BASIS_t`, `EXP_t`, `ROLL_t` khi có dữ liệu chỉ số cơ sở + lịch đáo hạn.
- Chốt giá trị tham số chính thức cho 8 kịch bản mô phỏng S1-S8.
- Chạy ước lượng MLE/VB/MCMC đầy đủ (nhiều điểm khởi tạo, nhiều bước, ≥4 chuỗi
  phân tán) trên toàn bộ mẫu thật và lưu kết quả vào các bảng trong bài báo
  (Bảng 7.x).
- Dự báo cuốn chiếu ngoài mẫu (`inference/forecast.py`) + đánh giá
  QLIKE/điểm log/VaR ngoài mẫu trên dữ liệu thật (hiện có ví dụ hàm, chưa có
  script điều phối rolling window đầy đủ).
- Mô hình mở rộng `nu_s` theo trạng thái (R1) đã được `ModelConfig(state_specific_nu=True)`
  hỗ trợ ở tầng model/inference, nhưng chưa có script demo riêng.
