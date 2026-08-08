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

scripts/     Script demo chạy trực tiếp (xem bên dưới)
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
Số liệu đo được (CPU, T cố định, sau khi "làm nóng"):

| T | thời gian 1 lần forward+backward |
|---|---|
| 80 | ~27 ms |
| 150 | ~51 ms |
| 2170 (toàn bộ mẫu thật) | ~0.5-1 s |

Hệ quả thực tế:
- MLE một điểm khởi tạo trên toàn mẫu thật (~2170 quan sát), vài trăm bước
  Adam + L-BFGS: **vài phút**. Nhân với số điểm khởi tạo (`n_starts`).
- **MCMC (NUTS) đo trực tiếp tại T=2170** (không ngoại suy): với
  `max_tree_depth=5` (giới hạn tối đa 32 bước leapfrog/mẫu, thấp hơn nhiều so
  với mặc định 1024 của NUTS), trung bình **~53.5 giây/vòng lặp NUTS** trong
  giai đoạn warmup+sampling (hình học hậu nghiệm khá "cứng": step size hội tụ
  về ~1e-4). Với cấu hình 4 chuỗi × 400 mẫu + warmup 250 = 2600 vòng lặp, tổng
  thời gian ước tính **~35-40 giờ (1.5-2 ngày)** trên CPU thông thường. Đây là
  con số đo tại đúng quy mô dữ liệu thật, không phải ước lượng suy diễn.
- Các script demo (`02`, `03`, `04`) và preset `dry-run` của `run_full_study.py`
  đều dùng cấu hình RÚT GỌN để chạy trong vài phút. Preset `full` dùng ngân
  sách đầy đủ (đã chốt cùng người dùng) và **nên chạy trên máy mạnh / qua
  đêm nhiều ngày**, không phải trong một phiên tương tác.

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
