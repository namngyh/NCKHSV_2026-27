"""MCMC bằng NUTS (No-U-Turn Sampler) và chẩn đoán hội tụ.

LÝ THUYẾT
    MCMC (Markov Chain Monte Carlo) tạo một chuỗi Markov có phân phối dừng là
    hậu nghiệm p(theta | r); sau giai đoạn khởi động, các trạng thái của chuỗi
    được dùng như mẫu từ hậu nghiệm. Khác VB, MCMC không giả định dạng của hậu
    nghiệm, nên được dùng làm CHUẨN THAM CHIẾU để đánh giá VB (tỷ số R_sd).

    Hamiltonian Monte Carlo (HMC): coi U(theta) = -log p(theta | r) là thế năng,
    thêm động lượng p ~ N(0, M) và mô phỏng chuyển động Hamilton bằng tích phân
    leapfrog với bước eps:
        p <- p - (eps/2) grad U(theta);  theta <- theta + eps M^{-1} p;  p <- p - (eps/2) grad U(theta)
    Điểm cuối quỹ đạo được chấp nhận với xác suất Metropolis min(1, exp(-dH)).
    Nhờ dùng gradient, HMC đi xa trong không gian tham số mà vẫn chấp nhận cao.

    NUTS (Hoffman & Gelman, 2014) tự chọn độ dài quỹ đạo: nhân đôi quỹ đạo cho
    tới khi nó bắt đầu "quay đầu" (U-turn) hoặc đạt độ sâu cây tối đa (tối đa
    2^depth bước leapfrog mỗi mẫu). Trong giai đoạn khởi động (warmup), bước eps
    được thích nghi để xác suất chấp nhận trung bình đạt 0,85, và ma trận khối
    lượng M được ước lượng từ mẫu (full_mass = True: M đặc, nắm tương quan).

    Chẩn đoán hội tụ (thư viện arviz):
    - R-hat (Gelman-Rubin, bản chuẩn hóa xếp hạng của Vehtari và cs. 2021): so
      sánh phương sai giữa các chuỗi với trong chuỗi; gần 1 khi các chuỗi trộn
      tốt. Ngưỡng thường dùng: R-hat < 1,01.
    - ESS (cỡ mẫu hiệu dụng): số mẫu độc lập tương đương, tính đến tự tương quan.
    - Số bước phân kỳ (divergence): dấu hiệu hình học hậu nghiệm khó lấy mẫu.

THƯ VIỆN
    pyro.infer.NUTS, pyro.infer.MCMC : bộ lấy mẫu và vòng chạy chuỗi.
    arviz.rhat, arviz.ess            : chẩn đoán hội tụ.
    torch.Generator                  : bộ sinh số ngẫu nhiên riêng cho điểm khởi tạo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import pyro
import torch
from pyro.infer import MCMC, NUTS

from garch_vn30.mo_hinh.danh_muc import ModelSpec
from garch_vn30.suy_dien.bayes import SITES, PriorConfig, model, posterior_summary

# Trần độ lớn nhiễu loạn trên thang log/logit: phương pháp delta phóng đại vô
# lý khi tham số gần biên (x hoặc x(1-x) rất nhỏ).
TRAN_NHIEU = 1.0


@dataclass
class MCMCResult:
    """Kết quả MCMC."""
    samples: dict[str, torch.Tensor]           # mẫu gộp các chuỗi: (n_chains*n_samples, ...)
    samples_by_chain: dict[str, torch.Tensor]  # mẫu theo chuỗi: (n_chains, n_samples, ...)
    summary: dict[str, dict[str, torch.Tensor]]   # mean, sd, q2.5, q97.5
    rhat: dict[str, float]                     # R-hat lớn nhất của từng site
    ess_bulk: dict[str, float]                 # ESS nhỏ nhất của từng site
    max_rhat: float
    min_ess: float
    n_divergences: int


def khoi_tao_tu_mle(params, spec: ModelSpec, chain: int, se: dict | None = None,
                    base_seed: int = 5000, k_se: float = 2.0,
                    scale_du_phong: float = 0.1) -> dict:
    """Điểm xuất phát của chuỗi `chain`, đặt quanh nghiệm MLE.

    - Chuỗi 0 đặt ĐÚNG tại nghiệm MLE.
    - Các chuỗi khác nhiễu loạn k_se = 2 sai số chuẩn (từ mle.standard_errors):
      đủ phân tán để R-hat có ý nghĩa chẩn đoán (Gelman-Rubin cần điểm xuất phát
      phân tán hơn hậu nghiệm), nhưng không đẩy chuỗi sang vùng xa của không gian
      tham số. Không có sai số chuẩn thì nhiễu loạn 10% giá trị tham số.
    - Nhiễu loạn tôn trọng miền của từng tham số: tham số dương nhiễu trên thang
      log (độ lớn SE/x), tham số trong (0,1) nhiễu trên thang logit (độ lớn
      SE/(x(1-x))) -- quy đổi theo phương pháp delta.

    Trả về dict {tên site: giá trị} trên MIỀN HỢP LỆ; Pyro tự đổi sang không gian
    không ràng buộc.
    """
    g = torch.Generator().manual_seed(base_seed + chain)
    tai_mle = chain == 0

    def do_lon(ten, x):
        """Độ lớn nhiễu loạn trên thang tự nhiên (tensor, vì gamma là vector)."""
        x = torch.as_tensor(x, dtype=torch.float64)
        if tai_mle:
            return torch.zeros_like(x)
        if se is not None and ten in se:
            return k_se * torch.as_tensor(se[ten], dtype=torch.float64)
        return scale_du_phong * x.abs().clamp(min=1e-3)

    def cong(ten, x):
        """Tham số tự do: cộng nhiễu chuẩn."""
        x = torch.as_tensor(x, dtype=torch.float64)
        return x + do_lon(ten, x) * torch.randn(x.shape, generator=g, dtype=torch.float64)

    def qua_log(ten, x):
        """Tham số dương: nhiễu trên thang log."""
        x = torch.as_tensor(x, dtype=torch.float64)
        s = (do_lon(ten, x) / x.abs().clamp(min=1e-8)).clamp(max=TRAN_NHIEU)
        return x * torch.exp(s * torch.randn(x.shape, generator=g, dtype=torch.float64))

    def qua_logit(ten, x):
        """Tham số trong (0,1): nhiễu trên thang logit."""
        x = torch.as_tensor(x, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
        s = (do_lon(ten, x) / (x * (1 - x)).clamp(max=1.0, min=1e-8)).clamp(max=TRAN_NHIEU)
        return torch.sigmoid(torch.log(x) - torch.log1p(-x)
                             + s * torch.randn(x.shape, generator=g, dtype=torch.float64))

    ra = {"mu": cong("mu", params.mu),
          "alpha": qua_logit("alpha", params.alpha),
          "beta": qua_logit("beta", params.beta)}
    # omega: tự do với EGARCH (phương trình cho log h), dương với GARCH.
    ra["omega"] = (cong("omega", params.omega) if spec.family == "egarch"
                   else qua_log("omega", params.omega))
    if spec.family == "egarch":
        ra["theta"] = cong("theta", params.theta)
    if spec.use_exog:
        ra["gamma"] = cong("gamma", params.gamma)
    if spec.dist == "student_t":
        ra["nu_minus_2"] = qua_log("nu", torch.as_tensor(params.nu) - 2.0)
    return ra


def inference_data(post: dict):
    """Đóng gói mẫu {tên: mảng (n_chuỗi, n_mẫu, ...)} thành InferenceData của arviz.

    Cú pháp arviz.from_dict đổi ở phiên bản 1.0: bản < 1.0 nhận posterior=...,
    bản >= 1.0 nhận dict {"posterior": ...}. Truyền sai dạng KHÔNG báo lỗi mà tạo
    dữ liệu rác (R-hat thành NaN), nên chọn cú pháp theo phiên bản.
    """
    if int(az.__version__.split(".")[0]) >= 1:
        return az.from_dict({"posterior": post})
    return az.from_dict(posterior=post)


def _luu_tien_do(checkpoint_path, moi: int):
    """Tạo hàm "hook" mà Pyro gọi sau mỗi vòng lấy mẫu; cứ `moi` vòng thì ghi
    trạng thái sampler ra tệp (một chuỗi NUTS có thể chạy nhiều giờ)."""
    if checkpoint_path is None:
        return None
    cp = Path(checkpoint_path)
    cp.parent.mkdir(parents=True, exist_ok=True)

    def hook(kern, mau, giai_doan, i, **_):
        if i % moi:
            return
        try:
            torch.save({"giai_doan": giai_doan, "vong": i,
                        "step_size": float(getattr(kern, "step_size", float("nan"))),
                        "tham_so": {k: v.detach().clone() for k, v in (mau or {}).items()},
                        "thoi_diem": time.time()}, cp)
        except Exception:
            pass          # việc lưu tiến độ không được làm hỏng lần chạy
    return hook


def fit_mcmc(r: torch.Tensor, X: torch.Tensor, spec: ModelSpec,
             prior: PriorConfig | None = None, n_chains: int = 4, n_warmup: int = 500,
             n_samples: int = 1000, seed: int = 0, max_tree_depth: int = 8,
             full_mass: bool = False, initial_params=None, checkpoint_path=None,
             checkpoint_moi: int = 25, verbose: bool = False) -> MCMCResult:
    """Lấy mẫu hậu nghiệm bằng NUTS; các chuỗi chạy TUẦN TỰ (tránh lỗi tạo tiến trình trên Windows).

    Tham số:
        n_chains, n_warmup, n_samples: số chuỗi, số vòng khởi động, số mẫu giữ lại mỗi chuỗi.
        seed          : chuỗi c dùng hạt giống seed + c.
        max_tree_depth: tối đa 2^depth bước leapfrog mỗi mẫu -- mỗi bước là một
                        lần tính xuôi + lan truyền ngược qua cả chuỗi dữ liệu, nên
                        đây là nút chi phí chính.
        full_mass     : dùng ma trận khối lượng đặc.
        initial_params: None (Pyro tự khởi tạo), dict (dùng chung), list (phần tử
                        thứ c cho chuỗi c), hoặc hàm c -> dict (vd. khoi_tao_tu_mle).
        checkpoint_path, checkpoint_moi: tệp và tần suất lưu tiến độ.
    Trả về:
        MCMCResult với mẫu, tóm tắt, R-hat, ESS và số bước phân kỳ.
    """
    theo_chuoi: dict[str, list[torch.Tensor]] = {name: [] for name in SITES}
    n_div = 0
    for c in range(n_chains):
        pyro.set_rng_seed(seed + c)
        kernel = NUTS(model, full_mass=full_mass, target_accept_prob=0.85,
                      max_tree_depth=max_tree_depth)
        if initial_params is None or isinstance(initial_params, dict):
            init_c = initial_params
        elif isinstance(initial_params, (list, tuple)):
            init_c = initial_params[c]
        else:
            init_c = initial_params(c)
        mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=n_warmup, num_chains=1,
                    initial_params=init_c,
                    hook_fn=_luu_tien_do(checkpoint_path, checkpoint_moi),
                    disable_progbar=not verbose)
        mcmc.run(r, X, spec, prior)          # tham số truyền tiếp cho hàm bayes.model
        mau = mcmc.get_samples()
        for name in SITES:
            if name in mau:
                theo_chuoi[name].append(mau[name])
        try:
            # diagnostics() cần >= 4 mẫu; ở đây chỉ dùng để đếm số bước phân kỳ.
            diag = mcmc.diagnostics()
            for d in diag.get("divergences", {}).values():
                n_div += len(d) if hasattr(d, "__len__") else int(d)
        except (AssertionError, RuntimeError):
            pass

    by_chain = {name: torch.stack(ds) for name, ds in theo_chuoi.items() if ds}
    samples = {name: x.reshape(-1, *x.shape[2:]) for name, x in by_chain.items()}

    try:
        idata = inference_data({name: x.numpy() for name, x in by_chain.items()})
        rhat_ds, ess_ds = az.rhat(idata), az.ess(idata, method="bulk")
        rhat = {name: float(rhat_ds[name].values.max()) for name in by_chain}
        ess = {name: float(ess_ds[name].values.min()) for name in by_chain}
    except Exception:
        # Cấu hình quá nhỏ (vd. 1 chuỗi, ít mẫu) thì không tính được R-hat.
        rhat = {name: float("nan") for name in by_chain}
        ess = {name: float("nan") for name in by_chain}

    return MCMCResult(
        samples=samples, samples_by_chain=by_chain, summary=posterior_summary(samples),
        rhat=rhat, ess_bulk=ess,
        max_rhat=max(rhat.values()) if rhat else float("nan"),
        min_ess=min(ess.values()) if ess else float("nan"),
        n_divergences=n_div,
    )
