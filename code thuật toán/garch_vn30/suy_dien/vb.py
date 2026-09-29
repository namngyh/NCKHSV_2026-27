"""Bayes biến phân (Variational Bayes, VB).

LÝ THUYẾT
    Hậu nghiệm p(theta | r) không có dạng đóng. VB thay nó bằng phân phối gần
    đúng q_phi trong một họ đơn giản, chọn tham số phi để cực đại cận dưới bằng
    chứng (ELBO):

        ELBO(phi) = E_q[ log p(r, theta) - log q_phi(theta) ]
                  = log p(r) - KL( q_phi || p(theta | r) )   <=   log p(r).

    Cực đại ELBO tương đương cực tiểu độ phân kỳ Kullback-Leibler từ q tới hậu
    nghiệm thật. Gọi theta~ ∈ R^d là tham số sau khi Pyro biến đổi về R. Hai họ q:

    - mean-field : q(theta~) = tích các N(m_j, s_j^2) độc lập -- bỏ qua mọi tương
                   quan giữa các tham số (Pyro: AutoNormal).
    - full-rank  : q(theta~) = N(m, L L^T), L tam giác dưới -- nắm được tương
                   quan tuyến tính giữa các tham số (Pyro: AutoMultivariateNormal).

    Gradient của ELBO ước lượng bằng mẹo tham số hóa lại (reparameterization):
    theta~ = m + L * eps với eps ~ N(0, I), nên đạo hàm đi xuyên qua phép lấy
    mẫu; mỗi bước dùng một mẫu eps (Trace_ELBO, num_particles = 1). Tham số phi
    được cập nhật bằng Adam -- quy trình gọi là suy diễn biến phân ngẫu nhiên (SVI).

THƯ VIỆN (Pyro)
    AutoNormal / AutoMultivariateNormal : tự tạo họ q ("guide") từ mô hình.
    SVI(model, guide, optim, loss)       : vòng tối ưu ELBO; svi.step() làm một bước
                                           và trả về -ELBO của bước đó.
    Trace_ELBO                           : ước lượng ELBO bằng mẫu Monte Carlo.
    Predictive(model, guide, num_samples): rút mẫu tham số từ q sau khi khớp.
    pyro.set_rng_seed / clear_param_store: cố định hạt giống, xóa tham số guide cũ.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyro
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, AutoNormal

from garch_vn30.mo_hinh.danh_muc import ModelSpec
from garch_vn30.suy_dien.bayes import SITES, PriorConfig, model, posterior_summary


@dataclass
class VBResult:
    """Kết quả VB."""
    guide: object                              # phân phối q đã khớp (đối tượng Pyro)
    elbo_trace: list[float]                    # ELBO theo từng bước (theo dõi hội tụ)
    final_elbo: float                          # trung bình ELBO của 50 bước cuối
    samples: dict[str, torch.Tensor]           # mẫu tham số rút từ q, theo tên site
    summary: dict[str, dict[str, torch.Tensor]]   # mean, sd, q2.5, q97.5 của từng site
    guide_type: str                            # "mean_field" hoặc "full_rank"


def fit_vb(r: torch.Tensor, X: torch.Tensor, spec: ModelSpec, guide_type: str = "mean_field",
           prior: PriorConfig | None = None, n_steps: int = 3000, lr: float = 0.01,
           num_particles: int = 1, n_posterior_draws: int = 2000, seed: int = 0,
           verbose: bool = False) -> VBResult:
    """Khớp VB cho mô hình `spec` trên dữ liệu (r, X).

    Tham số:
        guide_type        : "mean_field" hoặc "full_rank".
        prior             : tiên nghiệm (mặc định PriorConfig()).
        n_steps, lr       : số bước SVI và tốc độ học của Adam.
        num_particles     : số mẫu Monte Carlo ước lượng ELBO mỗi bước.
        n_posterior_draws : số mẫu rút từ q sau khi khớp.
        seed              : hạt giống (kết quả tái lập được).
    Trả về:
        VBResult. ELBO chỉ so sánh được giữa các lần khớp của CÙNG một mô hình,
        không dùng thay AIC/BIC để xếp hạng các mô hình khác nhau.
    """
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    if guide_type == "mean_field":
        guide = AutoNormal(model)
    elif guide_type == "full_rank":
        guide = AutoMultivariateNormal(model)
    else:
        raise ValueError(f"guide_type khong hop le: {guide_type!r}")

    svi = SVI(model, guide, pyro.optim.Adam({"lr": lr}),
              loss=Trace_ELBO(num_particles=num_particles))
    elbo_trace = []
    for step in range(n_steps):
        loss = svi.step(r, X, spec, prior)      # một bước gradient; trả về -ELBO
        elbo_trace.append(-loss)
        if verbose and step % 200 == 0:
            print(f"[VB {guide_type}] buoc {step}: ELBO={-loss:.4f}")

    # Rút mẫu từ q đã khớp; chỉ giữ các site tham số (bỏ site "loglik").
    trace = Predictive(model, guide=guide, num_samples=n_posterior_draws)(r, X, spec, prior)
    samples = {name: trace[name].detach() for name in SITES if name in trace}
    return VBResult(
        guide=guide,
        elbo_trace=elbo_trace,
        final_elbo=float(sum(elbo_trace[-50:]) / min(50, len(elbo_trace))),
        samples=samples,
        summary=posterior_summary(samples),
        guide_type=guide_type,
    )
