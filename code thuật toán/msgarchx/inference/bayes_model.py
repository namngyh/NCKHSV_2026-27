"""Mo hinh Pyro dung chung cho Variational Bayes va MCMC.

Day la diem mau chot bao dam yeu cau "cung ham hop ly, cung co che thu gon"
(checklist Phu luc B, muc 7): ca variational.py va mcmc.py deu goi ham `model`
duy nhat o day, va ham nay goi thang `ms_garch_forward_filter` -- CUNG mot ham
duoc dung boi mle.py. Prior duoc dat truc tiep tren mien tham so hop le
(LogNormal, Beta, Gamma...) de Pyro tu dong xu ly phep bien doi sang khong
gian khong rang buoc khi chay NUTS/SVI, thay vi phai tu tay lam nhu o mle.py.

Tham chieu prior: eq:prior-mu, eq:prior-omega, eq:prior-gamma,
eq:prior-transition, eq:prior-df.
"""
from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch

from msgarchx.inference.param_spec import ModelConfig, PriorConfig
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter


def model(
    r: torch.Tensor,
    X: torch.Tensor,
    config: ModelConfig,
    prior: PriorConfig | None = None,
    variance_link: str = "clamp",
    floor: float = 1e-8,
    fixed_nu: float | None = None,
):
    """Mo hinh xac suat chung cho VB va MCMC.

    `fixed_nu`: neu duoc dat, nu KHONG duoc lay mau ma giu co dinh o gia tri
    nay. Dung cho MS-GARCH CHUAN: Student-t chuan hoa voi nu rat lon hoi tu ve
    phan phoi chuan, nen ta cai mo hinh chuan bang chinh khung MS-GARCH-X thay
    vi viet mot mo hinh rieng -- dung cach MLE dang lam cho mo hinh nay, nho do
    hai phuong phap uoc luong tren CUNG MOT ham likelihood (nguyen tac cot loi
    cua bai). nu co dinh thi khong tinh vao so tham so khi tinh AIC/BIC.
    Mac dinh None giu nguyen hanh vi cu.
    """
    prior = prior or PriorConfig()
    k = config.n_exog

    mu = pyro.sample("mu", dist.Normal(0.0, prior.sigma_mu).expand([2]).to_event(1))
    if prior.omega_prior == "half_cauchy":
        omega_dist = dist.HalfCauchy(prior.omega_hc_scale)
    elif prior.omega_prior == "lognormal":
        omega_dist = dist.LogNormal(prior.m_omega, prior.sigma_omega)
    else:
        raise ValueError(f"omega_prior khong hop le: {prior.omega_prior!r}")
    omega = pyro.sample("omega", omega_dist.expand([2]).to_event(1))
    a = prior.alpha_beta_concentration
    alpha = pyro.sample("alpha", dist.Beta(a, a).expand([2]).to_event(1))
    beta = pyro.sample("beta", dist.Beta(a, a).expand([2]).to_event(1))
    gamma = pyro.sample(
        "gamma", dist.Normal(0.0, prior.sigma_gamma).expand([2, k]).to_event(2)
    )

    stay, switch = prior.transition_prior_stay, prior.transition_prior_switch
    p11 = pyro.sample("p11", dist.Beta(stay, switch))
    p22 = pyro.sample("p22", dist.Beta(stay, switch))
    P = torch.stack(
        [
            torch.stack([p11, 1.0 - p11]),
            torch.stack([1.0 - p22, p22]),
        ]
    )

    if fixed_nu is not None:
        nu = torch.as_tensor(float(fixed_nu))
    elif config.state_specific_nu:
        nu_minus_2 = pyro.sample(
            "nu_minus_2",
            dist.Gamma(prior.nu_shape, prior.nu_rate).expand([2]).to_event(1),
        )
        nu = nu_minus_2 + 2.0
    else:
        nu_minus_2 = pyro.sample("nu_minus_2", dist.Gamma(prior.nu_shape, prior.nu_rate))
        nu = nu_minus_2 + 2.0

    params = MSGARCHXParams(mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma, P=P, nu=nu)
    out = ms_garch_forward_filter(params, r, X, variance_link=variance_link, floor=floor)
    pyro.factor("loglik", out["loglik"])
    return params, out


def params_from_trace(trace_dict: dict, config: ModelConfig) -> MSGARCHXParams:
    """Ghep cac site sample cua Pyro (1 mau) thanh MSGARCHXParams."""
    p11, p22 = trace_dict["p11"], trace_dict["p22"]
    P = torch.stack(
        [
            torch.stack([p11, 1.0 - p11]),
            torch.stack([1.0 - p22, p22]),
        ]
    )
    nu = trace_dict["nu_minus_2"] + 2.0
    return MSGARCHXParams(
        mu=trace_dict["mu"],
        omega=trace_dict["omega"],
        alpha=trace_dict["alpha"],
        beta=trace_dict["beta"],
        gamma=trace_dict["gamma"],
        P=P,
        nu=nu,
    )
