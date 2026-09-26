"""Mo hinh Bayes (Pyro) cho GARCH/EGARCH mot trang thai va tom tat hau nghiem.

Tien nghiem dat TRUC TIEP tren mien hop le cua tung tham so; Pyro tu lo phep
bien doi sang khong gian khong rang buoc cho VB va NUTS. Log-hop-ly goi DUNG
ham garch_loglik / egarch_loglik ma MLE dung, qua pyro.factor, nen ba phuong
phap toi uu/lay mau tren cung mot ham muc tieu:

    p(theta | r) ∝ exp(l(theta)) * p(theta).

Ten "site" trong Pyro: mu, gamma, nu_minus_2 (= nu - 2), omega, alpha, theta
(chi EGARCH), beta.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import torch

from garch_vn30.mo_hinh.danh_muc import ModelSpec, loglik_cua
from garch_vn30.mo_hinh.egarch import EGARCHParams
from garch_vn30.mo_hinh.garch import GARCHParams

SITES = ["mu", "omega", "alpha", "theta", "beta", "gamma", "nu_minus_2"]


@dataclass(frozen=True)
class PriorConfig:
    """Sieu tham so cua tien nghiem (yeu, gan nhu khong thong tin)."""
    sigma_mu: float = 1.0                  # mu ~ N(0, 1)
    m_omega: float = 0.0                   # GARCH: omega ~ LogNormal(0, 1)
    sigma_omega: float = 1.0               # EGARCH: omega ~ N(0, 1)
    alpha_beta_concentration: float = 2.0  # alpha, beta ~ Beta(2, 2) tren (0, 1)
    sigma_gamma: float = 1.0               # gamma_j ~ N(0, 1)
    nu_shape: float = 2.0                  # nu - 2 ~ Gamma(2; 0,2), trung binh nu = 12
    nu_rate: float = 0.2
    # theta ~ N(0, 1) DOI XUNG quanh 0: tien nghiem khong gia dinh truoc dau
    # don bay, nen "du lieu co ung ho don bay khong" van la cau hoi mo.
    sigma_theta: float = 1.0


def model(r: torch.Tensor, X: torch.Tensor, spec: ModelSpec,
          prior: PriorConfig | None = None):
    """Mo hinh sinh: rut tham so tu tien nghiem, cong log-hop-ly vao mat do.

    Tra ve (params, loglik) de tien kiem tra; phan dong gop vao hau nghiem
    nam o pyro.factor.
    """
    prior = prior or PriorConfig()
    k = X.shape[1]
    a = prior.alpha_beta_concentration

    mu = pyro.sample("mu", dist.Normal(0.0, prior.sigma_mu))
    if spec.use_exog and k > 0:
        gamma = pyro.sample("gamma", dist.Normal(0.0, prior.sigma_gamma).expand([k]).to_event(1))
    else:
        gamma = torch.zeros(0)
    nu = None
    if spec.dist == "student_t":
        nu = pyro.sample("nu_minus_2", dist.Gamma(prior.nu_shape, prior.nu_rate)) + 2.0

    if spec.family == "garch":
        omega = pyro.sample("omega", dist.LogNormal(prior.m_omega, prior.sigma_omega))
        alpha = pyro.sample("alpha", dist.Beta(a, a))
        beta = pyro.sample("beta", dist.Beta(a, a))
        params = GARCHParams(mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu)
    elif spec.family == "egarch":
        # log h khong can omega duong -> tien nghiem Normal.
        omega = pyro.sample("omega", dist.Normal(0.0, prior.sigma_omega))
        alpha = pyro.sample("alpha", dist.Beta(a, a))
        theta = pyro.sample("theta", dist.Normal(0.0, prior.sigma_theta))
        beta = pyro.sample("beta", dist.Beta(a, a))
        params = EGARCHParams(mu=mu, omega=omega, alpha=alpha, theta=theta, beta=beta,
                              gamma=gamma, nu=nu)
    else:
        raise ValueError(f"family khong hop le: {spec.family!r}")

    loglik = loglik_cua(spec.family)(params, r, X)
    pyro.factor("loglik", loglik)
    return params, loglik


def params_tu_mau(samples: dict[str, torch.Tensor], spec: ModelSpec, i: int):
    """Bo tham so thu i trong tap mau hau nghiem (VB hoac MCMC)."""
    gamma = samples["gamma"][i] if "gamma" in samples else torch.zeros(0)
    nu = samples["nu_minus_2"][i] + 2.0 if spec.dist == "student_t" else None
    if spec.family == "garch":
        return GARCHParams(mu=samples["mu"][i], omega=samples["omega"][i],
                           alpha=samples["alpha"][i], beta=samples["beta"][i],
                           gamma=gamma, nu=nu)
    return EGARCHParams(mu=samples["mu"][i], omega=samples["omega"][i],
                        alpha=samples["alpha"][i], theta=samples["theta"][i],
                        beta=samples["beta"][i], gamma=gamma, nu=nu)


def rut_bo_tham_so(samples: dict[str, torch.Tensor], spec: ModelSpec, n_draws: int) -> list:
    """Lay ~n_draws bo tham so cach deu nhau tu tap mau hau nghiem."""
    tong = samples["mu"].shape[0]
    buoc = max(1, tong // n_draws)
    return [params_tu_mau(samples, spec, i) for i in range(0, tong, buoc)][:n_draws]


def posterior_summary(samples: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """{ten: {mean, sd, q2.5, q97.5}} cho moi site (bo qua site rong)."""
    summary = {}
    for name, draws in samples.items():
        if draws.numel() == 0:
            continue
        summary[name] = {
            "mean": draws.mean(dim=0),
            "sd": draws.std(dim=0),
            "q2.5": draws.quantile(0.025, dim=0),
            "q97.5": draws.quantile(0.975, dim=0),
        }
    return summary


def sd_ratio(summary_vb: dict, summary_mcmc: dict) -> dict[str, torch.Tensor]:
    """R_sd,j = sd_VB(theta_j) / sd_MCMC(theta_j). R_sd < 1: VB danh gia thap bat dinh."""
    return {name: summary_vb[name]["sd"] / summary_mcmc[name]["sd"]
            for name in summary_vb if name in summary_mcmc}
