"""Mo hinh Pyro cho cac mo hinh MOT trang thai (GARCH va EGARCH).

File nay la ban song song cua bayes_model.py -- von chi phuc vu MS-GARCH-X hai
trang thai. Cung nguyen tac: dat prior TRUC TIEP tren mien tham so hop le de
Pyro tu lo phep bien doi sang khong gian khong rang buoc, va goi thang cung
mot ham likelihood ma mle.py/garch_mle.py dung (garch_loglik / egarch_loglik),
de VB, MCMC va MLE deu toi uu tren DUNG mot ham muc tieu.

Cach dung (khong can functools.partial -- truyen thang spec qua model_args):

    from msgarchx.inference.bayes_single import model as single_model
    res = fit_vb(r, X, config=None, guide_type="mean_field",
                 model_fn=single_model, model_args=(spec, prior))

Thu tu trong model_args phai khop dung thu tu tham so cua `model` sau (r, X),
tuc la (spec, prior).
"""
from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch

from msgarchx.inference.model_spec import ModelSpec
from msgarchx.inference.param_spec import PriorConfig
from msgarchx.models.egarch import EGARCHParams, egarch_loglik
from msgarchx.models.garch import GARCHParams, garch_loglik


def _sample_gamma(spec: ModelSpec, k: int, prior: PriorConfig) -> torch.Tensor:
    """He so ngoai sinh; mo hinh khong dung X thi tra ve vector rong."""
    if not spec.use_exog or k == 0:
        return torch.zeros(0)
    return pyro.sample("gamma", dist.Normal(0.0, prior.sigma_gamma).expand([k]).to_event(1))


def _sample_nu(spec: ModelSpec, prior: PriorConfig):
    """Bac tu do; None neu sai so phan phoi chuan (eq:prior-df)."""
    if spec.dist != "student_t":
        return None
    nu_minus_2 = pyro.sample("nu_minus_2", dist.Gamma(prior.nu_shape, prior.nu_rate))
    return nu_minus_2 + 2.0


def model(
    r: torch.Tensor,
    X: torch.Tensor,
    spec: ModelSpec,
    prior: PriorConfig | None = None,
):
    """Mo hinh sinh cho mot chuoi loi suat, theo dac ta trong `spec`.

    Tra ve (params, loglik) de tien kiem tra; phan dong gop vao hau nghiem
    duoc ghi qua pyro.factor giong bayes_model.py.
    """
    prior = prior or PriorConfig()
    k = X.shape[1]
    a = prior.alpha_beta_concentration

    mu = pyro.sample("mu", dist.Normal(0.0, prior.sigma_mu))
    gamma = _sample_gamma(spec, k, prior)
    nu = _sample_nu(spec, prior)

    if spec.family == "garch":
        # Giong bayes_model.py: omega > 0 (LogNormal), alpha/beta tren (0,1).
        omega = pyro.sample("omega", dist.LogNormal(prior.m_omega, prior.sigma_omega))
        alpha = pyro.sample("alpha", dist.Beta(a, a))
        beta = pyro.sample("beta", dist.Beta(a, a))
        params = GARCHParams(mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu)
        loglik = garch_loglik(params, r, X)
    elif spec.family == "egarch":
        # EGARCH viet cho log h nen omega KHONG can duong -> prior Normal.
        # alpha (do lon) va beta (dai dang) van dat tren (0,1) de dung cung ho
        # prior voi GARCH, giu phep so sanh giua hai ho cong bang.
        # theta doi xung quanh 0: xem ghi chu sigma_theta trong PriorConfig.
        omega = pyro.sample("omega", dist.Normal(0.0, prior.sigma_omega))
        alpha = pyro.sample("alpha", dist.Beta(a, a))
        theta = pyro.sample("theta", dist.Normal(0.0, prior.sigma_theta))
        beta = pyro.sample("beta", dist.Beta(a, a))
        params = EGARCHParams(
            mu=mu, omega=omega, alpha=alpha, theta=theta, beta=beta, gamma=gamma, nu=nu
        )
        loglik = egarch_loglik(params, r, X)
    else:
        raise ValueError(f"family khong hop le: {spec.family!r}")

    pyro.factor("loglik", loglik)
    return params, loglik


def params_from_trace(trace_dict: dict, spec: ModelSpec, k: int):
    """Ghep cac site sample cua Pyro (1 mau) thanh Params tuong ung `spec`."""
    gamma = trace_dict.get("gamma", torch.zeros(0))
    nu = trace_dict["nu_minus_2"] + 2.0 if spec.dist == "student_t" else None
    if spec.family == "garch":
        return GARCHParams(
            mu=trace_dict["mu"],
            omega=trace_dict["omega"],
            alpha=trace_dict["alpha"],
            beta=trace_dict["beta"],
            gamma=gamma,
            nu=nu,
        )
    return EGARCHParams(
        mu=trace_dict["mu"],
        omega=trace_dict["omega"],
        alpha=trace_dict["alpha"],
        theta=trace_dict["theta"],
        beta=trace_dict["beta"],
        gamma=gamma,
        nu=nu,
    )
