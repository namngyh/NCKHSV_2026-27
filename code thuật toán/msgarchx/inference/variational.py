"""Variational Bayes (Muc 4.4): mean-field va Gaussian day du tren khong gian
tham so khong rang buoc, toi da hoa ELBO (eq:elbo).

AutoNormal cua Pyro dat q(theta_tilde) = prod_j N(m_j, s_j^2) tren khong gian
da bien doi -- dung chinh xac cau truc mean-field trong eq:mean-field.
AutoMultivariateNormal dat q(theta_tilde) = N(m, L L^T) -- dung eq:full-rank-vi.
Ca hai deu goi lai `bayes_model.model`, tuc la cung mot ham hop ly va cung mot
co che thu gon nhu MLE va MCMC.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoMultivariateNormal, AutoNormal

from msgarchx.inference.bayes_model import model as bayes_model
from msgarchx.inference.param_spec import ModelConfig, PriorConfig
from msgarchx.inference.summary import posterior_summary

_GUIDE_SITES = ["mu", "omega", "alpha", "beta", "gamma", "p11", "p22", "nu_minus_2"]


@dataclass
class VBResult:
    guide: object
    elbo_trace: list[float]
    final_elbo: float
    samples: dict[str, torch.Tensor]
    summary: dict[str, dict[str, torch.Tensor]]
    config: ModelConfig
    guide_type: str


def fit_vb(
    r: torch.Tensor,
    X: torch.Tensor,
    config: ModelConfig,
    guide_type: str = "mean_field",
    prior: PriorConfig | None = None,
    n_steps: int = 3000,
    lr: float = 0.01,
    num_particles: int = 1,
    n_posterior_draws: int = 2000,
    seed: int = 0,
    variance_link: str = "clamp",
    floor: float = 1e-8,
    verbose: bool = False,
) -> VBResult:
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    if guide_type == "mean_field":
        guide = AutoNormal(bayes_model)
    elif guide_type == "full_rank":
        guide = AutoMultivariateNormal(bayes_model)
    else:
        raise ValueError(f"guide_type khong hop le: {guide_type!r}")

    optim = pyro.optim.Adam({"lr": lr})
    svi = SVI(bayes_model, guide, optim, loss=Trace_ELBO(num_particles=num_particles))

    elbo_trace = []
    for step in range(n_steps):
        loss = svi.step(r, X, config, prior, variance_link, floor)
        elbo_trace.append(-loss)
        if verbose and step % 200 == 0:
            print(f"[VB {guide_type}] step {step}: ELBO={-loss:.4f}")

    predictive = Predictive(bayes_model, guide=guide, num_samples=n_posterior_draws)
    trace = predictive(r, X, config, prior, variance_link, floor)
    samples = {name: trace[name].detach() for name in _GUIDE_SITES if name in trace}

    summary = posterior_summary(samples)
    final_elbo = float(sum(elbo_trace[-50:]) / min(50, len(elbo_trace)))

    return VBResult(
        guide=guide,
        elbo_trace=elbo_trace,
        final_elbo=final_elbo,
        samples=samples,
        summary=summary,
        config=config,
        guide_type=guide_type,
    )
