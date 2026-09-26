"""Variational Bayes: xap xi hau nghiem bang mot phan phoi Gauss tren khong
gian tham so da bien doi ve R^d, toi da hoa ELBO bang gradient ngau nhien.

    ELBO(q) = E_q[ log p(r, theta) ] - E_q[ log q(theta) ]  <=  log p(r)

  mean_field: q = prod_j N(m_j, s_j^2)       (AutoNormal, bo qua tuong quan)
  full_rank : q = N(m, L L^T)                (AutoMultivariateNormal, L tam giac duoi)

Gradient uoc luong bang meo tham so hoa lai (theta = m + L eps, eps ~ N(0, I))
qua Trace_ELBO cua Pyro; toi uu bang Adam voi toc do hoc co dinh.
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
    guide: object
    elbo_trace: list[float]
    final_elbo: float                          # trung binh 50 buoc cuoi
    samples: dict[str, torch.Tensor]           # mau rut tu q
    summary: dict[str, dict[str, torch.Tensor]]
    guide_type: str


def fit_vb(r: torch.Tensor, X: torch.Tensor, spec: ModelSpec, guide_type: str = "mean_field",
           prior: PriorConfig | None = None, n_steps: int = 3000, lr: float = 0.01,
           num_particles: int = 1, n_posterior_draws: int = 2000, seed: int = 0,
           verbose: bool = False) -> VBResult:
    """Chay SVI `n_steps` buoc roi rut `n_posterior_draws` mau tu q da khop."""
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
        loss = svi.step(r, X, spec, prior)
        elbo_trace.append(-loss)
        if verbose and step % 200 == 0:
            print(f"[VB {guide_type}] buoc {step}: ELBO={-loss:.4f}")

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
