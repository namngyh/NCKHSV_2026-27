"""Tom tat hau nghiem (mean, sd, KTC 95%) va so sanh Variational Bayes voi MCMC.

Dung chung cho variational.py va mcmc.py de dien vao Bang so sanh uoc luong
tham so (tab:parameter-template) va ty so do lech chuan R_sd,j (eq:sd-ratio).
"""
from __future__ import annotations

import torch


def posterior_summary(samples: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """samples[name] co shape (n_draws, *event_shape) -> {name: {mean, sd, q2.5, q97.5}}."""
    summary = {}
    for name, draws in samples.items():
        summary[name] = {
            "mean": draws.mean(dim=0),
            "sd": draws.std(dim=0),
            "q2.5": draws.quantile(0.025, dim=0),
            "q97.5": draws.quantile(0.975, dim=0),
        }
    return summary


def sd_ratio(
    summary_q: dict[str, dict[str, torch.Tensor]],
    summary_mcmc: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """R_sd,j = sd_q(theta_j) / sd_MCMC(theta_j), eq:sd-ratio.

    Gia tri < 1 cho thay Variational Bayes danh gia thap phuong sai hau nghiem
    so voi MCMC (dung lam chuan tham chieu chi sau khi hoi tu dat yeu cau).
    """
    ratios = {}
    for name in summary_q:
        if name not in summary_mcmc:
            continue
        ratios[name] = summary_q[name]["sd"] / summary_mcmc[name]["sd"]
    return ratios
