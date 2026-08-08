"""Danh gia du bao mat do ngoai mau (Muc 6.8): diem log am, CRPS, PIT.

eq:negative-log-score, eq:crps, eq:pit, eq:mixture-predictive-density.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import optimize, stats

from msgarchx.distributions import std_t_logpdf


def mixture_log_density(
    r: torch.Tensor,
    mu: torch.Tensor,
    h: torch.Tensor,
    weights: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """log f(r | Info) = logsumexp_s [log w_s + log f_s(r)], eq:mixture-predictive-density.

    mu, h, weights: (2,) tai mot thoi diem; nu: scalar hoac (2,).
    """
    z = (r - mu) / torch.sqrt(h)
    if nu.dim() > 0 and nu.shape[0] == 2:
        log_f_s = torch.stack([std_t_logpdf(z[s], nu[s]) for s in range(2)])
    else:
        log_f_s = std_t_logpdf(z, nu)
    log_f_s = log_f_s - 0.5 * torch.log(h)
    return torch.logsumexp(torch.log(weights) + log_f_s, dim=0)


def negative_log_score(log_densities: torch.Tensor) -> torch.Tensor:
    """NLS = -mean_t log f_t(r_t), eq:negative-log-score. Nho hon la tot hon."""
    return -log_densities.mean()


def crps_from_samples(samples: torch.Tensor, r_obs: float) -> float:
    """Uoc luong CRPS bang cong thuc energy score tu mau Monte Carlo:

    CRPS(F, r) ~= E|X - r| - 0.5 E|X - X'|, voi X, X' la hai mau doc lap tu F.
    Day la uoc luong khong lech cua eq:crps (Gneiting & Raftery, 2007).
    """
    x = samples.detach().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
    term1 = np.mean(np.abs(x - r_obs))
    n = len(x)
    x_sorted = np.sort(x)
    idx = np.arange(1, n + 1)
    term2 = 2.0 * np.sum((2 * idx - n - 1) * x_sorted) / (n * n)
    return float(term1 - 0.5 * term2)


def pit_from_samples(samples: torch.Tensor, r_obs: float) -> float:
    """u_t = F_hat_t(r_t) uoc luong bang phan bo thuc nghiem cua mau du bao, eq:pit."""
    x = samples.detach().numpy() if isinstance(samples, torch.Tensor) else np.asarray(samples)
    return float(np.mean(x <= r_obs))


def mixture_cdf(
    r: float,
    mu: np.ndarray,
    h: np.ndarray,
    weights: np.ndarray,
    nu: np.ndarray | float,
) -> float:
    """Dang dong cua F_t(r) khi co cong thuc CDF Student-t chuan hoa (dung
    khi chi co mot bo tham so diem, vd. du bao cam-tham-so cua MLE).

    Student-t chuan hoa co scale = sqrt((nu-2)/nu) so voi Student-t chuan
    cua scipy (df=nu, loc=0, scale=1).
    """
    mu = np.atleast_1d(mu)
    h = np.atleast_1d(h)
    weights = np.atleast_1d(weights)
    nu_arr = np.broadcast_to(np.atleast_1d(nu), mu.shape)
    cdf = 0.0
    for s in range(len(mu)):
        scale = np.sqrt(h[s]) * np.sqrt((nu_arr[s] - 2.0) / nu_arr[s])
        cdf += weights[s] * stats.t.cdf(r, df=nu_arr[s], loc=mu[s], scale=scale)
    return float(cdf)


def mixture_quantile(
    alpha: float,
    mu: np.ndarray,
    h: np.ndarray,
    weights: np.ndarray,
    nu: np.ndarray | float,
) -> float:
    """q_alpha sao cho mixture_cdf(q_alpha, ...) = alpha, dung tim bang
    brentq tren F(x)-alpha=0 (F don dieu tang). Dung cho eq:predictive-quantile
    khi mat do du bao la hon hop nhieu trang thai (MS-GARCH-X)."""
    mu = np.atleast_1d(mu)
    h = np.atleast_1d(h)
    lo = float(mu.min() - 20.0 * np.sqrt(h.max()))
    hi = float(mu.max() + 20.0 * np.sqrt(h.max()))
    f = lambda x: mixture_cdf(x, mu, h, weights, nu) - alpha
    return float(optimize.brentq(f, lo, hi, xtol=1e-6))
