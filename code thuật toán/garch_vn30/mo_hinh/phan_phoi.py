"""Phan phoi cua sai so chuan hoa z_t: chuan N(0,1) va Student-t chuan hoa t*_nu.

t*_nu co E[z] = 0 va Var[z] = 1 voi moi nu > 2, nen trong ca GARCH lan EGARCH
phuong sai co dieu kien cua r_t luon la h_t, bat ke phan phoi nao.
"""
from __future__ import annotations

import math

import torch

_LOG_PI = math.log(math.pi)


def std_t_log_const(nu: torch.Tensor) -> torch.Tensor:
    """Hang so chuan hoa log[Gamma((nu+1)/2) / (Gamma(nu/2) sqrt(pi (nu-2)))]."""
    return (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu - 2.0)
        - 0.5 * _LOG_PI
    )


def std_t_logpdf(z: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """log f(z | nu) cua t*_nu:  const - (nu+1)/2 * log(1 + z^2/(nu-2))."""
    return std_t_log_const(nu) - ((nu + 1.0) / 2.0) * torch.log1p(z * z / (nu - 2.0))


def normal_logpdf(z: torch.Tensor) -> torch.Tensor:
    """log-mat do N(0,1): -z^2/2 - log(2 pi)/2."""
    return -0.5 * (z * z) - 0.5 * math.log(2.0 * math.pi)


def expected_abs_normal() -> float:
    """E|z| voi z ~ N(0,1), bang sqrt(2/pi) ~ 0,7979."""
    return math.sqrt(2.0 / math.pi)


def expected_abs_std_t(nu: torch.Tensor) -> torch.Tensor:
    """E|z| voi z ~ t*_nu:

        E|z| = 2 sqrt(nu-2) Gamma((nu+1)/2) / [ (nu-1) sqrt(pi) Gamma(nu/2) ].

    Tinh qua lgamma de khong tran so khi nu lon; khi nu -> vo cung gia tri
    hoi tu ve sqrt(2/pi) (co test rieng).
    """
    log_ratio = torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
    log_val = (
        math.log(2.0)
        + 0.5 * torch.log(nu - 2.0)
        + log_ratio
        - torch.log(nu - 1.0)
        - 0.5 * _LOG_PI
    )
    return torch.exp(log_val)


def std_t_sample(nu: torch.Tensor) -> torch.Tensor:
    """Sinh z ~ t*_nu cho MOI phan tu cua `nu` (cung shape voi `nu`).

    T = Z / sqrt(V/nu) voi Z ~ N(0,1), V ~ chi2(nu), roi chia cho
    sqrt(nu/(nu-2)) de phuong sai bang 1. Muon n mau tu mot nu vo huong thi
    truyen nu.expand(n). Dat torch.manual_seed truoc de tai lap.
    """
    normal = torch.randn(nu.shape)
    chi2 = torch.distributions.Chi2(nu).sample()
    raw = normal * torch.sqrt(nu / chi2)
    return raw / torch.sqrt(nu / (nu - 2.0))
