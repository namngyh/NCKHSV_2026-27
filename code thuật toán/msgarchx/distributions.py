"""Phan phoi Student-t chuan hoa (eq:standardized-t-density trong bai bao).

z ~ t*_nu co E[z] = 0, Var[z] = 1 voi moi nu > 2. Day la phan phoi cua sai so
chuan hoa dung trong phuong trinh sai so cua MS-GARCH-X: eps_t = sqrt(h_t) * z_t.
"""
from __future__ import annotations

import math

import torch

_LOG_PI = math.log(math.pi)


def std_t_log_const(nu: torch.Tensor) -> torch.Tensor:
    """log[Gamma((nu+1)/2) / (Gamma(nu/2) sqrt(pi (nu-2)))]."""
    return (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu - 2.0)
        - 0.5 * _LOG_PI
    )


def std_t_logpdf(z: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Log-mat do cua t*_nu tai z, dung eq:standardized-t-density.

    Ho tro broadcasting: z co the la (T,) va nu la scalar hoac (T,)/(S,).
    """
    const = std_t_log_const(nu)
    return const - ((nu + 1.0) / 2.0) * torch.log1p(z * z / (nu - 2.0))


def normal_logpdf(z: torch.Tensor) -> torch.Tensor:
    """Log-mat do N(0,1), dung lam doi chung khi nu -> infty."""
    return -0.5 * (z * z) - 0.5 * math.log(2.0 * math.pi)


def std_t_sample(nu: torch.Tensor) -> torch.Tensor:
    """Sinh mot mau z ~ t*_nu (phuong sai bang 1) cho MOI phan tu cua `nu`.

    Output co cung shape voi `nu` (broadcast theo kieu elementwise). De sinh
    nhieu mau iid tu mot nu vo huong, truyen `nu.expand(shape)`. Dung phep
    bien doi Normal/Chi-binh-phuong chuan (T = Z / sqrt(V/df)) roi chuan hoa
    phuong sai ve 1 bang sqrt(nu/(nu-2)). Dung `torch.manual_seed` truoc khi
    goi ham de tai lap ket qua, thay vi truyen generator rieng.
    """
    normal = torch.randn(nu.shape)
    chi2 = torch.distributions.Chi2(nu).sample()
    raw = normal * torch.sqrt(nu / chi2)
    scale = torch.sqrt(nu / (nu - 2.0))
    return raw / scale
