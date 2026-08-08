"""GARCH(1,1) va GARCH-X mot trang thai (mo hinh doi chung B1-B3 trong Bang 5.2).

Quy uoc canh chinh du lieu: X[t] la vector ngoai sinh da biet truoc thoi diem
du bao r[t] -- tuc la X_{t-1} trong ky hieu bai bao, da duoc dich chi so de
X[t] cung hang voi r[t] trong mang du lieu. Xem eq:garchx.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from msgarchx.distributions import normal_logpdf, std_t_logpdf
from msgarchx.transforms import positive_variance_link


class GARCHParams(NamedTuple):
    mu: torch.Tensor       # scalar
    omega: torch.Tensor    # scalar > 0
    alpha: torch.Tensor    # scalar >= 0
    beta: torch.Tensor     # scalar >= 0
    gamma: torch.Tensor    # (k,)
    nu: Optional[torch.Tensor] = None  # scalar > 2, None neu phan phoi chuan


def garch_forward(
    params: GARCHParams,
    r: torch.Tensor,
    X: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    variance_link: str = "clamp",
    floor: float = 1e-8,
) -> torch.Tensor:
    """Tinh duong phuong sai co dieu kien h_t theo eq:garchx (h[0] la dieu kien dau)."""
    T = r.shape[0]
    if h0 is None:
        h0 = torch.var(r).detach()
    eps = r - params.mu
    h_list = [h0]
    for t in range(1, T):
        h_raw = (
            params.omega
            + params.alpha * eps[t - 1] ** 2
            + params.beta * h_list[t - 1]
            + X[t] @ params.gamma
        )
        h_list.append(positive_variance_link(h_raw, floor=floor, mode=variance_link))
    return torch.stack(h_list)


def garch_loglik(
    params: GARCHParams,
    r: torch.Tensor,
    X: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    variance_link: str = "clamp",
    floor: float = 1e-8,
) -> torch.Tensor:
    """Tong log-hop-ly tu t=1 (t=0 dung lam dieu kien dau, khong tinh vao likelihood)."""
    h = garch_forward(params, r, X, h0=h0, variance_link=variance_link, floor=floor)
    eps = r - params.mu
    z = eps / torch.sqrt(h)
    if params.nu is None:
        log_f = normal_logpdf(z) - 0.5 * torch.log(h)
    else:
        log_f = std_t_logpdf(z, params.nu) - 0.5 * torch.log(h)
    return log_f[1:].sum()
