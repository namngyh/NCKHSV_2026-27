"""EGARCH(1,1) va EGARCH-X mot trang thai (Nelson, 1991).

    log h_t = omega + alpha (|z_{t-1}| - E|z|) + theta z_{t-1}
              + beta log h_{t-1} + gamma^T X_{t-1},      z_{t-1} = eps_{t-1}/sqrt(h_{t-1})

  alpha: hieu ung DO LON (bien do cu soc, da tru ky vong E|z|).
  theta: hieu ung DAU (don bay); theta < 0 thi cu soc am tang bien dong manh hon.
  beta : do dai dang cua log-phuong sai; can |beta| < 1 de qua trinh dung.

Vi viet cho log h nen h_t = exp(...) > 0 theo cau truc, khong can rang buoc
dau cho omega, alpha, theta, gamma. Doi lai z_{t-1} phu thuoc h_{t-1} nen phai
tinh trong vong lap.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch

from garch_vn30.mo_hinh.garch import log_mat_do
from garch_vn30.mo_hinh.phan_phoi import expected_abs_normal, expected_abs_std_t

# Luoi an toan so hoc de exp() khong tran so; KHONG phai rang buoc mo hinh.
# Loi suat theo phan tram cho log h thuc te quanh 0-1, xa hai bien nay.
LOG_H_MIN = -30.0
LOG_H_MAX = 30.0


class EGARCHParams(NamedTuple):
    mu: torch.Tensor       # vo huong
    omega: torch.Tensor    # vo huong, tu do
    alpha: torch.Tensor    # vo huong, hieu ung do lon
    theta: torch.Tensor    # vo huong, hieu ung don bay
    beta: torch.Tensor     # vo huong, trong (0, 1) theo quy uoc cua de tai
    gamma: torch.Tensor    # (k,)
    nu: Optional[torch.Tensor] = None  # > 2; None = phan phoi chuan


def expected_abs_z(nu: Optional[torch.Tensor]) -> Union[torch.Tensor, float]:
    """E|z| cua phan phoi sai so: sqrt(2/pi) neu chuan, cong thuc t*_nu neu Student-t."""
    if nu is None:
        return expected_abs_normal()
    return expected_abs_std_t(nu)


def egarch_forward(params: EGARCHParams, r: torch.Tensor, X: torch.Tensor,
                   h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Duong phuong sai h_0, ..., h_{T-1} (tra ve h, khong phai log h)."""
    T = r.shape[0]
    if h0 is None:
        h0 = torch.var(r).detach()
    eps = r - params.mu
    gX = X @ params.gamma
    e_abs_z = expected_abs_z(params.nu)

    h_list = [h0]
    h_prev = h0
    for t in range(1, T):
        z_prev = eps[t - 1] / torch.sqrt(h_prev)
        log_h = (
            params.omega
            + params.alpha * (torch.abs(z_prev) - e_abs_z)
            + params.theta * z_prev
            + params.beta * torch.log(h_prev)
            + gX[t]
        )
        h_prev = torch.exp(torch.clamp(log_h, LOG_H_MIN, LOG_H_MAX))
        h_list.append(h_prev)
    return torch.stack(h_list)


def egarch_loglik(params: EGARCHParams, r: torch.Tensor, X: torch.Tensor,
                  h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """log-hop-ly, cung dang voi garch_loglik; chi khac cach tinh h."""
    h = egarch_forward(params, r, X, h0=h0)
    return log_mat_do(r, params.mu, h, params.nu)[1:].sum()
