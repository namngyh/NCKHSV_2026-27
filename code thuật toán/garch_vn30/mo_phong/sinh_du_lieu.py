"""Co che sinh du lieu (DGP) tu EGARCH-X mot trang thai.

Cung phuong trinh voi mo_hinh/egarch.py, nhung chay "xuoi": rut z_t roi tinh
h_t, r_t. Bien ngoai sinh sinh iid N(0,1) (mac dinh ar_coef = 0). Loai bo
`burn_in` quan sat dau de xoa anh huong cua dieu kien dau.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from garch_vn30.mo_hinh.egarch import LOG_H_MAX, LOG_H_MIN, EGARCHParams, expected_abs_z
from garch_vn30.mo_hinh.phan_phoi import std_t_sample


@dataclass
class DuLieuMoPhong:
    r: torch.Tensor   # (T,) loi suat
    h: torch.Tensor   # (T,) phuong sai co dieu kien that
    X: torch.Tensor   # (T, k) bien ngoai sinh


def generate_exogenous(total_T: int, n_exog: int, ar_coef: float = 0.0,
                       scale: float = 1.0) -> torch.Tensor:
    """X_t = ar_coef X_{t-1} + e_t, e_t ~ N(0, scale^2) (iid khi ar_coef = 0)."""
    X = torch.zeros(total_T, n_exog)
    eps = torch.randn(total_T, n_exog) * scale
    X[0] = eps[0]
    for t in range(1, total_T):
        X[t] = ar_coef * X[t - 1] + eps[t]
    return X


def simulate_egarch(params: EGARCHParams, T: int, dist: str = "student_t",
                    n_exog: int | None = None, X: torch.Tensor | None = None,
                    burn_in: int = 200, seed: int | None = None) -> DuLieuMoPhong:
    """Sinh T quan sat (sau burn_in) tu EGARCH-X.

    Dieu kien dau: log h_0 = omega / (1 - beta), trung binh vo dieu kien cua
    log-phuong sai khi E[X] = 0. Thu tu rut so ngau nhien co dinh (X truoc,
    z sau) de cung seed luon cho cung du lieu.
    """
    if seed is not None:
        torch.manual_seed(seed)
    total_T = T + burn_in
    if X is None:
        X = generate_exogenous(total_T, params.gamma.shape[0] if n_exog is None else n_exog)

    z_all = torch.randn(total_T) if dist == "normal" else std_t_sample(params.nu.expand(total_T))
    e_abs_z = expected_abs_z(None if dist == "normal" else params.nu)

    h_list = [torch.exp(torch.clamp(params.omega / (1.0 - params.beta), LOG_H_MIN, LOG_H_MAX))]
    r_list = [params.mu + torch.sqrt(h_list[0]) * z_all[0]]
    for t in range(1, total_T):
        z_prev = z_all[t - 1]
        log_h = (params.omega
                 + params.alpha * (torch.abs(z_prev) - e_abs_z)
                 + params.theta * z_prev
                 + params.beta * torch.log(h_list[t - 1])
                 + (params.gamma * X[t]).sum())
        h_list.append(torch.exp(torch.clamp(log_h, LOG_H_MIN, LOG_H_MAX)))
        r_list.append(params.mu + torch.sqrt(h_list[t]) * z_all[t])

    r, h = torch.stack(r_list), torch.stack(h_list)
    return DuLieuMoPhong(r=r[burn_in:], h=h[burn_in:], X=X[burn_in:])
