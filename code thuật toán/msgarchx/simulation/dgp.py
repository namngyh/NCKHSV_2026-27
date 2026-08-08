"""Co che sinh du lieu (DGP) cho nghien cuu mo phong (Muc 6.2).

Thuc hien dung 6 buoc trong bai bao: sinh S_1 tu phan phoi dung; sinh S_t
tu hang S_{t-1} cua P0; sinh bien ngoai sinh; cap nhat h_{t,S_t} THEO DUONG DI
THAT (khong can thu gon vi trang thai la da biet khi mo phong, khac voi bo
loc trong models/ms_garch.py); sinh z_t ~ t*_nu0 va r_t; roi loai bo giai
doan khoi dong (burn-in).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from msgarchx.distributions import std_t_sample
from msgarchx.models.ms_garch import MSGARCHXParams, stationary_distribution
from msgarchx.transforms import positive_variance_link


@dataclass
class SimulatedData:
    r: torch.Tensor       # (T,)
    h: torch.Tensor       # (T,) phuong sai thuc su tai trang thai da tham
    state: torch.Tensor   # (T,) long, trang thai that (0/1)
    X: torch.Tensor       # (T, k)


def generate_exogenous(total_T: int, n_exog: int, ar_coef: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """Sinh bien ngoai sinh iid hoac AR(1) don gian, dung khi khong tai su
    dung khoi du lieu that (xem thao luan sau eq:collapsed-residual, Muc 6.2 buoc 3)."""
    X = torch.zeros(total_T, n_exog)
    eps = torch.randn(total_T, n_exog) * scale
    X[0] = eps[0]
    for t in range(1, total_T):
        X[t] = ar_coef * X[t - 1] + eps[t]
    return X


def simulate_ms_garch_x(
    params: MSGARCHXParams,
    P0: torch.Tensor,
    T: int,
    n_exog: int | None = None,
    X: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    burn_in: int = 200,
    variance_link: str = "clamp",
    floor: float = 1e-8,
    seed: int | None = None,
) -> SimulatedData:
    if seed is not None:
        torch.manual_seed(seed)

    total_T = T + burn_in
    if X is None:
        if n_exog is None:
            n_exog = params.gamma.shape[1]
        X = generate_exogenous(total_T, n_exog)

    stationary = stationary_distribution(P0)
    state = torch.empty(total_T, dtype=torch.long)
    state[0] = torch.multinomial(stationary, 1)
    for t in range(1, total_T):
        state[t] = torch.multinomial(P0[state[t - 1]], 1)

    if h0 is None:
        denom = (1.0 - params.alpha - params.beta).clamp(min=0.05)
        h0 = params.omega / denom

    mu_path = params.mu[state]
    omega_path = params.omega[state]
    alpha_path = params.alpha[state]
    beta_path = params.beta[state]
    gamma_path = params.gamma[state]
    nu_path = params.nu[state] if params.state_specific_nu else params.nu.expand(total_T)
    z_all = std_t_sample(nu_path)

    h_list = [h0[state[0]]]
    r_list = [mu_path[0] + torch.sqrt(h_list[0]) * z_all[0]]
    for t in range(1, total_T):
        eps_prev = r_list[t - 1] - mu_path[t - 1]
        h_raw = (
            omega_path[t]
            + alpha_path[t] * eps_prev**2
            + beta_path[t] * h_list[t - 1]
            + (gamma_path[t] * X[t]).sum()
        )
        h_t = positive_variance_link(h_raw, floor=floor, mode=variance_link)
        h_list.append(h_t)
        r_list.append(mu_path[t] + torch.sqrt(h_t) * z_all[t])

    r = torch.stack(r_list)
    h = torch.stack(h_list)

    return SimulatedData(
        r=r[burn_in:], h=h[burn_in:], state=state[burn_in:], X=X[burn_in:],
    )
