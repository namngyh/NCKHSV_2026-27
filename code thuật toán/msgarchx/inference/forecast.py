"""Du bao hau nghiem va mo phong nhieu buoc (Muc 4.6).

Voi mot tham so diem (MLE) hoac nhieu mau hau nghiem (VB/MCMC), ham
`simulate_forecast_paths` mo phong dong thoi bon thanh phan duoc bai bao yeu
cau: trang thai tuong lai theo P, sai so Student-t chuan hoa, phuong sai co
dieu kien de quy, va bien ngoai sinh tuong lai (truyen vao qua X_future).

Du bao Bayes (eq:posterior-predictive, eq:mcmc-predictive) duoc xap xi bang
cach gop mo phong tren nhieu mau tham so hau nghiem trong
`posterior_predictive_paths`.
"""
from __future__ import annotations

import torch

from msgarchx.distributions import std_t_sample
from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.transforms import positive_variance_link


def predictive_variance_path(params: MSGARCHXParams, xi_pred: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Phuong sai du bao MOT BUOC h_hat_t = Var(r_t | Info_{t-1}), dung
    xi_pred (xac suat DU BAO, tinh TRUOC khi quan sat r_t) chu khong dung
    xi_filt (xac suat LOC, tinh SAU khi quan sat r_t). Dung xi_filt o day se
    "ro ri" thong tin cua chinh r_t vao du bao cua no, lam sai lech danh gia
    QLIKE/MAE/RMSE ngoai mau.

    xi_pred, h: dau ra tu ms_garch_forward_filter (shape (T, 2)).
    """
    mu = params.mu
    bar_mu = (xi_pred * mu).sum(dim=1)
    bar_h = (xi_pred * (h + mu**2)).sum(dim=1) - bar_mu**2
    return bar_h


def simulate_forecast_paths(
    params: MSGARCHXParams,
    xi_last: torch.Tensor,
    h_last: torch.Tensor,
    r_last: torch.Tensor,
    X_future: torch.Tensor,
    horizon: int,
    n_paths: int = 2000,
    variance_link: str = "clamp",
    floor: float = 1e-8,
    generator: torch.Generator | None = None,
) -> dict:
    """Mo phong `n_paths` duong du bao r_{T+1},...,r_{T+horizon}.

    xi_last : (2,) xac suat loc xi_{T|T}, dung de mau S_T.
    h_last  : (2,) phuong sai theo trang thai h_{T,s} tu bo loc.
    r_last  : scalar, gia tri quan sat cuoi cung r_T.
    X_future: (horizon, k) bien ngoai sinh da biet hoac theo kich ban.
    """
    device = params.mu.device
    state_T = torch.multinomial(xi_last, n_paths, replacement=True, generator=generator)

    eps_prev = r_last - params.mu[state_T]
    h_prev = h_last[state_T]
    state_prev = state_T

    r_paths = torch.empty(n_paths, horizon)
    h_paths = torch.empty(n_paths, horizon)
    state_paths = torch.empty(n_paths, horizon, dtype=torch.long)

    for step in range(horizon):
        u = torch.rand(n_paths, generator=generator)
        cdf1 = params.P[state_prev, 0]
        state_t = torch.where(u < cdf1, torch.zeros_like(state_prev), torch.ones_like(state_prev))

        h_raw = (
            params.omega[state_t]
            + params.alpha[state_t] * eps_prev**2
            + params.beta[state_t] * h_prev
            + (X_future[step] * params.gamma[state_t]).sum(dim=-1)
        )
        h_t = positive_variance_link(h_raw, floor=floor, mode=variance_link)

        nu_t = params.nu[state_t] if params.state_specific_nu else params.nu.expand(n_paths)
        z_t = std_t_sample(nu_t)
        eps_t = torch.sqrt(h_t) * z_t
        r_t = params.mu[state_t] + eps_t

        r_paths[:, step] = r_t
        h_paths[:, step] = h_t
        state_paths[:, step] = state_t

        eps_prev, h_prev, state_prev = eps_t, h_t, state_t

    return {"r": r_paths, "h": h_paths, "state": state_paths}


def posterior_predictive_paths(
    params_samples: list[MSGARCHXParams],
    xi_last_samples: list[torch.Tensor],
    h_last_samples: list[torch.Tensor],
    r_last: torch.Tensor,
    X_future: torch.Tensor,
    horizon: int,
    n_paths_per_sample: int = 200,
    variance_link: str = "clamp",
    floor: float = 1e-8,
) -> torch.Tensor:
    """Gop mo phong tren M mau hau nghiem -> xap xi Monte Carlo cua
    p(r_{T+1:T+h} | Data_T) theo eq:mcmc-predictive / eq:posterior-predictive.

    Tra ve tensor (M * n_paths_per_sample, horizon).
    """
    all_paths = []
    for params, xi_last, h_last in zip(params_samples, xi_last_samples, h_last_samples):
        out = simulate_forecast_paths(
            params, xi_last, h_last, r_last, X_future, horizon,
            n_paths=n_paths_per_sample, variance_link=variance_link, floor=floor,
        )
        all_paths.append(out["r"])
    return torch.cat(all_paths, dim=0)


def var_from_samples(samples: torch.Tensor, alpha: float) -> torch.Tensor:
    """VaR_{t,alpha} = -q_{t,alpha}, voi q_{t,alpha} la phan vi muc alpha cua
    mau du bao (eq:predictive-quantile, eq:var-positive)."""
    q = torch.quantile(samples, alpha)
    return -q
