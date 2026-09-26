"""GARCH(1,1) va GARCH-X mot trang thai.

    r_t = mu + eps_t,        eps_t = sqrt(h_t) z_t
    h_t = omega + alpha eps_{t-1}^2 + beta h_{t-1} + gamma^T X_{t-1}

Quy uoc canh chinh du lieu: hang X[t] chua bien ngoai sinh DA BIET truoc khi
quan sat r[t] (tuc X_{t-1} trong cong thuc), da duoc dich tre san khi dung du
lieu (xem du_lieu/bien_ngoai_sinh.py). Mo hinh khong co X thi truyen X co 0 cot.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from garch_vn30.mo_hinh.phan_phoi import normal_logpdf, std_t_logpdf

# gamma^T X co the am nen ve phai co the am: cat duoi tai H_MIN de h_t > 0.
H_MIN = 1e-8


class GARCHParams(NamedTuple):
    mu: torch.Tensor       # vo huong
    omega: torch.Tensor    # vo huong > 0
    alpha: torch.Tensor    # vo huong >= 0
    beta: torch.Tensor     # vo huong >= 0
    gamma: torch.Tensor    # (k,), k = 0 neu khong co bien ngoai sinh
    nu: Optional[torch.Tensor] = None  # > 2; None = phan phoi chuan


def garch_forward(params: GARCHParams, r: torch.Tensor, X: torch.Tensor,
                  h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Duong phuong sai co dieu kien h_0, ..., h_{T-1}.

    h_0 la dieu kien dau (mac dinh: phuong sai mau cua r), khong tinh vao
    log-hop-ly. eps_t^2 va gamma^T X_t khong phu thuoc h nen tinh san mot lan.
    """
    T = r.shape[0]
    if h0 is None:
        h0 = torch.var(r).detach()
    eps = r - params.mu
    eps_sq = eps * eps
    gX = X @ params.gamma
    h_list = [h0]
    h_prev = h0
    for t in range(1, T):
        h_raw = params.omega + params.alpha * eps_sq[t - 1] + params.beta * h_prev + gX[t]
        h_prev = torch.clamp(h_raw, min=H_MIN)
        h_list.append(h_prev)
    return torch.stack(h_list)


def log_mat_do(r: torch.Tensor, mu: torch.Tensor, h: torch.Tensor,
               nu: Optional[torch.Tensor]) -> torch.Tensor:
    """log f(r_t | F_{t-1}) cho tung t:  log f_z(z_t) - log(h_t)/2,  z_t = (r_t-mu)/sqrt(h_t).

    So hang -log(h_t)/2 la Jacobian cua phep doi bien z -> r. Dung chung cho
    GARCH, EGARCH va danh gia ngoai mau.
    """
    z = (r - mu) / torch.sqrt(h)
    if nu is None:
        return normal_logpdf(z) - 0.5 * torch.log(h)
    return std_t_logpdf(z, nu) - 0.5 * torch.log(h)


def garch_loglik(params: GARCHParams, r: torch.Tensor, X: torch.Tensor,
                 h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """log-hop-ly  l = sum_{t>=1} log f(r_t | F_{t-1})  (bo t = 0 lam dieu kien dau)."""
    h = garch_forward(params, r, X, h0=h0)
    return log_mat_do(r, params.mu, h, params.nu)[1:].sum()
