"""Bien doi tham so khong rang buoc <-> mien hop le.

Cac cong thuc bam theo Muc 4.2 cua bai bao:
- p_ss = sigmoid(eta_s)                       (eq:logit-transition)
- nu   = 2 + softplus(eta_nu)                 (eq:df-transform)
- omega_s = exp(log_omega_s)                  (khop voi prior log-normal, eq:prior-omega)
- alpha_s, beta_s = softplus(.) >= 0          (mien duong, khong dac ta prior rieng trong bai)

Ham `positive_variance_link` xu ly tinh huong h_t co the am khi gamma^T X_{t-1}
mang dau am (xem thao luan sau eq:garchx). Mac dinh dung floor cung ("clamp")
de giu nguyen dang cong eq:main-variance; co the doi sang "softplus" de co
rang buoc duong tuyet doi bang mot ham lien ket.
"""
from __future__ import annotations

import torch

_SOFTPLUS = torch.nn.functional.softplus


def softplus(x: torch.Tensor) -> torch.Tensor:
    return _SOFTPLUS(x)


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Nghich dao softplus, dung de khoi tao raw-param tu gia tri mong muon."""
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    return y + torch.log(-torch.expm1(-y))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def logit(p: torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.get_default_dtype())
    return torch.log(p) - torch.log1p(-p)


def nu_from_raw(eta_nu: torch.Tensor) -> torch.Tensor:
    """eq:df-transform: nu = 2 + softplus(eta_nu), bao dam nu > 2."""
    return 2.0 + softplus(eta_nu)


def raw_from_nu(nu: torch.Tensor) -> torch.Tensor:
    return inv_softplus(torch.as_tensor(nu, dtype=torch.get_default_dtype()) - 2.0)


def p_from_raw(eta_p: torch.Tensor) -> torch.Tensor:
    """eq:logit-transition: p_ss = sigmoid(eta_s)."""
    return sigmoid(eta_p)


def raw_from_p(p: torch.Tensor) -> torch.Tensor:
    return logit(p)


def positive_variance_link(
    h_raw: torch.Tensor,
    floor: float = 1e-8,
    mode: str = "clamp",
) -> torch.Tensor:
    """Bao dam h_t > 0 khi h_raw = omega + alpha*eps^2 + beta*h_prev + gamma'X.

    mode="clamp": giu nguyen dang cong, chi cat duoi tai `floor` (kha vi hau
        khap noi, phu hop khi gamma'X hiem khi am manh).
    mode="softplus": ham lien ket duong tuyet doi h_t = softplus(h_raw - floor) + floor,
        bao dam duong o moi diem nhung lam h_t khong con tuyen tinh voi cac
        thanh phan cong nhu trong eq:main-variance.
    """
    if mode == "clamp":
        return torch.clamp(h_raw, min=floor)
    if mode == "softplus":
        return softplus(h_raw - floor) + floor
    raise ValueError(f"mode khong hop le: {mode!r}")
