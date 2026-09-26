"""Ba truong hop khoi phuc tham so cho EGARCH-X Student-t.

    E1  don bay ro     theta_0 = -0,10, nu_0 = 8  -- co khoi phuc duoc don bay khong?
    E2  KHONG don bay  theta_0 =  0,    nu_0 = 8  -- co BIA RA don bay khi khong co khong?
    E3  duoi day       theta_0 = -0,10, nu_0 = 4  -- con dung khi duoi rat day (vung VN30F1M)?

Cac tham so con lai chung: mu = 0,05; omega = 0,04; alpha = 0,15; beta = 0,95;
gamma = (0,05; 0,05). Muc log-phuong sai vo dieu kien omega/(1-beta) = 0,8,
tuc h ~ 2,2 -- cung bac voi phuong sai mau cua VN30F1M (~2,27).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from garch_vn30.mo_hinh.egarch import EGARCHParams

# Thu tu va ten cac tham so trong moi bang/hinh cua nghien cuu mo phong.
THU_TU = ["mu", "omega", "alpha", "theta", "beta", "gamma_0", "gamma_1", "nu"]


@dataclass(frozen=True)
class TruongHop:
    code: str
    mo_ta: str
    params: EGARCHParams
    dist: str
    T: int
    B: int
    burn_in: int = 200


def _tham_so(theta: float, nu: float) -> EGARCHParams:
    return EGARCHParams(mu=torch.tensor(0.05), omega=torch.tensor(0.04),
                        alpha=torch.tensor(0.15), theta=torch.tensor(theta),
                        beta=torch.tensor(0.95), gamma=torch.tensor([0.05, 0.05]),
                        nu=torch.tensor(nu))


def truong_hop_mac_dinh(T: int = 2000, B: int = 100) -> dict[str, TruongHop]:
    return {
        "E1": TruongHop("E1", "Don bay ro: theta_0 = -0,10", _tham_so(-0.10, 8.0), "student_t", T, B),
        "E2": TruongHop("E2", "KHONG don bay: theta_0 = 0", _tham_so(0.0, 8.0), "student_t", T, B),
        "E3": TruongHop("E3", "Duoi day: nu_0 = 4, co don bay", _tham_so(-0.10, 4.0), "student_t", T, B),
    }


def tham_so_that(params: EGARCHParams) -> dict[str, float]:
    """Bo tham so duoi dang {ten: gia tri} (gamma tach thanh gamma_0, gamma_1, ...)."""
    d = {"mu": float(params.mu), "omega": float(params.omega), "alpha": float(params.alpha),
         "theta": float(params.theta), "beta": float(params.beta)}
    for j in range(params.gamma.shape[0]):
        d[f"gamma_{j}"] = float(params.gamma[j])
    if params.nu is not None:
        d["nu"] = float(params.nu)
    return d
