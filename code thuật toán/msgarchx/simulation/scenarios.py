"""Cac kich ban mo phong S1-S8 (Bang 6.3, tab:simulation-scenarios).

Gia tri tham so cu the o day la GIA TRI VI DU hop ly ve mat dinh tinh (dung
de chay thu toan bo pipeline tren du lieu mo phong ngay khi chua co du lieu
VN30F1M that). Nguoi dung can thay bang gia tri chinh thuc truoc khi bao cao
ket qua mo phong trong ban thao (xem \\CanBoSung sau Bang 6.3 trong bai bao).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from msgarchx.models.ms_garch import MSGARCHXParams


@dataclass(frozen=True)
class SimulationScenario:
    code: str
    description: str
    params: MSGARCHXParams
    P0: torch.Tensor
    T: int
    B: int
    burn_in: int = 200
    misspecified: bool = False


def _base_params(
    omega1: float,
    omega2: float,
    alpha1: float,
    alpha2: float,
    beta1: float,
    beta2: float,
    nu0: float,
    n_exog: int = 2,
    gamma_scale: float = 0.05,
) -> MSGARCHXParams:
    return MSGARCHXParams(
        mu=torch.tensor([0.0, 0.0]),
        omega=torch.tensor([omega1, omega2]),
        alpha=torch.tensor([alpha1, alpha2]),
        beta=torch.tensor([beta1, beta2]),
        gamma=torch.full((2, n_exog), gamma_scale),
        P=torch.eye(2),  # khong dung truc tiep, P0 rieng duoc truyen vao DGP
        nu=torch.tensor(nu0),
    )


def _P0(p11: float, p22: float) -> torch.Tensor:
    return torch.tensor([[p11, 1 - p11], [1 - p22, p22]])


def default_scenarios(T_default: int = 2000, B_default: int = 200) -> dict[str, SimulationScenario]:
    scenarios = {}

    scenarios["S1"] = SimulationScenario(
        code="S1",
        description="Hai trang thai phan biet ro, nu0=8, p11 va p22 cao",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=8.0),
        P0=_P0(0.98, 0.95),
        T=T_default, B=B_default,
    )
    scenarios["S2"] = SimulationScenario(
        code="S2",
        description="Hai trang thai gan nhau, nu0=8",
        params=_base_params(0.08, 0.14, 0.06, 0.08, 0.85, 0.83, nu0=8.0),
        P0=_P0(0.95, 0.90),
        T=T_default, B=B_default,
    )
    scenarios["S3"] = SimulationScenario(
        code="S3",
        description="Hai trang thai phan biet ro, nu0=4 (duoi rat day)",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=4.0),
        P0=_P0(0.98, 0.95),
        T=T_default, B=B_default,
    )
    scenarios["S4"] = SimulationScenario(
        code="S4",
        description="Hai trang thai phan biet ro, nu0 lon (gan phan phoi chuan)",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=30.0),
        P0=_P0(0.98, 0.95),
        T=T_default, B=B_default,
    )
    scenarios["S5"] = SimulationScenario(
        code="S5",
        description="Trang thai bien dong cao hiem, p11 rat cao, p22 trung binh",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=8.0),
        P0=_P0(0.99, 0.85),
        T=T_default, B=B_default,
    )
    scenarios["S6"] = SimulationScenario(
        code="S6",
        description="Co mau nho, tham so nhu S1",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=8.0),
        P0=_P0(0.98, 0.95),
        T=500, B=B_default,
    )
    scenarios["S7"] = SimulationScenario(
        code="S7",
        description="Co mau lon, tham so nhu S1",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=8.0),
        P0=_P0(0.98, 0.95),
        T=5000, B=max(B_default // 2, 50),
    )
    scenarios["S8"] = SimulationScenario(
        code="S8",
        description="Sai dac ta: du lieu them thanh phan tu tuong quan trong trung binh",
        params=_base_params(0.05, 0.40, 0.05, 0.10, 0.85, 0.80, nu0=8.0),
        P0=_P0(0.98, 0.95),
        T=T_default, B=B_default,
        misspecified=True,
    )
    return scenarios
