"""Ham mat mat du bao bien dong ngoai mau (Muc 6.7): MAE, RMSE, QLIKE.

eq:mae-vol, eq:rmse-vol, eq:qlike. Gia tri nho hon la du bao tot hon o ca ba
thuoc do. RV_t va h_hat_t phai cung don vi phuong sai (xem checklist Phu luc B).
"""
from __future__ import annotations

import torch


def mae_vol(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    return (rv - h_hat).abs().mean()


def rmse_vol(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(((rv - h_hat) ** 2).mean())


def qlike(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """QLIKE = mean[log(h_hat) + RV/h_hat], on dinh hon voi bien dong dai
    dien co nhieu (Patton, 2011). h_hat phai duong tuyet doi."""
    return (torch.log(h_hat) + rv / h_hat).mean()
