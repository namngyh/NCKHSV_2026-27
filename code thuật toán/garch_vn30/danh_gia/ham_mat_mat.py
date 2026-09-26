"""Ham mat mat du bao ngoai mau. Moi thuoc do: nho hon la tot hon.

RV_t (bien dong thuc te) va h_hat_t phai cung don vi phuong sai (%^2).
"""
from __future__ import annotations

import torch


def mae_vol(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """MAE = mean |RV_t - h_hat_t|."""
    return (rv - h_hat).abs().mean()


def rmse_vol(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """RMSE = sqrt(mean (RV_t - h_hat_t)^2)."""
    return torch.sqrt(((rv - h_hat) ** 2).mean())


def qlike(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """QLIKE = mean[log h_hat_t + RV_t / h_hat_t]  (Patton, 2011).

    Ben vung voi nhieu cua dai dien bien dong: thu tu xep hang cac mo hinh
    khong doi khi thay RV_t bang mot dai dien khong chech co nhieu.
    """
    return (torch.log(h_hat) + rv / h_hat).mean()


def negative_log_score(log_densities: torch.Tensor) -> torch.Tensor:
    """NLS = -mean_t log f_t(r_t): diem log am cua mat do du bao."""
    return -log_densities.mean()
