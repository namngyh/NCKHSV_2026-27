"""Hàm mất mát đánh giá dự báo ngoài mẫu. Với mọi thước đo: nhỏ hơn là tốt hơn.

LÝ THUYẾT
    Phương sai thật của lợi suất không quan sát được; biến động thực tế RV_t
    (tổng bình phương lợi suất phút trong ngày) được dùng làm đại diện. Với n
    quan sát kiểm định, dự báo phương sai h_hat_t:

        MAE   = (1/n) sum |RV_t - h_hat_t|
        RMSE  = sqrt( (1/n) sum (RV_t - h_hat_t)^2 )
        QLIKE = (1/n) sum [ log h_hat_t + RV_t / h_hat_t ]
        NLS   = -(1/n) sum log f_t(r_t)       (điểm log âm của mật độ dự báo)

    QLIKE (Patton, 2011) giữ nguyên thứ tự xếp hạng các mô hình khi RV_t là đại
    diện không chệch nhưng có nhiễu của phương sai thật; MAE/RMSE thì không có
    tính chất này và bị các ngày biến động lớn chi phối. NLS đánh giá cả mật độ
    dự báo (bao gồm đuôi), không chỉ phương sai.

    RV_t và h_hat_t phải cùng đơn vị (%^2).
"""
from __future__ import annotations

import torch


def mae_vol(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """MAE = trung bình |RV_t - h_hat_t|."""
    return (rv - h_hat).abs().mean()


def rmse_vol(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """RMSE = căn của trung bình (RV_t - h_hat_t)^2."""
    return torch.sqrt(((rv - h_hat) ** 2).mean())


def qlike(rv: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """QLIKE = trung bình [log h_hat_t + RV_t / h_hat_t]. Đạt nhỏ nhất khi h_hat_t = RV_t."""
    return (torch.log(h_hat) + rv / h_hat).mean()


def negative_log_score(log_densities: torch.Tensor) -> torch.Tensor:
    """NLS = -trung bình log f_t(r_t); đầu vào là vector log-mật độ dự báo tại giá trị thực."""
    return -log_densities.mean()
