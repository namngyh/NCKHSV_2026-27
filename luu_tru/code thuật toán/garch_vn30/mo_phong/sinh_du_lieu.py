"""Cơ chế sinh dữ liệu (data generating process, DGP) từ EGARCH-X một trạng thái.

MỤC ĐÍCH
    Nghiên cứu mô phỏng cần dữ liệu mà ta BIẾT tham số thật theta_0: sinh dữ liệu
    từ theta_0, ước lượng lại, rồi so ước lượng với theta_0 để đo độ chệch, RMSE
    và độ phủ của khoảng tin cậy.

CÁCH SINH (cùng phương trình với mo_hinh/egarch.py nhưng chạy "xuôi")
    1. X_t ~ N(0, I_k) độc lập (hoặc AR(1) nếu ar_coef != 0).
    2. z_t ~ N(0,1) hoặc t*_nu (Student-t chuẩn hóa).
    3. Điều kiện đầu log h_0 = omega / (1 - beta): trung bình vô điều kiện của
       log-phương sai khi E[X] = 0.
    4. Lặp: log h_t theo phương trình EGARCH với z_{t-1} là giá trị vừa rút,
       r_t = mu + sqrt(h_t) z_t.
    5. Bỏ `burn_in` quan sát đầu để xóa ảnh hưởng của điều kiện đầu.
    Thứ tự rút số ngẫu nhiên cố định (X trước, z sau), nên cùng seed luôn cho
    cùng một bộ dữ liệu.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from garch_vn30.mo_hinh.egarch import LOG_H_MAX, LOG_H_MIN, EGARCHParams, expected_abs_z
from garch_vn30.mo_hinh.phan_phoi import std_t_sample


@dataclass
class DuLieuMoPhong:
    """Một bộ dữ liệu mô phỏng (sau khi đã bỏ burn-in)."""
    r: torch.Tensor   # (T,) lợi suất
    h: torch.Tensor   # (T,) phương sai có điều kiện THẬT
    X: torch.Tensor   # (T, k) biến ngoại sinh


def generate_exogenous(total_T: int, n_exog: int, ar_coef: float = 0.0,
                       scale: float = 1.0) -> torch.Tensor:
    """Sinh ma trận ngoại sinh X_t = ar_coef X_{t-1} + e_t, e_t ~ N(0, scale^2 I)
    (độc lập theo thời gian khi ar_coef = 0, mặc định). Trả về tensor (total_T, n_exog)."""
    X = torch.zeros(total_T, n_exog)
    eps = torch.randn(total_T, n_exog) * scale
    X[0] = eps[0]
    for t in range(1, total_T):
        X[t] = ar_coef * X[t - 1] + eps[t]
    return X


def simulate_egarch(params: EGARCHParams, T: int, dist: str = "student_t",
                    n_exog: int | None = None, X: torch.Tensor | None = None,
                    burn_in: int = 200, seed: int | None = None) -> DuLieuMoPhong:
    """Sinh T quan sát từ EGARCH-X với tham số thật `params`.

    Tham số:
        params : tham số thật theta_0 (EGARCHParams).
        T      : số quan sát giữ lại sau burn-in.
        dist   : "normal" hoặc "student_t" (khi đó dùng params.nu).
        n_exog : số biến ngoại sinh (mặc định = độ dài params.gamma).
        X      : có thể truyền sẵn ma trận ngoại sinh độ dài T + burn_in.
        burn_in: số quan sát đầu bị bỏ.
        seed   : hạt giống (torch.manual_seed) để tái lập.
    Trả về: DuLieuMoPhong(r, h, X).
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
