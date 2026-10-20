"""Ba trường hợp của nghiên cứu khôi phục tham số cho EGARCH-X Student-t.

    E1  đòn bẩy rõ     theta_0 = -0,10, nu_0 = 8  -- bộ ước lượng có khôi phục được đòn bẩy không?
    E2  KHÔNG đòn bẩy  theta_0 =  0,    nu_0 = 8  -- có "bịa ra" đòn bẩy khi dữ liệu không có không?
    E3  đuôi dày       theta_0 = -0,10, nu_0 = 4  -- còn khôi phục đúng khi đuôi rất dày
                                                     (gần vùng nu ước lượng của VN30F1M) không?

Các tham số còn lại dùng chung: mu = 0,05; omega = 0,04; alpha = 0,15; beta = 0,95;
gamma = (0,05; 0,05). Khi đó omega/(1-beta) = 0,8, tức h ≈ 2,2 -- cùng bậc với
phương sai mẫu của VN30F1M (≈ 2,27). E2 kiểm tra chiều ngược lại của E1: nếu
theta ước lượng lệch khỏi 0 một cách hệ thống khi dữ liệu không có đòn bẩy, thì
kết luận "VN30F1M có hiệu ứng đòn bẩy" sẽ không đáng tin.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from garch_vn30.mo_hinh.egarch import EGARCHParams

# Tên và thứ tự các tham số trong mọi bảng/hình của nghiên cứu mô phỏng
# (gamma được tách thành gamma_0, gamma_1).
THU_TU = ["mu", "omega", "alpha", "theta", "beta", "gamma_0", "gamma_1", "nu"]


@dataclass(frozen=True)
class TruongHop:
    """Một trường hợp mô phỏng."""
    code: str             # "E1", "E2", "E3"
    mo_ta: str            # mô tả ngắn
    params: EGARCHParams  # tham số thật theta_0
    dist: str             # phân phối sai số khi sinh dữ liệu
    T: int                # số quan sát mỗi lần lặp (sau burn-in)
    B: int                # số lần lặp Monte Carlo
    burn_in: int = 200


def _tham_so(theta: float, nu: float) -> EGARCHParams:
    """Tham số thật chung, chỉ khác theta và nu."""
    return EGARCHParams(mu=torch.tensor(0.05), omega=torch.tensor(0.04),
                        alpha=torch.tensor(0.15), theta=torch.tensor(theta),
                        beta=torch.tensor(0.95), gamma=torch.tensor([0.05, 0.05]),
                        nu=torch.tensor(nu))


def truong_hop_mac_dinh(T: int = 2000, B: int = 100) -> dict[str, TruongHop]:
    """Ba trường hợp E1-E3 với T quan sát và B lần lặp (mặc định T = 2000, B = 100)."""
    return {
        "E1": TruongHop("E1", "Don bay ro: theta_0 = -0,10", _tham_so(-0.10, 8.0), "student_t", T, B),
        "E2": TruongHop("E2", "KHONG don bay: theta_0 = 0", _tham_so(0.0, 8.0), "student_t", T, B),
        "E3": TruongHop("E3", "Duoi day: nu_0 = 4, co don bay", _tham_so(-0.10, 4.0), "student_t", T, B),
    }


def tham_so_that(params: EGARCHParams) -> dict[str, float]:
    """Bộ tham số dưới dạng {tên: giá trị số}, gamma tách thành gamma_0, gamma_1, ...
    Dạng này dùng để so ước lượng với giá trị thật từng tham số."""
    d = {"mu": float(params.mu), "omega": float(params.omega), "alpha": float(params.alpha),
         "theta": float(params.theta), "beta": float(params.beta)}
    for j in range(params.gamma.shape[0]):
        d[f"gamma_{j}"] = float(params.gamma[j])
    if params.nu is not None:
        d["nu"] = float(params.nu)
    return d
