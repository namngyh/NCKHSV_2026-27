"""Mô hình EGARCH(1,1) và EGARCH-X một trạng thái (Nelson, 1991).

LÝ THUYẾT
    EGARCH viết phương trình cho LOGARIT của phương sai:

        log h_t = omega + alpha * (|z_{t-1}| - E|z|) + theta * z_{t-1}
                  + beta * log h_{t-1} + gamma^T X_{t-1},
        z_{t-1} = eps_{t-1} / sqrt(h_{t-1})      (sai số chuẩn hóa phiên trước)

    - alpha: hiệu ứng ĐỘ LỚN -- cú sốc có |z| lớn hơn mức kỳ vọng làm tăng biến động.
    - theta: hiệu ứng DẤU (đòn bẩy) -- với theta < 0, cú sốc âm z = -c làm log h
             tăng (alpha - theta) c, nhiều hơn mức (alpha + theta) c của cú sốc
             dương cùng độ lớn: tin xấu làm thị trường biến động mạnh hơn tin tốt.
    - beta : độ dai dẳng của log-phương sai; quá trình dừng khi |beta| < 1, và khi
             đó E[log h] = omega / (1 - beta) nếu E[X] = 0.
    - gamma: hệ số biến ngoại sinh.

    Ưu điểm so với GARCH: h_t = exp(...) luôn dương THEO CẤU TRÚC, nên không cần
    ràng buộc dấu cho omega, alpha, theta, gamma. Đề tài quy ước beta ∈ (0, 1) để
    trùng với tiên nghiệm Beta của Bayes (xem suy_dien/mle.py).

    Khác GARCH: z_{t-1} phụ thuộc h_{t-1} vừa tính, nên mọi số hạng phải tính bên
    trong vòng lặp (không vector hóa trước được).

THƯ VIỆN
    torch: phép toán tensor khả vi; torch.clamp chặn giá trị; torch.exp/log.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch

from garch_vn30.mo_hinh.garch import log_mat_do
from garch_vn30.mo_hinh.phan_phoi import expected_abs_normal, expected_abs_std_t

# Lưới an toàn số học để exp() không tràn số; KHÔNG phải ràng buộc mô hình.
# Lợi suất theo phần trăm cho log h thực tế quanh 0-1, rất xa hai biên này.
LOG_H_MIN = -30.0
LOG_H_MAX = 30.0


class EGARCHParams(NamedTuple):
    """Bộ tham số của EGARCH(-X). Truy cập theo tên, bất biến."""
    mu: torch.Tensor       # trung bình lợi suất
    omega: torch.Tensor    # hệ số chặn của log-phương sai (tự do dấu)
    alpha: torch.Tensor    # hiệu ứng độ lớn
    theta: torch.Tensor    # hiệu ứng đòn bẩy (dấu)
    beta: torch.Tensor     # độ dai dẳng, trong (0, 1)
    gamma: torch.Tensor    # vector hệ số ngoại sinh, shape (k,)
    nu: Optional[torch.Tensor] = None  # bậc tự do Student-t; None = phân phối chuẩn


def expected_abs_z(nu: Optional[torch.Tensor]) -> Union[torch.Tensor, float]:
    """E|z| theo phân phối sai số: sqrt(2/pi) nếu chuẩn (nu là None), công thức t*_nu nếu Student-t."""
    if nu is None:
        return expected_abs_normal()
    return expected_abs_std_t(nu)


def egarch_forward(params: EGARCHParams, r: torch.Tensor, X: torch.Tensor,
                   h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Tính đường phương sai h_0, ..., h_{T-1} theo đệ quy EGARCH.

    Tham số giống garch_forward: r shape (T,), X shape (T, k), h0 mặc định là
    phương sai mẫu của r.
    Trả về: tensor h shape (T,) -- trả về h (không phải log h) để mọi hàm phía
    sau (log-hợp lý, đánh giá, dự báo) dùng chung giao diện với GARCH.
    """
    T = r.shape[0]
    if h0 is None:
        h0 = torch.var(r).detach()
    eps = r - params.mu
    gX = X @ params.gamma                 # chỉ số hạng ngoại sinh tính trước được
    e_abs_z = expected_abs_z(params.nu)   # hằng số E|z| (phụ thuộc nu)

    h_list = [h0]
    h_prev = h0
    for t in range(1, T):
        z_prev = eps[t - 1] / torch.sqrt(h_prev)      # sai số chuẩn hóa phiên trước
        log_h = (
            params.omega
            + params.alpha * (torch.abs(z_prev) - e_abs_z)   # hiệu ứng độ lớn
            + params.theta * z_prev                          # hiệu ứng đòn bẩy
            + params.beta * torch.log(h_prev)                # độ dai dẳng
            + gX[t]                                          # biến ngoại sinh
        )
        h_prev = torch.exp(torch.clamp(log_h, LOG_H_MIN, LOG_H_MAX))
        h_list.append(h_prev)
    return torch.stack(h_list)


def egarch_loglik(params: EGARCHParams, r: torch.Tensor, X: torch.Tensor,
                  h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Log-hợp lý có điều kiện của EGARCH(-X): cùng công thức với garch_loglik,
    chỉ khác cách tính đường phương sai h_t."""
    h = egarch_forward(params, r, X, h0=h0)
    return log_mat_do(r, params.mu, h, params.nu)[1:].sum()
