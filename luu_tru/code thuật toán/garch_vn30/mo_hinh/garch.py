"""Mô hình GARCH(1,1) và GARCH-X một trạng thái (Bollerslev, 1986).

LÝ THUYẾT
    Lợi suất:        r_t = mu + eps_t,        eps_t = sqrt(h_t) * z_t
    Phương sai:      h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1} + gamma^T X_{t-1}

    - omega > 0 : mức nền của phương sai.
    - alpha >= 0: phản ứng của phương sai với cú sốc mới (eps_{t-1}^2 lớn -> h_t tăng).
    - beta  >= 0: độ dai dẳng -- biến động hôm nay kéo dài sang ngày mai.
    - gamma     : hệ số của biến ngoại sinh X_{t-1} (khối lượng, RV của phiên trước).
    Khi gamma = 0 và alpha + beta < 1, phương sai dài hạn là omega / (1 - alpha - beta).

    Vì gamma^T X có thể âm nên vế phải có thể âm; mã nguồn cắt dưới h_t >= 1e-8.

QUY ƯỚC DỮ LIỆU
    Hàng X[t] chứa biến ngoại sinh ĐÃ BIẾT trước khi quan sát r[t] (tức X_{t-1}
    trong công thức), đã được dịch trễ sẵn ở khâu chuẩn bị dữ liệu
    (du_lieu/chuan_bi.py). Mô hình không dùng X thì truyền X có 0 cột, khi đó
    gamma là vector rỗng và gamma^T X = 0.

THƯ VIỆN
    torch: các phép tính đều là phép toán tensor nên PyTorch tự tính được đạo
    hàm của log-hợp lý theo mọi tham số (autograd) -- đây là điều kiện để dùng
    Adam/L-BFGS (MLE), Bayes biến phân và NUTS (MCMC) trên cùng một hàm.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from garch_vn30.mo_hinh.phan_phoi import normal_logpdf, std_t_logpdf

# Cận dưới của phương sai: gamma^T X có thể âm nên vế phải có thể âm.
H_MIN = 1e-8


class GARCHParams(NamedTuple):
    """Bộ tham số của GARCH(-X). NamedTuple: truy cập bằng tên (params.omega), bất biến."""
    mu: torch.Tensor       # trung bình lợi suất (vô hướng)
    omega: torch.Tensor    # hệ số chặn của phương sai (> 0)
    alpha: torch.Tensor    # hệ số ARCH (>= 0)
    beta: torch.Tensor     # hệ số GARCH, độ dai dẳng (>= 0)
    gamma: torch.Tensor    # vector hệ số ngoại sinh, shape (k,); k = 0 nếu không có X
    nu: Optional[torch.Tensor] = None  # bậc tự do Student-t (> 2); None = phân phối chuẩn


def garch_forward(params: GARCHParams, r: torch.Tensor, X: torch.Tensor,
                  h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Tính đường phương sai có điều kiện h_0, h_1, ..., h_{T-1} bằng đệ quy GARCH.

    Tham số:
        params: bộ tham số GARCHParams.
        r     : chuỗi lợi suất, tensor shape (T,).
        X     : ma trận ngoại sinh, shape (T, k); X[t] đã là giá trị trễ một phiên.
        h0    : điều kiện đầu h_0; mặc định là phương sai mẫu của r.
    Trả về:
        tensor h shape (T,). h[0] chỉ là điều kiện đầu, không dùng trong log-hợp lý.

    Kỹ thuật: eps_{t-1}^2 và gamma^T X_t không phụ thuộc h nên được tính một lần
    cho cả chuỗi (phép toán vector) trước vòng lặp; vòng lặp chỉ còn phép cộng.
    `.detach()` trên h0 để điều kiện đầu không tham gia vào đạo hàm.
    """
    T = r.shape[0]
    if h0 is None:
        h0 = torch.var(r).detach()
    eps = r - params.mu            # phần dư eps_t = r_t - mu
    eps_sq = eps * eps             # eps_t^2 cho mọi t
    gX = X @ params.gamma          # gamma^T X_t cho mọi t (phép nhân ma trận-vector)
    h_list = [h0]
    h_prev = h0
    for t in range(1, T):
        h_raw = params.omega + params.alpha * eps_sq[t - 1] + params.beta * h_prev + gX[t]
        h_prev = torch.clamp(h_raw, min=H_MIN)      # bảo đảm h_t > 0
        h_list.append(h_prev)
    return torch.stack(h_list)     # ghép danh sách thành một tensor (T,)


def log_mat_do(r: torch.Tensor, mu: torch.Tensor, h: torch.Tensor,
               nu: Optional[torch.Tensor]) -> torch.Tensor:
    """Log-mật độ có điều kiện log f(r_t | thông tin đến t-1) cho từng t.

        z_t = (r_t - mu) / sqrt(h_t)
        log f(r_t) = log f_z(z_t) - 0,5 * log(h_t)

    Số hạng -0,5 log h_t là Jacobian của phép đổi biến từ z sang r (vì
    r = mu + sqrt(h) z, nên dz/dr = 1/sqrt(h)). f_z là N(0,1) nếu nu là None,
    ngược lại là Student-t chuẩn hóa. Hàm dùng chung cho GARCH, EGARCH và cho
    đánh giá ngoài mẫu (danh_gia/ngoai_mau.py).

    Trả về: tensor shape giống r.
    """
    z = (r - mu) / torch.sqrt(h)
    if nu is None:
        return normal_logpdf(z) - 0.5 * torch.log(h)
    return std_t_logpdf(z, nu) - 0.5 * torch.log(h)


def garch_loglik(params: GARCHParams, r: torch.Tensor, X: torch.Tensor,
                 h0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Log-hợp lý có điều kiện của GARCH(-X):

        l(theta) = sum_{t=1}^{T-1} log f(r_t | F_{t-1})

    (bỏ t = 0 vì quan sát đầu chỉ làm điều kiện đầu). Kết quả là tensor vô hướng,
    khả vi theo mọi tham số trong `params` -- MLE cực đại hóa hàm này, còn mô hình
    Bayes (suy_dien/bayes.py) cộng nó vào log-mật độ hậu nghiệm.
    """
    h = garch_forward(params, r, X, h0=h0)
    return log_mat_do(r, params.mu, h, params.nu)[1:].sum()
