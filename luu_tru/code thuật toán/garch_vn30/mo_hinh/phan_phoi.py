"""Phân phối của sai số chuẩn hóa z_t trong các mô hình GARCH/EGARCH.

LÝ THUYẾT
    Mọi mô hình trong đề tài có dạng
        r_t = mu + eps_t,      eps_t = sqrt(h_t) * z_t,
    trong đó h_t là phương sai có điều kiện và z_t là "sai số chuẩn hóa": độc lập,
    cùng phân phối, E[z] = 0, Var[z] = 1. Hai lựa chọn cho z_t:

    1. Chuẩn N(0, 1):  f(z) = exp(-z^2 / 2) / sqrt(2 pi).
    2. Student-t CHUẨN HÓA t*_nu (nu > 2 là bậc tự do):
           f(z | nu) = Gamma((nu+1)/2) / [Gamma(nu/2) sqrt(pi (nu-2))]
                       * (1 + z^2 / (nu-2))^(-(nu+1)/2).
       Đây là Student-t thông thường co lại theo hệ số sqrt((nu-2)/nu) để phương
       sai đúng bằng 1. Nhờ vậy h_t luôn là phương sai của r_t, dù chọn phân phối
       nào. nu càng nhỏ đuôi càng dày (xác suất biến động cực đoan càng lớn);
       nu -> vô cùng thì t*_nu tiến về N(0, 1).

    EGARCH cần thêm kỳ vọng E|z| (xem hàm expected_abs_std_t).

THƯ VIỆN
    torch  : mọi hàm nhận/trả tensor PyTorch để dùng được vi phân tự động
             (autograd) khi tối ưu log-hợp lý; torch.lgamma = log Gamma(x).
    math   : hằng số pi, sqrt cho các giá trị không cần đạo hàm.
"""
from __future__ import annotations

import math

import torch

_LOG_PI = math.log(math.pi)   # log(pi), tính sẵn một lần


def std_t_log_const(nu: torch.Tensor) -> torch.Tensor:
    """Log của hằng số chuẩn hóa trong mật độ t*_nu.

        log C(nu) = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0,5 log(nu-2) - 0,5 log(pi)

    Kỹ thuật: tính trên thang log bằng torch.lgamma thay vì Gamma rồi lấy log,
    vì Gamma(nu/2) tràn số (quá lớn) khi nu lớn.

    Tham số:
        nu: bậc tự do (tensor, > 2), có thể là vô hướng hoặc vector.
    Trả về:
        tensor cùng shape với nu.
    """
    return (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu - 2.0)
        - 0.5 * _LOG_PI
    )


def std_t_logpdf(z: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Log-mật độ của phân phối Student-t chuẩn hóa tại điểm z.

        log f(z | nu) = log C(nu) - (nu+1)/2 * log(1 + z^2 / (nu-2))

    Kỹ thuật: torch.log1p(x) = log(1 + x) chính xác hơn khi x rất nhỏ.
    Hàm hỗ trợ "broadcasting" của PyTorch: z có thể là vector (T,) còn nu là vô
    hướng, kết quả là vector (T,) -- tính cho cả chuỗi trong một lệnh.

    Tham số:
        z : sai số chuẩn hóa (tensor).
        nu: bậc tự do (tensor, > 2).
    Trả về:
        tensor log-mật độ, cùng shape với z.
    """
    return std_t_log_const(nu) - ((nu + 1.0) / 2.0) * torch.log1p(z * z / (nu - 2.0))


def normal_logpdf(z: torch.Tensor) -> torch.Tensor:
    """Log-mật độ chuẩn N(0,1):  log f(z) = -z^2/2 - log(2 pi)/2."""
    return -0.5 * (z * z) - 0.5 * math.log(2.0 * math.pi)


def expected_abs_normal() -> float:
    """E|z| khi z ~ N(0, 1), bằng sqrt(2/pi) ≈ 0,7979. Dùng trong phương trình EGARCH."""
    return math.sqrt(2.0 / math.pi)


def expected_abs_std_t(nu: torch.Tensor) -> torch.Tensor:
    """E|z| khi z ~ t*_nu (Student-t chuẩn hóa).

    Công thức:
        E|z| = 2 sqrt(nu-2) Gamma((nu+1)/2) / [ (nu-1) sqrt(pi) Gamma(nu/2) ].

    Dẫn xuất: với T ~ t_nu thông thường, E|T| = 2 sqrt(nu) Gamma((nu+1)/2) /
    [(nu-1) sqrt(pi) Gamma(nu/2)]; vì z = T sqrt((nu-2)/nu) nên nhân thêm hệ số
    đó ta được công thức trên.

    Vai trò: EGARCH dùng số hạng alpha (|z_{t-1}| - E|z|); trừ E|z| để số hạng
    này có trung bình 0. Sai hằng số này làm lệch mức nền của phương sai mà không
    gây lỗi số học nào, nên có kiểm thử riêng: khi nu lớn giá trị phải hội tụ về
    sqrt(2/pi).

    Kỹ thuật: tính trên thang log (lgamma) rồi lấy exp để tránh tràn số.
    """
    log_ratio = torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0)
    log_val = (
        math.log(2.0)
        + 0.5 * torch.log(nu - 2.0)
        + log_ratio
        - torch.log(nu - 1.0)
        - 0.5 * _LOG_PI
    )
    return torch.exp(log_val)


def std_t_sample(nu: torch.Tensor) -> torch.Tensor:
    """Sinh mẫu ngẫu nhiên z ~ t*_nu, một mẫu cho MỖI phần tử của `nu`.

    Kỹ thuật (định nghĩa Student-t): nếu Z ~ N(0,1) và V ~ chi2(nu) độc lập thì
    T = Z / sqrt(V / nu) ~ t_nu. Sau đó chia cho sqrt(nu / (nu-2)) để phương sai
    bằng 1.

    Cách dùng: muốn n mẫu với cùng một nu thì truyền nu.expand(n). Đặt
    torch.manual_seed(...) trước khi gọi để kết quả tái lập được.

    Thư viện: torch.randn sinh N(0,1); torch.distributions.Chi2(nu).sample()
    sinh biến khi bình phương.
    """
    normal = torch.randn(nu.shape)
    chi2 = torch.distributions.Chi2(nu).sample()
    raw = normal * torch.sqrt(nu / chi2)        # Student-t thông thường, phương sai nu/(nu-2)
    return raw / torch.sqrt(nu / (nu - 2.0))    # chuẩn hóa về phương sai 1
