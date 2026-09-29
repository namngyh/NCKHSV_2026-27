"""Mô hình Bayes (viết bằng Pyro) cho GARCH/EGARCH một trạng thái, và tóm tắt hậu nghiệm.

LÝ THUYẾT
    Suy diễn Bayes coi tham số theta là biến ngẫu nhiên. Định lý Bayes:
        p(theta | r)  ∝  p(r | theta) * p(theta)
        hậu nghiệm        hợp lý        tiên nghiệm
    với p(r | theta) = exp(l(theta)) là CÙNG hàm hợp lý mà MLE dùng. Như vậy khác
    biệt giữa Bayes và MLE chỉ đến từ tiên nghiệm và cách suy diễn, không đến từ
    cách viết mô hình.

    Tiên nghiệm (PriorConfig) là tiên nghiệm yếu, đặt trực tiếp trên miền hợp lệ:
        mu ~ N(0,1);  gamma_j ~ N(0,1);  nu - 2 ~ Gamma(2; 0,2)  (kỳ vọng nu = 12)
        GARCH : omega ~ LogNormal(0,1)  (bảo đảm omega > 0)
        EGARCH: omega ~ N(0,1)          (không cần dương vì phương trình cho log h)
        alpha, beta ~ Beta(2,2) trên (0,1)
        theta ~ N(0,1): ĐỐI XỨNG quanh 0 -- không giả định trước dấu đòn bẩy, nên
                        "dữ liệu có ủng hộ đòn bẩy không" vẫn là câu hỏi mở.

CÁCH PYRO HOẠT ĐỘNG (thư viện lập trình xác suất trên PyTorch)
    - pyro.sample("ten", phan_phoi): khai báo một tham số ngẫu nhiên ("site") có
      tiên nghiệm phan_phoi. Khi chạy VB hoặc NUTS, Pyro thay giá trị site bằng
      giá trị đang thử và tự cộng log p(theta) vào mật độ.
    - pyro.factor("loglik", l): cộng thêm l vào log-mật độ -- đây là cách đưa
      log-hợp lý GARCH/EGARCH vào mô hình.
    - Pyro tự biến đổi mỗi site có miền giới hạn về R (log, logit, ...) và cộng
      log-Jacobian; VB (vb.py) và NUTS (mcmc.py) làm việc trên không gian đã biến đổi.
    - pyro.distributions: các phân phối Normal, LogNormal, Beta, Gamma;
      .expand([k]).to_event(1) biến một phân phối vô hướng thành vector k chiều độc lập.
"""
from __future__ import annotations

from dataclasses import dataclass

import pyro
import pyro.distributions as dist
import torch

from garch_vn30.mo_hinh.danh_muc import ModelSpec, loglik_cua
from garch_vn30.mo_hinh.egarch import EGARCHParams
from garch_vn30.mo_hinh.garch import GARCHParams

# Tên các site có thể xuất hiện trong mô hình (nu_minus_2 = nu - 2).
SITES = ["mu", "omega", "alpha", "theta", "beta", "gamma", "nu_minus_2"]


@dataclass(frozen=True)
class PriorConfig:
    """Siêu tham số của tiên nghiệm (xem đầu tệp)."""
    sigma_mu: float = 1.0                  # mu ~ N(0, sigma_mu)
    m_omega: float = 0.0                   # GARCH: omega ~ LogNormal(m_omega, sigma_omega)
    sigma_omega: float = 1.0               # EGARCH: omega ~ N(0, sigma_omega)
    alpha_beta_concentration: float = 2.0  # alpha, beta ~ Beta(a, a)
    sigma_gamma: float = 1.0               # gamma_j ~ N(0, sigma_gamma)
    nu_shape: float = 2.0                  # nu - 2 ~ Gamma(shape, rate)
    nu_rate: float = 0.2
    sigma_theta: float = 1.0               # theta ~ N(0, sigma_theta)


def model(r: torch.Tensor, X: torch.Tensor, spec: ModelSpec,
          prior: PriorConfig | None = None):
    """Mô hình sinh Pyro: rút tham số từ tiên nghiệm rồi cộng log-hợp lý.

    Đây là hàm DUY NHẤT mà cả VB (vb.py) và NUTS (mcmc.py) gọi. Thứ tự các lệnh
    pyro.sample cố định (mu, gamma, nu, omega, alpha, theta, beta) để cùng seed
    luôn cho cùng kết quả.

    Tham số:
        r, X : dữ liệu (lợi suất (T,), ngoại sinh (T, k)).
        spec : mô hình trong danh mục (họ, phân phối, có dùng X).
        prior: siêu tham số tiên nghiệm (mặc định PriorConfig()).
    Trả về:
        (params, loglik) để tiện kiểm tra; phần đóng góp vào hậu nghiệm nằm ở pyro.factor.
    """
    prior = prior or PriorConfig()
    k = X.shape[1]
    a = prior.alpha_beta_concentration

    mu = pyro.sample("mu", dist.Normal(0.0, prior.sigma_mu))
    if spec.use_exog and k > 0:
        gamma = pyro.sample("gamma", dist.Normal(0.0, prior.sigma_gamma).expand([k]).to_event(1))
    else:
        gamma = torch.zeros(0)
    nu = None
    if spec.dist == "student_t":
        nu = pyro.sample("nu_minus_2", dist.Gamma(prior.nu_shape, prior.nu_rate)) + 2.0

    if spec.family == "garch":
        omega = pyro.sample("omega", dist.LogNormal(prior.m_omega, prior.sigma_omega))
        alpha = pyro.sample("alpha", dist.Beta(a, a))
        beta = pyro.sample("beta", dist.Beta(a, a))
        params = GARCHParams(mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu)
    elif spec.family == "egarch":
        omega = pyro.sample("omega", dist.Normal(0.0, prior.sigma_omega))
        alpha = pyro.sample("alpha", dist.Beta(a, a))
        theta = pyro.sample("theta", dist.Normal(0.0, prior.sigma_theta))
        beta = pyro.sample("beta", dist.Beta(a, a))
        params = EGARCHParams(mu=mu, omega=omega, alpha=alpha, theta=theta, beta=beta,
                              gamma=gamma, nu=nu)
    else:
        raise ValueError(f"family khong hop le: {spec.family!r}")

    loglik = loglik_cua(spec.family)(params, r, X)     # cùng hàm log-hợp lý với MLE
    pyro.factor("loglik", loglik)                       # cộng l(theta) vào log-mật độ
    return params, loglik


def params_tu_mau(samples: dict[str, torch.Tensor], spec: ModelSpec, i: int):
    """Lấy bộ tham số thứ i trong tập mẫu hậu nghiệm (VB hoặc MCMC) thành GARCHParams/EGARCHParams."""
    gamma = samples["gamma"][i] if "gamma" in samples else torch.zeros(0)
    nu = samples["nu_minus_2"][i] + 2.0 if spec.dist == "student_t" else None
    if spec.family == "garch":
        return GARCHParams(mu=samples["mu"][i], omega=samples["omega"][i],
                           alpha=samples["alpha"][i], beta=samples["beta"][i],
                           gamma=gamma, nu=nu)
    return EGARCHParams(mu=samples["mu"][i], omega=samples["omega"][i],
                        alpha=samples["alpha"][i], theta=samples["theta"][i],
                        beta=samples["beta"][i], gamma=gamma, nu=nu)


def rut_bo_tham_so(samples: dict[str, torch.Tensor], spec: ModelSpec, n_draws: int) -> list:
    """Rút khoảng n_draws bộ tham số CÁCH ĐỀU nhau từ tập mẫu hậu nghiệm
    (dùng cho đánh giá ngoài mẫu theo hậu nghiệm, danh_gia/ngoai_mau.py)."""
    tong = samples["mu"].shape[0]
    buoc = max(1, tong // n_draws)
    return [params_tu_mau(samples, spec, i) for i in range(0, tong, buoc)][:n_draws]


def posterior_summary(samples: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """Tóm tắt hậu nghiệm cho từng site: trung bình, độ lệch chuẩn, phân vị 2,5% và 97,5%
    (hai phân vị này tạo khoảng hậu nghiệm 95%). Site rỗng (vd. gamma khi không có X) bị bỏ qua."""
    summary = {}
    for name, draws in samples.items():
        if draws.numel() == 0:
            continue
        summary[name] = {
            "mean": draws.mean(dim=0),
            "sd": draws.std(dim=0),
            "q2.5": draws.quantile(0.025, dim=0),
            "q97.5": draws.quantile(0.975, dim=0),
        }
    return summary


def sd_ratio(summary_vb: dict, summary_mcmc: dict) -> dict[str, torch.Tensor]:
    """R_sd,j = sd_VB(theta_j) / sd_MCMC(theta_j).

    MCMC được dùng làm chuẩn tham chiếu (khi đã hội tụ); R_sd < 1 nghĩa là VB báo
    bất định hậu nghiệm THẤP hơn thực tế (khoảng tin cậy quá hẹp).
    """
    return {name: summary_vb[name]["sd"] / summary_mcmc[name]["sd"]
            for name in summary_vb if name in summary_mcmc}
