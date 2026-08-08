"""MS-GARCH-X Student-t: bo loc tien (Phu luc A) va logic thu gon kieu Gray.

Day la mo hinh chinh cua bai bao (eq:main-mean .. eq:main-variance). Ham
`ms_garch_forward_filter` la NOI DUY NHAT tinh log-likelihood; MLE, Variational
Bayes va MCMC deu goi lai ham nay (qua inference/param_spec.py va
inference/bayes_model.py) de dam bao ca ba phuong phap dung cung mot ham hop ly
va cung mot co che thu gon (checklist Phu luc B, muc 7 trong ban thao).

Quy uoc canh chinh du lieu giong garch.py: X[t] la vector ngoai sinh da biet
truoc thoi diem du bao r[t] (tuc la X_{t-1} trong ky hieu bai bao).
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from msgarchx.distributions import std_t_logpdf
from msgarchx.transforms import positive_variance_link


class MSGARCHXParams(NamedTuple):
    mu: torch.Tensor       # (2,)
    omega: torch.Tensor    # (2,) > 0
    alpha: torch.Tensor    # (2,) >= 0
    beta: torch.Tensor     # (2,) >= 0
    gamma: torch.Tensor    # (2, k)
    P: torch.Tensor        # (2, 2) hang ngau nhien, P[i, j] = p_ij
    nu: torch.Tensor       # scalar (mo hinh chinh) hoac (2,) (mo hinh mo rong R1)

    @property
    def state_specific_nu(self) -> bool:
        return self.nu.dim() > 0 and self.nu.shape[0] == 2

    def nu_for_state(self, s: int) -> torch.Tensor:
        return self.nu[s] if self.state_specific_nu else self.nu


def stationary_distribution(P: torch.Tensor) -> torch.Tensor:
    """Giai bpi^T = bpi^T P, bpi_1+bpi_2=1 (eq:stationary-markov) cho ma tran 2x2."""
    p12 = P[0, 1]
    p21 = P[1, 0]
    denom = p12 + p21
    pi1 = p21 / denom
    return torch.stack([pi1, 1.0 - pi1])


def expected_duration(P: torch.Tensor) -> torch.Tensor:
    """d_s = 1/(1-p_ss), eq:expected-duration."""
    return 1.0 / (1.0 - torch.diagonal(P))


def ms_garch_forward_filter(
    params: MSGARCHXParams,
    r: torch.Tensor,
    X: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    xi0: Optional[torch.Tensor] = None,
    variance_link: str = "clamp",
    floor: float = 1e-8,
) -> dict:
    """Thuc hien bo loc tien kieu Gray mo ta trong Phu luc A.

    Tra ve dict gom:
      loglik        : tensor scalar, tong log-hop-ly tu t=1 (t=0 la dieu kien dau)
      xi_filt        : (T, 2) xac suat loc xi_{t|t}
      xi_pred        : (T, 2) xac suat du bao xi_{t|t-1} (xi_pred[0] khong dinh nghia, dat = xi0)
      h              : (T, 2) phuong sai co dieu kien theo tung trang thai h_{t,s}
      collapsed_mean : (T,) trung binh hon hop bar_mu_t
      collapsed_var  : (T,) phuong sai hon hop bar_h_t
    """
    T = r.shape[0]
    dtype = r.dtype
    if xi0 is None:
        xi0 = stationary_distribution(params.P).detach()
    if h0 is None:
        h0 = torch.var(r).detach().expand(2).clone()

    mu, omega, alpha, beta, gamma = params.mu, params.omega, params.alpha, params.beta, params.gamma

    bar_mu0 = (xi0 * mu).sum()
    bar_h0 = (xi0 * (h0 + mu**2)).sum() - bar_mu0**2
    bar_eps0 = r[0] - bar_mu0

    xi_filt_list = [xi0]
    xi_pred_list = [xi0]
    h_list = [h0]
    bar_mu_list = [bar_mu0]
    bar_h_list = [bar_h0]
    bar_eps_list = [bar_eps0]

    loglik = torch.zeros((), dtype=dtype)

    for t in range(1, T):
        xi_pred_t = xi_filt_list[t - 1] @ params.P  # eq:predicted-state-prob
        bar_eps_prev = bar_eps_list[t - 1]
        bar_h_prev = bar_h_list[t - 1]

        h_raw = (
            omega
            + alpha * bar_eps_prev**2
            + beta * bar_h_prev
            + gamma @ X[t]
        )
        h_t = positive_variance_link(h_raw, floor=floor, mode=variance_link)

        z_t = (r[t] - mu) / torch.sqrt(h_t)
        if params.state_specific_nu:
            log_f = torch.stack(
                [std_t_logpdf(z_t[s], params.nu[s]) for s in range(2)]
            )
        else:
            log_f = std_t_logpdf(z_t, params.nu)
        log_f = log_f - 0.5 * torch.log(h_t)

        log_joint = torch.log(xi_pred_t) + log_f
        loglik_t = torch.logsumexp(log_joint, dim=0)
        xi_filt_t = torch.exp(log_joint - loglik_t)  # eq:filtered-state-prob

        bar_mu_t = (xi_filt_t * mu).sum()
        bar_h_t = (xi_filt_t * (h_t + mu**2)).sum() - bar_mu_t**2
        bar_eps_t = r[t] - bar_mu_t

        loglik = loglik + loglik_t
        xi_filt_list.append(xi_filt_t)
        xi_pred_list.append(xi_pred_t)
        h_list.append(h_t)
        bar_mu_list.append(bar_mu_t)
        bar_h_list.append(bar_h_t)
        bar_eps_list.append(bar_eps_t)

    return {
        "loglik": loglik,
        "xi_filt": torch.stack(xi_filt_list),
        "xi_pred": torch.stack(xi_pred_list),
        "h": torch.stack(h_list),
        "collapsed_mean": torch.stack(bar_mu_list),
        "collapsed_var": torch.stack(bar_h_list),
    }


def ms_garch_smoother(xi_filt: torch.Tensor, xi_pred: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Bo lam tron kieu Kim (1994): xi_{t|T}(i) = xi_filt[t,i] * sum_j P[i,j] xi_smooth[t+1,j] / xi_pred[t+1,j].

    Ap dung cho HMM co xac suat loc/du bao da tinh san; day la mot xap xi hop
    ly cho MS-GARCH thu gon vi likelihood chi phu thuoc xi_filt (khong phu
    thuoc toan bo duong di trang thai). Xem eq:smoothed-state-prob.
    """
    T = xi_filt.shape[0]
    xi_smooth = [None] * T
    xi_smooth[T - 1] = xi_filt[T - 1]
    for t in range(T - 2, -1, -1):
        ratio = xi_smooth[t + 1] / xi_pred[t + 1]
        xi_smooth[t] = xi_filt[t] * (P @ ratio)
    return torch.stack(xi_smooth)


def label_switch_correction(params: MSGARCHXParams, h: torch.Tensor) -> MSGARCHXParams:
    """Sap xep lai trang thai 1/2 de thoa eq:label-identification: E(h|S=1) < E(h|S=2).

    Dung trung binh theo thoi gian cua duong h_{t,s} lam uoc luong cua E(h_t|S_t=s).
    """
    mean_h = h.mean(dim=0)
    if mean_h[0] <= mean_h[1]:
        return params
    perm = torch.tensor([1, 0])
    P_swapped = params.P[perm][:, perm]
    nu_swapped = params.nu[perm] if params.state_specific_nu else params.nu
    return MSGARCHXParams(
        mu=params.mu[perm],
        omega=params.omega[perm],
        alpha=params.alpha[perm],
        beta=params.beta[perm],
        gamma=params.gamma[perm],
        P=P_swapped,
        nu=nu_swapped,
    )
