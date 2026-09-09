"""EGARCH(1,1) va EGARCH-X mot trang thai (eq:egarchx trong bai bao).

File nay co y viet SONG SONG voi garch.py de doi chieu duoc tung dong. Khac
biet co ban nam o phuong trinh phuong sai:

    GARCH :      h_t = omega + alpha eps_{t-1}^2 + beta h_{t-1}     + gamma^T X
    EGARCH: log h_t = omega + alpha (|z_{t-1}| - E|z|)
                            + theta z_{t-1}      + beta log h_{t-1} + gamma^T X

voi z_{t-1} = eps_{t-1} / sqrt(h_{t-1}).

Hai he qua thuc te:

1. h_t = exp(...) > 0 THEO CAU TRUC. Khong can positive_variance_link, khong
   can rang buoc dau tren (omega, alpha, beta), khong can kiem tra hau nghiem
   -- xem rem:egarch-positivity. Doi lai can chan log h de exp() khong tran so.
2. z_{t-1} PHU THUOC h_{t-1}, nen khong tinh san ngoai vong lap duoc nhu
   eps_sq ben GARCH -- xem rem:egarch-cost.

Y nghia tung he so:
    alpha  -- hieu ung DO LON: phan ung voi bien do cu soc, da tru ky vong.
    theta  -- hieu ung DAU (don bay): theta < 0 nghia la cu soc am lam tang
              bien dong nhieu hon cu soc duong cung do lon.
    beta   -- do dai dang cua log-phuong sai; can |beta| < 1 de dung.

Quy uoc canh chinh du lieu giong garch.py: X[t] la vector ngoai sinh da biet
truoc thoi diem du bao r[t], tuc X_{t-1} trong ky hieu bai bao.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch

from msgarchx.distributions import (
    expected_abs_normal,
    expected_abs_std_t,
    normal_logpdf,
    std_t_logpdf,
)

# Chan log-phuong sai de exp() khong tran so. Voi loi suat tinh theo phan tram,
# h thuc te nam quanh 1-3 nen log h quanh 0-1: khoang [-30, 30] la RAT rong.
# Day thuan tuy la luoi an toan so hoc, KHONG phai rang buoc mo hinh. Neu
# nghiem toi uu cham chan nay thi do la dau hieu qua trinh toi uu di lac, can
# xem lai diem khoi tao chu khong phai noi long chan.
LOG_H_MIN = -30.0
LOG_H_MAX = 30.0


class EGARCHParams(NamedTuple):
    mu: torch.Tensor       # scalar
    omega: torch.Tensor    # scalar, KHONG rang buoc dau
    alpha: torch.Tensor    # scalar, he so hieu ung do lon, KHONG rang buoc dau
    theta: torch.Tensor    # scalar, he so don bay, KHONG rang buoc dau
    beta: torch.Tensor     # scalar, can |beta| < 1
    gamma: torch.Tensor    # (k,)
    nu: Optional[torch.Tensor] = None  # scalar > 2, None neu phan phoi chuan


def expected_abs_z(nu: Optional[torch.Tensor]) -> Union[torch.Tensor, float]:
    """E|z| ung voi phan phoi sai so: chuan neu nu is None, Student-t neu khong."""
    if nu is None:
        return expected_abs_normal()
    return expected_abs_std_t(nu)


def egarch_forward(
    params: EGARCHParams,
    r: torch.Tensor,
    X: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Tinh duong phuong sai co dieu kien h_t theo eq:egarchx.

    Tra ve h (KHONG phai log h) de dung chung giao dien voi garch_forward:
    moi ham danh gia/du bao phia sau deu lam viec tren thang phuong sai.
    h[0] la dieu kien dau, khong tinh vao likelihood.
    """
    T = r.shape[0]
    if h0 is None:
        h0 = torch.var(r).detach()

    eps = r - params.mu
    # Chi rieng phan ngoai sinh la KHONG phu thuoc h, nen tinh san duoc mot lan.
    gX = X @ params.gamma          # (T,)
    e_abs_z = expected_abs_z(params.nu)

    h_list = [h0]
    h_prev = h0
    for t in range(1, T):
        # z_{t-1} phai tinh TRONG vong lap: no phu thuoc h_{t-1} vua tinh xong.
        z_prev = eps[t - 1] / torch.sqrt(h_prev)
        log_h = (
            params.omega
            + params.alpha * (torch.abs(z_prev) - e_abs_z)
            + params.theta * z_prev
            + params.beta * torch.log(h_prev)
            + gX[t]
        )
        h_prev = torch.exp(torch.clamp(log_h, LOG_H_MIN, LOG_H_MAX))
        h_list.append(h_prev)
    return torch.stack(h_list)


def egarch_loglik(
    params: EGARCHParams,
    r: torch.Tensor,
    X: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Tong log-hop-ly tu t=1 (t=0 dung lam dieu kien dau, khong tinh vao tong).

    Cau truc giong het garch_loglik -- chi khac cach tinh h. So hang
    -0.5*log(h) la Jacobian cua phep doi bien tu z sang r.
    """
    h = egarch_forward(params, r, X, h0=h0)
    eps = r - params.mu
    z = eps / torch.sqrt(h)
    if params.nu is None:
        log_f = normal_logpdf(z) - 0.5 * torch.log(h)
    else:
        log_f = std_t_logpdf(z, params.nu) - 0.5 * torch.log(h)
    return log_f[1:].sum()
