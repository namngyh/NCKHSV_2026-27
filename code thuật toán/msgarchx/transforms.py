"""Bien doi tham so khong rang buoc <-> mien hop le.

Cac cong thuc bam theo Muc 4.2 cua bai bao:
- p_ss = sigmoid(eta_s)                       (eq:logit-transition)
- nu   = 2 + softplus(eta_nu)                 (eq:df-transform)
- omega_s = exp(log_omega_s)                  (khop voi prior log-normal, eq:prior-omega)
- alpha_s, beta_s = softplus(.) >= 0          (mien duong, khong dac ta prior rieng trong bai)

Ham `positive_variance_link` xu ly tinh huong h_t co the am khi gamma^T X_{t-1}
mang dau am (xem thao luan sau eq:garchx). Mac dinh dung floor cung ("clamp")
de giu nguyen dang cong eq:main-variance; co the doi sang "softplus" de co
rang buoc duong tuyet doi bang mot ham lien ket.
"""
from __future__ import annotations

import torch

_SOFTPLUS = torch.nn.functional.softplus


def softplus(x: torch.Tensor) -> torch.Tensor:
    return _SOFTPLUS(x)


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Nghich dao softplus, dung de khoi tao raw-param tu gia tri mong muon."""
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    return y + torch.log(-torch.expm1(-y))


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def logit(p: torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(p, dtype=torch.get_default_dtype())
    return torch.log(p) - torch.log1p(-p)


def nu_from_raw(eta_nu: torch.Tensor) -> torch.Tensor:
    """eq:df-transform: nu = 2 + softplus(eta_nu), bao dam nu > 2."""
    return 2.0 + softplus(eta_nu)


def raw_from_nu(nu: torch.Tensor) -> torch.Tensor:
    return inv_softplus(torch.as_tensor(nu, dtype=torch.get_default_dtype()) - 2.0)


def p_from_raw(eta_p: torch.Tensor) -> torch.Tensor:
    """eq:logit-transition: p_ss = sigmoid(eta_s)."""
    return sigmoid(eta_p)


def raw_from_p(p: torch.Tensor) -> torch.Tensor:
    return logit(p)


def positive_variance_link(
    h_raw: torch.Tensor,
    floor: float = 1e-8,
    mode: str = "clamp",
) -> torch.Tensor:
    """Bao dam h_t > 0 khi h_raw = omega + alpha*eps^2 + beta*h_prev + gamma'X.

    mode="clamp": giu nguyen dang cong, chi cat duoi tai `floor` (kha vi hau
        khap noi, phu hop khi gamma'X hiem khi am manh).
    mode="softplus": ham lien ket duong tuyet doi h_t = softplus(h_raw - floor) + floor,
        bao dam duong o moi diem nhung lam h_t khong con tuyen tinh voi cac
        thanh phan cong nhu trong eq:main-variance.
    """
    if mode == "clamp":
        return torch.clamp(h_raw, min=floor)
    if mode == "softplus":
        return softplus(h_raw - floor) + floor
    raise ValueError(f"mode khong hop le: {mode!r}")


# ---------------------------------------------------------------------------
# Ham lien ket cho alpha_s, beta_s
#
# "softplus" la dac ta CHINH THUC cua ban thao (eq:variance-coef-transform,
# Bai bao chinh.tex dong 877-879): alpha_s, beta_s = softplus(.) thuoc (0, +vc).
#
# "sigmoid" la bien the KIEM TRA DO BEN: alpha_s, beta_s thuoc (0, 1), trung
# voi mien cua tien nghiem Beta(a,a) ma bayes_model.py dat cho hai tham so nay.
# Voi softplus, MLE tim tren (0,+vc) con VB/MCMC tim tren (0,1) -- hai phuong
# phap lam viec tren HAI khong gian tham so khac nhau, nen so sanh M1 voi
# M2/M3/M4 khong hoan toan cung dieu kien. Bien the sigmoid xoa bo khac biet
# do, va dong thoi loai bo theo cau truc cac dinh gia co beta_s > 1.
#
# LUU Y: rang buoc alpha_s, beta_s < 1 KHONG phai yeu cau ly thuyet cua
# MS-GARCH (mot che do rieng le duoc phep no ma qua trinh tong the van dung);
# day la mot rang buoc dieu chuan do nguoi nghien cuu ap dat va PHAI duoc khai
# bao neu dung lam ket qua chinh.
#
# Chon bang bien moi truong MSGARCHX_AB_LINK = "softplus" (mac dinh) | "sigmoid".
# ---------------------------------------------------------------------------
import os as _os

AB_LINK = _os.environ.get("MSGARCHX_AB_LINK", "softplus").lower()
if AB_LINK not in ("softplus", "sigmoid"):
    raise ValueError(f"MSGARCHX_AB_LINK khong hop le: {AB_LINK!r}")


def ab_from_raw(x: torch.Tensor) -> torch.Tensor:
    """Doi eta -> alpha/beta theo ham lien ket dang chon."""
    return sigmoid(x) if AB_LINK == "sigmoid" else softplus(x)


def raw_from_ab(y: torch.Tensor) -> torch.Tensor:
    """Nghich dao cua ab_from_raw."""
    y = torch.as_tensor(y, dtype=torch.get_default_dtype())
    return logit(y) if AB_LINK == "sigmoid" else torch.log(torch.expm1(y))


def ab_init_center() -> tuple[float, float]:
    """Tam khoi tao (eta_alpha, eta_beta) cua init_raw.

    softplus: GIU NGUYEN (-1.5, +1.0) nhu code goc, de moi ket qua da tinh
      truoc do van tai lap duoc chinh xac tu ma nguon nay.
      LUU Y da biet: softplus(+1.0) = 1.313, tuc voi ham lien ket softplus thi
      MOI diem khoi tao ngau nhien deu bat dau TRONG vung no alpha+beta > 1.
      Day co the la mot nguyen nhan khien bo toi uu bi hut ve cac dinh gia co
      beta_s lon. Doi tam nay se lam thay doi ket qua cua cac lan chay cu nen
      KHONG tu y doi -- can quyet dinh rieng va khai bao neu doi.

    sigmoid: dat tam theo dung gia tri muc tieu alpha ~ 0.1, beta ~ 0.8.
    """
    if AB_LINK == "sigmoid":
        return (-2.1972, 1.3863)        # logit(0.1), logit(0.8)
    return (-1.5, 1.0)                  # nguyen ban cua code goc
