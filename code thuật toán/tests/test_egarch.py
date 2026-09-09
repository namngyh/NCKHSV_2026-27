import math

import torch

from msgarchx.distributions import expected_abs_normal, expected_abs_std_t, std_t_sample
from msgarchx.models.egarch import (
    LOG_H_MAX,
    EGARCHParams,
    egarch_forward,
    egarch_loglik,
)


def _make_params():
    return EGARCHParams(
        mu=torch.tensor(0.0),
        omega=torch.tensor(-0.05),   # KHONG can duong, khac GARCH
        alpha=torch.tensor(0.15),
        theta=torch.tensor(-0.08),   # don bay: cu soc am tang bien dong nhieu hon
        beta=torch.tensor(0.90),
        gamma=torch.tensor([0.1, -0.05]),
        nu=torch.tensor(8.0),
    )


# --------------------------------------------------------------- E|z|

def test_expected_abs_std_t_hoi_tu_ve_chuan_khi_nu_lon():
    """Phep kiem tra BAT BUOC: nu -> vo cung thi E|z| cua t* -> sqrt(2/pi).

    Sai o hang so nay lam lech muc nen cua phuong trinh phuong sai ma khong
    sinh ra bat ky canh bao so hoc nao, nen phai co test rieng.
    """
    target = expected_abs_normal()
    gap_100 = abs(float(expected_abs_std_t(torch.tensor(100.0))) - target)
    gap_10000 = abs(float(expected_abs_std_t(torch.tensor(10_000.0))) - target)

    assert gap_10000 < gap_100          # cang lon cang gan
    assert gap_10000 < 1e-4             # va rat gan o nu = 10^4


def test_expected_abs_std_t_khop_mo_phong_monte_carlo():
    """Doi chieu cong thuc dong voi trung binh mau |z| sinh tu chinh t*_nu."""
    torch.manual_seed(0)
    nu = torch.tensor(6.0)
    z = std_t_sample(nu.expand(400_000))
    mc = float(z.abs().mean())
    formula = float(expected_abs_std_t(nu))
    assert abs(mc - formula) < 0.01


def test_expected_abs_normal_dung_gia_tri_giai_tich():
    assert abs(expected_abs_normal() - math.sqrt(2.0 / math.pi)) < 1e-12


# --------------------------------------------------------------- forward

def test_egarch_forward_duong_va_tat_dinh():
    torch.manual_seed(0)
    T = 200
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()

    h = egarch_forward(params, r, X, h0=torch.tensor(0.25))
    assert h.shape == (T,)
    assert (h > 0).all()
    assert torch.isfinite(h).all()

    h2 = egarch_forward(params, r, X, h0=torch.tensor(0.25))
    assert torch.allclose(h, h2)


def test_tinh_duong_giu_nguyen_khi_ngoai_sinh_am_manh():
    """h > 0 THEO CAU TRUC ke ca khi gamma^T X am manh.

    Day dung la tinh chat ma GARCH-X KHONG co: ben eq:garchx ve phai co the
    am nen phai cat san (rem:positivity). Ben EGARCH thi h = exp(...) nen
    khong co duong nao de am (rem:egarch-positivity).
    """
    torch.manual_seed(0)
    T = 150
    r = torch.randn(T) * 0.5
    X = torch.ones(T, 2)
    params = _make_params()._replace(gamma=torch.tensor([-1.0, -1.0]))  # gamma^T X = -2

    h = egarch_forward(params, r, X, h0=torch.tensor(1.0))
    assert (h > 0).all()
    assert torch.isfinite(h).all()


def test_chan_log_h_chan_duoc_tran_so():
    """Gia tri cuc doan phai bi chan lai chu khong sinh ra inf."""
    T = 50
    r = torch.zeros(T)
    X = torch.ones(T, 2)
    params = _make_params()._replace(gamma=torch.tensor([500.0, 500.0]))

    h = egarch_forward(params, r, X, h0=torch.tensor(1.0))
    assert torch.isfinite(h).all()
    assert (h <= math.exp(LOG_H_MAX) + 1.0).all()


# --------------------------------------------------------------- don bay

def test_don_bay_cu_soc_am_lam_tang_phuong_sai_nhieu_hon():
    """theta < 0: cu soc AM phai lam h tang nhieu hon cu soc DUONG cung do lon."""
    params = _make_params()._replace(theta=torch.tensor(-0.30))
    X = torch.zeros(3, 2)
    h0 = torch.tensor(1.0)
    shock = 2.0

    h_am = egarch_forward(params, torch.tensor([-shock, 0.0, 0.0]), X, h0=h0)
    h_duong = egarch_forward(params, torch.tensor([shock, 0.0, 0.0]), X, h0=h0)

    assert h_am[1] > h_duong[1]


def test_theta_bang_khong_thi_doi_xung():
    """theta = 0: mo hinh tro lai doi xung, cu soc am va duong cho h bang nhau."""
    params = _make_params()._replace(theta=torch.tensor(0.0))
    X = torch.zeros(3, 2)
    h0 = torch.tensor(1.0)
    shock = 2.0

    h_am = egarch_forward(params, torch.tensor([-shock, 0.0, 0.0]), X, h0=h0)
    h_duong = egarch_forward(params, torch.tensor([shock, 0.0, 0.0]), X, h0=h0)

    assert torch.allclose(h_am[1], h_duong[1])


# --------------------------------------------------------------- loglik

def test_egarch_loglik_huu_han_va_kha_vi():
    torch.manual_seed(1)
    T = 150
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()._replace(theta=torch.tensor(-0.08, requires_grad=True))

    ll = egarch_loglik(params, r, X, h0=torch.tensor(0.25))
    assert torch.isfinite(ll)

    ll.backward()
    assert params.theta.grad is not None
    assert torch.isfinite(params.theta.grad)


def test_egarch_loglik_chay_duoc_voi_phan_phoi_chuan():
    """nu = None -> dung normal_logpdf va E|z| = sqrt(2/pi)."""
    torch.manual_seed(2)
    T = 120
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()._replace(nu=None)

    ll = egarch_loglik(params, r, X, h0=torch.tensor(0.25))
    assert torch.isfinite(ll)
