"""Test phan phoi sai so, phuong trinh GARCH/EGARCH va danh muc mo hinh."""
import math

import numpy as np
import pytest
import torch
from scipy import stats

from garch_vn30.mo_hinh import danh_muc
from garch_vn30.mo_hinh.egarch import LOG_H_MAX, EGARCHParams, egarch_forward, egarch_loglik
from garch_vn30.mo_hinh.garch import GARCHParams, garch_forward, garch_loglik
from garch_vn30.mo_hinh.phan_phoi import (expected_abs_normal, expected_abs_std_t, normal_logpdf,
                                          std_t_logpdf, std_t_sample)


def _garch():
    return GARCHParams(mu=torch.tensor(0.0), omega=torch.tensor(0.05), alpha=torch.tensor(0.08),
                       beta=torch.tensor(0.85), gamma=torch.tensor([0.1, -0.05]), nu=torch.tensor(8.0))


def _egarch():
    return EGARCHParams(mu=torch.tensor(0.0), omega=torch.tensor(-0.05), alpha=torch.tensor(0.15),
                        theta=torch.tensor(-0.08), beta=torch.tensor(0.90),
                        gamma=torch.tensor([0.1, -0.05]), nu=torch.tensor(8.0))


def _du_lieu(T=200, seed=0):
    torch.manual_seed(seed)
    return torch.randn(T) * 0.5, torch.randn(T, 2) * 0.1


# ------------------------------------------------------------- phan phoi
def test_mat_do_student_t_khop_scipy():
    z = torch.linspace(-3, 3, 25)
    ky_vong = stats.t.logpdf(z.numpy(), df=6.0, scale=np.sqrt(4.0 / 6.0))
    assert np.allclose(std_t_logpdf(z, torch.tensor(6.0)).numpy(), ky_vong, atol=1e-10)


def test_student_t_hoi_tu_ve_chuan_khi_nu_lon():
    z = torch.linspace(-2, 2, 9)
    assert torch.allclose(std_t_logpdf(z, torch.tensor(1e6)), normal_logpdf(z), atol=1e-4)


def test_mau_student_t_co_phuong_sai_1():
    torch.manual_seed(0)
    z = std_t_sample(torch.full((200_000,), 6.0))
    assert abs(z.mean().item()) < 0.02 and abs(z.var().item() - 1.0) < 0.05


def test_E_abs_z_hoi_tu_ve_sqrt_2_chia_pi():
    """Kiem tra BAT BUOC: sai hang so nay lam lech muc nen cua EGARCH ma khong bao loi."""
    gap = abs(float(expected_abs_std_t(torch.tensor(1e4))) - math.sqrt(2 / math.pi))
    assert gap < 1e-4
    assert abs(expected_abs_normal() - math.sqrt(2 / math.pi)) < 1e-12


def test_E_abs_z_khop_monte_carlo():
    torch.manual_seed(0)
    nu = torch.tensor(6.0)
    assert abs(float(std_t_sample(nu.expand(400_000)).abs().mean()) - float(expected_abs_std_t(nu))) < 0.01


# ------------------------------------------------------------- GARCH
def test_garch_phuong_sai_duong_va_tat_dinh():
    r, X = _du_lieu()
    h = garch_forward(_garch(), r, X, h0=torch.tensor(0.25))
    assert h.shape == (200,) and (h > 0).all()
    assert torch.equal(h, garch_forward(_garch(), r, X, h0=torch.tensor(0.25)))


def test_garch_loglik_khop_cong_thuc_viet_tay():
    """Tinh lai l bang vong lap numpy doc lap, so voi garch_loglik."""
    r, X = _du_lieu(T=50)
    p = _garch()
    h, eps = [0.25], (r - p.mu).numpy()
    for t in range(1, 50):
        h.append(max(0.05 + 0.08 * eps[t - 1] ** 2 + 0.85 * h[t - 1] + float(X[t] @ p.gamma), 1e-8))
    h = np.array(h)
    ll = np.sum(stats.t.logpdf(eps[1:] / np.sqrt(h[1:]), df=8, scale=np.sqrt(6 / 8)) - 0.5 * np.log(h[1:]))
    assert abs(float(garch_loglik(p, r, X, h0=torch.tensor(0.25))) - ll) < 1e-9


def test_garch_loglik_kha_vi():
    r, X = _du_lieu()
    p = _garch()._replace(omega=torch.tensor(0.05, requires_grad=True))
    garch_loglik(p, r, X).backward()
    assert torch.isfinite(p.omega.grad)


# ------------------------------------------------------------- EGARCH
def test_egarch_duong_ke_ca_khi_ngoai_sinh_am_manh():
    r, _ = _du_lieu(T=150)
    h = egarch_forward(_egarch()._replace(gamma=torch.tensor([-1.0, -1.0])), r, torch.ones(150, 2),
                       h0=torch.tensor(1.0))
    assert (h > 0).all() and torch.isfinite(h).all()


def test_egarch_chan_log_h_chong_tran_so():
    h = egarch_forward(_egarch()._replace(gamma=torch.tensor([500.0, 500.0])), torch.zeros(50),
                       torch.ones(50, 2), h0=torch.tensor(1.0))
    assert torch.isfinite(h).all() and (h <= math.exp(LOG_H_MAX) + 1).all()


@pytest.mark.parametrize("theta, am_lon_hon", [(-0.3, True), (0.0, None)])
def test_hieu_ung_don_bay(theta, am_lon_hon):
    """theta < 0: cu soc am tang h nhieu hon cu soc duong; theta = 0: doi xung."""
    p, X, h0 = _egarch()._replace(theta=torch.tensor(theta)), torch.zeros(3, 2), torch.tensor(1.0)
    h_am = egarch_forward(p, torch.tensor([-2.0, 0.0, 0.0]), X, h0=h0)[1]
    h_duong = egarch_forward(p, torch.tensor([2.0, 0.0, 0.0]), X, h0=h0)[1]
    assert (h_am > h_duong) if am_lon_hon else torch.allclose(h_am, h_duong)


def test_egarch_loglik_huu_han_ca_hai_phan_phoi():
    r, X = _du_lieu(T=120)
    assert torch.isfinite(egarch_loglik(_egarch(), r, X))
    assert torch.isfinite(egarch_loglik(_egarch()._replace(nu=None), r, X))


# ------------------------------------------------------------- danh muc
def test_so_tham_so_theo_danh_muc():
    mong_doi = {"garch_n": 4, "garch_t": 5, "garch_x_t": 7, "egarch_n": 5, "egarch_t": 6, "egarch_x_t": 8}
    assert {s.code: s.n_params(2) for s in danh_muc.CATALOGUE} == mong_doi


def test_get_bao_loi_khi_sai_ten():
    with pytest.raises(KeyError, match="Khong co mo hinh"):
        danh_muc.get("ms_garch_x_t")


def test_tham_so_dict_hai_chieu():
    for p, fam in ((_garch(), "garch"), (_egarch(), "egarch")):
        q = danh_muc.tham_so_tu_dict(danh_muc.tham_so_sang_dict(p), fam)
        assert all(torch.equal(a, b) for a, b in zip(p, q))
