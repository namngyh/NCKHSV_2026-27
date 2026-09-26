"""Test ba phuong phap suy dien (MLE, VB, MCMC) va cac ham danh gia ngoai mau."""
import numpy as np
import pytest
import torch

from garch_vn30.danh_gia import (christoffersen_test, danh_gia_hau_nghiem, danh_gia_mot_diem,
                                 diebold_mariano_test, kupiec_test, mae_vol, qlike, rmse_vol)
from garch_vn30.mo_hinh import danh_muc
from garch_vn30.mo_hinh.garch import GARCHParams
from garch_vn30.suy_dien import fit_mcmc, fit_mle, fit_vb, khoi_tao_tu_mle, standard_errors
from garch_vn30.suy_dien.mle import pack, unpack


def _du_lieu(T=200, k=2, seed=1):
    torch.manual_seed(seed)
    return torch.randn(T) * 1.1, torch.randn(T, k) * 0.2, (torch.randn(T) * 0.3 + 1.5).abs()


# ------------------------------------------------------------- MLE
@pytest.mark.parametrize("family", ["garch", "egarch"])
def test_pack_unpack_la_nghich_dao(family):
    torch.manual_seed(0)
    raw = {"mu": torch.tensor(0.1), "eta_beta": torch.tensor(1.2), "gamma": torch.randn(2),
           "eta_nu": torch.tensor(1.5)}
    raw.update({"log_omega": torch.tensor(-1.0), "eta_alpha": torch.tensor(-1.5)} if family == "garch"
               else {"omega": torch.tensor(-0.1), "alpha": torch.tensor(0.2), "theta": torch.tensor(-0.1)})
    lai = pack(unpack(raw, "student_t", family), "student_t", family)
    assert all(torch.allclose(raw[k], lai[k]) for k in raw)


@pytest.mark.parametrize("code", ["garch_t", "egarch_x_t"])
def test_mle_so_tham_so_aic_bic(code):
    spec = danh_muc.get(code)
    r, X, _ = _du_lieu(T=150)
    X = danh_muc.lay_X(spec, X)
    res = fit_mle(r, X, dist=spec.dist, family=spec.family, n_starts=1, n_adam=30, n_lbfgs=10)
    d = spec.n_params(2)
    assert res.n_params == d
    assert abs(res.aic - (2 * d - 2 * res.loglik)) < 1e-9
    assert abs(res.bic - (d * np.log(150) - 2 * res.loglik)) < 1e-9


def test_mle_tai_lap_voi_cung_seed():
    r, X, _ = _du_lieu(T=150)
    a = fit_mle(r, X, family="egarch", n_starts=2, n_adam=30, n_lbfgs=10)
    b = fit_mle(r, X, family="egarch", n_starts=2, n_adam=30, n_lbfgs=10)
    assert a.loglik == b.loglik


def test_sai_so_chuan_du_ten_va_duong():
    r, X, _ = _du_lieu(T=300)
    res = fit_mle(r, X, family="egarch", n_starts=1, n_adam=100, n_lbfgs=50)
    se = standard_errors(res.params, r, X, family="egarch")
    if se is not None:                        # None hop le khi Hessian khong xac dinh duong
        assert set(se) == {"mu", "omega", "alpha", "theta", "beta", "gamma", "nu"}
        assert se["gamma"].shape == (2,)
        assert all(float(torch.as_tensor(v).min()) >= 0 for v in se.values())


# ------------------------------------------------------------- VB, MCMC
@pytest.mark.parametrize("code", [s.code for s in danh_muc.CATALOGUE])
def test_vb_moi_mo_hinh_co_dung_site(code):
    spec = danh_muc.get(code)
    r, X, _ = _du_lieu(T=120)
    res = fit_vb(r, danh_muc.lay_X(spec, X), spec, n_steps=15, n_posterior_draws=10)
    assert np.isfinite(res.final_elbo)
    assert ("theta" in res.samples) == (spec.family == "egarch")
    assert ("gamma" in res.samples) == spec.use_exog
    assert ("nu_minus_2" in res.samples) == (spec.dist == "student_t")


def test_mcmc_chay_va_chuoi_0_bat_dau_tai_mle():
    spec = danh_muc.get("egarch_t")
    r, _, _ = _du_lieu(T=100)
    X = torch.zeros(100, 0)
    p = fit_mle(r, X, family="egarch", n_starts=1, n_adam=30, n_lbfgs=10).params
    init0 = khoi_tao_tu_mle(p, spec, chain=0)
    assert torch.allclose(init0["theta"], p.theta) and torch.allclose(init0["beta"], p.beta)
    res = fit_mcmc(r, X, spec, n_chains=2, n_warmup=5, n_samples=5, max_tree_depth=3,
                   initial_params=lambda c: khoi_tao_tu_mle(p, spec, c))
    assert res.samples["theta"].shape == (10,)
    assert np.isfinite(res.max_rhat)          # R-hat that su duoc tinh, khong am tham ra NaN


# ------------------------------------------------------------- danh gia
def test_ham_mat_mat_khi_du_bao_hoan_hao():
    rv = torch.tensor([1.0, 2.0, 0.5])
    assert mae_vol(rv, rv) == 0 and rmse_vol(rv, rv) == 0
    assert abs(float(qlike(rv, rv)) - float(np.mean(np.log(rv.numpy()) + 1))) < 1e-12
    assert qlike(torch.tensor([1.0]), torch.tensor([0.1])) > qlike(torch.tensor([1.0]), torch.tensor([1.0]))


def test_kupiec_bac_bo_ty_le_sai_va_chap_nhan_ty_le_dung():
    hits = np.zeros(1000)
    hits[:100] = 1
    assert kupiec_test(hits, 0.01)["p_value"] < 0.01
    dung = (np.random.default_rng(0).uniform(size=5000) < 0.01).astype(float)
    assert kupiec_test(dung, 0.01)["p_value"] > 0.01


def test_christoffersen_phat_hien_vi_pham_theo_cum():
    hits = np.zeros(200)
    hits[50:60] = 1
    assert christoffersen_test(hits, 0.01)["p_value_ind"] < 0.01


def test_diebold_mariano_chon_mo_hinh_mat_mat_thap():
    rng = np.random.default_rng(0)
    res = diebold_mariano_test(rng.normal(1.0, 0.1, 500), rng.normal(0.5, 0.1, 500))
    assert res["DM_stat"] > 0 and res["p_value"] < 0.01


def test_danh_gia_hau_nghiem_mot_mau_trung_ban_mot_diem():
    """Voi DUNG MOT mau, mat do hon hop rut ve mat do cua chinh mau do."""
    r, X, rv = _du_lieu()
    p = GARCHParams(mu=torch.tensor(0.0), omega=torch.tensor(0.1), alpha=torch.tensor(0.08),
                    beta=torch.tensor(0.85), gamma=torch.tensor([0.05, -0.02]), nu=torch.tensor(8.0))
    a = danh_gia_mot_diem(p, "garch", "student_t", r, X, rv, 150)
    b = danh_gia_hau_nghiem([p], "garch", "student_t", r, X, rv, 150)
    assert a["n_test"] == 50
    for k in ("qlike", "nls", "mae", "n_violations_1pct", "n_violations_5pct"):
        assert abs(a[k] - b[k]) < 1e-12
