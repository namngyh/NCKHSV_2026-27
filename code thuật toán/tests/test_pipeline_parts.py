"""Test cho cac thanh phan moi cua pipeline: danh muc mo hinh, tang Bayes mot
trang thai, va danh gia ngoai mau."""
import pytest
import torch

from msgarchx.evaluation.oos_single import (
    evaluate_single_regime,
    evaluate_single_regime_posterior,
)
from msgarchx.inference import model_spec
from msgarchx.inference.bayes_single import model as single_model
from msgarchx.inference.garch_mle import fit_garch_mle
from msgarchx.inference.variational import fit_vb
from msgarchx.models.egarch import EGARCHParams
from msgarchx.models.garch import GARCHParams


# ------------------------------------------------------------ danh muc

# So tham so ky vong, doi chieu voi bang trong bai bao (k = 2 bien ngoai sinh).
SO_THAM_SO_MONG_DOI = {
    "garch_n": 4, "garch_t": 5, "garch_x_t": 7,
    "egarch_n": 5, "egarch_t": 6, "egarch_x_t": 8,
    "ms_garch_n": 10, "ms_garch_x_t": 15,
}


def test_so_tham_so_khop_bai_bao():
    for spec in model_spec.CATALOGUE:
        assert spec.n_params(2) == SO_THAM_SO_MONG_DOI[spec.code], spec.code


def test_egarch_luon_hon_garch_dung_mot_tham_so_moi_trang_thai():
    """Chenh lech giua hai ho phai dung bang so trang thai (he so don bay)."""
    for garch_code, egarch_code, n_regimes in [
        ("garch_n", "egarch_n", 1),
        ("garch_t", "egarch_t", 1),
        ("garch_x_t", "egarch_x_t", 1),
    ]:
        g, e = model_spec.get(garch_code), model_spec.get(egarch_code)
        assert e.n_params(2) - g.n_params(2) == n_regimes


def test_get_bao_loi_ro_rang_khi_sai_ten():
    with pytest.raises(KeyError, match="Khong co mo hinh"):
        model_spec.get("khong_ton_tai")


def test_danh_muc_khong_trung_code():
    codes = [s.code for s in model_spec.CATALOGUE]
    assert len(codes) == len(set(codes))


# --------------------------------------------------- tang Bayes 1 trang thai

@pytest.mark.parametrize("code", [s.code for s in model_spec.CATALOGUE if s.n_regimes == 1])
def test_vb_chay_duoc_cho_moi_mo_hinh_mot_trang_thai(code):
    """VB phai chay va tra ve dung cac site ma dac ta doi hoi."""
    spec = model_spec.get(code)
    torch.manual_seed(0)
    T = 120
    r = torch.randn(T) * 1.0
    X = torch.randn(T, 2) * 0.2 if spec.use_exog else torch.zeros(T, 0)

    res = fit_vb(r, X, config=None, guide_type="mean_field", n_steps=15,
                 n_posterior_draws=10, seed=0,
                 model_fn=single_model, model_args=(spec, None))

    assert torch.isfinite(torch.tensor(res.final_elbo))
    assert ("theta" in res.samples) == (spec.family == "egarch")
    assert ("gamma" in res.samples) == spec.use_exog
    assert ("nu_minus_2" in res.samples) == (spec.dist == "student_t")


# ------------------------------------------------------- danh gia ngoai mau

def _du_lieu_gia(T=200, k=2):
    torch.manual_seed(1)
    r = torch.randn(T) * 1.1
    X = torch.randn(T, k) * 0.2
    rv = (torch.randn(T) * 0.3 + 1.5).abs()
    return r, X, rv


def test_danh_gia_ngoai_mau_garch_tra_ve_du_chi_tieu():
    r, X, rv = _du_lieu_gia()
    params = GARCHParams(
        mu=torch.tensor(0.0), omega=torch.tensor(0.1), alpha=torch.tensor(0.08),
        beta=torch.tensor(0.85), gamma=torch.tensor([0.05, -0.02]), nu=torch.tensor(8.0),
    )
    out = evaluate_single_regime(params, "garch", "student_t", r, X, rv, n_train=150)

    for khoa in ["mae", "rmse", "qlike", "nls", "n_test", "kupiec_1pct", "n_violations_5pct"]:
        assert khoa in out
    assert out["n_test"] == 50
    for khoa in ["mae", "rmse", "qlike", "nls"]:
        assert torch.isfinite(torch.tensor(out[khoa]))


def test_danh_gia_ngoai_mau_egarch_chay_duoc():
    r, X, rv = _du_lieu_gia()
    params = EGARCHParams(
        mu=torch.tensor(0.0), omega=torch.tensor(-0.05), alpha=torch.tensor(0.15),
        theta=torch.tensor(-0.08), beta=torch.tensor(0.90),
        gamma=torch.tensor([0.05, -0.02]), nu=torch.tensor(8.0),
    )
    out = evaluate_single_regime(params, "egarch", "student_t", r, X, rv, n_train=150)
    assert torch.isfinite(torch.tensor(out["qlike"]))
    assert (out["h_hat"] > 0).all()


def test_danh_gia_theo_hau_nghiem_khop_ban_mot_diem_khi_chi_co_mot_mau():
    """Voi DUNG MOT mau hau nghiem, mat do hon hop rut ve mat do cua chinh mau
    do, nen hai ham phai cho cung ket qua. Day la phep kiem tra tinh nhat quan
    giua hai duong danh gia."""
    r, X, rv = _du_lieu_gia()
    params = GARCHParams(
        mu=torch.tensor(0.0), omega=torch.tensor(0.1), alpha=torch.tensor(0.08),
        beta=torch.tensor(0.85), gamma=torch.tensor([0.05, -0.02]), nu=torch.tensor(8.0),
    )
    mot_diem = evaluate_single_regime(params, "garch", "student_t", r, X, rv, 150)
    hau_nghiem = evaluate_single_regime_posterior([params], "garch", "student_t", r, X, rv, 150)

    assert abs(mot_diem["qlike"] - hau_nghiem["qlike"]) < 1e-10
    assert abs(mot_diem["nls"] - hau_nghiem["nls"]) < 1e-10


# ------------------------------------------- khoi phuc tham so cho EGARCH

def test_sinh_du_lieu_egarch_hop_le_va_tai_lap_duoc():
    from msgarchx.simulation.dgp import simulate_egarch
    from msgarchx.simulation.run_egarch import default_egarch_scenarios

    scn = default_egarch_scenarios(T=300, B=1)["E1"]
    a = simulate_egarch(scn.params, T=300, dist="student_t", n_exog=2, burn_in=50, seed=7)
    b = simulate_egarch(scn.params, T=300, dist="student_t", n_exog=2, burn_in=50, seed=7)

    assert a.r.shape == (300,) and a.X.shape == (300, 2)
    assert torch.isfinite(a.r).all()
    assert (a.h > 0).all()                      # tinh duong theo cau truc
    assert torch.allclose(a.r, b.r)             # cung seed -> cung du lieu


def test_ba_kich_ban_egarch_dung_thiet_ke():
    """E2 phai co theta = 0 -- day la kich ban kiem tra don bay gia."""
    from msgarchx.simulation.run_egarch import default_egarch_scenarios

    scn = default_egarch_scenarios()
    assert set(scn) == {"E1", "E2", "E3"}
    assert float(scn["E2"].params.theta) == 0.0
    assert float(scn["E1"].params.theta) < 0        # don bay: cu soc am tang bien dong
    assert float(scn["E3"].params.nu) < float(scn["E1"].params.nu)   # E3 duoi day hon


def test_khoi_phuc_tham_so_egarch_tra_ve_dung_cau_truc():
    from msgarchx.simulation.run_egarch import default_egarch_scenarios, run_egarch_scenario

    spec = model_spec.get("egarch_x_t")
    scn = default_egarch_scenarios(T=150, B=2)["E2"]
    out = run_egarch_scenario(
        scn, spec, methods=("mle", "vb_mean_field"), base_seed=0,
        method_kwargs={"mle": {"n_starts": 1, "n_adam": 15, "n_lbfgs": 5},
                       "vb_mean_field": {"n_steps": 15, "n_posterior_draws": 10}},
    )

    assert set(out) == {"mle", "vb_mean_field"}
    for m, blk in out.items():
        assert blk["n_replications"] == 2
        assert "theta" in blk["bias"] and "theta" in blk["rmse"]
        assert blk["tham_so_that"]["theta"] == 0.0
    # MLE khong sinh khoang tin cay -> khong co coverage; VB thi co.
    assert out["mle"]["coverage"] == {}
    assert "theta" in out["vb_mean_field"]["coverage"]


def test_mle_egarch_cho_so_tham_so_dung():
    r, X, rv = _du_lieu_gia(T=150)
    res = fit_garch_mle(r, X, dist="student_t", family="egarch",
                        n_starts=1, n_adam=20, n_lbfgs=10)
    assert res.n_params == model_spec.get("egarch_x_t").n_params(2)
    assert torch.isfinite(torch.tensor(res.loglik))
