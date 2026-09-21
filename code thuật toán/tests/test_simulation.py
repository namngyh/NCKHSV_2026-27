import pytest
import numpy as np
import torch

from msgarchx.inference.param_spec import ModelConfig
from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.simulation.dgp import simulate_ms_garch_x
from msgarchx.simulation.metrics import bias, brier_state, coverage, parameter_rmse
from msgarchx.simulation.run_simulation import run_scenario
from msgarchx.simulation.scenarios import SimulationScenario, default_scenarios


def _make_params():
    return MSGARCHXParams(
        mu=torch.tensor([0.0, 0.0]),
        omega=torch.tensor([0.05, 0.30]),
        alpha=torch.tensor([0.05, 0.10]),
        beta=torch.tensor([0.85, 0.80]),
        gamma=torch.tensor([[0.05, -0.02], [0.10, 0.03]]),
        P=torch.tensor([[0.97, 0.03], [0.05, 0.95]]),
        nu=torch.tensor(8.0),
    )


def test_simulate_ms_garch_x_shapes_and_validity():
    params = _make_params()
    P0 = params.P
    sim = simulate_ms_garch_x(params, P0, T=300, n_exog=2, burn_in=100, seed=42)
    assert sim.r.shape == (300,)
    assert sim.h.shape == (300,)
    assert sim.state.shape == (300,)
    assert sim.X.shape == (300, 2)
    assert (sim.h > 0).all()
    assert set(sim.state.unique().tolist()).issubset({0, 1})


def test_simulate_is_reproducible_with_seed():
    params = _make_params()
    sim1 = simulate_ms_garch_x(params, params.P, T=200, n_exog=2, seed=7)
    sim2 = simulate_ms_garch_x(params, params.P, T=200, n_exog=2, seed=7)
    assert torch.allclose(sim1.r, sim2.r)


def test_default_scenarios_cover_s1_to_s8():
    scenarios = default_scenarios(T_default=100, B_default=2)
    assert set(scenarios.keys()) == {f"S{i}" for i in range(1, 9)}
    for scenario in scenarios.values():
        sim = simulate_ms_garch_x(
            scenario.params, scenario.P0, T=scenario.T, n_exog=2, burn_in=20, seed=1
        )
        assert torch.isfinite(sim.r).all()


def test_metrics_basic_formulas():
    theta_hat = np.array([1.1, 0.9, 1.0, 1.2])
    assert abs(bias(theta_hat, 1.0) - 0.05) < 1e-8
    assert parameter_rmse(theta_hat, 1.0) > 0

    lo, hi = np.array([0.5, 1.5]), np.array([1.5, 1.9])
    assert coverage(lo, hi, 1.0) == 0.5

    xi_hat = np.array([[0.9, 0.1], [0.2, 0.8]])
    true_state = np.array([0, 1])
    assert brier_state(xi_hat, true_state) < 0.05


# ---------------------------------------------------------------------------
# Sap nhan trang thai truoc khi gop (doi xung hoan vi nhan)
# ---------------------------------------------------------------------------
def _rep_mau(omega1, omega2, method="vb_mean_field"):
    from msgarchx.simulation.run_simulation import SimulationReplicationResult
    est = {"mu1": 0.1, "mu2": 0.2, "omega1": omega1, "omega2": omega2,
           "alpha1": 0.05, "alpha2": 0.10, "beta1": 0.85, "beta2": 0.80,
           "p11": 0.98, "p22": 0.95, "gamma1_0": 1.0, "gamma1_1": 2.0,
           "gamma2_0": 3.0, "gamma2_1": 4.0, "nu": 8.0}
    lo = {k: v - 1.0 for k, v in est.items()}
    hi = {k: v + 1.0 for k, v in est.items()}
    return SimulationReplicationResult(method, est, lo, hi, 0.3)


def test_sap_nhan_giu_nguyen_khi_dung_thu_tu():
    """Trang thai 1 da la bien dong thap -> khong duoc dong vao."""
    from msgarchx.simulation.run_simulation import sap_nhan_trang_thai
    rep = _rep_mau(0.05, 0.40)
    assert sap_nhan_trang_thai(rep).point_estimate == rep.point_estimate


def test_sap_nhan_hoan_vi_khi_dao_thu_tu():
    """Trang thai 1 dang la bien dong CAO -> phai doi cho ca diem lan khoang."""
    from msgarchx.simulation.run_simulation import sap_nhan_trang_thai
    ra = sap_nhan_trang_thai(_rep_mau(0.40, 0.05)).point_estimate
    assert ra["omega1"] == 0.05 and ra["omega2"] == 0.40
    assert ra["alpha1"] == 0.10 and ra["beta1"] == 0.80      # di kem trang thai
    assert ra["p11"] == 0.95 and ra["p22"] == 0.98           # p_ss doi cho nhau
    assert ra["gamma1_0"] == 3.0 and ra["gamma2_0"] == 1.0   # ca hang gamma
    assert ra["nu"] == 8.0                                    # vo huong, khong doi


def test_sap_nhan_hoan_vi_ca_khoang_tin_cay():
    """Coverage doc tu ci_lower/ci_upper nen chung phai duoc hoan vi CUNG NHAU."""
    from msgarchx.simulation.run_simulation import sap_nhan_trang_thai
    ra = sap_nhan_trang_thai(_rep_mau(0.40, 0.05))
    assert ra.ci_lower["omega1"] == 0.05 - 1.0
    assert ra.ci_upper["omega1"] == 0.05 + 1.0


def test_sap_nhan_bo_qua_mo_hinh_mot_trang_thai():
    """EGARCH mot trang thai khong co khoa omega1/omega2 -> tra ve nguyen ven."""
    from msgarchx.simulation.run_simulation import (
        SimulationReplicationResult, sap_nhan_trang_thai)
    rep = SimulationReplicationResult("mle", {"omega": 0.04, "theta": -0.1}, None, None, None)
    assert sap_nhan_trang_thai(rep) is rep


def test_sap_nhan_luy_dang():
    """Sap nhan hai lan phai bang sap nhan mot lan (idempotent)."""
    from msgarchx.simulation.run_simulation import sap_nhan_trang_thai
    mot = sap_nhan_trang_thai(_rep_mau(0.40, 0.05))
    hai = sap_nhan_trang_thai(mot)
    assert mot.point_estimate == hai.point_estimate


def test_run_scenario_true_p11_p22_come_from_P0_not_placeholder():
    """`scenario.params.P` chi la placeholder (xem docstring `_base_params`
    trong scenarios.py, "khong dung truc tiep, P0 rieng duoc truyen vao
    DGP"). Bias/RMSE cua p11/p22 phai tinh so voi `scenario.P0` -- KHONG
    phai `scenario.params.P`.

    Regression cho loi da tim thay va sua (10/2026): `run_scenario` truoc do
    goi `_named_params(scenario.params)` truc tiep de lay theta_0, khien
    Bias/RMSE cua p11/p22 tren TOAN BO S1-S8 da bao cao bi tinh nham so voi
    p11=p22=1.0 (eye(2)) thay vi P0 that. Vi DGP chi doc P0 (khong doc
    params.P), hai kich ban duoi day sinh CUNG MOT du lieu (cung seed, cung
    P0) nhung co placeholder params.P khac nhau -- Bias phai GIONG HET nhau.
    """
    P0_that = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    base = _make_params()
    config = ModelConfig(n_exog=2, state_specific_nu=False)
    method_kwargs = {"mle": {"n_starts": 1, "n_adam": 30, "n_lbfgs": 10}}

    scn_eye = SimulationScenario(
        code="T1", description="placeholder eye(2)",
        params=base._replace(P=torch.eye(2)), P0=P0_that,
        T=100, B=1, burn_in=20,
    )
    scn_khac = SimulationScenario(
        code="T2", description="placeholder khac",
        params=base._replace(P=torch.tensor([[0.1, 0.9], [0.9, 0.1]])), P0=P0_that,
        T=100, B=1, burn_in=20,
    )

    out1 = run_scenario(scn_eye, config, methods=("mle",), base_seed=0, method_kwargs=method_kwargs)
    out2 = run_scenario(scn_khac, config, methods=("mle",), base_seed=0, method_kwargs=method_kwargs)

    assert out1["mle"]["bias"]["p11"] == pytest.approx(out2["mle"]["bias"]["p11"])
    assert out1["mle"]["bias"]["p22"] == pytest.approx(out2["mle"]["bias"]["p22"])
