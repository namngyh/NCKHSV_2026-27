import numpy as np
import torch

from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.simulation.dgp import simulate_ms_garch_x
from msgarchx.simulation.metrics import bias, brier_state, coverage, parameter_rmse
from msgarchx.simulation.scenarios import default_scenarios


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
