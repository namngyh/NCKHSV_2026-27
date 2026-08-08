import torch

from msgarchx.inference.forecast import (
    posterior_predictive_paths,
    predictive_variance_path,
    simulate_forecast_paths,
    var_from_samples,
)
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter


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


def test_simulate_forecast_paths_shapes_and_positivity():
    torch.manual_seed(0)
    params = _make_params()
    T = 100
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    with torch.no_grad():
        out = ms_garch_forward_filter(params, r, X)

    horizon = 10
    X_future = torch.randn(horizon, 2) * 0.1
    paths = simulate_forecast_paths(
        params, out["xi_filt"][-1], out["h"][-1], r[-1], X_future,
        horizon=horizon, n_paths=500,
    )
    assert paths["r"].shape == (500, horizon)
    assert paths["h"].shape == (500, horizon)
    assert (paths["h"] > 0).all()
    assert torch.isfinite(paths["r"]).all()
    assert set(paths["state"].unique().tolist()).issubset({0, 1})


def test_posterior_predictive_paths_aggregates_multiple_samples():
    torch.manual_seed(1)
    params_list = [_make_params(), _make_params()]
    T = 60
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    xi_list, h_list = [], []
    for p in params_list:
        with torch.no_grad():
            out = ms_garch_forward_filter(p, r, X)
        xi_list.append(out["xi_filt"][-1])
        h_list.append(out["h"][-1])

    horizon = 5
    X_future = torch.zeros(horizon, 2)
    all_paths = posterior_predictive_paths(
        params_list, xi_list, h_list, r[-1], X_future, horizon, n_paths_per_sample=100
    )
    assert all_paths.shape == (200, horizon)


def test_predictive_variance_path_uses_xi_pred_not_xi_filt():
    torch.manual_seed(3)
    params = _make_params()
    T = 50
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    with torch.no_grad():
        out = ms_garch_forward_filter(params, r, X)

    h_hat = predictive_variance_path(params, out["xi_pred"], out["h"])
    assert h_hat.shape == (T,)
    assert (h_hat > 0).all()
    # Voi tham so trong _make_params, mu gan 0 nen h_hat phai gan trung binh
    # co trong so cua h theo tung trang thai (kiem tra cong thuc, khong chi hinh dang).
    expected = (out["xi_pred"] * out["h"]).sum(dim=1)
    assert torch.allclose(h_hat, expected, atol=1e-6)


def test_var_from_samples_matches_quantile():
    samples = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    var_50 = var_from_samples(samples, 0.5)
    assert torch.allclose(var_50, -torch.quantile(samples, 0.5))
