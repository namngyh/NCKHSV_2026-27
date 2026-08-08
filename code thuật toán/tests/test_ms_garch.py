import torch

from msgarchx.models.ms_garch import (
    MSGARCHXParams,
    expected_duration,
    label_switch_correction,
    ms_garch_forward_filter,
    ms_garch_smoother,
    stationary_distribution,
)


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


def test_stationary_distribution_sums_to_one_and_solves_balance():
    P = torch.tensor([[0.97, 0.03], [0.05, 0.95]])
    pi = stationary_distribution(P)
    assert torch.allclose(pi.sum(), torch.tensor(1.0))
    assert torch.allclose(pi @ P, pi, atol=1e-6)


def test_expected_duration_formula():
    P = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    d = expected_duration(P)
    assert torch.allclose(d, torch.tensor([10.0, 5.0]))


def test_forward_filter_shapes_and_validity():
    torch.manual_seed(0)
    T = 100
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()
    out = ms_garch_forward_filter(params, r, X)

    assert out["xi_filt"].shape == (T, 2)
    assert out["h"].shape == (T, 2)
    assert torch.isfinite(out["loglik"])
    assert torch.allclose(out["xi_filt"].sum(dim=1), torch.ones(T), atol=1e-5)
    assert (out["h"] > 0).all()


def test_forward_filter_is_differentiable_wrt_params():
    torch.manual_seed(1)
    T = 80
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()
    omega = params.omega.clone().requires_grad_(True)
    params = params._replace(omega=omega)
    out = ms_garch_forward_filter(params, r, X)
    out["loglik"].backward()
    assert omega.grad is not None
    assert torch.isfinite(omega.grad).all()


def test_smoother_returns_valid_probabilities():
    torch.manual_seed(2)
    T = 60
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()
    out = ms_garch_forward_filter(params, r, X)
    xi_smooth = ms_garch_smoother(out["xi_filt"], out["xi_pred"], params.P)
    assert xi_smooth.shape == (T, 2)
    assert torch.allclose(xi_smooth.sum(dim=1), torch.ones(T), atol=1e-5)
    assert (xi_smooth >= -1e-8).all()
    # thoi diem cuoi, lam tron phai trung voi loc
    assert torch.allclose(xi_smooth[-1], out["xi_filt"][-1])


def test_label_switch_correction_swaps_when_needed():
    params = _make_params()  # state 0 la low-vol, state 1 la high-vol -> khong doi
    h = torch.tensor([[0.1, 0.5]] * 10)
    same = label_switch_correction(params, h)
    assert torch.allclose(same.omega, params.omega)

    h_swapped = torch.tensor([[0.5, 0.1]] * 10)
    corrected = label_switch_correction(params, h_swapped)
    assert torch.allclose(corrected.omega, params.omega.flip(0))
    assert torch.allclose(corrected.P, params.P.flip(0).flip(1))
