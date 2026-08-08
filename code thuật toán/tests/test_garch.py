import torch

from msgarchx.models.garch import GARCHParams, garch_forward, garch_loglik


def _make_params():
    return GARCHParams(
        mu=torch.tensor(0.0),
        omega=torch.tensor(0.05),
        alpha=torch.tensor(0.08),
        beta=torch.tensor(0.85),
        gamma=torch.tensor([0.1, -0.05]),
        nu=torch.tensor(8.0),
    )


def test_garch_forward_positive_and_deterministic():
    torch.manual_seed(0)
    T = 200
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()
    h = garch_forward(params, r, X, h0=torch.tensor(0.25))
    assert h.shape == (T,)
    assert (h > 0).all()
    h2 = garch_forward(params, r, X, h0=torch.tensor(0.25))
    assert torch.allclose(h, h2)


def test_garch_loglik_is_finite_and_differentiable():
    torch.manual_seed(1)
    T = 150
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params = _make_params()._replace(omega=torch.tensor(0.05, requires_grad=True))
    ll = garch_loglik(params, r, X, h0=torch.tensor(0.25))
    assert torch.isfinite(ll)
    ll.backward()
    assert params.omega.grad is not None
    assert torch.isfinite(params.omega.grad)


def test_garch_normal_vs_student_t_different_loglik():
    torch.manual_seed(2)
    T = 150
    r = torch.randn(T) * 0.5
    X = torch.randn(T, 2) * 0.1
    params_t = _make_params()
    params_n = params_t._replace(nu=None)
    ll_t = garch_loglik(params_t, r, X, h0=torch.tensor(0.25))
    ll_n = garch_loglik(params_n, r, X, h0=torch.tensor(0.25))
    assert ll_t.item() != ll_n.item()
