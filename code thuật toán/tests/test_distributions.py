import torch

from msgarchx.distributions import normal_logpdf, std_t_logpdf, std_t_sample


def test_std_t_logpdf_matches_scipy():
    from scipy import stats
    import numpy as np

    nu = torch.tensor(6.0)
    z = torch.linspace(-3, 3, 25)
    scale = np.sqrt((6.0 - 2.0) / 6.0)
    expected = stats.t.logpdf(z.numpy(), df=6.0, loc=0.0, scale=scale)
    got = std_t_logpdf(z, nu).numpy()
    assert np.allclose(got, expected, atol=1e-6)


def test_std_t_converges_to_normal_for_large_nu():
    nu = torch.tensor(1e6)
    z = torch.linspace(-2, 2, 9)
    assert torch.allclose(std_t_logpdf(z, nu), normal_logpdf(z), atol=1e-4)


def test_std_t_sample_has_unit_variance():
    torch.manual_seed(0)
    nu = torch.full((200_000,), 6.0)
    z = std_t_sample(nu)
    assert abs(z.mean().item()) < 0.02
    assert abs(z.var().item() - 1.0) < 0.05
