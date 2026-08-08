import torch

from msgarchx.transforms import (
    nu_from_raw,
    p_from_raw,
    positive_variance_link,
    raw_from_nu,
    raw_from_p,
)


def test_nu_transform_roundtrip_and_domain():
    eta = torch.tensor([-5.0, 0.0, 3.0])
    nu = nu_from_raw(eta)
    assert (nu > 2.0).all()
    back = raw_from_nu(nu)
    assert torch.allclose(nu_from_raw(back), nu, atol=1e-5)


def test_p_transform_roundtrip_and_domain():
    eta = torch.tensor([-2.0, 0.0, 2.0])
    p = p_from_raw(eta)
    assert (p > 0).all() and (p < 1).all()
    back = raw_from_p(p)
    assert torch.allclose(p_from_raw(back), p, atol=1e-5)


def test_positive_variance_link_clamp_and_softplus():
    h_raw = torch.tensor([-1.0, 0.0, 5.0])
    clamped = positive_variance_link(h_raw, floor=1e-4, mode="clamp")
    assert (clamped >= 1e-4).all()
    assert clamped[2] == 5.0

    soft = positive_variance_link(h_raw, floor=1e-4, mode="softplus")
    assert (soft > 0).all()
