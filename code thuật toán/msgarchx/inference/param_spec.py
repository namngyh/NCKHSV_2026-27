"""Dac ta vector tham so dung chung cho MLE (raw vector khong rang buoc)
va cho prior cua Variational Bayes / MCMC (bayes_model.py).

Vector raw cho MLE bam theo eq:parameter-vector va cac bien doi trong Muc 4.2:

    [mu_1, mu_2, log_omega_1, log_omega_2,
     eta_alpha_1, eta_alpha_2, eta_beta_1, eta_beta_2,
     gamma_1 (k gia tri), gamma_2 (k gia tri),
     eta_p11, eta_p22,
     eta_nu (1 gia tri, hoac 2 gia tri neu state_specific_nu=True)]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.transforms import (
    ab_from_raw,
    ab_init_center,
    nu_from_raw,
    p_from_raw,
    raw_from_ab,
    raw_from_nu,
    raw_from_p,
)


@dataclass(frozen=True)
class ModelConfig:
    """Cau hinh khong doi cua mo hinh: so bien ngoai sinh va co dung nu_s rieng hay khong."""

    n_exog: int
    state_specific_nu: bool = False

    @property
    def raw_dim(self) -> int:
        base = 2 + 2 + 2 + 2 + 2 * self.n_exog + 2
        return base + (2 if self.state_specific_nu else 1)


@dataclass(frozen=True)
class PriorConfig:
    """Sieu tham so prior tham khao, dung trong inference/bayes_model.py.

    Bam theo eq:prior-mu, eq:prior-omega, eq:prior-gamma, eq:prior-transition,
    eq:prior-df. alpha_s, beta_s khong co dac ta prior rieng trong bai bao;
    o day dung Beta(a,a) tren (0,1) lam prior yeu, hop ly voi mien gia tri
    thong thuong cua he so ARCH/GARCH.
    """

    sigma_mu: float = 1.0
    m_omega: float = 0.0
    sigma_omega: float = 1.0
    alpha_beta_concentration: float = 2.0
    sigma_gamma: float = 1.0
    transition_prior_stay: float = 8.0
    transition_prior_switch: float = 2.0
    nu_shape: float = 2.0
    nu_rate: float = 0.2


def param_slices(config: ModelConfig) -> dict:
    k = config.n_exog
    n_nu = 2 if config.state_specific_nu else 1
    idx = {}
    pos = 0
    for name, size in [
        ("mu", 2),
        ("log_omega", 2),
        ("eta_alpha", 2),
        ("eta_beta", 2),
        ("gamma", 2 * k),
        ("eta_p", 2),
        ("eta_nu", n_nu),
    ]:
        idx[name] = slice(pos, pos + size)
        pos += size
    return idx


def unpack_raw(raw: torch.Tensor, config: ModelConfig) -> MSGARCHXParams:
    """Bien doi vector raw khong rang buoc thanh MSGARCHXParams hop le."""
    idx = param_slices(config)
    mu = raw[idx["mu"]]
    omega = torch.exp(raw[idx["log_omega"]])
    alpha = ab_from_raw(raw[idx["eta_alpha"]])
    beta = ab_from_raw(raw[idx["eta_beta"]])
    gamma = raw[idx["gamma"]].reshape(2, config.n_exog)

    p11 = p_from_raw(raw[idx["eta_p"]][0])
    p22 = p_from_raw(raw[idx["eta_p"]][1])
    P = torch.stack(
        [
            torch.stack([p11, 1.0 - p11]),
            torch.stack([1.0 - p22, p22]),
        ]
    )

    eta_nu = raw[idx["eta_nu"]]
    nu = nu_from_raw(eta_nu) if config.state_specific_nu else nu_from_raw(eta_nu[0])

    return MSGARCHXParams(mu=mu, omega=omega, alpha=alpha, beta=beta, gamma=gamma, P=P, nu=nu)


def pack_params(params: MSGARCHXParams, config: ModelConfig) -> torch.Tensor:
    """Chieu nguoc: MSGARCHXParams hop le -> vector raw (dung de khoi tao MLE)."""
    p11 = params.P[0, 0]
    p22 = params.P[1, 1]
    eta_nu = raw_from_nu(params.nu) if config.state_specific_nu else raw_from_nu(params.nu).reshape(1)
    parts = [
        params.mu,
        torch.log(params.omega),
        raw_from_ab(params.alpha),
        raw_from_ab(params.beta),
        params.gamma.reshape(-1),
        torch.stack([raw_from_p(p11), raw_from_p(p22)]),
        eta_nu,
    ]
    return torch.cat([p.reshape(-1) for p in parts])


def init_raw(config: ModelConfig, seed: int | None = None) -> torch.Tensor:
    """Khoi tao raw vector hop ly de bat dau MLE (nhieu diem khoi tao -> doi seed).

    Trang thai 1 duoc khoi tao voi omega nho hon trang thai 2 de goi y quy uoc
    bien dong thap/cao ngay tu dau (van can label_switch_correction sau khi hoi tu).
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def randn(*shape, scale=1.0):
        return scale * torch.randn(*shape, generator=generator)

    mu = randn(2, scale=0.05)
    log_omega = torch.stack(
        [torch.log(torch.tensor(0.5)) + randn(()), torch.log(torch.tensor(2.0)) + randn(())]
    )
    c_alpha, c_beta = ab_init_center()
    eta_alpha = randn(2, scale=0.3) + c_alpha
    eta_beta = randn(2, scale=0.3) + c_beta
    gamma = randn(2 * config.n_exog, scale=0.05)
    eta_p = torch.stack([raw_from_p(torch.tensor(0.97)), raw_from_p(torch.tensor(0.95))]) + randn(2, scale=0.2)
    n_nu = 2 if config.state_specific_nu else 1
    eta_nu = raw_from_nu(torch.tensor(8.0)).expand(n_nu).clone() + randn(n_nu, scale=0.3)
    return torch.cat([mu, log_omega, eta_alpha, eta_beta, gamma, eta_p, eta_nu])


# Bien the omega (thap, cao) va duong cheo P cho cac diem khoi tao AM.
# Khong dat hai trang thai TRUNG KHIT nhau: ban thao canh bao rang khi do ham
# hop ly doi xung qua phep hoan vi nhan va thuat toan gradient co the hoi tu
# ve diem yen ngua ung voi cau hinh doi xung do (Muc 4.2).
# (he so nhan omega, duong cheo P, quy tac dat nu tu nu cua mo hinh 1 trang thai)
_BIEN_THE_AM = [
    ((0.5, 2.0), (0.95, 0.95), lambda nu: nu),
    ((0.25, 4.0), (0.98, 0.90), lambda nu: nu),
    ((0.7, 1.5), (0.90, 0.85), lambda nu: max(nu, 6.0)),
    ((0.4, 3.0), (0.99, 0.94), lambda nu: nu + 2.0),
]


def warm_starts_from_garch(garch_params, config: ModelConfig,
                            n_variants: int = 4) -> list[torch.Tensor]:
    """Sinh cac diem khoi tao AM cho mo hinh 2 trang thai tu nghiem MOT trang thai.

    Bam theo chien luoc "khoi tao tuan tu theo thu bac mo hinh" ma ban thao yeu
    cau (Muc 4.2): uoc luong B1 -> B2 -> B3 truoc, roi dung nghiem mot trang
    thai do khoi tao mo hinh hai trang thai bang cach NHAN DOI bo tham so va
    tach hai ban sao ra bang cach nhan omega len/xuong.

    `garch_params` la GARCHParams (mu, omega, alpha, beta, gamma, nu vo huong).
    """
    from msgarchx.models.ms_garch import MSGARCHXParams

    k = config.n_exog
    if k and garch_params.gamma.numel() == k:
        gam = garch_params.gamma.reshape(1, k).expand(2, k).clone()
    else:
        gam = torch.zeros(2, k)
    nu0 = float(garch_params.nu.item()) if garch_params.nu is not None else 8.0
    mu0 = float(garch_params.mu.item())
    om0 = float(garch_params.omega.item())
    al0 = max(float(garch_params.alpha.item()), 1e-3)
    be0 = max(float(garch_params.beta.item()), 1e-3)

    out = []
    for (w_lo, w_hi), (p11, p22), nu_rule in _BIEN_THE_AM[:n_variants]:
        nu_i = float(nu_rule(nu0))
        params = MSGARCHXParams(
            mu=torch.tensor([mu0, mu0]),
            omega=torch.tensor([om0 * w_lo, om0 * w_hi]),
            alpha=torch.tensor([al0, al0]),
            beta=torch.tensor([be0, be0]),
            gamma=gam,
            P=torch.tensor([[p11, 1.0 - p11], [1.0 - p22, p22]]),
            nu=torch.tensor(nu_i if not config.state_specific_nu else [nu_i, nu_i]),
        )
        out.append(pack_params(params, config))
    return out
