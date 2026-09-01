"""MCMC bang NUTS (Muc 4.5): lay tong trang thai roi rac bang bo loc tien
(chien luoc thu nhat trong bai bao), sau do NUTS chi can lay mau tren cac
tham so lien tuc vi trang thai da duoc "tich hop" ben trong `ms_garch_forward_filter`.

Chan doan toi thieu theo bai bao: it nhat 4 chuoi khoi tao phan tan, R-hat,
ESS (dung arviz, tham chieu Vehtari et al. 2021).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import pyro
import torch
from pyro.infer import MCMC, NUTS

from msgarchx.inference.bayes_model import model as bayes_model
from msgarchx.inference.param_spec import ModelConfig, PriorConfig
from msgarchx.inference.summary import posterior_summary

_SITES = ["mu", "omega", "alpha", "beta", "gamma", "p11", "p22", "nu_minus_2"]


@dataclass
class MCMCResult:
    samples: dict[str, torch.Tensor]  # gop cac chuoi, shape (n_chains*n_samples, ...)
    samples_by_chain: dict[str, torch.Tensor]  # shape (n_chains, n_samples, ...)
    summary: dict[str, dict[str, torch.Tensor]]
    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    max_rhat: float
    min_ess: float
    n_divergences: int
    config: ModelConfig


def _init_cho_chuoi(initial_params, c: int):
    """Chon diem khoi tao cho chuoi thu `c`.

    `initial_params` co the la:
      - None            : de Pyro tu khoi tao (init_to_uniform)
      - dict            : DUNG CHUNG cho moi chuoi (chi hop ly khi n_chains=1,
                          vi 4 chuoi xuat phat cung mot diem thi R-hat mat y
                          nghia chan doan -- ban thao yeu cau "4 chuoi khoi tao
                          PHAN TAN")
      - list/tuple      : phan tu thu c
      - callable        : goi initial_params(c)
    """
    if initial_params is None or isinstance(initial_params, dict):
        return initial_params
    if isinstance(initial_params, (list, tuple)):
        return initial_params[c]
    return initial_params(c)


def initial_params_from_mle(params, chain: int, base_seed: int = 5000,
                            scale: float = 0.3) -> dict:
    """Diem khoi tao cua chuoi `chain`: nghiem MLE cua M1, co nhieu loan.

    Khoi tao tu TIEN NGHIEM khien cac chuoi bat dau o nhung vung vo ly va khong
    bao giu gap nhau (xem chan doan lan chay dau: R-hat ~ 4, ESS ~ 4,4). O day
    dat chung quanh nghiem MLE da thoa dieu kien chinh quy -- nhung VAN PHAN TAN
    (moi chuoi mot nhieu loan khac nhau) de R-hat con y nghia chan doan. Chuoi 0
    dat DUNG TAI nghiem MLE, cac chuoi con lai nhieu loan quanh no.

    Tra ve dict tren MIEN HOP LE (khong phai khong gian raw); Pyro tu bien doi.
    """
    g = torch.Generator().manual_seed(base_seed + chain)
    s_ = 0.0 if chain == 0 else scale

    def jitter(x, s=s_):
        return x + s * torch.randn(x.shape, generator=g)

    def logit(x):
        return torch.log(x) - torch.log1p(-x)

    return {
        "mu": jitter(params.mu, 0.05 if chain else 0.0),
        "omega": params.omega * torch.exp(jitter(torch.zeros(2))),
        "alpha": torch.sigmoid(jitter(logit(params.alpha.clamp(1e-4, 1 - 1e-4)))),
        "beta": torch.sigmoid(jitter(logit(params.beta.clamp(1e-4, 1 - 1e-4)))),
        "gamma": jitter(params.gamma, 0.05 if chain else 0.0),
        "p11": torch.sigmoid(jitter(logit(params.P[0, 0].clamp(1e-4, 1 - 1e-4)).reshape(1))).squeeze(),
        "p22": torch.sigmoid(jitter(logit(params.P[1, 1].clamp(1e-4, 1 - 1e-4)).reshape(1))).squeeze(),
        "nu_minus_2": (params.nu - 2.0) * torch.exp(jitter(torch.zeros(()))),
    }


def fit_mcmc(
    r: torch.Tensor,
    X: torch.Tensor,
    config: ModelConfig,
    prior: PriorConfig | None = None,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_samples: int = 1000,
    seed: int = 0,
    variance_link: str = "clamp",
    floor: float = 1e-8,
    verbose: bool = False,
    max_tree_depth: int = 8,
    full_mass: bool = False,
    initial_params=None,
    checkpoint_path=None,
    checkpoint_moi: int = 25,
) -> MCMCResult:
    """max_tree_depth gioi han so buoc leapfrog toi da moi mau (2^depth) --
    mac dinh cua NUTS la 10 (toi 1024 buoc/mau), rat ton kem vi bo loc tien
    MS-GARCH-X la mot vong lap Python tuan tu theo T (moi buoc leapfrog can
    mot lan forward+backward qua toan bo T). Gia tri 8 (toi da 256 buoc/mau)
    la mot thoa hiep hop ly; giam them neu T lon (du lieu thuc VN30F1M).

    LUU Y HIEU NANG: thu nghiem cho thay torch.jit.trace/`jit_compile=True`
    KHONG giup ich (thuc te cham hon) cho vong lap Python tuan tu nay tren
    CPU, nen ham nay chay hoan toan o che do eager.
    """
    # Chay tuan tu tung chuoi (khong dung multiprocessing) de tranh cac van
    # de spawn-process cua Windows khi ham nay duoc goi tu mot script khong
    # co bao ve `if __name__ == "__main__"`.
    per_chain_samples: dict[str, list[torch.Tensor]] = {name: [] for name in _SITES}
    n_div = 0
    for c in range(n_chains):
        pyro.set_rng_seed(seed + c)
        # full_mass=True: ma tran khoi luong DAC thay vi duong cheo. Can thiet
        # khi cac tham so co TUONG QUAN MANH voi nhau -- voi MS-GARCH-X thi
        # omega_s, alpha_s, beta_s tuong quan rat manh (chung cung xac dinh muc
        # va do dai dang cua phuong sai). Ma tran duong cheo khong the hien
        # duoc tuong quan do, nen NUTS buoc phai ha buoc tich phan xuong rat
        # nho va khong di chuyen duoc (xem chan doan lan chay dau: R-hat ~ 4,
        # ESS ~ 4.4, buoc tich phan ~ 8e-4).
        kernel = NUTS(
            bayes_model,
            full_mass=full_mass,
            target_accept_prob=0.85,
            max_tree_depth=max_tree_depth,
        )
        # LUU TRANG THAI DINH KY. Ly do: mot chuoi NUTS tren du lieu that mat
        # NHIEU NGAY, va Pyro chi tra ve ket qua o CUOI. Ngay 26/08/2026,
        # Windows tu khoi dong lai luc 00:35 (Kernel-Power 577) da xoa sach
        # ~65% tien do cua ca 4 chuoi. Voi hook nay, lan chay sau it nhat con
        # doc duoc buoc tich phan da thich nghi va diem tham so cuoi cung de
        # khoi tao lai, thay vi bat dau tu con so khong.
        hook = None
        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            cp.parent.mkdir(parents=True, exist_ok=True)

            def hook(kern, mau, giai_doan, i, **_):
                if i % checkpoint_moi:
                    return
                try:
                    torch.save({"giai_doan": giai_doan, "vong": i,
                                "step_size": float(getattr(kern, "step_size", float("nan"))),
                                "tham_so": {k: v.detach().clone()
                                            for k, v in (mau or {}).items()},
                                "thoi_diem": time.time()}, cp)
                except Exception:
                    pass          # khong bao gio de viec luu lam hong lan chay

        mcmc_chain = MCMC(
            kernel,
            num_samples=n_samples,
            warmup_steps=n_warmup,
            num_chains=1,
            initial_params=_init_cho_chuoi(initial_params, c),
            hook_fn=hook,
            disable_progbar=not verbose,
        )
        mcmc_chain.run(r, X, config, prior, variance_link, floor)
        chain_samples = mcmc_chain.get_samples()
        for name in _SITES:
            if name in chain_samples:
                per_chain_samples[name].append(chain_samples[name])
        try:
            # mcmc_chain.diagnostics() tinh ca R-hat noi bo cua Pyro, co the
            # nem AssertionError khi n_samples < 4 (Pyro can >=4 mau de split
            # Gelman-Rubin). Chi dung ham nay de lay so phan ky nen bo qua
            # loi nay mot cach an toan, khong lam hong toan bo lan chay.
            diag = mcmc_chain.diagnostics()
            if "divergences" in diag:
                for chain_div in diag["divergences"].values():
                    n_div += len(chain_div) if hasattr(chain_div, "__len__") else int(chain_div)
        except (AssertionError, RuntimeError):
            pass

    samples_by_chain = {
        name: torch.stack(draws_list) for name, draws_list in per_chain_samples.items() if draws_list
    }
    samples = {name: draws.reshape(-1, *draws.shape[2:]) for name, draws in samples_by_chain.items()}

    try:
        post = {name: draws.numpy() for name, draws in samples_by_chain.items()}
        try:
            idata = az.from_dict({"posterior": post})      # arviz >= 1.0
        except TypeError:
            idata = az.from_dict(posterior=post)            # arviz < 1.0
        rhat_ds = az.rhat(idata)
        ess_ds = az.ess(idata, method="bulk")
        rhat = {name: float(rhat_ds[name].values.max()) for name in samples_by_chain}
        ess_bulk = {name: float(ess_ds[name].values.min()) for name in samples_by_chain}
    except Exception:
        # Chan doan hoi tu chi co y nghia khi du cum/mau (vd. >=4 chuoi cho
        # split R-hat); voi cau hinh rat nho (demo/dry-run) co the that bai,
        # khong nen lam hong toan bo ket qua MCMC vi ly do nay.
        rhat = {name: float("nan") for name in samples_by_chain}
        ess_bulk = {name: float("nan") for name in samples_by_chain}

    summary = posterior_summary(samples)

    return MCMCResult(
        samples=samples,
        samples_by_chain=samples_by_chain,
        summary=summary,
        rhat=rhat,
        ess_bulk=ess_bulk,
        max_rhat=max(rhat.values()) if rhat else float("nan"),
        min_ess=min(ess_bulk.values()) if ess_bulk else float("nan"),
        n_divergences=n_div,
        config=config,
    )
