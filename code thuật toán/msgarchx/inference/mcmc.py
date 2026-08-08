"""MCMC bang NUTS (Muc 4.5): lay tong trang thai roi rac bang bo loc tien
(chien luoc thu nhat trong bai bao), sau do NUTS chi can lay mau tren cac
tham so lien tuc vi trang thai da duoc "tich hop" ben trong `ms_garch_forward_filter`.

Chan doan toi thieu theo bai bao: it nhat 4 chuoi khoi tao phan tan, R-hat,
ESS (dung arviz, tham chieu Vehtari et al. 2021).
"""
from __future__ import annotations

from dataclasses import dataclass

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
        kernel = NUTS(
            bayes_model,
            target_accept_prob=0.85,
            max_tree_depth=max_tree_depth,
        )
        mcmc_chain = MCMC(
            kernel,
            num_samples=n_samples,
            warmup_steps=n_warmup,
            num_chains=1,
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
        idata = az.from_dict(
            posterior={name: draws.numpy() for name, draws in samples_by_chain.items()}
        )
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
