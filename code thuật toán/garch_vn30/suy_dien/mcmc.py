"""MCMC bang NUTS (No-U-Turn Sampler, Hoffman & Gelman 2014) va chan doan hoi tu.

NUTS la Hamiltonian Monte Carlo tu dong chon do dai quy dao: coi -log hau
nghiem la the nang U(theta), them dong luong p ~ N(0, M), mo phong phuong
trinh Hamilton bang tich phan leapfrog va dung khi quy dao bat dau quay dau.
Buoc tich phan va ma tran khoi luong M duoc thich nghi trong giai doan warmup.

Chan doan: R-hat tach doi (Gelman-Rubin) va ESS khoi (bulk) tinh bang arviz
tren >= 4 chuoi khoi tao PHAN TAN.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import pyro
import torch
from pyro.infer import MCMC, NUTS

from garch_vn30.mo_hinh.danh_muc import ModelSpec
from garch_vn30.suy_dien.bayes import SITES, PriorConfig, model, posterior_summary

# Tran do lon nhieu loan tren thang log/logit: phuong phap delta phong dai vo
# ly khi tham so gan bien (x hoac x(1-x) rat nho).
TRAN_NHIEU = 1.0


@dataclass
class MCMCResult:
    samples: dict[str, torch.Tensor]           # gop cac chuoi: (n_chains*n_samples, ...)
    samples_by_chain: dict[str, torch.Tensor]  # (n_chains, n_samples, ...)
    summary: dict[str, dict[str, torch.Tensor]]
    rhat: dict[str, float]
    ess_bulk: dict[str, float]
    max_rhat: float
    min_ess: float
    n_divergences: int


def khoi_tao_tu_mle(params, spec: ModelSpec, chain: int, se: dict | None = None,
                    base_seed: int = 5000, k_se: float = 2.0,
                    scale_du_phong: float = 0.1) -> dict:
    """Diem xuat phat cua chuoi `chain`, dat quanh nghiem MLE.

    Chuoi 0 dat DUNG tai nghiem MLE; cac chuoi khac nhieu loan k_se = 2 sai so
    chuan (tu mle.standard_errors) -- du phan tan de R-hat co y nghia, nhung
    khong nem chuoi sang vung khac cua khong gian tham so. Neu khong co sai so
    chuan thi nhieu loan 10% gia tri tham so.

    Nhieu loan ton trong mien cua tung tham so: tham so duong nhieu tren thang
    log (do lon SE/x), tham so trong (0,1) nhieu tren thang logit (SE/(x(1-x))).
    Tra ve dict tren MIEN HOP LE; Pyro tu doi sang khong gian khong rang buoc.
    """
    g = torch.Generator().manual_seed(base_seed + chain)
    tai_mle = chain == 0

    def do_lon(ten, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        if tai_mle:
            return torch.zeros_like(x)
        if se is not None and ten in se:
            return k_se * torch.as_tensor(se[ten], dtype=torch.float64)
        return scale_du_phong * x.abs().clamp(min=1e-3)

    def cong(ten, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        return x + do_lon(ten, x) * torch.randn(x.shape, generator=g, dtype=torch.float64)

    def qua_log(ten, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        s = (do_lon(ten, x) / x.abs().clamp(min=1e-8)).clamp(max=TRAN_NHIEU)
        return x * torch.exp(s * torch.randn(x.shape, generator=g, dtype=torch.float64))

    def qua_logit(ten, x):
        x = torch.as_tensor(x, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
        s = (do_lon(ten, x) / (x * (1 - x)).clamp(max=1.0, min=1e-8)).clamp(max=TRAN_NHIEU)
        return torch.sigmoid(torch.log(x) - torch.log1p(-x)
                             + s * torch.randn(x.shape, generator=g, dtype=torch.float64))

    ra = {"mu": cong("mu", params.mu),
          "alpha": qua_logit("alpha", params.alpha),
          "beta": qua_logit("beta", params.beta)}
    # omega: tu do voi EGARCH (log h), duong voi GARCH.
    ra["omega"] = (cong("omega", params.omega) if spec.family == "egarch"
                   else qua_log("omega", params.omega))
    if spec.family == "egarch":
        ra["theta"] = cong("theta", params.theta)
    if spec.use_exog:
        ra["gamma"] = cong("gamma", params.gamma)
    if spec.dist == "student_t":
        ra["nu_minus_2"] = qua_log("nu", torch.as_tensor(params.nu) - 2.0)
    return ra


def inference_data(post: dict):
    """Dong goi mau {ten: mang (n_chuoi, n_mau, ...)} thanh InferenceData cua arviz.

    Cu phap from_dict doi o arviz 1.0: ban < 1.0 nhan posterior=..., ban >= 1.0
    nhan dict {"posterior": ...}. Truyen sai dang KHONG bao loi ma tao du lieu
    rac, nen chon theo phien ban.
    """
    if int(az.__version__.split(".")[0]) >= 1:
        return az.from_dict({"posterior": post})
    return az.from_dict(posterior=post)


def _luu_tien_do(checkpoint_path, moi: int):
    """Hook ghi trang thai sampler dinh ky (mot chuoi NUTS co the chay nhieu gio)."""
    if checkpoint_path is None:
        return None
    cp = Path(checkpoint_path)
    cp.parent.mkdir(parents=True, exist_ok=True)

    def hook(kern, mau, giai_doan, i, **_):
        if i % moi:
            return
        try:
            torch.save({"giai_doan": giai_doan, "vong": i,
                        "step_size": float(getattr(kern, "step_size", float("nan"))),
                        "tham_so": {k: v.detach().clone() for k, v in (mau or {}).items()},
                        "thoi_diem": time.time()}, cp)
        except Exception:
            pass          # viec luu tien do khong duoc lam hong lan chay
    return hook


def fit_mcmc(r: torch.Tensor, X: torch.Tensor, spec: ModelSpec,
             prior: PriorConfig | None = None, n_chains: int = 4, n_warmup: int = 500,
             n_samples: int = 1000, seed: int = 0, max_tree_depth: int = 8,
             full_mass: bool = False, initial_params=None, checkpoint_path=None,
             checkpoint_moi: int = 25, verbose: bool = False) -> MCMCResult:
    """Chay `n_chains` chuoi NUTS TUAN TU (tranh loi spawn tien trinh tren Windows).

    initial_params: None (Pyro tu khoi tao), dict (dung chung), list (phan tu
    thu c cho chuoi c) hoac ham c -> dict (vd. lambda c: khoi_tao_tu_mle(...)).
    max_tree_depth: toi da 2^depth buoc leapfrog moi mau -- moi buoc la mot
    lan forward+backward qua ca chuoi du lieu, nen day la nut chi phi chinh.
    full_mass: ma tran khoi luong DAC (nam duoc tuong quan giua cac tham so).
    """
    theo_chuoi: dict[str, list[torch.Tensor]] = {name: [] for name in SITES}
    n_div = 0
    for c in range(n_chains):
        pyro.set_rng_seed(seed + c)
        kernel = NUTS(model, full_mass=full_mass, target_accept_prob=0.85,
                      max_tree_depth=max_tree_depth)
        if initial_params is None or isinstance(initial_params, dict):
            init_c = initial_params
        elif isinstance(initial_params, (list, tuple)):
            init_c = initial_params[c]
        else:
            init_c = initial_params(c)
        mcmc = MCMC(kernel, num_samples=n_samples, warmup_steps=n_warmup, num_chains=1,
                    initial_params=init_c,
                    hook_fn=_luu_tien_do(checkpoint_path, checkpoint_moi),
                    disable_progbar=not verbose)
        mcmc.run(r, X, spec, prior)
        mau = mcmc.get_samples()
        for name in SITES:
            if name in mau:
                theo_chuoi[name].append(mau[name])
        try:
            # diagnostics() can >= 4 mau; chi dung de dem so buoc phan ky.
            diag = mcmc.diagnostics()
            for d in diag.get("divergences", {}).values():
                n_div += len(d) if hasattr(d, "__len__") else int(d)
        except (AssertionError, RuntimeError):
            pass

    by_chain = {name: torch.stack(ds) for name, ds in theo_chuoi.items() if ds}
    samples = {name: x.reshape(-1, *x.shape[2:]) for name, x in by_chain.items()}

    try:
        idata = inference_data({name: x.numpy() for name, x in by_chain.items()})
        rhat_ds, ess_ds = az.rhat(idata), az.ess(idata, method="bulk")
        rhat = {name: float(rhat_ds[name].values.max()) for name in by_chain}
        ess = {name: float(ess_ds[name].values.min()) for name in by_chain}
    except Exception:
        # Cau hinh qua nho (vd. 1 chuoi, it mau) thi khong tinh duoc R-hat.
        rhat = {name: float("nan") for name in by_chain}
        ess = {name: float("nan") for name in by_chain}

    return MCMCResult(
        samples=samples, samples_by_chain=by_chain, summary=posterior_summary(samples),
        rhat=rhat, ess_bulk=ess,
        max_rhat=max(rhat.values()) if rhat else float("nan"),
        min_ess=min(ess.values()) if ess else float("nan"),
        n_divergences=n_div,
    )
