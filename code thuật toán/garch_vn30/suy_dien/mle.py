"""Uoc luong hop ly cuc dai (MLE) cho GARCH/EGARCH mot trang thai.

Toi uu tren khong gian KHONG rang buoc, roi bien doi ve mien hop le:

    GARCH : omega = exp(log_omega), alpha = softplus(eta_alpha), beta = softplus(eta_beta)
    EGARCH: omega, alpha, theta tu do;  beta = sigmoid(eta_beta) trong (0, 1)
    ca hai: nu = 2 + softplus(eta_nu)  (neu Student-t)

Moi diem khoi tao chay Adam (buoc tho, co cat chuan gradient) roi L-BFGS voi
tim kiem duong strong-Wolfe (buoc tinh); chon nghiem co log-hop-ly lon nhat.
Sai so chuan lay tu Hessian quan sat qua phuong phap delta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from garch_vn30.mo_hinh.danh_muc import loglik_cua
from garch_vn30.mo_hinh.egarch import EGARCHParams
from garch_vn30.mo_hinh.garch import GARCHParams

_softplus = torch.nn.functional.softplus

# Thu tu cac bien khong rang buoc (dung khi phang hoa de tinh Hessian) va
# ten tham so tu nhien tuong ung, theo dung thu tu.
_BIEN_RAW = {
    "garch": ["mu", "log_omega", "eta_alpha", "eta_beta", "gamma"],
    "egarch": ["mu", "omega", "alpha", "theta", "eta_beta", "gamma"],
}
_TEN_TU_NHIEN = {
    "garch": ["mu", "omega", "alpha", "beta", "gamma"],
    "egarch": ["mu", "omega", "alpha", "theta", "beta", "gamma"],
}


@dataclass
class LanChay:
    """Chan doan cua mot diem khoi tao."""
    seed: int
    loglik: float
    converged: bool      # log-hop-ly huu han va ||gradient|| < 1 tai diem dung
    grad_norm: float


@dataclass
class MLEResult:
    params: GARCHParams | EGARCHParams
    loglik: float
    aic: float
    bic: float
    n_params: int
    runs: list[LanChay] = field(default_factory=list)


# ------------------------------------------------------------ bien doi tham so
def unpack(raw: dict, dist: str, family: str):
    """Bien khong rang buoc -> bo tham so tren mien hop le."""
    nu = 2.0 + _softplus(raw["eta_nu"]) if dist == "student_t" else None
    if family == "garch":
        return GARCHParams(
            mu=raw["mu"],
            omega=torch.exp(raw["log_omega"]),
            alpha=_softplus(raw["eta_alpha"]),
            beta=_softplus(raw["eta_beta"]),
            gamma=raw["gamma"],
            nu=nu,
        )
    if family == "egarch":
        # beta trong (0,1) chu khong phai (-1,1): tien nghiem Bayes cua beta la
        # Beta(a,a) tren (0,1), nen MLE cung tim tren (0,1) de ba phuong phap
        # lam viec tren CUNG mot khong gian tham so.
        return EGARCHParams(
            mu=raw["mu"],
            omega=raw["omega"],
            alpha=raw["alpha"],
            theta=raw["theta"],
            beta=torch.sigmoid(raw["eta_beta"]),
            gamma=raw["gamma"],
            nu=nu,
        )
    raise ValueError(f"family khong hop le: {family!r}")


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y))


def pack(params, dist: str, family: str) -> dict:
    """Chieu nguoc cua unpack: bo tham so tu nhien -> bien khong rang buoc."""
    if family == "garch":
        raw = {"mu": params.mu, "log_omega": torch.log(params.omega),
               "eta_alpha": _inv_softplus(params.alpha),
               "eta_beta": _inv_softplus(params.beta), "gamma": params.gamma}
    else:
        raw = {"mu": params.mu, "omega": params.omega, "alpha": params.alpha,
               "theta": params.theta,
               "eta_beta": torch.log(params.beta / (1.0 - params.beta)),
               "gamma": params.gamma}
    if dist == "student_t":
        raw["eta_nu"] = _inv_softplus(params.nu - 2.0)
    return raw


def _khoi_tao(dist: str, family: str, k: int, h0: torch.Tensor) -> dict:
    """Diem khoi tao ngau nhien quanh mot cau hinh hop ly (dat seed truoc khi goi)."""
    if family == "garch":
        raw = {
            "mu": torch.zeros(()),
            "log_omega": torch.log(h0 * 0.1) + 0.1 * torch.randn(()),
            "eta_alpha": torch.tensor(-1.5) + 0.2 * torch.randn(()),
            "eta_beta": torch.tensor(1.0) + 0.2 * torch.randn(()),
            "gamma": 0.01 * torch.randn(k),
        }
    else:
        # Nham vao trang thai dung: beta ~ 0,9 va omega ~ (1 - beta) log h0.
        beta0 = 0.9
        raw = {
            "mu": torch.zeros(()),
            "omega": (1.0 - beta0) * torch.log(h0) + 0.05 * torch.randn(()),
            "alpha": torch.tensor(0.10) + 0.02 * torch.randn(()),
            "theta": torch.tensor(0.0) + 0.02 * torch.randn(()),  # khong gia dinh dau don bay
            "eta_beta": torch.log(torch.tensor(beta0 / (1.0 - beta0))) + 0.1 * torch.randn(()),
            "gamma": 0.01 * torch.randn(k),
        }
    if dist == "student_t":
        raw["eta_nu"] = torch.tensor(1.5) + 0.2 * torch.randn(())
    return raw


# ------------------------------------------------------------------ toi uu
def _mot_diem_khoi_tao(seed, r, X, dist, family, h0, n_adam, n_lbfgs, lr_adam, verbose):
    torch.manual_seed(seed)
    loglik = loglik_cua(family)
    raw = _khoi_tao(dist, family, X.shape[1], h0)
    vars_ = {ten: v.clone().requires_grad_(True) for ten, v in raw.items()}
    ds_bien = list(vars_.values())

    def nll():
        return -loglik(unpack(vars_, dist, family), r, X, h0=h0)

    # Giai doan 1 -- Adam: on dinh khi con xa nghiem; cat chuan gradient o 50.
    opt = torch.optim.Adam(ds_bien, lr=lr_adam)
    for it in range(n_adam):
        opt.zero_grad()
        loss = nll()
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ds_bien, 50.0)
        opt.step()
        if verbose and it % 200 == 0:
            print(f"  [seed={seed}] adam {it}: nll={loss.item():.4f}")

    # Giai doan 2 -- L-BFGS (tua Newton) voi tim kiem duong strong-Wolfe.
    lbfgs = torch.optim.LBFGS(ds_bien, lr=0.5, max_iter=n_lbfgs, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = nll()
        if torch.isnan(loss):
            return loss
        loss.backward()
        return loss

    truoc_lbfgs = {ten: v.detach().clone() for ten, v in vars_.items()}
    try:
        lbfgs.step(closure)
    except (IndexError, RuntimeError) as exc:
        # L-BFGS cua torch co the vo khi mot buoc tim kiem duong cho NaN: quay
        # ve nghiem cua giai doan Adam thay vi lam hong ca qua trinh.
        print(f"  [seed={seed}] L-BFGS that bai ({type(exc).__name__}); giu nghiem Adam.")
        with torch.no_grad():
            for ten, v in vars_.items():
                v.copy_(truoc_lbfgs[ten])
                v.grad = None
        lai = nll()
        if torch.isfinite(lai):
            lai.backward()

    with torch.no_grad():
        final_loss = nll()
    grad_norm = float(sum((v.grad.norm() ** 2 for v in ds_bien if v.grad is not None),
                          torch.tensor(0.0)).sqrt())
    converged = bool(torch.isfinite(final_loss)) and grad_norm < 1.0
    params = unpack({ten: v.detach() for ten, v in vars_.items()}, dist, family)
    ll = float(-final_loss.item())
    return params, ll, LanChay(seed=seed, loglik=ll, converged=converged, grad_norm=grad_norm)


def fit_mle(r: torch.Tensor, X: torch.Tensor, dist: str = "student_t", family: str = "garch",
            n_starts: int = 5, base_seed: int = 0, n_adam: int = 400, n_lbfgs: int = 200,
            lr_adam: float = 0.02, h0: torch.Tensor | None = None,
            verbose: bool = False) -> MLEResult:
    """MLE voi `n_starts` diem khoi tao (seed = base_seed, base_seed+1, ...).

    Tra ve nghiem co log-hop-ly HUU HAN lon nhat, kem AIC = 2d - 2l va
    BIC = d log n - 2l. Mo hinh khong co bien ngoai sinh: truyen X co 0 cot.
    """
    if h0 is None:
        h0 = torch.var(r).detach()
    best, runs = None, []
    for i in range(n_starts):
        params, ll, lan = _mot_diem_khoi_tao(base_seed + i, r, X, dist, family, h0,
                                             n_adam, n_lbfgs, lr_adam, verbose)
        runs.append(lan)
        if math.isfinite(ll) and (best is None or ll > best[1]):
            best = (params, ll)
    if best is None:
        raise RuntimeError("Tat ca cac diem khoi tao MLE deu that bai (NaN/Inf).")

    params, ll = best
    d = 4 + (1 if family == "egarch" else 0) + X.shape[1] + (1 if dist == "student_t" else 0)
    aic = -2.0 * ll + 2.0 * d
    bic = -2.0 * ll + d * torch.log(torch.tensor(float(r.shape[0]))).item()
    return MLEResult(params=params, loglik=ll, aic=aic, bic=bic, n_params=d, runs=runs)


# ------------------------------------------------------------ sai so chuan
def _phang(raw: dict, dist: str, family: str) -> torch.Tensor:
    ten = _BIEN_RAW[family] + (["eta_nu"] if dist == "student_t" else [])
    return torch.cat([raw[t].reshape(-1) for t in ten])


def _dung_lai(vec: torch.Tensor, k: int, dist: str, family: str) -> dict:
    ten = _BIEN_RAW[family] + (["eta_nu"] if dist == "student_t" else [])
    raw, pos = {}, 0
    for t in ten:
        n = k if t == "gamma" else 1
        raw[t] = vec[pos] if n == 1 else vec[pos:pos + n]
        pos += n
    return raw


def standard_errors(params, r: torch.Tensor, X: torch.Tensor, dist: str = "student_t",
                    family: str = "garch", h0: torch.Tensor | None = None
                    ) -> dict[str, torch.Tensor] | None:
    """Sai so chuan tren thang TU NHIEN bang Hessian quan sat + phuong phap delta.

        Cov(eta_hat) ~ H^{-1},  H = Hessian cua -l theo bien khong rang buoc eta
        Cov(theta_hat) ~ J H^{-1} J^T,  J = d theta / d eta

    Tra ve None khi Hessian khong kha nghich hoac cho phuong sai am (nghiem
    khong phai cuc dai trong, vd. sat bien): khi do tiem can chuan khong ap
    dung va khoang Wald khong hop le.
    """
    k = X.shape[1]
    if h0 is None:
        h0 = r.var()
    loglik = loglik_cua(family)
    vec = _phang(pack(params, dist, family), dist, family).detach().clone()

    def nll(v):
        return -loglik(unpack(_dung_lai(v, k, dist, family), dist, family), r, X, h0=h0)

    hess = torch.autograd.functional.hessian(nll, vec)
    try:
        cov_raw = torch.linalg.inv(hess)
    except RuntimeError:
        return None
    if (torch.diagonal(cov_raw) < 0).any():
        return None

    def sang_tu_nhien(v):
        p = unpack(_dung_lai(v, k, dist, family), dist, family)
        phan = [p.mu.reshape(-1), p.omega.reshape(-1), p.alpha.reshape(-1)]
        if family == "egarch":
            phan.append(p.theta.reshape(-1))
        phan += [p.beta.reshape(-1), p.gamma.reshape(-1)]
        if dist == "student_t":
            phan.append(p.nu.reshape(-1))
        return torch.cat(phan)

    J = torch.autograd.functional.jacobian(sang_tu_nhien, vec)
    se_phang = torch.sqrt(torch.diagonal(J @ cov_raw @ J.T).clamp(min=0.0))

    ten = _TEN_TU_NHIEN[family] + (["nu"] if dist == "student_t" else [])
    ra, pos = {}, 0
    for t in ten:
        n = k if t == "gamma" else 1
        ra[t] = se_phang[pos:pos + n] if t == "gamma" else se_phang[pos]
        pos += n
    return ra
