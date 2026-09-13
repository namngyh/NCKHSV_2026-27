"""MLE cho GARCH(1,1)/GARCH-X mot trang thai (mo hinh doi chung B1-B3, Bang 5.6).

Cung quy trinh nhu inference/mle.py (nhieu diem khoi tao, Adam + L-BFGS, chan
doan gradient) nhung ap dung cho models.garch (mot trang thai) thay vi
models.ms_garch (hai trang thai).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from msgarchx.models.egarch import EGARCHParams, egarch_loglik
from msgarchx.models.garch import GARCHParams, garch_loglik


@dataclass
class GARCHMLERunDiagnostics:
    seed: int
    loglik: float
    converged: bool
    grad_norm: float


@dataclass
class GARCHMLEResult:
    params: GARCHParams | EGARCHParams
    loglik: float
    aic: float
    bic: float
    n_params: int
    runs: list[GARCHMLERunDiagnostics] = field(default_factory=list)


def _unpack_garch(vars_: dict, dist: str) -> GARCHParams:
    """GARCH: omega > 0 qua exp, alpha/beta >= 0 qua softplus (eq:garchx)."""
    nu = 2.0 + torch.nn.functional.softplus(vars_["eta_nu"]) if dist == "student_t" else None
    return GARCHParams(
        mu=vars_["mu"],
        omega=torch.exp(vars_["log_omega"]),
        alpha=torch.nn.functional.softplus(vars_["eta_alpha"]),
        beta=torch.nn.functional.softplus(vars_["eta_beta"]),
        gamma=vars_["gamma"],
        nu=nu,
    )


def _unpack_egarch(vars_: dict, dist: str) -> EGARCHParams:
    """EGARCH: omega/alpha/theta TU DO, beta bi rang buoc vao (0, 1) qua sigmoid.

    Khac han GARCH: vi phuong trinh viet cho log h nen khong he so nao can
    duong ca (as:egarch).

    VI SAO (0,1) CHU KHONG PHAI (-1,1): dieu kien dung ve ly thuyet chi doi
    hoi |beta| < 1. Nhung tien nghiem Bayes cua beta trong bayes_single.py la
    Beta(a,a) tren (0,1), nen neu MLE tim tren (-1,1) thi BA PHUONG PHAP SE
    TIM TREN HAI KHONG GIAN THAM SO KHAC NHAU -- dung cai ma nguyen tac "cung
    mot ham hop ly" cua bai bao cam. Do dai dang AM cua log-phuong sai cung
    khong co y nghia kinh te. Nen ca ba phuong phap cung dung (0,1).
    """
    nu = 2.0 + torch.nn.functional.softplus(vars_["eta_nu"]) if dist == "student_t" else None
    return EGARCHParams(
        mu=vars_["mu"],
        omega=vars_["omega"],
        alpha=vars_["alpha"],
        theta=vars_["theta"],
        beta=torch.sigmoid(vars_["eta_beta"]),
        gamma=vars_["gamma"],
        nu=nu,
    )


def _unpack(vars_: dict, dist: str, family: str):
    if family == "garch":
        return _unpack_garch(vars_, dist)
    if family == "egarch":
        return _unpack_egarch(vars_, dist)
    raise ValueError(f"family khong hop le: {family!r} (chi nhan 'garch' hoac 'egarch')")


def _loglik(params, r, X, h0, family: str) -> torch.Tensor:
    if family == "garch":
        return garch_loglik(params, r, X, h0=h0)
    return egarch_loglik(params, r, X, h0=h0)


def _init_raw(dist: str, family: str, k: int, h0: torch.Tensor) -> dict:
    """Diem khoi tao ngau nhien tren khong gian KHONG rang buoc cua tung ho."""
    if family == "garch":
        raw = {
            "mu": torch.zeros(()),
            "log_omega": torch.log(h0 * 0.1) + 0.1 * torch.randn(()),
            "eta_alpha": torch.tensor(-1.5) + 0.2 * torch.randn(()),
            "eta_beta": torch.tensor(1.0) + 0.2 * torch.randn(()),
            "gamma": 0.01 * torch.randn(k),
        }
    else:
        # Khoi tao EGARCH: nham vao trang thai dung log h ~ log(h0). Voi
        # beta ~ 0.9 thi omega ~ (1 - beta) * log(h0) de muc nen khop du lieu.
        # eta_beta khoi tao tai logit(0.9) vi beta = sigmoid(eta_beta).
        beta0 = 0.9
        logit_beta0 = torch.log(torch.tensor(beta0 / (1.0 - beta0)))
        raw = {
            "mu": torch.zeros(()),
            "omega": (1.0 - beta0) * torch.log(h0) + 0.05 * torch.randn(()),
            "alpha": torch.tensor(0.10) + 0.02 * torch.randn(()),
            "theta": torch.tensor(0.0) + 0.02 * torch.randn(()),   # khong gia dinh dau don bay
            "eta_beta": logit_beta0 + 0.1 * torch.randn(()),
            "gamma": 0.01 * torch.randn(k),
        }
    if dist == "student_t":
        raw["eta_nu"] = torch.tensor(1.5) + 0.2 * torch.randn(())
    return raw


def _single_start(seed, r, X, dist, family, h0, n_adam, n_lbfgs, lr_adam, verbose):
    torch.manual_seed(seed)
    k = X.shape[1]
    raw = _init_raw(dist, family, k, h0)
    vars_ = {name: v.clone().requires_grad_(True) for name, v in raw.items()}
    params_list = list(vars_.values())

    opt = torch.optim.Adam(params_list, lr=lr_adam)
    for it in range(n_adam):
        opt.zero_grad()
        loss = -_loglik(_unpack(vars_, dist, family), r, X, h0, family)
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_list, 50.0)
        opt.step()
        if verbose and it % 200 == 0:
            print(f"  [seed={seed}] adam {it}: nll={loss.item():.4f}")

    lbfgs = torch.optim.LBFGS(params_list, lr=0.5, max_iter=n_lbfgs, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss = -_loglik(_unpack(vars_, dist, family), r, X, h0, family)
        if torch.isnan(loss):
            return loss
        loss.backward()
        return loss

    before_lbfgs = {k_: v.detach().clone() for k_, v in vars_.items()}
    try:
        lbfgs.step(closure)
    except (IndexError, RuntimeError) as exc:
        # torch.optim.LBFGS._strong_wolfe co the vo hieu (IndexError tren
        # `bracket`) khi ham muc tieu tra ve NaN/Inf giua luc do duong -- xay
        # ra khi mot buoc lon day tham so ra ngoai vung hop le. Closure o tren
        # da luong truoc NaN nhung L-BFGS cua torch thi khong. Bat loi, quay
        # ve nghiem cua giai doan Adam va danh dau diem khoi tao nay la KHONG
        # hoi tu, thay vi lam sap ca qua trinh nhieu diem khoi tao.
        print(f"  [seed={seed}] L-BFGS that bai ({type(exc).__name__}: {exc}); "
              f"giu nghiem cua giai doan Adam.")
        with torch.no_grad():
            for k_, v in vars_.items():
                v.copy_(before_lbfgs[k_])
                v.grad = None
        recomputed = -_loglik(_unpack(vars_, dist, family), r, X, h0, family)
        if torch.isfinite(recomputed):
            recomputed.backward()

    with torch.no_grad():
        final_loss = -_loglik(_unpack(vars_, dist, family), r, X, h0, family)
    grad_norm = float(
        sum((v.grad.norm() ** 2 for v in params_list if v.grad is not None), torch.tensor(0.0)).sqrt()
    )
    converged = bool(torch.isfinite(final_loss)) and grad_norm < 1.0
    params = _unpack({k_: v.detach() for k_, v in vars_.items()}, dist, family)
    diag = GARCHMLERunDiagnostics(seed=seed, loglik=float(-final_loss.item()), converged=converged, grad_norm=grad_norm)
    return params, float(-final_loss.item()), diag


def fit_garch_mle(
    r: torch.Tensor,
    X: torch.Tensor,
    dist: str = "student_t",
    family: str = "garch",
    n_starts: int = 5,
    base_seed: int = 0,
    n_adam: int = 400,
    n_lbfgs: int = 200,
    lr_adam: float = 0.02,
    h0: torch.Tensor | None = None,
    verbose: bool = False,
) -> GARCHMLEResult:
    """MLE cho mo hinh MOT trang thai, ho `family` ("garch" hoac "egarch").

    Mo hinh khong dung bien ngoai sinh thi truyen X co 0 cot: khi do gamma la
    vector rong va so hang ngoai sinh bang 0.
    """
    if h0 is None:
        h0 = torch.var(r).detach()
    best = None
    runs = []
    for i in range(n_starts):
        params, loglik, diag = _single_start(
            base_seed + i, r, X, dist, family, h0, n_adam, n_lbfgs, lr_adam, verbose
        )
        runs.append(diag)
        # `loglik > best[1]` luon False khi loglik la NaN, nen neu diem khoi
        # tao DAU TIEN cho NaN thi `best` se ket lai o NaN vinh vien. Kiem tra
        # huu han truoc, giong cach fit_mle (2 trang thai) da lam.
        if torch.isfinite(torch.tensor(loglik)) and (best is None or loglik > best[1]):
            best = (params, loglik)

    if best is None:
        raise RuntimeError("Tat ca cac diem khoi tao MLE deu that bai (NaN/Inf).")
    params, loglik = best
    # mu, omega, alpha, beta (+ theta neu EGARCH) + gamma + (nu)
    n_params = (
        4
        + (1 if family == "egarch" else 0)
        + X.shape[1]
        + (1 if dist == "student_t" else 0)
    )
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + n_params * torch.log(torch.tensor(float(r.shape[0]))).item()
    return GARCHMLEResult(params=params, loglik=loglik, aic=aic, bic=bic, n_params=n_params, runs=runs)

# ===========================================================================
# Sai so chuan tu Hessian quan sat (mo hinh MOT trang thai)
#
# Ban song song cua natural_scale_standard_errors() trong mle.py, von chi lam
# viec voi MS-GARCH-X. Tach rieng vi hai ho dung cau truc tham so khac nhau:
# MS-GARCH-X dung mot vector `raw` phang, con ho mot trang thai dung mot dict
# cac bien khong rang buoc.
# ===========================================================================

# Thu tu phang hoa, dung chung cho ca hai chieu (phang hoa va dat ten lai).
_THU_TU = {
    "garch": ["mu", "log_omega", "eta_alpha", "eta_beta", "gamma"],
    "egarch": ["mu", "omega", "alpha", "theta", "eta_beta", "gamma"],
}
# Ten tham so TU NHIEN tuong ung, theo dung thu tu tren.
_TEN_TU_NHIEN = {
    "garch": ["mu", "omega", "alpha", "beta", "gamma"],
    "egarch": ["mu", "omega", "alpha", "theta", "beta", "gamma"],
}


def _nghich_dao_softplus(x: torch.Tensor) -> torch.Tensor:
    """Nghich dao cua softplus. Dung expm1 thay vi exp(x)-1 de on dinh khi x nho."""
    return torch.log(torch.expm1(x))


def _raw_tu_params(params, dist: str, family: str) -> dict:
    """Dua tham so TU NHIEN ve lai khong gian KHONG rang buoc.

    Can thiet vi GARCHMLEResult chi luu tham so tu nhien, trong khi Hessian
    phai lay tren khong gian toi uu hoa that su dien ra (khong rang buoc) --
    Hessian tho tren thang tu nhien se sai o nhung tham so co rang buoc.
    """
    if family == "garch":
        raw = {
            "mu": params.mu,
            "log_omega": torch.log(params.omega),
            "eta_alpha": _nghich_dao_softplus(params.alpha),
            "eta_beta": _nghich_dao_softplus(params.beta),
            "gamma": params.gamma,
        }
    else:
        raw = {
            "mu": params.mu,
            "omega": params.omega,
            "alpha": params.alpha,
            "theta": params.theta,
            "eta_beta": torch.log(params.beta / (1.0 - params.beta)),  # nghich dao sigmoid
            "gamma": params.gamma,
        }
    if dist == "student_t":
        raw["eta_nu"] = _nghich_dao_softplus(params.nu - 2.0)
    return raw


def _phang(raw: dict, dist: str, family: str) -> torch.Tensor:
    ten = list(_THU_TU[family]) + (["eta_nu"] if dist == "student_t" else [])
    return torch.cat([raw[t].reshape(-1) for t in ten])


def _dung_lai(vec: torch.Tensor, k: int, dist: str, family: str) -> dict:
    ten = list(_THU_TU[family]) + (["eta_nu"] if dist == "student_t" else [])
    raw, pos = {}, 0
    for t in ten:
        n = k if t == "gamma" else 1
        raw[t] = vec[pos] if n == 1 else vec[pos:pos + n]
        pos += n
    return raw


def standard_errors_single(
    params, r: torch.Tensor, X: torch.Tensor, dist: str = "student_t",
    family: str = "garch", h0: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | None:
    """Sai so chuan tren thang do TU NHIEN cho mo hinh MOT trang thai.

    Phuong phap delta: Cov_tu_nhien = J Cov_raw J^T, voi Cov_raw la nghich dao
    Hessian quan sat cua -log-likelihood theo cac bien KHONG rang buoc, va
    J = d(tham so tu nhien)/d(bien khong rang buoc).

    Tra ve None neu Hessian khong kha nghich hoac cho phuong sai am -- truong
    hop do nghia la nghiem khong phai cuc dai trong (vd. nam sat bien), khi ay
    tiem can chuan cua uoc luong hop ly cuc dai khong ap dung va sai so chuan
    tu Hessian KHONG hop le. Nguoi goi phai xu ly None chu dung coi la 0.
    """
    k = X.shape[1]
    if h0 is None:
        h0 = r.var()
    vec = _phang(_raw_tu_params(params, dist, family), dist, family).detach().clone()

    def nll(v):
        return -_loglik(_unpack(_dung_lai(v, k, dist, family), dist, family), r, X, h0, family)

    hess = torch.autograd.functional.hessian(nll, vec)
    try:
        cov_raw = torch.linalg.inv(hess)
    except RuntimeError:
        return None
    if (torch.diagonal(cov_raw) < 0).any():
        return None

    def sang_tu_nhien(v):
        p = _unpack(_dung_lai(v, k, dist, family), dist, family)
        phan = [p.mu.reshape(-1), p.omega.reshape(-1), p.alpha.reshape(-1)]
        if family == "egarch":
            phan.append(p.theta.reshape(-1))
        phan += [p.beta.reshape(-1), p.gamma.reshape(-1)]
        if dist == "student_t":
            phan.append(p.nu.reshape(-1))
        return torch.cat(phan)

    J = torch.autograd.functional.jacobian(sang_tu_nhien, vec)
    se_phang = torch.sqrt(torch.diagonal(J @ cov_raw @ J.T).clamp(min=0.0))

    ten = list(_TEN_TU_NHIEN[family]) + (["nu"] if dist == "student_t" else [])
    ra, pos = {}, 0
    for t in ten:
        n = k if t == "gamma" else 1
        ra[t] = se_phang[pos:pos + n] if t == "gamma" else se_phang[pos]
        pos += n
    return ra
