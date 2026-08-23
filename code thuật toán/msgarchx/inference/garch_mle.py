"""MLE cho GARCH(1,1)/GARCH-X mot trang thai (mo hinh doi chung B1-B3, Bang 5.6).

Cung quy trinh nhu inference/mle.py (nhieu diem khoi tao, Adam + L-BFGS, chan
doan gradient) nhung ap dung cho models.garch (mot trang thai) thay vi
models.ms_garch (hai trang thai).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from msgarchx.models.garch import GARCHParams, garch_loglik


@dataclass
class GARCHMLERunDiagnostics:
    seed: int
    loglik: float
    converged: bool
    grad_norm: float


@dataclass
class GARCHMLEResult:
    params: GARCHParams
    loglik: float
    aic: float
    bic: float
    n_params: int
    runs: list[GARCHMLERunDiagnostics] = field(default_factory=list)


def _unpack(vars_: dict, dist: str) -> GARCHParams:
    nu = 2.0 + torch.nn.functional.softplus(vars_["eta_nu"]) if dist == "student_t" else None
    return GARCHParams(
        mu=vars_["mu"],
        omega=torch.exp(vars_["log_omega"]),
        alpha=torch.nn.functional.softplus(vars_["eta_alpha"]),
        beta=torch.nn.functional.softplus(vars_["eta_beta"]),
        gamma=vars_["gamma"],
        nu=nu,
    )


def _single_start(seed, r, X, dist, h0, n_adam, n_lbfgs, lr_adam, verbose):
    torch.manual_seed(seed)
    k = X.shape[1]
    raw = {
        "mu": torch.zeros(()),
        "log_omega": torch.log(h0 * 0.1) + 0.1 * torch.randn(()),
        "eta_alpha": torch.tensor(-1.5) + 0.2 * torch.randn(()),
        "eta_beta": torch.tensor(1.0) + 0.2 * torch.randn(()),
        "gamma": 0.01 * torch.randn(k),
    }
    if dist == "student_t":
        raw["eta_nu"] = torch.tensor(1.5) + 0.2 * torch.randn(())
    vars_ = {name: v.clone().requires_grad_(True) for name, v in raw.items()}
    params_list = list(vars_.values())

    opt = torch.optim.Adam(params_list, lr=lr_adam)
    for it in range(n_adam):
        opt.zero_grad()
        loss = -garch_loglik(_unpack(vars_, dist), r, X, h0=h0)
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
        loss = -garch_loglik(_unpack(vars_, dist), r, X, h0=h0)
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
        recomputed = -garch_loglik(_unpack(vars_, dist), r, X, h0=h0)
        if torch.isfinite(recomputed):
            recomputed.backward()

    with torch.no_grad():
        final_loss = -garch_loglik(_unpack(vars_, dist), r, X, h0=h0)
    grad_norm = float(
        sum((v.grad.norm() ** 2 for v in params_list if v.grad is not None), torch.tensor(0.0)).sqrt()
    )
    converged = bool(torch.isfinite(final_loss)) and grad_norm < 1.0
    params = _unpack({k_: v.detach() for k_, v in vars_.items()}, dist)
    diag = GARCHMLERunDiagnostics(seed=seed, loglik=float(-final_loss.item()), converged=converged, grad_norm=grad_norm)
    return params, float(-final_loss.item()), diag


def fit_garch_mle(
    r: torch.Tensor,
    X: torch.Tensor,
    dist: str = "student_t",
    n_starts: int = 5,
    base_seed: int = 0,
    n_adam: int = 400,
    n_lbfgs: int = 200,
    lr_adam: float = 0.02,
    h0: torch.Tensor | None = None,
    verbose: bool = False,
) -> GARCHMLEResult:
    if h0 is None:
        h0 = torch.var(r).detach()
    best = None
    runs = []
    for i in range(n_starts):
        params, loglik, diag = _single_start(base_seed + i, r, X, dist, h0, n_adam, n_lbfgs, lr_adam, verbose)
        runs.append(diag)
        # `loglik > best[1]` luon False khi loglik la NaN, nen neu diem khoi
        # tao DAU TIEN cho NaN thi `best` se ket lai o NaN vinh vien. Kiem tra
        # huu han truoc, giong cach fit_mle (2 trang thai) da lam.
        if torch.isfinite(torch.tensor(loglik)) and (best is None or loglik > best[1]):
            best = (params, loglik)

    if best is None:
        raise RuntimeError("Tat ca cac diem khoi tao MLE deu that bai (NaN/Inf).")
    params, loglik = best
    n_params = 4 + X.shape[1] + (1 if dist == "student_t" else 0)  # mu, omega, alpha, beta + gamma + (nu)
    aic = -2.0 * loglik + 2.0 * n_params
    bic = -2.0 * loglik + n_params * torch.log(torch.tensor(float(r.shape[0]))).item()
    return GARCHMLEResult(params=params, loglik=loglik, aic=aic, bic=bic, n_params=n_params, runs=runs)
