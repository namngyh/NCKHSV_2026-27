"""Uoc luong hop ly cuc dai (Muc 4.2): nhieu diem khoi tao + Adam roi tinh L-BFGS.

Quy trinh bam theo cac buoc liet ke trong bai bao:
1. bien doi tham so khong rang buoc sang mien hop le (param_spec.unpack_raw);
2. nhieu diem khoi tao (init_raw voi seed khac nhau);
3. luu log-hop-ly, ma hoi tu va chuan gradient tai nghiem;
4. kiem tra nghiem gan bien (p_ss gan 1, alpha+beta gan 1, nu gan 2);
5. AIC/BIC (eq:aic, eq:bic).

Sai so chuan tu nghich dao Hessian quan sat (buoc 5 trong bai) duoc tinh
rieng trong `hessian_standard_errors` vi day la buoc tuy chon, co the ton
kem voi nhieu tham so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from msgarchx.inference.param_spec import ModelConfig, init_raw, param_slices, unpack_raw
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter
from msgarchx.transforms import raw_from_nu


@dataclass
class MLERunDiagnostics:
    seed: int
    loglik: float
    converged: bool
    grad_norm: float
    near_boundary: bool


@dataclass
class MLEResult:
    params: MSGARCHXParams
    raw: torch.Tensor
    loglik: float
    aic: float
    bic: float
    config: ModelConfig
    runs: list[MLERunDiagnostics] = field(default_factory=list)
    filter_output: dict | None = None


def _neg_loglik(raw: torch.Tensor, config: ModelConfig, r: torch.Tensor, X: torch.Tensor, **filter_kwargs):
    params = unpack_raw(raw, config)
    out = ms_garch_forward_filter(params, r, X, **filter_kwargs)
    return -out["loglik"], out


def _near_boundary(params: MSGARCHXParams) -> bool:
    p_diag = torch.diagonal(params.P)
    persistence = params.alpha + params.beta
    nu_min = params.nu.min() if params.nu.dim() > 0 else params.nu
    return bool(
        (p_diag > 0.995).any()
        or (persistence > 0.995).any()
        or (nu_min < 2.05)
    )


def _single_start(
    seed: int,
    config: ModelConfig,
    r: torch.Tensor,
    X: torch.Tensor,
    n_adam: int,
    n_lbfgs: int,
    lr_adam: float,
    filter_kwargs: dict,
    verbose: bool,
    fixed_nu: float | None = None,
) -> tuple[torch.Tensor, float, MLERunDiagnostics]:
    raw = init_raw(config, seed=seed).clone()
    nu_slice = param_slices(config)["eta_nu"] if fixed_nu is not None else None
    if fixed_nu is not None:
        with torch.no_grad():
            raw[nu_slice] = raw_from_nu(torch.tensor(float(fixed_nu))).expand(nu_slice.stop - nu_slice.start)
    raw.requires_grad_(True)

    def _mask_fixed_nu_grad():
        if fixed_nu is not None and raw.grad is not None:
            raw.grad[nu_slice] = 0.0

    opt = torch.optim.Adam([raw], lr=lr_adam)
    for it in range(n_adam):
        opt.zero_grad()
        loss, _ = _neg_loglik(raw, config, r, X, **filter_kwargs)
        if torch.isnan(loss):
            break
        loss.backward()
        _mask_fixed_nu_grad()
        torch.nn.utils.clip_grad_norm_([raw], max_norm=50.0)
        opt.step()
        if verbose and it % 100 == 0:
            print(f"[seed={seed}] adam iter {it}: nll={loss.item():.4f}")

    lbfgs = torch.optim.LBFGS(
        [raw], lr=0.5, max_iter=n_lbfgs, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        loss, _ = _neg_loglik(raw, config, r, X, **filter_kwargs)
        if torch.isnan(loss):
            return loss
        loss.backward()
        _mask_fixed_nu_grad()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        final_loss, out = _neg_loglik(raw, config, r, X, **filter_kwargs)
    raw_grad = raw.grad
    grad_norm = float(raw_grad.norm().item()) if raw_grad is not None else float("nan")
    converged = bool(torch.isfinite(final_loss)) and grad_norm < 1.0
    params = unpack_raw(raw.detach(), config)
    diag = MLERunDiagnostics(
        seed=seed,
        loglik=float(-final_loss.item()),
        converged=converged,
        grad_norm=grad_norm,
        near_boundary=_near_boundary(params),
    )
    return raw.detach(), float(-final_loss.item()), diag


def fit_mle(
    r: torch.Tensor,
    X: torch.Tensor,
    config: ModelConfig,
    n_starts: int = 5,
    base_seed: int = 0,
    n_adam: int = 400,
    n_lbfgs: int = 200,
    lr_adam: float = 0.02,
    variance_link: str = "clamp",
    floor: float = 1e-8,
    verbose: bool = False,
    fixed_nu: float | None = None,
) -> MLEResult:
    """Chay MLE voi `n_starts` diem khoi tao doc lap, giu nghiem tot nhat.

    Loglik cua cac diem khoi tao khac nhau duoc so sanh de kiem tra tinh da
    dinh cua ham hop ly MS-GARCH (buoc 1 trong Muc 4.2).

    `fixed_nu`: neu duoc dat (vd. 1e6), nu KHONG duoc uoc luong ma giu co
    dinh o gia tri lon, dua Student-t chuan hoa xap xi phan phoi chuan. Dung
    de fit mo hinh doi chung B4 (MS-GARCH chuan) bang chinh khung MS-GARCH-X
    hien co, ma khong can mot mo hinh phan phoi chuan rieng.
    """
    filter_kwargs = {"variance_link": variance_link, "floor": floor}
    best_raw, best_loglik, best_diag = None, -float("inf"), None
    runs = []
    for i in range(n_starts):
        raw, loglik, diag = _single_start(
            base_seed + i, config, r, X, n_adam, n_lbfgs, lr_adam, filter_kwargs, verbose,
            fixed_nu=fixed_nu,
        )
        runs.append(diag)
        if loglik > best_loglik and torch.isfinite(torch.tensor(loglik)):
            best_raw, best_loglik, best_diag = raw, loglik, diag

    if best_raw is None:
        raise RuntimeError("Tat ca cac diem khoi tao MLE deu that bai (NaN/Inf).")

    params = unpack_raw(best_raw, config)
    with torch.no_grad():
        filter_out = ms_garch_forward_filter(params, r, X, **filter_kwargs)

    n_nu_params = 2 if config.state_specific_nu else 1
    k = config.raw_dim - n_nu_params if fixed_nu is not None else config.raw_dim
    aic = -2.0 * best_loglik + 2.0 * k
    bic = -2.0 * best_loglik + k * torch.log(torch.tensor(float(r.shape[0]))).item()

    return MLEResult(
        params=params,
        raw=best_raw,
        loglik=best_loglik,
        aic=aic,
        bic=bic,
        config=config,
        runs=runs,
        filter_output=filter_out,
    )


def hessian_standard_errors(raw: torch.Tensor, config: ModelConfig, r: torch.Tensor, X: torch.Tensor, **filter_kwargs) -> torch.Tensor:
    """Sai so chuan tu nghich dao Hessian quan sat cua raw-vector tai nghiem MLE.

    Tra ve None neu Hessian khong xac dinh duong (can bootstrap thay the,
    xem Muc 4.2 buoc 6).
    """
    raw = raw.clone().requires_grad_(True)

    def nll(x):
        loss, _ = _neg_loglik(x, config, r, X, **filter_kwargs)
        return loss

    hess = torch.autograd.functional.hessian(nll, raw)
    try:
        cov = torch.linalg.inv(hess)
    except RuntimeError:
        return None
    diag = torch.diagonal(cov)
    if (diag < 0).any():
        return None
    return torch.sqrt(diag)


def _flatten_natural(params: MSGARCHXParams) -> torch.Tensor:
    parts = [
        params.mu, params.omega, params.alpha, params.beta, params.gamma.reshape(-1),
        params.P[0, 0].reshape(1), params.P[1, 1].reshape(1), params.nu.reshape(-1),
    ]
    return torch.cat(parts)


def natural_scale_standard_errors(
    raw: torch.Tensor, config: ModelConfig, r: torch.Tensor, X: torch.Tensor, **filter_kwargs
) -> dict[str, torch.Tensor] | None:
    """Sai so chuan tren THANG DO TU NHIEN (mu, omega, alpha, beta, gamma,
    p11, p22, nu) bang phuong phap delta: Cov_tu_nhien = J Cov_raw J^T, voi
    J = d(tham so tu nhien)/d(raw) va Cov_raw = nghich dao Hessian quan sat
    cua -log-hop-ly theo raw. Can thiet vi toi uu hoa dien ra tren khong gian
    RAW khong rang buoc (Muc 4.2), nen Hessian tho tren raw KHONG the dung
    truc tiep lam sai so chuan cho tham so tu nhien.

    Tra ve None neu Hessian raw khong xac dinh duong.
    """
    raw = raw.clone().requires_grad_(True)

    def nll(x):
        loss, _ = _neg_loglik(x, config, r, X, **filter_kwargs)
        return loss

    hess = torch.autograd.functional.hessian(nll, raw)
    try:
        cov_raw = torch.linalg.inv(hess)
    except RuntimeError:
        return None
    if (torch.diagonal(cov_raw) < 0).any():
        return None

    def natural_map(x):
        return _flatten_natural(unpack_raw(x, config))

    J = torch.autograd.functional.jacobian(natural_map, raw.detach())
    cov_natural = J @ cov_raw.detach() @ J.T
    diag = torch.diagonal(cov_natural).clamp(min=0.0)
    se_flat = torch.sqrt(diag)

    k = config.n_exog
    n_nu = 2 if config.state_specific_nu else 1
    idx, pos = {}, 0
    for name, size in [
        ("mu", 2), ("omega", 2), ("alpha", 2), ("beta", 2), ("gamma", 2 * k),
        ("p11", 1), ("p22", 1), ("nu", n_nu),
    ]:
        idx[name] = slice(pos, pos + size)
        pos += size
    return {name: se_flat[sl] for name, sl in idx.items()}
