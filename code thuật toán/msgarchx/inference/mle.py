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



# ===========================================================================
# Dieu kien chinh quy cua nghiem (Muc 4.2, buoc chon nghiem)
#
# Ham hop ly cua mo hinh chuyen trang thai co CAC DINH GIA voi gia tri tuy y
# cao: mot trang thai thoai hoa thanh "bay bat diem ngoai le" chi ton tai dung
# mot phien se keo log-hop-ly len ma khong mang y nghia kinh te nao. Day la
# benh da biet cua lop mo hinh hon hop, khong phai loi cai dat. Thuc hanh
# chuan la bao cao CUC DAI DIA PHUONG LON NHAT THOA DIEU KIEN CHINH QUY.
#
#   C1  Da hoi tu       : ||grad|| < 1 tren thang khong rang buoc
#   C2  Moi trang thai la MOT CHE DO : d_s = 1/(1-p_ss) >= MIN_DURATION phien
#   C3  p_ss khong sat bien          : P_LOW < p_ss < P_HIGH
#
# KHONG ap dieu kien alpha_s + beta_s < 1 cho tung trang thai: trong MS-GARCH
# mot che do rieng le duoc phep no ma qua trinh tong the van dung, nen do se la
# rang buoc do nguoi nghien cuu tu dat chu khong phai yeu cau ly thuyet.
# ===========================================================================
MIN_DURATION = 2.0
P_LOW, P_HIGH = 0.01, 0.999


def kiem_tra_chinh_quy(params: MSGARCHXParams) -> tuple[bool, str]:
    """Tra ve (dat_chinh_quy, ly_do_neu_khong_dat)."""
    p_diag = torch.diagonal(params.P)
    dur = 1.0 / (1.0 - p_diag)
    ly_do = []
    if bool((dur < MIN_DURATION).any()):
        ly_do.append(f"d_s={[round(v, 2) for v in dur.tolist()]} < {MIN_DURATION}")
    if bool((p_diag <= P_LOW).any() or (p_diag >= P_HIGH).any()):
        ly_do.append(f"p_ss={[round(v, 4) for v in p_diag.tolist()]} sat bien")
    return (not ly_do), "; ".join(ly_do)


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
    init_raw_vec: torch.Tensor | None = None,
    n_cycles: int = 1,
) -> tuple[torch.Tensor, float, MLERunDiagnostics]:
    raw = (init_raw_vec.detach().clone() if init_raw_vec is not None
           else init_raw(config, seed=seed).clone())
    nu_slice = param_slices(config)["eta_nu"] if fixed_nu is not None else None
    if fixed_nu is not None:
        with torch.no_grad():
            raw[nu_slice] = raw_from_nu(torch.tensor(float(fixed_nu))).expand(nu_slice.stop - nu_slice.start)
    raw.requires_grad_(True)

    def _mask_fixed_nu_grad():
        if fixed_nu is not None and raw.grad is not None:
            raw.grad[nu_slice] = 0.0

    # Mot "chu ky" = Adam roi L-BFGS. Lap lai toi da `n_cycles` lan, dung
    # ngay khi ||grad|| < 1 (tieu chi hoi tu o Muc 4.2). Ly do: tren du lieu
    # that, mot chu ky duy nhat thuong dung lai o diem co ||grad|| rat lon --
    # tuc L-BFGS dung tren vach chu khong tren dinh. Danh bong them vai chu ky
    # KHONG doi bai toan uoc luong, chi co gang giai no ky hon.
    for cycle in range(max(1, n_cycles)):
        lr_cycle = lr_adam / (2.0 ** cycle)
        _run_one_cycle(raw, config, r, X, n_adam, n_lbfgs, lr_cycle, filter_kwargs,
                       verbose, seed, cycle, _mask_fixed_nu_grad)
        with torch.no_grad():
            probe, _ = _neg_loglik(raw, config, r, X, **filter_kwargs)
        raw.grad = None
        probe2, _ = _neg_loglik(raw, config, r, X, **filter_kwargs)
        if torch.isfinite(probe2):
            probe2.backward()
            _mask_fixed_nu_grad()
        gn = float(raw.grad.norm().item()) if raw.grad is not None else float("nan")
        if verbose:
            print(f"[seed={seed}] het chu ky {cycle}: nll={float(probe):.4f} ||grad||={gn:.4f}")
        if gn == gn and gn < 1.0:
            break

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


def _run_one_cycle(raw, config, r, X, n_adam, n_lbfgs, lr_adam, filter_kwargs,
                   verbose, seed, cycle, _mask_fixed_nu_grad):
    """Mot chu ky Adam roi L-BFGS, cap nhat `raw` tai cho."""
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

    raw_before_lbfgs = raw.detach().clone()
    try:
        lbfgs.step(closure)
    except (IndexError, RuntimeError) as exc:
        # torch.optim.LBFGS._strong_wolfe co the vo hieu (IndexError tren
        # `bracket`) khi ham muc tieu tra ve NaN/Inf giua luc do duong -- xay
        # ra khi mot buoc lon day tham so ra ngoai vung hop le. Closure o tren
        # da luong truoc NaN nhung L-BFGS cua torch thi khong. Bat loi, quay
        # ve nghiem cua giai doan Adam va danh dau diem khoi tao nay la KHONG
        # hoi tu, thay vi lam sap ca qua trinh nhieu diem khoi tao.
        print(f"[seed={seed}] L-BFGS that bai ({type(exc).__name__}: {exc}); "
              f"giu nghiem cua giai doan Adam.")
        with torch.no_grad():
            raw.copy_(raw_before_lbfgs)
        raw.grad = None
        recomputed, _ = _neg_loglik(raw, config, r, X, **filter_kwargs)
        if torch.isfinite(recomputed):
            recomputed.backward()
            _mask_fixed_nu_grad()


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
    init_raws: list[torch.Tensor] | None = None,
    n_cycles: int = 1,
    select: str = "max_loglik",
) -> MLEResult:
    """Chay MLE voi `n_starts` diem khoi tao doc lap, giu nghiem tot nhat.

    Loglik cua cac diem khoi tao khac nhau duoc so sanh de kiem tra tinh da
    dinh cua ham hop ly MS-GARCH (buoc 1 trong Muc 4.2).

    `fixed_nu`: neu duoc dat (vd. 1e6), nu KHONG duoc uoc luong ma giu co
    dinh o gia tri lon, dua Student-t chuan hoa xap xi phan phoi chuan. Dung
    de fit mo hinh doi chung B4 (MS-GARCH chuan) bang chinh khung MS-GARCH-X
    hien co, ma khong can mot mo hinh phan phoi chuan rieng.
    """
    if select not in ("max_loglik", "regular"):
        raise ValueError(f"select khong hop le: {select!r}")
    filter_kwargs = {"variance_link": variance_link, "floor": floor}
    runs, ket_qua = [], []
    for i in range(n_starts):
        iv = init_raws[i] if (init_raws is not None and i < len(init_raws)) else None
        raw, loglik, diag = _single_start(
            base_seed + i, config, r, X, n_adam, n_lbfgs, lr_adam, filter_kwargs, verbose,
            fixed_nu=fixed_nu, init_raw_vec=iv, n_cycles=n_cycles,
        )
        runs.append(diag)
        if torch.isfinite(torch.tensor(loglik)):
            ket_qua.append({"raw": raw, "loglik": loglik, "diag": diag,
                            "params": unpack_raw(raw, config)})

    if not ket_qua:
        raise RuntimeError("Tat ca cac diem khoi tao MLE deu that bai (NaN/Inf).")

    if select == "regular":
        for k in ket_qua:
            k["ok"], k["ly_do"] = kiem_tra_chinh_quy(k["params"])
        chinh_quy = [k for k in ket_qua if k["diag"].converged and k["ok"]]
        hoi_tu = [k for k in ket_qua if k["diag"].converged]
        pool = chinh_quy or hoi_tu or ket_qua
        if verbose and not chinh_quy:
            print("[fit_mle] CANH BAO: khong diem khoi tao nao vua hoi tu vua chinh quy")
    else:
        pool = ket_qua

    best = max(pool, key=lambda k: k["loglik"])
    if verbose and select == "regular":
        bo_qua = [k["loglik"] for k in ket_qua if k["loglik"] > best["loglik"]]
        if bo_qua:
            print(f"[fit_mle] da BO QUA {len(bo_qua)} nghiem log-hop-ly cao hon "
                  f"(cao nhat {max(bo_qua):.4f}) vi khong thoa dieu kien chinh quy")
    best_raw, best_loglik, best_diag = best["raw"], best["loglik"], best["diag"]

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
