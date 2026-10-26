"""Ước lượng và suy diễn: MLE, sai số chuẩn, khả nghịch, bootstrap, VB mean-field/full-rank, NUTS, dự báo.

LÝ THUYẾT:
  MLE (mục 4.2): xi_hat = argmax l_T(g(xi)) bằng L-BFGS với gradient vi phân tự động; ít nhất 20 điểm khởi tạo,
    điểm đầu là nghiệm mô hình lồng (phân phối chuẩn, gamma = 0), các điểm còn lại ngẫu nhiên quanh đó.
  Sai số chuẩn: Hessian A^-1 và bánh kẹp A^-1 B A^-1 với A = -tổng Hessian, B = tổng tích điểm số (mục 4.2).
  Khả nghịch (3.3.5): lọc ln h từ hai giá trị khởi tạo, chênh lệch phải tắt dần.
  Bootstrap tham số khi beta sát biên (4.2).
  VB (4.3.3): q_MF = N(m, diag s^2), q_FR = N(m, L L'), cực đại ELBO; chẩn đoán Pareto k_hat của trọng số
    ln p(xi | D) - ln q(xi), và độ ổn định ELBO qua nhiều lần khởi tạo.
  NUTS (4.3.4): ma trận khối lượng đầy đủ thích nghi; R_hat chuẩn hóa thứ hạng, ESS phần thân và phần đuôi, số phân kỳ.
  Dự báo (5.3, 5.4): mật độ cắm tham số của MLE; mật độ dự báo hậu nghiệm f_t(r) = (1/S) tổng f(r | psi^(s)).

THƯ VIỆN: jax, numpyro (SVI, AutoNormal, AutoMultivariateNormal, NUTS), scipy.optimize, arviz (rhat, ess, psislw).
"""
from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

from . import du_lieu as DL
from . import phan_phoi as PP
from .mo_hinh_jax import TIEN_NGHIEM_GOC, DacTa, e_abs, g_nghich, ham, mo_hinh_numpyro


# ----------------------------------------------------------------------------------------
# Dữ liệu đầu vào cho hàm hợp lý
# ----------------------------------------------------------------------------------------
def chuan_bi(f: dict, bien, s_cuoi: int):
    """Bộ (r, J_soc, Z, mat_na, lh0) từ phiên BO_QUA đến hết bảng; cửa sổ ước lượng = [BO_QUA, s_cuoi).

    f là kết quả du_lieu.dac_trung(d, s) với s = s_cuoi - 1 (thống kê chuẩn hóa chỉ dùng dữ liệu đến s)."""
    bq = DL.BO_QUA
    r = f["r"][bq:]
    J = (np.asarray(f["J"], bool) | np.isnan(f["r"]))[bq:]
    cot = [DL.TEN_BIEN.index(b) for b in bien]
    Z = np.column_stack([f["X"][bq:, cot], f["D"][bq:]]) if cot else f["D"][bq:]
    mask = (~J) & (np.arange(len(r)) + bq < s_cuoi)
    lh0 = math.log(float(np.nanvar(f["r"][bq:bq + 20], ddof=1)))
    return (jnp.asarray(np.nan_to_num(r)), jnp.asarray(J), jnp.asarray(np.ascontiguousarray(Z, float)),
            jnp.asarray(mask), jnp.asarray(lh0))


def mat_na_doan(dl, dau: int, cuoi: int):
    """Đổi mặt nạ thành các phiên có lợi suất trong [dau, cuoi) (chỉ số trên bảng ngày)."""
    n = dl[0].shape[0]
    idx = np.arange(n) + DL.BO_QUA
    m = (~np.asarray(dl[1])) & (idx >= dau) & (idx < cuoi)
    return (dl[0], dl[1], dl[2], jnp.asarray(m), dl[4])


def bo_bien(dl, nx: int):
    """Dữ liệu cho mô hình lồng gamma = 0 (giữ biến giả lịch)."""
    return (dl[0], dl[1], dl[2][:, nx:], dl[3], dl[4])


# ----------------------------------------------------------------------------------------
# MLE
# ----------------------------------------------------------------------------------------
def _toi_uu(spec: DacTa, dl, diem_xi):
    h = ham(spec.pp, spec.nx)

    def f(x):
        v, gr = h.am_ll_xi_grad(jnp.asarray(x), dl)
        v, gr = float(v), np.asarray(gr, float)
        if not (math.isfinite(v) and np.isfinite(gr).all()):
            return 1e20, np.zeros_like(gr)
        return v, gr

    kq = []
    for x0 in diem_xi:
        try:
            kq.append(optimize.minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                                        options={"maxiter": 3000, "gtol": 1e-6, "ftol": 1e-13, "maxls": 50}))
        except (ValueError, FloatingPointError, OverflowError):
            continue
    kq = [k for k in kq if np.isfinite(k.fun) and k.fun < 1e19]
    if not kq:
        raise RuntimeError(f"MLE {spec}: khong co nghiem huu han")
    return min(kq, key=lambda k: k.fun), np.array([-k.fun for k in kq])


def _psi_long(dl, nx: int):
    r = np.asarray(dl[0])[np.asarray(dl[3])]
    var, mu = float(r.var()), float(r.mean())
    return [np.array([mu, math.log(var) * (1 - be), 0.1, -0.05, be, 0.0, 0.0, 0.0]) for be in (0.5, 0.8, 0.9, 0.95, 0.98)]


def _mo_rong(psi0: np.ndarray, spec: DacTa) -> np.ndarray:
    """Nghiệm mô hình lồng (chuẩn, gamma = 0) -> điểm khởi tạo của cấu hình đầy đủ."""
    p = list(psi0[:5]) + [0.0] * spec.nx + list(psi0[5:8])
    p += {"chuan": [], "student": [8.0], "skewt": [8.0, 0.0], "ged": [1.5]}[spec.pp]
    return np.array(p)


def mle(spec: DacTa, dl, so_diem: int = 20, do_tan: float = 0.3, rng=None, xi_am=None) -> dict:
    """MLE nhiều điểm khởi tạo (mục 4.2). xi_am: nghiệm lần ước lượng trước (khởi tạo ấm, thêm vào danh sách)."""
    rng = rng or np.random.default_rng(0)
    long = DacTa("chuan", ())
    k_long, _ = _toi_uu(long, bo_bien(dl, spec.nx), [g_nghich(p, "chuan", 0) for p in _psi_long(dl, spec.nx)])
    psi_long = np.asarray(ham("chuan", 0).g(jnp.asarray(k_long.x)))
    goc = g_nghich(_mo_rong(psi_long, spec), spec.pp, spec.nx)
    diem = [goc] + [goc + rng.normal(0, do_tan, spec.d) for _ in range(max(so_diem - 1, 0))]
    if xi_am is not None:
        diem = [np.asarray(xi_am, float)] + diem
    t = time.time()
    tot, lls = _toi_uu(spec, dl, diem)
    psi = np.asarray(ham(spec.pp, spec.nx).g(jnp.asarray(tot.x)))
    return {"psi": psi, "xi": np.asarray(tot.x), "ll": float(-tot.fun), "ll_cac_diem": lls,
            "so_gan_tot": int((lls >= -tot.fun - 1e-3).sum()), "so_diem": len(diem),
            "do_tan_ll": float(lls.max() - np.median(lls)), "hoi_tu": bool(tot.success), "ten": spec.ten(),
            "ll_long": float(-k_long.fun), "giay": time.time() - t}


def sai_so_chuan(spec: DacTa, psi, dl) -> dict:
    h = ham(spec.pp, spec.nx)
    p = jnp.asarray(psi)
    S = np.asarray(h.diem(p, dl))
    A = -np.asarray(h.hessian(p, dl))
    B = S.T @ S
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        Ai = np.linalg.pinv(A)
    VS = Ai @ B @ Ai
    # -Hessian không xác định dương (nghiệm ở biên, beta sát 1) thì sai số chuẩn không đáng tin; trị tuyệt đối của đường
    # chéo che mất điều này, nên ghi cờ để báo cáo và để mô phỏng đếm số lần Hessian hợp lệ.
    tri_rieng_nho_nhat = float(np.linalg.eigvalsh((A + A.T) / 2).min())
    return {"se_hessian": np.sqrt(np.abs(np.diag(Ai))), "se_banh_kep": np.sqrt(np.abs(np.diag(VS))),
            "cov_banh_kep": VS, "cov_hessian": Ai, "hessian_xac_dinh_duong": bool(tri_rieng_nho_nhat > 0),
            "tri_rieng_nho_nhat": tri_rieng_nho_nhat}


def khoi_tao_phan_tan(spec: DacTa, m: dict, dl, rng, he_so: float = 2.0) -> np.ndarray:
    """Điểm khởi tạo phân tán cho một chuỗi NUTS: xi_hat + he_so * se_xi * N(0, 1).

    LÝ THUYẾT: R_hat chỉ phát hiện được chuỗi chưa hội tụ khi các chuỗi xuất phát phân tán hơn hậu nghiệm (Vehtari và cộng
    sự 2021). se_xi là sai số chuẩn Hessian đổi sang không gian xi theo đạo hàm của phép đổi biến g (18): se_xi = se_psi /
    |d psi / d xi| với beta: beta(1 - beta); nu: sigmoid(xi); lambda: 1 - lambda^2; các tham số khác: 1. Sai số chuẩn không
    hữu hạn hoặc Hessian không xác định dương thì dùng 0,1; se_xi được chặn trong [0,01; 1] để tránh điểm đầu vô lý."""
    psi, xi = np.asarray(m["psi"], float), np.asarray(m["xi"], float)
    se = sai_so_chuan(spec, psi, dl)
    se_psi = np.asarray(se["se_hessian"], float)
    dao = np.ones(spec.d)
    dao[4] = psi[4] * (1 - psi[4])
    k = 5 + spec.nx + 3
    if spec.pp in ("student", "skewt", "ged"):
        dao[k] = 1 / (1 + math.exp(-xi[k]))
    if spec.pp == "skewt":
        dao[k + 1] = 1 - psi[k + 1] ** 2
    se_xi = se_psi / np.maximum(np.abs(dao), 1e-12)
    if not se["hessian_xac_dinh_duong"]:
        se_xi = np.full(spec.d, 0.1)
    se_xi = np.clip(np.where(np.isfinite(se_xi), se_xi, 0.1), 0.01, 1.0)
    return xi + he_so * se_xi * rng.normal(size=spec.d)


def kha_nghich(spec: DacTa, psi, dl, lech: float = 2.0) -> dict:
    """Kiểm tra thực nghiệm điều kiện khả nghịch (3.3.5): hai đường ln h từ ln h_1 và ln h_1 + lech."""
    h = ham(spec.pp, spec.nx)
    l1, _ = h.loc(jnp.asarray(psi), dl)
    l2, _ = h.loc(jnp.asarray(psi), (dl[0], dl[1], dl[2], dl[3], dl[4] + lech))
    cuoi = int(np.flatnonzero(np.asarray(dl[3]))[-1])
    dd = np.abs(np.asarray(l1) - np.asarray(l2))[:cuoi + 1]
    duoi = np.flatnonzero(dd < 1e-6)
    return {"chenh_dau": float(dd[0]), "chenh_cuoi": float(dd[-1]),
            "phien_den_1e-6": int(duoi[0]) if len(duoi) else None, "dat": bool(dd[-1] < 1e-6)}


def sinh_tu_mo_hinh(spec: DacTa, psi, Z, J, lh0: float, rng, can: float = 20.0):
    """Sinh chuỗi lợi suất từ mô hình đã biết tham số (dùng cho bootstrap và mô phỏng); r = 0 tại phiên J."""
    nx = spec.nx
    mu, om, ka, th, be = (float(x) for x in psi[:5])
    b, sh = np.asarray(psi[5:5 + nx + 3]), np.asarray(psi[5 + nx + 3:])
    mabs = float(e_abs(spec.pp, jnp.asarray(sh)))
    n = len(J)
    z = PP.lay_mau(n, spec.pp, sh, rng)
    lnh = np.empty(n)
    lnh[0] = lh0
    for t in range(1, n):
        s = 0.0 if J[t - 1] else ka * (abs(z[t - 1]) - mabs) + th * z[t - 1]
        lnh[t] = min(can, max(-can, om + s + be * lnh[t - 1] + Z[t] @ b))
    r = mu + np.exp(0.5 * lnh) * z
    return np.where(J, 0.0, r), lnh


def bootstrap_tham_so(spec: DacTa, psi, dl, B: int, rng, so_diem: int = 4) -> dict:
    Z, J, lh0 = np.asarray(dl[2]), np.asarray(dl[1]), float(dl[4])
    mau = []
    for _ in range(B):
        r, _ = sinh_tu_mo_hinh(spec, psi, Z, J, lh0, rng)
        dlb = (jnp.asarray(r), dl[1], dl[2], dl[3], dl[4])
        try:
            mau.append(mle(spec, dlb, so_diem, 0.2, rng, xi_am=g_nghich(psi, spec.pp, spec.nx))["psi"])
        except RuntimeError:
            continue
    M = np.array(mau)
    return {"B": len(M), "q2.5": np.quantile(M, 0.025, axis=0), "q97.5": np.quantile(M, 0.975, axis=0)}


# ----------------------------------------------------------------------------------------
# VB và NUTS
# ----------------------------------------------------------------------------------------
def _luu(p, obj):
    import pickle
    from .khung import ghi_nguyen_tu
    ghi_nguyen_tu(p, pickle.dumps(obj))


def _nap(p):
    import pickle
    return pickle.loads(p.read_bytes())


def vb(spec: DacTa, dl, xi0, kieu: str, so_buoc: int, toc_do: float, so_mau: int, seed: int,
       thu_muc=None, buoc_moi_doan: int = 2000, pa=TIEN_NGHIEM_GOC) -> dict:
    """VB mean-field (kieu='vb_mf') hoặc full-rank ('vb_fr') trên xi; khởi tạo trung bình tại xi0.

    Nếu có `thu_muc`: lưu trạng thái tối ưu sau mỗi `buoc_moi_doan` bước (điểm kiểm tra tầng 3); hết giờ thì ném
    HetGio sau khi lưu, lần sau chạy tiếp từ bước đã lưu."""
    import arviz as az
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
    from numpyro.infer.initialization import init_to_value
    from numpyro.optim import Adam

    from .khung import HetGio, con_thoi_gian
    h = ham(spec.pp, spec.nx, pa)

    def model():
        mo_hinh_numpyro(dl, spec.pp, spec.nx, spec.d, pa)

    init = init_to_value(values={"xi": jnp.asarray(xi0)})
    lop = AutoNormal if kieu == "vb_mf" else AutoMultivariateNormal
    guide = lop(model, init_loc_fn=init, init_scale=0.05)
    svi = SVI(model, guide, Adam(toc_do), Trace_ELBO())
    t = time.time()
    trang_thai = svi.init(jax.random.PRNGKey(seed))           # cũng thiết lập hàm ràng buộc của guide
    da, losses, giay = 0, [], 0.0
    p_tt = None if thu_muc is None else thu_muc / "svi.pkl"
    if p_tt is not None and p_tt.exists():
        x = _nap(p_tt)
        trang_thai = jax.tree.map(jnp.asarray, x["trang_thai"])
        da, losses, giay = x["da"], list(x["losses"]), x["giay"]
    buoc = so_buoc if thu_muc is None else buoc_moi_doan
    while da < so_buoc:
        m = min(buoc, so_buoc - da)
        res = svi.run(jax.random.PRNGKey(seed + 1 + da), m, init_state=trang_thai, progress_bar=False)
        trang_thai, da = res.state, da + m
        losses += list(np.asarray(res.losses))
        if p_tt is not None:
            _luu(p_tt, {"trang_thai": jax.device_get(trang_thai), "da": da, "losses": losses,
                        "giay": giay + time.time() - t})
            if da < so_buoc and not con_thoi_gian():
                raise HetGio()
    p = svi.get_params(trang_thai)
    res_losses = np.asarray(losses)
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=(so_mau, spec.d))
    tham = [np.asarray(v) for v in p.values()]
    if not all(np.isfinite(v).all() for v in tham) or not np.isfinite(res_losses).any():
        # tối ưu phân kỳ (tham số biến phân thành NaN): ghi nhận thất bại thay vì làm sập cả bước; b08 chọn lần khởi tạo khác
        return {"psi_mau": np.full((so_mau, spec.d), np.nan), "xi_mau": np.full((so_mau, spec.d), np.nan),
                "elbo": float("-inf"), "k_hat": float("inf"), "loss_moc": res_losses[::max(1, len(res_losses) // 200)],
                "giay": giay + time.time() - t, "so_buoc": int(da), "that_bai": True}
    if kieu == "vb_mf":
        loc, sc = np.asarray(p["xi_auto_loc"]), np.asarray(p["xi_auto_scale"])
        xs = loc + sc * eps
        logq = stats.norm.logpdf(xs, loc, sc).sum(1)
    else:
        loc, L = np.asarray(p["auto_loc"]), np.asarray(p["auto_scale_tril"])
        xs = loc + eps @ L.T
        logq = stats.multivariate_normal.logpdf(xs, loc, L @ L.T)
    lp = np.asarray(h.lp_xi_vmap(jnp.asarray(xs), dl))
    _, k_hat = az.psislw(lp - logq)
    losses = res_losses
    return {"psi_mau": np.asarray(h.g_vmap(jnp.asarray(xs))), "xi_mau": xs,
            "elbo": float(-losses[-max(1, len(losses) // 20):].mean()), "k_hat": float(np.asarray(k_hat)),
            "loss_moc": losses[::max(1, len(losses) // 200)], "giay": giay + time.time() - t,
            "so_buoc": int(da)}


def nuts_chuoi(spec: DacTa, dl, xi0, khoi_dong: int, so_mau: int, dich: float, do_sau: int, seed: int,
               thu_muc=None, vong_moi_doan: int = 100, pa=TIEN_NGHIEM_GOC) -> dict:
    """Một chuỗi NUTS, ma trận khối lượng đầy đủ thích nghi.

    Nếu có `thu_muc` (điểm kiểm tra tầng 3): lưu trạng thái sau khởi động (khoi_dong.pkl), rồi lấy mẫu theo đoạn
    `vong_moi_doan` vòng và lưu mẫu + trạng thái sau mỗi đoạn (doan_###.pkl). Trạng thái gồm bước nhảy và ma trận
    khối lượng đã thích nghi, nên chuỗi chạy tiếp ở tiến trình hoặc máy khác vẫn là cùng một chuỗi. Hết giờ thì ném
    HetGio sau khi lưu."""
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    from .khung import HetGio, con_thoi_gian
    h = ham(spec.pp, spec.nx, pa)

    def model():
        mo_hinh_numpyro(dl, spec.pp, spec.nx, spec.d, pa)

    ker = NUTS(model, dense_mass=True, target_accept_prob=dich, max_tree_depth=do_sau,
               init_strategy=init_to_value(values={"xi": jnp.asarray(xi0)}))
    t = time.time()
    if thu_muc is None:
        mc = MCMC(ker, num_warmup=khoi_dong, num_samples=so_mau, num_chains=1, progress_bar=False)
        mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
        xs = np.asarray(mc.get_samples()["xi"])
        ex = mc.get_extra_fields()
        return {"psi_mau": np.asarray(h.g_vmap(jnp.asarray(xs))), "xi_mau": xs,
                "phan_ky": int(np.asarray(ex["diverging"]).sum()),
                "buoc_tb": float(np.asarray(ex["num_steps"]).mean()), "giay": time.time() - t}
    p_kd = thu_muc / "khoi_dong.pkl"
    if p_kd.exists():
        kd = _nap(p_kd)
    else:
        mc = MCMC(ker, num_warmup=khoi_dong, num_samples=1, num_chains=1, progress_bar=False)
        mc.warmup(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"), collect_warmup=False)
        kd = {"trang_thai": jax.device_get(mc.post_warmup_state), "giay": time.time() - t}
        _luu(p_kd, kd)
        if not con_thoi_gian():
            raise HetGio()
    trang_thai = kd["trang_thai"]
    so_doan = int(math.ceil(so_mau / vong_moi_doan))
    mc = None
    xs, pk, buoc, giay = [], 0, [], kd["giay"]
    for j in range(so_doan):
        p = thu_muc / f"doan_{j:03d}.pkl"
        if p.exists():
            x = _nap(p)
        else:
            n = min(vong_moi_doan, so_mau - j * vong_moi_doan)
            if mc is None or mc.num_samples != n:
                mc = MCMC(ker, num_warmup=khoi_dong, num_samples=n, num_chains=1, progress_bar=False)
            t1 = time.time()
            mc.post_warmup_state = jax.tree.map(jnp.asarray, trang_thai)
            mc.run(jax.random.PRNGKey(seed + 1 + j), extra_fields=("diverging", "num_steps"))
            ex = mc.get_extra_fields()
            x = {"xi": np.asarray(mc.get_samples()["xi"]), "trang_thai": jax.device_get(mc.last_state),
                 "phan_ky": int(np.asarray(ex["diverging"]).sum()), "buoc": np.asarray(ex["num_steps"]),
                 "giay": time.time() - t1}
            _luu(p, x)
            if j < so_doan - 1 and not con_thoi_gian():
                raise HetGio()
        trang_thai = x["trang_thai"]
        xs.append(x["xi"]); pk += x["phan_ky"]; buoc.append(x["buoc"]); giay += x["giay"]
    xs = np.concatenate(xs)
    return {"psi_mau": np.asarray(h.g_vmap(jnp.asarray(xs))), "xi_mau": xs, "phan_ky": pk,
            "buoc_tb": float(np.concatenate(buoc).mean()), "giay": giay, "so_doan": so_doan}


def chan_doan_nuts(cac_chuoi: list[dict], rhat_max: float, ess_min: float) -> dict:
    import arviz as az
    P = np.stack([c["psi_mau"] for c in cac_chuoi])           # (chuỗi, mẫu, d)
    d = P.shape[2]
    rh = [float(az.rhat(P[:, :, j])) for j in range(d)]
    eb = [float(az.ess(P[:, :, j], method="bulk")) for j in range(d)]
    et = [float(az.ess(P[:, :, j], method="tail")) for j in range(d)]
    pk = int(sum(c["phan_ky"] for c in cac_chuoi))
    return {"rhat": rh, "ess_bulk": eb, "ess_tail": et, "phan_ky": pk, "rhat_max": max(rh),
            "ess_min": min(min(eb), min(et)),
            "dat": bool(max(rh) < rhat_max and min(eb) >= ess_min and min(et) >= ess_min and pk == 0)}


# ----------------------------------------------------------------------------------------
# Dự báo một phiên (mục 5.2 - 5.4)
# ----------------------------------------------------------------------------------------
def _hinh(spec, psi):
    return np.asarray(psi[5 + spec.nx + 3:])


def du_bao_cam(spec: DacTa, psi, dl, dau: int, cuoi: int, alphas) -> dict:
    """Dự báo cắm tham số MLE cho các phiên [dau, cuoi): ln h, LS, PIT, VaR/ES (Phụ lục A)."""
    h = ham(spec.pp, spec.nx)
    lnh, z = (np.asarray(x) for x in h.loc(jnp.asarray(psi), dl))
    a0, a1 = dau - DL.BO_QUA, cuoi - DL.BO_QUA
    lnh, z, J = lnh[a0:a1], z[a0:a1], np.asarray(dl[1])[a0:a1]
    sh, mu = _hinh(spec, psi), float(psi[0])
    s = np.exp(0.5 * lnh)
    out = {"lnh": lnh, "h": np.exp(lnh), "mu": np.full(len(lnh), mu),
           "ls": np.where(J, np.nan, PP.log_mat_do(z, spec.pp, sh) - 0.5 * lnh),
           "pit": np.where(J, np.nan, PP.cdf(z, spec.pp, sh)), "var": {}, "es": {}}
    for a in alphas:
        q, e = float(PP.phan_vi(a, spec.pp, sh)), PP.ky_vong_duoi_trai(a, spec.pp, sh)
        qp, ep = float(PP.phan_vi(1 - a, spec.pp, sh)), PP.ky_vong_duoi_phai(a, spec.pp, sh)
        out["var"][a] = {"mua": -mu - s * q, "ban": mu + s * qp}
        out["es"][a] = {"mua": -mu - s * e, "ban": mu + s * ep}
    return out


def du_bao_hau_nghiem(spec: DacTa, psi_mau, dl, dau: int, cuoi: int, alphas, M: int, rng) -> dict:
    """Mật độ dự báo hậu nghiệm (hỗn hợp theo tham số) cho các phiên [dau, cuoi)."""
    h = ham(spec.pp, spec.nx)
    P = jnp.asarray(psi_mau)
    lnh_s, z_s = (np.asarray(x) for x in h.loc_vmap(P, dl))
    a0, a1 = dau - DL.BO_QUA, cuoi - DL.BO_QUA
    lnh_s, z_s = lnh_s[:, a0:a1], z_s[:, a0:a1]
    J = np.asarray(dl[1])[a0:a1]
    S = len(psi_mau)
    mu_s = np.asarray(psi_mau)[:, 0]
    ll_s = np.empty_like(lnh_s)
    pit_s = np.empty_like(lnh_s)
    K = max(1, int(math.ceil(M / S)))
    pool = np.empty((S, K))
    for i in range(S):
        sh = _hinh(spec, psi_mau[i])
        ll_s[i] = PP.log_mat_do(z_s[i], spec.pp, sh) - 0.5 * lnh_s[i]
        pit_s[i] = PP.cdf(z_s[i], spec.pp, sh)
        pool[i] = PP.lay_mau(K, spec.pp, sh, rng)
    out = {"lnh": np.log(np.exp(lnh_s).mean(0)), "h": np.exp(lnh_s).mean(0) + mu_s.var(),
           "mu": np.full(a1 - a0, mu_s.mean()),
           "ls": np.where(J, np.nan, logsumexp(ll_s, axis=0) - math.log(S)),
           "pit": np.where(J, np.nan, pit_s.mean(0)), "var": {}, "es": {}}
    for a in alphas:
        out["var"][a] = {"mua": np.empty(a1 - a0), "ban": np.empty(a1 - a0)}
        out["es"][a] = {"mua": np.empty(a1 - a0), "ban": np.empty(a1 - a0)}
    for t in range(a1 - a0):
        r = (mu_s[:, None] + np.exp(0.5 * lnh_s[:, t])[:, None] * pool).ravel()
        for a in alphas:
            for ben, L in (("mua", -r), ("ban", r)):
                k = max(1, int(math.ceil(a * len(L))))
                top = np.partition(L, len(L) - k)[len(L) - k:]
                out["var"][a][ben][t], out["es"][a][ben][t] = top.min(), top.mean()
    return out
