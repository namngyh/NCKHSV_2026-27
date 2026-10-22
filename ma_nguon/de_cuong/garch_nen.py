"""GARCH(1,1) làm mô hình nền (đối chứng) theo đề cương đã đăng ký (BM02, mục 3 và 5): GARCH, EGARCH, EGARCH-X.

LÝ THUYẾT:
  r_t = mu + sqrt(h_t) z_t,  h_t = omega + alpha (r_{t-1} - mu)^2 + beta h_{t-1}  (Bollerslev 1986), z_t ~ G chuẩn hóa
  (chuẩn, Student-t, Student-t lệch Hansen, GED; cùng mật độ với EGARCH-X nên so sánh được).
  Phiên chuyển hợp đồng J_t: lợi suất khuyết, nên số hạng sốc của phiên kế tiếp được thay bằng kỳ vọng của nó,
  (r_{t-1} - mu)^2 -> h_{t-1}; log-hợp lý bỏ phiên J_t (cùng cách xử lý với EGARCH-X, mục 2.3).
  Đổi biến để ước lượng không ràng buộc: omega = exp(x1), alpha + beta = sigmoid(x2) < 1 (dừng), alpha = (alpha + beta)
  * sigmoid(x3), nu = 2 + softplus, lambda = tanh, GED: nu = softplus. h_1 = phương sai mẫu của 20 lợi suất đầu.
  Mô hình nền không có biến ngoại sinh và không có biến giả lịch (GARCH chuẩn).
  MLE: L-BFGS với gradient vi phân tự động, nhiều điểm khởi tạo, giữ nghiệm có log-hợp lý lớn nhất (mục 4.2).
  Dự báo một phiên cắm tham số: h_{t+1} cho cả phiên dự báo bằng đệ quy với tham số cố định, VaR/ES từ phân vị của G.

THƯ VIỆN: jax (lax.scan, jit, value_and_grad), scipy.optimize, numpy. Mô-đun này độc lập với mo_hinh_jax để không làm
đổi mã băm của các bước đã chạy (mo_hinh_jax chỉ được đọc: log_mat_do).
"""
from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy import optimize

from . import du_lieu as DL
from . import phan_phoi as PP
from .mo_hinh_jax import log_mat_do

CAN = 20.0          # chặn |ln h|, cùng giá trị với EGARCH-X ([mo_hinh] can_lnh)


def ten_tham_so(pp: str) -> list[str]:
    return ["mu", "omega", "alpha", "beta"] + {"chuan": [], "student": ["nu"], "skewt": ["nu", "lambda"], "ged": ["nu"]}[pp]


def _g(xi, pp: str):
    """Tham số không ràng buộc xi -> psi = (mu, omega, alpha, beta, hình dạng)."""
    omega = jnp.exp(xi[1])
    tong = jax.nn.sigmoid(xi[2])
    alpha = tong * jax.nn.sigmoid(xi[3])
    psi = [xi[0], omega, alpha, tong - alpha]
    if pp in ("student", "skewt"):
        psi.append(2.0 + jax.nn.softplus(xi[4]))
    if pp == "skewt":
        psi.append(jnp.tanh(xi[5]))
    if pp == "ged":
        psi.append(jax.nn.softplus(xi[4]))
    return jnp.stack(psi)


def _g_nghich(psi, pp: str) -> np.ndarray:
    mu, om, al, be = (float(x) for x in psi[:4])
    tong = al + be
    xi = [mu, math.log(om), math.log(tong / (1 - tong)), math.log((al / tong) / (1 - al / tong))]
    if pp in ("student", "skewt"):
        xi.append(math.log(math.expm1(float(psi[4]) - 2.0)))
    if pp == "skewt":
        xi.append(math.atanh(float(psi[5])))
    if pp == "ged":
        xi.append(math.log(math.expm1(float(psi[4]))))
    return np.array(xi)


def loc(psi, du_lieu):
    """Trả về (h_t, z_t) cho mọi phiên của bảng. du_lieu = (r, J, Z, mat_na, lh0) như EGARCH-X (Z không dùng)."""
    r, J, _, _, lh0 = du_lieu
    mu, om, al, be = psi[0], psi[1], psi[2], psi[3]
    hmin, hmax = math.exp(-CAN), math.exp(CAN)

    def buoc(carry, x):
        h_p, e2_p, dau = carry
        r_t, J_t = x
        h = jnp.where(dau, jnp.exp(lh0), jnp.clip(om + al * e2_p + be * h_p, hmin, hmax))
        e = r_t - mu
        z = jnp.where(J_t, 0.0, e * lax.rsqrt(h))
        return (h, jnp.where(J_t, h, e * e), False), (h, z)

    h0 = jnp.exp(jnp.asarray(lh0, float))
    _, (h, z) = lax.scan(buoc, (h0, h0, True), (r, J))
    return h, z


def ll_tung_phien(psi, pp: str, du_lieu):
    h, z = loc(psi, du_lieu)
    sh = psi[4:]
    z_an = jnp.where(du_lieu[3], z, 1.0)       # tránh gradient NaN tại z = 0 ở phiên bị che (xem mo_hinh_jax.ll_tung_phien)
    return jnp.where(du_lieu[3], log_mat_do(z_an, pp, sh) - 0.5 * jnp.log(h), 0.0)


@lru_cache(maxsize=None)
def _ham(pp: str):
    def am_ll(xi, dl):
        return -jnp.sum(ll_tung_phien(_g(xi, pp), pp, dl))

    class H:
        pass

    h = H()
    h.am_ll_grad = jax.jit(jax.value_and_grad(am_ll))
    h.g = jax.jit(lambda xi: _g(xi, pp))
    h.loc = jax.jit(lambda psi, dl: loc(psi, dl))
    return h


def mle(pp: str, dl, so_diem: int = 20, do_tan: float = 0.3, rng=None) -> dict:
    """MLE nhiều điểm khởi tạo: lưới (alpha, beta) quanh các giá trị thường gặp, cộng nhiễu ngẫu nhiên."""
    rng = rng or np.random.default_rng(0)
    h = _ham(pp)
    r = np.asarray(dl[0])[np.asarray(dl[3])]
    var, mu = float(r.var()), float(r.mean())
    hinh = {"chuan": [], "student": [8.0], "skewt": [8.0, 0.0], "ged": [1.5]}[pp]
    luoi = [(0.05, 0.90), (0.10, 0.80), (0.03, 0.95), (0.15, 0.70), (0.08, 0.88)]
    goc = [_g_nghich([mu, var * (1 - a - b), a, b] + hinh, pp) for a, b in luoi]
    diem = list(goc) + [goc[0] + rng.normal(0, do_tan, len(goc[0])) for _ in range(max(so_diem - len(goc), 0))]

    def f(x):
        v, gr = h.am_ll_grad(jnp.asarray(x), dl)
        v, gr = float(v), np.asarray(gr, float)
        if not (math.isfinite(v) and np.isfinite(gr).all()):
            return 1e20, np.zeros_like(gr)
        return v, gr

    kq = []
    for x0 in diem:
        try:
            kq.append(optimize.minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                                        options={"maxiter": 3000, "gtol": 1e-6, "ftol": 1e-13, "maxls": 50}))
        except (ValueError, FloatingPointError, OverflowError):
            continue
    kq = [k for k in kq if np.isfinite(k.fun) and k.fun < 1e19]
    if not kq:
        raise RuntimeError(f"MLE GARCH {pp}: khong co nghiem huu han")
    tot = min(kq, key=lambda k: k.fun)
    lls = np.array([-k.fun for k in kq])
    return {"psi": np.asarray(h.g(jnp.asarray(tot.x))), "xi": np.asarray(tot.x), "ll": float(-tot.fun),
            "so_gan_tot": int((lls >= -tot.fun - 1e-3).sum()), "so_diem": len(diem),
            "do_tan_ll": float(lls.max() - np.median(lls)), "hoi_tu": bool(tot.success), "ten": ten_tham_so(pp)}


def du_bao_cam(pp: str, psi, dl, dau: int, cuoi: int, alphas) -> dict:
    """Dự báo cắm tham số cho các phiên [dau, cuoi): ln h, LS, PIT, VaR/ES. Cùng khuôn với uoc_luong.du_bao_cam."""
    h, z = (np.asarray(x) for x in _ham(pp).loc(jnp.asarray(psi), dl))
    a0, a1 = dau - DL.BO_QUA, cuoi - DL.BO_QUA
    h, z, J = h[a0:a1], z[a0:a1], np.asarray(dl[1])[a0:a1]
    sh, mu = np.asarray(psi[4:]), float(psi[0])
    s = np.sqrt(h)
    out = {"lnh": np.log(h), "h": h, "mu": np.full(len(h), mu),
           "ls": np.where(J, np.nan, PP.log_mat_do(z, pp, sh) - 0.5 * np.log(h)),
           "pit": np.where(J, np.nan, PP.cdf(z, pp, sh)), "var": {}, "es": {}}
    for a in alphas:
        q, e = float(PP.phan_vi(a, pp, sh)), PP.ky_vong_duoi_trai(a, pp, sh)
        qp, ep = float(PP.phan_vi(1 - a, pp, sh)), PP.ky_vong_duoi_phai(a, pp, sh)
        out["var"][a] = {"mua": -mu - s * q, "ban": mu + s * qp}
        out["es"][a] = {"mua": -mu - s * e, "ban": mu + s * ep}
    return out
