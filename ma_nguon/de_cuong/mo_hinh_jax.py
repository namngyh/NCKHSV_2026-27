"""EGARCH-X bằng JAX: một hàm log-hợp lý dùng chung cho MLE, VB và NUTS (mục 3, 4.1, 4.3).

LÝ THUYẾT:
  r_t = mu + sqrt(h_t) z_t;  ln h_t = omega + s_{t-1} + beta ln h_{t-1} + gamma'X_{t-1} + delta'D_t        (10)-(11)
  s_{t-1} = (1 - J_{t-1}) [kappa (|z_{t-1}| - E|z|) + theta z_{t-1}]                                       (12)
  l_T(psi) = tổng trên phiên có lợi suất và thuộc cửa sổ ước lượng của  ln f(z_t) - 0.5 ln h_t            (16)
  Đổi biến (18): beta = sigmoid(xi), nu = 2 + softplus(xi) (GED: nu = softplus(xi)), lambda = tanh(xi).
  Hậu nghiệm trên xi (20): l_T(g(xi)) + ln pi(g(xi)) + tổng ln|dg/dxi|, tiên nghiệm theo Bảng 5 và Phụ lục B.
  Phân phối z: chuẩn, Student-t chuẩn hóa (13), Student-t lệch Hansen (15), GED (Phụ lục B).
  Mảng luôn có độ dài bằng cả bảng ngày, cửa sổ ước lượng chọn bằng mặt nạ, nên JAX chỉ biên dịch một lần.

THƯ VIỆN: jax (lax.scan, jit, grad, jacfwd, hessian), numpyro (mô hình xác suất cho SVI và NUTS).
"""
from __future__ import annotations

import math
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy import stats as jst
from jax.scipy.special import gammaln

jax.config.update("jax_enable_x64", True)

SO_HINH_DANG = {"chuan": 0, "student": 1, "skewt": 2, "ged": 1}
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


class DacTa:
    """Đặc tả cấu hình: phân phối và danh sách biến ngoại sinh. d = 5 + d_X + 3 + dim(psi_G) (mục 3.5)."""

    def __init__(self, pp: str, bien=()):
        self.pp, self.bien = pp, tuple(bien)
        self.nx = len(self.bien)
        self.ns = SO_HINH_DANG[pp]
        self.d = 5 + self.nx + 3 + self.ns

    def ten(self) -> list[str]:
        t = ["mu", "omega", "kappa", "theta", "beta"] + [f"gamma_{b}" for b in self.bien]
        t += ["delta_dh", "delta_sau", "delta_nghi"]
        t += {"chuan": [], "student": ["nu"], "skewt": ["nu", "lambda"], "ged": ["nu"]}[self.pp]
        return t

    def __repr__(self):
        return f"{self.pp}|{'+'.join(self.bien) or 'EGARCH'}"


# ----------------------------------------------------------------------------------------
# Phân phối chuẩn hóa (kỳ vọng 0, phương sai 1)
# ----------------------------------------------------------------------------------------
def _tcdf(x, nu):
    """Hàm phân phối Student-t chưa chuẩn hóa = 1/2 + tích phân mật độ từ 0 đến x (Gauss-Legendre 64 điểm).
    Dùng cầu phương vì betainc của JAX không có đạo hàm theo nu; ở đây |x| < 2 nên sai số ở mức làm tròn."""
    s = 0.5 * x * (1.0 + _GL_X)
    logc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi)
    return 0.5 + 0.5 * x * jnp.sum(_GL_W * jnp.exp(logc - 0.5 * (nu + 1) * jnp.log1p(s * s / nu)))


def _hang_skew(nu, lam):
    logk = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(jnp.pi * (nu - 2))
    a = 4 * lam * jnp.exp(logk) * (nu - 2) / (nu - 1)
    b = jnp.sqrt(1 + 3 * lam * lam - a * a)
    return a, b, logk


def _ged_lam(nu):
    return jnp.sqrt(2.0 ** (-2.0 / nu) * jnp.exp(gammaln(1.0 / nu) - gammaln(3.0 / nu)))


def log_mat_do(z, pp: str, sh):
    if pp == "chuan":
        return -0.5 * jnp.log(2 * jnp.pi) - 0.5 * z * z
    if pp == "student":
        nu = sh[0]
        logk = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(jnp.pi * (nu - 2))
        return logk - 0.5 * (nu + 1) * jnp.log1p(z * z / (nu - 2))
    if pp == "skewt":
        nu, lam = sh[0], sh[1]
        a, b, logk = _hang_skew(nu, lam)
        u = (b * z + a) / jnp.where(b * z + a < 0, 1 - lam, 1 + lam)
        return jnp.log(b) + logk - 0.5 * (nu + 1) * jnp.log1p(u * u / (nu - 2))
    if pp == "ged":
        nu = sh[0]
        lv = _ged_lam(nu)
        return (jnp.log(nu) - 0.5 * jnp.abs(z / lv) ** nu - jnp.log(lv) - (1 + 1 / nu) * jnp.log(2.0)
                - gammaln(1 / nu))
    raise ValueError(pp)


def e_abs(pp: str, sh):
    """mu_|z| = E|z| (mục 3.2, Phụ lục B)."""
    if pp == "chuan":
        return jnp.sqrt(2 / jnp.pi)
    if pp == "student":
        nu = sh[0]
        return 2 * jnp.sqrt(nu - 2) * jnp.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / ((nu - 1) * jnp.sqrt(jnp.pi))
    if pp == "ged":
        nu = sh[0]
        return _ged_lam(nu) * 2.0 ** (1 / nu) * jnp.exp(gammaln(2 / nu) - gammaln(1 / nu))
    nu, lam = sh[0], sh[1]
    a, b, logk = _hang_skew(nu, lam)
    q = nu - 2
    m1 = jnp.exp(logk) * q / (nu - 1)
    u1 = a / (1 + lam)
    duong_a = (1 + lam) / b * ((1 + lam) * m1 * (1 + u1 * u1 / q) ** (-(nu - 1) / 2)
                               - a * _tcdf(-u1 * jnp.sqrt(nu / q), nu))
    u2 = -a / (1 - lam)
    thap = (1 - lam) / b * (-(1 - lam) * (m1 - m1 * (1 + u2 * u2 / q) ** (-(nu - 1) / 2))
                            - a * (_tcdf(u2 * jnp.sqrt(nu / q), nu) - 0.5))
    cao = (1 + lam) / b * ((1 + lam) * m1 - 0.5 * a)
    return 2 * jnp.where(lam >= 0, duong_a, thap + cao)


# ----------------------------------------------------------------------------------------
# Đổi biến (18), tiên nghiệm (Bảng 5, Phụ lục B), Jacobian (20)
# ----------------------------------------------------------------------------------------
def g(xi, pp: str, nx: int):
    psi = xi.at[4].set(jax.nn.sigmoid(xi[4]))
    k = 5 + nx + 3
    if pp in ("student", "skewt"):
        psi = psi.at[k].set(2.0 + jax.nn.softplus(xi[k]))
    if pp == "skewt":
        psi = psi.at[k + 1].set(jnp.tanh(xi[k + 1]))
    if pp == "ged":
        psi = psi.at[k].set(jax.nn.softplus(xi[k]))
    return psi


def g_nghich(psi, pp: str, nx: int) -> np.ndarray:
    xi = np.array(psi, float).copy()
    xi[4] = math.log(xi[4] / (1 - xi[4]))
    k = 5 + nx + 3
    if pp in ("student", "skewt"):
        xi[k] = math.log(math.expm1(psi[k] - 2.0))
    if pp == "skewt":
        xi[k + 1] = math.atanh(psi[k + 1])
    if pp == "ged":
        xi[k] = math.log(math.expm1(psi[k]))
    return xi


def log_jacobian(xi, pp: str, nx: int):
    k = 5 + nx + 3
    lj = -jax.nn.softplus(-xi[4]) - jax.nn.softplus(xi[4])           # d sigmoid = s(1-s)
    if pp in ("student", "skewt", "ged"):
        lj = lj - jax.nn.softplus(-xi[k])                              # d softplus = sigmoid
    if pp == "skewt":                                                  # d tanh = 1 - tanh^2 = 4 s(2x)(1 - s(2x))
        lj = lj + jnp.log(4.0) - jax.nn.softplus(-2 * xi[k + 1]) - jax.nn.softplus(2 * xi[k + 1])
    return lj


def _log_beta_xi(x, a, b):
    """ln Beta(sigmoid(x); a, b) tính theo x để ổn định khi sigmoid(x) làm tròn thành 0 hoặc 1."""
    return (-(a - 1) * jax.nn.softplus(-x) - (b - 1) * jax.nn.softplus(x)
            - (gammaln(a) + gammaln(b) - gammaln(a + b)))


TIEN_NGHIEM_GOC = (1.0, 20.0, 1.5, 0.1)   # (hệ số nhân độ lệch chuẩn, Beta a, Beta b của beta, tốc độ Gamma của nu-2)


def log_tien_nghiem(psi, pp: str, nx: int, xi=None, pa=TIEN_NGHIEM_GOC):
    """ln pi(psi) theo Bảng 5 (Phụ lục B cho GED). Nếu có xi thì các số hạng của beta, lambda tính theo xi.
    pa: phương án tiên nghiệm cho phân tích độ nhạy (mục 4.3.1); mặc định đúng Bảng 5."""
    c, ba, bb, nr = pa
    n = jst.norm.logpdf
    lp = n(psi[0], 0, 0.5 * c) + n(psi[1], 0, 0.5 * c) + n(psi[2], 0, 0.5 * c) + n(psi[3], 0, 0.5 * c)
    lp = lp + (_log_beta_xi(xi[4], ba, bb) if xi is not None else jst.beta.logpdf(psi[4], ba, bb))
    lp = lp + jnp.sum(n(psi[5:5 + nx], 0, 0.5 * c)) + jnp.sum(n(psi[5 + nx:5 + nx + 3], 0, 1.0 * c))
    k = 5 + nx + 3
    if pp in ("student", "skewt"):
        lp = lp + jst.gamma.logpdf(psi[k] - 2.0, 2.0, scale=1.0 / nr)  # nu - 2 ~ Gamma(2; 0,1)
    if pp == "skewt":                                                 # (lambda + 1)/2 = sigmoid(2 xi) ~ Beta(2, 2)
        lp = lp + (_log_beta_xi(2 * xi[k + 1], 2.0, 2.0) if xi is not None
                   else jst.beta.logpdf((psi[k + 1] + 1) / 2, 2.0, 2.0)) + jnp.log(0.5)
    if pp == "ged":
        lp = lp + jst.gamma.logpdf(psi[k], 4.0, scale=0.5)            # nu ~ Gamma(4; 2)
    return lp


# ----------------------------------------------------------------------------------------
# Đệ quy, hàm hợp lý, các hàm đã biên dịch
# ----------------------------------------------------------------------------------------
def tach(psi, nx: int):
    return psi[0], psi[1], psi[2], psi[3], psi[4], psi[5:5 + nx + 3], psi[5 + nx + 3:]


def loc(psi, pp: str, nx: int, du_lieu, can: float = 20.0):
    """Trả về (ln h_t, z_t) cho mọi phiên. du_lieu = (r, J_soc, Z, mat_na, lh0); r đã thay NaN bằng 0."""
    r, J, Z, _, lh0 = du_lieu
    mu, om, ka, th, be, b, sh = tach(psi, nx)
    mabs = e_abs(pp, sh)

    def buoc(carry, x):
        lnh_p, z_p, J_p, dau = carry
        r_t, J_t, Z_t = x
        s = jnp.where(J_p, 0.0, ka * (jnp.abs(z_p) - mabs) + th * z_p)
        lnh = jnp.where(dau, lh0, jnp.clip(om + s + be * lnh_p + Z_t @ b, -can, can))
        z = jnp.where(J_t, 0.0, (r_t - mu) * jnp.exp(-0.5 * lnh))
        return (lnh, z, J_t, False), (lnh, z)

    _, (lnh, z) = lax.scan(buoc, (jnp.asarray(lh0, float), 0.0, True, True), (r, J, Z))
    return lnh, z


def ll_tung_phien(psi, pp: str, nx: int, du_lieu):
    lnh, z = loc(psi, pp, nx, du_lieu)
    sh = psi[5 + nx + 3:]
    return jnp.where(du_lieu[3], log_mat_do(z, pp, sh) - 0.5 * lnh, 0.0)


@lru_cache(maxsize=None)
def ham(pp: str, nx: int, pa: tuple = TIEN_NGHIEM_GOC):
    """Các hàm đã biên dịch cho một cấu hình (pp, nx) và phương án tiên nghiệm pa."""
    ll = jax.jit(lambda psi, dl: jnp.sum(ll_tung_phien(psi, pp, nx, dl)))
    llv = jax.jit(lambda psi, dl: ll_tung_phien(psi, pp, nx, dl))

    def am_ll_xi(xi, dl):
        return -jnp.sum(ll_tung_phien(g(xi, pp, nx), pp, nx, dl))

    def lp_xi(xi, dl):
        psi = g(xi, pp, nx)
        return (jnp.sum(ll_tung_phien(psi, pp, nx, dl)) + log_tien_nghiem(psi, pp, nx, xi, pa)
                + log_jacobian(xi, pp, nx))

    class H:
        pass

    h = H()
    h.ll, h.ll_vec = ll, llv
    h.loc = jax.jit(lambda psi, dl: loc(psi, pp, nx, dl))
    h.am_ll_xi_grad = jax.jit(jax.value_and_grad(am_ll_xi))
    h.lp_xi = jax.jit(lp_xi)
    h.lp_xi_vmap = jax.jit(jax.vmap(lp_xi, in_axes=(0, None)))
    h.diem = jax.jit(jax.jacfwd(lambda psi, dl: ll_tung_phien(psi, pp, nx, dl)))
    h.hessian = jax.jit(jax.hessian(lambda psi, dl: jnp.sum(ll_tung_phien(psi, pp, nx, dl))))
    h.loc_vmap = jax.jit(jax.vmap(lambda psi, dl: loc(psi, pp, nx, dl), in_axes=(0, None)))
    h.ll_vec_vmap = jax.jit(jax.vmap(lambda psi, dl: ll_tung_phien(psi, pp, nx, dl), in_axes=(0, None)))
    h.g = jax.jit(lambda xi: g(xi, pp, nx))
    h.g_vmap = jax.jit(jax.vmap(lambda xi: g(xi, pp, nx)))
    return h


def mo_hinh_numpyro(du_lieu, pp: str, nx: int, d: int, pa: tuple = TIEN_NGHIEM_GOC):
    """Mô hình NumPyro trên xi (không gian không ràng buộc); mật độ = (20)."""
    import numpyro
    import numpyro.distributions as dist
    from numpyro.distributions import constraints
    xi = numpyro.sample("xi", dist.ImproperUniform(constraints.real, (), event_shape=(d,)))
    numpyro.factor("log_hau_nghiem", ham(pp, nx, pa).lp_xi(xi, du_lieu))
