"""Hàm log-hợp lý EGARCH-X bằng JAX (vi phân tự động) và mô hình Bayes NumPyro theo mục 4.1 - 4.3.

LÝ THUYẾT: cùng hàm hợp lý với `mo_hinh.py` (phương trình 10-16), viết bằng jax.lax.scan để dùng vi phân
tự động cho L-BFGS, VB và NUTS. Tiên nghiệm theo Bảng 5; NumPyro tự cộng số hạng Jacobian khi đổi biến
sang không gian không ràng buộc (phương trình 20). Hiện hỗ trợ phân phối Student-t lệch Hansen
(Student-t chuẩn khi lambda = 0).

THƯ VIỆN: jax (scan, jit, grad), numpyro (NUTS, SVI), jax.scipy.special (gammaln, betainc).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax
from jax.scipy.special import betainc, gammaln

jax.config.update("jax_enable_x64", True)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _tcdf(x, nu):
    """Hàm phân phối Student-t (chưa chuẩn hóa) = 1/2 + tích phân mật độ từ 0 đến x (Gauss-Legendre).

    Dùng cầu phương vì jax.scipy.special.betainc không có đạo hàm theo bậc tự do; đối số ở đây nhỏ
    (|x| < 2) nên 64 điểm cho sai số ở mức làm tròn."""
    s = 0.5 * x * (1.0 + _GL_X)
    logc = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi)
    dens = jnp.exp(logc - 0.5 * (nu + 1) * jnp.log1p(s * s / nu))
    return 0.5 + 0.5 * x * jnp.sum(_GL_W * dens)


def hang_skew(nu, lam):
    logk = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(jnp.pi * (nu - 2))
    a = 4 * lam * jnp.exp(logk) * (nu - 2) / (nu - 1)
    b = jnp.sqrt(1 + 3 * lam * lam - a * a)
    return a, b, logk


def e_abs_skew(nu, lam):
    """E|z| của Student-t lệch Hansen (cùng công thức với bản NumPy, viết liền không rẽ nhánh)."""
    a, b, logk = hang_skew(nu, lam)
    q = nu - 2
    m1 = jnp.exp(logk) * q / (nu - 1)
    u1 = a / (1 + lam)                                   # trường hợp lam >= 0
    duoi0 = _tcdf(-u1 * jnp.sqrt(nu / q), nu)
    duoi1 = m1 * (1 + u1 * u1 / q) ** (-(nu - 1) / 2)
    duong_a = (1 + lam) / b * ((1 + lam) * duoi1 - a * duoi0)
    u2 = -a / (1 - lam)                                  # trường hợp lam < 0
    giua0 = _tcdf(u2 * jnp.sqrt(nu / q), nu) - 0.5
    duoi1b = m1 * (1 + u2 * u2 / q) ** (-(nu - 1) / 2)
    thap = (1 - lam) / b * (-(1 - lam) * (m1 - duoi1b) - a * giua0)
    cao = (1 + lam) / b * ((1 + lam) * m1 - 0.5 * a)
    duong_b = thap + cao
    return 2 * jnp.where(lam >= 0, duong_a, duong_b)


def log_mat_do_skew(z, nu, lam):
    a, b, logk = hang_skew(nu, lam)
    ben = jnp.where(b * z + a < 0, 1 - lam, 1 + lam)
    u = (b * z + a) / ben
    return jnp.log(b) + logk - 0.5 * (nu + 1) * jnp.log1p(u * u / (nu - 2))


def log_hop_ly(mu, omega, kappa, theta, beta, bvec, nu, lam, r, J, Z, lh0):
    """l(psi) = tổng trên các phiên có lợi suất của ln f(z_t) - 0.5 ln h_t (phương trình 16)."""
    mabs = e_abs_skew(nu, lam)

    def buoc(carry, x):
        lnh_p, z_p, J_p, dau = carry
        r_t, J_t, Z_t = x
        s = jnp.where(J_p, 0.0, kappa * (jnp.abs(z_p) - mabs) + theta * z_p)
        lnh = jnp.where(dau, lh0, jnp.clip(omega + s + beta * lnh_p + Z_t @ bvec, -20.0, 20.0))
        z = jnp.where(J_t, 0.0, (r_t - mu) * jnp.exp(-0.5 * lnh))
        ll = jnp.where(J_t, 0.0, log_mat_do_skew(z, nu, lam) - 0.5 * lnh)
        return (lnh, z, J_t, False), ll

    init = (lh0, 0.0, True, True)
    _, ll = lax.scan(buoc, init, (r, J, Z))
    return ll.sum()


def mo_hinh_bayes(r, J, Z, lh0, nx):
    """Mô hình NumPyro với tiên nghiệm Bảng 5 (Student-t lệch)."""
    mu = numpyro.sample("mu", dist.Normal(0.0, 0.5))
    omega = numpyro.sample("omega", dist.Normal(0.0, 0.5))
    kappa = numpyro.sample("kappa", dist.Normal(0.0, 0.5))
    theta = numpyro.sample("theta", dist.Normal(0.0, 0.5))
    beta = numpyro.sample("beta", dist.Beta(20.0, 1.5))
    gamma = numpyro.sample("gamma", dist.Normal(0.0, 0.5).expand([nx]).to_event(1)) if nx else jnp.zeros(0)
    delta = numpyro.sample("delta", dist.Normal(0.0, 1.0).expand([3]).to_event(1))
    nu = 2.0 + numpyro.sample("nu_tru_2", dist.Gamma(2.0, 0.1))
    u = numpyro.sample("lam_u", dist.Beta(2.0, 2.0))
    lam = 2.0 * u - 1.0
    bvec = jnp.concatenate([jnp.atleast_1d(gamma), delta])
    numpyro.factor("hop_ly", log_hop_ly(mu, omega, kappa, theta, beta, bvec, nu, lam, r, J, Z, lh0))
