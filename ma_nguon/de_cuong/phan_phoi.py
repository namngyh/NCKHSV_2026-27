"""Phân phối sai số chuẩn hóa bằng NumPy/SciPy: mật độ, hàm phân phối, phân vị, ES, lấy mẫu.

LÝ THUYẾT: bốn phân phối có kỳ vọng 0 và phương sai 1 (mục 3.2, Phụ lục B): chuẩn; Student-t chuẩn hóa
t*_nu; Student-t lệch Hansen SkT(nu, lambda); GED(nu) theo tham số hóa của Phụ lục B. Phân vị q_alpha và
kỳ vọng đuôi e_alpha = E[z | z <= q_alpha] dùng cho VaR/ES một phiên theo Phụ lục A:
VaR = -mu - sqrt(h) q_alpha, ES = -mu - sqrt(h) e_alpha (vị thế mua); vị thế bán dùng đuôi phải.
Hàm `sh` là vectơ tham số hình dạng: [] (chuẩn), [nu] (student, ged), [nu, lambda] (skewt).

THƯ VIỆN: scipy.stats (t, norm, gennorm), scipy.integrate.quad (kỳ vọng đuôi bằng cầu phương).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special, stats


def _skew_hang(nu, lam):
    logk = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    a = 4 * lam * math.exp(logk) * (nu - 2) / (nu - 1)
    return a, math.sqrt(1 + 3 * lam * lam - a * a), logk


def _ged_scale(nu):
    """Thang đo của scipy.stats.gennorm tương ứng GED Phụ lục B: s = lambda_nu 2^(1/nu)."""
    lam = math.sqrt(2 ** (-2 / nu) * math.exp(special.gammaln(1 / nu) - special.gammaln(3 / nu)))
    return lam * 2 ** (1 / nu)


def log_mat_do(z, pp, sh):
    z = np.asarray(z, float)
    if pp == "chuan":
        return stats.norm.logpdf(z)
    if pp == "student":
        nu = float(sh[0]); c = math.sqrt((nu - 2) / nu)
        return stats.t.logpdf(z / c, nu) - math.log(c)
    if pp == "ged":
        nu = float(sh[0])
        return stats.gennorm.logpdf(z, nu, scale=_ged_scale(nu))
    nu, lam = float(sh[0]), float(sh[1])
    a, b, logk = _skew_hang(nu, lam)
    u = (b * z + a) / np.where(b * z + a < 0, 1 - lam, 1 + lam)
    return math.log(b) + logk - 0.5 * (nu + 1) * np.log1p(u * u / (nu - 2))


def cdf(z, pp, sh):
    z = np.asarray(z, float)
    if pp == "chuan":
        return stats.norm.cdf(z)
    if pp == "student":
        nu = float(sh[0])
        return stats.t.cdf(z / math.sqrt((nu - 2) / nu), nu)
    if pp == "ged":
        nu = float(sh[0])
        return stats.gennorm.cdf(z, nu, scale=_ged_scale(nu))
    nu, lam = float(sh[0]), float(sh[1])
    a, b, _ = _skew_hang(nu, lam)
    c = math.sqrt((nu - 2) / nu)
    trai = b * z + a < 0
    Ft = stats.t.cdf((b * z + a) / np.where(trai, 1 - lam, 1 + lam) / c, nu)
    return np.where(trai, (1 - lam) * Ft, (1 + lam) * Ft - lam)


def phan_vi(p, pp, sh):
    p = np.asarray(p, float)
    if pp == "chuan":
        return stats.norm.ppf(p)
    if pp == "student":
        nu = float(sh[0])
        return stats.t.ppf(p, nu) * math.sqrt((nu - 2) / nu)
    if pp == "ged":
        nu = float(sh[0])
        return stats.gennorm.ppf(p, nu, scale=_ged_scale(nu))
    nu, lam = float(sh[0]), float(sh[1])
    a, b, _ = _skew_hang(nu, lam)
    c = math.sqrt((nu - 2) / nu)
    with np.errstate(invalid="ignore"):
        thap = ((1 - lam) * c * stats.t.ppf(np.clip(p / (1 - lam), 0, 1), nu) - a) / b
        cao = ((1 + lam) * c * stats.t.ppf(np.clip((p + lam) / (1 + lam), 0, 1), nu) - a) / b
    return np.where(p < (1 - lam) / 2, thap, cao)


def e_abs(pp, sh):
    """E|z| bằng cầu phương (dùng để kiểm tra chéo bản JAX, Phụ lục C.5)."""
    f = lambda x: abs(x) * math.exp(float(log_mat_do(np.array([x]), pp, sh)[0]))
    return integrate.quad(f, -np.inf, 0, limit=200)[0] + integrate.quad(f, 0, np.inf, limit=200)[0]


def ky_vong_duoi_trai(alpha, pp, sh):
    """e_alpha = E[z | z <= q_alpha] (Phụ lục A): công thức đóng cho chuẩn, t*, SkT; cầu phương cho GED."""
    if pp == "chuan":
        return -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    if pp == "student":
        nu = float(sh[0]); x = stats.t.ppf(alpha, nu)
        return -math.sqrt((nu - 2) / nu) * stats.t.pdf(x, nu) / alpha * (nu + x * x) / (nu - 1)
    if pp == "skewt":
        nu, lam = float(sh[0]), float(sh[1])
        a, b, _ = _skew_hang(nu, lam)
        if float(phan_vi(alpha, pp, sh)) < -a / b:
            c = math.sqrt((nu - 2) / nu); x = stats.t.ppf(alpha / (1 - lam), nu)
            return (1 - lam) / (alpha * b) * (-(1 - lam) * c * stats.t.pdf(x, nu) * (nu + x * x) / (nu - 1)
                                             - a * alpha / (1 - lam))
    return integrate.quad(lambda u: float(phan_vi(u, pp, sh)), 0, alpha, limit=200)[0] / alpha


def ky_vong_duoi_phai(alpha, pp, sh):
    """E[z | z >= Q(1 - alpha)] cho vị thế bán."""
    if pp in ("chuan", "student", "ged"):
        return -ky_vong_duoi_trai(alpha, pp, sh)
    return integrate.quad(lambda u: float(phan_vi(u, pp, sh)), 1 - alpha, 1, limit=200)[0] / alpha


def lay_mau(n, pp, sh, rng: np.random.Generator):
    return phan_vi(rng.uniform(1e-12, 1 - 1e-12, n), pp, sh)
