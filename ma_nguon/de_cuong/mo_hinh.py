"""EGARCH-X theo mục 3 và 4.2 của đề cương, ước lượng hợp lý cực đại, dự báo cuốn chiếu.

LÝ THUYẾT: r_t = mu + sqrt(h_t) z_t;  ln h_t = omega + s_{t-1} + beta ln h_{t-1} + gamma'X_{t-1}
+ delta'D_t;  s_{t-1} = (1 - J_{t-1}) [kappa(|z_{t-1}| - E|z|) + theta z_{t-1}] (phương trình 10-12).
Phiên chuyển hợp đồng (J_t = 1) không có lợi suất: không đóng góp vào hàm hợp lý và cú sốc
của nó bằng không (mục 3.4). beta bị chặn trong (0, 1) (mục 3.3). ln h_1 = logarit phương sai
của 20 lợi suất đầu. Ba phân phối của z: chuẩn, Student-t chuẩn hóa, Student-t lệch Hansen.

THƯ VIỆN: numba (đệ quy ln h), scipy.optimize (L-BFGS-B), scipy.special/stats (mật độ, phân vị).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import integrate, optimize, special, stats

from .du_lieu import BO_QUA, dac_trung

PHAN_PHOI = ("chuan", "student", "skewt")


# ----------------------------------------------------------------------------------------
# Phân phối sai số chuẩn hóa (kỳ vọng 0, phương sai 1)
# ----------------------------------------------------------------------------------------
def _skew_hang(nu: float, lam: float):
    """Hằng số Hansen (1994): a, b, log K_nu (phương trình 15)."""
    logk = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    a = 4 * lam * math.exp(logk) * (nu - 2) / (nu - 1)
    b = math.sqrt(1 + 3 * lam * lam - a * a)
    return a, b, logk


def e_abs(dist: str, sp: np.ndarray) -> float:
    """mu_|z| = E|z| theo phân phối (mục 3.2)."""
    if dist == "chuan":
        return math.sqrt(2 / math.pi)
    nu = float(sp[0])
    lam = float(sp[1]) if dist == "skewt" else 0.0
    a, b, logk = _skew_hang(nu, lam)
    q = nu - 2
    m1 = math.exp(logk) * q / (nu - 1)
    if lam >= 0:
        u = a / (1 + lam)
        duoi0 = special.stdtr(nu, -u * math.sqrt(nu / q))
        duoi1 = m1 * (1 + u * u / q) ** (-(nu - 1) / 2)
        duong = (1 + lam) / b * ((1 + lam) * duoi1 - a * duoi0)
    else:
        u = -a / (1 - lam)
        giua0 = special.stdtr(nu, u * math.sqrt(nu / q)) - 0.5
        duoi1 = m1 * (1 + u * u / q) ** (-(nu - 1) / 2)
        thap = (1 - lam) / b * (-(1 - lam) * (m1 - duoi1) - a * giua0)
        cao = (1 + lam) / b * ((1 + lam) * m1 - 0.5 * a)
        duong = thap + cao
    return 2 * duong


def log_mat_do(z: np.ndarray, dist: str, sp: np.ndarray) -> np.ndarray:
    """ln f_G(z)."""
    if dist == "chuan":
        return -0.5 * math.log(2 * math.pi) - 0.5 * z * z
    nu = float(sp[0])
    lam = float(sp[1]) if dist == "skewt" else 0.0
    a, b, logk = _skew_hang(nu, lam)
    ben = np.where(b * z + a < 0, 1 - lam, 1 + lam)
    u = (b * z + a) / ben
    return math.log(b) + logk - 0.5 * (nu + 1) * np.log1p(u * u / (nu - 2))


def phan_vi(p, dist: str, sp: np.ndarray) -> np.ndarray:
    """Q_G(p), vectơ hóa."""
    p = np.asarray(p, float)
    if dist == "chuan":
        return stats.norm.ppf(p)
    nu = float(sp[0])
    he = math.sqrt((nu - 2) / nu)
    if dist == "student":
        return stats.t.ppf(p, nu) * he
    lam = float(sp[1])
    a, b, _ = _skew_hang(nu, lam)
    thap = ((1 - lam) * stats.t.ppf(np.clip(p / (1 - lam), 0, 1), nu) * he - a) / b
    cao = ((1 + lam) * stats.t.ppf(np.clip((p + lam) / (1 + lam), 0, 1), nu) * he - a) / b
    return np.where(p < (1 - lam) / 2, thap, cao)


def ky_vong_duoi_trai(alpha: float, dist: str, sp: np.ndarray) -> float:
    """e_alpha = E[z | z <= Q(alpha)] (Phụ lục A)."""
    if dist == "chuan":
        return -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    if dist == "student":
        nu = float(sp[0])
        x = stats.t.ppf(alpha, nu)
        return -math.sqrt((nu - 2) / nu) * (nu + x * x) / (nu - 1) * stats.t.pdf(x, nu) / alpha
    return integrate.quad(lambda u: float(phan_vi(u, dist, sp)), 0, alpha, limit=200)[0] / alpha


def ky_vong_duoi_phai(alpha: float, dist: str, sp: np.ndarray) -> float:
    """E[z | z >= Q(1-alpha)] cho vị thế bán; đối xứng với đuôi trái trừ khi lệch."""
    if dist != "skewt":
        return -ky_vong_duoi_trai(alpha, dist, sp)
    return integrate.quad(lambda u: float(phan_vi(u, dist, sp)), 1 - alpha, 1, limit=200)[0] / alpha


def lay_mau(n: int, dist: str, sp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mẫu z ~ G bằng biến đổi nghịch đảo."""
    return phan_vi(rng.uniform(1e-12, 1 - 1e-12, n), dist, sp)


# ----------------------------------------------------------------------------------------
# Đệ quy ln h và hàm hợp lý
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def loc_lnh(r, J, Z, b, mu, omega, kappa, theta, beta, mabs, lh0):
    """Đệ quy ln h_t (phương trình 11-12). Z = [X_{t-1}, D_t], b = [gamma, delta]."""
    n = len(r)
    lnh = np.empty(n)
    z = np.zeros(n)
    lnh[0] = lh0
    for t in range(n):
        if t > 0:
            s = 0.0
            if not J[t - 1]:
                s = kappa * (abs(z[t - 1]) - mabs) + theta * z[t - 1]
            v = omega + s + beta * lnh[t - 1]
            for j in range(Z.shape[1]):
                v += Z[t, j] * b[j]
            lnh[t] = min(20.0, max(-20.0, v))
        if not J[t]:
            z[t] = (r[t] - mu) * math.exp(-0.5 * lnh[t])
    return lnh, z


class Cau_hinh:
    """Mô tả cấu hình: phân phối và tập cột biến ngoại sinh (chỉ số trong TEN_BIEN)."""

    def __init__(self, dist: str, cot: tuple[int, ...]):
        self.dist, self.cot = dist, tuple(cot)
        self.nx = len(cot)
        self.n_shape = {"chuan": 0, "student": 1, "skewt": 2}[dist]
        self.k = 5 + self.nx + 3 + self.n_shape

    def tach(self, p):
        nx = self.nx
        return (p[0], p[1], p[2], p[3], p[4], p[5:5 + nx + 3], p[5 + nx + 3:])

    def can(self):
        nx = self.nx
        c = [(-5, 5), (-10, 10), (-3, 3), (-3, 3), (1e-4, 0.9999)] + [(-3, 3)] * (nx + 3)
        if self.dist != "chuan":
            c.append((2.05, 150.0))
        if self.dist == "skewt":
            c.append((-0.95, 0.95))
        return c

    def ten(self):
        t = ["mu", "omega", "kappa", "theta", "beta"]
        t += [f"gamma{j}" for j in self.cot] + ["delta_dh", "delta_sau", "delta_nghi"]
        t += {"chuan": [], "student": ["nu"], "skewt": ["nu", "lambda"]}[self.dist]
        return t


def tao_Z(cfg: Cau_hinh, f: dict):
    """Ma trận Z = [X_{t-1} (cột đã chọn), D_t]."""
    return np.ascontiguousarray(np.column_stack([f["X"][:, list(cfg.cot)], f["D"]]))


def duong_lnh(p, cfg: Cau_hinh, r, J, Z, lh0):
    mu, om, ka, th, be, b, sp = cfg.tach(p)
    mabs = e_abs(cfg.dist, sp)
    return loc_lnh(r, J, Z, np.asarray(b, float), mu, om, ka, th, be, mabs, lh0), sp


def am_log_hop_ly(p, cfg: Cau_hinh, r, J, Z, lh0) -> float:
    """-l(psi); trả 1e20 khi nghiệm không hợp lệ."""
    try:
        (lnh, z), sp = duong_lnh(p, cfg, r, J, Z, lh0)
        m = ~J
        ll = log_mat_do(z[m], cfg.dist, sp) - 0.5 * lnh[m]
        if not np.isfinite(ll).all():
            return 1e20
        return float(-ll.sum())
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        return 1e20


def diem_khoi_tao(cfg: Cau_hinh, r_ul: np.ndarray, n_starts: int, rng) -> list[np.ndarray]:
    """Các điểm khởi tạo đa dạng cho lần ước lượng đầu."""
    var = float(np.nanvar(r_ul, ddof=1))
    out = []
    for i in range(n_starts):
        be = [0.5, 0.8, 0.9, 0.95][i % 4] + (0 if i < 4 else rng.normal(0, 0.03))
        be = float(np.clip(be, 0.05, 0.98))
        p = [float(np.nanmean(r_ul)), math.log(var) * (1 - be), 0.1 + (0.1 * rng.normal() if i >= 4 else 0),
             -0.05 + (0.05 * rng.normal() if i >= 4 else 0), be]
        p += [0.0] * (cfg.nx + 3)
        if cfg.dist != "chuan":
            p.append([8.0, 5.0, 12.0][i % 3])
        if cfg.dist == "skewt":
            p.append([0.0, -0.1, 0.1][i % 3])
        out.append(np.array(p, float))
    return out


def uoc_luong(cfg: Cau_hinh, r, J, Z, lh0, khoi_tao: list[np.ndarray]):
    """MLE đa điểm khởi tạo bằng L-BFGS-B. Trả về (p_tốt_nhất, -log-hợp-lý, số điểm gần tốt nhất, số điểm hữu hạn)."""
    can = cfg.can()
    kq = []
    for p0 in khoi_tao:
        p0 = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(p0, can)])
        f = optimize.minimize(am_log_hop_ly, p0, args=(cfg, r, J, Z, lh0), method="L-BFGS-B", bounds=can,
                              options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-5, "maxls": 40})
        if np.isfinite(f.fun) and f.fun < 1e19:
            kq.append(f)
    if not kq:
        raise RuntimeError("Không có nghiệm hữu hạn")
    tot = min(kq, key=lambda f: f.fun)
    gan = sum(1 for f in kq if f.fun - tot.fun < 1e-3)
    return tot.x, float(tot.fun), gan, len(kq)


# ----------------------------------------------------------------------------------------
# Dự báo cuốn chiếu (mục 5.1): ước lượng lại mỗi `buoc` phiên trên cửa sổ mở rộng
# ----------------------------------------------------------------------------------------
def chay_cuon_chieu(d, cfg: Cau_hinh, dau: int, cuoi: int, buoc: int = 21, n_starts: int = 8,
                    alphas=(0.01, 0.025, 0.05), seed: int = 20261020, in_ra: bool = False) -> dict:
    """Dự báo cho các phiên có chỉ số trong [dau, cuoi) (chỉ số trên bảng ngày).

    Tại mỗi mốc s0 (đầu mỗi đoạn `buoc` phiên) ước lượng lại trên [BO_QUA, s0-1]; thống kê chuẩn
    hóa và hệ số thang đo chỉ dùng dữ liệu đến s0-1; đoạn kế tiếp được dự báo bằng đệ quy cố định tham số.
    """
    rng = np.random.default_rng(seed)
    n_ra = cuoi - dau
    lnh_db = np.full(n_ra, np.nan)
    mu_db = np.full(n_ra, np.nan)
    rho_sao = np.full(n_ra, np.nan)
    ls_db = np.full(n_ra, np.nan)       # ln f_hat_t(r_t) (điểm logarit mật độ)
    z_db = np.full(n_ra, np.nan)        # phần dư chuẩn hóa ngoài mẫu
    mk = lambda: {a: np.full(n_ra, np.nan) for a in alphas}
    q, e, qp, ep = mk(), mk(), mk(), mk()   # Q(a), E[z|duoi trai], Q(1-a), E[z|duoi phai]
    params, mocs, p_truoc = [], [], None
    for s0 in range(dau, cuoi, buoc):
        s1 = min(s0 + buoc, cuoi)
        f = dac_trung(d, s0 - 1)
        r = np.nan_to_num(f["r"], nan=0.0)
        J = np.asarray(f["J"], bool) | np.isnan(f["r"])
        Z = tao_Z(cfg, f)
        lh0 = math.log(float(np.nanvar(f["r"][BO_QUA:BO_QUA + 20], ddof=1)))
        st = diem_khoi_tao(cfg, f["r"][BO_QUA:s0], n_starts, rng)
        if p_truoc is not None:
            st = [p_truoc] + st[:max(2, n_starts // 3)]
        p, nll, gan, tong = uoc_luong(cfg, r[BO_QUA:s0], J[BO_QUA:s0], Z[BO_QUA:s0], lh0, st)
        p_truoc = p
        sl = slice(BO_QUA, s1)
        (lnh, _), sp = duong_lnh(p, cfg, r[sl], J[sl], Z[sl], lh0)
        off = s0 - BO_QUA
        i0, i1 = s0 - dau, s1 - dau
        lnh_db[i0:i1] = lnh[off:off + (s1 - s0)]
        mu_db[i0:i1] = p[0]
        rho_sao[i0:i1] = f["rho_sao"][s0:s1]
        lnh_b = lnh[off:off + (s1 - s0)]
        z_b = (f["r"][s0:s1] - p[0]) * np.exp(-0.5 * lnh_b)           # NaN ở phiên chuyển
        ls_db[i0:i1] = log_mat_do(z_b, cfg.dist, sp) - 0.5 * lnh_b
        z_db[i0:i1] = z_b
        for a in alphas:
            q[a][i0:i1] = float(phan_vi(a, cfg.dist, sp))
            e[a][i0:i1] = ky_vong_duoi_trai(a, cfg.dist, sp)
            qp[a][i0:i1] = float(phan_vi(1 - a, cfg.dist, sp))
            ep[a][i0:i1] = ky_vong_duoi_phai(a, cfg.dist, sp)
        params.append(p)
        mocs.append({"s0": s0, "nll": nll, "gan_tot": gan, "tong": tong, "c_rho": f["c_rho"]})
        if in_ra:
            print(f"  {cfg.dist} cot={cfg.cot} moc {s0}: nll={nll:.3f} gan_tot={gan}/{tong}", flush=True)
    return {"lnh": lnh_db, "mu": mu_db, "rho_sao": rho_sao, "ls": ls_db, "z": z_db, "q": q, "e": e, "qp": qp, "ep": ep,
            "params": np.array(params), "moc": mocs, "dau": dau, "cuoi": cuoi, "cfg": (cfg.dist, cfg.cot)}
