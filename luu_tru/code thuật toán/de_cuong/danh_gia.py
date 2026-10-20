"""Đánh giá theo mục 4.2 (chẩn đoán, sai số chuẩn) và mục 5 (điểm số, kiểm định, so sánh dự báo).

LÝ THUYẾT: QLIKE trên rho* (5.2); điểm logarit mật độ (5.3); VaR/ES một phiên từ phương trình
(25), kiểm định Kupiec, Christoffersen, phân vị động (Engle-Manganelli), điểm FZ0 (28), kiểm
định ES dạng McNeil-Frey (5.4); Diebold-Mariano với phương sai Newey-West (5.5); đèn giao thông
Basel; sai số chuẩn bánh kẹp (4.2); Berkowitz, Ljung-Box, hồi quy z^2-1 lên biến lịch (4.2).

THƯ VIỆN: numpy, scipy.stats (phân phối chi bình phương, chuẩn), statsmodels (Ljung-Box, OLS).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .mo_hinh import Cau_hinh, duong_lnh, log_mat_do


# ---------------------------------------------------------------------------------------
# VaR, ES một phiên và điểm số
# ---------------------------------------------------------------------------------------
def var_es_mot_phien(res: dict, alpha: float, ben: str = "mua"):
    """VaR và ES của tổn thất (dương) một phiên: phương trình (25). ben = 'mua' hoặc 'ban'."""
    s = np.exp(0.5 * res["lnh"])
    mu = res["mu"]
    if ben == "mua":
        return -mu - s * res["q"][alpha], -mu - s * res["e"][alpha]
    return mu + s * res["qp"][alpha], mu + s * res["ep"][alpha]


def qlike(res: dict, khoa=None) -> float:
    h = np.exp(res["lnh"])
    m = np.isfinite(res["rho_sao"])
    if khoa is not None:
        m &= ~khoa
    return float(np.mean(np.log(h[m]) + res["rho_sao"][m] / h[m]))


def kupiec(n1: int, n: int, a: float) -> tuple[float, float]:
    pi = n1 / n
    ll0 = (n - n1) * math.log(1 - a) + n1 * math.log(a)
    ll1 = (n - n1) * math.log(1 - pi) + (n1 * math.log(pi) if n1 > 0 else 0.0)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(I: np.ndarray, a: float) -> dict:
    """LR độc lập và LR có điều kiện; chỉ dùng các cặp phiên liền kề cùng có chỉ thị."""
    x = I[:-1]
    y = I[1:]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok].astype(int), y[ok].astype(int)
    n00 = int(((x == 0) & (y == 0)).sum()); n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum()); n11 = int(((x == 1) & (y == 1)).sum())

    def lg(p, k):
        return k * math.log(p) if k > 0 else 0.0

    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = lg(1 - pi, n00 + n10) + lg(pi, n01 + n11)
    l1 = lg(1 - pi01, n00) + lg(pi01, n01) + lg(1 - pi11, n10) + lg(pi11, n11)
    lr_ind = max(-2 * (l0 - l1), 0.0)
    n1 = int(np.nansum(I)); n = int(np.isfinite(I).sum())
    lr_uc, _ = kupiec(n1, n, a)
    lr_cc = lr_uc + lr_ind
    return {"lr_ind": lr_ind, "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)),
            "lr_cc": lr_cc, "p_cc": float(1 - stats.chi2.cdf(lr_cc, 2))}


def phan_vi_dong(I: np.ndarray, var: np.ndarray, a: float, lags: int = 4) -> tuple[float, float]:
    """Kiểm định phân vị động Engle-Manganelli: hit_t - a hồi quy lên hit trễ và VaR."""
    h = I - a
    ok = np.isfinite(h) & np.isfinite(var)
    idx = np.flatnonzero(ok)
    idx = idx[idx >= lags]
    rows = []
    for t in idx:
        if np.isfinite(h[t - lags:t]).all():
            rows.append(t)
    rows = np.array(rows)
    X = np.column_stack([np.ones(len(rows))] + [h[rows - k] for k in range(1, lags + 1)] + [var[rows]])
    y = h[rows]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    stat = float(b @ (X.T @ X) @ b / (a * (1 - a)))
    return stat, float(1 - stats.chi2.cdf(stat, X.shape[1]))


def fz0(L, v, e, a) -> np.ndarray:
    """Điểm nhất quán VaR-ES (28); L tổn thất dương, 0 < v <= e."""
    v = np.maximum(v, 1e-8)
    e = np.maximum(e, v * (1 + 1e-9))
    return (L > v) * (L - v) / (a * e) + v / e + np.log(e) - 1.0


def mcneil_frey(L, es, sigma, vp) -> tuple[float, float, int]:
    """Kiểm định ES: trung bình (L - ES)/sigma trên các phiên vượt VaR bằng 0 (t một mẫu)."""
    m = vp & np.isfinite(L)
    n = int(m.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    u = (L[m] - es[m]) / sigma[m]
    t = float(u.mean() / (u.std(ddof=1) / math.sqrt(n)))
    return t, float(2 * (1 - stats.t.cdf(abs(t), n - 1))), n


def dm(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano với HAC Newey-West. Trả về (thống kê, p một phía cho H1: E[la-lb] < 0)."""
    dl = la - lb
    dl = dl[np.isfinite(dl)]
    n = len(dl)
    J = int(n ** (1 / 3))
    c = dl - dl.mean()
    v = c @ c / n
    for j in range(1, J + 1):
        v += 2 * (1 - j / (J + 1)) * (c[j:] @ c[:-j]) / n
    se = math.sqrt(max(v, 1e-18) / n)
    t = dl.mean() / se
    return float(t), float(stats.norm.cdf(t))


def den_giao_thong(I: np.ndarray) -> dict:
    """Đèn giao thông Basel (99%, 250 phiên): xanh <= 4, vàng 5-9, đỏ >= 10 vi phạm."""
    v = np.nan_to_num(I, nan=0.0)
    if len(v) < 250:
        return {"cua_so": 0}
    tong = np.convolve(v, np.ones(250), mode="valid")
    mau = np.where(tong <= 4, 0, np.where(tong <= 9, 1, 2))
    return {"cua_so": int(len(tong)), "xanh": int((mau == 0).sum()), "vang": int((mau == 1).sum()),
            "do": int((mau == 2).sum()), "vi_pham_lon_nhat": int(tong.max())}


def tom_tat_khoi(res: dict, r: np.ndarray, khoa: np.ndarray, mask: np.ndarray | None = None,
                 alphas=(0.01, 0.025, 0.05)) -> dict:
    """Mọi chỉ tiêu 5.2-5.4 trên một khối. Với nhóm phiên chọn bằng `mask` (mục 5.6, không liên tục)
    chỉ tính các chỉ tiêu không phụ thuộc thứ tự thời gian (không có Christoffersen, phân vị động)."""
    toan = mask is None
    m = np.ones(len(r), bool) if toan else mask
    h = np.exp(res["lnh"])
    q_ok = m & np.isfinite(res["rho_sao"]) & ~khoa
    out = {"n": int((m & np.isfinite(r)).sum()),
           "QLIKE": float(np.mean(np.log(h[q_ok]) + res["rho_sao"][q_ok] / h[q_ok])) if q_ok.any() else float("nan"),
           "LS": float(np.nanmean(res["ls"][m]))}
    s = np.exp(0.5 * res["lnh"])
    L = -r
    ok = np.isfinite(L)
    for a in alphas:
        v, e = var_es_mot_phien(res, a, "mua")
        I = np.where(ok, (L > v).astype(float), np.nan)
        I_s = I[m]
        n1, n = int(np.nansum(I_s)), int(np.isfinite(I_s).sum())
        if n == 0:
            continue
        lr, p = kupiec(n1, n, a)
        fz = np.where(ok, fz0(np.where(ok, L, 0.0), v, e, a), np.nan)
        t_es, p_es, _ = mcneil_frey(L[m], e[m], s[m], (I_s == 1))
        row = {"vi_pham": n1, "ty_le": n1 / n, "kupiec_p": p, "FZ0": float(np.nanmean(fz[m])),
               "es_t": t_es, "es_p": p_es, "tb_VaR": float(np.nanmean(v[m]))}
        if toan:
            ch = christoffersen(I_s, a)
            row.update({"ind_p": ch["p_ind"], "cc_p": ch["p_cc"], "dqr_p": phan_vi_dong(I_s, v, a)[1]})
        out[f"a{a}"] = row
    if toan:
        out["den_giao_thong_1pct"] = den_giao_thong(np.where(ok, (L > var_es_mot_phien(res, 0.01)[0]).astype(float), np.nan))
    return out


def fz0_chuoi(res: dict, r: np.ndarray, a: float) -> np.ndarray:
    v, e = var_es_mot_phien(res, a, "mua")
    L = -r
    ok = np.isfinite(L)
    return np.where(ok, fz0(np.where(ok, L, 0.0), v, e, a), np.nan)


# ---------------------------------------------------------------------------------------
# Sai số chuẩn bánh kẹp và chẩn đoán trong mẫu
# ---------------------------------------------------------------------------------------
def _ll_t(p, cfg: Cau_hinh, r, J, Z, lh0):
    (lnh, z), sp = duong_lnh(p, cfg, r, J, Z, lh0)
    m = ~J
    return log_mat_do(z[m], cfg.dist, sp) - 0.5 * lnh[m]


def sai_so_chuan(p, cfg: Cau_hinh, r, J, Z, lh0) -> dict:
    """Sai số chuẩn Hessian (A^-1) và bánh kẹp (A^-1 B A^-1) bằng sai phân hữu hạn (mục 4.2)."""
    k = len(p)
    hh = 1e-5 * np.maximum(1.0, np.abs(p))
    can = cfg.can()
    lo = np.array([c[0] for c in can]); hi = np.array([c[1] for c in can])

    def diem(pp):
        return _ll_t(np.clip(pp, lo, hi), cfg, r, J, Z, lh0)

    def diem_luot(pp):
        S = np.empty((len(diem(pp)), k))
        for i in range(k):
            e = np.zeros(k); e[i] = hh[i]
            S[:, i] = (diem(pp + e) - diem(pp - e)) / (2 * hh[i])
        return S

    S = diem_luot(p)
    B = S.T @ S
    A = np.empty((k, k))
    for j in range(k):
        e = np.zeros(k); e[j] = 1e-4 * max(1.0, abs(p[j]))
        gp, gm = diem_luot(p + e).sum(0), diem_luot(p - e).sum(0)
        A[:, j] = -(gp - gm) / (2 * e[j])
    A = 0.5 * (A + A.T)
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return {"se_hessian": np.full(k, np.nan), "se_banh_kep": np.full(k, np.nan)}
    return {"se_hessian": np.sqrt(np.abs(np.diag(Ai))), "se_banh_kep": np.sqrt(np.abs(np.diag(Ai @ B @ Ai)))}


def cdf_chuan_hoa(z, dist: str, sp) -> np.ndarray:
    from .mo_hinh import _skew_hang
    z = np.asarray(z, float)
    if dist == "chuan":
        return stats.norm.cdf(z)
    nu = float(sp[0]); he = math.sqrt((nu - 2) / nu)
    if dist == "student":
        return stats.t.cdf(z / he, nu)
    lam = float(sp[1]); a, b, _ = _skew_hang(nu, lam)
    ben = np.where(b * z + a < 0, 1 - lam, 1 + lam)
    Ft = stats.t.cdf((b * z + a) / ben / he, nu)
    return np.where(b * z + a < 0, (1 - lam) * Ft, (1 + lam) * Ft - lam)


def berkowitz(u: np.ndarray) -> tuple[float, float]:
    """LR của Berkowitz: Phi^-1(u) ~ AR(1) với trung bình 0, phương sai 1, hệ số 0 (3 bậc tự do)."""
    x = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    y, xl = x[1:], x[:-1]
    n = len(y)
    X = np.column_stack([np.ones(n), xl])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.mean((y - X @ b) ** 2)
    ll1 = -0.5 * n * (math.log(2 * math.pi * s2) + 1)
    ll0 = -0.5 * n * math.log(2 * math.pi) - 0.5 * np.sum(y ** 2)
    lr = max(2 * (ll1 - ll0), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 3))


def diem_gay_phuong_sai(z: np.ndarray) -> tuple[float, int]:
    """Thống kê Inclan-Tiao: sqrt(n/2) max_k |D_k|; ngưỡng bác bỏ 5% xấp xỉ 1,358."""
    c = np.cumsum(z ** 2)
    k = np.arange(1, len(z) + 1)
    d = c / c[-1] - k / len(z)
    return float(math.sqrt(len(z) / 2) * np.abs(d).max()), int(np.argmax(np.abs(d)))


def chan_doan_trong_mau(p, cfg: Cau_hinh, r, J, Z, lh0, tau, D) -> dict:
    """Ljung-Box, hồi quy z^2-1 lên D và lên nhóm tau, Berkowitz, điểm gãy (mục 4.2)."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import acorr_ljungbox
    (lnh, z), sp = duong_lnh(p, cfg, r, J, Z, lh0)
    m = ~J
    zz = z[m]
    out = {}
    for lag in (10, 20):
        out[f"LB_z_{lag}_p"] = float(acorr_ljungbox(zz, lags=[lag])["lb_pvalue"].iloc[0])
        out[f"LB_z2_{lag}_p"] = float(acorr_ljungbox(zz ** 2, lags=[lag])["lb_pvalue"].iloc[0])
    y = zz ** 2 - 1
    Dm = np.column_stack([np.ones(len(y)), D[m][:, [0, 2]]])   # D_sau = J luôn bằng 0 trên các phiên có lợi suất
    f = OLS(y, Dm).fit()
    out["hoi_quy_D_F_p"] = float(f.f_test(np.eye(Dm.shape[1])[1:]).pvalue)
    nhom = np.minimum(tau[m].astype(int), 5)       # nhóm tau = 0,1,2,3,4,>=5
    G = np.column_stack([np.ones(len(y))] + [(nhom == g).astype(float) for g in range(5)])
    f2 = OLS(y, G).fit()
    out["hoi_quy_tau_F_p"] = float(f2.f_test(np.eye(G.shape[1])[1:]).pvalue)
    out["z2_tb_theo_tau"] = {int(g): float(np.mean(y[nhom == g]) + 1) for g in range(6) if (nhom == g).any()}
    out["berkowitz_lr"], out["berkowitz_p"] = berkowitz(cdf_chuan_hoa(zz, cfg.dist, sp))
    out["gay_IT"], out["gay_vitri"] = diem_gay_phuong_sai(zz)
    out["so_phien_cham_can_lnh"] = int(((lnh <= -19.99) | (lnh >= 19.99)).sum())
    return out
