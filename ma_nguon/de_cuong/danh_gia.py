"""Đánh giá theo mục 4.2 (chẩn đoán) và mục 5 (điểm số, kiểm định, so sánh dự báo), mục 6 (VIF, SPA).

LÝ THUYẾT:
  QLIKE trên rho* (5.2); điểm logarit mật độ LS và PIT với kiểm định Berkowitz (5.3);
  VaR/ES: Kupiec, Christoffersen, phân vị động Engle-Manganelli, điểm FZ0 (28), kiểm định ES dạng McNeil-Frey (5.4);
  So sánh: Diebold-Mariano với HAC Newey-West, Giacomini-White có điều kiện, tập mô hình tin cậy (MCS, thống kê
  T_max với bootstrap dừng), kiểm định SPA của Hansen (5.5, 6); VIF (6); đèn giao thông Basel;
  Chẩn đoán trong mẫu (4.2): Ljung-Box trên z và z^2, hồi quy z^2 - 1 lên D và nhóm tau, Berkowitz, điểm gãy
  phương sai Inclan-Tiao.
Định dạng dự báo `db`: {"h", "ls", "pit", "var": {alpha: {"mua", "ban"}}, "es": {...}} theo từng phiên.

THƯ VIỆN: numpy, scipy.stats, statsmodels (Ljung-Box, OLS).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------------------
# Kiểm định độ phủ và điểm số
# ---------------------------------------------------------------------------------------
def kupiec(n1: int, n: int, a: float) -> tuple[float, float]:
    pi = n1 / n
    ll0 = (n - n1) * math.log(1 - a) + n1 * math.log(a)
    ll1 = ((n - n1) * math.log(1 - pi) if pi < 1 else 0.0) + (n1 * math.log(pi) if n1 > 0 else 0.0)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return lr, float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(I: np.ndarray, a: float) -> dict:
    """LR độc lập và LR có điều kiện; chỉ dùng các cặp phiên liền kề cùng có chỉ thị."""
    x, y = I[:-1], I[1:]
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok].astype(int), y[ok].astype(int)
    n00 = int(((x == 0) & (y == 0)).sum()); n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum()); n11 = int(((x == 1) & (y == 1)).sum())
    lg = lambda p, k: k * math.log(p) if k > 0 else 0.0
    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = lg(1 - pi, n00 + n10) + lg(pi, n01 + n11)
    l1 = lg(1 - pi01, n00) + lg(pi01, n01) + lg(1 - pi11, n10) + lg(pi11, n11)
    lr_ind = max(-2 * (l0 - l1), 0.0)
    lr_uc, _ = kupiec(int(np.nansum(I)), int(np.isfinite(I).sum()), a)
    return {"lr_ind": lr_ind, "p_ind": float(1 - stats.chi2.cdf(lr_ind, 1)),
            "lr_cc": lr_uc + lr_ind, "p_cc": float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2))}


def phan_vi_dong(I: np.ndarray, var: np.ndarray, a: float, lags: int = 4) -> tuple[float, float]:
    """Kiểm định phân vị động Engle-Manganelli: hit_t - a hồi quy lên hằng số, hit trễ và VaR."""
    h = I - a
    rows = [t for t in range(lags, len(h)) if np.isfinite(h[t - lags:t + 1]).all() and np.isfinite(var[t])]
    rows = np.array(rows)
    if len(rows) < 30:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(rows))] + [h[rows - k] for k in range(1, lags + 1)] + [var[rows]])
    b = np.linalg.lstsq(X, h[rows], rcond=None)[0]
    stat = float(b @ (X.T @ X) @ b / (a * (1 - a)))
    return stat, float(1 - stats.chi2.cdf(stat, X.shape[1]))


def phan_vi_dong_hac(I: np.ndarray, var: np.ndarray, a: float, H: int, lags: int = 4) -> tuple[float, float]:
    """Kiểm định phân vị động cho chỉ thị vượt của cửa sổ H phiên chồng lấn (Nhận xét 3 của đề cương).

    hit_t - a hồi quy lên hằng số, các hit trễ H, H+1, ..., H+lags-1 (các hit gần hơn tương quan cơ học do chồng lấn)
    và VaR_t. Dưới H0, hit có phương sai đã biết a(1-a) và chỉ tự tương quan tới bậc H-1 (cửa sổ chồng lấn), độc lập
    với thông tin trễ từ bậc H trở đi. Do đó Var(X'y) = a(1-a) X'X + tổng_{j=1..H-1} g_j (C_j + C_j'), với g_j là tự
    hiệp phương sai mẫu của hit ở độ trễ j và C_j là mô-men chéo của X ở độ trễ j. Khi H = 1 thống kê trùng kiểm định
    phân vị động gốc. (Ước lượng vững kiểu White cho điểm số bị sai cỡ nặng khi hồi quy lên chỉ thị hiếm: b02k.)
    Thống kê Wald b' V^-1 b, tiệm cận chi bình phương với số bậc tự do bằng số hồi quy."""
    h = I - a
    L0 = max(H, 1)
    rows = [t for t in range(L0 + lags - 1, len(h))
            if np.isfinite(h[t]) and np.isfinite(var[t]) and np.isfinite(h[t - L0 - lags + 1:t - L0 + 1]).all()]
    rows = np.array(rows)
    if len(rows) < 50:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(len(rows))] + [h[rows - k] for k in range(L0, L0 + lags)] + [var[rows]])
    y = h[rows]
    XtX_i = np.linalg.pinv(X.T @ X)
    b = XtX_i @ X.T @ y
    S = a * (1 - a) * (X.T @ X)
    Xf = np.zeros((len(h), X.shape[1]))
    Xf[rows] = X
    for j in range(1, L0):
        ok = np.isfinite(h[j:]) & np.isfinite(h[:-j])
        gj = float(np.mean(h[j:][ok] * h[:-j][ok])) if ok.any() else 0.0
        Cj = Xf[j:].T @ Xf[:-j]
        S += gj * (Cj + Cj.T)
    V = XtX_i @ S @ XtX_i
    stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, float(1 - stats.chi2.cdf(stat, X.shape[1]))


def fz0(L, v, e, a) -> np.ndarray:
    """Điểm nhất quán VaR-ES (28) theo quy ước tổn thất dương, 0 < v <= e."""
    v = np.maximum(v, 1e-8)
    e = np.maximum(e, v * (1 + 1e-9))
    return (L > v) * (L - v) / (a * e) + v / e + np.log(e) - 1.0


def mcneil_frey(L, es, sigma, vp) -> tuple[float, float, int]:
    m = vp & np.isfinite(L)
    n = int(m.sum())
    if n < 5:
        return float("nan"), float("nan"), n
    u = (L[m] - es[m]) / sigma[m]
    t = float(u.mean() / (u.std(ddof=1) / math.sqrt(n)))
    return t, float(2 * (1 - stats.t.cdf(abs(t), n - 1))), n


def den_giao_thong(I: np.ndarray) -> dict:
    """Đèn giao thông Basel (99%, 250 phiên): xanh <= 4, vàng 5-9, đỏ >= 10 vi phạm."""
    v = np.nan_to_num(I, nan=0.0)
    if len(v) < 250:
        return {"cua_so": 0}
    tong = np.convolve(v, np.ones(250), mode="valid")
    mau = np.where(tong <= 4, 0, np.where(tong <= 9, 1, 2))
    return {"cua_so": int(len(tong)), "xanh": int((mau == 0).sum()), "vang": int((mau == 1).sum()),
            "do": int((mau == 2).sum()), "vi_pham_lon_nhat": int(tong.max())}


def berkowitz(u: np.ndarray) -> tuple[float, float]:
    """LR của Berkowitz (2001): Phi^-1(u) ~ AR(1), H0: trung bình 0, phương sai 1, hệ số 0 (3 bậc tự do)."""
    u = u[np.isfinite(u)]
    x = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
    y, xl = x[1:], x[:-1]
    X = np.column_stack([np.ones(len(y)), xl])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = np.mean((y - X @ b) ** 2)
    ll1 = -0.5 * len(y) * (math.log(2 * math.pi * s2) + 1)
    ll0 = -0.5 * len(y) * math.log(2 * math.pi) - 0.5 * np.sum(y ** 2)
    lr = max(2 * (ll1 - ll0), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 3))


def fz0_chuoi(db: dict, r: np.ndarray, a: float, ben: str = "mua") -> np.ndarray:
    L = -r if ben == "mua" else r
    ok = np.isfinite(L)
    return np.where(ok, fz0(np.where(ok, L, 0.0), db["var"][a][ben], db["es"][a][ben], a), np.nan)


def tom_tat(db: dict, r: np.ndarray, rho_sao: np.ndarray, khoa: np.ndarray, alphas, mask=None, lags: int = 4) -> dict:
    """Mọi chỉ tiêu 5.2 - 5.4 trên một khối; với nhóm phiên (mask, không liên tục) chỉ tính chỉ tiêu
    không phụ thuộc thứ tự thời gian."""
    toan = mask is None
    m = np.ones(len(r), bool) if toan else mask
    h = db["h"]
    q_ok = m & np.isfinite(rho_sao) & ~khoa & np.isfinite(h)
    out = {"n": int((m & np.isfinite(r)).sum()),
           "QLIKE": float(np.mean(np.log(h[q_ok]) + rho_sao[q_ok] / h[q_ok])) if q_ok.any() else float("nan"),
           "LS": float(np.nanmean(db["ls"][m]))}
    if toan:
        out["berkowitz_lr"], out["berkowitz_p"] = berkowitz(db["pit"])
    s = np.sqrt(h)
    L = -r
    ok = np.isfinite(L)
    for a in alphas:
        v, e = db["var"][a]["mua"], db["es"][a]["mua"]
        I = np.where(ok, (L > v).astype(float), np.nan)
        I_s = I[m]
        n1, n = int(np.nansum(I_s)), int(np.isfinite(I_s).sum())
        if n == 0:
            continue
        _, p = kupiec(n1, n, a)
        t_es, p_es, _ = mcneil_frey(L[m], e[m], s[m], I_s == 1)
        row = {"vi_pham": n1, "ty_le": n1 / n, "kupiec_p": p, "FZ0": float(np.nanmean(fz0_chuoi(db, r, a)[m])),
               "es_t": t_es, "es_p": p_es, "tb_VaR": float(np.nanmean(v[m]))}
        if toan:
            ch = christoffersen(I_s, a)
            row.update({"ind_p": ch["p_ind"], "cc_p": ch["p_cc"], "dqr_p": phan_vi_dong(I_s, v, a, lags)[1]})
        out[f"a{a}"] = row
    if toan and 0.01 in alphas:
        out["den_giao_thong_1pct"] = den_giao_thong(np.where(ok, (L > db["var"][0.01]["mua"]).astype(float), np.nan))
    return out


# ---------------------------------------------------------------------------------------
# So sánh dự báo
# ---------------------------------------------------------------------------------------
def dm(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano với HAC Newey-West; trả về (thống kê, p một phía cho H1: E[la - lb] < 0)."""
    dl = la - lb
    dl = dl[np.isfinite(dl)]
    n = len(dl)
    J = int(n ** (1 / 3))
    c = dl - dl.mean()
    v = c @ c / n
    for j in range(1, J + 1):
        v += 2 * (1 - j / (J + 1)) * (c[j:] @ c[:-j]) / n
    t = dl.mean() / math.sqrt(max(v, 1e-18) / n)
    return float(t), float(stats.norm.cdf(t))


def giacomini_white(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    """Kiểm định khả năng dự báo có điều kiện Giacomini-White (2006), hàm thử h_{t-1} = (1, dl_{t-1})."""
    dl = la - lb
    ok = np.isfinite(dl[1:]) & np.isfinite(dl[:-1])
    y, x = dl[1:][ok], dl[:-1][ok]
    Zt = np.column_stack([y, x * y])
    n = len(Zt)
    zb = Zt.mean(0)
    Om = (Zt - zb).T @ (Zt - zb) / n
    stat = float(n * zb @ np.linalg.pinv(Om) @ zb)
    return stat, float(1 - stats.chi2.cdf(stat, 2))


def _chi_so_bootstrap(n: int, B: int, khoi: float, rng) -> np.ndarray:
    idx = np.empty((B, n), int)
    for b in range(B):
        t = rng.integers(n)
        for i in range(n):
            t = rng.integers(n) if rng.random() < 1 / khoi else (t + 1) % n
            idx[b, i] = t
    return idx


def spa(Dm: np.ndarray, rng, B: int = 2000, khoi: float = 5) -> dict:
    """SPA của Hansen (2005): Dm[t, k] = L_chuẩn - L_ứng_viên_k; H0: không ứng viên nào tốt hơn chuẩn."""
    n, k = Dm.shape
    dbar = Dm.mean(0)
    bm = Dm[_chi_so_bootstrap(n, B, khoi, rng)].mean(1)
    om = np.maximum(np.sqrt(n) * bm.std(0, ddof=1), 1e-12)
    tk = np.sqrt(n) * dbar / om
    T = max(tk.max(), 0.0)
    mu_c = dbar * (tk <= -math.sqrt(2 * math.log(math.log(n))))
    Tb = np.maximum((np.sqrt(n) * (bm - dbar + mu_c) / om).max(1), 0.0)
    return {"T": float(T), "p": float((Tb >= T).mean()), "tk": tk}


def mcs(L: np.ndarray, ten: list[str], rng, muc: float = 0.10, B: int = 2000, khoi: float = 5) -> dict:
    """Tập mô hình tin cậy Hansen-Lunde-Nason (2011), thống kê T_max; L[t, i] là mất mát của mô hình i."""
    L = L[np.isfinite(L).all(1)]
    n = len(L)
    idx = _chi_so_bootstrap(n, B, khoi, rng)
    con = list(range(L.shape[1]))
    p_mcs, p_chay = {}, 0.0
    while len(con) > 1:
        Lc = L[:, con]
        di = Lc - Lc.mean(1, keepdims=True)
        dbar = di.mean(0)
        bm = di[idx].mean(1)
        se = np.maximum(bm.std(0, ddof=1), 1e-12)
        t = dbar / se
        p = float((((bm - dbar) / se).max(1) >= t.max()).mean())
        p_chay = max(p_chay, p)
        if p >= muc:
            break
        loai = con[int(np.argmax(t))]
        p_mcs[ten[loai]] = p_chay
        con.remove(loai)
    for i in con:
        p_mcs[ten[i]] = 1.0 if len(con) == 1 else max(p_chay, muc)
    return {"tap_tin_cay": [ten[i] for i in con], "p_mcs": p_mcs, "muc": muc}


def vif(X: np.ndarray) -> list[float]:
    """Hệ số phóng đại phương sai của từng cột (mục 6, đa cộng tuyến)."""
    X = X[np.isfinite(X).all(1)]
    out = []
    for j in range(X.shape[1]):
        if X.shape[1] == 1:
            out.append(1.0)
            continue
        A = np.column_stack([np.ones(len(X)), np.delete(X, j, 1)])
        b = np.linalg.lstsq(A, X[:, j], rcond=None)[0]
        r2 = 1 - np.var(X[:, j] - A @ b) / np.var(X[:, j])
        out.append(float(1 / max(1 - r2, 1e-12)))
    return out


# ---------------------------------------------------------------------------------------
# Chẩn đoán trong mẫu (mục 4.2)
# ---------------------------------------------------------------------------------------
def diem_gay_phuong_sai(z: np.ndarray) -> tuple[float, int]:
    """Thống kê Inclan-Tiao sqrt(n/2) max_k |D_k| cho thay đổi phương sai; ngưỡng 5% xấp xỉ 1,358."""
    c = np.cumsum(z ** 2)
    k = np.arange(1, len(z) + 1)
    d = c / c[-1] - k / len(z)
    return float(math.sqrt(len(z) / 2) * np.abs(d).max()), int(np.argmax(np.abs(d)))


def chan_doan(z: np.ndarray, pit: np.ndarray, tau: np.ndarray, D: np.ndarray, lnh: np.ndarray) -> dict:
    """Chẩn đoán (i)-(iv) của mục 4.2 trên các phiên có lợi suất trong cửa sổ ước lượng."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.diagnostic import acorr_ljungbox
    out = {}
    for lag in (10, 20):
        out[f"LB_z_{lag}_p"] = float(acorr_ljungbox(z, lags=[lag])["lb_pvalue"].iloc[0])
        out[f"LB_z2_{lag}_p"] = float(acorr_ljungbox(z ** 2, lags=[lag])["lb_pvalue"].iloc[0])
    y = z ** 2 - 1
    Dm = np.column_stack([np.ones(len(y)), D[:, [0, 2]]])   # D_sau = J luôn bằng 0 trên các phiên có lợi suất
    out["hoi_quy_D_F_p"] = float(OLS(y, Dm).fit().f_test(np.eye(Dm.shape[1])[1:]).pvalue)
    nhom = np.minimum(tau.astype(int), 5)
    G = np.column_stack([np.ones(len(y))] + [(nhom == g).astype(float) for g in range(5)])
    out["hoi_quy_tau_F_p"] = float(OLS(y, G).fit().f_test(np.eye(G.shape[1])[1:]).pvalue)
    out["z2_tb_theo_tau"] = {int(g): float(np.mean(z[nhom == g] ** 2)) for g in range(6) if (nhom == g).any()}
    out["berkowitz_lr"], out["berkowitz_p"] = berkowitz(pit)
    out["gay_IT"], out["gay_vi_tri"] = diem_gay_phuong_sai(z)
    out["so_phien_cham_can_lnh"] = int((np.abs(lnh) >= 19.999).sum())
    return out
