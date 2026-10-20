"""Giai đoạn 2: cơ chế ký quỹ động theo mục 7 của đề cương.

LÝ THUYẾT: Ký quỹ cơ sở là phân vị có điều kiện của tổn thất H phiên (7.1). Với H >= 2 phân phối của
lợi suất tích lũy không có công thức đóng nên được xấp xỉ bằng mô phỏng S đường (7.2): tại mỗi phiên
gốc t lấy ln h_{t+1} đã lọc, sinh z từ phân phối đã ước lượng (mu = 0), sinh biến ngoại sinh tương lai từ
AR(1) với phần dư lấy mẫu lại, cập nhật ln h theo (11) và biến giả lịch. Quy tắc năm bước (7.3):
  m1 = (1-w) VaR + w ES;  m2 = (1+b) m1;  m3 = max(m2, phi VaR_LS);
  m4_t = max(m3_t, eta m4_{t-1} + (1-eta) m3_t);  m_t = max(sàn, g ceil(max(m4_t, m_{t-1} - D)/g)).
Chi phí kỳ vọng C = mean[c_v m + c_th (L - m)+] với mức đuôi tối ưu alpha* = c_v / c_th (7.4).
Đối chứng (7.5): cố định, EWMA 0,94, mô phỏng lịch sử 250 phiên, mô phỏng lịch sử có lọc EGARCH-X, mốc trên
H x biên độ giá. Phiên gốc có cửa sổ (t, t+H] chứa phiên chuyển hợp đồng bị loại khỏi đánh giá (Nhận xét 4).

THƯ VIỆN: numpy (mô phỏng vectơ hóa, phân hoạch để lấy phân vị đuôi), scipy.stats.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from .du_lieu import BO_QUA, dac_trung
from .mo_hinh import Cau_hinh, duong_lnh, e_abs, phan_vi, tao_Z

ALPHAS = (0.05, 0.025, 0.01)
HS = (1, 2, 3)
BUOC_UL = 21


# ----------------------------------------------------------------------------------------
# Chuẩn bị nguồn theo từng mốc ước lượng lại
# ----------------------------------------------------------------------------------------
def nguon_theo_moc(d, cfg: Cau_hinh, res: dict) -> list[dict]:
    """Với mỗi mốc ước lượng lại: tham số, phần dư chuẩn hóa trong mẫu, AR(1) và phần dư của biến ngoại sinh."""
    out = []
    for k, m in enumerate(res["moc"]):
        s0 = m["s0"]
        p = res["params"][k]
        f = dac_trung(d, s0 - 1)
        r = np.nan_to_num(f["r"], nan=0.0)
        J = np.asarray(f["J"], bool) | np.isnan(f["r"])
        Z = tao_Z(cfg, f)
        lh0 = math.log(float(np.nanvar(f["r"][BO_QUA:BO_QUA + 20], ddof=1)))
        (lnh, z), sp = duong_lnh(p, cfg, r[BO_QUA:s0], J[BO_QUA:s0], Z[BO_QUA:s0], lh0)
        pool = z[~J[BO_QUA:s0]]
        nx = cfg.nx
        Xz = f["X"][:, list(cfg.cot)] if nx else np.zeros((len(r), 0))
        c, phi, resid = np.zeros(nx), np.zeros(nx), np.zeros((1, 0))
        if nx:
            x = Xz[BO_QUA:s0]
            y, xl = x[1:], x[:-1]
            resid = np.empty_like(y)
            for i in range(nx):
                A = np.column_stack([np.ones(len(xl)), xl[:, i]])
                b = np.linalg.lstsq(A, y[:, i], rcond=None)[0]
                c[i], phi[i] = b
                resid[:, i] = y[:, i] - A @ b
        out.append({"s0": s0, "p": p, "sp": sp, "pool": pool, "c": c, "phi": phi, "resid": resid, "Xz": Xz})
    return out


# ----------------------------------------------------------------------------------------
# Mô phỏng VaR và ES H phiên (7.2)
# ----------------------------------------------------------------------------------------
def _duoi(L: np.ndarray, a: float):
    S = len(L)
    k = max(int(math.ceil(a * S)), 1)
    top = np.partition(L, S - k)[S - k:]
    return float(top.min()), float(top.mean())


def mo_phong_ky_quy(d, cfg: Cau_hinh, res: dict, S: int = 10000, che_do: str = "tham_so",
                    Hmax: int = 3, seed: int = 20261021) -> dict:
    """VaR và ES H phiên (mua, bán) cho mọi phiên gốc t trong [dau-1, cuoi-2].

    Trả về mảng var, es có hình (số gốc, Hmax, 3 mức alpha, 2 phía [mua, bán]) tính theo tỷ lệ giá trị hợp đồng.
    che_do = 'tham_so' dùng phân phối tham số G; 'loc' lấy mẫu lại phần dư chuẩn hóa (mô phỏng lịch sử có lọc).
    """
    nguon = nguon_theo_moc(d, cfg, res)
    dau, cuoi = res["dau"], res["cuoi"]
    D = d[["D_dh", "D_sau", "D_nghi"]].to_numpy(float)
    J = d["J"].to_numpy()
    n_goc = cuoi - dau
    var = np.full((n_goc, Hmax, len(ALPHAS), 2), np.nan)
    es = np.full_like(var, np.nan)
    rng = np.random.default_rng(seed)
    nx = cfg.nx
    for o in range(n_goc):
        t = dau - 1 + o
        ng = nguon[o // BUOC_UL]
        mu, om, ka, th, be, b, sp = cfg.tach(ng["p"])
        gam, dl = np.asarray(b[:nx]), np.asarray(b[nx:])
        mabs = e_abs(cfg.dist, sp)
        lnh = np.full(S, res["lnh"][o])               # ln h_{t+1}: đã biết tại cuối phiên t
        X = np.tile(ng["Xz"][t + 1], (S, 1)) if nx else np.zeros((S, 0))   # giá trị biến ở phiên t
        cum = np.zeros(S)
        for j in range(1, Hmax + 1):
            if che_do == "loc":
                z = ng["pool"][rng.integers(len(ng["pool"]), size=S)]
            else:
                z = phan_vi(rng.uniform(1e-12, 1 - 1e-12, S), cfg.dist, sp)
            cum = cum + np.exp(0.5 * lnh) * z         # mu = 0 (lựa chọn thận trọng)
            R = np.expm1(cum / 100.0)
            for ia, a in enumerate(ALPHAS):
                var[o, j - 1, ia, 0], es[o, j - 1, ia, 0] = _duoi(-R, a)
                var[o, j - 1, ia, 1], es[o, j - 1, ia, 1] = _duoi(R, a)
            if j < Hmax:
                if nx:
                    X = ng["c"] + ng["phi"] * X + ng["resid"][rng.integers(len(ng["resid"]), size=S)]
                s = 0.0 if (t + j < len(J) and J[t + j]) else ka * (np.abs(z) - mabs) + th * z
                dj = D[min(t + j + 1, len(D) - 1)]
                lnh = np.clip(om + s + be * lnh + (X @ gam if nx else 0.0) + dj @ dl, -20, 20)
    return {"var": var, "es": es}


# ----------------------------------------------------------------------------------------
# Tổn thất thực tế, sàn mô phỏng lịch sử, đối chứng
# ----------------------------------------------------------------------------------------
def tong_that_thuc(d, Hmax: int = 3):
    """R_{t,H} = P_{t+H}/P_t - 1 và mặt nạ hợp lệ (cửa sổ (t, t+H] không chứa phiên chuyển hợp đồng)."""
    P = d["close"].to_numpy(float)
    J = d["J"].to_numpy()
    n = len(P)
    R = np.full((n, Hmax), np.nan)
    ok = np.zeros((n, Hmax), bool)
    for h in range(1, Hmax + 1):
        R[:n - h, h - 1] = P[h:] / P[:n - h] - 1
        for t in range(n - h):
            ok[t, h - 1] = not J[t + 1:t + h + 1].any()
    R = np.where(ok, R, np.nan)
    return R, ok


def san_lich_su(R: np.ndarray, tmoc: np.ndarray, H: int, a: float) -> np.ndarray:
    """VaR H phiên theo mô phỏng lịch sử trên toàn bộ dữ liệu đến t (cửa sổ mở rộng): max hai phía."""
    out = np.full(len(tmoc), np.nan)
    for i, t in enumerate(tmoc):
        r = R[:t - H + 1, H - 1]                     # cửa sổ kết thúc muộn nhất tại t
        r = r[np.isfinite(r)]
        if len(r) > 100:
            out[i] = max(-np.quantile(r, a), np.quantile(r, 1 - a))
    return out


def lich_su_250(R: np.ndarray, tmoc: np.ndarray, H: int, a: float, cua_so: int = 250) -> np.ndarray:
    out = np.full(len(tmoc), np.nan)
    for i, t in enumerate(tmoc):
        r = R[max(0, t - H + 1 - cua_so):t - H + 1, H - 1]
        r = r[np.isfinite(r)]
        if len(r) > 100:
            out[i] = max(-np.quantile(r, a), np.quantile(r, 1 - a))
    return out


def ewma_var(d, tmoc: np.ndarray, H: int, a: float, lam: float = 0.94) -> np.ndarray:
    """VaR chuẩn theo trung bình trượt hàm mũ: sigma_{t+1} sqrt(H) z_{1-a} (đơn vị tỷ lệ)."""
    r = d["r"].to_numpy()
    s2 = np.full(len(r), np.nan)
    cur = float(np.nanvar(r[1:21]))
    for t in range(len(r)):
        if t > 0 and np.isfinite(r[t]):
            cur = lam * cur + (1 - lam) * r[t] ** 2
        s2[t] = cur
    return stats.norm.ppf(1 - a) * np.sqrt(s2[tmoc] * H) / 100.0


# ----------------------------------------------------------------------------------------
# Quy tắc năm bước và chỉ tiêu (7.3, 7.6)
# ----------------------------------------------------------------------------------------
def quy_tac(var, es, san, w, b, phi, eta, giam=0.01, g=0.005):
    """Áp dụng quy tắc năm bước cho các chuỗi var, es, san (cùng độ dài, theo thứ tự thời gian)."""
    n = len(var)
    m = np.full(n, np.nan)
    m1 = (1 - w) * var + w * es
    m3 = np.maximum((1 + b) * m1, phi * san)
    prev4 = prev = None
    for i in range(n):
        if not np.isfinite(m3[i]):
            continue
        if prev4 is None:
            prev4, prev = m3[i], math.ceil(m3[i] / g) * g
        m4 = max(m3[i], eta * prev4 + (1 - eta) * m3[i])
        mi = max(phi * san[i], g * math.ceil(max(m4, prev - giam) / g - 1e-12))
        prev4, prev = m4, mi
        m[i] = mi
    return m


def chi_tieu(m: np.ndarray, L: np.ndarray, hop_le: np.ndarray, H: int, mask: np.ndarray | None = None) -> dict:
    """Chỉ tiêu mục 7.6 trên các phiên gốc hợp lệ."""
    ok = hop_le & np.isfinite(m) & np.isfinite(L)
    if mask is not None:
        ok = ok & mask
    mm, LL = m[ok], L[ok]
    B = LL > mm
    thieu = np.maximum(LL - mm, 0)
    out = {"n": int(ok.sum()), "ty_le_vuot": float(B.mean()) if len(B) else float("nan"),
           "m_tb": float(mm.mean()) if len(mm) else float("nan"),
           "thieu_hut_tb": float(thieu[B].mean()) if B.any() else 0.0,
           "thieu_hut_max": float(thieu.max()) if len(thieu) else 0.0,
           "m_max": float(mm.max()) if len(mm) else float("nan")}
    idx = np.flatnonzero(ok)
    mf = m[idx[0]:idx[-1] + 1] if len(idx) > 1 else mm      # đoạn liên tục của khối đang đánh giá
    mf = mf[np.isfinite(mf)]
    dm = np.diff(mf)
    out["sd_dm"] = float(dm.std()) if len(dm) else float("nan")
    out["so_lan_dieu_chinh"] = int((np.abs(dm) > 1e-12).sum())
    out["thuan_chu_ky"] = float(np.max(mf[20:] / mf[:-20])) if len(mf) > 25 else float("nan")
    return out


def kiem_dinh_vuot(m, L, hop_le, H, a):
    """Kupiec và Christoffersen trên dãy lấy thưa mỗi H phiên (Nhận xét 3)."""
    from .danh_gia import christoffersen, kupiec
    idx = np.arange(0, len(m), H)
    ok = hop_le[idx] & np.isfinite(m[idx]) & np.isfinite(L[idx])
    I = np.where(ok, (L[idx] > m[idx]).astype(float), np.nan)
    n1, n = int(np.nansum(I)), int(ok.sum())
    if n < 20:
        return {}
    lr, p = kupiec(n1, n, a)
    ch = christoffersen(I, a)
    return {"n": n, "vuot": n1, "kupiec_p": p, "ind_p": ch["p_ind"], "cc_p": ch["p_cc"]}


def chi_phi(m, L, hop_le, c_v=1.0, c_th=100.0):
    ok = hop_le & np.isfinite(m) & np.isfinite(L)
    return float(np.mean(c_v * m[ok] + c_th * np.maximum(L[ok] - m[ok], 0)))
