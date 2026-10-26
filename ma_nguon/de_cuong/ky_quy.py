"""Giai đoạn 2: cơ chế ký quỹ động theo mục 7 của đề cương.

LÝ THUYẾT: Ký quỹ cơ sở là phân vị có điều kiện của tổn thất H phiên (7.1). Với H >= 2 phân phối của lợi suất
tích lũy không có công thức đóng nên được xấp xỉ bằng mô phỏng S đường (7.2):
  1. mỗi đường lấy một bộ tham số psi^(s): rút từ hậu nghiệm (VB, NUTS) hoặc psi^(s) = psi_hat (MLE);
  2. ln h_{t+1} là giá trị lọc tại cuối phiên t theo chính bộ tham số đó;
  3. với j = 1..H: z ~ G (hoặc lấy mẫu lại phần dư chuẩn hóa: mô phỏng lịch sử có lọc), r = sqrt(h) z (mu = 0),
     X_{t+j} sinh từ AR(1) ước lượng trên cùng cửa sổ với phần dư lấy mẫu lại, cập nhật ln h theo (11) và lịch;
     với x_tuong_lai = "ar1_kem_z" (26/10/2026), X_{t+j+1} = c + phi X_{t+j} + a |z_{t+j}| + b z_{t+j} + e: biến đo đến cuối
     phiên t+j (như RV, G, O) phụ thuộc cú sốc của chính phiên đó, nên cú sốc lớn kéo phương sai các phiên sau lên;
     "ar1" là cách của đề cương (X độc lập với z cùng phiên, có thể làm đuôi lợi suất tích lũy bị đánh giá thấp);
  4. R = exp(tổng r / 100) - 1, tổn thất mua -R, bán R; VaR = L^(j*), ES = trung bình j* tổn thất lớn nhất.
Quy tắc sáu bước (7.3): m1 = (1-w) VaR + w ES; m2 = (1+b) m1; m3 = max(m2, phi VaR_LS);
  m4_t = max(m3_t, eta m4_{t-1} + (1-eta) m3_t); m5_t = max(sàn, g ceil(max(m4_t, m5_{t-1} - D) / g));
  bước 6 (vùng không điều chỉnh): mức áp dụng chỉ đổi khi m5_t lệch khỏi nó tối thiểu ng_tang hoặc ng_giam.
Chi phí ky vọng C = mean[c_v m + c_th (L - m)+], mức đuôi tối ưu alpha* = c_v / c_th (7.4).
Đối chứng (7.5) và chỉ tiêu (7.6); cửa sổ (t, t+H] chứa phiên chuyển hợp đồng bị loại khỏi đánh giá (Nhận xét 4).

THƯ VIỆN: numpy (mô phỏng vectơ hóa, phân hoạch lấy phân vị đuôi), scipy.stats, jax (lọc ln h theo từng bộ tham số).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy import stats

from . import du_lieu as DL
from . import phan_phoi as PP
from .mo_hinh_jax import DacTa, e_abs, ham


# ----------------------------------------------------------------------------------------
# Nguồn mô phỏng của một mốc ước lượng lại
# ----------------------------------------------------------------------------------------
def nguon_moc(spec: DacTa, P: np.ndarray, f: dict, dl, s0: int, s1: int, x_tuong_lai: str = "ar1_kem_z") -> dict:
    """Chuẩn bị cho các phiên gốc t = s0-1 .. s1-2 (dự báo cho phiên s0 .. s1-1) của một mốc.

    P: (S_p, d) các bộ tham số (S_p = 1 với MLE). f, dl: dữ liệu và thống kê chuẩn hóa tại mốc.
    x_tuong_lai: "ar1" (đề cương) hoặc "ar1_kem_z" (thêm |z| và z của phiên vừa qua vào hồi quy của X)."""
    h = ham(spec.pp, spec.nx)
    lnh, z = (np.asarray(x) for x in h.loc_vmap(jnp.asarray(P), dl))
    a0, a1 = s0 - DL.BO_QUA, s1 - DL.BO_QUA
    cot = [DL.TEN_BIEN.index(b) for b in spec.bien]
    nx = spec.nx
    c, phi, a_z, b_z, resid = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros((1, nx))
    J = np.asarray(dl[1])
    if nx:
        x = f["X"][DL.BO_QUA:s0][:, cot]
        y, xl = x[1:], x[:-1]
        if x_tuong_lai == "ar1_kem_z":
            # X ở hàng k+1 đo đến cuối phiên BO_QUA+k, nên hồi quy theo z của phiên đó (bộ tham số đầu = MLE); bỏ phiên J
            zk = np.asarray(z[0, :a0 - 1], float)
            dung = ~J[:a0 - 1] & np.isfinite(zk)
            y, xl, zk = y[dung], xl[dung], zk[dung]
        elif x_tuong_lai != "ar1":
            raise ValueError(f"x_tuong_lai = {x_tuong_lai!r} khong hop le")
        resid = np.empty_like(y)
        for i in range(nx):
            cot_A = [np.ones(len(xl)), xl[:, i]] + ([np.abs(zk), zk] if x_tuong_lai == "ar1_kem_z" else [])
            A = np.column_stack(cot_A)
            b = np.linalg.lstsq(A, y[:, i], rcond=None)[0]
            c[i], phi[i] = b[0], b[1]
            if x_tuong_lai == "ar1_kem_z":
                a_z[i], b_z[i] = b[2], b[3]
            resid[:, i] = y[:, i] - A @ b
    pool_fhs = z[0, :a0][~J[:a0]]                    # phần dư chuẩn hóa trong mẫu (bộ tham số đầu = MLE)
    sh = P[:, 5 + nx + 3:]
    return {"P": P, "lnh_tiep": lnh[:, a0:a1], "X0": f["X"][s0:s1][:, cot] if nx else np.zeros((s1 - s0, 0)),
            "c": c, "phi": phi, "a_z": a_z, "b_z": b_z, "resid": resid, "pool_fhs": pool_fhs,
            "mabs": np.array([float(e_abs(spec.pp, jnp.asarray(s))) for s in sh]), "s0": s0}


def mo_phong_moc(spec: DacTa, ng: dict, D: np.ndarray, J: np.ndarray, S: int, Hmax: int, alphas, rng,
                 che_do: str = "tham_so") -> dict:
    """VaR, ES H phiên cho mọi phiên gốc của một mốc. Trả về mảng (số phiên, Hmax, số alpha, 2 phía)."""
    P = ng["P"]
    Sp, nx = len(P), spec.nx
    idx = rng.integers(Sp, size=S)                   # bộ tham số của từng đường
    om, ka, th, be = (P[idx, k] for k in range(1, 5))
    gam, dl = P[idx, 5:5 + nx], P[idx, 5 + nx:5 + nx + 3]
    mabs = ng["mabs"][idx]
    # kho z dùng chung cho các phiên của mốc (số ngẫu nhiên chung): z[đường, bước]
    if che_do == "loc":
        Zk = ng["pool_fhs"][rng.integers(len(ng["pool_fhs"]), size=(S, Hmax))]
    else:
        Zk = np.empty((S, Hmax))
        for s in np.unique(idx):
            m = idx == s
            Zk[m] = PP.lay_mau(int(m.sum()) * Hmax, spec.pp, P[s, 5 + nx + 3:], rng).reshape(-1, Hmax)
    n_ngay = ng["lnh_tiep"].shape[1]
    var = np.full((n_ngay, Hmax, len(alphas), 2), np.nan)
    es = np.full_like(var, np.nan)
    for i in range(n_ngay):
        t = ng["s0"] - 1 + i                          # phiên gốc
        lnh = ng["lnh_tiep"][idx, i]
        X = np.repeat(ng["X0"][i][None, :], S, 0)
        cum = np.zeros(S)
        for j in range(1, Hmax + 1):
            z = Zk[:, j - 1]
            cum = cum + np.exp(0.5 * lnh) * z
            R = np.expm1(cum / 100.0)
            for ia, a in enumerate(alphas):
                var[i, j - 1, ia, 0], es[i, j - 1, ia, 0] = _duoi(-R, a)
                var[i, j - 1, ia, 1], es[i, j - 1, ia, 1] = _duoi(R, a)
            if j < Hmax:
                if nx:
                    X = (ng["c"] + ng["phi"] * X + ng["a_z"] * np.abs(z)[:, None] + ng["b_z"] * z[:, None]
                         + ng["resid"][rng.integers(len(ng["resid"]), size=S)])
                s_ = np.where(J[min(t + j, len(J) - 1)], 0.0, ka * (np.abs(z) - mabs) + th * z)
                dj = D[min(t + j + 1, len(D) - 1)]
                lnh = np.clip(om + s_ + be * lnh + (np.sum(X * gam, 1) if nx else 0.0) + dl @ dj, -20, 20)
    return {"var": var, "es": es}


def _duoi(L: np.ndarray, a: float):
    S = len(L)
    k = max(int(math.ceil(a * S)), 1)
    top = np.partition(L, S - k)[S - k:]
    return float(top.min()), float(top.mean())


# ----------------------------------------------------------------------------------------
# Tổn thất thực tế, sàn mô phỏng lịch sử, đối chứng
# ----------------------------------------------------------------------------------------
def tong_that_thuc(d, Hmax: int = 3):
    """R_{t,H} = exp(tổng r_{t+1..t+H} / 100) - 1 trên chuỗi lợi suất CÙNG HỢP ĐỒNG, và mặt nạ hợp lệ.

    Lợi suất r đã được lấp tại phiên chuyển bằng giá F2M (xu_ly_dao_han = "f2m"), nên tổn thất qua phiên chuyển là
    tổn thất của một vị thế thật được cuộn sang hợp đồng mới. Cửa sổ (t, t+H] chứa lợi suất khuyết (thiếu F2M, hoặc
    cách xử lý "loai") bị đánh dấu không hợp lệ, đúng như cách cũ (Nhận xét 4 của đề cương chi tiết)."""
    r = d["r"].to_numpy(float)
    n = len(r)
    R = np.full((n, Hmax), np.nan)
    ok = np.zeros((n, Hmax), bool)
    for h in range(1, Hmax + 1):
        for t in range(n - h):
            w = r[t + 1:t + h + 1]
            if np.isfinite(w).all():
                R[t, h - 1] = math.expm1(float(w.sum()) / 100.0)
                ok[t, h - 1] = True
    return R, ok


def san_lich_su(R: np.ndarray, tmoc: np.ndarray, H: int, a: float) -> np.ndarray:
    """VaR H phiên theo mô phỏng lịch sử trên toàn bộ dữ liệu đến t (cửa sổ mở rộng): max hai phía."""
    out = np.full(len(tmoc), np.nan)
    for i, t in enumerate(tmoc):
        r = R[:t - H + 1, H - 1]
        r = r[np.isfinite(r)]
        if len(r) > 100:
            out[i] = max(-np.quantile(r, a), np.quantile(r, 1 - a))
    return out


def lich_su_cua_so(R: np.ndarray, tmoc: np.ndarray, H: int, a: float, cua_so: int = 250) -> np.ndarray:
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
def quy_tac(var, es, san, w, b, phi, eta, giam=0.01, g=0.005, ng_tang=0.0, ng_giam=0.0):
    """Áp dụng quy tắc sáu bước (năm bước của đề cương và vùng không điều chỉnh) cho các chuỗi var, es, san.

    Bước 6 (vùng không điều chỉnh): mức mục tiêu mi do năm bước tính; mức áp dụng chỉ đổi khi mi lệch khỏi mức đang áp
    dụng tối thiểu ng_tang (tăng) hoặc ng_giam (giảm), khi đó nhảy thẳng tới mi; luôn giữ mức sàn phi * san.
    Với ng_tang = ng_giam = 0, kết quả trùng quy tắc năm bước. Đệ quy của năm bước dùng mức mục tiêu."""
    n = len(var)
    m = np.full(n, np.nan)
    m1 = (1 - w) * var + w * es
    m3 = np.maximum((1 + b) * m1, phi * san)
    prev4 = prev = ap = None
    for i in range(n):
        if not np.isfinite(m3[i]):
            continue
        if prev4 is None:
            prev4, prev = m3[i], math.ceil(m3[i] / g) * g
        m4 = max(m3[i], eta * prev4 + (1 - eta) * m3[i])
        mi = max(phi * san[i], g * math.ceil(max(m4, prev - giam) / g - 1e-12))
        prev4, prev = m4, mi
        if ap is None or mi >= ap + ng_tang - 1e-12 or mi <= ap - ng_giam + 1e-12:
            ap = mi
        ap = max(ap, phi * san[i])
        m[i] = ap
    return m


def theo_lich_vsdc(m, san, ngay, moc=(1, 10, 20)):
    """Cập nhật theo lịch VSDC: mức áp dụng chỉ đổi ở phiên đầu tiên có ngày >= 1, 10, 20 của tháng (kể từ lần
    cập nhật trước), nhảy tới mức mục tiêu m; giữa hai ky giữ nguyên; luôn giữ mức sàn san. ngay: chuỗi ngày cùng độ dài."""
    import pandas as pd
    ngay = pd.DatetimeIndex(ngay)
    ky = [(t.year, t.month, sum(t.day >= x for x in moc)) for t in ngay]
    out = np.full(len(m), np.nan)
    ap, ky_truoc = None, None
    for i in range(len(m)):
        if not np.isfinite(m[i]):
            continue
        if ap is None or ky[i] != ky_truoc:
            ap, ky_truoc = m[i], ky[i]
        ap = max(ap, san[i])
        out[i] = ap
    return out


def _cua_so_vsdc(x: np.ndarray, t: int, N: int):
    """N giá trị tổn thất hữu hạn ngay trước và tại chỉ số t (dùng dữ liệu đến t); None nếu chưa đủ N."""
    if t < N - 1:
        return None
    w = x[t - N + 1:t + 1]
    return w if np.isfinite(w).all() else None


def vsdc_gia_tri(r, chi_so, N: int, kieu: str, n: float, zc: float = 2.89, hai_ngay: bool = False) -> dict:
    """Tỷ lệ ký quỹ ban đầu theo văn bản VSDC tại các chỉ số phiên chi_so (đối chứng, không phải chuỗi VSDC công bố).

    LÝ THUYẾT: kieu = "mvar" (QĐ 61/QĐ-VSD 2022 và QĐ 12/QĐ-HĐTV 2023): IM = MVaR sqrt(n), MVaR = mean + Z delta,
        Z = zc + (zc^2 - 1) S / 6 + (zc^3 - 3 zc) K / 24 - (2 zc^3 - 5 zc) S^2 / 36, S độ lệch, K độ nhọn thừa.
        kieu = "var3" (QĐ 26/QĐ-HĐTV 2025): IM = VaR sqrt(n), VaR = mean + 3 delta, biến động 2 ngày (hai_ngay).
        Hai phía: tổn thất của vị thế mua (-u) và vị thế bán (+u); lấy phía lớn hơn, sàn 0. u là tỷ lệ biến động giá
        thường (expm1 của tổng lợi suất log). Văn bản không nêu độ nhọn là thô hay thừa; dùng độ nhọn thừa.
    r: lợi suất log (%) theo phiên (khuyết = NaN). Trả về {chi_so: tỷ lệ ký quỹ hoặc NaN}."""
    r = np.asarray(r, float)
    u = np.expm1((r + np.r_[np.nan, r[:-1]] if hai_ngay else r) / 100.0)
    out = {}
    for t in chi_so:
        best = np.nan
        for dau in (-1.0, 1.0):
            w = _cua_so_vsdc(dau * u, int(t), N)
            if w is None:
                continue
            mean, sd = float(w.mean()), float(w.std(ddof=1))
            if kieu == "mvar":
                S = float(stats.skew(w, bias=False))
                Kt = float(stats.kurtosis(w, bias=False))
                Z = zc + (zc ** 2 - 1) * S / 6 + (zc ** 3 - 3 * zc) * Kt / 24 - (2 * zc ** 3 - 5 * zc) * S ** 2 / 36
            else:
                Z = 3.0
            v = max((mean + Z * sd) * math.sqrt(n), 0.0)
            best = v if not np.isfinite(best) else max(best, v)
        out[int(t)] = best
    return out


def vsdc_theo_lich(ngay, tmoc, gia_tri_tai, tre: int = 2, moc=(1, 10, 20)) -> np.ndarray:
    """Cập nhật theo lịch VSDC: tại phiên đầu tiên của mỗi kỳ (ngày 1, 10, 20) tính lại bằng dữ liệu đến `tre` phiên
    trước đó (công bố trước >= 2 ngày làm việc), giữ nguyên đến kỳ sau. gia_tri_tai(t) -> giá trị tính bằng dữ liệu đến t."""
    import pandas as pd
    ngay = pd.DatetimeIndex(ngay)
    ky = [(t.year, t.month, sum(t.day >= x for x in moc)) for t in ngay]
    out = np.full(len(tmoc), np.nan)
    hien, ky_truoc = np.nan, None
    for i, t in enumerate(tmoc):
        if ky[i] != ky_truoc:
            ky_truoc = ky[i]
            hien = gia_tri_tai(int(t) - tre)
        out[i] = hien
    return out


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
           "thieu_hut_moi_phien": float(thieu.mean()) if len(thieu) else float("nan"),
           "thieu_hut_max": float(thieu.max()) if len(thieu) else 0.0,
           "m_max": float(mm.max()) if len(mm) else float("nan")}
    idx = np.flatnonzero(ok)
    mf = m[idx[0]:idx[-1] + 1] if len(idx) > 1 else mm
    mf = mf[np.isfinite(mf)]
    dm = np.diff(mf)
    out["sd_dm"] = float(dm.std()) if len(dm) else float("nan")
    out["so_lan_dieu_chinh"] = int((np.abs(dm) > 1e-12).sum())
    out["thuan_chu_ky"] = float(np.max(mf[20:] / mf[:-20])) if len(mf) > 25 else float("nan")
    return out


def kiem_dinh_vuot(m, L, hop_le, H, a):
    """Kupiec và Christoffersen trên dãy lấy thưa mỗi H phiên, và kiểm định phân vị động với phương sai HAC trên
    toàn bộ dãy chồng lấn (Nhận xét 3)."""
    from .danh_gia import christoffersen, kupiec, phan_vi_dong_hac
    idx = np.arange(0, len(m), H)
    ok = hop_le[idx] & np.isfinite(m[idx]) & np.isfinite(L[idx])
    I = np.where(ok, (L[idx] > m[idx]).astype(float), np.nan)
    n1, n = int(np.nansum(I)), int(ok.sum())
    if n < 20:
        return {}
    _, p = kupiec(n1, n, a)
    ch = christoffersen(I, a)
    ok_d = hop_le & np.isfinite(m) & np.isfinite(L)
    I_d = np.where(ok_d, (L > m).astype(float), np.nan)
    dq, p_dq = phan_vi_dong_hac(I_d, m, a, H)
    return {"n": n, "vuot": n1, "kupiec_p": p, "ind_p": ch["p_ind"], "cc_p": ch["p_cc"],
            "dq_hac_thong_ke": dq, "dq_hac_p": p_dq, "n_chong_lan": int(ok_d.sum())}


def chi_phi(m, L, hop_le, c_v=1.0, c_th=100.0):
    ok = hop_le & np.isfinite(m) & np.isfinite(L)
    return float(np.mean(c_v * m[ok] + c_th * np.maximum(L[ok] - m[ok], 0)))


def duong_bien(luoi: list[dict]) -> list[dict]:
    """Đường biên đánh đổi (7.4): các cấu hình không bị trội về (m_tb, thiếu hụt mỗi phiên)."""
    diem = sorted(luoi, key=lambda x: (x["m_tb"], x["thieu_hut_moi_phien"]))
    out, tot = [], float("inf")
    for x in diem:
        if x["thieu_hut_moi_phien"] < tot:
            out.append(x)
            tot = x["thieu_hut_moi_phien"]
    return out
