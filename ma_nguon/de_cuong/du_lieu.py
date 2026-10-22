"""Dữ liệu và xây dựng biến theo mục 2 của đề cương.

LÝ THUYẾT: Chuỗi ngày gồm bảy trường (giá mở, cao, thấp, đóng, V, V+, V-) lấy từ nến phút.
Phiên chuyển hợp đồng J_t = 1 (phiên đầu tiên sau ngày đáo hạn, thứ Năm thứ ba của tháng): P_{t-1} của F1M
thuộc hợp đồng cũ. Cách xử lý chọn bằng [du_lieu].xu_ly_dao_han (22/10/2026):
  "f2m"  : lợi suất cùng hợp đồng r_t = 100 ln(P_t / P^{F2M}_{t-1}) và r^dem_t = 100 ln(O_t / P^{F2M}_{t-1}),
           vì hợp đồng là F2M ở phiên t-1 chính là F1M ở phiên t; chỉ khuyết khi thiếu giá F2M phiên t-1;
  "loai" : lợi suất phiên chuyển là quan sát khuyết (cách cũ, mục 2.3 của đề cương chi tiết).
Cột `J` luôn là chỉ thị phiên chuyển theo lịch (dùng cho biến giả D_sau, nhóm phiên); cột `khuyet` là cờ lợi suất
khuyết (dùng cho hàm hợp lý và đệ quy phương sai). Đại lượng đại diện phương sai là Garman-Klass cộng
phần qua đêm, nhân hệ số thang đo c_rho (mục 2.4). Biến ngoại sinh và các thống kê chuẩn hóa
chỉ dùng dữ liệu đến thời điểm ước lượng lại s (mục 2.7), nên hàm `dac_trung(ngay, s)` nhận s.

THƯ VIỆN: pandas (gộp nến phút thành ngày), numpy.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
CSV = GOC / "du_lieu" / "goc" / "ohlc_export.csv"
CSV_F2M = GOC / "du_lieu" / "goc" / "VN30F2M_1d.csv"
# Giá trị mặc định; `ap_dung_cau_hinh` ghi đè bằng cau_hinh/de_cuong.toml [du_lieu].
N_V = 60            # cửa sổ khối lượng trung bình (mục 2.6)
EPS = 1e-6          # hằng số tránh logarit của không (Bảng 4)
BO_QUA = 61         # số phiên đầu bỏ đi vì thiếu lịch sử 60 phiên cho biến ngoại sinh
NGHI = 3            # số ngày làm việc liên tiếp không giao dịch để coi là kỳ nghỉ dài (mục 2.5)
XU_LY_DAO_HAN = "loai"   # "f2m" hoặc "loai"; cấu hình chính thức đặt "f2m"
TEN_BIEN = ["A", "G", "F", "B", "O"]


def ap_dung_cau_hinh(cfg: dict | None):
    """Đặt các hằng số của mục 2 theo cấu hình (một nguồn sự thật duy nhất)."""
    global N_V, EPS, BO_QUA, NGHI, XU_LY_DAO_HAN, CSV_F2M
    if cfg:
        c = cfg["du_lieu"]
        N_V, EPS, BO_QUA, NGHI = int(c["n_V"]), float(c["eps"]), int(c["bo_qua_dau"]), int(c["nghi_dai_toi_thieu"])
        XU_LY_DAO_HAN = c.get("xu_ly_dao_han", "loai")
        if c.get("du_lieu_f2m"):
            CSV_F2M = GOC / c["du_lieu_f2m"]


def doc_f2m(csv: Path) -> pd.Series:
    """Giá đóng cửa ngày của VN30F2M, chỉ mục là ngày giao dịch (giờ Việt Nam)."""
    b = pd.read_csv(csv, usecols=["datetime_vn", "close"])
    s = pd.Series(b["close"].to_numpy(float), index=pd.to_datetime(b["datetime_vn"].str[:10]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def doc_ngay(csv: Path = CSV) -> pd.DataFrame:
    """Gộp nến phút thành bảng ngày: open, high, low, close, vol, buy, sell."""
    b = pd.read_csv(csv, usecols=["TRADING_DATE", "OPEN_PX", "HIGH_PX", "LOW_PX", "CLOSE_PX",
                                  "VOL", "TRADING_TIME", "BUY_VOL", "SELL_VOL"])
    b = b.sort_values(["TRADING_DATE", "TRADING_TIME"], kind="stable")
    g = b.groupby("TRADING_DATE", sort=True)
    n = pd.DataFrame({"open": g["OPEN_PX"].first(), "high": g["HIGH_PX"].max(),
                      "low": g["LOW_PX"].min(), "close": g["CLOSE_PX"].last(),
                      "vol": g["VOL"].sum(), "buy": g["BUY_VOL"].sum(),
                      "sell": g["SELL_VOL"].sum()})
    n.index = pd.to_datetime(n.index.astype(str), format="%Y%m%d")
    n.index.name = "date"
    return n.reset_index()


def thu_nam_thu_ba(nam: int, thang: int) -> pd.Timestamp:
    """Thứ Năm thứ ba của tháng (ngày giao dịch cuối cùng theo Bảng 3 của đề cương)."""
    thu = [t[calendar.THURSDAY] for t in calendar.monthcalendar(nam, thang) if t[calendar.THURSDAY] > 0]
    return pd.Timestamp(nam, thang, thu[2])


def chi_so_dao_han(ngay: pd.Series) -> list[int]:
    """Chỉ số phiên đáo hạn: thứ Năm thứ ba; nếu nghỉ thì phiên quan sát liền trước trong tháng."""
    kq = []
    for nam in range(ngay.iloc[0].year, ngay.iloc[-1].year + 1):
        for thang in range(1, 13):
            m = thu_nam_thu_ba(nam, thang)
            if m < ngay.iloc[0] or m > ngay.iloc[-1]:
                continue
            ok = np.flatnonzero((ngay <= m) & (ngay.dt.year == nam) & (ngay.dt.month == thang))
            kq.append(int(ok[-1]))
    return kq


def xay_ngay(csv: Path = CSV, cfg: dict | None = None) -> pd.DataFrame:
    """Bảng ngày đầy đủ biến thô: lợi suất, J, tau, biến giả lịch, rho_P, rho_GK, rho_RS, rho (cả qua đêm)."""
    ap_dung_cau_hinh(cfg)
    d = doc_ngay(csv)
    n = len(d)
    dh = chi_so_dao_han(d["date"])
    J = np.zeros(n, bool)
    D_dh = np.zeros(n, bool)
    tau = np.full(n, np.nan)
    for e in dh:
        D_dh[e] = True
        if e + 1 < n:
            J[e + 1] = True
    moc = np.array(dh)
    for t in range(n):
        sau = moc[moc >= t]
        if len(sau):
            tau[t] = sau[0] - t
        else:   # sau kỳ đáo hạn cuối của mẫu: đếm ngày làm việc tới thứ Năm thứ ba tháng kế tiếp
            ngay_t = d["date"][t]
            th = ngay_t.month % 12 + 1
            moc_sau = thu_nam_thu_ba(ngay_t.year + (ngay_t.month == 12), th)
            tau[t] = len(pd.bdate_range(ngay_t, moc_sau)) - 1
    # kỳ nghỉ: số ngày làm việc (thứ 2-6) không có phiên giữa hai phiên liên tiếp
    nghi = np.zeros(n, bool)
    for t in range(1, n):
        kc = pd.bdate_range(d["date"][t - 1] + pd.Timedelta(days=1), d["date"][t] - pd.Timedelta(days=1))
        nghi[t] = len(kc) >= NGHI
    d["J"], d["D_dh"], d["D_sau"], d["D_nghi"], d["tau"] = J, D_dh, J.copy(), nghi, tau
    lo, hi, op, cl = (np.log(d[c].to_numpy()) for c in ("low", "high", "open", "close"))
    u, dd, c = hi - op, lo - op, cl - op
    d["rho_p"] = 1e4 * (u - dd) ** 2 / (4 * np.log(2))                                   # (3)
    d["rho_gk"] = np.maximum(1e4 * (0.5 * (u - dd) ** 2 - (2 * np.log(2) - 1) * c ** 2), 0.0)  # (4)
    d["rho_rs"] = np.maximum(1e4 * (u * (u - c) + dd * (dd - c)), 0.0)                    # (5)
    d["r"] = 100 * np.log(d["close"] / d["close"].shift(1))
    d["dem"] = 100 * np.log(d["open"] / d["close"].shift(1))
    d.loc[d["J"], ["r", "dem"]] = np.nan
    if XU_LY_DAO_HAN == "f2m":
        # giá F2M của phiên liền trước = giá của chính hợp đồng mới (F1M ở phiên t)
        f2 = doc_f2m(CSV_F2M)
        p2 = f2.reindex(d["date"].shift(1)).to_numpy()
        cl, op_ = d["close"].to_numpy(float), d["open"].to_numpy(float)
        lap = d["J"].to_numpy() & np.isfinite(p2) & (p2 > 0)
        d.loc[lap, "r"] = 100 * np.log(cl[lap] / p2[lap])
        d.loc[lap, "dem"] = 100 * np.log(op_[lap] / p2[lap])
    elif XU_LY_DAO_HAN != "loai":
        raise ValueError(f"xu_ly_dao_han = {XU_LY_DAO_HAN!r} khong hop le (chi 'f2m' hoac 'loai')")
    d["khuyet"] = d["r"].isna().to_numpy()
    d["rho"] = d["rho_gk"] + d["dem"] ** 2
    d["khoa_bien"] = (d["open"] == d["high"]) & (d["high"] == d["low"]) & (d["low"] == d["close"])
    return d


def dac_trung(d: pd.DataFrame, s: int) -> dict:
    """Biến ngoại sinh và đại lượng đại diện, thống kê chỉ dùng chỉ số phiên <= s (mục 2.4-2.7).

    Trả về dict với: Xtho (n x 5 chưa chuẩn hóa, đã trễ một phiên), X (đã chuẩn hóa),
    rho_sao (rho* = c_rho * rho), c_rho, tau, D (n x 3), r, J.
    """
    n = len(d)
    V = d["vol"].to_numpy(float)
    vbar = np.full(n, np.nan)          # trung bình V_{t-1..t-60}
    cs = np.cumsum(np.r_[0.0, V])
    for t in range(N_V, n):
        vbar[t] = (cs[t] - cs[t - N_V]) / N_V
    tau = d["tau"].to_numpy()
    ti = np.arange(n)
    hop_le = (ti >= N_V) & (ti <= s) & ~np.isnan(vbar)
    ti_tau = tau.astype(int)
    vs = np.ones(int(np.nanmax(tau)) + 1)
    for k in range(len(vs)):
        m = hop_le & (ti_tau == k)
        if m.any():
            vs[k] = np.mean(V[m] / vbar[m])
    A = np.log(V) - np.log(vbar * vs[ti_tau])
    r = d["r"].to_numpy()
    rho = d["rho"].to_numpy()
    khoa = d["khoa_bien"].to_numpy()
    ok = (~np.isnan(r)) & (~np.isnan(rho)) & (ti <= s) & ~khoa
    c_rho = float(np.sum(r[ok] ** 2) / np.sum(rho[ok]))
    gk = d["rho_gk"].to_numpy()
    gk_thay = np.where(khoa, np.where(np.isnan(r), 0.0, r ** 2), gk)   # Nhận xét 1
    G = np.log(gk_thay + EPS)
    cg = np.cumsum(np.r_[0.0, gk_thay])
    F = np.full(n, np.nan)
    for t in range(N_V - 1, n):
        F[t] = np.log((cg[t + 1] - cg[t + 1 - N_V]) / N_V + EPS)
    buy, sell = d["buy"].to_numpy(float), d["sell"].to_numpy(float)
    B = np.abs((buy - sell) / np.maximum(buy + sell, 1.0))
    dem = d["dem"].to_numpy()
    O = np.log(dem ** 2 + EPS)
    tho = np.column_stack([A, G, F, B, O])
    X = np.full_like(tho, np.nan)
    X[1:] = tho[:-1]                     # biến vào phương trình của phiên t đo đến cuối phiên t-1
    Xz = np.full_like(X, np.nan)
    for j in range(X.shape[1]):
        m = (ti >= BO_QUA) & (ti <= s) & ~np.isnan(X[:, j])
        Xz[:, j] = (X[:, j] - X[m, j].mean()) / X[m, j].std(ddof=1)
    Xz = np.where(np.isnan(Xz) & (ti[:, None] >= BO_QUA), 0.0, Xz)   # O khuyết ở phiên chuyển -> 0
    D = d[["D_dh", "D_sau", "D_nghi"]].to_numpy(float)
    return {"Xtho": X, "X": Xz, "rho_sao": c_rho * rho, "c_rho": c_rho, "tau": tau, "D": D,
            "r": r, "J": d["khuyet"].to_numpy(), "J_chuyen": d["J"].to_numpy(), "khoa": khoa, "vs": vs}


def chia_khoi(n_phien: int, ul: int = 700, xt: int = 420) -> dict:
    """Ba khối thời gian (Bảng 7), chỉ số trên chuỗi đã bỏ BO_QUA phiên đầu."""
    return {"UL": (0, ul), "XT": (ul, ul + xt), "KT": (ul + xt, n_phien)}
