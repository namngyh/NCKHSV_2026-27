"""Biến ngoại sinh bổ sung RV: biến động thực tế đóng-đến-đóng tách buổi sáng, nghỉ trưa, buổi chiều và qua đêm.

LÝ THUYẾT (theo `uoc_luong_mle_khoang_nghi.py` và `uoc_luong_mle_roll_f2m.py` trong luu_tru): mỗi ngày tách thành các buổi tại
khoảng trống trên 30 phút giữa hai nến (nghỉ trưa; mỗi ngày của mẫu có đúng một khoảng trống). Trong mỗi buổi, nến đầu dùng
lợi suất mở-đến-đóng, các nến sau dùng lợi suất giữa hai giá đóng cửa liền nhau; bình phương rồi cộng thành RV của buổi
(rv_sang, rv_chieu). Hai khoảng còn lại của ngày được cộng như hai số hạng riêng: lợi suất nghỉ trưa (từ đóng cửa buổi sáng đến
mở cửa buổi chiều) và lợi suất qua đêm (từ đóng cửa phiên trước đến mở cửa phiên này):
    RV_t = rv_sang + rv_chieu + nghi_trua^2 + qua_dem^2        (đơn vị %^2).
Bốn đoạn nối tiếp nhau nên cộng đúng bằng lợi suất đóng-đến-đóng r_t: RV_t phủ cùng khoảng với r_t, không bỏ sót nghỉ trưa
hay qua đêm. Ở phiên đầu tiên sau đáo hạn, lợi suất qua đêm lấy từ bảng ngày `d` (cột `dem`), vốn đã tính bằng giá VN30F2M của
phiên trước (cùng hợp đồng), nên cú nhảy giữa hai hợp đồng không lọt vào RV; nếu thiếu giá F2M thì `dem` khuyết và số hạng
qua đêm bằng 0, như mã cũ.
Biến vào phương trình phương sai của phiên t là RV_{t-1} = log(RV_{t-1} + eps), chuẩn hóa bằng thống kê của các phiên đến thời
điểm ước lượng, giống năm biến của đề cương. Phiên khóa biên (bốn mức giá bằng nhau) thay RV bằng bình phương lợi suất, như G.
VÌ SAO LÀ MÔ-ĐUN RIÊNG: mã băm của b01 và b02 gồm tệp du_lieu.py; b02 lại ghi kèm tên máy, giờ chạy vào đầu ra, nên chạy
lại b02 đổi mã băm đầu ra và làm mất hiệu lực b03a, b03b (mô phỏng bằng dữ liệu giả, không dùng RV). Để nguyên du_lieu.py
và gắn RV bằng `gan()` giữ được tiến độ b03a, b03b; các bước b04 trở đi chạy lại vì tệp bước của chúng đã đổi.
`gan()` thêm "RV" vào cuối du_lieu.TEN_BIEN (thứ tự năm biến cũ không đổi) và bọc du_lieu.dac_trung để thêm cột thứ sáu.
THƯ VIỆN: pandas (đọc CSV, gộp theo ngày, theo buổi), numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import du_lieu as DL

TEN = "RV"
NGUONG_NGHI = 30.0   # phút: khoảng trống giữa hai nến lớn hơn ngưỡng này là khoảng nghỉ trưa
_RV: dict = {}       # bộ nhớ đệm theo đường dẫn tệp nến phút


def doc_rv(csv=None) -> pd.DataFrame:
    """Bảng ngày gồm: date, rv_sang, rv_chieu (%^2, trong buổi), nghi_trua (%, lợi suất qua khoảng nghỉ), so_nen."""
    csv = DL.CSV if csv is None else csv
    k = str(csv)
    if k not in _RV:
        b = pd.read_csv(csv, usecols=["TRADING_DATE", "TRADING_TIME", "OPEN_PX", "CLOSE_PX"])
        b = b.sort_values(["TRADING_DATE", "TRADING_TIME"], kind="stable").reset_index(drop=True)
        ngay = b["TRADING_DATE"]
        phut = pd.to_timedelta(b["TRADING_TIME"]).dt.total_seconds() / 60
        buoi = (phut.groupby(ngay).diff() > NGUONG_NGHI).groupby(ngay).cumsum()
        dau_buoi = b.groupby([ngay, buoi]).cumcount() == 0
        dong_truoc = b.groupby([ngay, buoi])["CLOSE_PX"].shift(1)
        gia_truoc = dong_truoc.where(~dau_buoi, b["OPEN_PX"])        # nến đầu buổi: mở-đến-đóng
        bp = (100 * np.log(b["CLOSE_PX"] / gia_truoc)) ** 2
        # lợi suất qua khoảng nghỉ: từ đóng cửa nến liền trước (cuối buổi trước) đến mở cửa nến đầu buổi sau
        dong_ngay_truoc = b.groupby(ngay)["CLOSE_PX"].shift(1)
        nghi = (100 * np.log(b["OPEN_PX"] / dong_ngay_truoc)).where(dau_buoi & (buoi > 0))
        r = pd.DataFrame({"rv_sang": bp.where(buoi == 0, 0.0).groupby(ngay).sum(),
                          "rv_chieu": bp.where(buoi > 0, 0.0).groupby(ngay).sum(),
                          "nghi_trua": nghi.fillna(0.0).groupby(ngay).sum(),
                          "so_nen": b.groupby(ngay).size()})
        r.index = pd.to_datetime(r.index.astype(str), format="%Y%m%d")
        r.index.name = "date"
        _RV[k] = r.reset_index()
    return _RV[k]


def them_rv(d: pd.DataFrame, csv=None) -> pd.DataFrame:
    """Bản sao của bảng ngày `d` có thêm rv_sang, rv_chieu, nghi_trua, so_nen và rv = RV toàn ngày đóng-đến-đóng."""
    r = doc_rv(csv).set_index("date")
    d = d.copy()
    for c in ("rv_sang", "rv_chieu", "nghi_trua", "so_nen"):
        d[c] = d["date"].map(r[c])
    d["rv"] = d["rv_sang"] + d["rv_chieu"] + d["nghi_trua"] ** 2 + np.nan_to_num(d["dem"].to_numpy(float)) ** 2
    return d


def _them_cot(f: dict, d: pd.DataFrame, s: int) -> dict:
    """Thêm cột RV vào Xtho (thô, đã trễ một phiên) và X (đã chuẩn hóa bằng thống kê đến phiên s)."""
    if "rv" not in d.columns:
        d = them_rv(d)
    r, khoa = f["r"], f["khoa"]
    rv_thay = np.where(khoa, np.where(np.isnan(r), 0.0, r ** 2), d["rv"].to_numpy(float))
    sao = np.log(rv_thay + DL.EPS)
    n = len(d)
    ti = np.arange(n)
    x = np.full(n, np.nan)
    x[1:] = sao[:-1]
    m = (ti >= DL.BO_QUA) & (ti <= s) & ~np.isnan(x)
    z = (x - x[m].mean()) / x[m].std(ddof=1)
    z = np.where(np.isnan(z) & (ti >= DL.BO_QUA), 0.0, z)
    f["Xtho"] = np.column_stack([f["Xtho"], x])
    f["X"] = np.column_stack([f["X"], z])
    return f


def gan() -> None:
    """Gắn RV vào du_lieu: thêm tên vào TEN_BIEN và bọc dac_trung. Gọi lại nhiều lần vẫn an toàn."""
    if TEN not in DL.TEN_BIEN:
        DL.TEN_BIEN.append(TEN)
    if getattr(DL.dac_trung, "co_rv", False):
        return
    goc = DL.dac_trung

    def dac_trung(d, s):
        return _them_cot(goc(d, s), d, s)

    dac_trung.co_rv = True
    dac_trung.__doc__ = goc.__doc__
    DL.dac_trung = dac_trung
