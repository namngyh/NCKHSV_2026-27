"""Kiểm thử biến RV đóng-đến-đóng (tách buổi sáng, nghỉ trưa, buổi chiều, qua đêm) và việc gắn vào ma trận X.

LÝ THUYẾT: RV_t = rv_sang + rv_chieu + nghi_trua^2 + qua_dem^2. Trong mỗi buổi, nến đầu dùng lợi suất mở-đến-đóng, các nến sau
dùng lợi suất giữa hai giá đóng cửa liền nhau; nghi_trua là lợi suất từ đóng cửa buổi sáng đến mở cửa buổi chiều; qua_dem là
cột `dem` của bảng ngày (đã sửa bằng giá F2M ở phiên sau đáo hạn).
THƯ VIỆN: pytest, numpy, pandas; ma_nguon.de_cuong.bien_rv, du_lieu.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import bien_rv as BRV  # noqa: E402
from de_cuong import du_lieu as DL  # noqa: E402


def _csv_nen(tmp_path, ngay_nen):
    """CSV nến phút tổng hợp. `ngay_nen`: mỗi ngày là danh sách (giờ, giá mở, giá đóng)."""
    dong = []
    for i, nen in enumerate(ngay_nen):
        for gio, mo, dong_ in nen:
            dong.append({"SYMBOL": "VN30F1M", "TRADING_DATE": 20240102 + i, "OPEN_PX": mo, "HIGH_PX": max(mo, dong_),
                         "LOW_PX": min(mo, dong_), "CLOSE_PX": dong_, "VOL": 10, "TRADING_TIME": gio, "BUY_VOL": 5,
                         "BUY_VAL": 0, "SELL_VOL": 5, "SELL_VAL": 0})
    f = tmp_path / "phut.csv"
    pd.DataFrame(dong).to_csv(f, index=False)
    return f


NGAY = [("09:00:00", 100.0, 101.0), ("09:01:00", 101.0, 100.0), ("13:00:00", 102.0, 103.0), ("13:01:00", 103.0, 102.0)]


def test_rv_tach_buoi_va_nghi_trua(tmp_path):
    d = BRV.doc_rv(_csv_nen(tmp_path, [NGAY]))
    sang = 1e4 * (np.log(101 / 100) ** 2 + np.log(100 / 101) ** 2)
    chieu = 1e4 * (np.log(103 / 102) ** 2 + np.log(102 / 103) ** 2)    # nến đầu buổi chiều: mở-đến-đóng
    assert np.isclose(d["rv_sang"].iloc[0], sang)
    assert np.isclose(d["rv_chieu"].iloc[0], chieu)
    assert np.isclose(d["nghi_trua"].iloc[0], 100 * np.log(102 / 100))   # đóng cửa 100 -> mở cửa chiều 102
    assert d["so_nen"].iloc[0] == 4


def test_rv_khong_gom_qua_dem_trong_doc_rv(tmp_path):
    # Giá nhảy 100 -> 150 giữa hai ngày, trong mỗi buổi giá đứng yên: mọi thành phần trong ngày bằng 0.
    n1 = [("09:00:00", 100.0, 100.0), ("13:00:00", 100.0, 100.0)]
    n2 = [("09:00:00", 150.0, 150.0), ("13:00:00", 150.0, 150.0)]
    d = BRV.doc_rv(_csv_nen(tmp_path, [n1, n2]))
    assert (d[["rv_sang", "rv_chieu", "nghi_trua"]] == 0).all().all()


def test_rv_toan_ngay_cong_qua_dem(tmp_path):
    f = _csv_nen(tmp_path, [NGAY, NGAY])
    d = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "dem": [np.nan, 0.5]})
    e = BRV.them_rv(d, f)
    trong = e["rv_sang"] + e["rv_chieu"] + e["nghi_trua"] ** 2
    assert np.isclose(e["rv"].iloc[0], trong.iloc[0])                 # dem khuyết: số hạng qua đêm bằng 0
    assert np.isclose(e["rv"].iloc[1], trong.iloc[1] + 0.5 ** 2)


def test_gan_them_rv_o_cuoi_va_an_toan_khi_goi_lai():
    BRV.gan()
    BRV.gan()
    assert DL.TEN_BIEN.count("RV") == 1 and DL.TEN_BIEN[-1] == "RV"
    assert DL.TEN_BIEN[:5] == ["A", "G", "F", "B", "O"]   # thứ tự năm biến của đề cương không đổi
