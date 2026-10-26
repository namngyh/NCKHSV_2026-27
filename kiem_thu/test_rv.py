"""Kiểm thử biến RV (tách buổi sáng, nghỉ trưa, buổi chiều) và điều kiện (i) của cổng bằng kiểm định nhị thức.

LÝ THUYẾT: RV_t = rv_sang + rv_chieu + nghi_trua^2 + qua_dem^2. Trong mỗi buổi, nến đầu dùng lợi suất mở-đến-đóng, các nến sau
dùng lợi suất giữa hai giá đóng cửa liền nhau; nghi_trua là lợi suất từ đóng cửa buổi sáng đến mở cửa buổi chiều. Điều kiện
(i): mỗi ô là một kiểm định nhị thức một phía so với ngưỡng, hiệu chỉnh Holm.
THƯ VIỆN: pytest, numpy, pandas; ma_nguon.de_cuong.du_lieu, mo_phong.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import du_lieu as DL  # noqa: E402
from de_cuong import mo_phong as MP  # noqa: E402


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
    d = DL.doc_ngay(_csv_nen(tmp_path, [NGAY]))
    sang = 1e4 * (np.log(101 / 100) ** 2 + np.log(100 / 101) ** 2)
    chieu = 1e4 * (np.log(103 / 102) ** 2 + np.log(102 / 103) ** 2)    # nến đầu buổi chiều: mở-đến-đóng
    assert np.isclose(d["rv_sang"].iloc[0], sang)
    assert np.isclose(d["rv_chieu"].iloc[0], chieu)
    assert np.isclose(d["nghi_trua"].iloc[0], 100 * np.log(102 / 100))   # đóng cửa 100 -> mở cửa chiều 102
    assert d["so_nen"].iloc[0] == 4


def test_rv_trong_ngay_khong_gom_qua_dem(tmp_path):
    # Giá nhảy 100 -> 150 giữa hai ngày, trong mỗi buổi giá đứng yên: mọi thành phần trong ngày bằng 0.
    n1 = [("09:00:00", 100.0, 100.0), ("13:00:00", 100.0, 100.0)]
    n2 = [("09:00:00", 150.0, 150.0), ("13:00:00", 150.0, 150.0)]
    d = DL.doc_ngay(_csv_nen(tmp_path, [n1, n2]))
    assert (d[["rv_sang", "rv_chieu", "nghi_trua"]] == 0).all().all()


def test_rv_la_cot_cuoi_cua_X():
    assert DL.TEN_BIEN == ["A", "G", "F", "B", "O", "RV"]   # thứ tự năm biến của đề cương không đổi


CG = {"do_phu_min": 0.90, "do_phu_max": 0.99, "bac_bo_sai_max": 0.10, "muc_y_nghia_do_phu": 0.05,
      "kiem_dinh_do_phu": "nhi_thuc"}


def test_cong_nhi_thuc_khong_loai_vi_nhieu():
    dk = MP.dieu_kien_cong({"E2|kappa": (0.899, 1000), "E1|beta": (0.95, 1000)}, {"E2|theta": (0.05, 1000)}, CG)
    assert dk["dat_nguong_cung"] is False and dk["dat_nhi_thuc"] is True and dk["dat"] is True


def test_cong_nhi_thuc_loai_khi_lech_ro():
    dk = MP.dieu_kien_cong({"E2|kappa": (0.85, 1000), "E1|beta": (0.95, 1000)}, {}, CG)
    assert dk["dat"] is False and "E2|kappa|duoi" in dk["o_khong_dat"]
    dk = MP.dieu_kien_cong({"E1|beta": (0.95, 1000)}, {"E2|theta": (0.15, 1000)}, CG)
    assert dk["dat"] is False


def test_cong_nguong_cung_giu_cach_de_cuong():
    dk = MP.dieu_kien_cong({"E2|kappa": (0.899, 1000)}, {}, {**CG, "kiem_dinh_do_phu": "nguong_cung"})
    assert dk["dat"] is False
