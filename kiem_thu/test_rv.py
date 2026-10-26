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


def test_nghi_trua2_cong_binh_phuong_tung_khoang(tmp_path):
    # hai khoảng trống > 30 phút trong ngày (mất dữ liệu giữa buổi): 100 -> 102 và 102 -> 99
    ngay = [("09:00:00", 100.0, 100.0), ("10:00:00", 102.0, 102.0), ("13:00:00", 99.0, 99.0)]
    d = DL.doc_ngay(_csv_nen(tmp_path, [ngay]))
    g1, g2 = 100 * np.log(102 / 100), 100 * np.log(99 / 102)
    assert np.isclose(d["nghi_trua2"].iloc[0], g1 ** 2 + g2 ** 2)
    assert not np.isclose(d["nghi_trua2"].iloc[0], (g1 + g2) ** 2)


def test_tau_cuoi_mau_tinh_toi_dao_han_thang_nay(tmp_path):
    # mẫu kết thúc 10/07/2024, trước kỳ đáo hạn tháng 7 (thứ Năm thứ ba = 18/07/2024): tau phải đếm tới 18/07, không tới tháng 8
    dong = []
    for ngay in pd.bdate_range("2024-07-01", "2024-07-10"):
        dong.append({"SYMBOL": "VN30F1M", "TRADING_DATE": int(ngay.strftime("%Y%m%d")), "OPEN_PX": 100.0, "HIGH_PX": 101.0,
                     "LOW_PX": 99.0, "CLOSE_PX": 100.5, "VOL": 10, "TRADING_TIME": "09:00:00", "BUY_VOL": 5,
                     "BUY_VAL": 0, "SELL_VOL": 5, "SELL_VAL": 0})
    f = tmp_path / "phut.csv"
    pd.DataFrame(dong).to_csv(f, index=False)
    cfg = {"du_lieu": {"n_V": 60, "eps": 1e-6, "bo_qua_dau": 61, "nghi_dai_toi_thieu": 3, "xu_ly_dao_han": "loai"}}
    d = DL.xay_ngay(f, cfg)
    assert d["tau"].iloc[-1] == len(pd.bdate_range("2024-07-10", "2024-07-18")) - 1


def test_ky_quy_hai_phia_la_phan_vi_cua_tri_tuyet_doi():
    from de_cuong import ky_quy as KQ
    R = np.random.default_rng(0).normal(size=200_000)
    a = 0.01
    m_dung, m_cu = KQ._phan_vi_ls(R, a, True), KQ._phan_vi_ls(R, a, False)
    assert abs(np.mean(np.abs(R) > m_dung) - a) < 0.001            # P(|R| > m) = alpha
    assert abs(np.mean(np.abs(R) > m_cu) - 2 * a) < 0.002          # cách max hai phía: xấp xỉ 2 alpha


def test_sbc_tan_so_ky_vong_theo_so_gia_tri_moi_ngan():
    # thứ hạng đều trên 0..100 (101 giá trị, 10 ngăn): ngăn có 11 giá trị phải có tần số kỳ vọng lớn hơn
    L, n = 100, 101 * 200
    H = np.tile(np.arange(L + 1), n // (L + 1))[:, None]
    kq = MP.tong_hop_sbc([{"hang": h, "L": L, "ten": ["x"]} for h in H])
    assert kq["tham_so"]["x"]["chi2_p"] > 0.999                    # dữ liệu đều hoàn hảo không bị bác bỏ
    assert np.allclose(kq["tham_so"]["x"]["dem"], kq["tham_so"]["x"]["ky_vong"])


def test_bam_gom_chung_va_tep_phu():
    import quy_trinh.b03b_so_lan_lap  # noqa: F401  (đăng ký b03a, b03b và tệp vá)
    import quy_trinh.b06c_mo_phong_chon  # noqa: F401
    from de_cuong import khung as K
    t = [Path(p).name for p in K._tep_ma(K.DANH_MUC["b06c"])]
    assert {"_chung.py", "b04_b07_giai_doan_1.py", "mo_phong.py"} <= set(t)
    t = [Path(p).name for p in K._tep_ma(K.DANH_MUC["b03b"])]
    assert {"b03b_so_lan_lap.py", "b03a_ged.py"} <= set(t)


def test_khoi_tao_nuts_phan_tan_theo_sai_so_chuan():
    import jax.numpy as jnp
    from de_cuong import uoc_luong as U
    from de_cuong.mo_hinh_jax import DacTa
    rng = np.random.default_rng(2)
    spec = DacTa("student", ())
    psi0 = np.array([0.05, 0.04, 0.15, -0.1, 0.95, 0.0, 0.0, 0.0, 8.0])
    n = 1500
    Z, J = np.zeros((n, 3)), np.zeros(n, bool)
    r, _ = U.sinh_tu_mo_hinh(spec, psi0, Z, J, 0.8, rng)
    dl = (jnp.asarray(r), jnp.asarray(J), jnp.asarray(Z), jnp.asarray(~J), jnp.asarray(0.8))
    m = U.mle(spec, dl, 4, 0.3, rng)
    X = np.array([U.khoi_tao_phan_tan(spec, m, dl, np.random.default_rng(k)) for k in range(200)])
    sd_beta_xi = X[:, 4].std()
    assert sd_beta_xi > 0.1                                   # nhiễu cũ là 0,05 cho mọi tham số
    assert np.isfinite(X).all()


def test_nhom_cang_thang_dung_lich_su_truoc_khoi():
    import pandas as pd
    import quy_trinh.b08_b09_ngoai_mau as B8
    n_truoc, n = 300, 100
    rho = np.ones(n_truoc + n)
    rho[n_truoc + 5] = 50.0                                   # phiên thứ 6 của khối: căng thẳng rõ rệt
    d = pd.DataFrame({"tau": np.full(n_truoc + n, 10.0), "J": np.zeros(n_truoc + n, bool),
                      "D_nghi": np.zeros(n_truoc + n, bool)})
    nm = B8.nhom_phien(d, rho, n_truoc, n_truoc + n, lech=n_truoc)
    assert nm["cang_thang"][5] and nm["cang_thang"].sum() == 1


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
