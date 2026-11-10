"""Kiểm thử de_cuong/chan_doan.py (chuỗi NUTS kẹt, phân loại chẩn đoán) và bảo đảm mô-đun này không lọt vào mã băm
của b03b (sửa nó không được làm mô phỏng b03b chạy lại).

Chạy: python -m pytest kiem_thu -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import chan_doan as CD     # noqa: E402


def _mo_hinh_nho():
    """EGARCH chuẩn không biến ngoại sinh, 400 phiên mô phỏng, và MLE trên đó."""
    import jax.numpy as jnp
    from de_cuong import uoc_luong as U
    from de_cuong.mo_hinh_jax import DacTa
    spec = DacTa("chuan", ())
    rng = np.random.default_rng(3)
    n = 400
    Z, J = np.zeros((n, 3)), np.zeros(n, bool)
    r, _ = U.sinh_tu_mo_hinh(spec, np.array([0.05, 0.04, 0.15, -0.1, 0.9, 0, 0, 0]), Z, J, 0.4, rng)
    dl = (jnp.asarray(r), jnp.asarray(J), jnp.asarray(Z), jnp.asarray(~J), jnp.asarray(float(np.log(r[:20].var()))))
    return spec, dl, U.mle(spec, dl, 2, 0.2, rng)


def test_nguong():
    assert abs(CD.nguong_thieu_hut(11, 5.0) - (5.5 + 5 * np.sqrt(5.5))) < 1e-12
    assert CD.nguong_thieu_hut(8, 0.0) == 4.0


def test_kiem_tra_chuoi_phan_biet_chuoi_tot_va_chuoi_ket():
    """Mẫu sát MLE: không kẹt. Mẫu lệch xa (beta rất nhỏ, omega lớn): kẹt. Mẫu có NaN: coi là kẹt."""
    spec, dl, m = _mo_hinh_nho()
    rng = np.random.default_rng(0)
    tot = m["xi"] + rng.normal(0, 0.01, (200, spec.d))
    kt = CD.kiem_tra_chuoi(spec, tot, m["xi"], dl)
    assert not kt["ket"] and kt["thieu_hut"] < kt["nguong"]
    xa = tot.copy()
    xa[:, 4] = -3.0                       # beta = sigmoid(-3) ~ 0,05 so với 0,9
    xa[:, 1] = 2.0
    kt = CD.kiem_tra_chuoi(spec, xa, m["xi"], dl)
    assert kt["ket"] and kt["thieu_hut"] > 10 * kt["nguong"]
    nan = tot.copy()
    nan[0, 0] = np.nan
    assert CD.kiem_tra_chuoi(spec, nan, m["xi"], dl)["ket"]


def test_chon_chuoi_va_phan_loai():
    ds = [{"chuoi_ket": {"ket": False}}, {"chuoi_ket": {"ket": True}}, {}, {"chuoi_ket": {"ket": False}}]
    giu, loai = CD.chon_chuoi(ds)
    assert loai == [1] and len(giu) == 3
    assert CD.phan_loai({"rhat_max": 1.003, "ess_min": 1500, "phan_ky": 2}, 1.01, 400) == \
        {"hoi_tu_dat": True, "phan_ky": 2, "chi_vi_phan_ky": True}
    assert CD.phan_loai({"rhat_max": 1.59, "ess_min": 7, "phan_ky": 0}, 1.01, 400)["hoi_tu_dat"] is False


def test_chan_doan_khong_vao_ma_bam_b03b():
    """b03b (và các bước theo MLE) không khai báo chan_doan hay [chuoi_ket]; b08, b08b có khai báo."""
    import quy_trinh.b03_mo_phong  # noqa: F401  (đăng ký các bước)
    import quy_trinh.b03b_so_lan_lap  # noqa: F401
    import quy_trinh.b04_b07_giai_doan_1  # noqa: F401
    import quy_trinh.b08_b09_ngoai_mau  # noqa: F401
    from de_cuong import khung as K
    for ma in ("b03a", "b03b", "b04", "b05", "b06", "b07"):
        b = K.DANH_MUC[ma]
        assert "chan_doan" not in b.thu_vien and "chuoi_ket" not in b.cau_hinh, ma
        assert not any(Path(t).name == "chan_doan.py" for t in K._tep_ma(b)), ma
    for ma in ("b08", "b08b"):
        assert "chan_doan" in K.DANH_MUC[ma].thu_vien and "chuoi_ket" in K.DANH_MUC[ma].cau_hinh
