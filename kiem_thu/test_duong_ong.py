"""Kiểm thử đường ống: phân phối (Phụ lục A, C.5), khớp số JAX - numba, khóa khối KT (C.7), điểm kiểm tra.

Chạy: python -m pytest kiem_thu -q
"""
from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

GOC = Path(__file__).resolve().parents[1]
for p in (GOC / "ma_nguon", GOC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from de_cuong import khung as K          # noqa: E402
from de_cuong import phan_phoi as P      # noqa: E402

BANG_11 = {("chuan", (), "mua"): [2.326, 2.665, 1.960, 2.338, 1.645, 2.063],
           ("student", (5.0,), "mua"): [2.606, 3.449, 1.991, 2.728, 1.561, 2.239],
           ("skewt", (5.0, -0.1), "mua"): [2.783, 3.720, 2.102, 2.919, 1.627, 2.377],
           ("skewt", (5.0, -0.1), "ban"): [2.416, 3.159, 1.871, 2.522, 1.488, 2.089]}


@pytest.mark.parametrize("khoa", list(BANG_11))
def test_bang_11(khoa):
    pp, sh, ben = khoa
    tinh = []
    for a in (0.01, 0.025, 0.05):
        if ben == "mua":
            tinh += [-float(P.phan_vi(a, pp, sh)), -P.ky_vong_duoi_trai(a, pp, sh)]
        else:
            tinh += [float(P.phan_vi(1 - a, pp, sh)), P.ky_vong_duoi_phai(a, pp, sh)]
    assert np.max(np.abs(np.array(tinh) - BANG_11[khoa])) < 0.0015


@pytest.mark.parametrize("pp,sh", [("chuan", []), ("student", [4.5]), ("skewt", [4.5, -0.15]), ("ged", [1.3])])
def test_e_abs_jax_khop_cau_phuong(pp, sh):
    import jax.numpy as jnp
    from de_cuong import mo_hinh_jax as M
    assert abs(float(M.e_abs(pp, jnp.array(sh))) - P.e_abs(pp, sh)) < 1e-8


def test_gioi_han():
    import jax.numpy as jnp
    from de_cuong import mo_hinh_jax as M
    assert abs(float(M.e_abs("student", jnp.array([1e7]))) - math.sqrt(2 / math.pi)) < 1e-6
    z = jnp.linspace(-4, 4, 21)
    assert float(jnp.max(jnp.abs(M.log_mat_do(z, "skewt", jnp.array([5.0, 0.0]))
                                 - M.log_mat_do(z, "student", jnp.array([5.0]))))) < 1e-10


def test_tien_nghiem_on_dinh_tai_bien():
    """beta = sigmoid(40) làm tròn thành 1 nhưng log tiên nghiệm theo xi vẫn hữu hạn."""
    import jax.numpy as jnp
    from de_cuong import mo_hinh_jax as M
    xi = jnp.array([0.0, 0.0, 0.1, -0.1, 40.0, 0.0, 0.0, 0.0, 1.0, 30.0])
    psi = M.g(xi, "skewt", 0)
    assert math.isfinite(float(M.log_tien_nghiem(psi, "skewt", 0, xi)))


# ---------------------------------------------------------------------------------------
@pytest.fixture()
def goc_tam(tmp_path, monkeypatch):
    shutil.copytree(GOC / "cau_hinh", tmp_path / "cau_hinh")
    monkeypatch.setattr(K, "GOC", tmp_path)
    return tmp_path


def test_khoa_KT_cat_du_lieu():
    ctx = K.NguCanh("day_du", K.Buoc("t_cat", "cat", "-", __file__, _lam, _th, song_song=False))
    d = ctx.du_lieu()
    assert len(d) == ctx.khoi()["KT"][0]           # không có quan sát nào của KT


def test_buoc_can_khoa_tu_choi(goc_tam):
    """Bước cần khóa cấu hình không chạy khi chưa có KHOA_CAU_HINH.json (Phụ lục C.7)."""
    DEM["n"] = 0
    K.Buoc("t_kt", "can_khoa", "-", __file__, _lam, _th, can_khoa_KT=True, song_song=False)
    assert K.chay_buoc("t_kt", "day_du") == "cho_buoc_truoc" and DEM["n"] == 0


def test_phan_manh_roi_nhau_va_du():
    """Mỗi nhóm đơn vị có đúng một chủ; nhóm NUTS chỉ giao cho máy trong may_nuts; mọi máy đều có việc."""
    cfg = K.doc_cau_hinh("day_du")
    nhom = [f"mle|skewt|G_{i}" for i in range(400)] + [f"nuts|{k:03d}|c{c}" for k in range(68) for c in range(4)]
    chu = {n: K.chu_so_huu(n, cfg) for n in nhom}
    assert all(chu[n] in cfg["phan_tan"]["may"] for n in nhom)
    assert all(chu[n] in cfg["phan_tan"]["may_nuts"] for n in nhom if K.la_nhom_nuts(n))
    assert set(chu[n] for n in nhom if not K.la_nhom_nuts(n)) == set(cfg["phan_tan"]["may"])
    assert chu == {n: K.chu_so_huu(n, cfg) for n in nhom}          # tất định


def test_nuts_chay_tiep_sau_khi_dung(tmp_path, monkeypatch):
    """Chuỗi NUTS dừng sau khởi động (hết giờ) rồi chạy tiếp ở lần gọi sau, cho đủ số mẫu."""
    import jax.numpy as jnp
    from de_cuong import uoc_luong as U
    from de_cuong.mo_hinh_jax import DacTa
    spec = DacTa("chuan", ())
    rng = np.random.default_rng(3)
    n = 400
    Z, J = np.zeros((n, 3)), np.zeros(n, bool)
    r, _ = U.sinh_tu_mo_hinh(spec, np.array([0.05, 0.04, 0.15, -0.1, 0.9, 0, 0, 0]), Z, J, 0.4, rng)
    dl = (jnp.asarray(r), jnp.asarray(J), jnp.asarray(Z), jnp.asarray(~J), jnp.asarray(float(np.log(r[:20].var()))))
    m = U.mle(spec, dl, 2, 0.2, rng)
    monkeypatch.setenv("NCKHSV_HAN", "1")
    with pytest.raises(K.HetGio):
        U.nuts_chuoi(spec, dl, m["xi"], 20, 30, 0.8, 4, 1, thu_muc=tmp_path, vong_moi_doan=10)
    assert (tmp_path / "khoi_dong.pkl").exists() and not (tmp_path / "doan_000.pkl").exists()
    monkeypatch.delenv("NCKHSV_HAN")
    c = U.nuts_chuoi(spec, dl, m["xi"], 20, 30, 0.8, 4, 1, thu_muc=tmp_path, vong_moi_doan=10)
    assert c["psi_mau"].shape == (30, spec.d) and c["so_doan"] == 3


DEM = {"n": 0}


def _lam(ctx, khoa, truoc):
    DEM["n"] += 1
    k = int(khoa.split("::")[1])
    return {"k": k, "x": (0 if truoc is None else truoc["x"]) + k}


def _th(ctx, kq):
    ctx.ghi("tong.json", {k: v["x"] for k, v in sorted(kq.items())})


def test_diem_kiem_tra_tiep_tuc(goc_tam):
    K.Buoc("t_dk", "diem_kiem", "-", __file__, _lam, _th,
           don_vi=lambda ctx: [f"a::{i}" for i in range(5)] + [f"b::{i}" for i in range(3)], song_song=False)
    DEM["n"] = 0
    assert K.chay_buoc("t_dk", "day_du") == "xong" and DEM["n"] == 8
    ctx = K.NguCanh("day_du", K.DANH_MUC["t_dk"])
    truoc = (ctx.thu_muc() / "tong.json").read_text(encoding="utf-8")
    hs = K.doc_ho_so(ctx, "t_dk")
    (ctx.ket_qua / hs["thu_muc_diem_kiem_tra"] / "a_3.pkl").unlink()   # giả lập dừng giữa chừng
    DEM["n"] = 0
    K.chay_buoc("t_dk", "day_du", chay_lai=True)
    assert DEM["n"] == 1                                              # chỉ tính lại đơn vị bị mất
    assert (ctx.thu_muc() / "tong.json").read_text(encoding="utf-8") == truoc
    DEM["n"] = 0
    assert K.chay_buoc("t_dk", "day_du") == "bo_qua" and DEM["n"] == 0


def test_mo_rong_don_vi_chi_chay_phan_thieu(goc_tam):
    """Bước mo_rong_don_vi (b03b): tăng số lần lặp thì chỉ chạy các đơn vị mới; giữ nguyên số thì bỏ qua."""
    import json
    SO = {"B": 3}
    K.Buoc("t_mr", "mo_rong", "-", __file__, _lam, _th, don_vi=lambda ctx: [f"a::{i}" for i in range(SO["B"])],
           song_song=False, mo_rong_don_vi=True)
    DEM["n"] = 0
    assert K.chay_buoc("t_mr", "day_du") == "xong" and DEM["n"] == 3
    SO["B"] = 5
    DEM["n"] = 0
    assert K.chay_buoc("t_mr", "day_du") == "xong" and DEM["n"] == 2       # chỉ a::3, a::4
    ctx = K.NguCanh("day_du", K.DANH_MUC["t_mr"])
    assert len(json.loads((ctx.thu_muc() / "tong.json").read_text(encoding="utf-8"))) == 5
    DEM["n"] = 0
    assert K.chay_buoc("t_mr", "day_du") == "bo_qua" and DEM["n"] == 0


def test_xu_ly_dao_han_bang_f2m():
    """Phiên chuyển: r_t = 100 ln(F1M_t / F2M_{t-1}); chỉ khuyết khi thiếu F2M; tổn thất H phiên đi qua phiên chuyển."""
    from de_cuong import du_lieu as DL
    from de_cuong import ky_quy as KQ
    cfg = K.doc_cau_hinh("day_du")
    if cfg["du_lieu"].get("xu_ly_dao_han") != "f2m":
        pytest.skip("cau hinh khong dung F2M")
    d = DL.xay_ngay(K.GOC / cfg["chung"]["du_lieu_goc"], cfg)
    f2 = DL.doc_f2m(K.GOC / cfg["du_lieu"]["du_lieu_f2m"])
    J = d["J"].to_numpy()
    idx = np.flatnonzero(J & ~d["khuyet"].to_numpy())
    assert len(idx) >= 100 and d["khuyet"].sum() <= 3
    for t in idx[:5]:
        p2 = f2[d["date"].iloc[t - 1]]
        assert math.isclose(d["r"].iloc[t], 100 * math.log(d["close"].iloc[t] / p2), rel_tol=1e-12)
    assert np.nanstd(d["r"].to_numpy()[J]) < 2 * np.nanstd(d["r"].to_numpy()[~J])   # không còn cú nhảy cơ sở
    R, ok = KQ.tong_that_thuc(d, 3)
    t = idx[3] - 1                                                                  # cửa sổ (t, t+2] chứa phiên chuyển
    assert ok[t, 1] and math.isclose(R[t, 1], math.expm1((d["r"].iloc[t + 1] + d["r"].iloc[t + 2]) / 100))


def test_b03b_so_lan_lap_rieng():
    """b03b: VB dùng B_vb, NUTS dùng B_nuts; [mo_phong_bayes] không nằm trong mã băm của b03a, b03b."""
    import importlib
    importlib.import_module("quy_trinh.b03_mo_phong")
    from quy_trinh import b03b_so_lan_lap as BL
    b = K.DANH_MUC["b03b"]
    assert b.mo_rong_don_vi and "mo_phong_bayes" not in b.cau_hinh and "mo_phong_bayes" not in K.DANH_MUC["b03a"].cau_hinh
    cfg = K.doc_cau_hinh("day_du")
    assert BL.so_lan_lap(cfg, "vb_mf") == cfg["mo_phong_bayes"]["B_vb"]
    assert BL.so_lan_lap(cfg, "nuts") == cfg["mo_phong_bayes"]["B_nuts"]
    ds = b.don_vi(K.NguCanh("day_du", b))
    n_nuts = sum(1 for k in ds if k.startswith("mpb|skewt|nuts|E1|"))
    n_vb = sum(1 for k in ds if k.startswith("mpb|skewt|vb_mf|E1|"))
    assert n_nuts == cfg["mo_phong_bayes"]["B_nuts"] * cfg["bayes"]["nuts_so_chuoi"]
    assert n_vb == cfg["mo_phong_bayes"]["B_vb"] * cfg["bayes"]["vb_so_lan_khoi_tao"]


def test_b03a_them_ged():
    """b03a: GED được thêm vào mô phỏng khôi phục khi GED tham gia chọn (BM02); đơn vị cũ giữ nguyên tên, tệp băm của
    b03a không đổi, và b03a được đánh dấu mo_rong_don_vi để chỉ chạy phần GED còn thiếu."""
    import importlib
    B3 = importlib.import_module("quy_trinh.b03_mo_phong")
    tep_cu = K.DANH_MUC["b03a"].tep
    importlib.import_module("quy_trinh.b03a_ged")
    b = K.DANH_MUC["b03a"]
    cfg = K.doc_cau_hinh("day_du")
    ctx = K.NguCanh("day_du", b)
    ds = b.don_vi(ctx)
    cu = B3._dv_b03a(ctx)
    assert b.mo_rong_don_vi and b.tep == tep_cu
    assert set(cu) <= set(ds)
    them = [k for k in ds if k.startswith("mp|ged|mle|")]
    assert len(them) == 4 * cfg["mo_phong"]["B_mle"] and len(ds) == len(cu) + len(them)
    # b03b cũng có GED (VB, NUTS, SBC), cùng số đơn vị mỗi phân phối, tệp băm không đổi
    importlib.import_module("quy_trinh.b03b_so_lan_lap")
    bb = K.DANH_MUC["b03b"]
    db = bb.don_vi(K.NguCanh("day_du", bb))
    n_pp = {pp: sum(1 for k in db if k.split("|")[1] == pp) for pp in ("chuan", "ged")}
    assert n_pp["ged"] == n_pp["chuan"] > 0 and bb.tep == B3.TEP and bb.mo_rong_don_vi


def test_quy_tac_vung_khong_dieu_chinh():
    """Ngưỡng 0 trùng năm bước; ngưỡng dương giảm số lần đổi mức, giữ mức sàn, và tăng/giảm đúng bất đối xứng."""
    import numpy as np
    from de_cuong import ky_quy as KQ
    rng = np.random.default_rng(1)
    v = 0.08 + 0.03 * np.abs(rng.standard_normal(300))
    e, san = 1.2 * v, np.full(300, 0.05)
    m0 = KQ.quy_tac(v, e, san, 0.5, 0.1, 0.5, 0.8)
    assert np.allclose(m0, KQ.quy_tac(v, e, san, 0.5, 0.1, 0.5, 0.8, ng_tang=0.0, ng_giam=0.0))
    m1 = KQ.quy_tac(v, e, san, 0.5, 0.1, 0.5, 0.8, ng_tang=0.01, ng_giam=0.02)
    assert (np.abs(np.diff(m1)) > 1e-12).sum() < (np.abs(np.diff(m0)) > 1e-12).sum()
    assert (m1 >= 0.5 * san - 1e-12).all()
