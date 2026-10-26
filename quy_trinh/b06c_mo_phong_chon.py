"""b06b: ước lượng cấu hình được chọn trên UL + XT; b06c: mô phỏng khôi phục tham số quanh chính ước lượng đó (26/10/2026).

LÝ THUYẾT: Điều kiện (i) của cổng (mục 8.1) dựa vào mô phỏng khôi phục b03a, vốn sinh dữ liệu theo thiết kế cố định của
đề cương (hai biến X giả AR(1), beta_0 = 0,95, kappa_0 = 0,15). Mô hình được chọn trên dữ liệu thật có thể nằm ở vùng tham
số khác hẳn (lần chạy 25/10: một biến RV, beta = 0,70, kappa = 0,06), nên b03a không chứng minh được độ tin cậy của khoảng
tin cậy cho chính mô hình sẽ báo cáo. b06c bổ sung bootstrap tham số: lặp B lần, sinh lợi suất từ mô hình được chọn với
psi = ước lượng b06b, dùng đúng chuỗi biến ngoại sinh, biến giả lịch và cờ phiên khuyết của UL + XT; ước lượng lại bằng MLE;
ghi độ phủ của khoảng Wald 95% (sai số chuẩn bánh kẹp và Hessian), độ chệch, RMSE. Điều kiện của b06c dùng cùng kiểm định
nhị thức với b03a (mo_phong.dieu_kien_cong) trên omega, kappa, theta, beta và các gamma.
Giới hạn: chuỗi X được cố định theo dữ liệu thật, nên phản hồi của X (như RV) với phương sai mô phỏng không được tái tạo.
b06b cũng là ước lượng mà b07 dùng cho điều kiện (ii) (trước đây b07 tự ước lượng lại).
THƯ VIỆN: numpy, jax (qua uoc_luong).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import quy_trinh._chung as C  # noqa: F401
import quy_trinh.b04_b07_giai_doan_1 as B4
from de_cuong import du_lieu as DL
from de_cuong import mo_phong as MP
from de_cuong import uoc_luong as U
from de_cuong.khung import Buoc
from de_cuong.mo_hinh_jax import DacTa, g_nghich

TEP = __file__
LOI = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


# ---------------------------------------------------------------------------------------- b06b
def _lam_b06b(ctx, khoa, truoc):
    cb = ctx.doc("chon_bien.json", "b05")
    cp = ctx.doc("chon_phan_phoi.json", "b06")
    pp, bien = cp["phan_phoi"], tuple(cb["chon"])
    x = B4.uoc_luong_day_du(ctx, pp, bien, ctx.khoi()["XT"][1], ("b06b", pp), bootstrap=False)
    return {**x, "phan_phoi": pp, "bien": list(bien)}


def _th_b06b(ctx, kq):
    ctx.ghi("uoc_luong_chon.json", kq["tat_ca::0"])


Buoc("b06b", "uoc_luong_chon", "8.1", TEP, _lam_b06b, _th_b06b, phu_thuoc=("b05", "b06"),
     cau_hinh=("du_lieu", "khoi", "mo_hinh", "mle"), thu_vien=B4.TV, song_song=False, chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b06c
def _dv_b06c(ctx):
    return [f"mp|{b:04d}::0" for b in range(ctx.cfg["mo_phong_chon"]["B"])]


def _lam_b06c(ctx, khoa, truoc):
    b = int(khoa.split("::")[0].split("|")[1])
    x = ctx.doc("uoc_luong_chon.json", "b06b")
    pp, bien, psi = x["phan_phoi"], tuple(x["bien"]), np.asarray(x["psi"], float)
    spec = DacTa(pp, bien)
    xt1 = ctx.khoi()["XT"][1]
    dl = U.chuan_bi(DL.dac_trung(ctx.du_lieu(), xt1 - 1), spec.bien, xt1)
    nw = xt1 - DL.BO_QUA                                       # chỉ cửa sổ UL + XT
    Z, J = np.asarray(dl[2])[:nw], np.asarray(dl[1])[:nw]
    rng = ctx.rng("mp", b)
    try:
        r, _ = U.sinh_tu_mo_hinh(spec, psi, Z, J, float(dl[4]), rng)
        dlb = (jnp.asarray(r), jnp.asarray(J), jnp.asarray(Z), jnp.asarray(~J), dl[4])
        cm = ctx.cfg["mle"]
        m = U.mle(spec, dlb, cm["so_diem_khoi_tao"], cm["do_tan_khoi_tao"], rng, xi_am=g_nghich(psi, pp, spec.nx))
        se = U.sai_so_chuan(spec, m["psi"], dlb)
        return {"uoc": m["psi"], "se_h": se["se_hessian"], "se_s": se["se_banh_kep"]}
    except LOI as e:
        ctx.log(f"don vi {khoa} THAT BAI: {e}")
        return {"loi": str(e)}


def _th_b06c(ctx, kq):
    x = ctx.doc("uoc_luong_chon.json", "b06b")
    ten, psi = x["ten"], np.asarray(x["psi"], float)
    ok = [v for v in kq.values() if "loi" not in v]
    U_ = np.array([v["uoc"] for v in ok])
    SS, SH = np.array([v["se_s"] for v in ok]), np.array([v["se_h"] for v in ok])
    B = len(ok)
    bang = {}
    for i, t in enumerate(ten):
        bang[t] = {"that": float(psi[i]), "chech": float((U_[:, i] - psi[i]).mean()),
                   "rmse": float(np.sqrt(((U_[:, i] - psi[i]) ** 2).mean())),
                   "do_phu": float(np.mean(np.abs(U_[:, i] - psi[i]) <= 1.959964 * SS[:, i])),
                   "do_phu_hessian": float(np.mean(np.abs(U_[:, i] - psi[i]) <= 1.959964 * SH[:, i]))}
    xet = [t for t in ten if t in ("omega", "kappa", "theta", "beta") or t.startswith("gamma_")]
    dk = MP.dieu_kien_cong({t: (bang[t]["do_phu"], B) for t in xet}, {}, ctx.cfg["cong"])
    dk["so_that_bai"] = len(kq) - B
    ctx.ghi("mo_phong_cau_hinh_chon.json", {"phan_phoi": x["phan_phoi"], "bien": x["bien"], "B": B,
                                            "T": ctx.khoi()["XT"][1] - DL.BO_QUA, "tham_so_xet_cong": xet,
                                            "bang": bang, "dieu_kien": dk,
                                            "loi_that_bai": [v["loi"] for v in kq.values() if "loi" in v]})


Buoc("b06c", "mo_phong_cau_hinh_chon", "8.1 (bổ sung 26/10/2026)", TEP, _lam_b06c, _th_b06c, don_vi=_dv_b06c,
     phu_thuoc=("b06b",), cau_hinh=("du_lieu", "khoi", "mo_hinh", "mle", "mo_phong_chon", "cong"), thu_vien=B4.TV)
