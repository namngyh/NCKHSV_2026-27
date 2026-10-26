"""b03a: mô phỏng khôi phục tham số bằng MLE; b03b: mô phỏng khôi phục bằng VB và NUTS, và SBC (mục 4.4, Bảng 6).

Tách theo tiến độ của đề cương (mục 8.2): mô phỏng bằng MLE trước (điều kiện (i) của mục 8.1 dựa vào đây), suy diễn
Bayes và hiệu chuẩn dựa trên mô phỏng sau. b03b không chặn đường ống; báo cáo cuối (b13) được làm lại khi b03b xong.
Đơn vị điểm kiểm tra:
  b03a: một lần lặp MLE (phân phối, kịch bản, b).
  b03b: một chuỗi NUTS hoặc một lần khởi tạo VB của một lần lặp; một chuỗi/lần khởi tạo của một lần SBC.
Hạt giống dữ liệu của lần lặp (phân phối, kịch bản, b) dùng chung giữa b03a và b03b: MLE và Bayes được đánh giá trên
cùng các bộ dữ liệu.
THƯ VIỆN: numpy.
"""
from __future__ import annotations

import numpy as np

import quy_trinh._chung as C  # noqa: F401
from de_cuong import mo_phong as MP
from de_cuong.khung import Buoc, bam

TEP = __file__
THAM_SO_CONG = ("omega", "kappa", "theta", "beta", "gamma_X1", "gamma_X2")
LOI = (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError)


def _hat_giong(ctx, *k) -> int:
    """Hạt giống không phụ thuộc bước: cùng (loại, phân phối, kịch bản, b) cho cùng hạt giống ở b03a và b03b."""
    return int(bam([ctx.cfg["chung"]["hat_giong"], "mo_phong", *k])[:8], 16)


# ---------------------------------------------------------------------------------------- b03a
def _dv_b03a(ctx):
    c = ctx.cfg
    return [f"mp|{pp}|mle|{kb}|{b:03d}::0" for pp in c["mo_hinh"]["phan_phoi"] for kb in MP.KICH_BAN
            for b in range(c["mo_phong"]["B_mle"])]


def _lam_b03a(ctx, khoa, truoc):
    _, pp, _, kb, b = khoa.split("::")[0].split("|")
    try:
        spec, psi0, dl = MP.du_lieu_lan(kb, pp, ctx.cfg, _hat_giong(ctx, "du_lieu", pp, kb, b))
        return MP.don_vi_mle(spec, psi0, dl, ctx.cfg, _hat_giong(ctx, "mle", pp, kb, b))
    except LOI as e:
        ctx.log(f"don vi {khoa} THAT BAI: {e}")
        return {"loi": str(e)}


def _th_b03a(ctx, kq):
    nhom, that_bai = {}, {}
    for k, v in kq.items():
        _, pp, _, kb, _ = k.split("::")[0].split("|")
        if "loi" in v:
            that_bai.setdefault(f"{pp}|{kb}", []).append(v["loi"])
        else:
            nhom.setdefault((pp, kb), []).append(v)
    bang = {f"{pp}|mle|{kb}": MP.tong_hop(v, "mle") for (pp, kb), v in nhom.items()}
    cg = ctx.cfg["cong"]
    dieu_kien = {}
    for pp in ctx.cfg["mo_hinh"]["phan_phoi"]:
        phu = [bang[f"{pp}|mle|{kb}"]["tham_so"][t]["do_phu"] for kb in MP.KICH_BAN for t in THAM_SO_CONG
               if f"{pp}|mle|{kb}" in bang]
        bb = ([bang[f"{pp}|mle|E2"]["bac_bo_theta0"], bang[f"{pp}|mle|E4"]["bac_bo_gamma0"]]
              if f"{pp}|mle|E2" in bang and f"{pp}|mle|E4" in bang else [])
        o_phu = {f"{kb}|{t}": (bang[f"{pp}|mle|{kb}"]["tham_so"][t]["do_phu"], bang[f"{pp}|mle|{kb}"]["B"])
                 for kb in MP.KICH_BAN for t in THAM_SO_CONG if f"{pp}|mle|{kb}" in bang}
        o_bb = ({"E2|theta": (bb[0], bang[f"{pp}|mle|E2"]["B"]), "E4|gamma": (bb[1], bang[f"{pp}|mle|E4"]["B"])}
                if bb else {})
        dieu_kien[pp] = {**MP.dieu_kien_cong(o_phu, o_bb, cg),
                         "bac_bo_sai_E2_theta": bb[0] if bb else None, "bac_bo_sai_E4_gamma": bb[1] if bb else None,
                         "so_that_bai": sum(len(v) for k, v in that_bai.items() if k.startswith(pp + "|"))}
        dieu_kien[pp]["dat"] = bool(phu and bb and dieu_kien[pp]["dat"])
    ctx.ghi("mo_phong_khoi_phuc.json", {"B": ctx.cfg["mo_phong"]["B_mle"], "T": ctx.cfg["mo_phong"]["T"],
                                        "kich_ban": {k: list(v) for k, v in MP.KICH_BAN.items()}, "bang": bang,
                                        "dieu_kien_8_1_i": dieu_kien, "tham_so_xet_cong": THAM_SO_CONG,
                                        "so_lan_that_bai": {k: len(v) for k, v in that_bai.items()},
                                        "loi_that_bai": that_bai})


Buoc("b03a", "mo_phong_mle", "4.4, Bảng 6 (MLE)", TEP, _lam_b03a, _th_b03a, don_vi=_dv_b03a, phu_thuoc=("b02",),
     cau_hinh=("mo_phong", "mle", "mo_hinh", "cong"), thu_vien=("mo_phong", "uoc_luong", "mo_hinh_jax", "phan_phoi"))


# ---------------------------------------------------------------------------------------- b03b
def _hau_to(ctx, pt):
    b = ctx.cfg["bayes"]
    return [f"c{c}" for c in range(b["nuts_so_chuoi"])] if pt == "nuts" else \
        [f"l{i}" for i in range(b["vb_so_lan_khoi_tao"])]


def _dv_b03b(ctx):
    c = ctx.cfg
    ds = []
    for pp in c["mo_hinh"]["phan_phoi"]:
        for pt in c["bayes"]["phuong_phap"]:
            ds += [f"mpb|{pp}|{pt}|{kb}|{b:03d}|{h}::0" for kb in MP.KICH_BAN for b in range(c["mo_phong"]["B"])
                   for h in _hau_to(ctx, pt)]
            ds += [f"sbc|{pp}|{pt}|{l:03d}|{h}::0" for l in range(c["mo_phong"]["sbc_so_lan"]) for h in _hau_to(ctx, pt)]
    return ds


def _lam_b03b(ctx, khoa, truoc):
    p = khoa.split("::")[0].split("|")
    loai, pp, pt, h = p[0], p[1], p[2], p[-1]
    chi_so = int(h[1:])
    try:
        if loai == "mpb":
            kb, b = p[3], p[4]
            spec, psi0, dl = MP.du_lieu_lan(kb, pp, ctx.cfg, _hat_giong(ctx, "du_lieu", pp, kb, b))
            s_kt = _hat_giong(ctx, "mle", pp, kb, b)
        else:
            l = p[3]
            spec, psi0, dl = MP.du_lieu_sbc(pp, ctx.cfg, _hat_giong(ctx, "sbc", pp, l))
            s_kt = _hat_giong(ctx, "sbc_mle", pp, l)
        return MP.don_vi_bayes(spec, psi0, dl, ctx.cfg, pt, s_kt, ctx.hat_giong(khoa), chi_so, ctx.thu_muc_doan(khoa))
    except LOI as e:
        ctx.log(f"don vi {khoa} THAT BAI: {e}")
        return {"loi": str(e)}


def _th_b03b(ctx, kq):
    lan, that_bai = {}, {}
    for k, v in kq.items():
        p = k.split("::")[0].split("|")
        khoa_lan = "|".join(p[:-1])
        if "loi" in v:
            that_bai.setdefault(khoa_lan, []).append(v["loi"])
        else:
            lan.setdefault(khoa_lan, []).append(v)
    so_can = {pt: len(_hau_to(ctx, pt)) for pt in ctx.cfg["bayes"]["phuong_phap"]}
    mp_nhom, sbc_nhom, thieu = {}, {}, []
    for khoa_lan, ds in lan.items():
        p = khoa_lan.split("|")
        pt = p[2]
        if pt == "nuts" and len(ds) < so_can[pt]:          # thiếu chuỗi do đơn vị thất bại
            thieu.append(khoa_lan)
            continue
        g = MP.gop_bayes(ds, pt, ctx.cfg)
        if p[0] == "mpb":
            mp_nhom.setdefault((p[1], pt, p[3]), []).append(g)
        else:
            sbc_nhom.setdefault((p[1], pt), []).append(MP.hang_sbc(g))
    ctx.ghi("mo_phong_bayes.json", {"bang": {f"{pp}|{pt}|{kb}": MP.tong_hop(v, pt) for (pp, pt, kb), v in mp_nhom.items()},
                                    "lan_lap_thieu_chuoi": thieu, "so_don_vi_that_bai": {k: len(v) for k, v in that_bai.items()},
                                    "loi_that_bai": that_bai})
    ctx.ghi("sbc.json", {f"{pp}|{pt}": MP.tong_hop_sbc(v) for (pp, pt), v in sbc_nhom.items()})


Buoc("b03b", "mo_phong_bayes_sbc", "4.4, 4.3.3 (Bayes, SBC)", TEP, _lam_b03b, _th_b03b, don_vi=_dv_b03b,
     phu_thuoc=("b02",), cau_hinh=("mo_phong", "mle", "bayes", "mo_hinh"),
     thu_vien=("mo_phong", "uoc_luong", "mo_hinh_jax", "phan_phoi"))
