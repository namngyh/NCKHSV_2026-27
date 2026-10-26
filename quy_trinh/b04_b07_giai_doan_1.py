"""Giai đoạn 1 trước khi nhìn khối kiểm tra: b04 MLE ban đầu, b05 chọn biến, b06 chọn phân phối, b07 chốt cấu hình.

b04 (mục 3.6, 4.2, 8.2 tháng 11/2026): MLE EGARCH-X cấu hình chính trên khối UL cho ba phân phối (cộng GED theo
    Phụ lục B để báo cáo), sai số chuẩn Hessian và bánh kẹp, LR cho gamma = 0, chẩn đoán (i)-(iv), khả nghịch,
    bootstrap tham số khi beta sát biên.
b05 (mục 6): dự báo cuốn chiếu trên XT cho mọi tập biến thỏa ngưỡng tương quan; chọn tiến theo LS với DM một phía
    và không làm xấu FZ0; SPA cho "ít nhất một ứng viên cải thiện"; LR trong mẫu từng biến hiệu chỉnh Holm; VIF;
    độ ổn định của gamma qua các lần ước lượng lại; báo cáo mọi ứng viên.
b06 (mục 5, 6): dự báo cuốn chiếu trên XT cho tập biến được chọn và cấu hình chính với mọi phân phối; chọn phân phối
    có LS cao nhất (GED tham gia khi [bam_sat_bm02].ged_tham_gia_chon = true, bám BM02; ngược lại chỉ báo cáo).
b07 (mục 8.1, Phụ lục C.7): kiểm tra ba điều kiện chuyển giai đoạn rồi ghi KHOA_CAU_HINH.json; từ đây khối KT mới
    được mở.
THƯ VIỆN: numpy, scipy.stats, jax.
"""
from __future__ import annotations

import itertools
import json
import math

import jax.numpy as jnp
import numpy as np
from scipy import stats

import quy_trinh._chung as C
from de_cuong import danh_gia as DG
from de_cuong import du_lieu as DL
from de_cuong import phan_phoi as PP
from de_cuong import uoc_luong as U
from de_cuong.khung import Buoc, ghi_nguyen_tu, js
from de_cuong.mo_hinh_jax import DacTa, ham

TEP = __file__
TV = ("du_lieu", "mo_hinh_jax", "uoc_luong", "phan_phoi", "danh_gia")


def _cac_pp(ctx, them_thay_the=True):
    pps = list(ctx.cfg["mo_hinh"]["phan_phoi"])
    return pps + [ctx.cfg["mo_hinh"]["phan_phoi_thay_the"]] if them_thay_the else pps


def uoc_luong_day_du(ctx, pp, bien, s1, khoa_rng, bootstrap=True) -> dict:
    """MLE trên [BO_QUA, s1) kèm mọi đại lượng của mục 4.2."""
    d = ctx.du_lieu()
    f = DL.dac_trung(d, s1 - 1)
    spec = DacTa(pp, bien)
    dl = U.chuan_bi(f, spec.bien, s1)
    cm = ctx.cfg["mle"]
    rng = ctx.rng(*khoa_rng)
    m = U.mle(spec, dl, cm["so_diem_khoi_tao"], cm["do_tan_khoi_tao"], rng)
    n = int(np.asarray(dl[3]).sum())
    se = U.sai_so_chuan(spec, m["psi"], dl)
    out = {"cau_hinh": repr(spec), "ten": spec.ten(), "psi": m["psi"], "se_hessian": se["se_hessian"],
           "se_banh_kep": se["se_banh_kep"], "ll": m["ll"], "k": spec.d, "n": n,
           "aic": 2 * spec.d - 2 * m["ll"], "bic": spec.d * math.log(n) - 2 * m["ll"],
           "so_gan_tot": m["so_gan_tot"], "so_diem": m["so_diem"], "do_tan_ll": m["do_tan_ll"]}
    if spec.nx:
        m0 = U.mle(DacTa(pp, ()), U.bo_bien(dl, spec.nx), cm["so_diem_khoi_tao"], cm["do_tan_khoi_tao"], rng)
        lr = 2 * (m["ll"] - m0["ll"])
        out.update({"LR_gamma0": lr, "LR_p": float(1 - stats.chi2.cdf(lr, spec.nx)), "ll_EGARCH": m0["ll"]})
    lnh, z = (np.asarray(x) for x in ham(pp, spec.nx).loc(jnp.asarray(m["psi"]), dl))
    mk = np.asarray(dl[3])
    sh = m["psi"][5 + spec.nx + 3:]
    out["chan_doan"] = DG.chan_doan(z[mk], PP.cdf(z[mk], pp, sh), f["tau"][DL.BO_QUA:][mk],
                                    f["D"][DL.BO_QUA:][mk], lnh[mk])
    out["kha_nghich"] = U.kha_nghich(spec, m["psi"], dl, ctx.cfg["mo_hinh"]["lech_khoi_tao_kha_nghich"])
    if bootstrap and m["psi"][4] > cm["nguong_beta_bien"]:
        out["bootstrap"] = U.bootstrap_tham_so(spec, m["psi"], dl, cm["so_lan_bootstrap"], rng)
    return out


# ---------------------------------------------------------------------------------------- b04
def _lam_b04(ctx, khoa, truoc):
    pp = khoa.split("::")[0]
    return uoc_luong_day_du(ctx, pp, tuple(ctx.cfg["mo_hinh"]["cau_hinh_chinh"]), ctx.khoi()["UL"][1], ("b04", pp))


def _th_b04(ctx, kq):
    ctx.ghi("mle_UL.json", {k.split("::")[0]: v for k, v in kq.items()})


Buoc("b04", "mle_ban_dau", "3.6, 4.2", TEP, _lam_b04, _th_b04,
     don_vi=lambda ctx: [f"{pp}::0" for pp in _cac_pp(ctx)], phu_thuoc=("b02",),
     cau_hinh=("du_lieu", "khoi", "mo_hinh", "mle"), thu_vien=TV, chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b04b
def _thong_ke_chuoi(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    c = r - r.mean()
    return {"do_lech_chuan": float(r.std()), "nhon": float((c ** 4).mean() / max(c.var() ** 2, 1e-300)),
            "max_tuyet_doi": float(np.abs(r).max()), "q01": float(np.quantile(r, 0.01)), "q99": float(np.quantile(r, 0.99))}


def _lam_b04b(ctx, khoa, truoc):
    """Kiểm tra dự báo tiên nghiệm (mục 4.3.1): psi ~ tiên nghiệm Bảng 5, sinh lợi suất trên lịch và biến ngoại sinh
    thật của khối UL (cấu hình chính), so thống kê mô phỏng với thống kê dữ liệu."""
    from de_cuong import mo_phong as MP
    pp = khoa.split("::")[0]
    d = ctx.du_lieu()
    s1 = ctx.khoi()["UL"][1]
    f = DL.dac_trung(d, s1 - 1)
    spec = DacTa(pp, tuple(ctx.cfg["mo_hinh"]["cau_hinh_chinh"]))
    dl = U.chuan_bi(f, spec.bien, s1)
    a1 = s1 - DL.BO_QUA
    Z, J, lh0 = np.asarray(dl[2])[:a1], np.asarray(dl[1])[:a1], float(dl[4])
    rng = ctx.rng(pp)
    tk, can = [], 0
    for _ in range(ctx.cfg["do_nhay_tien_nghiem"]["so_lan_du_bao_tien_nghiem"]):
        psi = MP.rut_tien_nghiem(pp, rng)
        r, lnh = U.sinh_tu_mo_hinh(spec, psi, Z, J, lh0, rng)
        can += int((np.abs(lnh) >= 19.999).any())
        tk.append(_thong_ke_chuoi(np.where(J, np.nan, r)))
    du_lieu = _thong_ke_chuoi(np.asarray(f["r"][DL.BO_QUA:s1]))
    out = {"so_lan": len(tk), "ty_le_cham_can_lnh": can / len(tk), "du_lieu": du_lieu, "tien_nghiem": {}}
    for k in du_lieu:
        x = np.array([t[k] for t in tk])
        out["tien_nghiem"][k] = {"q05": float(np.quantile(x, 0.05)), "q50": float(np.median(x)),
                                 "q95": float(np.quantile(x, 0.95)), "phan_vi_cua_du_lieu": float((x < du_lieu[k]).mean())}
    return out


def _th_b04b(ctx, kq):
    ctx.ghi("du_bao_tien_nghiem.json", {k.split("::")[0]: v for k, v in kq.items()})


Buoc("b04b", "du_bao_tien_nghiem", "4.3.1", TEP, _lam_b04b, _th_b04b,
     don_vi=lambda ctx: [f"{pp}::0" for pp in _cac_pp(ctx)], phu_thuoc=("b02",),
     cau_hinh=("du_lieu", "khoi", "mo_hinh", "do_nhay_tien_nghiem"), thu_vien=TV + ("mo_phong",), chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b05
def _tap_hop_le(ctx):
    d = ctx.du_lieu()
    xt1 = ctx.khoi()["XT"][1]
    f = DL.dac_trung(d, xt1 - 1)
    ten = ctx.cfg["mo_hinh"]["bien_ung_vien"]
    X = f["X"][DL.BO_QUA:xt1][:, [DL.TEN_BIEN.index(b) for b in ten]]
    R = np.corrcoef(X.T)
    nguong = ctx.cfg["chon_bien"]["tuong_quan_toi_da"]
    taps = [c for k in range(len(ten) + 1) for c in itertools.combinations(ten, k)
            if all(abs(R[ten.index(a), ten.index(b)]) <= nguong for a, b in itertools.combinations(c, 2))]
    return taps, R, X


def _moc_khoi(ctx, ten_khoi):
    dau, cuoi = ctx.khoi()[ten_khoi]
    return C.cac_moc(ctx, dau, cuoi), cuoi


def _dv_b05(ctx):
    taps, _, _ = _tap_hop_le(ctx)
    moc, _ = _moc_khoi(ctx, "XT")
    don = [(b,) for b in ctx.cfg["mo_hinh"]["bien_ung_vien"]]
    return ([f"cc|{C.ten_tap(t)}::{k:03d}" for t in taps for k in range(len(moc))]
            + [f"lr|{C.ten_tap(t)}::0" for t in [()] + don])


def _lam_b05(ctx, khoa, truoc):
    nhom, k = khoa.split("::")
    loai, tap = nhom.split("|")
    pp = ctx.cfg["chon_bien"]["phan_phoi"]
    moc, cuoi = _moc_khoi(ctx, "XT")
    if loai == "cc":
        s0 = moc[int(k)]
        return C.mle_mot_moc(ctx, pp, C.tu_ten_tap(tap), s0, min(s0 + ctx.cfg["khoi"]["buoc_uoc_luong_lai"], cuoi), truoc)
    d = ctx.du_lieu()
    f = DL.dac_trung(d, cuoi - 1)
    spec = DacTa(pp, C.tu_ten_tap(tap))
    m = U.mle(spec, U.chuan_bi(f, spec.bien, cuoi), ctx.cfg["mle"]["so_diem_khoi_tao"],
              ctx.cfg["mle"]["do_tan_khoi_tao"], ctx.rng("lr", tap))
    return {"ll": m["ll"], "psi": m["psi"]}


def _th_b05(ctx, kq):
    cfg = ctx.cfg
    ten = cfg["mo_hinh"]["bien_ung_vien"]
    taps, R, X = _tap_hop_le(ctx)
    ten_taps = [C.ten_tap(t) for t in taps]
    dau, cuoi = ctx.khoi()["XT"]
    r, _ = C.thuc_te(ctx.du_lieu(), dau, cuoi)
    db = {t: C.ghep([v for k, v in kq.items() if k.split("::")[0] == f"cc|{t}"]) for t in ten_taps}
    a_fz, muc = cfg["chon_bien"]["alpha_fz"], cfg["chon_bien"]["muc_y_nghia"]
    ls = {t: float(np.nanmean(db[t]["ls"])) for t in ten_taps}
    fz = {t: float(np.nanmean(DG.fz0_chuoi(db[t], r, a_fz))) for t in ten_taps}
    sap = lambda c: tuple(sorted(c, key=ten.index))
    cur, nhat_ky = (), []
    while True:
        ung = [sap(cur + (b,)) for b in ten if b not in cur and C.ten_tap(sap(cur + (b,))) in ten_taps]
        if not ung:
            break
        diem = {C.ten_tap(c): ls[C.ten_tap(c)] for c in ung}
        tot = max(diem, key=diem.get)
        t_dm, p_dm = DG.dm(-db[tot]["ls"], -db[C.ten_tap(cur)]["ls"])
        nhan = bool(p_dm < muc and fz[tot] <= fz[C.ten_tap(cur)])
        nhat_ky.append({"tap_hien_tai": C.ten_tap(cur), "ung_vien_LS": diem, "tot_nhat": tot, "DM_t": t_dm,
                        "DM_p_mot_phia": p_dm, "FZ0_hien_tai": fz[C.ten_tap(cur)], "FZ0_ung_vien": fz[tot],
                        "chap_nhan": nhan})
        ctx.log(f"chon tien: {C.ten_tap(cur)} -> {tot}, DM p = {p_dm:.4f}, chap nhan = {nhan}")
        if not nhan:
            break
        cur = C.tu_ten_tap(tot)
    chon = C.ten_tap(cur)
    so_sanh = [C.ten_tap((b,)) for b in ten] + ([chon] if len(cur) > 1 else [])
    L0 = -db["EGARCH"]["ls"]
    Dm = np.column_stack([L0 + db[c]["ls"] for c in so_sanh])
    Dm = Dm[np.isfinite(Dm).all(1)]
    spa = DG.spa(Dm, ctx.rng("spa"), cfg["danh_gia"]["so_lan_bootstrap"], cfg["danh_gia"]["do_dai_khoi_tb"])
    ll0 = kq["lr|EGARCH::0"]["ll"]
    p_lr = {b: float(1 - stats.chi2.cdf(2 * (kq[f"lr|{b}::0"]["ll"] - ll0), 1)) for b in ten}
    holm, mx = {}, 0.0
    for i, b in enumerate(sorted(p_lr, key=p_lr.get)):
        mx = max(mx, min(1.0, (len(p_lr) - i) * p_lr[b]))
        holm[b] = mx
    on_dinh = {}
    if cur:
        P = db[chon]["psi_cac_moc"]
        for i, b in enumerate(cur):
            g = P[:, 5 + i]
            on_dinh[b] = {"min": float(g.min()), "max": float(g.max()), "doi_dau": bool(g.min() < 0 < g.max())}
    ctx.ghi("chon_bien.json", {
        "chon": list(cur), "ten_chon": chon, "phan_phoi_dung_khi_chon": cfg["chon_bien"]["phan_phoi"],
        "nhat_ky_chon_tien": nhat_ky, "SPA": {**spa, "ung_vien": so_sanh},
        "LR_trong_mau_p": p_lr, "LR_Holm_p": holm, "VIF_tap_chon": DG.vif(X[:, [ten.index(b) for b in cur]]) if cur else [],
        "tuong_quan": R, "on_dinh_gamma": on_dinh, "LS_moi_ung_vien": ls, f"FZ0_{a_fz}_moi_ung_vien": fz,
        "ghi_chu": "Cấu hình chính luôn được báo cáo; nếu d_X khác 2 thì đề cương phải ghi nhận thay đổi (mục 6)."})
    ctx.ghi("du_bao_XT.pkl", db)


Buoc("b05", "chon_bien", "6", TEP, _lam_b05, _th_b05, don_vi=_dv_b05, phu_thuoc=("b02",),
     cau_hinh=("du_lieu", "khoi", "mo_hinh", "mle", "chon_bien", "danh_gia"), thu_vien=TV)


# ---------------------------------------------------------------------------------------- b06
def _taps_b06(ctx):
    chon = C.ten_tap(ctx.doc("chon_bien.json", "b05")["chon"])
    return list(dict.fromkeys([chon, C.ten_tap(ctx.cfg["mo_hinh"]["cau_hinh_chinh"])]))


def _dv_b06(ctx):
    moc, _ = _moc_khoi(ctx, "XT")
    return [f"{pp}|{t}::{k:03d}" for pp in _cac_pp(ctx) for t in _taps_b06(ctx) for k in range(len(moc))]


def _lam_b06(ctx, khoa, truoc):
    nhom, k = khoa.split("::")
    pp, tap = nhom.split("|")
    moc, cuoi = _moc_khoi(ctx, "XT")
    s0 = moc[int(k)]
    return C.mle_mot_moc(ctx, pp, C.tu_ten_tap(tap), s0, min(s0 + ctx.cfg["khoi"]["buoc_uoc_luong_lai"], cuoi), truoc)


def _th_b06(ctx, kq):
    dau, cuoi = ctx.khoi()["XT"]
    r, khoa = C.thuc_te(ctx.du_lieu(), dau, cuoi)
    nhom = sorted({k.split("::")[0] for k in kq})
    db = {g: C.ghep([v for k, v in kq.items() if k.split("::")[0] == g]) for g in nhom}
    al = ctx.cfg["danh_gia"]["alphas"]
    bang = {g: DG.tom_tat(db[g], r, db[g]["rho_sao"], khoa, al, lags=ctx.cfg["danh_gia"]["tre_phan_vi_dong"]) for g in nhom}
    chon_tap = _taps_b06(ctx)[0]
    pps = _cac_pp(ctx, ctx.cfg["bam_sat_bm02"]["ged_tham_gia_chon"])     # BM02: GED là một trong bốn phân phối
    tot = max(pps, key=lambda pp: bang[f"{pp}|{chon_tap}"]["LS"])
    dmb = {f"{a}-{b}": dict(zip(("t", "p_mot_phia"), DG.dm(-db[f"{a}|{chon_tap}"]["ls"], -db[f"{b}|{chon_tap}"]["ls"])))
           for a in _cac_pp(ctx) for b in _cac_pp(ctx) if a != b}
    ctx.ghi("chon_phan_phoi.json", {"tap": chon_tap, "phan_phoi": tot, "bang_XT": bang, "DM_LS": dmb,
                                    "ghi_chu": ("GED tham gia lựa chọn (bám BM02)." if ctx.cfg["bam_sat_bm02"]["ged_tham_gia_chon"]
                                                else "GED (Phụ lục B) chỉ báo cáo, không tham gia lựa chọn."),
                                    "ung_vien": pps})
    ctx.ghi("du_bao_XT.pkl", db)


Buoc("b06", "chon_phan_phoi", "5, 6", TEP, _lam_b06, _th_b06, don_vi=_dv_b06, phu_thuoc=("b05",),
     cau_hinh=("du_lieu", "khoi", "mo_hinh", "mle", "danh_gia", "bam_sat_bm02"), thu_vien=TV)


# ---------------------------------------------------------------------------------------- b07
def _cong_b07(ctx):
    cb = ctx.doc("chon_bien.json", "b05")
    cp = ctx.doc("chon_phan_phoi.json", "b06")
    mp = ctx.doc("mo_phong_khoi_phuc.json", "b03a")
    pp, bien = cp["phan_phoi"], tuple(cb["chon"])
    cg = ctx.cfg["cong"]
    dk1a = mp["dieu_kien_8_1_i"].get(pp, {"dat": False, "ghi_chu": f"phan phoi {pp} chua co mo phong khoi phuc tham so (b03a)"})
    if cg.get("kiem_tra_mo_hinh_chon", False):
        dk1b = ctx.doc("mo_phong_cau_hinh_chon.json", "b06c")["dieu_kien"]
        dk1 = {"b03a_thiet_ke_de_cuong": dk1a, "b06c_mo_hinh_chon": dk1b, "dat": bool(dk1a["dat"] and dk1b["dat"])}
    else:
        dk1 = dk1a
    x = ctx.doc("uoc_luong_chon.json", "b06b")      # ước lượng UL + XT của cấu hình được chọn (b06b)
    dk2 = {"hoi_quy_tau_F_p": x["chan_doan"]["hoi_quy_tau_F_p"], "z2_tb_theo_tau": x["chan_doan"]["z2_tb_theo_tau"]}
    dk2["dat"] = bool(dk2["hoi_quy_tau_F_p"] > cg["p_tau"])
    row = cp["bang_XT"][f"{pp}|{C.ten_tap(bien)}"]
    dk3 = {f"a{a}": {"kupiec_p": row[f"a{a}"]["kupiec_p"], "cc_p": row[f"a{a}"]["cc_p"]} for a in ctx.cfg["danh_gia"]["alphas"]}
    dk3["dat"] = bool(all(v["kupiec_p"] > cg["p_var_xt"] and v["cc_p"] > cg["p_var_xt"]
                          for k, v in dk3.items() if k.startswith("a")))
    b00 = ctx.thu_muc("b00") / "cong.json"
    c1 = json.loads(b00.read_text(encoding="utf-8")) if b00.exists() else {}
    return {"dat": bool(dk1["dat"] and dk2["dat"] and dk3["dat"]), "i_mo_phong": dk1, "ii_z2_theo_tau": dk2,
            "iii_VaR_XT": dk3, "C1_the_che": c1, "uoc_luong_UL_XT": x}


def _lam_b07(ctx, khoa, truoc):
    cb = ctx.doc("chon_bien.json", "b05")
    cp = ctx.doc("chon_phan_phoi.json", "b06")
    return {"phan_phoi": cp["phan_phoi"], "bien": cb["chon"], "cau_hinh_chinh": ctx.cfg["mo_hinh"]["cau_hinh_chinh"],
            "khoi": ctx.khoi(), "ho_so": ctx.ho_so}


def _th_b07(ctx, kq):
    x = kq["tat_ca::0"]
    ctx.ghi("khoa_cau_hinh.json", x)
    ghi_nguyen_tu(ctx.tep_khoa, json.dumps(js(x), ensure_ascii=False, indent=1).encode("utf-8"))
    ctx.log(f"DA KHOA CAU HINH: {x['phan_phoi']} | {C.ten_tap(x['bien'])}. Khoi KT duoc mo tu buoc b08.")


Buoc("b07", "chot_cau_hinh", "8.1, C.7", TEP, _lam_b07, _th_b07,
     phu_thuoc=("b03a", "b04", "b04b", "b05", "b06", "b06b", "b06c"),
     cau_hinh=("cong", "khoi"), thu_vien=TV, cong=_cong_b07, song_song=False, chi_chu_tri=True)
