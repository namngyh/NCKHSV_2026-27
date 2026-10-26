"""Giai đoạn 2 (mục 7): b10 dự báo rủi ro H phiên, b11 quy tắc ký quỹ và đối chứng, b12 kiểm tra độ bền.

b10 (7.2): tại mọi phiên gốc của XT + KT, mô phỏng S đường cho H = 1..Hmax từ: MLE cắm tham số, mô phỏng lịch sử có lọc
    (phần dư chuẩn hóa), hậu nghiệm VB-MF, VB-FR, NUTS (hỗn hợp theo tham số), và EGARCH không biến ngoại sinh (MLE).
b11 (7.3 - 7.6): với mỗi cặp thiết kế (H, alpha): hiệu chỉnh (w, b, phi, eta) trên XT bằng chi phí kỳ vọng với
    c_th / c_v = 1 / alpha; đường biên đánh đổi trên XT; giữ cố định trên KT; sáu đối chứng; so sánh tại cùng ký quỹ
    trung bình (giả thuyết 3); kiểm định trên dãy lấy thưa; đèn giao thông; nhóm phiên.
b12 (7.6 cuối): đổi w, eta quanh cấu hình chọn; tổng hợp theo H, alpha; MLE so với hậu nghiệm; nhóm phiên.
THƯ VIỆN: numpy, pandas (đọc chuỗi ký quỹ lịch sử nếu có).
"""
from __future__ import annotations

import itertools

import numpy as np

import quy_trinh._chung as C
from de_cuong import danh_gia as DG
from de_cuong import du_lieu as DL
from de_cuong import ky_quy as KQ
from de_cuong import uoc_luong as U
from de_cuong.khung import GOC, Buoc
from de_cuong.mo_hinh_jax import DacTa

TEP = __file__
TV = ("du_lieu", "mo_hinh_jax", "uoc_luong", "phan_phoi", "ky_quy", "danh_gia")


def _moc(ctx):
    dau, cuoi = ctx.khoi()["XT"][0], ctx.khoi()["KT"][1]
    return C.cac_moc(ctx, dau, cuoi), dau, cuoi


def _nguon(ctx) -> list[str]:
    return ["mle", "fhs", "egarch_mle"] + list(ctx.cfg["bayes"]["phuong_phap"])


def _the_che(ctx, ten):
    return ctx.cfg["the_che"][ten]


def mot_ty_le(ctx, x: np.ndarray, H: int, ia: int) -> np.ndarray:
    """Một tỷ lệ cho cả hai phía từ mảng (phiên, Hmax, alpha, 3) của b10. [ky_quy].phia = "tri_tuyet_doi": phân vị của
    |R| (P(|R| > m) = alpha, khớp L = |R| của 7.4); "max_hai_phia" (đề cương): max(VaR mua, VaR bán), vượt xấp xỉ 2 alpha."""
    if ctx.cfg["ky_quy"].get("phia", "tri_tuyet_doi") == "tri_tuyet_doi":
        return x[:, H - 1, ia, 2]
    return np.max(x[:, H - 1, ia, :2], axis=1)


# ---------------------------------------------------------------------------------------- b10
def _lam_b10(ctx, khoa, truoc):
    ng, k = khoa.split("::")[0].split("|")
    k = int(k)
    moc, _, cuoi = _moc(ctx)
    s0 = moc[k]
    s1 = min(s0 + ctx.cfg["khoi"]["buoc_uoc_luong_lai"], cuoi)
    kh = ctx.khoa()
    pp = kh["phan_phoi"]
    bien = () if ng == "egarch_mle" else tuple(kh["bien"])
    spec = DacTa(pp, bien)
    d = ctx.du_lieu()
    f = DL.dac_trung(d, s0 - 1)
    dl = U.chuan_bi(f, spec.bien, s0)
    if ng in ("mle", "fhs", "egarch_mle"):
        P = ctx.doc("mle_cuon_chieu.pkl", "b08")[f"{pp}|{C.ten_tap(bien)}"]["psi_cac_moc"][k][None, :]
    else:
        P = ctx.doc("bayes_cuon_chieu.pkl", "b08")[k][ng]["psi_mau"]
        P = P[np.linspace(0, len(P) - 1, min(len(P), ctx.cfg["bayes"]["so_mau_hau_nghiem"])).astype(int)]
    kq = ctx.cfg["ky_quy"]
    nguon = KQ.nguon_moc(spec, P, f, dl, s0, s1, kq.get("x_tuong_lai", "ar1"))
    sim = KQ.mo_phong_moc(spec, nguon, d[["D_dh", "D_sau", "D_nghi"]].to_numpy(float), d["khuyet"].to_numpy(),
                          kq["S"], max(kq["H"]), kq["alpha"], ctx.rng(khoa), "loc" if ng == "fhs" else "tham_so")
    return {"s0": s0, **sim}


def _th_b10(ctx, kq):
    moc, dau, cuoi = _moc(ctx)
    out = {"tmoc": np.arange(dau - 1, cuoi - 1), "H": ctx.cfg["ky_quy"]["H"], "alpha": ctx.cfg["ky_quy"]["alpha"]}
    for ng in _nguon(ctx):
        ds = sorted([v for k, v in kq.items() if k.startswith(ng + "|")], key=lambda x: x["s0"])
        out[ng] = {"var": np.concatenate([x["var"] for x in ds]), "es": np.concatenate([x["es"] for x in ds])}
    ctx.ghi("du_bao_H_phien.pkl", out)


Buoc("b10", "du_bao_H_phien", "7.2", TEP, _lam_b10, _th_b10,
     don_vi=lambda ctx: [f"{ng}|{k:03d}::0" for ng in _nguon(ctx) for k in range(len(_moc(ctx)[0]))],
     phu_thuoc=("b08",), cau_hinh=("khoi", "bayes", "ky_quy"), thu_vien=TV, can_khoa_KT=True)


# ---------------------------------------------------------------------------------------- b11
def _can_bang(m, muc_tieu, hop):
    ok = hop & np.isfinite(m)
    return m * (muc_tieu / m[ok].mean())


def _ky_quy_lich_su(d, tmoc):
    p = GOC / "du_lieu" / "ngoai" / "ky_quy_lich_su.csv"
    if not p.exists():
        return None
    import pandas as pd
    s = pd.read_csv(p, parse_dates=["ngay"]).set_index("ngay")["ty_le"].sort_index()
    return s.reindex(d["date"].iloc[tmoc], method="ffill").to_numpy(float)


def _lam_b11(ctx, khoa, truoc):
    H, a = khoa.split("::")[0].split("|")
    H, a = int(H[1:]), float(a[1:])
    kqc = ctx.cfg["ky_quy"]
    ia = kqc["alpha"].index(a)
    db = ctx.doc("du_bao_H_phien.pkl", "b10")
    tmoc = db["tmoc"]
    d = ctx.du_lieu()
    kt0 = ctx.khoi()["KT"][0]
    R, ok_h = KQ.tong_that_thuc(d, max(kqc["H"]))
    L = np.abs(R[tmoc, H - 1])
    hop = ok_h[tmoc, H - 1] & np.isfinite(L)
    o_kt = tmoc + 1 >= kt0
    hop_xt = hop & ~o_kt & (tmoc + H < kt0)            # XT không nhìn sang KT
    hop_kt = hop & o_kt
    g = float(_the_che(ctx, "buoc_lam_tron_ky_quy")["gia_tri"])
    giam = float(_the_che(ctx, "gioi_han_giam_moi_phien")["gia_tri"])
    nt, nd = kqc["nguong_tang"], kqc["nguong_giam"]
    ttd = kqc.get("phia", "tri_tuyet_doi") == "tri_tuyet_doi"
    san = KQ.san_lich_su(R, tmoc, H, a, ttd)
    luoi = list(itertools.product(kqc["w"], kqc["b"], kqc["phi"], kqc["eta"]))
    vmax = lambda ng: mot_ty_le(ctx, db[ng]["var"], H, ia)
    emax = lambda ng: mot_ty_le(ctx, db[ng]["es"], H, ia)
    dong, chon, luoi_xt = {}, {}, {}
    for ng in ["mle", "egarch_mle"] + list(ctx.cfg["bayes"]["phuong_phap"]):
        tot, bang = None, []
        for w, b, phi, eta in luoi:
            m = KQ.quy_tac(vmax(ng), emax(ng), san, w, b, phi, eta, giam, g, nt, nd)
            c = KQ.chi_phi(m, L, hop_xt, 1.0, 1.0 / a)
            ct = KQ.chi_tieu(m, L, hop_xt, H)
            bang.append({"w": w, "b": b, "phi": phi, "eta": eta, "chi_phi": c, **{x: ct[x] for x in
                         ("m_tb", "ty_le_vuot", "thieu_hut_tb", "thieu_hut_moi_phien")}})
            if tot is None or c < tot[0]:
                tot = (c, (w, b, phi, eta), m)
        dong[f"dong_{ng}"], chon[ng], luoi_xt[ng] = tot[2], dict(zip(("w", "b", "phi", "eta"), tot[1])), bang
    m_dong = dong["dong_mle"]
    vuot_xt = KQ.chi_tieu(m_dong, L, hop_xt, H)["ty_le_vuot"]
    nang = dict(dong)
    c_ = chon["mle"]
    nang["dong_mle_khong_nguong"] = KQ.quy_tac(vmax("mle"), emax("mle"), san, c_["w"], c_["b"], c_["phi"], c_["eta"], giam, g)
    nang["VaR_tho_mle"] = vmax("mle")
    nang["FHS_EGARCH-X"] = vmax("fhs")
    nang["EWMA"] = KQ.ewma_var(d, tmoc, H, a, kqc["lambda_ewma"], ttd)
    nang["lich_su_cua_so"] = KQ.lich_su_cua_so(R, tmoc, H, a, kqc["cua_so_lich_su"], ttd)
    nang["co_dinh_khop_XT"] = np.full(len(tmoc), float(np.quantile(L[hop_xt], 1 - max(vuot_xt, 1 / hop_xt.sum()))))
    vsd = _the_che(ctx, "ty_le_ky_quy_vsdc")
    nang["VSD_hang_so" + ("" if vsd.get("nguon") else "_chua_xac_minh")] = np.full(len(tmoc), float(vsd["gia_tri"]))
    ls = _ky_quy_lich_su(d, tmoc)
    if ls is not None:
        nang["VSD_lich_su"] = ls
    nang["moc_tren_H_x_bien_do"] = np.full(len(tmoc), H * float(_the_che(ctx, "bien_do_gia_ngay")["gia_tri"]))
    m_tb = KQ.chi_tieu(m_dong, L, hop_kt, H)["m_tb"]
    th_d = np.where(hop_kt & np.isfinite(m_dong), np.maximum(L - m_dong, 0), np.nan)
    bang_kt = {}
    for ten, m in nang.items():
        row = KQ.chi_tieu(m, L, hop_kt, H)
        row["kiem_dinh_thua"] = KQ.kiem_dinh_vuot(np.where(o_kt, m, np.nan), np.where(o_kt, L, np.nan), hop_kt, H, a)
        row["chi_phi"] = KQ.chi_phi(m, L, hop_kt, 1.0, 1.0 / a)
        mc = m if ten == "dong_mle" else _can_bang(m, m_tb, hop_kt)
        rc = KQ.chi_tieu(mc, L, hop_kt, H)
        th_m = np.where(hop_kt & np.isfinite(mc), np.maximum(L - mc, 0), np.nan)
        t, p = DG.dm(th_d, th_m)
        row["cung_m_tb"] = {**{x: rc[x] for x in ("m_tb", "ty_le_vuot", "thieu_hut_tb", "thieu_hut_moi_phien", "thieu_hut_max")},
                            "DM_thieu_hut_moi_phien_dong_mle_thap_hon": {"t": t, "p_mot_phia": p}}
        bang_kt[ten] = row
    nhom = ctx.doc("du_bao_KT.pkl", "b09")["nhom"]
    kt_rel = int(np.argmax(o_kt))
    theo_nhom = {}
    for gname, mk in nhom.items():
        mask = np.zeros(len(tmoc), bool)
        mask[kt_rel:kt_rel + len(mk)] = mk[:len(tmoc) - kt_rel]
        theo_nhom[gname] = {ten: KQ.chi_tieu(nang[ten], L, hop_kt, H, mask) for ten in ("dong_mle", "co_dinh_khop_XT")}
    B = np.where(hop_kt & np.isfinite(m_dong), (L > m_dong).astype(float), np.nan)
    return {"H": H, "alpha": a, "chon": chon, "XT_dong_mle": KQ.chi_tieu(m_dong, L, hop_xt, H),
            "duong_bien_XT": {ng: KQ.duong_bien(v) for ng, v in luoi_xt.items()}, "luoi_XT": luoi_xt,
            "KT": bang_kt, "nhom_KT": theo_nhom, "den_giao_thong": DG.den_giao_thong(B[o_kt]),
            "chuoi": {**nang, "L": L, "hop_kt": hop_kt, "tmoc": tmoc}, "c_th_tren_c_v": 1 / a, "g": g, "giam": giam,
            "nguong_tang": nt, "nguong_giam": nd}


def _th_b11(ctx, kq):
    tom, chuoi = {}, {}
    for k, v in kq.items():
        v = dict(v)
        chuoi[k.split("::")[0]] = v.pop("chuoi")
        tom[k.split("::")[0]] = v
    kqc = ctx.cfg["ky_quy"]
    ctx.ghi("ky_quy.json", {"thiet_ke": tom, "trinh_bay": f"H{kqc['trinh_bay_H']}|a{kqc['trinh_bay_alpha']}",
                            "ghi_chu": "H và alpha là tham số thiết kế (7.1); (w, b, phi, eta) chọn trên XT, cố định trên KT."})
    ctx.ghi("chuoi_ky_quy.pkl", chuoi)


Buoc("b11", "ky_quy", "7.3 - 7.6", TEP, _lam_b11, _th_b11,
     don_vi=lambda ctx: [f"H{H}|a{a}::0" for H in ctx.cfg["ky_quy"]["H"] for a in ctx.cfg["ky_quy"]["alpha"]],
     phu_thuoc=("b09", "b10", "b00"), cau_hinh=("khoi", "ky_quy", "the_che"), thu_vien=TV, can_khoa_KT=True,
     chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b12
def _lam_b12(ctx, khoa, truoc):
    kqc = ctx.cfg["ky_quy"]
    tk = ctx.doc("ky_quy.json", "b11")["thiet_ke"]
    ch = ctx.doc("chuoi_ky_quy.pkl", "b11")
    db = ctx.doc("du_bao_H_phien.pkl", "b10")
    H, a = kqc["trinh_bay_H"], kqc["trinh_bay_alpha"]
    key = f"H{H}|a{a}"
    ia = kqc["alpha"].index(a)
    s = ch[key]
    L, hop, tmoc = s["L"], s["hop_kt"], s["tmoc"]
    d = ctx.du_lieu()
    R, _ = KQ.tong_that_thuc(d, max(kqc["H"]))
    san = KQ.san_lich_su(R, tmoc, H, a, kqc.get("phia", "tri_tuyet_doi") == "tri_tuyet_doi")
    c0 = tk[key]["chon"]["mle"]
    g = tk[key]["g"]
    giam = tk[key]["giam"]
    v = mot_ty_le(ctx, db["mle"]["var"], H, ia)
    e = mot_ty_le(ctx, db["mle"]["es"], H, ia)
    doi = {}
    for ten, luoi in (("w", kqc["w"]), ("eta", kqc["eta"])):
        doi[ten] = {}
        for x in luoi:
            c = {**c0, ten: x}
            m = KQ.quy_tac(v, e, san, c["w"], c["b"], c["phi"], c["eta"], giam, g, kqc["nguong_tang"], kqc["nguong_giam"])
            doi[ten][str(x)] = KQ.chi_tieu(m, L, hop, H)
    doi["nguong"] = {}
    for nt_, nd_ in kqc["nguong_do_ben"]:
        m = KQ.quy_tac(v, e, san, c0["w"], c0["b"], c0["phi"], c0["eta"], giam, g, nt_, nd_)
        doi["nguong"][f"{nt_}|{nd_}"] = KQ.chi_tieu(m, L, hop, H)
    m0 = KQ.quy_tac(v, e, san, c0["w"], c0["b"], c0["phi"], c0["eta"], giam, g)
    doi["lich_vsdc"] = KQ.chi_tieu(KQ.theo_lich_vsdc(m0, c0["phi"] * san, d["date"].iloc[tmoc].to_numpy()), L, hop, H)
    doi["vsdc"] = _doi_chung_vsdc(ctx, d, tmoc, L, hop, H, a)
    doi["tien"] = _quy_doi_tien(ctx, d, tmoc, L, hop, s)
    theo_thiet_ke = {k: {x: v_["KT"]["dong_mle"][x] for x in ("m_tb", "ty_le_vuot", "thieu_hut_tb", "thieu_hut_moi_phien")}
                     for k, v_ in tk.items()}
    mle_vs_hau_nghiem = {k: {ng: v_["KT"][f"dong_{ng}"] for ng in ["mle"] + list(ctx.cfg["bayes"]["phuong_phap"])
                             if f"dong_{ng}" in v_["KT"]} for k, v_ in tk.items()}
    return {"thiet_ke_trinh_bay": key, "doi_w_eta": doi, "theo_H_alpha": theo_thiet_ke,
            "mle_vs_hau_nghiem": mle_vs_hau_nghiem, "nhom_phien": tk[key]["nhom_KT"]}


def _quy_doi_tien(ctx, d, tmoc, L, hop, s) -> dict:
    """Quy đổi ra đồng mỗi hợp đồng: giá trị hợp đồng = giá đóng cửa phiên gốc x hệ số nhân (mục 7.1). Ký quỹ = m x giá trị;
    thiếu hụt = max(L - m, 0) x giá trị. Trung bình trên các phiên gốc hợp lệ của KT."""
    he_so = float(_the_che(ctx, "he_so_nhan")["gia_tri"])
    V = d["close"].to_numpy(float)[tmoc] * he_so
    out = {"he_so_nhan": he_so, "gia_tri_hop_dong_tb": float(np.nanmean(V[hop]))}
    for ten in ("dong_mle", "dong_mle_khong_nguong", "VSD_hang_so", "co_dinh_khop_XT", "VaR_tho_mle", "EWMA", "lich_su_cua_so"):
        if ten not in s:
            continue
        m = np.asarray(s[ten], float)
        ok = hop & np.isfinite(m) & np.isfinite(L)
        th = np.maximum(L[ok] - m[ok], 0.0) * V[ok]
        out[ten] = {"n": int(ok.sum()), "ky_quy_tb": float(np.mean(m[ok] * V[ok])), "ky_quy_max": float(np.max(m[ok] * V[ok])),
                    "thieu_hut_moi_phien_tb": float(th.mean()), "thieu_hut_max": float(th.max()), "tong_thieu_hut": float(th.sum()),
                    "thieu_hut_moi_lan_vuot": float(th[th > 0].mean()) if (th > 0).any() else 0.0}
    return out


def _doi_chung_vsdc(ctx, d, tmoc, L, hop, H, a) -> dict:
    """Đối chứng cách tính VSDC (bản 2022/2023 và 2025) trên các phiên gốc của KT; n, kỳ quan sát, zc lấy từ [vsdc]."""
    cv = ctx.cfg["vsdc"]
    r = d["r"].to_numpy(float)
    ngay = d["date"].iloc[tmoc].to_numpy()
    cac = [("2022-2023 (MVaR)", "mvar", False, cv["n"], N) for N in cv["ky_quan_sat_2022"]]
    cac += [(f"2025 (VaR 3 sigma), n = {n:g}", "var3", True, n, N) for n in (cv["n"], cv["n_bien_the_2025"])
            for N in cv["ky_quan_sat_2025"]]
    out = {}
    for ten, kieu, hai, n, N in cac:
        gia = KQ.vsdc_gia_tri(r, np.unique(np.r_[tmoc, tmoc - cv["tre_cong_bo"]]), N, kieu, n, cv["zc"], hai)
        for che_do, m in (("hằng ngày", np.array([gia.get(int(t), np.nan) for t in tmoc])),
                          ("theo lịch 1, 10, 20", KQ.vsdc_theo_lich(ngay, tmoc, lambda t: gia.get(t, np.nan), cv["tre_cong_bo"]))):
            x = KQ.chi_tieu(m, L, hop, H)
            x["kiem_dinh_thua"] = KQ.kiem_dinh_vuot(m, L, hop, H, a)
            x["chi_phi"] = KQ.chi_phi(m, L, hop, 1.0, 1.0 / a)
            out[f"{ten}; N = {N}; {che_do}"] = x
    return out


def _th_b12(ctx, kq):
    ctx.ghi("do_ben.json", kq["tat_ca::0"])


Buoc("b12", "do_ben", "7.6", TEP, _lam_b12, _th_b12, phu_thuoc=("b11",), cau_hinh=("ky_quy", "vsdc"), thu_vien=TV,
     can_khoa_KT=True, song_song=False, chi_chu_tri=True)
