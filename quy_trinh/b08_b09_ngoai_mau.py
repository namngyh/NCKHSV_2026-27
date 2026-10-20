"""Sau khi khóa cấu hình: b08 dự báo cuốn chiếu bốn phương pháp, b08b độ nhạy tiên nghiệm, b09 đánh giá trên KT.

b08 (mục 4.3, 5.1): ước lượng lại mỗi 21 phiên, cửa sổ mở rộng, trên XT + KT.
    MLE cho: EGARCH (gamma = 0), cấu hình chính (A, G), cấu hình được chọn với mọi phân phối (và GED để báo cáo).
    VB mean-field, VB full-rank (mỗi lần khởi tạo là một đơn vị) và NUTS (mỗi chuỗi là một đơn vị, điểm kiểm tra
    tầng 3 sau khởi động và mỗi đoạn lấy mẫu) cho cấu hình được chọn ở mọi mốc.
b08b (mục 4.3.1): phân tích độ nhạy tiên nghiệm bằng NUTS tại mốc đầu khối KT (dữ liệu đến hết XT) với các phương án
    trong [do_nhay_tien_nghiem]; so trung bình và độ lệch chuẩn hậu nghiệm với tiên nghiệm Bảng 5.
b09 (mục 4.3.5, 5): so sánh bốn phương pháp tại từng mốc (tỷ số độ lệch chuẩn biên so với NUTS, chênh lệch trung bình
    theo đơn vị độ lệch chuẩn NUTS, chuẩn Frobenius của hiệu hai ma trận tương quan, thời gian); mật độ dự báo hậu
    nghiệm; QLIKE, LS, PIT/Berkowitz, VaR/ES và các kiểm định, FZ0; DM, Giacomini-White, tập mô hình tin cậy;
    CH1 (EGARCH-X so với EGARCH), CH2 (phân phối, phương pháp suy diễn); nhóm phiên (5.6).
THƯ VIỆN: numpy, jax.
"""
from __future__ import annotations

import numpy as np

import quy_trinh._chung as C
from de_cuong import danh_gia as DG
from de_cuong import du_lieu as DL
from de_cuong import uoc_luong as U
from de_cuong.khung import Buoc
from de_cuong.mo_hinh_jax import TIEN_NGHIEM_GOC, DacTa

TEP = __file__
TV = ("du_lieu", "mo_hinh_jax", "uoc_luong", "phan_phoi", "danh_gia")


def _moc(ctx):
    dau, cuoi = ctx.khoi()["XT"][0], ctx.khoi()["KT"][1]
    return C.cac_moc(ctx, dau, cuoi), dau, cuoi


def cau_hinh_mle(ctx) -> list[tuple[str, str]]:
    kh = ctx.khoa()
    pp, chon, chinh = kh["phan_phoi"], C.ten_tap(kh["bien"]), C.ten_tap(kh["cau_hinh_chinh"])
    ds = [(pp, "EGARCH"), (pp, chinh)] + [(p, chon) for p in ctx.cfg["mo_hinh"]["phan_phoi"]]
    ds.append((ctx.cfg["mo_hinh"]["phan_phoi_thay_the"], chon))
    return list(dict.fromkeys(ds))


def _hau_to(ctx, pt):
    b = ctx.cfg["bayes"]
    return [f"c{c}" for c in range(b["nuts_so_chuoi"])] if pt == "nuts" else \
        [f"l{i}" for i in range(b["vb_so_lan_khoi_tao"])]


def _mle_khoi_tao(ctx, s0):
    """MLE của cấu hình được chọn trên [BO_QUA, s0): điểm khởi tạo cho VB và NUTS (tất định theo s0)."""
    kh = ctx.khoa()
    spec = DacTa(kh["phan_phoi"], tuple(kh["bien"]))
    f = DL.dac_trung(ctx.du_lieu(), s0 - 1)
    dl = U.chuan_bi(f, spec.bien, s0)
    m = U.mle(spec, dl, ctx.cfg["mle"]["so_diem_khoi_tao"], ctx.cfg["mle"]["do_tan_khoi_tao"],
              np.random.default_rng(int(ctx.cfg["chung"]["hat_giong"]) + s0))
    return spec, dl, m


def _bayes_mot(ctx, khoa, pt, spec, dl, m, chi_so, pa=TIEN_NGHIEM_GOC):
    b = ctx.cfg["bayes"]
    seed = ctx.hat_giong(khoa)
    rng = ctx.rng(khoa, "khoi_tao")
    if pt == "nuts":
        return U.nuts_chuoi(spec, dl, m["xi"] + rng.normal(0, 0.05, spec.d), b["nuts_khoi_dong"], b["nuts_so_mau"],
                            b["nuts_dich_chap_nhan"], b["nuts_do_sau_cay"], seed, thu_muc=ctx.thu_muc_doan(khoa),
                            vong_moi_doan=b["nuts_vong_moi_doan"], pa=pa)
    return U.vb(spec, dl, m["xi"] + rng.normal(0, 0.02 * chi_so, spec.d), pt, b["vb_so_buoc"], b["vb_toc_do_hoc"],
                b["so_mau_hau_nghiem"], seed, thu_muc=ctx.thu_muc_doan(khoa), buoc_moi_doan=b["vb_buoc_moi_doan"], pa=pa)


# ---------------------------------------------------------------------------------------- b08
def _dv_b08(ctx):
    moc, _, _ = _moc(ctx)
    ds = [f"mle|{pp}|{t}::{k:03d}" for pp, t in cau_hinh_mle(ctx) for k in range(len(moc))]
    for pt in ctx.cfg["bayes"]["phuong_phap"]:
        ds += [f"{pt}|{k:03d}|{h}::0" for k in range(len(moc)) for h in _hau_to(ctx, pt)]
    return ds


def _lam_b08(ctx, khoa, truoc):
    nhom = khoa.split("::")
    p = nhom[0].split("|")
    moc, _, cuoi = _moc(ctx)
    buoc = ctx.cfg["khoi"]["buoc_uoc_luong_lai"]
    if p[0] == "mle":
        s0 = moc[int(nhom[1])]
        return C.mle_mot_moc(ctx, p[1], C.tu_ten_tap(p[2]), s0, min(s0 + buoc, cuoi), truoc)
    k, h = int(p[1]), p[2]
    s0 = moc[k]
    spec, dl, m = _mle_khoi_tao(ctx, s0)
    out = _bayes_mot(ctx, khoa, p[0], spec, dl, m, int(h[1:]))
    out.pop("xi_mau", None)
    out.update({"s0": s0, "s1": min(s0 + buoc, cuoi), "psi_mle": m["psi"]})
    return out


def _gop_bayes_moc(ctx, kq) -> tuple[dict, dict]:
    """Gộp theo mốc: VB chọn lần khởi tạo có ELBO lớn nhất; NUTS ghép các chuỗi và chẩn đoán R_hat, ESS, phân kỳ."""
    b = ctx.cfg["bayes"]
    theo = {}
    for k, v in kq.items():
        p = k.split("::")[0].split("|")
        if p[0] in b["phuong_phap"]:
            theo.setdefault(int(p[1]), {}).setdefault(p[0], []).append(v)
    gop, chan_doan = {}, {}
    for km, x in sorted(theo.items()):
        gop[km], chan_doan[km] = {}, {}
        for pt, ds in x.items():
            if pt == "nuts":
                cd = U.chan_doan_nuts(ds, b["rhat_toi_da"], b["ess_toi_thieu"]) if len(ds) > 1 else {"dat": None}
                gop[km][pt] = {"psi_mau": np.concatenate([c["psi_mau"] for c in ds]), "s0": ds[0]["s0"],
                               "s1": ds[0]["s1"], "giay": max(c["giay"] for c in ds),
                               "buoc_tb": float(np.mean([c["buoc_tb"] for c in ds])), "chan_doan": cd}
                chan_doan[km][pt] = cd
            else:
                tot = max(ds, key=lambda v: v["elbo"])
                gop[km][pt] = {**tot, "elbo_cac_lan": [v["elbo"] for v in ds]}
                chan_doan[km][pt] = {"k_hat": tot["k_hat"], "elbo_cac_lan": [v["elbo"] for v in ds],
                                     "dat": bool(tot["k_hat"] < b["k_hat_toi_da"])}
    return gop, chan_doan


def _th_b08(ctx, kq):
    mle = {}
    for pp, t in cau_hinh_mle(ctx):
        mle[f"{pp}|{t}"] = C.ghep([v for k, v in kq.items() if k.split("::")[0] == f"mle|{pp}|{t}"])
    ctx.ghi("mle_cuon_chieu.pkl", mle)
    gop, chan_doan = _gop_bayes_moc(ctx, kq)
    ctx.ghi("bayes_cuon_chieu.pkl", gop)
    ctx.ghi("chan_doan_bayes.json", {
        "theo_moc": chan_doan,
        "nuts_dat_moi_moc": all(v.get("nuts", {}).get("dat") for v in chan_doan.values() if "nuts" in v),
        "so_moc_nuts_khong_dat": [k for k, v in chan_doan.items() if "nuts" in v and not v["nuts"].get("dat")],
        "vb_k_hat_dat": {pt: [v[pt]["dat"] for v in chan_doan.values() if pt in v] for pt in ("vb_mf", "vb_fr")}})


Buoc("b08", "cuon_chieu_bon_phuong_phap", "4.3, 5.1", TEP, _lam_b08, _th_b08, don_vi=_dv_b08, phu_thuoc=("b07",),
     cau_hinh=("du_lieu", "khoi", "mo_hinh", "mle", "bayes", "danh_gia"), thu_vien=TV, can_khoa_KT=True)


# ---------------------------------------------------------------------------------------- b08b
def _phuong_an(ctx) -> dict:
    out = {}
    for ten, v in ctx.cfg["do_nhay_tien_nghiem"]["phuong_an"].items():
        out[ten] = (float(v["he_so_sd"]), float(v["beta_a"]), float(v["beta_b"]), float(v["nu_toc_do"]))
    return out


def _dv_b08b(ctx):
    return [f"dn|{ten}|nuts|{h}::0" for ten in _phuong_an(ctx) for h in _hau_to(ctx, "nuts")]


def _lam_b08b(ctx, khoa, truoc):
    p = khoa.split("::")[0].split("|")
    s0 = ctx.khoi()["KT"][0]
    spec, dl, m = _mle_khoi_tao(ctx, s0)
    out = _bayes_mot(ctx, khoa, "nuts", spec, dl, m, int(p[3][1:]), pa=_phuong_an(ctx)[p[1]])
    out.pop("xi_mau", None)
    out["s0"] = s0
    return out


def _th_b08b(ctx, kq):
    b = ctx.cfg["bayes"]
    s0 = ctx.khoi()["KT"][0]
    goc = None
    gop = ctx.doc("bayes_cuon_chieu.pkl", "b08")
    for x in gop.values():
        if "nuts" in x and x["nuts"]["s0"] == s0:
            goc = x["nuts"]["psi_mau"]
    kh = ctx.khoa()
    ten = DacTa(kh["phan_phoi"], tuple(kh["bien"])).ten()
    out = {"moc_s0": s0, "ten": ten, "tien_nghiem_goc": TIEN_NGHIEM_GOC, "phuong_an": {}}
    for pa in _phuong_an(ctx):
        ds = [v for k, v in kq.items() if k.split("|")[1] == pa]
        M = np.concatenate([c["psi_mau"] for c in ds])
        cd = U.chan_doan_nuts(ds, b["rhat_toi_da"], b["ess_toi_thieu"]) if len(ds) > 1 else {}
        row = {"tham_so": _phuong_an(ctx)[pa], "trung_binh": M.mean(0), "do_lech_chuan": M.std(0),
               "chan_doan": {k: cd.get(k) for k in ("rhat_max", "ess_min", "phan_ky", "dat")}}
        if goc is not None:
            row["chenh_tb_don_vi_sd_goc"] = (M.mean(0) - goc.mean(0)) / goc.std(0)
            row["ty_so_sd_so_voi_goc"] = M.std(0) / goc.std(0)
        out["phuong_an"][pa] = row
    if goc is not None:
        out["goc"] = {"trung_binh": goc.mean(0), "do_lech_chuan": goc.std(0)}
    ctx.ghi("do_nhay_tien_nghiem.json", out)


Buoc("b08b", "do_nhay_tien_nghiem", "4.3.1", TEP, _lam_b08b, _th_b08b, don_vi=_dv_b08b, phu_thuoc=("b08",),
     cau_hinh=("khoi", "mle", "bayes", "do_nhay_tien_nghiem"), thu_vien=TV, can_khoa_KT=True)


# ---------------------------------------------------------------------------------------- b09
def _dv_b09(ctx):
    moc, _, _ = _moc(ctx)
    ds = [f"{pt}|{k:03d}::0" for pt in ctx.cfg["bayes"]["phuong_phap"] for k in range(len(moc))]
    if "nuts" in ctx.cfg["bayes"]["phuong_phap"]:
        ds += [f"ss|{k:03d}::0" for k in range(len(moc))]
    return ds


def _lam_b09(ctx, khoa, truoc):
    pt, k = khoa.split("::")[0].split("|")
    x = ctx.doc("bayes_cuon_chieu.pkl", "b08")[int(k)]
    if pt == "ss":                                    # mục 4.3.5 tại một mốc
        N = x["nuts"]["psi_mau"]
        sd_n, m_n, R_n = N.std(0), N.mean(0), np.corrcoef(N.T)
        spec, dl, m = _mle_khoi_tao(ctx, x["nuts"]["s0"])
        se = U.sai_so_chuan(spec, m["psi"], dl)
        row = {"s0": x["nuts"]["s0"], "ten": spec.ten(), "giay_nuts": x["nuts"]["giay"],
               "mle": {"ty_so_sd": se["se_banh_kep"] / sd_n, "lech_tb_don_vi_sd": (m["psi"] - m_n) / sd_n}}
        for p in ("vb_mf", "vb_fr"):
            if p in x:
                Q = x[p]["psi_mau"]
                row[p] = {"ty_so_sd": Q.std(0) / sd_n, "lech_tb_don_vi_sd": (Q.mean(0) - m_n) / sd_n,
                          "frobenius_tuong_quan": float(np.linalg.norm(np.corrcoef(Q.T) - R_n)),
                          "k_hat": x[p]["k_hat"], "giay": x[p]["giay"]}
        return row
    x = x[pt]
    kh = ctx.khoa()
    spec = DacTa(kh["phan_phoi"], tuple(kh["bien"]))
    s0, s1 = x["s0"], x["s1"]
    f = DL.dac_trung(ctx.du_lieu(), s0 - 1)
    dl = U.chuan_bi(f, spec.bien, s0)
    P = x["psi_mau"]
    P = P[np.linspace(0, len(P) - 1, min(len(P), ctx.cfg["bayes"]["so_mau_hau_nghiem"])).astype(int)]
    db = U.du_bao_hau_nghiem(spec, P, dl, s0, s1, ctx.cfg["danh_gia"]["alphas"],
                             ctx.cfg["danh_gia"]["mau_du_bao_hau_nghiem"], ctx.rng(khoa))
    return {"s0": s0, "s1": s1, "db": db, "rho_sao": f["rho_sao"][s0:s1]}


def nhom_phien(d, rho_sao: np.ndarray, dau: int, cuoi: int) -> dict:
    """Mục 5.6: quanh đáo hạn (tau <= 2 hoặc phiên chuyển), sau kỳ nghỉ dài, căng thẳng (rho* > P95 trượt 250 phiên)."""
    tau = d["tau"].to_numpy()[dau:cuoi]
    cang = np.zeros(cuoi - dau, bool)
    for i in range(cuoi - dau):
        qk = rho_sao[max(0, i - 250):i]
        qk = qk[np.isfinite(qk)]
        cang[i] = len(qk) > 50 and np.isfinite(rho_sao[i]) and rho_sao[i] > np.quantile(qk, 0.95)
    return {"quanh_dao_han": (tau <= 2) | d["J"].to_numpy()[dau:cuoi],
            "sau_nghi_dai": d["D_nghi"].to_numpy()[dau:cuoi], "cang_thang": cang}


def _qlike_chuoi(db, khoa):
    q = np.log(db["h"]) + db["rho_sao"] / db["h"]
    return np.where(np.isfinite(db["rho_sao"]) & ~khoa, q, np.nan)


def _th_b09(ctx, kq):
    _, dau, cuoi = _moc(ctx)
    kt0, kt1 = ctx.khoi()["KT"]
    a0 = kt0 - dau
    d = ctx.du_lieu()
    r, khoa = C.thuc_te(d, kt0, kt1)
    al = ctx.cfg["danh_gia"]["alphas"]
    kh = ctx.khoa()
    chon = f"{kh['phan_phoi']}|{C.ten_tap(kh['bien'])}"
    mo_hinh = {f"MLE|{k}": C.cat(v, a0, v["lnh"].shape[0]) for k, v in ctx.doc("mle_cuon_chieu.pkl", "b08").items()}
    for pt in ctx.cfg["bayes"]["phuong_phap"]:
        g = C.ghep([v for k, v in kq.items() if k.startswith(pt + "|")])
        mo_hinh[f"{pt}|{chon}"] = C.cat(g, a0, g["lnh"].shape[0])
    ctx.ghi("so_sanh_bon_phuong_phap.json", [v for k, v in sorted(kq.items()) if k.startswith("ss|")])
    lags = ctx.cfg["danh_gia"]["tre_phan_vi_dong"]
    bang = {k: DG.tom_tat(v, r, v["rho_sao"], khoa, al, lags=lags) for k, v in mo_hinh.items()}
    nm = nhom_phien(d, mo_hinh[f"MLE|{chon}"]["rho_sao"], kt0, kt1)
    theo_nhom = {k: {g: DG.tom_tat(v, r, v["rho_sao"], khoa, al, mask=m) for g, m in nm.items()} for k, v in mo_hinh.items()}
    ten = list(mo_hinh)
    mat = {"LS": {k: -v["ls"] for k, v in mo_hinh.items()}, "QLIKE": {k: _qlike_chuoi(v, khoa) for k, v in mo_hinh.items()}}
    for a in (0.01, 0.025):
        mat[f"FZ0_{a}"] = {k: DG.fz0_chuoi(v, r, a) for k, v in mo_hinh.items()}
    goc = f"MLE|{chon}"
    so_sanh = {ham: {k: {"DM": dict(zip(("t", "p_mot_phia_goc_tot_hon"), DG.dm(L[goc], L[k]))),
                         "GW": dict(zip(("thong_ke", "p"), DG.giacomini_white(L[goc], L[k])))}
                     for k in ten if k != goc} for ham, L in mat.items()}
    rng = ctx.rng("mcs")
    tin_cay = {ham: DG.mcs(np.column_stack([L[k] for k in ten]), ten, rng, ctx.cfg["danh_gia"]["muc_mcs"],
                           ctx.cfg["danh_gia"]["so_lan_bootstrap"], ctx.cfg["danh_gia"]["do_dai_khoi_tb"])
               for ham, L in mat.items()}
    egarch = f"MLE|{kh['phan_phoi']}|EGARCH"
    ctx.ghi("danh_gia_KT.json", {"khoi_KT": [kt0, kt1], "ngay": [str(d.date[kt0].date()), str(d.date[kt1 - 1].date())],
                                 "cau_hinh_chon": chon, "bang": bang, "nhom_phien": theo_nhom,
                                 "so_sanh_voi_cau_hinh_chon": so_sanh, "tap_mo_hinh_tin_cay": tin_cay,
                                 "CH1_EGARCH-X_vs_EGARCH": {ham: so_sanh[ham].get(egarch) for ham in mat}})
    ctx.ghi("du_bao_KT.pkl", {"mo_hinh": mo_hinh, "r": r, "khoa": khoa, "nhom": nm})


Buoc("b09", "danh_gia_KT", "4.3.5, 5", TEP, _lam_b09, _th_b09, don_vi=_dv_b09, phu_thuoc=("b08",),
     cau_hinh=("khoi", "mle", "bayes", "danh_gia"), thu_vien=TV, can_khoa_KT=True)
