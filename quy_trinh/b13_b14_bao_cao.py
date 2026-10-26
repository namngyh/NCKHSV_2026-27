"""b13 báo cáo (bảng Markdown, hình, phụ lục sai lệch); b15 kiểm toán; b14 đóng gói.

Mọi con số đọc thẳng từ các tệp kết quả của bước trước (không chép tay). b13 không chờ b03b (mô phỏng Bayes, chạy
nền); khi b03b xong, máy chủ trì chạy lại b13 - b15 để báo cáo có đủ mục 4.4.
b15 tính lại các con số chính từ tệp trung gian và so với báo cáo, kiểm tra không rò rỉ khối KT và nguồn gốc; có
chênh lệch là bước thất bại.
THƯ VIỆN: matplotlib (hình), zipfile.
"""
from __future__ import annotations

import shutil
import time
import zipfile

import numpy as np

import quy_trinh._chung as C  # noqa: F401
from de_cuong import danh_gia as DG
from de_cuong import ky_quy as KQ
from de_cuong.khung import GOC, Buoc, doc_ho_so

TEP = __file__
DANH = ("b00", "b01", "b02", "b02k", "b03a", "b03b", "b04", "b04b", "b05", "b06", "b06b", "b06c", "b07", "b08", "b08b", "b09", "b10",
        "b11", "b12")


def _f(x, n=4):
    return "" if x is None else (f"{x:.{n}f}" if isinstance(x, (float, np.floating)) else str(x))


def _bang(tieu_de, cot, hang):
    s = [f"| {' | '.join(tieu_de)} |", "|" + "---|" * len(tieu_de)]
    s += ["| " + " | ".join(_f(h.get(c)) if isinstance(h, dict) else "" for c in cot) + " |" for h in hang]
    return "\n".join(s)


# ---------------------------------------------------------------------------------------- phụ lục sai lệch
def phu_luc_sai_lech(ctx) -> str:
    cfg = ctx.cfg
    s = ["# Phụ lục: sai lệch so với đề cương và các sự kiện cần công bố", "",
         f"Hồ sơ chạy: `{ctx.ho_so}`" + (" (CHẠY THỬ, không phải kết quả nghiên cứu)." if ctx.ho_so != "day_du" else "."), ""]
    lech = []
    for k, v in cfg["de_cuong_goc"].items():
        muc, ten = k.split(".", 1)
        if cfg[muc][ten] != v:
            lech.append(f"- `{k}`: đề cương {v}, đang dùng {cfg[muc][ten]}")
    s += ["## Tham số khác đề cương chi tiết", "",
          "Giá trị tham chiếu là của đề cương chi tiết (`tai_lieu/de_cuong/`). Các điểm về khối UL/XT/KT và GED tham gia chọn phân phối",
          "là có chủ đích: nhóm bám đề cương đã đăng ký BM02, vốn khác đề cương chi tiết ở các điểm này.", ""] + (lech or ["- Không có."]) + [""]
    cong = []
    for ma in DANH:
        hs = doc_ho_so(ctx, ma) or {}
        if hs.get("chap_nhan_cong"):
            cong.append(f"- Bước {ma}: cổng không đạt nhưng được người dùng chủ động chấp nhận "
                        f"(chi tiết trong `cong.json` của bước).")
    s += ["## Cổng chuyển giai đoạn được chấp nhận khi không đạt", ""] + (cong or ["- Không có."]) + [""]
    tc = ctx.doc("cong.json", "b00") if ctx.co_tep("cong.json", "b00") else {}
    s += ["## Thông số thể chế (Bảng 3) chưa xác minh", ""] + (
        [f"- {x}" for x in tc.get("thong_so_chua_xac_minh", [])] or ["- Không có."]) + [""]
    tb = []
    for ma, ten in (("b03a", "mo_phong_khoi_phuc.json"), ("b03b", "mo_phong_bayes.json")):
        if ctx.co_tep(ten, ma):
            x = ctx.doc(ten, ma)
            for k, n in (x.get("so_lan_that_bai") or x.get("so_don_vi_that_bai") or {}).items():
                tb.append(f"- {ma} {k}: {n} đơn vị thất bại")
            for k in x.get("lan_lap_thieu_chuoi", []):
                tb.append(f"- {ma} {k}: thiếu chuỗi NUTS, lần lặp bị loại khỏi tổng hợp")
    s += ["## Đơn vị mô phỏng thất bại", ""] + (tb or ["- Không có."]) + [""]
    if ctx.co_tep("chan_doan_bayes.json", "b08"):
        cd = ctx.doc("chan_doan_bayes.json", "b08")
        s += ["## Chẩn đoán Bayes không đạt (b08)", "",
              f"- Các mốc NUTS không đạt (R_hat, ESS, phân kỳ): {cd.get('so_moc_nuts_khong_dat')}",
              f"- Pareto k_hat của VB đạt theo mốc: {cd.get('vb_k_hat_dat')}", ""]
    kq = cfg["ky_quy"]
    s += ["## Bổ sung ngoài đề cương (lựa chọn thiết kế của nhóm nghiên cứu, chốt trước khi mở khối KT)", "",
          f"- Vùng không điều chỉnh của tỷ lệ ký quỹ (bước 6): tăng khi mức mục tiêu cao hơn mức áp dụng từ {100 * kq['nguong_tang']:g} điểm %, "
          f"giảm khi thấp hơn từ {100 * kq['nguong_giam']:g} điểm %; đối chứng không ngưỡng và hai mức ngưỡng khác ở b12. BM02 chỉ nêu \"giới hạn thay đổi giữa hai phiên\".",
          f"- Đối chứng cách tính của VSDC (QĐ 61/QĐ-VSD 2022, QĐ 12/QĐ-HĐTV 2023, QĐ 26/QĐ-HĐTV 2025), n = {cfg['vsdc']['n']}, kỳ quan sát 90/120/250 ngày, "
          "cập nhật hằng ngày và theo lịch 1, 10, 20 (b12). Đây là tính lại theo văn bản, không phải chuỗi VSDC công bố.",
          "- Quy đổi ký quỹ và thiếu hụt ra đồng mỗi hợp đồng bằng hệ số nhân 100.000 đồng/điểm (b12).", "",
          "## Thay đổi sau khi khối KT đã được nhìn thấy và các sự kiện cần công bố", ""]
    s += [f"- {x}" for x in cfg.get("cong_bo", {}).get("su_kien", [])] or ["- Không có."]
    s += ["", "## Lựa chọn của nhóm nghiên cứu ở những chỗ đề cương không quy định", "",
          "Xem `cau_hinh/de_cuong.toml` (chú thích từng giá trị) và THIET_KE_CODE.md mục 6b.", ""]
    return "\n".join(s)


# ---------------------------------------------------------------------------------------- b13
def _bang_khoi(ctx) -> dict:
    """Bảng 7 tính lại từ cấu hình hiện hành (kiem_tra_du_lieu.json của b01 không phụ thuộc XT nên có thể cũ)."""
    d = ctx.du_lieu()
    return {k: {"tu": str(d.date[v[0]].date()), "den": str(d.date[v[1] - 1].date()), "so_phien": v[1] - v[0],
                "so_phien_chuyen": int(d.J[v[0]:v[1]].sum())} for k, v in ctx.khoi().items()}


def _lam_b13(ctx, khoa, truoc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tm = ctx.thu_muc()
    tm.mkdir(parents=True, exist_ok=True)
    md = [f"# Kết quả theo đề cương EGARCH-X ký quỹ động (hồ sơ: {ctx.ho_so})", "",
          f"Sinh tự động lúc {time.strftime('%Y-%m-%d %H:%M')}. Hồ sơ `nhanh` chỉ để kiểm tra đường ống.", ""]
    q = ctx.doc("kiem_tra_du_lieu.json", "b01")
    md += ["## 1. Dữ liệu (mục 2)", "", f"{q['so_phien']} phiên, {q['tu']} – {q['den']}; "
           f"{len(q['phien_chuyen_hop_dong'])} phiên chuyển hợp đồng; {len(q['phien_cham_bien_khoa'])} phiên khóa biên; "
           f"c_rho (UL) = {_f(q['dai_dien_phuong_sai']['c_rho_UL'])}.", "",
           _bang(["Khối", "Từ", "Đến", "Số phiên", "Phiên chuyển"], ["k", "tu", "den", "so_phien", "so_phien_chuyen"],
                 [{"k": k, **v} for k, v in _bang_khoi(ctx).items()]), ""]
    kc = ctx.doc("kiem_tra_cai_dat.json", "b02")
    da = kc["doi_chieu_arch"]
    md += ["## 2. Kiểm tra cài đặt và đối chiếu độc lập (Phụ lục A, C.5)", "",
           f"Bảng 11 lệch lớn nhất {_f(kc['bang_11']['lech_lon_nhat'], 5)}; đối chiếu gói arch: EGARCH chênh tham số lớn nhất "
           f"{_f(float(np.max(da['egarch_chuan']['chenh_tuyet_doi'])), 4)}, MCS trùng tập = {da['mcs']['dat']}, "
           f"SPA cùng kết luận = {da['spa']['dat']}.", ""]
    cl = ctx.doc("co_luc_kiem_dinh.json", "b02k")
    md += ["### Cỡ và lực của các kiểm định (mô phỏng, N = khối KT)", "",
           _bang(["Kiểm định", "Tỷ lệ bác bỏ 5%", "Sai số MC"], ["k", "t", "s"],
                 [{"k": k, "t": v["ty_le_bac_bo_5pct"], "s": v["sai_so_chuan_mc"]} for k, v in sorted(cl.items())]), ""]
    mp = ctx.doc("mo_phong_khoi_phuc.json", "b03a")
    md += ["## 3. Mô phỏng khôi phục tham số bằng MLE (mục 4.4)", "", _bang(
        ["Cấu hình", "theta chệch", "theta độ phủ", "beta độ phủ", "bác bỏ theta=0", "bác bỏ gamma=0"],
        ["k", "tc", "tp", "bp", "bt", "bg"],
        [{"k": k, "tc": v["tham_so"]["theta"]["chech"], "tp": v["tham_so"]["theta"]["do_phu"],
          "bp": v["tham_so"]["beta"]["do_phu"], "bt": v["bac_bo_theta0"], "bg": v["bac_bo_gamma0"]}
         for k, v in sorted(mp["bang"].items())]), ""]
    if ctx.co_tep("mo_phong_bayes.json", "b03b"):
        mb = ctx.doc("mo_phong_bayes.json", "b03b")
        md += ["### Mô phỏng khôi phục bằng VB và NUTS", "", _bang(
            ["Cấu hình", "theta độ phủ", "beta độ phủ", "omega độ phủ", "chẩn đoán đạt", "bác bỏ theta=0"],
            ["k", "tp", "bp", "op", "cd", "bt"],
            [{"k": k, "tp": v["tham_so"]["theta"]["do_phu"], "bp": v["tham_so"]["beta"]["do_phu"],
              "op": v["tham_so"]["omega"]["do_phu"], "cd": v.get("chan_doan_dat"), "bt": v["bac_bo_theta0"]}
             for k, v in sorted(mb["bang"].items())]), ""]
        sb = ctx.doc("sbc.json", "b03b")
        md += ["### Hiệu chuẩn dựa trên mô phỏng (p của kiểm định đều cho thứ hạng)", "",
               _bang(["Cấu hình", "beta", "omega", "theta"], ["k", "b", "o", "t"],
                     [{"k": k, "b": v["tham_so"]["beta"]["chi2_p"], "o": v["tham_so"]["omega"]["chi2_p"],
                       "t": v["tham_so"]["theta"]["chi2_p"]} for k, v in sorted(sb.items())]), ""]
    else:
        md += ["*Mô phỏng Bayes và SBC (b03b) đang chạy; báo cáo sẽ được làm lại khi xong.*", ""]
    ml = ctx.doc("mle_UL.json", "b04")
    md += ["## 4. MLE trên khối UL (mục 4.2)", ""]
    for pp, v in ml.items():
        md += [f"**{v['cau_hinh']}**: logL = {_f(v['ll'], 2)}, AIC = {_f(v['aic'], 2)}, BIC = {_f(v['bic'], 2)}, "
               f"LR(gamma=0) p = {_f(v.get('LR_p'))}, {v['so_gan_tot']}/{v['so_diem']} điểm khởi tạo về cùng nghiệm.", "",
               _bang(["Tham số", "Ước lượng", "SE Hessian", "SE bánh kẹp"], ["t", "p", "h", "s"],
                     [{"t": t, "p": p, "h": h, "s": s} for t, p, h, s in zip(v["ten"], v["psi"], v["se_hessian"], v["se_banh_kep"])]), ""]
    tn = ctx.doc("du_bao_tien_nghiem.json", "b04b")
    md += ["### Kiểm tra dự báo tiên nghiệm (mục 4.3.1): phân vị của thống kê dữ liệu trong phân phối dự báo tiên nghiệm", "",
           _bang(["Phân phối", "độ lệch chuẩn", "độ nhọn", "max |r|", "chạm chặn ln h"], ["k", "s", "n", "m", "c"],
                 [{"k": k, "s": v["tien_nghiem"]["do_lech_chuan"]["phan_vi_cua_du_lieu"],
                   "n": v["tien_nghiem"]["nhon"]["phan_vi_cua_du_lieu"],
                   "m": v["tien_nghiem"]["max_tuyet_doi"]["phan_vi_cua_du_lieu"], "c": v["ty_le_cham_can_lnh"]}
                  for k, v in tn.items()]), ""]
    cb = ctx.doc("chon_bien.json", "b05")
    cp = ctx.doc("chon_phan_phoi.json", "b06")
    md += ["## 5. Chọn biến và phân phối trên XT (mục 6)", "",
           f"Tập biến được chọn: **{cb['ten_chon']}**; SPA p = {_f(cb['SPA']['p'])}; Holm: {cb['LR_Holm_p']}; "
           f"VIF: {cb['VIF_tap_chon']}.", f"Phân phối được chọn: **{cp['phan_phoi']}**.", ""]
    cg = ctx.doc("cong.json", "b07")
    md += ["## 6. Cổng chuyển giai đoạn (mục 8.1)", "",
           f"(i) {cg['i_mo_phong'].get('dat')}; (ii) {cg['ii_z2_theo_tau']['dat']}; (iii) {cg['iii_VaR_XT']['dat']}; "
           f"tổng: **{cg['dat']}**.", ""]
    dn = ctx.doc("do_nhay_tien_nghiem.json", "b08b")
    md += ["## 7. Độ nhạy tiên nghiệm (mục 4.3.1): chênh lệch trung bình hậu nghiệm theo đơn vị độ lệch chuẩn gốc", "",
           _bang(["Phương án"] + dn["ten"], ["k"] + dn["ten"],
                 [{"k": pa, **dict(zip(dn["ten"], v.get("chenh_tb_don_vi_sd_goc", [None] * len(dn["ten"]))))}
                  for pa, v in dn["phuong_an"].items()]), ""]
    dg = ctx.doc("danh_gia_KT.json", "b09")
    hang = [{"k": k, "q": v["QLIKE"], "ls": v["LS"], "v1": v.get("a0.01", {}).get("vi_pham"),
             "p1": v.get("a0.01", {}).get("kupiec_p"), "fz": v.get("a0.025", {}).get("FZ0"), "bk": v.get("berkowitz_p")}
            for k, v in dg["bang"].items()]
    md += [f"## 8. Đánh giá ngoài mẫu trên KT ({dg['ngay'][0]} – {dg['ngay'][1]}) (mục 5)", "",
           _bang(["Mô hình", "QLIKE", "LS", "Vi phạm 1%", "Kupiec p", "FZ0 2,5%", "Berkowitz p"],
                 ["k", "q", "ls", "v1", "p1", "fz", "bk"], hang), "",
           "Tập mô hình tin cậy: " + str({h: v["tap_tin_cay"] for h, v in dg["tap_mo_hinh_tin_cay"].items()}), "",
           "CH1 (EGARCH-X so với EGARCH): " + str({h: (v or {}).get("DM") for h, v in dg["CH1_EGARCH-X_vs_EGARCH"].items()}), ""]
    ss = ctx.doc("so_sanh_bon_phuong_phap.json", "b09")
    if ss:
        x = ss[-1]
        md += [f"### So sánh bốn phương pháp tại mốc cuối (s0 = {x['s0']}): tỷ số độ lệch chuẩn so với NUTS", "",
               _bang(["Phương pháp"] + x["ten"], ["k"] + x["ten"],
                     [{"k": p, **dict(zip(x["ten"], x[p]["ty_so_sd"]))} for p in ("mle", "vb_mf", "vb_fr") if p in x]), ""]
    kq = ctx.doc("ky_quy.json", "b11")
    key = kq["trinh_bay"]
    t = kq["thiet_ke"][key]
    md += [f"## 9. Ký quỹ động (mục 7), thiết kế trình bày {key}", "", f"Siêu tham số chọn trên XT: {t['chon']['mle']}.", "",
           _bang(["Quy tắc", "m tb", "Tỷ lệ vượt", "Thiếu hụt tb", "Cùng m_tb: thiếu hụt/phiên", "DM p"],
                 ["k", "m", "v", "th", "cb", "p"],
                 [{"k": k, "m": r["m_tb"], "v": r["ty_le_vuot"], "th": r["thieu_hut_tb"],
                   "cb": r["cung_m_tb"]["thieu_hut_moi_phien"],
                   "p": r["cung_m_tb"]["DM_thieu_hut_moi_phien_dong_mle_thap_hon"]["p_mot_phia"]} for k, r in t["KT"].items()]), ""]
    ch = ctx.doc("chuoi_ky_quy.pkl", "b11")[key]
    o = ch["hop_kt"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ch["L"]))
    ax.plot(x[o], ch["L"][o], ".", ms=2, color="0.6", label="|R| thực tế")
    for ten, mau in (("dong_mle", "C0"), ("co_dinh_khop_XT", "C3"), ("EWMA", "C2")):
        ax.plot(x, np.where(x >= np.argmax(o), ch[ten], np.nan), lw=1, color=mau, label=ten)
    ax.set_xlabel("phiên gốc"); ax.set_ylabel("tỷ lệ giá trị hợp đồng"); ax.legend(); fig.tight_layout()
    fig.savefig(tm / "hinh_ky_quy.png", dpi=150); plt.close(fig)
    bien, luoi = t["duong_bien_XT"]["mle"], t["luoi_XT"]["mle"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([v["m_tb"] for v in luoi], [v["thieu_hut_moi_phien"] for v in luoi], ".", color="0.7")
    ax.plot([v["m_tb"] for v in bien], [v["thieu_hut_moi_phien"] for v in bien], "-o", ms=3)
    ax.set_xlabel("ký quỹ trung bình"); ax.set_ylabel("thiếu hụt mỗi phiên"); ax.set_title("Đường biên trên XT")
    fig.tight_layout(); fig.savefig(tm / "hinh_duong_bien.png", dpi=150); plt.close(fig)
    md += ["![Ký quỹ](hinh_ky_quy.png)", "", "![Đường biên](hinh_duong_bien.png)", ""]
    return {"md": "\n".join(md), "phu_luc": phu_luc_sai_lech(ctx)}


def _th_b13(ctx, kq):
    ctx.ghi("bao_cao_ket_qua.md", kq["tat_ca::0"]["md"])
    ctx.ghi("phu_luc_sai_lech.md", kq["tat_ca::0"]["phu_luc"])
    dich = GOC / "bao_cao" / ctx.ho_so
    if dich.exists():       # OneDrive/Explorer có thể giữ thư mục: xóa tệp con, không xóa chính thư mục, rồi ghi đè
        for p in sorted(dich.rglob("*"), key=lambda q: len(q.parts), reverse=True):
            try:
                p.unlink() if p.is_file() or p.is_symlink() else p.rmdir()
            except OSError:
                pass
    shutil.copytree(ctx.thu_muc(), dich, dirs_exist_ok=True)


Buoc("b13", "bao_cao", "8.3", TEP, _lam_b13, _th_b13, phu_thuoc=("b12", "b02k", "b04b", "b08b"),
     cau_hinh=("cong_bo", "de_cuong_goc"),        # phụ lục sai lệch đọc hai mục này: sửa chúng phải làm lại báo cáo
     song_song=False, can_khoa_KT=True, chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b15
def _lech(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def _lam_b15(ctx, khoa, truoc):
    """Kiểm toán: tính lại con số từ tệp trung gian, so với tệp báo cáo; kiểm tra rò rỉ khối KT và nguồn gốc."""
    al = ctx.cfg["danh_gia"]["alphas"]
    out = {}
    dk = ctx.doc("du_bao_KT.pkl", "b09")
    dg = ctx.doc("danh_gia_KT.json", "b09")["bang"]
    m09 = 0.0
    for k, db in dk["mo_hinh"].items():
        t = DG.tom_tat(db, dk["r"], db["rho_sao"], dk["khoa"], al, lags=ctx.cfg["danh_gia"]["tre_phan_vi_dong"])
        m09 = max(m09, _lech(t["QLIKE"], dg[k]["QLIKE"]), _lech(t["LS"], dg[k]["LS"]),
                  *[_lech(t[f"a{a}"]["vi_pham"], dg[k][f"a{a}"]["vi_pham"]) for a in al],
                  *[_lech(t[f"a{a}"]["kupiec_p"], dg[k][f"a{a}"]["kupiec_p"]) for a in al])
    out["b09_lech_lon_nhat"] = m09
    ch = ctx.doc("chuoi_ky_quy.pkl", "b11")
    tk = ctx.doc("ky_quy.json", "b11")["thiet_ke"]
    m11 = 0.0
    for key, s in ch.items():
        H = tk[key]["H"]
        for ten, row in tk[key]["KT"].items():
            c = KQ.chi_tieu(s[ten], s["L"], s["hop_kt"], H)
            m11 = max(m11, _lech(c["m_tb"], row["m_tb"]), _lech(c["ty_le_vuot"], row["ty_le_vuot"]),
                      _lech(c["thieu_hut_tb"], row["thieu_hut_tb"]))
    out["b11_lech_lon_nhat"] = m11
    xt = ctx.doc("du_bao_XT.pkl", "b05")
    ls = ctx.doc("chon_bien.json", "b05")["LS_moi_ung_vien"]
    out["b05_lech_lon_nhat"] = max(_lech(np.nanmean(v["ls"]), ls[k]) for k, v in xt.items())
    kt0 = ctx.khoi()["KT"][0]
    out["b05_b06_chi_truoc_KT"] = bool(all(v["cuoi"] <= kt0 for v in xt.values()) and
                                      all(v["cuoi"] <= kt0 for v in ctx.doc("du_bao_XT.pkl", "b06").values()))
    hs07, hs08 = doc_ho_so(ctx, "b07") or {}, doc_ho_so(ctx, "b08") or {}
    out["khoa_truoc_b08"] = bool(hs07.get("ket_thuc") and hs08.get("bat_dau") and hs07["ket_thuc"] <= hs08["bat_dau"])
    commit, thieu = {}, []
    for ma in DANH + ("b13",):
        if ctx.co_tep("nguon_goc.json", ma):
            commit.setdefault(ctx.doc("nguon_goc.json", ma)["commit"], []).append(ma)
        elif (doc_ho_so(ctx, ma) or {}).get("trang_thai") == "xong":
            thieu.append(ma)
    out["commit_theo_buoc"] = commit
    out["buoc_thieu_nguon_goc"] = thieu
    out["dat"] = bool(m09 < 1e-9 and m11 < 1e-9 and out["b05_lech_lon_nhat"] < 1e-9 and out["b05_b06_chi_truoc_KT"]
                      and out["khoa_truoc_b08"] and not thieu)
    return out


def _th_b15(ctx, kq):
    x = kq["tat_ca::0"]
    ctx.ghi("kiem_toan.json", x)
    if not x["dat"]:
        raise RuntimeError("KIEM TOAN KHONG DAT (xem kiem_toan.json)")


Buoc("b15", "kiem_toan", "8.5, C.7", TEP, _lam_b15, _th_b15, phu_thuoc=("b13",), cau_hinh=("danh_gia",),
     song_song=False, can_khoa_KT=True, chi_chu_tri=True, thu_vien=("danh_gia", "ky_quy"))


# ---------------------------------------------------------------------------------------- b14
def _lam_b14(ctx, khoa, truoc):
    dich = GOC / "dong_goi"
    dich.mkdir(exist_ok=True)
    ten = dich / f"NCKHSV_EGARCHX_{ctx.ho_so}_{time.strftime('%Y%m%d_%H%M')}.zip"
    with zipfile.ZipFile(ten, "w", zipfile.ZIP_DEFLATED) as z:
        for goc in ("cau_hinh", "ma_nguon", "quy_trinh", "kiem_thu", "colab", "tai_lieu/de_cuong", f"bao_cao/{ctx.ho_so}"):
            if (GOC / goc).exists():
                for p in (GOC / goc).rglob("*"):
                    if p.is_file() and "__pycache__" not in p.parts and p.name != "python.txt":
                        z.write(p, p.relative_to(GOC))
        for p in ctx.ket_qua.rglob("*"):
            if p.is_file() and "_diem_kiem_tra" not in p.parts:
                z.write(p, p.relative_to(GOC))
        for f in ("chay.bat", "requirements.txt", "README.md", "THIET_KE_CODE.md", "GHI_CHU_DU_AN.md"):
            if (GOC / f).exists():
                z.write(GOC / f, f)
    return {"tep": str(ten)}


def _th_b14(ctx, kq):
    ctx.ghi("dong_goi.json", kq["tat_ca::0"])
    ctx.log(f"Da dong goi: {kq['tat_ca::0']['tep']}")


Buoc("b14", "dong_goi", "8.3, 8.5", TEP, _lam_b14, _th_b14, phu_thuoc=("b15",), song_song=False, can_khoa_KT=True,
     chi_chu_tri=True)
