"""Các bước chuẩn bị: b00 thông số thể chế, b01 dữ liệu, b02 kiểm tra cài đặt.

b00 (mục 2.2, Bảng 3, Phụ lục C.1): đọc cau_hinh/the_che.toml; cổng không đạt ở hồ sơ "day_du" nếu còn thông số
    chưa có nguồn hoặc ngày xác minh.
b01 (mục 2, 2.8, Phụ lục C.2 - C.3): bảng ngày, phiên chuyển hợp đồng, đáo hạn, nghỉ dài, chạm biên; kiểm tra
    chất lượng dữ liệu. Được thấy cả khối KT vì chỉ kiểm tra chất lượng, không lựa chọn gì.
b02 (Phụ lục C.5, Phụ lục A): tích phân số của bốn phân phối, trường hợp giới hạn, Bảng 11, và khớp số giữa hàm
    log-hợp lý JAX với bản numba độc lập. Không đạt thì bước thất bại.
THƯ VIỆN: numpy, scipy.integrate.
"""
from __future__ import annotations

import math

import numpy as np

import quy_trinh._chung as C  # noqa: F401  (đặt đường dẫn)
from de_cuong import du_lieu as DL
from de_cuong.khung import Buoc

TEP = __file__


# ---------------------------------------------------------------------------------------- b00
def _cong_b00(ctx):
    thieu = [k for k, v in ctx.cfg["the_che"].items() if not v.get("nguon") or not v.get("ngay_xac_minh")]
    dat = not thieu or ctx.ho_so != "day_du"
    return {"dat": dat, "thong_so_chua_xac_minh": thieu,
            "ghi_chu": "Hồ sơ nhanh chỉ cảnh báo." if thieu and ctx.ho_so != "day_du" else ""}


def _lam_b00(ctx, khoa, truoc):
    return {"the_che": ctx.cfg["the_che"]}


def _th_b00(ctx, kq):
    ctx.ghi("the_che.json", kq["tat_ca::0"])


Buoc("b00", "the_che", "2.2, Bảng 3, C.1", TEP, _lam_b00, _th_b00, cong=_cong_b00, song_song=False,
     cau_hinh=("the_che",), chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b01
def _lam_b01(ctx, khoa, truoc):
    d = ctx.du_lieu()
    k = ctx.khoi()
    n = len(d)
    ohlc = (d.low > np.minimum(d.open, d.close)) | (d.high < np.maximum(d.open, d.close))
    lech_v = (d.vol - d.buy - d.sell).abs() / d.vol
    nguong = ctx.cfg["du_lieu"]["nguong_r_bat_thuong"]
    lon = d.loc[d.r.abs() > nguong, ["date", "r", "open", "high", "low", "close"]]
    dem = 100 * np.log(d.open / d.close.shift(1))
    rr = d.r.dropna().to_numpy()
    gap = [int(np.busday_count(d.date[t - 1].date(), d.date[t].date())) - 1 for t in range(1, n)]
    f = DL.dac_trung(d, k["UL"][1] - 1)
    ok = np.isfinite(d.rho_gk) & np.isfinite(d.r)
    return {
        "so_phien": n, "tu": str(d.date.iloc[0].date()), "den": str(d.date.iloc[-1].date()),
        "OHLC_vi_pham": int(ohlc.sum()), "phien_trung": int(d.date.duplicated().sum()),
        "V_lech_tren_1pct": int((lech_v > 0.01).sum()),
        "V_lech_tren_20pct_ngay": [str(x.date()) for x in d.date[lech_v > 0.2]],
        "khoang_trong_ngay_lam_viec": {"so_khoang": int(sum(g > 0 for g in gap)), "dai_nhat": int(max(gap))},
        f"r_vuot_{nguong}pct_ngoai_phien_chuyen": [{"ngay": str(x.date.date()), "r": float(x.r)} for x in lon.itertuples()],
        "phien_chuyen_hop_dong": [str(x.date()) for x in d.date[d.J]],
        "phien_dao_han": [str(x.date()) for x in d.date[d.D_dh]],
        "phien_sau_nghi_dai": [str(x.date()) for x in d.date[d.D_nghi]],
        "phien_cham_bien_khoa": [str(x.date()) for x in d.date[d.khoa_bien]],
        "qua_dem_trung_vi_tuyet_doi": {"phien_chuyen": float(dem[d.J].abs().median()),
                                      "phien_thuong": float(dem[~d.J].abs().median())},
        "r_mo_ta": {"n": len(rr), "trung_binh": rr.mean(), "phuong_sai": rr.var(ddof=1),
                    "lech": float(((rr - rr.mean()) ** 3).mean() / rr.std() ** 3),
                    "nhon": float(((rr - rr.mean()) ** 4).mean() / rr.std() ** 4)},
        "dai_dien_phuong_sai": {"c_rho_UL": f["c_rho"],
                                "tuong_quan_voi_r2": {c: float(np.corrcoef(d[c][ok], d.r[ok] ** 2)[0, 1])
                                                       for c in ("rho_p", "rho_gk", "rho_rs")}},
        "khoi": {kk: {"chi_so": v, "tu": str(d.date[v[0]].date()), "den": str(d.date[v[1] - 1].date()),
                      "so_phien": v[1] - v[0], "so_phien_chuyen": int(d.J[v[0]:v[1]].sum())} for kk, v in k.items()},
        "bang": d,
    }


def _th_b01(ctx, kq):
    x = dict(kq["tat_ca::0"])
    d = x.pop("bang")
    ctx.ghi("kiem_tra_du_lieu.json", x)
    ctx.ghi("ngay.csv", d)


Buoc("b01", "du_lieu", "2, 2.8, C.2, C.3", TEP, _lam_b01, _th_b01, phu_thuoc=("b00",),
     cau_hinh=("chung", "du_lieu", "khoi"), thu_vien=("du_lieu",), truy_cap_KT=True, song_song=False,
     chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b02
BANG_11 = {("chuan", (), "mua"): [2.326, 2.665, 1.960, 2.338, 1.645, 2.063],
           ("student", (5.0,), "mua"): [2.606, 3.449, 1.991, 2.728, 1.561, 2.239],
           ("skewt", (5.0, -0.1), "mua"): [2.783, 3.720, 2.102, 2.919, 1.627, 2.377],
           ("skewt", (5.0, -0.1), "ban"): [2.416, 3.159, 1.871, 2.522, 1.488, 2.089]}


def _lam_b02(ctx, khoa, truoc):
    import jax.numpy as jnp
    from scipy import integrate

    from de_cuong import mo_hinh as NB
    from de_cuong import mo_hinh_jax as M
    from de_cuong import phan_phoi as P
    from de_cuong import uoc_luong as U
    kq, dat = {}, True
    # (C.5) tích phân số: mật độ có tổng 1, kỳ vọng 0, phương sai 1; E|z| JAX khớp cầu phương
    for pp, sh in (("chuan", []), ("student", [4.5]), ("skewt", [4.5, -0.15]), ("skewt", [6.0, 0.3]), ("ged", [1.2])):
        fz = lambda x: math.exp(float(M.log_mat_do(jnp.array(x), pp, jnp.array(sh))))
        mom = [integrate.quad(lambda x: x ** j * fz(x), -np.inf, np.inf, limit=300)[0] for j in (0, 1, 2)]
        ea = abs(float(M.e_abs(pp, jnp.array(sh))) - P.e_abs(pp, sh))
        ok = abs(mom[0] - 1) < 1e-6 and abs(mom[1]) < 1e-6 and abs(mom[2] - 1) < 1e-6 and ea < 1e-8
        kq[f"tich_phan_{pp}_{sh}"] = {"moment": mom, "chenh_E|z|": ea, "dat": ok}
        dat &= ok
    # trường hợp giới hạn: nu -> vô cùng cho chuẩn; lambda = 0 cho Student-t
    zz = jnp.linspace(-4, 4, 41)
    gh1 = abs(float(M.e_abs("student", jnp.array([1e7]))) - math.sqrt(2 / math.pi))
    gh2 = float(jnp.max(jnp.abs(M.log_mat_do(zz, "skewt", jnp.array([5.0, 0.0])) - M.log_mat_do(zz, "student", jnp.array([5.0])))))
    kq["gioi_han"] = {"t_nu_vo_cung_E|z|": gh1, "skewt_lambda0_vs_t": gh2, "dat": gh1 < 1e-6 and gh2 < 1e-10}
    dat &= kq["gioi_han"]["dat"]
    # Phụ lục A, Bảng 11
    lech = 0.0
    for (pp, sh, ben), ky_vong in BANG_11.items():
        tinh = []
        for a in (0.01, 0.025, 0.05):
            if ben == "mua":
                tinh += [-float(P.phan_vi(a, pp, sh)), -P.ky_vong_duoi_trai(a, pp, sh)]
            else:
                tinh += [float(P.phan_vi(1 - a, pp, sh)), P.ky_vong_duoi_phai(a, pp, sh)]
        lech = max(lech, float(np.max(np.abs(np.array(tinh) - ky_vong))))
    kq["bang_11"] = {"lech_lon_nhat": lech, "dat": lech < 0.0015}
    dat &= kq["bang_11"]["dat"]
    # khớp số JAX - numba trên khối UL, cấu hình chính
    d = ctx.du_lieu()
    s = ctx.khoi()["UL"][1]
    f = DL.dac_trung(d, s - 1)
    r = np.nan_to_num(f["r"]); J = np.asarray(f["J"], bool) | np.isnan(f["r"])
    lh0 = math.log(float(np.nanvar(f["r"][DL.BO_QUA:DL.BO_QUA + 20], ddof=1)))
    bq = DL.BO_QUA
    for pp, sh in (("chuan", []), ("student", [5.0]), ("skewt", [5.0, -0.1])):
        psi = np.array([0.02, 0.1, 0.15, -0.15, 0.8, -0.05, 0.3, 0.5, 0.1, 1.0] + sh)
        cfgnb = NB.Cau_hinh(pp, (0, 1))
        Z = NB.tao_Z(cfgnb, f)
        ll_nb = -NB.am_log_hop_ly(psi, cfgnb, r[bq:s], J[bq:s], Z[bq:s], lh0)
        spec = M.DacTa(pp, ("A", "G"))
        ll_jx = float(M.ham(pp, 2).ll(jnp.asarray(psi), U.chuan_bi(f, spec.bien, s)))
        kq[f"khop_numba_{pp}"] = {"numba": ll_nb, "jax": ll_jx, "chenh": abs(ll_nb - ll_jx), "dat": abs(ll_nb - ll_jx) < 1e-6}
        dat &= kq[f"khop_numba_{pp}"]["dat"]
    # đối chiếu độc lập với gói arch (Sheppard): EGARCH chuẩn, tập mô hình tin cậy, SPA
    kq["doi_chieu_arch"] = _doi_chieu_arch(ctx)
    dat &= kq["doi_chieu_arch"]["dat"]
    kq["dat"] = bool(dat)
    return kq


def _doi_chieu_arch(ctx) -> dict:
    import jax.numpy as jnp
    from arch import arch_model
    from arch.bootstrap import MCS, SPA

    from de_cuong import danh_gia as DG
    from de_cuong import mo_hinh_jax as M
    from de_cuong import uoc_luong as U
    out = {}
    rng = np.random.default_rng(20261020)
    # (1) EGARCH(1,1) phân phối chuẩn, không biến ngoại sinh, không phiên khuyết: cùng đặc tả với arch
    spec = M.DacTa("chuan", ())
    psi0 = np.array([0.05, 0.04, 0.15, -0.10, 0.95, 0.0, 0.0, 0.0])
    n = 3000
    Z, J = np.zeros((n, 3)), np.zeros(n, bool)
    r, _ = U.sinh_tu_mo_hinh(spec, psi0, Z, J, 0.8, rng)
    dl = (jnp.asarray(r), jnp.asarray(J), jnp.asarray(Z), jnp.asarray(~J), jnp.asarray(math.log(np.var(r[:20], ddof=1))))
    m = U.mle(spec, dl, 8, 0.3, rng)
    a = arch_model(r, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")
    ta = np.array([a.params["mu"], a.params["omega"], a.params["alpha[1]"], a.params["gamma[1]"], a.params["beta[1]"]])
    chenh = np.abs(m["psi"][:5] - ta)
    out["egarch_chuan"] = {"ta": m["psi"][:5], "arch": ta, "chenh_tuyet_doi": chenh, "dat": bool(chenh.max() < 0.03)}
    # (2) tập mô hình tin cậy: 6 mô hình, 3 tốt như nhau, 3 kém rõ rệt
    base = rng.normal(size=(800, 1))
    L = base + rng.normal(scale=0.5, size=(800, 6)) + np.array([0, 0, 0, 0.4, 0.6, 1.0])
    ten = [f"m{i}" for i in range(6)]
    ta_mcs = set(DG.mcs(L, ten, np.random.default_rng(1), 0.10, 1000, 5)["tap_tin_cay"])
    mc = MCS(L, size=0.10, reps=1000, block_size=5, method="max", seed=1)
    mc.compute()
    arch_mcs = {ten[i] for i in mc.included}
    out["mcs"] = {"ta": sorted(ta_mcs), "arch": sorted(arch_mcs), "dat": ta_mcs == arch_mcs}
    # (3) SPA (mất mát, nhỏ hơn là tốt hơn): có ứng viên tốt hơn rõ rệt -> bác bỏ; mọi ứng viên kém hơn -> không bác bỏ
    kq_spa = []
    for lech, co_tot_hon in (([0.2, 0.4, -0.3], True), ([0.2, 0.4, 0.1], False)):
        b0 = rng.normal(size=600)
        mods = b0[:, None] + rng.normal(scale=0.5, size=(600, 3)) + np.array(lech)[None, :]
        p_ta = DG.spa(b0[:, None] - mods, np.random.default_rng(2), 1000, 5)["p"]
        s = SPA(b0, mods, reps=1000, block_size=5, seed=2)
        s.compute()
        p_arch = float(s.pvalues["consistent"])
        kq_spa.append({"co_ung_vien_tot_hon": co_tot_hon, "p_ta": p_ta, "p_arch": p_arch,
                       "cung_ket_luan": (p_ta < 0.05) == (p_arch < 0.05), "dung_ky_vong": (p_ta < 0.05) == co_tot_hon})
    out["spa"] = {"truong_hop": kq_spa, "dat": all(x["cung_ket_luan"] and x["dung_ky_vong"] for x in kq_spa)}
    out["dat"] = bool(out["egarch_chuan"]["dat"] and out["mcs"]["dat"] and out["spa"]["dat"])
    return out


def _th_b02(ctx, kq):
    x = kq["tat_ca::0"]
    ctx.ghi("kiem_tra_cai_dat.json", x)
    if not x["dat"]:
        raise RuntimeError("Kiem tra cai dat KHONG DAT (xem kiem_tra_cai_dat.json)")


Buoc("b02", "kiem_tra_cai_dat", "C.5, Phụ lục A", TEP, _lam_b02, _th_b02, phu_thuoc=("b01",),
     cau_hinh=("du_lieu", "khoi"), thu_vien=("mo_hinh_jax", "phan_phoi", "mo_hinh", "uoc_luong", "du_lieu", "danh_gia"),
     song_song=False, chi_chu_tri=True)


# ---------------------------------------------------------------------------------------- b02k
def _dv_b02k(ctx):
    ds = []
    for a in ctx.cfg["danh_gia"]["alphas"]:
        ds += [f"kupiec|a{a}|co::0", f"kupiec|a{a}|luc::0", f"christoffersen|a{a}|co::0", f"dq|a{a}|co::0"]
        ds += [f"dq_hac|a{a}|H{H}::0" for H in ctx.cfg["ky_quy"]["H"] if H > 1]
    return ds + ["berkowitz|co::0", "dm|iid::0", "dm|ar1::0", "gw|iid::0"]


def _lam_b02k(ctx, khoa, truoc):
    """Tỷ lệ bác bỏ ở mức 5% qua R lần mô phỏng, cỡ mẫu N bằng khối KT: 'co' là cỡ (dưới H0), 'luc' là lực khi tỷ lệ
    vi phạm thực gấp đôi mức danh nghĩa (tương ứng Bảng 8 của đề cương)."""
    from scipy import stats as st

    from de_cuong import danh_gia as DG
    N = ctx.khoi()["KT"][1] - ctx.khoi()["KT"][0]
    R = ctx.cfg["kiem_dinh_co_luc"]["so_lan"]
    rng = ctx.rng(khoa)
    p = khoa.split("::")[0].split("|")
    bac_bo = []
    for _ in range(R):
        if p[0] in ("kupiec", "christoffersen", "dq"):
            a = float(p[1][1:])
            I = (rng.random(N) < (2 * a if p[2] == "luc" else a)).astype(float)
            if p[0] == "kupiec":
                pv = DG.kupiec(int(I.sum()), N, a)[1]
            elif p[0] == "christoffersen":
                pv = DG.christoffersen(I, a)["p_cc"]
            else:
                pv = DG.phan_vi_dong(I, rng.normal(size=N), a)[1]
        elif p[0] == "dq_hac":
            a, H = float(p[1][1:]), int(p[2][1:])
            x = rng.normal(size=N + H)
            L = np.array([x[t + 1:t + H + 1].sum() for t in range(N)])
            v = np.full(N, math.sqrt(H) * st.norm.ppf(1 - a))           # VaR đúng của tổng H lợi suất chuẩn
            # biến hồi quy "VaR" phải có biến thiên (giống dữ liệu thật); dùng biến độc lập với chỉ thị, như kiểm định dq
            pv = DG.phan_vi_dong_hac((L > v).astype(float), rng.normal(size=N), a, H)[1]
        elif p[0] == "berkowitz":
            pv = DG.berkowitz(rng.random(N))[1]
        elif p[0] == "dm":
            e = rng.normal(size=N)
            if p[1] == "ar1":
                for t in range(1, N):
                    e[t] = 0.5 * e[t - 1] + math.sqrt(0.75) * e[t]
            pv = 2 * min(DG.dm(e, np.zeros(N))[1], 1 - DG.dm(e, np.zeros(N))[1])
        else:
            pv = DG.giacomini_white(rng.normal(size=N), np.zeros(N))[1]
        bac_bo.append(np.isfinite(pv) and pv < 0.05)
    ty_le = float(np.mean(bac_bo))
    return {"kiem_dinh": khoa.split("::")[0], "N": N, "R": R, "ty_le_bac_bo_5pct": ty_le,
            "sai_so_chuan_mc": math.sqrt(max(ty_le * (1 - ty_le), 1e-12) / R)}


def _th_b02k(ctx, kq):
    ctx.ghi("co_luc_kiem_dinh.json", {k.split("::")[0]: v for k, v in kq.items()})


Buoc("b02k", "co_luc_kiem_dinh", "5.1 (Bảng 8), 5.4, 5.5, Nhận xét 3", TEP, _lam_b02k, _th_b02k, don_vi=_dv_b02k,
     phu_thuoc=("b02",), cau_hinh=("khoi", "danh_gia", "ky_quy", "kiem_dinh_co_luc"), thu_vien=("danh_gia",))
