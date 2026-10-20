"""Điều phối Giai đoạn 1 theo đề cương EGARCH-X (mục 2 - 6), chỉ dùng MLE.

LÝ THUYẾT: xem `de_cuong/*.py`. Các bước (mỗi bước ghi vào results/de_cuong/):
  1 du_lieu       kiểm tra chất lượng dữ liệu (mục 2.8) và thống kê mô tả
  2 uoc_luong     MLE trên khối UL và UL+XT, sai số chuẩn bánh kẹp, chẩn đoán, kiểm định gamma = 0 (3.6, 4.2)
  3 chon_bien     chọn tiến biến ngoại sinh trên khối XT, SPA, Holm (mục 6)
  4 chon_pp       chọn phân phối sai số trên khối XT (mục 5, 6)
  5 kiem_tra      đánh giá ngoài mẫu trên khối KT, so sánh có kiểm định, nhóm phiên (mục 5)
Chạy:  python scripts/dc_chay.py --buoc 1 2 3 4 5   (--nhanh: ít khởi tạo hơn để thử đường chạy)
THƯ VIỆN: numpy, scipy, concurrent.futures (song song hóa các cấu hình độc lập).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))

from de_cuong import danh_gia as dg                                   # noqa: E402
from de_cuong.du_lieu import BO_QUA, TEN_BIEN, chia_khoi, dac_trung, xay_ngay  # noqa: E402
from de_cuong.mo_hinh import (PHAN_PHOI, Cau_hinh, chay_cuon_chieu, diem_khoi_tao,  # noqa: E402
                              tao_Z, uoc_luong)

RA = GOC / "results" / "de_cuong"
SEED = 20261020


def js(o):
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, np.ndarray):
        return js(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def luu_json(ten, obj):
    RA.mkdir(parents=True, exist_ok=True)
    (RA / ten).write_text(json.dumps(js(obj), ensure_ascii=False, indent=1), encoding="utf-8")
    print("  ->", ten, flush=True)


def luu_pkl(ten, obj):
    RA.mkdir(parents=True, exist_ok=True)
    with open(RA / ten, "wb") as f:
        pickle.dump(obj, f)


def doc_pkl(ten):
    with open(RA / ten, "rb") as f:
        return pickle.load(f)


def nap():
    d = xay_ngay()
    blk = chia_khoi(len(d) - BO_QUA)
    return d, {k: (a + BO_QUA, b + BO_QUA) for k, (a, b) in blk.items()}


def job_cuon(args):
    """Một cấu hình cuốn chiếu (chạy trong tiến trình con)."""
    dist, cot, dau, cuoi, n_starts = args
    d, _ = nap()
    return chay_cuon_chieu(d, Cau_hinh(dist, tuple(cot)), dau, cuoi, n_starts=n_starts, seed=SEED)


def song_song(jobs, nhan=6):
    with ProcessPoolExecutor(max_workers=min(nhan, len(jobs))) as ex:
        return list(ex.map(job_cuon, jobs))


# ------------------------------------------------------------------------------------------
def buoc1(d, blk, a):
    print("== Bước 1: kiểm tra dữ liệu (mục 2.8)")
    n = len(d)
    ohlc = ((d.low > np.minimum(d.open, d.close)) | (d.high < np.maximum(d.open, d.close)))
    lech_v = (d.vol - d.buy - d.sell).abs() / d.vol
    r = d.r
    lon = d.loc[r.abs() > 7, ["date", "r"]]
    gap = []
    for t in range(1, n):
        nlv = int(np.busday_count(d.date[t - 1].date(), d.date[t].date())) - 1
        if nlv > 0:
            gap.append(nlv)
    rr = r.dropna().to_numpy()
    dem = 100 * np.log(d.open / d.close.shift(1))
    kq = {"so_phien": n, "tu": str(d.date.iloc[0].date()), "den": str(d.date.iloc[-1].date()),
          "OHLC_vi_pham": int(ohlc.sum()), "phien_trung": int(d.date.duplicated().sum()),
          "V_lech_buy_sell_tren_1pct": int((lech_v > 0.01).sum()),
          "V_lech_buy_sell_ngay": [str(x.date()) for x in d.date[lech_v > 0.2]],
          "phien_chuyen_hop_dong": int(d.J.sum()), "phien_dao_han": int(d.D_dh.sum()),
          "phien_sau_nghi_dai": int(d.D_nghi.sum()), "phien_cham_bien_khoa": int(d.khoa_bien.sum()),
          "|r|>7_ngoai_phien_chuyen": [{"ngay": str(x.date.date()), "r": float(x.r)} for x in lon.itertuples()],
          "khoang_trong_ngay_lam_viec_(so_khoang)": len(gap), "khoang_trong_dai_nhat": max(gap) if gap else 0,
          "|qua_dem|_trung_vi_phien_chuyen": float(dem[d.J].abs().median()),
          "|qua_dem|_trung_vi_phien_thuong": float(dem[~d.J].abs().median()),
          "r_mo_ta": {"n": len(rr), "trung_binh": rr.mean(), "phuong_sai": rr.var(ddof=1),
                      "lech": float(((rr - rr.mean()) ** 3).mean() / rr.std() ** 3),
                      "nhon": float(((rr - rr.mean()) ** 4).mean() / rr.std() ** 4)},
          "khoi": {k: {"chi_so": v, "tu": str(d.date[v[0]].date()), "den": str(d.date[v[1] - 1].date()),
                       "so_phien": v[1] - v[0], "so_phien_chuyen": int(d.J[v[0]:v[1]].sum())}
                   for k, v in blk.items()},
          "bo_qua_dau": BO_QUA,
          "ghi_chu": "Phiên chuyển hợp đồng xác định theo lịch (thứ Năm thứ ba); không thể đối chiếu lịch giao dịch chính thức "
                     "nên các khoảng trống ngày làm việc (lễ, nghỉ bù) chưa được phân biệt với phiên thiếu."}
    luu_json("01_du_lieu.json", kq)
    d.to_csv(RA / "01_ngay.csv", index=False)


def lay_mau(d, s0, s1):
    f = dac_trung(d, s1 - 1)
    r = np.nan_to_num(f["r"], nan=0.0)
    J = np.asarray(f["J"], bool) | np.isnan(f["r"])
    lh0 = math.log(float(np.nanvar(f["r"][BO_QUA:BO_QUA + 20], ddof=1)))
    return f, r, J, lh0


def ml(cfg, f, r, J, lh0, s1, n_starts, rng):
    Z = tao_Z(cfg, f)
    sl = slice(BO_QUA, s1)
    st = diem_khoi_tao(cfg, f["r"][sl], n_starts, rng)
    p, nll, gan, tong = uoc_luong(cfg, r[sl], J[sl], Z[sl], lh0, st)
    return p, nll, gan, tong, Z[sl]


def buoc2(d, blk, a):
    from scipy import stats
    print("== Bước 2: ước lượng MLE, sai số chuẩn, chẩn đoán, kiểm định gamma = 0")
    ns = 8 if a.nhanh else 20
    rng = np.random.default_rng(SEED)
    out = {}
    for ten_khoi, s1 in (("UL", blk["UL"][1]), ("UL+XT", blk["XT"][1])):
        f, r, J, lh0 = lay_mau(d, BO_QUA, s1)
        sl = slice(BO_QUA, s1)
        rs = {}
        for dist in PHAN_PHOI:
            cfg = Cau_hinh(dist, (0, 1))
            p, nll, gan, tong, Zs = ml(cfg, f, r, J, lh0, s1, ns, rng)
            se = dg.sai_so_chuan(p, cfg, r[sl], J[sl], Zs, lh0)
            cfg0 = Cau_hinh(dist, ())
            p0, nll0, _, _, _ = ml(cfg0, f, r, J, lh0, s1, ns, rng)
            lr = 2 * (nll0 - nll)
            rs[dist] = {"ten": cfg.ten(), "p": p, "se_hessian": se["se_hessian"], "se_banh_kep": se["se_banh_kep"],
                        "loglik": -nll, "aic": 2 * cfg.k + 2 * nll, "bic": cfg.k * math.log((~J[sl]).sum()) + 2 * nll,
                        "k": cfg.k, "n_hop_ly": int((~J[sl]).sum()), "gan_tot": [gan, tong],
                        "LR_gamma0": lr, "LR_p": float(1 - stats.chi2.cdf(lr, 2)), "loglik_EGARCH": -nll0,
                        "chan_doan": dg.chan_doan_trong_mau(p, cfg, r[sl], J[sl], Zs, lh0, f["tau"][sl], f["D"][sl])}
            print(f"  {ten_khoi} {dist}: AIC={rs[dist]['aic']:.2f} LR(gamma=0)={lr:.2f} p={rs[dist]['LR_p']:.4f}", flush=True)
        out[ten_khoi] = rs
    luu_json("02_uoc_luong.json", out)


# ------------------------------------------------------------------------------------------
def buoc3(d, blk, a):
    from scipy import stats
    print("== Bước 3: chọn biến ngoại sinh (mục 6) trên khối XT, phân phối Student-t chuẩn hóa")
    ns = 4 if a.nhanh else 8
    dau, cuoi = blk["XT"]
    r = d.r.to_numpy()[dau:cuoi]
    f_all = dac_trung(d, blk["XT"][1] - 1)
    Xc = f_all["X"][BO_QUA:blk["XT"][1]]
    tuong_quan = np.corrcoef(Xc.T)
    nhat_ky, cache = [], {}

    def chay(cot_list):
        jobs = [("student", c, dau, cuoi, ns) for c in cot_list if tuple(c) not in cache]
        if jobs:
            for j, res in zip(jobs, song_song(jobs)):
                cache[tuple(j[1])] = res
        return {tuple(c): cache[tuple(c)] for c in cot_list}

    chay([()])
    cur = ()
    while True:
        ung_vien = [j for j in range(5) if j not in cur and all(abs(tuong_quan[j, k]) <= 0.8 for k in cur)]
        if not ung_vien:
            break
        thu = chay([tuple(sorted(cur + (j,))) for j in ung_vien])
        diem = {c: float(np.nanmean(rs["ls"])) for c, rs in thu.items()}
        tot = max(diem, key=diem.get)
        t, p = dg.dm(-cache[tot]["ls"], -cache[cur]["ls"])
        fz_c = np.nanmean(dg.fz0_chuoi(cache[cur], r, 0.025))
        fz_t = np.nanmean(dg.fz0_chuoi(cache[tot], r, 0.025))
        nhan = bool(p < 0.05 and fz_t <= fz_c)
        nhat_ky.append({"buoc": len(cur) + 1, "tap_hien_tai": [TEN_BIEN[j] for j in cur],
                        "ung_vien": {"+".join(TEN_BIEN[j] for j in c): v for c, v in diem.items()},
                        "tot_nhat": [TEN_BIEN[j] for j in tot], "LS_hien_tai": float(np.nanmean(cache[cur]["ls"])),
                        "DM_t": t, "DM_p_mot_phia": p, "FZ0_2.5_hien_tai": float(fz_c), "FZ0_2.5_ung_vien": float(fz_t),
                        "chap_nhan": nhan})
        print(f"  bước {len(cur)+1}: tốt nhất {[TEN_BIEN[j] for j in tot]} DM p={p:.4f} chấp nhận={nhan}", flush=True)
        if not nhan:
            break
        cur = tot
    chon = list(cur)
    chay([(j,) for j in range(5)])
    tap = [(j,) for j in range(5)] + ([tuple(chon)] if len(chon) > 1 else [])
    for c in tap:
        chay([c])
    L0 = -cache[()]["ls"]
    Dm = np.column_stack([L0 - (-cache[c]["ls"]) for c in tap])
    Dm = Dm[np.isfinite(Dm).all(1)]
    spa = spa_test(Dm, np.random.default_rng(SEED))
    f, rr, J, lh0 = lay_mau(d, BO_QUA, blk["XT"][1])
    rng = np.random.default_rng(SEED + 1)
    s1 = blk["XT"][1]
    nll_cfg = {}
    for c in [()] + [(j,) for j in range(5)]:
        _, nll, _, _, _ = ml(Cau_hinh("student", c), f, rr, J, lh0, s1, 8 if a.nhanh else 20, rng)
        nll_cfg[c] = nll
    pj = {TEN_BIEN[c[0]]: float(1 - stats.chi2.cdf(2 * (nll_cfg[()] - nll_cfg[c]), 1)) for c in nll_cfg if c}
    holm, mx = {}, 0.0
    for i, k in enumerate(sorted(pj, key=pj.get)):
        mx = max(mx, min(1.0, (len(pj) - i) * pj[k]))
        holm[k] = mx
    on_dinh = {}
    if chon:
        P = cache[tuple(chon)]["params"]
        for i, j in enumerate(chon):
            g = P[:, 5 + i]
            on_dinh[TEN_BIEN[j]] = {"min": float(g.min()), "max": float(g.max()), "doi_dau": bool(g.min() < 0 < g.max())}
    luu_json("03_chon_bien.json", {"chon": [TEN_BIEN[j] for j in chon], "nhat_ky": nhat_ky, "SPA": spa,
                                   "tuong_quan_X_chuan_hoa": tuong_quan, "LR_tung_bien_p": pj, "Holm_p": holm,
                                   "on_dinh_gamma": on_dinh,
                                   "LS_XT": {"+".join(TEN_BIEN[j] for j in c) or "EGARCH": float(np.nanmean(v["ls"]))
                                             for c, v in cache.items()}})
    luu_pkl("03_cuon_chieu_XT.pkl", cache)


def spa_test(Dm, rng, B=2000, mean_block=5):
    """Kiểm định SPA của Hansen (2005): d = L_chuẩn - L_ứng viên; H0: không ứng viên nào tốt hơn."""
    n, k = Dm.shape
    dbar = Dm.mean(0)
    idx = np.empty((B, n), int)
    for b in range(B):
        t = rng.integers(n)
        for i in range(n):
            t = rng.integers(n) if rng.random() < 1 / mean_block else (t + 1) % n
            idx[b, i] = t
    bm = Dm[idx].mean(1)
    omega = np.sqrt(n) * bm.std(0, ddof=1)
    omega = np.where(omega > 1e-12, omega, 1e-12)
    tk = np.sqrt(n) * dbar / omega
    T = max(tk.max(), 0.0)
    mu_c = dbar * (tk <= -math.sqrt(2 * math.log(math.log(n))))
    z = np.sqrt(n) * (bm - dbar + mu_c) / omega
    Tb = np.maximum(z.max(1), 0.0)
    return {"T": float(T), "p": float((Tb >= T).mean()), "tk": tk, "ung_vien": k}


# ------------------------------------------------------------------------------------------
def buoc4(d, blk, a):
    print("== Bước 4: chọn phân phối sai số trên khối XT")
    ns = 4 if a.nhanh else 8
    sel = json.loads((RA / "03_chon_bien.json").read_text(encoding="utf-8"))["chon"]
    cot = tuple(TEN_BIEN.index(x) for x in sel) or (0, 1)
    dau, cuoi = blk["XT"]
    r = d.r.to_numpy()[dau:cuoi]
    kh = d.khoa_bien.to_numpy()[dau:cuoi]
    jobs = [(dist, cot, dau, cuoi, ns) for dist in PHAN_PHOI] + [("student", (0, 1), dau, cuoi, ns)]
    rs = song_song(jobs)
    tab = {}
    for (dist, c, *_), res in zip(jobs, rs):
        tab[f"{dist}|{'+'.join(TEN_BIEN[j] for j in c)}"] = dg.tom_tat_khoi(res, r, kh)
    ls = {dist: rs[i]["ls"] for i, dist in enumerate(PHAN_PHOI)}
    tot = max(PHAN_PHOI, key=lambda x: np.nanmean(ls[x]))
    dm_so_sanh = {f"{x}-{y}": dict(zip(("t", "p_mot_phia"), dg.dm(-ls[x], -ls[y])))
                  for x in PHAN_PHOI for y in PHAN_PHOI if x != y}
    luu_json("04_chon_phan_phoi.json", {"cot": [TEN_BIEN[j] for j in cot], "bang": tab, "chon": tot,
                                        "DM_LS": dm_so_sanh})
    print("  chọn:", tot, "cột:", [TEN_BIEN[j] for j in cot])


# ------------------------------------------------------------------------------------------
def nhom_phien(d, dau, cuoi):
    """Mặt nạ các nhóm phiên theo mục 5.6, trên khối [dau, cuoi)."""
    tau = d.tau.to_numpy()[dau:cuoi]
    m = {"quanh_dao_han(tau<=2 hoac sau chuyen)": (tau <= 2) | d.J.to_numpy()[dau:cuoi],
         "sau_nghi_dai": d.D_nghi.to_numpy()[dau:cuoi]}
    f = dac_trung(d, cuoi - 1)
    rho = f["rho_sao"]
    st = np.zeros(cuoi - dau, bool)
    for i, t in enumerate(range(dau, cuoi)):
        qk = rho[max(0, t - 250):t]
        qk = qk[np.isfinite(qk)]
        st[i] = len(qk) > 50 and np.isfinite(rho[t]) and rho[t] > np.quantile(qk, 0.95)
    m["cang_thang(rho*>P95 truot)"] = st
    return m


def buoc5(d, blk, a):
    print("== Bước 5: đánh giá ngoài mẫu trên khối KT")
    ns = 4 if a.nhanh else 8
    sel = json.loads((RA / "03_chon_bien.json").read_text(encoding="utf-8"))["chon"]
    cot = tuple(TEN_BIEN.index(x) for x in sel) or (0, 1)
    pp = json.loads((RA / "04_chon_phan_phoi.json").read_text(encoding="utf-8"))["chon"]
    dau, cuoi = blk["XT"][0], blk["KT"][1]
    kt0 = blk["KT"][0] - dau
    cfgs = {"EGARCH|" + pp: (pp, ()), "chinh(A,G)|" + pp: (pp, (0, 1)), "chon|" + pp: (pp, cot)}
    for dist in PHAN_PHOI:
        cfgs[f"chon|{dist}"] = (dist, cot)
    ten = list(dict.fromkeys(cfgs))
    rs = dict(zip(ten, song_song([(cfgs[k][0], cfgs[k][1], dau, cuoi, ns) for k in ten])))
    r = d.r.to_numpy()[dau:cuoi][kt0:]
    kh = d.khoa_bien.to_numpy()[dau:cuoi][kt0:]

    def cat(res):
        o = dict(res)
        for k in ("lnh", "mu", "rho_sao", "ls", "z"):
            o[k] = res[k][kt0:]
        for k in ("q", "e", "qp", "ep"):
            o[k] = {al: v[kt0:] for al, v in res[k].items()}
        return o

    kt = {k: cat(v) for k, v in rs.items()}
    bang = {k: dg.tom_tat_khoi(v, r, kh) for k, v in kt.items()}
    nm = nhom_phien(d, blk["KT"][0], blk["KT"][1])
    theo_nhom = {k: {g: dg.tom_tat_khoi(v, r, kh, mask=m) for g, m in nm.items()} for k, v in kt.items()}
    so_sanh, chuan = {}, "chon|" + pp
    for k in ten:
        if k == chuan:
            continue
        so_sanh[f"{chuan} vs {k}"] = {
            "LS_DM": dict(zip(("t", "p"), dg.dm(-kt[chuan]["ls"], -kt[k]["ls"]))),
            "FZ0_2.5_DM": dict(zip(("t", "p"), dg.dm(dg.fz0_chuoi(kt[chuan], r, 0.025), dg.fz0_chuoi(kt[k], r, 0.025)))),
            "FZ0_1_DM": dict(zip(("t", "p"), dg.dm(dg.fz0_chuoi(kt[chuan], r, 0.01), dg.fz0_chuoi(kt[k], r, 0.01))))}
    on_dinh = {k: {"beta_min": float(v["params"][:, 4].min()), "beta_max": float(v["params"][:, 4].max()),
                   "gamma_dau_doi": [bool(v["params"][:, 5 + i].min() < 0 < v["params"][:, 5 + i].max())
                                     for i in range(len(cfgs[k][1]))]} for k, v in rs.items()}
    luu_json("05_kiem_tra_KT.json", {"cau_hinh": {k: [v[0], [TEN_BIEN[j] for j in v[1]]] for k, v in cfgs.items()},
                                     "KT": bang, "nhom": theo_nhom, "so_sanh": so_sanh, "on_dinh": on_dinh,
                                     "tu": str(d.date[blk["KT"][0]].date()), "den": str(d.date[blk["KT"][1] - 1].date())})
    luu_pkl("05_cuon_chieu_XT_KT.pkl", {"rs": rs, "cfgs": cfgs, "dau": dau, "cuoi": cuoi, "kt0": kt0})
    for k, v in bang.items():
        print(f"  {k:24s} QLIKE={v['QLIKE']:.4f} LS={v['LS']:.4f} vp1%={v['a0.01']['vi_pham']} "
              f"(p={v['a0.01']['kupiec_p']:.3f}) FZ0(2.5%)={v['a0.025']['FZ0']:.4f}", flush=True)


BUOC = {1: buoc1, 2: buoc2, 3: buoc3, 4: buoc4, 5: buoc5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--nhanh", action="store_true")
    a = ap.parse_args()
    d, blk = nap()
    for b in a.buoc:
        t = time.time()
        BUOC[b](d, blk, a)
        print(f"  (bước {b}: {time.time()-t:.0f} giây)", flush=True)


if __name__ == "__main__":
    main()
