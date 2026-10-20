"""Mô phỏng khôi phục tham số EGARCH-X theo mục 4.4 (Bảng 6): E1 - E4, chỉ MLE.

LÝ THUYẾT: Sinh dữ liệu từ EGARCH-X Student-t lệch với tham số đã biết (T = 2000 sau 200 quan sát
khởi động, B = 100 lần lặp mỗi kịch bản), ước lượng lại bằng MLE và đo độ chệch, RMSE, độ phủ khoảng
Wald 95% (sai số chuẩn Hessian và bánh kẹp). E2 kiểm tra độ đặc hiệu của theta (tỷ lệ bác bỏ sai
H0: theta = 0), E4 kiểm tra độ đặc hiệu của gamma (LR, H0: gamma = 0). Biến lịch: phiên đáo hạn mỗi
21 phiên, phiên kế tiếp là phiên chuyển hợp đồng (J = 1, lợi suất khuyết), kỳ nghỉ dài ngẫu nhiên.
Chạy: python scripts/dc_mo_phong.py [--B 100] [--jobs 8] [--nhanh]
THƯ VIỆN: numpy, scipy, concurrent.futures.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))

from de_cuong import danh_gia as dg                                        # noqa: E402
from de_cuong.mo_hinh import (Cau_hinh, diem_khoi_tao, e_abs, lay_mau,      # noqa: E402
                              uoc_luong)

CHUNG = {"mu": 0.05, "omega": 0.04, "kappa": 0.15, "beta": 0.95}
KICH_BAN = {"E1": (-0.10, (0.05, 0.05), 8.0, -0.1), "E2": (0.0, (0.05, 0.05), 8.0, -0.1),
            "E3": (-0.10, (0.05, 0.05), 4.0, -0.1), "E4": (-0.10, (0.0, 0.0), 8.0, -0.1)}
T, KHOI_DONG = 2000, 200
CFG = Cau_hinh("skewt", (0, 1))


def sinh(kb, rng):
    th, gam, nu, lam = KICH_BAN[kb]
    n = T + KHOI_DONG
    X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + math.sqrt(1 - 0.25) * rng.normal(size=2)
    D = np.zeros((n, 3))
    D[20::21, 0] = 1
    D[21::21, 1] = 1
    D[:, 2] = rng.random(n) < 0.006
    J = D[:, 1] > 0
    Z = np.zeros((n, 5))
    Z[1:, :2] = X[:-1]
    Z[:, 2:] = D
    sp = np.array([nu, lam])
    z = lay_mau(n, "skewt", sp, rng)
    mabs = e_abs("skewt", sp)
    b = np.array([*gam, 0.0, 0.0, 0.0])
    lnh = np.empty(n)
    lnh[0] = CHUNG["omega"] / (1 - CHUNG["beta"])
    r = np.empty(n)
    for t in range(n):
        if t > 0:
            s = 0.0 if J[t - 1] else CHUNG["kappa"] * (abs(z[t - 1]) - mabs) + th * z[t - 1]
            lnh[t] = min(20.0, max(-20.0, CHUNG["omega"] + s + CHUNG["beta"] * lnh[t - 1] + Z[t] @ b))
        r[t] = CHUNG["mu"] + math.exp(0.5 * lnh[t]) * z[t]
    sl = slice(KHOI_DONG, n)
    return r[sl], J[sl], np.ascontiguousarray(Z[sl])


def mot_lan(args):
    kb, b, nhanh = args
    rng = np.random.default_rng(20261020 + 1000 * int(kb[1]) + b)
    th, gam, nu, lam = KICH_BAN[kb]
    that = np.array([CHUNG["mu"], CHUNG["omega"], CHUNG["kappa"], th, CHUNG["beta"], *gam, 0, 0, 0, nu, lam])
    try:
        r, J, Z = sinh(kb, rng)
        r_f = np.where(J, 0.0, r)
        lh0 = math.log(float(np.var(r[~J][:20], ddof=1)))
        st = diem_khoi_tao(CFG, r[:200], 3 if nhanh else 6, rng)
        p, nll, gan, tong = uoc_luong(CFG, r_f, J, Z, lh0, st)
        se = dg.sai_so_chuan(p, CFG, r_f, J, Z, lh0)
        cfg0 = Cau_hinh("skewt", ())
        Z0 = np.ascontiguousarray(Z[:, 2:])
        p0, nll0, _, _ = uoc_luong(cfg0, r_f, J, Z0, lh0, diem_khoi_tao(cfg0, r[:200], 3 if nhanh else 6, rng))
        return {"ok": True, "p": p, "se_h": se["se_hessian"], "se_s": se["se_banh_kep"], "that": that,
                "LR_gamma": 2 * (nll0 - nll)}
    except Exception as e:   # noqa: BLE001
        return {"ok": False, "loi": str(e)}


def tom_tat(kq):
    ok = [x for x in kq if x["ok"]]
    ten = CFG.ten()
    P = np.array([x["p"] for x in ok]); T0 = np.array([x["that"] for x in ok])
    SH = np.array([x["se_h"] for x in ok]); SS = np.array([x["se_s"] for x in ok])
    z975 = 1.959964
    tham_so = {}
    for i, t in enumerate(ten):
        phu = lambda se: float(np.mean(np.abs(P[:, i] - T0[:, i]) <= z975 * se[:, i]))
        tham_so[t] = {"that": float(T0[0, i]), "tb": float(P[:, i].mean()), "chech": float((P[:, i] - T0[:, i]).mean()),
                      "rmse": float(np.sqrt(((P[:, i] - T0[:, i]) ** 2).mean())),
                      "phu_hessian": phu(SH), "phu_banh_kep": phu(SS)}
    ith = ten.index("theta")
    return {"B_thanh_cong": len(ok), "B_that_bai": len(kq) - len(ok), "tham_so": tham_so,
            "bac_bo_theta0_banh_kep": float(np.mean(np.abs(P[:, ith] / SS[:, ith]) > z975)),
            "bac_bo_theta0_hessian": float(np.mean(np.abs(P[:, ith] / SH[:, ith]) > z975)),
            "theta_am_ty_le": float(np.mean(P[:, ith] < 0)),
            "bac_bo_gamma0_LR": float(np.mean(np.array([x["LR_gamma"] for x in ok]) > stats.chi2.ppf(0.95, 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--nhanh", action="store_true")
    a = ap.parse_args()
    out = {}
    for kb in KICH_BAN:
        t = time.time()
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            kq = list(ex.map(mot_lan, [(kb, b, a.nhanh) for b in range(a.B)]))
        out[kb] = tom_tat(kq)
        o = out[kb]
        print(f"{kb}: thành công {o['B_thanh_cong']}/{a.B}, theta chệch {o['tham_so']['theta']['chech']:+.4f}, "
              f"phủ theta (bánh kẹp) {o['tham_so']['theta']['phu_banh_kep']:.2f}, bác bỏ theta=0 {o['bac_bo_theta0_banh_kep']:.2f}, "
              f"bác bỏ gamma=0 {o['bac_bo_gamma0_LR']:.2f}  [{time.time()-t:.0f} s]", flush=True)
    ra = GOC / "results" / "de_cuong"
    ra.mkdir(parents=True, exist_ok=True)
    ten = "06_mo_phong_nhanh.json" if a.nhanh else "06_mo_phong.json"
    (ra / ten).write_text(json.dumps({"B": a.B, "T": T, "khoi_dong": KHOI_DONG, "tham_so_that": {
        k: {"theta": v[0], "gamma": list(v[1]), "nu": v[2], "lambda": v[3]} for k, v in KICH_BAN.items()},
        "ket_qua": out}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
