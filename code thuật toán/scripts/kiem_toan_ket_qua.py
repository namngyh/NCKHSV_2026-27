"""Kiem toan doc lap: tinh LAI moi con so trong bao cao tu nguon tho roi doi chieu.

  Buoc 1: log-hop-ly, AIC, BIC tai tham so MLE da luu; QLIKE, NLS, so vi pham
          VaR tai tham so MLE va tu mau hau nghiem VB da luu.
  Buoc 2: Bias, RMSE, Coverage tu tung file lan lap mo phong.
  Buoc 3: tom tat hau nghiem va R-hat tu mau MCMC da luu.

CACH CHAY (tu "code thuật toán"):  python scripts/kiem_toan_ket_qua.py
In "DAT"/"HONG" cho tung phep doi chieu; ket thuc voi ma thoat 1 neu co HONG.
"""
from __future__ import annotations

import math

import arviz as az
import torch

from chung import PHUONG_PHAP, RESULTS, nap

from garch_vn30.danh_gia import danh_gia_hau_nghiem, danh_gia_mot_diem
from garch_vn30.mo_hinh import danh_muc
from garch_vn30.mo_phong import tham_so_that, tong_hop, truong_hop_mac_dinh
from garch_vn30.suy_dien import posterior_summary
from garch_vn30.suy_dien.bayes import rut_bo_tham_so
from garch_vn30.suy_dien.mcmc import inference_data

dem = {"DAT": 0, "HONG": 0}


def kiem(ten: str, tinh_lai, da_luu, tol: float = 1e-8) -> None:
    lech = abs(float(tinh_lai) - float(da_luu))
    kq = "DAT" if lech <= tol * max(1.0, abs(float(da_luu))) else "HONG"
    dem[kq] += 1
    if kq == "HONG":
        print(f"  [HONG] {ten}: tinh lai {float(tinh_lai):.10g} / da luu {float(da_luu):.10g}")


def buoc_1() -> None:
    print("Buoc 1: 6 mo hinh x 3 phuong phap")
    d = nap(RESULTS / "00_data.pt")
    r, X_all, rv, n = d["r_all"], d["X_all"], d["rv_all"], d["n_train"]
    for spec in danh_muc.CATALOGUE:
        X = danh_muc.lay_X(spec, X_all)
        for pp in PHUONG_PHAP:
            o = nap(RESULTS / "pipeline" / f"{spec.code}_{pp}.pt")
            if pp == "mle":
                p = danh_muc.tham_so_tu_dict(o["params"], spec.family)
                ll = float(danh_muc.loglik_cua(spec.family)(p, r[:n], X[:n], h0=torch.var(r[:n])))
                k = spec.n_params(X_all.shape[1])
                kiem(f"{spec.code} loglik", ll, o["loglik"])
                kiem(f"{spec.code} AIC", 2 * k - 2 * ll, o["aic"])
                kiem(f"{spec.code} BIC", k * math.log(n) - 2 * ll, o["bic"])
                oos = danh_gia_mot_diem(p, spec.family, spec.dist, r, X, rv, n)
            else:
                draws = rut_bo_tham_so(o["samples"], spec, 200)
                oos = danh_gia_hau_nghiem(draws, spec.family, spec.dist, r, X, rv, n)
            for key in ("qlike", "nls", "mae", "rmse", "n_violations_1pct", "n_violations_5pct"):
                kiem(f"{spec.code}/{pp} {key}", oos[key], o["oos"][key])


def buoc_2() -> None:
    print("Buoc 2: mo phong E1-E3")
    for ma, th in truong_hop_mac_dinh().items():
        gop = nap(RESULTS / "mo_phong_egarch" / f"{ma}.pt")
        that = tham_so_that(th.params)
        for pp in PHUONG_PHAP:
            files = sorted((RESULTS / "mo_phong_egarch" / "parts").glob(f"{ma}_{pp}_*.pt"))
            b_so = sorted(nap(f)["b"] for f in files)
            kiem(f"{ma}/{pp} du B lan lap khong trung", len(set(b_so)) == th.B == len(b_so), True)
            moi = tong_hop([nap(f) for f in files], that)
            for loai in ("bias", "rmse", "coverage"):
                for t, v in gop[pp][loai].items():
                    kiem(f"{ma}/{pp} {loai}({t})", moi[loai][t], v, 1e-12)


def buoc_3() -> None:
    print("Buoc 3: MCMC tren EGARCH-X Student-t")
    m = nap(RESULTS / "pipeline" / "mcmc_egarch_x_t.pt")
    tt = posterior_summary(m["samples"])
    for ten, s in m["summary"].items():
        for key in ("mean", "sd", "q2.5", "q97.5"):
            kiem(f"mcmc {ten} {key}", (tt[ten][key] - s[key]).abs().max(), 0.0, 1e-12)
    # R-hat: mau gop duoc xep chuoi noi tiep chuoi (4 chuoi x 400 mau).
    n_chuoi = 4
    post = {k: v.reshape(n_chuoi, -1, *v.shape[1:]).numpy() for k, v in m["samples"].items()}
    rhat = max(float(az.rhat(inference_data(post))[k].values.max()) for k in post)
    kiem("mcmc R-hat lon nhat", rhat, m["max_rhat"], 1e-10)


if __name__ == "__main__":
    buoc_1()
    buoc_2()
    buoc_3()
    print(f"\nKET QUA: {dem['DAT']} DAT, {dem['HONG']} HONG")
    raise SystemExit(1 if dem["HONG"] else 0)
