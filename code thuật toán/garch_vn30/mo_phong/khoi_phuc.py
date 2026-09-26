"""Nghien cuu khoi phuc tham so: uoc luong tren MOT bo du lieu mo phong, va gop
B lan lap thanh Bias / RMSE / Coverage.

Voi tham so thu j, gia tri that theta_0j va uoc luong theta_hat_{j,b}, b = 1..B:

    Bias_j     = (1/B) sum_b (theta_hat_{j,b} - theta_0j)
    RMSE_j     = sqrt( (1/B) sum_b (theta_hat_{j,b} - theta_0j)^2 )
    Coverage_j = (1/B') sum_b 1{ L_{j,b} <= theta_0j <= U_{j,b} }

[L, U] la khoang 95%: phan vi 2,5%-97,5% cua hau nghiem (VB), hoac khoang
Wald theta_hat +- 1,96 SE (MLE, SE tu Hessian). Voi MLE chi dem nhung lan lap
co Hessian xac dinh duong (B' <= B).
"""
from __future__ import annotations

import numpy as np
import torch

from garch_vn30.mo_hinh.danh_muc import ModelSpec
from garch_vn30.mo_phong.truong_hop import tham_so_that
from garch_vn30.suy_dien.mle import fit_mle, standard_errors
from garch_vn30.suy_dien.vb import fit_vb

Z_95 = 1.96

# Cau hinh uoc luong trong mo phong (nhe hon Buoc 1 vi phai lap 900 lan).
CAU_HINH_MAC_DINH = {
    "mle": {"n_starts": 3},
    "vb_mean_field": {"n_steps": 1500},
    "vb_full_rank": {"n_steps": 1500},
}


def _tu_summary(summary: dict, key: str, spec: ModelSpec, k: int) -> dict[str, float]:
    """Doc mot thong ke (mean / q2.5 / q97.5) cua VB thanh dict co ten."""
    d = {ten: float(summary[ten][key]) for ten in ["mu", "omega", "alpha", "theta", "beta"]}
    if spec.use_exog:
        for j in range(k):
            d[f"gamma_{j}"] = float(summary["gamma"][key][j])
    if spec.dist == "student_t":
        d["nu"] = float(summary["nu_minus_2"][key]) + 2.0
    return d


def _se_theo_ten(se: dict, ten: str) -> float:
    if ten.startswith("gamma_"):
        return float(se["gamma"][int(ten.split("_")[1])])
    return float(se[ten])


def chay_mot_lan_lap(method: str, r: torch.Tensor, X: torch.Tensor, spec: ModelSpec,
                     seed: int, kwargs: dict | None = None) -> dict:
    """Uoc luong EGARCH bang `method` tren (r, X); tra ve uoc luong diem va khoang 95%."""
    kw = {**CAU_HINH_MAC_DINH.get(method, {}), **(kwargs or {})}
    if method == "mle":
        res = fit_mle(r, X, dist=spec.dist, family=spec.family, base_seed=seed, **kw)
        uoc_luong = tham_so_that(res.params)
        se = standard_errors(res.params, r, X, dist=spec.dist, family=spec.family)
        if se is None:            # Hessian khong xac dinh duong -> khong co khoang Wald
            return {"uoc_luong": uoc_luong, "can_duoi": None, "can_tren": None}
        return {"uoc_luong": uoc_luong,
                "can_duoi": {t: v - Z_95 * _se_theo_ten(se, t) for t, v in uoc_luong.items()},
                "can_tren": {t: v + Z_95 * _se_theo_ten(se, t) for t, v in uoc_luong.items()}}

    if method in ("vb_mean_field", "vb_full_rank"):
        guide = "mean_field" if method == "vb_mean_field" else "full_rank"
        res = fit_vb(r, X, spec, guide_type=guide, seed=seed, **kw)
        k = X.shape[1]
        return {"uoc_luong": _tu_summary(res.summary, "mean", spec, k),
                "can_duoi": _tu_summary(res.summary, "q2.5", spec, k),
                "can_tren": _tu_summary(res.summary, "q97.5", spec, k)}

    raise ValueError(f"Phuong phap khong ho tro: {method!r}")


def tong_hop(lan_lap: list[dict], that: dict[str, float]) -> dict:
    """Gop cac lan lap cua MOT phuong phap thanh Bias, RMSE, Coverage theo tham so."""
    co_khoang = [x for x in lan_lap if x["can_duoi"] is not None]
    ra = {"bias": {}, "rmse": {}, "coverage": {}, "n_replications": len(lan_lap),
          "n_co_khoang": len(co_khoang), "tham_so_that": that}
    for ten, gia_tri_that in that.items():
        sai = np.array([x["uoc_luong"][ten] for x in lan_lap]) - gia_tri_that
        ra["bias"][ten] = float(np.mean(sai))
        ra["rmse"][ten] = float(np.sqrt(np.mean(sai ** 2)))
        if co_khoang:
            duoi = np.array([x["can_duoi"][ten] for x in co_khoang])
            tren = np.array([x["can_tren"][ten] for x in co_khoang])
            ra["coverage"][ten] = float(np.mean((duoi <= gia_tri_that) & (gia_tri_that <= tren)))
    return ra
