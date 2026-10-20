"""Nghiên cứu khôi phục tham số: ước lượng trên MỘT bộ dữ liệu mô phỏng, và gộp
B lần lặp thành các chỉ tiêu Bias / RMSE / Coverage.

LÝ THUYẾT
    Với tham số thứ j, giá trị thật theta_0j và ước lượng theta_hat_{j,b} ở lần lặp b = 1..B:

        Bias_j     = (1/B) sum_b (theta_hat_{j,b} - theta_0j)             độ chệch trung bình
        RMSE_j     = sqrt( (1/B) sum_b (theta_hat_{j,b} - theta_0j)^2 )   sai số tổng (chệch + phương sai)
        Coverage_j = (1/B') sum_b 1{ L_{j,b} <= theta_0j <= U_{j,b} }    tỷ lệ khoảng 95% chứa giá trị thật

    [L, U] là khoảng 95%:
    - VB : phân vị 2,5% và 97,5% của mẫu hậu nghiệm;
    - MLE: khoảng Wald theta_hat ± 1,96 SE, SE từ Hessian (suy_dien/mle.standard_errors).
      Chỉ tính trên B' <= B lần lặp có Hessian xác định dương.
    Coverage gần 0,95 nghĩa là khoảng tin cậy được hiệu chỉnh tốt; nhỏ hơn nhiều
    nghĩa là phương pháp báo bất định thấp hơn thực tế.
"""
from __future__ import annotations

import numpy as np
import torch

from garch_vn30.mo_hinh.danh_muc import ModelSpec
from garch_vn30.mo_phong.truong_hop import tham_so_that
from garch_vn30.suy_dien.mle import fit_mle, standard_errors
from garch_vn30.suy_dien.vb import fit_vb

Z_95 = 1.96   # phân vị 97,5% của N(0,1)

# Cấu hình ước lượng trong mô phỏng (nhẹ hơn Bước 1 vì phải lặp 900 lần).
CAU_HINH_MAC_DINH = {
    "mle": {"n_starts": 3},
    "vb_mean_field": {"n_steps": 1500},
    "vb_full_rank": {"n_steps": 1500},
}


def _tu_summary(summary: dict, key: str, spec: ModelSpec, k: int) -> dict[str, float]:
    """Đọc một thống kê (mean / q2.5 / q97.5) từ tóm tắt hậu nghiệm VB thành dict có tên."""
    d = {ten: float(summary[ten][key]) for ten in ["mu", "omega", "alpha", "theta", "beta"]}
    if spec.use_exog:
        for j in range(k):
            d[f"gamma_{j}"] = float(summary["gamma"][key][j])
    if spec.dist == "student_t":
        d["nu"] = float(summary["nu_minus_2"][key]) + 2.0
    return d


def _se_theo_ten(se: dict, ten: str) -> float:
    """Sai số chuẩn theo tên phẳng (gamma_0 -> se["gamma"][0])."""
    if ten.startswith("gamma_"):
        return float(se["gamma"][int(ten.split("_")[1])])
    return float(se[ten])


def chay_mot_lan_lap(method: str, r: torch.Tensor, X: torch.Tensor, spec: ModelSpec,
                     seed: int, kwargs: dict | None = None) -> dict:
    """Ước lượng EGARCH trên MỘT bộ dữ liệu mô phỏng (r, X) bằng `method`.

    Tham số:
        method: "mle", "vb_mean_field" hoặc "vb_full_rank".
        spec  : mô hình ước lượng (egarch_x_t).
        seed  : hạt giống cho bộ ước lượng.
        kwargs: ghi đè cấu hình (vd. {"n_starts": 1} khi chạy thử).
    Trả về dict:
        uoc_luong: {tên: ước lượng điểm};
        can_duoi, can_tren: {tên: cận khoảng 95%}, hoặc None nếu MLE không có
                            Hessian hợp lệ.
    """
    kw = {**CAU_HINH_MAC_DINH.get(method, {}), **(kwargs or {})}
    if method == "mle":
        res = fit_mle(r, X, dist=spec.dist, family=spec.family, base_seed=seed, **kw)
        uoc_luong = tham_so_that(res.params)
        se = standard_errors(res.params, r, X, dist=spec.dist, family=spec.family)
        if se is None:            # Hessian không xác định dương -> không có khoảng Wald
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
    """Gộp các lần lặp của MỘT phương pháp thành Bias, RMSE, Coverage theo từng tham số.

    Tham số:
        lan_lap: danh sách kết quả chay_mot_lan_lap.
        that   : giá trị thật {tên: số} (từ truong_hop.tham_so_that).
    Trả về dict: bias, rmse, coverage (dict theo tên tham số), n_replications (B),
    n_co_khoang (B', số lần lặp có khoảng 95%), tham_so_that.
    """
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
