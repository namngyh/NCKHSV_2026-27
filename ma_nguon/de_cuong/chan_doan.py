"""Phát hiện chuỗi NUTS bị kẹt và phân loại kết quả chẩn đoán hội tụ (dùng ở b08, b08b).

Mô-đun riêng, không sửa uoc_luong.py: uoc_luong.py nằm trong mã băm của b02, b03a, b03b và mọi bước theo MLE, nên sửa
nó buộc chạy lại toàn bộ mô phỏng b03b. Chỉ b08 và b08b khai báo mô-đun này trong `thu_vien`.

LÝ THUYẾT
  Chuỗi kẹt. Lần chạy b08 ngày 27/10/2026 có một chuỗi (mốc 37, s0 = 1588, chuỗi 0) nằm suốt 1.000 mẫu ở vùng có log
  hậu nghiệm thấp hơn các chuỗi khác khoảng 3.680 đơn vị; R_hat = 1,59 phát hiện được nhưng đường ống vẫn gộp chuỗi đó.
  Phép thử dưới đây kiểm tra TỪNG chuỗi, không cần các chuỗi khác, nên có thể chạy lại chuỗi kẹt ngay trong đơn vị.

  Với hậu nghiệm gần chuẩn d chiều quanh mốt x*, lp(x) = lp(x*) - Q/2 với Q ~ chi2_d, nên trung vị lp của mẫu xấp xỉ
  lp(x*) - (d - 2/3)/2. Mốt là cực đại nên lp(x*) >= lp(xi_MLE) với mọi điểm tham chiếu, kể cả khi tiên nghiệm làm mốt
  lệch khỏi MLE. Do đó thiếu hụt = lp(xi_MLE) - trung vị lp(chuỗi) của một chuỗi đã hội tụ không vượt quá khoảng d/2
  (có thể âm). Chuỗi bị coi là kẹt khi thiếu hụt > d/2 + he_so * sqrt(d/2) (sqrt(d/2) là độ lệch chuẩn của Q/2).
  Trên 260 chuỗi thật của lần chạy 27/10 (d = 11, ngưỡng 17,2 với he_so = 5): 259 chuỗi có thiếu hụt từ -0,84 đến 4,71;
  chuỗi kẹt có 3.679.

  Phân loại chẩn đoán. Điều kiện của đề cương (mục 4.3.4) gộp hai thứ khác bản chất: hội tụ (R_hat < 1,01 và ESS >= 400)
  và không có bước phân kỳ nào. `phan_loai` giữ nguyên kết luận "dat" của đề cương và tách thêm hai phần đó để báo cáo.

THƯ VIỆN: numpy, jax (log hậu nghiệm trong không gian xi từ mo_hinh_jax.ham).
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .mo_hinh_jax import TIEN_NGHIEM_GOC, DacTa, ham


def nguong_thieu_hut(d: int, he_so: float) -> float:
    """Ngưỡng thiếu hụt log hậu nghiệm để coi một chuỗi là kẹt.

    Tham số: d là số tham số; he_so là số độ lệch chuẩn của Q/2 (Q ~ chi2_d) cộng thêm vào d/2.
    Trả về: d/2 + he_so * sqrt(d/2)."""
    return d / 2 + he_so * math.sqrt(d / 2)


def kiem_tra_chuoi(spec: DacTa, xi_mau, xi_tham_chieu, dl, he_so: float = 5.0, pa=TIEN_NGHIEM_GOC) -> dict:
    """Kiểm tra một chuỗi NUTS có bị kẹt ở vùng log hậu nghiệm thấp hay không.

    Tham số: spec là đặc tả mô hình; xi_mau là mẫu của chuỗi trong không gian xi (n x d); xi_tham_chieu là điểm tham
        chiếu (xi của MLE trên cùng dữ liệu); dl là bộ dữ liệu của uoc_luong.chuan_bi; he_so như nguong_thieu_hut; pa là
        bộ tiên nghiệm mà chuỗi dùng.
    Trả về: dict gồm lp_tham_chieu, lp_trung_vi, thieu_hut, nguong và ket (True khi thiếu hụt vượt ngưỡng hoặc không
        tính được)."""
    h = ham(spec.pp, spec.nx, pa)
    lp = np.asarray(h.lp_xi_vmap(jnp.asarray(xi_mau), dl), float)
    lp0 = float(h.lp_xi(jnp.asarray(xi_tham_chieu), dl))
    tv = float(np.median(lp)) if np.isfinite(lp).all() else float("nan")
    thieu = lp0 - tv
    ng = nguong_thieu_hut(spec.d, he_so)
    return {"lp_tham_chieu": lp0, "lp_trung_vi": tv, "thieu_hut": thieu, "nguong": ng, "ket": not thieu <= ng}


def chon_chuoi(cac_chuoi: list[dict]) -> tuple[list[dict], list[int]]:
    """Bỏ các chuỗi vẫn kẹt sau khi đã chạy lại (cờ chuoi_ket.ket của đơn vị b08/b08b).

    Tham số: cac_chuoi là danh sách kết quả chuỗi theo thứ tự chỉ số chuỗi.
    Trả về: (các chuỗi giữ lại, chỉ số các chuỗi bị loại). Chuỗi không có cờ được giữ."""
    giu, loai = [], []
    for i, c in enumerate(cac_chuoi):
        if c.get("chuoi_ket", {}).get("ket"):
            loai.append(i)
        else:
            giu.append(c)
    return giu, loai


def phan_loai(cd: dict, rhat_max: float, ess_min: float) -> dict:
    """Tách kết luận "dat" của đề cương thành hội tụ (R_hat, ESS) và phân kỳ.

    Tham số: cd là kết quả uoc_luong.chan_doan_nuts; rhat_max, ess_min là ngưỡng của [bayes].
    Trả về: dict gồm hoi_tu_dat (R_hat và ESS đạt), phan_ky (số bước phân kỳ) và chi_vi_phan_ky (hội tụ đạt nhưng
        không đạt đề cương chỉ vì có bước phân kỳ)."""
    hoi_tu = bool(cd["rhat_max"] < rhat_max and cd["ess_min"] >= ess_min)
    return {"hoi_tu_dat": hoi_tu, "phan_ky": int(cd["phan_ky"]), "chi_vi_phan_ky": bool(hoi_tu and cd["phan_ky"] > 0)}
