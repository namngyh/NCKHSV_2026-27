"""Kiểm định Diebold-Mariano (1995): hai mô hình có độ chính xác dự báo như nhau không?

LÝ THUYẾT
    Với hàm mất mát L (vd. QLIKE từng ngày) của hai mô hình, đặt d_t = L_1t - L_2t.
    H0: E[d_t] = 0 (hai mô hình dự báo tốt như nhau).

        DM = mean(d) / sqrt(S / n),
        S  = gamma_0 + 2 sum_{k=1}^{h-1} (1 - k/h) gamma_k       (Newey-West)

    với gamma_k là hiệp phương sai trễ k của d_t; h là tầm dự báo (h = 1 cho dự
    báo một bước). Hiệu chỉnh mẫu nhỏ Harvey-Leybourne-Newbold (1997): nhân DM
    với sqrt([n + 1 - 2h + h(h-1)/n] / n) và so với phân phối t(n-1).
    DM > 0: mô hình 1 có mất mát trung bình lớn hơn (tệ hơn) mô hình 2.

    Hàm đã được cài đặt và kiểm thử; chưa được dùng trong các bảng hiện có.

THƯ VIỆN
    numpy (trung bình, hiệp phương sai), scipy.stats (phân phối t và chuẩn).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west(d: np.ndarray, lag: int) -> float:
    """Phương sai dài hạn Newey-West của chuỗi d với trọng số Bartlett (1 - k/(lag+1))."""
    d_c = d - d.mean()
    var = np.mean(d_c ** 2)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * np.mean(d_c[k:] * d_c[:-k])
    return var


def diebold_mariano_test(loss_1: np.ndarray, loss_2: np.ndarray, h: int = 1,
                         small_sample_correction: bool = True) -> dict:
    """Kiểm định DM giữa hai chuỗi mất mát cùng độ dài.

    Trả về dict: DM_stat, p_value (hai phía), mean_loss_diff, n_obs.
    """
    d = np.asarray(loss_1, dtype=float) - np.asarray(loss_2, dtype=float)
    n = len(d)
    dm = d.mean() / np.sqrt(_newey_west(d, max(h - 1, 0)) / n)
    if small_sample_correction:
        dm = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = 2.0 * (1.0 - stats.t.cdf(np.abs(dm), df=n - 1))
    else:
        p = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm)))
    return {"DM_stat": float(dm), "p_value": float(p),
            "mean_loss_diff": float(d.mean()), "n_obs": n}
