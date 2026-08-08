"""Kiem dinh Diebold--Mariano (1995) so sanh do chinh xac du bao giua hai mo hinh.

H0: E[d_t] = 0 voi d_t = L(e_1t) - L(e_2t) la chenh lech ham mat mat. Dung
hieu chinh mau nho cua Harvey-Leybourne-Newbold (1997) va sai so chuan HAC
(Newey-West) khi chan doan h > 1 buoc gay tu tuong quan chuoi.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west_long_run_var(d: np.ndarray, lag: int) -> float:
    n = len(d)
    d_centered = d - d.mean()
    gamma0 = np.mean(d_centered**2)
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        gamma_k = np.mean(d_centered[k:] * d_centered[:-k])
        var += 2.0 * w * gamma_k
    return var


def diebold_mariano_test(
    loss_1: np.ndarray,
    loss_2: np.ndarray,
    h: int = 1,
    small_sample_correction: bool = True,
) -> dict:
    """DM = mean(d) / sqrt(Var_hat(mean(d))), voi lag HAC = h-1.

    Gia tri DM > 0 cho thay mo hinh 1 co mat mat trung binh lon hon (te hon)
    mo hinh 2. p-value hai phia tinh tu phan phoi chuan hoac t(n-1) neu dung
    hieu chinh mau nho.
    """
    d = np.asarray(loss_1, dtype=float) - np.asarray(loss_2, dtype=float)
    n = len(d)
    lag = max(h - 1, 0)
    long_run_var = _newey_west_long_run_var(d, lag)
    d_mean = d.mean()
    dm_stat = d_mean / np.sqrt(long_run_var / n)

    if small_sample_correction:
        correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = dm_stat * correction
        p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    else:
        p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(dm_stat)))

    return {
        "DM_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_mean),
        "n_obs": n,
    }
