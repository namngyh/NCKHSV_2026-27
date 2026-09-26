"""Kiem dinh Diebold-Mariano (1995) so sanh do chinh xac du bao cua hai mo hinh.

d_t = L_1t - L_2t (chenh lech mat mat), H0: E[d_t] = 0.
DM = mean(d) / sqrt(S / n), S la phuong sai dai han Newey-West voi do tre h-1;
hieu chinh mau nho Harvey-Leybourne-Newbold (1997), so voi t(n-1).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _newey_west(d: np.ndarray, lag: int) -> float:
    d_c = d - d.mean()
    var = np.mean(d_c ** 2)
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * np.mean(d_c[k:] * d_c[:-k])
    return var


def diebold_mariano_test(loss_1: np.ndarray, loss_2: np.ndarray, h: int = 1,
                         small_sample_correction: bool = True) -> dict:
    """DM > 0: mo hinh 1 co mat mat trung binh lon hon (te hon) mo hinh 2."""
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
