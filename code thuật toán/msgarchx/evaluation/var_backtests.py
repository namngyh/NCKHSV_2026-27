"""Kiem dinh VaR ngoai mau (Muc 6.9): Kupiec (ty le vi pham) va Christoffersen
(bao phu co dieu kien = ty le + doc lap).

eq:var-hit. Ket qua tra ve dict co thong ke LR va p-value (chi binh phuong).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def kupiec_test(hits: np.ndarray, alpha: float) -> dict:
    """Kiem dinh ty le vi pham vo dieu kien (Kupiec, 1995).

    H0: P(vi pham) = alpha. LR_uc ~ chi2(1) duoi H0.
    """
    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    x = hits.sum()
    pi_hat = x / n if n > 0 else float("nan")

    def _term(count, prob):
        if count == 0:
            return 0.0
        return count * np.log(prob)

    log_l_alpha = _term(n - x, 1 - alpha) + _term(x, alpha)
    log_l_hat = _term(n - x, 1 - pi_hat) + _term(x, pi_hat) if 0 < pi_hat < 1 else 0.0
    lr_uc = -2.0 * (log_l_alpha - log_l_hat)
    p_value = 1.0 - stats.chi2.cdf(lr_uc, df=1)
    return {
        "n_obs": n,
        "n_violations": int(x),
        "violation_rate": pi_hat,
        "alpha": alpha,
        "LR_uc": float(lr_uc),
        "p_value": float(p_value),
    }


def christoffersen_test(hits: np.ndarray, alpha: float) -> dict:
    """Kiem dinh doc lap va bao phu co dieu kien (Christoffersen, 1998).

    LR_ind kiem dinh cac lan vi pham khong tao cum (doc lap Markov bac 1);
    LR_cc = LR_uc + LR_ind ~ chi2(2) la kiem dinh bao phu co dieu kien.
    """
    hits = np.asarray(hits, dtype=int)
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _safe_log_term(count, prob):
        if count == 0:
            return 0.0
        return count * np.log(prob) if prob > 0 else -np.inf

    log_l_restricted = _safe_log_term(n00 + n10, 1 - pi) + _safe_log_term(n01 + n11, pi)
    log_l_unrestricted = (
        _safe_log_term(n00, 1 - pi01)
        + _safe_log_term(n01, pi01)
        + _safe_log_term(n10, 1 - pi11)
        + _safe_log_term(n11, pi11)
    )
    lr_ind = -2.0 * (log_l_restricted - log_l_unrestricted)
    p_ind = 1.0 - stats.chi2.cdf(lr_ind, df=1)

    kupiec = kupiec_test(hits, alpha)
    lr_cc = kupiec["LR_uc"] + lr_ind
    p_cc = 1.0 - stats.chi2.cdf(lr_cc, df=2)

    return {
        "LR_ind": float(lr_ind),
        "p_value_ind": float(p_ind),
        "LR_cc": float(lr_cc),
        "p_value_cc": float(p_cc),
    }
