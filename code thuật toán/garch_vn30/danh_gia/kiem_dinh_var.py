"""Gia tri chiu rui ro (VaR) va kiem dinh hau kiem (backtest).

VaR muc a cua vi the mua tai t:  VaR_t(a) = -(mu + sqrt(h_t) q_a),
voi q_a la phan vi muc a cua z_t. Vi pham: I_t = 1{ r_t < -VaR_t(a) }.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def phan_vi_z(a: float, dist: str, nu: float | None) -> float:
    """Phan vi muc a cua sai so chuan hoa z (phuong sai 1).

    Student-t chuan hoa: q_a = t_{nu}^{-1}(a) * sqrt((nu-2)/nu).
    """
    if dist == "normal":
        return float(stats.norm.ppf(a))
    return float(stats.t.ppf(a, df=nu) * ((nu - 2.0) / nu) ** 0.5)


def kupiec_test(hits: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995), bao phu vo dieu kien. H0: P(I_t = 1) = alpha.

    LR_uc = -2 [ log L(alpha) - log L(pi_hat) ] ~ chi2(1),  pi_hat = x/n.
    """
    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    x = hits.sum()
    pi_hat = x / n if n > 0 else float("nan")

    def _term(count, prob):
        return 0.0 if count == 0 else count * np.log(prob)

    log_l_alpha = _term(n - x, 1 - alpha) + _term(x, alpha)
    log_l_hat = _term(n - x, 1 - pi_hat) + _term(x, pi_hat) if 0 < pi_hat < 1 else 0.0
    lr_uc = -2.0 * (log_l_alpha - log_l_hat)
    return {
        "n_obs": n,
        "n_violations": int(x),
        "violation_rate": pi_hat,
        "alpha": alpha,
        "LR_uc": float(lr_uc),
        "p_value": float(1.0 - stats.chi2.cdf(lr_uc, df=1)),
    }


def christoffersen_test(hits: np.ndarray, alpha: float) -> dict:
    """Christoffersen (1998): doc lap (xich Markov bac 1) va bao phu co dieu kien.

    LR_ind so sanh mo hinh Markov (pi_01, pi_11) voi mo hinh doc lap (pi);
    LR_cc = LR_uc + LR_ind ~ chi2(2).
    """
    hits = np.asarray(hits, dtype=int)
    n00 = np.sum((hits[:-1] == 0) & (hits[1:] == 0))
    n01 = np.sum((hits[:-1] == 0) & (hits[1:] == 1))
    n10 = np.sum((hits[:-1] == 1) & (hits[1:] == 0))
    n11 = np.sum((hits[:-1] == 1) & (hits[1:] == 1))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _term(count, prob):
        if count == 0:
            return 0.0
        return count * np.log(prob) if prob > 0 else -np.inf

    log_l_doc_lap = _term(n00 + n10, 1 - pi) + _term(n01 + n11, pi)
    log_l_markov = (_term(n00, 1 - pi01) + _term(n01, pi01)
                    + _term(n10, 1 - pi11) + _term(n11, pi11))
    lr_ind = -2.0 * (log_l_doc_lap - log_l_markov)
    lr_cc = kupiec_test(hits, alpha)["LR_uc"] + lr_ind
    return {
        "LR_ind": float(lr_ind),
        "p_value_ind": float(1.0 - stats.chi2.cdf(lr_ind, df=1)),
        "LR_cc": float(lr_cc),
        "p_value_cc": float(1.0 - stats.chi2.cdf(lr_cc, df=2)),
    }
