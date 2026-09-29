"""Giá trị chịu rủi ro (Value at Risk, VaR) và kiểm định hậu kiểm (backtest).

LÝ THUYẾT
    VaR mức a (a = 1% hoặc 5%) của vị thế mua là mức lỗ chỉ bị vượt với xác suất a:
        VaR_t(a) = -( mu + sqrt(h_t) * q_a ),
    với q_a là phân vị mức a của sai số chuẩn hóa z. Một "vi phạm" xảy ra khi
    lợi suất thực tế tệ hơn mức dự báo:
        I_t = 1{ r_t < -VaR_t(a) }.
    Mô hình tốt thì tỷ lệ vi phạm xấp xỉ a và các vi phạm không dồn thành cụm.

    Kiểm định Kupiec (1995) -- bao phủ vô điều kiện, H0: P(I_t = 1) = a.
        LR_uc = -2 [ log L(a) - log L(pi_hat) ] ~ chi2(1),   pi_hat = x / n,
        L(p) = (1-p)^(n-x) p^x  (x = số vi phạm trong n phiên).

    Kiểm định Christoffersen (1998) -- tính độc lập và bao phủ có điều kiện.
        n_ij = số lần chuỗi I_t chuyển từ trạng thái i sang j;
        pi_01 = n_01/(n_00+n_01), pi_11 = n_11/(n_10+n_11): xác suất vi phạm
        sau một ngày không vi phạm / sau một ngày vi phạm.
        LR_ind so sánh mô hình xích Markov bậc 1 (pi_01, pi_11) với mô hình độc
        lập (một xác suất pi chung) ~ chi2(1); bác bỏ nghĩa là vi phạm dồn cụm.
        LR_cc = LR_uc + LR_ind ~ chi2(2).

THƯ VIỆN
    scipy.stats: norm.ppf, t.ppf (phân vị), chi2.cdf (giá trị p). numpy: đếm, log.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def phan_vi_z(a: float, dist: str, nu: float | None) -> float:
    """Phân vị mức a của sai số chuẩn hóa z (phương sai 1).

    Chuẩn: q_a = Phi^{-1}(a).  Student-t chuẩn hóa: q_a = t_nu^{-1}(a) * sqrt((nu-2)/nu).
    """
    if dist == "normal":
        return float(stats.norm.ppf(a))
    return float(stats.t.ppf(a, df=nu) * ((nu - 2.0) / nu) ** 0.5)


def kupiec_test(hits: np.ndarray, alpha: float) -> dict:
    """Kiểm định Kupiec cho chuỗi vi phạm `hits` (mảng 0/1) ở mức `alpha`.

    Trả về dict: n_obs, n_violations, violation_rate, alpha, LR_uc, p_value.
    """
    hits = np.asarray(hits, dtype=float)
    n = len(hits)
    x = hits.sum()
    pi_hat = x / n if n > 0 else float("nan")

    def _term(count, prob):
        """count * log(prob), quy ước 0 * log(0) = 0."""
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
    """Kiểm định Christoffersen: trả về LR_ind, p_value_ind, LR_cc, p_value_cc."""
    hits = np.asarray(hits, dtype=int)
    # Đếm các cặp (ngày trước, ngày sau).
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
