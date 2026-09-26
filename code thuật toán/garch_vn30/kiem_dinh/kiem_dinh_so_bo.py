"""Kiem dinh thong ke ban dau cho chuoi loi suat, truoc khi dung mo hinh GARCH.

  ADF, Phillips-Perron : H0 co nghiem don vi (khong dung)
  KPSS                 : H0 dung (dao chieu gia thuyet so voi ADF/PP)
  Ljung-Box tren r, r^2: H0 khong tu tuong quan den do tre m
  ARCH-LM (Engle)      : H0 khong co hieu ung ARCH
  Jarque-Bera          : H0 phan phoi chuan (do lech 0, do nhon 3)
"""
from __future__ import annotations

import numpy as np
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss


def _ket_luan(p: float, bac_bo: str) -> str:
    return bac_bo if p < 0.05 else "khong du bang chung bac bo H0"


def adf_test(x: np.ndarray, regression: str = "c", autolag: str = "AIC") -> dict:
    stat, p, lag, n_obs, crit, _ = adfuller(x, regression=regression, autolag=autolag)
    return {"test": "ADF", "statistic": float(stat), "p_value": float(p), "used_lag": int(lag),
            "n_obs": int(n_obs), "critical_values": crit,
            "decision": _ket_luan(p, "bac bo H0 (co the dung)")}


def pp_test(x: np.ndarray, trend: str = "c") -> dict:
    res = PhillipsPerron(np.asarray(x), trend=trend)
    return {"test": "Phillips-Perron", "statistic": float(res.stat), "p_value": float(res.pvalue),
            "lags": int(res.lags), "decision": _ket_luan(res.pvalue, "bac bo H0 (co the dung)")}


def kpss_test(x: np.ndarray, regression: str = "c", nlags: str = "auto") -> dict:
    """p-value cua KPSS chi noi suy trong [0,01; 0,10]; ngoai khoang bi cat o hai dau."""
    stat, p, lags, crit = kpss(x, regression=regression, nlags=nlags)
    return {"test": "KPSS", "statistic": float(stat), "p_value": float(p), "lags": int(lags),
            "critical_values": crit, "decision": _ket_luan(p, "bac bo H0 (co the khong dung)")}


def ljung_box_test(x: np.ndarray, lags: int = 10, model_df: int = 0) -> dict:
    """Q(m) = n(n+2) sum_{k=1}^m rho_k^2 / (n-k) ~ chi2(m - model_df)."""
    bang = acorr_ljungbox(x, lags=lags, model_df=model_df, return_df=True)
    cuoi = bang.iloc[-1]
    return {"test": "Ljung-Box", "lags": int(bang.index[-1]), "statistic": float(cuoi["lb_stat"]),
            "p_value": float(cuoi["lb_pvalue"]), "full_table": bang,
            "decision": _ket_luan(cuoi["lb_pvalue"], "bac bo H0 (co tu tuong quan)")}


def arch_lm_test(resid: np.ndarray, nlags: int = 5) -> dict:
    """Hoi quy e_t^2 len e_{t-1}^2..e_{t-q}^2; LM = n R^2 ~ chi2(q)."""
    lm, p, f, f_p = het_arch(resid, nlags=nlags)
    return {"test": "ARCH-LM", "lags": nlags, "statistic": float(lm), "p_value": float(p),
            "f_statistic": float(f), "f_p_value": float(f_p),
            "decision": _ket_luan(p, "bac bo H0 (co hieu ung ARCH)")}


def jarque_bera_test(x: np.ndarray) -> dict:
    """JB = n/6 [S^2 + (K-3)^2 / 4] ~ chi2(2)."""
    jb, p, skew, kurt = jarque_bera(x)
    return {"test": "Jarque-Bera", "statistic": float(jb), "p_value": float(p),
            "skewness": float(skew), "kurtosis": float(kurt),
            "decision": _ket_luan(p, "bac bo H0 chuan")}


def run_all_preliminary_tests(r: np.ndarray, lags_ljung_box: int = 10,
                              lags_arch_lm: int = 5) -> dict:
    """Chay toan bo kiem dinh tren chuoi loi suat r."""
    r = np.asarray(r)
    return {
        "adf_r": adf_test(r),
        "pp_r": pp_test(r),
        "kpss_r": kpss_test(r),
        "ljung_box_r": ljung_box_test(r, lags=lags_ljung_box),
        "ljung_box_r2": ljung_box_test(r ** 2, lags=lags_ljung_box),
        "arch_lm": arch_lm_test(r - r.mean(), nlags=lags_arch_lm),
        "jarque_bera": jarque_bera_test(r),
    }
