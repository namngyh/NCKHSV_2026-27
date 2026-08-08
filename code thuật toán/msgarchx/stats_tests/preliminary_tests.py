"""Kiem dinh thong ke ban dau (Muc 5): tinh dung, tu tuong quan, hieu ung
ARCH, phan phoi. Tuong ung eq:adf, eq:ljung-box, eq:arch-lm, eq:jarque-bera.

Ket qua tra ve la dict de co the do truc tiep vao Bang 5.5
(tab:initial-tests-template) ma khong can dinh dang lai.
"""
from __future__ import annotations

import numpy as np
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(x: np.ndarray, regression: str = "c", autolag: str = "AIC") -> dict:
    """Augmented Dickey-Fuller: H0 co nghiem don vi (khong dung). eq:adf."""
    stat, pvalue, used_lag, n_obs, crit, _ = adfuller(x, regression=regression, autolag=autolag)
    return {
        "test": "ADF",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": crit,
        "decision": "bac bo H0 (co the dung)" if pvalue < 0.05 else "khong du bang chung bac bo H0",
    }


def pp_test(x: np.ndarray, trend: str = "c") -> dict:
    """Phillips-Perron: H0 co nghiem don vi, hieu chinh phi tham so cho tu
    tuong quan va phuong sai thay doi cua nhieu."""
    res = PhillipsPerron(np.asarray(x), trend=trend)
    return {
        "test": "Phillips-Perron",
        "statistic": float(res.stat),
        "p_value": float(res.pvalue),
        "lags": int(res.lags),
        "decision": "bac bo H0 (co the dung)" if res.pvalue < 0.05 else "khong du bang chung bac bo H0",
    }


def kpss_test(x: np.ndarray, regression: str = "c", nlags: str = "auto") -> dict:
    """KPSS: H0 dung quanh muc (hoac quanh xu the). eq (dao chieu gia thuyet so voi ADF/PP).

    Luu y: p-value cua KPSS chi noi suy chinh xac trong khoang [0.01, 0.10];
    statsmodels tra ve canh bao va cat gia tri o hai dau khoang nay.
    """
    stat, pvalue, lags, crit = kpss(x, regression=regression, nlags=nlags)
    return {
        "test": "KPSS",
        "statistic": float(stat),
        "p_value": float(pvalue),
        "lags": int(lags),
        "critical_values": crit,
        "decision": "bac bo H0 (co the khong dung)" if pvalue < 0.05 else "khong du bang chung bac bo H0",
    }


def ljung_box_test(x: np.ndarray, lags: int | list[int] = 10, model_df: int = 0) -> dict:
    """Thong ke Q(m) cho tu tuong quan den do tre m, eq:ljung-box.

    Ap dung cho r_t (dong luc phuong trinh trung binh) hoac r_t^2 (dong luc
    bien dong). `model_df` la so tham so da uoc luong can tru bac tu do.
    """
    df = acorr_ljungbox(x, lags=lags, model_df=model_df, return_df=True)
    last = df.iloc[-1]
    return {
        "test": "Ljung-Box",
        "lags": int(df.index[-1]),
        "statistic": float(last["lb_stat"]),
        "p_value": float(last["lb_pvalue"]),
        "full_table": df,
        "decision": "bac bo H0 (co tu tuong quan)" if last["lb_pvalue"] < 0.05 else "khong du bang chung bac bo H0",
    }


def arch_lm_test(resid: np.ndarray, nlags: int = 5) -> dict:
    """Kiem dinh ARCH-LM (hoi quy binh phuong phan du), eq:arch-lm."""
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(resid, nlags=nlags)
    return {
        "test": "ARCH-LM",
        "lags": nlags,
        "statistic": float(lm_stat),
        "p_value": float(lm_pvalue),
        "f_statistic": float(f_stat),
        "f_p_value": float(f_pvalue),
        "decision": "bac bo H0 (co hieu ung ARCH)" if lm_pvalue < 0.05 else "khong du bang chung bac bo H0",
    }


def jarque_bera_test(x: np.ndarray) -> dict:
    """Kiem dinh Jarque-Bera cho do lech va do nhon, eq:jarque-bera."""
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(x)
    return {
        "test": "Jarque-Bera",
        "statistic": float(jb_stat),
        "p_value": float(jb_pvalue),
        "skewness": float(skew),
        "kurtosis": float(kurtosis),
        "decision": "bac bo H0 chuan" if jb_pvalue < 0.05 else "khong du bang chung bac bo H0 chuan",
    }


def run_all_preliminary_tests(
    r: np.ndarray,
    lags_ljung_box: int = 10,
    lags_arch_lm: int = 5,
) -> dict:
    """Tong hop toan bo kiem dinh trong Muc 5, dien truc tiep vao Bang 5.5."""
    r = np.asarray(r)
    r2 = r**2
    return {
        "adf_r": adf_test(r),
        "pp_r": pp_test(r),
        "kpss_r": kpss_test(r),
        "ljung_box_r": ljung_box_test(r, lags=lags_ljung_box),
        "ljung_box_r2": ljung_box_test(r2, lags=lags_ljung_box),
        "arch_lm": arch_lm_test(r - r.mean(), nlags=lags_arch_lm),
        "jarque_bera": jarque_bera_test(r),
    }
