"""Kiểm định thống kê ban đầu cho chuỗi lợi suất, trước khi dùng mô hình GARCH.

MỤC ĐÍCH VÀ LÝ THUYẾT
    Các kiểm định này trả lời: chuỗi lợi suất có phù hợp để mô hình hóa bằng
    GARCH không?
    - ADF (Dickey-Fuller mở rộng) và Phillips-Perron: hồi quy
          Delta r_t = c + rho r_{t-1} + ...;   H0: rho = 0 (có nghiệm đơn vị, chuỗi không dừng).
      PP hiệu chỉnh tự tương quan và phương sai thay đổi bằng phương pháp phi tham số.
    - KPSS: đảo chiều giả thuyết, H0: chuỗi dừng quanh mức. Dùng cùng ADF/PP để
      kết luận chắc hơn.
    - Ljung-Box: Q(m) = n(n+2) sum_{k=1}^m rho_k^2 / (n-k) ~ chi2(m);
      H0: không có tự tương quan đến độ trễ m. Áp dụng trên r_t (động lực trung
      bình) và trên r_t^2 (động lực biến động).
    - ARCH-LM (Engle, 1982): hồi quy e_t^2 lên e_{t-1}^2, ..., e_{t-q}^2;
      LM = n R^2 ~ chi2(q). Bác bỏ H0 nghĩa là có hiệu ứng ARCH (biến động
      theo cụm) -- cơ sở để dùng mô hình GARCH.
    - Jarque-Bera: JB = n/6 [S^2 + (K-3)^2 / 4] ~ chi2(2), S là độ lệch, K là
      độ nhọn; bác bỏ H0 phân phối chuẩn gợi ý dùng sai số đuôi dày (Student-t).

THƯ VIỆN
    statsmodels: adfuller, kpss, acorr_ljungbox, het_arch, jarque_bera.
    arch       : PhillipsPerron.
    Mỗi hàm trả về dict gồm thống kê, giá trị p và kết luận ở mức ý nghĩa 5%.
"""
from __future__ import annotations

import numpy as np
from arch.unitroot import PhillipsPerron
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss


def _ket_luan(p: float, bac_bo: str) -> str:
    """Kết luận ở mức 5%: trả về `bac_bo` nếu p < 0,05."""
    return bac_bo if p < 0.05 else "khong du bang chung bac bo H0"


def adf_test(x: np.ndarray, regression: str = "c", autolag: str = "AIC") -> dict:
    """Kiểm định ADF (hằng số trong hồi quy; độ trễ chọn theo AIC)."""
    stat, p, lag, n_obs, crit, _ = adfuller(x, regression=regression, autolag=autolag)
    return {"test": "ADF", "statistic": float(stat), "p_value": float(p), "used_lag": int(lag),
            "n_obs": int(n_obs), "critical_values": crit,
            "decision": _ket_luan(p, "bac bo H0 (co the dung)")}


def pp_test(x: np.ndarray, trend: str = "c") -> dict:
    """Kiểm định Phillips-Perron."""
    res = PhillipsPerron(np.asarray(x), trend=trend)
    return {"test": "Phillips-Perron", "statistic": float(res.stat), "p_value": float(res.pvalue),
            "lags": int(res.lags), "decision": _ket_luan(res.pvalue, "bac bo H0 (co the dung)")}


def kpss_test(x: np.ndarray, regression: str = "c", nlags: str = "auto") -> dict:
    """Kiểm định KPSS. Giá trị p chỉ được nội suy trong [0,01; 0,10]; ngoài khoảng
    này statsmodels cắt ở hai đầu và phát cảnh báo."""
    stat, p, lags, crit = kpss(x, regression=regression, nlags=nlags)
    return {"test": "KPSS", "statistic": float(stat), "p_value": float(p), "lags": int(lags),
            "critical_values": crit, "decision": _ket_luan(p, "bac bo H0 (co the khong dung)")}


def ljung_box_test(x: np.ndarray, lags: int = 10, model_df: int = 0) -> dict:
    """Kiểm định Ljung-Box đến độ trễ `lags`; trả về cả bảng mọi độ trễ (full_table)."""
    bang = acorr_ljungbox(x, lags=lags, model_df=model_df, return_df=True)
    cuoi = bang.iloc[-1]
    return {"test": "Ljung-Box", "lags": int(bang.index[-1]), "statistic": float(cuoi["lb_stat"]),
            "p_value": float(cuoi["lb_pvalue"]), "full_table": bang,
            "decision": _ket_luan(cuoi["lb_pvalue"], "bac bo H0 (co tu tuong quan)")}


def arch_lm_test(resid: np.ndarray, nlags: int = 5) -> dict:
    """Kiểm định ARCH-LM với `nlags` độ trễ của phần dư bình phương."""
    lm, p, f, f_p = het_arch(resid, nlags=nlags)
    return {"test": "ARCH-LM", "lags": nlags, "statistic": float(lm), "p_value": float(p),
            "f_statistic": float(f), "f_p_value": float(f_p),
            "decision": _ket_luan(p, "bac bo H0 (co hieu ung ARCH)")}


def jarque_bera_test(x: np.ndarray) -> dict:
    """Kiểm định Jarque-Bera, kèm độ lệch và độ nhọn mẫu."""
    jb, p, skew, kurt = jarque_bera(x)
    return {"test": "Jarque-Bera", "statistic": float(jb), "p_value": float(p),
            "skewness": float(skew), "kurtosis": float(kurt),
            "decision": _ket_luan(p, "bac bo H0 chuan")}


def run_all_preliminary_tests(r: np.ndarray, lags_ljung_box: int = 10,
                              lags_arch_lm: int = 5) -> dict:
    """Chạy toàn bộ kiểm định trên chuỗi lợi suất r (nội dung results/01_preliminary_tests.pt).

    Khóa: adf_r, pp_r, kpss_r, ljung_box_r, ljung_box_r2, arch_lm, jarque_bera.
    """
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
