"""Chuan bi du lieu mo hinh: bien ngoai sinh, chuan hoa, chia tap theo thoi gian.

Bien ngoai sinh (da dich tre mot phien -- chi dung thong tin da biet truoc
ngay t):
    log_volume_lag1 = log(1 + VOL_{t-1})
    log_rv_lag1     = log(RV_{t-1} + 1e-6)
Ca hai duoc chuan hoa (x - mean) / sd voi mean, sd tinh CHI tren tap huan
luyen, de khong ro ri thong tin tap kiem dinh vao mo hinh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from garch_vn30.du_lieu.doc_du_lieu import load_vn30f1m_daily

COT_X = ["log_volume_lag1", "log_rv_lag1"]


def build_exogenous_matrix(daily: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Hai bien ngoai sinh da dich tre, kem cot `date`."""
    out = pd.DataFrame({"date": daily["date"]})
    out["log_volume_lag1"] = np.log1p(daily["volume"]).shift(1)
    out["log_rv_lag1"] = np.log(daily["RV"] + eps).shift(1)
    return out


def standardize_with_train_stats(X: pd.DataFrame, train_mask: pd.Series
                                 ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X - mean_train) / sd_train cho moi cot so (tru `date`)."""
    cot = [c for c in X.columns if c != "date" and pd.api.types.is_numeric_dtype(X[c])]
    mean = X.loc[train_mask, cot].mean()
    std = X.loc[train_mask, cot].std().replace(0.0, 1.0)
    out = X.copy()
    out[cot] = (X[cot] - mean) / std
    return out, mean, std


def chuan_bi_du_lieu(csv_path: str, train_frac: float = 0.8) -> dict:
    """Doc du lieu, dung X, chia 80/20 THEO THOI GIAN (khong xao tron).

    Mat na huan luyen dat tren toan bo phien goc TRUOC khi loc cac dong thieu
    (ngay dau khong co loi suat, ngay dau/ngay thu hai khong co bien tre).
    """
    daily, bao_cao = load_vn30f1m_daily(csv_path)
    X_raw = build_exogenous_matrix(daily)
    n = len(daily)
    train_mask = pd.Series(False, index=X_raw.index)
    train_mask.iloc[: int(n * train_frac)] = True
    X_std, x_mean, x_std = standardize_with_train_stats(X_raw, train_mask)

    hop_le = (daily["log_return"].notna() & X_std["log_volume_lag1"].notna()
              & X_std["log_rv_lag1"].notna() & daily["RV"].notna())
    return {
        "r_all": torch.tensor(daily.loc[hop_le, "log_return"].to_numpy()),
        "X_all": torch.tensor(X_std.loc[hop_le, COT_X].to_numpy()),
        "rv_all": torch.tensor(daily.loc[hop_le, "RV"].to_numpy()),
        "dates_all": daily.loc[hop_le, "date"].reset_index(drop=True),
        "n_train": int(train_mask.loc[hop_le].sum()),
        "quality_report": bao_cao,
        "exog_mean": x_mean,
        "exog_std": x_std,
    }
