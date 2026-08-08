"""Xay dung bien ngoai sinh X_{t-1} cho phuong trinh phuong sai (Bang 5.4).

Tat ca bien lien tuc duoc chuan hoa CHI bang thong ke cua tap huan luyen
(Muc 5.6, Muc 5.9): `standardize_with_train_stats` nhan mot mat na tap huan
luyen va tuyet doi khong dung thong ke cua toan mau.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_exogenous_matrix(
    df: pd.DataFrame,
    date_col: str = "date",
    volume_col: str = "volume",
    rv_col: str = "RV",
    basis_col: str | None = "basis",
    expiry_dummy_col: str | None = "is_expiry",
    roll_dummy_col: str | None = "is_roll_day",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Tra ve DataFrame voi cac cot da dich ve dang X_{t-1} (Bang 5.4):

    - log_volume_lag1   = log(1+VOL_{t-1})
    - log_rv_lag1       = log(RV_{t-1} + eps)
    - basis_lag1        = BASIS_{t-1} (neu co)
    - is_expiry         = EXP_t (bien lich, biet truoc nen KHONG dich)
    - is_roll_day       = ROLL_t (bien lich, KHONG dich)
    """
    out = pd.DataFrame({date_col: df[date_col]})
    out["log_volume_lag1"] = np.log1p(df[volume_col]).shift(1)
    out["log_rv_lag1"] = np.log(df[rv_col] + eps).shift(1)
    if basis_col is not None and basis_col in df.columns:
        out["basis_lag1"] = df[basis_col].shift(1)
    if expiry_dummy_col is not None and expiry_dummy_col in df.columns:
        out["is_expiry"] = df[expiry_dummy_col].astype(float)
    if roll_dummy_col is not None and roll_dummy_col in df.columns:
        out["is_roll_day"] = df[roll_dummy_col].astype(float)
    return out


def standardize_with_train_stats(
    X: pd.DataFrame, train_mask: pd.Series, exclude_cols: tuple[str, ...] = ()
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Chuan hoa (X - mean_train) / std_train, voi mean/std chi tinh tren
    `train_mask`. Cac cot trong `exclude_cols` (vd. cot ngay, bien nhi phan
    da biet truoc) duoc giu nguyen.
    """
    numeric_cols = [c for c in X.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(X[c])]
    mean = X.loc[train_mask, numeric_cols].mean()
    std = X.loc[train_mask, numeric_cols].std().replace(0.0, 1.0)
    out = X.copy()
    out[numeric_cols] = (X[numeric_cols] - mean) / std
    return out, mean, std
