"""Bien dong thuc te tu du lieu noi ngay (Muc 5.4). eq:realized-volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_volatility_from_intraday(
    df_intraday: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "intraday_log_return",
) -> pd.Series:
    """RV_t = sum_j r_{t,j}^2, gom nhom theo ngay (eq:realized-volatility).

    Du lieu dau vao phai da duoc lay mau theo mot tan suat co dinh (vd. 5
    hoac 15 phut) duoc quyet dinh TRUOC va nhat quan toan bo mau.
    """
    return df_intraday.groupby(date_col)[return_col].apply(lambda x: np.sum(x**2))


def realized_volatility_proxy(daily_return: pd.Series) -> pd.Series:
    """Khi khong co du lieu noi ngay du tin cay, dung r_t^2 lam dai dien bien
    dong (nhieu hon RV thuc su, xem thao luan Muc 5.4)."""
    return daily_return**2
