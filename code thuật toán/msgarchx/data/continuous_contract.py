"""Xay dung chuoi hop dong tuong lai lien tuc VN30F1M (Muc 5.2).

Chien luoc: (1) chon hop dong chinh moi ngay theo mot tieu chi ro rang;
(2) noi chuoi LOI SUAT thay vi noi chuoi MUC GIA de tranh sai lech tai ngay
chuyen hop dong (Muc 5.2: "noi chuoi theo loi suat thuong it gay sai lech
hon"); (3) danh dau ngay chuyen hop dong bang ROLL_t de mo hinh tu kiem soat
hieu ung ky thuat qua Bang 5.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_front_contract(
    df: pd.DataFrame,
    date_col: str = "date",
    contract_col: str = "contract_code",
    expiry_col: str = "expiry_date",
    volume_col: str = "volume",
    rule: str = "nearest_expiry",
) -> pd.DataFrame:
    """Voi moi ngay, chon mot hop dong chinh theo `rule`:

    - "nearest_expiry": hop dong co ngay dao han gan nhat (>= ngay hien tai).
    - "max_volume": hop dong co khoi luong giao dich cao nhat trong ngay.
    """
    df = df.copy()
    if rule == "nearest_expiry":
        df = df[df[expiry_col] >= df[date_col]]
        idx = df.groupby(date_col)[expiry_col].idxmin()
    elif rule == "max_volume":
        idx = df.groupby(date_col)[volume_col].idxmax()
    else:
        raise ValueError(f"rule khong hop le: {rule!r}")
    return df.loc[idx].sort_values(date_col).reset_index(drop=True)


def build_continuous_series(
    df: pd.DataFrame,
    date_col: str = "date",
    contract_col: str = "contract_code",
    close_col: str = "close",
    expiry_col: str = "expiry_date",
    volume_col: str = "volume",
    rule: str = "nearest_expiry",
) -> pd.DataFrame:
    """Tra ve DataFrame [date, contract_code, close, log_return, is_roll_day,
    price_level] voi price_level la chuoi muc gia da dieu chinh kieu
    "back-adjusted" (Panama) chi dung de ve bieu do, KHONG dung de tinh loi
    suat (loi suat da duoc noi truc tiep, khong qua price_level).
    """
    front = select_front_contract(df, date_col, contract_col, expiry_col, volume_col, rule)
    front = front.sort_values(date_col).reset_index(drop=True)

    front["is_roll_day"] = front[contract_col].ne(front[contract_col].shift(1)) & front.index.to_series().gt(0)
    log_close = np.log(front[close_col])
    front["log_return"] = log_close.diff()
    front.loc[front["is_roll_day"], "log_return"] = np.nan  # xem ham duoi de xu ly rieng ngay roll

    # Tai ngay roll, tinh lai loi suat bang chinh hop dong moi (dung du lieu
    # gia hop dong moi tai t va t-1 neu co); neu khong co du lieu chong lan,
    # giu NaN va ghi log de nguoi dung tu quyet dinh quy tac thay the.
    for i in front.index[front["is_roll_day"]]:
        new_contract = front.loc[i, contract_col]
        prev_date = front.loc[i - 1, date_col] if i > 0 else None
        prev_row = df[(df[contract_col] == new_contract) & (df[date_col] == prev_date)]
        if not prev_row.empty:
            prev_close = prev_row[close_col].iloc[0]
            front.loc[i, "log_return"] = np.log(front.loc[i, close_col] / prev_close)

    first_close = front[close_col].iloc[0]
    cum_return = front["log_return"].fillna(0.0).cumsum()
    front["price_level"] = first_close * np.exp(cum_return)

    return front[[date_col, contract_col, close_col, "log_return", "is_roll_day", "price_level"]]
