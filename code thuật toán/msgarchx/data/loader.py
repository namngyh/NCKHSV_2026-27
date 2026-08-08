"""Nap du lieu VN30F1M tu file xuat "ohlc_export.csv" (Muc 5.1).

Dinh dang nguon: SYMBOL, TRADING_DATE (YYYYMMDD), OPEN_PX, HIGH_PX, LOW_PX,
CLOSE_PX, VOL, TRADING_TIME (HH:MM:SS), BUY_VOL, BUY_VAL, SELL_VOL, SELL_VAL
-- day la du lieu THEO PHUT cho MOT ky hieu lien tuc duy nhat "VN30F1M" (nha
cung cap da tu chon hop dong gan nhat, khong co cot ma hop dong/ngay dao han
rieng). Vi vay module `continuous_contract.py` (can nhieu hop dong dong thoi
de tu xay dung ROLL_t/EXP_t) CHUA the ap dung truc tiep tren file nay; khi co
them du lieu ma hop dong + ngay dao han + chi so VN30 co so, nen chuyen sang
dung `continuous_contract.py` va `exogenous.py` (BASIS, EXP, ROLL) day du.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_intraday_csv(path: str) -> pd.DataFrame:
    """Doc file xuat va tra ve DataFrame voi cot `datetime` da ghep tu
    TRADING_DATE + TRADING_TIME, sap xep theo thoi gian tang dan."""
    df = pd.read_csv(path, dtype={"TRADING_DATE": str})
    df["date"] = pd.to_datetime(df["TRADING_DATE"], format="%Y%m%d")
    df["datetime"] = pd.to_datetime(
        df["TRADING_DATE"] + " " + df["TRADING_TIME"], format="%Y%m%d %H:%M:%S"
    )
    df = df.rename(
        columns={
            "OPEN_PX": "open", "HIGH_PX": "high", "LOW_PX": "low",
            "CLOSE_PX": "close", "VOL": "volume",
        }
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["date", "datetime", "SYMBOL", "open", "high", "low", "close", "volume"]]


def data_quality_report(df: pd.DataFrame) -> dict:
    """Bao cao chat luong toi thieu truoc khi lam sach (Muc 5.3)."""
    dup = df.duplicated(subset=["datetime"]).sum()
    ohlc_bad = ~(
        (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df[["open", "close"]].max(axis=1) <= df["high"])
    )
    return {
        "n_rows": len(df),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "n_trading_days": df["date"].nunique(),
        "n_duplicate_datetime": int(dup),
        "n_null": int(df.isna().sum().sum()),
        "n_nonpositive_price": int((df[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "n_nonpositive_volume": int((df["volume"] <= 0).sum()),
        "n_ohlc_violation": int(ohlc_bad.sum()),
        "bars_per_day_min": int(df.groupby("date").size().min()),
        "bars_per_day_median": float(df.groupby("date").size().median()),
    }


def build_daily_bars(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """Gop nen phut thanh nen ngay: open=nen dau, high=max, low=min,
    close=nen cuoi, volume=tong khoi luong (chuan bi cho eq:log-return)."""
    daily = (
        df_intraday.sort_values("datetime")
        .groupby("date")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            n_bars=("volume", "size"),
        )
        .reset_index()
    )
    return daily


def compute_daily_log_returns(daily: pd.DataFrame, close_col: str = "close", scale: float = 100.0) -> pd.DataFrame:
    """r_t = c * (log P_t - log P_{t-1}), eq:log-return, mac dinh c=100."""
    out = daily.copy()
    out["log_return"] = scale * np.log(out[close_col] / out[close_col].shift(1))
    return out


def compute_intraday_log_returns(df_intraday: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Loi suat log giua hai nen phut lien tiep TRONG CUNG MOT NGAY, dung de
    tinh RV_t (eq:realized-volatility). Nen dau tien moi ngay khong co loi
    suat noi ngay truoc do nen duoc dat NaN (khong tinh vao RV)."""
    out = df_intraday.sort_values("datetime").copy()
    out["intraday_log_return"] = np.log(out[close_col] / out.groupby("date")[close_col].shift(1))
    return out


def load_vn30f1m_daily(
    path: str, scale: float = 100.0
) -> tuple[pd.DataFrame, dict]:
    """Ham tien ich tong hop: nap file, gop ngay, tinh loi suat ngay va RV_t
    thuc tu du lieu phut. Tra ve (DataFrame ngay, bao cao chat luong)."""
    raw = load_intraday_csv(path)
    report = data_quality_report(raw)

    daily = build_daily_bars(raw)
    daily = compute_daily_log_returns(daily, scale=scale)

    intraday_ret = compute_intraday_log_returns(raw)
    rv = intraday_ret.groupby("date")["intraday_log_return"].apply(
        lambda x: float(np.sum(x.dropna() ** 2)) * scale**2
    )
    daily = daily.merge(rv.rename("RV"), on="date", how="left")

    return daily, report
