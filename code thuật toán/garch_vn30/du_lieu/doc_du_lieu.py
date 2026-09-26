"""Doc du lieu phut VN30F1M tu "ohlc_export.csv" va dung chuoi ngay.

Dinh dang nguon (moi dong mot nen 1 phut cua ky hieu lien tuc VN30F1M do nha
cung cap ghep san):
    SYMBOL, TRADING_DATE (YYYYMMDD), OPEN_PX, HIGH_PX, LOW_PX, CLOSE_PX, VOL,
    TRADING_TIME (HH:MM:SS), BUY_VOL, BUY_VAL, SELL_VOL, SELL_VAL

Tu du lieu phut tinh:
    r_t  = 100 (log P_t - log P_{t-1})            loi suat log ngay (%)
    RV_t = 100^2 sum_j (log P_{t,j} - log P_{t,j-1})^2   bien dong thuc te (%^2)
voi P_t la gia dong cua cua nen cuoi ngay t, P_{t,j} la gia dong cua nen phut
thu j trong ngay t (khong noi loi suat qua dem vao RV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_intraday_csv(path: str) -> pd.DataFrame:
    """Doc file, ghep cot `datetime`, doi ten cot gia/khoi luong, sap xep theo thoi gian."""
    df = pd.read_csv(path, dtype={"TRADING_DATE": str})
    df["date"] = pd.to_datetime(df["TRADING_DATE"], format="%Y%m%d")
    df["datetime"] = pd.to_datetime(df["TRADING_DATE"] + " " + df["TRADING_TIME"],
                                    format="%Y%m%d %H:%M:%S")
    df = df.rename(columns={"OPEN_PX": "open", "HIGH_PX": "high", "LOW_PX": "low",
                            "CLOSE_PX": "close", "VOL": "volume"})
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["date", "datetime", "SYMBOL", "open", "high", "low", "close", "volume"]]


def data_quality_report(df: pd.DataFrame) -> dict:
    """Bao cao chat luong du lieu phut: trung lap, gia tri thieu, gia/khoi luong khong duong,
    nen vi pham Low <= min(Open, Close) <= max(Open, Close) <= High."""
    ohlc_bad = ~((df["low"] <= df[["open", "close"]].min(axis=1))
                 & (df[["open", "close"]].max(axis=1) <= df["high"]))
    so_nen = df.groupby("date").size()
    return {
        "n_rows": len(df),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "n_trading_days": df["date"].nunique(),
        "n_duplicate_datetime": int(df.duplicated(subset=["datetime"]).sum()),
        "n_null": int(df.isna().sum().sum()),
        "n_nonpositive_price": int((df[["open", "high", "low", "close"]] <= 0).sum().sum()),
        "n_nonpositive_volume": int((df["volume"] <= 0).sum()),
        "n_ohlc_violation": int(ohlc_bad.sum()),
        "bars_per_day_min": int(so_nen.min()),
        "bars_per_day_median": float(so_nen.median()),
    }


def build_daily_bars(df_intraday: pd.DataFrame) -> pd.DataFrame:
    """Gop nen phut thanh nen ngay: open dau, high max, low min, close cuoi, volume tong."""
    return (df_intraday.sort_values("datetime")
            .groupby("date")
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum"),
                 n_bars=("volume", "size"))
            .reset_index())


def compute_daily_log_returns(daily: pd.DataFrame, close_col: str = "close",
                              scale: float = 100.0) -> pd.DataFrame:
    """r_t = scale * log(P_t / P_{t-1}); ngay dau tien la NaN."""
    out = daily.copy()
    out["log_return"] = scale * np.log(out[close_col] / out[close_col].shift(1))
    return out


def compute_intraday_log_returns(df_intraday: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Loi suat log giua hai nen phut lien tiep TRONG CUNG MOT NGAY (nen dau ngay = NaN)."""
    out = df_intraday.sort_values("datetime").copy()
    out["intraday_log_return"] = np.log(out[close_col] / out.groupby("date")[close_col].shift(1))
    return out


def load_vn30f1m_daily(path: str, scale: float = 100.0) -> tuple[pd.DataFrame, dict]:
    """Nap file -> (DataFrame ngay co cot log_return va RV, bao cao chat luong)."""
    raw = load_intraday_csv(path)
    report = data_quality_report(raw)
    daily = compute_daily_log_returns(build_daily_bars(raw), scale=scale)
    rv = (compute_intraday_log_returns(raw)
          .groupby("date")["intraday_log_return"]
          .apply(lambda x: float(np.sum(x.dropna() ** 2)) * scale ** 2))
    return daily.merge(rv.rename("RV"), on="date", how="left"), report
