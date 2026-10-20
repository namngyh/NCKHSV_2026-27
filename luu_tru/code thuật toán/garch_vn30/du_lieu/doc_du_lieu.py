"""Đọc dữ liệu phút VN30F1M từ "ohlc_export.csv" và dựng chuỗi ngày.

DỮ LIỆU NGUỒN
    Mỗi dòng là một nến 1 phút của chuỗi hợp đồng liên tục VN30F1M (nhà cung cấp
    đã ghép sẵn hợp đồng gần nhất). Các cột:
        SYMBOL, TRADING_DATE (YYYYMMDD), OPEN_PX, HIGH_PX, LOW_PX, CLOSE_PX, VOL,
        TRADING_TIME (HH:MM:SS), BUY_VOL, BUY_VAL, SELL_VOL, SELL_VAL
    Ví dụ (giá trị giả định): VN30F1M, 20171107, 850.1, 850.5, 849.9, 850.3, 120, 09:01:00, ...

CÔNG THỨC
    Gọi P_t là giá đóng cửa của nến cuối ngày t, P_{t,j} là giá đóng cửa nến phút
    thứ j trong ngày t:
        lợi suất log ngày (%):         r_t  = 100 (log P_t - log P_{t-1})
        biến động thực tế (%^2):       RV_t = 100^2 sum_j (log P_{t,j} - log P_{t,j-1})^2
    RV_t chỉ cộng lợi suất giữa hai nến TRONG CÙNG một ngày (không gồm khoảng qua
    đêm); nó là đại diện quan sát được cho phương sai ngày, dùng để đánh giá dự
    báo phương sai h_t.

THƯ VIỆN
    pandas: đọc CSV (read_csv), chuyển chuỗi thành ngày giờ (to_datetime), gom
            nhóm theo ngày (groupby ... agg), dịch dữ liệu (shift), ghép bảng (merge).
    numpy : log, tổng bình phương.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_intraday_csv(path: str) -> pd.DataFrame:
    """Đọc tệp CSV phút, tạo cột `date` và `datetime`, đổi tên cột giá/khối lượng,
    sắp xếp theo thời gian tăng dần.

    Trả về DataFrame với các cột: date, datetime, SYMBOL, open, high, low, close, volume.
    """
    df = pd.read_csv(path, dtype={"TRADING_DATE": str})
    df["date"] = pd.to_datetime(df["TRADING_DATE"], format="%Y%m%d")
    df["datetime"] = pd.to_datetime(df["TRADING_DATE"] + " " + df["TRADING_TIME"],
                                    format="%Y%m%d %H:%M:%S")
    df = df.rename(columns={"OPEN_PX": "open", "HIGH_PX": "high", "LOW_PX": "low",
                            "CLOSE_PX": "close", "VOL": "volume"})
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["date", "datetime", "SYMBOL", "open", "high", "low", "close", "volume"]]


def data_quality_report(df: pd.DataFrame) -> dict:
    """Báo cáo chất lượng dữ liệu phút: số dòng trùng thời điểm, giá trị thiếu,
    giá hoặc khối lượng không dương, nến vi phạm Low <= min(Open, Close) <=
    max(Open, Close) <= High, số nến mỗi ngày. Chỉ báo cáo, không xóa dòng nào."""
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
    """Gộp nến phút thành nến ngày: open = nến đầu ngày, high = cao nhất, low = thấp
    nhất, close = nến cuối ngày, volume = tổng khối lượng, n_bars = số nến."""
    return (df_intraday.sort_values("datetime")
            .groupby("date")
            .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum"),
                 n_bars=("volume", "size"))
            .reset_index())


def compute_daily_log_returns(daily: pd.DataFrame, close_col: str = "close",
                              scale: float = 100.0) -> pd.DataFrame:
    """Thêm cột log_return: r_t = scale * log(P_t / P_{t-1}); ngày đầu là NaN
    (shift(1) lấy giá của hàng phía trên, tức ngày trước)."""
    out = daily.copy()
    out["log_return"] = scale * np.log(out[close_col] / out[close_col].shift(1))
    return out


def compute_intraday_log_returns(df_intraday: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Thêm cột intraday_log_return = log lợi suất giữa hai nến phút liên tiếp TRONG
    CÙNG NGÀY. groupby("date").shift(1) chỉ dịch trong từng ngày, nên nến đầu ngày
    nhận NaN và lợi suất qua đêm không bị tính vào RV."""
    out = df_intraday.sort_values("datetime").copy()
    out["intraday_log_return"] = np.log(out[close_col] / out.groupby("date")[close_col].shift(1))
    return out


def load_vn30f1m_daily(path: str, scale: float = 100.0) -> tuple[pd.DataFrame, dict]:
    """Hàm tổng hợp: đọc tệp phút -> bảng ngày có cột log_return và RV, kèm báo cáo chất lượng.

    Trả về (daily, report); daily có các cột date, open, high, low, close, volume,
    n_bars, log_return, RV.
    """
    raw = load_intraday_csv(path)
    report = data_quality_report(raw)
    daily = compute_daily_log_returns(build_daily_bars(raw), scale=scale)
    rv = (compute_intraday_log_returns(raw)
          .groupby("date")["intraday_log_return"]
          .apply(lambda x: float(np.sum(x.dropna() ** 2)) * scale ** 2))
    return daily.merge(rv.rename("RV"), on="date", how="left"), report
