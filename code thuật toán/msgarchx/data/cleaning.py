"""Lam sach va dong bo du lieu (Muc 5.3): loai trung, kiem tra OHLC, luoi thoi gian.

Moi ham tra ve them mot bao cao (`report`) ghi lai so quan sat bi loai/thay
doi, dung cho nhat ky bat buoc trong bai bao ("luu nhat ky moi quan sat bi
loai hoac thay doi").
"""
from __future__ import annotations

import pandas as pd


def remove_duplicates(df: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, dict]:
    n_before = len(df)
    out = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    return out, {"n_before": n_before, "n_after": len(out), "n_removed": n_before - len(out)}


def validate_ohlc(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> tuple[pd.DataFrame, dict]:
    """Kiem tra Low <= min(Open,Close) <= max(Open,Close) <= High. Cac dong
    vi pham duoc danh dau (`ohlc_valid` = False) chu KHONG tu dong xoa, vi
    ngoai le thi truong that khong nen bi xoa chi vi do lon cao (Muc 5.3)."""
    lo_ok = df[low_col] <= df[[open_col, close_col]].min(axis=1)
    hi_ok = df[[open_col, close_col]].max(axis=1) <= df[high_col]
    valid = lo_ok & hi_ok
    out = df.copy()
    out["ohlc_valid"] = valid
    n_invalid = int((~valid).sum())
    return out, {"n_rows": len(df), "n_invalid": n_invalid}


def flag_zero_price_or_volume(
    df: pd.DataFrame, price_col: str = "close", volume_col: str = "volume"
) -> pd.DataFrame:
    out = df.copy()
    out["zero_price"] = out[price_col] <= 0
    out["zero_volume"] = out[volume_col] <= 0
    return out


def build_time_grid(
    df: pd.DataFrame, date_col: str = "date", freq: str = "B"
) -> tuple[pd.DataFrame, dict]:
    """Tao luoi thoi gian chuan (mac dinh ngay lam viec) va GHI NHAN khoang
    trong bang mot cot `is_missing`, khong tu y noi suy (Muc 5.3)."""
    full_index = pd.date_range(df[date_col].min(), df[date_col].max(), freq=freq)
    grid = pd.DataFrame({date_col: full_index})
    merged = grid.merge(df, on=date_col, how="left", indicator=True)
    merged["is_missing"] = merged["_merge"] == "left_only"
    merged = merged.drop(columns="_merge")
    return merged, {"n_grid": len(grid), "n_missing": int(merged["is_missing"].sum())}
