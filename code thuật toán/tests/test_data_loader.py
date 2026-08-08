import numpy as np
import pandas as pd

from msgarchx.data.loader import (
    build_daily_bars,
    compute_daily_log_returns,
    compute_intraday_log_returns,
    data_quality_report,
)


def _make_synthetic_intraday(n_days=3, bars_per_day=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    price = 1000.0
    for d in range(n_days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(bars_per_day):
            price *= float(np.exp(rng.normal(0, 0.001)))
            rows.append(
                {
                    "date": date,
                    "datetime": date + pd.Timedelta(minutes=b),
                    "SYMBOL": "VN30F1M",
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": 10 + b,
                }
            )
    return pd.DataFrame(rows)


def test_data_quality_report_clean_synthetic_data():
    df = _make_synthetic_intraday()
    report = data_quality_report(df)
    assert report["n_duplicate_datetime"] == 0
    assert report["n_null"] == 0
    assert report["n_nonpositive_price"] == 0
    assert report["n_ohlc_violation"] == 0
    assert report["n_trading_days"] == 3


def test_build_daily_bars_and_log_returns():
    df = _make_synthetic_intraday(n_days=4, bars_per_day=6)
    daily = build_daily_bars(df)
    assert len(daily) == 4
    assert set(["open", "high", "low", "close", "volume", "n_bars"]).issubset(daily.columns)
    assert (daily["n_bars"] == 6).all()

    daily_ret = compute_daily_log_returns(daily)
    assert daily_ret["log_return"].isna().sum() == 1  # ngay dau khong co loi suat
    assert np.isfinite(daily_ret["log_return"].dropna()).all()


def test_compute_intraday_log_returns_resets_at_day_boundary():
    df = _make_synthetic_intraday(n_days=2, bars_per_day=4)
    out = compute_intraday_log_returns(df)
    # nen dau tien moi ngay phai la NaN (khong noi loi suat qua ngay)
    first_bar_mask = out.groupby("date").cumcount() == 0
    assert out.loc[first_bar_mask, "intraday_log_return"].isna().all()
    assert out.loc[~first_bar_mask, "intraday_log_return"].notna().all()
