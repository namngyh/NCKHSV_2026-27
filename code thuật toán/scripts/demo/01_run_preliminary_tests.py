"""Demo: nap du lieu VN30F1M that (ohlc_export.csv), gop ngay, tinh loi suat
va RV_t, roi chay toan bo kiem dinh thong ke ban dau (Muc 5).

Chay: python scripts/demo/01_run_preliminary_tests.py [duong_dan_csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from msgarchx.data.loader import load_vn30f1m_daily
from msgarchx.stats_tests.preliminary_tests import run_all_preliminary_tests


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2] / "ohlc_export.csv")
    print(f"Nap du lieu tu: {csv_path}")

    daily, report = load_vn30f1m_daily(csv_path)
    print("\n== Bao cao chat luong du lieu (Muc 5.3) ==")
    for k, v in report.items():
        print(f"  {k}: {v}")

    r = daily["log_return"].dropna().to_numpy()
    print(f"\nSo quan sat loi suat ngay (sau khi loai NaN dau chuoi): {len(r)}")
    print(f"Khoang thoi gian: {daily['date'].min().date()} -> {daily['date'].max().date()}")
    print(f"Trung binh r_t: {r.mean():.4f}  |  Do lech chuan r_t: {r.std():.4f}")
    print(f"Do lech (skewness): {((r - r.mean())**3).mean() / r.std()**3:.4f}")
    print(f"Do nhon (kurtosis): {((r - r.mean())**4).mean() / r.std()**4:.4f}")

    print("\n== Kiem dinh thong ke ban dau (Muc 5) ==")
    results = run_all_preliminary_tests(r)
    header = f"{'Kiem dinh':<18}{'Thong ke':>14}{'p-value':>12}   Quyet dinh"
    print(header)
    print("-" * len(header))
    for key, res in results.items():
        name = res.get("test", key)
        print(f"{name:<18}{res['statistic']:>14.4f}{res['p_value']:>12.4f}   {res['decision']}")

    rv = daily["RV"].dropna().to_numpy()
    print(f"\nRV_t (tu du lieu phut): trung binh={rv.mean():.4f}, do lech chuan={rv.std():.4f}")
    print(f"So sanh voi dai dien r_t^2: trung binh(r_t^2)={np.mean(r**2):.4f}")


if __name__ == "__main__":
    main()
