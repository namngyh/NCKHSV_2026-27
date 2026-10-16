"""Correct front-month expiry bridges with next-month EOD closes and refit MLE.

This is a *bridge correction*: it removes the price-level jump between contracts
on the first trading day after expiry. It deliberately retains actual trading
returns on the expiry day, including unusual auction prices. Replacing those
requires contract-specific minute data and an earlier roll rule.

Run from ``code thuật toán``::

    python scripts/uoc_luong_mle_roll_f2m.py --fetch --starts 30
    python scripts/uoc_luong_mle_roll_f2m.py --starts 30  # use frozen cache
"""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garch_vn30.du_lieu.chuan_bi import build_exogenous_matrix, standardize_with_train_stats
from uoc_luong_mle_khoang_nghi import (
    PARAM_NAMES, construct_daily, fit_multistart, serialize_model, verify_student_t,
)


ENDPOINT = "https://dchart-api.vndirect.com.vn/dchart/history"
FROZEN = ROOT / "results" / "00_data.pt"
BASE = ROOT / "results" / "phan_phoi_moi" / "khoang_nghi" / "egarch_x_full_day_mle.json"
OUT = ROOT / "results" / "phan_phoi_moi" / "dao_han_f2m"
CACHE = OUT / "vndirect_f1m_f2m_daily.csv"


def fetch_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "resolution": "D", "symbol": symbol,
        "from": int(start.tz_localize("UTC").timestamp()),
        "to": int(end.tz_localize("UTC").timestamp()),
    }
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("s") != "ok" or not payload.get("t"):
        raise RuntimeError(f"No daily {symbol} history: {payload.get('s')}")
    dates = pd.to_datetime(payload["t"], unit="s", utc=True).tz_convert(
        "Asia/Ho_Chi_Minh").tz_localize(None).normalize()
    frame = pd.DataFrame({"date": dates, f"{symbol.lower()}_close": payload["c"]})
    if frame.date.duplicated().any() or (frame.iloc[:, 1] <= 0).any():
        raise AssertionError(f"Invalid {symbol} daily history")
    return frame


def load_or_fetch_cache(first: pd.Timestamp, last: pd.Timestamp, fetch: bool) -> pd.DataFrame:
    if fetch:
        OUT.mkdir(parents=True, exist_ok=True)
        start = first - pd.Timedelta(days=7)
        end = last + pd.Timedelta(days=3)
        f1 = fetch_history("VN30F1M", start, end)
        f2 = fetch_history("VN30F2M", start, end)
        frame = f1.merge(f2, on="date", how="outer").sort_values("date")
        frame.to_csv(CACHE, index=False, float_format="%.10g")
        meta = {"source": ENDPOINT, "resolution": "D", "symbols": ["VN30F1M", "VN30F2M"],
                "from": str(start.date()), "to": str(end.date()),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(frame)}
        (OUT / "source_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing {CACHE}; run with --fetch")
    return pd.read_csv(CACHE, parse_dates=["date"])


def third_thursday(year: int, month: int) -> pd.Timestamp:
    thursdays = [week[calendar.THURSDAY] for week in calendar.monthcalendar(year, month)
                 if week[calendar.THURSDAY] > 0]
    return pd.Timestamp(year, month, thursdays[2])


def expiry_boundaries(dates: pd.Series) -> list[tuple[int, int]]:
    """HNX third Thursday; use preceding observed session if it is a holiday."""
    bounds = []
    for year in range(dates.iloc[0].year, dates.iloc[-1].year + 1):
        for month in range(1, 13):
            target = third_thursday(year, month)
            if target < dates.iloc[0] or target >= dates.iloc[-1]:
                continue
            eligible = np.flatnonzero((dates <= target) & (dates > target - pd.Timedelta(days=8))
                                  & (dates.dt.year == year) & (dates.dt.month == month))
            if not len(eligible):
                raise AssertionError(f"No observed expiry session near {target.date()}")
            i = int(eligible[-1])
            if i + 1 >= len(dates):
                continue
            bounds.append((i, i + 1))
    return bounds


def prepare(fetch: bool = False) -> tuple[dict, dict, pd.DataFrame]:
    daily = construct_daily()
    prices = load_or_fetch_cache(daily.date.iloc[0], daily.date.iloc[-1], fetch)
    daily = daily.merge(prices, on="date", how="left", validate="one_to_one")
    f1_present = daily.vn30f1m_close.notna()
    if f1_present.sum() < len(daily) - 2:
        raise AssertionError("Too many missing VN30F1M source closes")
    if not np.allclose(daily.loc[f1_present, "close"], daily.loc[f1_present, "vn30f1m_close"],
                       atol=1e-8, rtol=0):
        raise AssertionError("External F1M daily closes do not match the minute CSV")

    boundaries = expiry_boundaries(daily.date)
    is_expiry = np.zeros(len(daily), dtype=bool)
    is_roll_next = np.zeros(len(daily), dtype=bool)
    prior_close = daily.close.shift(1).to_numpy(copy=True)
    for i, j in boundaries:
        if j != i + 1 or not np.isfinite(daily.vn30f2m_close.iloc[i]):
            raise AssertionError(f"Missing F2M close for expiry {daily.date.iloc[i]}")
        is_expiry[i] = True
        is_roll_next[j] = True
        prior_close[j] = daily.vn30f2m_close.iloc[i]
    daily["is_expiry"] = is_expiry
    daily["is_roll_next"] = is_roll_next
    daily["close_for_previous_day"] = prior_close
    daily["raw_log_return"] = daily.log_return
    daily["raw_overnight_return"] = daily.overnight_return
    daily["raw_RV"] = daily.RV
    daily["log_return"] = 100.0 * np.log(daily.close / prior_close)
    daily["overnight_return"] = 100.0 * np.log(daily.open / prior_close)
    daily["RV"] = daily.rv_morning + daily.rv_afternoon + daily.lunch_return.pow(2) + \
                  daily.overnight_return.pow(2).fillna(0.0)

    frozen = torch.load(FROZEN, map_location="cpu", weights_only=False)
    X_raw = build_exogenous_matrix(daily)
    train_mask = pd.Series(False, index=daily.index)
    train_mask.iloc[:int(len(daily) * .8)] = True
    X_std, _, _ = standardize_with_train_stats(X_raw, train_mask)
    valid = daily.log_return.notna() & daily.RV.notna() & X_std.iloc[:, 1:].notna().all(axis=1)
    r = daily.loc[valid, "log_return"].to_numpy(np.float64)
    rv = daily.loc[valid, "RV"].to_numpy(np.float64)
    X = X_std.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy(np.float64)
    dates = daily.loc[valid, "date"].reset_index(drop=True)
    n_train = int(train_mask.loc[valid].sum())
    if (len(r) != len(frozen["r_all"]) or n_train != frozen["n_train"] or
            not np.array_equal(dates.to_numpy(dtype="datetime64[ns]"),
                               frozen["dates_all"].to_numpy(dtype="datetime64[ns]"))):
        raise AssertionError("Sample dates/split changed")
    if not np.isfinite(r).all() or not np.isfinite(rv).all() or not np.isfinite(X).all():
        raise AssertionError("Non-finite corrected data")
    changed = daily.loc[is_roll_next]
    audit = {
        "n_roll_boundaries": len(boundaries),
        "n_source_f1m_close_matches": int(f1_present.sum()),
        "n_source_f1m_close_missing": int((~f1_present).sum()),
        "n_train": n_train, "n_test": len(r) - n_train,
        "median_abs_overnight_raw_roll_pct": float(changed.raw_overnight_return.abs().median()),
        "median_abs_overnight_corrected_roll_pct": float(changed.overnight_return.abs().median()),
        "max_abs_overnight_raw_roll_pct": float(changed.raw_overnight_return.abs().max()),
        "max_abs_overnight_corrected_roll_pct": float(changed.overnight_return.abs().max()),
        "sum_rv_raw": float(daily.raw_RV.iloc[1:].sum()),
        "sum_rv_corrected": float(daily.RV.iloc[1:].sum()),
        "expiry_price_retained": True,
        "correction": "At the first session after expiry, replace previous F1M close with prior-day F2M close",
    }
    return {"r": r, "rv": rv, "X": X, "n_train": n_train, "dates": dates}, audit, daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="refresh and freeze public VNDIRECT daily prices")
    parser.add_argument("--starts", type=int, default=30)
    args = parser.parse_args()
    if args.starts < 1:
        parser.error("--starts must be positive")
    data, audit, daily = prepare(fetch=args.fetch)
    print(f"Checked {audit['n_roll_boundaries']} roll bridges; F1M source closes match on "
          f"{audit['n_source_f1m_close_matches']} dates.", flush=True)
    prior = json.loads(BASE.read_text(encoding="utf-8"))["models"]["student_t"]["params"]
    start = {k: prior[k] for k in ("mu", "omega", "alpha", "theta", "beta", "nu")}
    start["gamma"] = [prior["gamma_volume"], prior["gamma_rv"]]
    n = data["n_train"]
    t_p, t_diag = fit_multistart(data["r"][:n], data["X"][:n], start, "student_t", args.starts)
    refreshed = dict(zip(["mu", "omega", "alpha", "theta", "beta"], t_p[:5]))
    refreshed["gamma"] = list(t_p[5:7])
    refreshed["nu"] = float(t_p[7])
    results = {"student_t": serialize_model(t_p, "student_t", t_diag, data)}
    verify_student_t(t_p, data, results["student_t"])
    for dist in ("skewt", "ged"):
        p, diag = fit_multistart(data["r"][:n], data["X"][:n], refreshed, dist, args.starts)
        results[dist] = serialize_model(p, dist, diag, data)
    output = {"method": "EGARCH-X MLE; corrected overnight roll bridge using next-month close",
              "audit": audit, "models": results,
              "notes": ["Expiry-day traded close remains in daily return and intraday RV.",
                        "F1M minute CSV closes match VNDIRECT daily F1M closes except one missing API EOD date; F2M closes come from VNDIRECT.",
                        "Do not directly compare AIC/NLS with models fitted to uncorrected returns."]}
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["date", "close", "vn30f1m_close", "vn30f2m_close", "is_expiry",
               "is_roll_next", "raw_log_return", "log_return", "raw_overnight_return",
               "overnight_return", "raw_RV", "RV"]
    daily[columns].to_csv(OUT / "corrected_daily_data.csv", index=False, float_format="%.12g")
    path = OUT / "egarch_x_roll_bridge_mle.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    for name, row in results.items():
        print(f"{name:10s} AIC={row['aic']:.3f} BIC={row['bic']:.3f} "
              f"QLIKE={row['oos']['qlike']:.6f} NLS={row['oos']['nls']:.6f} "
              f"near_best={row['optimization']['n_near_best']}/{args.starts}")


if __name__ == "__main__":
    main()
