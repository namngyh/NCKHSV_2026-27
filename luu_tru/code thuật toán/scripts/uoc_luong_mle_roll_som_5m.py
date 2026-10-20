"""Compare expiry-day early roll with post-expiry roll on matched 5-minute data.

At the close of the session before expiry, a hypothetical strategy switches
from F1M to F2M. It measures the expiry-day return from F2M's own preceding
close and uses F2M 5-minute bars for that whole day. On the next day the F1M
generic series represents the same contract. The comparator retains F1M on
expiry day and bridges to F2M's expiry-day close the following morning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garch_vn30.du_lieu.chuan_bi import build_exogenous_matrix, standardize_with_train_stats
from garch_vn30.du_lieu.doc_du_lieu import load_intraday_csv
from uoc_luong_egarch_x_phan_phoi_moi import model_parts
from uoc_luong_mle_khoang_nghi import fit_multistart, serialize_model, verify_student_t
from uoc_luong_mle_roll_f2m import CACHE as DAILY_PRICE_CACHE, expiry_boundaries


OUT = ROOT / "results" / "phan_phoi_moi" / "dao_han_f2m_5m"
SOURCE = OUT / "source_vn30f2m_5m.csv"
F1_SOURCE = ROOT / "ohlc_export.csv"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bars(import_source: bool, external: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    if import_source:
        if external is None:
            raise ValueError("Pass --source with the F2M 5-minute CSV when using --import-source")
        if not external.exists():
            raise FileNotFoundError(external)
        shutil.copyfile(external, SOURCE)
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}; use --import-source")
    f2 = pd.read_csv(SOURCE, parse_dates=["datetime_vn"])
    needed = {"datetime_vn", "open", "high", "low", "close", "volume"}
    if not needed.issubset(f2.columns):
        raise AssertionError("F2M 5-minute CSV lacks OHLCV or datetime_vn")
    f2 = f2.rename(columns={"datetime_vn": "datetime"})
    f2["date"] = f2.datetime.dt.normalize()
    f2 = f2.sort_values("datetime")
    if f2.datetime.duplicated().any() or (f2[["open", "high", "low", "close"]] <= 0).any().any():
        raise AssertionError("Invalid F2M timestamps or prices")
    f1 = load_intraday_csv(str(F1_SOURCE))
    start = max(f1.date.min(), f2.date.min())
    end = min(f1.date.max(), f2.date.max())
    f1 = f1.loc[f1.date.between(start, end)].copy()
    f2 = f2.loc[f2.date.between(start, end)].copy()
    if not np.array_equal(f1.date.drop_duplicates().to_numpy(dtype="datetime64[ns]"),
                          f2.date.drop_duplicates().to_numpy(dtype="datetime64[ns]")):
        raise AssertionError("F1M and F2M trading dates do not align")
    # The 08:59 F1M record has no 5-minute F2M peer; both series start at 09:00.
    f1 = f1.loc[f1.datetime.dt.time >= pd.Timestamp("09:00").time()].copy()
    f1["datetime"] = f1.datetime.dt.floor("5min")
    f1 = (f1.groupby(["date", "datetime"], as_index=False, sort=True)
          .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
               close=("close", "last"), volume=("volume", "sum")))
    f1["contract"] = "F1M"
    f2 = f2[["date", "datetime", "open", "high", "low", "close", "volume"]].copy()
    f2["contract"] = "F2M"
    f1_dates = f1.groupby("date").close.last()
    f2_dates = f2.groupby("date").close.last()
    if not f1_dates.index.equals(f2_dates.index):
        raise AssertionError("Daily dates changed when resampling F1M")
    price = pd.read_csv(DAILY_PRICE_CACHE, parse_dates=["date"]).set_index("date")
    check = price.loc[f1_dates.index]
    if not np.allclose(f1_dates, check.vn30f1m_close, atol=1e-9, rtol=0):
        raise AssertionError("Resampled F1M closes differ from VNDIRECT daily F1M")
    if not np.allclose(f2_dates, check.vn30f2m_close, atol=1e-9, rtol=0):
        raise AssertionError("F2M 5-minute closes differ from VNDIRECT daily F2M")
    dates = pd.Series(f1_dates.index)
    expiry_pairs = expiry_boundaries(dates)
    expiry = {dates.iloc[i] for i, _ in expiry_pairs}
    preexpiry = {dates.iloc[i - 1] for i, _ in expiry_pairs}
    f2_daily = f2.groupby("date").agg(bars=("close", "size"),
                                       first=("datetime", "first"),
                                       last=("datetime", "last"),
                                       volume=("volume", "sum"))
    selected = f2_daily.loc[sorted(expiry | preexpiry)]
    if (selected.bars.min() < 49 or
            not selected["first"].dt.time.eq(pd.Timestamp("09:00").time()).all() or
            not selected["last"].dt.time.eq(pd.Timestamp("14:45").time()).all()):
        raise AssertionError("F2M bar coverage is incomplete around an early roll")
    f1_volume = f1.groupby("date").volume.sum()
    audit = {"first_date": str(start.date()), "last_date": str(end.date()),
             "n_daily_dates": len(dates), "n_expiry": len(expiry),
             "n_f1_5m_bars": len(f1), "n_f2_5m_bars": len(f2),
             "n_f2_daily_closes_matched": len(f2_dates),
             "min_f2_bars_preexpiry": int(f2_daily.loc[sorted(preexpiry)].bars.min()),
             "min_f2_bars_expiry": int(f2_daily.loc[sorted(expiry)].bars.min()),
             "median_f2_to_f1_volume_preexpiry": float(
                 (f2_daily.loc[sorted(preexpiry)].volume /
                  f1_volume.loc[sorted(preexpiry)]).median()),
             "f2_source_sha256": hash_file(SOURCE),
             "f1_source_sha256": hash_file(F1_SOURCE)}
    return f1, f2, {"dates": dates, "expiry_pairs": expiry_pairs,
                    "expiry_dates": expiry, "audit": audit}


def daily_from_bars(bars: pd.DataFrame, bridge_dates: set[pd.Timestamp],
                    prior_f2_close: pd.Series) -> pd.DataFrame:
    bars = bars.sort_values("datetime").reset_index(drop=True)
    gaps = bars.groupby("date").datetime.diff().dt.total_seconds().div(60)
    lunch = gaps > 30
    if not lunch.groupby(bars.date).sum().eq(1).all():
        raise AssertionError("Each selected trading day must have one lunch gap")
    bars["session"] = lunch.groupby(bars.date).cumsum().astype(np.int8)
    group = bars.groupby(["date", "session"], sort=True)
    first_bar = group.cumcount().eq(0)
    prior = group.close.shift(1).where(~first_bar, bars.open)
    squared = (100 * np.log(bars.close / prior)) ** 2
    rv_s = squared.groupby([bars.date, bars.session]).sum().unstack()
    first_open = group.open.first().unstack()
    last_close = group.close.last().unstack()
    daily = (bars.groupby("date", sort=True)
             .agg(open=("open", "first"), close=("close", "last"),
                  volume=("volume", "sum"), n_bars=("close", "size"),
                  contract=("contract", "first")))
    close_prev = daily.close.shift(1).copy()
    is_bridge = daily.index.isin(bridge_dates)
    close_prev.loc[is_bridge] = prior_f2_close.shift(1).loc[is_bridge]
    if close_prev.iloc[1:].isna().any():
        raise AssertionError("Missing F2M close at roll boundary")
    daily["is_bridge"] = is_bridge
    daily["prior_close_used"] = close_prev
    daily["log_return"] = 100 * np.log(daily.close / close_prev)
    daily["overnight_return"] = 100 * np.log(first_open[0] / close_prev)
    daily["lunch_return"] = 100 * np.log(first_open[1] / last_close[0])
    daily["rv_morning"] = rv_s[0]
    daily["rv_afternoon"] = rv_s[1]
    daily["RV"] = (rv_s[0] + rv_s[1] + daily.lunch_return.pow(2)
                   + daily.overnight_return.pow(2).fillna(0.0))
    morning = 100 * np.log(last_close[0] / first_open[0])
    afternoon = 100 * np.log(last_close[1] / first_open[1])
    error = (daily.log_return - (daily.overnight_return + morning
                                 + daily.lunch_return + afternoon)).abs().iloc[1:]
    if error.max() > 1e-10 or (daily.RV <= 0).any():
        raise AssertionError("Return decomposition or RV invalid")
    return daily.reset_index()


def prepare_variant(daily: pd.DataFrame) -> dict:
    mask_train = pd.Series(False, index=daily.index)
    mask_train.iloc[:int(.8 * len(daily))] = True
    raw_x = build_exogenous_matrix(daily)
    std_x, means, stds = standardize_with_train_stats(raw_x, mask_train)
    valid = daily.log_return.notna() & daily.RV.notna() & std_x.iloc[:, 1:].notna().all(axis=1)
    out = {"r": daily.loc[valid, "log_return"].to_numpy(np.float64),
           "rv": daily.loc[valid, "RV"].to_numpy(np.float64),
           "X": std_x.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy(np.float64),
           "n_train": int(mask_train.loc[valid].sum()),
           "dates": daily.loc[valid, "date"].reset_index(drop=True),
           "x_mean": means.to_dict(), "x_std": stds.to_dict()}
    if not all(np.isfinite(out[k]).all() for k in ("r", "rv", "X")):
        raise AssertionError("Non-finite model data")
    return out


def paired_scores(p: np.ndarray, dist: str, data: dict, mask: np.ndarray) -> dict:
    n = data["n_train"]
    h0 = float(np.var(data["r"][:n], ddof=1))
    h, logf = model_parts(p, "skewt" if dist == "student_t" else dist,
                          data["r"], data["X"], h0)
    return {"n_test": int(mask.sum()),
            "qlike": float(np.mean(np.log(h[mask]) + data["rv"][mask] / h[mask])),
            "nls": float(-np.mean(logf[mask])),
            "mae": float(np.mean(np.abs(data["rv"][mask] - h[mask]))),
            "rmse": float(np.sqrt(np.mean((data["rv"][mask] - h[mask]) ** 2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-source", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--starts", type=int, default=30)
    args = parser.parse_args()
    if args.starts < 1:
        parser.error("--starts must be positive")
    f1, f2, info = load_bars(args.import_source, args.source)
    dates = info["dates"]
    expiry = info["expiry_dates"]
    f2_close = f2.groupby("date").close.last()
    late_bridge_dates = {dates.iloc[j] for _, j in info["expiry_pairs"]}
    early_bars = pd.concat([f1.loc[~f1.date.isin(expiry)], f2.loc[f2.date.isin(expiry)]],
                           ignore_index=True)
    early_daily = daily_from_bars(early_bars, expiry, f2_close)
    late_daily = daily_from_bars(f1, late_bridge_dates, f2_close)
    early_daily["is_expiry"] = early_daily.date.isin(expiry)
    late_daily["is_expiry"] = late_daily.date.isin(expiry)
    early, late = prepare_variant(early_daily), prepare_variant(late_daily)
    if (not early["dates"].equals(late["dates"]) or
            early["n_train"] != late["n_train"]):
        raise AssertionError("Variant observation dates or splits differ")
    common = ~early["dates"].isin(expiry).to_numpy()
    if (not np.allclose(early["r"][common], late["r"][common], atol=1e-10, rtol=0)
            or not np.allclose(early["rv"][common], late["rv"][common], atol=1e-10, rtol=0)):
        raise AssertionError("Non-expiry return/RV targets differ between roll methods")
    common[:early["n_train"]] = False
    print(f"Matched {len(dates)} dates, {len(expiry)} expiries; "
          f"train={early['n_train']}, test={len(early['r'])-early['n_train']}, "
          f"paired non-expiry test={common.sum()}", flush=True)
    base_json = json.loads((ROOT / "results" / "phan_phoi_moi" / "dao_han_f2m"
                            / "egarch_x_roll_bridge_mle.json").read_text(encoding="utf-8"))
    base_row = base_json["models"]["student_t"]["params"]
    base = {k: base_row[k] for k in ("mu", "omega", "alpha", "theta", "beta", "nu")}
    base["gamma"] = [base_row["gamma_volume"], base_row["gamma_rv"]]
    results = {}
    for name, data in (("late", late), ("early", early)):
        n = data["n_train"]
        tp, td = fit_multistart(data["r"][:n], data["X"][:n], base, "student_t", args.starts)
        refreshed = dict(zip(("mu", "omega", "alpha", "theta", "beta"), tp[:5]))
        refreshed["gamma"] = list(tp[5:7])
        refreshed["nu"] = float(tp[7])
        models = {"student_t": serialize_model(tp, "student_t", td, data)}
        verify_student_t(tp, data, models["student_t"])
        models["student_t"]["oos_nonexpiry_paired"] = paired_scores(tp, "student_t", data, common)
        for dist in ("skewt", "ged"):
            p, diag = fit_multistart(data["r"][:n], data["X"][:n], refreshed, dist, args.starts)
            models[dist] = serialize_model(p, dist, diag, data)
            models[dist]["oos_nonexpiry_paired"] = paired_scores(p, dist, data, common)
        results[name] = {"x_train_mean": data["x_mean"], "x_train_std": data["x_std"],
                         "models": models}
        print(f"{name} finished", flush=True)
    audit = {**info["audit"], "n_train": early["n_train"],
             "n_test": len(early["r"]) - early["n_train"],
             "n_test_nonexpiry_paired": int(common.sum()),
             "first_test_date": str(early["dates"].iloc[early["n_train"]].date()),
             "early_roll_expiry_dates": [str(d.date()) for d in sorted(expiry)]}
    output = {"method": "5-minute matched-period EGARCH-X MLE; early versus late roll",
              "audit": audit,
              "notes": ["F1M 1-minute bars aggregated to 5-minute; 08:59 record excluded to match F2M's 09:00 start.",
                        "Early roll selects F2M for the entire expiry day and uses prior-day F2M close for that day's return and overnight RV.",
                        "Late roll selects F1M on expiry day and uses expiry-day F2M close for the following day's bridge.",
                        "AIC/BIC across early and late variants are not directly comparable because expiry-day target returns differ.",
                        "Paired non-expiry OOS metrics use identical return and RV targets and dates.",
                        "Trading costs and spread for contract switch are not modeled."],
              "variants": results}
    (OUT / "early_daily_data.csv").write_text(
        early_daily.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="\n")
    (OUT / "late_daily_data.csv").write_text(
        late_daily.to_csv(index=False, lineterminator="\n"), encoding="utf-8", newline="\n")
    path = OUT / "egarch_x_5m_early_vs_late_mle.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    for variant, block in results.items():
        for dist, row in block["models"].items():
            paired = row["oos_nonexpiry_paired"]
            print(f"{variant:5s} {dist:9s} AIC={row['aic']:.3f} BIC={row['bic']:.3f} "
                  f"paired QLIKE={paired['qlike']:.6f} NLS={paired['nls']:.6f} "
                  f"near_best={row['optimization']['n_near_best']}/{args.starts}")


if __name__ == "__main__":
    main()
