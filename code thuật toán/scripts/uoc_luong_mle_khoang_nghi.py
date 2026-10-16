"""Refit daily EGARCH-X MLE after separating lunch and overnight returns.

Run from ``code thuật toán``::

    python scripts/uoc_luong_mle_khoang_nghi.py

The daily close-to-close return and the chronological train/test split are unchanged.
The realized-variance proxy now covers that same close-to-close interval. The
input has only the generic VN30F1M symbol, so contract rolls cannot be labeled.
Results are written separately from the original study.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import optimize
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garch_vn30.du_lieu.chuan_bi import build_exogenous_matrix, standardize_with_train_stats
from garch_vn30.du_lieu.doc_du_lieu import (
    build_daily_bars, compute_daily_log_returns, load_intraday_csv,
)
from garch_vn30.danh_gia.ngoai_mau import danh_gia_mot_diem
from garch_vn30.mo_hinh.egarch import EGARCHParams, egarch_loglik
from uoc_luong_egarch_x_phan_phoi_moi import (
    evaluate, ged_eabs, negloglik, skew_eabs,
)


CSV = ROOT / "ohlc_export.csv"
FROZEN = ROOT / "results" / "00_data.pt"
OLD = ROOT / "results" / "phan_phoi_moi" / "egarch_x_skewt_ged_mle.json"
OUT = ROOT / "results" / "phan_phoi_moi" / "khoang_nghi"
PARAM_NAMES = ["mu", "omega", "alpha", "theta", "beta", "gamma_volume", "gamma_rv"]


def construct_daily(gap_threshold_minutes: float = 30.0) -> pd.DataFrame:
    """Build session RV plus separate lunch and previous-close-to-open jumps.

Session RV includes first bar's open-to-close return, then adjacent bar closes.
The source has exactly one >30 minute within-day gap on every observed date.
"""
    bars = load_intraday_csv(str(CSV))
    if (bars[["open", "close"]] <= 0).any().any():
        raise ValueError("Cannot take log returns from nonpositive prices")
    elapsed = bars.groupby("date", sort=False)["datetime"].diff().dt.total_seconds() / 60
    is_lunch = elapsed > gap_threshold_minutes
    gaps_per_day = is_lunch.groupby(bars["date"]).sum()
    if not gaps_per_day.eq(1).all():
        raise ValueError(f"Expected one lunch gap per day: {gaps_per_day.value_counts().to_dict()}")

    bars["session"] = is_lunch.groupby(bars["date"]).cumsum().astype(np.int8)
    by_session = bars.groupby(["date", "session"], sort=True)
    prior_close = by_session["close"].shift(1)
    first_bar = by_session.cumcount().eq(0)
    prior_price = prior_close.where(~first_bar, bars["open"])
    squared_bar_return = (100.0 * np.log(bars["close"] / prior_price)) ** 2
    rv_session = squared_bar_return.groupby([bars["date"], bars["session"]]).sum().unstack()
    first_open = by_session["open"].first().unstack()
    last_close = by_session["close"].last().unstack()

    daily = compute_daily_log_returns(build_daily_bars(bars)).set_index("date")
    overnight = 100.0 * np.log(first_open[0] / daily["close"].shift(1))
    lunch = 100.0 * np.log(first_open[1] / last_close[0])
    daily["rv_morning"] = rv_session[0]
    daily["rv_afternoon"] = rv_session[1]
    daily["overnight_return"] = overnight
    daily["lunch_return"] = lunch
    # The first date has no preceding close in the file. Its return is excluded,
    # but its observed within-day RV must remain available as the next day's lag.
    daily["RV"] = rv_session[0] + rv_session[1] + overnight.pow(2).fillna(0.0) + lunch.pow(2)

    # Both return paths must represent the identical close-to-close interval.
    morning = 100.0 * np.log(last_close[0] / first_open[0])
    afternoon = 100.0 * np.log(last_close[1] / first_open[1])
    decomposition_error = (daily["log_return"] - (overnight + morning + lunch + afternoon)).abs()
    if decomposition_error.iloc[1:].max() > 1e-10:
        raise AssertionError("Return decomposition does not match close-to-close returns")
    daily.attrs["gap_minutes"] = {
        "min": float(elapsed[is_lunch].min()),
        "median": float(elapsed[is_lunch].median()),
        "max": float(elapsed[is_lunch].max()),
    }
    return daily.reset_index()


def prepare() -> tuple[dict, dict]:
    daily = construct_daily()
    saved = torch.load(FROZEN, map_location="cpu", weights_only=False)
    X_raw = build_exogenous_matrix(daily)
    train_mask = pd.Series(False, index=daily.index)
    train_mask.iloc[:int(len(daily) * .8)] = True
    X_std, means, stds = standardize_with_train_stats(X_raw, train_mask)
    valid = daily["log_return"].notna() & daily["RV"].notna() & X_std.iloc[:, 1:].notna().all(axis=1)
    dates = daily.loc[valid, "date"].reset_index(drop=True)
    r = daily.loc[valid, "log_return"].to_numpy(np.float64)
    rv = daily.loc[valid, "RV"].to_numpy(np.float64)
    X = X_std.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy(np.float64)
    n_train = int(train_mask.loc[valid].sum())
    if len(r) != len(saved["r_all"]) or n_train != saved["n_train"]:
        raise AssertionError("Sample size or split differs from the original study")
    if not np.array_equal(dates.to_numpy(dtype="datetime64[ns]"),
                          saved["dates_all"].to_numpy(dtype="datetime64[ns]")):
        raise AssertionError("Dates differ from the original study")
    if not np.allclose(r, saved["r_all"].numpy(), atol=1e-12, rtol=0):
        raise AssertionError("Daily returns differ from the original study")
    if not np.isfinite(rv).all() or not np.isfinite(X).all() or (rv <= 0).any():
        raise AssertionError("Invalid full-day RV or regressors")
    data = {"r": r, "rv": rv, "X": X, "n_train": n_train, "dates": dates}
    audit = {
        "n_days": len(daily), "n_observations": len(r), "n_train": n_train,
        "n_test": len(r) - n_train, "first_test_date": str(dates.iloc[n_train].date()),
        "lunch_gap_minutes": daily.attrs["gap_minutes"],
        "rv_mean_old": float(saved["rv_all"].numpy().mean()),
        "rv_mean_new": float(rv.mean()),
        "rv_median_old": float(np.median(saved["rv_all"].numpy())),
        "rv_median_new": float(np.median(rv)),
        "overnight_sq_share": float(np.nansum(daily["overnight_return"].to_numpy()[1:] ** 2) / rv.sum()),
        "lunch_sq_share": float(np.nansum(daily["lunch_return"].to_numpy()[1:] ** 2) / rv.sum()),
        "x_train_mean": dict(zip(["volume", "rv"], map(float, means.iloc[:2]))),
        "x_train_std": dict(zip(["volume", "rv"], map(float, stds.iloc[:2]))),
        "roll_status": "Unknown: source contains only the continuous VN30F1M symbol",
    }
    return data, audit


def diverse_starting_points(base: dict, dist: str, count: int) -> list[np.ndarray]:
    """Distinct deterministic starts, including tail/shape and broad EGARCH values."""
    common = np.array([base[k] for k in ("mu", "omega", "alpha", "theta", "beta")]
                      + list(base["gamma"]), dtype=float)
    if dist == "student_t":
        tails = [(nu,) for nu in (base["nu"], 2.5, 3.0, 6.0, 10.0, 20.0)]
    elif dist == "skewt":
        tails = [(base["nu"], lam) for lam in (0.0, -.2, .2, -.5, .5, -.8, .8)]
        tails += [(nu, lam) for nu, lam in ((2.5, -.4), (8.0, .4), (15.0, 0.0))]
    else:
        tails = [(shape,) for shape in (1.1, .65, .9, 1.5, 2.0, 3.0, 4.0)]
    points = [np.r_[common, *tail] for tail in tails[:count]]
    rng = np.random.default_rng(20261016 + {"student_t": 0, "skewt": 1, "ged": 2}[dist])
    while len(points) < count:
        p = common.copy()
        p += rng.normal(0, [.18, .35, .18, .22, .20, .25, .25])
        p[4] = np.clip(p[4], .05, .95)
        if dist == "student_t":
            extra = (rng.uniform(2.2, 25.0),)
        elif dist == "skewt":
            extra = (rng.uniform(2.2, 25.0), rng.uniform(-.85, .85))
        else:
            extra = (rng.uniform(.6, 5.0),)
        points.append(np.r_[p, *extra])
    return points


def fit_multistart(r: np.ndarray, X: np.ndarray, base: dict, dist: str,
                   starts: int) -> tuple[np.ndarray, dict]:
    """Fit Student-t (skew=0), skew-t, or GED from diverse free-parameter starts."""
    h0 = float(np.var(r, ddof=1))
    bounds = [(-5, 5), (-10, 10), (-3, 3), (-3, 3), (1e-4, .9999),
              (-3, 3), (-3, 3), (2.05, 150) if dist != "ged" else (.5, 10)]
    if dist == "skewt":
        bounds.append((-.95, .95))

    def objective(p: np.ndarray) -> float:
        return negloglik(np.r_[p, 0.0] if dist == "student_t" else p,
                         "skewt" if dist == "student_t" else dist, r, X, h0)

    runs = []
    best = None
    points = diverse_starting_points(base, dist, starts)
    if len({tuple(p) for p in points}) != starts:
        raise AssertionError("Repeated starting point")
    for i, p0 in enumerate(points):
        result = optimize.minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 1500, "ftol": 1e-11,
                                            "gtol": 1e-5, "maxls": 40})
        runs.append({"start": i, "success": bool(result.success),
                     "message": str(result.message), "loglik": float(-result.fun),
                     "iterations": int(result.nit),
                     "initial_params": list(map(float, p0)),
                     "fitted_params": list(map(float, result.x))})
        print(f"{dist} start {i+1}/{starts}: ll={-result.fun:.6f}, "
              f"success={result.success}, iters={result.nit}", flush=True)
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"No finite {dist} MLE")
    p_best = np.r_[best.x, 0.0] if dist == "student_t" else best.x
    near = [run for run in runs if -best.fun - run["loglik"] < 1e-4]
    fitted = np.asarray([run["fitted_params"] for run in near])
    return p_best, {
        "runs": runs, "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message), "nll": float(best.fun),
        "n_starts": starts, "near_best_tolerance_loglik": 1e-4,
        "n_near_best": len(near), "n_far_from_best": starts - len(near),
        "near_best_loglik_spread": float(-best.fun - min(run["loglik"] for run in near)),
        "near_best_max_param_range": float(np.max(np.ptp(fitted, axis=0))),
    }


def serialize_model(p: np.ndarray, dist: str, diag: dict, data: dict) -> dict:
    is_t = dist == "student_t"
    dist_core = "skewt" if is_t else dist
    k = 8 if is_t or dist == "ged" else 9
    names = PARAM_NAMES + (["nu"] if is_t else ["eta", "lambda_skew"] if dist == "skewt"
                           else ["ged_shape"])
    ll = -diag["nll"]
    n = data["n_train"]
    result = {
        "params": dict(zip(names, map(float, p[:k]))),
        "e_abs_z": float(skew_eabs(p[7], p[8]) if dist == "skewt"
                         else skew_eabs(p[7], 0.0) if is_t else ged_eabs(p[7])),
        "loglik": ll, "n_params": k, "aic": 2*k - 2*ll,
        "bic": k*math.log(n) - 2*ll,
        "oos": evaluate(p, dist_core, data["r"], data["X"], data["rv"], n),
        "optimization": diag,
    }
    return result


def verify_student_t(p: np.ndarray, data: dict, row: dict) -> None:
    """Cross-check the fast t-nested likelihood and OOS scores against repo code."""
    to_torch = lambda x: torch.tensor(x, dtype=torch.float64)
    params = EGARCHParams(
        mu=to_torch(p[0]), omega=to_torch(p[1]), alpha=to_torch(p[2]),
        theta=to_torch(p[3]), beta=to_torch(p[4]), gamma=to_torch(p[5:7]),
        nu=to_torch(p[7]),
    )
    n = data["n_train"]
    r, X, rv = map(to_torch, (data["r"], data["X"], data["rv"]))
    ll = float(egarch_loglik(params, r[:n], X[:n], h0=torch.var(r[:n])))
    if abs(ll - row["loglik"]) > 1e-7:
        raise AssertionError(f"Student-t likelihood mismatch: {ll} vs {row['loglik']}")
    check = danh_gia_mot_diem(params, "egarch", "student_t", r, X, rv, n)
    for name in ("qlike", "nls", "mae", "rmse"):
        if abs(check[name] - row["oos"][name]) > 1e-7:
            raise AssertionError(f"Student-t OOS {name} mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=30)
    args = parser.parse_args()
    if args.starts < 1:
        parser.error("--starts must be positive")
    data, audit = prepare()
    old = json.loads(OLD.read_text(encoding="utf-8"))
    base = old["baseline"]["params"]
    base_for_fit = {**base, "gamma": [base["gamma_volume"], base["gamma_rv"]]}
    old_p = np.array([base[k] for k in PARAM_NAMES + ["nu"]] + [0.0])
    old_ll = -negloglik(old_p, "skewt", data["r"][:data["n_train"]],
                        torch.load(FROZEN, map_location="cpu", weights_only=False)["X_all"][:data["n_train"]].numpy(),
                        float(np.var(data["r"][:data["n_train"]], ddof=1)))
    if abs(old_ll - old["baseline"]["loglik"]) > 1e-6:
        raise AssertionError("The original Student-t likelihood cannot be reproduced")
    print(f"Prepared {len(data['r'])} daily returns, {data['n_train']} train and "
          f"{len(data['r'])-data['n_train']} test. Original LL check: {old_ll:.6f}", flush=True)

    r_train = data["r"][:data["n_train"]]
    X_train = data["X"][:data["n_train"]]
    t_p, t_diag = fit_multistart(r_train, X_train, base_for_fit, "student_t", args.starts)
    refreshed = dict(zip(["mu", "omega", "alpha", "theta", "beta"], t_p[:5]))
    refreshed["gamma"] = list(t_p[5:7])
    refreshed["nu"] = float(t_p[7])
    results = {"student_t": serialize_model(t_p, "student_t", t_diag, data)}
    verify_student_t(t_p, data, results["student_t"])
    for name in ("skewt", "ged"):
        p, diag = fit_multistart(r_train, X_train, refreshed, name, args.starts)
        results[name] = serialize_model(p, name, diag, data)

    output = {
        "method": "Daily EGARCH-X MLE, session-separated RV with overnight and lunch squares",
        "audit": audit, "models": results,
        "old_results_reference": str(OLD.relative_to(ROOT)).replace("\\", "/"),
        "notes": [
            "Daily returns and train/test dates are unchanged; RV and lagged log RV are recomputed.",
            "Lunch split uses the single within-day gap greater than 30 minutes.",
            "Session RV includes each session's first bar open-to-close return.",
            "No contract-roll adjustment is possible from the generic symbol alone.",
            "AIC/BIC use n_train as in the original study; likelihood omits the first initialization observation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    daily_columns = ["date", "open", "close", "volume", "log_return",
                     "rv_morning", "rv_afternoon", "overnight_return", "lunch_return", "RV"]
    construct_daily()[daily_columns].to_csv(OUT / "daily_rv_components.csv", index=False,
                                            float_format="%.12g")
    path = OUT / "egarch_x_full_day_mle.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    for name, row in results.items():
        print(f"{name:10s} AIC={row['aic']:.3f} BIC={row['bic']:.3f} "
              f"QLIKE={row['oos']['qlike']:.6f} NLS={row['oos']['nls']:.6f}")


if __name__ == "__main__":
    main()
