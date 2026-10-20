"""Conditional EGARCH-X fit on segments outside expiry event windows.

Exclude expiry dates from the likelihood. The following trading date is used
only to restart the EGARCH recursion, so its likelihood is excluded too. The
next date can then be evaluated using a corrected same-contract return and RV.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
from numba import njit
from scipy import optimize

from uoc_luong_egarch_x_phan_phoi_moi import (
    ged_eabs, ged_logpdf, skew_eabs, skew_logpdf,
)
from uoc_luong_mle_khoang_nghi import PARAM_NAMES, diverse_starting_points
from uoc_luong_mle_roll_f2m import OUT, prepare


@njit(cache=True)
def variance_path_segmented(r, X, reset, mu, omega, alpha, theta, beta,
                            gamma0, gamma1, eabs, h0):
    h = np.empty(len(r), dtype=np.float64)
    h[0] = h0
    for t in range(1, len(r)):
        if reset[t]:
            h[t] = h0
        else:
            zprev = (r[t - 1] - mu) / math.sqrt(h[t - 1])
            logh = (omega + alpha * (abs(zprev) - eabs) + theta * zprev
                    + beta * math.log(h[t - 1]) + gamma0 * X[t, 0] + gamma1 * X[t, 1])
            h[t] = math.exp(min(30.0, max(-30.0, logh)))
    return h


def parts(p, dist, r, X, reset, h0):
    eabs = skew_eabs(p[7], p[8]) if dist == "skewt" else ged_eabs(p[7])
    h = variance_path_segmented(r, X, reset, *p[:7], eabs, h0)
    z = (r - p[0]) / np.sqrt(h)
    logfz = skew_logpdf(z, p[7], p[8]) if dist == "skewt" else ged_logpdf(z, p[7])
    return h, logfz - .5 * np.log(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=30)
    args = ap.parse_args()
    if args.starts < 1:
        ap.error("--starts must be positive")
    data, audit, daily = prepare()
    source = json.loads((OUT / "egarch_x_roll_bridge_mle.json").read_text(encoding="utf-8"))
    r, X, rv, n, dates = (data[k] for k in ("r", "X", "rv", "n_train", "dates"))
    event = daily.set_index("date").loc[dates, ["is_expiry", "is_roll_next"]]
    expiry = event.is_expiry.to_numpy(bool)
    reset = event.is_roll_next.to_numpy(bool).copy()
    reset[0] = True
    # The expiry observation is excluded, and the next observation initializes
    # a fresh variance path. This leaves no state or lagged RV from expiry.
    included = ~(expiry | reset)
    included[0] = False
    # prepare() uses all training dates to standardize X. Recenter the already
    # standardized matrix on retained training dates only. Algebraically this
    # equals standardizing raw lagged X on the retained dates, while preserving
    # the frozen source's dates and data checks.
    clean_train = included[:n]
    x_center = X[:n][clean_train].mean(axis=0)
    x_scale = X[:n][clean_train].std(axis=0, ddof=1)
    if not np.isfinite(x_scale).all() or (x_scale <= 0).any():
        raise AssertionError("Invalid clean training X scale")
    X = (X - x_center) / x_scale
    h0 = float(np.var(r[:n][clean_train], ddof=1))
    n_effective = int(included[:n].sum())
    n_test = int(included[n:].sum())
    assert n_effective > 1000 and n_test > 300
    assert not np.any(expiry & reset)
    assert np.all(np.flatnonzero(expiry) + 1 == np.flatnonzero(reset)[1:])
    assert not np.any(expiry[:-1] & included[1:])
    results = {}
    for dist in ("student_t", "skewt", "ged"):
        row = source["models"][dist]
        base = row["params"]
        b = {k: base[k] for k in ("mu", "omega", "alpha", "theta", "beta")}
        original_gamma = np.array([base["gamma_volume"], base["gamma_rv"]])
        b["omega"] += float(original_gamma @ x_center)
        b["gamma"] = list(original_gamma * x_scale)
        b["nu"] = base.get("nu", base.get("eta", 7.0))
        bounds = [(-5, 5), (-10, 10), (-3, 3), (-3, 3), (1e-4, .9999),
                  (-3, 3), (-3, 3), (2.05, 150) if dist != "ged" else (.5, 10)]
        if dist == "skewt":
            bounds.append((-.95, .95))
        starts = diverse_starting_points(b, dist, args.starts)
        if dist == "skewt":
            starts[0][8] = base["lambda_skew"]
        elif dist == "ged":
            starts[0][7] = base["ged_shape"]

        def objective(free):
            p = np.r_[free, 0.0] if dist == "student_t" else free
            core = "skewt" if dist == "student_t" else dist
            try:
                h, logf = parts(p, core, r[:n], X[:n], reset[:n], h0)
                if not np.isfinite(h).all() or not np.isfinite(logf[included[:n]]).all():
                    return 1e20
                return float(-logf[included[:n]].sum())
            except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
                return 1e20

        runs = []
        best = None
        for i, start in enumerate(starts):
            fit = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": 1500, "ftol": 1e-11,
                                             "gtol": 1e-5, "maxls": 40})
            runs.append({"start": i, "loglik": float(-fit.fun), "success": bool(fit.success),
                         "iterations": int(fit.nit), "fitted_params": list(map(float, fit.x))})
            print(f"{dist} {i+1}/{args.starts}: LL={-fit.fun:.6f}, success={fit.success}", flush=True)
            if np.isfinite(fit.fun) and (best is None or fit.fun < best.fun):
                best = fit
        assert best is not None and best.fun < 1e19
        p = np.r_[best.x, 0.0] if dist == "student_t" else best.x
        core = "skewt" if dist == "student_t" else dist
        h, logf = parts(p, core, r, X, reset, h0)
        mask = included.copy()
        mask[:n] = False
        oos = {"n_test": n_test,
               "qlike": float(np.mean(np.log(h[mask]) + rv[mask]/h[mask])),
               "nls": float(-np.mean(logf[mask])),
               "mae": float(np.mean(np.abs(rv[mask] - h[mask]))),
               "rmse": float(np.sqrt(np.mean((rv[mask] - h[mask])**2)))}
        k = len(best.x)
        names = PARAM_NAMES + (["nu"] if dist == "student_t" else
                               ["eta", "lambda_skew"] if dist == "skewt" else ["ged_shape"])
        near = [run for run in runs if -best.fun - run["loglik"] < 1e-4]
        results[dist] = {"params": dict(zip(names, map(float, best.x))),
                         "loglik": float(-best.fun), "n_params": k,
                         "aic": float(2*k + 2*best.fun),
                         "bic": float(k*math.log(n_effective) + 2*best.fun),
                         "oos": oos,
                         "optimization": {"n_starts": args.starts,
                                          "n_near_best": len(near), "runs": runs}}
        print(f"{dist}: AIC={results[dist]['aic']:.3f} BIC={results[dist]['bic']:.3f} "
              f"QLIKE={oos['qlike']:.6f} NLS={oos['nls']:.6f}", flush=True)
    output = {"method": "EGARCH-X segmented conditional MLE outside expiry windows",
              "audit": {**audit, "n_train_likelihood": n_effective,
                        "n_test_scored": n_test,
                        "n_expiry_train": int(expiry[:n].sum()),
                        "n_expiry_test": int(expiry[n:].sum()),
                        "n_restart_train": int(reset[:n].sum()),
                        "n_restart_test": int(reset[n:].sum()),
                        "h0_clean_train_variance": h0,
                        "x_clean_train_center_in_old_std_units": x_center.tolist(),
                        "x_clean_train_scale_in_old_std_units": x_scale.tolist()},
              "notes": ["Expiry day excluded from likelihood.",
                        "First trading date after expiry initializes a fresh variance path and is not scored.",
                        "Initial variance and X standardization use retained training observations only.",
                        "Subsequent dates use the roll-bridged return and realized variance.",
                        "This segmented conditional likelihood does not model volatility on excluded expiry dates.",
                        "AIC/BIC are comparable among these three segmented fits, not directly to the full-sample fits."],
              "models": results}
    path = OUT / "egarch_x_expiry_window_sensitivity.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
