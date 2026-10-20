"""MLE EGARCH-X with standardized Hansen skew-t or GED innovations.

Uses the frozen 00_data.pt split and the same EGARCH recursion as the paper.
Run from `code thuật toán`:
    python scripts/uoc_luong_egarch_x_phan_phoi_moi.py
    python scripts/uoc_luong_egarch_x_phan_phoi_moi.py --self-test

Results are written to results/phan_phoi_moi/; existing study results are untouched.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numba import njit
from scipy import integrate, optimize, special, stats
import torch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "00_data.pt"
BASE = ROOT / "results" / "pipeline" / "egarch_x_t_mle.pt"
OUT = ROOT / "results" / "phan_phoi_moi"


def skew_constants(eta: float, lam: float) -> tuple[float, float, float]:
    """Hansen (1994) constants; the resulting innovation has mean 0, variance 1."""
    log_c = (special.gammaln((eta + 1) / 2) - special.gammaln(eta / 2)
             - 0.5 * math.log(math.pi * (eta - 2)))
    c = math.exp(log_c)
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = math.sqrt(1 + 3 * lam * lam - a * a)
    return a, b, log_c


def skew_eabs(eta: float, lam: float) -> float:
    """Exact E|Z| from the two half-t pieces of Hansen's density."""
    a, b, log_c = skew_constants(eta, lam)
    q = eta - 2
    m1 = math.exp(log_c) * q / (eta - 1)  # integral_0^infinity u f_t*(u) du
    if lam >= 0:
        u = a / (1 + lam)
        tail0 = special.stdtr(eta, -u * math.sqrt(eta / q))
        tail1 = m1 * (1 + u * u / q) ** (-(eta - 1) / 2)
        positive = (1 + lam) / b * ((1 + lam) * tail1 - a * tail0)
    else:
        u = -a / (1 - lam)  # positive magnitude of lower-piece threshold
        middle0 = special.stdtr(eta, u * math.sqrt(eta / q)) - 0.5
        tail1 = m1 * (1 + u * u / q) ** (-(eta - 1) / 2)
        lower = (1 - lam) / b * (-(1 - lam) * (m1 - tail1) - a * middle0)
        upper = (1 + lam) / b * ((1 + lam) * m1 - 0.5 * a)
        positive = lower + upper
    return 2 * positive  # E[Z]=0, hence E|Z|=2 E[Z_+]


def skew_logpdf(z: np.ndarray, eta: float, lam: float) -> np.ndarray:
    a, b, log_c = skew_constants(eta, lam)
    side = np.where(b * z + a < 0, 1 - lam, 1 + lam)
    u = (b * z + a) / side
    return math.log(b) + log_c - 0.5 * (eta + 1) * np.log1p(u * u / (eta - 2))


def skew_ppf(p: float, eta: float, lam: float) -> float:
    a, b, _ = skew_constants(eta, lam)
    if p < (1 - lam) / 2:
        u = stats.t.ppf(p / (1 - lam), eta) * math.sqrt((eta - 2) / eta)
        return ((1 - lam) * u - a) / b
    u = stats.t.ppf((p + lam) / (1 + lam), eta) * math.sqrt((eta - 2) / eta)
    return ((1 + lam) * u - a) / b


def ged_scale(shape: float) -> float:
    return math.exp(0.5 * (special.gammaln(1 / shape) - special.gammaln(3 / shape)))


def ged_eabs(shape: float) -> float:
    return ged_scale(shape) * math.exp(special.gammaln(2 / shape) - special.gammaln(1 / shape))


def ged_logpdf(z: np.ndarray, shape: float) -> np.ndarray:
    scale = ged_scale(shape)
    return (math.log(shape) - math.log(2 * scale) - special.gammaln(1 / shape)
            - (np.abs(z) / scale) ** shape)


def ged_ppf(p: float, shape: float) -> float:
    return float(stats.gennorm.ppf(p, shape, scale=ged_scale(shape)))


@njit(cache=True)
def variance_path(r, X, mu, omega, alpha, theta, beta, gamma0, gamma1, eabs, h0):
    """Mirror egarch_forward: X[t] already contains t-1 values; clamp log h."""
    h = np.empty(len(r), dtype=np.float64)
    h[0] = h0
    for t in range(1, len(r)):
        zprev = (r[t - 1] - mu) / math.sqrt(h[t - 1])
        logh = (omega + alpha * (abs(zprev) - eabs) + theta * zprev
                + beta * math.log(h[t - 1]) + gamma0 * X[t, 0] + gamma1 * X[t, 1])
        h[t] = math.exp(min(30.0, max(-30.0, logh)))
    return h


def model_parts(params: np.ndarray, dist: str, r: np.ndarray, X: np.ndarray,
                h0: float) -> tuple[np.ndarray, np.ndarray]:
    mu, omega, alpha, theta, beta, g0, g1 = params[:7]
    eabs = skew_eabs(params[7], params[8]) if dist == "skewt" else ged_eabs(params[7])
    h = variance_path(r, X, mu, omega, alpha, theta, beta, g0, g1, eabs, h0)
    z = (r - mu) / np.sqrt(h)
    logfz = skew_logpdf(z, params[7], params[8]) if dist == "skewt" else ged_logpdf(z, params[7])
    return h, logfz - 0.5 * np.log(h)


def negloglik(params: np.ndarray, dist: str, r: np.ndarray, X: np.ndarray,
              h0: float) -> float:
    try:
        h, logf = model_parts(params, dist, r, X, h0)
        if not np.isfinite(h).all() or not np.isfinite(logf).all():
            return 1e20
        return float(-logf[1:].sum())  # observation zero initializes the recursion
    except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
        return 1e20


def starting_points(base: dict, dist: str, count: int, seed: int) -> list[np.ndarray]:
    b = np.array([float(base[k]) for k in ("mu", "omega", "alpha", "theta", "beta")]
                 + [float(v) for v in base["gamma"]], dtype=float)
    rng = np.random.default_rng(seed)
    if dist == "skewt":
        initial = [np.r_[b, float(base["nu"]), lam] for lam in (0.0, -0.15, 0.15)]
    else:
        initial = [np.r_[b, shape] for shape in (1.2, 1.5, 2.0)]
    while len(initial) < count:
        p = initial[len(initial) % 3].copy()
        p[:7] += rng.normal(0, [0.03, 0.05, 0.04, 0.04, 0.03, 0.04, 0.04])
        p[4] = np.clip(p[4], .02, .98)
        p[7] = max(2.1, p[7] + rng.normal(0, 2)) if dist == "skewt" else np.clip(p[7] + rng.normal(0, .3), .6, 3)
        if dist == "skewt":
            p[8] = np.clip(p[8] + rng.normal(0, .1), -.8, .8)
        initial.append(p)
    return initial[:count]


def fit(r: np.ndarray, X: np.ndarray, base: dict, dist: str,
        starts: int = 10) -> tuple[np.ndarray, dict]:
    h0 = float(np.var(r, ddof=1))  # torch.var default, matching original study
    bounds = [(-5, 5), (-10, 10), (-3, 3), (-3, 3), (1e-4, .9999),
              (-3, 3), (-3, 3), (2.05, 150) if dist == "skewt" else (.5, 10)]
    if dist == "skewt":
        bounds.append((-.95, .95))
    runs = []
    best = None
    for i, start in enumerate(starting_points(base, dist, starts, 20261014)):
        result = optimize.minimize(negloglik, start, args=(dist, r, X, h0),
                                   method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 1500, "ftol": 1e-11, "gtol": 1e-5, "maxls": 40})
        runs.append({"start": i, "success": bool(result.success),
                     "message": str(result.message), "loglik": float(-result.fun),
                     "iterations": int(result.nit)})
        print(f"{dist} start {i+1}/{starts}: ll={-result.fun:.6f}, "
              f"success={result.success}, iters={result.nit}", flush=True)
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError(f"No finite {dist} fit")
    return best.x, {"runs": runs, "optimizer_success": bool(best.success),
                    "optimizer_message": str(best.message), "nll": float(best.fun)}


def evaluate(p: np.ndarray, dist: str, r: np.ndarray, X: np.ndarray,
             rv: np.ndarray, n_train: int) -> dict:
    h0 = float(np.var(r[:n_train], ddof=1))
    h, logf = model_parts(p, dist, r, X, h0)
    h, logf, rt, rvt = h[n_train:], logf[n_train:], r[n_train:], rv[n_train:]
    shape = p[7]
    out = {"mae": float(np.mean(np.abs(rvt - h))),
           "rmse": float(np.sqrt(np.mean((rvt - h) ** 2))),
           "qlike": float(np.mean(np.log(h) + rvt / h)),
           "nls": float(-np.mean(logf)), "n_test": len(rt)}
    for level in (.01, .05):
        q = skew_ppf(level, shape, p[8]) if dist == "skewt" else ged_ppf(level, shape)
        hits = rt < p[0] + np.sqrt(h) * q
        out[f"var_{int(level*100)}pct_quantile_z"] = float(q)
        out[f"var_{int(level*100)}pct_violations"] = int(hits.sum())
    return out


def self_test() -> None:
    """Numerically check the density, moments, centering, and t nesting."""
    for eta, lam in ((3.5, -.6), (6.0, 0.0), (11.0, .4)):
        fn = lambda x: math.exp(float(skew_logpdf(np.array([x]), eta, lam)[0]))
        split = -skew_constants(eta, lam)[0] / skew_constants(eta, lam)[1]
        for power, expected in ((0, 1.0), (1, 0.0), (2, 1.0)):
            got = integrate.quad(lambda x: x**power * fn(x), -np.inf, split,
                                 epsabs=1e-9)[0] + integrate.quad(
                                     lambda x: x**power * fn(x), split, np.inf, epsabs=1e-9)[0]
            assert abs(got - expected) < 1e-7, (eta, lam, power, got)
        eabs = integrate.quad(lambda x: abs(x)*fn(x), -np.inf, split)[0] + integrate.quad(
            lambda x: abs(x)*fn(x), split, np.inf)[0]
        assert abs(eabs - skew_eabs(eta, lam)) < 1e-7, (eta, lam, eabs, skew_eabs(eta, lam))
    for shape in (.75, 1.0, 1.5, 2.0, 3.0):
        assert abs(stats.gennorm.var(shape, scale=ged_scale(shape)) - 1) < 1e-10
        assert abs(stats.gennorm.expect(lambda x: abs(x), args=(shape,),
                                       scale=ged_scale(shape)) - ged_eabs(shape)) < 1e-8
    z = np.linspace(-5, 5, 100)
    assert np.max(np.abs(skew_logpdf(z, 5.0, 0.0)
                         - stats.t.logpdf(z / math.sqrt(3/5), 5.0)
                         + math.log(math.sqrt(3/5)))) < 1e-12
    for eta, lam in ((4.0, -.4), (7.0, .3)):
        for probability in (.01, .05, .5, .95):
            quantile = skew_ppf(probability, eta, lam)
            split = -skew_constants(eta, lam)[0] / skew_constants(eta, lam)[1]
            density = lambda x: math.exp(float(skew_logpdf(np.array([x]), eta, lam)[0]))
            if quantile <= split:
                cdf = integrate.quad(density, -np.inf, quantile)[0]
            else:
                cdf = (1-lam)/2 + integrate.quad(density, split, quantile)[0]
            assert abs(cdf - probability) < 1e-7, (eta, lam, probability, cdf)
    print("Distribution moment, centering, and skew-t nesting checks passed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--starts", type=int, default=10)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    d = torch.load(DATA, map_location="cpu", weights_only=False)
    baseline = torch.load(BASE, map_location="cpu", weights_only=False)
    r = d["r_all"].numpy().astype(np.float64)
    X = d["X_all"].numpy().astype(np.float64)
    rv = d["rv_all"].numpy().astype(np.float64)
    n = int(d["n_train"])
    if X.shape != (len(r), 2) or n < 100 or n >= len(r):
        raise ValueError("Unexpected saved sample/split")
    b = baseline["params"]
    b_np = np.array([float(b[k]) for k in ("mu", "omega", "alpha", "theta", "beta")]
                    + [float(v) for v in b["gamma"]] + [float(b["nu"]), 0.0])
    nested_ll = -negloglik(b_np, "skewt", r[:n], X[:n], float(np.var(r[:n], ddof=1)))
    if abs(nested_ll - float(baseline["loglik"])) > 1e-6:
        raise RuntimeError(f"Student-t baseline not reproduced: {nested_ll} vs {baseline['loglik']}")
    baseline_names = ["mu", "omega", "alpha", "theta", "beta", "gamma_volume", "gamma_rv", "nu"]
    output = {"sample": {"n_total": len(r), "n_train": n, "n_test": len(r)-n},
              "baseline": {"model": "EGARCH-X Student-t",
                           "params": dict(zip(baseline_names, map(float, b_np[:8]))),
                           "e_abs_z": float(skew_eabs(b_np[7], 0.0)),
                           "loglik": float(baseline["loglik"]),
                           "aic": float(baseline["aic"]), "bic": float(baseline["bic"]),
                           "oos": {**{k: float(baseline["oos"][k]) for k in
                                      ("mae", "rmse", "qlike", "nls")},
                                   **{f"var_{level}pct_violations": int(baseline["oos"][f"n_violations_{level}pct"])
                                      for level in (1, 5)}}},
              "models": {}}
    for dist in ("skewt", "ged"):
        p, diag = fit(r[:n], X[:n], b, dist, args.starts)
        names = ["mu", "omega", "alpha", "theta", "beta", "gamma_volume", "gamma_rv",
                 "eta" if dist == "skewt" else "ged_shape"]
        if dist == "skewt":
            names.append("lambda_skew")
        ll = -diag["nll"]
        k = len(p)
        result = {"model": f"EGARCH-X {dist}", "params": dict(zip(names, map(float, p))),
                  "e_abs_z": float(skew_eabs(p[7], p[8]) if dist == "skewt" else ged_eabs(p[7])),
                  "loglik": ll, "n_params": k, "aic": 2*k - 2*ll,
                  "bic": k*math.log(n) - 2*ll,
                  "oos": evaluate(p, dist, r, X, rv, n), "optimization": diag}
        output["models"][dist] = result
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "egarch_x_skewt_ged_mle.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved results/phan_phoi_moi/{path.name}")
    for key, row in [("student_t", output["baseline"]), *output["models"].items()]:
        print(f"{key:10s} AIC={row['aic']:.3f} BIC={row['bic']:.3f} "
              f"QLIKE={row['oos']['qlike']:.5f} NLS={row['oos']['nls']:.5f}")


if __name__ == "__main__":
    main()
