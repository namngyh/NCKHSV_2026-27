"""Variational Bayes for EGARCH-X with Hansen skew-t and standardized GED.

Keeps the original study's priors for shared coefficients, AutoNormal and
AutoMultivariateNormal guides, 3000 Adam steps, and mixture forecast evaluation.
The custom autograd operation differentiates the exact MLE likelihood by
central finite differences; the likelihood itself is unchanged.

Run from `code thuật toán`:
    python scripts/uoc_luong_vb_egarch_x_phan_phoi_moi.py --self-test
    python scripts/uoc_luong_vb_egarch_x_phan_phoi_moi.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal, AutoNormal
from scipy import special

from uoc_luong_egarch_x_phan_phoi_moi import (
    DATA, OUT, ROOT, ged_ppf, model_parts, negloglik, skew_ppf,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
torch.set_default_dtype(torch.float64)
SITE_NAMES = ("mu", "gamma", "nu_minus_2", "lambda_unit", "shape", "omega",
              "alpha", "theta", "beta")


class LogLikelihood(torch.autograd.Function):
    """Exact log likelihood; central-difference derivatives of that likelihood."""

    @staticmethod
    def forward(ctx, params, returns, exog, h0, family):
        p = params.detach().cpu().numpy().copy()
        ctx.p, ctx.returns, ctx.exog, ctx.h0, ctx.family = p, returns, exog, h0, family
        value = -negloglik(p, family, returns, exog, h0)
        return params.new_tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        p = ctx.p
        gradients = np.empty_like(p)
        for i in range(len(p)):
            step = 1e-5 * max(1.0, abs(p[i]))
            if i == 7:  # eta > 2 or GED shape > 0
                step = min(step, .25 * (p[i] - (2 if ctx.family == "skewt" else 0)))
            elif i == 8:  # |lambda| < 1
                step = min(step, .25 * (1 - abs(p[i])))
            step = max(step, 1e-9)
            plus, minus = p.copy(), p.copy()
            plus[i], minus[i] = p[i] + step, p[i] - step
            gradients[i] = (-negloglik(plus, ctx.family, ctx.returns, ctx.exog, ctx.h0)
                            + negloglik(minus, ctx.family, ctx.returns, ctx.exog, ctx.h0)) / (2 * step)
        return grad_output * torch.from_numpy(gradients).to(grad_output), None, None, None, None


def model(r, X, r_np, X_np, h0, family):
    """Original EGARCH-X priors plus one distribution-specific prior."""
    mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
    gamma = pyro.sample("gamma", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
    if family == "skewt":
        eta = pyro.sample("nu_minus_2", dist.Gamma(2.0, 0.2)) + 2.0
        lam = 2 * pyro.sample("lambda_unit", dist.Beta(2.0, 2.0)) - 1
    elif family == "ged":
        shape = pyro.sample("shape", dist.Gamma(2.0, 1.0))
    else:
        raise ValueError(family)
    omega = pyro.sample("omega", dist.Normal(0.0, 1.0))
    alpha = pyro.sample("alpha", dist.Beta(2.0, 2.0))
    theta = pyro.sample("theta", dist.Normal(0.0, 1.0))
    beta = pyro.sample("beta", dist.Beta(2.0, 2.0))
    fields = [mu, omega, alpha, theta, beta, gamma[0], gamma[1],
              eta if family == "skewt" else shape]
    if family == "skewt":
        fields.append(lam)
    p = torch.stack(fields)
    pyro.factor("loglik", LogLikelihood.apply(p, r_np, X_np, h0, family))
    return p


def vector_from_draws(draws: dict, family: str, i: int) -> np.ndarray:
    fields = [float(draws[k][i]) for k in ("mu", "omega", "alpha", "theta", "beta")]
    fields.extend(float(x) for x in draws["gamma"][i])
    if family in ("skewt", "student_t"):
        fields.extend((float(draws["nu_minus_2"][i]) + 2.0,
                       0.0 if family == "student_t" else 2 * float(draws["lambda_unit"][i]) - 1))
    else:
        fields.append(float(draws["shape"][i]))
    return np.array(fields, dtype=np.float64)


def summarize(draws: dict, family: str) -> dict:
    matrix = np.stack([vector_from_draws(draws, family, i) for i in range(len(draws["mu"]))])
    names = ["mu", "omega", "alpha", "theta", "beta", "gamma_volume", "gamma_rv",
             "eta" if family == "skewt" else "ged_shape"]
    if family == "skewt":
        names.append("lambda_skew")
    return {name: {"mean": float(matrix[:, j].mean()),
                   "sd": float(matrix[:, j].std(ddof=1)),
                   "q2.5": float(np.quantile(matrix[:, j], .025)),
                   "q97.5": float(np.quantile(matrix[:, j], .975))}
            for j, name in enumerate(names)}


def evaluate_draws(draws: dict, family: str, r: np.ndarray, X: np.ndarray,
                   rv: np.ndarray, n_train: int, max_draws: int = 200) -> dict:
    total = len(draws["mu"])
    ids = np.arange(0, total, max(1, total // max_draws))[:max_draws]
    h0 = float(np.var(r[:n_train], ddof=1))
    paths, logdensities, thresholds = [], [], {0.01: [], 0.05: []}
    for i in ids:
        p = vector_from_draws(draws, family, int(i))
        density = "skewt" if family == "student_t" else family
        h, logf = model_parts(p, density, r, X, h0)
        h, logf = h[n_train:], logf[n_train:]
        paths.append(h)
        logdensities.append(logf)
        for level in thresholds:
            q = skew_ppf(level, p[7], p[8]) if density == "skewt" else ged_ppf(level, p[7])
            thresholds[level].append(p[0] + np.sqrt(h) * q)
    paths = np.stack(paths)
    logdensities = np.stack(logdensities)
    mean_h = paths.mean(0)
    rv_test, r_test = rv[n_train:], r[n_train:]
    # Same posterior mixture NLS and average component quantiles as ngoai_mau.py.
    out = {"mae": float(np.mean(np.abs(rv_test - mean_h))),
           "rmse": float(np.sqrt(np.mean((rv_test - mean_h)**2))),
           "qlike": float(np.mean(np.log(mean_h) + rv_test / mean_h)),
           "nls": float(-np.mean(special.logsumexp(logdensities, axis=0) - math.log(len(ids)))),
           "n_test": len(rv_test), "n_draws_used": len(ids)}
    for level, components in thresholds.items():
        threshold = np.stack(components).mean(0)
        out[f"var_{int(level*100)}pct_violations"] = int(np.sum(r_test < threshold))
    return out


def fit_one(family: str, guide_type: str, data: dict, steps: int, draws_count: int,
            seed: int = 0, learning_rate: float = .01) -> tuple[dict, dict]:
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    n = int(data["n_train"])
    r = data["r_all"].to(torch.float64)
    X = data["X_all"].to(torch.float64)
    rv = data["rv_all"].numpy().astype(np.float64)
    r_np, X_np = r[:n].numpy(), X[:n].numpy()
    h0 = float(np.var(r_np, ddof=1))
    guide = AutoNormal(model) if guide_type == "mean_field" else AutoMultivariateNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": learning_rate}), loss=Trace_ELBO())
    args = (r[:n], X[:n], r_np, X_np, h0, family)
    trace = []
    began = time.monotonic()
    for step in range(steps):
        loss = svi.step(*args)
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite ELBO at {family}/{guide_type} step {step}")
        trace.append(-float(loss))
        if (step + 1) % 200 == 0:
            print(f"{family}/{guide_type} step {step+1}/{steps}: "
                  f"ELBO50={np.mean(trace[-50:]):.3f}", flush=True)
    predictive = Predictive(model, guide=guide, num_samples=draws_count)
    samples_raw = predictive(*args)
    samples = {name: samples_raw[name].detach().cpu() for name in SITE_NAMES if name in samples_raw}
    summary = summarize(samples, family)
    oos = evaluate_draws(samples, family, r.numpy(), X.numpy(), rv, n)
    result = {"family": family, "guide": guide_type, "steps": steps,
              "n_posterior_draws": draws_count, "seed": seed,
              "final_elbo": float(np.mean(trace[-50:])), "summary": summary,
              "oos": oos, "elapsed_seconds": time.monotonic() - began}
    artifact = {"result": result, "samples": samples, "elbo_trace": trace,
                "param_store": pyro.get_param_store().get_state()}
    return result, artifact


def self_test() -> None:
    """Check finite-difference gradients against the original t autograd path."""
    from garch_vn30.mo_hinh.egarch import EGARCHParams, egarch_loglik
    d = torch.load(DATA, weights_only=False)
    r, X = d["r_all"][:80], d["X_all"][:80]
    p = torch.tensor([.08, .16, .07, -.14, .68, -.07, .3, 4.3, 0.0],
                     requires_grad=True)
    val = LogLikelihood.apply(p, r.numpy(), X.numpy(), float(torch.var(r)), "skewt")
    val.backward()
    original = EGARCHParams(mu=p[0], omega=p[1], alpha=p[2], theta=p[3], beta=p[4],
                            gamma=p[5:7], nu=p[7])
    reference = egarch_loglik(original, r, X)
    gradient = torch.autograd.grad(reference, p)[0]
    assert abs(float((val - reference).detach())) < 1e-8
    assert np.max(np.abs(p.grad.detach().numpy()[:8] - gradient.detach().numpy()[:8])) < 1e-3
    original_vb = torch.load(ROOT / "results" / "pipeline" / "egarch_x_t_vb_mean_field.pt",
                             weights_only=False)
    own = evaluate_draws(original_vb["samples"], "student_t",
                         d["r_all"].numpy(), d["X_all"].numpy(),
                         d["rv_all"].numpy(), int(d["n_train"]))
    for name in ("qlike", "nls", "mae", "rmse"):
        assert abs(own[name] - float(original_vb["oos"][name])) < 1e-8, name
    for level in (1, 5):
        assert own[f"var_{level}pct_violations"] == original_vb["oos"][f"n_violations_{level}pct"]
    print("VB likelihood and numerical gradient match the original Student-t implementation.")
    print("Posterior mixture forecast scores reproduce the original Student-t VB result.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--family", choices=("skewt", "ged", "both"), default="both")
    ap.add_argument("--guide", choices=("mean_field", "full_rank", "both"), default="both")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    data = torch.load(DATA, weights_only=False)
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / "egarch_x_skewt_ged_vb.json"
    results = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    families = ("skewt", "ged") if args.family == "both" else (args.family,)
    guides = ("mean_field", "full_rank") if args.guide == "both" else (args.guide,)
    for family in families:
        for guide in guides:
            key = f"{family}_{guide}"
            if key in results and results[key]["steps"] >= args.steps:
                print(f"Existing {key} skipped", flush=True)
                continue
            result, artifact = fit_one(family, guide, data, args.steps, args.draws)
            torch.save(artifact, OUT / f"egarch_x_{key}_vb.pt")
            results[key] = result
            result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved {key}: ELBO={result['final_elbo']:.3f}, "
                  f"QLIKE={result['oos']['qlike']:.5f}, NLS={result['oos']['nls']:.5f}", flush=True)


if __name__ == "__main__":
    main()
