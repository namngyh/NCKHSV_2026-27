"""Dieu phoi nghien cuu mo phong: B lan lap x {MLE, Variational Bayes, MCMC}
tren cung mot kich ban DGP (Muc 6).

Voi moi lan lap, du lieu duoc sinh tu `simulate_ms_garch_x`, sau do uoc luong
lai bang cac phuong phap trong `methods`. MLE chi bao cao Bias/RMSE (diem uoc
luong); Variational Bayes va MCMC bao cao them Coverage tu khoang tin cay hau
nghiem 95% (q2.5/q97.5) vi day la khoang tu nhien co san tu suy dien Bayes.
Brier score (eq:brier-state) so sanh xac suat trang thai loc/hau nghiem trung
binh voi trang thai that.

CANH BAO CHI PHI: MCMC va Variational Bayes voi nhieu lan lap co the ton thoi
gian dang ke; nghien cuu mo phong day du (B lon nhu Bang 6.3) nen chay ngoai
tuyen / tren nhieu tien trinh, khong phai trong mot phien demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from msgarchx.inference.mcmc import fit_mcmc
from msgarchx.inference.mle import fit_mle
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.inference.variational import fit_vb
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter
from msgarchx.simulation.dgp import simulate_ms_garch_x
from msgarchx.simulation.metrics import bias, brier_state, parameter_rmse, coverage
from msgarchx.simulation.scenarios import SimulationScenario


def _named_params(params: MSGARCHXParams) -> dict[str, float]:
    d = {
        "mu1": float(params.mu[0]), "mu2": float(params.mu[1]),
        "omega1": float(params.omega[0]), "omega2": float(params.omega[1]),
        "alpha1": float(params.alpha[0]), "alpha2": float(params.alpha[1]),
        "beta1": float(params.beta[0]), "beta2": float(params.beta[1]),
        "p11": float(params.P[0, 0]), "p22": float(params.P[1, 1]),
    }
    for s in range(2):
        for j in range(params.gamma.shape[1]):
            d[f"gamma{s+1}_{j}"] = float(params.gamma[s, j])
    if params.state_specific_nu:
        d["nu1"], d["nu2"] = float(params.nu[0]), float(params.nu[1])
    else:
        d["nu"] = float(params.nu)
    return d


def _apply_misspecification(r: torch.Tensor, misspecified: bool, phi: float = 0.3) -> torch.Tensor:
    if not misspecified:
        return r
    r_shift = torch.roll(r, shifts=1)
    r_shift[0] = 0.0
    return r + phi * r_shift


@dataclass
class SimulationReplicationResult:
    method: str
    point_estimate: dict[str, float]
    ci_lower: dict[str, float] | None
    ci_upper: dict[str, float] | None
    brier: float | None


_DEFAULT_METHOD_KWARGS = {
    "mle": {"n_starts": 3},
    "vb_mean_field": {"n_steps": 1500},
    "vb_full_rank": {"n_steps": 1500},
    "mcmc": {"n_chains": 2, "n_warmup": 250, "n_samples": 500},
}


def _run_one_method(
    method: str,
    r: torch.Tensor,
    X: torch.Tensor,
    config: ModelConfig,
    true_state: torch.Tensor,
    seed: int,
    method_kwargs: dict | None = None,
) -> SimulationReplicationResult:
    kwargs = {**_DEFAULT_METHOD_KWARGS.get(method, {}), **(method_kwargs or {})}

    if method == "mle":
        res = fit_mle(r, X, config, base_seed=seed, **kwargs)
        est = _named_params(res.params)
        xi = res.filter_output["xi_filt"].detach().numpy()
        return SimulationReplicationResult("mle", est, None, None, brier_state(xi, true_state.numpy()))

    if method in ("vb_mean_field", "vb_full_rank"):
        guide_type = "mean_field" if method == "vb_mean_field" else "full_rank"
        res = fit_vb(r, X, config, guide_type=guide_type, seed=seed, **kwargs)
        mean_params = params_from_summary_mean(res.summary, config)
        with torch.no_grad():
            out = ms_garch_forward_filter(mean_params, r, X)
        est = _summary_to_named(res.summary, key="mean", config=config)
        lo = _summary_to_named(res.summary, key="q2.5", config=config)
        hi = _summary_to_named(res.summary, key="q97.5", config=config)
        return SimulationReplicationResult(method, est, lo, hi, brier_state(out["xi_filt"].numpy(), true_state.numpy()))

    if method == "mcmc":
        # verbose=True de thanh tien trinh tqdm ghi ra nhat ky: khong co no
        # thi mot lan lap MCMC chay HANG CHUC GIO ma khong in gi, khong biet
        # dang o dau va con bao lau.
        kwargs.setdefault("verbose", True)
        res = fit_mcmc(r, X, config, seed=seed, **kwargs)
        mean_params = params_from_summary_mean(res.summary, config)
        with torch.no_grad():
            out = ms_garch_forward_filter(mean_params, r, X)
        est = _summary_to_named(res.summary, key="mean", config=config)
        lo = _summary_to_named(res.summary, key="q2.5", config=config)
        hi = _summary_to_named(res.summary, key="q97.5", config=config)
        return SimulationReplicationResult(method, est, lo, hi, brier_state(out["xi_filt"].numpy(), true_state.numpy()))

    raise ValueError(f"Phuong phap khong ho tro: {method!r}")


def params_from_summary_mean(summary: dict, config: ModelConfig) -> MSGARCHXParams:
    p11, p22 = summary["p11"]["mean"], summary["p22"]["mean"]
    P = torch.stack([torch.stack([p11, 1 - p11]), torch.stack([1 - p22, p22])])
    nu = summary["nu_minus_2"]["mean"] + 2.0
    return MSGARCHXParams(
        mu=summary["mu"]["mean"], omega=summary["omega"]["mean"],
        alpha=summary["alpha"]["mean"], beta=summary["beta"]["mean"],
        gamma=summary["gamma"]["mean"], P=P, nu=nu,
    )


def _summary_to_named(summary: dict, key: str, config: ModelConfig) -> dict[str, float]:
    mu, omega, alpha, beta, gamma = (summary[n][key] for n in ["mu", "omega", "alpha", "beta", "gamma"])
    d = {
        "mu1": float(mu[0]), "mu2": float(mu[1]),
        "omega1": float(omega[0]), "omega2": float(omega[1]),
        "alpha1": float(alpha[0]), "alpha2": float(alpha[1]),
        "beta1": float(beta[0]), "beta2": float(beta[1]),
        "p11": float(summary["p11"][key]), "p22": float(summary["p22"][key]),
    }
    for s in range(2):
        for j in range(config.n_exog):
            d[f"gamma{s+1}_{j}"] = float(gamma[s, j])
    nu_val = summary["nu_minus_2"][key] + 2.0
    if config.state_specific_nu:
        d["nu1"], d["nu2"] = float(nu_val[0]), float(nu_val[1])
    else:
        d["nu"] = float(nu_val)
    return d


def run_scenario(
    scenario: SimulationScenario,
    config: ModelConfig,
    methods: tuple[str, ...] = ("mle",),
    base_seed: int = 0,
    method_kwargs: dict[str, dict] | None = None,
) -> dict:
    """Chay `scenario.B` lan lap; tra ve dict[method] -> {"replications": [...], "bias": {...}, "rmse": {...}, "coverage": {...} (neu co CI), "mean_brier": float}.

    `method_kwargs`: vd. {"mcmc": {"n_chains": 4, "n_warmup": 100, "n_samples": 150}}
    de ghi de sieu tham so mac dinh (_DEFAULT_METHOD_KWARGS) theo tung phuong phap.
    """
    # scenario.params.P chi la placeholder (eye(2)) -- P0 rieng moi la ma tran
    # chuyen trang thai THAT dung de sinh du lieu (xem scenarios.py). Phai ghi
    # de p11/p22 sau khi goi _named_params, neu khong Bias/RMSE se tinh nham
    # so voi p11=p22=1.0.
    true_named = _named_params(scenario.params)
    true_named["p11"] = float(scenario.P0[0, 0])
    true_named["p22"] = float(scenario.P0[1, 1])
    results = {m: [] for m in methods}
    method_kwargs = method_kwargs or {}

    for b in range(scenario.B):
        sim = simulate_ms_garch_x(
            scenario.params, scenario.P0, scenario.T,
            n_exog=config.n_exog, burn_in=scenario.burn_in, seed=base_seed + b,
        )
        r = _apply_misspecification(sim.r, scenario.misspecified)
        for method in methods:
            results[method].append(
                _run_one_method(method, r, sim.X, config, sim.state, seed=base_seed + b,
                                 method_kwargs=method_kwargs.get(method))
            )

    # Sap nhan truoc khi gop, giong het duong chay song song (combine() trong
    # scripts/chay_mo_phong_song_song.py) -- hai duong phai cho cung ket qua.
    return {method: _summarize_method([sap_nhan_trang_thai(r) for r in reps], true_named)
            for method, reps in results.items()}


def phuong_sai_vo_dieu_kien(est: dict[str, float], s: int) -> float:
    """h_s = omega_s / (1 - alpha_s - beta_s), phuong sai vo dieu kien cua trang thai s.

    Dung lam tieu chi sap nhan. Mau duoc chan duoi vi nghiem uoc luong co the
    cho alpha+beta >= 1 (khong dung), khi do ty so mat nghia; chan lai van giu
    dung THU TU giua hai trang thai, la thu duy nhat ta can o day.
    """
    return est[f"omega{s}"] / max(1.0 - est[f"alpha{s}"] - est[f"beta{s}"], 0.05)


def _hoan_vi(d: dict[str, float] | None) -> dict[str, float] | None:
    """Doi cho hai trang thai trong mot dict ten -> gia tri."""
    if d is None:
        return None
    ra = dict(d)
    for goc in ("mu", "omega", "alpha", "beta", "nu"):
        a, b = f"{goc}1", f"{goc}2"
        if a in d and b in d:
            ra[a], ra[b] = d[b], d[a]
    if "p11" in d and "p22" in d:
        ra["p11"], ra["p22"] = d["p22"], d["p11"]
    for ten in d:
        if ten.startswith("gamma1_"):
            doi = "gamma2_" + ten.split("_", 1)[1]
            if doi in d:
                ra[ten], ra[doi] = d[doi], d[ten]
    return ra


def sap_nhan_trang_thai(rep: SimulationReplicationResult) -> SimulationReplicationResult:
    """Dua mot lan lap ve quy uoc "trang thai 1 = bien dong THAP".

    VI SAO BAT BUOC. Mo hinh chuyen trang thai co doi xung hoan vi nhan: doi
    cho hai trang thai cho ra dung mot ham likelihood. Bo uoc luong vi the tra
    ve nhan theo thu tu TUY Y. Gop 100 lan lap ma khong sap nhan truoc thi
    omega_1 va omega_2 bi tron vao nhau, va moi Bias/RMSE/Coverage cua tham so
    theo trang thai deu vo nghia -- do chinh la loi da xay ra: VB dao nhan o
    khoang mot nua so lan lap, tuc dung bang tung dong xu.

    Khong ap dung cho mo hinh MOT trang thai (EGARCH): khong co khoa "omega1"
    thi ham tra ve nguyen ven.
    """
    est = rep.point_estimate
    if "omega1" not in est or "omega2" not in est:
        return rep
    if phuong_sai_vo_dieu_kien(est, 1) <= phuong_sai_vo_dieu_kien(est, 2):
        return rep
    return SimulationReplicationResult(
        rep.method, _hoan_vi(est), _hoan_vi(rep.ci_lower), _hoan_vi(rep.ci_upper), rep.brier,
    )


def _summarize_method(reps: list[SimulationReplicationResult], true_named: dict[str, float]) -> dict:
    if not reps:
        return {}
    names = reps[0].point_estimate.keys()
    out = {"bias": {}, "rmse": {}, "coverage": {}, "mean_brier": None}
    for name in names:
        theta0 = true_named.get(name)
        if theta0 is None:
            continue
        theta_hat = np.array([rep.point_estimate[name] for rep in reps])
        out["bias"][name] = bias(theta_hat, theta0)
        out["rmse"][name] = parameter_rmse(theta_hat, theta0)
        if reps[0].ci_lower is not None:
            lo = np.array([rep.ci_lower[name] for rep in reps])
            hi = np.array([rep.ci_upper[name] for rep in reps])
            out["coverage"][name] = coverage(lo, hi, theta0)
    briers = [rep.brier for rep in reps if rep.brier is not None]
    out["mean_brier"] = float(np.mean(briers)) if briers else None
    return out
