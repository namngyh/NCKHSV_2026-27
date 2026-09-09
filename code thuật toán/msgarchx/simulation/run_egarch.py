"""Khoi phuc tham so cho EGARCH mot trang thai.

Ban song song cua run_simulation.py (von danh cho MS-GARCH-X hai trang thai).
Tach rieng thay vi nhet chung mot file vi hai ho khac nhau ca ve bo sinh du
lieu, danh sach tham so lan cach danh gia.

BA KICH BAN, trong do kich ban E2 la quan trong nhat va khong duoc bo:

    E1  Don bay ro     theta_0 = -0,10   -- co khoi phuc duoc don bay khong?
    E2  KHONG don bay  theta_0 =  0      -- co BIA RA don bay khi khong co khong?
    E3  Duoi day       nu_0 = 4          -- khoi phuc con dung khi duoi rat day?

E2 kiem tra chieu nguoc lai cua E1. Neu uoc luong theta lech khoi 0 mot cach
he thong tren du lieu VON KHONG CO don bay, thi moi ket luan "VN30F1M co hieu
ung don bay" deu mat gia tri. Day la loai cau hoi phan bien se duoc hoi, va co
san cau tra loi dinh luong thi rat manh.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from msgarchx.inference.garch_mle import fit_garch_mle
from msgarchx.inference.bayes_single import model as single_model
from msgarchx.inference.model_spec import ModelSpec
from msgarchx.inference.variational import fit_vb
from msgarchx.models.egarch import EGARCHParams
from msgarchx.simulation.dgp import simulate_egarch
# Ham gop ket qua (bias/rmse/coverage) khong phu thuoc ho mo hinh nen dung
# chung voi duong MS thay vi viet lai -- tranh hai cong thuc cho cung mot dai luong.
from msgarchx.simulation.run_simulation import (
    SimulationReplicationResult,
    _summarize_method,
)

_DEFAULT_KWARGS = {
    "mle": {"n_starts": 3},
    "vb_mean_field": {"n_steps": 1500},
    "vb_full_rank": {"n_steps": 1500},
}


@dataclass(frozen=True)
class EGARCHScenario:
    code: str
    description: str
    params: EGARCHParams
    dist: str
    T: int
    B: int
    burn_in: int = 200


def default_egarch_scenarios(T: int = 2000, B: int = 100) -> dict[str, EGARCHScenario]:
    """Ba kich ban khoi phuc tham so cho EGARCH-X Student-t.

    Muc phuong sai duoc chon de log h vo dieu kien = omega/(1-beta) = 0,8,
    tuc h ~ 2,2 -- cung bac do lon voi phuong sai mau cua VN30F1M (~2,27).
    """
    def _p(theta: float, nu: float) -> EGARCHParams:
        return EGARCHParams(
            mu=torch.tensor(0.05),
            omega=torch.tensor(0.04),
            alpha=torch.tensor(0.15),
            theta=torch.tensor(theta),
            beta=torch.tensor(0.95),
            gamma=torch.tensor([0.05, 0.05]),
            nu=torch.tensor(nu),
        )

    return {
        "E1": EGARCHScenario("E1", "Don bay ro: theta_0 = -0,10", _p(-0.10, 8.0),
                             "student_t", T, B),
        "E2": EGARCHScenario("E2", "KHONG don bay: theta_0 = 0", _p(0.0, 8.0),
                             "student_t", T, B),
        "E3": EGARCHScenario("E3", "Duoi day: nu_0 = 4, co don bay", _p(-0.10, 4.0),
                             "student_t", T, B),
    }


def named_params_egarch(params: EGARCHParams) -> dict[str, float]:
    """Bo tham so duoi dang dict ten -> gia tri, de doi chieu voi uoc luong."""
    d = {
        "mu": float(params.mu), "omega": float(params.omega),
        "alpha": float(params.alpha), "theta": float(params.theta),
        "beta": float(params.beta),
    }
    for j in range(params.gamma.shape[0]):
        d[f"gamma_{j}"] = float(params.gamma[j])
    if params.nu is not None:
        d["nu"] = float(params.nu)
    return d


def _tu_summary(summary: dict, key: str, spec: ModelSpec, k: int) -> dict[str, float]:
    """Doc mot thong ke (mean / q2.5 / q97.5) tu ket qua VB thanh dict co ten."""
    d = {name: float(summary[name][key]) for name in ["mu", "omega", "alpha", "theta", "beta"]}
    if spec.use_exog:
        for j in range(k):
            d[f"gamma_{j}"] = float(summary["gamma"][key][j])
    if spec.dist == "student_t":
        d["nu"] = float(summary["nu_minus_2"][key]) + 2.0
    return d


def _chay_mot_phuong_phap(method, r, X, spec, seed, kwargs) -> SimulationReplicationResult:
    kwargs = {**_DEFAULT_KWARGS.get(method, {}), **(kwargs or {})}

    if method == "mle":
        res = fit_garch_mle(r, X, dist=spec.dist, family="egarch", base_seed=seed, **kwargs)
        # MLE khong sinh khoang tin cay trong quy trinh nay -> khong tinh coverage.
        return SimulationReplicationResult(
            "mle", named_params_egarch(res.params), None, None, None)

    if method in ("vb_mean_field", "vb_full_rank"):
        guide = "mean_field" if method == "vb_mean_field" else "full_rank"
        res = fit_vb(r, X, config=None, guide_type=guide, seed=seed,
                     model_fn=single_model, model_args=(spec, None), **kwargs)
        k = X.shape[1]
        return SimulationReplicationResult(
            method,
            _tu_summary(res.summary, "mean", spec, k),
            _tu_summary(res.summary, "q2.5", spec, k),
            _tu_summary(res.summary, "q97.5", spec, k),
            None,
        )

    raise ValueError(f"Phuong phap khong ho tro cho EGARCH: {method!r}")


def run_egarch_scenario(
    scenario: EGARCHScenario,
    spec: ModelSpec,
    methods: tuple[str, ...] = ("mle", "vb_mean_field", "vb_full_rank"),
    base_seed: int = 0,
    method_kwargs: dict[str, dict] | None = None,
) -> dict:
    """Chay `scenario.B` lan lap; tra ve dict[method] -> bias/rmse/coverage."""
    if spec.family != "egarch":
        raise ValueError(f"spec phai thuoc ho egarch, nhan duoc {spec.family!r}")
    if spec.dist != scenario.dist:
        # Neu de lot, ta se sinh du lieu theo mot phan phoi roi uoc luong bang
        # mot phan phoi khac -- do la nghien cuu SAI DAC TA, khong phai khoi
        # phuc tham so, va ket qua bias/coverage se bi doc nham hoan toan.
        raise ValueError(
            f"Phan phoi khong khop: kich ban sinh du lieu bang {scenario.dist!r} "
            f"nhung mo hinh uoc luong la {spec.dist!r}. Khoi phuc tham so doi hoi "
            f"hai thu nay TRUNG NHAU."
        )
    that = named_params_egarch(scenario.params)
    ket_qua = {m: [] for m in methods}
    method_kwargs = method_kwargs or {}
    n_exog = scenario.params.gamma.shape[0]

    for b in range(scenario.B):
        sim = simulate_egarch(
            scenario.params, T=scenario.T, dist=scenario.dist,
            n_exog=n_exog, burn_in=scenario.burn_in, seed=base_seed + b,
        )
        X = sim.X if spec.use_exog else torch.zeros(sim.r.shape[0], 0)
        for method in methods:
            ket_qua[method].append(
                _chay_mot_phuong_phap(method, sim.r, X, spec, base_seed + b,
                                      method_kwargs.get(method))
            )

    ra = {}
    for m, reps in ket_qua.items():
        blk = _summarize_method(reps, that)
        # _summarize_method khong tu ghi so lan lap; them vao day de ket qua
        # cung dang voi cac file 07_simulation_S*.pt cua duong MS.
        blk["n_replications"] = len(reps)
        blk["tham_so_that"] = that
        ra[m] = blk
    return ra
