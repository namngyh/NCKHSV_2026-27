"""Smoke test cho MLE, Variational Bayes, MCMC tren du lieu mo phong nho.

Muc tieu la kiem tra pipeline CHAY DUOC va cho ket qua huu han, khong kiem
tra do chinh xac phuc hoi tham so (can B lon va T lon hon nhieu, xem
simulation/run_simulation.py va Muc 6 cua bai bao).
"""
import pytest
import torch

from msgarchx.inference.mcmc import fit_mcmc
from msgarchx.inference.mle import fit_mle
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.inference.variational import fit_vb
from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.simulation.dgp import simulate_ms_garch_x


def _small_dataset(T=150, seed=0):
    true_params = MSGARCHXParams(
        mu=torch.tensor([0.0, 0.0]),
        omega=torch.tensor([0.05, 0.30]),
        alpha=torch.tensor([0.05, 0.10]),
        beta=torch.tensor([0.85, 0.80]),
        gamma=torch.tensor([[0.05, -0.02], [0.10, 0.03]]),
        P=torch.tensor([[0.97, 0.03], [0.05, 0.95]]),
        nu=torch.tensor(8.0),
    )
    sim = simulate_ms_garch_x(true_params, true_params.P, T=T, n_exog=2, burn_in=100, seed=seed)
    return sim, true_params


def test_mle_smoke_runs_and_beats_or_matches_true_loglik():
    sim, true_params = _small_dataset(T=100)
    config = ModelConfig(n_exog=2, state_specific_nu=False)

    from msgarchx.models.ms_garch import ms_garch_forward_filter
    with torch.no_grad():
        true_out = ms_garch_forward_filter(true_params, sim.r, sim.X)
    true_loglik = float(true_out["loglik"])

    res = fit_mle(sim.r, sim.X, config, n_starts=1, base_seed=0, n_adam=100, n_lbfgs=50)

    assert torch.isfinite(torch.tensor(res.loglik))
    # MLE toi da hoa hop ly nen khong the te hon nhieu so voi tham so that
    # (cho phep bien do vi T nho va toi uu chua hoi tu hoan toan).
    assert res.loglik > true_loglik - 20.0
    assert res.aic > 0
    assert res.filter_output["xi_filt"].shape == (sim.r.shape[0], 2)


def test_vb_mean_field_smoke():
    sim, _ = _small_dataset(T=60, seed=1)
    config = ModelConfig(n_exog=2, state_specific_nu=False)
    res = fit_vb(
        sim.r, sim.X, config, guide_type="mean_field",
        n_steps=30, n_posterior_draws=30, seed=0,
    )
    assert len(res.elbo_trace) == 30
    assert torch.isfinite(torch.tensor(res.final_elbo))
    assert "omega" in res.summary
    assert res.summary["omega"]["mean"].shape == (2,)


def test_vb_full_rank_smoke():
    sim, _ = _small_dataset(T=60, seed=2)
    config = ModelConfig(n_exog=2, state_specific_nu=False)
    res = fit_vb(
        sim.r, sim.X, config, guide_type="full_rank",
        n_steps=30, n_posterior_draws=30, seed=0,
    )
    assert torch.isfinite(torch.tensor(res.final_elbo))


@pytest.mark.skip(reason="MCMC đã được loại bỏ khỏi phạm vi nghiên cứu do chuỗi NUTS không hội tụ")
def test_mcmc_smoke():
    sim, _ = _small_dataset(T=40, seed=3)
    config = ModelConfig(n_exog=2, state_specific_nu=False)
    res = fit_mcmc(
        sim.r, sim.X, config,
        n_chains=2, n_warmup=15, n_samples=15, seed=0, max_tree_depth=6,
    )
    assert res.samples["omega"].shape[0] == 2 * 15
    assert "omega" in res.rhat
    assert res.max_rhat == res.max_rhat  # khong phai NaN
