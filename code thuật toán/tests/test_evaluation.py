import numpy as np
import torch

from msgarchx.evaluation.dm_test import diebold_mariano_test
from msgarchx.evaluation.density_scores import crps_from_samples, pit_from_samples
from msgarchx.evaluation.var_backtests import christoffersen_test, kupiec_test
from msgarchx.evaluation.volatility_loss import mae_vol, qlike, rmse_vol


def test_volatility_losses_zero_when_perfect():
    rv = torch.tensor([1.0, 2.0, 0.5])
    h_hat = rv.clone()
    assert mae_vol(rv, h_hat).item() == 0.0
    assert rmse_vol(rv, h_hat).item() == 0.0
    expected_qlike = float(np.mean(np.log(rv.numpy()) + 1.0))
    assert abs(qlike(rv, h_hat).item() - expected_qlike) < 1e-8


def test_qlike_penalizes_bad_forecast_more():
    rv = torch.tensor([1.0])
    good = qlike(rv, torch.tensor([1.0]))
    bad = qlike(rv, torch.tensor([0.1]))
    assert bad > good


def test_crps_is_small_for_tight_correct_samples():
    torch.manual_seed(0)
    samples = torch.randn(5000) * 0.01
    val = crps_from_samples(samples, 0.0)
    assert val < 0.05


def test_pit_uniform_marginal_roughly():
    torch.manual_seed(0)
    samples = torch.randn(10000)
    u = pit_from_samples(samples, 0.0)
    assert 0.4 < u < 0.6


def test_kupiec_rejects_wrong_rate():
    hits = np.zeros(1000)
    hits[:100] = 1  # 10% vi pham nhung ky vong 1%
    res = kupiec_test(hits, alpha=0.01)
    assert res["p_value"] < 0.01


def test_kupiec_accepts_correct_rate():
    rng = np.random.default_rng(0)
    hits = (rng.uniform(size=5000) < 0.01).astype(float)
    res = kupiec_test(hits, alpha=0.01)
    assert res["p_value"] > 0.01


def test_christoffersen_detects_clustering():
    hits = np.zeros(200)
    hits[50:60] = 1  # cum vi pham lien tuc
    res = christoffersen_test(hits, alpha=0.01)
    assert res["LR_ind"] > 0


def test_diebold_mariano_prefers_lower_loss_model():
    rng = np.random.default_rng(0)
    loss1 = rng.normal(1.0, 0.1, size=500)
    loss2 = rng.normal(0.5, 0.1, size=500)
    res = diebold_mariano_test(loss1, loss2)
    assert res["DM_stat"] > 0
    assert res["p_value"] < 0.01
