"""Test cho msgarchx/simulation/calibration_bias.py -- luoi 3x3 kiem dinh
"do thien vi hieu chinh" (calibration bias)."""
import pytest
import torch

from msgarchx.models.ms_garch import MSGARCHXParams
from msgarchx.simulation.calibration_bias import (
    CAC_NGUON,
    build_calibration_scenarios,
    ma_kich_ban,
)


def _fake_params(p11: float, p22: float) -> MSGARCHXParams:
    return MSGARCHXParams(
        mu=torch.tensor([0.0, 0.3]),
        omega=torch.tensor([0.1, 1.0]),
        alpha=torch.tensor([0.1, 0.3]),
        beta=torch.tensor([0.8, 0.3]),
        gamma=torch.tensor([[-0.02, 0.05], [-0.17, 0.62]]),
        P=torch.tensor([[p11, 1 - p11], [1 - p22, p22]]),
        nu=torch.tensor(4.08),
    )


def test_build_calibration_scenarios_ba_nguon():
    theta = {
        "mle": _fake_params(0.9986, 0.9932),
        "vb_mean_field": _fake_params(0.7182, 0.9346),
        "vb_full_rank": _fake_params(0.8675, 0.9571),
    }
    scenarios = build_calibration_scenarios(theta, T=2000, B=100)

    assert set(scenarios) == {ma_kich_ban(n) for n in CAC_NGUON}
    for nguon, params in theta.items():
        scn = scenarios[ma_kich_ban(nguon)]
        assert scn.T == 2000 and scn.B == 100
        assert scn.misspecified is False
        # P0 dung de sinh du lieu phai LA nghiem that (params.P), khong phai
        # placeholder -- khac voi quy uoc cua S1-S8 (xem test_simulation.py).
        assert torch.allclose(scn.P0, params.P)
        assert torch.allclose(scn.params.omega, params.omega)


def test_build_calibration_scenarios_tu_choi_nguon_la():
    with pytest.raises(ValueError):
        build_calibration_scenarios({"khong_hop_le": _fake_params(0.9, 0.9)})


def test_ma_kich_ban_duy_nhat_va_on_dinh():
    ma = {ma_kich_ban(n) for n in CAC_NGUON}
    assert len(ma) == len(CAC_NGUON)
    assert ma_kich_ban("mle") == "CAL_mle"
