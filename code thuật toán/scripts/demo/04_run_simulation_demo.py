"""Demo nghien cuu mo phong (Muc 6): chay kich ban S1 voi so lan lap nho de
minh hoa quy trinh Bias/RMSE/Coverage/Brier. Ban thao chinh thuc can B lon
hon nhieu (vd. 200+, xem scenarios.py) va nen chay ngoai tuyen/nhieu tien
trinh do chi phi MLE/VB/MCMC tren nhieu lan lap.

Chay: python scripts/demo/04_run_simulation_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from msgarchx.inference.param_spec import ModelConfig
from msgarchx.simulation.run_simulation import run_scenario
from msgarchx.simulation.scenarios import default_scenarios


def main():
    scenarios = default_scenarios(T_default=150, B_default=2)
    scenario = scenarios["S1"]
    config = ModelConfig(n_exog=2, state_specific_nu=False)

    print(f"Kich ban {scenario.code}: {scenario.description}")
    print(f"T={scenario.T}, B={scenario.B} (GIA TRI DEMO nhanh)")

    t0 = time.time()
    method_kwargs = {"mle": {"n_starts": 1, "n_adam": 40, "n_lbfgs": 15}}
    results = run_scenario(scenario, config, methods=("mle",), base_seed=0, method_kwargs=method_kwargs)
    print(f"(thoi gian: {time.time()-t0:.1f}s)")

    mle_res = results["mle"]
    print("\n== Bias / RMSE tren tham so (MLE, eq:bias, eq:parameter-rmse) ==")
    for name in ["omega1", "omega2", "alpha1", "alpha2", "beta1", "beta2", "p11", "p22", "nu"]:
        if name in mle_res["bias"]:
            print(f"  {name:<10} bias={mle_res['bias'][name]:+.4f}   rmse={mle_res['rmse'][name]:.4f}")
    print(f"\nBrier score trung binh (xac suat trang thai, eq:brier-state): {mle_res['mean_brier']:.4f}")


if __name__ == "__main__":
    main()
