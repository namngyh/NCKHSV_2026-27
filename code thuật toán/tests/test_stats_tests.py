import numpy as np

from msgarchx.stats_tests.preliminary_tests import (
    adf_test,
    arch_lm_test,
    jarque_bera_test,
    kpss_test,
    ljung_box_test,
    run_all_preliminary_tests,
)


def test_adf_rejects_unit_root_for_stationary_series():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    res = adf_test(x)
    assert res["p_value"] < 0.05


def test_kpss_runs_and_returns_dict():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    res = kpss_test(x)
    assert "p_value" in res


def test_arch_lm_detects_garch_effect():
    rng = np.random.default_rng(0)
    n = 2000
    h = np.ones(n)
    eps = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.02 + 0.15 * eps[t - 1] ** 2 + 0.8 * h[t - 1]
        eps[t] = np.sqrt(h[t]) * rng.normal()
    res = arch_lm_test(eps, nlags=5)
    assert res["p_value"] < 0.01


def test_jarque_bera_rejects_normality_for_heavy_tailed():
    rng = np.random.default_rng(0)
    x = rng.standard_t(df=3, size=2000)
    res = jarque_bera_test(x)
    assert res["p_value"] < 0.01


def test_run_all_preliminary_tests_returns_all_keys():
    rng = np.random.default_rng(0)
    r = rng.normal(size=500)
    out = run_all_preliminary_tests(r)
    assert set(out.keys()) == {
        "adf_r", "pp_r", "kpss_r", "ljung_box_r", "ljung_box_r2", "arch_lm", "jarque_bera",
    }
