"""Test doc du lieu, chuan bi bien ngoai sinh, kiem dinh ban dau va nghien cuu mo phong."""
import numpy as np
import pandas as pd
import torch

from garch_vn30.du_lieu.chuan_bi import build_exogenous_matrix, standardize_with_train_stats
from garch_vn30.du_lieu.doc_du_lieu import (build_daily_bars, compute_daily_log_returns,
                                            compute_intraday_log_returns, data_quality_report)
from garch_vn30.kiem_dinh.kiem_dinh_so_bo import arch_lm_test, run_all_preliminary_tests
from garch_vn30.mo_hinh import danh_muc
from garch_vn30.mo_phong import (chay_mot_lan_lap, simulate_egarch, tham_so_that, tong_hop,
                                 truong_hop_mac_dinh)


def _du_lieu_phut(n_ngay=3, n_nen=5, seed=0):
    rng = np.random.default_rng(seed)
    dong, gia = [], 1000.0
    for d in range(n_ngay):
        ngay = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        for b in range(n_nen):
            gia *= float(np.exp(rng.normal(0, 0.001)))
            dong.append({"date": ngay, "datetime": ngay + pd.Timedelta(minutes=b), "SYMBOL": "VN30F1M",
                         "open": gia, "high": gia * 1.001, "low": gia * 0.999, "close": gia,
                         "volume": 10 + b})
    return pd.DataFrame(dong)


# ------------------------------------------------------------- du lieu
def test_bao_cao_chat_luong_du_lieu_sach():
    bc = data_quality_report(_du_lieu_phut())
    assert bc["n_duplicate_datetime"] == bc["n_null"] == bc["n_ohlc_violation"] == 0
    assert bc["n_trading_days"] == 3


def test_gop_ngay_va_loi_suat():
    ngay = compute_daily_log_returns(build_daily_bars(_du_lieu_phut(4, 6)))
    assert len(ngay) == 4 and (ngay["n_bars"] == 6).all()
    assert ngay["log_return"].isna().sum() == 1


def test_loi_suat_noi_ngay_khong_noi_qua_dem():
    out = compute_intraday_log_returns(_du_lieu_phut(2, 4))
    dau_ngay = out.groupby("date").cumcount() == 0
    assert out.loc[dau_ngay, "intraday_log_return"].isna().all()
    assert out.loc[~dau_ngay, "intraday_log_return"].notna().all()


def test_bien_ngoai_sinh_dich_tre_va_chuan_hoa_chi_bang_tap_huan_luyen():
    daily = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10),
                          "volume": np.arange(1, 11) * 100.0, "RV": np.linspace(1, 2, 10)})
    X = build_exogenous_matrix(daily)
    assert np.isnan(X["log_volume_lag1"].iloc[0])
    assert X["log_volume_lag1"].iloc[1] == np.log1p(100.0)       # gia tri cua ngay TRUOC
    mask = pd.Series([True] * 6 + [False] * 4)
    Xs, mean, _ = standardize_with_train_stats(X, mask)
    assert abs(Xs.loc[mask, "log_rv_lag1"].mean()) < 1e-12        # trung binh 0 tren tap huan luyen
    assert abs(mean["log_rv_lag1"] - X.loc[mask, "log_rv_lag1"].mean()) < 1e-15


def test_kiem_dinh_ban_dau():
    rng = np.random.default_rng(0)
    out = run_all_preliminary_tests(rng.normal(size=500))
    assert set(out) == {"adf_r", "pp_r", "kpss_r", "ljung_box_r", "ljung_box_r2", "arch_lm", "jarque_bera"}
    assert out["adf_r"]["p_value"] < 0.05
    h, e = np.ones(2000), np.zeros(2000)
    for t in range(1, 2000):
        h[t] = 0.02 + 0.15 * e[t - 1] ** 2 + 0.8 * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.normal()
    assert arch_lm_test(e)["p_value"] < 0.01


# ------------------------------------------------------------- mo phong
def test_ba_truong_hop_dung_thiet_ke():
    th = truong_hop_mac_dinh()
    assert set(th) == {"E1", "E2", "E3"}
    assert float(th["E2"].params.theta) == 0.0          # kiem tra don bay gia
    assert float(th["E1"].params.theta) < 0
    assert float(th["E3"].params.nu) < float(th["E1"].params.nu)


def test_sinh_du_lieu_tai_lap_va_hop_le():
    p = truong_hop_mac_dinh()["E1"].params
    a = simulate_egarch(p, T=300, burn_in=50, seed=7)
    b = simulate_egarch(p, T=300, burn_in=50, seed=7)
    assert a.r.shape == (300,) and a.X.shape == (300, 2) and (a.h > 0).all()
    assert torch.equal(a.r, b.r)


def test_lan_lap_va_tong_hop_bias_rmse_coverage():
    th = truong_hop_mac_dinh(T=150, B=2)["E2"]
    spec, that = danh_muc.get("egarch_x_t"), tham_so_that(th.params)
    lan_lap = {"mle": [], "vb_mean_field": []}
    for b in range(2):
        sim = simulate_egarch(th.params, T=150, burn_in=50, seed=b)
        lan_lap["mle"].append(chay_mot_lan_lap("mle", sim.r, sim.X, spec, b,
                                               {"n_starts": 1, "n_adam": 15, "n_lbfgs": 5}))
        lan_lap["vb_mean_field"].append(chay_mot_lan_lap("vb_mean_field", sim.r, sim.X, spec, b,
                                                         {"n_steps": 15, "n_posterior_draws": 10}))
    for ds in lan_lap.values():
        kq = tong_hop(ds, that)
        assert kq["n_replications"] == 2 and set(kq["bias"]) == set(that)
        sai = np.array([x["uoc_luong"]["theta"] for x in ds]) - that["theta"]
        assert abs(kq["rmse"]["theta"] - np.sqrt(np.mean(sai ** 2))) < 1e-12
    assert "theta" in tong_hop(lan_lap["vb_mean_field"], that)["coverage"]


def test_tong_hop_bo_qua_lan_lap_khong_co_khoang():
    that = {"theta": 0.0}
    ds = [{"uoc_luong": {"theta": 0.1}, "can_duoi": {"theta": -0.1}, "can_tren": {"theta": 0.2}},
          {"uoc_luong": {"theta": 0.3}, "can_duoi": None, "can_tren": None}]
    kq = tong_hop(ds, that)
    assert kq["n_co_khoang"] == 1 and kq["coverage"]["theta"] == 1.0
    assert abs(kq["bias"]["theta"] - 0.2) < 1e-12
