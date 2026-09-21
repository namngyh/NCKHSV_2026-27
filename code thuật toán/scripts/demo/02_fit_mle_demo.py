"""Demo: uoc luong MLE cho GARCH(1,1) chuan (B1), GARCH(1,1) Student-t (B2)
va MS-GARCH-X Student-t (M1) tren du lieu VN30F1M that (Bang 5.6).

Bien ngoai sinh hien co tu ohlc_export.csv: log(1+VOL_{t-1}), log(RV_{t-1}+eps).
BASIS/EXP/ROLL can them du lieu chi so VN30 co so va lich dao han hop dong
(chua co trong file nay) nen chua duoc dua vao o ban demo nay.

Chay: python scripts/demo/02_fit_mle_demo.py [duong_dan_csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from msgarchx.data.exogenous import build_exogenous_matrix, standardize_with_train_stats
from msgarchx.data.loader import load_vn30f1m_daily
from msgarchx.inference.mle import fit_mle
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.models.garch import GARCHParams, garch_loglik
from msgarchx.models.ms_garch import expected_duration, label_switch_correction


def prepare_data(csv_path: str):
    daily, _ = load_vn30f1m_daily(csv_path)
    X_raw = build_exogenous_matrix(
        daily, volume_col="volume", rv_col="RV",
        basis_col=None, expiry_dummy_col=None, roll_dummy_col=None,
    )
    train_mask = pd.Series(False, index=X_raw.index)
    train_mask.iloc[: int(len(daily) * 0.8)] = True
    X_scaled, mean, std = standardize_with_train_stats(X_raw, train_mask, exclude_cols=("date",))

    valid = daily["log_return"].notna() & X_scaled["log_volume_lag1"].notna() & X_scaled["log_rv_lag1"].notna()
    r = torch.tensor(daily.loc[valid, "log_return"].to_numpy())
    X = torch.tensor(X_scaled.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy())
    return r, X, daily.loc[valid, "date"].reset_index(drop=True)


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=str(Path(__file__).resolve().parents[2] / "ohlc_export.csv"))
    parser.add_argument("--n-starts", type=int, default=2, help="So diem khoi tao MLE cho M1 (mac dinh nho de chay nhanh; tang len 4-8 cho ket qua bao cao chinh thuc)")
    parser.add_argument("--n-adam", type=int, default=150)
    parser.add_argument("--n-lbfgs", type=int, default=60)
    args = parser.parse_args()

    r, X, dates = prepare_data(args.csv_path)
    print(f"So quan sat dung cho uoc luong: {len(r)} ({dates.iloc[0].date()} -> {dates.iloc[-1].date()})")
    print(
        f"[Luu y hieu nang] Bo loc tien MS-GARCH-X la vong lap tuan tu theo T "
        f"(khong vector hoa); voi T={len(r)}, moi lan danh gia hop ly ton ~0.5-1s "
        f"(CPU). Cau hinh mac dinh cua demo nay (n_starts={args.n_starts}, "
        f"n_adam={args.n_adam}, n_lbfgs={args.n_lbfgs}) uu tien toc do; tang cac "
        f"gia tri nay (vd. n_starts=8, n_adam=500+) khi chay uoc luong chinh thuc "
        f"ngoai tuyen cho ban thao."
    )

    h0 = torch.var(r)

    print("\n== B1: GARCH(1,1) chuan (MLE) ==")
    t0 = time.time()
    b1 = _fit_garch_baseline(r, X, dist="normal", h0=h0, n_starts=args.n_starts)
    _report_garch(b1, k=4)
    print(f"(thoi gian: {time.time()-t0:.1f}s)")

    print("\n== B2: GARCH(1,1) Student-t (MLE) ==")
    t0 = time.time()
    b2 = _fit_garch_baseline(r, X, dist="student_t", h0=h0, n_starts=args.n_starts)
    _report_garch(b2, k=5)
    print(f"(thoi gian: {time.time()-t0:.1f}s)")

    print("\n== M1: MS-GARCH-X Student-t (MLE) ==")
    t0 = time.time()
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    res = fit_mle(r, X, config, n_starts=args.n_starts, base_seed=0, n_adam=args.n_adam, n_lbfgs=args.n_lbfgs, verbose=True)
    print(f"(thoi gian: {time.time()-t0:.1f}s)")
    params = label_switch_correction(res.params, res.filter_output["h"])
    print(f"loglik={res.loglik:.2f}  AIC={res.aic:.2f}  BIC={res.bic:.2f}")
    print(f"mu={params.mu.tolist()}")
    print(f"omega={params.omega.tolist()}")
    print(f"alpha={params.alpha.tolist()}")
    print(f"beta={params.beta.tolist()}")
    print(f"gamma=\n{params.gamma}")
    print(f"P=\n{params.P}")
    print(f"nu={params.nu.item():.3f}")
    d = expected_duration(params.P)
    print(f"Thoi gian ton tai ky vong: trang thai 1 (bien dong thap)={d[0]:.1f} ngay, trang thai 2 (bien dong cao)={d[1]:.1f} ngay")
    print("Chan doan cac lan khoi tao MLE:")
    for run in res.runs:
        print(f"  seed={run.seed} loglik={run.loglik:.2f} converged={run.converged} grad_norm={run.grad_norm:.4f} near_boundary={run.near_boundary}")


def _fit_garch_baseline(r, X, dist, h0, n_starts=3):
    best = None
    for seed in range(n_starts):
        torch.manual_seed(seed)
        raw = {
            "mu": torch.zeros(()), "log_omega": torch.log(h0 * 0.1),
            "eta_alpha": torch.tensor(-1.5), "eta_beta": torch.tensor(1.0),
            "gamma": torch.zeros(X.shape[1]) + 0.01 * torch.randn(X.shape[1]),
            "eta_nu": torch.tensor(1.5),
        }
        vars_ = {k: v.clone().requires_grad_(True) for k, v in raw.items()}
        opt = torch.optim.Adam(list(vars_.values()), lr=0.02)
        for _ in range(250):
            opt.zero_grad()
            params = _unpack_garch(vars_, dist)
            loss = -garch_loglik(params, r, X, h0=h0)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vars_.values()), 50.0)
            opt.step()
        with torch.no_grad():
            params = _unpack_garch(vars_, dist)
            ll = garch_loglik(params, r, X, h0=h0).item()
        if best is None or ll > best[0]:
            best = (ll, params)
    return best


def _unpack_garch(vars_, dist):
    nu = 2.0 + torch.nn.functional.softplus(vars_["eta_nu"]) if dist == "student_t" else None
    return GARCHParams(
        mu=vars_["mu"],
        omega=torch.exp(vars_["log_omega"]),
        alpha=torch.nn.functional.softplus(vars_["eta_alpha"]),
        beta=torch.nn.functional.softplus(vars_["eta_beta"]),
        gamma=vars_["gamma"],
        nu=nu,
    )


def _report_garch(result, k):
    ll, params = result
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * torch.log(torch.tensor(2000.0)).item()
    print(f"loglik={ll:.2f}  AIC={aic:.2f}  BIC={bic:.2f}")
    print(f"mu={params.mu.item():.4f} omega={params.omega.item():.4f} alpha={params.alpha.item():.4f} beta={params.beta.item():.4f}")
    print(f"gamma={params.gamma.tolist()}")
    if params.nu is not None:
        print(f"nu={params.nu.item():.3f}")


if __name__ == "__main__":
    main()
