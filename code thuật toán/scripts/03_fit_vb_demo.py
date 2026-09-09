"""Demo: Variational Bayes (mean-field va full-rank) cho
MS-GARCH-X Student-t tren du lieu VN30F1M that (Phan thu hai cua bai bao).

Minh hoa quy trinh uoc luong xap xi hau nghiem bang Stochastic Variational Inference (SVI)
su dung Pyro. Mac dinh dung mot cua so con (vd. 300 quan sat gan nhat) de minh hoa nhanh;
uoc luong chinh thuc tren toan bo 2.000 quan sat da duoc luu tai results/03_vb.pt.

Chay: python scripts/03_fit_vb_demo.py [duong_dan_csv] [so_quan_sat]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msgarchx.data.exogenous import build_exogenous_matrix, standardize_with_train_stats
from msgarchx.data.loader import load_vn30f1m_daily
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.inference.variational import fit_vb


def prepare_data(csv_path: str, window: int):
    daily, _ = load_vn30f1m_daily(csv_path)
    X_raw = build_exogenous_matrix(
        daily, volume_col="volume", rv_col="RV",
        basis_col=None, expiry_dummy_col=None, roll_dummy_col=None,
    )
    train_mask = pd.Series(True, index=X_raw.index)
    X_scaled, _, _ = standardize_with_train_stats(X_raw, train_mask, exclude_cols=("date",))

    valid = daily["log_return"].notna() & X_scaled["log_volume_lag1"].notna() & X_scaled["log_rv_lag1"].notna()
    r_full = torch.tensor(daily.loc[valid, "log_return"].to_numpy())
    X_full = torch.tensor(X_scaled.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy())
    return r_full[-window:], X_full[-window:]


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "ohlc_export.csv")
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    r, X = prepare_data(csv_path, window)
    print(f"Su dung {len(r)} quan sat gan nhat cho demo Variational Bayes.")
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)

    print("\n== 1. Variational Bayes Mean-Field (M2) ==")
    t0 = time.time()
    vb_mf = fit_vb(r, X, config, guide_type="mean_field", n_steps=800, n_posterior_draws=500, seed=0, verbose=True)
    t_mf = time.time() - t0
    print(f"ELBO cuoi (trung binh 50 buoc cuoi): {vb_mf.final_elbo:.2f}  (thoi gian: {t_mf:.1f}s)")
    print("Tham so hau nghiem M2 (Mean-Field):")
    for name in ["omega", "alpha", "beta", "p11", "p22", "nu"]:
        if name in vb_mf.summary:
            m = vb_mf.summary[name]["mean"].tolist()
            s = vb_mf.summary[name]["sd"].tolist()
            print(f"  {name:<6}: mean = {m} | sd = {s}")

    print("\n== 2. Variational Bayes Full-Rank (M3) ==")
    t0 = time.time()
    vb_fr = fit_vb(r, X, config, guide_type="full_rank", n_steps=800, n_posterior_draws=500, seed=0, verbose=True)
    t_fr = time.time() - t0
    print(f"ELBO cuoi (trung binh 50 buoc cuoi): {vb_fr.final_elbo:.2f}  (thoi gian: {t_fr:.1f}s)")
    print("Tham so hau nghiem M3 (Full-Rank):")
    for name in ["omega", "alpha", "beta", "p11", "p22", "nu"]:
        if name in vb_fr.summary:
            m = vb_fr.summary[name]["mean"].tolist()
            s = vb_fr.summary[name]["sd"].tolist()
            print(f"  {name:<6}: mean = {m} | sd = {s}")

    print("\n== So sanh ELBO va chat luong xap xi ==")
    print(f"ELBO Mean-Field : {vb_mf.final_elbo:.2f}")
    print(f"ELBO Full-Rank  : {vb_fr.final_elbo:.2f}")
    diff = vb_fr.final_elbo - vb_mf.final_elbo
    print(f"Chenh lech ELBO (Full-Rank - Mean-Field): {diff:+.2f} "
          f"({'Full-Rank xap xi chat hon' if diff > 0 else 'Tuong duong'})")


if __name__ == "__main__":
    main()
