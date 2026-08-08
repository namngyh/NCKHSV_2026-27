"""Demo: Variational Bayes (mean-field, full-rank) va MCMC (NUTS) cho
MS-GARCH-X Student-t tren du lieu VN30F1M that (Muc 4.4, 4.5).

CANH BAO HIEU NANG: bo loc tien la vong lap Python tuan tu theo T nen MCMC
(hang nghin lan danh gia gradient) co the mat nhieu thoi gian voi T lon. Demo
nay mac dinh chi dung MOT CUA SO con (vd. 300 quan sat gan nhat) de minh hoa
quy trinh trong thoi gian hop ly; ban thao chinh thuc can chay tren toan bo
mau (hoac cua so huan luyen day du) o che do ngoai tuyen, chap nhan thoi gian
chay dai hon nhieu (xem ghi chu trong scripts/02_fit_mle_demo.py).

Chay: python scripts/03_fit_vb_mcmc_demo.py [duong_dan_csv]
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
from msgarchx.inference.mcmc import fit_mcmc
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.inference.summary import sd_ratio
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
    print(f"Dung {len(r)} quan sat gan nhat cho demo VB/MCMC (T lon hon se cham hon dang ke).")
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)

    print("\n== Variational Bayes (mean-field) ==")
    t0 = time.time()
    vb_mf = fit_vb(r, X, config, guide_type="mean_field", n_steps=800, n_posterior_draws=500, seed=0, verbose=True)
    print(f"ELBO cuoi (trung binh 50 buoc cuoi): {vb_mf.final_elbo:.2f}  (thoi gian: {time.time()-t0:.1f}s)")

    print("\n== Variational Bayes (full-rank) ==")
    t0 = time.time()
    vb_fr = fit_vb(r, X, config, guide_type="full_rank", n_steps=800, n_posterior_draws=500, seed=0, verbose=True)
    print(f"ELBO cuoi (trung binh 50 buoc cuoi): {vb_fr.final_elbo:.2f}  (thoi gian: {time.time()-t0:.1f}s)")

    print("\n== MCMC (NUTS, 4 chuoi) ==")
    t0 = time.time()
    mcmc_res = fit_mcmc(r, X, config, n_chains=4, n_warmup=200, n_samples=300, seed=0, max_tree_depth=7, verbose=True)
    print(f"(thoi gian: {time.time()-t0:.1f}s)")
    print(f"R-hat lon nhat: {mcmc_res.max_rhat:.4f}  |  ESS nho nhat: {mcmc_res.min_ess:.1f}")
    for name in ["omega", "alpha", "beta", "p11", "p22"]:
        print(f"  {name}: mean={mcmc_res.summary[name]['mean'].tolist()}  sd={mcmc_res.summary[name]['sd'].tolist()}")

    print("\n== So sanh do lech chuan hau nghiem: VB / MCMC (eq:sd-ratio) ==")
    for label, vb_res in [("mean-field", vb_mf), ("full-rank", vb_fr)]:
        ratios = sd_ratio(vb_res.summary, mcmc_res.summary)
        avg_ratio = sum(float(v.mean()) for v in ratios.values()) / len(ratios)
        print(f"  {label}: ty le do lech chuan trung binh (VB/MCMC) = {avg_ratio:.3f} "
              f"({'danh gia thap' if avg_ratio < 1 else 'danh gia cao'} phuong sai hau nghiem so voi MCMC)")


if __name__ == "__main__":
    main()
