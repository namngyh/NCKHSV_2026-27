"""Dieu phoi TOAN BO qua trinh uoc luong chinh thuc cho bai bao MS-GARCH-X.

Trinh tu (moi buoc duoc CHECKPOINT vao results/ -- neu file da ton tai, buoc
do duoc BO QUA va nap lai ket qua cu, nen script co the dung/chay lai bat cu
luc nao ma khong mat tien do):

  0. Nap du lieu VN30F1M that (ohlc_export.csv), kiem dinh thong ke ban dau.
  1. MLE: B1 (GARCH chuan), B2 (GARCH Student-t), B3 (GARCH-X Student-t),
     B4 (MS-GARCH chuan, nu co dinh lon), M1 (MS-GARCH-X Student-t).
  2. Variational Bayes: M2 (mean-field), M3 (full-rank).
  3. MCMC: M4 (NUTS, nhieu chuoi).
  4. R1: MS-GARCH-X voi nu_s theo trang thai (VB mean-field) + xac suat hau
     nghiem P(nu_2 < nu_1 | Data) (eq:tail-probability).
  5. Danh gia ngoai mau (QLIKE/MAE/RMSE, diem log, VaR Kupiec/Christoffersen)
     cho tat ca cac mo hinh o buoc 1-4, dung THAM SO CO DINH tu tap huan
     luyen + chi cap nhat bo loc tren tap kiem dinh (nhu da chot voi nguoi
     dung).
  6. Nghien cuu mo phong 8 kich ban (S1-S8): MLE (B lon), VB (B vua), MCMC
     (B nho, chi mang tinh doi chieu xu huong).

CANH BAO THOI GIAN: cau hinh mac dinh (--preset full) co the mat 1.5-2 NGAY
tren laptop thong thuong (xem README.md, phan "Luu y hieu nang"). Dung
`--preset dry-run` de kiem tra toan bo pipeline chay dung trong vai phut
truoc khi chay that.

Chay: python scripts/run_full_study.py --preset full
      python scripts/run_full_study.py --preset dry-run
      python scripts/run_full_study.py --preset full --only simulation
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msgarchx.data.exogenous import build_exogenous_matrix, standardize_with_train_stats
from msgarchx.data.loader import load_vn30f1m_daily
from msgarchx.evaluation.density_scores import mixture_quantile, negative_log_score
from msgarchx.evaluation.var_backtests import christoffersen_test, kupiec_test
from msgarchx.evaluation.volatility_loss import mae_vol, qlike, rmse_vol
from msgarchx.inference.forecast import predictive_variance_path
from msgarchx.inference.garch_mle import fit_garch_mle
from msgarchx.inference.mcmc import fit_mcmc, initial_params_from_mle
from msgarchx.inference.mle import fit_mle, natural_scale_standard_errors
from msgarchx.inference.param_spec import ModelConfig, warm_starts_from_garch
from msgarchx.inference.summary import relabel_posterior_samples
from msgarchx.inference.variational import fit_vb
from msgarchx.models.garch import garch_forward
from msgarchx.models.ms_garch import ms_garch_forward_filter
from msgarchx.simulation.run_simulation import run_scenario
from msgarchx.simulation.scenarios import default_scenarios
from msgarchx.stats_tests.preliminary_tests import run_all_preliminary_tests


# ============================================================
# Cau hinh
# ============================================================
@dataclass
class StudyConfig:
    csv_path: str
    results_dir: Path
    train_frac: float = 0.8
    seed: int = 0

    mle_n_starts: int = 10
    mle_n_adam: int = 500
    mle_n_lbfgs: int = 250
    mle_n_cycles: int = 3      # so chu ky Adam+L-BFGS moi diem khoi tao

    vb_n_steps: int = 3000
    vb_n_draws: int = 1000

    mcmc_n_chains: int = 4
    # warmup 250 la QUA IT de thich nghi ma tran khoi luong DAC; Stan mac dinh
    # 1000. Lan chay dau (250 vong, ma tran duong cheo) cho R-hat ~ 4, ESS ~ 4.4.
    mcmc_n_warmup: int = 1000
    mcmc_n_samples: int = 400
    mcmc_max_tree_depth: int = 7
    mcmc_full_mass: bool = True

    oos_n_posterior_draws: int = 200

    sim_T: int = 2000
    sim_burn_in: int = 200
    sim_B_mle: int = 30
    sim_B_vb: int = 15
    sim_B_mcmc: int = 3
    sim_n_exog: int = 2
    sim_mle_n_starts: int = 3
    sim_mle_n_adam: int = 200
    sim_mle_n_lbfgs: int = 100
    sim_vb_n_steps: int = 800
    # fit_vb mac dinh lay 2000 mau hau nghiem; trong mo phong ta chi can
    # trung binh/do lech chuan/khoang 2.5-97.5% de tinh bias, RMSE va coverage,
    # 500 mau la du va tiet kiem hon mot nua chi phi cua buoc nay.
    sim_vb_n_draws: int = 500
    sim_mcmc_n_chains: int = 4
    sim_mcmc_n_warmup: int = 100
    sim_mcmc_n_samples: int = 150
    sim_mcmc_max_tree_depth: int = 6


PRESETS = {
    "full": StudyConfig(csv_path=str(ROOT / "ohlc_export.csv"), results_dir=ROOT / "results"),
    "dry-run": StudyConfig(
        csv_path=str(ROOT / "ohlc_export.csv"),
        results_dir=ROOT / "results_dryrun",
        mle_n_starts=1, mle_n_adam=20, mle_n_lbfgs=10, mle_n_cycles=1,
        vb_n_steps=20, vb_n_draws=20,
        mcmc_n_chains=1, mcmc_n_warmup=5, mcmc_n_samples=5, mcmc_max_tree_depth=4,
        mcmc_full_mass=True,
        oos_n_posterior_draws=5,
        sim_T=80, sim_burn_in=20, sim_B_mle=2, sim_B_vb=2, sim_B_mcmc=1,
        sim_mle_n_starts=1, sim_mle_n_adam=15, sim_mle_n_lbfgs=5,
        sim_vb_n_steps=10, sim_mcmc_n_chains=1, sim_mcmc_n_warmup=3, sim_mcmc_n_samples=3,
        sim_mcmc_max_tree_depth=3,
    ),
}


# ============================================================
# Checkpoint tien ich
# ============================================================
def _record_timing(cfg: StudyConfig, name: str, elapsed_seconds: float) -> None:
    """Ghi thoi gian chay cua moi buoc vao mot file rieng (09_timing.pt),
    dung cho Bang hieu qua tinh toan trong bao cao (khong lam thay doi cau
    truc ket qua tra ve cua tung buoc)."""
    timing_path = cfg.results_dir / "09_timing.pt"
    timing = torch.load(timing_path, weights_only=False) if timing_path.exists() else {}
    timing[name] = elapsed_seconds
    torch.save(timing, timing_path)


def checkpoint(cfg: StudyConfig, name: str, compute_fn):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.results_dir / f"{name}.pt"
    if path.exists():
        print(f"[bo qua] {name} da co san tai {path}")
        return torch.load(path, weights_only=False)
    print(f"[chay] {name} ...")
    t0 = time.time()
    result = compute_fn()
    elapsed = time.time() - t0
    torch.save(result, path)
    _record_timing(cfg, name, elapsed)
    print(f"[xong] {name} ({elapsed:.1f}s) -> {path}")
    return result


def safe_checkpoint(cfg: StudyConfig, name: str, compute_fn):
    """Nhu checkpoint, nhung bat loi va ghi log thay vi lam sap toan bo pipeline."""
    try:
        return checkpoint(cfg, name, compute_fn)
    except Exception:
        print(f"[LOI] Buoc '{name}' that bai, bo qua va tiep tuc pipeline:")
        traceback.print_exc()
        return None


# ============================================================
# Buoc 0: Du lieu va kiem dinh so bo
# ============================================================
def prepare_data(cfg: StudyConfig) -> dict:
    daily, quality_report = load_vn30f1m_daily(cfg.csv_path)
    X_raw = build_exogenous_matrix(
        daily, volume_col="volume", rv_col="RV",
        basis_col=None, expiry_dummy_col=None, roll_dummy_col=None,
    )
    n = len(daily)
    train_mask = pd.Series(False, index=X_raw.index)
    train_mask.iloc[: int(n * cfg.train_frac)] = True
    X_scaled, x_mean, x_std = standardize_with_train_stats(X_raw, train_mask, exclude_cols=("date",))

    valid = (
        daily["log_return"].notna()
        & X_scaled["log_volume_lag1"].notna()
        & X_scaled["log_rv_lag1"].notna()
        & daily["RV"].notna()
    )
    r_all = torch.tensor(daily.loc[valid, "log_return"].to_numpy())
    X_all = torch.tensor(X_scaled.loc[valid, ["log_volume_lag1", "log_rv_lag1"]].to_numpy())
    rv_all = torch.tensor(daily.loc[valid, "RV"].to_numpy())
    dates_all = daily.loc[valid, "date"].reset_index(drop=True)
    n_train = int(train_mask.loc[valid].sum())

    return {
        "r_all": r_all, "X_all": X_all, "rv_all": rv_all, "dates_all": dates_all,
        "n_train": n_train, "quality_report": quality_report,
        "exog_mean": x_mean, "exog_std": x_std,
    }


def run_preliminary_tests_step(cfg: StudyConfig, data: dict) -> dict:
    r_np = data["r_all"].numpy()
    return run_all_preliminary_tests(r_np)


# ============================================================
# Buoc 1: MLE (B1-B4, M1)
# ============================================================
def run_mle_step(cfg: StudyConfig, r_train: torch.Tensor, X_train: torch.Tensor) -> dict:
    """`verbose=True` o day CO CHU DICH: voi preset "full" (6 diem khoi tao x
    500 Adam x 250 L-BFGS, vong lap Python qua ~1735 quan sat cho B4/M1), moi
    model co the chay lieu vai chuc phut MA KHONG in gi neu tat verbose --
    de nguoi dung tuong cua so bi treo va dong nham giua chung. Bat verbose
    de in tien do moi ~100-200 buoc/diem khoi tao, lam "nhip tim" hien thi
    lien tuc (xem README.md / cai_dat_va_chay_full.bat)."""
    zero_X = torch.zeros(r_train.shape[0], 0)
    kw = dict(n_starts=cfg.mle_n_starts, n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs,
              base_seed=cfg.seed, verbose=True)

    print("  -- B1: GARCH(1,1) chuan, khong X --")
    b1 = fit_garch_mle(r_train, zero_X, dist="normal", **kw)
    print(f"     loglik={b1.loglik:.2f} AIC={b1.aic:.2f}")

    print("  -- B2: GARCH(1,1) Student-t, khong X --")
    b2 = fit_garch_mle(r_train, zero_X, dist="student_t", **kw)
    print(f"     loglik={b2.loglik:.2f} AIC={b2.aic:.2f} nu={b2.params.nu.item():.3f}")

    print("  -- B3: GARCH-X Student-t --")
    b3 = fit_garch_mle(r_train, X_train, dist="student_t", **kw)
    print(f"     loglik={b3.loglik:.2f} AIC={b3.aic:.2f} nu={b3.params.nu.item():.3f}")

    # Hai mo hinh 2 trang thai duoi day dung THEM ba thu so voi mac dinh cua
    # fit_mle, deu bam theo Muc 4.2 cua ban thao:
    #   (a) khoi tao AM tu nghiem mot trang thai (B1 cho B4, B3 cho M1) --
    #       "khoi tao tuan tu theo thu bac mo hinh";
    #   (b) nhieu chu ky Adam+L-BFGS, dung som khi ||grad|| < 1;
    #   (c) chon CUC DAI DIA PHUONG LON NHAT THOA DIEU KIEN CHINH QUY
    #       (select="regular") thay vi log-hop-ly cao nhat -- ham hop ly
    #       MS-GARCH co dinh gia tuong ung voi trang thai chi ton tai 1 phien.
    # Khong co (a)-(c), ca B4 lan M1 deu hoi tu ve nghiem suy biến.
    print("  -- B4: MS-GARCH chuan (nu co dinh lon), khong X --")
    config_b4 = ModelConfig(n_exog=0, state_specific_nu=False)
    b4 = fit_mle(r_train, zero_X, config_b4, n_starts=cfg.mle_n_starts, base_seed=cfg.seed,
                 n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs, fixed_nu=1e6, verbose=True,
                 init_raws=warm_starts_from_garch(b1.params, config_b4),
                 n_cycles=cfg.mle_n_cycles, select="regular")
    print(f"     loglik={b4.loglik:.2f} AIC={b4.aic:.2f}")

    print("  -- M1: MS-GARCH-X Student-t --")
    config_m1 = ModelConfig(n_exog=X_train.shape[1], state_specific_nu=False)
    m1 = fit_mle(r_train, X_train, config_m1, n_starts=cfg.mle_n_starts, base_seed=cfg.seed,
                 n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs, verbose=True,
                 init_raws=warm_starts_from_garch(b3.params, config_m1),
                 n_cycles=cfg.mle_n_cycles, select="regular")
    print(f"     loglik={m1.loglik:.2f} AIC={m1.aic:.2f} nu={m1.params.nu.item():.3f}")

    print("     Tinh sai so chuan (delta method tu Hessian) cho M1 ...")
    m1_se = natural_scale_standard_errors(m1.raw, config_m1, r_train, X_train)
    if m1_se is None:
        print("     Hessian khong xac dinh duong tai nghiem MLE cua M1 -- bo qua SE (Muc 4.2 buoc 6).")

    return {"b1": b1, "b2": b2, "b3": b3, "b4": b4, "m1": m1, "m1_se": m1_se, "config_m1": config_m1}


# ============================================================
# Buoc 2: Variational Bayes (M2, M3)
# ============================================================
def run_vb_step(cfg: StudyConfig, r_train: torch.Tensor, X_train: torch.Tensor, config_m1: ModelConfig) -> dict:
    print("  -- M2: MS-GARCH-X Student-t, VB mean-field --")
    m2 = fit_vb(r_train, X_train, config_m1, guide_type="mean_field",
                n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws, seed=cfg.seed, verbose=True)
    print(f"     ELBO cuoi = {m2.final_elbo:.2f}")

    print("  -- M3: MS-GARCH-X Student-t, VB full-rank --")
    m3 = fit_vb(r_train, X_train, config_m1, guide_type="full_rank",
                n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws, seed=cfg.seed, verbose=True)
    print(f"     ELBO cuoi = {m3.final_elbo:.2f}")

    return {"m2": m2, "m3": m3}


# ============================================================
# Buoc 3: MCMC (M4)
# ============================================================
def run_mcmc_step(cfg: StudyConfig, r_train: torch.Tensor, X_train: torch.Tensor, mle_res: dict) -> dict:
    """M4 bang NUTS.

    LUU Y TAI LAP: truoc day ham nay KHONG truyen `full_mass` (du StudyConfig co
    truong `mcmc_full_mass`) va KHONG truyen `initial_params`, nen chay lai M4 o
    day se tai lap dung cau hinh DA THAT BAI o lan 1 (ma tran khoi luong duong
    cheo + khoi tao tu tien nghiem: R-hat ~ 4, ESS ~ 4,4, cac chuoi dung yen),
    chu khong phai cau hinh da tao ra ket qua duoc bao cao. Nay ca hai deu duoc
    truyen, dung chung bo khoi tao voi scripts/chay_tiep_m1_mcmc_oos.py.
    """
    config_m1 = mle_res["config_m1"]
    m1 = mle_res.get("m1")

    if m1 is not None:
        # Moi chuoi mot diem khoi tao khac nhau: chuoi 0 dat dung tai nghiem MLE,
        # cac chuoi con lai nhieu loan quanh no. Truyen mot dict DUY NHAT se cho
        # ca 4 chuoi xuat phat cung mot cho, lam R-hat mat y nghia chan doan.
        init = lambda c: initial_params_from_mle(m1.params, c)
        mo_ta_init = "nghiem MLE cua M1 (chuoi 0 dung tai nghiem, cac chuoi khac nhieu loan)"
    else:
        init = None
        mo_ta_init = "mac dinh cua Pyro (init_to_uniform) -- KHONG co nghiem MLE de khoi tao"

    print("  -- M4: MS-GARCH-X Student-t, MCMC (NUTS) --")
    print(f"     n_chains={cfg.mcmc_n_chains}  n_warmup={cfg.mcmc_n_warmup}  "
          f"n_samples={cfg.mcmc_n_samples}  max_tree_depth={cfg.mcmc_max_tree_depth}  "
          f"full_mass={cfg.mcmc_full_mass}  target_accept_prob=0.85  seed={cfg.seed}")
    print(f"     khoi tao: {mo_ta_init}")
    m4 = fit_mcmc(r_train, X_train, config_m1, n_chains=cfg.mcmc_n_chains,
                  n_warmup=cfg.mcmc_n_warmup, n_samples=cfg.mcmc_n_samples, seed=cfg.seed,
                  max_tree_depth=cfg.mcmc_max_tree_depth, verbose=True,
                  full_mass=cfg.mcmc_full_mass, initial_params=init)
    print(f"     R-hat lon nhat={m4.max_rhat:.4f}  ESS nho nhat={m4.min_ess:.1f}  phan ky={m4.n_divergences}")
    return {"m4": m4}


# ============================================================
# Buoc 4: R1 (nu theo trang thai)
# ============================================================
def run_r1_step(cfg: StudyConfig, r_train: torch.Tensor, X_train: torch.Tensor) -> dict:
    print("  -- R1: MS-GARCH-X voi nu_s rieng theo trang thai, VB mean-field --")
    config_r1 = ModelConfig(n_exog=X_train.shape[1], state_specific_nu=True)
    r1 = fit_vb(r_train, X_train, config_r1, guide_type="mean_field",
                n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws, seed=cfg.seed, verbose=True)
    # PHAI sap xep nhan trang thai TRUOC khi tinh eq:tail-probability. Mau hau
    # nghiem khong duoc tu dong sap xep, ma dai luong nay duoc dinh nghia theo
    # quy uoc trang thai 2 = bien dong CAO (eq:label-identification). Tinh
    # truc tiep tren mau tho co the cho ket qua NGUOC HAN.
    samples_da_sap, ty_le_hoan_vi = relabel_posterior_samples(r1.samples)
    nu_samples = samples_da_sap["nu_minus_2"] + 2.0
    p_tail = float((nu_samples[:, 1] < nu_samples[:, 0]).float().mean())
    p_tail_tho = float(((r1.samples["nu_minus_2"] + 2.0)[:, 1]
                        < (r1.samples["nu_minus_2"] + 2.0)[:, 0]).float().mean())
    print(f"     P(nu_2 < nu_1 | Data) ~= {p_tail:.4f} (eq:tail-probability, "
          f"sau khi sap xep nhan; tren mau tho la {p_tail_tho:.4f})")
    print(f"     ty le mau bi hoan vi nhan: {ty_le_hoan_vi:.3f}"
          + ("  <-- gan 0.5: hau nghiem dang doi nhan qua lai, can bao cao ro"
             if 0.2 < ty_le_hoan_vi < 0.8 else ""))
    return {"r1": r1, "config_r1": config_r1, "p_tail": p_tail,
            "p_tail_mau_tho": p_tail_tho, "ty_le_hoan_vi_nhan": ty_le_hoan_vi,
            "samples_da_sap_xep": samples_da_sap}


# ============================================================
# Buoc 5: Danh gia ngoai mau
# ============================================================
def _var_hits_and_backtests(r_test: torch.Tensor, var_1pct: torch.Tensor, var_5pct: torch.Tensor) -> dict:
    import numpy as np
    out = {}
    for alpha, var_series, label in [(0.01, var_1pct, "1pct"), (0.05, var_5pct, "5pct")]:
        hits = (r_test.numpy() < -var_series.numpy()).astype(float)
        out[f"kupiec_{label}"] = kupiec_test(hits, alpha)
        out[f"christoffersen_{label}"] = christoffersen_test(hits, alpha)
        out[f"n_violations_{label}"] = int(hits.sum())
    return out


def _garch_var_quantile(alpha: float, dist: str, mu: float, nu: float | None) -> float:
    """Phan vi chuan hoa (phuong sai=1) cua z tai muc alpha, dung cho VaR mot trang thai."""
    from scipy import stats as sstats

    if dist == "normal":
        return sstats.norm.ppf(alpha)
    return sstats.t.ppf(alpha, df=nu) * ((nu - 2.0) / nu) ** 0.5


def _eval_garch_point(params, dist, r_all, X_all, rv_all, n_train) -> dict:
    from msgarchx.distributions import normal_logpdf, std_t_logpdf

    h0 = torch.var(r_all[:n_train]).detach()
    h_path = garch_forward(params, r_all, X_all, h0=h0)
    test = slice(n_train, None)
    h_hat, rv_test, r_test = h_path[test], rv_all[test], r_all[test]

    z = (r_test - params.mu) / torch.sqrt(h_hat)
    if dist == "normal":
        log_f = normal_logpdf(z) - 0.5 * torch.log(h_hat)
    else:
        log_f = std_t_logpdf(z, params.nu) - 0.5 * torch.log(h_hat)

    nu_val = params.nu.item() if dist == "student_t" else None
    q1 = _garch_var_quantile(0.01, dist, params.mu.item(), nu_val)
    q5 = _garch_var_quantile(0.05, dist, params.mu.item(), nu_val)
    var_1pct = -(params.mu.item() + torch.sqrt(h_hat) * q1)
    var_5pct = -(params.mu.item() + torch.sqrt(h_hat) * q5)

    result = {
        "mae": float(mae_vol(rv_test, h_hat)), "rmse": float(rmse_vol(rv_test, h_hat)),
        "qlike": float(qlike(rv_test, h_hat)), "nls": float(negative_log_score(log_f)),
        "h_hat": h_hat, "n_test": int(rv_test.shape[0]),
    }
    result.update(_var_hits_and_backtests(r_test, var_1pct, var_5pct))
    return result


def _eval_msgarchx_point(params, r_all, X_all, rv_all, n_train) -> dict:
    import numpy as np

    from msgarchx.evaluation.density_scores import mixture_log_density

    with torch.no_grad():
        out = ms_garch_forward_filter(params, r_all, X_all)
    h_hat_full = predictive_variance_path(params, out["xi_pred"], out["h"])
    test = slice(n_train, None)
    h_hat, rv_test, r_test = h_hat_full[test], rv_all[test], r_all[test]

    nu_state = params.nu.numpy() if params.state_specific_nu else np.full(2, params.nu.item())
    mu_np = params.mu.numpy()
    var_1pct, var_5pct = [], []
    for t in range(n_train, r_all.shape[0]):
        h_np = out["h"][t].numpy()
        w_np = out["xi_pred"][t].numpy()
        var_1pct.append(mixture_quantile(0.01, mu_np, h_np, w_np, nu_state))
        var_5pct.append(mixture_quantile(0.05, mu_np, h_np, w_np, nu_state))

    log_f = torch.stack([
        mixture_log_density(r_all[t], params.mu, out["h"][t], out["xi_pred"][t], params.nu)
        for t in range(n_train, r_all.shape[0])
    ])
    var_1pct_t = -torch.tensor(var_1pct)
    var_5pct_t = -torch.tensor(var_5pct)

    result = {
        "mae": float(mae_vol(rv_test, h_hat)), "rmse": float(rmse_vol(rv_test, h_hat)),
        "qlike": float(qlike(rv_test, h_hat)), "nls": float(negative_log_score(log_f)),
        "h_hat": h_hat, "n_test": int(rv_test.shape[0]),
    }
    result.update(_var_hits_and_backtests(r_test, var_1pct_t, var_5pct_t))
    return result


def _eval_msgarchx_posterior(samples: dict, config: ModelConfig, r_all, X_all, rv_all, n_train, n_draws: int) -> dict:
    from msgarchx.inference.bayes_model import params_from_trace
    import numpy as np

    n_available = samples["mu"].shape[0]
    idx = torch.randperm(n_available)[: min(n_draws, n_available)]

    h_hat_draws, log_f_draws, var1_draws, var5_draws = [], [], [], []
    for i in idx.tolist():
        draw = {name: samples[name][i] for name in samples}
        params = params_from_trace(draw, config)
        with torch.no_grad():
            out = ms_garch_forward_filter(params, r_all, X_all)
        h_hat = predictive_variance_path(params, out["xi_pred"], out["h"])
        h_hat_draws.append(h_hat[n_train:])

        from msgarchx.evaluation.density_scores import mixture_log_density
        nu_state = params.nu.numpy() if params.state_specific_nu else np.full(2, params.nu.item())
        log_f_t = torch.stack([
            mixture_log_density(r_all[t], params.mu, out["h"][t], out["xi_pred"][t], params.nu)
            for t in range(n_train, r_all.shape[0])
        ])
        log_f_draws.append(log_f_t)

        mu_np = params.mu.numpy()
        v1, v5 = [], []
        for t in range(n_train, r_all.shape[0]):
            h_np = out["h"][t].numpy()
            w_np = out["xi_pred"][t].numpy()
            v1.append(mixture_quantile(0.01, mu_np, h_np, w_np, nu_state))
            v5.append(mixture_quantile(0.05, mu_np, h_np, w_np, nu_state))
        var1_draws.append(torch.tensor(v1))
        var5_draws.append(torch.tensor(v5))

    h_hat = torch.stack(h_hat_draws).mean(dim=0)
    log_f_stack = torch.stack(log_f_draws)  # (n_draws, n_test)
    log_f_bayes = torch.logsumexp(log_f_stack, dim=0) - torch.log(torch.tensor(float(len(idx))))
    var_1pct = -torch.stack(var1_draws).mean(dim=0)
    var_5pct = -torch.stack(var5_draws).mean(dim=0)

    test = slice(n_train, None)
    rv_test, r_test = rv_all[test], r_all[test]
    result = {
        "mae": float(mae_vol(rv_test, h_hat)), "rmse": float(rmse_vol(rv_test, h_hat)),
        "qlike": float(qlike(rv_test, h_hat)), "nls": float(negative_log_score(log_f_bayes)),
        "h_hat": h_hat, "n_test": int(rv_test.shape[0]), "n_draws_used": len(idx),
    }
    result.update(_var_hits_and_backtests(r_test, var_1pct, var_5pct))
    return result


def run_oos_step(cfg: StudyConfig, data: dict, mle_res: dict, vb_res: dict | None, mcmc_res: dict | None) -> dict:
    r_all, X_all, rv_all, n_train = data["r_all"], data["X_all"], data["rv_all"], data["n_train"]
    results = {}

    print("  -- OOS: B1 --")
    results["b1"] = _eval_garch_point(mle_res["b1"].params, "normal", r_all, torch.zeros(r_all.shape[0], 0), rv_all, n_train)
    print("  -- OOS: B2 --")
    results["b2"] = _eval_garch_point(mle_res["b2"].params, "student_t", r_all, torch.zeros(r_all.shape[0], 0), rv_all, n_train)
    print("  -- OOS: B3 --")
    results["b3"] = _eval_garch_point(mle_res["b3"].params, "student_t", r_all, X_all, rv_all, n_train)
    print("  -- OOS: B4 --")
    results["b4"] = _eval_msgarchx_point(mle_res["b4"].params, r_all, torch.zeros(r_all.shape[0], 0), rv_all, n_train)
    print("  -- OOS: M1 --")
    results["m1"] = _eval_msgarchx_point(mle_res["m1"].params, r_all, X_all, rv_all, n_train)

    if vb_res is not None:
        print("  -- OOS: M2 (VB mean-field) --")
        results["m2"] = _eval_msgarchx_posterior(vb_res["m2"].samples, mle_res["config_m1"], r_all, X_all, rv_all, n_train, cfg.oos_n_posterior_draws)
        print("  -- OOS: M3 (VB full-rank) --")
        results["m3"] = _eval_msgarchx_posterior(vb_res["m3"].samples, mle_res["config_m1"], r_all, X_all, rv_all, n_train, cfg.oos_n_posterior_draws)

    if mcmc_res is not None:
        print("  -- OOS: M4 (MCMC) --")
        results["m4"] = _eval_msgarchx_posterior(mcmc_res["m4"].samples, mle_res["config_m1"], r_all, X_all, rv_all, n_train, cfg.oos_n_posterior_draws)

    for name, res in results.items():
        print(f"     {name}: QLIKE={res['qlike']:.4f} MAE={res['mae']:.4f} RMSE={res['rmse']:.4f} NLS={res['nls']:.4f}")

    return results


# ============================================================
# Buoc 6: Nghien cuu mo phong S1-S8
#
# MOI KICH BAN duoc checkpoint RIENG (07_simulation_S1.pt ... S8.pt) thay vi
# gop chung 1 file: neu mot kich ban loi (vd. khong hoi tu do T lon + ngan
# sach toi uu nho), cac kich ban da chay xong truoc do KHONG bi mat, va lan
# chay lai se tu dong bo qua nhung kich ban da co san (giong co che checkpoint
# chinh cua toan pipeline).
# ============================================================
def run_one_scenario(cfg: StudyConfig, code: str, scenario, config: ModelConfig) -> dict:
    print(f"  -- Kich ban {code}: {scenario.description} --")
    scenario_result = {}

    mle_scn = replace(scenario, B=cfg.sim_B_mle, burn_in=cfg.sim_burn_in)
    t0 = time.time()
    mle_kwargs = {"mle": {
        "n_starts": cfg.sim_mle_n_starts, "n_adam": cfg.sim_mle_n_adam, "n_lbfgs": cfg.sim_mle_n_lbfgs,
    }}
    scenario_result["mle"] = run_scenario(
        mle_scn, config, methods=("mle",), base_seed=cfg.seed, method_kwargs=mle_kwargs
    )["mle"]
    print(f"     MLE (B={cfg.sim_B_mle}): {time.time()-t0:.1f}s, mean_brier={scenario_result['mle']['mean_brier']}")

    vb_scn = replace(scenario, B=cfg.sim_B_vb, burn_in=cfg.sim_burn_in)
    t0 = time.time()
    vb_kwargs = {"vb_mean_field": {"n_steps": cfg.sim_vb_n_steps,
                                   "n_posterior_draws": cfg.sim_vb_n_draws}}
    scenario_result["vb_mean_field"] = run_scenario(
        vb_scn, config, methods=("vb_mean_field",), base_seed=cfg.seed, method_kwargs=vb_kwargs
    )["vb_mean_field"]
    print(f"     VB (B={cfg.sim_B_vb}): {time.time()-t0:.1f}s, mean_brier={scenario_result['vb_mean_field']['mean_brier']}")

    mcmc_scn = replace(scenario, B=cfg.sim_B_mcmc, burn_in=cfg.sim_burn_in)
    t0 = time.time()
    mcmc_kwargs = {"mcmc": {
        "n_chains": cfg.sim_mcmc_n_chains, "n_warmup": cfg.sim_mcmc_n_warmup,
        "n_samples": cfg.sim_mcmc_n_samples, "max_tree_depth": cfg.sim_mcmc_max_tree_depth,
    }}
    scenario_result["mcmc"] = run_scenario(
        mcmc_scn, config, methods=("mcmc",), base_seed=cfg.seed, method_kwargs=mcmc_kwargs
    )["mcmc"]
    print(f"     MCMC (B={cfg.sim_B_mcmc}): {time.time()-t0:.1f}s, mean_brier={scenario_result['mcmc']['mean_brier']}")

    return scenario_result


def run_simulation_all_scenarios(cfg: StudyConfig) -> dict:
    """Chay va checkpoint TUNG kich ban rieng le; tra ve dict gop tat ca ket
    qua da co (co the la mot phan neu vai kich ban chua chay/that bai)."""
    scenarios = default_scenarios(T_default=cfg.sim_T, B_default=cfg.sim_B_mle)
    config = ModelConfig(n_exog=cfg.sim_n_exog, state_specific_nu=False)

    results = {}
    for code, scenario in scenarios.items():
        res = safe_checkpoint(cfg, f"07_simulation_{code}", lambda code=code, scenario=scenario: run_one_scenario(cfg, code, scenario, config))
        if res is not None:
            results[code] = res

    if results:
        torch.save(results, cfg.results_dir / "07_simulation.pt")
        print(f"[xong] 07_simulation (tong hop {len(results)}/{len(scenarios)} kich ban) -> {cfg.results_dir / '07_simulation.pt'}")
    return results


# ============================================================
# Main
# ============================================================
STAGE_ORDER = ["data", "prelim", "mle", "vb", "mcmc", "r1", "oos", "simulation"]


class _Tee:
    """Nhan doi stdout/stderr sang mot file nhat ky, KHONG dung pipe cua shell.

    Ly do khong dung `| Tee-Object` cua PowerShell trong file .bat: (1) khi
    stdout bi noi qua pipe, Python chuyen sang block-buffering nen tien do
    hien thi tre rat lau; (2) PowerShell 5.1 boc moi dong stderr cua chuong
    trinh ngoai (vd. thanh tien do tqdm cua MCMC) thanh NativeCommandError khi
    dung `2>&1`, lam nhieu output va sai ma loi. Tee trong Python tranh ca hai.
    """

    def __init__(self, stream, log_handle):
        self._stream = stream
        self._log = log_handle

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        try:
            self._log.write(text)
            self._log.flush()
        except (ValueError, OSError):
            pass
        return len(text)

    def flush(self):
        self._stream.flush()
        try:
            self._log.flush()
        except (ValueError, OSError):
            pass

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="full")
    parser.add_argument("--only", nargs="*", choices=STAGE_ORDER, default=None,
                         help="Chi chay cac buoc duoc chi dinh (mac dinh: tat ca theo thu tu)")
    parser.add_argument("--log-file", default=None,
                         help="Ghi toan bo man hinh ra file nay (van hien binh thuong tren cmd)")
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8", errors="replace")
        sys.stdout = _Tee(sys.stdout, log_handle)
        sys.stderr = _Tee(sys.stderr, log_handle)
        print(f"[nhat ky] Ghi toan bo man hinh vao: {log_path}")

    cfg = PRESETS[args.preset]
    stages = args.only if args.only else STAGE_ORDER
    print(f"=== run_full_study.py | preset={args.preset} | results_dir={cfg.results_dir} ===")
    print(f"Cac buoc se chay: {stages}")

    t_start = time.time()

    # "data" luon can thiet cho hau het cac buoc khac nen luon duoc nap/tinh.
    data = checkpoint(cfg, "00_data", lambda: prepare_data(cfg))

    if "prelim" in stages:
        safe_checkpoint(cfg, "01_preliminary_tests", lambda: run_preliminary_tests_step(cfg, data))

    r_train = data["r_all"][: data["n_train"]]
    X_train = data["X_all"][: data["n_train"]]

    mle_res = None
    if "mle" in stages:
        mle_res = safe_checkpoint(cfg, "02_mle", lambda: run_mle_step(cfg, r_train, X_train))
    else:
        mle_path = cfg.results_dir / "02_mle.pt"
        mle_res = torch.load(mle_path, weights_only=False) if mle_path.exists() else None

    vb_res = None
    if "vb" in stages and mle_res is not None:
        vb_res = safe_checkpoint(cfg, "03_vb", lambda: run_vb_step(cfg, r_train, X_train, mle_res["config_m1"]))
    elif mle_res is not None:
        vb_path = cfg.results_dir / "03_vb.pt"
        vb_res = torch.load(vb_path, weights_only=False) if vb_path.exists() else None

    mcmc_res = None
    if "mcmc" in stages and mle_res is not None:
        mcmc_res = safe_checkpoint(cfg, "04_mcmc", lambda: run_mcmc_step(cfg, r_train, X_train, mle_res))
    elif mle_res is not None:
        mcmc_path = cfg.results_dir / "04_mcmc.pt"
        mcmc_res = torch.load(mcmc_path, weights_only=False) if mcmc_path.exists() else None

    if "r1" in stages:
        safe_checkpoint(cfg, "05_r1", lambda: run_r1_step(cfg, r_train, X_train))

    if "oos" in stages and mle_res is not None:
        safe_checkpoint(cfg, "06_oos", lambda: run_oos_step(cfg, data, mle_res, vb_res, mcmc_res))

    if "simulation" in stages:
        # Checkpoint theo tung kich ban duoc xu ly BEN TRONG ham nay (xem
        # run_simulation_all_scenarios); khong bao ngoai bang safe_checkpoint
        # nua de tranh 1 kich ban loi lam mat ket qua cac kich ban khac.
        run_simulation_all_scenarios(cfg)

    print(f"=== Hoan tat trong {(time.time() - t_start)/60:.1f} phut. Ket qua tai: {cfg.results_dir} ===")


if __name__ == "__main__":
    main()
