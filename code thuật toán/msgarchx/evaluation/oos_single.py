"""Danh gia ngoai mau cho mo hinh MOT trang thai (GARCH va EGARCH).

Thiet ke ngoai mau giong het phan da dung cho MS-GARCH-X (Muc 3.5 trong
README): tham so uoc luong MOT LAN tren tap huan luyen, sau do vong lap
phuong sai chay tiep qua tap kiem dinh voi THAM SO CO DINH. Tai moi t trong
tap kiem dinh, h_t chi dung thong tin den t-1, nen day van la du bao ngoai
mau hop le.
"""
from __future__ import annotations

import torch

from msgarchx.distributions import normal_logpdf, std_t_logpdf
from msgarchx.evaluation.density_scores import negative_log_score
from msgarchx.evaluation.var_backtests import christoffersen_test, kupiec_test
from msgarchx.evaluation.volatility_loss import mae_vol, qlike, rmse_vol
from msgarchx.models.egarch import egarch_forward
from msgarchx.models.garch import garch_forward


def _var_quantile(alpha: float, dist: str, nu: float | None) -> float:
    """Phan vi muc alpha cua sai so chuan hoa z (phuong sai = 1)."""
    from scipy import stats as sstats

    if dist == "normal":
        return float(sstats.norm.ppf(alpha))
    return float(sstats.t.ppf(alpha, df=nu) * ((nu - 2.0) / nu) ** 0.5)


def _duong_h_va_logf(params, family, dist, r_all, X_all, n_train):
    """Tinh (h_hat, log_f) tren TAP KIEM DINH cho MOT bo tham so.

    Dung chung cho ca hai duong danh gia ben duoi, de chung chac chan nhat
    quan voi nhau: cam diem chi la truong hop mot bo tham so, danh gia hau
    nghiem la lap lai ham nay qua tung mau.

    Dieu kien dau h0 lay tu TAP HUAN LUYEN, sau do vong lap phuong sai chay
    tiep qua tap kiem dinh voi tham so co dinh -- tai moi t, h_t chi dung
    thong tin den t-1 nen van la du bao ngoai mau hop le.
    """
    forward = garch_forward if family == "garch" else egarch_forward
    with torch.no_grad():
        h0 = torch.var(r_all[:n_train]).detach()
        h_path = forward(params, r_all, X_all, h0=h0)

    test = slice(n_train, None)
    h_hat, r_test = h_path[test], r_all[test]
    z = (r_test - params.mu) / torch.sqrt(h_hat)
    if dist == "normal":
        log_f = normal_logpdf(z) - 0.5 * torch.log(h_hat)
    else:
        log_f = std_t_logpdf(z, params.nu) - 0.5 * torch.log(h_hat)
    return h_hat, log_f


def evaluate_single_regime(
    params,
    family: str,
    dist: str,
    r_all: torch.Tensor,
    X_all: torch.Tensor,
    rv_all: torch.Tensor,
    n_train: int,
) -> dict:
    """Tra ve dict cac chi tieu ngoai mau: MAE, RMSE, QLIKE, NLS va kiem dinh VaR.

    `params` phai la GARCHParams (family="garch") hoac EGARCHParams
    (family="egarch"). `rv_all` la bien dong thuc te, cung thang phuong sai
    voi h.
    """
    h_hat, log_f = _duong_h_va_logf(params, family, dist, r_all, X_all, n_train)
    test = slice(n_train, None)
    rv_test, r_test = rv_all[test], r_all[test]
    nu_val = None if dist == "normal" else float(params.nu)

    result = {
        "mae": float(mae_vol(rv_test, h_hat)),
        "rmse": float(rmse_vol(rv_test, h_hat)),
        "qlike": float(qlike(rv_test, h_hat)),
        "nls": float(negative_log_score(log_f)),
        "h_hat": h_hat,
        "n_test": int(rv_test.shape[0]),
    }

    mu_val = float(params.mu)
    for alpha, label in [(0.01, "1pct"), (0.05, "5pct")]:
        q = _var_quantile(alpha, dist, nu_val)
        var_series = -(mu_val + torch.sqrt(h_hat) * q)
        hits = (r_test.numpy() < -var_series.numpy()).astype(float)
        result[f"kupiec_{label}"] = kupiec_test(hits, alpha)
        result[f"christoffersen_{label}"] = christoffersen_test(hits, alpha)
        result[f"n_violations_{label}"] = int(hits.sum())

    return result


def evaluate_single_regime_posterior(
    draws: list,
    family: str,
    dist: str,
    r_all: torch.Tensor,
    X_all: torch.Tensor,
    rv_all: torch.Tensor,
    n_train: int,
) -> dict:
    """Danh gia ngoai mau khi CO ca phan phoi hau nghiem, khong chi mot diem.

    `draws` la danh sach cac bo tham so rut tu hau nghiem (VB hoac MCMC).
    Khac biet co ban so voi ban cam-diem o tren:

      - Du bao diem cua phuong sai lay TRUNG BINH h qua cac mau.
      - Diem log (NLS) lay mat do HON HOP: -mean_t log[(1/S) sum_s f_s(r_t)],
        chu khong phai mat do tai mot bo tham so. Day chinh la kenh ma bat
        dinh tham so di vao du bao (eq:posterior-predictive), va la ly do
        phuong phap Bayes co the thang o NLS ngay ca khi khong thang o QLIKE.
    """
    test = slice(n_train, None)
    rv_test, r_test = rv_all[test], r_all[test]

    # Lap lai DUNG phep tinh cua ban cam diem qua tung mau hau nghiem.
    h_stack, logf_stack = [], []
    for params in draws:
        h_hat, log_f = _duong_h_va_logf(params, family, dist, r_all, X_all, n_train)
        h_stack.append(h_hat)
        logf_stack.append(log_f)

    h_draws = torch.stack(h_stack)          # (S, n_test)
    logf_draws = torch.stack(logf_stack)    # (S, n_test)

    h_mean = h_draws.mean(dim=0)
    # log[(1/S) sum_s exp(log f_s)] tinh on dinh bang logsumexp.
    log_mix = torch.logsumexp(logf_draws, dim=0) - torch.log(
        torch.tensor(float(logf_draws.shape[0]))
    )

    result = {
        "mae": float(mae_vol(rv_test, h_mean)),
        "rmse": float(rmse_vol(rv_test, h_mean)),
        "qlike": float(qlike(rv_test, h_mean)),
        "nls": float(negative_log_score(log_mix)),
        "h_hat": h_mean,
        "n_test": int(rv_test.shape[0]),
        "n_draws_used": int(h_draws.shape[0]),
    }

    # VaR lay tu phan vi thuc nghiem cua chinh phan phoi du bao hon hop:
    # mo phong r tu tung mau roi lay phan vi, thay vi dung cong thuc dong.
    dt = r_all.dtype
    mu_draws = torch.stack([p.mu.reshape(()) for p in draws])          # (S,)
    if dist == "normal":
        q_draws = torch.full((len(draws),), _var_quantile(0.01, "normal", None), dtype=dt)
        q5_draws = torch.full((len(draws),), _var_quantile(0.05, "normal", None), dtype=dt)
    else:
        q_draws = torch.tensor([_var_quantile(0.01, dist, float(p.nu)) for p in draws], dtype=dt)
        q5_draws = torch.tensor([_var_quantile(0.05, dist, float(p.nu)) for p in draws], dtype=dt)

    for q_vec, alpha, label in [(q_draws, 0.01, "1pct"), (q5_draws, 0.05, "5pct")]:
        # Phan vi cua hon hop xap xi bang trung binh phan vi cac thanh phan --
        # du cho muc dich kiem dinh vi pham VaR o day.
        var_series = -(
            mu_draws.mean() + (torch.sqrt(h_draws) * q_vec[:, None]).mean(dim=0)
        )
        hits = (r_test.numpy() < -var_series.numpy()).astype(float)
        result[f"kupiec_{label}"] = kupiec_test(hits, alpha)
        result[f"christoffersen_{label}"] = christoffersen_test(hits, alpha)
        result[f"n_violations_{label}"] = int(hits.sum())

    return result
