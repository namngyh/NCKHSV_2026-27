"""Danh gia ngoai mau cho GARCH/EGARCH mot trang thai.

Thiet ke: tham so uoc luong MOT LAN tren tap huan luyen; vong lap phuong sai
chay tiep qua tap kiem dinh voi tham so co dinh. Tai moi t, h_t chi dung
thong tin den t-1 nen day la du bao mot buoc ngoai mau hop le (khong tai uoc
luong theo tung ngay).
"""
from __future__ import annotations

import torch

from garch_vn30.danh_gia.ham_mat_mat import mae_vol, negative_log_score, qlike, rmse_vol
from garch_vn30.danh_gia.kiem_dinh_var import christoffersen_test, kupiec_test, phan_vi_z
from garch_vn30.mo_hinh.danh_muc import forward_cua
from garch_vn30.mo_hinh.garch import log_mat_do

MUC_VAR = [(0.01, "1pct"), (0.05, "5pct")]


def duong_h_va_logf(params, family: str, r_all: torch.Tensor, X_all: torch.Tensor,
                    n_train: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h_hat, log f) tren TAP KIEM DINH cho mot bo tham so.

    Dieu kien dau h_0 = phuong sai mau cua tap huan luyen.
    """
    with torch.no_grad():
        h0 = torch.var(r_all[:n_train]).detach()
        h_path = forward_cua(family)(params, r_all, X_all, h0=h0)
    h_hat, r_test = h_path[n_train:], r_all[n_train:]
    return h_hat, log_mat_do(r_test, params.mu, h_hat, params.nu)


def _kiem_dinh_var(ket_qua: dict, r_test: torch.Tensor, var_theo_muc: dict) -> None:
    """Them so vi pham va kiem dinh Kupiec/Christoffersen cho tung muc VaR."""
    for alpha, nhan in MUC_VAR:
        hits = (r_test.numpy() < -var_theo_muc[alpha].numpy()).astype(float)
        ket_qua[f"kupiec_{nhan}"] = kupiec_test(hits, alpha)
        ket_qua[f"christoffersen_{nhan}"] = christoffersen_test(hits, alpha)
        ket_qua[f"n_violations_{nhan}"] = int(hits.sum())


def danh_gia_mot_diem(params, family: str, dist: str, r_all: torch.Tensor,
                      X_all: torch.Tensor, rv_all: torch.Tensor, n_train: int) -> dict:
    """Danh gia cho MOT bo tham so (du bao cam tham so, vd. MLE).

    Tra ve: mae, rmse, qlike (du bao phuong sai so voi RV), nls (mat do),
    h_hat, n_test, va kiem dinh VaR 1%, 5%.
    """
    h_hat, log_f = duong_h_va_logf(params, family, r_all, X_all, n_train)
    rv_test, r_test = rv_all[n_train:], r_all[n_train:]
    ket_qua = {
        "mae": float(mae_vol(rv_test, h_hat)),
        "rmse": float(rmse_vol(rv_test, h_hat)),
        "qlike": float(qlike(rv_test, h_hat)),
        "nls": float(negative_log_score(log_f)),
        "h_hat": h_hat,
        "n_test": int(rv_test.shape[0]),
    }
    nu = None if dist == "normal" else float(params.nu)
    var = {a: -(float(params.mu) + torch.sqrt(h_hat) * phan_vi_z(a, dist, nu))
           for a, _ in MUC_VAR}
    _kiem_dinh_var(ket_qua, r_test, var)
    return ket_qua


def danh_gia_hau_nghiem(draws: list, family: str, dist: str, r_all: torch.Tensor,
                        X_all: torch.Tensor, rv_all: torch.Tensor, n_train: int) -> dict:
    """Danh gia khi co S bo tham so rut tu hau nghiem (VB hoac MCMC).

    - Du bao phuong sai: h_bar_t = (1/S) sum_s h_t^(s).
    - Mat do du bao HON HOP: f_t(r) = (1/S) sum_s f_t(r | theta^(s)); NLS lay
      tren mat do nay (tinh on dinh bang logsumexp). Day la kenh bat dinh
      tham so di vao du bao.
    - VaR: trung binh phan vi cac thanh phan (xap xi phan vi cua hon hop).
    """
    rv_test, r_test = rv_all[n_train:], r_all[n_train:]
    h_ds, logf_ds = [], []
    for params in draws:
        h_hat, log_f = duong_h_va_logf(params, family, r_all, X_all, n_train)
        h_ds.append(h_hat)
        logf_ds.append(log_f)
    h_draws = torch.stack(h_ds)            # (S, n_test)
    logf_draws = torch.stack(logf_ds)      # (S, n_test)

    h_mean = h_draws.mean(dim=0)
    log_mix = torch.logsumexp(logf_draws, dim=0) - torch.log(torch.tensor(float(len(draws))))
    ket_qua = {
        "mae": float(mae_vol(rv_test, h_mean)),
        "rmse": float(rmse_vol(rv_test, h_mean)),
        "qlike": float(qlike(rv_test, h_mean)),
        "nls": float(negative_log_score(log_mix)),
        "h_hat": h_mean,
        "n_test": int(rv_test.shape[0]),
        "n_draws_used": int(h_draws.shape[0]),
    }

    dt = r_all.dtype
    mu_draws = torch.stack([p.mu.reshape(()) for p in draws])
    var = {}
    for a, _ in MUC_VAR:
        if dist == "normal":
            q = torch.full((len(draws),), phan_vi_z(a, "normal", None), dtype=dt)
        else:
            q = torch.tensor([phan_vi_z(a, dist, float(p.nu)) for p in draws], dtype=dt)
        var[a] = -(mu_draws.mean() + (torch.sqrt(h_draws) * q[:, None]).mean(dim=0))
    _kiem_dinh_var(ket_qua, r_test, var)
    return ket_qua
