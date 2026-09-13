"""Tom tat hau nghiem (mean, sd, KTC 95%) va so sanh Variational Bayes voi MCMC.

Dung chung cho variational.py va mcmc.py de dien vao Bang so sanh uoc luong
tham so (tab:parameter-template) va ty so do lech chuan R_sd,j (eq:sd-ratio).
"""
from __future__ import annotations

import torch


def posterior_summary(samples: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """samples[name] co shape (n_draws, *event_shape) -> {name: {mean, sd, q2.5, q97.5}}."""
    summary = {}
    for name, draws in samples.items():
        if draws.numel() == 0:
            # Site RONG. Mo hinh khong dung bien ngoai sinh (vd. MS-GARCH chuan,
            # GARCH chuan) van khai bao site "gamma" nhung voi shape
            # (n_draws, 2, 0). quantile() bao loi tren tensor rong, va thong ke
            # cua mot tap rong cung khong co nghia -> bo qua han site do.
            continue
        summary[name] = {
            "mean": draws.mean(dim=0),
            "sd": draws.std(dim=0),
            "q2.5": draws.quantile(0.025, dim=0),
            "q97.5": draws.quantile(0.975, dim=0),
        }
    return summary


def sd_ratio(
    summary_q: dict[str, dict[str, torch.Tensor]],
    summary_mcmc: dict[str, dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """R_sd,j = sd_q(theta_j) / sd_MCMC(theta_j), eq:sd-ratio.

    Gia tri < 1 cho thay Variational Bayes danh gia thap phuong sai hau nghiem
    so voi MCMC (dung lam chuan tham chieu chi sau khi hoi tu dat yeu cau).
    """
    ratios = {}
    for name in summary_q:
        if name not in summary_mcmc:
            continue
        ratios[name] = summary_q[name]["sd"] / summary_mcmc[name]["sd"]
    return ratios


# ===========================================================================
# Sap xep lai nhan trang thai cho MAU HAU NGHIEM (VB / MCMC)
#
# VI SAO CAN: eq:label-identification quy dinh trang thai 1 la trang thai bien
# dong THAP. `label_switch_correction` trong models/ms_garch.py chi ap dung cho
# MOT bo tham so diem (ket qua MLE). Mau hau nghiem cua VB/MCMC KHONG duoc sap
# xep o bat ky dau trong pipeline, nen:
#   - Pr(nu_2 < nu_1 | Data) co the bi tinh NGUOC (eq:tail-probability);
#   - bang tham so hau nghiem co the in trang thai 1/2 dao nhau;
#   - ty so do lech chuan R_sd,j giua VB va MCMC co the so trang thai THAP cua
#     phuong phap nay voi trang thai CAO cua phuong phap kia.
#
# TIEU CHI: dung phuong sai KHONG dieu kien ham y cua tung trang thai,
#     v_s = omega_s / (1 - alpha_s - beta_s),
# lam dai dien cho E(h_t | S_t = s). Day la dai dien re (dang dong, khong phai
# chay bo loc cho tung mau) va nhat quan voi y nghia cua eq:label-identification.
# Khi alpha_s + beta_s >= 1 (che do no) thi v_s = +vo cung, tuc chac chan la
# trang thai bien dong cao.
#
# Ham tra ve ca TY LE MAU DA BI HOAN VI: neu ty le nay gan 0.5 thi chinh cac
# chuoi/mau dang bi doi nhan qua lai (label switching) chu khong phai chi mot
# phep hoan vi toan cuc -- can bao cao ro trong phan chan doan.
# ===========================================================================
def implied_unconditional_variance(omega: torch.Tensor, alpha: torch.Tensor,
                                   beta: torch.Tensor) -> torch.Tensor:
    """v_s = omega_s / (1 - alpha_s - beta_s), = +inf khi alpha_s+beta_s >= 1."""
    denom = 1.0 - alpha - beta
    v = torch.where(denom > 0, omega / denom.clamp(min=1e-12),
                    torch.full_like(omega, float("inf")))
    return v


def relabel_posterior_samples(samples: dict[str, torch.Tensor]) -> tuple[dict, float]:
    """Sap xep lai nhan trang thai cho TUNG mau, thoa E(h|S=1) < E(h|S=2).

    `samples[name]` co shape (n_draws, *event_shape) dung nhu VBResult.samples
    va MCMCResult.samples. Tra ve (mau_da_sap_xep, ty_le_mau_bi_hoan_vi).
    """
    need = ("mu", "omega", "alpha", "beta", "gamma", "p11", "p22")
    if any(k not in samples for k in need):
        return samples, 0.0

    v = implied_unconditional_variance(samples["omega"], samples["alpha"], samples["beta"])
    swap = v[:, 0] > v[:, 1]                      # (n_draws,) bool
    ty_le = float(swap.double().mean())
    if ty_le == 0.0:
        return samples, 0.0

    idx = torch.where(swap.unsqueeze(1),
                      torch.tensor([1, 0]).expand(len(swap), 2),
                      torch.tensor([0, 1]).expand(len(swap), 2))
    out = dict(samples)
    for name in ("mu", "omega", "alpha", "beta"):
        out[name] = torch.gather(samples[name], 1, idx)
    # gamma: (n_draws, 2, k) -> hoan vi theo chieu trang thai
    g = samples["gamma"]
    out["gamma"] = torch.gather(g, 1, idx.unsqueeze(-1).expand(-1, -1, g.shape[-1]))
    # nu theo trang thai (mo hinh R1)
    nu = samples.get("nu_minus_2")
    if nu is not None and nu.dim() == 2 and nu.shape[1] == 2:
        out["nu_minus_2"] = torch.gather(nu, 1, idx)
    # P hoan vi ca hang lan cot => p11 <-> p22
    p11, p22 = samples["p11"], samples["p22"]
    out["p11"] = torch.where(swap, p22, p11)
    out["p22"] = torch.where(swap, p11, p22)
    return out, ty_le
