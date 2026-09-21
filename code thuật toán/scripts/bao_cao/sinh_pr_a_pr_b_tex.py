"""Sinh phan PR-A + PR-B (LaTeX) cho `Bai bao/8_kich_ban_mo_phong.tex` truc tiep
tu `results/07_simulation_S1..S3.pt` va `results/mo_phong_egarch/E1..E3.pt`.

Khong nhap tay so lieu -- dung dinh dang bang giong voi phan Phu luc S1-S8 da co
(Mean, Do chech, RMSE, SE, CP 95%) de nguoi doc doi chieu duoc hai nghien cuu.

Chay:
    python "scripts/bao_cao/sinh_pr_a_pr_b_tex.py"
In ra doan LaTeX (section PR-A + PR-B) ra stdout; script khac ghep vao file
hoan chinh (xem `cap_nhat_8_kich_ban.py`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from msgarchx.simulation.scenarios import default_scenarios

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

TEN_PP = {
    "mle": "Hợp lý cực đại (MLE)",
    "vb_mean_field": "Bayes biến phân mean-field",
    "vb_full_rank": "Bayes biến phân full-rank",
}

NHAN_PRA = {
    "mu1": r"$\mu_1$", "mu2": r"$\mu_2$",
    "omega1": r"$\omega_1$", "omega2": r"$\omega_2$",
    "alpha1": r"$\alpha_1$", "alpha2": r"$\alpha_2$",
    "beta1": r"$\beta_1$", "beta2": r"$\beta_2$",
    "p11": r"$p_{11}$", "p22": r"$p_{22}$",
    "gamma1_0": r"$\gamma_{1,1}$", "gamma1_1": r"$\gamma_{1,2}$",
    "gamma2_0": r"$\gamma_{2,1}$", "gamma2_1": r"$\gamma_{2,2}$",
    "nu": r"$\nu$",
}
THU_TU_PRA = ["mu1", "mu2", "omega1", "omega2", "alpha1", "alpha2", "beta1",
              "beta2", "gamma1_0", "gamma1_1", "gamma2_0", "gamma2_1",
              "p11", "p22", "nu"]

NHAN_PRB = {
    "mu": r"$\mu$", "omega": r"$\omega$", "alpha": r"$\alpha$",
    "theta": r"$\theta$", "beta": r"$\beta$",
    "gamma_0": r"$\gamma_1$", "gamma_1": r"$\gamma_2$", "nu": r"$\nu$",
}
THU_TU_PRB = ["mu", "omega", "alpha", "theta", "beta", "gamma_0", "gamma_1", "nu"]


def _so(x, chu_so=3):
    if x is None:
        return "--"
    s = f"{x:.{chu_so}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]  # tranh hien thi "-0,000" (am khong) do lam tron
    s = s.replace(".", ",")
    if s.startswith("-"):
        return "$-$" + s[1:]
    return s


def _true_values_pra(code: str) -> dict:
    scn = default_scenarios()[code]
    p = scn.params
    P0 = scn.P0
    return {
        "mu1": float(p.mu[0]), "mu2": float(p.mu[1]),
        "omega1": float(p.omega[0]), "omega2": float(p.omega[1]),
        "alpha1": float(p.alpha[0]), "alpha2": float(p.alpha[1]),
        "beta1": float(p.beta[0]), "beta2": float(p.beta[1]),
        "gamma1_0": float(p.gamma[0, 0]), "gamma1_1": float(p.gamma[0, 1]),
        "gamma2_0": float(p.gamma[1, 0]), "gamma2_1": float(p.gamma[1, 1]),
        "p11": float(P0[0, 0]), "p22": float(P0[1, 1]),
        "nu": float(p.nu),
    }


def _bang_tham_so(ten_params, nhan, thuat, method_dict, theta0, brier=None):
    bias = method_dict["bias"]
    rmse = method_dict["rmse"]
    cov = method_dict.get("coverage") or {}
    n = method_dict["n_replications"]
    co_cp = bool(cov)
    cols = "l C C C C C" + (" C" if co_cp else "")
    out = []
    out.append(r"\begin{table}[H]\centering\scriptsize")
    brier_txt = f", Brier trung bình $={_so(brier, 4)}$" if brier is not None else ""
    out.append(
        rf"\caption{{{thuat} ($n={n}$ lần lặp{brier_txt})}}"
    )
    out.append(rf"\begin{{tabularx}}{{\textwidth}}{{{cols}}}")
    out.append(r"\toprule")
    hdr = r"\textbf{Tham số} & \textbf{Thật ($\theta_0$)} & \textbf{Mean} & \textbf{Độ chệch} & \textbf{RMSE} & \textbf{SE}"
    if co_cp:
        hdr += r" & \textbf{CP 95\%}"
    out.append(hdr + r" \\")
    out.append(r"\midrule")
    for k in ten_params:
        t0 = theta0[k]
        b = bias[k]
        r_ = rmse[k]
        mean = t0 + b
        se = max(r_ * r_ - b * b, 0.0) ** 0.5
        row = f"{nhan[k]} & {_so(t0)} & {_so(mean)} & {_so(b)} & {_so(r_)} & {_so(se)}"
        if co_cp:
            c = cov.get(k)
            row += f" & {int(round(c * 100))}\\%" if c is not None else " & --"
        out.append(row + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    out.append(r"\end{table}")
    out.append("")
    return "\n".join(out)


def _bang_theta0_pra(code: str, theta0: dict) -> str:
    out = [r"\begin{table}[H]\centering\small",
           rf"\caption{{{code}: toàn bộ tham số thật $\theta_0$ dùng để sinh dữ liệu}}",
           r"\begin{tabularx}{\textwidth}{l C  l C}",
           r"\toprule",
           r"\textbf{Tham số} & \textbf{Giá trị} & \textbf{Tham số} & \textbf{Giá trị} \\",
           r"\midrule"]
    trai = ["mu1", "mu2", "omega1", "omega2", "alpha1", "alpha2", "beta1", "beta2"]
    phai = ["gamma1_0", "gamma1_1", "gamma2_0", "gamma2_1", "p11", "p22", "nu"]
    for i in range(max(len(trai), len(phai))):
        l = f"{NHAN_PRA[trai[i]]} & {_so(theta0[trai[i]])}" if i < len(trai) else " & "
        r_ = f"{NHAN_PRA[phai[i]]} & {_so(theta0[phai[i]])}" if i < len(phai) else " & "
        out.append(f"{l} & {r_} \\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    out.append(r"\vspace{0.2em}\raggedright\scriptsize \emph{Ghi chú:} $T=2000$, "
               r"burn-in $=200$, $B=100$ lần lặp (mục tiêu và thực đạt đều bằng nhau, "
               r"trừ vài lần lặp phân kỳ NaN đã dùng quy tắc thử lại điểm khởi tạo).")
    out.append(r"\end{table}")
    out.append("")
    return "\n".join(out)


def sinh_pra() -> str:
    out = []
    mo_ta = {
        "S1": "Mốc nền --- hai trạng thái phân biệt rõ, $\\nu_0=8$",
        "S2": "Nhận dạng yếu --- hai trạng thái gần nhau, $\\nu_0=8$",
        "S3": "Đuôi rất dày --- $\\nu_0=4$, sát biên tồn tại phương sai",
    }
    for code in ["S1", "S2", "S3"]:
        d = torch.load(RESULTS / f"07_simulation_{code}.pt", weights_only=False)
        theta0 = _true_values_pra(code)
        out.append(rf"\subsection*{{Kịch bản {code} --- {mo_ta[code]}}}")
        out.append(_bang_theta0_pra(code, theta0))
        for meth in ["mle", "vb_mean_field", "vb_full_rank"]:
            out.append(_bang_tham_so(THU_TU_PRA, NHAN_PRA, TEN_PP[meth], d[meth],
                                      theta0, brier=d[meth].get("mean_brier")))
        n_dao = {m: d[m].get("n_dao_nhan") for m in ["mle", "vb_mean_field", "vb_full_rank"]}
        out.append(
            rf"\emph{{Đảo nhãn trạng thái trước khi gộp (đã sửa):}} "
            rf"MLE {n_dao['mle']}/100, VB-MF {n_dao['vb_mean_field']}/100, "
            rf"VB-FR {n_dao['vb_full_rank']}/100 lần lặp trả về nhãn trạng thái "
            rf"ngược quy ước (đã hoán vị trước khi tính Bias/RMSE/CP ở trên)."
        )
        out.append(r"\bigskip")
        out.append("")
    return "\n".join(out)


def sinh_prb() -> str:
    out = []
    mo_ta = {
        "E1": r"Đòn bẩy rõ, $\theta_0=-0{,}10$",
        "E2": r"Không đòn bẩy, $\theta_0=0$ --- kịch bản quan trọng nhất",
        "E3": r"Đuôi dày, $\nu_0=4$, có đòn bẩy",
    }
    for code in ["E1", "E2", "E3"]:
        d = torch.load(RESULTS / "mo_phong_egarch" / f"{code}.pt", weights_only=False)
        theta0 = d["mle"]["tham_so_that"]
        theta0 = {k: float(v) for k, v in theta0.items()}
        out.append(rf"\subsection*{{Kịch bản {code} --- {mo_ta[code]}}}")
        tbl = [r"\begin{table}[H]\centering\small",
               rf"\caption{{{code}: toàn bộ tham số thật $\theta_0$ dùng để sinh dữ liệu}}",
               r"\begin{tabularx}{\textwidth}{l C  l C}",
               r"\toprule",
               r"\textbf{Tham số} & \textbf{Giá trị} & \textbf{Tham số} & \textbf{Giá trị} \\",
               r"\midrule"]
        trai = ["mu", "omega", "alpha", "theta"]
        phai = ["beta", "gamma_0", "gamma_1", "nu"]
        for i in range(4):
            tbl.append(f"{NHAN_PRB[trai[i]]} & {_so(theta0[trai[i]])} & "
                       f"{NHAN_PRB[phai[i]]} & {_so(theta0[phai[i]])} \\\\")
        tbl.append(r"\bottomrule")
        tbl.append(r"\end{tabularx}")
        tbl.append(r"\vspace{0.2em}\raggedright\scriptsize \emph{Ghi chú:} $T=2000$, "
                   r"$B=100$ lần lặp, mô hình một trạng thái (không có đối xứng "
                   r"hoán vị nhãn).")
        tbl.append(r"\end{table}")
        tbl.append("")
        out.append("\n".join(tbl))
        for meth in ["mle", "vb_mean_field", "vb_full_rank"]:
            out.append(_bang_tham_so(THU_TU_PRB, NHAN_PRB, TEN_PP[meth], d[meth],
                                      theta0, brier=None))
        out.append(r"\bigskip")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print("%% ===== PR-A =====")
    print(sinh_pra())
    print("%% ===== PR-B =====")
    print(sinh_prb())
