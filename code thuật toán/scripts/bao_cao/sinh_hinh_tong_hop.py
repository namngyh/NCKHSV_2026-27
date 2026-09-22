"""Sinh hinh cho Bai bao/tong_hop_ket_qua.tex.

QUAN TRONG: file .tex do NGUOI DUNG da tu tay sua (khong con la ban sinh tu
dong nguyen goc). Script nay CHI ve hinh va luu ra PDF trong hinh_tong_hop/;
KHONG dong vao viec sinh lai tong_hop_ket_qua.tex (xem sinh_tong_hop_ket_qua.py
cho viec do). So lieu dung trong hinh duoc CHEP TAY tu chinh cac bang da co
trong tong_hop_ket_qua.tex (khong doc lai tu results/*.pt) de dam bao hinh
khop tuyet doi voi bang nguoi dung dang xem, ke ca sau khi ho da chinh sua.

Chay (tu `code thuat toan`):  py scripts/bao_cao/sinh_hinh_tong_hop.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FIG = ROOT.parent / "Bài báo" / "hinh_tong_hop"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 140,
})
C_MLE, C_MF, C_FR, C_MCMC = "#2E5A88", "#C77B30", "#7A9A5B", "#9B4B4B"


# ---------------------------------------------------------- 0. Gia VN30F1M
def hinh_gia():
    from msgarchx.data.loader import load_vn30f1m_daily
    daily, _ = load_vn30f1m_daily(str(ROOT / "ohlc_export.csv"))
    daily = daily.iloc[1:]  # bo phien dau (log_return = NaN, khong nam trong r_all)
    d = torch.load(ROOT / "results" / "00_data.pt", weights_only=False)
    n_tr = int(d["n_train"])
    dates = daily["date"].to_numpy()
    close = daily["close"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(dates, close, color=C_MLE, lw=0.7)
    ax.axvline(dates[n_tr], color="#9B4B4B", ls="--", lw=1,
               label="Ranh giới huấn luyện / kiểm định")
    ax.set_ylabel("Giá đóng cửa (điểm)")
    ax.set_title("Giá hợp đồng tương lai VN30F1M (chuỗi hợp đồng liên tục, giá đóng cửa ngày)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "gia_vn30f1m.pdf")
    plt.close(fig)


# ---------------------------------------------------------- 1. Du lieu
def hinh_du_lieu():
    d = torch.load(ROOT / "results" / "00_data.pt", weights_only=False)
    r = d["r_all"].numpy()
    rv = d["rv_all"].numpy()
    dates = list(d["dates_all"])
    n_tr = int(d["n_train"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax1.plot(dates, r, color=C_MLE, lw=0.5)
    ax1.axvline(dates[n_tr], color="#9B4B4B", ls="--", lw=1)
    ax1.set_ylabel("Lợi suất ngày (%)")
    ax1.set_title("Lợi suất và biến động thực tế VN30F1M", fontsize=10)
    ax2.plot(dates, rv, color=C_FR, lw=0.5)
    ax2.axvline(dates[n_tr], color="#9B4B4B", ls="--", lw=1,
                label="Ranh giới huấn luyện / kiểm định")
    ax2.set_ylabel(r"$RV_t$")
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "du_lieu_thoi_gian.pdf")
    plt.close(fig)


# ---------------------------------------------------------- 2. Buoc 1: AIC
def hinh_buoc1_aic():
    ten = ["GARCH\nchuẩn", "GARCH\nStudent-t", "GARCH-X\nStudent-t", "EGARCH\nchuẩn",
           "EGARCH\nStudent-t", "EGARCH-X\nStudent-t", "MS-GARCH\nchuẩn", "MS-GARCH-X\nStudent-t"]
    aic = [5897.80, 5661.62, 5648.87, 5888.09, 5656.08, 5606.87, 5728.29, 5650.58]
    mau = [C_MLE] * 5 + [C_FR] + [C_MLE, C_MLE]
    mau[5] = "#1F7A4D"  # EGARCH-X Student-t: mo hinh thang cuoc

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(ten))
    bars = ax.bar(x, aic, color=mau)
    ax.set_xticks(x)
    ax.set_xticklabels(ten, fontsize=7.5)
    ax.set_ylabel("AIC (MLE)")
    ax.set_ylim(min(aic) - 40, max(aic) + 40)
    for b, v in zip(bars, aic):
        ax.annotate(f"{v:,.1f}".replace(",", "."), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=6.5, rotation=0)
    ax.set_title("AIC trong mẫu theo mô hình (nhỏ hơn là tốt hơn) --- EGARCH-X Student-$t$ thắng cuộc", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "buoc1_aic.pdf")
    plt.close(fig)


# ---------------------------------------------------------- 3. MS-GARCH-X: coverage S1-S3
def hinh_msgarch_coverage():
    # CP 95% cua p11, p22 (tham so nhan dang yeu nhat) -- doc thang tu bang S1-S3
    kich_ban = ["S1", "S2", "S3"]
    p11 = {"MLE": [52, 45, 62], "VB-MF": [0, 5, 0], "VB-FR": [40, 69, 24]}
    p22 = {"MLE": [92, 59, 81], "VB-MF": [3, 58, 0], "VB-FR": [23, 25, 25]}

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    x = np.arange(len(kich_ban))
    w = 0.25
    for ax, dat, ten in [(axs[0], p11, "$p_{11}$"), (axs[1], p22, "$p_{22}$")]:
        for i, (pp, mau) in enumerate([("MLE", C_MLE), ("VB-MF", C_MF), ("VB-FR", C_FR)]):
            ax.bar(x + (i - 1) * w, dat[pp], width=w, label=pp, color=mau)
        ax.axhline(95, color="#9B4B4B", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(kich_ban)
        ax.set_title(f"Coverage 95% của {ten}", fontsize=9.5)
        ax.set_ylim(0, 105)
    axs[0].set_ylabel("Coverage (%)")
    axs[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("MS-GARCH-X Student-$t$: coverage của $p_{11},p_{22}$ theo kịch bản và phương pháp "
                 "(mục tiêu 95%, đường đứt nét)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "msgarch_coverage.pdf")
    plt.close(fig)


# ---------------------------------------------------------- 4. EGARCH-X: theta
def hinh_egarch_theta():
    kich_ban = ["E1\n($\\theta_0=-0{,}10$)", "E2\n($\\theta_0=0$)", "E3\n($\\theta_0=-0{,}10$, đuôi dày)"]
    theta0 = [-0.100, 0.000, -0.100]
    mean_ = {"MLE": [-0.103, -0.001, -0.103], "VB-MF": [-0.111, 0.000, -0.113], "VB-FR": [-0.114, -0.003, -0.117]}
    se = {"MLE": [0.018, 0.017, 0.021], "VB-MF": [0.020, 0.021, 0.022], "VB-FR": [0.022, 0.023, 0.023]}

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(kich_ban))
    w = 0.25
    for i, (pp, mau) in enumerate([("MLE", C_MLE), ("VB-MF", C_MF), ("VB-FR", C_FR)]):
        ax.errorbar(x + (i - 1) * w, mean_[pp], yerr=[1.96 * s for s in se[pp]], fmt="o",
                    color=mau, label=pp, capsize=3, markersize=5)
    ax.scatter(x, theta0, marker="_", s=400, color="black", zorder=5, label=r"$\theta_0$ thật")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(kich_ban, fontsize=8)
    ax.set_ylabel(r"$\hat\theta$ (khoảng 95%)")
    ax.set_title("EGARCH-X Student-$t$: khôi phục hệ số đòn bẩy $\\theta$ theo kịch bản và phương pháp", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(FIG / "egarch_theta.pdf")
    plt.close(fig)


# ---------------------------------------------------------- 5. Buoc 3: R_sd
def hinh_rsd():
    tham = ["$\\mu$", "$\\omega$", "$\\alpha$", "$\\theta$", "$\\beta$", "$\\gamma_1$", "$\\gamma_2$", "$\\nu$"]
    r_mf = [0.95, 0.37, 0.88, 0.92, 0.29, 0.82, 0.28, 0.63]
    r_fr = [0.94, 0.75, 1.01, 1.00, 0.75, 0.86, 0.82, 0.97]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(tham))
    w = 0.35
    ax.bar(x - w / 2, r_mf, width=w, label="VB-MF", color=C_MF)
    ax.bar(x + w / 2, r_fr, width=w, label="VB-FR", color=C_FR)
    ax.axhline(1.0, color="#9B4B4B", ls="--", lw=1, label=r"$R_{sd}=1$ (khớp MCMC)")
    ax.set_xticks(x)
    ax.set_xticklabels(tham)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(r"$R_{sd}=\mathrm{sd}_{VB}/\mathrm{sd}_{MCMC}$")
    ax.set_title("Bước 3: VB đánh giá thấp bất định của $\\omega,\\beta$ rõ nhất "
                 "(dữ liệu VN30F1M thật, EGARCH-X Student-$t$)", fontsize=9.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "rsd_bar.pdf")
    plt.close(fig)


if __name__ == "__main__":
    hinh_gia()
    hinh_du_lieu()
    hinh_buoc1_aic()
    hinh_msgarch_coverage()
    hinh_egarch_theta()
    hinh_rsd()
    print("-> hinh_tong_hop/: 6 file PDF")
