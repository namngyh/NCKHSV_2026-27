"""Ve cac hinh minh hoa ket qua vao "Bài báo/hinh_tong_hop/" -- moi so lieu doc thang tu results/.

    gia_vn30f1m.pdf        gia dong cua ngay VN30F1M, ranh gioi huan luyen / kiem dinh
    du_lieu_thoi_gian.pdf  loi suat ngay va bien dong thuc te RV
    buoc1_aic.pdf          AIC (MLE) cua 6 mo hinh o Buoc 1
    egarch_theta.pdf       khoi phuc he so don bay theta o truong hop E1-E3 (Buoc 2)
    rsd_bar.pdf            R_sd = sd_VB / sd_MCMC tren EGARCH-X Student-t (Buoc 3)

Chay (tu "code thuật toán"):  python scripts/bao_cao/sinh_hinh_tong_hop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from garch_vn30.du_lieu import load_vn30f1m_daily  # noqa: E402
from garch_vn30.mo_hinh import danh_muc  # noqa: E402

RES = ROOT / "results"
FIG = ROOT.parent / "Bài báo" / "hinh_tong_hop"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
MAU = {"mle": "#2E5A88", "vb_mean_field": "#C77B30", "vb_full_rank": "#7A9A5B"}
NHAN = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR"}
nap = lambda p: torch.load(p, weights_only=False)  # noqa: E731


def hinh_gia() -> None:
    daily, _ = load_vn30f1m_daily(str(ROOT / "ohlc_export.csv"))
    daily = daily.iloc[1:]                      # ngay dau khong co loi suat
    n_tr = int(nap(RES / "00_data.pt")["n_train"])
    ngay = daily["date"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(ngay, daily["close"].to_numpy(), color=MAU["mle"], lw=0.7)
    ax.axvline(ngay[n_tr], color="#9B4B4B", ls="--", lw=1, label="Ranh giới huấn luyện / kiểm định")
    ax.set_ylabel("Giá đóng cửa (điểm)")
    ax.set_title("Giá hợp đồng tương lai VN30F1M (giá đóng cửa ngày)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "gia_vn30f1m.pdf")
    plt.close(fig)


def hinh_du_lieu() -> None:
    d = nap(RES / "00_data.pt")
    ngay, n_tr = list(d["dates_all"]), int(d["n_train"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax1.plot(ngay, d["r_all"].numpy(), color=MAU["mle"], lw=0.5)
    ax1.set_ylabel("Lợi suất ngày (%)")
    ax2.plot(ngay, d["rv_all"].numpy(), color=MAU["vb_full_rank"], lw=0.5)
    ax2.set_ylabel(r"$RV_t$")
    for ax in (ax1, ax2):
        ax.axvline(ngay[n_tr], color="#9B4B4B", ls="--", lw=1)
    ax1.set_title("Lợi suất và biến động thực tế VN30F1M", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "du_lieu_thoi_gian.pdf")
    plt.close(fig)


def hinh_buoc1_aic() -> None:
    ten = [s.name.replace(" ", "\n", 1) for s in danh_muc.CATALOGUE]
    aic = [nap(RES / "pipeline" / f"{s.code}_mle.pt")["aic"] for s in danh_muc.CATALOGUE]
    mau = ["#1F7A4D" if a == min(aic) else MAU["mle"] for a in aic]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    cot = ax.bar(np.arange(len(aic)), aic, color=mau)
    ax.set_xticks(np.arange(len(aic)))
    ax.set_xticklabels(ten, fontsize=7.5)
    ax.set_ylim(min(aic) - 40, max(aic) + 40)
    ax.set_ylabel("AIC (MLE)")
    for c, v in zip(cot, aic):
        ax.annotate(f"{v:.1f}".replace(".", ","), (c.get_x() + c.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_title("AIC trong mẫu theo mô hình (nhỏ hơn là tốt hơn)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "buoc1_aic.pdf")
    plt.close(fig)


def hinh_egarch_theta() -> None:
    """Trung binh uoc luong theta +- 1,96 SE Monte Carlo, SE = sqrt(RMSE^2 - Bias^2)."""
    ma = ["E1", "E2", "E3"]
    gop = {m: nap(RES / "mo_phong_egarch" / f"{m}.pt") for m in ma}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(ma))
    for i, pp in enumerate(NHAN):
        that = np.array([gop[m][pp]["tham_so_that"]["theta"] for m in ma])
        bias = np.array([gop[m][pp]["bias"]["theta"] for m in ma])
        rmse = np.array([gop[m][pp]["rmse"]["theta"] for m in ma])
        se = np.sqrt(np.maximum(rmse ** 2 - bias ** 2, 0))
        ax.errorbar(x + (i - 1) * 0.25, that + bias, yerr=1.96 * se, fmt="o", color=MAU[pp],
                    label=NHAN[pp], capsize=3, markersize=5)
    ax.scatter(x, [gop[m]["mle"]["tham_so_that"]["theta"] for m in ma], marker="_", s=400,
               color="black", zorder=5, label=r"$\theta_0$ thật")
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ma)
    ax.set_ylabel(r"$\hat\theta$")
    ax.set_title(r"EGARCH-X Student-$t$: khôi phục hệ số đòn bẩy $\theta$", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(FIG / "egarch_theta.pdf")
    plt.close(fig)


def hinh_rsd() -> None:
    mcmc = nap(RES / "pipeline" / "mcmc_egarch_x_t.pt")["samples"]
    vb = {pp: nap(RES / "pipeline" / f"egarch_x_t_{pp}.pt")["samples"]
          for pp in ("vb_mean_field", "vb_full_rank")}

    def cot(s, ten):
        if ten == "nu":
            return s["nu_minus_2"].double() + 2
        if ten.startswith("gamma"):
            return s["gamma"][:, int(ten[-1])].double()
        return s[ten].double()

    ten = ["mu", "omega", "alpha", "theta", "beta", "gamma0", "gamma1", "nu"]
    nhan = [r"$\mu$", r"$\omega$", r"$\alpha$", r"$\theta$", r"$\beta$", r"$\gamma_1$", r"$\gamma_2$", r"$\nu$"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(ten))
    for i, pp in enumerate(vb):
        r = [float(cot(vb[pp], t).std() / cot(mcmc, t).std()) for t in ten]
        ax.bar(x + (i - 0.5) * 0.35, r, width=0.35, color=MAU[pp], label=NHAN[pp])
    ax.axhline(1.0, color="#9B4B4B", ls="--", lw=1, label=r"$R_{sd}=1$ (khớp MCMC)")
    ax.set_xticks(x)
    ax.set_xticklabels(nhan)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(r"$R_{sd}=\mathrm{sd}_{VB}/\mathrm{sd}_{MCMC}$")
    ax.set_title(r"Độ lệch chuẩn hậu nghiệm VB so với MCMC (EGARCH-X Student-$t$)", fontsize=9.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "rsd_bar.pdf")
    plt.close(fig)


if __name__ == "__main__":
    for ham in (hinh_gia, hinh_du_lieu, hinh_buoc1_aic, hinh_egarch_theta, hinh_rsd):
        ham()
    print(f"-> {FIG}: 5 hinh PDF")
