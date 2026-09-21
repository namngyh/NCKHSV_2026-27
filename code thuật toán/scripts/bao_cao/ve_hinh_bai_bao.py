"""Ve cac hinh cho bai bao tu ket qua da co, xuat PDF vector vao `Bai bao/hinh/`.

CHI ve nhung hinh co du lieu CHAC CHAN. Hinh nao can ket qua chua tin duoc
(vd. M4 khi chuoi MCMC chua hoi tu) thi bo qua, khong ve.

Chay:
    python scripts/bao_cao/ve_hinh_bai_bao.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HINH = ROOT.parent / "Bài báo" / "hinh"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from msgarchx.evaluation.density_scores import mixture_cdf
from msgarchx.models.ms_garch import (
    label_switch_correction,
    ms_garch_forward_filter,
    ms_garch_smoother,
)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False,
})
XANH, DO, XAM = "#1d4ed8", "#b91c1c", "#6b7280"


def _nap(ten):
    p = ROOT / "results" / f"{ten}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def _luu(fig, ten):
    HINH.mkdir(parents=True, exist_ok=True)
    duong = HINH / f"{ten}.pdf"
    fig.savefig(duong)                      # PDF vector cho LaTeX
    fig.savefig(HINH / f"{ten}.png", dpi=150)  # PNG de xem nhanh / bao cao HTML
    plt.close(fig)
    print(f"  da ve {duong.relative_to(ROOT.parent)}")
    return duong


# ============================================================
def hinh_gia_loi_suat_rv(data):
    """Ba bang: gia dong cua, loi suat logarit, bien dong thuc te."""
    from msgarchx.data.loader import load_vn30f1m_daily
    daily, _ = load_vn30f1m_daily(str(ROOT / "ohlc_export.csv"))
    ngay_gia = daily["date"].to_numpy()
    gia = daily["close"].to_numpy()

    ngay = np.asarray(data["dates_all"])
    r = data["r_all"].numpy()
    rv = data["rv_all"].numpy()
    n_train = data["n_train"]
    moc = ngay[n_train]

    fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    ax[0].plot(ngay_gia, gia, lw=0.7, color=XANH)
    ax[0].set_ylabel("Giá đóng cửa")
    ax[1].plot(ngay, r, lw=0.5, color=XAM)
    ax[1].set_ylabel("Lợi suất $r_t$ (%)")
    ax[2].plot(ngay, rv, lw=0.5, color=DO)
    ax[2].set_ylabel("$RV_t$ (%$^2$)")
    ax[2].set_yscale("log")
    for a in ax:
        a.axvline(moc, color="k", ls="--", lw=0.8)
    ax[0].text(moc, ax[0].get_ylim()[1], " bắt đầu tập kiểm định",
               va="top", fontsize=7.5)
    ax[2].set_xlabel("Thời gian")
    fig.align_ylabels(ax)
    return _luu(fig, "gia_loi_suat_rv")


def hinh_phan_phoi(data, mle):
    """Histogram co phu mat do chuan va Student-t, kem do thi Q--Q."""
    r = data["r_all"].numpy()
    z = (r - r.mean()) / r.std(ddof=1)
    nu = float(mle["b2"].params.nu) if mle else 5.0

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax[0].hist(z, bins=80, density=True, color="#cbd5e1", edgecolor="none")
    x = np.linspace(z.min(), z.max(), 400)
    ax[0].plot(x, stats.norm.pdf(x), color=XAM, lw=1.4, label="Chuẩn")
    sc = np.sqrt((nu - 2) / nu)
    ax[0].plot(x, stats.t.pdf(x / sc, df=nu) / sc, color=DO, lw=1.4,
               label=f"Student-t ($\\nu={nu:.2f}$)")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-4, 1)
    ax[0].set_xlabel("Lợi suất chuẩn hóa")
    ax[0].set_ylabel("Mật độ (thang log)")
    ax[0].legend(fontsize=7.5, frameon=False)

    # Q--Q so voi Student-t chuan hoa
    n = len(z)
    p = (np.arange(1, n + 1) - 0.5) / n
    lt = stats.t.ppf(p, df=nu) * sc
    ax[1].scatter(lt, np.sort(z), s=3, color=XANH, alpha=0.5, edgecolors="none")
    lim = [min(lt.min(), z.min()), max(lt.max(), z.max())]
    ax[1].plot(lim, lim, color=DO, lw=1.0)
    ax[1].set_xlabel(f"Phân vị lý thuyết $t^*_{{{nu:.2f}}}$")
    ax[1].set_ylabel("Phân vị thực nghiệm")
    return _luu(fig, "phan_phoi_loi_suat")


def hinh_xac_suat_trang_thai(data, mle):
    """Xac suat LOC va LAM TRON cua trang thai bien dong cao, tren TOAN mau.

    Tham so uoc luong tren tap huan luyen, roi bo loc chay tiep qua tap kiem
    dinh voi tham so co dinh -- dung thiet ke o Muc 3.5 cua README.
    """
    m1 = mle["m1"]
    p = label_switch_correction(m1.params, m1.filter_output["h"])
    r, X = data["r_all"], data["X_all"]
    with torch.no_grad():
        out = ms_garch_forward_filter(p, r, X)
        smo = ms_garch_smoother(out["xi_filt"], out["xi_pred"], p.P)
    ngay = np.asarray(data["dates_all"])
    moc = ngay[data["n_train"]]

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax[0].fill_between(ngay, 0, out["xi_filt"][:, 1].numpy(), color=DO, lw=0, alpha=0.85)
    ax[0].set_ylabel("$\\xi_{t|t}(2)$")
    ax[0].set_title("Xác suất lọc của trạng thái biến động cao", fontsize=9, loc="left")
    ax[1].fill_between(ngay, 0, smo[:, 1].numpy(), color=XANH, lw=0, alpha=0.85)
    ax[1].set_ylabel("$\\xi_{t|T}(2)$")
    ax[1].set_title("Xác suất làm trơn (chỉ mô tả lịch sử, không dùng dự báo)",
                    fontsize=9, loc="left")
    for a in ax:
        a.set_ylim(0, 1)
        a.axvline(moc, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("Thời gian")
    fig.align_ylabels(ax)
    return _luu(fig, "xac_suat_trang_thai")


def hinh_pit(data, mle):
    """Histogram PIT tren tap kiem dinh cho cac mo hinh diem dang tin."""
    r, X = data["r_all"], data["X_all"]
    n_train = data["n_train"]
    mo_hinh = [("B3: GARCH-X Student-t", "b3"), ("M1: MS-GARCH-X Student-t", "m1")]

    def pit_ms(res):
        p = label_switch_correction(res.params, res.filter_output["h"])
        with torch.no_grad():
            o = ms_garch_forward_filter(p, r, X)
        mu, nu = p.mu.numpy(), float(p.nu)
        u = [mixture_cdf(float(r[t]), mu, o["h"][t].numpy(),
                         o["xi_pred"][t].numpy(), np.full(2, nu))
             for t in range(n_train, r.shape[0])]
        return np.asarray(u)

    def pit_garch(res, X_dung):
        from msgarchx.models.garch import garch_forward
        p = res.params
        with torch.no_grad():
            h = garch_forward(p, r, X_dung)
        nu = float(p.nu)
        sc = np.sqrt((nu - 2) / nu)
        u = [stats.t.cdf((float(r[t]) - float(p.mu)) / (np.sqrt(float(h[t])) * sc), df=nu)
             for t in range(n_train, r.shape[0])]
        return np.asarray(u)

    dl = {}
    dl["B3: GARCH-X Student-t"] = pit_garch(mle["b3"], X)
    dl["M1: MS-GARCH-X Student-t"] = pit_ms(mle["m1"])

    fig, ax = plt.subplots(1, len(dl), figsize=(3.6 * len(dl), 2.8), sharey=True)
    ax = np.atleast_1d(ax)
    for a, (ten, u) in zip(ax, dl.items()):
        a.hist(u, bins=20, range=(0, 1), density=True, color="#cbd5e1",
               edgecolor="white", linewidth=0.6)
        a.axhline(1.0, color=DO, lw=1.2, ls="--")
        ks = stats.kstest(u, "uniform")
        a.set_title(f"{ten}\nKS $p={ks.pvalue:.3f}$", fontsize=8.5)
        a.set_xlabel("$u_t = \\widehat F_t(r_t)$")
    ax[0].set_ylabel("Mật độ")
    return _luu(fig, "pit")


def main() -> int:
    data, mle = _nap("00_data"), _nap("02_mle")
    if data is None or mle is None:
        print("[LOI] thieu 00_data.pt hoac 02_mle.pt")
        return 1
    print("Dang ve:")
    hinh_gia_loi_suat_rv(data)
    hinh_phan_phoi(data, mle)
    hinh_xac_suat_trang_thai(data, mle)
    hinh_pit(data, mle)
    print("\nXong. Cac hinh nam trong 'Bai bao/hinh/'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
