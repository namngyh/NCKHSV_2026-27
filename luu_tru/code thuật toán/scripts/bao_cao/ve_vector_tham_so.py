"""Ve trung binh tich luy cua uoc luong tung tham so qua cac lan lap mo phong
(truong hop E1-E3):  theta_bar_{j,n} = (1/n) sum_{b<=n} theta_hat_{j,b}.

Moi duong doc thang tu cac file lan lap results/mo_phong_egarch/parts/, va
diem cuoi cua duong duoc doi chieu voi Bias trong file gop (sai lech > 5e-4
thi dung lai bao loi), de hinh va bang luon cung mot nguon.

Chay (tu "code thuật toán"):  python scripts/bao_cao/ve_vector_tham_so.py
Dau ra: Bài báo/hinh_tong_hop/vector_E1.pdf ... vector_E3.pdf (kem .png).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from garch_vn30.mo_phong import THU_TU, tham_so_that, truong_hop_mac_dinh  # noqa: E402

RES = ROOT / "results" / "mo_phong_egarch"
OUT = ROOT.parent / "Bài báo" / "hinh_tong_hop"
PP = ("mle", "vb_mean_field", "vb_full_rank")
MAU = {"mle": "#1f4e79", "vb_mean_field": "#bd6b27", "vb_full_rank": "#377e65"}
NHAN = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR"}
KY_HIEU = {"mu": r"$\mu$", "omega": r"$\omega$", "alpha": r"$\alpha$", "theta": r"$\theta$",
           "beta": r"$\beta$", "gamma_0": r"$\gamma_1$", "gamma_1": r"$\gamma_2$", "nu": r"$\nu$"}


def uoc_luong(ma: str, pp: str) -> list[dict]:
    files = sorted((RES / "parts").glob(f"{ma}_{pp}_*.pt"))
    return [torch.load(f, weights_only=False)["uoc_luong"] for f in files]


def ve(ma: str) -> None:
    that = tham_so_that(truong_hop_mac_dinh()[ma].params)
    gop = torch.load(RES / f"{ma}.pt", weights_only=False)
    duong = {pp: uoc_luong(ma, pp) for pp in PP}
    for pp, ds in duong.items():                     # hinh phai khop bang
        for t in THU_TU:
            cuoi = np.mean([x[t] for x in ds])
            if not np.isclose(cuoi, that[t] + gop[pp]["bias"][t], atol=5e-4):
                raise ValueError(f"Trung binh cuoi khong khop Bias: {ma}/{pp}/{t}")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "legend.frameon": False})
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.89, wspace=0.34, hspace=0.52)
    n_max = max(len(ds) for ds in duong.values())
    for j, (ax, t) in enumerate(zip(axes.ravel(), THU_TU)):
        ax.axhline(that[t], color="#333333", lw=1.1, ls="--")
        for pp, ds in duong.items():
            v = np.array([x[t] for x in ds])
            ax.plot(np.arange(1, len(v) + 1), np.cumsum(v) / np.arange(1, len(v) + 1),
                    color=MAU[pp], lw=1.25)
        ax.set_xlim(1, n_max)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_title(f"{chr(97 + j)}  {KY_HIEU[t]}", loc="left", fontweight="bold", pad=3)
        if j >= 6:
            ax.set_xlabel("Số lần lặp")
    tay = [mpl.lines.Line2D([], [], color=MAU[pp], lw=1.7, label=NHAN[pp]) for pp in PP]
    tay.append(mpl.lines.Line2D([], [], color="#333333", lw=1.2, ls="--", label="Giá trị thật"))
    fig.legend(handles=tay, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94), fontsize=8)
    fig.suptitle(f"{ma}: Trung bình tích lũy của vector tham số qua các lần lặp", y=0.985, fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"vector_{ma}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"vector_{ma}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"-> vector_{ma}.pdf  (" + ", ".join(f"{NHAN[pp]} {len(duong[pp])}" for pp in PP) + " lan lap)")


if __name__ == "__main__":
    for ma in ("E1", "E2", "E3"):
        ve(ma)
