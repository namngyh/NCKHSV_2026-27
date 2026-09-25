"""Ve trung binh tich luy cua vector tham so qua cac lan lap mo phong.

Moi duong dung dung cac checkpoint tung lan lap. Voi MS-GARCH-X, sap nhan
trang thai truoc khi lay trung binh, giong buoc combine chinh thuc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / ".agents" / "skills" / "nature-figure" / "scripts"))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from msgarchx.simulation.run_simulation import sap_nhan_trang_thai  # noqa: E402
from msgarchx.simulation.scenarios import default_scenarios  # noqa: E402
from msgarchx.simulation.run_egarch import default_egarch_scenarios, named_params_egarch  # noqa: E402

RES = ROOT / "results"
OUT = ROOT.parent / "Bài báo" / "hinh_tong_hop"
METHODS = ("mle", "vb_mean_field", "vb_full_rank")
COLORS = {"mle": "#1f4e79", "vb_mean_field": "#bd6b27", "vb_full_rank": "#377e65"}
LABELS = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR"}
ORDER_A = ("mu1", "mu2", "omega1", "omega2", "alpha1", "alpha2", "beta1", "beta2",
           "gamma1_0", "gamma1_1", "gamma2_0", "gamma2_1", "p11", "p22", "nu")
ORDER_B = ("mu", "omega", "alpha", "theta", "beta", "gamma_0", "gamma_1", "nu")
SYMBOLS = {"mu1": r"$\mu_1$", "mu2": r"$\mu_2$", "omega1": r"$\omega_1$",
           "omega2": r"$\omega_2$", "alpha1": r"$\alpha_1$", "alpha2": r"$\alpha_2$",
           "beta1": r"$\beta_1$", "beta2": r"$\beta_2$", "gamma1_0": r"$\gamma_{1,1}$",
           "gamma1_1": r"$\gamma_{1,2}$", "gamma2_0": r"$\gamma_{2,1}$",
           "gamma2_1": r"$\gamma_{2,2}$", "p11": r"$p_{11}$", "p22": r"$p_{22}$",
           "nu": r"$\nu$", "mu": r"$\mu$", "omega": r"$\omega$",
           "alpha": r"$\alpha$", "theta": r"$\theta$", "beta": r"$\beta$",
           "gamma_0": r"$\gamma_1$", "gamma_1": r"$\gamma_2$"}


def true_values(code: str) -> dict[str, float]:
    if code.startswith("S"):
        s = default_scenarios()[code]
        p = s.params
        out = {}
        for name in ("mu", "omega", "alpha", "beta"):
            out.update({f"{name}1": float(getattr(p, name)[0]),
                        f"{name}2": float(getattr(p, name)[1])})
        for state in range(2):
            for j in range(p.gamma.shape[1]):
                out[f"gamma{state + 1}_{j}"] = float(p.gamma[state, j])
        out.update({"p11": float(s.P0[0, 0]), "p22": float(s.P0[1, 1]), "nu": float(p.nu)})
        return out
    return named_params_egarch(default_egarch_scenarios()[code].params)


def estimates(code: str, method: str) -> tuple[np.ndarray, list[dict[str, float]]]:
    directory = RES / ("parts" if code.startswith("S") else "mo_phong_egarch/parts")
    if code.startswith("S"):
        directory = directory / "sim"
    files = sorted(directory.glob(f"{code}_{method}_*.pt"))
    ids, values = [], []
    for file in files:
        record = torch.load(file, weights_only=False)
        rep = record["rep"]
        if code.startswith("S"):
            rep = sap_nhan_trang_thai(rep)
        ids.append(int(record["b"]))
        values.append(rep.point_estimate)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Trung chi so lan lap: {code}/{method}")
    return np.asarray(ids), values


def plot_one(code: str) -> dict:
    actual = true_values(code)
    order = ORDER_A if code.startswith("S") else ORDER_B
    runs = {method: estimates(code, method) for method in METHODS}
    aggregate_path = (RES / f"07_simulation_{code}.pt" if code.startswith("S")
                      else RES / "mo_phong_egarch" / f"{code}.pt")
    aggregate = torch.load(aggregate_path, weights_only=False)
    for method, (_, values) in runs.items():
        if method not in aggregate:
            continue
        if len(values) != aggregate[method]["n_replications"]:
            raise ValueError(f"So lan lap khong khop: {code}/{method}")
        for name in order:
            empirical_mean = np.mean([float(item[name]) for item in values])
            reported_mean = actual[name] + float(aggregate[method]["bias"][name])
            if not np.isclose(empirical_mean, reported_mean, atol=5e-4):
                raise ValueError(f"Trung binh cuoi khong khop bang Bias: {code}/{method}/{name}")
    n_cols = 3 if code.startswith("S") else 2
    n_rows = int(np.ceil(len(order) / n_cols))
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 8.8 if n_rows == 5 else 6.8))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.07, top=0.89,
                        wspace=0.34, hspace=0.52)
    axes = np.asarray(axes).ravel()
    max_n = max(len(values) for _, values in runs.values())
    for j, name in enumerate(order):
        ax = axes[j]
        ax.axhline(actual[name], color="#333333", lw=1.1, ls="--", zorder=2)
        for method, (_, values) in runs.items():
            if not values:
                continue
            v = np.array([float(item[name]) for item in values])
            n = np.arange(1, len(v) + 1)
            avg = np.cumsum(v) / n
            ax.plot(n, avg, color=COLORS[method], lw=1.25, label=LABELS[method])
        ax.set_xlim(1, max_n)
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.set_title(f"{chr(97 + j)}  {SYMBOLS[name]}", loc="left", fontweight="bold", pad=3)
        if j // n_cols == n_rows - 1:
            ax.set_xlabel("Số lần lặp")
        ax.tick_params(length=2)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], lw=1.7, label=LABELS[m]) for m in METHODS]
    handles.append(mpl.lines.Line2D([], [], color="#333333", lw=1.2, ls="--", label="Giá trị thật"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.94), fontsize=8)
    fig.suptitle(f"{code}: Trung bình tích lũy của vector tham số qua các lần lặp", y=0.985, fontsize=11)
    fig.canvas.draw()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"vector_{code}"
    require_matplotlib_panel_alignment(fig, json_out=str(stem) + ".alignment-layout.json",
                                       overlay_svg=str(stem) + ".alignment.svg",
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       require_panel_labels=False, strict=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"code": code, "n": {LABELS[m]: len(runs[m][1]) for m in METHODS},
            "parameters": len(order), "source": "individual replication checkpoints",
            "label_correction": code.startswith("S")}


def main() -> None:
    metadata = [plot_one(code) for code in [*(f"S{i}" for i in range(1, 9)),
                                               *(f"E{i}" for i in range(1, 4))]]
    (OUT / "vector_tham_so_nguon.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in metadata:
        print(item)


if __name__ == "__main__":
    main()
