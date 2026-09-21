# -*- coding: utf-8 -*-
"""Sinh hinh cho bao cao ket qua tu cac file .pt that."""
import io, os, sys, math

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from msgarchx.inference.summary import posterior_summary, relabel_posterior_samples

R = os.path.join(ROOT, "results")
BAIBAO = os.path.join(REPO, "Bài báo")
FIG = os.path.join(BAIBAO, "hinh_bao_cao")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})

C_MLE, C_VB, C_VB2, C_MCMC = "#2E5A88", "#C77B30", "#7A9A5B", "#9B4B4B"
C_BEST, C_BASE = "#2E5A88", "#9AA5B1"

mle = torch.load(os.path.join(R, "02_mle.pt"), weights_only=False)
vb = torch.load(os.path.join(R, "03_vb.pt"), weights_only=False)
mc2_p = os.path.join(R, "04_mcmc.pt")
mc1_p = os.path.join(R, "lich_su_sua_loi", "04_mcmc_lan1_duongcheo.pt")
mc2 = torch.load(mc2_p, weights_only=False)["m4"] if os.path.exists(mc2_p) else None
mc1 = torch.load(mc1_p, weights_only=False)["m4"] if os.path.exists(mc1_p) else None
oos = torch.load(os.path.join(R, "06_oos.pt"), weights_only=False)
timing = torch.load(os.path.join(R, "09_timing.pt"), weights_only=False)
r1d = torch.load(os.path.join(R, "05_r1.pt"), weights_only=False)

DATA = {}   # gom moi so lieu de xuat sang .tex


def npar(res):
    return int(round((res.aic + 2 * res.loglik) / 2))


# ---------------------------------------------------------------- Hinh 1: AIC/BIC
models_is = ["b1", "b2", "b3", "b4", "m1"]
lab_is = ["B1\nGARCH", "B2\nGARCH-t", "B3\nGARCH-X-t", "B4\nMS-GARCH", "M1\nMS-GARCH-X-t"]
aic = [mle[m].aic for m in models_is]
bic = [mle[m].bic for m in models_is]
ll = [mle[m].loglik for m in models_is]
k = [npar(mle[m]) for m in models_is]
DATA["is"] = {m: dict(loglik=mle[m].loglik, aic=mle[m].aic, bic=mle[m].bic, k=npar(mle[m]))
              for m in models_is}

fig, ax = plt.subplots(figsize=(7.0, 3.1))
x = np.arange(len(models_is)); w = 0.38
b1 = ax.bar(x - w/2, aic, w, label="AIC", color=C_MLE)
b2 = ax.bar(x + w/2, bic, w, label="BIC", color="#A8C0D8")
ax.set_xticks(x); ax.set_xticklabels(lab_is)
ax.set_ylim(min(min(aic), min(bic)) - 40, max(max(aic), max(bic)) + 45)
ax.set_ylabel("Giá trị tiêu chí (nhỏ hơn = tốt hơn)")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
ia, ib = int(np.argmin(aic)), int(np.argmin(bic))
b1[ia].set_edgecolor("black"); b1[ia].set_linewidth(1.6)
b2[ib].set_edgecolor("black"); b2[ib].set_linewidth(1.6)
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.set_title("Tiêu chí thông tin trong mẫu (viền đậm = tốt nhất)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_ic.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Hinh 2: OOS
mo = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4"]
lab_o = ["B1", "B2", "B3", "B4", "M1", "M2", "M3", "M4"]
qlike = [oos[m]["qlike"] for m in mo]
nls = [oos[m]["nls"] for m in mo]
mae = [oos[m]["mae"] for m in mo]
rmse = [oos[m]["rmse"] for m in mo]
DATA["oos"] = {m: dict(mae=oos[m]["mae"], rmse=oos[m]["rmse"],
                       qlike=oos[m]["qlike"], nls=oos[m]["nls"]) for m in mo}

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, vals, name in [(axs[0], qlike, "QLIKE"), (axs[1], nls, "Điểm log âm (NLS)")]:
    order = np.argsort(vals)
    cols = [C_BASE] * len(vals); cols[int(order[0])] = C_BEST
    bars = ax.bar(np.arange(len(vals)), vals, color=cols)
    bars[mo.index("m4")].set_hatch("///")
    bars[mo.index("m4")].set_edgecolor("#6b7280")
    ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(lab_o)
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo - (hi - lo) * 0.35, hi + (hi - lo) * 0.18)
    ax.set_title(f"{name} ngoài mẫu (nhỏ hơn = tốt hơn)", fontsize=9)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}".replace(".", ","), (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=6.5, rotation=90)
axs[0].legend(handles=[Patch(facecolor=C_BEST, label="Tốt nhất"),
                       Patch(facecolor=C_BASE, hatch="///", edgecolor="#6b7280",
                             label="M4: chưa hội tụ")],
              frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_oos.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Hinh 3: VaR
fig, ax = plt.subplots(figsize=(7.0, 2.8))
mv = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4"]
v1 = [oos[m]["n_violations_1pct"] for m in mv]
v5 = [oos[m]["n_violations_5pct"] for m in mv]
n_test = oos["b1"]["n_test"]
DATA["var"] = {m: dict(v1=oos[m]["n_violations_1pct"], v5=oos[m]["n_violations_5pct"],
                       kup1=oos[m]["kupiec_1pct"]["p_value"], kup5=oos[m]["kupiec_5pct"]["p_value"],
                       cc1=oos[m]["christoffersen_1pct"]["p_value_cc"],
                       cc5=oos[m]["christoffersen_5pct"]["p_value_cc"]) for m in mv}
x = np.arange(len(mv)); w = 0.38
ax.bar(x - w/2, v1, w, label="Vi phạm ở mức 1%", color=C_MLE)
ax.bar(x + w/2, v5, w, label="Vi phạm ở mức 5%", color="#A8C0D8")
ax.axhline(0.01 * n_test, color=C_MCMC, ls="--", lw=1.2)
ax.axhline(0.05 * n_test, color="#C77B30", ls="--", lw=1.2)
ax.annotate(f"kỳ vọng 1% = {0.01*n_test:.2f}".replace(".", ","), (len(mv) - 0.4, 0.01 * n_test),
            fontsize=6.5, color=C_MCMC, va="bottom", ha="right")
ax.annotate(f"kỳ vọng 5% = {0.05*n_test:.2f}".replace(".", ","), (len(mv) - 0.4, 0.05 * n_test),
            fontsize=6.5, color="#C77B30", va="bottom", ha="right")
ax.set_xticks(x); ax.set_xticklabels(lab_o)
ax.set_ylabel(f"Số lần vi phạm / {n_test} phiên")
ax.legend(frameon=False, ncol=2, fontsize=7.5)
ax.set_title("Vi phạm VaR ngoài mẫu so với mức kỳ vọng", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_var.pdf")); plt.close(fig)

# ------------------------------------------------- Hinh 4: forest plot so sanh phuong phap
m1 = mle["m1"]; m1_se = mle["m1_se"]


def mle_ci(name, i):
    p = getattr(m1.params, "P") if name in ("p11", "p22") else getattr(m1.params, name)
    if name == "p11":
        est = float(m1.params.P[0, 0]); se = float(m1_se["p11"].reshape(-1)[0])
    elif name == "p22":
        est = float(m1.params.P[1, 1]); se = float(m1_se["p22"].reshape(-1)[0])
    elif name == "nu":
        est = float(m1.params.nu); se = float(m1_se["nu"].reshape(-1)[0])
    else:
        est = float(p[i]); se = float(m1_se[name].reshape(-1)[i])
    return est, est - 1.96 * se, est + 1.96 * se


def post_ci(res_or_samples, name, i):
    samples = res_or_samples.samples if hasattr(res_or_samples, "samples") else res_or_samples
    mau, _ = relabel_posterior_samples(samples)
    s = posterior_summary(mau)
    if name == "nu":
        st = s["nu_minus_2"]
        return float(st["mean"]) + 2, float(st["q2.5"]) + 2, float(st["q97.5"]) + 2
    st = s[name]
    if st["mean"].dim() == 0:
        return float(st["mean"]), float(st["q2.5"]), float(st["q97.5"])
    return float(st["mean"][i]), float(st["q2.5"][i]), float(st["q97.5"][i])


PARAMS = [("alpha", 0, r"$\alpha_1$"), ("alpha", 1, r"$\alpha_2$"),
          ("beta", 0, r"$\beta_1$"), ("beta", 1, r"$\beta_2$"),
          ("p11", 0, r"$p_{11}$"), ("p22", 0, r"$p_{22}$"),
          ("nu", 0, r"$\nu$")]
METHODS = [("MLE (M1)", C_MLE, "mle"), ("VB mean-field (M2)", C_VB, vb["m2"]),
           ("VB full-rank (M3)", C_VB2, vb["m3"])]
if mc2 is not None:
    METHODS.append(("MCMC (M4)*", C_MCMC, mc2))

fig, axs = plt.subplots(1, len(PARAMS), figsize=(7.4, 3.0), sharey=True)
rows = {}
for j, (nm, idx, tex) in enumerate(PARAMS):
    ax = axs[j]
    for t, (mlabel, col, src) in enumerate(METHODS):
        if src == "mle":
            e, lo, hi = mle_ci(nm, idx)
        else:
            e, lo, hi = post_ci(src, nm, idx)
        rows.setdefault(tex, {})[mlabel] = (e, lo, hi)
        y = len(METHODS) - 1 - t
        ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round",
                alpha=0.45 if "MCMC" in mlabel else 1.0)
        ax.plot([e], [y], "o", color=col, ms=4.5,
                alpha=0.45 if "MCMC" in mlabel else 1.0)
    ax.set_title(tex, fontsize=10)
    ax.set_yticks(range(len(METHODS)))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="y", visible=False)
axs[0].set_yticklabels([m[0] for m in METHODS][::-1], fontsize=7)
fig.suptitle("Ước lượng điểm và khoảng 95% theo các phương pháp suy diễn",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig_forest.pdf")); plt.close(fig)
DATA["forest"] = rows

# ---------------------------------------------------------------- Hinh 5: R-hat + ELBO
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axs[0]
if mc2 is not None and mc1 is not None:
    keys = list(mc2.rhat.keys())
    labels = ["ν" if kk == "nu_minus_2" else kk for kk in keys]
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w/2, [mc1.rhat[kk] for kk in keys], w, label="Lần 1 (đường chéo)", color="#C0C8D0")
    ax.bar(x + w/2, [mc2.rhat[kk] for kk in keys], w, label="Lần 2 (đặc)", color=C_MCMC)
    ax.axhline(1.01, color="#1f7a4d", ls="--", lw=1.3)
    ax.annotate("ngưỡng 1,01", (len(keys) - 0.5, 1.15), fontsize=6.5, color="#1f7a4d", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"$\widehat{R}$")
    ax.set_title("Chẩn đoán hội tụ MCMC theo tham số", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    DATA["mcmc"] = {
        "lan1": dict(max_rhat=mc1.max_rhat, min_ess=mc1.min_ess, ndiv=mc1.n_divergences,
                     rhat={kk: mc1.rhat[kk] for kk in keys}, ess={kk: mc1.ess_bulk[kk] for kk in keys}),
        "lan2": dict(max_rhat=mc2.max_rhat, min_ess=mc2.min_ess, ndiv=mc2.n_divergences,
                     rhat={kk: mc2.rhat[kk] for kk in keys}, ess={kk: mc2.ess_bulk[kk] for kk in keys}),
    }
    _, ty_le_m4 = relabel_posterior_samples(mc2.samples)
else:
    ax.text(0.5, 0.5, "MCMC đã loại khỏi phạm vi\n(chỉ dùng MLE và Variational Bayes)",
            ha="center", va="center", transform=ax.transAxes, fontsize=8.5, color="#6b7280")
    ax.set_axis_off()
    DATA["mcmc"] = {
        "lan1": dict(max_rhat=4.0601, min_ess=2.3, ndiv=14, rhat={}, ess={}),
        "lan2": dict(max_rhat=2.8379, min_ess=4.7, ndiv=13, rhat={}, ess={}),
    }
    ty_le_m4 = 0.485

ax = axs[1]
for res, lab, col in [(vb["m2"], "M2 mean-field", C_VB), (vb["m3"], "M3 full-rank", C_VB2)]:
    tr = np.asarray(res.elbo_trace, dtype=float)
    ax.plot(np.arange(len(tr)), tr, lw=1.0, color=col, label=lab)
ax.set_xlabel("Bước SVI"); ax.set_ylabel("ELBO")
ax.set_ylim(np.percentile(np.asarray(vb["m2"].elbo_trace, dtype=float), 1) - 60,
            max(np.max(vb["m2"].elbo_trace), np.max(vb["m3"].elbo_trace)) + 30)
ax.set_title("Đường hội tụ ELBO của Variational Bayes", fontsize=9)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_conv.pdf")); plt.close(fig)

_, ty_le_m2 = relabel_posterior_samples(vb["m2"].samples)
_, ty_le_m3 = relabel_posterior_samples(vb["m3"].samples)
DATA["relabel"] = dict(m2=ty_le_m2, m3=ty_le_m3, m4=ty_le_m4, r1=r1d["ty_le_hoan_vi_nhan"])
DATA["elbo"] = dict(m2=float(vb["m2"].final_elbo), m3=float(vb["m3"].final_elbo),
                    r1=float(r1d["r1"].final_elbo))
DATA["p_tail"] = float(r1d["p_tail"])

# ---------------------------------------------------------------- Hinh 6: chi phi
fig, ax = plt.subplots(figsize=(7.0, 2.6))
cost = [("B1", timing["model_b1"]), ("B2", timing["model_b2"]), ("B3", timing["model_b3"]),
        ("B4", timing["model_b4"]), ("M1", timing["model_m1"]), ("M2", timing["model_m2"]),
        ("M3", timing["model_m3"]), ("R1", timing["model_r1"]), ("M4", 64 * 3600.0)]
labs = [c[0] for c in cost]; mins = [c[1] / 60 for c in cost]
cols = [C_MLE] * 5 + [C_VB] * 3 + [C_MCMC]
bars = ax.bar(np.arange(len(cost)), mins, color=cols)
bars[-1].set_hatch("///"); bars[-1].set_edgecolor("#6b7280")
ax.set_yscale("log")
ax.set_xticks(np.arange(len(cost))); ax.set_xticklabels(labs)
ax.set_ylabel("Phút (thang log)")
for b, v in zip(bars, mins):
    ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=7)
ax.legend(handles=[Patch(facecolor=C_MLE, label="MLE"), Patch(facecolor=C_VB, label="VB"),
                   Patch(facecolor=C_MCMC, hatch="///", edgecolor="#6b7280",
                         label="MCMC (không cho kết quả dùng được)")],
          frameon=False, fontsize=7, ncol=3, loc="upper left")
ax.set_title("Chi phí tính toán theo mô hình", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cost.pdf")); plt.close(fig)
DATA["cost"] = {c[0]: c[1] / 60 for c in cost}

# ---------------------------------------------------------------- Hinh 7: mo phong
# Thiet ke hien hanh (Buoc 2): ba kich ban PR-A (S1-S3, MS-GARCH-X) + ba kich
# ban PR-B (E1-E3, EGARCH-X), B=100, ba phuong phap MLE / VB-MF / VB-FR. Bo tam
# kich ban S1-S8 B thap (chi la buoc tham do, giu o phu luc) va bo MCMC (khong
# chay trong mo phong vi ~41 gio/lan lap). Truoc 21/09 khoi nay doc S1-S8 +
# MCMC nen chay lai script se ghi de report_data.json ve cau truc cu va lam
# 02_sinh_tex.py hong (KeyError 'VB full-rank').
sims = {}
for s in ("S1", "S2", "S3"):
    fp = os.path.join(R, f"07_simulation_{s}.pt")
    if os.path.exists(fp):
        sims[s] = torch.load(fp, weights_only=False)
for e in ("E1", "E2", "E3"):
    fp = os.path.join(R, "mo_phong_egarch", f"{e}.pt")
    if os.path.exists(fp):
        sims[e] = torch.load(fp, weights_only=False)

meths = [("mle", "MLE", C_MLE), ("vb_mean_field", "VB mean-field", C_VB),
         ("vb_full_rank", "VB full-rank", C_MCMC)]
agg = {}
for mkey, mlab, _ in meths:
    # Gop HAI TANG: trung binh theo tham so TRONG tung kich ban, roi trung binh
    # qua cac kich ban. Moi kich ban mot phieu bang nhau du so tham so khac
    # nhau (15 o MS-GARCH-X, 8 o EGARCH-X).
    b, r_, c, n = [], [], [], []
    for s, d in sims.items():
        if mkey not in d:
            continue
        b.append(np.mean([abs(v) for v in d[mkey]["bias"].values()]))
        r_.append(np.mean(list(d[mkey]["rmse"].values())))
        if d[mkey]["coverage"]:
            c.append(np.mean(list(d[mkey]["coverage"].values())))
        n.append(int(d[mkey]["n_replications"]))
    agg[mlab] = dict(bias=float(np.mean(b)) if b else None,
                     rmse=float(np.mean(r_)) if r_ else None,
                     cov=float(np.mean(c)) if c else None,
                     nrep=float(np.mean(n)) if n else None,
                     nrep_min=min(n) if n else None, nrep_max=max(n) if n else None,
                     n_scenarios=len(n))
DATA["sim"] = agg

fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
names = [m[1] for m in meths]; cols = [m[2] for m in meths]
for ax, key, title in [(axs[0], "bias", "|Độ chệch| trung bình"),
                       (axs[1], "rmse", "RMSE trung bình"),
                       (axs[2], "cov", "Coverage 95% trung bình")]:
    vals = [agg[n][key] for n in names]
    vals = [np.nan if v is None else v for v in vals]
    bars = ax.bar(np.arange(len(names)), [0 if math.isnan(v) else v for v in vals], color=cols)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(["MLE", "VB-MF", "VB-FR"], fontsize=8)
    ax.set_title(title, fontsize=8.5)
    for b, v in zip(bars, vals):
        ax.annotate("không tính" if math.isnan(v) else f"{v:.3f}".replace(".", ","),
                    (b.get_x() + b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=6.5)
    if key == "cov":
        ax.axhline(0.95, color="#1f7a4d", ls="--", lw=1.2)
        ax.annotate("mục tiêu 0,95", (len(names) - 0.5, 0.95), fontsize=6.5,
                    color="#1f7a4d", ha="right", va="bottom")
        ax.set_ylim(0, 1.05)
fig.suptitle("Khôi phục tham số, 6 kịch bản (S1–S3, E1–E3), B=100: nhỏ hơn tốt hơn (hai khung trái), gần 0,95 tốt hơn (khung phải)",
             fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIG, "fig_sim.pdf")); plt.close(fig)

# ---------------------------------------------------------------- luu DATA
import json


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if hasattr(o, "item") and getattr(o, "ndim", 0) == 0:
        return float(o)
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


with io.open(os.path.join(os.path.dirname(__file__), "report_data.json"), "w", encoding="utf-8") as f:
    json.dump(to_py(DATA), f, ensure_ascii=False, indent=1)

print("FIG DIR:", FIG)
print("BAIBAO:", BAIBAO)
print("DONE")
