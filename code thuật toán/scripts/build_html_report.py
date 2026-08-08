"""Tong hop ket qua trong thu muc results/ (tao boi run_full_study.py) thanh
mot bao cao HTML don, tu chua (khong phu thuoc CDN), co the mo truc tiep bang
trinh duyet hoac xuat ban qua Artifact.

Script nay CHI DOC cac file .pt da co san; neu mot buoc chua chay xong (file
khong ton tai), phan tuong ung trong bao cao se ghi "chua co du lieu" thay vi
loi.

Chay: python scripts/build_html_report.py [--results-dir results] [--out report.html]
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from msgarchx.evaluation.density_scores import mixture_cdf
from msgarchx.evaluation.dm_test import diebold_mariano_test
from msgarchx.models.ms_garch import label_switch_correction, ms_garch_forward_filter


def _load(results_dir: Path, name: str):
    path = results_dir / f"{name}.pt"
    if not path.exists():
        return None
    return torch.load(path, weights_only=False)


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, nd=4) -> str:
    if x is None:
        return "--"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(x)


def _table(headers: list[str], rows: list[list], caption: str = "") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell if isinstance(cell, str) and cell.startswith('<') else _esc(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return f'<table class="tbl">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _label_corrected(params, h):
    """Ap dung eq:label-identification (trang thai 1 luon la bien dong thap)
    va cho biet co xay ra doi nhan hay khong, de con permute lai SE/xac suat
    trang thai tuong ung cho nhat quan voi bang tham so da hien thi."""
    corrected = label_switch_correction(params, h)
    swapped = not bool(torch.equal(corrected.omega, params.omega))
    return corrected, swapped


def _permute_se(se: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    perm = torch.tensor([1, 0])
    k = se["gamma"].shape[0] // 2
    return {
        "mu": se["mu"][perm], "omega": se["omega"][perm], "alpha": se["alpha"][perm],
        "beta": se["beta"][perm], "gamma": se["gamma"].reshape(2, k)[perm].reshape(-1),
        "p11": se["p22"], "p22": se["p11"], "nu": se["nu"],
    }


def _img_tag(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;display:block;margin:0.6rem 0 1.2rem">'


# ============================================================
# Cac phan bao cao
# ============================================================
def section_overview(data) -> str:
    if data is None:
        return "<p>Chua co du lieu (buoc 'data' chua chay).</p>"
    q = data["quality_report"]
    rows = [[k, v] for k, v in q.items()]
    dates = data["dates_all"]
    info = (
        f"<p>So quan sat hop le: <b>{data['r_all'].shape[0]}</b> "
        f"({dates.iloc[0].date()} &rarr; {dates.iloc[-1].date()}). "
        f"Tap huan luyen: <b>{data['n_train']}</b> quan sat dau; "
        f"tap kiem dinh: <b>{data['r_all'].shape[0] - data['n_train']}</b> quan sat cuoi.</p>"
    )
    return info + _table(["Chi so chat luong du lieu", "Gia tri"], rows)


def _desc_stats_row(name: str, x: np.ndarray) -> list:
    x = x[~np.isnan(x)]
    mean, sd = float(x.mean()), float(x.std())
    skew = float(((x - mean) ** 3).mean() / sd**3) if sd > 0 else float("nan")
    kurt = float(((x - mean) ** 4).mean() / sd**4) if sd > 0 else float("nan")
    return [name, len(x), _fmt(mean), _fmt(sd), _fmt(skew), _fmt(kurt), _fmt(x.min()), _fmt(x.max())]


def section_descriptive_stats(data) -> str:
    if data is None:
        return "<p>Chua co du lieu (buoc 'data' chua chay).</p>"
    r = data["r_all"].numpy()
    rv = data["rv_all"].numpy()
    rows = [
        _desc_stats_row("r_t (loi suat ngay, %)", r),
        _desc_stats_row("RV_t (bien dong thuc te)", rv),
    ]
    return _table(
        ["Bien", "So quan sat", "Trung binh", "Do lech chuan", "Do lech (skew)", "Do nhon (kurtosis)", "Nho nhat", "Lon nhat"],
        rows, caption="Bang 7.1 -- Thong ke mo ta cua loi suat va RV_t",
    )


def section_preliminary(prelim) -> str:
    if prelim is None:
        return "<p>Chua co du lieu (buoc 'prelim' chua chay).</p>"
    rows = []
    for key, res in prelim.items():
        rows.append([
            res.get("test", key), _fmt(res.get("statistic")), _fmt(res.get("p_value")),
            res.get("decision", "--"),
        ])
    return _table(["Kiem dinh", "Thong ke", "p-value", "Quyet dinh"], rows,
                   caption="Bang 5.5 -- Kiem dinh thong ke ban dau")


def _garch_param_row(name, res) -> list:
    p = res.params
    nu = f"{p.nu.item():.3f}" if getattr(p, "nu", None) is not None else "--"
    return [name, _fmt(res.loglik, 2), _fmt(res.aic, 2), _fmt(res.bic, 2),
            _fmt(p.mu.item()), _fmt(p.omega.item()), _fmt(p.alpha.item()), _fmt(p.beta.item()), nu]


def section_mle_baselines(mle) -> str:
    if mle is None:
        return "<p>Chua co du lieu (buoc 'mle' chua chay).</p>"
    rows = [
        _garch_param_row("B1: GARCH chuan", mle["b1"]),
        _garch_param_row("B2: GARCH Student-t", mle["b2"]),
        _garch_param_row("B3: GARCH-X Student-t", mle["b3"]),
    ]
    b1_2_5 = _table(
        ["Mo hinh", "Log-hop-ly", "AIC", "BIC", "mu", "omega", "alpha", "beta", "nu"],
        rows, caption="Bang 7.1 -- Mo hinh doi chung mot trang thai (B1-B3, MLE)",
    )

    b4, m1 = mle["b4"], mle["m1"]
    m1_se = mle.get("m1_se")
    swapped_note = []
    ms_rows = []
    for key, label, res in [("b4", "B4: MS-GARCH chuan", b4), ("m1", "M1: MS-GARCH-X Student-t", m1)]:
        p = res.params
        h = res.filter_output["h"] if res.filter_output is not None else None
        if h is not None:
            p, swapped = _label_corrected(p, h)
            if swapped:
                swapped_note.append(key.upper())
        else:
            swapped = False
        nu = f"{p.nu.item():.3f}" if p.nu.dim() == 0 else str([round(v, 3) for v in p.nu.tolist()])
        ms_rows.append([
            label, _fmt(res.loglik, 2), _fmt(res.aic, 2), _fmt(res.bic, 2),
            str([round(v, 3) for v in p.mu.tolist()]), str([round(v, 3) for v in p.omega.tolist()]),
            str([round(v, 3) for v in p.alpha.tolist()]), str([round(v, 3) for v in p.beta.tolist()]),
            str([[round(v, 3) for v in row] for row in p.P.tolist()]), nu,
        ])
        if key == "m1" and m1_se is not None:
            se = _permute_se(m1_se) if swapped else m1_se
            ms_rows.append([
                "M1: sai so chuan (delta method)", "--", "--", "--",
                str([round(v, 4) for v in se["mu"].tolist()]), str([round(v, 4) for v in se["omega"].tolist()]),
                str([round(v, 4) for v in se["alpha"].tolist()]), str([round(v, 4) for v in se["beta"].tolist()]),
                f"p11={se['p11'].item():.4f}, p22={se['p22'].item():.4f}", _fmt(se["nu"].item(), 4),
            ])
    ms_tbl = _table(
        ["Mo hinh", "Log-hop-ly", "AIC", "BIC", "mu", "omega", "alpha", "beta", "P", "nu"],
        ms_rows, caption="Bang 7.2 -- Mo hinh hai trang thai (B4, M1, MLE; trang thai da chuan hoa theo eq:label-identification)",
    )
    note = ""
    if swapped_note:
        note = f'<p class="note">Da tu dong doi nhan trang thai 1/2 cho {", ".join(swapped_note)} de trang thai 1 luon la bien dong thap (eq:label-identification).</p>'
    if m1_se is None:
        note += '<p class="note">Khong tinh duoc sai so chuan cho M1: Hessian quan sat tai nghiem MLE khong xac dinh duong (xem Muc 4.2 buoc 6 -- can bootstrap thay the).</p>'

    runs_rows = []
    for name in ["b1", "b2", "b3"]:
        for r in mle[name].runs:
            runs_rows.append([name, r.seed, _fmt(r.loglik, 2), r.converged, _fmt(r.grad_norm, 4)])
    for name in ["b4", "m1"]:
        for r in mle[name].runs:
            runs_rows.append([name, r.seed, _fmt(r.loglik, 2), r.converged, _fmt(r.grad_norm, 4)])
    diag_tbl = _table(["Mo hinh", "seed", "Log-hop-ly", "Hoi tu", "||gradient||"], runs_rows,
                       caption="Chan doan cac lan khoi tao MLE (Muc 4.2)")

    return b1_2_5 + ms_tbl + note + diag_tbl


def _vb_summary_rows(res, param_names=("mu", "omega", "alpha", "beta", "p11", "p22")) -> list:
    rows = []
    s = res.summary
    for name in param_names:
        if name not in s:
            continue
        mean, sd = s[name]["mean"], s[name]["sd"]
        if mean.dim() == 0:
            rows.append([name, _fmt(mean.item()), _fmt(sd.item())])
        else:
            for i in range(mean.shape[0]):
                rows.append([f"{name}[{i+1}]", _fmt(mean[i].item()), _fmt(sd[i].item())])
    nu_mean = s["nu_minus_2"]["mean"] + 2.0
    nu_sd = s["nu_minus_2"]["sd"]
    if nu_mean.dim() == 0:
        rows.append(["nu", _fmt(nu_mean.item()), _fmt(nu_sd.item())])
    else:
        for i in range(nu_mean.shape[0]):
            rows.append([f"nu[{i+1}]", _fmt(nu_mean[i].item()), _fmt(nu_sd[i].item())])
    return rows


def section_vb(vb) -> str:
    if vb is None:
        return "<p>Chua co du lieu (buoc 'vb' chua chay).</p>"
    out = f"<p>ELBO cuoi cung (trung binh 50 buoc cuoi) -- Mean-field: <b>{_fmt(vb['m2'].final_elbo, 2)}</b>, Full-rank: <b>{_fmt(vb['m3'].final_elbo, 2)}</b></p>"
    for label, res in [("M2: VB mean-field", vb["m2"]), ("M3: VB full-rank", vb["m3"])]:
        out += _table(["Tham so", "Trung binh hau nghiem", "Do lech chuan"], _vb_summary_rows(res), caption=label)
    return out


def section_mcmc(mcmc) -> str:
    if mcmc is None:
        return "<p>Chua co du lieu (buoc 'mcmc' chua chay).</p>"
    m4 = mcmc["m4"]
    out = (
        f"<p>R-hat lon nhat: <b>{_fmt(m4.max_rhat, 4)}</b> "
        f"(chuan &lt; 1.01) &nbsp;|&nbsp; ESS nho nhat: <b>{_fmt(m4.min_ess, 1)}</b> "
        f"&nbsp;|&nbsp; So phan ky: <b>{m4.n_divergences}</b></p>"
    )
    rhat_rows = [[name, _fmt(v, 4)] for name, v in m4.rhat.items()]
    ess_rows = [[name, _fmt(v, 1)] for name, v in m4.ess_bulk.items()]
    out += _table(["Tham so", "R-hat"], rhat_rows, caption="Chan doan hoi tu R-hat")
    out += _table(["Tham so", "ESS (bulk)"], ess_rows, caption="Co mau hieu dung (ESS)")
    out += _table(["Tham so", "Trung binh hau nghiem", "Do lech chuan"], _vb_summary_rows(m4), caption="M4: MCMC (NUTS)")
    return out


def section_r1(r1_data) -> str:
    if r1_data is None:
        return "<p>Chua co du lieu (buoc 'r1' chua chay).</p>"
    p_tail = r1_data["p_tail"]
    rows = _vb_summary_rows(r1_data["r1"], param_names=("mu", "omega", "alpha", "beta", "p11", "p22"))
    return (
        f"<p>P(nu_2 &lt; nu_1 | Data) &asymp; <b>{_fmt(p_tail, 4)}</b> "
        f"(eq:tail-probability -- xac suat trang thai bien dong cao co duoi day hon).</p>"
        + _table(["Tham so", "Trung binh hau nghiem", "Do lech chuan"], rows, caption="R1: MS-GARCH-X voi nu_s theo trang thai (VB mean-field)")
    )


def section_oos(oos) -> str:
    if oos is None:
        return "<p>Chua co du lieu (buoc 'oos' chua chay).</p>"
    rows = []
    for name, res in oos.items():
        rows.append([
            name.upper(), _fmt(res["mae"]), _fmt(res["rmse"]), _fmt(res["qlike"]), _fmt(res["nls"]),
            res["n_test"],
        ])
    vol_tbl = _table(["Mo hinh", "MAE", "RMSE", "QLIKE", "NLS (diem log am)", "So quan sat OOS"], rows,
                      caption="Bang 7.6 -- Du bao bien dong va mat do ngoai mau")

    var_rows = []
    for name, res in oos.items():
        for label in ["1pct", "5pct"]:
            kup = res.get(f"kupiec_{label}", {})
            chr_ = res.get(f"christoffersen_{label}", {})
            var_rows.append([
                name.upper(), label, res.get(f"n_violations_{label}", "--"),
                _fmt(kup.get("violation_rate")), _fmt(kup.get("p_value")), _fmt(chr_.get("p_value_cc")),
            ])
    var_tbl = _table(
        ["Mo hinh", "Muc", "So vi pham", "Ty le vi pham", "Kupiec p", "Christoffersen (CC) p"],
        var_rows, caption="Bang 7.8 -- Kiem dinh VaR ngoai mau",
    )
    return vol_tbl + var_tbl


_DM_PAIRS = [
    ("b1", "b2"), ("b2", "b3"), ("b3", "m1"),
    ("m1", "m2"), ("m1", "m3"), ("m1", "m4"), ("m2", "m4"), ("m3", "m4"),
]


def section_dm_tests(oos, data) -> str:
    """Kiem dinh Diebold-Mariano (1995) tren mat mat QLIKE, tinh lai tu h_hat
    da luu trong ket qua OOS (khong can chay lai MLE/VB/MCMC)."""
    if oos is None or data is None:
        return "<p>Chua co du lieu (can ca 'oos' va 'data').</p>"
    n_train = data["n_train"]
    rv_test = data["rv_all"][n_train:].numpy()

    rows = []
    for a, b in _DM_PAIRS:
        if a not in oos or b not in oos:
            continue
        ha = oos[a]["h_hat"].numpy()
        hb = oos[b]["h_hat"].numpy()
        loss_a = np.log(ha) + rv_test / ha
        loss_b = np.log(hb) + rv_test / hb
        res = diebold_mariano_test(loss_a, loss_b)
        better = a.upper() if res["mean_loss_diff"] < 0 else b.upper()
        rows.append([a.upper(), b.upper(), _fmt(res["DM_stat"]), _fmt(res["p_value"]), better])
    return _table(
        ["Mo hinh 1", "Mo hinh 2", "DM stat", "p-value", "QLIKE thap hon (tot hon)"],
        rows, caption="Kiem dinh Diebold-Mariano (1995) tren mat mat QLIKE ngoai mau",
    )


def section_figures(data, mle) -> str:
    if data is None:
        return "<p>Chua co du lieu.</p>"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ""
    dates = data["dates_all"]
    r = data["r_all"].numpy()
    rv = data["rv_all"].numpy()

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(dates, r, lw=0.5, color="#1d4ed8")
    axes[0].set_ylabel("$r_t$ (%)")
    axes[0].axhline(0, color="grey", lw=0.4)
    axes[1].plot(dates, rv, lw=0.5, color="#b91c1c")
    axes[1].set_ylabel("$RV_t$")
    fig.suptitle("Loi suat va bien dong thuc te VN30F1M theo thoi gian")
    out += _img_tag(fig)

    if mle is not None and mle.get("m1") is not None and mle["m1"].filter_output is not None:
        m1 = mle["m1"]
        params, swapped = _label_corrected(m1.params, m1.filter_output["h"])
        xi_filt = m1.filter_output["xi_filt"]
        if swapped:
            xi_filt = xi_filt[:, [1, 0]]
        high_vol_prob = xi_filt[:, 1].numpy()
        n_train_dates = dates.iloc[: len(high_vol_prob)]
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.plot(n_train_dates, high_vol_prob, lw=0.7, color="#ea580c")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(trang thai cao)")
        ax.set_title("Xac suat loc cua trang thai bien dong cao -- M1 (tren tap huan luyen)")
        out += _img_tag(fig)

        with torch.no_grad():
            out_full = ms_garch_forward_filter(params, data["r_all"], data["X_all"])
        n_train = data["n_train"]
        xi_pred_test = out_full["xi_pred"][n_train:]
        h_test = out_full["h"][n_train:]
        r_test = data["r_all"][n_train:]
        mu_np = params.mu.numpy()
        nu_np = params.nu.numpy() if params.state_specific_nu else np.full(2, params.nu.item())
        pit_vals = [
            mixture_cdf(float(r_test[t]), mu_np, h_test[t].numpy(), xi_pred_test[t].numpy(), nu_np)
            for t in range(len(r_test))
        ]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.hist(pit_vals, bins=20, range=(0, 1), color="#0d9488", edgecolor="white")
        ax.axhline(len(pit_vals) / 20, color="grey", ls="--", lw=1)
        ax.set_xlabel("PIT $u_t = \\hat F_t(r_t)$")
        ax.set_ylabel("Tan suat")
        ax.set_title("Histogram PIT ngoai mau -- M1 (duong ngang = phan bo deu ky vong)")
        out += _img_tag(fig)

    return out


def section_computation(timing) -> str:
    if timing is None:
        return "<p>Chua co du lieu (chua co file 09_timing.pt).</p>"
    rows = [[name, f"{secs/60:.1f} phut ({secs:.0f}s)"] for name, secs in timing.items()]
    return _table(["Buoc", "Thoi gian chay (khi tinh moi, khong tinh lan nap lai checkpoint)"], rows,
                  caption="Bang 5.9 -- Chi phi tinh toan cac buoc")


def section_simulation(sim) -> str:
    if sim is None:
        return "<p>Chua co du lieu (buoc 'simulation' chua chay).</p>"
    rows = []
    for code, scenario_result in sim.items():
        for method, res in scenario_result.items():
            if not res:
                continue
            avg_bias = sum(abs(v) for v in res["bias"].values()) / max(len(res["bias"]), 1)
            avg_rmse = sum(res["rmse"].values()) / max(len(res["rmse"]), 1)
            avg_cov = (
                sum(res["coverage"].values()) / len(res["coverage"]) if res.get("coverage") else None
            )
            rows.append([
                code, method, _fmt(avg_bias), _fmt(avg_rmse),
                _fmt(avg_cov) if avg_cov is not None else "--", _fmt(res.get("mean_brier")),
            ])
    return _table(
        ["Kich ban", "Phuong phap", "|Bias| trung binh", "RMSE trung binh", "Coverage 95% trung binh", "Brier score"],
        rows, caption="Bang 6.x -- Nghien cuu mo phong S1-S8: Bias/RMSE/Coverage/Brier",
    )


# ============================================================
# Trang HTML
# ============================================================
_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --border: #e5e7eb;
  --accent: #1d4ed8; --tbl-head: #f3f4f6; --tbl-stripe: #fafafa;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --border: #2a2e37;
    --accent: #60a5fa; --tbl-head: #1a1d24; --tbl-stripe: #161920;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --border: #2a2e37;
  --accent: #60a5fa; --tbl-head: #1a1d24; --tbl-stripe: #161920;
}
body { background: var(--bg); color: var(--fg); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 5rem; line-height: 1.55; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.25rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
p.subtitle { color: var(--muted); margin-top: -0.5rem; }
.tbl { border-collapse: collapse; width: 100%; margin: 0.8rem 0 1.6rem; font-size: 0.88rem; overflow-x: auto; display: block; }
.tbl caption { text-align: left; font-weight: 600; margin-bottom: 0.4rem; caption-side: top; }
.tbl th, .tbl td { border: 1px solid var(--border); padding: 0.35rem 0.6rem; text-align: left; white-space: nowrap; }
.tbl thead th { background: var(--tbl-head); position: sticky; top: 0; }
.tbl tbody tr:nth-child(even) { background: var(--tbl-stripe); }
.note { color: var(--muted); font-size: 0.85rem; }
nav.toc { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1rem 0 2rem; }
nav.toc a { color: var(--accent); text-decoration: none; font-size: 0.85rem; border: 1px solid var(--border); padding: 0.25rem 0.6rem; border-radius: 999px; }
"""


def build_report(results_dir: Path) -> str:
    data = _load(results_dir, "00_data")
    prelim = _load(results_dir, "01_preliminary_tests")
    mle = _load(results_dir, "02_mle")
    vb = _load(results_dir, "03_vb")
    mcmc = _load(results_dir, "04_mcmc")
    r1 = _load(results_dir, "05_r1")
    oos = _load(results_dir, "06_oos")
    sim = _load(results_dir, "07_simulation")
    timing = _load(results_dir, "09_timing")

    sections = [
        ("overview", "Tong quan du lieu", section_overview(data)),
        ("desc", "Thong ke mo ta", section_descriptive_stats(data)),
        ("figures", "Bieu do", section_figures(data, mle)),
        ("prelim", "Kiem dinh thong ke ban dau", section_preliminary(prelim)),
        ("mle", "Uoc luong MLE (B1-B4, M1)", section_mle_baselines(mle)),
        ("vb", "Variational Bayes (M2, M3)", section_vb(vb)),
        ("mcmc", "MCMC (M4)", section_mcmc(mcmc)),
        ("r1", "Mo hinh mo rong nu theo trang thai (R1)", section_r1(r1)),
        ("oos", "Danh gia ngoai mau", section_oos(oos)),
        ("dm", "Kiem dinh Diebold-Mariano", section_dm_tests(oos, data)),
        ("sim", "Nghien cuu mo phong (S1-S8)", section_simulation(sim)),
        ("cost", "Chi phi tinh toan", section_computation(timing)),
    ]

    toc = "".join(f'<a href="#{sid}">{_esc(title)}</a>' for sid, title, _ in sections)
    body = "".join(f'<h2 id="{sid}">{_esc(title)}</h2>{content}' for sid, title, content in sections)

    return f"""<title>Ket qua MS-GARCH-X VN30F1M</title>
<style>{_CSS}</style>
<h1>Ket qua uoc luong MS-GARCH-X Student-t cho VN30F1M</h1>
<p class="subtitle">Tong hop tu MLE, Variational Bayes va MCMC tren cung mot ham hop ly. Nguon: {_esc(results_dir)}</p>
<nav class="toc">{toc}</nav>
{body}
<p class="note">Bao cao sinh tu drive scripts/build_html_report.py. Cac bang so hoa dua tren checkpoint trong thu muc results/.</p>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "report.html"))
    args = parser.parse_args()

    html_content = build_report(Path(args.results_dir))
    Path(args.out).write_text(html_content, encoding="utf-8")
    print(f"Da ghi bao cao vao: {args.out}")


if __name__ == "__main__":
    main()
