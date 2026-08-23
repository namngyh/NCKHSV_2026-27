"""Tong hop ket qua trong thu muc results/ (tao boi run_full_study.py) thanh
mot bao cao HTML don, tu chua (khong phu thuoc CDN), co the mo truc tiep bang
trinh duyet hoac xuat ban qua Artifact.

Script nay CHI DOC cac file .pt da co san; neu mot buoc chua chay xong (file
khong ton tai), phan tuong ung trong bao cao se ghi "chua co du lieu" thay vi
loi. Ngoai ra MOI muc deu duoc boc trong `_safe` -- neu mot muc gap loi bat
ngo, bao cao van duoc tao va in traceback ngay tai cho, giong triet ly
`safe_checkpoint` cua run_full_study.py.

Chay: python scripts/build_html_report.py [--results-dir results] [--out report.html]
      python scripts/build_html_report.py --fragment   # bo <html>/<head>, de dan vao Artifact
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from msgarchx.evaluation.density_scores import mixture_cdf
from msgarchx.evaluation.dm_test import diebold_mariano_test
from msgarchx.inference.summary import sd_ratio
from msgarchx.models.ms_garch import (
    expected_duration,
    label_switch_correction,
    ms_garch_forward_filter,
    stationary_distribution,
)


# ============================================================
# Tien ich chung
# ============================================================
class Raw(str):
    """Chuoi da la HTML hop le, KHONG escape lai khi do vao bang."""


def _load(results_dir: Path, name: str):
    """Nap mot checkpoint .pt. Tra ve None neu chua co; neu file hong thi bao
    loi ra stderr va cung tra ve None (de khong lam hong ca bao cao)."""
    path = results_dir / f"{name}.pt"
    if not path.exists():
        return None
    try:
        return torch.load(path, weights_only=False)
    except Exception as exc:  # file hong / cat ngang khi dang ghi
        print(f"[canh bao] Khong doc duoc {path}: {exc}", file=sys.stderr)
        return None


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, nd=4) -> str:
    """Dinh dang so; None va gia tri khong huu han (NaN/Inf) deu hien '--'."""
    if x is None:
        return "--"
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            return _esc([round(float(v), nd) for v in x.reshape(-1).tolist()])
        x = x.item()
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _esc(x)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{nd}f}"


def _num(x) -> float | None:
    """Doi sang float neu la mot so huu han, nguoc lai tra ve None (dung de so
    sanh/xep hang an toan khi mot chi so bi thieu hoac la NaN)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            return None
        x = x.item()
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _vec(x, nd=3) -> str:
    """Vector/ma tran torch -> chuoi gon cho o bang."""
    if x is None:
        return "--"
    t = x.detach() if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.dim() == 0:
        return _fmt(t.item(), nd)
    if t.dim() == 1:
        return "[" + ", ".join(_fmt(v, nd) for v in t.tolist()) + "]"
    return "[" + "; ".join("[" + ", ".join(_fmt(v, nd) for v in row) + "]" for row in t.tolist()) + "]"


def _table(headers: list[str], rows: list[list], caption: str = "") -> str:
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="empty">(khong co dong du lieu)</td></tr>'
    else:
        body = "".join(
            "<tr>"
            + "".join(f"<td>{cell if isinstance(cell, Raw) else _esc(cell)}</td>" for cell in row)
            + "</tr>"
            for row in rows
        )
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (
        f'<div class="twrap"><table class="tbl">{cap}'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _safe(title: str, fn, *args, **kwargs) -> str:
    """Chay mot ham dung bao cao; neu loi thi tra ve khoi loi thay vi lam
    sap toan bo script (bao cao van duoc tao voi cac muc con lai)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        tb = traceback.format_exc()
        print(f"[LOI] Muc '{title}' that bai:\n{tb}", file=sys.stderr)
        return (
            f'<div class="err"><b>Muc nay gap loi khi dung bao cao</b>'
            f"<pre>{_esc(tb)}</pre></div>"
        )


_MPL_READY = False


def _mpl():
    """Nap matplotlib o che do Agg (khong can man hinh), chi mot lan."""
    global _MPL_READY
    import matplotlib

    if not _MPL_READY:
        matplotlib.use("Agg")
        _MPL_READY = True
    import matplotlib.pyplot as plt

    return plt


def _img_tag(fig) -> str:
    plt = _mpl()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f'<img alt="bieu do" src="data:image/png;base64,{b64}" '
        'style="max-width:100%;height:auto;display:block;margin:0.6rem 0 1.2rem">'
    )


# ============================================================
# Nhan tham so / hieu chinh doi nhan trang thai
# ============================================================
def _exog_names(data, n_exog: int) -> list[str]:
    """Ten cac bien ngoai sinh (lay tu thong ke chuan hoa da luu trong buoc
    'data'); rut ve x1..xk neu khong khop."""
    if data is not None:
        mean = data.get("exog_mean")
        if mean is not None and len(getattr(mean, "index", [])) == n_exog:
            return [str(c) for c in mean.index]
    return [f"x{j+1}" for j in range(n_exog)]


def _labels_for(name: str, shape: tuple, exog_names: list[str] | None = None) -> list[str]:
    """Nhan cho tung phan tu cua mot tham so co shape bat ky."""
    if len(shape) == 0:
        return [name]
    if len(shape) == 1:
        return [f"{name}[{i+1}]" for i in range(shape[0])]
    if len(shape) == 2:
        n_state, k = shape
        cols = exog_names if exog_names and len(exog_names) == k else [f"x{j+1}" for j in range(k)]
        return [f"{name}[tt{s+1}, {cols[j]}]" for s in range(n_state) for j in range(k)]
    return [f"{name}[{i}]" for i in range(int(np.prod(shape)))]


def _label_corrected(params, h):
    """Ap dung eq:label-identification (trang thai 1 luon la bien dong thap)
    va cho biet co xay ra doi nhan hay khong, de con permute lai SE/xac suat
    trang thai tuong ung cho nhat quan voi bang tham so da hien thi.

    `label_switch_correction` tra ve CHINH doi tuong params khi khong can doi,
    nen kiem tra bang `is` la chinh xac tuyet doi (khong phu thuoc gia tri)."""
    corrected = label_switch_correction(params, h)
    return corrected, corrected is not params


def _permute_se(se: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Hoan vi sai so chuan theo cung phep doi nhan trang thai 1<->2."""
    perm = torch.tensor([1, 0])
    k = se["gamma"].shape[0] // 2
    gamma = se["gamma"].reshape(2, k)[perm].reshape(-1) if k > 0 else se["gamma"]
    return {
        "mu": se["mu"][perm], "omega": se["omega"][perm], "alpha": se["alpha"][perm],
        "beta": se["beta"][perm], "gamma": gamma,
        "p11": se["p22"], "p22": se["p11"], "nu": se["nu"],
    }


# ============================================================
# Cac phan bao cao
# ============================================================
def section_overview(data) -> str:
    if data is None:
        return "<p>Chua co du lieu (buoc 'data' chua chay).</p>"
    q = data["quality_report"]
    rows = [[k, Raw(_fmt(v, 4)) if isinstance(v, float) else v] for k, v in q.items()]
    dates = data["dates_all"]
    n = data["r_all"].shape[0]
    n_train = data["n_train"]
    info = (
        f"<p>So quan sat hop le: <b>{n}</b> "
        f"({dates.iloc[0].date()} &rarr; {dates.iloc[-1].date()}). "
        f"Tap huan luyen: <b>{n_train}</b> quan sat dau "
        f"({dates.iloc[0].date()} &rarr; {dates.iloc[n_train - 1].date()}); "
        f"tap kiem dinh: <b>{n - n_train}</b> quan sat cuoi "
        f"({dates.iloc[n_train].date()} &rarr; {dates.iloc[-1].date()}).</p>"
    )
    exog = _exog_names(data, data["X_all"].shape[1])
    info += f'<p class="note">Bien ngoai sinh X_(t-1): {", ".join(exog)} (chuan hoa bang thong ke tap huan luyen).</p>'
    return info + _table(["Chi so chat luong du lieu", "Gia tri"], rows)


def _desc_stats_row(name: str, x: np.ndarray) -> list:
    x = x[np.isfinite(x)]
    mean, sd = float(x.mean()), float(x.std())
    skew = float(((x - mean) ** 3).mean() / sd**3) if sd > 0 else float("nan")
    kurt = float(((x - mean) ** 4).mean() / sd**4) if sd > 0 else float("nan")
    return [name, len(x), _fmt(mean), _fmt(sd), _fmt(skew), _fmt(kurt), _fmt(x.min()), _fmt(x.max())]


def section_descriptive_stats(data) -> str:
    if data is None:
        return "<p>Chua co du lieu (buoc 'data' chua chay).</p>"
    r = data["r_all"].numpy()
    rv = data["rv_all"].numpy()
    n_train = data["n_train"]
    rows = [
        _desc_stats_row("r_t (loi suat ngay, %)", r),
        _desc_stats_row("r_t -- tap huan luyen", r[:n_train]),
        _desc_stats_row("r_t -- tap kiem dinh", r[n_train:]),
        _desc_stats_row("RV_t (bien dong thuc te)", rv),
        _desc_stats_row("RV_t -- tap huan luyen", rv[:n_train]),
        _desc_stats_row("RV_t -- tap kiem dinh", rv[n_train:]),
    ]
    return _table(
        ["Bien", "So quan sat", "Trung binh", "Do lech chuan", "Do lech (skew)",
         "Do nhon (kurtosis)", "Nho nhat", "Lon nhat"],
        rows, caption="Bang 7.1 -- Thong ke mo ta cua loi suat va RV_t",
    )


_PRELIM_LABEL = {
    "adf_r": "ADF tren r_t",
    "pp_r": "Phillips-Perron tren r_t",
    "kpss_r": "KPSS tren r_t",
    "ljung_box_r": "Ljung-Box Q(m) tren r_t",
    "ljung_box_r2": "Ljung-Box Q(m) tren r_t^2",
    "arch_lm": "ARCH-LM tren phan du",
    "jarque_bera": "Jarque-Bera tren r_t",
}


def section_preliminary(prelim) -> str:
    """Luu y: hai kiem dinh Ljung-Box (tren r_t va tren r_t^2) cung co truong
    'test' = 'Ljung-Box', nen phai dung KHOA cua dict de phan biet, neu khong
    bang se co hai dong trung ten khong the doc duoc."""
    if prelim is None:
        return "<p>Chua co du lieu (buoc 'prelim' chua chay).</p>"
    rows = []
    for key, res in prelim.items():
        label = _PRELIM_LABEL.get(key, res.get("test", key))
        lags = res.get("lags", res.get("used_lag"))
        rows.append([
            label, res.get("test", "--"), lags if lags is not None else "--",
            _fmt(res.get("statistic")), _fmt(res.get("p_value")), res.get("decision", "--"),
        ])
    return _table(["Kiem dinh", "Loai", "Do tre", "Thong ke", "p-value", "Quyet dinh"], rows,
                  caption="Bang 5.5 -- Kiem dinh thong ke ban dau")


def _garch_param_row(name, res, exog_names) -> list:
    p = res.params
    nu = _fmt(p.nu.item(), 3) if getattr(p, "nu", None) is not None else "--"
    gamma = p.gamma
    if gamma is None or gamma.numel() == 0:
        gamma_s = "--"
    else:
        vals = gamma.reshape(-1).tolist()
        names = exog_names if len(exog_names) == len(vals) else [f"x{j+1}" for j in range(len(vals))]
        gamma_s = ", ".join(f"{nm}={_fmt(v)}" for nm, v in zip(names, vals))
    persistence = float(p.alpha.item() + p.beta.item())
    return [name, _fmt(res.loglik, 2), _fmt(res.aic, 2), _fmt(res.bic, 2),
            _fmt(p.mu.item()), _fmt(p.omega.item()), _fmt(p.alpha.item()), _fmt(p.beta.item()),
            _fmt(persistence), gamma_s, nu]


def section_mle_baselines(mle, data) -> str:
    if mle is None:
        return "<p>Chua co du lieu (buoc 'mle' chua chay).</p>"
    exog = _exog_names(data, mle["config_m1"].n_exog if "config_m1" in mle else 0)

    rows = [
        _garch_param_row("B1: GARCH chuan", mle["b1"], []),
        _garch_param_row("B2: GARCH Student-t", mle["b2"], []),
        _garch_param_row("B3: GARCH-X Student-t", mle["b3"], exog),
    ]
    single_tbl = _table(
        ["Mo hinh", "Log-hop-ly", "AIC", "BIC", "mu", "omega", "alpha", "beta",
         "alpha+beta", "gamma", "nu"],
        rows, caption="Bang 7.1 -- Mo hinh doi chung mot trang thai (B1-B3, MLE)",
    )

    b4, m1 = mle["b4"], mle["m1"]
    m1_se = mle.get("m1_se")
    swapped_note, ms_rows, derived_rows = [], [], []

    for key, label, res in [("b4", "B4: MS-GARCH chuan", b4), ("m1", "M1: MS-GARCH-X Student-t", m1)]:
        p = res.params
        h = res.filter_output["h"] if res.filter_output is not None else None
        swapped = False
        if h is not None:
            p, swapped = _label_corrected(p, h)
            if swapped:
                swapped_note.append(key.upper())
        # B4 duoc fit voi nu co dinh = 1e6 (xap xi phan phoi chuan) nen khong
        # phai tham so uoc luong -- ghi ro thay vi in ra so 1000000.000.
        nu_s = "co dinh ~ chuan" if key == "b4" else _vec(p.nu, 3)
        gamma_s = "--" if p.gamma.numel() == 0 else _vec(p.gamma, 4)
        ms_rows.append([
            label, _fmt(res.loglik, 2), _fmt(res.aic, 2), _fmt(res.bic, 2),
            _vec(p.mu), _vec(p.omega), _vec(p.alpha), _vec(p.beta),
            gamma_s, _vec(p.P), nu_s,
        ])

        pi = stationary_distribution(p.P)
        dur = expected_duration(p.P)
        derived_rows.append([
            label, _vec(p.alpha + p.beta), _vec(pi), _vec(dur, 1),
            _fmt(p.P[0, 0].item()), _fmt(p.P[1, 1].item()),
        ])

        if key == "m1" and m1_se is not None:
            se = _permute_se(m1_se) if swapped else m1_se
            ms_rows.append([
                "M1: sai so chuan (delta method)", "--", "--", "--",
                _vec(se["mu"], 4), _vec(se["omega"], 4), _vec(se["alpha"], 4), _vec(se["beta"], 4),
                _vec(se["gamma"].reshape(p.gamma.shape) if se["gamma"].numel() else se["gamma"], 4),
                f"p11={_fmt(se['p11'].reshape(-1)[0].item())}, p22={_fmt(se['p22'].reshape(-1)[0].item())}",
                _fmt(se["nu"].reshape(-1)[0].item(), 4),
            ])

    ms_tbl = _table(
        ["Mo hinh", "Log-hop-ly", "AIC", "BIC", "mu", "omega", "alpha", "beta", "gamma", "P", "nu"],
        ms_rows,
        caption="Bang 7.2 -- Mo hinh hai trang thai (B4, M1, MLE; trang thai da chuan hoa theo eq:label-identification)",
    )
    derived_tbl = _table(
        ["Mo hinh", "alpha+beta (do dai dai bien dong)", "Phan bo dung (eq:stationary-markov)",
         "Thoi luong ky vong, ngay (eq:expected-duration)", "p11", "p22"],
        derived_rows, caption="Dai luong suy ra tu tham so hai trang thai",
    )

    note = ""
    if swapped_note:
        note = (f'<p class="note">Da tu dong doi nhan trang thai 1/2 cho {", ".join(swapped_note)} '
                "de trang thai 1 luon la bien dong thap (eq:label-identification); sai so chuan cua "
                "M1 duoc hoan vi tuong ung.</p>")
    if m1_se is None:
        note += ('<p class="note">Khong tinh duoc sai so chuan cho M1: Hessian quan sat tai nghiem MLE '
                 "khong xac dinh duong (xem Muc 4.2 buoc 6 -- can bootstrap thay the).</p>")

    runs_rows = []
    for name in ["b1", "b2", "b3", "b4", "m1"]:
        res = mle.get(name)
        if res is None:
            continue
        for r in res.runs:
            runs_rows.append([
                name.upper(), r.seed, _fmt(r.loglik, 2),
                "co" if r.converged else "KHONG", _fmt(r.grad_norm, 4),
                ("co" if r.near_boundary else "khong") if hasattr(r, "near_boundary") else "--",
            ])
    diag_tbl = _table(
        ["Mo hinh", "seed", "Log-hop-ly", "Hoi tu", "||gradient||", "Gan bien"],
        runs_rows, caption="Chan doan cac lan khoi tao MLE (Muc 4.2)",
    )
    spread_note = (
        '<p class="note">Chenh lech log-hop-ly giua cac diem khoi tao cua cung mot mo hinh cho biet '
        "muc do da dinh cua ham hop ly. Neu con chenh lech dang ke, hoac cot \"Hoi tu\" ghi KHONG "
        "va ||gradient|| lon, thi nghiem MLE CHUA dang tin cay: can tang mle_n_starts / mle_n_adam / "
        "mle_n_lbfgs trong StudyConfig cua run_full_study.py roi xoa 02_mle.pt de chay lai buoc nay.</p>"
    )
    return single_tbl + ms_tbl + derived_tbl + note + diag_tbl + spread_note


# ------------------------------------------------------------
# Bang hau nghiem dung chung cho VB (M2, M3), MCMC (M4) va R1
# ------------------------------------------------------------
_POSTERIOR_ORDER = ["mu", "omega", "alpha", "beta", "gamma", "p11", "p22"]


def _summary_da_sap_xep(res):
    """Tom tat hau nghiem SAU khi sap xep nhan trang thai (eq:label-identification).

    Mau hau nghiem cua VB/MCMC khong duoc sap xep o bat ky dau trong pipeline,
    nen bang tham so co the in trang thai 1/2 dao nhau. Tra ve
    (summary, ty_le_mau_bi_hoan_vi).
    """
    from msgarchx.inference.summary import posterior_summary, relabel_posterior_samples
    if not hasattr(res, "samples") or not res.samples:
        return res.summary, 0.0
    mau, ty_le = relabel_posterior_samples(res.samples)
    if ty_le == 0.0:
        return res.summary, 0.0
    return posterior_summary(mau), ty_le


def _posterior_rows(res, exog_names: list[str] | None = None) -> list[list]:
    """Bang hau nghiem day du: trung binh, do lech chuan va KTC 95%
    (tab:parameter-template). Bao gom CA gamma -- he so bien ngoai sinh, tham
    so trung tam cua mo hinh GARCH-X -- va nu doi lai tu nu_minus_2."""
    s, _ty_le = _summary_da_sap_xep(res)
    rows = []

    def add(name: str, stat: dict, shift: float = 0.0):
        mean = stat["mean"] + shift
        labels = _labels_for(name, tuple(mean.shape), exog_names)
        flat = {k: stat[k].reshape(-1) for k in ("sd", "q2.5", "q97.5") if k in stat}
        mean_f = mean.reshape(-1)
        for i, lab in enumerate(labels):
            rows.append([
                lab, _fmt(mean_f[i].item()),
                _fmt(flat["sd"][i].item()) if "sd" in flat else "--",
                _fmt(flat["q2.5"][i].item() + shift) if "q2.5" in flat else "--",
                _fmt(flat["q97.5"][i].item() + shift) if "q97.5" in flat else "--",
            ])

    for name in _POSTERIOR_ORDER:
        if name in s:
            add(name, s[name])
    if "nu_minus_2" in s:
        add("nu", s["nu_minus_2"], shift=2.0)
    return rows


_POSTERIOR_HEADERS = ["Tham so", "Trung binh hau nghiem", "Do lech chuan", "KTC 2.5%", "KTC 97.5%"]


def section_vb(vb, mle, data) -> str:
    if vb is None:
        return "<p>Chua co du lieu (buoc 'vb' chua chay).</p>"
    exog = _exog_names(data, mle["config_m1"].n_exog if mle else 0)
    out = (
        f"<p>ELBO cuoi cung (trung binh 50 buoc cuoi) -- Mean-field: "
        f"<b>{_fmt(vb['m2'].final_elbo, 2)}</b>, Full-rank: <b>{_fmt(vb['m3'].final_elbo, 2)}</b></p>"
    )
    for label, res in [("M2: VB mean-field", vb["m2"]), ("M3: VB full-rank", vb["m3"])]:
        out += _table(_POSTERIOR_HEADERS, _posterior_rows(res, exog), caption=label)
    return out


def section_mcmc(mcmc, mle, data) -> str:
    if mcmc is None:
        return "<p>Chua co du lieu (buoc 'mcmc' chua chay).</p>"
    m4 = mcmc["m4"]
    exog = _exog_names(data, mle["config_m1"].n_exog if mle else 0)
    out = (
        f"<p>R-hat lon nhat: <b>{_fmt(m4.max_rhat, 4)}</b> "
        f"(chuan &lt; 1.01) &nbsp;|&nbsp; ESS nho nhat: <b>{_fmt(m4.min_ess, 1)}</b> "
        f"&nbsp;|&nbsp; So phan ky: <b>{m4.n_divergences}</b></p>"
    )
    if not np.isfinite(m4.max_rhat):
        out += ('<p class="warn">Khong tinh duoc R-hat/ESS (thuong do so chuoi hoac so mau qua nho, '
                "vd. cau hinh dry-run). Chan doan hoi tu chi co y nghia voi >= 4 chuoi va du so mau.</p>")
    elif m4.max_rhat >= 1.01:
        out += ('<p class="warn">R-hat lon nhat >= 1.01: cac chuoi CHUA hoi tu dat chuan '
                "(Vehtari et al. 2021) -- can tang so mau/warmup truoc khi dung ket qua M4.</p>")
    if m4.n_divergences > 0:
        out += (f'<p class="warn">Co {m4.n_divergences} buoc phan ky (divergent transitions): hau nghiem '
                "co vung hinh hoc kho lay mau, nen tang target_accept_prob hoac tai tham so hoa.</p>")

    out += _table(["Tham so", "R-hat"], [[n, _fmt(v, 4)] for n, v in m4.rhat.items()],
                  caption="Chan doan hoi tu R-hat (lay max tren cac thanh phan cua moi tham so)")
    out += _table(["Tham so", "ESS (bulk)"], [[n, _fmt(v, 1)] for n, v in m4.ess_bulk.items()],
                  caption="Co mau hieu dung (ESS, lay min tren cac thanh phan)")
    out += _table(_POSTERIOR_HEADERS, _posterior_rows(m4, exog), caption="M4: MCMC (NUTS)")
    return out


def section_sd_ratio(vb, mcmc, mle, data) -> str:
    """R_sd,j = sd_VB / sd_MCMC (eq:sd-ratio): gia tri < 1 nghia la VB danh gia
    THAP phuong sai hau nghiem so voi MCMC -- ket qua trung tam khi so sanh hai
    phuong phap suy dien trong bai bao."""
    if vb is None or mcmc is None:
        return "<p>Chua co du lieu (can ca 'vb' va 'mcmc').</p>"
    exog = _exog_names(data, mle["config_m1"].n_exog if mle else 0)
    m4 = mcmc["m4"]
    # PHAI sap xep nhan trang thai cho CA HAI ben truoc khi lap ty so: neu
    # nhan cua VB va cua MCMC khac nhau thi R_sd,j se so trang thai bien dong
    # THAP cua ben nay voi trang thai CAO cua ben kia -- vo nghia.
    m4_summary, m4_ty_le = _summary_da_sap_xep(m4)
    canh_bao = []
    if m4_ty_le > 0.2:
        canh_bao.append(f"M4: {m4_ty_le:.1%} mau bi hoan vi nhan")
    rows = []
    for label, res in [("M2 (mean-field)", vb["m2"]), ("M3 (full-rank)", vb["m3"])]:
        res_summary, ty_le = _summary_da_sap_xep(res)
        if ty_le > 0.2:
            canh_bao.append(f"{label}: {ty_le:.1%} mau bi hoan vi nhan")
        ratios = sd_ratio(res_summary, m4_summary)
        for name in _POSTERIOR_ORDER + ["nu_minus_2"]:
            if name not in ratios:
                continue
            disp = "nu" if name == "nu_minus_2" else name
            val = ratios[name]
            labels = _labels_for(disp, tuple(val.shape), exog)
            flat = val.reshape(-1)
            for i, lab in enumerate(labels):
                rows.append([label, lab, _fmt(flat[i].item(), 3)])
    ghi_chu = ""
    if canh_bao:
        ghi_chu = ('<p class="note"><b>Luu y:</b> hau nghiem da duoc sap xep lai '
                   'nhan trang thai theo eq:label-identification truoc khi lap ty so. '
                   'Ty le mau phai hoan vi: ' + "; ".join(canh_bao) +
                   '. Ty le gan 50% nghia la hau nghiem doi nhan qua lai giua cac mau '
                   '(label switching), khong chi la mot phep hoan vi toan cuc -- can '
                   'neu ro khi bao cao.</p>')
    return (ghi_chu + (
        "<p>Ty so do lech chuan R_sd,j = sd_VB(theta_j) / sd_MCMC(theta_j) (eq:sd-ratio). "
        "Gia tri < 1 cho thay Variational Bayes danh gia thap do bat dinh hau nghiem; "
        "chi nen doc bang nay sau khi M4 dat chuan hoi tu.</p>"
        + _table(["Guide VB", "Tham so", "R_sd = sd_VB / sd_MCMC"], rows,
                 caption="So sanh do bat dinh hau nghiem: VB so voi MCMC (eq:sd-ratio)")
    ))


def section_r1(r1_data, mle, data) -> str:
    if r1_data is None:
        return "<p>Chua co du lieu (buoc 'r1' chua chay).</p>"
    exog = _exog_names(data, mle["config_m1"].n_exog if mle else 0)
    p_tail = r1_data["p_tail"]
    return (
        f"<p>P(nu_2 &lt; nu_1 | Data) &asymp; <b>{_fmt(p_tail, 4)}</b> "
        "(eq:tail-probability -- xac suat trang thai bien dong cao co duoi day hon "
        "trang thai bien dong thap).</p>"
        + _table(_POSTERIOR_HEADERS, _posterior_rows(r1_data["r1"], exog),
                 caption="R1: MS-GARCH-X voi nu_s theo trang thai (VB mean-field)")
    )


_MODEL_ORDER = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4"]


def _ordered(oos) -> list[str]:
    return [m for m in _MODEL_ORDER if m in oos] + [m for m in oos if m not in _MODEL_ORDER]


def section_oos(oos) -> str:
    if oos is None:
        return "<p>Chua co du lieu (buoc 'oos' chua chay).</p>"
    names = _ordered(oos)
    metrics = ("mae", "rmse", "qlike", "nls")
    # Chi so tot nhat chi lay tren cac gia tri so huu han, de mot mo hinh loi
    # (NaN/thieu chi so) khong lam hong ca bang.
    best = {}
    for k in metrics:
        vals = [v for v in (_num(oos[n].get(k)) for n in names) if v is not None]
        best[k] = min(vals) if vals else None

    rows = []
    for name in names:
        res = oos[name]
        cells = []
        for k in metrics:
            val = _num(res.get(k))
            txt = _fmt(res.get(k))
            cells.append(Raw(f"<b>{txt}</b>") if val is not None and val == best[k] else txt)
        rows.append([name.upper(), *cells, res.get("n_test", "--"), res.get("n_draws_used", "--")])
    vol_tbl = _table(
        ["Mo hinh", "MAE", "RMSE", "QLIKE", "NLS (diem log am)", "So quan sat OOS", "So mau hau nghiem"],
        rows, caption="Bang 7.6 -- Du bao bien dong va mat do ngoai mau (in dam = tot nhat moi cot)",
    )

    var_rows = []
    for name in names:
        res = oos[name]
        for label, target in [("1pct", 0.01), ("5pct", 0.05)]:
            kup = res.get(f"kupiec_{label}", {})
            chr_ = res.get(f"christoffersen_{label}", {})
            var_rows.append([
                name.upper(), label, res.get(f"n_violations_{label}", "--"),
                _fmt(kup.get("violation_rate")), _fmt(target, 4),
                _fmt(kup.get("p_value")), _fmt(chr_.get("p_value_cc")),
            ])
    var_tbl = _table(
        ["Mo hinh", "Muc", "So vi pham", "Ty le vi pham", "Ty le ky vong",
         "Kupiec p", "Christoffersen (CC) p"],
        var_rows, caption="Bang 7.8 -- Kiem dinh VaR ngoai mau",
    )
    note = ('<p class="note">p-value nho (&lt; 0.05) nghia la BAC BO tinh du bao VaR dung chuan: '
            "Kupiec kiem tra dung ty le vi pham, Christoffersen (CC) kiem tra dong thoi ty le va "
            "tinh doc lap theo thoi gian cua cac vi pham.</p>")
    return vol_tbl + var_tbl + note


_DM_PAIRS = [
    ("b1", "b2"), ("b2", "b3"), ("b3", "m1"), ("b4", "m1"),
    ("m1", "m2"), ("m1", "m3"), ("m1", "m4"), ("m2", "m4"), ("m3", "m4"),
]


def section_dm_tests(oos, data) -> str:
    """Kiem dinh Diebold-Mariano (1995) tren mat mat QLIKE, tinh lai tu h_hat
    da luu trong ket qua OOS (khong can chay lai MLE/VB/MCMC).

    QLIKE_t = log(h_t) + RV_t/h_t, dung dung cong thuc trong
    msgarchx.evaluation.volatility_loss.qlike."""
    if oos is None or data is None:
        return "<p>Chua co du lieu (can ca 'oos' va 'data').</p>"
    n_train = data["n_train"]
    rv_test = data["rv_all"][n_train:].numpy()

    rows = []
    for a, b in _DM_PAIRS:
        if a not in oos or b not in oos:
            continue
        h_a, h_b = oos[a].get("h_hat"), oos[b].get("h_hat")
        if h_a is None or h_b is None:
            continue
        ha, hb = h_a.numpy(), h_b.numpy()
        if ha.shape != rv_test.shape or hb.shape != rv_test.shape:
            continue
        loss_a = np.log(ha) + rv_test / ha
        loss_b = np.log(hb) + rv_test / hb
        res = diebold_mariano_test(loss_a, loss_b)
        # d_t = L_a - L_b; mean < 0 => mo hinh a co mat mat thap hon (tot hon).
        better = a.upper() if res["mean_loss_diff"] < 0 else b.upper()
        sig = "co" if res["p_value"] < 0.05 else "khong"
        rows.append([a.upper(), b.upper(), _fmt(res["mean_loss_diff"]), _fmt(res["DM_stat"]),
                     _fmt(res["p_value"]), better, sig])
    tbl = _table(
        ["Mo hinh 1", "Mo hinh 2", "Chenh lech QLIKE trung binh", "DM stat", "p-value",
         "QLIKE thap hon", "Khac biet co y nghia (5%)"],
        rows, caption="Kiem dinh Diebold-Mariano (1995) tren mat mat QLIKE ngoai mau",
    )
    return tbl + (
        '<p class="note">Luu y dien giai: DM duoc thiet ke cho hai du bao KHONG long nhau. '
        "Cac cap trong cung mot mo hinh nhung khac phuong phap suy dien (M1 so voi M2/M3/M4) nen "
        "duoc doc nhu so sanh mo ta giua cac du bao, khong phai kiem dinh chinh thuc.</p>"
    )


def section_figures(data, mle, vb) -> str:
    if data is None:
        return "<p>Chua co du lieu.</p>"
    plt = _mpl()

    out = ""
    dates = data["dates_all"]
    r = data["r_all"].numpy()
    rv = data["rv_all"].numpy()
    n_train = data["n_train"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(dates, r, lw=0.5, color="#1d4ed8")
    axes[0].set_ylabel("$r_t$ (%)")
    axes[0].axhline(0, color="grey", lw=0.4)
    axes[1].plot(dates, rv, lw=0.5, color="#b91c1c")
    axes[1].set_ylabel("$RV_t$")
    for ax in axes:
        ax.axvline(dates.iloc[n_train], color="black", ls="--", lw=0.8)
    axes[0].set_title("(duong dut = ranh gioi tap huan luyen / kiem dinh)", fontsize=9, color="grey")
    fig.suptitle("Loi suat va bien dong thuc te VN30F1M theo thoi gian")
    out += _img_tag(fig)

    if mle is not None and mle.get("m1") is not None and mle["m1"].filter_output is not None:
        m1 = mle["m1"]
        params, swapped = _label_corrected(m1.params, m1.filter_output["h"])
        xi_filt = m1.filter_output["xi_filt"]
        if swapped:
            xi_filt = xi_filt[:, [1, 0]]
        high_vol_prob = xi_filt[:, 1].numpy()
        train_dates = dates.iloc[: len(high_vol_prob)]

        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.plot(train_dates, high_vol_prob, lw=0.7, color="#ea580c")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(trang thai cao)")
        ax.set_title("Xac suat loc cua trang thai bien dong cao -- M1 (tren tap huan luyen)")
        out += _img_tag(fig)

        with torch.no_grad():
            out_full = ms_garch_forward_filter(params, data["r_all"], data["X_all"])
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

    if vb is not None:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        for label, res, color in [("M2 mean-field", vb["m2"], "#1d4ed8"), ("M3 full-rank", vb["m3"], "#b91c1c")]:
            trace = np.asarray(res.elbo_trace, dtype=float)
            if trace.size:
                ax.plot(trace, lw=0.8, color=color, label=label)
        ax.set_xlabel("Buoc SVI")
        ax.set_ylabel("ELBO")
        ax.set_title("Duong hoi tu ELBO (eq:elbo) -- kiem tra VB da on dinh chua")
        ax.legend(fontsize=8)
        out += _img_tag(fig)

    return out


def section_computation(timing) -> str:
    if timing is None:
        return "<p>Chua co du lieu (chua co file 09_timing.pt).</p>"
    rows = [[name, _fmt(secs / 60, 1), _fmt(secs, 0)] for name, secs in sorted(timing.items())]
    total = float(sum(timing.values()))
    rows.append([Raw("<b>Tong cong</b>"), Raw(f"<b>{_fmt(total/60, 1)}</b>"), Raw(f"<b>{_fmt(total, 0)}</b>")])
    return _table(["Buoc", "Phut", "Giay"], rows,
                  caption="Bang 5.9 -- Chi phi tinh toan (chi tinh lan tinh MOI, khong ke lan nap lai checkpoint)")


def section_simulation(sim) -> str:
    if sim is None:
        return "<p>Chua co du lieu (buoc 'simulation' chua chay).</p>"
    rows, detail = [], ""
    for code in sorted(sim):
        for method, res in sim[code].items():
            if not res:
                continue
            n_bias = max(len(res["bias"]), 1)
            avg_bias = sum(abs(v) for v in res["bias"].values()) / n_bias
            avg_rmse = sum(res["rmse"].values()) / max(len(res["rmse"]), 1)
            cov = res.get("coverage") or {}
            avg_cov = sum(cov.values()) / len(cov) if cov else None
            rows.append([
                code, method, _fmt(avg_bias), _fmt(avg_rmse),
                _fmt(avg_cov) if avg_cov is not None else "--", _fmt(res.get("mean_brier")),
            ])

            drows = [
                [name, _fmt(res["bias"][name]), _fmt(res["rmse"][name]),
                 _fmt(cov[name]) if name in cov else "--"]
                for name in res["bias"]
            ]
            detail += (f"<details><summary>{_esc(code)} / {_esc(method)} -- chi tiet theo tung tham so"
                       "</summary>"
                       + _table(["Tham so", "Bias", "RMSE", "Coverage 95%"], drows) + "</details>")

    summary_tbl = _table(
        ["Kich ban", "Phuong phap", "|Bias| trung binh", "RMSE trung binh",
         "Coverage 95% trung binh", "Brier score"],
        rows, caption="Bang 6.x -- Nghien cuu mo phong S1-S8: Bias/RMSE/Coverage/Brier",
    )
    caveat = ('<p class="note">Cot trung binh gop cac tham so co THANG DO khac nhau (vd. omega so voi '
              "p11 so voi nu) nen chi dung de so sanh tho giua cac phuong phap; hay mo phan chi tiet "
              "ben duoi de doc Bias/RMSE/Coverage cua tung tham so. Coverage bo trong voi MLE vi MLE "
              "khong sinh khoang tin cay trong quy trinh nay.</p>")
    return summary_tbl + caveat + detail


# ============================================================
# Trang HTML
# ============================================================
_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --border: #e5e7eb;
  --accent: #1d4ed8; --tbl-head: #f3f4f6; --tbl-stripe: #fafafa;
  --err-bg: #fef2f2; --err-fg: #991b1b; --warn-bg: #fffbeb; --warn-fg: #92400e;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --border: #2a2e37;
    --accent: #60a5fa; --tbl-head: #1a1d24; --tbl-stripe: #161920;
    --err-bg: #2a1416; --err-fg: #fca5a5; --warn-bg: #2a2110; --warn-fg: #fcd34d;
  }
}
:root[data-theme="dark"] {
  --bg: #0f1115; --fg: #e5e7eb; --muted: #9ca3af; --border: #2a2e37;
  --accent: #60a5fa; --tbl-head: #1a1d24; --tbl-stripe: #161920;
  --err-bg: #2a1416; --err-fg: #fca5a5; --warn-bg: #2a2110; --warn-fg: #fcd34d;
}
body { background: var(--bg); color: var(--fg); font-family: -apple-system, "Segoe UI", Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 5rem; line-height: 1.55; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.25rem; margin-top: 2.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
p.subtitle { color: var(--muted); margin-top: -0.5rem; }
.twrap { overflow-x: auto; margin: 0.8rem 0 1.6rem; }
.tbl { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
.tbl caption { text-align: left; font-weight: 600; margin-bottom: 0.4rem; caption-side: top; }
.tbl th, .tbl td { border: 1px solid var(--border); padding: 0.35rem 0.6rem; text-align: left; white-space: nowrap; }
.tbl thead th { background: var(--tbl-head); }
.tbl tbody tr:nth-child(even) { background: var(--tbl-stripe); }
.tbl td.empty { color: var(--muted); font-style: italic; }
.note { color: var(--muted); font-size: 0.85rem; }
.warn { background: var(--warn-bg); color: var(--warn-fg); padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.88rem; }
.err { background: var(--err-bg); color: var(--err-fg); padding: 0.6rem 0.9rem; border-radius: 6px; margin: 0.8rem 0; }
.err pre { white-space: pre-wrap; font-size: 0.78rem; overflow-x: auto; }
details { margin: 0.4rem 0; }
details summary { cursor: pointer; color: var(--accent); font-size: 0.88rem; }
nav.toc { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1rem 0 2rem; }
nav.toc a { color: var(--accent); text-decoration: none; font-size: 0.85rem; border: 1px solid var(--border); padding: 0.25rem 0.6rem; border-radius: 999px; }
"""

_TITLE = "Ket qua MS-GARCH-X VN30F1M"


def build_report(results_dir: Path, fragment: bool = False) -> str:
    data = _load(results_dir, "00_data")
    prelim = _load(results_dir, "01_preliminary_tests")
    mle = _load(results_dir, "02_mle")
    vb = _load(results_dir, "03_vb")
    mcmc = _load(results_dir, "04_mcmc")
    r1 = _load(results_dir, "05_r1")
    oos = _load(results_dir, "06_oos")
    sim = _load(results_dir, "07_simulation")
    timing = _load(results_dir, "09_timing")

    specs = [
        ("overview", "Tong quan du lieu", section_overview, (data,)),
        ("desc", "Thong ke mo ta", section_descriptive_stats, (data,)),
        ("figures", "Bieu do", section_figures, (data, mle, vb)),
        ("prelim", "Kiem dinh thong ke ban dau", section_preliminary, (prelim,)),
        ("mle", "Uoc luong MLE (B1-B4, M1)", section_mle_baselines, (mle, data)),
        ("vb", "Variational Bayes (M2, M3)", section_vb, (vb, mle, data)),
        ("mcmc", "MCMC (M4)", section_mcmc, (mcmc, mle, data)),
        ("sdratio", "So sanh VB voi MCMC (eq:sd-ratio)", section_sd_ratio, (vb, mcmc, mle, data)),
        ("r1", "Mo hinh mo rong nu theo trang thai (R1)", section_r1, (r1, mle, data)),
        ("oos", "Danh gia ngoai mau", section_oos, (oos,)),
        ("dm", "Kiem dinh Diebold-Mariano", section_dm_tests, (oos, data)),
        ("sim", "Nghien cuu mo phong (S1-S8)", section_simulation, (sim,)),
        ("cost", "Chi phi tinh toan", section_computation, (timing,)),
    ]
    sections = [(sid, title, _safe(title, fn, *args)) for sid, title, fn, args in specs]

    stages = [
        ("data", data), ("prelim", prelim), ("mle", mle), ("vb", vb),
        ("mcmc", mcmc), ("r1", r1), ("oos", oos), ("simulation", sim),
    ]
    done = [name for name, obj in stages if obj is not None]
    missing = [name for name, obj in stages if obj is None]
    status = f'<p class="note">Buoc da co ket qua: {", ".join(done) or "(chua co)"}.'
    if missing:
        status += f' Chua co: {", ".join(missing)}.'
    status += "</p>"

    toc = "".join(f'<a href="#{sid}">{_esc(title)}</a>' for sid, title, _ in sections)
    body = "".join(f'<h2 id="{sid}">{_esc(title)}</h2>{content}' for sid, title, content in sections)

    inner = f"""<h1>Ket qua uoc luong MS-GARCH-X Student-t cho VN30F1M</h1>
<p class="subtitle">Tong hop tu MLE, Variational Bayes va MCMC tren cung mot ham hop ly. Nguon: {_esc(results_dir)}</p>
{status}
<nav class="toc">{toc}</nav>
{body}
<p class="note">Bao cao sinh tu scripts/build_html_report.py. Cac bang so hoa dua tren checkpoint trong thu muc results/.</p>"""

    if fragment:
        return f"<title>{_TITLE}</title>\n<style>{_CSS}</style>\n{inner}\n"

    # Tai lieu day du: BAT BUOC co <meta charset> vi duong dan results_dir va
    # noi dung co the chua ky tu ngoai ASCII -- mo bang file:// khong co the
    # nay se hien sai font (mojibake) tren Windows.
    return f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_TITLE}</title>
<style>{_CSS}</style>
</head>
<body>
{inner}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Tao bao cao HTML tu thu muc results/")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "report.html"))
    parser.add_argument("--fragment", action="store_true",
                        help="Chi xuat phan noi dung (khong co <html>/<head>), de dan vao Artifact")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[LOI] Khong tim thay thu muc ket qua: {results_dir}", file=sys.stderr)
        return 1

    html_content = build_report(results_dir, fragment=args.fragment)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Da ghi bao cao vao: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
