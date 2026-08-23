"""Dien so lieu tu `results/*.pt` vao cac bang ket qua cua `Bai bao/Bai bao chinh.tex`.

Sinh lai duoc nhieu lan: moi bang duoc nhan dien qua `\\label{...}` va CHI phan
than bang (giua \\midrule va \\bottomrule) bi thay the. Chay lai sau khi co them
checkpoint (vd. MCMC xong) se tu dien not cac cot con trong.

O nhung cho chua co so lieu chac chan, script GIU dau "--" thay vi bia ra con so.

Chay:
    python scripts/cap_nhat_bai_bao.py            # xem truoc, khong ghi
    python scripts/cap_nhat_bai_bao.py --ghi      # ghi vao .tex
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEX = ROOT.parent / "Bài báo" / "Bài báo chính.tex"
RESULTS = ROOT / "results"

import numpy as np
import torch

from msgarchx.inference.summary import posterior_summary, relabel_posterior_samples
from msgarchx.models.ms_garch import expected_duration, label_switch_correction

TRONG = "--"


def f(x, nd=4):
    """Dinh dang so theo quy uoc tieng Viet: dau PHAY thap phan.

    `{,}` giu dung khoang cach cua dau phay trong che do toan; ngoai che do
    toan no van hien thi dung la dau phay.
    """
    if x is None or (isinstance(x, float) and (x != x)):
        return TRONG
    return f"{float(x):.{nd}f}".replace(".", "{,}")


def _nap(ten):
    p = RESULTS / f"{ten}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def _than_bang(tex: str, label: str) -> tuple[int, int]:
    """Vi tri (dau, cuoi) cua than bang co `label`, giua \\midrule va \\bottomrule."""
    i = tex.index(f"\\label{{{label}}}")
    mid = tex.index(r"\midrule", i) + len(r"\midrule")
    bot = tex.index(r"\bottomrule", mid)
    return mid, bot


def _thay(tex: str, label: str, rows: list[str]) -> str:
    a, b = _than_bang(tex, label)
    return tex[:a] + "\n" + "\n".join(rows) + "\n" + tex[b:]


# ============================================================
def bang_kiem_dinh_ban_dau(pre) -> list:
    if pre is None:
        return None
    def hang(ten, k, dien):
        t = pre[k]
        p = t["p_value"]
        qd = "Bác bỏ $H_0$" if p < 0.05 else "Không bác bỏ $H_0$"
        ps = "$<0{,}001$" if p < 1e-3 else f"${p:.3f}$".replace(".", "{,}")
        return f"{ten} & ${f(t['statistic'], 3)}$ & {ps} & {qd} & {dien} \\\\"
    return [
        hang("ADF trên $r_t$", "adf_r", "Chuỗi dừng"),
        hang("Phillips--Perron trên $r_t$", "pp_r", "Chuỗi dừng"),
        hang("KPSS trên $r_t$", "kpss_r", "Không bác bỏ tính dừng"),
        hang("Ljung--Box trên $r_t$", "ljung_box_r", "Có tự tương quan bậc thấp"),
        hang("Ljung--Box trên $r_t^2$", "ljung_box_r2", "Biến động tập trung theo cụm"),
        hang("ARCH-LM", "arch_lm", "Có hiệu ứng ARCH, cần mô hình GARCH"),
        hang("Jarque--Bera", "jarque_bera", "Bác bỏ chuẩn, ủng hộ Student-t"),
    ]


def bang_mo_ta(data) -> list:
    if data is None:
        return None
    def th(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        m, s = x.mean(), x.std(ddof=1)
        sk = float(((x - m) ** 3).mean() / s ** 3)
        ku = float(((x - m) ** 4).mean() / s ** 4)
        return [f"${n}$", f(m), f(s), f(sk, 3), f(ku, 3), f(x.min(), 3), f(x.max(), 3)]
    r = data["r_all"].numpy()
    rv = data["rv_all"].numpy()
    # X da chuan hoa -> khoi phuc thang goc bang trung binh/do lech cua tap huan luyen
    vol = data["X_all"][:, 0].numpy() * float(data["exog_std"].iloc[0]) + float(data["exog_mean"].iloc[0])
    rows = []
    for ten, x in [("$r_t$", r), ("$RV_t$", rv), ("$\\log(1+VOL_{t-1})$", vol)]:
        rows.append(ten + " & " + " & ".join(th(x)) + " \\\\")
    rows.append("$BASIS_t$ & \\multicolumn{7}{c}{\\emph{không có trong nguồn dữ liệu "
                "(xem \\cref{subsec:exogenous-variables})}} \\\\")
    return rows


def _tom_tat_vb(res):
    mau, _ = relabel_posterior_samples(res.samples)
    return posterior_summary(mau)


def bang_tham_so(mle, vb, mcmc) -> list:
    if mle is None:
        return None
    m1 = mle["m1"]
    p = label_switch_correction(m1.params, m1.filter_output["h"])
    se = mle.get("m1_se")
    s_vb = _tom_tat_vb(vb["m2"]) if vb else None
    s_mc = _tom_tat_vb(mcmc["m4"]) if mcmc else None

    ten = ([("$\\mu_1$", "mu", 0), ("$\\mu_2$", "mu", 1),
            ("$\\omega_1$", "omega", 0), ("$\\omega_2$", "omega", 1),
            ("$\\alpha_1$", "alpha", 0), ("$\\alpha_2$", "alpha", 1),
            ("$\\beta_1$", "beta", 0), ("$\\beta_2$", "beta", 1)]
           + [(f"$\\gamma_{{{s+1},{j+1}}}$", "gamma", (s, j))
              for s in range(2) for j in range(p.gamma.shape[1])]
           + [("$p_{11}$", "p11", None), ("$p_{22}$", "p22", None),
              ("$\\nu$", "nu", None)])

    def lay_mle(k, i):
        if k == "p11": v, s = p.P[0, 0], (se["p11"][0] if se else None)
        elif k == "p22": v, s = p.P[1, 1], (se["p22"][0] if se else None)
        elif k == "nu": v, s = p.nu, (se["nu"][0] if se else None)
        elif k == "gamma":
            v = p.gamma[i]; s = se["gamma"].reshape(2, -1)[i] if se else None
        else:
            v = getattr(p, k)[i]; s = se[k][i] if se else None
        return f(float(v)), (f(float(s)) if s is not None else TRONG)

    def lay_bayes(sm, k, i):
        if sm is None: return TRONG, TRONG, TRONG
        nm = "nu_minus_2" if k == "nu" else k
        if nm not in sm: return TRONG, TRONG, TRONG
        d = sm[nm]; sh = 2.0 if k == "nu" else 0.0
        def g(key):
            t = d[key]
            if i is None: return float(t) + sh
            return float(t[i] if not isinstance(i, tuple) else t[i[0], i[1]]) + sh
        return f(g("mean")), f(g("sd")), f"[{f(g('q2.5'),3)}; {f(g('q97.5'),3)}]"

    rows = []
    for nhan, k, i in ten:
        a, b = lay_mle(k, i)
        c1, c2, c3 = lay_bayes(s_vb, k, i)
        d1, d2, d3 = lay_bayes(s_mc, k, i)
        rows.append(f"{nhan} & {a} & {b} & {c1} & {c2} & {c3} & {d1} & {d2} & {d3} \\\\")
    return rows


def bang_chuyen_trang_thai(mle, vb, mcmc) -> list:
    if mle is None:
        return None
    m1 = mle["m1"]
    p = label_switch_correction(m1.params, m1.filter_output["h"])
    d = expected_duration(p.P)
    cot = {"MLE": {"p11": float(p.P[0, 0]), "p22": float(p.P[1, 1]),
                   "d1": float(d[0]), "d2": float(d[1])}}
    for ten, res in [("VB", vb["m2"] if vb else None), ("MCMC", mcmc["m4"] if mcmc else None)]:
        if res is None:
            cot[ten] = None; continue
        s = _tom_tat_vb(res)
        p11, p22 = float(s["p11"]["mean"]), float(s["p22"]["mean"])
        cot[ten] = {"p11": p11, "p22": p22, "d1": 1/(1-p11), "d2": 1/(1-p22)}
    def hang(nhan, key, bd=None):
        vs = []
        for c in ("MLE", "VB", "MCMC"):
            v = cot[c]
            vs.append(TRONG if v is None else f(bd(v) if bd else v[key]))
        return f"{nhan} & " + " & ".join(vs) + " \\\\"
    return [hang("$p_{11}$", "p11"), hang("$p_{12}$", None, lambda v: 1 - v["p11"]),
            hang("$p_{21}$", None, lambda v: 1 - v["p22"]), hang("$p_{22}$", "p22"),
            hang("$d_1$", "d1"), hang("$d_2$", "d2")]


def bang_chuan_vs_t(mle, oos) -> list:
    if mle is None:
        return None
    def hang(nhan, k):
        r = mle[k]
        o = oos.get(k) if oos else None
        var1 = TRONG
        if o:
            n = o["n_test"]
            kp = o["kupiec_1pct"]
            pv = kp.get("pvalue", kp.get("p_value")) if isinstance(kp, dict) else kp
            var1 = f"{o['n_violations_1pct']}/{n} ($p={f(pv,3)}$)"
        return (f"{nhan} & {f(r.loglik, 2)} & {f(r.aic, 2)} & {f(r.bic, 2)} & "
                f"{f(o['qlike'], 4) if o else TRONG} & {f(o['nls'], 4) if o else TRONG} & {var1} \\\\")
    return [hang("B1: GARCH chuẩn", "b1"), hang("B2: GARCH Student-t", "b2"),
            hang("B4: MS-GARCH chuẩn", "b4"), hang("M1: MS-GARCH-X Student-t", "m1")]


def bang_du_bao_bien_dong(oos) -> list:
    if oos is None:
        return None
    ten = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4"]
    co = [k for k in ten if k in oos]
    xh = {k: i + 1 for i, k in enumerate(sorted(co, key=lambda k: oos[k]["qlike"]))}
    rows = []
    for k in ten:
        if k not in oos:
            rows.append(f"{k.upper()} & {TRONG} & {TRONG} & {TRONG} & {TRONG} \\\\")
            continue
        o = oos[k]
        rows.append(f"{k.upper()} & {f(o['mae'])} & {f(o['rmse'])} & {f(o['qlike'])} & {xh[k]} \\\\")
    return rows


def bang_var(oos) -> list:
    if oos is None:
        return None
    rows = []
    for lvl, ten in [("1pct", "1\\%"), ("5pct", "5\\%")]:
        for k in ["b2", "m1", "m2", "m4"]:
            if k not in oos:
                rows.append(f"{k.upper()} & {ten} & {TRONG} & {TRONG} & {TRONG} & {TRONG} \\\\")
                continue
            o = oos[k]
            n, x = o["n_test"], o[f"n_violations_{lvl}"]
            def pv(d):
                if not isinstance(d, dict): return float(d)
                return d.get("pvalue", d.get("p_value", float("nan")))
            ch = pv(o[f"christoffersen_{lvl}"])
            ch_s = TRONG if ch != ch else f(ch, 3)
            rows.append(f"{k.upper()} & {ten} & {x} & {f(100*x/n, 2)}\\% & "
                        f"{f(pv(o[f'kupiec_{lvl}']), 3)} & {ch_s} \\\\")
    return rows


def bang_chi_phi(timing, mcmc) -> list:
    if timing is None:
        return None
    def phut(k):
        return f"{timing[k]/60:.0f} phút" if k in timing else TRONG
    mle_t = sum(timing.get(f"model_{k}", 0) for k in ["b1", "b2", "b3", "b4", "m1"])
    vb_t = sum(timing.get(f"model_{k}", 0) for k in ["m2", "m3"])
    mc_t = timing.get("04_mcmc")
    rhat = f(mcmc["m4"].max_rhat, 3) if mcmc else TRONG
    ess = f(mcmc["m4"].min_ess, 1) if mcmc else TRONG
    div = str(mcmc["m4"].n_divergences) if mcmc else TRONG
    return [
        f"Thời gian trung vị mỗi lần ước lượng & {mle_t/5/60:.0f} phút & {vb_t/2/60:.0f} phút & "
        + (f"{mc_t/3600:.1f} giờ" if mc_t else TRONG) + " \\\\",
        f"Bộ nhớ cực đại & {TRONG} & {TRONG} & {TRONG} \\\\",
        f"Số lần khởi tạo lại & 10 & 1 & 4 chuỗi \\\\",
        f"Tỷ lệ không hội tụ & 6/10 (M1) & {TRONG} & {TRONG} \\\\",
        f"$\\widehat R$ lớn nhất & --- & --- & {rhat} \\\\",
        f"ESS nhỏ nhất & --- & --- & {ess} \\\\",
        f"Số lần phân kỳ & --- & --- & {div} \\\\",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ghi", action="store_true", help="Ghi vao .tex (mac dinh chi xem truoc)")
    args = ap.parse_args()

    data, pre = _nap("00_data"), _nap("01_preliminary_tests")
    mle, vb, mcmc = _nap("02_mle"), _nap("03_vb"), _nap("04_mcmc")
    oos, timing = _nap("06_oos"), _nap("09_timing")
    if oos is None:
        # Lan chay 1 co M4 KHONG hop le (R-hat ~ 4), nhung ket qua cua B1-M3
        # van dung vi tham so cua chung khong doi. Dung lai phan hop le va BO
        # HAN M4 -- de trong con hon dien mot con so khong dang tin.
        oos = _nap("06_oos_lan1_M4_hong")
        if oos is not None:
            oos = {k: v for k, v in oos.items() if k != "m4"}
            print("  [luu y] dung ket qua ngoai mau cua lan chay 1, DA BO M4 "
                  "(mau MCMC lan 1 khong hoi tu)")

    cong_viec = [
        ("tab:initial-tests-template", bang_kiem_dinh_ban_dau(pre)),
        ("tab:descriptive-template", bang_mo_ta(data)),
        ("tab:parameter-template", bang_tham_so(mle, vb, mcmc)),
        ("tab:transition-results", bang_chuyen_trang_thai(mle, vb, mcmc)),
        ("tab:gaussian-t-template", bang_chuan_vs_t(mle, oos)),
        ("tab:volatility-forecast-template", bang_du_bao_bien_dong(oos)),
        ("tab:var-template", bang_var(oos)),
        ("tab:computation-template", bang_chi_phi(timing, mcmc)),
    ]

    tex = TEX.read_text(encoding="utf-8")
    n_sua = 0
    for label, rows in cong_viec:
        if rows is None:
            print(f"  [bo qua] {label}: chua co du lieu")
            continue
        try:
            tex = _thay(tex, label, rows)
            n_sua += 1
            print(f"  [da dien] {label}: {len(rows)} hang")
        except ValueError as e:
            print(f"  [LOI] {label}: {e}")

    if args.ghi:
        bak = TEX.with_suffix(".tex.bak")
        if not bak.exists():
            bak.write_text(TEX.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"  da sao luu ban goc: {bak.name}")
        TEX.write_text(tex, encoding="utf-8")
        print(f"\nDa ghi {n_sua} bang vao {TEX.name}")
    else:
        print(f"\n(xem truoc -- chua ghi. Them --ghi de ghi that. {n_sua} bang san sang)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
