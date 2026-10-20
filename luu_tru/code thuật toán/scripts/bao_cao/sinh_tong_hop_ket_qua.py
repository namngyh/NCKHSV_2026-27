"""Sinh "Bai bao/tong_hop_ket_qua.tex": bang, phuong trinh va dien giai ket qua.

Moi con so doc thang tu results/ (khong uoc luong lai, khong nhap tay).
Hinh vector tham so do ve_vector_tham_so.py ve tu tung lan lap mo phong.

Noi dung:
  1. Du lieu va cach chia.
  2. Buoc 1: so sanh 6 mo hinh GARCH/EGARCH x 3 phuong phap tren VN30F1M.
  3. Buoc 2: mo phong khoi phuc tham so EGARCH-X, truong hop E1-E3.
  4. Buoc 3: MCMC tren EGARCH-X Student-t, hau nghiem va R_sd.
  5. Bang dem thang RMSE va ket luan.

Chay (tu "code thuật toán"):  python scripts/bao_cao/sinh_tong_hop_ket_qua.py
Bien dich:                     xelatex tong_hop_ket_qua.tex  (trong "Bài báo", chay 2 lan)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from garch_vn30.mo_hinh import danh_muc                          # noqa: E402
from garch_vn30.mo_phong import THU_TU, truong_hop_mac_dinh     # noqa: E402

RES = ROOT / "results"
OUT = ROOT.parent / "Bài báo" / "tong_hop_ket_qua.tex"

PP = ("mle", "vb_mean_field", "vb_full_rank")
TOI_THIEU_SE = 10   # so lan lap co SE toi thieu de in CP 95% cua MLE
TEN_PP = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR", "mcmc": "MCMC"}
TEN_PP_DAI = {"mle": "Hợp lý cực đại (MLE)", "vb_mean_field": "Bayes biến phân mean-field (VB-MF)",
              "vb_full_rank": "Bayes biến phân full-rank (VB-FR)", "mcmc": "MCMC (NUTS)"}

NHAN_B = {"mu": r"$\mu$", "omega": r"$\omega$", "alpha": r"$\alpha$", "theta": r"$\theta$",
          "beta": r"$\beta$", "gamma_0": r"$\gamma_1$", "gamma_1": r"$\gamma_2$", "nu": r"$\nu$"}
THU_TU_B = THU_TU

# XeLaTeX + unicode-math loi khi hyperref doc ky tu toan trong tieu de
# \subsection de sinh bookmark: boc \texorpdfstring{toan}{chu}.
MO_TA_E = {
    "E1": r"đòn bẩy rõ, $\theta_0=-0{,}10$, $\nu_0=8$",
    "E2": r"không đòn bẩy, $\theta_0=0$, $\nu_0=8$",
    "E3": r"đuôi dày, $\theta_0=-0{,}10$, $\nu_0=4$",
}
MO_TA_E_CHU = {
    "E1": "don bay ro, theta0=-0,10, nu0=8",
    "E2": "khong don bay, theta0=0, nu0=8",
    "E3": "duoi day, theta0=-0,10, nu0=4",
}


# ------------------------------------------------------------------ dinh dang
def so(x, k: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    s = f"{float(x):.{k}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    s = s.replace(".", ",")
    return ("$-$" + s[1:]) if s.startswith("-") else s


def so_math(x, k: int = 3) -> str:
    """Decimal comma that remains correctly spaced within TeX math mode."""
    s = f"{float(x):.{k}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    return s.replace(".", "{,}")


def pct(x) -> str:
    return "-" if x is None else f"{int(round(100 * x))}\\%"


def bang(cot: str, dau: str, hang: list[str], caption: str, nhan: str,
         co: str = r"\footnotesize", ghi_chu: str | None = None, longtable: bool = False) -> str:
    """Mot bang booktabs; `cot` la dinh dang cot, `dau` la dong tieu de."""
    o = []
    if longtable:
        o += [rf"\begin{{longtable}}{{{cot}}}",
              rf"\caption{{{caption}}}\label{{{nhan}}}\\",
              r"\toprule", dau + r" \\", r"\midrule", r"\endfirsthead",
              r"\toprule", dau + r" \\", r"\midrule", r"\endhead",
              r"\bottomrule", r"\endfoot"]
        o += [h + r" \\" for h in hang]
        o += [r"\end{longtable}"]
        return "\n".join(o) + "\n"
    o += [r"\begin{table}[H]\centering" + co,
          rf"\caption{{{caption}}}\label{{{nhan}}}",
          rf"\begin{{tabular}}{{{cot}}}", r"\toprule", dau + r" \\", r"\midrule"]
    o += [h + r" \\" for h in hang]
    o += [r"\bottomrule", r"\end{tabular}"]
    if ghi_chu:
        o += [r"\par\vspace{0.25em}\begin{minipage}{0.95\textwidth}\scriptsize " + ghi_chu + r"\end{minipage}"]
    o += [r"\end{table}", ""]
    return "\n".join(o)


# ------------------------------------------------------------------ 1. du lieu
def phan_du_lieu() -> str:
    d = torch.load(RES / "00_data.pt", weights_only=False)
    n = int(d["r_all"].shape[0])
    n_tr = int(d["n_train"])
    dates = list(d["dates_all"])
    r = d["r_all"].numpy()
    rv = d["rv_all"].numpy()

    def dong(ten, x):
        return (f"{ten} & {len(x)} & {so(x.mean(), 4)} & {so(x.std(ddof=1), 4)} & "
                f"{so(np.min(x), 3)} & {so(np.max(x), 3)}")

    hang = [dong(r"$r_t$ (toàn bộ)", r), dong(r"$r_t$ (huấn luyện)", r[:n_tr]),
            dong(r"$r_t$ (kiểm định)", r[n_tr:]),
            dong(r"$RV_t$ (toàn bộ)", rv), dong(r"$RV_t$ (huấn luyện)", rv[:n_tr]),
            dong(r"$RV_t$ (kiểm định)", rv[n_tr:])]
    t1 = bang("l r r r r r",
              r"\textbf{Chuỗi} & $N$ & \textbf{Trung bình} & \textbf{Độ lệch chuẩn} & \textbf{Min} & \textbf{Max}",
              hang, "Thống kê mô tả lợi suất ngày và biến động thực tế VN30F1M", "tab:du-lieu",
              co=r"\normalsize")
    hang2 = [rf"Huấn luyện & {str(dates[0])[:10]} & {str(dates[n_tr - 1])[:10]} & {n_tr} & {so(100 * n_tr / n, 2)}\%",
             rf"Kiểm định & {str(dates[n_tr])[:10]} & {str(dates[-1])[:10]} & {n - n_tr} & {so(100 * (n - n_tr) / n, 2)}\%",
             rf"Tổng & {str(dates[0])[:10]} & {str(dates[-1])[:10]} & {n} & 100\%"]
    t2 = bang("l l l r r",
              r"\textbf{Tập} & \textbf{Từ} & \textbf{Đến} & \textbf{Số phiên} & \textbf{Tỷ lệ}",
              hang2, "Cách chia tập huấn luyện / kiểm định", "tab:chia-tap", co=r"\normalsize")
    return (r"\section{Dữ liệu}" + "\n"
            + rf"Dữ liệu gồm {n} lợi suất ngày, chia theo thời gian thành {n_tr} quan sát huấn luyện "
              rf"và {n-n_tr} quan sát kiểm định. Lợi suất toàn mẫu có trung bình "
              rf"{so(r.mean(),4)}\% và độ lệch chuẩn {so(r.std(ddof=1),4)}\%; "
              rf"RV trung bình {so(rv.mean(),4)} điểm phần trăm bình phương. "
              r"Sự phân tán của lợi suất và RV đặt ra yêu cầu đánh giá đồng thời "
              r"dự báo phương sai, mật độ lợi suất và rủi ro đuôi." + "\n\n" + t1 + t2)


# ------------------------------------------------------------------ 2. buoc 1
MO_HINH_BAO_CAO = danh_muc.CATALOGUE


def _nap_o(code: str, pp: str) -> dict | None:
    p = RES / "pipeline" / f"{code}_{pp}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def phan_buoc_1() -> str:
    d = torch.load(RES / "00_data.pt", weights_only=False)
    k = int(d["X_all"].shape[1])
    n_te = int(d["r_all"].shape[0]) - int(d["n_train"])
    hang1, hang2, hang3 = [], [], []
    for spec in MO_HINH_BAO_CAO:
        found = False
        for pp in PP:
            o = _nap_o(spec.code, pp)
            if o is None:
                continue
            found = True
            oos = o.get("oos") or {}
            ten = f"{spec.name} & {TEN_PP[pp]}"
            hang1.append(f"{ten} & {spec.n_params(k)} & {so(o.get('loglik'), 2)} & {so(o.get('aic'), 2)} & "
                         f"{so(o.get('bic'), 2)} & {so(o.get('final_elbo'), 2)}")
            hang2.append(f"{ten} & {so(oos.get('mae'), 4)} & {so(oos.get('rmse'), 4)} & "
                         f"{so(oos.get('qlike'), 4)} & {so(oos.get('nls'), 4)}")
            k1, k5 = oos.get("kupiec_1pct"), oos.get("kupiec_5pct")
            c1, c5 = oos.get("christoffersen_1pct"), oos.get("christoffersen_5pct")
            if k1 and k5:
                hang3.append(f"{ten} & {k1['n_violations']} & {so(k1['p_value'], 3)} & {so(c1['p_value_cc'], 3)} & "
                             f"{k5['n_violations']} & {so(k5['p_value'], 3)} & {so(c5['p_value_cc'], 3)}")
        if not found:
            raise FileNotFoundError(f"Thiếu kết quả cho mô hình {spec.name}")
    t1 = bang("l l r r r r r",
              r"\textbf{Mô hình} & \textbf{Phương pháp} & $d$ & \textbf{Log Likelihood} & \textbf{AIC} & \textbf{BIC} & \textbf{ELBO}",
              hang1, r"Bước 1: Khớp trong mẫu ($n_{\text{huấn luyện}}=1735$)", "tab:b1-trong-mau")
    t2 = bang("l l r r r r",
              r"\textbf{Mô hình} & \textbf{Phương pháp} & \textbf{MAE} & \textbf{RMSE} & \textbf{QLIKE} & \textbf{NLS}",
              hang2, rf"Bước 1: Dự báo ngoài mẫu ($n_{{\text{{kiểm định}}}}={n_te}$)", "tab:b1-ngoai-mau")
    t3 = bang("l l r r r r r r",
              r"\textbf{Mô hình} & \textbf{Phương pháp} & $n_{1\%}$ & $p_{\text{UC}}$ & $p_{\text{CC}}$ & $n_{5\%}$ & $p_{\text{UC}}$ & $p_{\text{CC}}$",
              hang3, rf"Bước 1: Kiểm định VaR ngoài mẫu ($n={n_te}$)",
              "tab:b1-var")
    g_n = _nap_o("garch_n", "mle")
    g_t = _nap_o("garch_t", "mle")
    g_x = _nap_o("garch_x_t", "mle")
    eg_x = _nap_o("egarch_x_t", "mle")
    lead = (r"\section{So sánh các mô hình GARCH và EGARCH trên dữ liệu VN30F1M}" + "\n"
            + rf"Student-$t$ làm AIC của GARCH giảm từ {so(g_n['aic'],2)} xuống "
              rf"{so(g_t['aic'],2)}. Khi thêm hai biến ngoại sinh, GARCH-X đạt "
              rf"AIC {so(g_x['aic'],2)}. EGARCH-X Student-$t$ với MLE "
              rf"đạt AIC {so(eg_x['aic'],2)}, BIC {so(eg_x['bic'],2)}, "
              rf"QLIKE {so(eg_x['oos']['qlike'],4)} và NLS {so(eg_x['oos']['nls'],4)}; "
              r"đây là các giá trị thấp nhất của bốn tiêu chí tương ứng trong sáu "
              r"mô hình được so sánh." + "\n\n"
            + r"Đối với VaR 1\%, mô hình GARCH chuẩn MLE có 10 vi phạm trên 435 phiên "
              r"($p_{\rm UC}=0{,}020$), trong khi GARCH Student-$t$ MLE có 5 "
              r"($p_{\rm UC}=0{,}760$). Điều này củng cố vai trò của phân phối đuôi dày "
              r"trong dự báo rủi ro. Mô hình EGARCH-X thắng QLIKE và NLS nhưng "
              r"không có số vi phạm VaR 1\% sát kỳ vọng nhất; không nên đồng nhất "
              r"xếp hạng điểm số với xếp hạng kiểm định VaR." + "\n\n")
    note = (r"AIC/BIC chỉ báo cáo cho MLE; ELBO không dùng để xếp hạng "
            r"giữa các mô hình khác đặc tả.")
    return lead + t1 + t2 + t3 + note + "\n"


# ------------------------------------------------------------------ 3. mo phong
def _mo_ta_theta_b(that: dict) -> str:
    return (r"Vector tham số sinh dữ liệu $\boldsymbol\theta_0$ được khai báo bằng "
            + ", ".join(rf"${NHAN_B[key][1:-1]}={so_math(that[key])}$" for key in THU_TU_B)
            + ".")


def _bang_khoi_phuc(code: str, pp: str, m: dict, that: dict, thu_tu: list, nhan: dict,
                    T: int, burn_in: int) -> str:
    cov = m.get("coverage") or {}
    n = m["n_replications"]
    n_ok = m.get("n_co_khoang")
    # MLE: CP Wald chi co nghia khi du lan lap co Hessian xac dinh duong
    if pp == "mle" and n_ok is not None and n_ok < TOI_THIEU_SE:
        cov = {}
    hang = []
    for kk in thu_tu:
        b, r_ = m["bias"][kk], m["rmse"][kk]
        se = max(r_ * r_ - b * b, 0.0) ** 0.5
        c = cov.get(kk)
        hang.append(f"{nhan[kk]} & {so(that[kk] + b)} & {so(b)} & {so(r_)} & {so(se)} & {pct(c)}")
    cap = rf"{code} --- {TEN_PP_DAI[pp]}: khôi phục tham số"
    ghi = rf"Số lần lặp hợp lệ $N={n}$; mỗi chuỗi có $T={T}$ quan sát sau burn-in $={burn_in}$. "
    if pp == "mle" and n_ok is not None:
        if cov:
            ghi += rf"CP 95\% Wald chỉ tính trên {n_ok}/{n} lần lặp có Hessian xác định dương."
        else:
            ghi += rf"CP 95\% để trống vì chỉ {n_ok}/{n} lần lặp có Hessian xác định dương."
    return bang("l r r r r r",
                r"\textbf{Tham số} & \textbf{Mean} & \textbf{Bias} & \textbf{RMSE} & \textbf{SE} & \textbf{CP 95\%}",
                hang, cap, f"tab:{code}-{pp}", ghi_chu=ghi)


# ------------------------------------------------------------------ 4. buoc 2
def phan_pr_b() -> tuple[str, dict]:
    scns = truong_hop_mac_dinh()
    out = [r"\section{Khôi phục tham số EGARCH-X Student-$t$}",
           r"Ba trường hợp E1--E3 cùng dùng 100 lần lặp cho mỗi phương pháp. "
           r"E2 đặt hệ số đòn bẩy thật bằng 0 để kiểm tra khả năng xuất hiện tín hiệu đòn bẩy giả; "
           r"E3 kiểm tra độ bền khi đuôi phân phối dày hơn."]
    ket_qua = {}
    for code, scn in scns.items():
        f = RES / "mo_phong_egarch" / f"{code}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        that = {k_: float(v) for k_, v in d["mle"]["tham_so_that"].items()}
        ket_qua[code] = d
        out.append(rf"\subsection{{{code} --- \texorpdfstring{{{MO_TA_E[code]}}}{{{MO_TA_E_CHU[code]}}}}}")
        out.append(_mo_ta_theta_b(that))
        for pp in PP:
            if pp in d:
                out.append(_bang_khoi_phuc(code, pp, d[pp], that, THU_TU_B, NHAN_B,
                                          scn.T, scn.burn_in))
        theta_text = "; ".join(
            rf"{TEN_PP[pp]}: Mean $={so(that['theta'] + d[pp]['bias']['theta'], 4)}$, "
            rf"RMSE $={so(d[pp]['rmse']['theta'], 4)}$"
            for pp in PP if pp in d)
        out.append(rf"Đối với hệ số đòn bẩy $\theta$, {theta_text}. "
                   r"Hình bên dưới cho thấy toàn bộ đường trung bình tích lũy, "
                   r"qua đó phân biệt sai lệch bền vững với dao động của vài lần lặp đầu.")
        out.append(rf"\begin{{figure}}[H]\centering\includegraphics[width=\textwidth]{{hinh_tong_hop/vector_{code}.pdf}}"
                   rf"\caption{{{code}: Trung bình tích lũy $\bar\theta_{{j,n}}$ của tám tham số EGARCH-X "
                   r"qua các lần lặp. Đường đứt nét là giá trị sinh dữ liệu; "
                   r"mỗi đường dùng đúng các checkpoint của phương pháp tương ứng.}"
                   rf"\label{{fig:vector-{code}}}\end{{figure}}")
    return "\n".join(out), ket_qua


# ------------------------------------------------------------------ 5. buoc 3
def phan_buoc_3() -> str:
    f = RES / "pipeline" / "mcmc_egarch_x_t.pt"
    if not f.exists():
        return ""
    m = torch.load(f, weights_only=False)
    S, sm = m["samples"], m["summary"]
    nhan = {"mu": r"$\mu$", "omega": r"$\omega$", "alpha": r"$\alpha$", "theta": r"$\theta$",
            "beta": r"$\beta$", "gamma0": r"$\gamma_1$", "gamma1": r"$\gamma_2$", "nu": r"$\nu$"}
    mle = torch.load(RES / "pipeline" / "egarch_x_t_mle.pt", weights_only=False)["params"]
    mle_v = {"mu": float(mle["mu"]), "omega": float(mle["omega"]), "alpha": float(mle["alpha"]),
             "theta": float(mle["theta"]), "beta": float(mle["beta"]),
             "gamma0": float(mle["gamma"][0]), "gamma1": float(mle["gamma"][1]), "nu": float(mle["nu"])}
    vb = {pp: torch.load(RES / "pipeline" / f"egarch_x_t_{pp}.pt", weights_only=False)["samples"]
          for pp in ("vb_mean_field", "vb_full_rank")}

    def cot(samples, ten):
        if ten == "nu":
            return samples["nu_minus_2"].double() + 2.0
        if ten.startswith("gamma"):
            return samples["gamma"][:, int(ten[-1])].double()
        return samples[ten].double()

    hang, hang_r = [], []
    for ten in ("mu", "omega", "alpha", "theta", "beta", "gamma0", "gamma1", "nu"):
        x = cot(S, ten).numpy()
        q = np.quantile(x, [0.025, 0.975])
        hang.append(f"{nhan[ten]} & {so(mle_v[ten], 4)} & {so(x.mean(), 4)} & {so(x.std(ddof=1), 4)} & "
                    f"[{so(q[0], 4)}; {so(q[1], 4)}]")
        sd_mc = x.std(ddof=1)
        r_mf = cot(vb["vb_mean_field"], ten).numpy().std(ddof=1) / sd_mc
        r_fr = cot(vb["vb_full_rank"], ten).numpy().std(ddof=1) / sd_mc
        hang_r.append(f"{nhan[ten]} & {so(sd_mc, 4)} & {so(r_mf, 2)} & {so(r_fr, 2)}")
    t1 = bang("l r r r c",
              r"\textbf{Tham số} & \textbf{MLE} & \textbf{TB hậu nghiệm} & \textbf{SD} & \textbf{KTC 95\%}",
              hang, r"Hậu nghiệm MCMC trên EGARCH-X Student-$t$ và nghiệm MLE đối chiếu",
              "tab:b3-mcmc", co=r"\normalsize",
              ghi_chu=rf"Dữ liệu huấn luyện $T=1735$; NUTS: 4 chuỗi, mỗi chuỗi 1.000 vòng warmup "
                       rf"và 400 mẫu giữ lại; $\max\hat R={so(m['max_rhat'], 4)}$, "
                       rf"ESS nhỏ nhất $={m['min_ess']:.0f}$, {m['n_divergences']} phân kỳ, "
                       rf"thời gian {m['giay'] / 3600:.1f} giờ.")
    t2 = bang("l r r r",
              r"\textbf{Tham số} & $\mathrm{sd}_{\text{MCMC}}$ & $R_{sd}$ VB-MF & $R_{sd}$ VB-FR",
              hang_r, r"Bước 3 --- $R_{sd}=\mathrm{sd}_{\text{VB}}/\mathrm{sd}_{\text{MCMC}}$ "
              r"(độ lệch chuẩn hậu nghiệm VB so với MCMC, cùng dữ liệu)", "tab:b3-rsd", co=r"\normalsize")
    lead = (r"\section{MCMC trên EGARCH-X Student-$t$}" + "\n"
            + rf"NUTS đạt $\max\hat R={so(m['max_rhat'],4)}$, ESS nhỏ nhất "
              rf"{m['min_ess']:.0f} và không có phân kỳ. Khoảng hậu nghiệm 95\% "
              r"của hệ số đòn bẩy $\theta$ loại trừ 0, phù hợp với tác động "
              r"bất đối xứng của cú sốc lên log-phương sai. "
              r"So với MCMC, VB-MF báo độ lệch chuẩn hậu nghiệm của $\omega$ "
              r"và $\beta$ lần lượt bằng 0,37 và 0,29 lần; VB-FR đạt khoảng "
              r"0,75 lần cho cả hai. Tâm hậu nghiệm gần nhau không đồng nghĩa "
              r"với mức bất định được hiệu chỉnh tốt." + "\n\n")
    return lead + t1 + t2


# ------------------------------------------------------------------ 6. dem thang
def phan_dem_thang(kq_b: dict) -> str:
    hang = []
    for code, d in kq_b.items():
        pps = [pp for pp in PP if pp in d]
        tham = list(d[pps[0]]["rmse"].keys())
        dem = {pp: 0 for pp in pps}
        for t in tham:
            best = min(pps, key=lambda pp: d[pp]["rmse"][t])
            dem[best] += 1
        n_txt = "/".join(str(d[pp]["n_replications"]) for pp in pps)
        hang.append(f"{code} & {len(tham)} & {n_txt} & " + " & ".join(
            str(dem[pp]) if pp in dem else "--" for pp in PP))
    return r"\section{Tổng hợp thứ hạng RMSE}" + "\n" + bang(
        "l r l r r r",
        r"\textbf{Trường hợp} & \textbf{Số tham số} & $B$ (MLE/MF/FR) & \textbf{MLE} & \textbf{VB-MF} & \textbf{VB-FR}",
        hang, "Số tham số EGARCH-X mà mỗi phương pháp đạt RMSE nhỏ nhất, theo trường hợp E1--E3.",
        "tab:dem-thang", co=r"\normalsize")


def ket_luan() -> str:
    return r"""
\section{Kết luận và giới hạn diễn giải}
Trên mẫu VN30F1M được khảo sát, EGARCH-X Student-$t$ với MLE đạt AIC,
BIC, QLIKE và NLS thấp nhất trong sáu mô hình một trạng thái được so sánh.
Phân phối đuôi dày và biến ngoại sinh có đóng góp trong những đối chiếu
hiện có. NUTS trên mô hình EGARCH-X Student-$t$
hội tụ và cho thấy hiệu ứng đòn bẩy âm; VB mean-field đánh giá thấp độ phân
tán hậu nghiệm của một số hệ số dai dẳng. Những kết luận này giới hạn ở
chuỗi VN30F1M, sáu đặc tả và cách chia mẫu đã dùng.

Mô phỏng EGARCH-X Student-$t$ ở ba trường hợp E1--E3 đánh giá khả năng khôi
phục tham số, nhưng không thay thế kiểm chứng ngoài mẫu trên dữ liệu khác.
Chuỗi hợp đồng liên tục do nhà cung cấp ghép sẵn nên nghiên cứu chưa tách
riêng được ảnh hưởng của ngày chuyển hợp đồng lên các quan sát cực trị.
"""


# ------------------------------------------------------------------ van ban
PREAMBLE = r"""% !TeX program = xelatex
% !TeX encoding = UTF-8
% SINH TU DONG boi scripts/bao_cao/sinh_tong_hop_ket_qua.py -- khong sua tay.
\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage{unicode-math}
\usepackage{amsmath}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Liberation Serif}}
\IfFontExistsTF{Cambria Math}{\setmathfont{Cambria Math}}{\setmathfont{Latin Modern Math}}
\usepackage[a4paper,top=2.2cm,bottom=2.2cm,left=1.8cm,right=1.8cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{array}
\usepackage{float}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\emergencystretch=3em
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\renewcommand{\tablename}{Bảng}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.3em}

\title{\vspace{-1.2cm}\textbf{Tổng hợp kết quả}\\[0.2em]\large Các mô hình GARCH và EGARCH một trạng thái cho VN30F1M}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-1.2cm}
\tableofcontents
\clearpage
"""

CHU_THICH = r"""
\section{Giải thích ký hiệu và quy ước}
\small
\renewcommand{\arraystretch}{1.00}
\begin{longtable}{@{}p{0.24\textwidth}p{0.71\textwidth}@{}}
\toprule
\textbf{Ký hiệu} & \textbf{Ý nghĩa} \\
\midrule
\endfirsthead
\toprule
\textbf{Ký hiệu} & \textbf{Ý nghĩa} \\
\midrule
\endhead
$P_t$ & Giá đóng cửa ở phiên $t$. \\
$r_t$ & Lợi suất log ngày, tính theo phần trăm. \\
$RV_t$ & Phương sai thực tế ngày $t$, tính từ các nến nội phiên. \\
$X_{t-1}$ & Vector biến ngoại sinh của phiên trước. \\
$h_t$ & Phương sai có điều kiện ở phiên $t$. \\
$\varepsilon_t$ & Phần dư lợi suất $r_t-\mu$. \\
$z_t$ & Phần dư chuẩn hóa $\varepsilon_t/\sqrt{h_t}$. \\
$\mu$ & Trung bình có điều kiện của lợi suất. \\
$\omega$ & Hệ số chặn trong phương trình phương sai hoặc log-phương sai. \\
$\alpha$ & Hệ số phản ứng của biến động với cú sốc trước đó. \\
$\beta$ & Hệ số dai dẳng của biến động. \\
$\theta$ & Hệ số bất đối xứng trong EGARCH. \\
$\boldsymbol\gamma$ & Vector hệ số của các biến ngoại sinh. \\
$\nu$ & Bậc tự do của phân phối Student-$t$. \\
$\boldsymbol\theta_0$ & Vector tham số thật dùng sinh dữ liệu mô phỏng. \\
$\theta_0$ & Giá trị thật của hệ số bất đối xứng EGARCH. \\
$\nu_0$ & Bậc tự do thật của phân phối Student-$t$ trong mô phỏng. \\
$\hat\theta_{j,b}$ & Ước lượng tham số thứ $j$ ở lần lặp mô phỏng $b$. \\
$\bar\theta_{j,n}$ & Trung bình ước lượng tham số thứ $j$ sau $n$ lần lặp. \\
$T$ & Số quan sát giữ lại trong mỗi chuỗi mô phỏng. \\
burn-in & Số quan sát đầu chuỗi mô phỏng bị loại bỏ. \\
$N$ & Số lần lặp mô phỏng hợp lệ. \\
$n_{\rm test}$ & Số quan sát của tập kiểm định ngoài mẫu. \\
$B$ & Số lần lặp của từng phương pháp trong bảng tổng hợp RMSE. \\
PP & Phương pháp ước lượng trong bảng kết quả. \\
MLE & Ước lượng hợp lý cực đại. \\
VB-MF & Bayes biến phân với hậu nghiệm xấp xỉ mean-field. \\
VB-FR & Bayes biến phân với hậu nghiệm xấp xỉ full-rank. \\
MCMC & Lấy mẫu Markov Monte Carlo từ hậu nghiệm. \\
NUTS & Thuật toán lấy mẫu No-U-Turn Sampler. \\
$d$ & Số tham số tự do của mô hình. \\
$\ell$ & Log-likelihood tại nghiệm ước lượng. \\
AIC & Tiêu chí thông tin Akaike. \\
BIC & Tiêu chí thông tin Bayes. \\
ELBO & Cận dưới bằng chứng trong suy diễn biến phân. \\
MAE & Trung bình sai số tuyệt đối của dự báo phương sai so với $RV_t$. \\
RMSE & Căn sai số bình phương trung bình; đối tượng được đo nêu tại từng bảng. \\
QLIKE & Hàm mất mát quasi-likelihood của dự báo phương sai. \\
NLS & Điểm log âm của mật độ dự báo lợi suất. \\
VaR & Giá trị chịu rủi ro tại mức xác suất được nêu. \\
$p_{\rm UC}$ & Giá trị $p$ của kiểm định bao phủ vô điều kiện Kupiec. \\
$p_{\rm CC}$ & Giá trị $p$ của kiểm định bao phủ có điều kiện Christoffersen. \\
Mean & Trung bình ước lượng qua các lần lặp mô phỏng. \\
Bias & Độ chệch của trung bình ước lượng so với tham số thật. \\
SE & Độ lệch chuẩn Monte Carlo của ước lượng. \\
CP 95\% & Tỷ lệ khoảng 95\% bao chứa tham số thật trong mô phỏng. \\
SD & Độ lệch chuẩn của mẫu hậu nghiệm MCMC. \\
KTC 95\% & Khoảng tin cậy hoặc khoảng hậu nghiệm 95\%, tùy phương pháp. \\
$R_{sd}$ & Tỷ số độ lệch chuẩn hậu nghiệm VB so với MCMC. \\
$\hat R$ & Chỉ số chẩn đoán hội tụ giữa các chuỗi MCMC. \\
ESS & Cỡ mẫu hiệu dụng của chuỗi MCMC. \\
\bottomrule
\end{longtable}
\normalsize
\clearpage
"""

PHUONG_TRINH = r"""
\section{Mô hình và cách đọc kết quả}
\subsection{Dữ liệu, phương sai và phân phối sai số}
Lợi suất ngày được đo theo phần trăm: $r_t=100(\log P_t-\log P_{t-1})$.
Biến động thực tế: $$RV_t=10^4\sum_j(\log P_{t,j}-\log P_{t,j-1})^2,$$
được tính từ các nến nội phiên và làm đại diện quan sát cho phương sai ngày.
Vector ngoại sinh $X_{t-1}$ gồm $\log(1+VOL_{t-1})$ và
$\log(RV_{t-1}+10^{-6})$; cả hai được chuẩn hóa bằng thống kê của tập huấn luyện.
Vì vậy thông tin ngày $t$ không đi vào dự báo của chính ngày đó.

Với $\varepsilon_t=r_t-\mu$ và $z_t=\varepsilon_t/\sqrt{h_t}$, mô hình
GARCH-X dùng
\begin{equation}\label{eq:garchx}
h_t=\omega+\alpha\varepsilon_{t-1}^{2}+\beta h_{t-1}
       +\boldsymbol\gamma^{\mathsf T}X_{t-1}.
\end{equation}
EGARCH-X thay phương trình phương sai bằng phương trình log-phương sai:
\begin{equation}\label{eq:egarchx}
\log h_t=\omega+\alpha\bigl(|z_{t-1}|-\mathbb E|z|\bigr)
       +\theta z_{t-1}+\beta\log h_{t-1}
       +\boldsymbol\gamma^{\mathsf T}X_{t-1}.
\end{equation}
Hệ số $\theta<0$ biểu thị cú sốc âm làm tăng biến động mạnh hơn cú sốc dương
cùng độ lớn. Dạng mũ bảo đảm $h_t>0$ theo cấu trúc. Các biến thể không có X
đặt $\boldsymbol\gamma=0$.

Sai số Student-$t$ được chuẩn hóa để $\mathbb E(z_t)=0$ và
$\operatorname{Var}(z_t)=1$ khi $\nu>2$:
\begin{equation}\label{eq:student}
f(z\mid\nu)=\frac{\Gamma((\nu+1)/2)}
 {\Gamma(\nu/2)\sqrt{\pi(\nu-2)}}
 \left(1+\frac{z^2}{\nu-2}\right)^{-(\nu+1)/2}.
\end{equation}
Các biến thể chuẩn dùng giới hạn $\nu\to\infty$. Phần so sánh thực nghiệm
gồm GARCH chuẩn, GARCH Student-$t$, GARCH-X Student-$t$, EGARCH chuẩn,
EGARCH Student-$t$ và EGARCH-X Student-$t$.

\subsection{Likelihood và suy diễn}
Với mật độ chuẩn hóa $f_z$ tương ứng phân phối chuẩn hoặc Student-$t$,
log-likelihood có điều kiện của mô hình một trạng thái là
\begin{equation}\label{eq:likelihood}
\ell(\boldsymbol\vartheta)=\sum_{t=1}^{T}
\left[\log f_z\left(\frac{r_t-\mu}{\sqrt{h_t}}\right)-\frac12\log h_t\right].
\end{equation}
MLE cực đại hóa $\ell$. Hai phương pháp VB xấp xỉ hậu nghiệm bằng Gaussian
mean-field hoặc Gaussian hiệp phương sai đầy đủ; MCMC dùng NUTS để lấy mẫu
hậu nghiệm. Ba phương pháp dùng cùng hàm likelihood; các cách Bayes còn dùng
tiên nghiệm nên chênh lệch kết quả không thể quy riêng cho thuật toán.

\subsection{Đánh giá và mô phỏng khôi phục tham số}
Trong mẫu, $\mathrm{AIC}=2d-2\ell$ và
$\mathrm{BIC}=d\log(n_{\rm train})-2\ell$; ELBO chỉ dùng để theo dõi VB
\emph{trong cùng một đặc tả}, không thay thế AIC/BIC để xếp hạng chéo mô hình.
Ngoài mẫu, QLIKE và điểm log âm (NLS) lần lượt đo dự báo phương sai và mật độ:
\begin{equation}\label{eq:scores}
\mathrm{QLIKE}=\frac1{n_{\rm test}}\sum_t\left(\log\hat h_t+\frac{RV_t}{\hat h_t}\right),
\qquad
\mathrm{NLS}=-\frac1{n_{\rm test}}\sum_t\log\hat f_t(r_t).
\end{equation}
VaR được kiểm tra ở mức 1\% và 5\% bằng số vi phạm cùng các kiểm định
Kupiec (bao phủ vô điều kiện) và Christoffersen (bao phủ có điều kiện).
Tham số chỉ được ước lượng trên tập huấn luyện một lần; bộ lọc tiếp tục cập
nhật theo quan sát kiểm định, không tái ước lượng tham số hằng ngày.

Trong mô phỏng, một vector tham số thật $\boldsymbol\theta_0$ sinh ra $N$
chuỗi độc lập. Sau khi bỏ burn-in, mỗi chuỗi còn $T$ quan sát và được ước
lượng riêng. Với tham số thứ $j$, các bảng báo cáo
\begin{equation}\label{eq:recovery}
\overline\theta_j=\frac1N\sum_{b=1}^N\hat\theta_{j,b},\quad
\mathrm{Bias}_j=\overline\theta_j-\theta_{0j},\quad
\mathrm{RMSE}_j=\sqrt{\frac1N\sum_{b=1}^N
 (\hat\theta_{j,b}-\theta_{0j})^2},\quad
\mathrm{SE}_{\rm MC,j}=\sqrt{\mathrm{RMSE}_j^2-\mathrm{Bias}_j^2}.
\end{equation}
CP 95\% là tỷ lệ khoảng 95\% chứa $\theta_{0j}$; khoảng MLE kiểu Wald
chỉ tính khi Hessian quan sát xác định dương. Hình vector dùng trung bình
tích lũy $\overline\theta_{j,n}=n^{-1}\sum_{b=1}^n\hat\theta_{j,b}$,
$1\le n\le N$; đường nằm xa $\theta_{0j}$ ở cuối trục là dấu hiệu chệch
vẫn còn sau toàn bộ lần lặp.
"""


def main() -> None:
    parts = [PREAMBLE, CHU_THICH, PHUONG_TRINH, phan_du_lieu(), phan_buoc_1()]
    b, kq_b = phan_pr_b()
    parts += [b, phan_buoc_3(), phan_dem_thang(kq_b), ket_luan(), r"\end{document}", ""]
    tex = "\n".join(parts)
    tex = tex.replace(" --- ", ": ").replace("---", ": ").replace("--", "-")
    tex = tex.replace("—", "-").replace("–", "-")
    OUT.write_text(tex, encoding="utf-8")
    # kiem tra can bang moi truong
    for env in ("table", "tabular", "longtable", "document"):
        mo, dong = tex.count(rf"\begin{{{env}}}"), tex.count(rf"\end{{{env}}}")
        assert mo == dong, f"{env}: {mo} begin / {dong} end"
    print(f"-> {OUT}  ({len(tex.splitlines())} dong, {tex.count(chr(92) + 'begin{table}') + tex.count(chr(92) + 'begin{longtable}')} bang)")


if __name__ == "__main__":
    main()
