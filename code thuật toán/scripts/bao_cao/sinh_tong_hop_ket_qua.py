"""Sinh `Bai bao/tong_hop_ket_qua.tex`: TOAN BO ket qua cua de tai, CHI BANG.

Khong nhan xet, khong dien giai -- moi bang co mot caption ngan noi ro bang
do la gi va n bao nhieu. Moi con so doc thang tu file ket qua; khong nhap tay.
Cac file nguon deu da qua `scripts/tien_ich/kiem_toan_ket_qua.py`.

Noi dung:
  1. Du lieu va cach chia.
  2. Buoc 1: 8 mo hinh x 3 phuong phap tren VN30F1M (trong mau + ngoai mau + VaR).
  3. Buoc 2 PR-A: MS-GARCH-X, 8 kich ban S1-S8 (theta_0, Bias/RMSE/SE/CP, Brier).
  4. Buoc 2 PR-B: EGARCH-X, 3 kich ban E1-E3.
  5. Buoc 3: MCMC tren EGARCH-X-t, hau nghiem va R_sd.
  6. Bang dem thang RMSE giua ba phuong phap.

Chay (tu `code thuat toan`):  py scripts/bao_cao/sinh_tong_hop_ket_qua.py
Bien dich:                     xelatex "Bai bao/tong_hop_ket_qua.tex"  (chay 2 lan)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from msgarchx.inference import model_spec                    # noqa: E402
from msgarchx.simulation.scenarios import default_scenarios  # noqa: E402
from msgarchx.simulation.run_egarch import default_egarch_scenarios  # noqa: E402

RES = ROOT / "results"
OUT = ROOT.parent / "Bài báo" / "tong_hop_ket_qua.tex"

PP = ("mle", "vb_mean_field", "vb_full_rank")
TOI_THIEU_SE = 10   # so lan lap co SE toi thieu de in CP 95% cua MLE
TEN_PP = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR", "mcmc": "MCMC"}
TEN_PP_DAI = {"mle": "Hợp lý cực đại (MLE)", "vb_mean_field": "Bayes biến phân mean-field (VB-MF)",
              "vb_full_rank": "Bayes biến phân full-rank (VB-FR)", "mcmc": "MCMC (NUTS)"}

NHAN_A = {"mu1": r"$\mu_1$", "mu2": r"$\mu_2$", "omega1": r"$\omega_1$", "omega2": r"$\omega_2$",
          "alpha1": r"$\alpha_1$", "alpha2": r"$\alpha_2$", "beta1": r"$\beta_1$", "beta2": r"$\beta_2$",
          "gamma1_0": r"$\gamma_{1,1}$", "gamma1_1": r"$\gamma_{1,2}$",
          "gamma2_0": r"$\gamma_{2,1}$", "gamma2_1": r"$\gamma_{2,2}$",
          "p11": r"$p_{11}$", "p22": r"$p_{22}$", "nu": r"$\nu$"}
THU_TU_A = ["mu1", "mu2", "omega1", "omega2", "alpha1", "alpha2", "beta1", "beta2",
            "gamma1_0", "gamma1_1", "gamma2_0", "gamma2_1", "p11", "p22", "nu"]
NHAN_B = {"mu": r"$\mu$", "omega": r"$\omega$", "alpha": r"$\alpha$", "theta": r"$\theta$",
          "beta": r"$\beta$", "gamma_0": r"$\gamma_1$", "gamma_1": r"$\gamma_2$", "nu": r"$\nu$"}
THU_TU_B = ["mu", "omega", "alpha", "theta", "beta", "gamma_0", "gamma_1", "nu"]

MO_TA_S = {
    "S1": r"hai trạng thái phân biệt rõ, $\nu_0=8$",
    "S2": r"hai trạng thái gần nhau, $\nu_0=8$",
    "S3": r"đuôi rất dày, $\nu_0=4$",
    "S4": r"$\nu_0=30$ (gần phân phối chuẩn)",
    "S5": r"trạng thái biến động cao hiếm, $p_{11}=0{,}99$, $p_{22}=0{,}85$",
    "S6": r"cỡ mẫu nhỏ, $T=500$",
    "S7": r"cỡ mẫu lớn, $T=5000$",
    "S8": r"sai đặc tả: $r_t + 0{,}3\,r_{t-1}$",
}
# Ban chu (khong toan) dung lam nhan dau trang PDF cho tieu de o tren. XeLaTeX +
# unicode-math bi loi "Improper alphabetic constant" khi hyperref co gang doc
# ky tu \nu (\mitnu) trong tieu de \subsection de sinh bookmark -- phai boc
# \texorpdfstring{toan}{chu} thay vi de toan tho trong \subsection{}.
MO_TA_S_CHU = {
    "S1": "hai trang thai phan biet ro, nu0=8",
    "S2": "hai trang thai gan nhau, nu0=8",
    "S3": "duoi rat day, nu0=4",
    "S4": "nu0=30 (gan phan phoi chuan)",
    "S5": "trang thai bien dong cao hiem, p11=0,99, p22=0,85",
    "S6": "co mau nho, T=500",
    "S7": "co mau lon, T=5000",
    "S8": "sai dac ta: r_t + 0,3 r_(t-1)",
}
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
        return "--"
    s = f"{float(x):.{k}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    s = s.replace(".", ",")
    return ("$-$" + s[1:]) if s.startswith("-") else s


def pct(x) -> str:
    return "--" if x is None else f"{int(round(100 * x))}\\%"


def bang(cot: str, dau: str, hang: list[str], caption: str, nhan: str,
         co: str = r"\scriptsize", ghi_chu: str | None = None, longtable: bool = False) -> str:
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
              co=r"\small")
    hang2 = [rf"Huấn luyện & {str(dates[0])[:10]} & {str(dates[n_tr - 1])[:10]} & {n_tr} & {so(100 * n_tr / n, 2)}\%",
             rf"Kiểm định & {str(dates[n_tr])[:10]} & {str(dates[-1])[:10]} & {n - n_tr} & {so(100 * (n - n_tr) / n, 2)}\%",
             rf"Tổng & {str(dates[0])[:10]} & {str(dates[-1])[:10]} & {n} & 100\%"]
    t2 = bang("l l l r r",
              r"\textbf{Tập} & \textbf{Từ} & \textbf{Đến} & \textbf{Số phiên} & \textbf{Tỷ lệ}",
              hang2, "Cách chia tập huấn luyện / kiểm định", "tab:chia-tap", co=r"\small")
    return r"\section{Dữ liệu}" + "\n" + t1 + t2


# ------------------------------------------------------------------ 2. buoc 1
MA_CU = {"garch_n": "b1", "garch_t": "b2", "garch_x_t": "b3", "ms_garch_n": "b4", "ms_garch_x_t": "m1"}


def _nap_o(code: str, pp: str) -> dict | None:
    p = RES / "pipeline" / f"{code}_{pp}.pt"
    if p.exists():
        return torch.load(p, weights_only=False)
    ma = MA_CU.get(code)
    if ma is None:
        return None
    o = torch.load(RES / "06_oos.pt", weights_only=False)
    if pp == "mle":
        m = torch.load(RES / "02_mle.pt", weights_only=False)
        if ma not in m:
            return None
        return {"loglik": m[ma].loglik, "aic": m[ma].aic, "bic": m[ma].bic, "oos": o.get(ma, {})}
    if code == "ms_garch_x_t":
        v = torch.load(RES / "03_vb.pt", weights_only=False)
        kho = {"vb_mean_field": "m2", "vb_full_rank": "m3"}[pp]
        return {"final_elbo": v[kho].final_elbo, "oos": o.get(kho, {})} if kho in v else None
    return None


def phan_buoc_1() -> str:
    d = torch.load(RES / "00_data.pt", weights_only=False)
    k = int(d["X_all"].shape[1])
    n_te = int(d["r_all"].shape[0]) - int(d["n_train"])
    hang1, hang2, hang3 = [], [], []
    for spec in model_spec.CATALOGUE:
        for pp in PP:
            o = _nap_o(spec.code, pp)
            if o is None:
                continue
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
    t1 = bang("l l r r r r r",
              r"\textbf{Mô hình} & \textbf{PP} & $d$ & \textbf{log-lik} & \textbf{AIC} & \textbf{BIC} & \textbf{ELBO}",
              hang1, r"Bước 1 --- khớp trong mẫu ($n_{\text{huấn luyện}}=1735$). $d$ = số tham số tự do; "
              r"AIC $=2d-2\ell$, BIC $=d\ln n-2\ell$; ELBO chỉ có với VB", "tab:b1-trong-mau")
    t2 = bang("l l r r r r",
              r"\textbf{Mô hình} & \textbf{PP} & \textbf{MAE} & \textbf{RMSE} & \textbf{QLIKE} & \textbf{NLS}",
              hang2, rf"Bước 1 --- dự báo ngoài mẫu ($n_{{\text{{kiểm định}}}}={n_te}$): MAE, RMSE, QLIKE "
              r"so với $RV_t$; NLS = điểm log âm trung bình. Nhỏ hơn là tốt hơn", "tab:b1-ngoai-mau")
    t3 = bang("l l r r r r r r",
              r"\textbf{Mô hình} & \textbf{PP} & $n_{1\%}$ & $p_{\text{UC}}$ & $p_{\text{CC}}$ & $n_{5\%}$ & $p_{\text{UC}}$ & $p_{\text{CC}}$",
              hang3, rf"Bước 1 --- kiểm định VaR ngoài mẫu ($n={n_te}$; kỳ vọng {so(0.01 * n_te, 1)} và "
              rf"{so(0.05 * n_te, 1)} vi phạm). $p_{{\text{{UC}}}}$: Kupiec; $p_{{\text{{CC}}}}$: Christoffersen",
              "tab:b1-var")
    return r"\section{Bước 1 --- tám mô hình, ba phương pháp, dữ liệu VN30F1M}" + "\n" + t1 + t2 + t3


# ------------------------------------------------------------------ 3. PR-A
def _that_A(scn) -> dict:
    p = scn.params
    d = {}
    for t in ("mu", "omega", "alpha", "beta"):
        v = getattr(p, t)
        d[f"{t}1"], d[f"{t}2"] = float(v[0]), float(v[1])
    for s in range(2):
        for j in range(p.gamma.shape[1]):
            d[f"gamma{s + 1}_{j}"] = float(p.gamma[s, j])
    d["p11"], d["p22"] = float(scn.P0[0, 0]), float(scn.P0[1, 1])
    d["nu"] = float(p.nu)
    return d


def _bang_theta0(code: str, that: dict, thu_tu: list, nhan: dict, T: int, ghi_chu: str) -> str:
    nua = (len(thu_tu) + 1) // 2
    trai, phai = thu_tu[:nua], thu_tu[nua:]
    hang = []
    for i in range(nua):
        l = f"{nhan[trai[i]]} & {so(that[trai[i]], 3)}"
        r_ = f"{nhan[phai[i]]} & {so(that[phai[i]], 3)}" if i < len(phai) else "&"
        hang.append(f"{l} & {r_}")
    return bang("l r l r", r"\textbf{Tham số} & $\theta_0$ & \textbf{Tham số} & $\theta_0$", hang,
                rf"{code}: tham số thật $\theta_0$; $T={T}${ghi_chu}", f"tab:theta0-{code}", co=r"\small")


def _bang_khoi_phuc(code: str, pp: str, m: dict, that: dict, thu_tu: list, nhan: dict) -> str:
    cov = m.get("coverage") or {}
    n = m["n_replications"]
    n_ok = m.get("n_se_ok")
    # MLE: CP Wald chi co nghia khi du lan lap co Hessian xac dinh duong
    if pp == "mle" and n_ok is not None and n_ok < TOI_THIEU_SE:
        cov = {}
    hang = []
    for kk in thu_tu:
        b, r_ = m["bias"][kk], m["rmse"][kk]
        se = max(r_ * r_ - b * b, 0.0) ** 0.5
        c = cov.get(kk)
        hang.append(f"{nhan[kk]} & {so(that[kk])} & {so(that[kk] + b)} & {so(b)} & {so(r_)} & {so(se)} & {pct(c)}")
    cap = rf"{code} --- {TEN_PP_DAI[pp]}, $B={n}$ lần lặp"
    if m.get("mean_brier") is not None:
        cap += rf", Brier $={so(m['mean_brier'], 4)}$"
    ghi = None
    if pp == "mle" and n_ok is not None:
        if cov:
            cap += rf"; CP 95\% Wald trên {n_ok}/{n} lần lặp có Hessian xác định dương"
        else:
            cap += rf"; CP 95\% bỏ trống vì chỉ {n_ok}/{n} lần lặp có Hessian xác định dương"
    return bang("l r r r r r r",
                r"\textbf{Tham số} & $\theta_0$ & \textbf{Mean} & \textbf{Bias} & \textbf{RMSE} & \textbf{SE} & \textbf{CP 95\%}",
                hang, cap, f"tab:{code}-{pp}", ghi_chu=ghi)


def phan_pr_a() -> tuple[str, dict]:
    scns = default_scenarios()
    out = [r"\section{Bước 2 --- PR-A: khôi phục tham số MS-GARCH-X Student-$t$ (8 kịch bản)}"]
    ket_qua = {}
    for code, scn in scns.items():
        f = RES / f"07_simulation_{code}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        that = _that_A(scn)
        ket_qua[code] = d
        out.append(rf"\subsection{{{code} --- \texorpdfstring{{{MO_TA_S[code]}}}{{{MO_TA_S_CHU[code]}}}}}")
        gc = r", burn-in $=200$"
        if scn.misspecified:
            gc += r"; chuỗi ước lượng là $r_t+0{,}3\,r_{t-1}$"
        out.append(_bang_theta0(code, that, THU_TU_A, NHAN_A, scn.T, gc))
        for pp in PP:
            if pp in d:
                out.append(_bang_khoi_phuc(code, pp, d[pp], that, THU_TU_A, NHAN_A))
        n_dao = " ; ".join(f"{TEN_PP[pp]} {d[pp].get('n_dao_nhan', '--')}/{d[pp]['n_replications']}"
                           for pp in PP if pp in d)
        out.append(rf"\noindent\scriptsize Số lần lặp đảo nhãn trạng thái (đã hoán vị trước khi gộp): {n_dao}.\normalsize")
        out.append("")
    return "\n".join(out), ket_qua


# ------------------------------------------------------------------ 4. PR-B
def phan_pr_b() -> tuple[str, dict]:
    scns = default_egarch_scenarios()
    out = [r"\section{Bước 2 --- PR-B: khôi phục tham số EGARCH-X Student-$t$ (3 kịch bản)}"]
    ket_qua = {}
    for code, scn in scns.items():
        f = RES / "mo_phong_egarch" / f"{code}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        that = {k_: float(v) for k_, v in d["mle"]["tham_so_that"].items()}
        ket_qua[code] = d
        out.append(rf"\subsection{{{code} --- \texorpdfstring{{{MO_TA_E[code]}}}{{{MO_TA_E_CHU[code]}}}}}")
        out.append(_bang_theta0(code, that, THU_TU_B, NHAN_B, scn.T, r", burn-in $=200$"))
        for pp in PP:
            if pp in d:
                out.append(_bang_khoi_phuc(code, pp, d[pp], that, THU_TU_B, NHAN_B))
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
    mle_v = {"mu": float(mle.mu), "omega": float(mle.omega), "alpha": float(mle.alpha),
             "theta": float(mle.theta), "beta": float(mle.beta),
             "gamma0": float(mle.gamma[0]), "gamma1": float(mle.gamma[1]), "nu": float(mle.nu)}
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
              hang, rf"Bước 3 --- MCMC (NUTS) trên EGARCH-X Student-$t$, dữ liệu huấn luyện: "
              rf"4 chuỗi $\times$ 400 mẫu, $\max\hat R={so(m['max_rhat'], 4)}$, "
              rf"$\min$ ESS $={m['min_ess']:.0f}$, {m['n_divergences']} phân kỳ, {m['giay'] / 3600:.1f} giờ",
              "tab:b3-mcmc", co=r"\small")
    t2 = bang("l r r r",
              r"\textbf{Tham số} & $\mathrm{sd}_{\text{MCMC}}$ & $R_{sd}$ VB-MF & $R_{sd}$ VB-FR",
              hang_r, r"Bước 3 --- $R_{sd}=\mathrm{sd}_{\text{VB}}/\mathrm{sd}_{\text{MCMC}}$ "
              r"(độ lệch chuẩn hậu nghiệm VB so với MCMC, cùng dữ liệu)", "tab:b3-rsd", co=r"\small")
    return r"\section{Bước 3 --- MCMC trên mô hình thắng cuộc}" + "\n" + t1 + t2


# ------------------------------------------------------------------ 6. dem thang
def phan_dem_thang(kq_a: dict, kq_b: dict) -> str:
    hang = []
    for code, d in list(kq_a.items()) + list(kq_b.items()):
        pps = [pp for pp in PP if pp in d]
        tham = list(d[pps[0]]["rmse"].keys())
        dem = {pp: 0 for pp in pps}
        for t in tham:
            best = min(pps, key=lambda pp: d[pp]["rmse"][t])
            dem[best] += 1
        n_txt = "/".join(str(d[pp]["n_replications"]) for pp in pps)
        hang.append(f"{code} & {len(tham)} & {n_txt} & " + " & ".join(
            str(dem[pp]) if pp in dem else "--" for pp in PP))
    return r"\section{Đếm số tham số có RMSE nhỏ nhất theo phương pháp}" + "\n" + bang(
        "l r l r r r",
        r"\textbf{Kịch bản} & \textbf{Số tham số} & $B$ (MLE/MF/FR) & \textbf{MLE} & \textbf{VB-MF} & \textbf{VB-FR}",
        hang, "Số tham số mà mỗi phương pháp đạt RMSE nhỏ nhất, theo kịch bản. "
        "Kịch bản không có VB-FR ghi $--$", "tab:dem-thang", co=r"\small")


# ------------------------------------------------------------------ van ban
PREAMBLE = r"""% !TeX program = xelatex
% !TeX encoding = UTF-8
% SINH TU DONG boi scripts/bao_cao/sinh_tong_hop_ket_qua.py -- khong sua tay.
\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage{unicode-math}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Liberation Serif}}
\IfFontExistsTF{Cambria Math}{\setmathfont{Cambria Math}}{\setmathfont{Latin Modern Math}}
\usepackage[a4paper,top=2.2cm,bottom=2.2cm,left=2.0cm,right=2.0cm]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{float}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\renewcommand{\tablename}{Bảng}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.3em}

\title{\vspace{-1.2cm}\textbf{Tổng hợp kết quả}\\[0.2em]\large Mô hình biến động VN30F1M: ước lượng, mô phỏng khôi phục tham số và MCMC}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-1.2cm}
\tableofcontents
\clearpage
"""

CHU_THICH = r"""
\section*{Ký hiệu}
\begin{tabular}{ll}
\toprule
MLE / VB-MF / VB-FR / MCMC & hợp lý cực đại / Bayes biến phân mean-field / full-rank / NUTS \\
$d$, $\ell$ & số tham số tự do, log-hợp-lý tại nghiệm \\
QLIKE & $\frac1N\sum_t[\log\hat h_t + RV_t/\hat h_t]$ \\
NLS & $-\frac1N\sum_t\log\hat f_t(r_t)$ (VB, MCMC: mật độ dự báo hỗn hợp qua mẫu hậu nghiệm) \\
Mean, Bias, RMSE, SE & $\bar\theta$, $\bar\theta-\theta_0$, $\sqrt{\frac1B\sum(\hat\theta_b-\theta_0)^2}$, $\sqrt{\text{RMSE}^2-\text{Bias}^2}$ \\
CP 95\% & tỷ lệ lần lặp mà khoảng 95\% (VB: phân vị hậu nghiệm; MLE: Wald từ Hessian quan sát) chứa $\theta_0$ \\
Brier & $\frac1T\sum_t\sum_{s=1}^2[\hat\xi_{t|t}(s)-\mathbb 1\{S_t=s\}]^2$, thang $[0,2]$, xác suất lọc sau khi sắp nhãn \\
$R_{sd}$ & $\mathrm{sd}_{\text{VB}}/\mathrm{sd}_{\text{MCMC}}$ trên cùng dữ liệu \\
\bottomrule
\end{tabular}
\clearpage
"""


def main() -> None:
    parts = [PREAMBLE, CHU_THICH, phan_du_lieu(), phan_buoc_1()]
    a, kq_a = phan_pr_a()
    b, kq_b = phan_pr_b()
    parts += [a, b, phan_buoc_3(), phan_dem_thang(kq_a, kq_b), r"\end{document}", ""]
    tex = "\n".join(parts)
    OUT.write_text(tex, encoding="utf-8")
    # kiem tra can bang moi truong
    for env in ("table", "tabular", "longtable", "document"):
        mo, dong = tex.count(rf"\begin{{{env}}}"), tex.count(rf"\end{{{env}}}")
        assert mo == dong, f"{env}: {mo} begin / {dong} end"
    print(f"-> {OUT}  ({len(tex.splitlines())} dong, {tex.count(chr(92) + 'begin{table}') + tex.count(chr(92) + 'begin{longtable}')} bang)")


if __name__ == "__main__":
    main()
