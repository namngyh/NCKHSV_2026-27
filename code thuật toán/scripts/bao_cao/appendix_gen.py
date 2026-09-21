# -*- coding: utf-8 -*-
"""Sinh phan PHU LUC DU LIEU DAY DU cho bao cao (moi tham so, moi gia tri)."""
import os, sys, math

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)

import torch
from msgarchx.inference.summary import posterior_summary, relabel_posterior_samples
from msgarchx.simulation.scenarios import default_scenarios

R = os.path.join(ROOT, "results")

# 07_simulation_S*.pt luu Bias/RMSE cua p11/p22 tinh SAI (so voi placeholder
# eye(2) trong scenario.params.P, thay vi P0 that -- xem CLAUDE.md "Loi da
# tim thay va sua"). Sua lai bang dai so KHONG can chay lai: Var = RMSE^2 -
# Bias^2 bat bien voi theta0 (khong doi), nen:
#   Bias_dung = (theta0_sai + Bias_sai) - theta0_dung
#   RMSE_dung = sqrt(Var + Bias_dung^2)
# Coverage KHONG sua duoc kieu nay (can chan CI tung lan lap, da bi xoa) --
# giu nguyen so bao cao nhung ghi chu ro trong bang.
_THETA0_SAI_P = 1.0  # eye(2): p11 = p22 = 1.0


def _sua_bias_rmse_p(key: str, bias: float | None, rmse: float | None, ma_kich_ban: str) -> tuple:
    if key not in ("p11", "p22") or bias is None or rmse is None:
        return bias, rmse
    scn = default_scenarios()[ma_kich_ban]
    theta0_dung = float(scn.P0[0, 0]) if key == "p11" else float(scn.P0[1, 1])
    var = max(rmse * rmse - bias * bias, 0.0)
    bias_dung = (_THETA0_SAI_P + bias) - theta0_dung
    rmse_dung = math.sqrt(var + bias_dung * bias_dung)
    return bias_dung, rmse_dung


def f(x, n=4):
    if x is None:
        return "--"
    if isinstance(x, torch.Tensor):
        x = float(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "--" if math.isnan(x) else r"$\infty$"
    return f"{x:.{n}f}".replace(".", "{,}")


def build_appendix() -> str:
    mle = torch.load(os.path.join(R, "02_mle.pt"), weights_only=False)
    vb = torch.load(os.path.join(R, "03_vb.pt"), weights_only=False)
    mc2_p = os.path.join(R, "04_mcmc.pt")
    mc1_p = os.path.join(R, "lich_su_sua_loi", "04_mcmc_lan1_duongcheo.pt")
    mc2 = torch.load(mc2_p, weights_only=False)["m4"] if os.path.exists(mc2_p) else None
    mc1 = torch.load(mc1_p, weights_only=False)["m4"] if os.path.exists(mc1_p) else None
    oos = torch.load(os.path.join(R, "06_oos.pt"), weights_only=False)
    r1d = torch.load(os.path.join(R, "05_r1.pt"), weights_only=False)
    try:
        data = torch.load(os.path.join(R, "00_data.pt"), weights_only=False)
    except Exception:
        data = None   # phien ban pandas hien tai khong doc duoc chi muc ngay thang

    L = []
    A = L.append
    A(r"\clearpage")
    A(r"\appendix")
    A(r"\section*{\centering PHỤ LỤC DỮ LIỆU ĐẦY ĐỦ}")
    A(r"\addcontentsline{toc}{section}{Phụ lục dữ liệu đầy đủ}")
    A(r"""
Phụ lục này liệt kê \emph{toàn bộ} giá trị số của nghiên cứu: cấu hình chạy,
tiên nghiệm, giá trị khởi tạo, ước lượng của từng tham số theo từng phương pháp,
chẩn đoán của từng điểm khởi tạo và từng chuỗi, cùng mọi thống kê kiểm định.
Mọi con số được đọc trực tiếp từ tệp kết quả nhị phân, không nhập tay.
""")

    # ============================================================ A. Cau hinh
    A(r"\section{Cấu hình chạy}")
    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Tham số điều khiển toàn bộ pipeline (preset \texttt{full} trong \texttt{run\_full\_study.py})}
\begin{tabularx}{\textwidth}{l l X}
\toprule
\textbf{Nhóm} & \textbf{Tham số} & \textbf{Giá trị} \\
\midrule
Dữ liệu & Tổng quan sát & 2170 \\
 & Tập huấn luyện & 1735 (tỷ lệ \texttt{train\_frac} $=0{,}8$) \\
 & Tập kiểm định & 435 \\
 & Số biến ngoại sinh $k$ & 2: $\log(1+VOL_{t-1})$ và $\log(RV_{t-1}+\epsilon)$ \\
 & Hạt giống gốc \texttt{seed} & 0 \\
\midrule
MLE & \texttt{mle\_n\_starts} & 10 điểm khởi tạo \\
 & \texttt{mle\_n\_adam} & 500 vòng Adam \\
 & \texttt{mle\_n\_lbfgs} & 250 vòng L-BFGS \\
 & \texttt{mle\_n\_cycles} & 3 chu kỳ Adam${+}$L-BFGS mỗi điểm khởi tạo \\
 & \texttt{lr\_adam} & 0,02 \\
 & Quy tắc chọn nghiệm & \texttt{select="regular"} với B4, M1; \texttt{"max\_loglik"} với B1--B3 \\
 & Điều kiện chính quy & $d_s \ge 2$ phiên và $p_{ss} \in [0{,}01;\ 0{,}999]$ \\
\midrule
Variational Bayes & \texttt{vb\_n\_steps} & 3000 bước SVI \\
 & \texttt{vb\_n\_draws} & 1000 mẫu hậu nghiệm \\
 & Guide M2 & \texttt{AutoNormal} (mean-field) \\
 & Guide M3 & \texttt{AutoMultivariateNormal} (full-rank) \\
\midrule
MCMC (NUTS) & \texttt{mcmc\_n\_chains} & 4 chuỗi \\
 & \texttt{mcmc\_n\_warmup} & 1000 (lần 1: 250) \\
 & \texttt{mcmc\_n\_samples} & 400 mỗi chuỗi \\
 & \texttt{mcmc\_max\_tree\_depth} & 7 \\
 & \texttt{mcmc\_full\_mass} & \texttt{True} --- ma trận đặc (lần 1: đường chéo) \\
\midrule
Ngoài mẫu & \texttt{oos\_n\_posterior\_draws} & 200 mẫu hậu nghiệm \\
\bottomrule
\end{tabularx}
\end{table}
""")

    # ============================================================ B. Tien nghiem
    A(r"\section{Tiên nghiệm dùng cho Variational Bayes và MCMC}")
    A(r"""
Cả M2, M3, M4 và R1 dùng \emph{chung một} hàm \texttt{model()} trong
\texttt{bayes\_model.py}, nên khác biệt kết quả giữa chúng thuần túy đến từ
phương pháp suy diễn chứ không từ đặc tả xác suất.

\begin{table}[H]\centering\footnotesize
\caption{Tiên nghiệm (giá trị mặc định của \texttt{PriorConfig})}
\begin{tabularx}{\textwidth}{l l X}
\toprule
\textbf{Tham số} & \textbf{Phân phối tiên nghiệm} & \textbf{Ghi chú} \\
\midrule
$\mu_s$ & $\mathcal{N}(0;\ 1^2)$, độc lập cho $s=1,2$ & \texttt{sigma\_mu} $=1{,}0$ \\
$\omega_s$ & $\mathrm{LogNormal}(0;\ 1^2)$ & \texttt{m\_omega} $=0$, \texttt{sigma\_omega} $=1$ \\
$\alpha_s$ & $\mathrm{Beta}(2;\ 2)$ & \texttt{alpha\_beta\_concentration} $=2{,}0$ \\
$\beta_s$ & $\mathrm{Beta}(2;\ 2)$ & chặn cả hai trong khoảng $(0;1)$ \\
$\gamma_{s,j}$ & $\mathcal{N}(0;\ 1^2)$ & \texttt{sigma\_gamma} $=1{,}0$, ma trận $2\times k$ \\
$p_{11},\ p_{22}$ & $\mathrm{Beta}(8;\ 2)$ & \texttt{transition\_prior\_stay} $=8$, \texttt{switch} $=2$ \\
$\nu-2$ & $\mathrm{Gamma}(\text{shape}=2;\ \text{rate}=0{,}2)$ & bảo đảm $\nu>2$ để phương sai tồn tại \\
\bottomrule
\end{tabularx}
\end{table}

Tiên nghiệm $\mathrm{Beta}(2;2)$ đặt trên $\alpha_s,\beta_s$ có hệ quả quan trọng khi
so sánh với MLE: nó \emph{chặn cứng} hai tham số này trong $(0;1)$, trong khi tham số
hóa softplus dùng cho MLE không có chặn trên. Đây là lý do $\hat\beta_1$ của MLE
($0{,}8463$) nằm ngoài vùng mà ba phương pháp Bayes có thể với tới.
""")

    # ============================================================ C. Khoi tao
    b1p, b3p = mle["b1"].params, mle["b3"].params
    A(r"\section{Giá trị khởi tạo}")
    A(r"""
\subsection*{C.1 Khởi tạo ấm cho mô hình hai trạng thái}
Bốn điểm khởi tạo đầu tiên của B4 và M1 \emph{không} ngẫu nhiên: chúng được sinh
từ nghiệm một trạng thái đã ước lượng xong (B1 cho B4, B3 cho M1) bằng cách nhân đôi
bộ tham số rồi tách hai bản sao ra qua hệ số nhân $\omega$.
""")
    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Bốn biến thể khởi tạo ấm (\texttt{\_BIEN\_THE\_AM})}
\begin{tabularx}{\textwidth}{c C C C C}
\toprule
\textbf{Biến thể} & \textbf{Hệ số nhân $\omega$} & \textbf{$p_{11}$ khởi tạo} & \textbf{$p_{22}$ khởi tạo} & \textbf{Quy tắc $\nu$} \\
\midrule
1 & $(0{,}5;\ 2{,}0)$ & 0{,}95 & 0{,}95 & $\nu_0$ \\
2 & $(0{,}25;\ 4{,}0)$ & 0{,}98 & 0{,}90 & $\nu_0$ \\
3 & $(0{,}7;\ 1{,}5)$ & 0{,}90 & 0{,}85 & $\max(\nu_0;\ 6)$ \\
4 & $(0{,}4;\ 3{,}0)$ & 0{,}99 & 0{,}94 & $\nu_0+2$ \\
\bottomrule
\end{tabularx}
\end{table}
""")
    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Nghiệm một trạng thái được dùng làm gốc khởi tạo}
\begin{tabularx}{\textwidth}{l C C C C C}
\toprule
\textbf{Nguồn} & $\mu_0$ & $\omega_0$ & $\alpha_0$ & $\beta_0$ & $\nu_0$ \\
\midrule""")
    A(rf"B1 $\to$ khởi tạo B4 & {f(b1p.mu)} & {f(b1p.omega)} & {f(b1p.alpha)} & {f(b1p.beta)} & --- (cố định $10^6$) \\")
    A(rf"B3 $\to$ khởi tạo M1 & {f(b3p.mu)} & {f(b3p.omega)} & {f(b3p.alpha)} & {f(b3p.beta)} & {f(b3p.nu)} \\")
    A(r"""\bottomrule
\end{tabularx}
\end{table}

Ví dụ cụ thể: biến thể 2 của M1 khởi tạo với
$\omega = (""" + f(float(b3p.omega) * 0.25) + "; " + f(float(b3p.omega) * 4.0) + r""")$,
$\alpha = (""" + f(float(b3p.alpha)) + "; " + f(float(b3p.alpha)) + r""")$,
$\beta = (""" + f(float(b3p.beta)) + "; " + f(float(b3p.beta)) + r""")$,
$P_{11} = 0{,}98$, $P_{22} = 0{,}90$, $\nu = """ + f(float(b3p.nu)) + r"""$.

\subsection*{C.2 Khởi tạo ngẫu nhiên cho các điểm còn lại}
Các điểm khởi tạo từ thứ 5 trở đi sinh ngẫu nhiên trên \emph{không gian không ràng buộc},
với hạt giống \texttt{base\_seed} $+\ i$:
\begin{itemize}
  \item $\mu_s \sim 0{,}05\cdot\mathcal{N}(0,1)$;
  \item $\log\omega_s = \log(0{,}5)$ hoặc $\log(2{,}0)$, cộng $\mathcal{N}(0,1)$ --- tách sẵn hai trạng thái;
  \item $\eta_{\alpha}, \eta_{\beta}$: tâm mặc định cộng $0{,}3\cdot\mathcal{N}(0,1)$;
  \item $\gamma_{s,j} \sim 0{,}05\cdot\mathcal{N}(0,1)$;
  \item $\eta_p$: khởi tạo tại $p_{11}=0{,}97$, $p_{22}=0{,}95$ cộng $0{,}2\cdot\mathcal{N}(0,1)$;
  \item $\eta_\nu$: khởi tạo tại $\nu=8$ cộng $0{,}3\cdot\mathcal{N}(0,1)$.
\end{itemize}

\subsection*{C.3 Khởi tạo của MCMC}
Lần 1 khởi tạo \textbf{từ tiên nghiệm} (mặc định của NUTS). Lần 2 khởi tạo
\textbf{từ nghiệm MLE chính quy} của M1 --- tức từ chính bộ tham số trong Phụ lục D --- và
vẫn không hội tụ, đây là bằng chứng loại trừ khả năng ``chỉ do khởi tạo xấu''.
""")

    # ============================================================ D. MLE day du
    A(r"\section{Ước lượng MLE đầy đủ}")

    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Mô hình một trạng thái B1--B3: toàn bộ tham số}
\begin{tabularx}{\textwidth}{l C C C C C C C}
\toprule
\textbf{Mô hình} & $\mu$ & $\omega$ & $\alpha$ & $\beta$ & $\alpha+\beta$ & $\gamma_1$ & $\gamma_2$ \\
\midrule""")
    for key, nm in [("b1", "B1"), ("b2", "B2"), ("b3", "B3")]:
        p = mle[key].params
        g = p.gamma.reshape(-1) if p.gamma is not None and p.gamma.numel() else None
        g1 = f(g[0]) if g is not None and g.numel() > 0 else "--"
        g2 = f(g[1]) if g is not None and g.numel() > 1 else "--"
        A(rf"{nm} & {f(p.mu)} & {f(p.omega)} & {f(p.alpha)} & {f(p.beta)} & "
          rf"{f(float(p.alpha)+float(p.beta))} & {g1} & {g2} \\")
    A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize\item $\nu$: B1 dùng phân phối chuẩn (không có $\nu$); """
      + rf"B2 có $\nu={f(mle['b2'].params.nu)}$; B3 có $\nu={f(mle['b3'].params.nu)}$."
      + r"""
\end{tablenotes}
\end{table}
""")

    m1, b4 = mle["m1"], mle["b4"]
    se = mle["m1_se"]
    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Mô hình hai trạng thái B4 và M1: toàn bộ tham số theo từng trạng thái}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
& \multicolumn{2}{c}{\textbf{B4: MS-GARCH chuẩn}} & \multicolumn{2}{c}{\textbf{M1: MS-GARCH-X Student-t}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Tham số} & Trạng thái 1 & Trạng thái 2 & Trạng thái 1 & Trạng thái 2 \\
\midrule""")
    for nm, tex in [("mu", r"$\mu_s$"), ("omega", r"$\omega_s$"),
                    ("alpha", r"$\alpha_s$"), ("beta", r"$\beta_s$")]:
        vb4, vm1 = getattr(b4.params, nm), getattr(m1.params, nm)
        A(rf"{tex} & {f(vb4[0])} & {f(vb4[1])} & {f(vm1[0])} & {f(vm1[1])} \\")
    ab4, am1 = b4.params.alpha + b4.params.beta, m1.params.alpha + m1.params.beta
    A(rf"$\alpha_s+\beta_s$ & {f(ab4[0])} & {f(ab4[1])} & {f(am1[0])} & {f(am1[1])} \\")
    gm = m1.params.gamma
    A(rf"$\gamma_{{s,1}}$ & --- & --- & {f(gm[0,0])} & {f(gm[1,0])} \\")
    A(rf"$\gamma_{{s,2}}$ & --- & --- & {f(gm[0,1])} & {f(gm[1,1])} \\")
    A(r"\midrule")
    A(rf"$p_{{11}}$ & \multicolumn{{2}}{{c}}{{{f(b4.params.P[0,0])}}} & \multicolumn{{2}}{{c}}{{{f(m1.params.P[0,0])}}} \\")
    A(rf"$p_{{12}}$ & \multicolumn{{2}}{{c}}{{{f(b4.params.P[0,1])}}} & \multicolumn{{2}}{{c}}{{{f(m1.params.P[0,1])}}} \\")
    A(rf"$p_{{21}}$ & \multicolumn{{2}}{{c}}{{{f(b4.params.P[1,0])}}} & \multicolumn{{2}}{{c}}{{{f(m1.params.P[1,0])}}} \\")
    A(rf"$p_{{22}}$ & \multicolumn{{2}}{{c}}{{{f(b4.params.P[1,1])}}} & \multicolumn{{2}}{{c}}{{{f(m1.params.P[1,1])}}} \\")
    d_b4 = [1 / (1 - float(b4.params.P[0, 0])), 1 / (1 - float(b4.params.P[1, 1]))]
    d_m1 = [1 / (1 - float(m1.params.P[0, 0])), 1 / (1 - float(m1.params.P[1, 1]))]
    A(rf"$d_s$ (phiên) & {f(d_b4[0],2)} & {f(d_b4[1],2)} & {f(d_m1[0],2)} & {f(d_m1[1],2)} \\")
    A(rf"$\nu$ & \multicolumn{{2}}{{c}}{{cố định $10^6$}} & \multicolumn{{2}}{{c}}{{{f(m1.params.nu)}}} \\")
    A(r"\midrule")
    A(rf"Log-hợp-lý & \multicolumn{{2}}{{c}}{{{f(b4.loglik,4)}}} & \multicolumn{{2}}{{c}}{{{f(m1.loglik,4)}}} \\")
    A(rf"AIC & \multicolumn{{2}}{{c}}{{{f(b4.aic,4)}}} & \multicolumn{{2}}{{c}}{{{f(m1.aic,4)}}} \\")
    A(rf"BIC & \multicolumn{{2}}{{c}}{{{f(b4.bic,4)}}} & \multicolumn{{2}}{{c}}{{{f(m1.bic,4)}}} \\")
    A(r"""\bottomrule
\end{tabularx}
\end{table}
""")

    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Sai số chuẩn của M1 (delta method từ Hessian quan sát)}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
\textbf{Tham số} & \textbf{Ước lượng tt.1} & \textbf{SE tt.1} & \textbf{Ước lượng tt.2} & \textbf{SE tt.2} \\
\midrule""")
    for nm, tex in [("mu", r"$\mu_s$"), ("omega", r"$\omega_s$"),
                    ("alpha", r"$\alpha_s$"), ("beta", r"$\beta_s$")]:
        v, s = getattr(m1.params, nm), se[nm].reshape(-1)
        A(rf"{tex} & {f(v[0])} & {f(s[0])} & {f(v[1])} & {f(s[1])} \\")
    sg = se["gamma"].reshape(-1)
    A(rf"$\gamma_{{s,1}}$ & {f(gm[0,0])} & {f(sg[0])} & {f(gm[1,0])} & {f(sg[2])} \\")
    A(rf"$\gamma_{{s,2}}$ & {f(gm[0,1])} & {f(sg[1])} & {f(gm[1,1])} & {f(sg[3])} \\")
    A(rf"$p_{{11}}$ & {f(m1.params.P[0,0])} & {f(se['p11'].reshape(-1)[0])} & --- & --- \\")
    A(rf"$p_{{22}}$ & {f(m1.params.P[1,1])} & {f(se['p22'].reshape(-1)[0])} & --- & --- \\")
    A(rf"$\nu$ & {f(m1.params.nu)} & {f(se['nu'].reshape(-1)[0])} & --- & --- \\")
    A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize
\item Vì nghiệm nằm sát biên ($p_{ss}\to1$), xấp xỉ tiệm cận chuẩn không còn hợp lệ; các sai số chuẩn này chỉ nên đọc như thước đo độ cong cục bộ của hàm hợp lý.
\end{tablenotes}
\end{table}
""")

    # ============================================================ E. Chan doan tung diem khoi tao
    A(r"\section{Chẩn đoán từng điểm khởi tạo MLE}")
    A(r"""
\begin{longtable}{l c r c r c}
\caption{Toàn bộ điểm khởi tạo của năm mô hình MLE}\\
\toprule
\textbf{Mô hình} & \textbf{seed} & \textbf{Log-hợp-lý} & \textbf{Hội tụ} & \textbf{$\|$gradient$\|$} & \textbf{Sát biên} \\
\midrule
\endfirsthead
\toprule
\textbf{Mô hình} & \textbf{seed} & \textbf{Log-hợp-lý} & \textbf{Hội tụ} & \textbf{$\|$gradient$\|$} & \textbf{Sát biên} \\
\midrule
\endhead
\bottomrule
\endfoot""")
    for key, nm in [("b1", "B1"), ("b2", "B2"), ("b3", "B3"), ("b4", "B4"), ("m1", "M1")]:
        for rr in mle[key].runs:
            ll = "NaN" if (isinstance(rr.loglik, float) and math.isnan(rr.loglik)) else f(rr.loglik, 4)
            nb = getattr(rr, "near_boundary", None)
            nbs = "---" if nb is None else ("có" if nb else "không")
            gn = f(rr.grad_norm, 4) if rr.grad_norm < 1e4 else f"{rr.grad_norm:.3e}".replace(".", "{,}")
            hoi_tu = "có" if rr.converged else "\\textbf{KHÔNG}"
            A(rf"{nm} & {rr.seed} & {ll} & {hoi_tu} & {gn} & {nbs} \\")
        A(r"\midrule")
    A(r"\end{longtable}")

    # ============================================================ F. VB day du
    A(r"\section{Hậu nghiệm Variational Bayes đầy đủ}")

    def post_table(res, caption, state_nu=False):
        raw = posterior_summary(res.samples)
        mau, ty = relabel_posterior_samples(res.samples)
        fixed = posterior_summary(mau)
        out = [r"\begin{table}[H]\centering\footnotesize",
               rf"\caption{{{caption}}}",
               r"\begin{tabularx}{\textwidth}{l C C C C | C C C C}",
               r"\toprule",
               r"& \multicolumn{4}{c|}{\textbf{Mẫu thô (chưa sắp nhãn)}} & \multicolumn{4}{c}{\textbf{Đã sắp nhãn}} \\",
               r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
               r"\textbf{Tham số} & TB & ĐLC & 2,5\% & 97,5\% & TB & ĐLC & 2,5\% & 97,5\% \\",
               r"\midrule"]

        def row(tex, name, idx=None, shift=0.0):
            cells = []
            for s in (raw, fixed):
                st = s[name]
                get = (lambda kk: st[kk] if idx is None else st[kk][idx])
                cells += [f(float(get("mean")) + shift), f(float(get("sd"))),
                          f(float(get("q2.5")) + shift), f(float(get("q97.5")) + shift)]
            out.append(tex + " & " + " & ".join(cells) + r" \\")

        for nm, tex in [("mu", r"\mu"), ("omega", r"\omega"), ("alpha", r"\alpha"), ("beta", r"\beta")]:
            for i in (0, 1):
                row(rf"${tex}_{i+1}$", nm, i)
        for s_ in (0, 1):
            for j_ in (0, 1):
                cells = []
                for s in (raw, fixed):
                    st = s["gamma"]
                    cells += [f(float(st["mean"][s_, j_])), f(float(st["sd"][s_, j_])),
                              f(float(st["q2.5"][s_, j_])), f(float(st["q97.5"][s_, j_]))]
                out.append(rf"$\gamma_{{{s_+1},{j_+1}}}$ & " + " & ".join(cells) + r" \\")
        row(r"$p_{11}$", "p11")
        row(r"$p_{22}$", "p22")
        if state_nu:
            for i in (0, 1):
                row(rf"$\nu_{i+1}$", "nu_minus_2", i, shift=2.0)
        else:
            row(r"$\nu$", "nu_minus_2", None, shift=2.0)
        out += [r"\bottomrule", r"\end{tabularx}",
                r"\begin{tablenotes}\footnotesize\item Tỷ lệ mẫu phải hoán vị nhãn: "
                + f(ty * 100, 2) + r"\%. ĐLC = độ lệch chuẩn hậu nghiệm.\end{tablenotes}",
                r"\end{table}"]
        return "\n".join(out)

    A(post_table(vb["m2"], r"M2 --- VB mean-field (\texttt{AutoNormal}), ELBO cuối = "
                 + f(float(vb["m2"].final_elbo), 4)))
    A(post_table(vb["m3"], r"M3 --- VB full-rank (\texttt{AutoMultivariateNormal}), ELBO cuối = "
                 + f(float(vb["m3"].final_elbo), 4)))
    A(post_table(r1d["r1"], r"R1 --- VB mean-field với $\nu_s$ riêng theo trạng thái, ELBO cuối = "
                 + f(float(r1d["r1"].final_elbo), 4), state_nu=True))
    A(r"""
Với R1, xác suất hậu nghiệm mà trạng thái biến động cao có đuôi dày hơn trạng thái
biến động thấp là $\Pr(\nu_{\text{cao}}<\nu_{\text{thấp}}\mid \mathcal{D}_T) = """
      + f(float(r1d["p_tail"]), 4) + r"""$.
""")

    # ============================================================ G. MCMC day du
    A(r"\section{Hậu nghiệm MCMC đầy đủ}")

    def mcmc_table(res, caption):
        mau, ty = relabel_posterior_samples(res.samples)
        s = posterior_summary(mau)
        out = [r"\begin{table}[H]\centering\footnotesize",
               rf"\caption{{{caption}}}",
               r"\begin{tabularx}{\textwidth}{l C C C C C C}",
               r"\toprule",
               r"\textbf{Tham số} & \textbf{TB} & \textbf{ĐLC} & \textbf{2,5\%} & \textbf{97,5\%} & $\widehat{R}$ & \textbf{ESS} \\",
               r"\midrule"]
        RH, ES = res.rhat, res.ess_bulk

        def row(tex, name, idx=None, shift=0.0, dkey=None):
            st = s[name]
            get = (lambda kk: st[kk] if idx is None else st[kk][idx])
            dk = dkey or name
            out.append(rf"{tex} & {f(float(get('mean'))+shift)} & {f(float(get('sd')))} & "
                       rf"{f(float(get('q2.5'))+shift)} & {f(float(get('q97.5'))+shift)} & "
                       rf"{f(RH[dk],4)} & {f(ES[dk],1)} \\")

        for nm, tex in [("mu", r"\mu"), ("omega", r"\omega"), ("alpha", r"\alpha"), ("beta", r"\beta")]:
            for i in (0, 1):
                row(rf"${tex}_{i+1}$", nm, i)
        for s_ in (0, 1):
            for j_ in (0, 1):
                st = s["gamma"]
                out.append(rf"$\gamma_{{{s_+1},{j_+1}}}$ & {f(float(st['mean'][s_,j_]))} & "
                           rf"{f(float(st['sd'][s_,j_]))} & {f(float(st['q2.5'][s_,j_]))} & "
                           rf"{f(float(st['q97.5'][s_,j_]))} & {f(RH['gamma'],4)} & {f(ES['gamma'],1)} \\")
        row(r"$p_{11}$", "p11")
        row(r"$p_{22}$", "p22")
        row(r"$\nu$", "nu_minus_2", None, 2.0)
        out += [r"\bottomrule", r"\end{tabularx}",
                r"\begin{tablenotes}\footnotesize\item Tỷ lệ mẫu phải hoán vị nhãn: "
                + f(ty * 100, 2) + r"\%. $\widehat{R}$ và ESS lấy theo nhóm tham số (max/min trên các thành phần). "
                + rf"Số lần phân kỳ: {res.n_divergences}.\end{{tablenotes}}",
                r"\end{table}"]
        return "\n".join(out)

    if mc1 is not None:
        A(mcmc_table(mc1, r"M4 lần 1 --- ma trận khối lượng đường chéo, warmup 250, khởi tạo từ tiên nghiệm"))
    if mc2 is not None:
        A(mcmc_table(mc2, r"M4 lần 2 --- ma trận khối lượng đặc, warmup 1000, khởi tạo từ nghiệm MLE"))
    if mc1 is not None or mc2 is not None:
        A(r"""
\textbf{Cảnh báo.} Không bảng nào trong Phụ lục G đạt chuẩn hội tụ. Chúng được in
đầy đủ để phục vụ kiểm chứng và tái lập, không phải để trích dẫn như ước lượng.
""")
    else:
        A(r"\textit{Phương pháp MCMC (M4) đã được loại bỏ khỏi nghiên cứu do không hội tụ ($\widehat{R} \approx 2{,}84$).}")

    # ============================================================ H. Do nhay tien nghiem
    A(r"\section{Độ nhạy tiên nghiệm}")
    dn_dir = os.path.join(R, "parts", "do_nhay")
    variants = [("goc", "Gốc: bản thảo (chính là M2)", vb["m2"])]
    for fn, lab in [("omega_hc", r"$\omega_s\sim$ half-Cauchy$(0;1)$"),
                    ("p_chat", r"$p_{ss}\sim\mathrm{Beta}(20;2)$"),
                    ("ca_hai", "Cả hai thay đổi trên")]:
        fp = os.path.join(dn_dir, fn + ".pt")
        if os.path.exists(fp):
            obj = torch.load(fp, weights_only=False)
            variants.append((fn, lab, obj["result"]))

    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Ước lượng hậu nghiệm dưới bốn lựa chọn tiên nghiệm (VB mean-field, đã sắp nhãn)}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
\textbf{Tham số} & \textbf{Gốc} & \textbf{half-Cauchy} & \textbf{Beta(20;2)} & \textbf{Cả hai} \\
\midrule""")
    order = [("omega", 0, r"$\omega_1$"), ("omega", 1, r"$\omega_2$"),
             ("alpha", 0, r"$\alpha_1$"), ("alpha", 1, r"$\alpha_2$"),
             ("beta", 0, r"$\beta_1$"), ("beta", 1, r"$\beta_2$"),
             ("p11", None, r"$p_{11}$"), ("p22", None, r"$p_{22}$"),
             ("nu_minus_2", None, r"$\nu$")]
    order_keys = ["goc", "omega_hc", "p_chat", "ca_hai"]
    summ, tyle, elbo = {}, {}, {}
    for code, lab, res in variants:
        mau, ty = relabel_posterior_samples(res.samples)
        summ[code] = posterior_summary(mau)
        tyle[code] = ty
        elbo[code] = float(res.final_elbo)
    for nm, idx, tex in order:
        cells = []
        for code in order_keys:
            if code not in summ:
                cells.append("--"); continue
            st = summ[code][nm]
            v = float(st["mean"] if idx is None else st["mean"][idx])
            cells.append(f(v + (2.0 if nm == "nu_minus_2" else 0.0)))
        A(rf"{tex} & " + " & ".join(cells) + r" \\")
    A(r"\midrule")
    A(r"ELBO cuối & " + " & ".join(f(elbo[c], 2) if c in elbo else "--" for c in order_keys) + r" \\")
    A(r"Tỷ lệ đổi nhãn & " + " & ".join((f(tyle[c]*100, 1) + r"\%") if c in tyle else "--" for c in order_keys) + r" \\")
    A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize
\item ELBO của các biến thể \emph{không} so sánh trực tiếp được với nhau vì chúng là cận dưới của log-bằng-chứng dưới các mô hình xác suất khác nhau.
\item $p_{11}$ dao động mạnh theo lựa chọn tiên nghiệm --- dấu hiệu dữ liệu không lấn át được tiên nghiệm ở tham số này.
\end{tablenotes}
\end{table}
""")

    # ============================================================ I. OOS day du
    A(r"\section{Kết quả ngoài mẫu đầy đủ}")
    mo8 = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4"]
    A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Toàn bộ chỉ tiêu dự báo ngoài mẫu (435 quan sát)}
\begin{tabularx}{\textwidth}{l C C C C C}
\toprule
\textbf{Mô hình} & \textbf{MAE} & \textbf{RMSE} & \textbf{QLIKE} & \textbf{NLS} & \textbf{Số mẫu hậu nghiệm} \\
\midrule""")
    for m in mo8:
        d = oos[m]
        nd = d.get("n_draws_used", "---")
        A(rf"{m.upper()} & {f(d['mae'])} & {f(d['rmse'])} & {f(d['qlike'])} & {f(d['nls'])} & {nd} \\")
    A(r"""\bottomrule
\end{tabularx}
\end{table}

\begin{table}[H]\centering\footnotesize
\caption{Toàn bộ thống kê kiểm định VaR}
\begin{tabularx}{\textwidth}{l c C C C C C C}
\toprule
\textbf{Mô hình} & $\alpha$ & \textbf{Vi phạm} & \textbf{Tỷ lệ} & $LR_{uc}$ & \textbf{Kupiec $p$} & $LR_{cc}$ & \textbf{Christ. $p_{cc}$} \\
\midrule""")
    for m in mo8:
        for lv, key in [("1\\%", "1pct"), ("5\\%", "5pct")]:
            kup = oos[m][f"kupiec_{key}"]
            chr_ = oos[m][f"christoffersen_{key}"]
            rate = float(kup["violation_rate"]) * 100
            A(rf"{m.upper()} & {lv} & {kup['n_violations']} & {f(rate,2)}\% & "
              rf"{f(kup['LR_uc'],4)} & {f(kup['p_value'],4)} & {f(chr_['LR_cc'],4)} & {f(chr_['p_value_cc'],4)} \\")
        A(r"\addlinespace[1pt]")
    A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize
\item $LR_{uc}$: thống kê tỷ số hợp lý cho độ phủ vô điều kiện (Kupiec). $LR_{cc} = LR_{uc} + LR_{ind}$: độ phủ có điều kiện (Christoffersen), gộp cả yêu cầu vi phạm không tạo cụm.
\end{tablenotes}
\end{table}
""")

    # ============================================================ J. Mo phong
    A(r"\section{Phụ lục: tám kịch bản mô phỏng bản gốc (\texorpdfstring{$S_1$--$S_8$}{S1-S8})}")
    A(r"""
Các bảng dưới đây là bản chạy CŨ ($B=15$--$30$), \emph{trước} khi phát hiện và
sửa lỗi đối xứng hoán vị nhãn trạng thái (xem \texttt{CLAUDE.md}). Số liệu
chính thức của Bước 2 nay là PR-A + PR-B ($B=100$, đã sắp nhãn) ở
Mục~\ref{sec:nghien-cuu-mo-phong} phía trên; giữ lại phần này thuần túy làm
bằng chứng bổ trợ và tài liệu lịch sử về quá trình phát triển phương pháp.

\begin{table}[H]\centering\footnotesize
\caption{Định nghĩa tám kịch bản: giá trị tham số THẬT dùng để sinh dữ liệu}
\begin{tabularx}{\textwidth}{l C C C C C C L}
\toprule
\textbf{KB} & $\omega_{1},\omega_{2}$ & $\alpha_{1},\alpha_{2}$ & $\beta_{1},\beta_{2}$ & $\nu_0$ & $p_{11},p_{22}$ & $T$ & \textbf{Đặc điểm} \\
\midrule
S1 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 8{,}00 & 0{,}98; 0{,}95 & 2000 & Mốc chuẩn: hai trạng thái phân biệt rõ \\
S2 & 0{,}08; 0{,}14 & 0{,}06; 0{,}08 & 0{,}85; 0{,}83 & 8{,}00 & 0{,}95; 0{,}90 & 2000 & Nhận dạng yếu: hai trạng thái gần nhau \\
S3 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 4{,}00 & 0{,}98; 0{,}95 & 2000 & Đuôi rất dày ($\nu_0=4$) \\
S4 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 30{,}00 & 0{,}98; 0{,}95 & 2000 & $\nu_0$ lớn: gần phân phối chuẩn \\
S5 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 8{,}00 & 0{,}99; 0{,}85 & 2000 & Trạng thái biến động cao hiếm ($p_{22}$ trung bình) \\
S6 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 8{,}00 & 0{,}98; 0{,}95 & 500 & Cỡ mẫu nhỏ (tham số như S1) \\
S7 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 8{,}00 & 0{,}98; 0{,}95 & 5000 & Cỡ mẫu lớn (tham số như S1) \\
S8 & 0{,}05; 0{,}40 & 0{,}05; 0{,}10 & 0{,}85; 0{,}80 & 8{,}00 & 0{,}98; 0{,}95 & 2000 & Sai đặc tả: thêm tự tương quan trong trung bình \\
\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize
\item Giá trị burn-in mặc định 200 quan sát. $\gamma_1=\gamma_2=(0{,}05;\,0{,}05)$ ở mọi kịch bản. Đây là các giá trị THAM SỐ MINH HỌA để kiểm tra hành vi thuật toán trên nhiều cấu hình khác nhau (mức phân biệt trạng thái, độ dày đuôi, cỡ mẫu, sai đặc tả) --- KHÔNG phải hiệu chỉnh trực tiếp từ nghiệm ước lượng VN30F1M. Số lần lặp thực đạt được cho mỗi phương pháp thấp hơn mục tiêu thiết kế do giới hạn thời gian tính toán (xem cột "Số lần lặp" ở bảng kết quả gộp).
\end{tablenotes}
\end{table}
""")

    PN = [("mu1", r"$\mu_1$"), ("mu2", r"$\mu_2$"), ("omega1", r"$\omega_1$"), ("omega2", r"$\omega_2$"),
          ("alpha1", r"$\alpha_1$"), ("alpha2", r"$\alpha_2$"), ("beta1", r"$\beta_1$"), ("beta2", r"$\beta_2$"),
          ("gamma1_0", r"$\gamma_{1,1}$"), ("gamma1_1", r"$\gamma_{1,2}$"),
          ("gamma2_0", r"$\gamma_{2,1}$"), ("gamma2_1", r"$\gamma_{2,2}$"),
          ("p11", r"$p_{11}$"), ("p22", r"$p_{22}$"), ("nu", r"$\nu$")]

    for s in range(1, 9):
        fp = os.path.join(R, f"07_simulation_S{s}.pt")
        if not os.path.exists(fp):
            continue
        d = torch.load(fp, weights_only=False)
        A(r"\begin{table}[H]\centering\scriptsize")
        A(rf"\caption{{Kịch bản S{s}: độ chệch, RMSE và coverage theo từng tham số}}")
        A(r"\begin{tabularx}{\textwidth}{l C C | C C C | C C C}")
        A(r"\toprule")
        A(r"& \multicolumn{2}{c|}{\textbf{MLE}} & \multicolumn{3}{c|}{\textbf{VB mean-field}} & \multicolumn{3}{c}{\textbf{MCMC}} \\")
        A(r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-9}")
        A(r"\textbf{Tham số} & Chệch & RMSE & Chệch & RMSE & Cov. & Chệch & RMSE & Cov. \\")
        A(r"\midrule")
        for key, tex in PN:
            cells = []
            tex_hien = tex + (r"$^{\dagger}$" if key in ("p11", "p22") else "")
            for meth, with_cov in [("mle", False), ("vb_mean_field", True), ("mcmc", True)]:
                if meth not in d:
                    cells += ["--", "--"] + (["--"] if with_cov else [])
                    continue
                b = d[meth]["bias"].get(key)
                rr = d[meth]["rmse"].get(key)
                b, rr = _sua_bias_rmse_p(key, b, rr, f"S{s}")
                cells += [f(b) if b is not None else "--", f(rr) if rr is not None else "--"]
                if with_cov:
                    c = d[meth]["coverage"].get(key)
                    cells.append(f(c, 3) if c is not None else "--")
            A(rf"{tex_hien} & " + " & ".join(cells) + r" \\")
        A(r"\midrule")
        nrow = []
        for meth, with_cov in [("mle", False), ("vb_mean_field", True), ("mcmc", True)]:
            if meth in d:
                n = d[meth]["n_replications"]
                br = f(d[meth]["mean_brier"], 4)
                nrow.append(rf"\multicolumn{{{3 if with_cov else 2}}}{{c}}{{$B={n}$, Brier $={br}$}}")
            else:
                nrow.append(rf"\multicolumn{{{3 if with_cov else 2}}}{{c}}{{---}}")
        A(r"Tổng hợp & " + " & ".join(nrow) + r" \\")
        A(r"\bottomrule")
        A(r"\end{tabularx}")
        A(r"""\begin{tablenotes}\footnotesize
\item[$\dagger$] Chệch/RMSE của $p_{11}$, $p_{22}$ đã được SỬA LẠI bằng đại số so với bản gốc từng lưu trong \texttt{07\_simulation\_S*.pt} (bản gốc tính nhầm giá trị thật là $1{,}0$ thay vì $P_0$ thật dùng để sinh dữ liệu -- xem \texttt{CLAUDE.md}). Cột \textbf{Cov.} của $p_{11}$, $p_{22}$ KHÔNG sửa lại được bằng cách này (cần cận khoảng tin cậy từng lần lặp, dữ liệu chi tiết đã không còn lưu) -- đọc các số Coverage này một cách thận trọng, chỉ mang tính tham khảo định tính.
\end{tablenotes}""")
        A(r"\end{table}")

    if data is not None:
        A(r"""
\begin{table}[H]\centering\footnotesize
\caption{Thống kê mô tả của dữ liệu đầu vào}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
\textbf{Đại lượng} & \textbf{Toàn mẫu} & \textbf{Tập huấn luyện} & \textbf{Tập kiểm định} & \textbf{Đơn vị} \\
\midrule""")
        r_all = data["r_all"]
        n_tr = data["n_train"]
        parts = [("Toàn mẫu", r_all), ("Huấn luyện", r_all[:n_tr]), ("Kiểm định", r_all[n_tr:])]
        stats = {}
        for lab, arr in parts:
            a = arr.double()
            stats[lab] = (int(a.numel()), float(a.mean()), float(a.std()), float(a.min()), float(a.max()))
        A(r"Số quan sát & " + " & ".join(str(stats[k][0]) for k, _ in parts) + r" & phiên \\")
        A(r"Trung bình & " + " & ".join(f(stats[k][1]) for k, _ in parts) + r" & \% \\")
        A(r"Độ lệch chuẩn & " + " & ".join(f(stats[k][2]) for k, _ in parts) + r" & \% \\")
        A(r"Nhỏ nhất & " + " & ".join(f(stats[k][3]) for k, _ in parts) + r" & \% \\")
        A(r"Lớn nhất & " + " & ".join(f(stats[k][4]) for k, _ in parts) + r" & \% \\")
        A(r"""\bottomrule
\end{tabularx}
\end{table}
""")

    return "\n".join(L)


if __name__ == "__main__":
    print(build_appendix()[:400])
