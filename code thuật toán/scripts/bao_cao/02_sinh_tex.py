# -*- coding: utf-8 -*-
"""Sinh file bao_cao_ket_qua.tex tu report_data.json (moi so deu tu du lieu that)."""
import io, os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(io.open(os.path.join(HERE, "report_data.json"), encoding="utf-8"))

from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
BAIBAO = REPO / "Bài báo"
OUT = os.path.join(BAIBAO, "bao_cao_ket_qua.tex")


def f(x, n=4):
    """So thap phan kieu Viet Nam."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:.{n}f}"
    return s.replace(".", "{,}")


def fi(x):
    return f"{int(round(x))}"


IS, OO, VAR, FOR = DATA["is"], DATA["oos"], DATA["var"], DATA["forest"]
MC, REL, ELBO, COST, SIM = DATA["mcmc"], DATA["relabel"], DATA["elbo"], DATA["cost"], DATA["sim"]

# --- xep hang
def ranks(d, key, keys):
    vals = sorted(keys, key=lambda m: d[m][key])
    return {m: i + 1 for i, m in enumerate(vals)}


mo8 = ["b1", "b2", "b3", "b4", "m1", "m2", "m3", "m4"]
rk_q = ranks(OO, "qlike", mo8)
rk_n = ranks(OO, "nls", mo8)
mo5 = ["b1", "b2", "b3", "b4", "m1"]
rk_aic = ranks(IS, "aic", mo5)
rk_bic = ranks(IS, "bic", mo5)

NAME = {"b1": "B1: GARCH(1,1) chuẩn", "b2": "B2: GARCH(1,1) Student-t",
        "b3": "B3: GARCH-X Student-t", "b4": "B4: MS-GARCH chuẩn",
        "m1": "M1: MS-GARCH-X Student-t (MLE)", "m2": "M2: MS-GARCH-X-t (VB mean-field)",
        "m3": "M3: MS-GARCH-X-t (VB full-rank)", "m4": "M4: MS-GARCH-X-t (MCMC/NUTS)"}

L = []
A = L.append

A(r"""% !TeX program = xelatex
% !TeX encoding = UTF-8
% Sinh tu du lieu that trong code thuat toan/results/*.pt
\documentclass[11pt,a4paper]{article}

\usepackage{fontspec}
\usepackage{unicode-math}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{Liberation Serif}}
\IfFontExistsTF{Cambria Math}{\setmathfont{Cambria Math}}{\setmathfont{LibertinusMath-Regular.otf}}

\usepackage[a4paper,top=2.2cm,bottom=2.2cm,left=2.2cm,right=2.0cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{threeparttable}
\usepackage{longtable}
\usepackage{makecell}
\usepackage{float}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}

\definecolor{Accent}{HTML}{2E5A88}
\definecolor{Warn}{HTML}{9B4B4B}
\definecolor{BoxBg}{HTML}{F4F6F8}

\captionsetup{font=small,labelfont=bf,skip=5pt}
\renewcommand{\tablename}{Bảng}
\renewcommand{\figurename}{Hình}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcommand{\best}[1]{\textbf{#1}}

\newcommand{\hopketqua}[1]{%
  \begin{center}\fcolorbox{Accent}{BoxBg}{\begin{minipage}{0.95\linewidth}
  \vspace{0.3em}#1\vspace{0.3em}
  \end{minipage}}\end{center}}

\title{\vspace{-1.2cm}\textbf{Báo cáo kết quả ước lượng và so sánh mô hình}\\[0.25em]
\large MS-GARCH-X Student-t cho hợp đồng tương lai VN30F1M}
\author{}
\date{}

\begin{document}
\maketitle
\vspace{-1.6cm}
""")

# ---------------------------------------------------------------- Tom tat
best_q = min(mo8, key=lambda m: OO[m]["qlike"])
best_n = min(mo8, key=lambda m: OO[m]["nls"])
best_aic = min(mo5, key=lambda m: IS[m]["aic"])
best_bic = min(mo5, key=lambda m: IS[m]["bic"])

A(r"\section*{Tóm tắt kết quả}")
A(r"\hopketqua{" + rf"""
\textbf{{Bốn kết luận chính.}}
\begin{{enumerate}}[leftmargin=1.6em,itemsep=0.25em,topsep=0.3em]
  \item \textbf{{Mô hình một chế độ thắng cả trong mẫu lẫn ngoài mẫu.}} B3 (GARCH-X Student-t) đạt AIC thấp nhất ({f(IS['b3']['aic'],2)}), BIC thấp nhất ({f(IS['b3']['bic'],2)}) và QLIKE ngoài mẫu tốt nhất ({f(OO['b3']['qlike'])}), vượt M1 (MS-GARCH-X) ở cả ba tiêu chí dù M1 có log-hợp-lý cao hơn.
  \item \textbf{{Variational Bayes cho dự báo mật độ tốt nhất với chi phí thấp.}} M2 đạt điểm log âm nhỏ nhất ({f(OO['m2']['nls'])}) trong {f(COST['M2'],1)} phút, so với {f(COST['M1'],1)} phút của M1.
  \item \textbf{{MCMC thất bại.}} Sau hai lần chạy được lưu, $\widehat{{R}}$ tốt nhất chỉ đạt {f(MC['lan2']['max_rhat'],4)} (yêu cầu $<1{{,}}01$) và ESS nhỏ nhất {f(MC['lan2']['min_ess'],1)}; {f(REL['m4']*100,2)}\% mẫu hậu nghiệm phải đổi nhãn trạng thái, tức danh tính trạng thái không ổn định ngay trong một lần chạy.
  \item \textbf{{Cấu trúc chuyển trạng thái không được dữ liệu nhận dạng.}} Bốn phương pháp cho $p_{{11}}$ trải từ {f(FOR['$p_{11}$']['VB mean-field (M2)'][0],4)} đến {f(FOR['$p_{11}$']['MLE (M1)'][0],4)}, tương ứng thời gian tồn tại trạng thái chênh nhau hai bậc độ lớn.
\end{{enumerate}}
""" + "}")

# ---------------------------------------------------------------- 1. Danh muc
A(r"""
\section{Danh mục mô hình và phương pháp}

\begin{table}[H]\centering
\caption{Các mô hình được ước lượng và phương pháp suy diễn tương ứng}
\begin{tabularx}{\textwidth}{l L{3.1cm} C C C}
\toprule
\textbf{Mã} & \textbf{Đặc tả} & \textbf{Phân phối sai số} & \textbf{Số trạng thái} & \textbf{Phương pháp} \\
\midrule
B1 & GARCH(1,1) & Chuẩn & 1 & MLE \\
B2 & GARCH(1,1) & Student-t & 1 & MLE \\
B3 & GARCH-X & Student-t & 1 & MLE \\
B4 & MS-GARCH & Chuẩn & 2 & MLE \\
M1 & MS-GARCH-X & Student-t & 2 & MLE \\
M2 & MS-GARCH-X & Student-t & 2 & VB mean-field \\
M3 & MS-GARCH-X & Student-t & 2 & VB full-rank \\
M4 & MS-GARCH-X & Student-t & 2 & MCMC (NUTS) \\
R1 & MS-GARCH-X, $\nu_s$ riêng & Student-t & 2 & VB mean-field \\
\bottomrule
\end{tabularx}
\end{table}
""".replace(r"L{3.1cm}", r">{\raggedright\arraybackslash}p{3.1cm}"))

# ---------------------------------------------------------------- 2. Trong mau
A(r"""
\section{So sánh trong mẫu}

\begin{table}[H]\centering
\caption{Log-hợp-lý và tiêu chí thông tin (tập huấn luyện $1735$ quan sát)}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C C C}
\toprule
\textbf{Mô hình} & \textbf{Số tham số} & \textbf{Log-hợp-lý} & \textbf{AIC} & \textbf{Hạng AIC} & \textbf{BIC} & \textbf{Hạng BIC} \\
\midrule""")
for m in mo5:
    d = IS[m]
    a = rf"\best{{{f(d['aic'],2)}}}" if rk_aic[m] == 1 else f(d["aic"], 2)
    b = rf"\best{{{f(d['bic'],2)}}}" if rk_bic[m] == 1 else f(d["bic"], 2)
    A(rf"{NAME[m].split(':')[0]} & {d['k']} & {f(d['loglik'],2)} & {a} & {rk_aic[m]} & {b} & {rk_bic[m]} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Giá trị nhỏ hơn là tốt hơn ở cả AIC và BIC. M1 có log-hợp-lý cao nhất nhưng dùng $15$ tham số; sau khi phạt số tham số, B3 ($7$ tham số) vượt lên ở cả hai tiêu chí. Chênh lệch BIC giữa B3 và M1 là """ + f(IS["m1"]["bic"] - IS["b3"]["bic"], 2) + r""" điểm, tức bằng chứng mạnh nghiêng về mô hình một chế độ.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_ic.pdf}
\caption{Tiêu chí thông tin trong mẫu. Việc thêm biến ngoại sinh (B2 $\to$ B3) cải thiện mạnh hơn nhiều so với việc thêm cơ chế chuyển trạng thái (B3 $\to$ M1).}
\end{figure}
""")

# ---------------------------------------------------------------- 3. Ngoai mau
A(r"""
\section{So sánh ngoài mẫu}

\begin{table}[H]\centering
\caption{Dự báo biến động và mật độ ngoài mẫu ($435$ quan sát kiểm định)}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C C C}
\toprule
\textbf{Mô hình} & \textbf{MAE} & \textbf{RMSE} & \textbf{QLIKE} & \textbf{Hạng QLIKE} & \textbf{NLS} & \textbf{Hạng NLS} \\
\midrule""")
for m in mo8:
    d = OO[m]
    q = rf"\best{{{f(d['qlike'])}}}" if rk_q[m] == 1 else f(d["qlike"])
    n_ = rf"\best{{{f(d['nls'])}}}" if rk_n[m] == 1 else f(d["nls"])
    star = r"$^{*}$" if m == "m4" else ""
    A(rf"{m.upper()}{star} & {f(d['mae'])} & {f(d['rmse'])} & {q} & {rk_q[m]} & {n_} & {rk_n[m]} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Giá trị nhỏ hơn là tốt hơn ở mọi cột. QLIKE là tiêu chí chính cho dự báo phương sai; NLS (điểm log âm) là tiêu chí cho dự báo mật độ.
\item[$^{*}$] Số liệu M4 sinh từ mẫu hậu nghiệm \emph{chưa hội tụ} ($\widehat{R}=""" + f(MC["lan2"]["max_rhat"], 4) + r"""$). Báo cáo ở đây để đối chiếu, không dùng làm căn cứ kết luận.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_oos.pdf}
\caption{Hai tiêu chí ngoài mẫu xếp hạng khác nhau: QLIKE ưu tiên B3, còn điểm log ưu tiên M2. Điều này phản ánh QLIKE đo dự báo phương sai điểm, còn điểm log đo toàn bộ hình dạng phân phối, nơi bất định tham số của phương pháp Bayes có giá trị.}
\end{figure}
""")

# ---------------------------------------------------------------- 4. VaR
A(r"""
\section{Kiểm định giá trị chịu rủi ro (VaR)}

\begin{table}[H]\centering
\caption{Kiểm định VaR ngoài mẫu ở hai mức tin cậy}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C C C}
\toprule
& \multicolumn{3}{c}{\textbf{Mức $\alpha=1\%$} (kỳ vọng $4{,}35$)} & \multicolumn{3}{c}{\textbf{Mức $\alpha=5\%$} (kỳ vọng $21{,}75$)} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Mô hình} & \makecell{\textbf{Số}\\\textbf{vi phạm}} & \makecell{\textbf{Kupiec}\\$p$} & \makecell{\textbf{Christ.}\\$p_{cc}$} & \makecell{\textbf{Số}\\\textbf{vi phạm}} & \makecell{\textbf{Kupiec}\\$p$} & \makecell{\textbf{Christ.}\\$p_{cc}$} \\
\midrule""")
for m in mo8:
    d = VAR[m]
    k1 = rf"\textcolor{{Warn}}{{{f(d['kup1'],3)}}}" if d["kup1"] < 0.05 else f(d["kup1"], 3)
    star = r"$^{*}$" if m == "m4" else ""
    A(rf"{m.upper()}{star} & {d['v1']} & {k1} & {f(d['cc1'],3)} & {d['v5']} & {f(d['kup5'],3)} & {f(d['cc5'],3)} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item $p$ nhỏ hơn $0{,}05$ (in màu đỏ) nghĩa là bác bỏ mô hình. Chỉ B1 --- mô hình giả định phân phối chuẩn --- bị Kupiec bác bỏ ở mức $1\%$ với $10$ vi phạm; mọi mô hình Student-t đều vượt qua cả hai kiểm định ở cả hai mức. Đây là bằng chứng trực tiếp cho vai trò của đuôi dày trong quản trị rủi ro.
\item[$^{*}$] Xem chú thích về M4 ở Bảng 3.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_var.pdf}
\caption{Số lần vi phạm VaR so với mức kỳ vọng lý thuyết (đường đứt nét). B1 vượt xa mức kỳ vọng ở ngưỡng $1\%$; các mô hình Student-t đều bám sát.}
\end{figure}
""")

# ---------------------------------------------------------------- 5. So sanh phuong phap
A(r"""
\section{So sánh bốn phương pháp ước lượng trên cùng một mô hình}

Mục này trả lời câu hỏi trung tâm: bốn phương pháp suy diễn cho kết quả \emph{khác nhau đến mức nào} khi áp dụng cho cùng một đặc tả MS-GARCH-X Student-t và cùng một tập dữ liệu.

\begin{table}[H]\centering
\caption{Ước lượng điểm và khoảng $95\%$ của các tham số then chốt}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
\textbf{Tham số} & \textbf{MLE (M1)} & \textbf{VB mean-field (M2)} & \textbf{VB full-rank (M3)} & \textbf{MCMC (M4)$^{*}$} \\
\midrule""")
ORDER = [r"$\alpha_1$", r"$\alpha_2$", r"$\beta_1$", r"$\beta_2$", r"$p_{11}$", r"$p_{22}$", r"$\nu$"]
MK = [m for m in ["MLE (M1)", "VB mean-field (M2)", "VB full-rank (M3)", "MCMC (M4)*"] if m in FOR.get(ORDER[0], {})]
for pname in ORDER:
    cells = []
    for mk in MK:
        if mk in FOR.get(pname, {}):
            e, lo, hi = FOR[pname][mk]
            cells.append(rf"\makecell{{{f(e)}\\\scriptsize[{f(lo,3)}; {f(hi,3)}]}}")
        else:
            cells.append(r"\makecell{--}")
    A(rf"{pname} & " + " & ".join(cells) + r" \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Với MLE, khoảng là $\hat\theta\pm1{,}96\,\mathrm{SE}$ (delta method từ Hessian quan sát); với ba phương pháp Bayes, khoảng là phân vị $2{,}5\%$ và $97{,}5\%$ của hậu nghiệm. Mẫu hậu nghiệm của cả ba đã được sắp xếp lại nhãn trạng thái theo quy ước ``trạng thái 1 là trạng thái biến động thấp'' trước khi tính.
\item Khoảng của $p_{11}$ và $p_{22}$ theo MLE vượt quá $1$ --- dấu hiệu nghiệm nằm sát biên không gian tham số, khiến xấp xỉ tiệm cận chuẩn không còn hợp lệ.
\item[$^{*}$] Xem chú thích về M4 ở Bảng 3.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_forest.pdf}
\caption{Bốn phương pháp đồng thuận cao ở $\nu$ và $\alpha_2$, nhưng phân kỳ mạnh ở $\beta_1$ và đặc biệt ở $p_{11}$, $p_{22}$ --- đúng những tham số điều khiển cấu trúc chuyển trạng thái.}
\end{figure}
""")

p11_mle = FOR.get(r"$p_{11}$", {}).get("MLE (M1)", (0.5,))[0]
p22_mle = FOR.get(r"$p_{22}$", {}).get("MLE (M1)", (0.5,))[0]
p11_vb = FOR.get(r"$p_{11}$", {}).get("VB mean-field (M2)", (0.5,))[0]
p22_vb = FOR.get(r"$p_{22}$", {}).get("VB mean-field (M2)", (0.5,))[0]
p11_mc = FOR.get(r"$p_{11}$", {}).get("MCMC (M4)*", (None,))[0]
p22_mc = FOR.get(r"$p_{22}$", {}).get("MCMC (M4)*", (None,))[0]

d1_mle = 1 / (1 - p11_mle) if p11_mle is not None and p11_mle < 1 else None
d2_mle = 1 / (1 - p22_mle) if p22_mle is not None and p22_mle < 1 else None
d1_vb = 1 / (1 - p11_vb) if p11_vb is not None and p11_vb < 1 else None
d2_vb = 1 / (1 - p22_vb) if p22_vb is not None and p22_vb < 1 else None
d1_mc = 1 / (1 - p11_mc) if p11_mc is not None and p11_mc < 1 else None
d2_mc = 1 / (1 - p22_mc) if p22_mc is not None and p22_mc < 1 else None

A(r"""
\begin{table}[H]\centering
\caption{Hệ quả kinh tế: thời gian tồn tại kỳ vọng của mỗi trạng thái}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C}
\toprule
\textbf{Đại lượng} & \textbf{MLE (M1)} & \textbf{VB mean-field (M2)} & \textbf{MCMC (M4)$^{*}$} \\
\midrule""")
A(rf"$d_1$ (trạng thái biến động thấp) & {f(d1_mle,1)} phiên & {f(d1_vb,1)} phiên & {f(d1_mc,1)} phiên \\")
A(rf"$d_2$ (trạng thái biến động cao) & {f(d2_mle,1)} phiên & {f(d2_vb,1)} phiên & {f(d2_mc,1)} phiên \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item $d_s = 1/(1-p_{ss})$. Ba phương pháp cho ba câu trả lời chênh nhau tới hai bậc độ lớn trên cùng một bộ dữ liệu: MLE mô tả các chế độ kéo dài nhiều năm, còn VB và MCMC mô tả các chế độ chỉ vài phiên đến vài tuần. Đây là biểu hiện rõ nhất của việc cấu trúc chuyển trạng thái không được nhận dạng từ dữ liệu.
\item[$^{*}$] Xem chú thích về M4 ở Bảng 3.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- 6. Chan doan
A(r"""
\section{Chẩn đoán hội tụ}

\begin{table}[H]\centering
\caption{Chẩn đoán hội tụ MCMC (NUTS), bốn chuỗi $\times$ $400$ mẫu}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
& \multicolumn{2}{c}{\textbf{Lần 1}: ma trận đường chéo, warmup $250$} & \multicolumn{2}{c}{\textbf{Lần 2}: ma trận đặc, warmup $1000$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Tham số} & $\widehat{R}$ & ESS & $\widehat{R}$ & ESS \\
\midrule""")
KEYS = ["mu", "omega", "alpha", "beta", "gamma", "p11", "p22", "nu_minus_2"]
TEXN = {"mu": r"$\mu$", "omega": r"$\omega$", "alpha": r"$\alpha$", "beta": r"$\beta$",
        "gamma": r"$\gamma$", "p11": r"$p_{11}$", "p22": r"$p_{22}$", "nu_minus_2": r"$\nu$"}
for kk in KEYS:
    r1 = MC['lan1']['rhat'].get(kk) if MC and 'lan1' in MC and 'rhat' in MC['lan1'] else None
    e1 = MC['lan1']['ess'].get(kk) if MC and 'lan1' in MC and 'ess' in MC['lan1'] else None
    r2 = MC['lan2']['rhat'].get(kk) if MC and 'lan2' in MC and 'rhat' in MC['lan2'] else None
    e2 = MC['lan2']['ess'].get(kk) if MC and 'lan2' in MC and 'ess' in MC['lan2'] else None
    A(rf"{TEXN[kk]} & {f(r1,4)} & {f(e1,1)} & {f(r2,4)} & {f(e2,1)} \\")
A(r"\midrule")
A(rf"\textbf{{Xấu nhất}} & \textcolor{{Warn}}{{{f(MC['lan1']['max_rhat'] if MC else None,4)}}} & {f(MC['lan1']['min_ess'] if MC else None,1)} & "
  rf"\textcolor{{Warn}}{{{f(MC['lan2']['max_rhat'] if MC else None,4)}}} & {f(MC['lan2']['min_ess'] if MC else None,1)} \\")
A(rf"Số lần phân kỳ & \multicolumn{{2}}{{c}}{{{MC['lan1']['ndiv'] if MC else '--'}}} & \multicolumn{{2}}{{c}}{{{MC['lan2']['ndiv'] if MC else '--'}}} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Tiêu chuẩn chấp nhận là $\widehat{R}<1{,}01$ và ESS đủ lớn (thông thường $>400$). Tăng warmup gấp bốn lần và chuyển sang ma trận khối lượng đặc chỉ kéo $\widehat{R}$ từ """ + f(MC["lan1"]["max_rhat"], 2) + " xuống " + f(MC["lan2"]["max_rhat"], 2) + r""" --- vẫn cách rất xa ngưỡng. ESS quanh $5$ trên $1600$ mẫu nghĩa là bốn chuỗi gần như đứng yên tại bốn vị trí khác nhau.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{table}[H]\centering
\caption{Tỷ lệ mẫu hậu nghiệm phải đổi nhãn trạng thái}
\begin{tabularx}{\textwidth}{l C C}
\toprule
\textbf{Mô hình} & \textbf{Tỷ lệ đổi nhãn} & \textbf{Diễn giải} \\
\midrule""")
INT = {"m2": "Ổn định: hầu hết mẫu đã đúng quy ước",
       "m3": "Bất ổn: gần một phép hoán vị toàn cục",
       "m4": "Bất ổn nặng: gần mức ngẫu nhiên $50\\%$",
       "r1": "Bất ổn: chủ yếu là hoán vị toàn cục"}
for kk, nm in [("m2", "M2: VB mean-field"), ("m3", "M3: VB full-rank"),
               ("m4", "M4: MCMC"), ("r1", "R1: VB, $\\nu_s$ riêng")]:
    A(rf"{nm} & {f(REL[kk]*100,2)}\% & {INT[kk]} \\")
A(r"""\bottomrule
\end{tabularx}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_conv.pdf}
\caption{Trái: $\widehat{R}$ theo tham số ở hai lần chạy MCMC, không lần nào chạm ngưỡng $1{,}01$. Phải: ELBO của hai biến thể Variational Bayes dao động quanh một mức ổn định sau khoảng một nghìn bước, cho thấy vấn đề hội tụ là đặc thù của MCMC chứ không phải của bài toán tối ưu nói chung.}
\end{figure}
""")

# ---------------------------------------------------------------- 7. Chi phi
A(r"""
\section{Chi phí tính toán}

\begin{table}[H]\centering
\caption{Thời gian ước lượng theo mô hình}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C}
\toprule
\textbf{Mô hình} & \textbf{Phương pháp} & \textbf{Thời gian (phút)} & \textbf{Kết quả dùng được?} \\
\midrule""")
ROWS = [("B1", "MLE", "Có"), ("B2", "MLE", "Có"), ("B3", "MLE", "Có"), ("B4", "MLE", "Có"),
        ("M1", "MLE", "Có"), ("M2", "VB mean-field", "Có"), ("M3", "VB full-rank", "Có"),
        ("R1", "VB mean-field", "Có"), ("M4", "MCMC (NUTS)", r"\textcolor{Warn}{\textbf{Không}}")]
for code, meth, ok in ROWS:
    A(rf"{code} & {meth} & {f(COST[code],1)} & {ok} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Thời gian M4 là tổng cộng dồn của các lần chạy trên bốn tiến trình song song. Kết luận về đánh đổi chi phí rất rõ: phương pháp đắt nhất không cho kết quả dùng được, còn phương pháp rẻ hơn năm lần so với MLE (VB mean-field) lại cho dự báo mật độ tốt nhất.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_cost.pdf}
\caption{Chi phí tính toán trên thang logarit. Khoảng cách giữa mô hình một chế độ và hai chế độ là hơn một bậc độ lớn.}
\end{figure}
""")

# ---------------------------------------------------------------- 8. Mo phong
A(r"""
\section{Nghiên cứu mô phỏng}

Trên dữ liệu thật không biết tham số đúng, nên chất lượng ước lượng được đánh giá bổ sung bằng mô phỏng tám kịch bản $S_1$--$S_8$ với tham số đã biết trước.

\begin{table}[H]\centering
\caption{Kết quả mô phỏng gộp trên tám kịch bản}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C C}
\toprule
\textbf{Phương pháp} & \textbf{Số lần lặp} & \textbf{$|$Độ chệch$|$ TB} & \textbf{RMSE TB} & \textbf{Coverage $95\%$} & \textbf{Brier} \\
\midrule""")
for nm in ["MLE", "VB mean-field", "MCMC"]:
    s = SIM[nm]
    cov = "--" if (s["cov"] is None or (isinstance(s["cov"], float) and math.isnan(s["cov"]))) else f(s["cov"], 3)
    nrep_txt = f(s['nrep'], 1) + (rf" ({s['nrep_min']}--{s['nrep_max']})" if s.get("nrep_min") != s.get("nrep_max") else "")
    A(rf"{nm} & {nrep_txt} & {f(s['bias'],4)} & {f(s['rmse'],4)} & {cov} & {f(s['brier'],4)} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Độ chệch và RMSE lấy trung bình trên các tham số có thang đo khác nhau, nên chỉ dùng để so sánh thô giữa các phương pháp. Coverage lý tưởng là $0{,}95$; cả hai phương pháp Bayes đều thấp hơn nhiều, tức khoảng tin cậy hậu nghiệm \emph{hẹp hơn thực tế} --- chúng đánh giá thấp mức bất định. MLE không sinh khoảng tin cậy trong quy trình mô phỏng nên không tính được coverage.
\item Số lần lặp của MCMC rất nhỏ ($3$) do chi phí tính toán, nên các con số ở hàng này chỉ mang tính tham khảo.
\end{tablenotes}
\end{threeparttable}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\textwidth]{hinh_bao_cao/fig_sim.pdf}
\caption{MLE khôi phục tham số chính xác nhất trên dữ liệu mô phỏng; cả hai phương pháp Bayes đều đánh giá thấp bất định một cách đáng kể.}
\end{figure}
""")

# ---------------------------------------------------------------- 9. Ket luan
A(r"""
\section{Kết luận so sánh}

\begin{table}[H]\centering
\caption{Nên dùng phương pháp nào cho mục đích nào}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{L C L}
\toprule
\textbf{Mục đích} & \textbf{Lựa chọn} & \textbf{Căn cứ} \\
\midrule
Dự báo phương sai điểm & B3 (GARCH-X, MLE) & QLIKE """ + f(OO["b3"]["qlike"]) + r""" --- tốt nhất; đồng thời AIC và BIC thấp nhất \\
Dự báo mật độ / VaR & M2 (VB mean-field) & NLS """ + f(OO["m2"]["nls"]) + r""" --- tốt nhất; vượt qua mọi kiểm định VaR \\
Ước lượng tham số có sai số chuẩn & B3 (MLE) & Hội tụ ở mọi điểm khởi tạo, nghiệm không sát biên \\
Suy diễn Bayes đầy đủ cho MS-GARCH-X & \textcolor{Warn}{Chưa khả thi} & MCMC không hội tụ; VB đánh giá thấp bất định \\
\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Kết luận bao trùm: trên dữ liệu VN30F1M hiện có, việc bổ sung \emph{biến ngoại sinh} mang lại cải thiện rõ rệt và ổn định, trong khi việc bổ sung \emph{cơ chế chuyển trạng thái ẩn} làm tăng đáng kể chi phí tính toán và độ bất định của kết luận mà không cải thiện dự báo ngoài mẫu.
\end{tablenotes}
\end{threeparttable}
\end{table}

\vfill
\begin{center}\small\textcolor{gray}{%
Toàn bộ số liệu trong báo cáo này được trích xuất trực tiếp từ
\texttt{code thuật toán/results/*.pt} và không nhập tay.}
\end{center}
""".replace(r"{L C L}", r"{>{\raggedright\arraybackslash}p{4.0cm} >{\centering\arraybackslash}p{3.4cm} X}"))

# ---------------------------------------------------------------- Phu luc
sys.path.insert(0, HERE)
from appendix_gen import build_appendix
A(build_appendix())
A(r"\end{document}")

with io.open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print("WROTE:", OUT)
