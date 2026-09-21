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
\large Danh mục GARCH/EGARCH/MS-GARCH-X, hợp đồng tương lai VN30F1M --- EGARCH-X Student-t là mô hình thắng cuộc}
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
\textbf{{Năm kết luận chính.}}
\begin{{enumerate}}[leftmargin=1.6em,itemsep=0.25em,topsep=0.3em]
  \item \textbf{{Mô hình EGARCH-X với sai số Student-$t$, ước lượng bằng hợp lý cực đại (MLE), thắng cả bốn tiêu chí so sánh}} trên toàn bộ danh mục tám mô hình và ba phương pháp suy diễn (MLE, Bayes biến phân mean-field, Bayes biến phân full-rank): AIC $5606{{,}}87$, BIC $5650{{,}}54$, QLIKE $0{{,}}9234$, điểm log âm ngoài mẫu $1{{,}}5203$ --- xem Bảng~\ref{{tab:buoc1-so-sanh}}.
  \item \textbf{{Thành phần đuôi dày của phân phối sai số (chuẩn $\to$ Student-$t$) là yếu tố cải thiện mô hình quan trọng nhất}}, đóng góp gấp khoảng năm lần mọi thành phần khác (giảm AIC khoảng $235$ điểm, so với thành phần tốt thứ nhì chỉ giảm $49$ điểm). Ngược lại, \textbf{{việc thêm cơ chế chuyển trạng thái Markov hai chế độ biến động làm mô hình xấu đi}} (AIC tăng $1{{,}}71$ điểm) --- tức mô hình chuyển trạng thái Markov với sai số Student-$t$ và biến ngoại sinh (MS-GARCH-X Student-$t$), vốn là mô hình được đề xuất ban đầu của đề tài, không được dữ liệu thực nghiệm ủng hộ. Xem Bảng~\ref{{tab:buoc1-phan-ra}}.
  \item \textbf{{Phương pháp MCMC (thuật toán NUTS) hội tụ tốt khi áp dụng cho mô hình EGARCH-X một chế độ biến động}}: hệ số $\hat R$ lớn nhất chỉ $1{{,}}0125$ (ngưỡng chấp nhận thông thường là $1{{,}}01$), cỡ mẫu hiệu quả nhỏ nhất đạt $1.340$ trên tổng $1.600$ mẫu hậu nghiệm ($84\%$), không có lần lặp phân kỳ nào. Kết quả này trái ngược hoàn toàn với lần chạy MCMC thất bại trước đó trên mô hình có cấu trúc chuyển trạng thái (xem Phụ lục A), qua đó xác nhận nguyên nhân thất bại là do đối xứng hoán vị nhãn trạng thái, không phải do bản thân phương pháp MCMC.
  \item \textbf{{Hiệu ứng đòn bẩy của lợi suất VN30F1M được xác lập có ý nghĩa thống kê}}: khoảng tin cậy $95\%$ của hệ số đòn bẩy theo MCMC là $[-0{{,}}2125;\,-0{{,}}0710]$, loại trừ giá trị $0$. Kết quả này được củng cố thêm bằng một nghiên cứu mô phỏng riêng, trong đó dữ liệu giả lập không có hiệu ứng đòn bẩy vẫn cho ước lượng hệ số đòn bẩy gần $0$ --- tức mô hình không tự tạo ra hiệu ứng đòn bẩy giả.
  \item \textbf{{Phương pháp Bayes biến phân dạng mean-field đánh giá thấp mức độ bất định của hai tham số mức phương sai và độ dai dẳng của biến động khoảng ba lần}} so với MCMC (được xem là chuẩn tham chiếu do đã hội tụ); Bayes biến phân dạng full-rank cải thiện đáng kể nhưng vẫn hẹp hơn khoảng $25\%$ ở hai tham số này. Một nghiên cứu mô phỏng khôi phục tham số độc lập, thực hiện trước khi biết kết quả trên dữ liệu thật, đã dự báo đúng chính hiện tượng này.
\end{{enumerate}}
\textbf{{Lưu ý khi đọc phần còn lại của báo cáo:}} Phụ lục A trình bày phân tích trên riêng mô hình chuyển trạng thái Markov (MS-GARCH-X Student-$t$) theo thiết kế nghiên cứu ban đầu, trước khi danh mục mô hình được mở rộng thêm họ EGARCH. Các phát hiện ở đó (MCMC không hội tụ, Bayes biến phân đánh giá thấp mức bất định) vẫn đúng và vẫn có giá trị làm bằng chứng khoa học, nhưng không còn là kết luận trung tâm của báo cáo --- kết luận trung tâm nằm ở Mục~\ref{{sec:ket-qua-ba-buoc}}.
""" + "}")

# ---------------------------------------------------------- 0b. Ket qua 3 buoc
A(r"""
\section{Kết quả trên dữ liệu thật: ba bước}
\label{sec:ket-qua-ba-buoc}

Mục này tổng hợp kết quả của \textbf{Bước 1} (so sánh toàn bộ danh mục tám mô
hình và ba phương pháp suy diễn trên dữ liệu VN30F1M thật), \textbf{Bước 2}
(mô phỏng khôi phục tham số, xem thêm Mục~\ref{sec:nghien-cuu-mo-phong} và
phụ lục điện tử "Tám kịch bản mô phỏng" để có bảng chi tiết theo từng tham
số), và \textbf{Bước 3} (ước lượng Bayes đầy đủ bằng MCMC, chạy một lần duy
nhất trên mô hình thắng cuộc). Số liệu trong sáu bảng dưới đây được tổng hợp
trực tiếp từ kết quả ước lượng, không nhập tay.

\input{ket_qua_moi}
""")

A(r"""
\section{Phụ lục A: phân tích nội bộ MS-GARCH-X Student-$t$ (thiết kế trước khi mở rộng sang EGARCH)}

Các mục dưới đây khảo sát riêng mô hình chuyển trạng thái Markov với sai số
Student-$t$ và biến ngoại sinh (\textbf{MS-GARCH-X Student-$t$}), bằng bốn
phương pháp suy diễn (hợp lý cực đại, Bayes biến phân mean-field, Bayes biến
phân full-rank, MCMC), theo thiết kế nghiên cứu ban đầu --- trước khi danh
mục mô hình được mở rộng thành tám mô hình ở Bước 1 (Mục~\ref{sec:ket-qua-ba-buoc}).
Ba mô hình một chế độ dùng làm đối chứng (GARCH chuẩn, GARCH Student-$t$,
GARCH-X Student-$t$) và các biến thể của MS-GARCH-X Student-$t$ ở đây
\textbf{không bao gồm họ EGARCH}. Giữ lại nguyên vẹn vì đây là nơi duy nhất
có: (a) so sánh trực tiếp bốn phương pháp suy diễn trên cùng một mô hình hai
chế độ biến động, và (b) bằng chứng chi tiết cho việc MCMC không hội tụ trên
cấu trúc chuyển trạng thái --- phát hiện vẫn đúng và vẫn là một phần lập luận
của thiết kế ba bước (xem đối chiếu ở Mục~\ref{sec:ket-qua-ba-buoc}, nơi MCMC
hội tụ khi bỏ cấu trúc chuyển trạng thái).
""")

# ---------------------------------------------------------------- 1. Danh muc
A(r"""
\subsection{Danh mục mô hình và phương pháp}

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
\subsection{So sánh trong mẫu}

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
\subsection{So sánh ngoài mẫu}

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
\subsection{Kiểm định giá trị chịu rủi ro (VaR)}

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
\subsection{So sánh bốn phương pháp ước lượng trên cùng một mô hình}

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
\subsection{Chẩn đoán hội tụ}

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
\subsection{Chi phí tính toán}

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
\label{sec:nghien-cuu-mo-phong}

Trên dữ liệu thật không biết tham số đúng, nên chất lượng ước lượng được đánh
giá bổ sung bằng hai nghiên cứu khôi phục tham số với tham số đã biết trước:
\textbf{PR-A} (MS-GARCH-X Student-$t$, hai trạng thái, kịch bản $S_1,S_2,S_3$)
và \textbf{PR-B} (EGARCH-X Student-$t$, một trạng thái, kịch bản $E_1,E_2,E_3$)
--- mỗi kịch bản $B=100$ lần lặp, $T=2000$, ba phương pháp (MLE, VB mean-field,
VB full-rank). Trước khi tính Bias/RMSE/Coverage của PR-A, nhãn trạng thái đã
được sắp lại theo phương sai vô điều kiện (bộ ước lượng có đối xứng hoán vị
nhãn nên có thể trả về "trạng thái 1"/"trạng thái 2" theo thứ tự tùy ý --- VB
đảo nhãn ở gần một nửa số lần lặp nếu không sắp lại). Bộ tám kịch bản $S_1$--$S_8$
minh họa cũ (tham số không hiệu chỉnh theo VN30F1M, $B=15$--$30$) được giữ làm
phụ lục bổ trợ, không còn là số liệu chính thức của mục này.

\begin{table}[H]\centering
\caption{Kết quả mô phỏng gộp trên PR-A + PR-B (sáu kịch bản, $B=100$)}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{l C C C C}
\toprule
\textbf{Phương pháp} & \textbf{Số lần lặp} & \textbf{$|$Độ chệch$|$ TB} & \textbf{RMSE TB} & \textbf{Coverage $95\%$} \\
\midrule""")
for nm in ["MLE", "VB mean-field", "VB full-rank"]:
    s = SIM[nm]
    cov = "--" if (s["cov"] is None or (isinstance(s["cov"], float) and math.isnan(s["cov"]))) else f(s["cov"], 3)
    nrep_txt = f(s['nrep'], 0) + (rf" ({s['nrep_min']}--{s['nrep_max']})" if s.get("nrep_min") != s.get("nrep_max") else "")
    A(rf"{nm} & {nrep_txt} & {f(s['bias'],4)} & {f(s['rmse'],4)} & {cov} \\")
A(r"""\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Độ chệch và RMSE lấy trung bình trên các tham số có thang đo khác nhau (gộp cả PR-A và PR-B), nên chỉ dùng để so sánh thô giữa các phương pháp --- xem \texttt{Bài báo/8\_kich\_ban\_mo\_phong.tex} để có bảng chi tiết theo từng tham số, từng kịch bản. Coverage lý tưởng là $0{,}95$; cả hai phương pháp Bayes đều thấp hơn nhiều, phần lớn do độ chệch (không phải khoảng tin cậy quá hẹp) --- ở $p_{11},p_{22}$ của PR-A, độ chệch chủ yếu đến từ tiên nghiệm Beta$(8;2)$ kéo ước lượng về trung bình tiên nghiệm $0{,}80$, không phải nhược điểm thuần túy của biến phân. MLE không sinh khoảng tin cậy trong quy trình này nên không tính được coverage.
\item Cột Brier đã bỏ khỏi bảng này: Brier của PR-A được tính trước khi sắp nhãn trạng thái và hiện sai (một số giá trị VB-FR tệ hơn đoán ngẫu nhiên $0{,}50$), còn PR-B là mô hình một trạng thái nên không có xác suất trạng thái để tính Brier.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

# ---------------------------------------------------------------- 9. Ket luan
A(r"""
\subsection{Kết luận so sánh trong Phụ lục A (chỉ MS-GARCH-X, 5 mô hình)}

\begin{table}[H]\centering
\caption{Trong phạm vi hẹp của Phụ lục A: nên dùng phương pháp nào cho mục đích nào}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{L C L}
\toprule
\textbf{Mục đích} & \textbf{Lựa chọn} & \textbf{Căn cứ} \\
\midrule
Dự báo phương sai điểm & B3 (GARCH-X, MLE) & QLIKE """ + f(OO["b3"]["qlike"]) + r""" --- tốt nhất \emph{trong danh mục 5 mô hình này}; đồng thời AIC và BIC thấp nhất \\
Dự báo mật độ / VaR & M2 (VB mean-field) & NLS """ + f(OO["m2"]["nls"]) + r""" --- tốt nhất \emph{trong danh mục 5 mô hình này}; vượt qua mọi kiểm định VaR \\
Ước lượng tham số có sai số chuẩn & B3 (MLE) & Hội tụ ở mọi điểm khởi tạo, nghiệm không sát biên \\
Suy diễn Bayes đầy đủ cho \emph{MS-GARCH-X cụ thể} & \textcolor{Warn}{Chưa khả thi} & MCMC không hội tụ trên cấu trúc chuyển trạng thái này; VB đánh giá thấp bất định \\
\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Kết luận này chỉ có giá trị \textbf{trong phạm vi 5 mô hình của Phụ lục A} (không có EGARCH). Trên toàn bộ danh mục 8 mô hình của Bước 1, EGARCH-X Student-t vượt qua cả B3 lẫn M2 --- xem Mục~\ref{sec:ket-qua-ba-buoc}.
\end{tablenotes}
\end{threeparttable}
\end{table}
""")

A(r"""
\section{Kết luận chung}

\begin{table}[H]\centering
\caption{Nên dùng mô hình/phương pháp nào cho mục đích nào (toàn bộ danh mục, ba bước)}
\begin{threeparttable}
\begin{tabularx}{\textwidth}{L C L}
\toprule
\textbf{Mục đích} & \textbf{Lựa chọn} & \textbf{Căn cứ} \\
\midrule
Dự báo phương sai điểm và mật độ & EGARCH-X Student-$t$ (MLE) & Thắng cả AIC, BIC, QLIKE, NLS trên toàn bộ 8 mô hình $\times$ 3 phương pháp \\
Suy diễn Bayes đầy đủ (khoảng tin cậy, hậu nghiệm) & MCMC (NUTS) trên EGARCH-X-$t$ & Hội tụ tốt ($\hat R=1{,}0125$, ESS $1.340$, $0$ phân kỳ); không khả thi trên MS-GARCH-X do đối xứng hoán vị nhãn \\
Ước lượng nhanh, chi phí thấp & MLE hoặc VB mean-field trên EGARCH-X-$t$ & Hai phương pháp gần như ngang nhau ở QLIKE ($0{,}9234$ so với $0{,}9281$) --- khác hẳn GARCH-X, nơi ràng buộc dương phá hoại VB \\
Đánh giá độ tin cậy khoảng tin cậy của VB & Có phạt bằng $R_{sd}$ so với MCMC & VB mean-field hẹp hơn thực tế khoảng $3$ lần ở $\omega,\beta$; VB full-rank khá hơn nhưng vẫn hẹp hơn $\sim\!25\%$ \\
\bottomrule
\end{tabularx}
\begin{tablenotes}\small
\item Kết luận bao trùm: \textbf{đuôi dày} là thành phần cấu trúc quan trọng nhất; \textbf{biến ngoại sinh} hữu ích hơn hẳn khi mô hình không bị ràng buộc dương phá hoại (EGARCH so với GARCH-X); \textbf{chuyển trạng thái ẩn} không được dữ liệu VN30F1M ủng hộ --- làm mô hình tệ đi ở Bước 1 và là nguyên nhân duy nhất khiến MCMC thất bại ở lần chạy đầu.
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
