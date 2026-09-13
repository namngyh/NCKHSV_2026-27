"""Sinh phan KET QUA MOI cua bai bao TU DONG tu results/*.pt.

VI SAO CO FILE NAY. `report_data.json` duoc duy tri THU CONG va khong tu dong
bo voi `results/`; no cung duoc to chuc quanh bo ma mo hinh CU (b1-b4, m1-m4)
nen khong bieu dien duoc danh muc tam mo hinh lan thiet ke ba buoc. Thay vi
chep tay so lieu moi vao do -- tao ra ban sao thu hai cua cung mot su that --
file nay doc THANG tu cac file ket qua va sinh ra LaTeX.

Chay lai bat cu luc nao; so lieu trong bai se tu khop voi so lieu da tinh.

    python scripts/bao_cao/sinh_ket_qua_moi.py

Ket qua: "Bai bao/ket_qua_moi.tex", dung de \\input vao ban thao.
Chi dung `booktabs` + `tabular` thuan de \\input duoc vao CA HAI ban thao ma
khong phu thuoc newcolumntype rieng cua tung file (xem bay so 7 trong CLAUDE.md:
bang dung cot `Y` chua dinh nghia tung lam file khong bien dich duoc).
"""
from __future__ import annotations

import glob
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT.parent / "Bài báo" / "ket_qua_moi.tex"

PP = ["mle", "vb_mean_field", "vb_full_rank"]
TEN_PP = {"mle": "MLE", "vb_mean_field": "VB-MF", "vb_full_rank": "VB-FR"}

_d: list[str] = []


def p(s: str = "") -> None:
    _d.append(s)


def so(v, n=2, dash="--"):
    return dash if v is None else f"{v:.{n}f}".replace(".", ",")


# ===========================================================================
# Buoc 1 -- so sanh danh muc mo hinh tren du lieu that
# ===========================================================================
def bang_buoc_1() -> dict:
    bang = torch.load(RESULTS / "pipeline" / "bang_so_sanh.pt", weights_only=False)

    p(r"\begin{table}[H]")
    p(r"\centering\footnotesize")
    p(r"\caption{So sánh toàn bộ danh mục mô hình trên dữ liệu VN30F1M, "
      r"ba phương pháp suy diễn}")
    p(r"\label{tab:buoc1-so-sanh}")
    p(r"\begin{tabular}{llrrrrrr}")
    p(r"\toprule")
    p(r"\textbf{Mô hình} & \textbf{PP} & $d$ & log-lik & AIC & BIC & QLIKE & NLS \\")
    p(r"\midrule")
    ten_truoc = None
    for r in bang:
        ten = r["ten"] if r["ten"] != ten_truoc else ""
        ten_truoc = r["ten"]
        p(f"{ten} & {TEN_PP[r['phuong_phap']]} & {r['d']} & "
          f"{so(r['loglik'])} & {so(r['aic'])} & {so(r['bic'])} & "
          f"{so(r['qlike'], 4)} & {so(r['nls'], 4)} \\\\")
    p(r"\bottomrule")
    p(r"\end{tabular}")
    p(r"\vspace{0.3em}")
    p(r"\raggedright\scriptsize")
    p(r"\emph{Ghi chú:} $d$ là số tham số tự do. Log-likelihood, AIC và BIC chỉ "
      r"định nghĩa cho ước lượng điểm nên để trống ở hai phương pháp Bayes biến "
      r"phân. QLIKE và NLS đo ngoài mẫu trên 435 phiên kiểm định; nhỏ hơn là tốt "
      r"hơn. Mô hình chuẩn cài bằng chính khung Student-$t$ với $\nu$ cố định ở "
      r"$10^6$, và $\nu$ đó không tính vào $d$.")
    p(r"\end{table}")
    p()

    tot = {}
    for tc in ("aic", "bic", "qlike", "nls"):
        hop = [r for r in bang if r.get(tc) is not None]
        tot[tc] = min(hop, key=lambda r: r[tc])
    return {"bang": bang, "tot": tot}


def bang_phan_ra(bang) -> None:
    """Tach dong gop cua tung yeu to. Chi so sanh duoc vi danh muc la mot LUOI:
    moi cap ke nhau chi khac DUNG MOT yeu to."""
    aic = {r["code"]: r["aic"] for r in bang if r["phuong_phap"] == "mle"}
    hang = [
        (r"Đuôi dày (chuẩn $\to$ Student-$t$), họ GARCH", "garch_n", "garch_t"),
        (r"Đuôi dày (chuẩn $\to$ Student-$t$), họ EGARCH", "egarch_n", "egarch_t"),
        (r"Biến ngoại sinh ($t \to$ X-$t$), họ GARCH", "garch_t", "garch_x_t"),
        (r"Biến ngoại sinh ($t \to$ X-$t$), họ EGARCH", "egarch_t", "egarch_x_t"),
        (r"Dạng hàm EGARCH, biến thể chuẩn", "garch_n", "egarch_n"),
        (r"Dạng hàm EGARCH, biến thể Student-$t$", "garch_t", "egarch_t"),
        (r"Dạng hàm EGARCH, biến thể X Student-$t$", "garch_x_t", "egarch_x_t"),
        (r"Chuyển trạng thái, biến thể X Student-$t$", "garch_x_t", "ms_garch_x_t"),
    ]
    p(r"\begin{table}[H]")
    p(r"\centering\footnotesize")
    p(r"\caption{Đóng góp riêng của từng thành phần cấu trúc, đo bằng chênh lệch AIC}")
    p(r"\label{tab:buoc1-phan-ra}")
    p(r"\begin{tabular}{lrrr}")
    p(r"\toprule")
    p(r"\textbf{Thành phần được thêm} & AIC trước & AIC sau & $\Delta$AIC \\")
    p(r"\midrule")
    for ten, a, b in hang:
        if a not in aic or b not in aic:
            continue
        d = aic[b] - aic[a]
        p(f"{ten} & {so(aic[a])} & {so(aic[b])} & {so(d)} \\\\")
    p(r"\bottomrule")
    p(r"\end{tabular}")
    p(r"\vspace{0.3em}")
    p(r"\raggedright\scriptsize")
    p(r"\emph{Ghi chú:} $\Delta$AIC âm nghĩa là thành phần được thêm \emph{cải "
      r"thiện} mô hình. Mọi giá trị lấy từ ước lượng hợp lý cực đại. Phép tách "
      r"này chỉ hợp lệ vì danh mục được thiết kế thành một lưới hai chiều, trong "
      r"đó mỗi cặp mô hình kề nhau chỉ khác \emph{đúng một} yếu tố.")
    p(r"\end{table}")
    p()


# ===========================================================================
# Buoc 2 -- mo phong khoi phuc tham so
# ===========================================================================
def _cov_tb(blk):
    v = [float(x) for x in blk.get("coverage", {}).values()]
    return sum(v) / len(v) if v else None


def bang_mo_phong_ms() -> None:
    mo_ta = {"S1": "Hai trạng thái phân biệt rõ",
             "S2": "Nhận dạng yếu, hai trạng thái gần nhau",
             "S3": r"Đuôi rất dày, $\nu_0 = 4$"}
    p(r"\begin{table}[H]")
    p(r"\centering\footnotesize")
    p(r"\caption{Khôi phục tham số, MS-GARCH-X Student-$t$: độ phủ trung bình của "
      r"khoảng tin cậy 95\%}")
    p(r"\label{tab:pra-coverage}")
    p(r"\begin{tabular}{llrrr}")
    p(r"\toprule")
    p(r"\textbf{KB} & \textbf{Đặc điểm} & $n$ & VB-MF & VB-FR \\")
    p(r"\midrule")
    for ma, mt in mo_ta.items():
        f = RESULTS / f"07_simulation_{ma}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        n = d["vb_mean_field"]["n_replications"]
        p(f"${ma[0]}_{{{ma[1]}}}$ & {mt} & {n} & "
          f"{so(_cov_tb(d['vb_mean_field']), 3)} & {so(_cov_tb(d['vb_full_rank']), 3)} \\\\")
    p(r"\bottomrule")
    p(r"\end{tabular}")
    p(r"\vspace{0.3em}")
    p(r"\raggedright\scriptsize")
    p(r"\emph{Ghi chú:} Giá trị lý tưởng là $0{,}95$. Mỗi kịch bản $T = 2000$ "
      r"quan sát, burn-in 200. Ước lượng hợp lý cực đại không có cột này vì quy "
      r"trình không sinh khoảng tin cậy cho nó. Nhãn trạng thái đã được sắp theo "
      r"phương sai vô điều kiện $\omega_s/(1-\alpha_s-\beta_s)$ \emph{trước khi} "
      r"gộp; không sắp nhãn thì mọi chỉ tiêu của tham số theo trạng thái đều vô "
      r"nghĩa vì hai phương pháp Bayes đảo nhãn ở khoảng một nửa số lần lặp.")
    p(r"\end{table}")
    p()


def bang_mo_phong_egarch() -> None:
    mo_ta = {"E1": (r"$\theta_0 = -0{,}10$", "Có đòn bẩy: phát hiện được không?"),
             "E2": (r"$\theta_0 = 0$", "Không đòn bẩy: có bịa ra không?"),
             "E3": (r"$\nu_0 = 4$", "Đuôi dày: còn khôi phục đúng không?")}
    p(r"\begin{table}[H]")
    p(r"\centering\footnotesize")
    p(r"\caption{Khôi phục tham số, EGARCH-X Student-$t$ một trạng thái: "
      r"hệ số đòn bẩy $\theta$}")
    p(r"\label{tab:prb-theta}")
    p(r"\begin{tabular}{llrrrr}")
    p(r"\toprule")
    p(r"\textbf{KB} & \textbf{Phương pháp} & $n$ & Độ chệch & RMSE & Độ phủ \\")
    p(r"\midrule")
    for ma, (gt, _) in mo_ta.items():
        f = RESULTS / "mo_phong_egarch" / f"{ma}.pt"
        if not f.exists():
            continue
        d = torch.load(f, weights_only=False)
        for i, m in enumerate(PP):
            if m not in d:
                continue
            b = d[m]
            cov = b["coverage"].get("theta")
            nhan = f"${ma[0]}_{{{ma[1]}}}$ ({gt})" if i == 0 else ""
            p(f"{nhan} & {TEN_PP[m]} & {b['n_replications']} & "
              f"{so(b['bias']['theta'], 4)} & {so(b['rmse']['theta'], 4)} & "
              f"{so(cov, 3)} \\\\")
        p(r"\addlinespace[0.2em]")
    p(r"\bottomrule")
    p(r"\end{tabular}")
    p(r"\vspace{0.3em}")
    p(r"\raggedright\scriptsize")
    p(r"\emph{Ghi chú:} $T = 2000$, $B = 100$ lần lặp mỗi ô, burn-in 200. Kịch bản "
      r"$E_2$ kiểm định chiều \emph{ngược lại} của $E_1$: nếu $\hat\theta$ lệch "
      r"khỏi 0 một cách hệ thống trên dữ liệu vốn không có đòn bẩy thì mọi kết "
      r"luận về đòn bẩy của VN30F1M đều mất giá trị. Mô hình một trạng thái không "
      r"có biến trạng thái ẩn nên điểm Brier không áp dụng.")
    p(r"\end{table}")
    p()


def main() -> None:
    p(r"% ==========================================================================")
    p(r"% FILE NAY DUOC SINH TU DONG -- KHONG SUA TAY.")
    p(r"% Sinh boi: code thuat toan/scripts/bao_cao/sinh_ket_qua_moi.py")
    p(f"% Sinh luc: {datetime.now():%Y-%m-%d %H:%M:%S}")
    p(r"% Muon doi noi dung: sua script roi chay lai.")
    p(r"% ==========================================================================")
    p()

    kq = bang_buoc_1()
    bang_phan_ra(kq["bang"])
    bang_mo_phong_ms()
    bang_mo_phong_egarch()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_d) + "\n", encoding="utf-8")
    t = kq["tot"]
    print(f"-> da ghi {OUT}  ({len(_d)} dong)")
    print("Mo hinh tot nhat theo tung tieu chi:")
    for tc, r in t.items():
        print(f"  {tc.upper():<6}: {r['ten']} + {TEN_PP[r['phuong_phap']]}")


if __name__ == "__main__":
    main()
