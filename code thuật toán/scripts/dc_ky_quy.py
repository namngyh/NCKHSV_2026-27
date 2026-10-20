"""Giai đoạn 2 theo đề cương (mục 7): ký quỹ động từ dự báo EGARCH-X, chỉ MLE.

LÝ THUYẾT: xem `de_cuong/ky_quy.py`. Quy trình:
  1. Mô phỏng VaR/ES H phiên (H = 1, 2, 3; alpha = 5%, 2,5%, 1%) tại mọi phiên gốc của XT và KT
     từ kết quả cuốn chiếu (tham số ước lượng lại mỗi 21 phiên).
  2. Với mỗi cặp thiết kế (H, alpha) hiệu chỉnh (w, b, phi, eta) trên XT bằng chi phí kỳ vọng với
     c_th / c_v = 1 / alpha (alpha* = c_v / c_th), cố định trên KT.
  3. So với đối chứng, báo cáo cả bản gốc và bản cân bằng ký quỹ trung bình (so sánh tại cùng m_tb).
Chạy:  python scripts/dc_ky_quy.py [--S 10000]
THƯ VIỆN: numpy, scipy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_chay import RA, doc_pkl, luu_json, luu_pkl, nap, nhom_phien   # noqa: E402
from de_cuong import danh_gia as dg                                    # noqa: E402
from de_cuong import ky_quy as kq                                      # noqa: E402
from de_cuong.mo_hinh import Cau_hinh                                  # noqa: E402

LUOI = {"w": (0.0, 0.5, 1.0), "b": (0.0, 0.1, 0.25), "phi": (0.5, 0.75, 1.0), "eta": (0.0, 0.8, 0.95)}
GIAM, BUOC_LAM_TRON = 0.01, 0.005      # D (giới hạn tốc độ giảm) và bước làm tròn g (chưa xác minh theo thông lệ)
GIOI_HAN_GIA, VSD = 0.07, 0.13         # biên độ giá ngày; mức 13% là giá trị làm việc chưa xác minh


def can_bang(m, muc_tieu, hop_le):
    """Nhân chuỗi m với hệ số để trung bình trên các phiên hợp lệ bằng muc_tieu (so sánh tại cùng m_tb)."""
    ok = hop_le & np.isfinite(m)
    return m * (muc_tieu / m[ok].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=10000)
    a = ap.parse_args()
    d, blk = nap()
    pk = doc_pkl("05_cuon_chieu_XT_KT.pkl")
    rs, cfgs, dau, cuoi, kt0 = pk["rs"], pk["cfgs"], pk["dau"], pk["cuoi"], pk["kt0"]
    pp = json.loads((RA / "04_chon_phan_phoi.json").read_text(encoding="utf-8"))["chon"]
    khoa_chinh, khoa_egarch = f"chon|{pp}", f"EGARCH|{pp}"
    n_goc = cuoi - dau
    tmoc = np.arange(dau - 1, cuoi - 1)
    R, ok_h = kq.tong_that_thuc(d)
    P_KT = blk["KT"][0]

    t0 = time.time()
    sim = {}
    for ten, kk, che_do in ((khoa_chinh, khoa_chinh, "tham_so"), (khoa_chinh + "|FHS", khoa_chinh, "loc"),
                            (khoa_egarch, khoa_egarch, "tham_so")):
        cfg = Cau_hinh(cfgs[kk][0], cfgs[kk][1])
        sim[ten] = kq.mo_phong_ky_quy(d, cfg, rs[kk], S=a.S, che_do=che_do)
        print(f"  mô phỏng {ten}: {time.time()-t0:.0f} s", flush=True)

    xt = np.arange(n_goc) < kt0
    kt = ~xt
    ket_qua, series = {}, {}
    nm = nhom_phien(d, blk["KT"][0], blk["KT"][1])
    for H in kq.HS:
        L = np.abs(R[tmoc, H - 1])
        hop = ok_h[tmoc, H - 1] & np.isfinite(L)
        hop_xt = hop & xt & (tmoc + H < P_KT)       # XT không được nhìn sang KT
        hop_kt = hop & kt
        for ia, al in enumerate(kq.ALPHAS):
            ten_tk = f"H{H}_a{al}"
            san = kq.san_lich_su(R, tmoc, H, al)
            vmax = np.max(sim[khoa_chinh]["var"][:, H - 1, ia, :], axis=1)
            emax = np.max(sim[khoa_chinh]["es"][:, H - 1, ia, :], axis=1)
            best, luoi_tm = None, []
            for w, b, phi, eta in itertools.product(*LUOI.values()):
                m = kq.quy_tac(vmax, emax, san, w, b, phi, eta, GIAM, BUOC_LAM_TRON)
                c = kq.chi_phi(m, L, hop_xt, 1.0, 1.0 / al)
                ct = kq.chi_tieu(m, L, hop_xt, H)
                luoi_tm.append({"w": w, "b": b, "phi": phi, "eta": eta, "chi_phi": c, "m_tb": ct["m_tb"],
                                "ty_le_vuot": ct["ty_le_vuot"], "thieu_hut_tb": ct["thieu_hut_tb"]})
                if best is None or c < best[0]:
                    best = (c, (w, b, phi, eta), m)
            w, b, phi, eta = best[1]
            m_dong = best[2]
            ct_xt = kq.chi_tieu(m_dong, L, hop_xt, H)
            nang = {"dong_EGARCH-X": m_dong}
            nang["dong_EGARCH(khong_X)"] = kq.quy_tac(
                np.max(sim[khoa_egarch]["var"][:, H - 1, ia, :], axis=1),
                np.max(sim[khoa_egarch]["es"][:, H - 1, ia, :], axis=1), san, w, b, phi, eta, GIAM, BUOC_LAM_TRON)
            nang["VaR_tho_EGARCH-X(khong_quy_tac)"] = vmax
            nang["FHS_EGARCH-X"] = np.max(sim[khoa_chinh + "|FHS"]["var"][:, H - 1, ia, :], axis=1)
            nang["EWMA_0.94"] = kq.ewma_var(d, tmoc, H, al)
            nang["lich_su_250"] = kq.lich_su_250(R, tmoc, H, al)
            q_co_dinh = float(np.quantile(L[hop_xt], 1 - max(ct_xt["ty_le_vuot"], 1.0 / hop_xt.sum())))
            nang["co_dinh(khop_ty_le_vuot_XT)"] = np.full(n_goc, q_co_dinh)
            nang["VSD_13%(chua_xac_minh)"] = np.full(n_goc, VSD)
            nang[f"muc_tran_{int(GIOI_HAN_GIA*100)}%xH"] = np.full(n_goc, GIOI_HAN_GIA * H)
            m_tb_dong = kq.chi_tieu(m_dong, L, hop_kt, H)["m_tb"]
            bang = {}
            for ten, m in nang.items():
                row = kq.chi_tieu(m, L, hop_kt, H)
                row["kiem_dinh"] = kq.kiem_dinh_vuot(np.where(kt, m, np.nan), np.where(kt, L, np.nan), hop_kt, H, al)
                row["chi_phi"] = kq.chi_phi(m, L, hop_kt, 1.0, 1.0 / al)
                mc = m if ten == "dong_EGARCH-X" else can_bang(m, m_tb_dong, hop_kt)
                rc = kq.chi_tieu(mc, L, hop_kt, H)
                row["can_bang_m_tb"] = {k: rc[k] for k in ("m_tb", "ty_le_vuot", "thieu_hut_tb", "thieu_hut_max")}
                th_m = np.where(hop_kt & np.isfinite(mc), np.maximum(L - mc, 0), np.nan)
                th_d = np.where(hop_kt & np.isfinite(m_dong), np.maximum(L - m_dong, 0), np.nan)
                t_dm, p_dm = dg.dm(th_d, th_m)         # H1: quy tắc động có thiếu hụt thấp hơn đối chứng
                row["thieu_hut_moi_phien_DM_dong_vs_doi_chung"] = {
                    "t": t_dm, "p_mot_phia": p_dm, "tb_dong": float(np.nanmean(th_d)),
                    "tb_doi_chung": float(np.nanmean(th_m))}
                bang[ten] = row
            nhom = {}
            for g, mk in nm.items():
                mask = np.zeros(n_goc, bool)
                mask[kt0:] = mk
                nhom[g] = {"dong": kq.chi_tieu(m_dong, L, hop_kt, H, mask),
                           "co_dinh": kq.chi_tieu(nang["co_dinh(khop_ty_le_vuot_XT)"], L, hop_kt, H, mask)}
            B = np.where(hop_kt & np.isfinite(m_dong), (L > m_dong).astype(float), np.nan)
            ket_qua[ten_tk] = {"H": H, "alpha": al, "chon": {"w": w, "b": b, "phi": phi, "eta": eta},
                               "XT": ct_xt, "KT": bang["dong_EGARCH-X"], "bang_KT": bang, "nhom_KT": nhom,
                               "den_giao_thong": dg.den_giao_thong(B[kt0:]), "luoi_XT": luoi_tm,
                               "ghi_chu_chi_phi": f"c_th/c_v = {1/al:.0f}"}
            series[ten_tk] = dict(nang)
            series[ten_tk]["L"] = L
            series[ten_tk]["hop_le_KT"] = hop_kt
            print(f"  H={H} alpha={al}: chọn (w,b,phi,eta)={best[1]}  KT: m_tb={bang['dong_EGARCH-X']['m_tb']:.4f} "
                  f"vượt={bang['dong_EGARCH-X']['ty_le_vuot']:.4f} thiếu hụt tb={bang['dong_EGARCH-X']['thieu_hut_tb']:.4f}",
                  flush=True)
    luu_json("07_ky_quy.json", {"mo_hinh": {"chinh": [cfgs[khoa_chinh][0], cfgs[khoa_chinh][1]],
                                            "so_sanh": [cfgs[khoa_egarch][0], cfgs[khoa_egarch][1]]},
                                "S": a.S, "luoi": LUOI, "giam_toi_da_moi_phien": GIAM, "buoc_lam_tron": BUOC_LAM_TRON,
                                "ket_qua": ket_qua, "tmoc_dau": int(tmoc[0]), "kt0": int(kt0),
                                "ngay_KT": [str(d.date[blk["KT"][0]].date()), str(d.date[blk["KT"][1] - 1].date())]})
    luu_pkl("07_ky_quy.pkl", {"series": series, "tmoc": tmoc, "kt0": kt0})


if __name__ == "__main__":
    main()
