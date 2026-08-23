"""Xuat MOT file text tong hop moi ket qua uoc luong da co trong `results/`.

Bo sung cho hai kenh bao cao san co, khong thay the chung:
  - `build_html_report.py` -> bao cao HTML day du (bang, bieu do);
  - `results/*.pt`         -> checkpoint may doc, dung cho cac buoc sau.
File text nay la ban ghi BEN, de doc bang mat, de grep, de dan vao bai bao,
va khong phu thuoc trinh duyet.

Doc `results/parts/*.pt` neu co (chay bang run_models_parallel.py), neu khong
thi doc thang `02_mle.pt` / `03_vb.pt` / `05_r1.pt`.

Chay:
    python scripts/tom_tat_ket_qua.py
    python scripts/tom_tat_ket_qua.py --results-dir results --out results/tom_tat.txt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from msgarchx.models.ms_garch import (
    MSGARCHXParams,
    expected_duration,
    label_switch_correction,
    ms_garch_forward_filter,
    stationary_distribution,
)

LINE = "=" * 78
SUB = "-" * 78

MLE_LABELS = {
    "b1": "B1  GARCH(1,1) chuan, khong X                     (1 trang thai)",
    "b2": "B2  GARCH(1,1) Student-t, khong X                 (1 trang thai)",
    "b3": "B3  GARCH-X Student-t                             (1 trang thai)",
    "b4": "B4  MS-GARCH chuan (nu co dinh lon), khong X      (2 trang thai)",
    "m1": "M1  MS-GARCH-X Student-t                          (2 trang thai)",
}
VB_LABELS = {
    "m2": "M2  MS-GARCH-X Student-t, VB mean-field",
    "m3": "M3  MS-GARCH-X Student-t, VB full-rank",
    "r1": "R1  MS-GARCH-X, nu_s rieng theo trang thai, VB mean-field",
}


def _v(t, nd=4):
    if t is None:
        return "--"
    if t.dim() == 0:
        return f"{t.item():.{nd}f}"
    return "[" + ", ".join(f"{x:.{nd}f}" for x in t.reshape(-1).tolist()) + "]"


def _load_parts(results_dir: Path) -> dict:
    """Tra ve {ten_mo_hinh: (ket_qua, thong_tin_phu)} tu parts/ hoac tu checkpoint gop."""
    out = {}
    pdir = results_dir / "parts"
    if pdir.exists():
        # CHI nhan cac tep co ten dung bang ten mo hinh. Cac ban sao luu nhu
        # `m1_lan_chay_truoc.pt` cung mang khoa d["model"] = "m1", neu nap ca
        # chung thi ban sao luu se GHI DE ket qua moi (thu tu chu cai dat
        # "m1.pt" truoc "m1_lan_chay_truoc.pt").
        hop_le = set(MLE_LABELS) | set(VB_LABELS)
        for p in sorted(pdir.glob("*.pt")):
            if p.stem not in hop_le:
                continue
            d = torch.load(p, weights_only=False)
            out[d["model"]] = (d["result"], d)
    for fname, keys in [("02_mle.pt", ["b1", "b2", "b3", "b4", "m1"]),
                        ("03_vb.pt", ["m2", "m3"])]:
        f = results_dir / fname
        if f.exists():
            d = torch.load(f, weights_only=False)
            for k in keys:
                if k in d and k not in out:
                    out[k] = (d[k], d)
    f = results_dir / "05_r1.pt"
    if f.exists() and "r1" not in out:
        d = torch.load(f, weights_only=False)
        out["r1"] = (d["r1"], d)
    return out



def _vb_state_order(res, r, X):
    """Hau nghiem VB/MCMC KHONG duoc tu dong sap xep nhan trang thai.

    Chay bo loc tien tai trung binh hau nghiem roi so E(h|S=1) voi E(h|S=2)
    de biet nhan co thoa eq:label-identification hay khong. Tra ve
    (mean_h, dung_quy_uoc) hoac (None, None) neu thieu du lieu.
    """
    if r is None or X is None:
        return None, None
    s = res.summary
    need = ("mu", "omega", "alpha", "beta", "gamma", "p11", "p22", "nu_minus_2")
    if any(k not in s for k in need):
        return None, None
    p11, p22 = s["p11"]["mean"], s["p22"]["mean"]
    P = torch.stack([torch.stack([p11, 1.0 - p11]), torch.stack([1.0 - p22, p22])])
    params = MSGARCHXParams(
        mu=s["mu"]["mean"], omega=s["omega"]["mean"], alpha=s["alpha"]["mean"],
        beta=s["beta"]["mean"], gamma=s["gamma"]["mean"], P=P,
        nu=s["nu_minus_2"]["mean"] + 2.0,
    )
    with torch.no_grad():
        out = ms_garch_forward_filter(params, r, X)
    mean_h = out["h"].mean(dim=0)
    return mean_h, bool(mean_h[0] < mean_h[1])


def _garch_block(name: str, res, meta: dict) -> list[str]:
    p = res.params
    L = [f"{MLE_LABELS.get(name, name)}", SUB,
         f"  log-hop-ly = {res.loglik:.4f}    AIC = {res.aic:.2f}    BIC = {res.bic:.2f}"
         f"    so tham so = {res.n_params}",
         f"  mu    = {_v(p.mu)}",
         f"  omega = {_v(p.omega)}",
         f"  alpha = {_v(p.alpha)}",
         f"  beta  = {_v(p.beta)}",
         f"  alpha + beta = {(p.alpha + p.beta).item():.4f}"
         f"{'   <-- GAN BIEN (>0.995)' if (p.alpha + p.beta).item() > 0.995 else ''}",
         f"  nu    = {_v(p.nu) if p.nu is not None else '-- (phan phoi chuan)'}"]
    if p.gamma.numel():
        L.append(f"  gamma = {_v(p.gamma)}")
    L += ["", "  Chan doan tung diem khoi tao:"]
    for d in res.runs:
        L.append(f"    seed={d.seed}  loglik={d.loglik:12.4f}  "
                 f"hoi_tu={'co ' if d.converged else 'KHONG'}  ||grad||={d.grad_norm:.4f}")
    nconv = sum(1 for d in res.runs if d.converged)
    L.append(f"  => hoi tu {nconv}/{len(res.runs)} diem khoi tao")
    if meta.get("elapsed"):
        L.append(f"  => thoi gian chay: {meta['elapsed']/60:.1f} phut")
    return L + [""]


def _ms_block(name: str, res, meta: dict) -> list[str]:
    p = res.params
    h = res.filter_output["h"] if res.filter_output is not None else None
    swapped = False
    if h is not None:
        p2 = label_switch_correction(p, h)
        swapped = not torch.equal(p2.omega, p.omega)
        p = p2
    L = [f"{MLE_LABELS.get(name, name)}", SUB,
         f"  log-hop-ly = {res.loglik:.4f}    AIC = {res.aic:.2f}    BIC = {res.bic:.2f}",
         f"  (trang thai da sap xep theo eq:label-identification E(h|S=1) < E(h|S=2)"
         f"{'; DA HOAN VI' if swapped else '; khong can hoan vi'})",
         f"  mu    = {_v(p.mu)}",
         f"  omega = {_v(p.omega)}",
         f"  alpha = {_v(p.alpha)}",
         f"  beta  = {_v(p.beta)}",
         f"  alpha + beta = {_v(p.alpha + p.beta)}"]
    if p.gamma.numel():
        L.append(f"  gamma = {_v(p.gamma)}   (hang 1 = trang thai 1, hang 2 = trang thai 2)")
    L += [f"  P     = [[{p.P[0,0]:.4f}, {p.P[0,1]:.4f}], [{p.P[1,0]:.4f}, {p.P[1,1]:.4f}]]",
          f"  thoi gian ton tai ky vong d_s = {_v(expected_duration(p.P), 2)} phien",
          f"  phan phoi dung pi = {_v(stationary_distribution(p.P))}",
          f"  nu    = {_v(p.nu)}"]
    se = meta.get("m1_se")
    if se is not None:
        L += ["", "  Sai so chuan (delta method tu Hessian quan sat):"]
        for k in ["mu", "omega", "alpha", "beta", "gamma", "p11", "p22", "nu"]:
            if k in se:
                L.append(f"    SE({k:<5}) = {_v(se[k])}")
    elif name == "m1":
        L.append("  Sai so chuan: KHONG CO (Hessian khong xac dinh duong tai nghiem)")
    L += ["", "  Chan doan tung diem khoi tao:"]
    for d in res.runs:
        L.append(f"    seed={d.seed}  loglik={d.loglik:12.4f}  "
                 f"hoi_tu={'co ' if d.converged else 'KHONG'}  ||grad||={d.grad_norm:10.4f}  "
                 f"gan_bien={'co' if d.near_boundary else 'khong'}")
    nconv = sum(1 for d in res.runs if d.converged)
    L.append(f"  => hoi tu {nconv}/{len(res.runs)} diem khoi tao")
    if meta.get("elapsed"):
        L.append(f"  => thoi gian chay: {meta['elapsed']/60:.1f} phut")
    return L + [""]


def _vb_block(name: str, res, meta: dict, r=None, X=None) -> list[str]:
    L = [f"{VB_LABELS.get(name, name)}", SUB,
         f"  ELBO cuoi (trung binh 50 buoc cuoi) = {res.final_elbo:.4f}",
         f"  so buoc SVI = {len(res.elbo_trace)}    so mau hau nghiem = "
         f"{len(next(iter(res.samples.values())))}",
         "",
         f"  {'Tham so':<12} {'trung binh':>12} {'do lech chuan':>14} "
         f"{'q2.5%':>12} {'q97.5%':>12}"]
    for k, s in res.summary.items():
        mean, sd = s["mean"].reshape(-1), s["sd"].reshape(-1)
        lo, hi = s["q2.5"].reshape(-1), s["q97.5"].reshape(-1)
        for i in range(mean.numel()):
            tag = k if mean.numel() == 1 else f"{k}[{i}]"
            L.append(f"  {tag:<12} {mean[i]:12.4f} {sd[i]:14.4f} {lo[i]:12.4f} {hi[i]:12.4f}")
    mean_h, ok = _vb_state_order(res, r, X)
    if mean_h is not None:
        L += ["",
              f"  Nhan trang thai: E(h|S=1) = {mean_h[0]:.4f}, E(h|S=2) = {mean_h[1]:.4f}"
              f"  -> quy uoc eq:label-identification {'DUOC THOA' if ok else 'BI VI PHAM'}"]
        if not ok:
            L += ["  !! CANH BAO: trang thai 1/2 o bang tren BI DAO so voi quy uoc cua bai",
                  "     bao (trang thai 1 phai la bien dong THAP). Hau nghiem VB/MCMC khong",
                  "     duoc sap xep nhan tu dong o bat ky dau trong pipeline, ke ca",
                  "     build_html_report.py::_posterior_rows. Phai hoan vi truoc khi dua",
                  "     vao bang cua bai bao va truoc khi tinh ty so do lech chuan VB/MCMC."]
    if name == "r1" and meta.get("p_tail") is not None:
        L += ["", f"  P(nu_2 < nu_1 | Data_T) tren mau THO = {meta['p_tail']:.4f}",
              "     (day la con so run_full_study.py::run_r1_step dang ghi lai --"
              " tinh truc tiep tu mau, KHONG sap xep nhan)"]
        if "nu_minus_2" in res.samples:
            # Sap xep nhan theo TUNG MAU (khong phai mot phep hoan vi toan cuc):
            # mot phan mau co the o thu tu nguoc lai, va phan do phai duoc tinh
            # dung thay vi bi ep theo da so.
            from msgarchx.inference.summary import relabel_posterior_samples
            mau, ty_le = relabel_posterior_samples(res.samples)
            nus = mau["nu_minus_2"] + 2.0
            p_fix = float((nus[:, 1] < nus[:, 0]).float().mean())
            L += [f"  P(nu_bien-dong-CAO < nu_bien-dong-THAP | Data_T) = {p_fix:.4f}"
                  "   <-- CON SO DUNG cho RQ6",
                  f"     (nu trung binh: bien dong thap = {float(nus[:,0].mean()):.3f}, "
                  f"bien dong cao = {float(nus[:,1].mean()):.3f})",
                  f"     ty le mau phai hoan vi nhan = {ty_le:.3f}"]
    if meta.get("elapsed"):
        L.append(f"  => thoi gian chay: {meta['elapsed']/60:.1f} phut")
    return L + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = Path(args.results_dir)
    out_path = Path(args.out) if args.out else rd / "tom_tat_ket_qua.txt"
    got = _load_parts(rd)
    if not got:
        print(f"[LOI] Khong tim thay ket qua nao trong {rd}", file=sys.stderr)
        return 1

    L = [LINE,
         "TOM TAT KET QUA UOC LUONG -- MS-GARCH-X Student-t cho VN30F1M",
         f"Sinh luc: {datetime.now():%Y-%m-%d %H:%M:%S}",
         f"Thu muc ket qua: {rd}",
         LINE, ""]

    r_train = X_train = None
    data_path = rd / "00_data.pt"
    if data_path.exists():
        d = torch.load(data_path, weights_only=False)
        r_all, n_train = d["r_all"], d["n_train"]
        r_train, X_train = r_all[:n_train], d["X_all"][:n_train]
        L += ["DU LIEU", SUB,
              f"  tong so quan sat loi suat = {r_all.shape[0]}",
              f"  tap huan luyen = {n_train}   tap kiem dinh = {r_all.shape[0]-n_train}",
              f"  bien ngoai sinh k = {d['X_all'].shape[1]}",
              f"  loi suat: trung binh = {r_all.mean():.4f}, do lech chuan = {r_all.std():.4f}, "
              f"min = {r_all.min():.4f}, max = {r_all.max():.4f}", ""]

    L += [LINE, "PHAN 1 -- UOC LUONG HOP LY CUC DAI (MLE)", LINE, ""]
    for n in ["b1", "b2", "b3"]:
        if n in got:
            L += _garch_block(n, *got[n])
    for n in ["b4", "m1"]:
        if n in got:
            L += _ms_block(n, *got[n])

    rows = [(n, got[n][0]) for n in ["b1", "b2", "b3", "b4", "m1"] if n in got]
    if rows:
        L += [SUB, "SO SANH MO HINH (nho hon la tot hon)", SUB,
              f"  {'Mo hinh':<10}{'log-hop-ly':>14}{'AIC':>12}{'BIC':>12}"]
        for n, res in rows:
            L.append(f"  {n.upper():<10}{res.loglik:14.2f}{res.aic:12.2f}{res.bic:12.2f}")
        best = min(rows, key=lambda x: x[1].aic)
        L += [f"  => AIC thap nhat: {best[0].upper()}", ""]

    if any(n in got for n in ["m2", "m3", "r1"]):
        L += [LINE, "PHAN 2 -- SUY DIEN BAYES BANG VARIATIONAL BAYES", LINE, ""]
        for n in ["m2", "m3", "r1"]:
            if n in got:
                L += _vb_block(n, *got[n], r=r_train, X=X_train)

    timing_path = rd / "09_timing.pt"
    if timing_path.exists():
        t = torch.load(timing_path, weights_only=False)
        L += [LINE, "CHI PHI TINH TOAN", LINE, ""]
        for k, v in sorted(t.items()):
            L.append(f"  {k:<24} {v/60:8.1f} phut  ({v:.0f} giay)")
        L += [f"  {'TONG (cong don, khong ke chay song song)':<24} "
              f"{sum(t.values())/60:8.1f} phut", ""]

    L += [LINE, "CAC BUOC CHUA CHAY", LINE,
          "  - M4 (MCMC/NUTS)      : chua chay -> results/04_mcmc.pt",
          "  - Danh gia ngoai mau  : CHO MCMC xong roi hay chay, neu khong bang so sanh",
          "                          se AM THAM thieu M4 (xem README muc 'Cach 2')",
          "  - Mo phong S1-S8      : chua chay", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Da ghi tom tat vao: {out_path}  ({len(L)} dong)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
