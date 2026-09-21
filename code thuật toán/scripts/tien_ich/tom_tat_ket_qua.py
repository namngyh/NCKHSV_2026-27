"""Xuat MOT file text tong hop moi ket qua uoc luong da co trong `results/`.

Bo sung cho hai kenh bao cao san co, khong thay the chung:
  - `build_html_report.py` -> bao cao HTML day du (bang, bieu do);
  - `results/*.pt`         -> checkpoint may doc, dung cho cac buoc sau.
File text nay la ban ghi BEN, de doc bang mat, de grep, de dan vao bai bao,
va khong phu thuoc trinh duyet.

Doc `results/parts/*.pt` neu co (chay bang run_models_parallel.py), neu khong
thi doc thang `02_mle.pt` / `03_vb.pt` / `05_r1.pt`.

Chay:
    python scripts/tien_ich/tom_tat_ket_qua.py
    python scripts/tien_ich/tom_tat_ket_qua.py --results-dir results --out results/tom_tat.txt
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from msgarchx.inference.summary import posterior_summary, relabel_posterior_samples
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


def f(x, nd=4):
    """Dinh dang mot so vo huong; tra ve '--' khi thieu hoac NaN."""
    if x is None:
        return "--"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if x != x else f"{x:.{nd}f}"


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
    # Co gan bien tinh LAI theo tieu chi HIEN TAI. Cac gia tri trong `res.runs`
    # duoc ghi luc CHAY, khi _near_boundary con chua kiem tra bien DUOI cua
    # alpha/beta -- nen chung co the bo lot truong hop alpha ~ 0.
    from msgarchx.inference.mle import ZERO_FLOOR, _near_boundary
    if _near_boundary(p):
        ly_do = []
        if bool((p.alpha < ZERO_FLOOR).any()):
            ly_do.append(f"alpha={[f'{v:.2e}' for v in p.alpha.tolist()]} cham day 0")
        if bool((p.beta < ZERO_FLOOR).any()):
            ly_do.append(f"beta={[f'{v:.2e}' for v in p.beta.tolist()]} cham day 0")
        pd = torch.diagonal(p.P)
        if bool((pd > 0.995).any()):
            ly_do.append(f"p_ss={[round(v,4) for v in pd.tolist()]} sat bien tren")
        if bool(((p.alpha + p.beta) > 0.995).any()):
            ly_do.append("alpha+beta > 0,995")
        L += ["", f"  !! NGHIEM NAM GAN BIEN (tieu chi hien tai): {'; '.join(ly_do)}",
              "     Hau qua: sai so chuan tu nghich dao Hessian KHONG con hop le,",
              "     vi tiem can chuan cua uoc luong hop ly cuc dai gia dinh nghiem",
              "     nam BEN TRONG khong gian tham so. Phai neu ro khi bao cao."]
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


# ============================================================
# Phan bo sung: do nhay tien nghiem, chan doan MCMC, mo phong
# ============================================================
def _muc_do_nhay(rd: Path) -> list:
    """Bang so sanh trung binh hau nghiem qua cac tien nghiem khac nhau."""
    d = rd / "parts" / "do_nhay"
    if not d.exists():
        return []
    co = {}
    for tep_bt in sorted(d.glob("*.pt")):
        p = torch.load(tep_bt, weights_only=False)
        co[p["bien_the"]] = p
    if not co:
        return []
    # moc so sanh: M2 cua pipeline chinh (cung tien nghiem ban thao)
    f_vb = rd / "03_vb.pt"
    if f_vb.exists():
        co["goc"] = {"result": torch.load(f_vb, weights_only=False)["m2"],
                     "mo_ta": "dac ta ban thao (M2 trong pipeline chinh)",
                     "elapsed": float("nan")}

    L = [LINE, "PHAN 3 -- DO NHAY VOI TIEN NGHIEM", LINE, "",
         "  Cau hoi: ket luan co doi khi doi ho tien nghiem khong? Chi dung VB",
         "  mean-field vi day la cau hoi ve DO ON DINH, khong phai ve do chinh xac",
         "  cua hau nghiem.", ""]
    for t, v in co.items():
        L.append(f"    {t:<10} = {v['mo_ta']}")
    L.append("")

    ten = list(co)
    tt = {}
    for t in ten:
        mau, ty_le = relabel_posterior_samples(co[t]["result"].samples)
        tt[t] = (posterior_summary(mau), ty_le)
    L.append("  " + f"{'Tham so':<12}" + "".join(f"{t:>14}" for t in ten))
    for h in ["omega", "alpha", "beta", "p11", "p22", "nu_minus_2"]:
        ten_h = "nu" if h == "nu_minus_2" else h
        sh = 2.0 if h == "nu_minus_2" else 0.0
        m0 = tt[ten[0]][0][h]["mean"]
        n = 1 if m0.dim() == 0 else m0.shape[0]
        for i in range(n):
            nhan = ten_h if n == 1 else f"{ten_h}_{i+1}"
            o = f"  {nhan:<12}"
            for t in ten:
                mm = tt[t][0][h]["mean"]
                o += f"{float(mm if mm.dim() == 0 else mm[i]) + sh:>14.4f}"
            L.append(o)
    L.append("  " + f"{'ELBO cuoi':<12}" + "".join(
        f"{co[t]['result'].final_elbo:>14.2f}" for t in ten))
    L.append("  " + f"{'% doi nhan':<12}" + "".join(f"{tt[t][1]:>13.1%}" for t in ten))
    L += ["",
          "  LUU Y 1: ELBO cua cac bien the KHONG so sanh truc tiep duoc voi nhau,",
          "           vi chung la can duoi cua log bang chung duoi CAC MO HINH",
          "           XAC SUAT KHAC NHAU.",
          "  LUU Y 2: neu tham so thay doi nhieu qua cac tien nghiem thi du lieu",
          "           KHONG lan at duoc tien nghiem -- mo hinh duoc nhan dang yeu,",
          "           va so sanh MLE voi VB/MCMC (RQ3, RQ4) bi nhieu boi chinh lua",
          "           chon tien nghiem. Phai khai bao dieu nay trong bai bao.", ""]
    return L


def _muc_mcmc(rd: Path) -> list:
    """Chan doan hoi tu MCMC, ke ca cac lan chay da that bai."""
    L = [LINE, "PHAN 4 -- CHAN DOAN MCMC (NUTS)", LINE, ""]
    lan = [("04_mcmc_lan1_duongcheo", "Lan 1: ma tran khoi luong DUONG CHEO, "
            "warmup 250, khoi tao tu tien nghiem"),
           ("04_mcmc", "Lan 2: ma tran khoi luong DAC, warmup 1000, "
            "khoi tao tu nghiem MLE chinh quy")]
    co_gi = False
    for ten, mo_ta in lan:
        tep_mc = rd / f"{ten}.pt"
        if not tep_mc.exists():
            L += [f"  {mo_ta}", "    -> chua co ket qua (dang chay hoac chua chay)", ""]
            continue
        co_gi = True
        m4 = torch.load(tep_mc, weights_only=False)["m4"]
        nc, ns = next(iter(m4.samples_by_chain.values())).shape[:2]
        dat = m4.max_rhat == m4.max_rhat and m4.max_rhat < 1.01
        L += [f"  {mo_ta}",
              f"    {nc} chuoi x {ns} mau = {nc*ns} mau hau nghiem",
              f"    R-hat lon nhat = {f(m4.max_rhat, 4)}   (yeu cau < 1,01)",
              f"    ESS nho nhat   = {f(m4.min_ess, 1)}",
              f"    So lan phan ky = {m4.n_divergences}",
              f"    KET LUAN: {'DAT' if dat else 'KHONG DAT -- mau KHONG dung duoc'}",
              ""]
        if not dat:
            L.append("    R-hat theo tung tham so:")
            for k, v in m4.rhat.items():
                nm = "nu" if k == "nu_minus_2" else k
                L.append(f"      {nm:<12} {f(v, 4):>10}   ESS = {f(m4.ess_bulk[k], 1)}")
            L.append("")
    if co_gi:
        L += ["  GHI CHU: da kiem tra rieng xem R-hat lon co phai do DOI NHAN TRANG",
              "  THAI gay ra khong. Voi lan 1: sap xep lai nhan cho tung mau chi dua",
              "  R-hat tu 4,0601 xuong 4,0516 -- KHONG phai nguyen nhan. Do phan tan",
              "  GIUA cac chuoi lon gap 10-800 lan do phan tan TRONG tung chuoi, tuc",
              "  moi chuoi dung yen tai mot diem khac nhau.", ""]
    return L


def _muc_mo_phong(rd: Path) -> list:
    """Ket qua mo phong theo tung kich ban da co."""
    tep = sorted(rd.glob("07_simulation_*.pt"))
    if not tep:
        return []
    L = [LINE, "PHAN 5 -- NGHIEN CUU MO PHONG (S1-S4)", LINE, "",
         "  Tham so 4 kich ban duoc hieu chuan tu ket qua thuc nghiem VN30F1M.",
         ""]
    for tep_mp in tep:
        code = tep_mp.stem.replace("07_simulation_", "")
        d = torch.load(tep_mp, weights_only=False)
        L.append(f"  --- Kich ban {code} ---")
        for pp, kq in d.items():
            n = kq.get("n_replications", "?")
            L.append(f"    {pp:<16} {n} lan lap   Brier trung binh = "
                     f"{f(kq.get('mean_brier'), 4)}")
            bias, rmse = kq.get("bias", {}), kq.get("rmse", {})
            cov = kq.get("coverage", {})
            ten_ts = [k for k in ("omega1", "omega2", "alpha1", "alpha2",
                                  "beta1", "beta2", "p11", "p22", "nu") if k in bias]
            if ten_ts:
                L.append(f"      {'tham so':<10}{'bias':>12}{'RMSE':>12}"
                         + (f"{'coverage':>12}" if cov else ""))
                for k in ten_ts:
                    o = f"      {k:<10}{f(bias[k], 4):>12}{f(rmse.get(k), 4):>12}"
                    if cov:
                        o += f"{f(cov.get(k), 3):>12}"
                    L.append(o)
        L.append("")
    L += ["  Coverage chi tinh duoc cho VB va MCMC (co khoang tin cay truc tiep tu",
          "  hau nghiem). Gia tri ly tuong la 0,95; thap hon dang ke nghia la phuong",
          "  phap danh gia THAP do bat dinh.", ""]
    return L

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

    L += _muc_do_nhay(rd)
    L += _muc_mcmc(rd)
    L += _muc_mo_phong(rd)

    timing_path = rd / "09_timing.pt"
    if timing_path.exists():
        t = torch.load(timing_path, weights_only=False)
        L += [LINE, "CHI PHI TINH TOAN", LINE, ""]
        for k, v in sorted(t.items()):
            L.append(f"  {k:<24} {v/60:8.1f} phut  ({v:.0f} giay)")
        L += [f"  {'TONG (cong don, khong ke chay song song)':<24} "
              f"{sum(t.values())/60:8.1f} phut", ""]

    L += [LINE, "CAC BUOC CHUA CHAY", LINE,
          "  - Mo phong S1-S4      : chua chay", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Da ghi tom tat vao: {out_path}  ({len(L)} dong)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
