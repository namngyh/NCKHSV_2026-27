"""Phan tich DO NHAY VOI TIEN NGHIEM: chay lai M1 bang Variational Bayes duoi
cac tien nghiem khac nhau, roi so ket qua voi dac ta goc.

VI SAO CHI DUNG VB, KHONG DUNG MCMC: cau hoi o day la "ket luan co doi khong
khi doi tien nghiem", chu khong phai "hau nghiem chinh xac bang bao nhieu".
VB mean-field chay ~1,3 gio moi bien the; MCMC dang mat nhieu NGAY va con chua
hoi tu. Neu VB cho thay ket luan on dinh thi do da du de viet mot doan do nhay.

CAC BIEN THE (chon bang --bien-the):
  goc          Dung dac ta ban thao: omega ~ LogNormal(0,1), p_ss ~ Beta(8,2).
  omega_hc     omega_s ~ half-Cauchy(0; 1). Ly do: LogNormal(0,1) co mat do
               tien ve 0 khi omega -> 0 nen ep manh cac gia tri nho; uoc luong
               omega_1 = 0,100 nam o phan vi 1,1% cua tien nghiem hien tai.
               half-Cauchy co mat do huu han duong tai 0 (Gelman 2006).
  p_chat       p_ss ~ Beta(20,2). Ly do: day la mau thuan tien nghiem--du lieu
               LON NHAT. Beta(8,2) co KTC 95% chi toi 0,972 trong khi MLE cho
               p_11 = 0,9986 (phan vi ~100%). Beta(20,2) hop hon voi thuc te
               che do bien dong thi truong rat dai dang.
  ca_hai       Ap dong thoi hai thay doi tren.

Chay:
    python scripts/pipeline/chay_do_nhay_prior.py --tat-ca --jobs 2
    python scripts/pipeline/chay_do_nhay_prior.py --bien-the omega_hc
    python scripts/pipeline/chay_do_nhay_prior.py --so-sanh
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from msgarchx.inference.param_spec import ModelConfig, PriorConfig
from msgarchx.inference.summary import posterior_summary, relabel_posterior_samples
from msgarchx.inference.variational import fit_vb
from scripts.pipeline.run_full_study import PRESETS

BIEN_THE = {
    "goc": (PriorConfig(), "đặc tả bản thảo: omega ~ LogNormal(0,1), p_ss ~ Beta(8,2)"),
    "omega_hc": (PriorConfig(omega_prior="half_cauchy", omega_hc_scale=1.0),
                 "omega_s ~ half-Cauchy(0; 1)"),
    "p_chat": (PriorConfig(transition_prior_stay=20.0, transition_prior_switch=2.0),
               "p_ss ~ Beta(20,2)"),
    "ca_hai": (PriorConfig(omega_prior="half_cauchy", omega_hc_scale=1.0,
                           transition_prior_stay=20.0, transition_prior_switch=2.0),
               "half-Cauchy cho omega VA Beta(20,2) cho p_ss"),
}


def _log(m):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


def _thu_muc(cfg):
    d = cfg.results_dir / "parts" / "do_nhay"
    d.mkdir(parents=True, exist_ok=True)
    return d


def chay_mot(cfg, ten: str) -> None:
    path = _thu_muc(cfg) / f"{ten}.pt"
    if path.exists():
        _log(f"[{ten}] da co, bo qua")
        return
    prior, mo_ta = BIEN_THE[ten]
    d = torch.load(cfg.results_dir / "00_data.pt", weights_only=False)
    r, X = d["r_all"][: d["n_train"]], d["X_all"][: d["n_train"]]
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    _log(f"[{ten}] bat dau -- {mo_ta}")
    t0 = time.time()
    res = fit_vb(r, X, config, guide_type="mean_field", prior=prior,
                 n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws,
                 seed=cfg.seed, verbose=True)
    torch.save({"result": res, "prior": prior, "mo_ta": mo_ta, "bien_the": ten,
                "elapsed": time.time() - t0}, path)
    _log(f"[{ten}] xong sau {(time.time()-t0)/60:.1f} phut, ELBO cuoi = {res.final_elbo:.2f}")


def so_sanh(cfg) -> None:
    d = _thu_muc(cfg)
    co = {t: torch.load(d / f"{t}.pt", weights_only=False)
          for t in BIEN_THE if (d / f"{t}.pt").exists()}
    if not co:
        _log("chua co bien the nao chay xong")
        return
    # M2 goc da chay trong pipeline chinh -> dung lam moc neu chua co "goc"
    if "goc" not in co and (cfg.results_dir / "03_vb.pt").exists():
        co["goc"] = {"result": torch.load(cfg.results_dir / "03_vb.pt",
                                          weights_only=False)["m2"],
                     "mo_ta": "M2 tu pipeline chinh (cung tien nghiem goc)",
                     "elapsed": float("nan")}

    print("\n" + "=" * 84)
    print("  DO NHAY VOI TIEN NGHIEM -- trung binh hau nghiem (da sap xep nhan trang thai)")
    print("=" * 84)
    hang = ["omega", "alpha", "beta", "p11", "p22", "nu_minus_2"]
    print(f"\n  {'Tham so':<12}" + "".join(f"{t:>17}" for t in co))
    tt = {}
    for t, v in co.items():
        mau, ty_le = relabel_posterior_samples(v["result"].samples)
        tt[t] = (posterior_summary(mau), ty_le)
    for h in hang:
        ten_h = "nu" if h == "nu_minus_2" else h
        sh = 2.0 if h == "nu_minus_2" else 0.0
        m = tt[list(co)[0]][0][h]["mean"]
        n = 1 if m.dim() == 0 else m.shape[0]
        for i in range(n):
            nhan = ten_h if n == 1 else f"{ten_h}_{i+1}"
            o = f"  {nhan:<12}"
            for t in co:
                mm = tt[t][0][h]["mean"]
                o += f"{float(mm if mm.dim()==0 else mm[i]) + sh:>17.4f}"
            print(o)
    print(f"\n  {'ELBO cuoi':<12}" + "".join(
        f"{co[t]['result'].final_elbo:>17.2f}" for t in co))
    print(f"  {'% doi nhan':<12}" + "".join(f"{tt[t][1]:>16.1%}" for t in co))
    print(f"  {'phut chay':<12}" + "".join(f"{co[t]['elapsed']/60:>17.1f}" for t in co))
    print("\n  LUU Y: ELBO cua cac bien the KHONG so sanh truc tiep duoc voi nhau vi")
    print("  chung la can duoi cua log bang chung duoi CAC MO HINH XAC SUAT KHAC NHAU.")
    print("  Cai can doc la tham so uoc luong co ON DINH qua cac tien nghiem hay khong.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bien-the", dest="bien_the", choices=list(BIEN_THE))
    ap.add_argument("--tat-ca", dest="tat_ca", action="store_true")
    ap.add_argument("--so-sanh", dest="so_sanh", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--preset", default="full")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]

    if args.bien_the:
        chay_mot(cfg, args.bien_the)
        return 0
    if args.so_sanh:
        so_sanh(cfg)
        return 0
    if not args.tat_ca:
        ap.print_help()
        return 0

    # "goc" khong can chay lai: M2 trong pipeline chinh dung dung tien nghiem do
    can = [t for t in BIEN_THE if t != "goc"
           and not (_thu_muc(cfg) / f"{t}.pt").exists()]
    if not can:
        so_sanh(cfg)
        return 0
    log_dir = ROOT / "logs"; log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    _log(f"Chay {len(can)} bien the tren toi da {args.jobs} tien trinh: {can}")
    queue, running = list(can), []
    while queue or running:
        while queue and len(running) < args.jobs:
            t = queue.pop(0)
            fh = open(log_dir / f"donhay_{t}_{stamp}.log", "w", encoding="utf-8",
                      errors="replace")
            pr = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--bien-the", t],
                stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            running.append((t, pr, fh))
            _log(f"  bat dau {t} (pid {pr.pid})")
        time.sleep(15)
        con = []
        for t, pr, fh in running:
            if pr.poll() is None:
                con.append((t, pr, fh))
            else:
                fh.close()
                _log(f"  {'XONG' if pr.returncode == 0 else 'LOI'} {t}")
        running = con
    so_sanh(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
