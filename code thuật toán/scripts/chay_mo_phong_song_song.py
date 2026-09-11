"""Chay nghien cuu mo phong S1-S4 SONG SONG theo tung LAN LAP.

VI SAO: `run_full_study.py --only simulation` chay tuan tu tung kich ban va lan lap.
Voi 4 kich ban hieu chinh VN30F1M (S1-S4) x cac phuong phap (MLE, VB-MF, VB-FR),
chay song song tren nhieu tien trinh giup tan dung toi da CPU da nhan.

MOI LAN LAP MOT CHECKPOINT (`results/parts/sim/<kich_ban>_<phuong_phap>_<b>.pt`)
nen dung giua chung khong mat gi, va co the chay bo sung tung phan.
Sau khi hoan tat, dung `--combine` de gop thanh `results/07_simulation_S*.pt`.

Chay:
    python scripts/chay_mo_phong_song_song.py --methods mle,vb_mean_field,vb_full_rank
    python scripts/chay_mo_phong_song_song.py --scenarios S1 --jobs 4
    python scripts/chay_mo_phong_song_song.py --combine
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from msgarchx.inference.param_spec import ModelConfig
from msgarchx.simulation.dgp import simulate_ms_garch_x
from msgarchx.simulation.run_simulation import (
    _apply_misspecification,
    _named_params,
    _run_one_method,
    _summarize_method,
    sap_nhan_trang_thai,
)
from msgarchx.simulation.scenarios import default_scenarios
from scripts.run_full_study import PRESETS

METHODS = ("mle", "vb_mean_field", "vb_full_rank")


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _scenarios(cfg):
    return default_scenarios(T_default=cfg.sim_T, B_default=cfg.sim_B_mle)


def _n_reps(cfg, method: str) -> int:
    return {"mle": cfg.sim_B_mle, "vb_mean_field": cfg.sim_B_vb,
            "vb_full_rank": cfg.sim_B_vb}[method]


def _method_kwargs(cfg, method: str) -> dict:
    if method == "mle":
        return {"n_starts": cfg.sim_mle_n_starts, "n_adam": cfg.sim_mle_n_adam,
                "n_lbfgs": cfg.sim_mle_n_lbfgs}
    if method in ("vb_mean_field", "vb_full_rank"):
        return {"n_steps": cfg.sim_vb_n_steps, "n_posterior_draws": cfg.sim_vb_n_draws}
    raise ValueError(f"Phuong phap khong hop le: {method}")


def _sim_dir(cfg) -> Path:
    d = cfg.results_dir / "parts" / "sim"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Worker: MOT lan lap
# ============================================================
def worker(cfg, code: str, method: str, b: int) -> None:
    path = _sim_dir(cfg) / f"{code}_{method}_{b:03d}.pt"
    if path.exists():
        _log(f"[{code}/{method}/{b}] da co, bo qua")
        return
    scenario = _scenarios(cfg)[code]
    config = ModelConfig(n_exog=cfg.sim_n_exog, state_specific_nu=False)
    scn = replace(scenario, burn_in=cfg.sim_burn_in)

    # Seed GIONG HET run_scenario: base_seed + b, dung cho ca sinh du lieu lan
    # uoc luong -> ket qua song song trung khop voi chay tuan tu.
    seed = cfg.seed + b
    t0 = time.time()
    sim = simulate_ms_garch_x(scn.params, scn.P0, scn.T, n_exog=config.n_exog,
                              burn_in=scn.burn_in, seed=seed)
    r = _apply_misspecification(sim.r, scn.misspecified)
    _log(f"[{code}/{method}/{b}] bat dau (T={scn.T}, seed={seed})")
    rep = _run_one_method(method, r, sim.X, config, sim.state, seed=seed,
                          method_kwargs=_method_kwargs(cfg, method))
    # Ghi ra file tam roi doi ten, thay vi ghi thang vao `path`. Neu may mat
    # dien dung luc dang ghi, ta chi co mot file .tmp thua -- vo hai vi `combine`
    # chi quet dung duoi .pt. Ghi thang thi co the sinh ra mot file .pt CUT DOI:
    # lan chay sau thay no "da ton tai" nen bo qua vinh vien, con `combine` thi
    # khong doc noi. Do la kieu hong am tham, dat nhat de phat hien.
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"rep": rep, "code": code, "method": method, "b": b,
                "elapsed": time.time() - t0}, tmp)
    os.replace(tmp, path)
    br = f"brier={rep.brier:.4f}" if rep.brier is not None else "brier=--"
    _log(f"[{code}/{method}/{b}] xong sau {(time.time()-t0)/60:.1f} phut ({br})")


# ============================================================
# Gop -> 07_simulation_<code>.pt (dung dinh dang run_one_scenario)
# ============================================================
def combine(cfg, codes: list[str]) -> None:
    sdir = _sim_dir(cfg)
    scenarios = _scenarios(cfg)
    for code in codes:
        true_named = _named_params(scenarios[code].params)
        out, tong = {}, 0
        for method in METHODS:
            files = sorted(sdir.glob(f"{code}_{method}_*.pt"))
            if not files:
                continue
            tho = [torch.load(f, weights_only=False)["rep"] for f in files]
            # SAP NHAN TRUOC KHI GOP. Bo uoc luong tra ve nhan trang thai theo
            # thu tu tuy y (doi xung hoan vi nhan), nen gop truc tiep se tron
            # omega_1 voi omega_2. Xem sap_nhan_trang_thai() de biet chi tiet.
            reps = [sap_nhan_trang_thai(r) for r in tho]
            n_dao = sum(1 for a, b in zip(tho, reps) if a is not b)
            out[method] = _summarize_method(reps, true_named)
            out[method]["n_replications"] = len(reps)
            out[method]["n_dao_nhan"] = n_dao
            tong += len(reps)
            can = _n_reps(cfg, method)
            trang_thai = "du" if len(reps) >= can else f"THIEU ({len(reps)}/{can})"
            _log(f"  {code}/{method}: {len(reps)} lan lap [{trang_thai}] "
                 f"dao nhan {n_dao}/{len(reps)} "
                 f"mean_brier={out[method]['mean_brier']}")
        if not out:
            _log(f"  {code}: chua co lan lap nao")
            continue
        torch.save(out, cfg.results_dir / f"07_simulation_{code}.pt")
        _log(f"  -> 07_simulation_{code}.pt ({tong} lan lap)")


# ============================================================
# Do thu toc do MCMC truoc khi cam ket chay dai
# ============================================================
def do_thu(cfg, code: str) -> None:
    _log("=" * 70)
    _log(f"DO THU: mot lan lap cua kich ban {code}")
    _log("=" * 70)
    for m in METHODS:
        worker(cfg, code, m, 0)


# ============================================================
def _spawn(tasks: list[list[str]], jobs: int, tag: str,
           script: Path | None = None, args_chung: list[str] | None = None) -> int:
    """Chay `tasks` tren toi da `jobs` tien trinh con, moi tac vu mot tien trinh.

    `script` mac dinh la chinh file nay. Truyen file khac de driver KHAC dung
    lai dung vong dieu phoi nay (PR-B dung chung) -- mot ban dieu phoi thay vi
    hai ban phai sua song song moi khi doi gi.
    `args_chung` la cac tham so dat truoc moi tac vu, vd. ["--preset", "full"].
    """
    script = Path(__file__).resolve() if script is None else Path(script).resolve()
    args_chung = list(args_chung or [])
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    queue, running, failed = list(tasks), [], []
    t0 = time.time()
    xong = 0
    while queue or running:
        while queue and len(running) < jobs:
            extra = queue.pop(0)
            name = "_".join(extra[1:4])
            fh = open(log_dir / f"{tag}_{name}_{stamp}.log", "w", encoding="utf-8",
                      errors="replace")
            proc = subprocess.Popen(
                [sys.executable, str(script)] + args_chung + extra,
                stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            running.append((name, proc, fh))
        time.sleep(10)
        still = []
        for name, proc, fh in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, fh))
                continue
            fh.close()
            xong += 1
            if rc != 0:
                failed.append(name)
                _log(f"  LOI {name} (ma thoat {rc})")
            if xong % 5 == 0 or rc != 0:
                con_lai = len(queue) + len(still)
                tb = (time.time() - t0) / max(xong, 1)
                _log(f"  tien do: {xong} xong, {con_lai} con lai, "
                     f"~{tb/60:.1f} phut/lan lap, con ~{con_lai*tb/max(jobs,1)/3600:.1f} gio")
        running = still
    _log(f"Ket thuc sau {(time.time()-t0)/3600:.2f} gio. Loi: {failed if failed else 'khong'}")
    return len(failed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", default="S1,S2,S3,S4")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="mle,vb_mean_field,vb_full_rank")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--do-thu", dest="do_thu", metavar="KICH_BAN")
    ap.add_argument("--worker", nargs=3, metavar=("KICH_BAN", "PHUONG_PHAP", "B"))
    ap.add_argument("--preset", default="full")
    ap.add_argument("--B", type=int, default=None,
                    help="ghi de so lan lap Monte Carlo cho MOI phuong phap "
                         "(mac dinh lay tu preset: 30 cho MLE, 15 cho VB)")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    # Buoc xac nhan cua thiet ke ba buoc dung B=100 cho CA BA phuong phap, thay
    # vi 30/15 cua buoc tham do. Ghi de o day (thay vi sua PRESETS) de mot lenh
    # duy nhat chon duoc ngan sach, va de `combine` biet dung nguong "du/thieu".
    if args.B is not None:
        cfg = replace(cfg, sim_B_mle=args.B, sim_B_vb=args.B)
    codes = [c.strip() for c in args.scenarios.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if any(m not in METHODS for m in methods):
        raise SystemExit(f"--methods khong hop le: {methods}")

    if args.worker:
        worker(cfg, args.worker[0], args.worker[1], int(args.worker[2]))
        return 0
    if args.do_thu:
        do_thu(cfg, args.do_thu)
        return 0
    if args.combine:
        combine(cfg, codes)
        return 0

    sdir = _sim_dir(cfg)
    tasks = []
    for code in codes:
        for method in methods:
            for b in range(_n_reps(cfg, method)):
                if not (sdir / f"{code}_{method}_{b:03d}.pt").exists():
                    tasks.append(["--worker", code, method, str(b)])
    if not tasks:
        _log("Khong con lan lap nao thieu -- chi can gop.")
        combine(cfg, codes)
        return 0
    _log(f"Se chay {len(tasks)} lan lap tren {args.jobs} tien trinh song song")
    _log(f"  kich ban: {codes}")
    _log(f"  phuong phap: {methods}")
    rc = _spawn(tasks, args.jobs, "sim", args_chung=["--preset", args.preset])
    combine(cfg, codes)
    return rc


if __name__ == "__main__":
    sys.exit(main())
