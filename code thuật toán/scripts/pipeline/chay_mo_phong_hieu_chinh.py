"""Chay nghien cuu "do thien vi hieu chinh" (calibration bias) cho MS-GARCH-X
Student-t -- xem msgarchx/simulation/calibration_bias.py de biet cau hoi
kiem dinh va thiet ke luoi 3x3.

theta_0 CUA BA KICH BAN LA NGHIEM THAT tren VN30F1M, lay tu
results/02_mle.pt (MLE) va results/03_vb.pt (VB mean-field, VB full-rank) --
KHONG phai tham so minh hoa nhu S1-S8. Neu hai file nay chua co (vd. tren may
chua chay Buoc 1), script se bao loi ro rang thay vi tu bia so.

CACH CHAY
    python scripts/pipeline/chay_mo_phong_hieu_chinh.py --nhanh
        Smoke test: T=200, B=2, sieu tham so rut gon -- kiem tra day du
        duong chay (nap theta that -> dung kich ban -> chay ca 3 phuong
        phap) khong loi, KHONG dung de bao cao ket qua.

    python scripts/pipeline/chay_mo_phong_hieu_chinh.py --jobs 8
        Chay THAT: T=2000, B=100, ca 3 nguon x 3 phuong phap = 900 lan uoc
        luong, song song tren 8 tien trinh. Moi lan lap MOT checkpoint rieng
        trong results/parts/hieu_chinh/ -- ngat giua chung chay lai se tu bo
        qua phan da xong.

    python scripts/pipeline/chay_mo_phong_hieu_chinh.py --combine
        Gop checkpoint thanh results/08_hieu_chinh_CAL_<nguon>.pt va in bang
        chan doan 3x3 (p11, nu).

    python scripts/pipeline/chay_mo_phong_hieu_chinh.py --bang
        Chi in bang 3x3 tu cac file 08_hieu_chinh_* da co, khong tinh gi moi.
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

import torch  # noqa: E402

from msgarchx.inference.param_spec import ModelConfig  # noqa: E402
from msgarchx.models.ms_garch import MSGARCHXParams  # noqa: E402
from msgarchx.simulation.calibration_bias import (  # noqa: E402
    CAC_NGUON,
    CALIBRATION_CONFIG,
    TEN_NGUON,
    build_calibration_scenarios,
    ma_kich_ban,
)
from msgarchx.simulation.dgp import simulate_ms_garch_x  # noqa: E402
from msgarchx.simulation.run_simulation import (  # noqa: E402
    _apply_misspecification,
    _named_params,
    _run_one_method,
    _summarize_method,
    params_from_summary_mean,
)

METHODS = ("mle", "vb_mean_field", "vb_full_rank")

# Giong het sieu tham so da dung cho S1-S8 / S1-S4 (chay_mo_phong_song_song.py
# preset "full") -- dong bo voi PR-A/PR-B nhu nguoi dung da chot.
METHOD_KWARGS_FULL = {
    "mle": {"n_starts": 3, "n_adam": 200, "n_lbfgs": 100},
    "vb_mean_field": {"n_steps": 800, "n_posterior_draws": 500},
    "vb_full_rank": {"n_steps": 800, "n_posterior_draws": 500},
}
METHOD_KWARGS_NHANH = {
    "mle": {"n_starts": 1, "n_adam": 20, "n_lbfgs": 10},
    "vb_mean_field": {"n_steps": 40, "n_posterior_draws": 50},
    "vb_full_rank": {"n_steps": 40, "n_posterior_draws": 50},
}


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def nap_diem_hieu_chinh(root: Path = ROOT) -> dict[str, MSGARCHXParams]:
    """Nap theta_0 THAT cua ca 3 phuong phap tu results/02_mle.pt, 03_vb.pt.

    Loi ro rang neu file chua co -- KHONG tu bia so lieu thay the.
    """
    mle_path = root / "results" / "02_mle.pt"
    vb_path = root / "results" / "03_vb.pt"
    if not mle_path.exists() or not vb_path.exists():
        raise FileNotFoundError(
            f"Can ca {mle_path.name} va {vb_path.name} trong {root / 'results'} "
            "de lay theta_0 THAT cua MS-GARCH-X Student-t (Buoc 1 chua chay xong?)."
        )
    mle_d = torch.load(mle_path, weights_only=False)
    vb_d = torch.load(vb_path, weights_only=False)
    return {
        "mle": mle_d["m1"].params,
        "vb_mean_field": params_from_summary_mean(vb_d["m2"].summary, CALIBRATION_CONFIG),
        "vb_full_rank": params_from_summary_mean(vb_d["m3"].summary, CALIBRATION_CONFIG),
    }


def _part_dir(root: Path) -> Path:
    d = root / "results" / "parts" / "hieu_chinh"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Worker: MOT lan lap (mot nguon hieu chinh x mot phuong phap x mot chi so b)
# ============================================================
def worker(root: Path, scenarios: dict, nguon: str, method: str, b: int,
           method_kwargs: dict) -> None:
    ma = ma_kich_ban(nguon)
    path = _part_dir(root) / f"{ma}_{method}_{b:03d}.pt"
    if path.exists():
        _log(f"[{ma}/{method}/{b}] da co, bo qua")
        return
    scn = scenarios[ma]
    seed = b  # base_seed=0, giong quy uoc cua run_scenario/chay_mo_phong_song_song
    t0 = time.time()
    sim = simulate_ms_garch_x(scn.params, scn.P0, scn.T, n_exog=CALIBRATION_CONFIG.n_exog,
                              burn_in=scn.burn_in, seed=seed)
    r = _apply_misspecification(sim.r, scn.misspecified)
    _log(f"[{ma}/{method}/{b}] bat dau (T={scn.T}, seed={seed})")
    rep = _run_one_method(method, r, sim.X, CALIBRATION_CONFIG, sim.state, seed=seed,
                          method_kwargs=method_kwargs.get(method))
    torch.save({"rep": rep, "nguon": nguon, "method": method, "b": b,
                "elapsed": time.time() - t0}, path)
    br = f"brier={rep.brier:.4f}" if rep.brier is not None else "brier=--"
    _log(f"[{ma}/{method}/{b}] xong sau {(time.time()-t0)/60:.1f} phut ({br})")


# ============================================================
# Gop -> results/08_hieu_chinh_<ma>.pt
# ============================================================
def combine(root: Path, scenarios: dict, nguon_list: list[str], B: int) -> dict:
    pdir = _part_dir(root)
    ket_qua = {}
    for nguon in nguon_list:
        ma = ma_kich_ban(nguon)
        scn = scenarios[ma]
        true_named = _named_params(scn.params)
        true_named["p11"] = float(scn.P0[0, 0])
        true_named["p22"] = float(scn.P0[1, 1])
        out, tong = {}, 0
        for method in METHODS:
            files = sorted(pdir.glob(f"{ma}_{method}_*.pt"))
            if not files:
                continue
            reps = [torch.load(f, weights_only=False)["rep"] for f in files]
            out[method] = _summarize_method(reps, true_named)
            out[method]["n_replications"] = len(reps)
            tong += len(reps)
            trang_thai = "du" if len(reps) >= B else f"THIEU ({len(reps)}/{B})"
            _log(f"  {ma}/{method}: {len(reps)} lan lap [{trang_thai}] "
                 f"mean_brier={out[method]['mean_brier']}")
        if not out:
            _log(f"  {ma}: chua co lan lap nao")
            continue
        out["theta0"] = true_named
        torch.save(out, root / "results" / f"08_hieu_chinh_{ma}.pt")
        _log(f"  -> 08_hieu_chinh_{ma}.pt ({tong} lan lap)")
        ket_qua[ma] = out
    return ket_qua


# ============================================================
# Bang chan doan 3x3: doc theo hang, duong cheo co "thang" cac o con lai
# ============================================================
def in_bang_chan_doan(root: Path, nguon_list: list[str]) -> None:
    print("\n" + "=" * 78)
    print("BANG CHAN DOAN 3x3 -- do thien vi hieu chinh (MS-GARCH-X Student-t)")
    print("Hang = nguon theta_0 that; Cot = phuong phap uoc luong lai.")
    print("O co dau * la O DUONG CHEO (phuong phap TRUNG nguon hieu chinh).")
    print("=" * 78)
    for nguon in nguon_list:
        ma = ma_kich_ban(nguon)
        path = root / "results" / f"08_hieu_chinh_{ma}.pt"
        if not path.exists():
            print(f"\n[{ma}] chua co ket qua ({path.name} khong ton tai)")
            continue
        d = torch.load(path, weights_only=False)
        print(f"\nHang: theta_0 = {TEN_NGUON[nguon]}")
        print(f"  {'phuong phap':<24}{'n':>4}{'bias(p11)':>12}{'rmse(p11)':>12}"
              f"{'bias(nu)':>11}{'rmse(nu)':>11}")
        for method in METHODS:
            if method not in d:
                continue
            b = d[method]
            danh_dau = "*" if method == nguon else " "
            print(f"  {danh_dau}{TEN_NGUON[method]:<23}{b['n_replications']:>4}"
                  f"{b['bias'].get('p11', float('nan')):>+12.4f}"
                  f"{b['rmse'].get('p11', float('nan')):>12.4f}"
                  f"{b['bias'].get('nu', float('nan')):>+11.4f}"
                  f"{b['rmse'].get('nu', float('nan')):>11.4f}")
    print("\nDOC KET QUA: neu o co dau (*) luon co |bias|/RMSE thap hon cac o")
    print("khac CUNG HANG, do la bang chung "+"\"chon theta_0 = nghiem cua mot"
          " phuong phap khien chinh phuong phap do co ve thang\" -- xac nhan")
    print("moi lo circularity. Neu khong co mo hinh ro rang, moi lo do khong")
    print("duoc xac nhan boi so lieu nay.")


# ============================================================
def _spawn(tasks: list[list[str]], jobs: int) -> int:
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
            fh = open(log_dir / f"hieuchinh_{name}_{stamp}.log", "w", encoding="utf-8",
                      errors="replace")
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve())] + extra,
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
    ap.add_argument("--nhanh", action="store_true", help="T=200,B=2, sieu tham so rut gon, chi de smoke test")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--bang", action="store_true", help="chi in bang chan doan tu ket qua da co")
    ap.add_argument("--nguon", default=",".join(CAC_NGUON))
    ap.add_argument("--worker", nargs=3, metavar=("NGUON", "PHUONG_PHAP", "B"))
    args = ap.parse_args()

    nguon_list = [n.strip() for n in args.nguon.split(",") if n.strip()]
    if any(n not in CAC_NGUON for n in nguon_list):
        raise SystemExit(f"--nguon khong hop le: {nguon_list}. Hop le: {CAC_NGUON}")

    if args.bang:
        in_bang_chan_doan(ROOT, nguon_list)
        return 0

    T, B = (200, 2) if args.nhanh else (2000, 100)
    method_kwargs = METHOD_KWARGS_NHANH if args.nhanh else METHOD_KWARGS_FULL

    theta_that = nap_diem_hieu_chinh(ROOT)
    scenarios = build_calibration_scenarios(theta_that, T=T, B=B)

    if args.worker:
        worker(ROOT, scenarios, args.worker[0], args.worker[1], int(args.worker[2]), method_kwargs)
        return 0

    if args.combine:
        combine(ROOT, scenarios, nguon_list, B)
        in_bang_chan_doan(ROOT, nguon_list)
        return 0

    if args.nhanh:
        _log("SMOKE TEST --nhanh: chay 1 lan lap cho ca 3 nguon x 3 phuong phap")
        for nguon in nguon_list:
            for method in METHODS:
                worker(ROOT, scenarios, nguon, method, 0, method_kwargs)
        combine(ROOT, scenarios, nguon_list, B)
        in_bang_chan_doan(ROOT, nguon_list)
        return 0

    pdir = _part_dir(ROOT)
    tasks = []
    for nguon in nguon_list:
        ma = ma_kich_ban(nguon)
        for method in METHODS:
            for b in range(B):
                if not (pdir / f"{ma}_{method}_{b:03d}.pt").exists():
                    tasks.append(["--worker", nguon, method, str(b)])
    if not tasks:
        _log("Khong con lan lap nao thieu -- chi can gop.")
        combine(ROOT, scenarios, nguon_list, B)
        in_bang_chan_doan(ROOT, nguon_list)
        return 0
    _log(f"Se chay {len(tasks)} lan lap tren {args.jobs} tien trinh song song "
         f"(T={T}, B={B}, nguon={nguon_list})")
    rc = _spawn(tasks, args.jobs)
    combine(ROOT, scenarios, nguon_list, B)
    in_bang_chan_doan(ROOT, nguon_list)
    return rc


if __name__ == "__main__":
    sys.exit(main())
