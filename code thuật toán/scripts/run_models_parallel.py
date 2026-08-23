"""Chay TUNG mo hinh MLE/VB rieng le trong 8 mo hinh cua pipeline chinh
(B1, B2, B3, B4, M1, M2, M3, R1), moi mo hinh mot checkpoint rieng, va co the
chay tat ca SONG SONG tren nhieu tien trinh.

VI SAO CO SCRIPT NAY (khac voi run_full_study.py):

  1. Tam mo hinh nay DOC LAP HOAN TOAN voi nhau -- khong mo hinh nao can ket
     qua cua mo hinh nao. (run_vb_step chi nhan `config_m1` tu buoc MLE, ma
     do chi la ModelConfig(n_exog=k), khong phai ket qua uoc luong.) Vi moi
     tien trinh chi dung ~1 nhan CPU (vong lap tuan tu theo T), chay ca 8 song
     song rut thoi gian tuong tu ~11.7 gio xuong con ~2.5 gio.

  2. `run_full_study.py` chi ghi `02_mle.pt` SAU KHI ca 5 mo hinh MLE xong
     (khoi ~6 gio) va `03_vb.pt` sau khi ca M2 lan M3 xong (khoi ~3.4 gio).
     Mat dien giua chung la mat toan bo khoi do. O day moi mo hinh ghi
     `results/parts/<ten>.pt` NGAY khi chay xong, nen chi mat mo hinh dang
     chay do.

CAU HINH VA SEED lay nguyen tu PRESETS["full"] cua run_full_study.py va goi
dung cac ham fit_* voi dung tham so nhu run_mle_step/run_vb_step/run_r1_step,
nen ket qua khong khac gi chay bang run_full_study.py.

Sau khi cac phan chay xong, `--merge` gop chung lai thanh dung ba file ma
buoc `oos` va `build_html_report.py` doc: `02_mle.pt`, `03_vb.pt`, `05_r1.pt`
(cong them `09_timing.pt`). Tu do tro di, chay tiep MCMC va OOS bang
run_full_study.py nhu binh thuong:

    python scripts/run_full_study.py --preset full --only mcmc
    python scripts/run_full_study.py --preset full --only oos simulation

Chay:
    python scripts/run_models_parallel.py --list
    python scripts/run_models_parallel.py --launch-all          # 8 tien trinh + tu gop
    python scripts/run_models_parallel.py --launch-all --jobs 4
    python scripts/run_models_parallel.py --model m1            # chay rieng 1 mo hinh
    python scripts/run_models_parallel.py --merge               # gop thu cong
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from msgarchx.inference.mle import fit_mle, natural_scale_standard_errors
from msgarchx.inference.garch_mle import fit_garch_mle
from msgarchx.inference.param_spec import ModelConfig
from msgarchx.inference.variational import fit_vb

# Dung lai nguyen cau hinh, ham chuan bi du lieu va thu tu cot cua pipeline
# chinh -- KHONG dinh nghia lai, de khong the lech nhau.
from scripts.run_full_study import PRESETS, prepare_data  # noqa: E402

MLE_PARTS = ["b1", "b2", "b3", "b4", "m1"]
VB_PARTS = ["m2", "m3"]
R1_PARTS = ["r1"]
ALL_PARTS = MLE_PARTS + VB_PARTS + R1_PARTS

LABELS = {
    "b1": "B1: GARCH(1,1) chuan, khong X            (1 trang thai, MLE)",
    "b2": "B2: GARCH(1,1) Student-t, khong X        (1 trang thai, MLE)",
    "b3": "B3: GARCH-X Student-t                    (1 trang thai, MLE)",
    "b4": "B4: MS-GARCH chuan (nu co dinh), khong X (2 trang thai, MLE)",
    "m1": "M1: MS-GARCH-X Student-t + sai so chuan  (2 trang thai, MLE)",
    "m2": "M2: MS-GARCH-X Student-t, VB mean-field",
    "m3": "M3: MS-GARCH-X Student-t, VB full-rank",
    "r1": "R1: MS-GARCH-X, nu_s theo trang thai, VB mean-field",
}

# Thu tu khoi chay: mo hinh lau nhat truoc, de khi so tien trinh it hon so mo
# hinh thi cac mo hinh nang khong bi don xuong cuoi hang doi.
LAUNCH_ORDER = ["m1", "b4", "r1", "m2", "m3", "b3", "b2", "b1"]


def parts_dir(results_dir: Path) -> Path:
    return results_dir / "parts"


# ============================================================
# Chay mot mo hinh
# ============================================================
def fit_one(name: str, cfg, r_train: torch.Tensor, X_train: torch.Tensor) -> dict:
    """Tra ve dict noi dung se luu vao parts/<name>.pt (khoa 'result' cua no
    duoc gop thang vao checkpoint chinh o buoc --merge)."""
    zero_X = torch.zeros(r_train.shape[0], 0, dtype=r_train.dtype)
    mle_kw = dict(n_starts=cfg.mle_n_starts, n_adam=cfg.mle_n_adam,
                  n_lbfgs=cfg.mle_n_lbfgs, base_seed=cfg.seed, verbose=True)
    config_m1 = ModelConfig(n_exog=X_train.shape[1], state_specific_nu=False)

    if name == "b1":
        return {"result": fit_garch_mle(r_train, zero_X, dist="normal", **mle_kw)}
    if name == "b2":
        return {"result": fit_garch_mle(r_train, zero_X, dist="student_t", **mle_kw)}
    if name == "b3":
        return {"result": fit_garch_mle(r_train, X_train, dist="student_t", **mle_kw)}
    if name == "b4":
        config_b4 = ModelConfig(n_exog=0, state_specific_nu=False)
        return {"result": fit_mle(r_train, zero_X, config_b4, fixed_nu=1e6, **mle_kw)}
    if name == "m1":
        m1 = fit_mle(r_train, X_train, config_m1, **mle_kw)
        print("     Tinh sai so chuan (delta method tu Hessian) cho M1 ...")
        m1_se = natural_scale_standard_errors(m1.raw, config_m1, r_train, X_train)
        if m1_se is None:
            print("     Hessian khong xac dinh duong tai nghiem MLE cua M1 -- bo qua SE.")
        return {"result": m1, "m1_se": m1_se}
    if name in ("m2", "m3"):
        guide = "mean_field" if name == "m2" else "full_rank"
        return {"result": fit_vb(r_train, X_train, config_m1, guide_type=guide,
                                 n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws,
                                 seed=cfg.seed, verbose=True)}
    if name == "r1":
        config_r1 = ModelConfig(n_exog=X_train.shape[1], state_specific_nu=True)
        r1 = fit_vb(r_train, X_train, config_r1, guide_type="mean_field",
                    n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws,
                    seed=cfg.seed, verbose=True)
        nu_samples = r1.samples["nu_minus_2"] + 2.0
        p_tail = float((nu_samples[:, 1] < nu_samples[:, 0]).float().mean())
        print(f"     P(nu_2 < nu_1 | Data) ~= {p_tail:.4f} (eq:tail-probability)")
        return {"result": r1, "p_tail": p_tail}
    raise ValueError(f"ten mo hinh khong hop le: {name!r}")


def run_model(name: str, cfg, force: bool = False) -> None:
    pdir = parts_dir(cfg.results_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / f"{name}.pt"
    if path.exists() and not force:
        print(f"[bo qua] {name} da co tai {path} (dung --force de chay lai)")
        return

    data = load_data(cfg)
    r_train = data["r_all"][: data["n_train"]]
    X_train = data["X_all"][: data["n_train"]]

    print(f"=== {name.upper()} | {LABELS[name]} ===")
    print(f"Bat dau: {datetime.now():%Y-%m-%d %H:%M:%S} | T_train = {r_train.shape[0]}")
    t0 = time.time()
    payload = fit_one(name, cfg, r_train, X_train)
    elapsed = time.time() - t0
    payload.update({"model": name, "elapsed": elapsed,
                    "finished_at": datetime.now().isoformat(timespec="seconds")})
    torch.save(payload, path)
    print(f"=== XONG {name.upper()} sau {elapsed/60:.1f} phut -> {path} ===")


def load_data(cfg) -> dict:
    """Nap 00_data.pt; KHONG tu tinh khi thieu, de nhieu tien trinh song song
    khong cung luc doc CSV va ghi de len nhau. Dung --prepare-data truoc."""
    path = cfg.results_dir / "00_data.pt"
    if not path.exists():
        raise SystemExit(
            f"[LOI] Chua co {path}. Chay truoc mot lan:\n"
            f"      python scripts/run_models_parallel.py --prepare-data"
        )
    return torch.load(path, weights_only=False)


# ============================================================
# Gop cac phan thanh checkpoint chuan cua pipeline
# ============================================================
def merge(cfg) -> None:
    pdir = parts_dir(cfg.results_dir)
    have = {n: (pdir / f"{n}.pt") for n in ALL_PARTS}
    loaded = {n: torch.load(p, weights_only=False) for n, p in have.items() if p.exists()}
    data = load_data(cfg)
    n_exog = data["X_all"].shape[1]

    timing_path = cfg.results_dir / "09_timing.pt"
    timing = torch.load(timing_path, weights_only=False) if timing_path.exists() else {}
    for n, pay in loaded.items():
        timing[f"model_{n}"] = pay["elapsed"]

    if all(n in loaded for n in MLE_PARTS):
        out = {n: loaded[n]["result"] for n in MLE_PARTS}
        out["m1_se"] = loaded["m1"].get("m1_se")
        out["config_m1"] = ModelConfig(n_exog=n_exog, state_specific_nu=False)
        torch.save(out, cfg.results_dir / "02_mle.pt")
        timing["02_mle"] = sum(loaded[n]["elapsed"] for n in MLE_PARTS)
        print(f"[gop] 02_mle.pt  <- {', '.join(MLE_PARTS)}")
    else:
        print(f"[cho] 02_mle.pt  thieu: {[n for n in MLE_PARTS if n not in loaded]}")

    if all(n in loaded for n in VB_PARTS):
        torch.save({n: loaded[n]["result"] for n in VB_PARTS}, cfg.results_dir / "03_vb.pt")
        timing["03_vb"] = sum(loaded[n]["elapsed"] for n in VB_PARTS)
        print(f"[gop] 03_vb.pt   <- {', '.join(VB_PARTS)}")
    else:
        print(f"[cho] 03_vb.pt   thieu: {[n for n in VB_PARTS if n not in loaded]}")

    if "r1" in loaded:
        torch.save({"r1": loaded["r1"]["result"],
                    "config_r1": ModelConfig(n_exog=n_exog, state_specific_nu=True),
                    "p_tail": loaded["r1"]["p_tail"]}, cfg.results_dir / "05_r1.pt")
        timing["05_r1"] = loaded["r1"]["elapsed"]
        print("[gop] 05_r1.pt   <- r1")
    else:
        print("[cho] 05_r1.pt   thieu: ['r1']")

    torch.save(timing, timing_path)
    print(f"[gop] 09_timing.pt cap nhat ({len(timing)} muc)")


# ============================================================
# Khoi chay song song
# ============================================================
def launch_all(cfg, jobs: int, force: bool, preset: str,
               only: list[str] | None = None) -> int:
    pdir = parts_dir(cfg.results_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    candidates = only if only else LAUNCH_ORDER
    todo = [n for n in LAUNCH_ORDER if n in candidates
            and (force or not (pdir / f"{n}.pt").exists())]
    if not todo:
        print("Tat ca cac phan da xong -- chi can gop.")
        merge(cfg)
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ)
    # Moi tien trinh chi lay 1 luong BLAS: vong lap tuan tu nay khong huong loi
    # tu da luong (xem Muc 13.1 README.md), nhung 8 tien trinh x 8 luong se
    # gianh nhau CPU va lam cham TAT CA.
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1"})

    print(f"Khoi chay {len(todo)} mo hinh, toi da {jobs} tien trinh song song:")
    for n in todo:
        print(f"   - {n}: {LABELS[n]}")
    print(f"Nhat ky tung mo hinh: {log_dir}\\model_<ten>_{stamp}.log\n")

    running: list[tuple[str, subprocess.Popen, object]] = []
    queue = list(todo)
    failed: list[str] = []
    t_start = time.time()

    while queue or running:
        while queue and len(running) < jobs:
            name = queue.pop(0)
            log_path = log_dir / f"model_{name}_{stamp}.log"
            fh = open(log_path, "w", encoding="utf-8", errors="replace")
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--model", name, "--preset", preset,
                   "--results-dir", str(cfg.results_dir)]
            if force:
                cmd.append("--force")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(ROOT))
            running.append((name, proc, fh))
            print(f"[{datetime.now():%H:%M:%S}] bat dau {name} (pid {proc.pid})")

        time.sleep(5)
        still = []
        for name, proc, fh in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, fh))
                continue
            fh.close()
            mins = (time.time() - t_start) / 60
            if rc == 0:
                print(f"[{datetime.now():%H:%M:%S}] XONG {name} (sau {mins:.0f} phut tu luc bat dau)")
            else:
                failed.append(name)
                print(f"[{datetime.now():%H:%M:%S}] LOI  {name} (ma thoat {rc}) "
                      f"-- xem logs/model_{name}_{stamp}.log")
        running = still

    print(f"\nTat ca ket thuc sau {(time.time()-t_start)/60:.1f} phut.")
    if failed:
        print(f"Cac mo hinh LOI: {failed}")
    merge(cfg)
    return 1 if failed else 0


def show_list(cfg) -> None:
    pdir = parts_dir(cfg.results_dir)
    print(f"Thu muc ket qua: {cfg.results_dir}\n")
    for n in ALL_PARTS:
        p = pdir / f"{n}.pt"
        if p.exists():
            pay = torch.load(p, weights_only=False)
            print(f"  {n:<3} [XONG {pay['elapsed']/60:6.1f} phut, {pay.get('finished_at','?')}]  {LABELS[n]}")
        else:
            print(f"  {n:<3} [chua chay              ]  {LABELS[n]}")
    print()
    for f in ("00_data.pt", "02_mle.pt", "03_vb.pt", "05_r1.pt"):
        print(f"  {f:<14} {'co' if (cfg.results_dir / f).exists() else 'chua co'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=ALL_PARTS, help="Chay dung mot mo hinh")
    ap.add_argument("--launch-all", action="store_true", help="Chay tat ca phan con thieu, song song")
    ap.add_argument("--merge", action="store_true", help="Gop cac phan da co thanh 02_mle/03_vb/05_r1")
    ap.add_argument("--prepare-data", action="store_true", help="Tao 00_data.pt (chay mot lan truoc)")
    ap.add_argument("--list", action="store_true", help="Xem tinh trang tung phan")
    ap.add_argument("--jobs", type=int, default=8, help="So tien trinh song song toi da (mac dinh 8)")
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--force", action="store_true", help="Chay lai du da co checkpoint")
    ap.add_argument("--models", default=None,
                    help="Chi khoi chay cac mo hinh nay, cach nhau bang dau phay (vd: m1,b1,b2)")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    if args.results_dir:
        from dataclasses import replace
        cfg = replace(cfg, results_dir=Path(args.results_dir))

    if args.prepare_data:
        path = cfg.results_dir / "00_data.pt"
        if path.exists():
            print(f"[bo qua] da co {path}")
        else:
            cfg.results_dir.mkdir(parents=True, exist_ok=True)
            torch.save(prepare_data(cfg), path)
            print(f"[xong] {path}")
        return 0
    if args.list:
        show_list(cfg)
        return 0
    if args.merge:
        merge(cfg)
        return 0
    if args.model:
        run_model(args.model, cfg, force=args.force)
        return 0
    if args.launch_all:
        only = [m.strip() for m in args.models.split(",")] if args.models else None
        if only and any(m not in ALL_PARTS for m in only):
            raise SystemExit(f"--models chua ten khong hop le: {only}")
        return launch_all(cfg, jobs=args.jobs, force=args.force,
                          preset=args.preset, only=only)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
