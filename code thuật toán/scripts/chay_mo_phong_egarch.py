"""Chay nghien cuu khoi phuc tham so cho EGARCH-X Student-t (PR-B).

CACH CHAY
    python scripts/chay_mo_phong_egarch.py                 # day du: 3 kich ban x B=100
    python scripts/chay_mo_phong_egarch.py --nhanh         # B=3, T=400, de thu duong chay
    python scripts/chay_mo_phong_egarch.py --kich-ban E2   # chi chay mot kich ban
    python scripts/chay_mo_phong_egarch.py --combine       # chi gop lai, khong tinh

BA KICH BAN (chi tiet trong msgarchx/simulation/run_egarch.py)
    E1  theta_0 = -0,10   co don bay  -> co phat hien ra khong?
    E2  theta_0 =  0      KHONG don bay -> co BIA RA khong?   <- quan trong nhat
    E3  nu_0 = 4          duoi day     -> con khoi phuc dung khong?

CHECKPOINT THEO TUNG LAN LAP, trong results/mo_phong_egarch/parts/. Ngat giua
chung roi chay lai se bo qua dung nhung lan lap da xong. Ban truoc chi luu
theo KICH BAN, nghia la ngat giua E1 thi mat ca 300 lan uoc luong cua E1.

Chay SONG SONG tren nhieu tien trinh (mac dinh 12). Chay tuan tu mot luong se
mat khoang 145 gio; day khong phai lua chon thuc te.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msgarchx.inference import model_spec                      # noqa: E402
from msgarchx.simulation.dgp import simulate_egarch            # noqa: E402
from msgarchx.simulation.run_egarch import (                   # noqa: E402
    _chay_mot_phuong_phap,
    default_egarch_scenarios,
    named_params_egarch,
)
from msgarchx.simulation.run_simulation import _summarize_method  # noqa: E402
# Dung chung vong dieu phoi song song cua PR-A thay vi viet ban thu hai.
from scripts.chay_mo_phong_song_song import _spawn             # noqa: E402

PHUONG_PHAP = ("mle", "vb_mean_field", "vb_full_rank")


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _cau_hinh(nhanh: bool, B_ghi_de: int | None) -> tuple[int, int, Path, dict | None]:
    """Tra ve (T, B, thu_muc_ket_qua, method_kwargs) cho che do dang chay."""
    T, B = (400, 3) if nhanh else (2000, 100)
    if B_ghi_de is not None:
        B = B_ghi_de
    out_dir = ROOT / "results" / ("mo_phong_egarch_nhanh" if nhanh else "mo_phong_egarch")
    if nhanh:
        mk = {"mle": {"n_starts": 1, "n_adam": 40, "n_lbfgs": 15},
              "vb_mean_field": {"n_steps": 80, "n_posterior_draws": 50},
              "vb_full_rank": {"n_steps": 80, "n_posterior_draws": 50}}
    else:
        mk = None
    return T, B, out_dir, mk


def _parts_dir(out_dir: Path) -> Path:
    d = out_dir / "parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ============================================================
# Worker: MOT lan lap cua MOT phuong phap
# ============================================================
def worker(ma: str, method: str, b: int, nhanh: bool, B_ghi_de: int | None) -> None:
    T, B, out_dir, mk = _cau_hinh(nhanh, B_ghi_de)
    path = _parts_dir(out_dir) / f"{ma}_{method}_{b:03d}.pt"
    if path.exists():
        _log(f"[{ma}/{method}/{b}] da co, bo qua")
        return

    spec = model_spec.get("egarch_x_t")
    scenario = default_egarch_scenarios(T=T, B=B)[ma]
    n_exog = scenario.params.gamma.shape[0]

    # Seed = b, dung cho ca sinh du lieu lan uoc luong -- giong het duong chay
    # tuan tu trong run_egarch_scenario, nen ket qua song song trung khop.
    seed = b
    t0 = time.time()
    sim = simulate_egarch(scenario.params, T=scenario.T, dist=scenario.dist,
                          n_exog=n_exog, burn_in=scenario.burn_in, seed=seed)
    X = sim.X if spec.use_exog else torch.zeros(sim.r.shape[0], 0)
    _log(f"[{ma}/{method}/{b}] bat dau (T={scenario.T}, seed={seed})")
    rep = _chay_mot_phuong_phap(method, sim.r, X, spec, seed,
                                (mk or {}).get(method))

    # Ghi qua file tam roi doi ten: mat dien giua chung chi de lai file .tmp
    # thua, khong de lai mot file .pt cut doi ma lan sau tuong la "da xong".
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"rep": rep, "code": ma, "method": method, "b": b,
                "elapsed": time.time() - t0}, tmp)
    os.replace(tmp, path)
    _log(f"[{ma}/{method}/{b}] xong sau {(time.time()-t0)/60:.1f} phut")


# ============================================================
# Gop -> mo_phong_egarch/<KB>.pt (dung dinh dang cu)
# ============================================================
def combine(codes: list[str], nhanh: bool, B_ghi_de: int | None) -> None:
    T, B, out_dir, _ = _cau_hinh(nhanh, B_ghi_de)
    pdir = _parts_dir(out_dir)
    scenarios = default_egarch_scenarios(T=T, B=B)
    for ma in codes:
        that = named_params_egarch(scenarios[ma].params)
        ra, tong, giay = {}, 0, 0.0
        for method in PHUONG_PHAP:
            files = sorted(pdir.glob(f"{ma}_{method}_*.pt"))
            if not files:
                continue
            reps = []
            for f in files:
                try:
                    o = torch.load(f, weights_only=False)
                except Exception:
                    _log(f"  !! file hong, bo qua: {f.name} (xoa di roi chay lai)")
                    continue
                reps.append(o["rep"])
                giay += o.get("elapsed", 0.0)
            blk = _summarize_method(reps, that)
            blk["n_replications"] = len(reps)
            blk["tham_so_that"] = that
            ra[method] = blk
            tong += len(reps)
            trang_thai = "du" if len(reps) >= B else f"THIEU ({len(reps)}/{B})"
            _log(f"  {ma}/{method}: {len(reps)} lan lap [{trang_thai}]")
        if not ra:
            _log(f"  {ma}: chua co lan lap nao")
            continue
        ra["giay"] = giay
        torch.save(ra, out_dir / f"{ma}.pt")
        _log(f"  -> {ma}.pt ({tong} lan lap)")
        in_ket_qua(ma, scenarios[ma], ra)


def in_ket_qua(ma: str, scenario, ket_qua: dict) -> None:
    that = named_params_egarch(scenario.params)
    print(f"\n  --- {ma}: {scenario.description} ---")
    print(f"      tham so that: theta={that['theta']:+.3f}  nu={that.get('nu', float('nan')):.2f}")
    print(f"      {'phuong phap':<16}{'n':>4}{'bias(theta)':>13}{'rmse(theta)':>13}{'cov(theta)':>12}")
    for pp in PHUONG_PHAP:
        if pp not in ket_qua:
            continue
        b = ket_qua[pp]
        cov = b["coverage"].get("theta")
        cov_txt = f"{cov:.3f}" if cov is not None else "--"
        print(f"      {pp:<16}{b['n_replications']:>4}"
              f"{b['bias']['theta']:>+13.4f}{b['rmse']['theta']:>13.4f}{cov_txt:>12}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nhanh", action="store_true", help="B=3, T=400, cau hinh nho de thu")
    ap.add_argument("--kich-ban", dest="kich_ban", default=None, choices=["E1", "E2", "E3"],
                    help="chi chay mot kich ban")
    ap.add_argument("--B", type=int, default=None, help="ghi de so lan lap Monte Carlo")
    ap.add_argument("--jobs", type=int, default=12, help="so tien trinh song song")
    ap.add_argument("--combine", action="store_true", help="chi gop, khong tinh toan")
    ap.add_argument("--worker", nargs=3, metavar=("KICH_BAN", "PHUONG_PHAP", "B"))
    args = ap.parse_args()

    T, B, out_dir, _ = _cau_hinh(args.nhanh, args.B)
    codes = [args.kich_ban] if args.kich_ban else ["E1", "E2", "E3"]

    if args.worker:
        worker(args.worker[0], args.worker[1], int(args.worker[2]), args.nhanh, args.B)
        return 0
    if args.combine:
        combine(codes, args.nhanh, args.B)
        return 0

    pdir = _parts_dir(out_dir)
    tasks = []
    for ma in codes:
        for method in PHUONG_PHAP:
            for b in range(B):
                if not (pdir / f"{ma}_{method}_{b:03d}.pt").exists():
                    tasks.append(["--worker", ma, method, str(b)])

    print(f"Khoi phuc tham so EGARCH-X Student-t | T={T}, B={B}, "
          f"{len(PHUONG_PHAP)} phuong phap")
    print(f"Ket qua luu vao: {out_dir}")
    if not tasks:
        _log("Khong con lan lap nao thieu -- chi can gop.")
        combine(codes, args.nhanh, args.B)
        return 0

    _log(f"Se chay {len(tasks)} lan lap tren {args.jobs} tien trinh song song")
    _log(f"  kich ban: {codes}")
    args_chung = ["--nhanh"] if args.nhanh else []
    if args.B is not None:
        args_chung += ["--B", str(args.B)]
    rc = _spawn(tasks, args.jobs, "egarch",
                script=Path(__file__).resolve(), args_chung=args_chung)
    combine(codes, args.nhanh, args.B)

    print("\nDOC KET QUA: voi E2 (theta_0 = 0), bias(theta) gan 0 nghia la mo hinh")
    print("KHONG bia ra don bay gia. Neu bias lech khoi 0 mot cach he thong thi")
    print("moi ket luan 've don bay cua VN30F1M' deu phai xem lai.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
