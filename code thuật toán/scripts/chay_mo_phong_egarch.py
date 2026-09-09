"""Chay nghien cuu khoi phuc tham so cho EGARCH-X Student-t (PR-B).

CACH CHAY
    python scripts/chay_mo_phong_egarch.py                 # day du: 3 kich ban x B=100
    python scripts/chay_mo_phong_egarch.py --nhanh         # B=3, T=400, de thu duong chay
    python scripts/chay_mo_phong_egarch.py --kich-ban E2   # chi chay mot kich ban

BA KICH BAN (chi tiet trong msgarchx/simulation/run_egarch.py)
    E1  theta_0 = -0,10   co don bay  -> co phat hien ra khong?
    E2  theta_0 =  0      KHONG don bay -> co BIA RA khong?   <- quan trong nhat
    E3  nu_0 = 4          duoi day     -> con khoi phuc dung khong?

MOI KICH BAN CHECKPOINT RIENG trong results/mo_phong_egarch/. Ngat giua chung
roi chay lai se tu bo qua kich ban da xong.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msgarchx.inference import model_spec                      # noqa: E402
from msgarchx.simulation.run_egarch import (                   # noqa: E402
    default_egarch_scenarios,
    run_egarch_scenario,
)

PHUONG_PHAP = ("mle", "vb_mean_field", "vb_full_rank")


def in_ket_qua(ma: str, scenario, ket_qua: dict) -> None:
    that = ket_qua[next(iter(ket_qua))]["tham_so_that"]
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nhanh", action="store_true", help="B=3, T=400, cau hinh nho de thu")
    ap.add_argument("--kich-ban", default=None, choices=["E1", "E2", "E3"],
                    help="chi chay mot kich ban")
    ap.add_argument("--B", type=int, default=None, help="ghi de so lan lap Monte Carlo")
    args = ap.parse_args()

    T, B = (400, 3) if args.nhanh else (2000, 100)
    if args.B is not None:
        B = args.B
    out_dir = ROOT / "results" / ("mo_phong_egarch_nhanh" if args.nhanh else "mo_phong_egarch")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = model_spec.get("egarch_x_t")
    scenarios = default_egarch_scenarios(T=T, B=B)
    can_chay = [args.kich_ban] if args.kich_ban else list(scenarios)

    print(f"Khoi phuc tham so EGARCH-X Student-t | T={T}, B={B}, "
          f"{len(PHUONG_PHAP)} phuong phap")
    print(f"Ket qua luu vao: {out_dir}")

    if args.nhanh:
        method_kwargs = {"mle": {"n_starts": 1, "n_adam": 40, "n_lbfgs": 15},
                         "vb_mean_field": {"n_steps": 80, "n_posterior_draws": 50},
                         "vb_full_rank": {"n_steps": 80, "n_posterior_draws": 50}}
    else:
        method_kwargs = None

    for ma in can_chay:
        path = out_dir / f"{ma}.pt"
        if path.exists():
            print(f"\n  [bo qua] {ma} da co {path.name}")
            ket_qua = torch.load(path, weights_only=False)
        else:
            print(f"\n  [chay] {ma} ...")
            t0 = time.time()
            ket_qua = run_egarch_scenario(
                scenarios[ma], spec, methods=PHUONG_PHAP,
                base_seed=0, method_kwargs=method_kwargs,
            )
            ket_qua["giay"] = time.time() - t0
            torch.save(ket_qua, path)
            print(f"      xong sau {ket_qua['giay']/60:.1f} phut -> {path.name}")
        in_ket_qua(ma, scenarios[ma], ket_qua)

    print("\nDOC KET QUA: voi E2 (theta_0 = 0), bias(theta) gan 0 nghia la mo hinh")
    print("KHONG bia ra don bay gia. Neu bias lech khoi 0 mot cach he thong thi")
    print("moi ket luan 've don bay cua VN30F1M' deu phai xem lai.")


if __name__ == "__main__":
    main()
