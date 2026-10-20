"""BUOC 2 -- Mo phong khoi phuc tham so cho EGARCH-X Student-t (truong hop E1-E3).

CACH CHAY (tu thu muc "code thuật toán"):
    python scripts/chay_buoc2_mo_phong.py                    # 3 truong hop x 3 phuong phap x B=100
    python scripts/chay_buoc2_mo_phong.py --jobs 8           # so tien trinh song song
    python scripts/chay_buoc2_mo_phong.py --truong-hop E2    # chi mot truong hop
    python scripts/chay_buoc2_mo_phong.py --gop              # chi gop ket qua da co
    python scripts/chay_buoc2_mo_phong.py --nhanh            # T=400, B=3, cau hinh nho de thu

Moi lan lap b: sinh du lieu T=2000 (seed = b) tu EGARCH-X voi tham so that
theta_0, uoc luong bang tung phuong phap, luu RIENG vao
results/mo_phong_egarch/parts/<TH>_<phuong_phap>_<b>.pt. Chay lai se bo qua
lan lap da co. Cuoi cung gop thanh results/mo_phong_egarch/<TH>.pt (Bias,
RMSE, Coverage cho tung tham so). Moi lan lap mat ~20-27 phut tren mot nhan.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from chung import PHUONG_PHAP, RESULTS, chay_song_song, log, luu, nap

from garch_vn30.mo_hinh import danh_muc
from garch_vn30.mo_phong import (chay_mot_lan_lap, simulate_egarch, tham_so_that, tong_hop,
                                 truong_hop_mac_dinh)

CAU_HINH_NHANH = {"mle": {"n_starts": 1, "n_adam": 40, "n_lbfgs": 15},
                  "vb_mean_field": {"n_steps": 80, "n_posterior_draws": 50},
                  "vb_full_rank": {"n_steps": 80, "n_posterior_draws": 50}}


def _cau_hinh(nhanh: bool, B: int | None):
    T, B_md = (400, 3) if nhanh else (2000, 100)
    thu_muc = RESULTS / ("mo_phong_egarch_nhanh" if nhanh else "mo_phong_egarch")
    return T, (B if B is not None else B_md), thu_muc, (CAU_HINH_NHANH if nhanh else {})


def worker(ma: str, method: str, b: int, nhanh: bool, B: int | None) -> None:
    """Mot lan lap: sinh du lieu seed = b, uoc luong bang `method`, luu ket qua."""
    T, B, thu_muc, kw = _cau_hinh(nhanh, B)
    path = thu_muc / "parts" / f"{ma}_{method}_{b:03d}.pt"
    if path.exists():
        return
    th = truong_hop_mac_dinh(T=T, B=B)[ma]
    spec = danh_muc.get("egarch_x_t")
    t0 = time.time()
    sim = simulate_egarch(th.params, T=th.T, dist=th.dist, burn_in=th.burn_in, seed=b)
    ket_qua = chay_mot_lan_lap(method, sim.r, sim.X, spec, seed=b, kwargs=kw.get(method))
    luu({"code": ma, "method": method, "b": b, "giay": time.time() - t0, **ket_qua}, path)
    log(f"[{ma}/{method}/{b}] xong sau {(time.time() - t0) / 60:.1f} phut")


def gop(codes: list[str], nhanh: bool, B: int | None) -> None:
    """Gop cac lan lap da co thanh results/mo_phong_egarch/<TH>.pt va in tom tat theta."""
    T, B, thu_muc, _ = _cau_hinh(nhanh, B)
    for ma in codes:
        that = tham_so_that(truong_hop_mac_dinh(T=T, B=B)[ma].params)
        ra, giay = {}, 0.0
        for method in PHUONG_PHAP:
            files = sorted((thu_muc / "parts").glob(f"{ma}_{method}_*.pt"))
            if not files:
                continue
            lan_lap = [nap(f) for f in files]
            giay += sum(x["giay"] for x in lan_lap)
            ra[method] = tong_hop(lan_lap, that)
            thieu = "" if len(lan_lap) >= B else f"  (THIEU: {len(lan_lap)}/{B})"
            m = ra[method]
            cov = m["coverage"].get("theta")
            log(f"  {ma}/{method:<14} n={m['n_replications']:>3}  bias(theta)={m['bias']['theta']:+.4f}"
                f"  rmse(theta)={m['rmse']['theta']:.4f}"
                f"  cov(theta)={'--' if cov is None else f'{cov:.2f}'}{thieu}")
        if ra:
            ra["giay"] = giay
            luu(ra, thu_muc / f"{ma}.pt")


def main() -> int:
    ap = argparse.ArgumentParser(description="Buoc 2: mo phong khoi phuc tham so EGARCH-X")
    ap.add_argument("--nhanh", action="store_true")
    ap.add_argument("--truong-hop", choices=["E1", "E2", "E3"], default=None)
    ap.add_argument("--B", type=int, default=None, help="ghi de so lan lap Monte Carlo")
    ap.add_argument("--jobs", type=int, default=8, help="so tien trinh song song")
    ap.add_argument("--gop", action="store_true", help="chi gop, khong tinh")
    ap.add_argument("--worker", nargs=3, metavar=("TH", "PHUONG_PHAP", "B"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], args.worker[1], int(args.worker[2]), args.nhanh, args.B)
        return 0
    codes = [args.truong_hop] if args.truong_hop else ["E1", "E2", "E3"]
    T, B, thu_muc, _ = _cau_hinh(args.nhanh, args.B)
    if not args.gop:
        tasks = [["--worker", ma, m, str(b)] for ma in codes for m in PHUONG_PHAP for b in range(B)
                 if not (thu_muc / "parts" / f"{ma}_{m}_{b:03d}.pt").exists()]
        log(f"T={T}, B={B}: con {len(tasks)} lan lap, {args.jobs} tien trinh -> {thu_muc}")
        if tasks:
            chung = (["--nhanh"] if args.nhanh else []) + (["--B", str(args.B)] if args.B else [])
            chay_song_song(tasks, args.jobs, Path(__file__).resolve(), "mo_phong", chung)
    gop(codes, args.nhanh, args.B)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
