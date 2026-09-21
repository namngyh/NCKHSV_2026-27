"""Bo sung HAI dai luong con thieu cho nghien cuu khoi phuc tham so, KHONG uoc luong lai.

    1. COVERAGE CHO MLE (ca PR-A lan PR-B). Quy trinh goc khong sinh khoang tin
       cay cho MLE, nen bang so sanh thien vi VB o cot coverage. Nay dung sai so
       chuan tu Hessian quan sat tai nghiem DA LUU: khoang Wald theta_hat +/-
       1,96 SE tren thang tu nhien.
         - PR-A (MS-GARCH-X): natural_scale_standard_errors() -- da co, chua goi.
         - PR-B (EGARCH-X)  : standard_errors_single() -- viet 13/09.
       Hessian co the khong xac dinh duong (nghiem sat bien) -> ghi None va DEM,
       khong coi la 0.

    2. BRIER DUNG cho PR-A. Brier goc tinh TRUOC khi sap nhan trang thai va
       checkpoint chi luu con so vo huong, nen khong sua duoc tu du lieu da co.
       Nay: sinh lai du lieu (cung seed b), SAP NHAN tham so da luu, chay lai bo
       loc tien, cham Brier so voi trang thai that. Chi bo loc, khong toi uu hoa.

Moi (kich ban, phuong phap, b) mot checkpoint rieng trong results/parts/bosung/;
chay song song bang _spawn dung chung; --combine ghep vao 07_simulation_S*.pt
va mo_phong_egarch/E*.pt (ban goc duoc sao luu truoc khi ghi de).

    python scripts/pipeline/bo_sung_coverage_brier.py --nhanh      # thu duong chay
    python scripts/pipeline/bo_sung_coverage_brier.py              # day du, 12 tien trinh
    python scripts/pipeline/bo_sung_coverage_brier.py --combine    # chi ghep
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from msgarchx.inference import model_spec                                  # noqa: E402
from msgarchx.inference.garch_mle import standard_errors_single            # noqa: E402
from msgarchx.inference.mle import natural_scale_standard_errors           # noqa: E402
from msgarchx.inference.param_spec import ModelConfig, pack_params         # noqa: E402
from msgarchx.models.egarch import EGARCHParams                            # noqa: E402
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter  # noqa: E402
from msgarchx.simulation.dgp import simulate_egarch, simulate_ms_garch_x   # noqa: E402
from msgarchx.simulation.metrics import brier_state, coverage              # noqa: E402
from msgarchx.simulation.run_egarch import default_egarch_scenarios, named_params_egarch  # noqa: E402
from msgarchx.simulation.run_simulation import (                           # noqa: E402
    _apply_misspecification, _named_params, sap_nhan_trang_thai)
from msgarchx.simulation.scenarios import default_scenarios                # noqa: E402
from scripts.pipeline.chay_mo_phong_song_song import _spawn                # noqa: E402

RESULTS = ROOT / "results"
SIM_DIR = RESULTS / "parts" / "sim"
EG_DIR = RESULTS / "mo_phong_egarch" / "parts"
OUT_DIR = RESULTS / "parts" / "bosung"
PP = ("mle", "vb_mean_field", "vb_full_rank")
Z = 1.96


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# ===========================================================================
# Dung lai doi tuong tham so tu dict da luu
# ===========================================================================
def _ms_params_tu_dict(e: dict, k: int) -> MSGARCHXParams:
    """Dict ten -> MSGARCHXParams. `e` PHAI da sap nhan (trang thai 1 = bien dong thap)."""
    P = torch.tensor([[e["p11"], 1 - e["p11"]], [1 - e["p22"], e["p22"]]], dtype=torch.float64)
    return MSGARCHXParams(
        mu=torch.tensor([e["mu1"], e["mu2"]], dtype=torch.float64),
        omega=torch.tensor([e["omega1"], e["omega2"]], dtype=torch.float64),
        alpha=torch.tensor([e["alpha1"], e["alpha2"]], dtype=torch.float64),
        beta=torch.tensor([e["beta1"], e["beta2"]], dtype=torch.float64),
        gamma=torch.tensor([[e[f"gamma1_{j}"] for j in range(k)],
                            [e[f"gamma2_{j}"] for j in range(k)]], dtype=torch.float64),
        P=P, nu=torch.tensor(e["nu"], dtype=torch.float64),
    )


def _eg_params_tu_dict(e: dict, k: int) -> EGARCHParams:
    return EGARCHParams(
        mu=torch.tensor(e["mu"], dtype=torch.float64),
        omega=torch.tensor(e["omega"], dtype=torch.float64),
        alpha=torch.tensor(e["alpha"], dtype=torch.float64),
        theta=torch.tensor(e["theta"], dtype=torch.float64),
        beta=torch.tensor(e["beta"], dtype=torch.float64),
        gamma=torch.tensor([e[f"gamma_{j}"] for j in range(k)], dtype=torch.float64),
        nu=torch.tensor(e["nu"], dtype=torch.float64),
    )


# ===========================================================================
# Worker: MOT (kich ban, phuong phap, b)
# ===========================================================================
def worker(kb: str, pp: str, b: int, T: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{kb}_{pp}_{b:03d}.pt"
    if path.exists():
        _log(f"[{kb}/{pp}/{b}] da co, bo qua")
        return
    t0 = time.time()
    ra: dict = {"code": kb, "method": pp, "b": b}

    if kb.startswith("S"):
        # ---------------- PR-A: MS-GARCH-X ----------------
        scn = default_scenarios(T_default=T, B_default=100)[kb]
        cfg = ModelConfig(n_exog=2, state_specific_nu=False)
        sim = simulate_ms_garch_x(scn.params, scn.P0, scn.T, n_exog=2,
                                  burn_in=scn.burn_in, seed=b)
        # S8 la kich ban SAI DAC TA: chuoi uoc luong la r + 0,3 r_{t-1}, khong
        # phai sim.r. Bo loc va Hessian phai chay tren dung chuoi da dung de
        # uoc luong (giong het worker cua chay_mo_phong_song_song.py).
        r = _apply_misspecification(sim.r, scn.misspecified)
        rep = torch.load(SIM_DIR / f"{kb}_{pp}_{b:03d}.pt", weights_only=False)["rep"]
        rep = sap_nhan_trang_thai(rep)
        params = _ms_params_tu_dict(rep.point_estimate, 2)

        # (2) Brier dung: bo loc tien voi tham so DA SAP NHAN, so voi trang thai that.
        with torch.no_grad():
            out = ms_garch_forward_filter(params, r, sim.X)
        ra["brier"] = brier_state(out["xi_filt"].numpy(), sim.state.numpy())

        # (1) Coverage cho MLE: sai so chuan Hessian tai nghiem da luu.
        if pp == "mle":
            raw = pack_params(params, cfg)
            se = natural_scale_standard_errors(raw, cfg, r, sim.X)
            ra["se_ok"] = se is not None
            if se is not None:
                # Ghep SE theo TEN, khong theo vi tri: _named_params() xep p11,p22
                # TRUOC gamma, con natural_scale_standard_errors() tra gamma truoc
                # p11,p22. Ghep theo vi tri se lech o dung cac khoa do.
                k = 2
                se_ten = {"mu1": se["mu"][0], "mu2": se["mu"][1],
                          "omega1": se["omega"][0], "omega2": se["omega"][1],
                          "alpha1": se["alpha"][0], "alpha2": se["alpha"][1],
                          "beta1": se["beta"][0], "beta2": se["beta"][1],
                          "p11": se["p11"].reshape(-1)[0], "p22": se["p22"].reshape(-1)[0],
                          "nu": se["nu"].reshape(-1)[0]}
                g = se["gamma"].reshape(2, k)           # hang s, cot j -> gamma{s+1}_{j}
                for s_ in range(2):
                    for j in range(k):
                        se_ten[f"gamma{s_+1}_{j}"] = g[s_, j]
                ra["ci_lower"] = {k_: float(v) - Z * float(se_ten[k_])
                                  for k_, v in rep.point_estimate.items()}
                ra["ci_upper"] = {k_: float(v) + Z * float(se_ten[k_])
                                  for k_, v in rep.point_estimate.items()}
    else:
        # ---------------- PR-B: EGARCH-X ----------------
        scn = default_egarch_scenarios(T=T, B=100)[kb]
        spec = model_spec.get("egarch_x_t")
        sim = simulate_egarch(scn.params, T=scn.T, dist=scn.dist, n_exog=2,
                              burn_in=scn.burn_in, seed=b)
        rep = torch.load(EG_DIR / f"{kb}_{pp}_{b:03d}.pt", weights_only=False)["rep"]
        # Mo hinh mot trang thai: khong co Brier (khong co trang thai an).
        if pp == "mle":
            params = _eg_params_tu_dict(rep.point_estimate, 2)
            se = standard_errors_single(params, sim.r, sim.X, dist=spec.dist, family=spec.family)
            ra["se_ok"] = se is not None
            if se is not None:
                def _se(k_):
                    if k_.startswith("gamma_"):
                        return float(se["gamma"][int(k_.split("_")[1])])
                    return float(se[k_])
                ra["ci_lower"] = {k_: v - Z * _se(k_) for k_, v in rep.point_estimate.items()}
                ra["ci_upper"] = {k_: v + Z * _se(k_) for k_, v in rep.point_estimate.items()}

    ra["elapsed"] = time.time() - t0
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ra, tmp)
    os.replace(tmp, path)
    _log(f"[{kb}/{pp}/{b}] xong {ra['elapsed']:.1f}s"
         + (f"  brier={ra['brier']:.4f}" if "brier" in ra else "")
         + (f"  SE={'ok' if ra.get('se_ok') else 'KHONG'}" if pp == "mle" else ""))


# ===========================================================================
# Ghep vao file da gop
# ===========================================================================
def combine(nhanh: bool) -> None:
    if nhanh:
        _log("--nhanh: khong ghep vao ket qua that.")
        return
    bk = RESULTS / "lich_su_sua_loi" / f"07_simulation_truoc_bosung_{datetime.now():%Y%m%d_%H%M}"
    bk.mkdir(parents=True, exist_ok=True)

    # ---- PR-A ----
    scen = default_scenarios(T_default=2000, B_default=100)
    for kb in scen:                                  # S1..S8, kich ban nao co file
        f = RESULTS / f"07_simulation_{kb}.pt"
        if not f.exists() or not any(OUT_DIR.glob(f"{kb}_*.pt")):
            continue
        shutil.copy(f, bk / f.name)
        d = torch.load(f, weights_only=False)
        that = _named_params(scen[kb].params)
        that["p11"], that["p22"] = float(scen[kb].P0[0, 0]), float(scen[kb].P0[1, 1])
        for pp in PP:
            fs = sorted(OUT_DIR.glob(f"{kb}_{pp}_*.pt"))
            if not fs:
                continue
            bs = [torch.load(x, weights_only=False) for x in fs]
            d[pp]["mean_brier"] = float(sum(x["brier"] for x in bs) / len(bs))
            d[pp]["brier_da_sap_nhan"] = True
            if pp == "mle":
                co = [x for x in bs if x.get("se_ok")]
                d[pp]["n_se_ok"], d[pp]["n_se_hong"] = len(co), len(bs) - len(co)
                if co:
                    import numpy as np
                    d[pp]["coverage"] = {
                        k_: coverage(np.array([x["ci_lower"][k_] for x in co]),
                                     np.array([x["ci_upper"][k_] for x in co]), that[k_])
                        for k_ in that}
                    d[pp]["coverage_nguon"] = "Wald tu Hessian quan sat, chi tren lan lap co Hessian xac dinh duong"
            _log(f"  {kb}/{pp}: brier={d[pp]['mean_brier']:.4f}"
                 + (f"  coverage MLE tren {d[pp].get('n_se_ok')} lan lap ({d[pp].get('n_se_hong')} Hessian hong)"
                    if pp == "mle" else ""))
        torch.save(d, f)

    # ---- PR-B ----
    scen_e = default_egarch_scenarios(T=2000, B=100)
    for kb in ("E1", "E2", "E3"):
        f = RESULTS / "mo_phong_egarch" / f"{kb}.pt"
        shutil.copy(f, bk / f.name)
        d = torch.load(f, weights_only=False)
        that = named_params_egarch(scen_e[kb].params)
        fs = sorted(OUT_DIR.glob(f"{kb}_mle_*.pt"))
        if not fs:
            continue
        bs = [torch.load(x, weights_only=False) for x in fs]
        co = [x for x in bs if x.get("se_ok")]
        d["mle"]["n_se_ok"], d["mle"]["n_se_hong"] = len(co), len(bs) - len(co)
        if co:
            import numpy as np
            d["mle"]["coverage"] = {
                k_: coverage(np.array([x["ci_lower"][k_] for x in co]),
                             np.array([x["ci_upper"][k_] for x in co]), that[k_])
                for k_ in that}
            d["mle"]["coverage_nguon"] = "Wald tu Hessian quan sat"
        _log(f"  {kb}/mle: coverage MLE tren {len(co)} lan lap ({len(bs)-len(co)} Hessian hong)")
        torch.save(d, f)
    _log(f"Ban goc truoc khi ghep: {bk}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nhanh", action="store_true", help="chi 2 lan lap moi o, de thu")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--worker", nargs=3, metavar=("KB", "PP", "B"))
    ap.add_argument("--kich-ban", nargs="+", default=["S1", "S2", "S3", "E1", "E2", "E3"],
                    help="kich ban can bo sung, vd. --kich-ban S4 S5 S6 S7 S8")
    args = ap.parse_args()
    T = 2000

    if args.worker:
        worker(args.worker[0], args.worker[1], int(args.worker[2]), T)
        return 0
    if args.combine:
        combine(args.nhanh)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    kb_S = [k for k in args.kich_ban if k.startswith("S")]
    kb_E = [k for k in args.kich_ban if k.startswith("E")]
    for kb in kb_S:
        for pp in PP:                       # Brier cho ca ba; SE chi cho mle
            # Chi nhung lan lap DA CO checkpoint uoc luong (S4-S8 co it hon 100).
            co_san = sorted(SIM_DIR.glob(f"{kb}_{pp}_*.pt"))
            if args.nhanh:
                co_san = co_san[:2]
            for f in co_san:
                b = int(f.stem.rsplit("_", 1)[1])
                if not (OUT_DIR / f"{kb}_{pp}_{b:03d}.pt").exists():
                    tasks.append(["--worker", kb, pp, str(b)])
    for kb in kb_E:
        co_san = sorted((RESULTS / "mo_phong_egarch" / "parts").glob(f"{kb}_mle_*.pt"))
        if args.nhanh:
            co_san = co_san[:2]
        for f in co_san:                    # chi mle (coverage)
            b = int(f.stem.rsplit("_", 1)[1])
            if not (OUT_DIR / f"{kb}_mle_{b:03d}.pt").exists():
                tasks.append(["--worker", kb, "mle", str(b)])
    if not tasks:
        _log("Khong con gi thieu -- chi ghep.")
        combine(args.nhanh)
        return 0
    _log(f"Se chay {len(tasks)} tac vu tren {args.jobs} tien trinh")
    rc = _spawn(tasks, args.jobs, "bosung", script=Path(__file__).resolve())
    combine(args.nhanh)
    return rc


if __name__ == "__main__":
    sys.exit(main())
