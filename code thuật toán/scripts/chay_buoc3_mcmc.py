"""BUOC 3 -- MCMC (NUTS) cho mo hinh thang cuoc cua Buoc 1.

CACH CHAY (tu thu muc "code thuật toán"):
    python scripts/chay_buoc3_mcmc.py                       # mo hinh tot nhat theo QLIKE o Buoc 1
    python scripts/chay_buoc3_mcmc.py --mo-hinh egarch_x_t  # chi dinh mo hinh
    python scripts/chay_buoc3_mcmc.py --nhanh               # 2 chuoi ngan, de thu (vai phut)

Cau hinh chinh: 4 chuoi x (1.000 warmup + 400 mau), do sau cay toi da 7, ma
tran khoi luong dac. Chuoi 0 xuat phat DUNG tai nghiem MLE, ba chuoi con lai
phan tan 2 sai so chuan quanh no. Mat khoang 8 gio cho EGARCH-X Student-t.
Ket qua: results/pipeline/mcmc_<mo_hinh>.pt (mau hau nghiem, tom tat, R-hat, ESS).
"""
from __future__ import annotations

import argparse

from chung import RESULTS, checkpoint, nap

from garch_vn30.mo_hinh import danh_muc
from garch_vn30.suy_dien import fit_mcmc, khoi_tao_tu_mle, standard_errors


def chon_mo_hinh(tieu_chi: str = "qlike") -> str:
    """Mo hinh (bat ky phuong phap) co tieu chi ngoai mau nho nhat trong bang Buoc 1."""
    hang = [h for h in nap(RESULTS / "pipeline" / "bang_so_sanh.pt") if h.get(tieu_chi) is not None]
    return min(hang, key=lambda h: h[tieu_chi])["code"]


def chay_mcmc(spec, nhanh: bool) -> dict:
    du_lieu = nap(RESULTS / "00_data.pt")
    n = du_lieu["n_train"]
    r, X = du_lieu["r_all"][:n], danh_muc.lay_X(spec, du_lieu["X_all"])[:n]

    mle = nap(RESULTS / "pipeline" / f"{spec.code}_mle.pt")
    params = danh_muc.tham_so_tu_dict(mle["params"], spec.family)
    se = standard_errors(params, r, X, dist=spec.dist, family=spec.family)
    if se is None:
        print("  [canh bao] Hessian khong xac dinh duong: nhieu loan 10% gia tri tham so.")

    cau_hinh = (dict(n_chains=2, n_warmup=30, n_samples=30, max_tree_depth=5) if nhanh else
                dict(n_chains=4, n_warmup=1000, n_samples=400, max_tree_depth=7))
    res = fit_mcmc(r, X, spec, full_mass=True, seed=0, verbose=True,
                   checkpoint_path=RESULTS / ("pipeline_nhanh" if nhanh else "pipeline")
                   / f"mcmc_{spec.code}_tiendo.pt",
                   initial_params=lambda c: khoi_tao_tu_mle(params, spec, c, se=se),
                   **cau_hinh)
    print(f"  R-hat lon nhat = {res.max_rhat:.4f}, ESS nho nhat = {res.min_ess:.0f}, "
          f"so phan ky = {res.n_divergences}")
    print("  -> " + ("DAT" if res.max_rhat < 1.01 else "CHUA DAT") + " nguong hoi tu R-hat < 1,01")
    return {"code": spec.code, "method": "mcmc", "summary": res.summary, "samples": res.samples,
            "max_rhat": res.max_rhat, "min_ess": res.min_ess, "n_divergences": res.n_divergences,
            "rhat": res.rhat, "ess_bulk": res.ess_bulk}


def main() -> None:
    ap = argparse.ArgumentParser(description="Buoc 3: MCMC cho mo hinh thang cuoc")
    ap.add_argument("--mo-hinh", default=None, help="code mo hinh; mac dinh: tot nhat theo QLIKE")
    ap.add_argument("--nhanh", action="store_true")
    args = ap.parse_args()

    spec = danh_muc.get(args.mo_hinh or chon_mo_hinh())
    print(f"MCMC cho {spec.name}")
    thu_muc = RESULTS / ("pipeline_nhanh" if args.nhanh else "pipeline")
    checkpoint(thu_muc / f"mcmc_{spec.code}.pt", lambda: chay_mcmc(spec, args.nhanh))


if __name__ == "__main__":
    main()
