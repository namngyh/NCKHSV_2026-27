"""BUOC 1 -- So sanh 6 mo hinh GARCH/EGARCH x 3 phuong phap tren du lieu VN30F1M.

CACH CHAY (tu thu muc "code thuật toán"):
    python scripts/chay_buoc1_so_sanh_mo_hinh.py             # chay day du (nhieu gio)
    python scripts/chay_buoc1_so_sanh_mo_hinh.py --kiem-ke   # chi xem se chay gi
    python scripts/chay_buoc1_so_sanh_mo_hinh.py --chi-bang  # chi in bang tu ket qua da co
    python scripts/chay_buoc1_so_sanh_mo_hinh.py --nhanh     # cau hinh nho de thu (vai phut)

QUY TRINH
  0. Du lieu: results/00_data.pt (loi suat, RV, bien ngoai sinh, chia 80/20) va
     kiem dinh ban dau results/01_preliminary_tests.pt.
  1. Voi moi mo hinh trong danh muc va moi phuong phap (MLE, VB mean-field,
     VB full-rank): uoc luong tren tap huan luyen, danh gia ngoai mau tren tap
     kiem dinh. Moi cap luu rieng results/pipeline/<mo_hinh>_<phuong_phap>.pt;
     chay lai se BO QUA cap da co, nen co the ngat giua chung.
  2. In bang so sanh (log-hop-ly, AIC, BIC, ELBO, QLIKE, NLS) va luu
     results/pipeline/bang_so_sanh.pt. Mo hinh thang cuoc chuyen sang Buoc 3.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

from chung import CSV, PHUONG_PHAP, RESULTS, checkpoint, luu, nap

from garch_vn30.danh_gia import danh_gia_hau_nghiem, danh_gia_mot_diem
from garch_vn30.du_lieu import chuan_bi_du_lieu
from garch_vn30.kiem_dinh import run_all_preliminary_tests
from garch_vn30.mo_hinh import danh_muc
from garch_vn30.suy_dien import fit_mle, fit_vb
from garch_vn30.suy_dien.bayes import rut_bo_tham_so


@dataclass
class CauHinh:
    thu_muc: str = "pipeline"
    seed: int = 0
    mle_n_starts: int = 10       # so diem khoi tao MLE
    mle_n_adam: int = 500
    mle_n_lbfgs: int = 250
    vb_n_steps: int = 3000       # so buoc SVI
    vb_n_draws: int = 1000       # so mau rut tu q sau khi khop
    vb_oos_draws: int = 200      # so bo tham so dung cho mat do du bao hon hop

    @classmethod
    def nhanh(cls) -> "CauHinh":
        """Cau hinh rut gon, ghi vao thu muc RIENG de khong lan voi ket qua that."""
        return cls(thu_muc="pipeline_nhanh", mle_n_starts=2, mle_n_adam=60, mle_n_lbfgs=25,
                   vb_n_steps=100, vb_n_draws=60, vb_oos_draws=30)


def nap_du_lieu() -> dict:
    du_lieu = checkpoint(RESULTS / "00_data.pt", lambda: chuan_bi_du_lieu(str(CSV)))
    checkpoint(RESULTS / "01_preliminary_tests.pt",
               lambda: run_all_preliminary_tests(du_lieu["r_all"].numpy()))
    return du_lieu


def chay_mle(cfg: CauHinh, spec, du_lieu: dict) -> dict:
    r, X_all, rv, n = du_lieu["r_all"], du_lieu["X_all"], du_lieu["rv_all"], du_lieu["n_train"]
    X = danh_muc.lay_X(spec, X_all)
    res = fit_mle(r[:n], X[:n], dist=spec.dist, family=spec.family, n_starts=cfg.mle_n_starts,
                  n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs, base_seed=cfg.seed, verbose=True)
    return {
        "code": spec.code, "method": "mle",
        "params": danh_muc.tham_so_sang_dict(res.params),
        "loglik": res.loglik, "aic": res.aic, "bic": res.bic, "n_params": res.n_params,
        "runs": [asdict(x) for x in res.runs],
        "oos": danh_gia_mot_diem(res.params, spec.family, spec.dist, r, X, rv, n),
    }


def chay_vb(cfg: CauHinh, spec, du_lieu: dict, guide: str) -> dict:
    r, X_all, rv, n = du_lieu["r_all"], du_lieu["X_all"], du_lieu["rv_all"], du_lieu["n_train"]
    X = danh_muc.lay_X(spec, X_all)
    res = fit_vb(r[:n], X[:n], spec, guide_type=guide, n_steps=cfg.vb_n_steps,
                 n_posterior_draws=cfg.vb_n_draws, seed=cfg.seed, verbose=True)
    draws = rut_bo_tham_so(res.samples, spec, cfg.vb_oos_draws)
    return {
        "code": spec.code, "method": f"vb_{guide}",
        "final_elbo": res.final_elbo, "summary": res.summary, "samples": res.samples,
        "oos": danh_gia_hau_nghiem(draws, spec.family, spec.dist, r, X, rv, n),
    }


def uoc_luong_tat_ca(cfg: CauHinh, du_lieu: dict) -> None:
    thu_muc = RESULTS / cfg.thu_muc
    k = du_lieu["X_all"].shape[1]
    for spec in danh_muc.CATALOGUE:
        print(f"\n=== {spec.name} (d = {spec.n_params(k)}) ===")
        checkpoint(thu_muc / f"{spec.code}_mle.pt", lambda s=spec: chay_mle(cfg, s, du_lieu))
        for guide in ("mean_field", "full_rank"):
            checkpoint(thu_muc / f"{spec.code}_vb_{guide}.pt",
                       lambda s=spec, g=guide: chay_vb(cfg, s, du_lieu, g))


def bang_so_sanh(cfg: CauHinh, k: int) -> list[dict]:
    hang = []
    for spec in danh_muc.CATALOGUE:
        for pp in PHUONG_PHAP:
            f = RESULTS / cfg.thu_muc / f"{spec.code}_{pp}.pt"
            if not f.exists():
                continue
            o = nap(f)
            oos = o["oos"]
            hang.append({"ten": spec.name, "code": spec.code, "phuong_phap": pp,
                         "d": spec.n_params(k), "loglik": o.get("loglik"), "aic": o.get("aic"),
                         "bic": o.get("bic"), "elbo": o.get("final_elbo"),
                         "qlike": oos["qlike"], "nls": oos["nls"], "mae": oos["mae"],
                         "rmse": oos["rmse"]})
    return hang


def in_bang(hang: list[dict]) -> None:
    def s(x, w, nd):
        return f"{x:>{w}.{nd}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"
    print("\n" + "=" * 100)
    print(f"{'Mo hinh':<22}{'Phuong phap':<15}{'d':>3}{'loglik':>12}{'AIC':>11}{'BIC':>11}"
          f"{'ELBO':>11}{'QLIKE':>9}{'NLS':>8}")
    print("-" * 100)
    for h in hang:
        print(f"{h['ten']:<22}{h['phuong_phap']:<15}{h['d']:>3}{s(h['loglik'], 12, 2)}"
              f"{s(h['aic'], 11, 2)}{s(h['bic'], 11, 2)}{s(h['elbo'], 11, 2)}"
              f"{s(h['qlike'], 9, 4)}{s(h['nls'], 8, 4)}")
    print("=" * 100)
    for tc in ("aic", "bic", "qlike", "nls"):
        hop_le = [h for h in hang if isinstance(h.get(tc), (int, float))]
        if hop_le:
            t = min(hop_le, key=lambda h: h[tc])
            print(f"  Tot nhat theo {tc.upper():<5}: {t['ten']} + {t['phuong_phap']} ({t[tc]:.4f})")


def kiem_ke(cfg: CauHinh) -> None:
    print(f"Thu muc ket qua: results/{cfg.thu_muc}/")
    for spec in danh_muc.CATALOGUE:
        for pp in PHUONG_PHAP:
            co = (RESULTS / cfg.thu_muc / f"{spec.code}_{pp}.pt").exists()
            print(f"  {spec.name:<22} {pp:<15} {'da co -> bo qua' if co else 'SE CHAY'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Buoc 1: so sanh 6 mo hinh x 3 phuong phap")
    ap.add_argument("--nhanh", action="store_true", help="cau hinh nho de thu duong chay")
    ap.add_argument("--chi-bang", action="store_true", help="chi in bang tu ket qua da co")
    ap.add_argument("--kiem-ke", action="store_true", help="chi liet ke se chay gi")
    args = ap.parse_args()
    cfg = CauHinh.nhanh() if args.nhanh else CauHinh()

    if args.kiem_ke:
        kiem_ke(cfg)
        return
    du_lieu = nap_du_lieu()
    n, T = du_lieu["n_train"], du_lieu["r_all"].shape[0]
    print(f"Du lieu: {T} phien, huan luyen {n}, kiem dinh {T - n}, k = {du_lieu['X_all'].shape[1]}")
    if not args.chi_bang:
        uoc_luong_tat_ca(cfg, du_lieu)
    hang = bang_so_sanh(cfg, du_lieu["X_all"].shape[1])
    in_bang(hang)
    luu(hang, RESULTS / cfg.thu_muc / "bang_so_sanh.pt")


if __name__ == "__main__":
    main()
