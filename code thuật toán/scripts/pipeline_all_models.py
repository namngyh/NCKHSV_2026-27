"""Pipeline: TOAN BO mo hinh x 3 phuong phap -> chon mo hinh tot nhat -> MCMC.

CACH CHAY
    python scripts/pipeline_all_models.py                 # chay day du
    python scripts/pipeline_all_models.py --nhanh         # cau hinh nho, de thu duong chay
    python scripts/pipeline_all_models.py --bo-qua-mcmc   # dung truoc Buoc 3
    python scripts/pipeline_all_models.py --chi-bang      # chi in bang tu checkpoint da co

THIET KE BA BUOC (xem CLAUDE.md muc "Cau truc nghien cuu ba buoc")
    Buoc 1  Moi mo hinh trong CATALOGUE x {MLE, VB mean-field, VB full-rank}
    Buoc 2  Mo phong khoi phuc tham so -- chay rieng bang run_full_study.py
    Buoc 3  MCMC chay MOT LAN duy nhat, cho mo hinh thang cuoc o Buoc 1

HAI NGUYEN TAC QUAN TRONG

1. KHONG CHAY LAI THU DA CO. Ket qua cua hai mo hinh chuyen trang thai da duoc
   tinh va kiem chung tu truoc (results/02_mle.pt, 03_vb.pt, 06_oos.pt), va
   duong MLE da duoc chung minh tai lap tung bit. Pipeline NAP LAI chung thay
   vi uoc luong lai -- tiet kiem khoang 9 gio va tranh rui ro lam lech nhung
   con so da bao cao.

2. MOI (mo hinh, phuong phap) DUOC CHECKPOINT RIENG trong results/pipeline/.
   Chay lai script se tu dong bo qua phan da xong. Can thiet vi toan bo qua
   trinh mat nhieu gio va co the bi ngat giua chung.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

try:                                   # in tieng Viet ra console Windows
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from msgarchx.evaluation.oos_single import (          # noqa: E402
    evaluate_single_regime,
    evaluate_single_regime_posterior,
)
from msgarchx.inference import model_spec              # noqa: E402
from msgarchx.inference.bayes_model import model as ms_model  # noqa: E402
from msgarchx.inference.bayes_single import model as single_model  # noqa: E402
from msgarchx.inference.param_spec import ModelConfig  # noqa: E402
from msgarchx.inference.garch_mle import (             # noqa: E402
    fit_garch_mle, standard_errors_single,
)
from msgarchx.inference.mcmc import (                 # noqa: E402
    fit_mcmc, initial_params_single_from_mle,
)
from msgarchx.inference.variational import fit_vb      # noqa: E402
from msgarchx.models.egarch import EGARCHParams        # noqa: E402
from msgarchx.models.garch import GARCHParams          # noqa: E402

PHUONG_PHAP = ("mle", "vb_mean_field", "vb_full_rank")

# Anh xa ma cu trong results/*.pt sang code moi trong model_spec.CATALOGUE.
MA_CU = {
    "garch_n": "b1",
    "garch_t": "b2",
    "garch_x_t": "b3",
    "ms_garch_n": "b4",
    "ms_garch_x_t": "m1",
}


@dataclass
class CauHinh:
    results_dir: Path = ROOT / "results"
    pipeline_dir: Path = ROOT / "results" / "pipeline"
    seed: int = 0
    # MLE
    mle_n_starts: int = 10
    mle_n_adam: int = 500
    mle_n_lbfgs: int = 250
    # VB
    vb_n_steps: int = 3000
    vb_n_draws: int = 1000
    vb_oos_draws: int = 200        # so mau dung khi danh gia ngoai mau theo hau nghiem
    # MCMC (chi chay cho mo hinh thang cuoc)
    mcmc_n_chains: int = 4
    mcmc_n_warmup: int = 1000
    mcmc_n_samples: int = 400
    mcmc_max_tree_depth: int = 7
    mcmc_full_mass: bool = True
    # Tieu chi chon mo hinh thang cuoc
    tieu_chi: str = "qlike"        # "qlike" | "nls" | "bic"

    def nhanh(self) -> "CauHinh":
        """Cau hinh rut gon de thu duong chay trong vai phut.

        Ghi vao THU MUC RIENG (results/pipeline_nhanh/). Neu dung chung thu muc
        voi lan chay that thi cac checkpoint chat luong thap nay se bi lan chay
        that coi la "da xong" va bo qua -- mot cai bay im lang.
        """
        self.mle_n_starts, self.mle_n_adam, self.mle_n_lbfgs = 2, 60, 25
        self.vb_n_steps, self.vb_n_draws, self.vb_oos_draws = 100, 60, 30
        self.mcmc_n_chains, self.mcmc_n_warmup, self.mcmc_n_samples = 2, 30, 30
        self.mcmc_max_tree_depth = 5
        self.pipeline_dir = self.results_dir / "pipeline_nhanh"
        return self


# ---------------------------------------------------------------- tien ich

def checkpoint(cfg: CauHinh, ten: str, tinh_fn):
    """Chay `tinh_fn` neu chua co file; nguoc lai nap lai tu dia."""
    cfg.pipeline_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.pipeline_dir / f"{ten}.pt"
    if path.exists():
        print(f"    [bo qua] da co {path.name}")
        return torch.load(path, weights_only=False)
    t0 = time.time()
    ket_qua = tinh_fn()
    ket_qua["giay"] = time.time() - t0
    torch.save(ket_qua, path)
    print(f"    [luu] {path.name}  ({ket_qua['giay']/60:.1f} phut)")
    return ket_qua


def nap_du_lieu(cfg: CauHinh) -> dict:
    path = cfg.results_dir / "00_data.pt"
    if not path.exists():
        raise SystemExit(
            f"Khong tim thay {path}.\n"
            "Chay buoc chuan bi du lieu truoc: python scripts/run_full_study.py --only data"
        )
    return torch.load(path, weights_only=False)


def lay_X(spec, X_all: torch.Tensor) -> torch.Tensor:
    """Mo hinh khong dung ngoai sinh thi nhan X co 0 cot."""
    if spec.use_exog:
        return X_all
    return torch.zeros(X_all.shape[0], 0, dtype=X_all.dtype)


def _draws_tu_vb(res, spec, n_draws: int) -> list:
    """Chuyen mau hau nghiem cua VB thanh danh sach bo tham so."""
    s = res.samples
    tong = s["mu"].shape[0]
    buoc = max(1, tong // n_draws)
    chi_so = range(0, tong, buoc)

    draws = []
    for i in chi_so:
        gamma = s["gamma"][i] if "gamma" in s else torch.zeros(0)
        nu = s["nu_minus_2"][i] + 2.0 if spec.dist == "student_t" else None
        if spec.family == "garch":
            draws.append(GARCHParams(
                mu=s["mu"][i], omega=s["omega"][i], alpha=s["alpha"][i],
                beta=s["beta"][i], gamma=gamma, nu=nu))
        else:
            draws.append(EGARCHParams(
                mu=s["mu"][i], omega=s["omega"][i], alpha=s["alpha"][i],
                theta=s["theta"][i], beta=s["beta"][i], gamma=gamma, nu=nu))
    return draws[:n_draws]


# ------------------------------------------------------- Buoc 1: uoc luong

def chay_mle(cfg: CauHinh, spec, du_lieu: dict) -> dict:
    n_train = du_lieu["n_train"]
    r_all, X_all, rv_all = du_lieu["r_all"], du_lieu["X_all"], du_lieu["rv_all"]
    X = lay_X(spec, X_all)

    res = fit_garch_mle(
        r_all[:n_train], X[:n_train],
        dist=spec.dist, family=spec.family,
        n_starts=cfg.mle_n_starts, n_adam=cfg.mle_n_adam,
        n_lbfgs=cfg.mle_n_lbfgs, base_seed=cfg.seed, verbose=True,
    )
    oos = evaluate_single_regime(res.params, spec.family, spec.dist,
                                 r_all, X, rv_all, n_train)
    return {
        "code": spec.code, "method": "mle", "params": res.params,
        "loglik": res.loglik, "aic": res.aic, "bic": res.bic,
        "n_params": res.n_params, "runs": res.runs, "oos": oos,
    }


def chay_vb(cfg: CauHinh, spec, du_lieu: dict, guide_type: str) -> dict:
    n_train = du_lieu["n_train"]
    r_all, X_all, rv_all = du_lieu["r_all"], du_lieu["X_all"], du_lieu["rv_all"]
    X = lay_X(spec, X_all)

    res = fit_vb(
        r_all[:n_train], X[:n_train], config=None, guide_type=guide_type,
        n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws,
        seed=cfg.seed, verbose=True,
        model_fn=single_model, model_args=(spec, None),
    )
    draws = _draws_tu_vb(res, spec, cfg.vb_oos_draws)
    oos = evaluate_single_regime_posterior(draws, spec.family, spec.dist,
                                           r_all, X, rv_all, n_train)
    return {
        "code": spec.code, "method": f"vb_{guide_type}",
        "final_elbo": res.final_elbo, "summary": res.summary,
        "samples": res.samples, "oos": oos,
    }


# Gia tri nu dung cho mo hinh CHUAN. Student-t chuan hoa voi nu rat lon hoi tu
# ve phan phoi chuan; dung 1e6 de khop dung cach MLE dang lam cho MS-GARCH chuan.
NU_CHUAN = 1.0e6


def chay_vb_hai_trang_thai(cfg: CauHinh, spec, du_lieu: dict, guide_type: str) -> dict:
    """VB cho mo hinh CHUYEN TRANG THAI (ms_garch_n, ms_garch_x_t).

    Tach khoi chay_vb() vi ho hai trang thai dung mot mo hinh Pyro khac
    (bayes_model thay vi bayes_single) va mot duong danh gia ngoai mau khac.
    """
    from scripts.run_full_study import _eval_msgarchx_posterior

    n_train = du_lieu["n_train"]
    r_all, X_all, rv_all = du_lieu["r_all"], du_lieu["X_all"], du_lieu["rv_all"]
    X = lay_X(spec, X_all)
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    fixed_nu = None if spec.dist == "student_t" else NU_CHUAN

    res = fit_vb(
        r_all[:n_train], X[:n_train], config=config, guide_type=guide_type,
        n_steps=cfg.vb_n_steps, n_posterior_draws=cfg.vb_n_draws,
        seed=cfg.seed, verbose=True,
        model_fn=ms_model, model_args=(config, None, "clamp", 1e-8, fixed_nu),
    )

    samples = dict(res.samples)
    if fixed_nu is not None:
        # params_from_trace() va duong danh gia ngoai mau deu doc khoa
        # "nu_minus_2". Voi mo hinh chuan thi nu khong duoc lay mau nen khoa do
        # vang mat; bo sung lai dung gia tri co dinh de phan con lai khong phai
        # sua gi. KHONG tinh nu nay vao so tham so.
        n = samples["mu"].shape[0]
        samples["nu_minus_2"] = torch.full((n,), float(fixed_nu) - 2.0)

    oos = _eval_msgarchx_posterior(samples, config, r_all, X, rv_all,
                                   n_train, cfg.vb_oos_draws)
    return {
        "code": spec.code, "method": f"vb_{guide_type}",
        "final_elbo": res.final_elbo, "summary": res.summary,
        "samples": samples, "oos": oos,
    }


def chay_buoc_1(cfg: CauHinh, du_lieu: dict, bo_qua: set) -> dict:
    """Uoc luong moi mo hinh MOT trang thai bang ca ba phuong phap.

    `bo_qua` chua cac cap (code, phuong_phap) da co ket qua tu truoc va da
    duoc kiem chung -- khong chay lai (xem nguyen tac 1 o dau file).
    """
    ket_qua = {}
    for spec in model_spec.CATALOGUE:
        print(f"\n  === {spec.name} (d={spec.n_params(du_lieu['X_all'].shape[1])}) ===")
        if (spec.code, "mle") in bo_qua:
            print("    [nap lai] MLE da co trong 02_mle.pt, khong chay lai")
        elif spec.n_regimes == 1:
            ket_qua[(spec.code, "mle")] = checkpoint(
                cfg, f"{spec.code}_mle", lambda s=spec: chay_mle(cfg, s, du_lieu))
        else:
            # MLE cho mo hinh chuyen trang thai qua dat de chay lai o day
            # (M1 mat 424,7 phut); nap lai tu 02_mle.pt, xem nap_ket_qua_cu.
            print("    [bo qua] MLE mo hinh chuyen trang thai: nap lai tu 02_mle.pt")
        for guide in ("mean_field", "full_rank"):
            if (spec.code, f"vb_{guide}") in bo_qua:
                print(f"    [nap lai] VB {guide} da co, khong chay lai")
                continue
            chay = chay_vb_hai_trang_thai if spec.n_regimes == 2 else chay_vb
            ket_qua[(spec.code, f"vb_{guide}")] = checkpoint(
                cfg, f"{spec.code}_vb_{guide}",
                lambda s=spec, g=guide, f=chay: f(cfg, s, du_lieu, g))
    return ket_qua


def kiem_ke(cfg: CauHinh, da_co: dict, k: int) -> None:
    """In ke hoach: cai gi se chay, cai gi bo qua, va vi sao. Khong tinh toan gi.

    Dung truoc khi bat dau mot lan chay dai (nhat la tren may moi) de biet
    truoc pipeline se lam gi.
    """
    se_chay, bo_qua_cu, bo_qua_checkpoint = [], [], []
    for spec in model_spec.CATALOGUE:
        for pp in PHUONG_PHAP:
            if (spec.code, pp) in da_co:
                bo_qua_cu.append((spec.name, pp))
            elif (cfg.pipeline_dir / f"{spec.code}_{pp}.pt").exists():
                bo_qua_checkpoint.append((spec.name, pp))
            elif pp == "mle" and spec.n_regimes == 2:
                # MLE cho mo hinh chuyen trang thai khong chay o day (qua dat,
                # M1 mat 424,7 phut) va cung da co san trong 02_mle.pt.
                continue
            else:
                se_chay.append((spec.name, pp))

    print("\n" + "=" * 72)
    print("KIEM KE -- pipeline se lam gi neu chay bay gio")
    print("=" * 72)

    print(f"\n[1] BO QUA, nap lai tu ket qua cu da kiem chung ({len(bo_qua_cu)} muc):")
    for ten, pp in bo_qua_cu:
        print(f"      {ten:<24} {pp}")

    if bo_qua_checkpoint:
        print(f"\n[2] BO QUA, da co checkpoint trong {cfg.pipeline_dir.name}/ "
              f"({len(bo_qua_checkpoint)} muc):")
        for ten, pp in bo_qua_checkpoint:
            print(f"      {ten:<24} {pp}")

    print(f"\n[3] SE CHAY ({len(se_chay)} muc):")
    for ten, pp in se_chay:
        print(f"      {ten:<24} {pp}")

    print("\n[4] SE CHAY: MCMC cho mo hinh thang cuoc (tru khi dung --bo-qua-mcmc)")
    print("\nMoc chi phi da DO tren may nay (de uoc luong; MLE mot trang thai ho GARCH):")
    print("      GARCH chuan 13,6 phut | GARCH Student-t 13,7 | GARCH-X Student-t 14,6")
    print("      EGARCH du kien cao hon ~10-20% (vong lap co them mot can va mot phep chia).")
    print("      VB cho mo hinh MOT trang thai: CHUA DO tren may nay.")
    print("=" * 72)


def nap_ket_qua_cu(cfg: CauHinh) -> dict:
    """Nap ket qua MLE/VB da co tu cac lan chay truoc, thay vi uoc luong lai.

    Nap duoc hai nhom:
      - Ba mo hinh GARCH mot trang thai (b1/b2/b3 trong 02_mle.pt). An toan vi
        da kiem chung: ban tai cau truc cua garch_mle.py cho log-hop-ly TRUNG
        TUNG BIT voi ban cu, va evaluate_single_regime tai lap CHINH XAC moi
        chi tieu ngoai mau da luu (chenh lech 0,00e+00).
      - Hai mo hinh chuyen trang thai (b4/m1, cung m2/m3 cho VB).
    """
    ra = {}
    mle_path, oos_path = cfg.results_dir / "02_mle.pt", cfg.results_dir / "06_oos.pt"
    if not (mle_path.exists() and oos_path.exists()):
        print("  [canh bao] khong thay 02_mle.pt / 06_oos.pt -- se uoc luong lai tat ca")
        return ra

    mle_cu, oos_cu = torch.load(mle_path, weights_only=False), torch.load(oos_path, weights_only=False)
    vb_path = cfg.results_dir / "03_vb.pt"
    vb_cu = torch.load(vb_path, weights_only=False) if vb_path.exists() else {}

    for spec in model_spec.CATALOGUE:
        ma = MA_CU.get(spec.code)
        if ma is None or ma not in mle_cu:
            continue                      # vd. ba mo hinh EGARCH: chua tung chay
        res = mle_cu[ma]
        ra[(spec.code, "mle")] = {
            "code": spec.code, "method": "mle", "params": res.params,
            "loglik": res.loglik, "aic": res.aic, "bic": res.bic,
            "n_params": getattr(res, "n_params", spec.n_params(2)),
            "oos": oos_cu.get(ma, {}), "nap_lai": True,
        }
        # m2/m3 la VB cho dung mo hinh MS-GARCH-X Student-t
        if spec.code == "ms_garch_x_t":
            for khoa_cu, ten in [("m2", "vb_mean_field"), ("m3", "vb_full_rank")]:
                if khoa_cu in vb_cu:
                    ra[(spec.code, ten)] = {
                        "code": spec.code, "method": ten,
                        "final_elbo": vb_cu[khoa_cu].final_elbo,
                        "oos": oos_cu.get(khoa_cu, {}), "nap_lai": True,
                    }
    return ra


# ------------------------------------------------------ Buoc 1b: bang so sanh

def bang_so_sanh(ket_qua: dict, k: int) -> list[dict]:
    hang = []
    for spec in model_spec.CATALOGUE:
        for pp in PHUONG_PHAP:
            r = ket_qua.get((spec.code, pp))
            if r is None:
                continue
            oos = r.get("oos") or {}
            hang.append({
                "ten": spec.name, "code": spec.code, "phuong_phap": pp,
                "d": spec.n_params(k),
                "loglik": r.get("loglik"), "aic": r.get("aic"), "bic": r.get("bic"),
                "elbo": r.get("final_elbo"),
                "qlike": oos.get("qlike"), "nls": oos.get("nls"),
                "mae": oos.get("mae"), "rmse": oos.get("rmse"),
                "nap_lai": r.get("nap_lai", False),
            })
    return hang


def in_bang(hang: list[dict]) -> None:
    print("\n" + "=" * 108)
    print(f"{'Mo hinh':<24}{'Phuong phap':<16}{'d':>3}{'loglik':>12}{'AIC':>11}{'BIC':>11}"
          f"{'QLIKE':>10}{'NLS':>9}{'':>3}")
    print("-" * 108)
    for h in hang:
        def s(x, w, nd=4):
            return f"{x:>{w}.{nd}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"
        dau = " *" if h["nap_lai"] else ""
        print(f"{h['ten']:<24}{h['phuong_phap']:<16}{h['d']:>3}"
              f"{s(h['loglik'],12,2)}{s(h['aic'],11,2)}{s(h['bic'],11,2)}"
              f"{s(h['qlike'],10)}{s(h['nls'],9)}{dau}")
    print("=" * 108)
    print("* = nap lai tu ket qua da tinh truoc do, khong chay lai (xem docstring dau file)")


def chon_thang_cuoc(hang: list[dict], tieu_chi: str) -> dict:
    """Chon (mo hinh, phuong phap) tot nhat theo tieu chi da khai bao.

    Moi tieu chi deu la "cang nho cang tot". Chi xet cac hang co gia tri hop le.
    """
    hop_le = [h for h in hang if isinstance(h.get(tieu_chi), (int, float))]
    if not hop_le:
        raise SystemExit(f"Khong co hang nao co tieu chi {tieu_chi!r} de xep hang.")
    return min(hop_le, key=lambda h: h[tieu_chi])


# ------------------------------------------------------------ Buoc 3: MCMC

def chay_mcmc_thang_cuoc(cfg: CauHinh, spec, du_lieu: dict) -> dict:
    n_train = du_lieu["n_train"]
    r_all, X_all = du_lieu["r_all"], du_lieu["X_all"]
    X = lay_X(spec, X_all)

    if spec.n_regimes != 1:
        raise SystemExit(
            f"Mo hinh thang cuoc la {spec.name} (hai trang thai). MCMC cho mo hinh nay\n"
            "da chay 3 lan va KHONG hoi tu (R-hat = 2,84) -- xem CLAUDE.md. Neu van\n"
            "muon chay, dung run_full_study.py --only mcmc thay vi script nay."
        )

    print(f"\n  MCMC cho mo hinh thang cuoc: {spec.name}")
    print("  Luu y: mo hinh MOT trang thai khong co doi xung hoan vi nhan,")
    print("  nen khong gap van de da dinh da lam hong lan chay MS-GARCH-X truoc day.")

    # KHOI TAO TU NGHIEM MLE, khong de Pyro tu khoi tao ngau nhien.
    # Lan chay 13/09 khong truyen initial_params: chuoi dau thich nghi tot (buoc
    # 0,70) nhung mot chuoi sau roi vao theta = +1,03, beta = 0,32 -- nguoc dau
    # don bay -- roi buoc sup xuong 1,8e-07 va dung yen. Bon chuoi van PHAN TAN
    # quanh nghiem MLE de R-hat con y nghia chan doan.
    mle_path = cfg.pipeline_dir / f"{spec.code}_mle.pt"
    if not mle_path.exists():
        raise SystemExit(
            f"Chua co nghiem MLE tai {mle_path.name} de khoi tao cac chuoi MCMC.\n"
            "Chay Buoc 1 truoc."
        )
    params_mle = torch.load(mle_path, weights_only=False)["params"]
    # Do lon nhieu loan lay theo SAI SO CHUAN tung tham so, khong phai mot hang
    # so chung -- xem giai thich trong initial_params_single_from_mle().
    se = standard_errors_single(params_mle, r_all[:n_train], X[:n_train],
                                dist=spec.dist, family=spec.family)
    if se is None:
        print("  [canh bao] Hessian khong xac dinh duong, khong lay duoc sai so")
        print("             chuan; lui ve nhieu loan ty le 10% gia tri tham so.")
    print("  Khoi tao 4 chuoi quanh nghiem MLE (chuoi 0 dat dung tai nghiem,")
    print("  ba chuoi con lai phan tan 2 sai so chuan).")

    res = fit_mcmc(
        r_all[:n_train], X[:n_train], config=None, prior=None,
        n_chains=cfg.mcmc_n_chains, n_warmup=cfg.mcmc_n_warmup,
        n_samples=cfg.mcmc_n_samples, seed=cfg.seed,
        max_tree_depth=cfg.mcmc_max_tree_depth, full_mass=cfg.mcmc_full_mass,
        verbose=True,
        checkpoint_path=cfg.pipeline_dir / f"mcmc_{spec.code}_tiendo.pt",
        model_fn=single_model, model_args=(spec, None),
        initial_params=lambda c: initial_params_single_from_mle(params_mle, spec, c, se=se),
    )
    print(f"    R-hat lon nhat = {res.max_rhat:.4f}   ESS nho nhat = {res.min_ess:.1f}"
          f"   phan ky = {res.n_divergences}")
    if res.max_rhat < 1.01:
        print("    -> DAT nguong hoi tu (R-hat < 1,01).")
    else:
        print("    -> CHUA dat nguong hoi tu; bao cao kem canh bao.")
    return {"code": spec.code, "method": "mcmc", "summary": res.summary,
            "samples": res.samples, "max_rhat": res.max_rhat,
            "min_ess": res.min_ess, "n_divergences": res.n_divergences}


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nhanh", action="store_true", help="cau hinh nho de thu duong chay")
    ap.add_argument("--bo-qua-mcmc", action="store_true", help="dung truoc Buoc 3")
    ap.add_argument("--chi-bang", action="store_true", help="chi in bang tu checkpoint da co")
    ap.add_argument("--kiem-ke", action="store_true",
                    help="chi in ke hoach (chay gi, bo qua gi) roi thoat, khong tinh toan")
    ap.add_argument("--chay-lai-tat-ca", action="store_true",
                    help="uoc luong lai ca nhung mo hinh da co ket qua tu truoc")
    ap.add_argument("--tieu-chi", default="qlike", choices=["qlike", "nls", "bic"])
    args = ap.parse_args()

    cfg = CauHinh(tieu_chi=args.tieu_chi)
    if args.nhanh:
        cfg = cfg.nhanh()

    # Tao thu muc ngay tu dau: duong --chi-bang khong goi checkpoint() nen neu
    # doi den luc torch.save(bang_so_sanh) moi tao thi se loi "Parent directory
    # does not exist".
    cfg.pipeline_dir.mkdir(parents=True, exist_ok=True)

    du_lieu = nap_du_lieu(cfg)
    k = du_lieu["X_all"].shape[1]
    print(f"Du lieu: T={du_lieu['r_all'].shape[0]} quan sat, "
          f"huan luyen={du_lieu['n_train']}, kiem dinh={du_lieu['r_all'].shape[0]-du_lieu['n_train']}, "
          f"k={k} bien ngoai sinh")

    print("\nNap ket qua da co tu cac lan chay truoc (khong uoc luong lai)")
    da_co = nap_ket_qua_cu(cfg)
    for (code, pp) in sorted(da_co):
        print(f"    [nap lai] {code:<14} {pp}")

    if args.kiem_ke:
        kiem_ke(cfg, da_co, k)
        return

    if args.chi_bang:
        ket_qua = {}
        for spec in model_spec.CATALOGUE:
            for pp in PHUONG_PHAP:
                p = cfg.pipeline_dir / f"{spec.code}_{pp}.pt"
                if p.exists():
                    ket_qua[(spec.code, pp)] = torch.load(p, weights_only=False)
    else:
        print("\nBUOC 1 -- uoc luong phan CON THIEU bang 3 phuong phap")
        bo_qua = set(da_co) if not args.chay_lai_tat_ca else set()
        ket_qua = chay_buoc_1(cfg, du_lieu, bo_qua)

    # Ket qua vua tinh duoc uu tien; phan nao thieu thi lay tu lan chay truoc.
    gop = dict(da_co)
    gop.update(ket_qua)
    ket_qua = gop

    hang = bang_so_sanh(ket_qua, k)
    in_bang(hang)
    torch.save(hang, cfg.pipeline_dir / "bang_so_sanh.pt")

    thang = chon_thang_cuoc(hang, cfg.tieu_chi)
    print(f"\nMo hinh tot nhat theo {cfg.tieu_chi.upper()}: "
          f"{thang['ten']} + {thang['phuong_phap']}  ({cfg.tieu_chi}={thang[cfg.tieu_chi]:.4f})")

    # In them xep hang theo cac tieu chi con lai de thay ro neu chung bat dong.
    for tc in ("qlike", "nls", "bic"):
        if tc == cfg.tieu_chi:
            continue
        try:
            t = chon_thang_cuoc(hang, tc)
            print(f"  (theo {tc.upper()} thi tot nhat lai la: {t['ten']} + {t['phuong_phap']})")
        except SystemExit:
            pass

    if args.bo_qua_mcmc:
        print("\n--bo-qua-mcmc: dung tai day.")
        return

    if args.chi_bang:
        # --chi-bang nghia la CHI IN BANG. Truoc day no chi bo qua Buoc 1 roi
        # chay tiep xuong day, nen mot lenh nguoi dung tuong la doc-va-in lai
        # am tham khoi dong mot lan MCMC dai 8 tieng. Da xay ra that ngay
        # 13/09: lenh `--chi-bang` chay nen suot toi, ghi de len chinh file
        # tien do cua mot lan chay khac, khien moi so doc tu file do khong con
        # dang tin va dan toi mot loat chan doan sai.
        print("\n--chi-bang: chi in bang, khong chay Buoc 3.")
        print("Muon chay MCMC thi bo --chi-bang.")
        return

    print("\nBUOC 3 -- MCMC cho mo hinh thang cuoc")
    spec_thang = model_spec.get(thang["code"])
    checkpoint(cfg, f"mcmc_{spec_thang.code}",
               lambda: chay_mcmc_thang_cuoc(cfg, spec_thang, du_lieu))
    print("\nXONG.")


if __name__ == "__main__":
    main()
