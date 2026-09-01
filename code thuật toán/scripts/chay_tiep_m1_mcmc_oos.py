"""Chay lien tuc ba buoc con lai (tru mo phong), khong can nguoi can thiep:

  BUOC 1  Uoc luong LAI M1 cho hoi tu
          - khoi tao AM tu nghiem B3 (GARCH-X Student-t da co) thay vi chi
            dung nhieu ngau nhien -- dung yeu cau o README muc 16 "Quy trinh
            uoc luong";
          - moi diem khoi tao chay trong MOT tien trinh rieng (song song);
          - moi diem khoi tao chay toi 3 chu ky Adam+L-BFGS, dung som khi
            ||grad|| < 1.
          Ly do phai lam truoc: buoc `oos` dung thang tham so M1, va M4 (MCMC)
          lay mau tren dung mo hinh nay.

  BUOC 2  MCMC (NUTS) cho M4 -- MOI CHUOI MOT TIEN TRINH
          4 chuoi doc lap hoan toan ve mat thong ke nen chay song song khong
          mat gi; mcmc.py goc chay tuan tu tung chuoi trong mot vong `for`.
          Ket qua gop lai thanh dung MCMCResult (R-hat/ESS tinh bang arviz
          tren ca 4 chuoi, giong het fit_mcmc).

  BUOC 3  Danh gia ngoai mau (`oos`) tren toan bo B1-B4, M1, M2, M3, M4.

Moi buoc ghi checkpoint rieng nen dung giua chung khong mat cac buoc truoc.

Chay:
    python scripts/chay_tiep_m1_mcmc_oos.py --all
    python scripts/chay_tiep_m1_mcmc_oos.py --step m1
    python scripts/chay_tiep_m1_mcmc_oos.py --step mcmc
    python scripts/chay_tiep_m1_mcmc_oos.py --step oos
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

from msgarchx.inference.mcmc import initial_params_from_mle
from msgarchx.inference.mle import MLEResult, fit_mle, natural_scale_standard_errors
from msgarchx.inference.param_spec import (ModelConfig, init_raw, pack_params,
                                            unpack_raw, warm_starts_from_garch)
from msgarchx.models.ms_garch import MSGARCHXParams, ms_garch_forward_filter
from scripts.run_full_study import PRESETS

N_WARM = 4          # so diem khoi tao lay tu nghiem B3
N_RANDOM = 6        # so diem khoi tao ngau nhien
N_CYCLES = 3        # so chu ky Adam+L-BFGS toi da moi diem khoi tao
JOBS = 8


def _log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def _data(cfg):
    d = torch.load(cfg.results_dir / "00_data.pt", weights_only=False)
    return d, d["r_all"][: d["n_train"]], d["X_all"][: d["n_train"]]


# ============================================================
# BUOC 1 -- M1
# ============================================================
def warm_starts_from_b3(cfg, config: ModelConfig) -> list[torch.Tensor]:
    """Bien nghiem B3 (mot trang thai) thanh cac diem khoi tao hai trang thai.

    B3 da uoc luong tot phan "trung binh" cua chuoi; hai trang thai chi khac
    nhau o muc do bien dong, nen nhan omega len/xuong quanh gia tri cua B3 la
    cach dat diem xuat phat hop ly hon nhieu so voi nhieu ngau nhien.
    """
    b3_path = cfg.results_dir / "parts" / "b3.pt"
    if not b3_path.exists():
        _log("khong tim thay parts/b3.pt -- bo qua khoi tao am, dung ngau nhien het")
        return []
    b3 = torch.load(b3_path, weights_only=False)["result"].params
    return warm_starts_from_garch(b3, config, n_variants=N_WARM)



# Dieu kien chinh quy de mot nghiem duoc coi la CHAP NHAN DUOC (Cach 2).
#
# Ham hop ly cua mo hinh hon hop / chuyen trang thai co CAC DINH GIA voi gia
# tri tuy y cao: mot trang thai co the co lai thanh "bay bat diem ngoai le"
# chi ton tai dung mot phien, keo log-hop-ly len ma khong mang y nghia kinh te
# nao. Day la benh da biet cua MS-GARCH, khong phai loi cai dat. Thuc hanh
# chuan trong tai lieu la bao cao CUC DAI DIA PHUONG LON NHAT THOA cac dieu
# kien chinh quy, chu khong phai gia tri lon nhat tim duoc.
#
# Ba dieu kien duoc ap dung o day (phai khai bao trong subsec:reproducibility
# cua bai bao):
#   C1  Da hoi tu: ||grad|| < 1 tren thang khong rang buoc.
#   C2  Moi trang thai phai la MOT CHE DO, khong phai mot cu soc don le:
#       thoi gian ton tai ky vong d_s = 1/(1-p_ss) >= 2 phien.
#   C3  Xac suat duy tri khong nam sat biên: 0.01 < p_ss < 0.999.
#
# LUU Y: KHONG ap dieu kien alpha_s + beta_s < 1 cho tung trang thai. Trong
# MS-GARCH mot che do rieng le duoc phep no ma qua trinh tong the van dung
# (dieu kien dung la dieu kien tren ban kinh pho lien quan ca P); ap rang buoc
# do se la mot han che do nguoi nghien cuu tu dat, khong phai yeu cau ly thuyet.
MIN_DURATION = 2.0
P_LOW, P_HIGH = 0.01, 0.999


def kiem_tra_chinh_quy(params) -> tuple[bool, str]:
    """Tra ve (dat_chinh_quy, ly_do_neu_khong_dat) cho mot bo tham so."""
    p_diag = torch.diagonal(params.P)
    dur = 1.0 / (1.0 - p_diag)
    ly_do = []
    if bool((dur < MIN_DURATION).any()):
        ly_do.append(f"d_s={[round(v,2) for v in dur.tolist()]} < {MIN_DURATION}")
    if bool((p_diag <= P_LOW).any() or (p_diag >= P_HIGH).any()):
        ly_do.append(f"p_ss={[round(v,4) for v in p_diag.tolist()]} sat bien")
    return (not ly_do), "; ".join(ly_do)


def m1_start_worker(cfg, k: int) -> None:
    sdir = cfg.results_dir / "parts" / "m1_starts"
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{k:02d}.pt"
    if path.exists():
        _log(f"[m1 start {k}] da co {path}, bo qua")
        return
    _, r, X = _data(cfg)
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    warm = warm_starts_from_b3(cfg, config)
    if k < len(warm):
        init, kind = warm[k], f"am tu B3 (bien the {k})"
    else:
        init, kind = init_raw(config, seed=1000 + k), f"ngau nhien (seed {1000+k})"
    _log(f"[m1 start {k}] khoi tao: {kind}")
    t0 = time.time()
    res = fit_mle(r, X, config, n_starts=1, base_seed=k,
                  n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs,
                  init_raws=[init], n_cycles=N_CYCLES, verbose=True)
    d = res.runs[0]
    torch.save({"raw": res.raw, "loglik": res.loglik, "diag": d, "kind": kind,
                "elapsed": time.time() - t0}, path)
    _log(f"[m1 start {k}] xong sau {(time.time()-t0)/60:.1f} phut: "
         f"loglik={res.loglik:.4f} ||grad||={d.grad_norm:.4f} "
         f"hoi_tu={'co' if d.converged else 'KHONG'}")


def m1_combine(cfg) -> None:
    sdir = cfg.results_dir / "parts" / "m1_starts"
    files = sorted(sdir.glob("*.pt"))
    if not files:
        raise SystemExit("[LOI] khong co ket qua diem khoi tao nao cho M1")
    _, r, X = _data(cfg)
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    loaded = [torch.load(f, weights_only=False) for f in files]

    runs = [d["diag"] for d in loaded]
    finite = [d for d in loaded if d["loglik"] == d["loglik"]]
    conv = [d for d in finite if d["diag"].converged]

    # Danh gia dieu kien chinh quy cho tung diem khoi tao
    for d in loaded:
        d["_params"] = unpack_raw(d["raw"], config)
        d["_ok"], d["_ly_do"] = kiem_tra_chinh_quy(d["_params"])
    regular = [d for d in conv if d["_ok"]]

    _log("  Bang chan doan tung diem khoi tao:")
    for d in sorted(loaded, key=lambda x: -x["loglik"] if x["loglik"] == x["loglik"] else 1e18):
        dur = 1.0 / (1.0 - torch.diagonal(d["_params"].P))
        _log(f"    loglik={d['loglik']:11.4f} ||grad||={d['diag'].grad_norm:12.4f} "
             f"d_s={[round(v,1) for v in dur.tolist()]:} "
             f"{'hoi tu' if d['diag'].converged else 'CHUA hoi tu'} "
             f"{'chinh quy' if d['_ok'] else 'SUY BIEN: ' + d['_ly_do']}")

    if regular:
        pool, nhan = regular, "DA HOI TU va THOA DIEU KIEN CHINH QUY"
    elif conv:
        pool, nhan = conv, "da hoi tu nhung KHONG diem nao thoa chinh quy (canh bao!)"
    elif finite:
        pool, nhan = finite, "KHONG diem nao hoi tu (canh bao nghiem trong!)"
    else:
        raise SystemExit("[LOI] moi diem khoi tao cua M1 deu NaN")
    best = max(pool, key=lambda d: d["loglik"])
    bo_qua = [d["loglik"] for d in finite if d["loglik"] > best["loglik"]]
    if bo_qua:
        _log(f"  Da BO QUA {len(bo_qua)} nghiem co log-hop-ly cao hon "
             f"(cao nhat {max(bo_qua):.4f}) vi khong thoa dieu kien chinh quy -- "
             f"day la dinh gia, xem ghi chu o dau ham kiem_tra_chinh_quy.")
    used_converged = bool(regular or conv)
    _log(f"  Chon nghiem: {nhan}")

    raw = best["raw"]
    params = unpack_raw(raw, config)
    with torch.no_grad():
        filt = ms_garch_forward_filter(params, r, X)
    kth = config.raw_dim
    aic = -2.0 * best["loglik"] + 2.0 * kth
    bic = -2.0 * best["loglik"] + kth * torch.log(torch.tensor(float(r.shape[0]))).item()
    res = MLEResult(params=params, raw=raw, loglik=best["loglik"], aic=aic, bic=bic,
                    config=config, runs=runs, filter_output=filt)

    _log(f"[m1] gop {len(loaded)} diem khoi tao | hoi tu {len(conv)}/{len(loaded)} | "
         f"chinh quy {len(regular)}/{len(loaded)} | "
         f"loglik={res.loglik:.4f} ||grad||={best['diag'].grad_norm:.4f}")
    _log("[m1] tinh sai so chuan (delta method tu Hessian) ...")
    se = natural_scale_standard_errors(raw, config, r, X)
    if se is None:
        _log("[m1] Hessian khong xac dinh duong -- khong co sai so chuan")

    pdir = cfg.results_dir / "parts"
    old = pdir / "m1.pt"
    if old.exists():
        bak = pdir / "m1_lan_chay_truoc.pt"
        if not bak.exists():
            old.rename(bak)
            _log(f"[m1] da giu lai ket qua cu tai {bak.name}")
    torch.save({"result": res, "m1_se": se, "model": "m1",
                "elapsed": sum(d["elapsed"] for d in loaded),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "n_starts_converged": len(conv), "n_starts_total": len(loaded),
                "n_starts_regular": len(regular),
                "quy_tac_chon": "cuc dai dia phuong lon nhat thoa C1 (hoi tu) + "
                                "C2 (d_s >= 2 phien) + C3 (p_ss khong sat bien)"},
               pdir / "m1.pt")
    _log(f"[m1] da ghi {pdir / 'm1.pt'}")


# ============================================================
# BUOC 2 -- MCMC, moi chuoi mot tien trinh
# ============================================================

def khoi_tao_chuoi_tu_mle(cfg, c: int) -> dict | None:
    """Diem khoi tao cua chuoi `c`: nghiem MLE chinh quy cua M1, co nhieu loan.

    Lan chay dau lay mau khoi tao tu TIEN NGHIEM nen cac chuoi bat dau o nhung
    vung vo ly va khong bao gio gap nhau. O day dat chung quanh nghiem MLE da
    thoa dieu kien chinh quy -- nhung VAN PHAN TAN (moi chuoi mot nhieu loan
    khac nhau), vi ban thao yeu cau "4 chuoi khoi tao phan tan" de R-hat con y
    nghia chan doan. Chuoi 0 dat dung tai nghiem MLE, ba chuoi con lai nhieu loan.
    """
    path = cfg.results_dir / "parts" / "m1.pt"
    if not path.exists():
        _log("[mcmc] khong tim thay parts/m1.pt -- dung khoi tao mac dinh cua Pyro")
        return None
    pm = torch.load(path, weights_only=False)["result"].params
    # Logic nhieu loan da chuyen vao msgarchx.inference.mcmc de
    # run_full_study.py dung CHUNG mot bo khoi tao (truoc day chi script
    # nay co, nen chay lai M4 qua run_full_study cho cau hinh KHAC).
    return initial_params_from_mle(pm, c)


def mcmc_chain_worker(cfg, c: int) -> None:
    from msgarchx.inference.mcmc import fit_mcmc
    cdir = cfg.results_dir / "parts" / "mcmc_chains"
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"chain{c}.pt"
    if path.exists():
        _log(f"[mcmc chuoi {c}] da co {path}, bo qua")
        return
    _, r, X = _data(cfg)
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)
    _log(f"[mcmc chuoi {c}] bat dau: warmup={cfg.mcmc_n_warmup} "
         f"samples={cfg.mcmc_n_samples} max_tree_depth={cfg.mcmc_max_tree_depth} "
         f"full_mass={cfg.mcmc_full_mass}")
    t0 = time.time()
    # seed = cfg.seed + c, GIONG HET cach fit_mcmc goc danh seed cho chuoi c.
    init = khoi_tao_chuoi_tu_mle(cfg, c)
    if init is not None:
        _log(f"[mcmc chuoi {c}] khoi tao tu nghiem MLE chinh quy"
             + (" (dung tai nghiem)" if c == 0 else " + nhieu loan"))
    # Neu lan chay truoc bi ngat (vd. Windows tu khoi dong lai), dung buoc tich
    # phan da thich nghi va diem tham so cuoi cung lam khoi dau -- ret ngan
    # giai doan warmup thay vi lam lai tu dau.
    cp = cfg.results_dir / "parts" / "mcmc_chains" / f"tiendo_chuoi{c}.pt"
    if cp.exists():
        tr = torch.load(cp, weights_only=False)
        _log(f"[mcmc chuoi {c}] TIM THAY tien do cu: {tr['giai_doan']} vong {tr['vong']}, "
             f"step_size={tr['step_size']:.3e} -- dung lam diem khoi dau")
        if tr.get("tham_so"):
            init = tr["tham_so"]
    res = fit_mcmc(r, X, config, n_chains=1, n_warmup=cfg.mcmc_n_warmup,
                   n_samples=cfg.mcmc_n_samples, seed=cfg.seed + c,
                   max_tree_depth=cfg.mcmc_max_tree_depth, verbose=True,
                   full_mass=cfg.mcmc_full_mass, initial_params=init,
                   checkpoint_path=cp, checkpoint_moi=25)
    torch.save({"samples_by_chain": res.samples_by_chain,
                "n_divergences": res.n_divergences,
                "chain": c, "elapsed": time.time() - t0}, path)
    _log(f"[mcmc chuoi {c}] xong sau {(time.time()-t0)/3600:.2f} gio, "
         f"phan ky={res.n_divergences}")


def mcmc_combine(cfg) -> None:
    import arviz as az
    from msgarchx.inference.mcmc import MCMCResult, _SITES
    from msgarchx.inference.summary import posterior_summary

    cdir = cfg.results_dir / "parts" / "mcmc_chains"
    files = sorted(cdir.glob("chain*.pt"))
    if not files:
        raise SystemExit("[LOI] khong co chuoi MCMC nao")
    loaded = [torch.load(f, weights_only=False) for f in files]
    _, _, X = _data(cfg)
    config = ModelConfig(n_exog=X.shape[1], state_specific_nu=False)

    samples_by_chain = {}
    for name in _SITES:
        parts = [d["samples_by_chain"][name] for d in loaded
                 if name in d["samples_by_chain"]]
        if parts:
            samples_by_chain[name] = torch.cat(parts, dim=0)   # (n_chains, n_samples, ...)
    samples = {n: v.reshape(-1, *v.shape[2:]) for n, v in samples_by_chain.items()}
    n_div = sum(d["n_divergences"] for d in loaded)

    try:
        post = {n: v.numpy() for n, v in samples_by_chain.items()}
        try:
            idata = az.from_dict({"posterior": post})      # arviz >= 1.0
        except TypeError:
            idata = az.from_dict(posterior=post)            # arviz < 1.0
        rhat = {n: float(az.rhat(idata)[n].values.max()) for n in samples_by_chain}
        ess = {n: float(az.ess(idata, method="bulk")[n].values.min()) for n in samples_by_chain}
    except Exception as exc:
        _log(f"[mcmc] arviz that bai ({exc}); R-hat/ESS = NaN")
        rhat = {n: float("nan") for n in samples_by_chain}
        ess = {n: float("nan") for n in samples_by_chain}

    res = MCMCResult(samples=samples, samples_by_chain=samples_by_chain,
                     summary=posterior_summary(samples), rhat=rhat, ess_bulk=ess,
                     max_rhat=max(rhat.values()) if rhat else float("nan"),
                     min_ess=min(ess.values()) if ess else float("nan"),
                     n_divergences=n_div, config=config)
    torch.save({"m4": res}, cfg.results_dir / "04_mcmc.pt")
    _log(f"[mcmc] gop {len(loaded)} chuoi -> 04_mcmc.pt | "
         f"R-hat lon nhat={res.max_rhat:.4f} ESS nho nhat={res.min_ess:.1f} "
         f"phan ky={n_div}")
    if res.max_rhat == res.max_rhat and res.max_rhat > 1.01:
        _log("[mcmc] CANH BAO: R-hat > 1.01 -- cac chuoi CHUA hoi tu ve cung hau nghiem")
    if n_div > 0:
        _log(f"[mcmc] CANH BAO: co {n_div} lan phan ky -- mau co the BI LECH")


# ============================================================
# Khoi chay song song
# ============================================================

def b4_worker(cfg) -> None:
    """B4 (MS-GARCH chuan, nu co dinh lon, khong X) -- 6 diem khoi tao trong
    MOT tien trinh, vi fit_mle da tu lap qua cac diem khoi tao."""
    pdir = cfg.results_dir / "parts"
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / "b4.pt"
    if path.exists():
        _log(f"[b4] da co {path}, bo qua")
        return
    _, r, X = _data(cfg)
    zero_X = torch.zeros(r.shape[0], 0, dtype=r.dtype)
    config_b4 = ModelConfig(n_exog=0, state_specific_nu=False)
    _log("[b4] bat dau")
    t0 = time.time()
    res = fit_mle(r, zero_X, config_b4, n_starts=cfg.mle_n_starts, base_seed=cfg.seed,
                  n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs, fixed_nu=1e6,
                  n_cycles=N_CYCLES, verbose=True)
    torch.save({"result": res, "model": "b4", "elapsed": time.time() - t0,
                "finished_at": datetime.now().isoformat(timespec="seconds")}, path)
    _log(f"[b4] xong sau {(time.time()-t0)/60:.1f} phut: loglik={res.loglik:.4f}")



# ============================================================
# B4 -- cung co che nhieu diem khoi tao song song + quy tac chinh quy nhu M1
# ============================================================
def warm_starts_from_b1(cfg, config: ModelConfig) -> list[torch.Tensor]:
    """Khoi tao am cho B4 (MS-GARCH chuan, khong X) tu nghiem B1."""
    b1_path = cfg.results_dir / "parts" / "b1.pt"
    if not b1_path.exists():
        return []
    b1 = torch.load(b1_path, weights_only=False)["result"].params
    return warm_starts_from_garch(b1, config, n_variants=N_WARM)


def b4_start_worker(cfg, k: int) -> None:
    sdir = cfg.results_dir / "parts" / "b4_starts"
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{k:02d}.pt"
    if path.exists():
        _log(f"[b4 start {k}] da co, bo qua")
        return
    _, r, X = _data(cfg)
    zero_X = torch.zeros(r.shape[0], 0, dtype=r.dtype)
    config = ModelConfig(n_exog=0, state_specific_nu=False)
    warm = warm_starts_from_b1(cfg, config)
    if k < len(warm):
        init, kind = warm[k], f"am tu B1 (bien the {k})"
    else:
        init, kind = init_raw(config, seed=2000 + k), f"ngau nhien (seed {2000+k})"
    _log(f"[b4 start {k}] khoi tao: {kind}")
    t0 = time.time()
    res = fit_mle(r, zero_X, config, n_starts=1, base_seed=k, fixed_nu=1e6,
                  n_adam=cfg.mle_n_adam, n_lbfgs=cfg.mle_n_lbfgs,
                  init_raws=[init], n_cycles=N_CYCLES, verbose=True)
    torch.save({"raw": res.raw, "loglik": res.loglik, "diag": res.runs[0], "kind": kind,
                "elapsed": time.time() - t0}, path)
    _log(f"[b4 start {k}] xong {(time.time()-t0)/60:.1f} phut: loglik={res.loglik:.4f} "
         f"||grad||={res.runs[0].grad_norm:.4f}")


def b4_combine(cfg) -> None:
    sdir = cfg.results_dir / "parts" / "b4_starts"
    files = sorted(sdir.glob("*.pt"))
    if not files:
        raise SystemExit("[LOI] khong co diem khoi tao nao cho B4")
    _, r, _ = _data(cfg)
    zero_X = torch.zeros(r.shape[0], 0, dtype=r.dtype)
    config = ModelConfig(n_exog=0, state_specific_nu=False)
    loaded = [torch.load(f, weights_only=False) for f in files]
    for d in loaded:
        d["_params"] = unpack_raw(d["raw"], config)
        d["_ok"], d["_ly_do"] = kiem_tra_chinh_quy(d["_params"])
    finite = [d for d in loaded if d["loglik"] == d["loglik"]]
    conv = [d for d in finite if d["diag"].converged]
    regular = [d for d in conv if d["_ok"]]
    _log("  Bang chan doan B4:")
    for d in sorted(loaded, key=lambda x: -x["loglik"]):
        dur = 1.0 / (1.0 - torch.diagonal(d["_params"].P))
        _log(f"    loglik={d['loglik']:11.4f} ||grad||={d['diag'].grad_norm:11.4f} "
             f"d_s={[round(v,1) for v in dur.tolist()]} "
             f"{'hoi tu' if d['diag'].converged else 'CHUA hoi tu'} "
             f"{'chinh quy' if d['_ok'] else 'SUY BIEN: ' + d['_ly_do']}")
    pool = regular or conv or finite
    if not pool:
        raise SystemExit("[LOI] moi diem khoi tao cua B4 deu NaN")
    best = max(pool, key=lambda d: d["loglik"])
    bo_qua = [d["loglik"] for d in finite if d["loglik"] > best["loglik"]]
    if bo_qua:
        _log(f"  Da BO QUA {len(bo_qua)} nghiem log-hop-ly cao hon (cao nhat "
             f"{max(bo_qua):.4f}) vi khong chinh quy")
    params = unpack_raw(best["raw"], config)
    with torch.no_grad():
        filt = ms_garch_forward_filter(params, r, zero_X)
    kth = config.raw_dim - 1          # nu co dinh, khong tinh vao so tham so
    aic = -2.0 * best["loglik"] + 2.0 * kth
    bic = -2.0 * best["loglik"] + kth * torch.log(torch.tensor(float(r.shape[0]))).item()
    res = MLEResult(params=params, raw=best["raw"], loglik=best["loglik"], aic=aic,
                    bic=bic, config=config, runs=[d["diag"] for d in loaded],
                    filter_output=filt)
    pdir = cfg.results_dir / "parts"
    old = pdir / "b4.pt"
    if old.exists() and not (pdir / "b4_lan_chay_truoc.pt").exists():
        old.rename(pdir / "b4_lan_chay_truoc.pt")
        _log("  da giu lai B4 cu tai b4_lan_chay_truoc.pt")
    torch.save({"result": res, "model": "b4",
                "elapsed": sum(d["elapsed"] for d in loaded),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "n_starts_converged": len(conv), "n_starts_regular": len(regular),
                "n_starts_total": len(loaded),
                "quy_tac_chon": "cuc dai dia phuong lon nhat thoa C1+C2+C3"}, pdir / "b4.pt")
    _log(f"[b4] hoi tu {len(conv)}/{len(loaded)}, chinh quy {len(regular)}/{len(loaded)}, "
         f"chon loglik={res.loglik:.4f} AIC={aic:.2f}")


def step_b4(cfg) -> None:
    _log("=" * 70)
    _log("B4: chay lai voi nhieu diem khoi tao song song + quy tac chinh quy")
    _log("=" * 70)
    jobs = int(os.environ.get("B4_JOBS", "4"))
    n = N_WARM + N_RANDOM
    _spawn(cfg, [["--worker-b4-start", str(k)] for k in range(n)], jobs, "b4start")
    b4_combine(cfg)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_models_parallel.py"),
                    "--merge"], cwd=str(ROOT), check=False)
    _log("B4 HOAN TAT")

def step_cach3(cfg_goc) -> None:
    """Bien the kiem tra do ben: alpha_s, beta_s thuoc (0,1) bang ham lien ket
    sigmoid, trung mien voi tien nghiem Beta(a,a) cua VB/MCMC.

    Ghi vao mot thu muc RIENG (results_sigmoid/) de ket qua chinh theo dung
    dac ta ban thao (softplus) khong bi ghi de.
    """
    from dataclasses import replace as dc_replace
    rd = ROOT / "results_sigmoid"
    rd.mkdir(parents=True, exist_ok=True)
    src = cfg_goc.results_dir / "00_data.pt"
    dst = rd / "00_data.pt"
    if not dst.exists():
        import shutil
        shutil.copy2(src, dst)
        _log(f"[cach3] da chep {src.name} sang {rd.name}/")
    # B3 dung lam khoi tao am: chep sang de warm_starts_from_b3 tim thay
    pdir = rd / "parts"; pdir.mkdir(exist_ok=True)
    b3_src = cfg_goc.results_dir / "parts" / "b3.pt"
    if b3_src.exists() and not (pdir / "b3.pt").exists():
        import shutil
        shutil.copy2(b3_src, pdir / "b3.pt")

    cfg = dc_replace(cfg_goc, results_dir=rd)
    _log("=" * 70)
    _log("CACH 3: chay lai M1 va B4 voi alpha_s, beta_s thuoc (0,1) [sigmoid]")
    _log(f"  thu muc rieng: {rd}")
    _log("=" * 70)
    n = N_WARM + N_RANDOM
    jobs = int(os.environ.get("CACH3_JOBS", "5"))
    tasks = [["--worker-m1-start", str(k), "--results-dir", str(rd)] for k in range(n)]
    tasks.append(["--worker-b4", "--results-dir", str(rd)])
    _spawn(cfg, tasks, jobs, "cach3", extra_env={"MSGARCHX_AB_LINK": "sigmoid"})
    # gop M1 cung phai chay voi sigmoid -> goi lai chinh file nay o tien trinh con
    env = dict(os.environ); env["MSGARCHX_AB_LINK"] = "sigmoid"
    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                    "--worker-m1-combine", "--results-dir", str(rd)],
                   cwd=str(ROOT), env=env, check=False)
    _log("CACH 3 HOAN TAT")

def _spawn(cfg, args_list: list[list[str]], jobs: int, tag: str,
           extra_env: dict | None = None) -> int:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    if extra_env:
        env.update(extra_env)
    queue, running, failed = list(args_list), [], []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < jobs:
            extra = queue.pop(0)
            # Ten log chi lay tu cac doi so NGAN (bo duong dan, vi duong dan
            # chua dau \ se lam hong ten tep tren Windows).
            name = "_".join(a.lstrip("-") for a in extra[:2]
                            if len(a) < 24 and "\\" not in a and "/" not in a) or "task"
            fh = open(log_dir / f"{tag}_{name}_{stamp}.log", "w", encoding="utf-8",
                      errors="replace")
            cmd = [sys.executable, str(Path(__file__).resolve())] + extra
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(ROOT))
            running.append((name, proc, fh))
            _log(f"  bat dau {name} (pid {proc.pid})")
        time.sleep(10)
        still = []
        for name, proc, fh in running:
            rc = proc.poll()
            if rc is None:
                still.append((name, proc, fh))
                continue
            fh.close()
            if rc == 0:
                _log(f"  XONG {name} ({(time.time()-t0)/60:.0f} phut tu dau buoc)")
            else:
                failed.append(name)
                _log(f"  LOI {name} (ma thoat {rc}) -- xem logs/{tag}_{name}_{stamp}.log")
        running = still
    if failed:
        _log(f"  cac tien trinh LOI: {failed}")
    return len(failed)


def step_m1(cfg) -> None:
    _log("=" * 70)
    _log(f"BUOC 1: uoc luong lai M1 ({N_WARM} diem am tu B3 + {N_RANDOM} ngau nhien, "
         f"toi da {N_CYCLES} chu ky moi diem)")
    _log("=" * 70)
    n = N_WARM + N_RANDOM
    _spawn(cfg, [["--worker-m1-start", str(k)] for k in range(n)], JOBS, "m1start")
    m1_combine(cfg)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_models_parallel.py"),
                    "--merge"], cwd=str(ROOT), check=False)
    _log("BUOC 1 HOAN TAT")


def step_mcmc(cfg) -> None:
    _log("=" * 70)
    _log(f"BUOC 2: MCMC (NUTS) -- {cfg.mcmc_n_chains} chuoi, moi chuoi mot tien trinh")
    _log("=" * 70)
    _spawn(cfg, [["--worker-mcmc-chain", str(c)] for c in range(cfg.mcmc_n_chains)],
           cfg.mcmc_n_chains, "mcmcchain")
    mcmc_combine(cfg)
    _log("BUOC 2 HOAN TAT")


def step_oos(cfg) -> None:
    _log("=" * 70)
    _log("BUOC 3: danh gia ngoai mau (oos)")
    _log("=" * 70)
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_full_study.py"),
                         "--preset", "full", "--only", "oos"],
                        cwd=str(ROOT), check=False).returncode
    _log(f"BUOC 3 {'HOAN TAT' if rc == 0 else f'LOI (ma thoat {rc})'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Chay lien tuc buoc 1 -> 2 -> 3")
    ap.add_argument("--step", choices=["m1", "m1-chon-lai", "b4", "mcmc", "oos", "cach3"])
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--worker-b4", action="store_true")
    ap.add_argument("--worker-b4-start", type=int)
    ap.add_argument("--worker-m1-combine", action="store_true")
    ap.add_argument("--worker-m1-start", type=int)
    ap.add_argument("--worker-mcmc-chain", type=int)
    ap.add_argument("--preset", default="full")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    if args.results_dir:
        from dataclasses import replace as _dc_replace
        cfg = _dc_replace(cfg, results_dir=Path(args.results_dir))

    if args.worker_b4:
        b4_worker(cfg)
        return 0
    if args.worker_b4_start is not None:
        b4_start_worker(cfg, args.worker_b4_start)
        return 0
    if args.worker_m1_combine:
        m1_combine(cfg)
        return 0
    if args.worker_m1_start is not None:
        m1_start_worker(cfg, args.worker_m1_start)
        return 0
    if args.worker_mcmc_chain is not None:
        mcmc_chain_worker(cfg, args.worker_mcmc_chain)
        return 0

    t0 = time.time()
    if args.all:
        step_m1(cfg)
        step_mcmc(cfg)
        step_oos(cfg)
        _log(f"TAT CA HOAN TAT sau {(time.time()-t0)/3600:.2f} gio")
        return 0
    if args.step == "m1":
        step_m1(cfg)
    elif args.step == "m1-chon-lai":
        # Chi ap dung LAI quy tac chon tren cac diem khoi tao DA CO, khong tinh lai
        m1_combine(cfg)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_models_parallel.py"),
                        "--merge"], cwd=str(ROOT), check=False)
    elif args.step == "mcmc":
        step_mcmc(cfg)
    elif args.step == "oos":
        step_oos(cfg)
    elif args.step == "b4":
        step_b4(cfg)
    elif args.step == "cach3":
        step_cach3(cfg)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
